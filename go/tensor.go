package demucs

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
)

// Tensor is a dense float32 tensor in row-major order.
type Tensor struct {
	Data  []float32
	Shape []int
}

func (t *Tensor) numel() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// safetensorsFile is the parsed content of a .safetensors file.
type safetensorsFile struct {
	metadata map[string]string
	tensors  map[string]*Tensor
}

type safetensorsEntry struct {
	DType       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets [2]int `json:"data_offsets"`
}

func readSafetensorsFile(path string) (*safetensorsFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return readSafetensors(f)
}

// readSafetensors parses the safetensors format: a little endian u64 header
// size, a JSON header, then the raw tensor bytes. F32, F16 and BF16 tensors
// are supported and always converted to float32.
func readSafetensors(r io.Reader) (*safetensorsFile, error) {
	var size uint64
	if err := binary.Read(r, binary.LittleEndian, &size); err != nil {
		return nil, fmt.Errorf("demucs: reading safetensors header size: %w", err)
	}
	if size > 100<<20 {
		return nil, fmt.Errorf("demucs: safetensors header too large (%d bytes)", size)
	}
	raw := make([]byte, size)
	if _, err := io.ReadFull(r, raw); err != nil {
		return nil, fmt.Errorf("demucs: reading safetensors header: %w", err)
	}
	var header map[string]json.RawMessage
	if err := json.Unmarshal(raw, &header); err != nil {
		return nil, fmt.Errorf("demucs: parsing safetensors header: %w", err)
	}
	out := &safetensorsFile{metadata: map[string]string{}, tensors: map[string]*Tensor{}}
	type named struct {
		name string
		safetensorsEntry
	}
	var entries []named
	for name, msg := range header {
		if name == "__metadata__" {
			if err := json.Unmarshal(msg, &out.metadata); err != nil {
				return nil, fmt.Errorf("demucs: parsing safetensors metadata: %w", err)
			}
			continue
		}
		var e safetensorsEntry
		if err := json.Unmarshal(msg, &e); err != nil {
			return nil, fmt.Errorf("demucs: parsing safetensors entry %q: %w", name, err)
		}
		entries = append(entries, named{name, e})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].DataOffsets[0] < entries[j].DataOffsets[0] })
	pos := 0
	for _, e := range entries {
		start, end := e.DataOffsets[0], e.DataOffsets[1]
		if start < pos || end < start {
			return nil, fmt.Errorf("demucs: tensor %q has invalid offsets %v", e.name, e.DataOffsets)
		}
		if _, err := io.CopyN(io.Discard, r, int64(start-pos)); err != nil {
			return nil, err
		}
		buf := make([]byte, end-start)
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("demucs: reading tensor %q: %w", e.name, err)
		}
		pos = end
		t := &Tensor{Shape: e.Shape}
		data, err := decodeFloats(buf, e.DType, t.numel())
		if err != nil {
			return nil, fmt.Errorf("demucs: tensor %q: %w", e.name, err)
		}
		t.Data = data
		out.tensors[e.name] = t
	}
	return out, nil
}

func decodeFloats(buf []byte, dtype string, n int) ([]float32, error) {
	width := map[string]int{"F32": 4, "F16": 2, "BF16": 2}[dtype]
	if width == 0 {
		return nil, fmt.Errorf("unsupported dtype %s", dtype)
	}
	if len(buf) != n*width {
		return nil, fmt.Errorf("expected %d bytes, got %d", n*width, len(buf))
	}
	out := make([]float32, n)
	for i := range out {
		switch dtype {
		case "F32":
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:]))
		case "F16":
			out[i] = float16to32(binary.LittleEndian.Uint16(buf[2*i:]))
		case "BF16":
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(buf[2*i:])) << 16)
		}
	}
	return out, nil
}

func float16to32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	switch {
	case exp == 0 && mant == 0:
		return math.Float32frombits(sign)
	case exp == 0:
		// Subnormal: renormalize.
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | e<<23 | mant<<13)
	case exp == 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | mant<<13)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
}

// weights is a view on the tensors of one model, under a name prefix.
type weights struct {
	prefix  string
	tensors map[string]*Tensor
	used    map[string]bool
}

func (w *weights) get(name string) (*Tensor, error) {
	t, ok := w.tensors[w.prefix+name]
	if !ok {
		return nil, fmt.Errorf("demucs: missing weight %s%s", w.prefix, name)
	}
	w.used[w.prefix+name] = true
	return t, nil
}

func (w *weights) has(name string) bool {
	_, ok := w.tensors[w.prefix+name]
	return ok
}

// unused lists tensors under the prefix that were never looked up, which
// signals a checkpoint with layers this port does not implement.
func (w *weights) unused() []string {
	var out []string
	for name := range w.tensors {
		if strings.HasPrefix(name, w.prefix) && !w.used[name] {
			out = append(out, name)
		}
	}
	sort.Strings(out)
	return out
}
