package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

const (
	formatPCM        = 1
	formatFloat      = 3
	formatExtensible = 0xfffe
)

// readWAV decodes a PCM (8/16/24/32 bit) or IEEE float (32/64 bit) WAV file
// into [channel][sample] float32 in [-1, 1].
func readWAV(path string) ([][]float32, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	var riff [12]byte
	if _, err := io.ReadFull(r, riff[:]); err != nil {
		return nil, 0, fmt.Errorf("%s: %w", path, err)
	}
	if string(riff[0:4]) != "RIFF" || string(riff[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("%s: not a RIFF/WAVE file", path)
	}
	var format, channels, bits int
	var rate int
	for {
		var hdr [8]byte
		if _, err := io.ReadFull(r, hdr[:]); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return nil, 0, fmt.Errorf("%s: no data chunk", path)
			}
			return nil, 0, err
		}
		id := string(hdr[0:4])
		size := int64(binary.LittleEndian.Uint32(hdr[4:8]))
		switch id {
		case "fmt ":
			buf := make([]byte, size)
			if _, err := io.ReadFull(r, buf); err != nil {
				return nil, 0, err
			}
			if len(buf) < 16 {
				return nil, 0, fmt.Errorf("%s: short fmt chunk", path)
			}
			format = int(binary.LittleEndian.Uint16(buf[0:2]))
			channels = int(binary.LittleEndian.Uint16(buf[2:4]))
			rate = int(binary.LittleEndian.Uint32(buf[4:8]))
			bits = int(binary.LittleEndian.Uint16(buf[14:16]))
			if format == formatExtensible && len(buf) >= 26 {
				format = int(binary.LittleEndian.Uint16(buf[24:26]))
			}
		case "data":
			if channels == 0 {
				return nil, 0, fmt.Errorf("%s: data chunk before fmt chunk", path)
			}
			buf := make([]byte, size)
			n, err := io.ReadFull(r, buf)
			if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) {
				return nil, 0, err
			}
			out, err := decodeSamples(buf[:n], format, channels, bits)
			if err != nil {
				return nil, 0, fmt.Errorf("%s: %w", path, err)
			}
			return out, rate, nil
		default:
			if _, err := io.CopyN(io.Discard, r, size+size%2); err != nil {
				return nil, 0, err
			}
		}
		if id == "fmt " && size%2 == 1 {
			r.ReadByte()
		}
	}
}

func decodeSamples(buf []byte, format, channels, bits int) ([][]float32, error) {
	width := bits / 8
	if width == 0 || (format != formatPCM && format != formatFloat) {
		return nil, fmt.Errorf("unsupported WAV encoding (format %d, %d bits)", format, bits)
	}
	if format == formatFloat && bits != 32 && bits != 64 {
		return nil, fmt.Errorf("unsupported float WAV with %d bits", bits)
	}
	frames := len(buf) / (width * channels)
	out := make([][]float32, channels)
	for c := range out {
		out[c] = make([]float32, frames)
	}
	for i := 0; i < frames; i++ {
		for c := 0; c < channels; c++ {
			b := buf[(i*channels+c)*width:]
			var v float32
			switch {
			case format == formatFloat && bits == 32:
				v = math.Float32frombits(binary.LittleEndian.Uint32(b))
			case format == formatFloat:
				v = float32(math.Float64frombits(binary.LittleEndian.Uint64(b)))
			case bits == 8:
				v = (float32(b[0]) - 128) / 128
			case bits == 16:
				v = float32(int16(binary.LittleEndian.Uint16(b))) / (1 << 15)
			case bits == 24:
				s := int32(uint32(b[0])<<8|uint32(b[1])<<16|uint32(b[2])<<24) >> 8
				v = float32(s) / (1 << 23)
			case bits == 32:
				v = float32(int32(binary.LittleEndian.Uint32(b))) / (1 << 31)
			default:
				return nil, fmt.Errorf("unsupported PCM WAV with %d bits", bits)
			}
			out[c][i] = v
		}
	}
	return out, nil
}

// writeWAV writes [channel][sample] audio as 32-bit float, or as clipped
// 16-bit PCM when int16 is set.
func writeWAV(path string, audio [][]float32, rate int, int16Out bool) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	channels := len(audio)
	frames := len(audio[0])
	format, bits := formatFloat, 32
	if int16Out {
		format, bits = formatPCM, 16
	}
	dataSize := frames * channels * bits / 8
	le := binary.LittleEndian
	hdr := make([]byte, 44)
	copy(hdr[0:], "RIFF")
	le.PutUint32(hdr[4:], uint32(36+dataSize))
	copy(hdr[8:], "WAVEfmt ")
	le.PutUint32(hdr[16:], 16)
	le.PutUint16(hdr[20:], uint16(format))
	le.PutUint16(hdr[22:], uint16(channels))
	le.PutUint32(hdr[24:], uint32(rate))
	le.PutUint32(hdr[28:], uint32(rate*channels*bits/8))
	le.PutUint16(hdr[32:], uint16(channels*bits/8))
	le.PutUint16(hdr[34:], uint16(bits))
	copy(hdr[36:], "data")
	le.PutUint32(hdr[40:], uint32(dataSize))
	w.Write(hdr)
	var b [4]byte
	for i := 0; i < frames; i++ {
		for c := 0; c < channels; c++ {
			v := audio[c][i]
			if int16Out {
				s := math.Round(float64(v) * (1 << 15))
				s = math.Max(-(1 << 15), math.Min(1<<15-1, s))
				le.PutUint16(b[:], uint16(int16(s)))
				w.Write(b[:2])
			} else {
				le.PutUint32(b[:], math.Float32bits(v))
				w.Write(b[:])
			}
		}
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
