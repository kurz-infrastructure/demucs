package demucs

import (
	"encoding/json"
	"math"
	"os"
	"strconv"
	"testing"
)

// The reference data is produced by tools/make_testdata.py from the PyTorch
// implementation, using a tiny randomly initialized HTDemucs.

// TestBackend lets CI assert which sgemm implementation was compiled in.
func TestBackend(t *testing.T) {
	t.Logf("backend: %s", Backend())
	if want := os.Getenv("DEMUCS_EXPECT_BACKEND"); want != "" && Backend() != want {
		t.Fatalf("built with the %s backend, want %s", Backend(), want)
	}
}

func loadReference(t *testing.T) *safetensorsFile {
	t.Helper()
	ref, err := readSafetensorsFile("testdata/reference.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	return ref
}

func loadTiny(t *testing.T, name string) *Model {
	t.Helper()
	m, err := Load("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// rows splits a [rows, n] tensor into per-row slices.
func rows(t *Tensor) [][]float32 {
	n := t.Shape[len(t.Shape)-1]
	var out [][]float32
	for i := 0; i < len(t.Data); i += n {
		out = append(out, t.Data[i:i+n])
	}
	return out
}

func flatten(x [][][]float32) []float32 {
	var out []float32
	for _, a := range x {
		for _, b := range a {
			out = append(out, b...)
		}
	}
	return out
}

// assertClose checks the max abs error relative to the reference RMS, which
// is robust to the float32 reduction order differences between backends.
func assertClose(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	assertCloseScale(t, name, got, want, tol, 0)
}

// assertCloseScale is assertClose with the tolerance relative to
// max(reference rms, floor), for references that can be almost silent.
func assertCloseScale(t *testing.T, name string, got, want []float32, tol, floor float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d values, want %d", name, len(got), len(want))
	}
	var maxErr, sq float64
	for i := range want {
		maxErr = math.Max(maxErr, math.Abs(float64(got[i]-want[i])))
		sq += float64(want[i]) * float64(want[i])
	}
	rms := math.Sqrt(sq / float64(len(want)))
	t.Logf("%s: max abs error %.3g, reference rms %.3g", name, maxErr, rms)
	if limit := tol * math.Max(rms, floor); maxErr > limit || math.IsNaN(maxErr) {
		t.Errorf("%s: max abs error %.3g exceeds %.3g (rms %.3g)", name, maxErr, limit, rms)
	}
}

func TestSTFT(t *testing.T) {
	ref := loadReference(t)
	x := ref.tensors["stft_input"].Data
	re, im := ref.tensors["stft_re"], ref.tensors["stft_im"]
	z := stft(x, 4096, 1024)
	frames, bins := re.Shape[2], re.Shape[1]
	if len(z) != frames || len(z[0]) != bins {
		t.Fatalf("stft shape [%d %d], want [%d %d]", len(z), len(z[0]), frames, bins)
	}
	var gotRe, gotIm []float32
	for f := 0; f < bins; f++ {
		for tt := 0; tt < frames; tt++ {
			gotRe = append(gotRe, real(z[tt][f]))
			gotIm = append(gotIm, imag(z[tt][f]))
		}
	}
	assertClose(t, "stft real", gotRe, re.Data, 1e-5)
	assertClose(t, "stft imag", gotIm, im.Data, 1e-5)

	wav := istft(z, 4096, 1024, len(x))
	assertClose(t, "istft", wav, ref.tensors["istft"].Data, 1e-5)
}

func TestForward(t *testing.T) {
	ref := loadReference(t)
	m := loadTiny(t, "tiny.safetensors")
	mix := rows(ref.tensors["mix"])
	seg := m.Segment()

	full := [][]float32{mix[0][:seg], mix[1][:seg]}
	out, err := m.models[0].forward(full)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "forward full segment", flatten(out), ref.tensors["forward_full"].Data, 1e-4)

	short, _ := strconv.Atoi(ref.metadata["forward_short_length"])
	out, err = m.models[0].forward([][]float32{mix[0][:short], mix[1][:short]})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "forward short segment", flatten(out), ref.tensors["forward_short"].Data, 1e-4)
}

func TestSeparateSplit(t *testing.T) {
	ref := loadReference(t)
	m := loadTiny(t, "tiny.safetensors")
	opts := DefaultOptions()
	opts.Shifts = 0
	sep, err := m.Separate(rows(ref.tensors["mix"]), opts)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "split", flatten(sep.Stems), ref.tensors["apply_split"].Data, 1e-4)
}

func TestSeparateShifts(t *testing.T) {
	ref := loadReference(t)
	m := loadTiny(t, "tiny.safetensors")
	var offsets []int
	if err := json.Unmarshal([]byte(ref.metadata["shift_offsets"]), &offsets); err != nil {
		t.Fatal(err)
	}
	mix := rows(ref.tensors["mix"])
	opts := DefaultOptions()
	opts.Shifts = len(offsets)
	// Run the runner directly to substitute the reference shift offsets.
	r := &runner{model: m, opts: *opts}
	r.shiftOffset = func(int) int {
		o := offsets[0]
		offsets = offsets[1:]
		return o
	}
	r.prepare(len(mix[0]))
	norm, scale, mean := normalizeLikeAPI(mix)
	out, err := r.applyBag(norm)
	if err != nil {
		t.Fatal(err)
	}
	got := flatten(out)
	for i := range got {
		got[i] = got[i]*scale + mean
	}
	assertClose(t, "shifts", got, ref.tensors["apply_shifts"].Data, 1e-4)
}

func normalizeLikeAPI(mix [][]float32) ([][]float32, float32, float32) {
	ref := make([]float32, len(mix[0]))
	for _, ch := range mix {
		for i, v := range ch {
			ref[i] += v / float32(len(mix))
		}
	}
	mean, std := meanStd(ref)
	scale := std + 1e-8
	out := make([][]float32, len(mix))
	for c, ch := range mix {
		out[c] = make([]float32, len(ch))
		for i, v := range ch {
			out[c][i] = (v - mean) / scale
		}
	}
	return out, scale, mean
}

func TestSeparateBag(t *testing.T) {
	ref := loadReference(t)
	m := loadTiny(t, "tiny_bag.safetensors")
	if len(m.models) != 2 {
		t.Fatalf("got %d models, want 2", len(m.models))
	}
	opts := DefaultOptions()
	opts.Shifts = 0
	sep, err := m.Separate(rows(ref.tensors["mix"]), opts)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "bag", flatten(sep.Stems), ref.tensors["apply_bag"].Data, 1e-4)
}

func TestSeparateAPI(t *testing.T) {
	m := loadTiny(t, "tiny.safetensors")
	if got := m.Sources; len(got) != 2 || got[0] != "drums" || got[1] != "vocals" {
		t.Fatalf("sources %v", got)
	}
	mono := make([]float32, m.Segment()/2)
	for i := range mono {
		mono[i] = float32(math.Sin(float64(i) * 0.01))
	}
	var calls int
	opts := DefaultOptions()
	opts.Progress = func(done, total int) { calls++ }
	sep, err := m.Separate([][]float32{mono}, opts)
	if err != nil {
		t.Fatal(err)
	}
	if calls == 0 {
		t.Error("progress callback never called")
	}
	vocals := sep.Stem("vocals")
	if len(vocals) != 2 || len(vocals[0]) != len(mono) {
		t.Fatalf("vocals stem has shape [%d %d]", len(vocals), len(vocals[0]))
	}
	if sep.Stem("guitar") != nil {
		t.Error("unknown stem should be nil")
	}
	if got := sep.Mono("drums"); len(got) != len(mono) {
		t.Errorf("mono drums has %d samples", len(got))
	}
	// Same seed, same result.
	again, err := m.Separate([][]float32{mono}, DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "determinism", flatten(again.Stems), flatten(sep.Stems), 0)
}

func TestFloat16(t *testing.T) {
	cases := map[uint16]float32{
		0x0000: 0, 0x3c00: 1, 0xc000: -2, 0x3555: 0.333251953125,
		0x0001: 5.960464477539063e-08, 0x03ff: 6.097555160522461e-05, 0x7bff: 65504,
	}
	for h, want := range cases {
		if got := float16to32(h); got != want {
			t.Errorf("float16to32(%#04x) = %g, want %g", h, got, want)
		}
	}
	if !math.IsInf(float64(float16to32(0x7c00)), 1) {
		t.Error("0x7c00 should be +Inf")
	}
}

func TestSgemm(t *testing.T) {
	// Compare every transpose combination against a naive product.
	m, n, k := 37, 29, 53
	a := make([]float32, m*k)
	b := make([]float32, k*n)
	for i := range a {
		a[i] = float32(math.Sin(float64(i)))
	}
	for i := range b {
		b[i] = float32(math.Cos(float64(i)))
	}
	want := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for p := 0; p < k; p++ {
				s += float64(a[i*k+p]) * float64(b[p*n+j])
			}
			want[i*n+j] = float32(2*s) + 0.5
		}
	}
	at := transposeNaive(a, m, k)
	bt := transposeNaive(b, k, n)
	for _, tc := range []struct {
		ta, tb bool
		a, b   []float32
		lda    int
		ldb    int
	}{
		{false, false, a, b, k, n},
		{true, false, at, b, m, n},
		{false, true, a, bt, k, k},
		{true, true, at, bt, m, k},
	} {
		c := make([]float32, m*n)
		for i := range c {
			c[i] = 1
		}
		sgemm(tc.ta, tc.tb, m, n, k, 2, tc.a, tc.lda, tc.b, tc.ldb, 0.5, c, n)
		assertClose(t, "sgemm", c, want, 1e-5)
	}
}

func transposeNaive(x []float32, r, c int) []float32 {
	out := make([]float32, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out[j*r+i] = x[i*c+j]
		}
	}
	return out
}

func TestExpf(t *testing.T) {
	var worst float64
	for x := float32(-87); x < 88; x += 0.0137 {
		want := math.Exp(float64(x))
		rel := math.Abs(float64(expf(x))-want) / want
		worst = math.Max(worst, rel)
	}
	if worst > 3e-7 {
		t.Errorf("expf max relative error %.3g", worst)
	}
	if expf(-100) != 0 || !math.IsInf(float64(expf(100)), 1) {
		t.Error("expf does not saturate")
	}
}
