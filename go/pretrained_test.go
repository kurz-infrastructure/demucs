package demucs

import (
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// TestPretrained compares against PyTorch on a released model. It needs
// files too large for the repository, so it only runs when both are given:
//
//	DEMUCS_TEST_MODEL      a model name (htdemucs: downloaded from the hub and cached)
//	                       or an htdemucs.safetensors from tools/export_safetensors.py
//	DEMUCS_TEST_REFERENCE  safetensors with "mix" [2, n] and "out" [4, 2, n], the
//	                       output of demucs.apply.apply_model with shifts=0,
//	                       split=True, overlap=0.25 on the normalized mix.
func TestPretrained(t *testing.T) {
	modelPath, refPath := os.Getenv("DEMUCS_TEST_MODEL"), os.Getenv("DEMUCS_TEST_REFERENCE")
	if modelPath == "" || refPath == "" {
		t.Skip("set DEMUCS_TEST_MODEL and DEMUCS_TEST_REFERENCE to run")
	}
	m, err := loadTestModel(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := readSafetensorsFile(refPath)
	if err != nil {
		t.Fatal(err)
	}
	mix := rows(ref.tensors["mix"])
	opts := DefaultOptions()
	opts.Shifts = 0
	start := time.Now()
	sep, err := m.Separate(mix, opts)
	if err != nil {
		t.Fatal(err)
	}
	elapsed := time.Since(start)
	t.Logf("backend %s: separated %.1fs of audio in %s", Backend(),
		float64(len(mix[0]))/float64(m.SampleRate), elapsed.Round(time.Millisecond))
	// Stems can be nearly silent on a short excerpt, so errors are measured
	// against the level of the mix rather than of each stem.
	var sq float64
	for _, v := range ref.tensors["mix"].Data {
		sq += float64(v) * float64(v)
	}
	mixRMS := math.Sqrt(sq / float64(len(ref.tensors["mix"].Data)))
	want := rows(ref.tensors["out"])
	for s, name := range m.Sources {
		var got, exp []float32
		for c := range sep.Stems[s] {
			got = append(got, sep.Stems[s][c]...)
			exp = append(exp, want[s*len(sep.Stems[s])+c]...)
		}
		assertCloseScale(t, name, got, exp, 1e-4, mixRMS)
	}
}

// loadTestModel takes DEMUCS_TEST_MODEL as a model name (fetched from the hub) or a file path.
func loadTestModel(nameOrPath string) (*Model, error) {
	if strings.ContainsAny(nameOrPath, "/.") {
		return Load(nameOrPath)
	}
	return Pretrained(nameOrPath, nil)
}
