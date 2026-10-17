package main

import (
	"math"
	"path/filepath"
	"testing"
)

func TestWAVRoundTrip(t *testing.T) {
	audio := [][]float32{make([]float32, 1000), make([]float32, 1000)}
	for i := range audio[0] {
		audio[0][i] = float32(math.Sin(float64(i) * 0.05))
		audio[1][i] = -0.5 * audio[0][i]
	}
	dir := t.TempDir()
	for _, int16Out := range []bool{false, true} {
		path := filepath.Join(dir, "x.wav")
		if err := writeWAV(path, audio, 44100, int16Out); err != nil {
			t.Fatal(err)
		}
		got, rate, err := readWAV(path)
		if err != nil {
			t.Fatal(err)
		}
		if rate != 44100 || len(got) != 2 || len(got[0]) != 1000 {
			t.Fatalf("int16=%v: rate %d, shape [%d %d]", int16Out, rate, len(got), len(got[0]))
		}
		tol := 0.0
		if int16Out {
			tol = 1.0 / (1 << 15)
		}
		for c := range audio {
			for i, v := range audio[c] {
				if d := math.Abs(float64(got[c][i] - v)); d > tol {
					t.Fatalf("int16=%v: sample %d/%d off by %g", int16Out, c, i, d)
				}
			}
		}
	}
}
