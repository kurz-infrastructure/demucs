package demucs

import (
	"os"
	"testing"
)

// BenchmarkForward times one segment of a released model, given by
// DEMUCS_TEST_MODEL (see TestPretrained).
func BenchmarkForward(b *testing.B) {
	path := os.Getenv("DEMUCS_TEST_MODEL")
	if path == "" {
		b.Skip("set DEMUCS_TEST_MODEL to run")
	}
	m, err := loadTestModel(path)
	if err != nil {
		b.Fatal(err)
	}
	seg := m.Segment()
	mix := make([][]float32, m.AudioChannels)
	for c := range mix {
		mix[c] = make([]float32, seg)
		for i := range mix[c] {
			mix[c][i] = float32((i*7919+c*104729)%2001-1000) / 4000
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.models[0].forward(mix); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSgemm(b *testing.B) {
	const m, n, k = 512, 2688, 512
	x := make([]float32, m*k)
	y := make([]float32, k*n)
	z := make([]float32, m*n)
	for i := range x {
		x[i] = float32(i%7) - 3
	}
	for i := range y {
		y[i] = float32(i%5) - 2
	}
	b.SetBytes(2 * m * n * k) // reported MB/s is MFLOP/s
	for i := 0; i < b.N; i++ {
		sgemm(false, false, m, n, k, 1, x, k, y, n, 0, z, n)
	}
}
