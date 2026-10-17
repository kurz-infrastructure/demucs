package demucs

import (
	"math"
)

// Resample converts audio between integer sample rates. It is a port of
// julius.resample_frac, which is what demucs.audio.convert_audio uses, with
// the same defaults (24 zero crossings, 0.945 rolloff): a windowed sinc,
// polyphase per output phase, replicate padding at the edges and an output
// of floor(len(x) * to / from) samples. Equal rates return x unchanged.
func Resample(x []float32, from, to int) []float32 {
	if from <= 0 || to <= 0 {
		panic("demucs: sample rates must be positive")
	}
	if from == to || len(x) == 0 {
		return x
	}
	g := gcd(from, to)
	old, nw := from/g, to/g
	const zeros, rolloff = 24, 0.945
	sr := float64(min(old, nw)) * rolloff
	width := int(math.Ceil(float64(zeros) * float64(old) / sr))
	klen := 2*width + old
	kernels := make([][]float32, nw)
	for i := range kernels {
		k := make([]float64, klen)
		var sum float64
		for j := 0; j < klen; j++ {
			idx := float64(j - width)
			t := (-float64(i)/float64(nw) + idx/float64(old)) * sr
			t = math.Max(-zeros, math.Min(zeros, t)) * math.Pi
			w := math.Cos(t / zeros / 2)
			k[j] = sinc(t) * w * w
			sum += k[j]
		}
		kernels[i] = make([]float32, klen)
		for j := range k {
			kernels[i][j] = float32(k[j] / sum)
		}
	}
	// replicate padding: width on the left, width+old on the right
	padded := make([]float32, width+len(x)+width+old)
	for i := 0; i < width; i++ {
		padded[i] = x[0]
	}
	copy(padded[width:], x)
	for i := width + len(x); i < len(padded); i++ {
		padded[i] = x[len(x)-1]
	}
	frames := (len(padded)-klen)/old + 1
	outLen := len(x) * nw / old // floor, like julius
	out := make([]float32, 0, frames*nw)
	for f := 0; f < frames; f++ {
		seg := padded[f*old : f*old+klen]
		for i := 0; i < nw; i++ {
			var acc float32
			k := kernels[i]
			for j, v := range seg {
				acc += v * k[j]
			}
			out = append(out, acc)
		}
	}
	if len(out) > outLen {
		out = out[:outLen]
	}
	return out
}

// ResampleChannels applies Resample to every channel.
func ResampleChannels(x [][]float32, from, to int) [][]float32 {
	out := make([][]float32, len(x))
	for c := range x {
		out[c] = Resample(x[c], from, to)
	}
	return out
}

func sinc(t float64) float64 {
	if t == 0 {
		return 1
	}
	return math.Sin(t) / t
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}
