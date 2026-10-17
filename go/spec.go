package demucs

import (
	"math"
	"math/bits"
	"math/cmplx"
	"sync"
)

// fftPlan holds twiddles and the bit reversal permutation for a radix-2 FFT.
type fftPlan struct {
	n       int
	twiddle []complex128
	rev     []int
	window  []float64 // periodic Hann window of length n
}

var (
	fftPlansMu sync.Mutex
	fftPlans   = map[int]*fftPlan{}
)

func getFFTPlan(n int) *fftPlan {
	fftPlansMu.Lock()
	defer fftPlansMu.Unlock()
	if p, ok := fftPlans[n]; ok {
		return p
	}
	if n&(n-1) != 0 {
		panic("demucs: FFT size must be a power of two")
	}
	p := &fftPlan{n: n, twiddle: make([]complex128, n/2), rev: make([]int, n), window: make([]float64, n)}
	for k := range p.twiddle {
		p.twiddle[k] = cmplx.Exp(complex(0, -2*math.Pi*float64(k)/float64(n)))
	}
	shift := 64 - bits.Len(uint(n)) + 1
	for i := range p.rev {
		p.rev[i] = int(bits.Reverse64(uint64(i)) >> shift)
	}
	for i := range p.window {
		p.window[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n))
	}
	fftPlans[n] = p
	return p
}

// transform runs an in-place forward FFT, or an unnormalized inverse one.
func (p *fftPlan) transform(x []complex128, inverse bool) {
	n := p.n
	for i, j := range p.rev {
		if i < j {
			x[i], x[j] = x[j], x[i]
		}
	}
	for size := 2; size <= n; size <<= 1 {
		half := size / 2
		step := n / size
		for start := 0; start < n; start += size {
			for k := 0; k < half; k++ {
				w := p.twiddle[k*step]
				if inverse {
					w = cmplx.Conj(w)
				}
				a := x[start+k]
				b := x[start+k+half] * w
				x[start+k] = a + b
				x[start+k+half] = a - b
			}
		}
	}
}

// reflectPad pads x on both sides with reflection (no edge repeat), like
// F.pad(mode="reflect"). Pads larger than the signal fall back to zero
// padding the excess first, as the reference `pad1d` helper does.
func reflectPad(x []float32, left, right int) []float32 {
	length := len(x)
	maxPad := max(left, right)
	if length <= maxPad {
		extra := maxPad - length + 1
		extraRight := min(right, extra)
		extraLeft := extra - extraRight
		padded := make([]float32, length+extra)
		copy(padded[extraLeft:], x)
		return reflectPad(padded, left-extraLeft, right-extraRight)
	}
	out := make([]float32, left+length+right)
	copy(out[left:], x)
	for i := 0; i < left; i++ {
		out[left-1-i] = x[i+1]
	}
	for i := 0; i < right; i++ {
		out[left+length+i] = x[length-2-i]
	}
	return out
}

// stft computes torch.stft(x, nfft, hop, window=hann(nfft), normalized=True,
// center=True, pad_mode="reflect", return_complex=True). The result is
// [frames][nfft/2+1].
func stft(x []float32, nfft, hop int) [][]complex64 {
	p := getFFTPlan(nfft)
	padded := reflectPad(x, nfft/2, nfft/2)
	frames := (len(padded)-nfft)/hop + 1
	bins := nfft/2 + 1
	norm := 1 / math.Sqrt(float64(nfft))
	out := make([][]complex64, frames)
	parallelFor(frames, workersFor(frames*nfft), func(lo, hi int) {
		buf := make([]complex128, nfft)
		for f := lo; f < hi; f++ {
			seg := padded[f*hop : f*hop+nfft]
			for i, v := range seg {
				buf[i] = complex(float64(v)*p.window[i], 0)
			}
			p.transform(buf, false)
			row := make([]complex64, bins)
			for k := range row {
				row[k] = complex64(buf[k] * complex(norm, 0))
			}
			out[f] = row
		}
	})
	return out
}

// istft inverts stft: torch.istft(z, nfft, hop, window=hann(nfft),
// normalized=True, center=True, length=length).
func istft(z [][]complex64, nfft, hop, length int) []float32 {
	p := getFFTPlan(nfft)
	frames := len(z)
	total := nfft + hop*(frames-1)
	signal := make([]float64, total)
	envelope := make([]float64, total)
	scale := math.Sqrt(float64(nfft)) / float64(nfft)
	frameOut := make([][]float64, frames)
	parallelFor(frames, workersFor(frames*nfft), func(lo, hi int) {
		buf := make([]complex128, nfft)
		for f := lo; f < hi; f++ {
			row := z[f]
			// Rebuild the Hermitian spectrum; like irfft, the imaginary parts
			// of the DC and Nyquist bins are ignored.
			buf[0] = complex(real(complex128(row[0])), 0)
			for k := 1; k < nfft/2; k++ {
				v := complex128(row[k])
				buf[k] = v
				buf[nfft-k] = cmplx.Conj(v)
			}
			buf[nfft/2] = complex(real(complex128(row[nfft/2])), 0)
			p.transform(buf, true)
			out := make([]float64, nfft)
			for i := range out {
				out[i] = real(buf[i]) * scale * p.window[i]
			}
			frameOut[f] = out
		}
	})
	for f, out := range frameOut {
		base := f * hop
		for i, v := range out {
			signal[base+i] += v
			envelope[base+i] += p.window[i] * p.window[i]
		}
	}
	start := nfft / 2
	res := make([]float32, length)
	for i := range res {
		j := start + i
		if j >= total {
			break
		}
		if envelope[j] > 1e-11 {
			res[i] = float32(signal[j] / envelope[j])
		}
	}
	return res
}
