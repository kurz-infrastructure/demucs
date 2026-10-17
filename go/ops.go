package demucs

import (
	"fmt"
	"math"
)

// act is a [C, H, W] activation. The frequency branch uses H for frequencies
// and W for time frames; the time branch and 1D convolutions use H == 1.
type act struct {
	data    []float32
	c, h, w int
}

func newAct(c, h, w int) act {
	return act{data: make([]float32, c*h*w), c: c, h: h, w: w}
}

func (a act) plane() int { return a.h * a.w }

// convParams describes a 2D convolution along (H, W). Conv1d layers are
// expressed as 1×K kernels acting on W.
type convParams struct {
	strideH, strideW int
	padH, padW       int
	dilH, dilW       int
}

func conv1dParams(stride, pad, dil int) convParams {
	return convParams{strideH: 1, strideW: stride, padW: pad, dilH: 1, dilW: dil}
}

// kernelDims returns (out, in, kh, kw) for a Conv1d [O, C, K] or Conv2d
// [O, C, KH, KW] weight.
func kernelDims(w *Tensor) (int, int, int, int) {
	switch len(w.Shape) {
	case 3:
		return w.Shape[0], w.Shape[1], 1, w.Shape[2]
	case 4:
		return w.Shape[0], w.Shape[1], w.Shape[2], w.Shape[3]
	}
	panic(fmt.Sprintf("demucs: unexpected conv weight shape %v", w.Shape))
}

// im2colTarget bounds the scratch matrix used by conv2d, in floats.
const im2colTarget = 1 << 22

// conv2d applies a convolution with weight [O, C, KH, KW] (or [O, C, K]) and an
// optional bias, via im2col and sgemm over tiles of output positions.
func conv2d(x act, w, bias *Tensor, p convParams) act {
	o, cin, kh, kw := kernelDims(w)
	if cin != x.c {
		panic(fmt.Sprintf("demucs: conv expects %d input channels, got %d", cin, x.c))
	}
	ho := (x.h+2*p.padH-p.dilH*(kh-1)-1)/p.strideH + 1
	wo := (x.w+2*p.padW-p.dilW*(kw-1)-1)/p.strideW + 1
	out := newAct(o, ho, wo)
	n := ho * wo
	ck := cin * kh * kw
	if kh == 1 && kw == 1 && p.strideH == 1 && p.strideW == 1 && p.padH == 0 && p.padW == 0 {
		sgemm(false, false, o, n, ck, 1, w.Data, ck, x.data, n, 0, out.data, n)
	} else {
		tile := max(256, im2colTarget/ck)
		tile = min(tile, n)
		cols := make([]float32, ck*tile)
		for p0 := 0; p0 < n; p0 += tile {
			p1 := min(p0+tile, n)
			tl := p1 - p0
			im2col(x, kh, kw, p, wo, p0, p1, cols)
			sgemm(false, false, o, tl, ck, 1, w.Data, ck, cols, tl, 0, out.data[p0:], n)
		}
	}
	addBias(out, bias)
	return out
}

// im2col fills cols [C*KH*KW, p1-p0] for output positions [p0, p1).
func im2col(x act, kh, kw int, p convParams, wo, p0, p1 int, cols []float32) {
	tl := p1 - p0
	rows := x.c * kh * kw
	parallelFor(rows, workersFor(rows*tl), func(lo, hi int) {
		for r := lo; r < hi; r++ {
			c := r / (kh * kw)
			i := (r / kw) % kh
			j := r % kw
			src := x.data[c*x.plane() : (c+1)*x.plane()]
			dst := cols[r*tl : (r+1)*tl]
			for q := range dst {
				pos := p0 + q
				oh := pos / wo
				ow := pos - oh*wo
				ih := oh*p.strideH - p.padH + i*p.dilH
				iw := ow*p.strideW - p.padW + j*p.dilW
				if ih < 0 || ih >= x.h || iw < 0 || iw >= x.w {
					dst[q] = 0
				} else {
					dst[q] = src[ih*x.w+iw]
				}
			}
		}
	})
}

// convTranspose2d applies a transposed convolution with weight
// [Cin, Cout, KH, KW] (or [Cin, Cout, K]), stride (sh, sw), no padding.
func convTranspose2d(x act, w, bias *Tensor, sh, sw int) act {
	cin, cout, kh, kw := kernelDims(w)
	if cin != x.c {
		panic(fmt.Sprintf("demucs: conv_tr expects %d input channels, got %d", cin, x.c))
	}
	n := x.plane()
	m := cout * kh * kw
	cols := make([]float32, m*n)
	sgemm(true, false, m, n, cin, 1, w.Data, m, x.data, n, 0, cols, n)
	ho := (x.h-1)*sh + kh
	wo := (x.w-1)*sw + kw
	out := newAct(cout, ho, wo)
	parallelFor(cout, workersFor(m*n), func(lo, hi int) {
		for co := lo; co < hi; co++ {
			dst := out.data[co*ho*wo : (co+1)*ho*wo]
			for i := 0; i < kh; i++ {
				for j := 0; j < kw; j++ {
					src := cols[((co*kh+i)*kw+j)*n:]
					for h := 0; h < x.h; h++ {
						row := dst[(h*sh+i)*wo+j:]
						s := src[h*x.w : (h+1)*x.w]
						for t, v := range s {
							row[t*sw] += v
						}
					}
				}
			}
		}
	})
	addBias(out, bias)
	return out
}

func addBias(x act, bias *Tensor) {
	if bias == nil {
		return
	}
	pl := x.plane()
	for c := 0; c < x.c; c++ {
		b := bias.Data[c]
		row := x.data[c*pl : (c+1)*pl]
		for i := range row {
			row[i] += b
		}
	}
}

// groupNorm normalizes over channel groups and all of (H, W), like
// nn.GroupNorm applied to a [1, C, H, W] tensor.
func groupNorm(x act, groups int, weight, bias *Tensor, eps float64) {
	cg := x.c / groups
	pl := x.plane()
	for g := 0; g < groups; g++ {
		seg := x.data[g*cg*pl : (g+1)*cg*pl]
		mean, inv := meanInvStd(seg, eps)
		for c := g * cg; c < (g+1)*cg; c++ {
			scale := float32(inv) * weight.Data[c]
			shift := bias.Data[c] - float32(mean)*scale
			row := x.data[c*pl : (c+1)*pl]
			for i, v := range row {
				row[i] = v*scale + shift
			}
		}
	}
}

// groupNormRows applies GroupNorm(1, C) independently to every H row, which is
// what the reference does when folding frequencies into the batch dimension.
func groupNormRows(x act, weight, bias *Tensor, eps float64) {
	if x.h == 1 {
		groupNorm(x, 1, weight, bias, eps)
		return
	}
	pl := x.plane()
	parallelFor(x.h, workersFor(len(x.data)), func(lo, hi int) {
		for h := lo; h < hi; h++ {
			var sum, sq float64
			for c := 0; c < x.c; c++ {
				row := x.data[c*pl+h*x.w : c*pl+(h+1)*x.w]
				for _, v := range row {
					sum += float64(v)
				}
			}
			cnt := float64(x.c * x.w)
			mean := sum / cnt
			for c := 0; c < x.c; c++ {
				row := x.data[c*pl+h*x.w : c*pl+(h+1)*x.w]
				for _, v := range row {
					d := float64(v) - mean
					sq += d * d
				}
			}
			inv := 1 / math.Sqrt(sq/cnt+eps)
			for c := 0; c < x.c; c++ {
				scale := float32(inv) * weight.Data[c]
				shift := bias.Data[c] - float32(mean)*scale
				row := x.data[c*pl+h*x.w : c*pl+(h+1)*x.w]
				for i, v := range row {
					row[i] = v*scale + shift
				}
			}
		}
	})
}

// meanInvStd returns the mean and 1/sqrt(biased variance + eps) of x.
func meanInvStd(x []float32, eps float64) (float64, float64) {
	var sum float64
	for _, v := range x {
		sum += float64(v)
	}
	mean := sum / float64(len(x))
	var sq float64
	for _, v := range x {
		d := float64(v) - mean
		sq += d * d
	}
	return mean, 1 / math.Sqrt(sq/float64(len(x))+eps)
}

// meanStd returns the mean and the unbiased standard deviation of x, as
// torch.Tensor.std does by default.
func meanStd(x []float32) (float32, float32) {
	var sum float64
	for _, v := range x {
		sum += float64(v)
	}
	mean := sum / float64(len(x))
	var sq float64
	for _, v := range x {
		d := float64(v) - mean
		sq += d * d
	}
	return float32(mean), float32(math.Sqrt(sq / float64(len(x)-1)))
}

// layerNorm normalizes each row of an [n, c] matrix in place.
func layerNorm(x []float32, n, c int, weight, bias *Tensor, eps float64) {
	parallelFor(n, workersFor(n*c), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			row := x[i*c : (i+1)*c]
			mean, inv := meanInvStd(row, eps)
			for j, v := range row {
				row[j] = float32((float64(v)-mean)*inv)*weight.Data[j] + bias.Data[j]
			}
		}
	})
}

// layerNormTo is layerNorm writing into dst instead of in place.
func layerNormTo(dst, x []float32, n, c int, weight, bias *Tensor, eps float64) {
	copy(dst, x)
	layerNorm(dst, n, c, weight, bias, eps)
}

// groupNormTokens applies GroupNorm(1, C) to a [n, c] token matrix, i.e. a
// single normalization over all tokens and channels with a per-channel affine.
func groupNormTokens(x []float32, n, c int, weight, bias *Tensor, eps float64) {
	mean, inv := meanInvStd(x, eps)
	parallelFor(n, workersFor(n*c), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			row := x[i*c : (i+1)*c]
			for j, v := range row {
				row[j] = float32((float64(v)-mean)*inv)*weight.Data[j] + bias.Data[j]
			}
		}
	})
}

// gelu applies the exact (erf based) GELU in place, as F.gelu does.
func gelu(x []float32) {
	parallelElems(len(x), func(lo, hi int) {
		for i, v := range x[lo:hi] {
			x[lo+i] = float32(0.5 * float64(v) * (1 + math.Erf(float64(v)/math.Sqrt2)))
		}
	})
}

// glu splits channels in two halves a, b and returns a * sigmoid(b).
func glu(x act) act {
	half := x.c / 2
	out := newAct(half, x.h, x.w)
	n := half * x.plane()
	a, b := x.data[:n], x.data[n:2*n]
	parallelElems(n, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			out.data[i] = a[i] * sigmoid(b[i])
		}
	})
	return out
}

func sigmoid(v float32) float32 {
	return 1 / (1 + expf(-v))
}

// linear computes y = x W^T + b for x [n, in] and W [out, in].
func linear(x []float32, n int, w, b *Tensor) []float32 {
	out, in := w.Shape[0], w.Shape[1]
	y := make([]float32, n*out)
	sgemm(false, true, n, out, in, 1, x, in, w.Data, in, 0, y, out)
	if b != nil {
		for i := 0; i < n; i++ {
			row := y[i*out : (i+1)*out]
			for j := range row {
				row[j] += b.Data[j]
			}
		}
	}
	return y
}

func addInPlace(dst, src []float32) {
	for i, v := range src {
		dst[i] += v
	}
}

// workersFor picks how many goroutines to use for a job touching n elements.
func workersFor(n int) int {
	if n < 1<<14 {
		return 1
	}
	return numWorkers()
}
