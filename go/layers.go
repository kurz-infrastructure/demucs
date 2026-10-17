package demucs

import "fmt"

const normEps = 1e-5

// norm is the affine part of a GroupNorm/LayerNorm.
type norm struct{ w, b *Tensor }

func loadNorm(w *weights, name string) (*norm, error) {
	wt, err := w.get(name + ".weight")
	if err != nil {
		return nil, err
	}
	bt, err := w.get(name + ".bias")
	if err != nil {
		return nil, err
	}
	return &norm{wt, bt}, nil
}

// conv is a Conv1d/Conv2d or ConvTranspose weight with its bias.
type conv struct{ w, b *Tensor }

func loadConv(w *weights, name string) (conv, error) {
	wt, err := w.get(name + ".weight")
	if err != nil {
		return conv{}, err
	}
	bt, err := w.get(name + ".bias")
	if err != nil {
		return conv{}, err
	}
	return conv{wt, bt}, nil
}

// dconvLayer is one residual block of DConv:
// conv(k=3, dilated) → GroupNorm(1) → GELU → conv(1×1) → GroupNorm(1) → GLU → LayerScale.
type dconvLayer struct {
	conv1, conv2 conv
	norm1, norm2 *norm
	scale        *Tensor
	dilation     int
	kernel       int
}

type dconv struct{ layers []dconvLayer }

func loadDConv(w *weights, name string, depth int) (*dconv, error) {
	d := &dconv{}
	dilate := depth > 0
	if depth < 0 {
		depth = -depth
	}
	for i := 0; i < depth; i++ {
		p := fmt.Sprintf("%s.layers.%d", name, i)
		var l dconvLayer
		var err error
		if l.conv1, err = loadConv(w, p+".0"); err != nil {
			return nil, err
		}
		if l.norm1, err = loadNorm(w, p+".1"); err != nil {
			return nil, err
		}
		if l.conv2, err = loadConv(w, p+".3"); err != nil {
			return nil, err
		}
		if l.norm2, err = loadNorm(w, p+".4"); err != nil {
			return nil, err
		}
		if l.scale, err = w.get(p + ".6.scale"); err != nil {
			return nil, err
		}
		l.dilation = 1
		if dilate {
			l.dilation = 1 << i
		}
		l.kernel = l.conv1.w.Shape[2]
		d.layers = append(d.layers, l)
	}
	return d, nil
}

// forward runs the residual branches. On the frequency branch (x.h > 1),
// every frequency row is an independent sequence, as in the reference which
// folds frequencies into the batch dimension.
func (d *dconv) forward(x act) {
	for _, l := range d.layers {
		pad := l.dilation * (l.kernel / 2)
		y := conv2d(x, l.conv1.w, l.conv1.b, conv1dParams(1, pad, l.dilation))
		groupNormRows(y, l.norm1.w, l.norm1.b, normEps)
		gelu(y.data)
		y = conv2d(y, l.conv2.w, l.conv2.b, conv1dParams(1, 0, 1))
		groupNormRows(y, l.norm2.w, l.norm2.b, normEps)
		y = glu(y)
		pl := y.plane()
		for c := 0; c < y.c; c++ {
			s := l.scale.Data[c]
			dst := x.data[c*pl : (c+1)*pl]
			for i, v := range y.data[c*pl : (c+1)*pl] {
				dst[i] += s * v
			}
		}
	}
}

// encLayer is HEncLayer.
type encLayer struct {
	freq, empty    bool
	kernel, stride int
	pad            int
	norm           bool
	normGroups     int
	conv           conv
	norm1, norm2   *norm
	rewrite        *conv
	dconv          *dconv
}

// decLayer is HDecLayer.
type decLayer struct {
	freq, empty, last bool
	kernel, stride    int
	pad               int
	chin              int
	norm              bool
	normGroups        int
	convTr            conv
	norm1, norm2      *norm
	rewrite           *conv
	dconv             *dconv
}

func (l *encLayer) forward(x act, inject *act) act {
	if !l.freq && x.h > 1 {
		x = act{data: x.data, c: x.c * x.h, h: 1, w: x.w}
	}
	var y act
	if l.freq {
		y = conv2d(x, l.conv.w, l.conv.b, convParams{
			strideH: l.stride, strideW: 1, padH: l.pad, dilH: 1, dilW: 1})
	} else {
		if r := x.w % l.stride; r != 0 {
			x = padRight(x, l.stride-r)
		}
		y = conv2d(x, l.conv.w, l.conv.b, conv1dParams(l.stride, l.pad, 1))
	}
	if l.empty {
		return y
	}
	if inject != nil {
		if inject.w != y.w {
			panic(fmt.Sprintf("demucs: inject length %d != %d", inject.w, y.w))
		}
		addBroadcastH(y, *inject)
	}
	if l.norm1 != nil {
		groupNorm(y, l.normGroups, l.norm1.w, l.norm1.b, normEps)
	}
	gelu(y.data)
	if l.dconv != nil {
		l.dconv.forward(y)
	}
	if l.rewrite == nil {
		return y
	}
	z := conv2d(y, l.rewrite.w, l.rewrite.b, samePadding(l.rewrite.w))
	if l.norm2 != nil {
		groupNorm(z, l.normGroups, l.norm2.w, l.norm2.b, normEps)
	}
	return glu(z)
}

// forward returns the layer output and the input of the final transposed
// convolution, which the time decoder consumes when the branches split.
func (l *decLayer) forward(x act, skip *act, length int) (act, act) {
	if l.freq && x.c != l.chin {
		x = act{data: x.data, c: l.chin, h: x.c * x.h / l.chin, w: x.w}
	}
	var y act
	if !l.empty {
		addInPlace(x.data, skip.data)
		y = x
		if l.rewrite != nil {
			r := conv2d(x, l.rewrite.w, l.rewrite.b, samePadding(l.rewrite.w))
			if l.norm1 != nil {
				groupNorm(r, l.normGroups, l.norm1.w, l.norm1.b, normEps)
			}
			y = glu(r)
		}
		if l.dconv != nil {
			l.dconv.forward(y)
		}
	} else {
		y = x
	}
	var z act
	if l.freq {
		z = convTranspose2d(y, l.convTr.w, l.convTr.b, l.stride, 1)
	} else {
		z = convTranspose2d(y, l.convTr.w, l.convTr.b, 1, l.stride)
	}
	if l.norm2 != nil {
		groupNorm(z, l.normGroups, l.norm2.w, l.norm2.b, normEps)
	}
	if l.freq {
		if l.pad > 0 {
			z = cropH(z, l.pad, z.h-l.pad)
		}
	} else {
		z = cropW(z, l.pad, l.pad+length)
	}
	if !l.last {
		gelu(z.data)
	}
	return z, y
}

// samePadding returns the padding of a stride 1 conv with odd kernel that
// keeps the spatial size, as used by the `rewrite` convolutions.
func samePadding(w *Tensor) convParams {
	_, _, kh, kw := kernelDims(w)
	return convParams{strideH: 1, strideW: 1, padH: (kh - 1) / 2, padW: (kw - 1) / 2, dilH: 1, dilW: 1}
}

func padRight(x act, n int) act {
	out := newAct(x.c, x.h, x.w+n)
	for c := 0; c < x.c; c++ {
		for h := 0; h < x.h; h++ {
			copy(out.data[(c*x.h+h)*out.w:], x.data[(c*x.h+h)*x.w:(c*x.h+h+1)*x.w])
		}
	}
	return out
}

func cropW(x act, lo, hi int) act {
	out := newAct(x.c, x.h, hi-lo)
	for c := 0; c < x.c; c++ {
		for h := 0; h < x.h; h++ {
			row := x.data[(c*x.h+h)*x.w:]
			copy(out.data[(c*x.h+h)*out.w:(c*x.h+h+1)*out.w], row[lo:hi])
		}
	}
	return out
}

func cropH(x act, lo, hi int) act {
	out := newAct(x.c, hi-lo, x.w)
	for c := 0; c < x.c; c++ {
		copy(out.data[c*out.plane():(c+1)*out.plane()], x.data[c*x.plane()+lo*x.w:c*x.plane()+hi*x.w])
	}
	return out
}

// addBroadcastH adds src [C, 1, W] (or [C, H, W]) to dst [C, H, W].
func addBroadcastH(dst, src act) {
	for c := 0; c < dst.c; c++ {
		for h := 0; h < dst.h; h++ {
			sh := 0
			if src.h > 1 {
				sh = h
			}
			d := dst.data[(c*dst.h+h)*dst.w : (c*dst.h+h+1)*dst.w]
			s := src.data[(c*src.h+sh)*src.w : (c*src.h+sh+1)*src.w]
			addInPlace(d, s)
		}
	}
}
