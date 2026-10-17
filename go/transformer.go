package demucs

import (
	"fmt"
	"math"
)

// tLayer is either MyTransformerEncoderLayer (self attention) or
// CrossTransformerEncoderLayer (cross attention), both with batch_first.
type tLayer struct {
	cross     bool
	normFirst bool
	gelu      bool
	heads     int

	inProjW, inProjB  *Tensor
	outProjW, outProj *Tensor
	lin1W, lin1B      *Tensor
	lin2W, lin2B      *Tensor
	norm1, norm2      *norm
	norm3             *norm // cross attention only, norm applied before the FFN
	normOut           *norm
	gamma1, gamma2    *Tensor
}

type crossTransformer struct {
	normIn, normInT *norm
	layers, layersT []*tLayer
	classicParity   int
	maxPeriod       float64
	weightPosEmbed  float32
}

func loadTLayer(w *weights, name string, cross bool, cfg *config) (*tLayer, error) {
	l := &tLayer{cross: cross, normFirst: cfg.TNormFirst, gelu: cfg.TGelu, heads: cfg.THeads}
	attn := name + ".self_attn"
	if cross {
		attn = name + ".cross_attn"
	}
	var err error
	get := func(dst **Tensor, n string) {
		if err == nil {
			*dst, err = w.get(n)
		}
	}
	get(&l.inProjW, attn+".in_proj_weight")
	get(&l.inProjB, attn+".in_proj_bias")
	get(&l.outProjW, attn+".out_proj.weight")
	get(&l.outProj, attn+".out_proj.bias")
	get(&l.lin1W, name+".linear1.weight")
	get(&l.lin1B, name+".linear1.bias")
	get(&l.lin2W, name+".linear2.weight")
	get(&l.lin2B, name+".linear2.bias")
	if err != nil {
		return nil, err
	}
	if l.norm1, err = loadNorm(w, name+".norm1"); err != nil {
		return nil, err
	}
	if l.norm2, err = loadNorm(w, name+".norm2"); err != nil {
		return nil, err
	}
	if cross {
		if l.norm3, err = loadNorm(w, name+".norm3"); err != nil {
			return nil, err
		}
	}
	if cfg.TNormFirst && cfg.TNormOut {
		if l.normOut, err = loadNorm(w, name+".norm_out"); err != nil {
			return nil, err
		}
	}
	if cfg.TLayerScale {
		get(&l.gamma1, name+".gamma_1.scale")
		get(&l.gamma2, name+".gamma_2.scale")
	}
	return l, err
}

func loadCrossTransformer(w *weights, cfg *config) (*crossTransformer, error) {
	t := &crossTransformer{maxPeriod: cfg.TMaxPeriod, weightPosEmbed: float32(cfg.TWeightPosEmbed)}
	if cfg.TCrossFirst {
		t.classicParity = 1
	}
	var err error
	if cfg.TNormIn {
		if t.normIn, err = loadNorm(w, "crosstransformer.norm_in"); err != nil {
			return nil, err
		}
		if t.normInT, err = loadNorm(w, "crosstransformer.norm_in_t"); err != nil {
			return nil, err
		}
	}
	for i := 0; i < cfg.TLayers; i++ {
		cross := i%2 != t.classicParity
		l, err := loadTLayer(w, fmt.Sprintf("crosstransformer.layers.%d", i), cross, cfg)
		if err != nil {
			return nil, err
		}
		lt, err := loadTLayer(w, fmt.Sprintf("crosstransformer.layers_t.%d", i), cross, cfg)
		if err != nil {
			return nil, err
		}
		t.layers = append(t.layers, l)
		t.layersT = append(t.layersT, lt)
	}
	return t, nil
}

// forward takes the spectral branch x [C, Fr, T1] and the temporal branch
// xt [C, 1, T2] and returns both, transformed, in the same layouts.
func (t *crossTransformer) forward(x, xt act) (act, act) {
	c, fr, t1 := x.c, x.h, x.w
	t2 := xt.w
	// Spectral tokens are ordered (t1, fr): "b c fr t1 -> b (t1 fr) c".
	n1 := fr * t1
	tok := make([]float32, n1*c)
	for ch := 0; ch < c; ch++ {
		for f := 0; f < fr; f++ {
			row := x.data[(ch*fr+f)*t1:]
			for tt := 0; tt < t1; tt++ {
				tok[(tt*fr+f)*c+ch] = row[tt]
			}
		}
	}
	if t.normIn != nil {
		layerNorm(tok, n1, c, t.normIn.w, t.normIn.b, normEps)
	}
	pos2d := sin2DEmbedding(c, fr, t1, t.maxPeriod)
	for i, v := range pos2d {
		tok[i] += t.weightPosEmbed * v
	}

	tokT := make([]float32, t2*c)
	for ch := 0; ch < c; ch++ {
		for tt := 0; tt < t2; tt++ {
			tokT[tt*c+ch] = xt.data[ch*t2+tt]
		}
	}
	if t.normInT != nil {
		layerNorm(tokT, t2, c, t.normInT.w, t.normInT.b, normEps)
	}
	pos1d := sin1DEmbedding(t2, c, t.maxPeriod)
	for i, v := range pos1d {
		tokT[i] += t.weightPosEmbed * v
	}

	for i := range t.layers {
		if !t.layers[i].cross {
			tok = t.layers[i].forward(tok, n1, nil, 0, c)
			tokT = t.layersT[i].forward(tokT, t2, nil, 0, c)
		} else {
			old := tok
			tok = t.layers[i].forward(tok, n1, tokT, t2, c)
			tokT = t.layersT[i].forward(tokT, t2, old, n1, c)
		}
	}

	outX := newAct(c, fr, t1)
	for ch := 0; ch < c; ch++ {
		for f := 0; f < fr; f++ {
			row := outX.data[(ch*fr+f)*t1:]
			for tt := 0; tt < t1; tt++ {
				row[tt] = tok[(tt*fr+f)*c+ch]
			}
		}
	}
	outT := newAct(c, 1, t2)
	for ch := 0; ch < c; ch++ {
		for tt := 0; tt < t2; tt++ {
			outT.data[ch*t2+tt] = tokT[tt*c+ch]
		}
	}
	return outX, outT
}

// forward applies the layer to x [n, c]. For cross attention layers, kv
// [nk, c] holds the other branch.
func (l *tLayer) forward(x []float32, n int, kv []float32, nk int, c int) []float32 {
	x = append([]float32(nil), x...)
	if l.normFirst {
		h := make([]float32, len(x))
		layerNormTo(h, x, n, c, l.norm1.w, l.norm1.b, normEps)
		var a []float32
		if l.cross {
			k := make([]float32, len(kv))
			layerNormTo(k, kv, nk, c, l.norm2.w, l.norm2.b, normEps)
			a = l.attention(h, n, k, nk, c)
		} else {
			a = l.attention(h, n, h, n, c)
		}
		addScaled(x, a, l.gamma1, c)
		ffNorm := l.norm2
		if l.cross {
			ffNorm = l.norm3
		}
		layerNormTo(h, x, n, c, ffNorm.w, ffNorm.b, normEps)
		addScaled(x, l.feedForward(h, n), l.gamma2, c)
		if l.normOut != nil {
			groupNormTokens(x, n, c, l.normOut.w, l.normOut.b, normEps)
		}
		return x
	}
	var a []float32
	if l.cross {
		a = l.attention(x, n, kv, nk, c)
	} else {
		a = l.attention(x, n, x, n, c)
	}
	addScaled(x, a, l.gamma1, c)
	layerNorm(x, n, c, l.norm1.w, l.norm1.b, normEps)
	addScaled(x, l.feedForward(x, n), l.gamma2, c)
	layerNorm(x, n, c, l.norm2.w, l.norm2.b, normEps)
	return x
}

func (l *tLayer) feedForward(x []float32, n int) []float32 {
	h := linear(x, n, l.lin1W, l.lin1B)
	if l.gelu {
		gelu(h)
	} else {
		for i, v := range h {
			h[i] = max(v, 0)
		}
	}
	return linear(h, n, l.lin2W, l.lin2B)
}

// attention is nn.MultiheadAttention(q, kv, kv) without masks.
func (l *tLayer) attention(q []float32, nq int, kv []float32, nk int, c int) []float32 {
	cc := c * c
	wq := &Tensor{Data: l.inProjW.Data[:cc], Shape: []int{c, c}}
	wk := &Tensor{Data: l.inProjW.Data[cc : 2*cc], Shape: []int{c, c}}
	wv := &Tensor{Data: l.inProjW.Data[2*cc : 3*cc], Shape: []int{c, c}}
	bq := &Tensor{Data: l.inProjB.Data[:c], Shape: []int{c}}
	bk := &Tensor{Data: l.inProjB.Data[c : 2*c], Shape: []int{c}}
	bv := &Tensor{Data: l.inProjB.Data[2*c : 3*c], Shape: []int{c}}
	Q := linear(q, nq, wq, bq)
	K := linear(kv, nk, wk, bk)
	V := linear(kv, nk, wv, bv)

	d := c / l.heads
	scale := float32(1 / math.Sqrt(float64(d)))
	out := make([]float32, nq*c)
	scores := make([]float32, nq*nk)
	for h := 0; h < l.heads; h++ {
		sgemm(false, true, nq, nk, d, scale, Q[h*d:], c, K[h*d:], c, 0, scores, nk)
		softmaxRows(scores, nq, nk)
		sgemm(false, false, nq, d, nk, 1, scores, nk, V[h*d:], c, 0, out[h*d:], c)
	}
	return linear(out, nq, l.outProjW, l.outProj)
}

func softmaxRows(x []float32, n, m int) {
	parallelFor(n, workersFor(n*m), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			row := x[i*m : (i+1)*m]
			mx := row[0]
			for _, v := range row {
				mx = max(mx, v)
			}
			var sum float64
			for j, v := range row {
				e := expf(v - mx)
				row[j] = e
				sum += float64(e)
			}
			inv := float32(1 / sum)
			for j := range row {
				row[j] *= inv
			}
		}
	})
}

// addScaled computes x += gamma * y, gamma being a per-channel LayerScale
// (or nil for identity).
func addScaled(x, y []float32, gamma *Tensor, c int) {
	if gamma == nil {
		addInPlace(x, y)
		return
	}
	g := gamma.Data
	for i := 0; i < len(x); i += c {
		xr := x[i : i+c]
		yr := y[i : i+c]
		for j := range xr {
			xr[j] += g[j] * yr[j]
		}
	}
}

// sin1DEmbedding is create_sin_embedding(length, dim) as [length, dim]. The
// float32 rounding steps of the reference are reproduced.
func sin1DEmbedding(length, dim int, maxPeriod float64) []float32 {
	half := dim / 2
	out := make([]float32, length*dim)
	for a := 0; a < half; a++ {
		e := float32(a) / float32(half-1)
		denom := float32(math.Pow(maxPeriod, float64(e)))
		for p := 0; p < length; p++ {
			phase := float64(float32(p) / denom)
			out[p*dim+a] = float32(math.Cos(phase))
			out[p*dim+half+a] = float32(math.Sin(phase))
		}
	}
	return out
}

// sin2DEmbedding is create_2d_sin_embedding(dim, height, width), returned
// directly as tokens [(width, height), dim] in the transformer order.
func sin2DEmbedding(dim, height, width int, maxPeriod float64) []float32 {
	dm := dim / 2
	coef := float32(-(math.Log(maxPeriod) / float64(dm)))
	out := make([]float32, height*width*dim)
	for i := 0; i < dm/2; i++ {
		div := float32(math.Exp(float64(float32(2*i) * coef)))
		for w := 0; w < width; w++ {
			arg := float64(float32(w) * div)
			s, cs := float32(math.Sin(arg)), float32(math.Cos(arg))
			for h := 0; h < height; h++ {
				base := (w*height + h) * dim
				out[base+2*i] = s
				out[base+2*i+1] = cs
			}
		}
		for h := 0; h < height; h++ {
			arg := float64(float32(h) * div)
			s, cs := float32(math.Sin(arg)), float32(math.Cos(arg))
			for w := 0; w < width; w++ {
				base := (w*height + h) * dim
				out[base+dm+2*i] = s
				out[base+dm+2*i+1] = cs
			}
		}
	}
	return out
}
