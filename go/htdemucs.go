package demucs

import "fmt"

// htdemucs is a single Hybrid Transformer Demucs model (demucs/htdemucs.py),
// inference only.
type htdemucs struct {
	cfg      config
	encoder  []*encLayer
	decoder  []*decLayer
	tencoder []*encLayer
	tdecoder []*decLayer

	freqEmb      *Tensor // [freqs, channels], already multiplied by emb_scale
	freqEmbScale float32

	upsampler, downsampler   *conv
	upsamplerT, downsamplerT *conv
	transformer              *crossTransformer
}

func loadHTDemucs(cfg config, w *weights) (*htdemucs, error) {
	m := &htdemucs{cfg: cfg}
	// Input channel counts are implied by the weight shapes; only the output
	// counts are tracked, for the decoders and the last frequency layer.
	chout := cfg.Channels
	if cfg.ChannelsTime != nil {
		chout = *cfg.ChannelsTime
	}
	choutZ := cfg.Channels
	freqs := cfg.NFFT / 2
	type layerKw struct {
		kernel, stride int
		freq, pad      bool
		norm           bool
	}
	var tdecs []*decLayer
	var decs []*decLayer
	for index := 0; index < cfg.Depth; index++ {
		kw := layerKw{kernel: cfg.KernelSize, stride: cfg.Stride, freq: freqs > 1, pad: true,
			norm: index >= cfg.NormStarts}
		if !kw.freq {
			if freqs != 1 {
				return nil, fmt.Errorf("demucs: invalid frequency count %d", freqs)
			}
			kw.kernel = cfg.TimeStride * 2
			kw.stride = cfg.TimeStride
		}
		lastFreq := false
		if kw.freq && freqs <= cfg.KernelSize {
			kw.kernel = freqs
			kw.pad = false
			lastFreq = true
		}
		if lastFreq {
			choutZ = max(chout, choutZ)
			chout = choutZ
		}
		padOf := func(k int, pad bool) int {
			if pad {
				return k / 4
			}
			return 0
		}

		enc := &encLayer{freq: kw.freq, kernel: kw.kernel, stride: kw.stride,
			pad: padOf(kw.kernel, kw.pad), norm: kw.norm, normGroups: cfg.NormGroups}
		if err := enc.load(w, fmt.Sprintf("encoder.%d", index), cfg); err != nil {
			return nil, err
		}
		m.encoder = append(m.encoder, enc)
		if kw.freq {
			tenc := &encLayer{freq: false, empty: lastFreq, kernel: cfg.KernelSize,
				stride: cfg.Stride, pad: padOf(cfg.KernelSize, true), norm: kw.norm,
				normGroups: cfg.NormGroups}
			if err := tenc.load(w, fmt.Sprintf("tencoder.%d", index), cfg); err != nil {
				return nil, err
			}
			m.tencoder = append(m.tencoder, tenc)
		}
		// Built outermost first; the reference inserts decoders at the front
		// of its list, so they are reversed below.
		decs = append(decs, &decLayer{freq: kw.freq, last: index == 0, kernel: kw.kernel,
			stride: kw.stride, pad: padOf(kw.kernel, kw.pad), chin: choutZ, norm: kw.norm,
			normGroups: cfg.NormGroups})
		if kw.freq {
			tdecs = append(tdecs, &decLayer{freq: false, empty: lastFreq, last: index == 0,
				kernel: cfg.KernelSize, stride: cfg.Stride, pad: padOf(cfg.KernelSize, true),
				chin: chout, norm: kw.norm, normGroups: cfg.NormGroups})
		}

		chout = int(cfg.Growth * float64(chout))
		choutZ = int(cfg.Growth * float64(choutZ))
		if kw.freq {
			if freqs <= cfg.KernelSize {
				freqs = 1
			} else {
				freqs /= cfg.Stride
			}
		}
		if index == 0 && cfg.FreqEmb != 0 {
			emb, err := w.get("freq_emb.embedding.weight")
			if err != nil {
				return nil, err
			}
			scaled := &Tensor{Data: make([]float32, len(emb.Data)), Shape: emb.Shape}
			for i, v := range emb.Data {
				scaled.Data[i] = v * float32(cfg.EmbScale)
			}
			m.freqEmb = scaled
			m.freqEmbScale = float32(cfg.FreqEmb)
		}
	}
	for i := len(decs) - 1; i >= 0; i-- {
		if err := decs[i].load(w, fmt.Sprintf("decoder.%d", len(decs)-1-i), cfg); err != nil {
			return nil, err
		}
		m.decoder = append(m.decoder, decs[i])
	}
	for i := len(tdecs) - 1; i >= 0; i-- {
		if err := tdecs[i].load(w, fmt.Sprintf("tdecoder.%d", len(tdecs)-1-i), cfg); err != nil {
			return nil, err
		}
		m.tdecoder = append(m.tdecoder, tdecs[i])
	}

	if cfg.BottomChans > 0 {
		for _, p := range []struct {
			dst  **conv
			name string
		}{
			{&m.upsampler, "channel_upsampler"}, {&m.downsampler, "channel_downsampler"},
			{&m.upsamplerT, "channel_upsampler_t"}, {&m.downsamplerT, "channel_downsampler_t"},
		} {
			c, err := loadConv(w, p.name)
			if err != nil {
				return nil, err
			}
			*p.dst = &c
		}
	}
	if cfg.TLayers > 0 {
		t, err := loadCrossTransformer(w, &cfg)
		if err != nil {
			return nil, err
		}
		m.transformer = t
	}
	if rest := w.unused(); len(rest) > 0 {
		return nil, fmt.Errorf("demucs: checkpoint has %d weights this port does not use, first: %s", len(rest), rest[0])
	}
	return m, nil
}

func (l *encLayer) load(w *weights, name string, cfg config) error {
	var err error
	if l.conv, err = loadConv(w, name+".conv"); err != nil {
		return err
	}
	if l.empty {
		return nil
	}
	if l.norm {
		if l.norm1, err = loadNorm(w, name+".norm1"); err != nil {
			return err
		}
	}
	if cfg.Rewrite {
		c, err := loadConv(w, name+".rewrite")
		if err != nil {
			return err
		}
		l.rewrite = &c
		if l.norm {
			if l.norm2, err = loadNorm(w, name+".norm2"); err != nil {
				return err
			}
		}
	}
	if cfg.DConvMode&1 != 0 {
		if l.dconv, err = loadDConv(w, name+".dconv", cfg.DConvDepth); err != nil {
			return err
		}
	}
	return nil
}

func (l *decLayer) load(w *weights, name string, cfg config) error {
	var err error
	if l.convTr, err = loadConv(w, name+".conv_tr"); err != nil {
		return err
	}
	if l.norm {
		if l.norm2, err = loadNorm(w, name+".norm2"); err != nil {
			return err
		}
	}
	if l.empty {
		return nil
	}
	if cfg.Rewrite {
		c, err := loadConv(w, name+".rewrite")
		if err != nil {
			return err
		}
		l.rewrite = &c
		if l.norm {
			if l.norm1, err = loadNorm(w, name+".norm1"); err != nil {
				return err
			}
		}
	}
	if cfg.DConvMode&2 != 0 {
		if l.dconv, err = loadDConv(w, name+".dconv", cfg.DConvDepth); err != nil {
			return err
		}
	}
	return nil
}

// forward separates one segment. mix is [channels][length] with length at
// most the training segment length; the result is [sources][channels][length].
func (m *htdemucs) forward(mix [][]float32) ([][][]float32, error) {
	cfg := &m.cfg
	if len(mix) != cfg.AudioChannels {
		return nil, fmt.Errorf("demucs: model expects %d channels, got %d", cfg.AudioChannels, len(mix))
	}
	length := len(mix[0])
	lengthPrePad := 0
	trainingLength := length
	if cfg.useTrainSegment() {
		trainingLength = cfg.trainingLength()
		if length > trainingLength {
			return nil, fmt.Errorf("demucs: segment of %d samples is longer than the training length %d", length, trainingLength)
		}
		if length < trainingLength {
			lengthPrePad = length
			padded := make([][]float32, len(mix))
			for c, ch := range mix {
				padded[c] = make([]float32, trainingLength)
				copy(padded[c], ch)
			}
			mix = padded
			length = trainingLength
		}
	}

	nfft := cfg.NFFT
	hl := nfft / 4
	// Spectrogram, complex as channels: channel 2c is the real part of audio
	// channel c, 2c+1 the imaginary part.
	le := (length + hl - 1) / hl
	freqs := nfft / 2
	x := newAct(2*cfg.AudioChannels, freqs, le)
	{
		pad := hl / 2 * 3
		for c, ch := range mix {
			padded := reflectPad(ch, pad, pad+le*hl-length)
			z := stft(padded, nfft, hl)
			if len(z) != le+4 {
				return nil, fmt.Errorf("demucs: internal error, %d frames instead of %d", len(z), le+4)
			}
			re := x.data[(2*c)*freqs*le:]
			im := x.data[(2*c+1)*freqs*le:]
			for t := 0; t < le; t++ {
				frame := z[t+2]
				for f := 0; f < freqs; f++ {
					re[f*le+t] = real(frame[f])
					im[f*le+t] = imag(frame[f])
				}
			}
		}
	}
	mean, std := meanStd(x.data)
	for i, v := range x.data {
		x.data[i] = (v - mean) / (1e-5 + std)
	}

	xt := newAct(cfg.AudioChannels, 1, length)
	for c, ch := range mix {
		copy(xt.data[c*length:], ch)
	}
	meanT, stdT := meanStd(xt.data)
	for i, v := range xt.data {
		xt.data[i] = (v - meanT) / (1e-5 + stdT)
	}

	var saved, savedT []act
	var lengths, lengthsT []int
	for idx, enc := range m.encoder {
		lengths = append(lengths, x.w)
		var inject *act
		if idx < len(m.tencoder) {
			lengthsT = append(lengthsT, xt.w)
			tenc := m.tencoder[idx]
			xt = tenc.forward(xt, nil)
			if !tenc.empty {
				savedT = append(savedT, xt)
			} else {
				inj := xt
				inject = &inj
			}
		}
		x = enc.forward(x, inject)
		if idx == 0 && m.freqEmb != nil {
			if m.freqEmb.Shape[0] != x.h || m.freqEmb.Shape[1] != x.c {
				return nil, fmt.Errorf("demucs: frequency embedding %v does not match [%d %d]", m.freqEmb.Shape, x.h, x.c)
			}
			for c := 0; c < x.c; c++ {
				for f := 0; f < x.h; f++ {
					e := m.freqEmbScale * m.freqEmb.Data[f*x.c+c]
					row := x.data[(c*x.h+f)*x.w : (c*x.h+f+1)*x.w]
					for t := range row {
						row[t] += e
					}
				}
			}
		}
		saved = append(saved, x)
	}

	if m.transformer != nil {
		if m.upsampler != nil {
			flat := act{data: x.data, c: x.c, h: 1, w: x.h * x.w}
			up := conv2d(flat, m.upsampler.w, m.upsampler.b, conv1dParams(1, 0, 1))
			x = act{data: up.data, c: up.c, h: x.h, w: x.w}
			xt = conv2d(xt, m.upsamplerT.w, m.upsamplerT.b, conv1dParams(1, 0, 1))
		}
		x, xt = m.transformer.forward(x, xt)
		if m.downsampler != nil {
			flat := act{data: x.data, c: x.c, h: 1, w: x.h * x.w}
			down := conv2d(flat, m.downsampler.w, m.downsampler.b, conv1dParams(1, 0, 1))
			x = act{data: down.data, c: down.c, h: x.h, w: x.w}
			xt = conv2d(xt, m.downsamplerT.w, m.downsamplerT.b, conv1dParams(1, 0, 1))
		}
	}

	offset := cfg.Depth - len(m.tdecoder)
	for idx, dec := range m.decoder {
		skip := saved[len(saved)-1]
		saved = saved[:len(saved)-1]
		l := lengths[len(lengths)-1]
		lengths = lengths[:len(lengths)-1]
		var pre act
		x, pre = dec.forward(x, &skip, l)
		if idx >= offset {
			tdec := m.tdecoder[idx-offset]
			lt := lengthsT[len(lengthsT)-1]
			lengthsT = lengthsT[:len(lengthsT)-1]
			if tdec.empty {
				if pre.h != 1 {
					return nil, fmt.Errorf("demucs: internal error, expected a single frequency, got %d", pre.h)
				}
				xt, _ = tdec.forward(pre, nil, lt)
			} else {
				skipT := savedT[len(savedT)-1]
				savedT = savedT[:len(savedT)-1]
				xt, _ = tdec.forward(xt, &skipT, lt)
			}
		}
	}

	sources := len(cfg.Sources)
	ac := cfg.AudioChannels
	for i, v := range x.data {
		x.data[i] = v*std + mean
	}
	out := make([][][]float32, sources)
	pad := hl / 2 * 3
	ilen := hl*((length+hl-1)/hl) + 2*pad
	for s := 0; s < sources; s++ {
		out[s] = make([][]float32, ac)
		for c := 0; c < ac; c++ {
			re := x.data[((s*ac+c)*2)*freqs*le:]
			im := x.data[((s*ac+c)*2+1)*freqs*le:]
			// Add back the Nyquist bin and two empty frames on each side.
			frames := make([][]complex64, le+4)
			for t := range frames {
				frames[t] = make([]complex64, freqs+1)
				if t < 2 || t >= le+2 {
					continue
				}
				for f := 0; f < freqs; f++ {
					frames[t][f] = complex(re[f*le+t-2], im[f*le+t-2])
				}
			}
			wav := istft(frames, nfft, hl, ilen)[pad : pad+length]
			tw := xt.data[(s*ac+c)*length : (s*ac+c+1)*length]
			res := make([]float32, length)
			for i := range res {
				res[i] = tw[i]*stdT + meanT + wav[i]
			}
			if lengthPrePad > 0 {
				res = res[:lengthPrePad]
			}
			out[s][c] = res
		}
	}
	return out, nil
}
