package demucs

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
)

// Options controls Separate. The zero value is valid but disables the
// random shift trick; DefaultOptions matches the Python CLI defaults.
type Options struct {
	// Shifts is the number of random time shifts (up to 0.5s) to average
	// over. Each shift costs a full pass over the track; the Python default
	// is 1, and 0 makes the output deterministic regardless of Seed.
	Shifts int
	// Seed seeds the shift offsets, so results are reproducible.
	Seed int64
	// NoSplit disables chunking; only valid for mixes no longer than the
	// model segment.
	NoSplit bool
	// Overlap between consecutive chunks, as a fraction of the segment.
	Overlap float64
	// Segment overrides the chunk length in seconds. It must not exceed the
	// model segment (7.8s for the released HTDemucs models). 0 keeps the
	// model segment.
	Segment float64
	// TransitionPower shapes the cross-fade between chunks; values below 1
	// are treated as 1.
	TransitionPower float64
	// Workers is the number of chunks evaluated concurrently. Each forward
	// pass is already multi-threaded, so 1 (the default) is usually best.
	Workers int
	// Progress, if set, is called after each chunk with the number of chunks
	// done and the estimated total, across shifts and bag models, and once
	// more at the end with done == total.
	Progress func(done, total int)
	// Context, if set, aborts the separation when cancelled.
	Context context.Context
}

// DefaultOptions returns the defaults of `python -m demucs`.
func DefaultOptions() *Options {
	return &Options{Shifts: 1, Overlap: 0.25, TransitionPower: 1}
}

// Separate splits a mix into stems. mix is [channels][samples] at
// m.SampleRate; mono input is duplicated to stereo. Like the Python API, the
// mix is normalized before inference and the stems are rescaled back.
// SeparateAt is Separate for audio at another sample rate: the mix is
// resampled to m.SampleRate first (demucs.api.Separator does the same with
// julius). The stems come back at m.SampleRate, as in Python; use Resample to
// bring them back to rate if needed.
func (m *Model) SeparateAt(mix [][]float32, rate int, opts *Options) (*Separation, error) {
	if rate != m.SampleRate {
		mix = ResampleChannels(mix, rate, m.SampleRate)
	}
	return m.Separate(mix, opts)
}

func (m *Model) Separate(mix [][]float32, opts *Options) (*Separation, error) {
	if opts == nil {
		opts = DefaultOptions()
	}
	if len(mix) == 0 || len(mix[0]) == 0 {
		return nil, fmt.Errorf("demucs: empty mix")
	}
	for _, ch := range mix {
		if len(ch) != len(mix[0]) {
			return nil, fmt.Errorf("demucs: channels have different lengths")
		}
	}
	wav, err := m.matchChannels(mix)
	if err != nil {
		return nil, err
	}
	length := len(wav[0])

	// ref = wav.mean(0); wav = (wav - ref.mean()) / (ref.std() + 1e-8)
	ref := make([]float32, length)
	for _, ch := range wav {
		for i, v := range ch {
			ref[i] += v
		}
	}
	for i := range ref {
		ref[i] /= float32(len(wav))
	}
	mean, std := meanStd(ref)
	if length == 1 || math.IsNaN(float64(std)) {
		std = 0
	}
	scale := std + 1e-8
	norm := make([][]float32, len(wav))
	for c, ch := range wav {
		norm[c] = make([]float32, length)
		for i, v := range ch {
			norm[c][i] = (v - mean) / scale
		}
	}

	r := &runner{model: m, opts: *opts}
	r.prepare(length)
	out, err := r.applyBag(norm)
	if err != nil {
		return nil, err
	}
	if opts.Progress != nil {
		// The total is estimated up front (shift offsets are random); settle it.
		opts.Progress(r.done, r.done)
	}
	for _, stem := range out {
		for _, ch := range stem {
			for i, v := range ch {
				ch[i] = v*scale + mean
			}
		}
	}
	return &Separation{Sources: append([]string(nil), m.Sources...), Stems: out}, nil
}

func (m *Model) matchChannels(mix [][]float32) ([][]float32, error) {
	switch {
	case len(mix) == m.AudioChannels:
		return mix, nil
	case len(mix) == 1:
		out := make([][]float32, m.AudioChannels)
		for c := range out {
			out[c] = mix[0]
		}
		return out, nil
	case m.AudioChannels == 1:
		mono := make([]float32, len(mix[0]))
		for _, ch := range mix {
			for i, v := range ch {
				mono[i] += v / float32(len(mix))
			}
		}
		return [][]float32{mono}, nil
	case len(mix) > m.AudioChannels:
		return mix[:m.AudioChannels], nil
	}
	return nil, fmt.Errorf("demucs: cannot convert %d channels to %d", len(mix), m.AudioChannels)
}

// chunk is a window on a longer signal, like TensorChunk in demucs/apply.py:
// padding a chunk pulls real samples from around it before falling back to
// zeros past the ends of the underlying signal.
type chunk struct {
	base           [][]float32
	offset, length int
}

func newChunk(base [][]float32, offset, length int) chunk {
	total := len(base[0])
	return chunk{base: base, offset: offset, length: min(length, total-offset)}
}

func (c chunk) sub(offset, length int) chunk {
	length = min(length, c.length-offset)
	return chunk{base: c.base, offset: c.offset + offset, length: length}
}

func (c chunk) padded(target int) [][]float32 {
	delta := target - c.length
	total := len(c.base[0])
	start := c.offset - delta/2
	end := start + target
	cs, ce := max(0, start), min(total, end)
	out := make([][]float32, len(c.base))
	for ch, src := range c.base {
		out[ch] = make([]float32, target)
		copy(out[ch][cs-start:], src[cs:ce])
	}
	return out
}

// runner carries the options and progress accounting of one Separate call.
type runner struct {
	model *Model
	opts  Options
	rng   *rand.Rand
	// shiftOffset draws a shift in [0, maxShift]; replaceable by tests.
	shiftOffset func(maxShift int) int

	mu         sync.Mutex
	done       int
	totalCalls int
}

func (r *runner) prepare(length int) {
	r.rng = rand.New(rand.NewSource(r.opts.Seed))
	if r.shiftOffset == nil {
		r.shiftOffset = func(maxShift int) int { return r.rng.Intn(maxShift + 1) }
	}
	if r.opts.TransitionPower < 1 {
		r.opts.TransitionPower = 1
	}
	if r.opts.Workers < 1 {
		r.opts.Workers = 1
	}
	// Count forward passes for progress reporting.
	for _, sub := range r.model.models {
		l := length
		if r.opts.Shifts > 0 {
			l += int(0.5 * float64(sub.cfg.SampleRate))
		}
		n := 1
		if !r.opts.NoSplit {
			seg := r.segmentLength(sub)
			stride := int((1 - r.opts.Overlap) * float64(seg))
			if stride > 0 {
				n = (l + stride - 1) / stride
			}
		}
		r.totalCalls += n * max(1, r.opts.Shifts)
	}
}

func (r *runner) segmentLength(sub *htdemucs) int {
	if r.opts.Segment > 0 {
		return int(r.opts.Segment * float64(sub.cfg.SampleRate))
	}
	return sub.cfg.trainingLength()
}

// applyBag averages the estimates of every model in the bag, weighted per
// source. A single model is a bag of one with unit weights.
func (r *runner) applyBag(mix [][]float32) ([][][]float32, error) {
	m := r.model
	length := len(mix[0])
	sources := len(m.Sources)
	totals := make([]float32, sources)
	var estimates [][][]float32
	for i, sub := range m.models {
		est, err := r.applyShifts(sub, mix)
		if err != nil {
			return nil, err
		}
		if estimates == nil {
			estimates = zeros3(sources, len(mix), length)
		}
		for k := 0; k < sources; k++ {
			w := m.weights[i][k]
			totals[k] += w
			for c := range est[k] {
				dst := estimates[k][c]
				for j, v := range est[k][c] {
					dst[j] += v * w
				}
			}
		}
	}
	for k := range estimates {
		for _, ch := range estimates[k] {
			for j := range ch {
				ch[j] /= totals[k]
			}
		}
	}
	return estimates, nil
}

func (r *runner) applyShifts(sub *htdemucs, mix [][]float32) ([][][]float32, error) {
	length := len(mix[0])
	if r.opts.Shifts <= 0 {
		return r.applySplit(sub, newChunk(mix, 0, length))
	}
	maxShift := int(0.5 * float64(sub.cfg.SampleRate))
	padded := newChunk(mix, 0, length).padded(length + 2*maxShift)
	out := zeros3(len(sub.cfg.Sources), len(mix), length)
	for s := 0; s < r.opts.Shifts; s++ {
		offset := r.shiftOffset(maxShift)
		shifted := newChunk(padded, offset, length+maxShift-offset)
		res, err := r.applySplit(sub, shifted)
		if err != nil {
			return nil, err
		}
		for k := range out {
			for c := range out[k] {
				dst := out[k][c]
				for i, v := range res[k][c][maxShift-offset:][:length] {
					dst[i] += v
				}
			}
		}
	}
	inv := 1 / float32(r.opts.Shifts)
	for k := range out {
		for _, ch := range out[k] {
			for i := range ch {
				ch[i] *= inv
			}
		}
	}
	return out, nil
}

func (r *runner) applySplit(sub *htdemucs, mix chunk) ([][][]float32, error) {
	if r.opts.NoSplit {
		return r.forward(sub, mix)
	}
	length := mix.length
	sources := len(sub.cfg.Sources)
	seg := r.segmentLength(sub)
	if seg <= 0 {
		return nil, fmt.Errorf("demucs: invalid segment length %d", seg)
	}
	stride := int((1 - r.opts.Overlap) * float64(seg))
	if stride <= 0 {
		return nil, fmt.Errorf("demucs: overlap %v leaves no stride", r.opts.Overlap)
	}
	// Triangular weight peaking mid segment, normalized, to the transition power.
	weight := make([]float32, seg)
	for i := 0; i < seg/2; i++ {
		weight[i] = float32(i + 1)
	}
	for i := seg / 2; i < seg; i++ {
		weight[i] = float32(seg - i)
	}
	var wmax float32
	for _, v := range weight {
		wmax = max(wmax, v)
	}
	for i, v := range weight {
		weight[i] = float32(math.Pow(float64(v/wmax), r.opts.TransitionPower))
	}

	var offsets []int
	for off := 0; off < length; off += stride {
		offsets = append(offsets, off)
	}
	results := make([][][][]float32, len(offsets))
	errs := make([]error, len(offsets))
	sem := make(chan struct{}, r.opts.Workers)
	var wg sync.WaitGroup
	for i, off := range offsets {
		if err := r.ctxErr(); err != nil {
			return nil, err
		}
		wg.Add(1)
		sem <- struct{}{}
		go func(i, off int) {
			defer wg.Done()
			defer func() { <-sem }()
			results[i], errs[i] = r.forward(sub, mix.sub(off, seg))
		}(i, off)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	out := zeros3(sources, len(mix.base), length)
	sumWeight := make([]float32, length)
	for i, off := range offsets {
		res := results[i]
		n := len(res[0][0])
		for k := range out {
			for c := range out[k] {
				dst := out[k][c][off : off+n]
				for j, v := range res[k][c] {
					dst[j] += weight[j] * v
				}
			}
		}
		for j := 0; j < n; j++ {
			sumWeight[off+j] += weight[j]
		}
	}
	for k := range out {
		for _, ch := range out[k] {
			for j := range ch {
				ch[j] /= sumWeight[j]
			}
		}
	}
	return out, nil
}

// forward evaluates the model on one chunk, padded with its surroundings to
// the model segment, then trimmed back to the chunk.
func (r *runner) forward(sub *htdemucs, c chunk) ([][][]float32, error) {
	if err := r.ctxErr(); err != nil {
		return nil, err
	}
	valid := c.length
	if r.opts.Segment > 0 {
		valid = int(r.opts.Segment * float64(sub.cfg.SampleRate))
	} else if sub.cfg.useTrainSegment() {
		valid = sub.cfg.trainingLength()
	}
	if valid < c.length {
		return nil, fmt.Errorf("demucs: chunk of %d samples exceeds the model segment of %d samples; enable splitting", c.length, valid)
	}
	out, err := sub.forward(c.padded(valid))
	if err != nil {
		return nil, err
	}
	// center_trim to the chunk length.
	delta := len(out[0][0]) - c.length
	for k := range out {
		for ch := range out[k] {
			out[k][ch] = out[k][ch][delta/2 : delta/2+c.length]
		}
	}
	r.mu.Lock()
	r.done++
	done, total := r.done, r.totalCalls
	r.mu.Unlock()
	if r.opts.Progress != nil {
		r.opts.Progress(done, max(total, done))
	}
	return out, nil
}

func (r *runner) ctxErr() error {
	if r.opts.Context == nil {
		return nil
	}
	return r.opts.Context.Err()
}

func zeros3(a, b, c int) [][][]float32 {
	out := make([][][]float32, a)
	for i := range out {
		out[i] = make([][]float32, b)
		for j := range out[i] {
			out[i][j] = make([]float32, c)
		}
	}
	return out
}
