// Package demucs is a Go port of Demucs (Hybrid Transformer Demucs v4)
// music source separation inference.
//
// Models are loaded from .safetensors files produced by
// go/tools/export_safetensors.py from the official PyTorch checkpoints:
//
//	m, err := demucs.Load("htdemucs.safetensors")
//	sep, err := m.Separate(mix, nil) // mix is [channels][samples] at m.SampleRate
//	vocals := sep.Stem("vocals")    // [channels][samples]
//
// On macOS with cgo enabled, matrix products run on Apple Accelerate. Other
// platforms (or the `demucs_purego` build tag) use a pure Go implementation.
package demucs

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// Model is a pretrained separation model: a single HTDemucs, or a bag of
// them whose estimates are averaged with per-source weights (htdemucs_ft).
type Model struct {
	// Sources lists the separated stems, e.g. drums, bass, other, vocals.
	Sources []string
	// SampleRate is the sample rate the model expects, in Hz.
	SampleRate int
	// AudioChannels is the number of channels the model expects.
	AudioChannels int

	models  []*htdemucs
	weights [][]float32
}

const formatName = "demucs-go/v1"

// Load reads a model exported by go/tools/export_safetensors.py.
func Load(path string) (*Model, error) {
	st, err := readSafetensorsFile(path)
	if err != nil {
		return nil, err
	}
	return fromSafetensors(st)
}

// LoadReader is Load from an io.Reader.
func LoadReader(r io.Reader) (*Model, error) {
	st, err := readSafetensors(r)
	if err != nil {
		return nil, err
	}
	return fromSafetensors(st)
}

// LoadFiles builds a model from one safetensors file per bag member in the
// Hugging Face hub layout (the constructor arguments in the metadata, as
// demucs/hf.py writes them). weights is the bag weight matrix
// [model][source]; nil means 1 everywhere, the BagOfModels default.
func LoadFiles(paths []string, bagWeights [][]float32) (*Model, error) {
	if len(paths) == 0 {
		return nil, fmt.Errorf("demucs: no model files")
	}
	m := &Model{}
	for i, p := range paths {
		st, err := readSafetensorsFile(p)
		if err != nil {
			return nil, fmt.Errorf("demucs: %s: %w", p, err)
		}
		if _, ok := st.metadata["klass"]; !ok {
			return nil, fmt.Errorf("demucs: %s is not a hub model file (no klass in its metadata)", p)
		}
		cfg, err := hubConfig(st.metadata)
		if err != nil {
			return nil, fmt.Errorf("demucs: %s: %w", p, err)
		}
		if err := m.add(cfg, &weights{tensors: st.tensors, used: map[string]bool{}}); err != nil {
			return nil, fmt.Errorf("demucs: %s: %w", p, err)
		}
		var w []float32
		if bagWeights != nil {
			if i >= len(bagWeights) {
				return nil, fmt.Errorf("demucs: %d bag weights for %d models", len(bagWeights), len(paths))
			}
			w = bagWeights[i]
		} else {
			w = make([]float32, len(cfg.Sources))
			for j := range w {
				w[j] = 1
			}
		}
		if len(w) != len(cfg.Sources) {
			return nil, fmt.Errorf("demucs: model %d has %d weights for %d sources", i, len(w), len(cfg.Sources))
		}
		m.weights = append(m.weights, w)
	}
	return m, nil
}

// hubConfig parses the metadata of a hub model file: klass names the model
// class, kwargs its constructor arguments as JSON (fractions encoded as
// {"_type": "fraction", "numerator", "denominator"}, which the fraction
// decoder accepts), structure marks a quantized state this port cannot read.
func hubConfig(meta map[string]string) (config, error) {
	if k := meta["klass"]; !strings.HasSuffix(k, ".HTDemucs") {
		return config{}, fmt.Errorf("only HTDemucs models are supported, got %s", k)
	}
	if _, quantized := meta["structure"]; quantized {
		return config{}, fmt.Errorf("quantized (diffq) model states are not supported")
	}
	return parseConfig(json.RawMessage(meta["kwargs"]))
}

// add loads one bag member and checks it agrees with the others.
func (m *Model) add(cfg config, w *weights) error {
	sub, err := loadHTDemucs(cfg, w)
	if err != nil {
		return err
	}
	if len(m.models) == 0 {
		m.Sources = cfg.Sources
		m.SampleRate = cfg.SampleRate
		m.AudioChannels = cfg.AudioChannels
	} else if !equalStrings(cfg.Sources, m.Sources) || cfg.SampleRate != m.SampleRate ||
		cfg.AudioChannels != m.AudioChannels {
		return fmt.Errorf("bag models disagree on sources, sample rate or channels")
	}
	m.models = append(m.models, sub)
	return nil
}

func fromSafetensors(st *safetensorsFile) (*Model, error) {
	if _, hub := st.metadata["klass"]; hub && st.metadata["format"] == "" {
		// a single hub model file, as downloaded by Pretrained
		cfg, err := hubConfig(st.metadata)
		if err != nil {
			return nil, fmt.Errorf("demucs: %w", err)
		}
		m := &Model{}
		if err := m.add(cfg, &weights{tensors: st.tensors, used: map[string]bool{}}); err != nil {
			return nil, fmt.Errorf("demucs: %w", err)
		}
		w := make([]float32, len(cfg.Sources))
		for j := range w {
			w[j] = 1
		}
		m.weights = [][]float32{w}
		return m, nil
	}
	if f := st.metadata["format"]; f != formatName {
		return nil, fmt.Errorf("demucs: unknown model format %q, expected %q", f, formatName)
	}
	var configs []json.RawMessage
	if err := json.Unmarshal([]byte(st.metadata["models"]), &configs); err != nil {
		return nil, fmt.Errorf("demucs: parsing model list: %w", err)
	}
	if len(configs) == 0 {
		return nil, fmt.Errorf("demucs: file contains no model")
	}
	var bagWeights [][]float32
	if err := json.Unmarshal([]byte(st.metadata["weights"]), &bagWeights); err != nil {
		return nil, fmt.Errorf("demucs: parsing bag weights: %w", err)
	}
	if len(bagWeights) != len(configs) {
		return nil, fmt.Errorf("demucs: %d bag weights for %d models", len(bagWeights), len(configs))
	}
	m := &Model{weights: bagWeights}
	for i, raw := range configs {
		cfg, err := parseConfig(raw)
		if err != nil {
			return nil, err
		}
		w := &weights{prefix: fmt.Sprintf("models.%d.", i), tensors: st.tensors, used: map[string]bool{}}
		sub, err := loadHTDemucs(cfg, w)
		if err != nil {
			return nil, fmt.Errorf("demucs: model %d: %w", i, err)
		}
		if i == 0 {
			m.Sources = cfg.Sources
			m.SampleRate = cfg.SampleRate
			m.AudioChannels = cfg.AudioChannels
		} else if !equalStrings(cfg.Sources, m.Sources) || cfg.SampleRate != m.SampleRate ||
			cfg.AudioChannels != m.AudioChannels {
			return nil, fmt.Errorf("demucs: bag models disagree on sources, sample rate or channels")
		}
		if len(bagWeights[i]) != len(cfg.Sources) {
			return nil, fmt.Errorf("demucs: model %d has %d weights for %d sources", i, len(bagWeights[i]), len(cfg.Sources))
		}
		m.models = append(m.models, sub)
	}
	return m, nil
}

// Segment returns the length, in samples, of the chunks the model processes
// at once. For a bag, it is the shortest of its models.
func (m *Model) Segment() int {
	seg := 0
	for _, sub := range m.models {
		if l := sub.cfg.trainingLength(); seg == 0 || l < seg {
			seg = l
		}
	}
	return seg
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Separation holds the separated stems of a mix.
type Separation struct {
	// Sources names each stem, in the same order as Stems.
	Sources []string
	// Stems is indexed [source][channel][sample].
	Stems [][][]float32
}

// Stem returns the [channel][sample] audio of a source by name, or nil.
func (s *Separation) Stem(name string) [][]float32 {
	for i, n := range s.Sources {
		if n == name {
			return s.Stems[i]
		}
	}
	return nil
}

// Mono returns the channel average of a source by name, or nil.
func (s *Separation) Mono(name string) []float64 {
	stem := s.Stem(name)
	if len(stem) == 0 {
		return nil
	}
	out := make([]float64, len(stem[0]))
	for _, ch := range stem {
		for i, v := range ch {
			out[i] += float64(v)
		}
	}
	for i := range out {
		out[i] /= float64(len(stem))
	}
	return out
}
