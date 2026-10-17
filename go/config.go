package demucs

import (
	"encoding/json"
	"fmt"
)

// fraction mirrors the {"numerator", "denominator"} encoding the exporter
// uses for Python Fraction values (the model segment length).
type fraction struct {
	Numerator   int64 `json:"numerator"`
	Denominator int64 `json:"denominator"`
}

func (f *fraction) UnmarshalJSON(b []byte) error {
	var num float64
	if err := json.Unmarshal(b, &num); err == nil {
		// Plain float: keep millisecond precision, as the exporter does.
		f.Numerator, f.Denominator = int64(num*1000+0.5), 1000
		return nil
	}
	type plain fraction
	return json.Unmarshal(b, (*plain)(f))
}

// config holds the HTDemucs constructor arguments stored by the exporter.
type config struct {
	Sources       []string `json:"sources"`
	AudioChannels int      `json:"audio_channels"`
	SampleRate    int      `json:"samplerate"`
	Segment       fraction `json:"segment"`
	Channels      int      `json:"channels"`
	ChannelsTime  *int     `json:"channels_time"`
	Growth        float64  `json:"growth"`
	NFFT          int      `json:"nfft"`
	WienerIters   int      `json:"wiener_iters"`
	Cac           bool     `json:"cac"`
	Depth         int      `json:"depth"`
	Rewrite       bool     `json:"rewrite"`
	MultiFreqs    []any    `json:"multi_freqs"`
	FreqEmb       float64  `json:"freq_emb"`
	EmbScale      float64  `json:"emb_scale"`
	KernelSize    int      `json:"kernel_size"`
	TimeStride    int      `json:"time_stride"`
	Stride        int      `json:"stride"`
	Context       int      `json:"context"`
	ContextEnc    int      `json:"context_enc"`
	NormStarts    int      `json:"norm_starts"`
	NormGroups    int      `json:"norm_groups"`
	DConvMode     int      `json:"dconv_mode"`
	DConvDepth    int      `json:"dconv_depth"`
	DConvComp     float64  `json:"dconv_comp"`
	BottomChans   int      `json:"bottom_channels"`
	UseTrainSeg   *bool    `json:"use_train_segment"`

	TLayers          int     `json:"t_layers"`
	THiddenScale     float64 `json:"t_hidden_scale"`
	THeads           int     `json:"t_heads"`
	TLayerScale      bool    `json:"t_layer_scale"`
	TGelu            bool    `json:"t_gelu"`
	TEmb             string  `json:"t_emb"`
	TMaxPeriod       float64 `json:"t_max_period"`
	TWeightPosEmbed  float64 `json:"t_weight_pos_embed"`
	TSinRandomShift  int     `json:"t_sin_random_shift"`
	TNormIn          bool    `json:"t_norm_in"`
	TNormInGroup     bool    `json:"t_norm_in_group"`
	TGroupNorm       bool    `json:"t_group_norm"`
	TNormFirst       bool    `json:"t_norm_first"`
	TNormOut         bool    `json:"t_norm_out"`
	TSparseSelfAttn  bool    `json:"t_sparse_self_attn"`
	TSparseCrossAttn bool    `json:"t_sparse_cross_attn"`
	TCrossFirst      bool    `json:"t_cross_first"`
}

func defaultConfig() config {
	// Defaults of HTDemucs.__init__, for keys an exporter may omit.
	return config{
		AudioChannels: 2, Channels: 48, Growth: 2, NFFT: 4096, Cac: true, Depth: 4,
		Rewrite: true, FreqEmb: 0.2, EmbScale: 10, KernelSize: 8, TimeStride: 2,
		Stride: 4, Context: 1, NormStarts: 4, NormGroups: 4, DConvMode: 1, DConvDepth: 2,
		DConvComp: 8, TLayers: 5, THiddenScale: 4, THeads: 8, TLayerScale: true,
		TGelu: true, TEmb: "sin", TMaxPeriod: 10000, TWeightPosEmbed: 1, TNormIn: true,
		TNormFirst: true, TNormOut: true, SampleRate: 44100,
		Segment: fraction{Numerator: 10, Denominator: 1},
	}
}

func parseConfig(raw json.RawMessage) (config, error) {
	cfg := defaultConfig()
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return cfg, fmt.Errorf("demucs: parsing model config: %w", err)
	}
	return cfg, cfg.validate()
}

// validate rejects HTDemucs options that the released models do not use and
// this port does not implement, instead of silently computing something else.
func (c *config) validate() error {
	unsupported := func(what string) error {
		return fmt.Errorf("demucs: unsupported model option: %s", what)
	}
	switch {
	case len(c.Sources) == 0:
		return fmt.Errorf("demucs: model config has no sources")
	case !c.Cac:
		return unsupported("cac=false (Wiener filtering)")
	case len(c.MultiFreqs) > 0:
		return unsupported("multi_freqs")
	case c.TLayers > 0 && c.TEmb != "sin":
		return unsupported("t_emb=" + c.TEmb)
	case c.TLayers > 0 && (c.TSparseSelfAttn || c.TSparseCrossAttn):
		return unsupported("sparse attention")
	case c.TLayers > 0 && (c.TNormInGroup && !c.TNormIn):
		return unsupported("t_norm_in_group")
	case c.TLayers > 0 && c.TGroupNorm:
		return unsupported("t_group_norm")
	case c.TSinRandomShift != 0:
		return unsupported("t_sin_random_shift")
	case c.NFFT < 16 || c.NFFT&(c.NFFT-1) != 0:
		return unsupported(fmt.Sprintf("nfft=%d", c.NFFT))
	case c.Segment.Denominator <= 0 || c.Segment.Numerator <= 0:
		return fmt.Errorf("demucs: invalid segment %d/%d", c.Segment.Numerator, c.Segment.Denominator)
	}
	return nil
}

// trainingLength is int(segment * samplerate), computed exactly.
func (c *config) trainingLength() int {
	return int(c.Segment.Numerator * int64(c.SampleRate) / c.Segment.Denominator)
}

func (c *config) useTrainSegment() bool {
	return c.UseTrainSeg == nil || *c.UseTrainSeg
}
