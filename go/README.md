# Demucs for Go

A Go port of [Hybrid Transformer Demucs](../README.md) (v4) inference, for
using source separation as a library in Go programs. It runs the released
`htdemucs`, `htdemucs_ft` and `htdemucs_6s` models and matches the PyTorch
implementation to float32 precision (max abs error ~3e-6 on `htdemucs`, see
[Testing](#testing)).

- **macOS**: with cgo enabled (the default), every matrix product runs on
  Apple Accelerate (`cblas_sgemm`, which uses the AMX units on Apple Silicon).
  Linking Accelerate adds no dependency: it ships with the OS.
- **Everywhere else**, or with `-tags demucs_purego`: a multi-threaded pure Go
  backend. Slower, but a single static binary.

The model forward pass, `apply_model` (chunking with overlap, random shifts,
bags of models) and the input normalization of `demucs.api.Separator` are
ported. Audio decoding and resampling are not: feed it audio at
`model.SampleRate` (44.1 kHz).

## Getting the weights

The released models (`htdemucs`, `htdemucs_ft`, `htdemucs_6s`) are fetched from
the Hugging Face hub on first use, from the same repositories the Python
package uses since 4.1 (`adefossez/HTDemucs` and friends: one safetensors
file per model, 84 MB each, with the constructor arguments in the metadata),
and cached under `DefaultCacheDir()` (`~/Library/Caches/demucs` on macOS,
`$XDG_CACHE_HOME/demucs` on Linux, or `$DEMUCS_CACHE`). Downloads are checked
against pinned sha256 hashes. Nothing else is needed: no Python, no export.

```go
model, err := demucs.Pretrained("htdemucs", nil) // or &demucs.FetchOptions{CacheDir: ..., Progress: ...}
```

`go/tools/export_safetensors.py` still converts a custom checkpoint (a `.th`
file from your own training) to a single file for `demucs.Load`. It works with
an installed `demucs` package or this checkout (which also needs `omegaconf`).

## Usage

```go
import demucs "github.com/kurz-infrastructure/demucs/go"

model, err := demucs.Load("htdemucs.safetensors")
// mix is [channels][samples] float32 at model.SampleRate. Mono is duplicated to stereo.
sep, err := model.Separate(mix, nil) // nil = Python CLI defaults: 1 shift, 25% overlap
drums := sep.Stem("drums")           // [channels][samples]
vocals := sep.Mono("vocals")         // []float64, channel average
```

`Separate` wants audio at `model.SampleRate` (44.1 kHz). `SeparateAt(mix, rate, opts)`
resamples any other rate first, with a port of the julius sinc resampler that
`demucs.api.Separator` uses; the stems come back at the model rate, as in
Python. `Resample` and `ResampleChannels` are exported for the way back.

`Options` mirrors `apply_model`: `Shifts`, `Overlap`, `Segment`, `NoSplit` and
`TransitionPower`, plus `Seed` (shift offsets are seeded, so results are
reproducible), `Workers`, `Progress` and `Context` for cancellation.
`Shifts: 0` skips the shift trick, which saves a pass over the track; the
original docs credit shifts with up to 0.2 points of SDR.

For an analyzer that works at a lower rate (e.g. lightgen's 22.05 kHz
`analyze.Stems`), decode the file in stereo at 44.1 kHz, separate, then
resample the `Mono` stems down with the analyzer's own resampler.

### CLI

```bash
go run ./go/cmd/demucs -o separated song.wav              # htdemucs, downloaded on first use
go run ./go/cmd/demucs -n htdemucs_ft -o separated song.wav
go run ./go/cmd/demucs -m custom.safetensors -o separated song.wav
```

writes `separated/song/{drums,bass,other,vocals}.wav` at the model sample
rate. Input is a WAV file at any sample rate (other rates are resampled in,
like the Python CLI). Run with `-h` for flags.

## Performance

Most of the time goes to matrix products (65–80% of the CPU profile with
the pure Go backend), which is what Accelerate takes over. Measured with
`go test -bench Forward` (see `bench_test.go`):

| Machine | Backend | Time per segment |
| --- | --- | --- |
| Raspberry Pi 5 (4× Cortex-A76) | pure Go | 44.6 s |
| Raspberry Pi 5, PyTorch 2.14 CPU, 4 threads | (reference) | ~23 s |
| Apple Silicon | Accelerate | reported by the `macos-pretrained` CI job |

A track needs about `duration / (0.75 × 7.8 s) × (shifts or 1)` segments.

## Testing

```bash
cd go && go test ./...
```

`testdata/` holds a tiny, randomly initialized HTDemucs and a bag of two,
with PyTorch outputs for the raw forward pass, chunked separation, shifts and
bag averaging. It is regenerated with `python3 go/tools/make_testdata.py
go/testdata`. To compare against the released weights:

```bash
python3 go/tools/make_pretrained_reference.py htdemucs test.mp3 /tmp/reference.safetensors
DEMUCS_TEST_MODEL=htdemucs DEMUCS_TEST_REFERENCE=/tmp/reference.safetensors \
  go test -run Pretrained -v ./go
```

`DEMUCS_TEST_MODEL` is a model name (downloaded through the Go package) or
the path of an exported file. `testdata/hf/` holds the tiny model in the hub
layout, written by `go/tools/make_hf_testdata.py` from the exporter files, so
both loaders are checked against each other offline.

CI (`.github/workflows/go.yml`) runs the tests on Linux (pure Go) and on
macOS with both backends, plus the comparison against PyTorch on the
released `htdemucs` weights.

## Module path

The module lives in a subdirectory, so it is `github.com/kurz-infrastructure/demucs/go`,
and release tags must be prefixed: `go/v0.1.0`.
