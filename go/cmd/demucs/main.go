// Command demucs separates a WAV file into stems with a model exported by
// tools/export_safetensors.py.
//
//	demucs -m htdemucs.safetensors -o separated song.wav
//
// writes separated/song/{drums,bass,other,vocals}.wav.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	demucs "github.com/kurz-infrastructure/demucs/go"
)

func main() {
	modelPath := flag.String("m", "", "model .safetensors file (default: download -n from the Hugging Face hub)")
	modelName := flag.String("n", "htdemucs", "pretrained model: htdemucs, htdemucs_ft or htdemucs_6s")
	cacheDir := flag.String("cache", "", "where downloaded models are kept (default "+demucs.DefaultCacheDir()+")")
	outDir := flag.String("o", "separated", "output directory")
	shifts := flag.Int("shifts", 1, "number of random shifts to average (0 for deterministic)")
	overlap := flag.Float64("overlap", 0.25, "overlap between chunks")
	segment := flag.Float64("segment", 0, "chunk length in seconds (0 = model segment)")
	seed := flag.Int64("seed", 0, "seed for the random shifts")
	jobs := flag.Int("j", 1, "chunks to process concurrently")
	int16Out := flag.Bool("int16", false, "write 16-bit PCM instead of 32-bit float")
	stems := flag.String("stems", "", "comma separated stems to write (default: all)")
	quiet := flag.Bool("q", false, "no progress output")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: demucs [-n htdemucs | -m model.safetensors] [flags] input.wav...\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() == 0 {
		flag.Usage()
		os.Exit(2)
	}
	start := time.Now()
	var model *demucs.Model
	var err error
	label := *modelPath
	if *modelPath != "" {
		model, err = demucs.Load(*modelPath)
	} else {
		label = *modelName
		fo := &demucs.FetchOptions{CacheDir: *cacheDir}
		if !*quiet {
			fo.Progress = func(file string, done, total int64) {
				fmt.Fprintf(os.Stderr, "\rdownloading %s: %d MB", file, done>>20)
				if done == total {
					fmt.Fprintln(os.Stderr)
				}
			}
		}
		model, err = demucs.Pretrained(*modelName, fo)
	}
	if err != nil {
		fatal(err)
	}
	if !*quiet {
		fmt.Fprintf(os.Stderr, "loaded %s (%s, backend %s) in %s\n", label,
			strings.Join(model.Sources, ", "), demucs.Backend(), time.Since(start).Round(time.Millisecond))
	}
	want := map[string]bool{}
	for _, s := range strings.Split(*stems, ",") {
		if s = strings.TrimSpace(s); s != "" {
			want[s] = true
		}
	}
	for name := range want {
		if !contains(model.Sources, name) {
			fatal(fmt.Errorf("unknown stem %q, the model has %s", name, strings.Join(model.Sources, ", ")))
		}
	}
	for _, in := range flag.Args() {
		mix, rate, err := readWAV(in)
		if err != nil {
			fatal(err)
		}
		if rate != model.SampleRate {
			// like the Python CLI: resample in, write the stems at the model rate
			mix = demucs.ResampleChannels(mix, rate, model.SampleRate)
			if !*quiet {
				fmt.Fprintf(os.Stderr, "%s: resampled %d Hz -> %d Hz\n", in, rate, model.SampleRate)
			}
			rate = model.SampleRate
		}
		opts := &demucs.Options{Shifts: *shifts, Overlap: *overlap, Segment: *segment,
			Seed: *seed, Workers: *jobs, TransitionPower: 1}
		if !*quiet {
			opts.Progress = func(done, total int) {
				fmt.Fprintf(os.Stderr, "\r%s: %d/%d chunks", filepath.Base(in), done, total)
			}
		}
		t0 := time.Now()
		sep, err := model.Separate(mix, opts)
		if err != nil {
			fatal(err)
		}
		if !*quiet {
			fmt.Fprintf(os.Stderr, "\n%s: %.1fs of audio in %s\n", in,
				float64(len(mix[0]))/float64(rate), time.Since(t0).Round(time.Millisecond))
		}
		dir := filepath.Join(*outDir, strings.TrimSuffix(filepath.Base(in), filepath.Ext(in)))
		if err := os.MkdirAll(dir, 0o755); err != nil {
			fatal(err)
		}
		for i, name := range sep.Sources {
			if len(want) > 0 && !want[name] {
				continue
			}
			path := filepath.Join(dir, name+".wav")
			if err := writeWAV(path, sep.Stems[i], rate, *int16Out); err != nil {
				fatal(err)
			}
			if !*quiet {
				fmt.Fprintf(os.Stderr, "wrote %s\n", path)
			}
		}
	}
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "demucs:", err)
	os.Exit(1)
}
