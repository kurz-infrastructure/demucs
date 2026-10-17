package demucs

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// The released models live on the Hugging Face hub, one repository per bag,
// as the Python package distributes them since 4.1: a yaml naming the models
// (and their weights for a bag) plus one safetensors file per model, with the
// HTDemucs constructor arguments in the file metadata. See demucs/hf.py.
const (
	DefaultHubURL   = "https://huggingface.co"
	defaultHubOwner = "adefossez"
)

// knownChecksums pins the sha256 of the released weights (the LFS object
// hashes of the hub repositories), so a download or a cached file that does
// not match is rejected. Files not listed here are loaded unverified.
var knownChecksums = map[string]string{
	"HTDemucs/955717e8.safetensors":    "d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd",
	"HTDemucs-ft/f7e0c4bc.safetensors": "2c85ab3c62dd6edd8e0b965e38b16fd1cdde357cc25de6b6bc9ce7c83f60925f",
	"HTDemucs-ft/d12395a8.safetensors": "5b01a97567ae9a3178a6236fb520251045c03eb8834bc8c24a4eec11d6c8fb56",
	"HTDemucs-ft/92cfc3b6.safetensors": "a241863551f30d01c42bd7b97da40839922ead3acb0f1fcab25682f55b4eeb59",
	"HTDemucs-ft/04573f0d.safetensors": "68854b0d7c2b3274723b5761f6fd9f5aec5f1bcd3f0de7c1669546fdb7871b7c",
	"HTDemucs-6s/5c90dfd2.safetensors": "d2a1745f0744721f6b8ca5bf469b67c651ea5ed1b52998cab033b2158609d411",
}

// FetchOptions controls Pretrained. The zero value downloads from the
// Hugging Face hub into DefaultCacheDir.
type FetchOptions struct {
	// CacheDir holds the downloaded files, under <owner>/<repo>/. Empty means DefaultCacheDir().
	CacheDir string
	// HubURL replaces DefaultHubURL (a mirror, or a test server).
	HubURL string
	// Client replaces http.DefaultClient.
	Client *http.Client
	// Context cancels downloads.
	Context context.Context
	// Checksums adds to or overrides the built-in sha256 pins, keyed "<repo>/<file>".
	Checksums map[string]string
	// Progress, if set, is called while a file downloads with the bytes so far
	// and the total (0 when unknown).
	Progress func(file string, done, total int64)
	// Offline fails instead of downloading when a file is not cached.
	Offline bool
}

// DefaultCacheDir is where Pretrained keeps the weights: the user cache
// directory (~/Library/Caches on macOS, $XDG_CACHE_HOME on Linux) plus
// "demucs", or the DEMUCS_CACHE environment variable.
func DefaultCacheDir() string {
	if d := os.Getenv("DEMUCS_CACHE"); d != "" {
		return d
	}
	base, err := os.UserCacheDir()
	if err != nil {
		base = os.TempDir()
	}
	return filepath.Join(base, "demucs")
}

// HubRepo maps a model name to its hub repository, like demucs.hf.hf_repo_name:
// htdemucs -> adefossez/HTDemucs, htdemucs_ft -> adefossez/HTDemucs-ft. A
// name with a slash names another owner: someone/htdemucs_ft.
func HubRepo(name string) (owner, repo, bag string) {
	owner, bag = defaultHubOwner, name
	if i := strings.IndexByte(name, '/'); i >= 0 {
		owner, bag = name[:i], name[i+1:]
	}
	switch {
	case bag == "htdemucs":
		repo = "HTDemucs"
	case strings.HasPrefix(bag, "htdemucs_"):
		repo = "HTDemucs-" + strings.TrimPrefix(bag, "htdemucs_")
	default:
		repo = "Demucs-" + bag
	}
	return owner, repo, bag
}

// Pretrained returns a released model by name (htdemucs, htdemucs_ft,
// htdemucs_6s), downloading its files on first use and reusing them after.
// It is the Go side of demucs.pretrained.get_model.
func Pretrained(name string, opts *FetchOptions) (*Model, error) {
	files, weights, err := Fetch(name, opts)
	if err != nil {
		return nil, err
	}
	return LoadFiles(files, weights)
}

// Fetch downloads (or finds cached) the bag definition and model files for
// name and returns the local safetensors paths with the bag weights.
func Fetch(name string, opts *FetchOptions) (files []string, weights [][]float32, err error) {
	if opts == nil {
		opts = &FetchOptions{}
	}
	owner, repo, bag := HubRepo(name)
	dir := opts.CacheDir
	if dir == "" {
		dir = DefaultCacheDir()
	}
	dir = filepath.Join(dir, owner, repo)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, nil, err
	}
	yamlPath, err := fetchFile(opts, owner, repo, bag+".yaml", dir, "")
	if err != nil {
		return nil, nil, err
	}
	raw, err := os.ReadFile(yamlPath)
	if err != nil {
		return nil, nil, err
	}
	models, weights, err := parseBagYAML(string(raw))
	if err != nil {
		return nil, nil, fmt.Errorf("demucs: %s: %w", yamlPath, err)
	}
	for _, sig := range models {
		fname := sig + ".safetensors"
		sum := knownChecksums[repo+"/"+fname]
		if s, ok := opts.Checksums[repo+"/"+fname]; ok {
			sum = s
		}
		p, err := fetchFile(opts, owner, repo, fname, dir, sum)
		if err != nil {
			return nil, nil, err
		}
		files = append(files, p)
	}
	return files, weights, nil
}

// fetchFile returns the cached path of a hub file, downloading it when it is
// missing or fails its checksum. Downloads go to a temporary name and are
// renamed into place once complete and verified.
func fetchFile(opts *FetchOptions, owner, repo, name, dir, sum string) (string, error) {
	dst := filepath.Join(dir, name)
	if st, err := os.Stat(dst); err == nil && st.Size() > 0 {
		if sum == "" {
			return dst, nil
		}
		if got, err := fileSHA256(dst); err == nil && got == sum {
			return dst, nil
		}
		// stale or corrupt: fall through and download again
	}
	if opts.Offline {
		return "", fmt.Errorf("demucs: %s/%s/%s is not cached in %s (offline)", owner, repo, name, dir)
	}
	hub := opts.HubURL
	if hub == "" {
		hub = DefaultHubURL
	}
	url := fmt.Sprintf("%s/%s/%s/resolve/main/%s", strings.TrimRight(hub, "/"), owner, repo, name)
	ctx := opts.Context
	if ctx == nil {
		ctx = context.Background()
	}
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", err
	}
	client := opts.Client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("demucs: downloading %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("demucs: downloading %s: HTTP %s", url, resp.Status)
	}
	tmp, err := os.CreateTemp(dir, name+".part-*")
	if err != nil {
		return "", err
	}
	defer os.Remove(tmp.Name())
	h := sha256.New()
	var done int64
	total := resp.ContentLength
	buf := make([]byte, 1<<20)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := tmp.Write(buf[:n]); werr != nil {
				tmp.Close()
				return "", werr
			}
			h.Write(buf[:n])
			done += int64(n)
			if opts.Progress != nil {
				opts.Progress(name, done, total)
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			tmp.Close()
			return "", fmt.Errorf("demucs: downloading %s: %w", url, rerr)
		}
	}
	if err := tmp.Close(); err != nil {
		return "", err
	}
	if got := hex.EncodeToString(h.Sum(nil)); sum != "" && got != sum {
		return "", fmt.Errorf("demucs: %s: sha256 %s, expected %s", name, got, sum)
	}
	os.Chmod(tmp.Name(), 0o644) // CreateTemp makes it private; the cache is shareable
	if err := os.Rename(tmp.Name(), dst); err != nil {
		return "", err
	}
	return dst, nil
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// parseBagYAML reads the two keys the bag files use, in the flow style the
// Python package writes:
//
//	models: ['f7e0c4bc', 'd12395a8']
//	weights: [
//	  [1., 0.],
//	  [0., 1.],
//	]
//
// A missing weights key means every model weighs 1 on every source (the
// BagOfModels default), which LoadFiles fills in.
func parseBagYAML(text string) (models []string, weights [][]float32, err error) {
	body := func(key string) (string, bool) {
		i := strings.Index(text, key+":")
		if i < 0 {
			return "", false
		}
		rest := text[i+len(key)+1:]
		open := strings.IndexByte(rest, '[')
		if open < 0 {
			return "", false
		}
		depth, end := 0, -1
		for j := open; j < len(rest); j++ {
			switch rest[j] {
			case '[':
				depth++
			case ']':
				depth--
				if depth == 0 {
					end = j
				}
			}
			if end >= 0 {
				break
			}
		}
		if end < 0 {
			return "", false
		}
		return rest[open+1 : end], true
	}
	list, ok := body("models")
	if !ok {
		return nil, nil, errors.New("no models list in bag yaml")
	}
	for _, s := range strings.Split(list, ",") {
		s = strings.Trim(strings.TrimSpace(s), `'"`)
		if s != "" {
			models = append(models, s)
		}
	}
	if len(models) == 0 {
		return nil, nil, errors.New("empty models list in bag yaml")
	}
	if rows, ok := body("weights"); ok {
		for _, row := range strings.Split(rows, "]") {
			row = strings.TrimSpace(strings.TrimLeft(strings.TrimSpace(row), ","))
			row = strings.TrimPrefix(strings.TrimSpace(row), "[")
			if strings.TrimSpace(row) == "" {
				continue
			}
			var ws []float32
			for _, v := range strings.Split(row, ",") {
				v = strings.TrimSpace(v)
				if v == "" {
					continue
				}
				f, err := strconv.ParseFloat(strings.TrimSuffix(v, "."), 64)
				if err != nil {
					return nil, nil, fmt.Errorf("bad weight %q", v)
				}
				ws = append(ws, float32(f))
			}
			weights = append(weights, ws)
		}
		if len(weights) != len(models) {
			return nil, nil, fmt.Errorf("%d weight rows for %d models", len(weights), len(models))
		}
	}
	return models, weights, nil
}
