package demucs

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

// hubServer serves testdata/hf like the Hugging Face hub: /<owner>/<repo>/resolve/main/<file>.
func hubServer(t *testing.T) (*httptest.Server, *int64) {
	t.Helper()
	var hits int64
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&hits, 1)
		parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
		if len(parts) != 5 || parts[2] != "resolve" || parts[3] != "main" {
			http.NotFound(w, r)
			return
		}
		http.ServeFile(w, r, filepath.Join("testdata", "hf", parts[4]))
	}))
	t.Cleanup(ts.Close)
	return ts, &hits
}

func TestHubRepo(t *testing.T) {
	for _, c := range []struct{ name, owner, repo, bag string }{
		{"htdemucs", "adefossez", "HTDemucs", "htdemucs"},
		{"htdemucs_ft", "adefossez", "HTDemucs-ft", "htdemucs_ft"},
		{"htdemucs_6s", "adefossez", "HTDemucs-6s", "htdemucs_6s"},
		{"mdx_extra_q", "adefossez", "Demucs-mdx_extra_q", "mdx_extra_q"},
		{"someone/htdemucs_ft", "someone", "HTDemucs-ft", "htdemucs_ft"},
	} {
		if o, r, b := HubRepo(c.name); o != c.owner || r != c.repo || b != c.bag {
			t.Fatalf("%s -> %s %s %s", c.name, o, r, b)
		}
	}
}

func TestParseBagYAML(t *testing.T) {
	models, weights, err := parseBagYAML("models: ['955717e8']\n")
	if err != nil || len(models) != 1 || models[0] != "955717e8" || weights != nil {
		t.Fatalf("%v %v %v", models, weights, err)
	}
	models, weights, err = parseBagYAML("models: ['f7e0c4bc', 'd12395a8']\nweights: [\n  [1., 0.],\n  [0., 1.],\n]\n")
	if err != nil || len(models) != 2 || len(weights) != 2 || weights[0][0] != 1 || weights[0][1] != 0 || weights[1][1] != 1 {
		t.Fatalf("%v %v %v", models, weights, err)
	}
	if _, _, err := parseBagYAML("weights: [[1.]]\n"); err == nil {
		t.Fatal("missing models must fail")
	}
	if _, _, err := parseBagYAML("models: ['a', 'b']\nweights: [[1.]]\n"); err == nil {
		t.Fatal("weight row count must match")
	}
}

func TestPretrainedDownloadsCachesAndVerifies(t *testing.T) {
	ts, hits := hubServer(t)
	cache := t.TempDir()
	opts := &FetchOptions{HubURL: ts.URL, CacheDir: cache}
	var progressed bool
	opts.Progress = func(file string, done, total int64) { progressed = done > 0 && total > 0 }
	m, err := Pretrained("tiny", opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Sources) != 2 || m.SampleRate != 44100 || !progressed {
		t.Fatalf("model %v %d progressed %v", m.Sources, m.SampleRate, progressed)
	}
	if _, err := os.Stat(filepath.Join(cache, "adefossez", "Demucs-tiny", "tiny0.safetensors")); err != nil {
		t.Fatal("file not cached under owner/repo")
	}
	n := atomic.LoadInt64(hits)
	if _, err := Pretrained("tiny", opts); err != nil || atomic.LoadInt64(hits) != n {
		t.Fatalf("second load must come from the cache (hits %d -> %d, err %v)", n, atomic.LoadInt64(hits), err)
	}
	if _, err := Pretrained("tiny", &FetchOptions{HubURL: ts.URL, CacheDir: t.TempDir(), Offline: true}); err == nil || !strings.Contains(err.Error(), "offline") {
		t.Fatalf("offline with an empty cache must fail: %v", err)
	}
	// a pinned checksum is enforced on download and on cached files
	bad := &FetchOptions{HubURL: ts.URL, CacheDir: t.TempDir(), Checksums: map[string]string{"Demucs-tiny/tiny0.safetensors": strings.Repeat("0", 64)}}
	if _, err := Pretrained("tiny", bad); err == nil || !strings.Contains(err.Error(), "sha256") {
		t.Fatalf("bad checksum must fail: %v", err)
	}
	raw, _ := os.ReadFile(filepath.Join("testdata", "hf", "tiny0.safetensors"))
	sum := sha256.Sum256(raw)
	good := &FetchOptions{HubURL: ts.URL, CacheDir: cache, Checksums: map[string]string{"Demucs-tiny/tiny0.safetensors": hex.EncodeToString(sum[:])}}
	if _, err := Pretrained("tiny", good); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(cache, "adefossez", "Demucs-tiny", "tiny0.safetensors"), []byte("corrupt"), 0o644)
	n = atomic.LoadInt64(hits)
	if _, err := Pretrained("tiny", good); err != nil || atomic.LoadInt64(hits) == n {
		t.Fatalf("a corrupt cached file must be downloaded again: %v", err)
	}
	if _, err := Pretrained("nope", opts); err == nil {
		t.Fatal("unknown model must fail")
	}
}

// The hub layout and the exporter layout must give the same model: same
// weights, same output.
func TestHubFilesMatchExportedModel(t *testing.T) {
	ref := loadReference(t)
	mix := rows(ref.tensors["mix"])
	opts := DefaultOptions()
	opts.Shifts = 0
	for _, c := range []struct {
		exported string
		hub      []string
		weights  [][]float32
	}{
		{"tiny.safetensors", []string{"testdata/hf/tiny0.safetensors"}, nil},
		{"tiny_bag.safetensors", []string{"testdata/hf/tiny_bag0.safetensors", "testdata/hf/tiny_bag1.safetensors"}, [][]float32{{1, 0.25}, {0.5, 1}}},
	} {
		want, err := loadTiny(t, c.exported).Separate(mix, opts)
		if err != nil {
			t.Fatal(err)
		}
		hub, err := LoadFiles(c.hub, c.weights)
		if err != nil {
			t.Fatal(err)
		}
		got, err := hub.Separate(mix, opts)
		if err != nil {
			t.Fatal(err)
		}
		assertClose(t, c.exported+" via hub files", flatten(got.Stems), flatten(want.Stems), 1e-6)
	}
	single, err := Load("testdata/hf/tiny0.safetensors")
	if err != nil || len(single.Sources) != 2 {
		t.Fatalf("Load must accept a single hub file: %v", err)
	}
	if _, err := LoadFiles([]string{"testdata/tiny.safetensors"}, nil); err == nil {
		t.Fatal("an exporter file is not a hub file")
	}
}

func TestSeparateAtResamples(t *testing.T) {
	m := loadTiny(t, "tiny.safetensors")
	n := 48000 / 2
	mix := [][]float32{make([]float32, n), make([]float32, n)}
	for i := range mix[0] {
		mix[0][i] = 0.1 * float32(i%100) / 100
		mix[1][i] = -mix[0][i]
	}
	opts := DefaultOptions()
	opts.Shifts = 0
	sep, err := m.SeparateAt(mix, 48000, opts)
	if err != nil {
		t.Fatal(err)
	}
	if want := n * 44100 / 48000; len(sep.Stems[0][0]) != want {
		t.Fatalf("stems at the model rate: %d samples, want %d", len(sep.Stems[0][0]), want)
	}
}
