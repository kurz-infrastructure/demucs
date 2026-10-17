package demucs

import (
	"math"
	"testing"
)

// Reference values from julius.resample_frac (julius 0.2.7, torch 2.14) on
// 0.5 sin(2 pi 440 t) + 0.25 sin(2 pi 3000 t) + 0.1 cos(2 pi 7 t) sampled at
// the old rate, and on arange(n)/n for the small integer case.
var juliusCases = []struct {
	old, nw, n      int
	outLen          int
	head, mid, tail []float32
}{
	{48000, 44100, 1000, 918,
		[]float32{0.1042974665760994, 0.23274189233779907, 0.3529747426509857, 0.4314899742603302, 0.4731743335723877, 0.4637582302093506, 0.4212794899940491, 0.34861546754837036},
		[]float32{0.09674613177776337, 0.06402178108692169, -0.011041104793548584, -0.11951441317796707},
		[]float32{0.6272552013397217, 0.6722253561019897, 0.6778014898300171, 0.6303335428237915}},
	{44100, 22050, 777, 388,
		[]float32{0.12466710805892944, 0.3456783890724182, 0.4749184548854828, 0.41748151183128357, 0.2739926874637604, 0.16500981152057648, 0.21470907330513, 0.40947723388671875},
		[]float32{-0.11550604552030563, -0.2711578905582428, -0.3890033960342407, -0.3538781702518463},
		[]float32{-0.10461495071649551, -0.21376946568489075, -0.43265095353126526, -0.6259666681289673}},
	{22050, 44100, 500, 1000,
		[]float32{0.1040152832865715, 0.20960259437561035, 0.3471592962741852, 0.4456009268760681, 0.4755275845527649, 0.45760032534599304, 0.4163355529308319, 0.35459792613983154},
		[]float32{0.07357887178659439, 0.20617862045764923, 0.31671714782714844, 0.39082369208335876},
		[]float32{-0.38361063599586487, -0.30896204710006714, -0.2391318529844284, -0.21732790768146515}},
	{3, 2, 20, 13,
		[]float32{0.005209659691900015, 0.07362610846757889, 0.15092483162879944, 0.22419004142284393, 0.3007897138595581, 0.3741966784000397, 0.45084840059280396, 0.5240760445594788},
		[]float32{0.45084840059280396, 0.5240760445594788, 0.6010416746139526, 0.6737667322158813},
		[]float32{0.6737667322158813, 0.751555860042572, 0.8227747678756714, 0.9041556715965271}},
}

func testSignal(old, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		if old <= 100 {
			x[i] = float32(i) / float32(n)
			continue
		}
		t := float64(i) / float64(old)
		x[i] = float32(0.5*math.Sin(2*math.Pi*440*t) + 0.25*math.Sin(2*math.Pi*3000*t) + 0.1*math.Cos(2*math.Pi*7*t))
	}
	return x
}

func TestResampleMatchesJulius(t *testing.T) {
	for _, c := range juliusCases {
		y := Resample(testSignal(c.old, c.n), c.old, c.nw)
		if len(y) != c.outLen {
			t.Fatalf("%d->%d: %d samples, want %d", c.old, c.nw, len(y), c.outLen)
		}
		check := func(what string, got, want []float32) {
			for i := range want {
				if d := math.Abs(float64(got[i] - want[i])); d > 2e-5 {
					t.Fatalf("%d->%d %s[%d] = %v, julius %v", c.old, c.nw, what, i, got[i], want[i])
				}
			}
		}
		if c.head != nil {
			check("head", y, c.head)
		}
		if c.mid != nil {
			check("mid", y[len(y)/2:], c.mid)
		}
		if c.tail != nil {
			check("tail", y[len(y)-len(c.tail):], c.tail)
		}
	}
}

func TestResampleIdentityAndChannels(t *testing.T) {
	x := testSignal(44100, 100)
	if y := Resample(x, 44100, 44100); &y[0] != &x[0] {
		t.Fatal("equal rates must return the input")
	}
	if y := Resample(nil, 48000, 44100); len(y) != 0 {
		t.Fatal("empty input")
	}
	ch := ResampleChannels([][]float32{x, x}, 44100, 48000)
	if len(ch) != 2 || len(ch[0]) != 108 || ch[0][50] != ch[1][50] {
		t.Fatalf("channels: %d x %d", len(ch), len(ch[0]))
	}
	// a pure tone survives a 48k -> 44.1k -> 48k round trip to within a few percent after the edges
	tone := make([]float32, 4800)
	for i := range tone {
		tone[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / 48000))
	}
	back := Resample(Resample(tone, 48000, 44100), 44100, 48000)
	for i := 200; i < len(back)-200; i++ {
		if d := math.Abs(float64(back[i] - tone[i])); d > 0.03 {
			t.Fatalf("round trip sample %d off by %v", i, d)
		}
	}
}
