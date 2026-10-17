package demucs

import "math"

// expf is a float32 exp (Cephes expf), accurate to about one ulp and several
// times faster than going through math.Exp in float64. It dominates softmax
// and sigmoid cost.
func expf(x float32) float32 {
	const (
		log2e = 1.44269504088896341
		c1    = 0.693359375
		c2    = -2.12194440e-4
	)
	if x > 88.3762626647949 {
		return float32(math.Inf(1))
	}
	if x < -87.3365447505531 {
		return 0
	}
	fx := float32(math.Floor(float64(x*log2e + 0.5)))
	x -= fx * c1
	x -= fx * c2
	z := x * x
	y := ((((1.9875691500e-4*x+1.3981999507e-3)*x+8.3334519073e-3)*x+4.1665795894e-2)*x+
		1.6666665459e-1)*x + 5.0000001201e-1
	y = y*z + x + 1
	return y * math.Float32frombits(uint32(int32(fx)+127)<<23)
}
