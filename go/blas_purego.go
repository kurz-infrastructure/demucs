//go:build !darwin || !cgo || demucs_purego

package demucs

import "runtime"

const backendName = "purego"

const (
	gemmBlockK = 256
	gemmBlockN = 256
)

func sgemm(transA, transB bool, m, n, k int, alpha float32, a []float32, lda int,
	b []float32, ldb int, beta float32, c []float32, ldc int) {
	if m == 0 || n == 0 {
		return
	}
	scaleMatrix(c, m, n, ldc, beta)
	if k == 0 || alpha == 0 {
		return
	}
	// Bring both operands to the non transposed layout; the copy is O(mk + kn),
	// negligible against the O(mnk) product.
	if transA {
		a = transpose(a, k, m, lda)
		lda = k
	}
	if transB {
		b = transpose(b, n, k, ldb)
		ldb = n
	}
	workers := runtime.GOMAXPROCS(0)
	if flops := m * n * k; flops < 1<<16 {
		workers = 1
	}
	// Split rows across workers when there are enough of them, columns otherwise.
	if m >= 4*workers || n < 2*gemmBlockN {
		parallelFor(m, workers, func(lo, hi int) {
			gemmNN(lo, hi, 0, n, k, alpha, a, lda, b, ldb, c, ldc)
		})
		return
	}
	parallelFor(n, workers, func(lo, hi int) {
		gemmNN(0, m, lo, hi, k, alpha, a, lda, b, ldb, c, ldc)
	})
}

// transpose returns the cols×rows transpose of a rows×cols matrix with stride ld.
func transpose(src []float32, rows, cols, ld int) []float32 {
	dst := make([]float32, rows*cols)
	const blk = 32
	for i0 := 0; i0 < rows; i0 += blk {
		i1 := min(i0+blk, rows)
		for j0 := 0; j0 < cols; j0 += blk {
			j1 := min(j0+blk, cols)
			for i := i0; i < i1; i++ {
				row := src[i*ld:]
				for j := j0; j < j1; j++ {
					dst[j*rows+i] = row[j]
				}
			}
		}
	}
	return dst
}

// gemmNN accumulates alpha*A[i0:i1,:]*B[:,j0:j1] into C. B is packed in
// panels of gemmBlockK×gemmBlockN, laid out as strips of 4 columns so the 4×4
// micro-kernel reads it sequentially while its accumulators stay in registers.
func gemmNN(i0, i1, j0, j1, k int, alpha float32, a []float32, lda int,
	b []float32, ldb int, c []float32, ldc int) {
	packed := make([]float32, gemmBlockK*(gemmBlockN+4))
	for jb := j0; jb < j1; jb += gemmBlockN {
		je := min(jb+gemmBlockN, j1)
		strips := (je - jb + 3) / 4
		for pb := 0; pb < k; pb += gemmBlockK {
			kl := min(pb+gemmBlockK, k) - pb
			packB(packed, b, ldb, pb, kl, jb, je)
			for i := i0; i < i1; i += 4 {
				rows := min(4, i1-i)
				for s := 0; s < strips; s++ {
					j := jb + 4*s
					cols := min(4, je-j)
					panel := packed[s*kl*4 : (s+1)*kl*4]
					if rows == 4 {
						kernel4x4(a[i*lda+pb:], lda, kl, panel, alpha, c[i*ldc+j:], ldc, cols)
					} else {
						kernel1x4(a[i*lda+pb:], lda, rows, kl, panel, alpha, c[i*ldc+j:], ldc, cols)
					}
				}
			}
		}
	}
}

// packB copies B[pb:pb+kl, jb:je] into strips of 4 columns, zero padded.
func packB(dst, b []float32, ldb, pb, kl, jb, je int) {
	for s := 0; jb+4*s < je; s++ {
		j := jb + 4*s
		cols := min(4, je-j)
		strip := dst[s*kl*4 : (s+1)*kl*4]
		for p := 0; p < kl; p++ {
			row := b[(pb+p)*ldb+j:]
			q := strip[p*4 : p*4+4]
			if cols == 4 {
				q[0], q[1], q[2], q[3] = row[0], row[1], row[2], row[3]
				continue
			}
			for t := range q {
				q[t] = 0
				if t < cols {
					q[t] = row[t]
				}
			}
		}
	}
}

func kernel4x4(a []float32, lda, kl int, panel []float32, alpha float32, c []float32, ldc, cols int) {
	a0 := a[:kl]
	a1 := a[lda : lda+kl]
	a2 := a[2*lda : 2*lda+kl]
	a3 := a[3*lda : 3*lda+kl]
	var c00, c01, c02, c03, c10, c11, c12, c13 float32
	var c20, c21, c22, c23, c30, c31, c32, c33 float32
	panel = panel[:4*kl]
	for p := range a0 {
		bp := panel[4*p : 4*p+4 : 4*p+4]
		b0, b1, b2, b3 := bp[0], bp[1], bp[2], bp[3]
		x0, x1, x2, x3 := a0[p], a1[p], a2[p], a3[p]
		c00 += x0 * b0
		c01 += x0 * b1
		c02 += x0 * b2
		c03 += x0 * b3
		c10 += x1 * b0
		c11 += x1 * b1
		c12 += x1 * b2
		c13 += x1 * b3
		c20 += x2 * b0
		c21 += x2 * b1
		c22 += x2 * b2
		c23 += x2 * b3
		c30 += x3 * b0
		c31 += x3 * b1
		c32 += x3 * b2
		c33 += x3 * b3
	}
	acc := [4][4]float32{
		{c00, c01, c02, c03}, {c10, c11, c12, c13},
		{c20, c21, c22, c23}, {c30, c31, c32, c33},
	}
	for r := 0; r < 4; r++ {
		row := c[r*ldc : r*ldc+cols]
		for t := range row {
			row[t] += alpha * acc[r][t]
		}
	}
}

// kernel1x4 handles the last rows of A when fewer than 4 remain.
func kernel1x4(a []float32, lda, rows, kl int, panel []float32, alpha float32, c []float32, ldc, cols int) {
	panel = panel[:4*kl]
	for r := 0; r < rows; r++ {
		ar := a[r*lda : r*lda+kl]
		var c0, c1, c2, c3 float32
		for p, x := range ar {
			bp := panel[4*p : 4*p+4 : 4*p+4]
			c0 += x * bp[0]
			c1 += x * bp[1]
			c2 += x * bp[2]
			c3 += x * bp[3]
		}
		acc := [4]float32{c0, c1, c2, c3}
		row := c[r*ldc : r*ldc+cols]
		for t := range row {
			row[t] += alpha * acc[t]
		}
	}
}
