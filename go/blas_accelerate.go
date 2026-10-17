//go:build darwin && cgo && !demucs_purego

package demucs

/*
#cgo CFLAGS: -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>

static void demucs_sgemm(int transA, int transB, int m, int n, int k, float alpha,
                         const float *a, int lda, const float *b, int ldb, float beta,
                         float *c, int ldc) {
	cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
	            transB ? CblasTrans : CblasNoTrans, m, n, k, alpha, a, lda, b, ldb,
	            beta, c, ldc);
}
*/
import "C"

const backendName = "accelerate"

func sgemm(transA, transB bool, m, n, k int, alpha float32, a []float32, lda int,
	b []float32, ldb int, beta float32, c []float32, ldc int) {
	if m == 0 || n == 0 {
		return
	}
	if k == 0 {
		scaleMatrix(c, m, n, ldc, beta)
		return
	}
	C.demucs_sgemm(cbool(transA), cbool(transB), C.int(m), C.int(n), C.int(k), C.float(alpha),
		(*C.float)(&a[0]), C.int(lda), (*C.float)(&b[0]), C.int(ldb), C.float(beta),
		(*C.float)(&c[0]), C.int(ldc))
}

func cbool(v bool) C.int {
	if v {
		return 1
	}
	return 0
}
