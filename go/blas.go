package demucs

// All matrix products in the model go through sgemm, which has two
// implementations selected at build time:
//
//   - blas_accelerate.go: Apple Accelerate (cblas_sgemm) via cgo, used on
//     darwin when cgo is enabled. This runs on the AMX units of Apple Silicon.
//   - blas_purego.go: a portable, multi-threaded pure Go fallback, used
//     everywhere else and when building with the `demucs_purego` tag.
//
// sgemm computes C = alpha*op(A)*op(B) + beta*C on row-major matrices, where
// op(A) is m×k, op(B) is k×n and C is m×n, matching the cblas_sgemm contract.

// Backend reports which sgemm implementation this binary was built with.
func Backend() string { return backendName }
