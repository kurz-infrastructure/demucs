package demucs

import (
	"runtime"
	"sync"
)

// parallelFor splits [0, n) into contiguous ranges, one per worker.
func parallelFor(n, workers int, fn func(lo, hi int)) {
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		fn(0, n)
		return
	}
	var wg sync.WaitGroup
	chunk := (n + workers - 1) / workers
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// parallelElems runs fn over [0, n) on all CPUs, unless n is too small to be
// worth the goroutine overhead.
func parallelElems(n int, fn func(lo, hi int)) {
	parallelFor(n, workersFor(n), fn)
}

func numWorkers() int { return runtime.GOMAXPROCS(0) }

// scaleMatrix computes C *= beta for an m×n matrix with stride ldc.
func scaleMatrix(c []float32, m, n, ldc int, beta float32) {
	if beta == 1 {
		return
	}
	for i := 0; i < m; i++ {
		row := c[i*ldc : i*ldc+n]
		if beta == 0 {
			clear(row)
			continue
		}
		for j := range row {
			row[j] *= beta
		}
	}
}
