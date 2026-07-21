package main

import (
	"fmt"
	"math"
	"time"

	"github.com/macroadster/trin/internal/weights"
)

func denseMatvec(W, x []float32, M, K int) []float32 {
	out := make([]float32, M)
	for i := 0; i < M; i++ {
		var acc float64
		base := i * K
		for t := 0; t < K; t++ {
			acc += float64(W[base+t]) * float64(x[t])
		}
		out[i] = float32(acc)
	}
	return out
}

func bench(fn func(), iters int) time.Duration {
	start := time.Now()
	for i := 0; i < iters; i++ {
		fn()
	}
	return time.Since(start)
}

func main() {
	shapes := [][2]int{{64, 64}, {64, 256}, {64, 1024}, {256, 1024}, {1024, 1024}, {4096, 1024}, {1024, 4096}, {512, 512}}
	for _, sh := range shapes {
		M, K := sh[0], sh[1]
		Wf := make([]float32, M*K)
		Wt := make([]int8, M*K)
		x := make([]float32, K)
		for i := 0; i < M*K; i++ {
			v := float32(math.Sin(float64(i) * 0.01))
			Wf[i] = v
			if v >= 0.3 {
				Wt[i] = 1
			} else if v <= -0.3 {
				Wt[i] = -1
			}
		}
		for i := 0; i < K; i++ {
			x[i] = float32(math.Cos(float64(i) * 0.02))
		}
		scale := float32(0.5)
		pos, neg, nw := weights.PackTritMatrix(Wt, M, K)
		_ = denseMatvec(Wf, x, M, K)
		_ = weights.LinearTritScalar(Wt, x, M, K, scale)
		_ = weights.LinearTritPlanesPacked(pos, neg, x, M, K, nw, scale)
		iters := 20
		if M*K < 1<<18 {
			iters = 200
		}
		if M*K < 1<<14 {
			iters = 2000
		}
		if M*K < 1<<12 {
			iters = 10000
		}
		td := bench(func() { _ = denseMatvec(Wf, x, M, K) }, iters)
		ts := bench(func() { _ = weights.LinearTritScalar(Wt, x, M, K, scale) }, iters)
		tp := bench(func() { _ = weights.LinearTritPlanesPacked(pos, neg, x, M, K, nw, scale) }, iters)
		f32B := M*K*4 + K*4
		tritB := M*((K+31)/32)*8 + K*4
		fmt.Printf("M=%d K=%d iters=%d dense_us=%.2f scalar_us=%.2f packed_us=%.2f ratio_dense/packed=%.3f mem_f32=%d mem_pack=%d mem_ratio=%.2f pack_bits_per_w=%.2f\n",
			M, K, iters,
			float64(td.Microseconds())/float64(iters),
			float64(ts.Microseconds())/float64(iters),
			float64(tp.Microseconds())/float64(iters),
			float64(td)/float64(tp),
			f32B, tritB, float64(f32B)/float64(tritB),
			float64(tritB*8)/float64(M*K),
		)
	}
}
