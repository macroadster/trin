package weights

import (
	"math"
	"testing"
)

func TestLinearTritScalarVsPlanes(t *testing.T) {
	// M=3 K=140 (covers 4-wide word chunks + residual)
	M, K := 3, 140
	W := make([]int8, M*K)
	x := make([]float32, K)
	pat := []int8{1, 0, -1, 1, -1, 0, 1, 1}
	for i := 0; i < M*K; i++ {
		W[i] = pat[i%len(pat)]
	}
	for i := 0; i < K; i++ {
		x[i] = 0.1*float32(i%7-3) + 0.05
	}
	scale := float32(0.37)
	a := LinearTritScalar(W, x, M, K, scale)
	b := LinearTritPlanes(W, x, M, K, scale)
	if len(a) != M || len(b) != M {
		t.Fatalf("len a=%d b=%d", len(a), len(b))
	}
	for i := 0; i < M; i++ {
		if math.Abs(float64(a[i]-b[i])) > 1e-5 {
			t.Fatalf("row %d scalar=%v planes=%v", i, a[i], b[i])
		}
	}
	// hand-check first element vs pure sum
	var ref float64
	for t0 := 0; t0 < K; t0++ {
		ref += float64(W[t0]) * float64(x[t0])
	}
	ref *= float64(scale)
	if math.Abs(ref-float64(a[0])) > 1e-5 {
		t.Fatalf("ref %v got %v", ref, a[0])
	}
}

func TestLinearTritEmpty(t *testing.T) {
	if len(LinearTritPlanes(nil, nil, 0, 0, 1)) != 0 {
		t.Fatal("empty")
	}
	if len(LinearTritScalar([]int8{1}, []float32{2}, 1, 1, 3) ) != 1 {
		t.Fatal("1x1")
	}
}
