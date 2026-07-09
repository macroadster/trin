package cuda

import (
	"math"
	"os"
	"testing"
)

func TestProbeForceOff(t *testing.T) {
	t.Setenv("TRIN_CUDA", "0")
	ResetProbeForTest()
	if Available() {
		t.Fatalf("expected Available=false under TRIN_CUDA=0")
	}
	if DeviceCount() != 0 {
		t.Fatalf("count=%d", DeviceCount())
	}
	if Backend() != BackendCPUFallback {
		t.Fatalf("backend=%s", Backend())
	}
	if Reason() == "" {
		t.Fatalf("empty reason")
	}
}

func TestLinearF32Known(t *testing.T) {
	// W 2x3: [[1,2,3],[4,5,6]] ; x=[1,0,-1] → y=[1*1+2*0+3*(-1), 4+0-6]=[-2,-2]
	W := []float32{1, 2, 3, 4, 5, 6}
	x := []float32{1, 0, -1}
	y, err := LinearF32(W, x, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(y) != 2 || y[0] != -2 || y[1] != -2 {
		t.Fatalf("y=%v", y)
	}
}

func TestGemmF32Identity(t *testing.T) {
	// 2x2 identity @ [[1,2],[3,4]] = same
	A := []float32{1, 0, 0, 1}
	B := []float32{1, 2, 3, 4}
	C, err := GemmF32(A, B, 2, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 2, 3, 4}
	for i := range want {
		if math.Abs(float64(C[i]-want[i])) > 1e-6 {
			t.Fatalf("C=%v want %v", C, want)
		}
	}
}

func TestGemmF32Larger(t *testing.T) {
	M, K, N := 17, 19, 13 // force multi-tile peels
	A := make([]float32, M*K)
	B := make([]float32, K*N)
	for i := range A {
		A[i] = float32((i%7)-3) * 0.5
	}
	for i := range B {
		B[i] = float32((i%5)-2) * 0.25
	}
	C, err := GemmF32(A, B, M, K, N)
	if err != nil {
		t.Fatal(err)
	}
	// reference naive
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var acc float32
			for k := 0; k < K; k++ {
				acc += A[i*K+k] * B[k*N+j]
			}
			if math.Abs(float64(C[i*N+j]-acc)) > 1e-4 {
				t.Fatalf("mismatch at %d,%d got %v want %v", i, j, C[i*N+j], acc)
			}
		}
	}
}

func TestLinearTrit(t *testing.T) {
	// W = [[1,0,-1],[0,1,1]] ; x=[1,2,3] ; scale=2
	// row0: 1*1 + 0*2 + (-1)*3 = -2 → -4
	// row1: 0*1 + 1*2 + 1*3 = 5 → 10
	W := []int32{1, 0, -1, 0, 1, 1}
	x := []float32{1, 2, 3}
	y, err := LinearTrit(W, x, 2, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	if y[0] != -4 || y[1] != 10 {
		t.Fatalf("y=%v", y)
	}
}

func TestDimMismatch(t *testing.T) {
	if _, err := LinearF32([]float32{1}, []float32{1, 2}, 2, 2); err == nil {
		t.Fatal("expected dim error")
	}
	if _, err := GemmF32([]float32{1}, []float32{1}, 2, 2, 2); err == nil {
		t.Fatal("expected dim error")
	}
}

func TestDeviceNameEmptyWithoutGPU(t *testing.T) {
	// Ensure force-off path; host has no GPU in CI.
	os.Setenv("TRIN_CUDA", "0")
	ResetProbeForTest()
	if DeviceName(0) != "" {
		t.Fatalf("name=%q", DeviceName(0))
	}
}
