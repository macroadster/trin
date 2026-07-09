package weights

import (
	"math"
	"path/filepath"
	"testing"
)

func approxEq(a, b, eps float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= eps
}

func TestLinearRowsF32(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "w.gguf")
	// W = [[1,2,3],[4,5,6]]  M=2 K=3
	vals := []float32{1, 2, 3, 4, 5, 6}
	if err := WriteGGUFF32(path, map[string][]float32{"weight": vals},
		map[string][]uint64{"weight": {2, 3}}, nil); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	x := []float32{1, 0, -1}
	// full
	y, err := s.Linear("weight", x, 2, 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	// y0 = 1-3 = -2; y1 = 4-6 = -2
	if !approxEq(y[0], -2, 1e-5) || !approxEq(y[1], -2, 1e-5) {
		t.Fatalf("full y=%v", y)
	}
	// tile row 1 only
	y1, err := s.LinearRows("weight", x, 3, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !approxEq(y1[0], -2, 1e-5) {
		t.Fatalf("tile y=%v", y1)
	}
	// multi-tile from one session
	a, _ := s.LinearRows("weight", x, 3, 0, 1)
	b, _ := s.LinearRows("weight", x, 3, 1, 1)
	if !approxEq(a[0]+b[0], -4, 1e-5) {
		t.Fatalf("a+b")
	}
	if _, err := s.LinearRows("weight", x, 3, 1, 2); err == nil {
		t.Fatal("expected OOB")
	}
	if _, err := s.LinearRows("weight", x, 3, -1, 1); err == nil {
		t.Fatal("expected neg OOB")
	}
}

func TestLinearRowsQ4(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "q.gguf")
	// 2x32 matrix from known pattern
	src := make([]float32, 64)
	for i := range src {
		src[i] = float32(i-32) * 0.1
	}
	raw := packQ4_0(src)
	if err := WriteGGUFTyped(path, "weight", GGML_Q4_0, []uint64{2, 32}, raw, nil); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	full, _, err := s.LoadF32("weight")
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float32, 32)
	for i := range x {
		x[i] = 0.05 * float32(i%5-2)
	}
	// reference matvec from full
	ref := make([]float32, 2)
	for r := 0; r < 2; r++ {
		acc := float32(0)
		for k := 0; k < 32; k++ {
			acc += full[r*32+k] * x[k]
		}
		ref[r] = acc
	}
	y, err := s.Linear("weight", x, 2, 32, 1) // one-row tiles
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		if math.Abs(float64(y[i]-ref[i])) > 1e-4 {
			t.Fatalf("row %d got %v want %v", i, y[i], ref[i])
		}
	}
	// single second row tile
	y1, err := s.LinearRows("weight", x, 32, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(y1[0]-ref[1])) > 1e-4 {
		t.Fatalf("tile row1 %v vs %v", y1[0], ref[1])
	}
}

func TestLinearGGUFOneShot(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "w.gguf")
	vals := []float32{1, 0, 0, 1, 0, 0} // 2x3 identity-ish rows
	if err := WriteGGUFF32(path, map[string][]float32{"W": vals},
		map[string][]uint64{"W": {2, 3}}, nil); err != nil {
		t.Fatal(err)
	}
	y, err := LinearGGUF(path, "W", []float32{3, 4, 5}, 2, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !approxEq(y[0], 3, 1e-5) || !approxEq(y[1], 3, 1e-5) {
		t.Fatalf("%v", y)
	}
}
