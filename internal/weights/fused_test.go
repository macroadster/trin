package weights

import (
	"math"
	"path/filepath"
	"testing"
)

func TestDotGGMLWindowF32(t *testing.T) {
	// W = [1,2,3,4], x partial
	raw := make([]byte, 16)
	for i, v := range []float32{1, 2, 3, 4} {
		binaryPutF32(raw[i*4:], v)
	}
	x := []float32{10, 1, 0.5}
	// sum W[1:4]*x = 2*10 + 3*1 + 4*0.5 = 20+3+2 = 25
	v, err := DotGGMLWindow(GGML_F32, raw, 4, 1, 3, x)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(v-25)) > 1e-5 {
		t.Fatalf("got %v want 25", v)
	}
	if _, err := DotGGMLWindow(GGML_F32, raw, 4, 2, 3, x); err == nil {
		t.Fatal("expected OOB")
	}
}

func binaryPutF32(b []byte, v float32) {
	u := math.Float32bits(v)
	b[0] = byte(u)
	b[1] = byte(u >> 8)
	b[2] = byte(u >> 16)
	b[3] = byte(u >> 24)
}

func TestLinearRowsFusedF32Parity(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "w.gguf")
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
	x := []float32{1, -1, 0.5}
	ref, err := s.Linear("weight", x, 2, 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	got, err := s.LinearFused("weight", x, 2, 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i := range ref {
		if math.Abs(float64(ref[i]-got[i])) > 1e-5 {
			t.Fatalf("i=%d ref=%v got=%v", i, ref[i], got[i])
		}
	}
	// multi-tile
	a, _ := s.LinearRowsFused("weight", x, 3, 0, 1)
	b, _ := s.LinearRowsFused("weight", x, 3, 1, 1)
	if math.Abs(float64(a[0]+b[0]-ref[0]-ref[1])) > 1e-5 {
		t.Fatal("multi-tile")
	}
	if _, err := s.LinearRowsFused("weight", x, 3, 1, 2); err == nil {
		t.Fatal("expected OOB")
	}
}

func TestLinearRowsFusedQ4Parity(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "q.gguf")
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
	x := make([]float32, 32)
	for i := range x {
		x[i] = 0.05 * float32(i%5-2)
	}
	ref, err := s.Linear("weight", x, 2, 32, 2) // non-fused full-window path
	if err != nil {
		t.Fatal(err)
	}
	got, err := s.LinearFused("weight", x, 2, 32, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		if math.Abs(float64(ref[i]-got[i])) > 1e-4 {
			t.Fatalf("row %d ref=%v got=%v", i, ref[i], got[i])
		}
	}
	// single second row tile
	y1, err := s.LinearRowsFused("weight", x, 32, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(y1[0]-ref[1])) > 1e-4 {
		t.Fatalf("tile %v vs %v", y1[0], ref[1])
	}
}

func TestLinearGGUFFusedOneShot(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "w.gguf")
	vals := []float32{2, 0, 0, 0, 3, 0}
	if err := WriteGGUFF32(path, map[string][]float32{"W": vals},
		map[string][]uint64{"W": {2, 3}}, nil); err != nil {
		t.Fatal(err)
	}
	y, err := LinearGGUFFused(path, "W", []float32{1, 1, 1}, 2, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(y[0]-2)) > 1e-5 || math.Abs(float64(y[1]-3)) > 1e-5 {
		t.Fatalf("%v", y)
	}
}
