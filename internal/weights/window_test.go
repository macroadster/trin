package weights

import (
	"math"
	"path/filepath"
	"testing"
)

func TestLoadF32WindowF32(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "w.gguf")
	vals := []float32{1, 2, 3, 4, 5, 6}
	if err := WriteGGUFF32(path, map[string][]float32{"weight": vals},
		map[string][]uint64{"weight": {2, 3}}, nil); err != nil {
		t.Fatal(err)
	}
	f, err := OpenGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	w, err := f.LoadF32Window("weight", 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(w) != 3 || w[0] != 3 || w[1] != 4 || w[2] != 5 {
		t.Fatalf("got %v", w)
	}
	// multi-window from same open
	a, _ := f.LoadF32Window("weight", 0, 2)
	b, _ := f.LoadF32Window("weight", 4, 2)
	if a[0] != 1 || b[1] != 6 {
		t.Fatalf("a=%v b=%v", a, b)
	}
	if _, err := f.LoadF32Window("weight", 5, 2); err == nil {
		t.Fatal("expected OOB")
	}
	if _, err := f.LoadF32Window("weight", -1, 1); err == nil {
		t.Fatal("expected neg OOB")
	}
}

func TestLoadF32WindowQ4(t *testing.T) {
	// 64 elems = 2 Q4_0 blocks
	src := make([]float32, 64)
	for i := range src {
		src[i] = float32(i-32) * 0.1
	}
	raw := packQ4_0(src)
	// full decode reference
	full, err := DecodeGGMLBlock(GGML_Q4_0, raw, 64)
	if err != nil {
		t.Fatal(err)
	}
	// window that straddles both blocks: elems 20..50 (30 elems)
	win, err := DecodeGGMLWindow(GGML_Q4_0, raw, 64, 20, 30)
	if err != nil {
		t.Fatal(err)
	}
	if len(win) != 30 {
		t.Fatalf("len %d", len(win))
	}
	for i := 0; i < 30; i++ {
		if math.Abs(float64(win[i]-full[20+i])) > 1e-6 {
			t.Fatalf("i=%d got %v want %v", i, win[i], full[20+i])
		}
	}
	// window only first block
	w0, err := DecodeGGMLWindow(GGML_Q4_0, raw, 64, 0, 16)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 16; i++ {
		if math.Abs(float64(w0[i]-full[i])) > 1e-6 {
			t.Fatalf("w0 i=%d", i)
		}
	}
	if _, err := DecodeGGMLWindow(GGML_Q4_0, raw, 64, 60, 10); err == nil {
		t.Fatal("expected OOB")
	}
}

func TestSessionWindowMulti(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "toy.gguf")
	// 64-elem Q4_0
	src := make([]float32, 64)
	for i := range src {
		src[i] = float32(i) * 0.05
	}
	raw := packQ4_0(src)
	if err := WriteGGUFTyped(path, "weight", GGML_Q4_0, []uint64{64}, raw, nil); err != nil {
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
	w1, err := s.LoadF32Window("weight", 0, 32)
	if err != nil {
		t.Fatal(err)
	}
	w2, err := s.LoadF32Window("weight", 32, 32)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 32; i++ {
		if math.Abs(float64(w1[i]-full[i])) > 1e-5 {
			t.Fatalf("w1 %d", i)
		}
		if math.Abs(float64(w2[i]-full[32+i])) > 1e-5 {
			t.Fatalf("w2 %d", i)
		}
	}
}

func TestSafeTensorsWindow(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "w.safetensors")
	vals := []float32{10, 20, 30, 40, 50}
	if err := WriteSafeTensors(path, map[string][]float32{"a": vals}, map[string][]int64{"a": {5}}); err != nil {
		t.Fatal(err)
	}
	f, err := OpenSafeTensors(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	w, err := f.LoadF32Window("a", 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(w) != 3 || w[0] != 20 || w[2] != 40 {
		t.Fatalf("%v", w)
	}
	if _, err := f.LoadF32Window("a", 4, 2); err == nil {
		t.Fatal("expected OOB")
	}
}
