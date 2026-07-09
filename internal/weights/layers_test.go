package weights

import (
	"math"
	"path/filepath"
	"testing"
)

func near(a, b float32) bool {
	return math.Abs(float64(a-b)) < 1e-4
}

func TestRMSNormF32(t *testing.T) {
	x := []float32{3, 4} // rms = 5/sqrt(1)? mean sq = (9+16)/2=12.5, sqrt=√12.5
	y := RMSNormF32(x, 0, nil)
	// inv = 1/sqrt(12.5)
	inv := float32(1 / math.Sqrt(12.5))
	if !near(y[0], 3*inv) || !near(y[1], 4*inv) {
		t.Fatalf("%v", y)
	}
	g := []float32{2, 0.5}
	yg := RMSNormF32(x, 0, g)
	if !near(yg[0], 3*inv*2) || !near(yg[1], 4*inv*0.5) {
		t.Fatalf("gain %v", yg)
	}
}

func TestLayerStackF32Parity(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "stack.gguf")
	// dim=3, two layers W0 and W1 (3x3 row-major)
	W0 := []float32{
		0.1, 0, 0,
		0, 0.2, 0,
		0, 0, 0.3,
	}
	W1 := []float32{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1, // identity → residual path tests cleanly
	}
	g0 := []float32{1, 1, 1}
	g1 := []float32{1, 1, 1}
	if err := WriteGGUFF32(path,
		map[string][]float32{"W0": W0, "W1": W1, "g0": g0, "g1": g1},
		map[string][]uint64{"W0": {3, 3}, "W1": {3, 3}, "g0": {3}, "g1": {3}},
		map[string]string{"general.name": "stack"},
	); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	x := []float32{1, 2, 3}
	eps := 1e-6
	// manual composition
	y0, err := s.LinearFused("W0", x, 3, 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	h0 := make([]float32, 3)
	for i := 0; i < 3; i++ {
		h0[i] = x[i] + y0[i]
	}
	g0v, _ := s.LoadF32Window("g0", 0, 3)
	n0 := RMSNormF32(h0, eps, g0v)
	y1, err := s.LinearFused("W1", n0, 3, 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	h1 := make([]float32, 3)
	for i := 0; i < 3; i++ {
		h1[i] = n0[i] + y1[i]
	}
	g1v, _ := s.LoadF32Window("g1", 0, 3)
	ref := RMSNormF32(h1, eps, g1v)

	got, err := s.LayerStackCSV("W0,W1", "g0,g1", "", x, 3, 1, eps)
	if err != nil {
		t.Fatal(err)
	}
	for i := range ref {
		if !near(ref[i], got[i]) {
			t.Fatalf("i=%d ref=%v got=%v", i, ref[i], got[i])
		}
	}

	// plain mode first layer only: residual disable
	plain, err := s.LayerStackCSV("W0", "g0", "plain", x, 3, 1, eps)
	if err != nil {
		t.Fatal(err)
	}
	// y=W0x then rms without residual
	pLin, _ := s.LinearFused("W0", x, 3, 3, 1)
	pref := RMSNormF32(pLin, eps, g0v)
	for i := range pref {
		if !near(pref[i], plain[i]) {
			t.Fatalf("plain i=%d %v vs %v", i, pref[i], plain[i])
		}
	}

	if _, err := s.LayerStackCSV("", "", "", x, 3, 1, eps); err == nil {
		t.Fatal("empty layers")
	}
	if _, err := s.LayerStackCSV("W0", "", "", x, 99, 1, eps); err == nil {
		t.Fatal("bad dim")
	}
}

func TestLayerStackQ4(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "qstack.gguf")
	// Use single Q4 weight via WriteGGUFTyped — only one tensor; for multi-layer
	// square Q needs nElem divisible: dim=32 weight 32x32 = 1024 elems = 32 blocks
	// For multi-tensor Q4, use two separate sessions or F32 stack already covered.
	// Here: one layer Q4 fused vs non-fused residual chain.
	src := make([]float32, 32*32)
	for i := range src {
		src[i] = float32((i%7)-3) * 0.05
	}
	raw := packQ4_0(src)
	if err := WriteGGUFTyped(path, "W", GGML_Q4_0, []uint64{32, 32}, raw, nil); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	x := make([]float32, 32)
	for i := range x {
		x[i] = 0.02 * float32(i-16)
	}
	// one residual block: fusion vs linear reference
	refLin, err := s.Linear("W", x, 32, 32, 4)
	if err != nil {
		t.Fatal(err)
	}
	h := make([]float32, 32)
	for i := 0; i < 32; i++ {
		h[i] = x[i] + refLin[i]
	}
	ref := RMSNormF32(h, 1e-6, nil)
	got, err := s.ForwardBlock(LayerSpec{WName: "W", Mode: "residual"}, x, 32, 4, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 32; i++ {
		if math.Abs(float64(ref[i]-got[i])) > 1e-3 {
			t.Fatalf("i=%d ref=%v got=%v", i, ref[i], got[i])
		}
	}
}

func TestLayerStackClosedSession(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "c.gguf")
	_ = WriteGGUFF32(path, map[string][]float32{"W": {1, 0, 0, 1}}, map[string][]uint64{"W": {2, 2}}, nil)
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	id := s.ID
	_ = CloseSession(id)
	if _, err := GetSession(id); err == nil {
		t.Fatal("expected closed")
	}
}
