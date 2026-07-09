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

func TestLayerStackTritParityAndCache(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "stack_trit.gguf")
	// dim=4 two square layers — larger than tiny so packed words exercise
	dim := 4
	W0 := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	W1 := []float32{
		0.5, 0, 0, 0,
		0, 0.5, 0, 0,
		0, 0, 0.5, 0,
		0, 0, 0.5, 0.5,
	}
	g0 := []float32{1, 1, 1, 1}
	g1 := []float32{1, 1, 1, 1}
	if err := WriteGGUFF32(path,
		map[string][]float32{"W0": W0, "W1": W1, "g0": g0, "g1": g1},
		map[string][]uint64{"W0": {4, 4}, "W1": {4, 4}, "g0": {4}, "g1": {4}},
		map[string]string{"general.name": "stack-trit"},
	); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)

	x := []float32{1, 2, 3, 4}
	eps := 1e-6

	// prepack once
	built, hits, err := s.PrepackTritCSV("W0,W1", dim)
	if err != nil {
		t.Fatal(err)
	}
	if built != 2 || hits != 0 {
		t.Fatalf("prepack first built=%d hits=%d", built, hits)
	}
	built2, hits2, err := s.PrepackTritCSV("W0,W1", dim)
	if err != nil {
		t.Fatal(err)
	}
	if built2 != 0 || hits2 != 2 {
		t.Fatalf("prepack second built=%d hits=%d", built2, hits2)
	}

	// manual composition via LinearTritCached full-tile absmean
	refManual := func(bitplane bool) []float32 {
		y0, _, err := s.LinearTritCached("W0", x, dim, dim, bitplane)
		if err != nil {
			t.Fatal(err)
		}
		h0 := make([]float32, dim)
		for i := 0; i < dim; i++ {
			h0[i] = x[i] + y0[i]
		}
		n0 := RMSNormF32(h0, eps, g0)
		y1, _, err := s.LinearTritCached("W1", n0, dim, dim, bitplane)
		if err != nil {
			t.Fatal(err)
		}
		h1 := make([]float32, dim)
		for i := 0; i < dim; i++ {
			h1[i] = n0[i] + y1[i]
		}
		return RMSNormF32(h1, eps, g1)
	}
	for _, bp := range []bool{false, true} {
		ref := refManual(bp)
		got, err := s.LayerStackTritCSV("W0,W1", "g0,g1", "residual,residual", x, dim, bp, eps)
		if err != nil {
			t.Fatal(err)
		}
		for i := range ref {
			if !near(ref[i], got[i]) {
				t.Fatalf("bp=%v i=%d ref=%v got=%v", bp, i, ref[i], got[i])
			}
		}
		// block stack compose
		b0, err := s.ForwardBlockTrit(LayerSpec{WName: "W0", GainName: "g0", Mode: "residual"}, x, dim, bp, eps)
		if err != nil {
			t.Fatal(err)
		}
		b1, err := s.ForwardBlockTrit(LayerSpec{WName: "W1", GainName: "g1", Mode: "residual"}, b0, dim, bp, eps)
		if err != nil {
			t.Fatal(err)
		}
		for i := range ref {
			if !near(ref[i], b1[i]) {
				t.Fatalf("block compose bp=%v i=%d", bp, i)
			}
		}
	}

	// plain mode: no residual
	plain, err := s.ForwardBlockTrit(LayerSpec{WName: "W0", GainName: "g0", Mode: "plain"}, x, dim, true, eps)
	if err != nil {
		t.Fatal(err)
	}
	ylin, _, err := s.LinearTritCached("W0", x, dim, dim, true)
	if err != nil {
		t.Fatal(err)
	}
	pref := RMSNormF32(ylin, eps, g0)
	for i := range pref {
		if !near(pref[i], plain[i]) {
			t.Fatalf("plain i=%d %v vs %v", i, pref[i], plain[i])
		}
	}

	// packs still present
	if !s.HasTritPack("W0") || !s.HasTritPack("W1") {
		t.Fatal("packs dropped unexpectedly")
	}

	if _, err := s.LayerStackTritCSV("", "", "", x, dim, true, eps); err == nil {
		t.Fatal("empty layers")
	}
	if _, err := s.LayerStackTritCSV("W0", "", "", x, 99, true, eps); err == nil {
		t.Fatal("bad dim")
	}
}

func TestAttnStackTritParityAndGrow(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "attn_stack.gguf")
	// 2 residual layers, dim=2, hidden=4 — shared identity-ish shapes.
	dim, hidden := 2, 4
	I2 := []float32{1, 0, 0, 1}
	K0 := []float32{1, 0.1, 0, 1}
	V0 := []float32{0.8, 0, 0, 0.8}
	// up: hidden×dim = 4×2
	Up0 := []float32{
		1, 0,
		0, 1,
		0.5, 0.5,
		0, 0.5,
	}
	// down: dim×hidden = 2×4
	Down0 := []float32{
		1, 0, 0.5, 0,
		0, 1, 0, 0.5,
	}
	// layer1 slight scale
	Q1 := []float32{0.9, 0, 0, 0.9}
	K1 := []float32{1, 0, 0.1, 1}
	V1 := []float32{0.7, 0, 0, 0.7}
	O1 := []float32{1, 0, 0, 1}
	Up1 := []float32{
		0.8, 0,
		0, 0.8,
		0.4, 0.4,
		0, 0.4,
	}
	Down1 := []float32{
		0.9, 0, 0.4, 0,
		0, 0.9, 0, 0.4,
	}
	g := []float32{1, 1}
	tensors := map[string][]float32{
		"L0_q": Q1, "L0_k": K0, "L0_v": V0, "L0_o": I2,
		"L0_up": Up0, "L0_down": Down0, "L0_an": g, "L0_fn": g,
		"L1_q": Q1, "L1_k": K1, "L1_v": V1, "L1_o": O1,
		"L1_up": Up1, "L1_down": Down1, "L1_an": g, "L1_fn": g,
	}
	// fix L0_q identity
	tensors["L0_q"] = I2
	shapes := map[string][]uint64{
		"L0_q": {2, 2}, "L0_k": {2, 2}, "L0_v": {2, 2}, "L0_o": {2, 2},
		"L0_up": {4, 2}, "L0_down": {2, 4}, "L0_an": {2}, "L0_fn": {2},
		"L1_q": {2, 2}, "L1_k": {2, 2}, "L1_v": {2, 2}, "L1_o": {2, 2},
		"L1_up": {4, 2}, "L1_down": {2, 4}, "L1_an": {2}, "L1_fn": {2},
	}
	if err := WriteGGUFF32(path, tensors, shapes, map[string]string{"general.name": "attn-stack"}); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)

	x := []float32{1.5, -0.5}
	eps := 1e-6
	// prepack all matrices (K = in features)
	built, hits, err := s.PrepackTritCSV("L0_q,L0_k,L0_v,L0_o,L1_q,L1_k,L1_v,L1_o", dim)
	if err != nil {
		t.Fatal(err)
	}
	if built != 8 || hits != 0 {
		t.Fatalf("attn packs built=%d hits=%d", built, hits)
	}
	bUp, hUp, err := s.PrepackTritCSV("L0_up,L1_up", dim)
	if err != nil {
		t.Fatal(err)
	}
	if bUp != 2 || hUp != 0 {
		t.Fatalf("up packs built=%d hits=%d", bUp, hUp)
	}
	bDn, hDn, err := s.PrepackTritCSV("L0_down,L1_down", hidden)
	if err != nil {
		t.Fatal(err)
	}
	if bDn != 2 || hDn != 0 {
		t.Fatalf("down packs built=%d hits=%d", bDn, hDn)
	}
	// second prepack all hits
	b2, h2, err := s.PrepackTritCSV("L0_q,L0_k,L0_v,L0_o,L0_up,L1_q,L1_k,L1_v,L1_o,L1_up", dim)
	if err != nil {
		t.Fatal(err)
	}
	if b2 != 0 || h2 != 10 {
		t.Fatalf("second pack built=%d hits=%d", b2, h2)
	}

	// manual one-layer compose matches ForwardAttnBlockTrit
	manual := func(spec AttnLayerSpec, xin, kp, vp []float32, pos float64, bp bool) ([]float32, []float32, []float32) {
		ag, err := s.loadGain(spec.AttnGain, dim)
		if err != nil {
			t.Fatal(err)
		}
		xn := RMSNormF32(xin, eps, ag)
		a, kn, vn, err := s.SessionAttnTrit(spec.Q, spec.K, spec.V, spec.O, xn, kp, vp, dim, dim, 1, pos, 0, bp)
		if err != nil {
			t.Fatal(err)
		}
		h := make([]float32, dim)
		for i := 0; i < dim; i++ {
			h[i] = xin[i] + a[i]
		}
		fg, err := s.loadGain(spec.FFNGain, dim)
		if err != nil {
			t.Fatal(err)
		}
		hn := RMSNormF32(h, eps, fg)
		up, _, err := s.LinearTritCached(spec.Up, hn, hidden, dim, bp)
		if err != nil {
			t.Fatal(err)
		}
		act := SiLUF32(up)
		down, _, err := s.LinearTritCached(spec.Down, act, dim, hidden, bp)
		if err != nil {
			t.Fatal(err)
		}
		y := make([]float32, dim)
		for i := 0; i < dim; i++ {
			y[i] = h[i] + down[i]
		}
		return y, kn, vn
	}
	l0 := AttnLayerSpec{Q: "L0_q", K: "L0_k", V: "L0_v", O: "L0_o", AttnGain: "L0_an", FFNGain: "L0_fn", Up: "L0_up", Down: "L0_down", Hidden: hidden}
	l1 := AttnLayerSpec{Q: "L1_q", K: "L1_k", V: "L1_v", O: "L1_o", AttnGain: "L1_an", FFNGain: "L1_fn", Up: "L1_up", Down: "L1_down", Hidden: hidden}

	for _, bp := range []bool{false, true} {
		y0m, k0, v0 := manual(l0, x, nil, nil, 0, bp)
		y0, k0b, v0b, err := s.ForwardAttnBlockTrit(l0, x, nil, nil, dim, 1, 0, 0, bp, eps)
		if err != nil {
			t.Fatal(err)
		}
		for i := range y0m {
			if !near(y0m[i], y0[i]) {
				t.Fatalf("block0 bp=%v i=%d %v vs %v", bp, i, y0m[i], y0[i])
			}
		}
		if len(k0) != len(k0b) || len(v0) != len(v0b) {
			t.Fatalf("kv0 len")
		}
		y1m, k1, v1 := manual(l1, y0m, nil, nil, 0, bp)
		ys, kns, vns, err := s.AttnStackTrit([]AttnLayerSpec{l0, l1}, x, nil, nil, dim, 1, 0, 0, bp, eps)
		if err != nil {
			t.Fatal(err)
		}
		for i := range y1m {
			if !near(y1m[i], ys[i]) {
				t.Fatalf("stack bp=%v i=%d %v vs %v", bp, i, y1m[i], ys[i])
			}
		}
		if len(kns) != 2 || len(vns) != 2 {
			t.Fatalf("stack kv parts")
		}
		if len(kns[0]) != len(k0) || len(kns[1]) != len(k1) {
			t.Fatalf("stack kv sizes")
		}
		// grow past
		ys2, kns2, _, err := s.AttnStackTrit([]AttnLayerSpec{l0, l1}, x, kns, vns, dim, 1, 1, 0, bp, eps)
		if err != nil {
			t.Fatal(err)
		}
		if len(kns2[0]) != 2*dim || len(kns2[1]) != 2*dim {
			t.Fatalf("grow want 2 toks/layer got %d %d", len(kns2[0])/dim, len(kns2[1])/dim)
		}
		if len(ys2) != dim {
			t.Fatalf("y2 len")
		}
		// CSV surface + flat KV roundtrip
		yCSV, kFlat, vFlat, err := s.AttnStackTritCSV(
			"L0_q,L1_q", "L0_k,L1_k", "L0_v,L1_v", "L0_o,L1_o",
			"L0_up,L1_up", "L0_down,L1_down", "L0_an,L1_an", "L0_fn,L1_fn",
			x, nil, nil, dim, hidden, 1, 0, 0, bp, eps,
		)
		if err != nil {
			t.Fatal(err)
		}
		for i := range ys {
			if !near(ys[i], yCSV[i]) {
				t.Fatalf("csv y bp=%v i=%d", bp, i)
			}
		}
		if len(kFlat) != 2*dim || len(vFlat) != 2*dim {
			// two layers × 1 token × dim
			t.Fatalf("flat kv len %d %d", len(kFlat), len(vFlat))
		}
		parts, err := SplitLayerKV(kFlat, 2, dim)
		if err != nil {
			t.Fatal(err)
		}
		if len(parts[0]) != dim || len(parts[1]) != dim {
			t.Fatalf("split lens")
		}
		// grow via CSV flat
		_, kFlat2, _, err := s.AttnStackTritCSV(
			"L0_q,L1_q", "L0_k,L1_k", "L0_v,L1_v", "L0_o,L1_o",
			"L0_up,L1_up", "L0_down,L1_down", "L0_an,L1_an", "L0_fn,L1_fn",
			x, kFlat, vFlat, dim, hidden, 1, 1, 0, bp, eps,
		)
		if err != nil {
			t.Fatal(err)
		}
		if len(kFlat2) != 4*dim {
			t.Fatalf("flat grow want %d got %d", 4*dim, len(kFlat2))
		}
		_ = k1
		_ = v1
	}

	if !s.HasTritPack("L0_q") || !s.HasTritPack("L1_down") {
		t.Fatal("packs dropped")
	}
	if _, _, _, err := s.AttnStackTritCSV("", "", "", "", "", "", "", "", x, nil, nil, dim, hidden, 1, 0, 0, true, eps); err == nil {
		t.Fatal("empty layers")
	}
	if _, _, _, err := s.AttnStackTritCSV("L0_q", "L0_k", "L0_v", "L0_o", "L0_up", "L0_down", "", "", x, nil, nil, 99, hidden, 1, 0, 0, true, eps); err == nil {
		t.Fatal("bad dim")
	}
}
