package weights

import (
	"math"
	"path/filepath"
	"testing"
	"time"
)

func approxEq(a, b, eps float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= eps
}

func TestLinearTritSessionParity(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "trit.gguf")
	// M=4 K=140 covers multiword 4-wide chunks + residual.
	M, K := 4, 140
	vals := make([]float32, M*K)
	pat := []float32{1.2, 0, -0.9, 0.4, -1.1, 0.05, 0.8, -0.3}
	for i := range vals {
		vals[i] = pat[i%len(pat)]
	}
	if err := WriteGGUFF32(path, map[string][]float32{"weight": vals},
		map[string][]uint64{"weight": {uint64(M), uint64(K)}}, nil); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	x := make([]float32, K)
	for i := range x {
		x[i] = 0.1*float32(i%7-3) + 0.05
	}
	ys, scs, err := s.LinearTrit("weight", x, M, K, 2, false)
	if err != nil {
		t.Fatal(err)
	}
	yp, scp, err := s.LinearTrit("weight", x, M, K, 2, true)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(scs-scp)) > 1e-6 {
		t.Fatalf("scale mismatch scalar=%v plane=%v", scs, scp)
	}
	// Reference: full-window quantize + scalar once
	full, _, err := s.LoadF32("weight")
	if err != nil {
		t.Fatal(err)
	}
	// Note: per-tile quant uses each window's absmean; tileRows=M forces one tile
	y1, sc1, err := s.LinearTrit("weight", x, M, K, M, false)
	if err != nil {
		t.Fatal(err)
	}
	W, scRef := QuantizeAbsMeanTrits(full)
	ref := LinearTritScalar(W, x, M, K, scRef)
	for i := 0; i < M; i++ {
		if math.Abs(float64(y1[i]-ref[i])) > 1e-4 {
			t.Fatalf("one-tile row %d got=%v ref=%v", i, y1[i], ref[i])
		}
		if math.Abs(float64(ys[i]-yp[i])) > 1e-4 {
			t.Fatalf("scalar/plane row %d %v vs %v", i, ys[i], yp[i])
		}
	}
	if math.Abs(float64(sc1-scRef)) > 1e-6 {
		t.Fatalf("scale one-tile %v ref %v", sc1, scRef)
	}
	// one-shot GGUF helper
	yo, _, err := LinearTritGGUF(path, "weight", x, M, K, 4, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(yo) != M {
		t.Fatalf("oneshot len %d", len(yo))
	}
}

func TestLinearTritBenchPaths(t *testing.T) {
	// Microbench: larger K/M so pack cost is amortized; report both paths.
	M, K := 64, 1024
	W := make([]int8, M*K)
	x := make([]float32, K)
	pat := []int8{1, 0, -1, 1, -1, 0, 1, 1}
	for i := range W {
		W[i] = pat[i%len(pat)]
	}
	for i := range x {
		x[i] = 0.05 * float32(i%9-4)
	}
	scale := float32(0.5)
	// warmup
	_ = LinearTritScalar(W, x, M, K, scale)
	_ = LinearTritPlanes(W, x, M, K, scale)
	pos, neg, nw := PackTritMatrix(W, M, K)
	_ = LinearTritPlanesPacked(pos, neg, x, M, K, nw, scale)
	const iters = 40
	t0 := time.Now()
	var sink float32
	for i := 0; i < iters; i++ {
		y := LinearTritScalar(W, x, M, K, scale)
		sink += y[0]
	}
	sScalar := time.Since(t0)
	t0 = time.Now()
	for i := 0; i < iters; i++ {
		y := LinearTritPlanes(W, x, M, K, scale)
		sink += y[0]
	}
	sPlane := time.Since(t0)
	t0 = time.Now()
	for i := 0; i < iters; i++ {
		y := LinearTritPlanesPacked(pos, neg, x, M, K, nw, scale)
		sink += y[0]
	}
	sPacked := time.Since(t0)
	// force sink use
	if sink == 1e30 {
		t.Fatal("unreachable")
	}
	t.Logf("linear_trit_bench M=%d K=%d iters=%d scalar=%s planes=%s packed=%s ratio_scalar/packed=%.3f ratio_planes/packed=%.3f",
		M, K, iters, sScalar, sPlane, sPacked, float64(sScalar)/float64(sPacked), float64(sPlane)/float64(sPacked))
}

func TestLinearTritPrepackCache(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cache.gguf")
	M, K := 8, 128
	vals := make([]float32, M*K)
	pat := []float32{1.1, 0, -0.8, 0.3, -1.2, 0.05, 0.7, -0.4}
	for i := range vals {
		vals[i] = pat[i%len(pat)]
	}
	if err := WriteGGUFF32(path, map[string][]float32{"weight": vals},
		map[string][]uint64{"weight": {uint64(M), uint64(K)}}, nil); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	if s.HasTritPack("weight") {
		t.Fatal("expected empty cache")
	}
	built, err := s.PrepackTrit("weight", K)
	if err != nil || !built {
		t.Fatalf("first prepack built=%v err=%v", built, err)
	}
	built2, err := s.PrepackTrit("weight", K)
	if err != nil || built2 {
		t.Fatalf("second prepack hit expected built=false got %v %v", built2, err)
	}
	if !s.HasTritPack("weight") {
		t.Fatal("expected pack present")
	}
	pm, pk, sc, ok := s.TritPackInfo("weight")
	if !ok || pm != M || pk != K || sc <= 0 {
		t.Fatalf("info M=%d K=%d sc=%v ok=%v", pm, pk, sc, ok)
	}
	x := make([]float32, K)
	for i := range x {
		x[i] = 0.07 * float32(i%5-2)
	}
	// uncached full-tile path vs cached scalar/planes
	ref, scRef, err := s.LinearTrit("weight", x, M, K, M, false)
	if err != nil {
		t.Fatal(err)
	}
	ys, scs, err := s.LinearTritCached("weight", x, M, K, false)
	if err != nil {
		t.Fatal(err)
	}
	yp, scp, err := s.LinearTritCached("weight", x, M, K, true)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(scRef-scs)) > 1e-6 || math.Abs(float64(scs-scp)) > 1e-6 {
		t.Fatalf("scale ref=%v s=%v p=%v", scRef, scs, scp)
	}
	for i := 0; i < M; i++ {
		if math.Abs(float64(ref[i]-ys[i])) > 1e-4 || math.Abs(float64(ys[i]-yp[i])) > 1e-4 {
			t.Fatalf("row %d ref=%v ys=%v yp=%v", i, ref[i], ys[i], yp[i])
		}
	}
	// Drop and re-prepack
	s.DropTritPack("weight")
	if s.HasTritPack("weight") {
		t.Fatal("drop failed")
	}
	built3, err := s.PrepackTrit("weight", K)
	if err != nil || !built3 {
		t.Fatalf("rebuild %v %v", built3, err)
	}
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
