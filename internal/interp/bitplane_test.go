package interp

import "testing"

func TestPackUnpackWords(t *testing.T) {
	xs := make([]Value, 40)
	pat := []int{1, 0, -1, 1, -1, 0, 1, 1}
	for i := range xs {
		xs[i] = Trit(pat[i%len(pat)])
	}
	pos, neg, n := packTritWords(xs)
	if n != 40 {
		t.Fatalf("n=%d", n)
	}
	if len(pos) != 2 || len(neg) != 2 {
		t.Fatalf("want 2 words, got %d/%d", len(pos), len(neg))
	}
	back := unpackTritWords(pos, neg, n)
	for i := 0; i < n; i++ {
		if back[i] != clampTrit(xs[i].AsInt()) {
			t.Fatalf("i=%d got %d want %d", i, back[i], xs[i].AsInt())
		}
	}
}

func TestTritDotPlanesMatchesScalar(t *testing.T) {
	a := []int{1, 0, -1, 1, -1, 0, 1, 1, -1, 0, 1, -1, 1, 0, -1, 1,
		0, 1, -1, 0, 1, 1, -1, 0, 1, -1, 0, 1, 1, -1, 0, 1, -1, 1, 0, -1}
	b := []int{-1, 1, 0, 1, 1, -1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0,
		1, -1, 1, 0, -1, 0, 1, 1, 0, 1, -1, 0, 1, 1, -1, 1, 0, -1, 1, 0}
	if len(a) != len(b) {
		t.Fatal("len")
	}
	A := make([]Value, len(a))
	B := make([]Value, len(b))
	want := 0
	for i := range a {
		A[i] = Trit(a[i])
		B[i] = Trit(b[i])
		want += a[i] * b[i]
	}
	ap, an, _ := packTritWords(A)
	bp, bn, _ := packTritWords(B)
	got := tritDotPlanes(ap, an, bp, bn, len(a))
	if got != want {
		t.Fatalf("dot planes=%d scalar=%d", got, want)
	}
}

func TestTritGemmBitplaneMatchesScalar(t *testing.T) {
	// 2x5 * 5x2 via both paths
	A := []Value{
		Trit(1), Trit(0), Trit(-1), Trit(1), Trit(-1),
		Trit(0), Trit(1), Trit(1), Trit(-1), Trit(0),
	}
	B := []Value{
		Trit(1), Trit(-1),
		Trit(0), Trit(1),
		Trit(-1), Trit(0),
		Trit(1), Trit(1),
		Trit(0), Trit(-1),
	}
	cs, err := tritGemm([]Value{Span(A), Span(B), I32(2), I32(5), I32(2)}, false, false)
	if err != nil {
		t.Fatal(err)
	}
	cb, err := tritGemm([]Value{Span(A), Span(B), I32(2), I32(5), I32(2)}, true, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(cs.Span) != len(cb.Span) {
		t.Fatalf("len")
	}
	for i := range cs.Span {
		if cs.Span[i].AsInt() != cb.Span[i].AsInt() {
			t.Fatalf("i=%d scalar=%d bitplane=%d", i, cs.Span[i].AsInt(), cb.Span[i].AsInt())
		}
	}
}

func TestTritDotPlanesSIMDMatchesScalar(t *testing.T) {
	// Cross multiple 128-lane blocks: 140 trits = 5 u32 words with residual.
	n := 140
	A := make([]Value, n)
	B := make([]Value, n)
	want := 0
	patA := []int{1, 0, -1, 1, -1, 0, 1, 1}
	patB := []int{-1, 1, 0, 1, 1, -1, 0, -1}
	for i := 0; i < n; i++ {
		a, b := patA[i%len(patA)], patB[i%len(patB)]
		A[i], B[i] = Trit(a), Trit(b)
		want += a * b
	}
	ap, an, _ := packTritWords(A)
	bp, bn, _ := packTritWords(B)
	got := tritDotPlanesSIMD(ap, an, bp, bn, n)
	gotS := tritDotPlanes(ap, an, bp, bn, n)
	if got != want || gotS != want {
		t.Fatalf("simd=%d scalar_plane=%d want=%d", got, gotS, want)
	}
	// GEMM SIMD path on 1×140 × 140×1
	cs, err := tritGemm([]Value{Span(A), Span(B), I32(1), I32(n), I32(1)}, false, false)
	if err != nil {
		t.Fatal(err)
	}
	cb, err := tritGemm([]Value{Span(A), Span(B), I32(1), I32(n), I32(1)}, true, true)
	if err != nil {
		t.Fatal(err)
	}
	if cs.Span[0].AsInt() != cb.Span[0].AsInt() || cs.Span[0].AsInt() != want {
		t.Fatalf("gemm scalar=%d simd=%d want=%d", cs.Span[0].AsInt(), cb.Span[0].AsInt(), want)
	}
	if SimdLaneWidth != 128 {
		t.Fatalf("lane width %d", SimdLaneWidth)
	}
}
