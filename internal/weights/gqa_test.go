package weights

import (
	"math"
	"path/filepath"
	"testing"
)

func nearG(a, b float32, tol float64) bool {
	return math.Abs(float64(a-b)) < tol
}

func TestGQAExpandKV(t *testing.T) {
	// 2 toks, 1 KV head, 2 Q heads, dh=2
	kv := []float32{1, 2, 3, 4}
	out, err := GQAExpandKV(kv, 2, 1, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 2, 1, 2, 3, 4, 3, 4}
	if len(out) != len(want) {
		t.Fatalf("len %d", len(out))
	}
	for i := range want {
		if !nearG(out[i], want[i], 1e-6) {
			t.Fatalf("%v vs %v", out, want)
		}
	}
	// 2 KV heads, 4 Q heads, dh=1, 1 tok → group=2
	kv2 := []float32{10, 20}
	out2, err := GQAExpandKV(kv2, 1, 2, 4, 1)
	if err != nil {
		t.Fatal(err)
	}
	want2 := []float32{10, 10, 20, 20}
	for i := range want2 {
		if !nearG(out2[i], want2[i], 1e-6) {
			t.Fatalf("%v vs %v", out2, want2)
		}
	}
	// identity expand
	id, err := GQAExpandKV([]float32{1, 2, 3, 4}, 1, 2, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range []float32{1, 2, 3, 4} {
		if !nearG(id[i], v, 1e-6) {
			t.Fatal(id)
		}
	}
	row, err := GQARepeatKVRow([]float32{7, 8}, 1, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !nearG(row[0], 7, 1e-6) || !nearG(row[2], 7, 1e-6) || len(row) != 4 {
		t.Fatal(row)
	}
	if _, err := GQAExpandKV(kv, 2, 3, 4, 2); err == nil {
		t.Fatal("expected nHeads%nKV fail")
	}
	if _, err := GQAExpandKV(kv[:3], 2, 1, 2, 2); err == nil {
		t.Fatal("expected len fail")
	}
}

func TestSessionAttnGQAParity(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "gqa.gguf")
	// dIn=2, nHeads=2, nKV=1, dh=2 → dQ=4, dKV=2
	// Q 4×2 rows: [[1,0],[0,1],[1,0],[0,1]] → q=[x0,x1,x0,x1]
	Q := []float32{
		1, 0,
		0, 1,
		1, 0,
		0, 1,
	}
	// K 2×2 identity
	K := []float32{1, 0, 0, 1}
	// V 2×2 *2
	V := []float32{2, 0, 0, 2}
	// O 4×4 identity
	O := make([]float32, 16)
	for i := 0; i < 4; i++ {
		O[i*4+i] = 1
	}
	if err := WriteGGUFF32(path,
		map[string][]float32{"Q": Q, "K": K, "V": V, "O": O},
		map[string][]uint64{"Q": {2, 4}, "K": {2, 2}, "V": {2, 2}, "O": {4, 4}},
		map[string]string{"general.name": "gqa"},
	); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	x := []float32{3, 4}
	q, k, v, err := s.AttnProjGQA("Q", "K", "V", x, 2, 2, 1, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	// q = [3,4,3,4], k=[3,4], v=[6,8]
	wantQ := []float32{3, 4, 3, 4}
	wantK := []float32{3, 4}
	wantV := []float32{6, 8}
	for i := range wantQ {
		if !nearG(q[i], wantQ[i], 1e-5) {
			t.Fatalf("q %v want %v", q, wantQ)
		}
	}
	for i := range wantK {
		if !nearG(k[i], wantK[i], 1e-5) || !nearG(v[i], wantV[i], 1e-5) {
			t.Fatalf("kv %v %v", k, v)
		}
	}
	// fused compose vs expand+mha
	qr, _ := s.LinearFused("Q", x, 4, 2, 1)
	kr, _ := s.LinearFused("K", x, 2, 2, 1)
	vr, _ := s.LinearFused("V", x, 2, 2, 1)
	for i := 0; i < 4; i++ {
		if !nearG(q[i], qr[i], 1e-5) {
			t.Fatalf("Q parity")
		}
	}
	for i := 0; i < 2; i++ {
		if !nearG(k[i], kr[i], 1e-5) || !nearG(v[i], vr[i], 1e-5) {
			t.Fatalf("KV parity")
		}
	}
	y, Knew, Vnew, err := s.SessionAttnGQA("Q", "K", "V", "O", x, nil, nil, 2, 2, 1, 2, 0, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(y) != 4 {
		t.Fatalf("y len %d", len(y))
	}
	if len(Knew) != 2 || len(Vnew) != 2 {
		t.Fatalf("compact kv %d %d", len(Knew), len(Vnew))
	}
	// manual: expand + mha + O
	Kexp, _ := GQAExpandKV(Knew, 1, 1, 2, 2)
	Vexp, _ := GQAExpandKV(Vnew, 1, 1, 2, 2)
	ym, err := MHAF32(q, Kexp, Vexp, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	yo, err := s.LinearFused("O", ym, 4, 4, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 4; i++ {
		if !nearG(y[i], yo[i], 1e-4) {
			t.Fatalf("session vs compose %v vs %v", y, yo)
		}
	}
	// grow KV second step stays compact
	y2, K2, V2, err := s.SessionAttnGQA("Q", "K", "V", "O", x, Knew, Vnew, 2, 2, 1, 2, 0, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(K2) != 4 || len(V2) != 4 {
		t.Fatalf("grown compact %d %d", len(K2), len(V2))
	}
	_ = y2
	// rope path
	yr, _, _, err := s.SessionAttnGQA("Q", "K", "V", "", x, nil, nil, 2, 2, 1, 2, 1.0, 10000, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(yr) != 4 {
		t.Fatal(yr)
	}
	// bad heads
	if _, _, _, err := s.SessionAttnGQA("Q", "K", "V", "", x, nil, nil, 2, 3, 2, 2, 0, 0, 1); err == nil {
		t.Fatal("expected heads mismatch")
	}
}
