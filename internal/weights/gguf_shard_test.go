package weights

import (
	"encoding/binary"
	"math"
	"path/filepath"
	"testing"
)

func f32raw(vals []float32) []byte {
	out := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

func writeShardPair(dir string) (string, string, error) {
	// 2 shards, 2 tensors total (A on 0, B on 1)
	a := []float32{1, 2, 3, 4}
	b := []float32{10, 20}
	meta0 := map[string]TypedMeta{
		"general.name":          {Kind: "string", Str: "trin-shard-demo"},
		MetaSplitNo:             {Kind: "u16", U64: 0},
		MetaSplitCount:          {Kind: "u16", U64: 2},
		MetaSplitTensorsCount:   {Kind: "u16", U64: 2},
	}
	meta1 := map[string]TypedMeta{
		MetaSplitNo:           {Kind: "u16", U64: 1},
		MetaSplitCount:        {Kind: "u16", U64: 2},
		MetaSplitTensorsCount: {Kind: "u16", U64: 2},
	}
	p0 := filepath.Join(dir, "toy-00000-of-00002.gguf")
	p1 := filepath.Join(dir, "toy-00001-of-00002.gguf")
	if err := WriteGGUFTypedMulti(p0, []string{"A"},
		map[string]uint32{"A": GGML_F32},
		map[string][]uint64{"A": {4}},
		map[string][]byte{"A": f32raw(a)},
		meta0); err != nil {
		return "", "", err
	}
	if err := WriteGGUFTypedMulti(p1, []string{"B"},
		map[string]uint32{"B": GGML_F32},
		map[string][]uint64{"B": {2}},
		map[string][]byte{"B": f32raw(b)},
		meta1); err != nil {
		return "", "", err
	}
	return p0, p1, nil
}

func TestOpenGGUFShardedListLoad(t *testing.T) {
	dir := t.TempDir()
	p0, p1, err := writeShardPair(dir)
	if err != nil {
		t.Fatal(err)
	}
	// open from first
	f, err := OpenGGUFSharded(p0)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if !f.IsSharded || f.SplitCount != 2 {
		t.Fatalf("sharded flags %+v", f.IsSharded)
	}
	names := f.List()
	if len(names) != 2 {
		t.Fatalf("names %v", names)
	}
	// order: shard0 tensors then shard1
	if names[0] != "A" || names[1] != "B" {
		t.Fatalf("order %v", names)
	}
	va, sa, err := f.LoadF32("A")
	if err != nil {
		t.Fatal(err)
	}
	if len(sa) != 1 || sa[0] != 4 || maxAbsDiff(va, []float32{1, 2, 3, 4}) > 1e-6 {
		t.Fatalf("A %v %v", va, sa)
	}
	vb, sb, err := f.LoadF32("B")
	if err != nil {
		t.Fatal(err)
	}
	if len(sb) != 1 || sb[0] != 2 || maxAbsDiff(vb, []float32{10, 20}) > 1e-6 {
		t.Fatalf("B %v %v", vb, sb)
	}
	// open from second shard also works (redirects to first)
	f2, err := OpenGGUFSharded(p1)
	if err != nil {
		t.Fatal(err)
	}
	defer f2.Close()
	if len(f2.List()) != 2 {
		t.Fatal(f2.List())
	}
	// plain OpenGGUF on shard0 only sees A
	s0, err := OpenGGUF(p0)
	if err != nil {
		t.Fatal(err)
	}
	defer s0.Close()
	if len(s0.List()) != 1 || s0.List()[0] != "A" {
		t.Fatal(s0.List())
	}
	// window on cross-shard tensor B
	w, err := f.LoadF32Window("B", 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if mathAbs(w[0]-20) > 1e-5 {
		t.Fatal(w)
	}
}

func mathAbs(v float32) float64 {
	if v < 0 {
		return float64(-v)
	}
	return float64(v)
}

func TestSessionMultiShardLinear(t *testing.T) {
	dir := t.TempDir()
	// W on shard1 only: 2x2 matrix row-major [[1,2],[3,4]] so M=2 K=2
	// put A dummy on 0, W on 1
	p0 := filepath.Join(dir, "lin-00000-of-00002.gguf")
	p1 := filepath.Join(dir, "lin-00001-of-00002.gguf")
	meta0 := map[string]TypedMeta{
		MetaSplitNo: {Kind: "u16", U64: 0}, MetaSplitCount: {Kind: "u16", U64: 2},
		MetaSplitTensorsCount: {Kind: "u16", U64: 2},
	}
	meta1 := map[string]TypedMeta{
		MetaSplitNo: {Kind: "u16", U64: 1}, MetaSplitCount: {Kind: "u16", U64: 2},
		MetaSplitTensorsCount: {Kind: "u16", U64: 2},
	}
	dummy := f32raw([]float32{0})
	W := f32raw([]float32{1, 2, 3, 4})
	if err := WriteGGUFTypedMulti(p0, []string{"bias"},
		map[string]uint32{"bias": GGML_F32},
		map[string][]uint64{"bias": {1}},
		map[string][]byte{"bias": dummy}, meta0); err != nil {
		t.Fatal(err)
	}
	if err := WriteGGUFTypedMulti(p1, []string{"W"},
		map[string]uint32{"W": GGML_F32},
		map[string][]uint64{"W": {2, 2}},
		map[string][]byte{"W": W}, meta1); err != nil {
		t.Fatal(err)
	}
	_ = p1
	s, err := OpenSession(p0)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	names, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(names) != 2 {
		t.Fatal(names)
	}
	x := []float32{1, 1}
	y, err := s.LinearFused("W", x, 2, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	// row0: 1+2=3, row1: 3+4=7
	if mathAbs(y[0]-3) > 1e-4 || mathAbs(y[1]-7) > 1e-4 {
		t.Fatalf("y=%v", y)
	}
}

func TestListShardPaths(t *testing.T) {
	paths, err := listShardPaths("/models/foo-00001-of-00003.gguf", 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) != 3 {
		t.Fatal(paths)
	}
	if filepath.Base(paths[0]) != "foo-00000-of-00003.gguf" || filepath.Base(paths[2]) != "foo-00002-of-00003.gguf" {
		t.Fatal(paths)
	}
}

func TestSingleFileShardedNoop(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "single.gguf")
	raw := f32raw([]float32{1, 2})
	if err := WriteGGUFTyped(p, "W", GGML_F32, []uint64{2}, raw, nil); err != nil {
		t.Fatal(err)
	}
	f, err := OpenGGUFSharded(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if f.IsSharded {
		t.Fatal("should not be sharded")
	}
	if len(f.List()) != 1 {
		t.Fatal(f.List())
	}
}
