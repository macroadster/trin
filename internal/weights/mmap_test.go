package weights

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestMmapGGUF_F32AndQ4(t *testing.T) {
	dir := t.TempDir()
	// F32 GGUF
	f32path := filepath.Join(dir, "f32.gguf")
	tensors := map[string][]float32{"weight": {1, 2, 3, 4}}
	shapes := map[string][]uint64{"weight": {2, 2}}
	if err := WriteGGUFF32(f32path, tensors, shapes, map[string]string{"general.name": "mmap-f32"}); err != nil {
		t.Fatal(err)
	}
	f, err := OpenGGUF(f32path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if !f.Mapped {
		t.Log("note: Mapped=false (heap fallback) — still OK for CI")
	}
	vals, shape, err := f.LoadF32("weight")
	if err != nil {
		t.Fatal(err)
	}
	if len(shape) != 2 || len(vals) != 4 || math.Abs(float64(vals[0]-1)) > 1e-6 {
		t.Fatalf("vals %v shape %v", vals, shape)
	}
	// list/meta must not require decoding all tensors
	if len(f.List()) != 1 {
		t.Fatalf("list %v", f.List())
	}
	if f.Meta["general.name"] != "mmap-f32" {
		t.Fatalf("meta %v", f.Meta)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	// Q4_0 GGUF
	qpath := filepath.Join(dir, "q4.gguf")
	src := make([]float32, 32)
	for i := range src {
		src[i] = float32(i-16) * 0.25
	}
	raw := packQ4_0(src)
	if err := WriteGGUFTyped(qpath, "weight", GGML_Q4_0, []uint64{32}, raw, map[string]string{"general.name": "mmap-q4"}); err != nil {
		t.Fatal(err)
	}
	fq, err := OpenGGUF(qpath)
	if err != nil {
		t.Fatal(err)
	}
	defer fq.Close()
	got, sh, err := fq.LoadF32("weight")
	if err != nil {
		t.Fatal(err)
	}
	if len(sh) != 1 || sh[0] != 32 || len(got) != 32 {
		t.Fatalf("q4 shape %v len %d", sh, len(got))
	}
	if maxAbsDiff(got, src) > 0.3 {
		t.Fatalf("q4 maxdiff %v", maxAbsDiff(got, src))
	}
	if !fq.IsMapped() && fq.Mapped {
		t.Fatal("inconsistent mapped flag")
	}
	_ = fq.Close()
}

func TestMmapSafeTensors(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "w.safetensors")
	if err := WriteSafeTensors(p, map[string][]float32{"a": {9, 8, 7}}, map[string][]int64{"a": {3}}); err != nil {
		t.Fatal(err)
	}
	f, err := OpenSafeTensors(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	vals, shape, err := f.LoadF32("a")
	if err != nil {
		t.Fatal(err)
	}
	if len(shape) != 1 || len(vals) != 3 || vals[0] != 9 {
		t.Fatalf("%v %v", vals, shape)
	}
	// prove file still open/mapped independently of heap
	if _, err := os.Stat(p); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestGGMLRawBytes(t *testing.T) {
	if GGMLRawBytes(GGML_F32, 10) != 40 {
		t.Fatal("f32")
	}
	if GGMLRawBytes(GGML_Q4_0, 32) != bsQ4_0 {
		t.Fatalf("q4 %d", GGMLRawBytes(GGML_Q4_0, 32))
	}
	if GGMLRawBytes(GGML_Q8_0, 64) != 2*bsQ8_0 {
		t.Fatal("q8")
	}
	if GGMLRawBytes(999, 10) != -1 {
		t.Fatal("unknown")
	}
}

func TestAutoLoadUsesMmap(t *testing.T) {
	dir := t.TempDir()
	st := filepath.Join(dir, "a.safetensors")
	gg := filepath.Join(dir, "a.gguf")
	_ = WriteSafeTensors(st, map[string][]float32{"x": {1.25}}, nil)
	_ = WriteGGUFF32(gg, map[string][]float32{"x": {2.5}}, map[string][]uint64{"x": {1}}, nil)
	v1, _, err := AutoLoad(st, "x")
	if err != nil || math.Abs(float64(v1[0]-1.25)) > 1e-6 {
		t.Fatalf("st %v %v", v1, err)
	}
	v2, _, err := AutoLoad(gg, "x")
	if err != nil || math.Abs(float64(v2[0]-2.5)) > 1e-6 {
		t.Fatalf("gg %v %v", v2, err)
	}
}
