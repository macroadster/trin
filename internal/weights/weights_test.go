package weights

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSafeTensorsRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "W.safetensors")
	tensors := map[string][]float32{
		"weight": {0.9, -1.1, 0.05, 1.0, -0.8, 0.2},
	}
	shapes := map[string][]int64{"weight": {2, 3}}
	if err := WriteSafeTensors(path, tensors, shapes); err != nil {
		t.Fatal(err)
	}
	f, err := OpenSafeTensors(path)
	if err != nil {
		t.Fatal(err)
	}
	names := f.List()
	if len(names) != 1 || names[0] != "weight" {
		t.Fatalf("names %v", names)
	}
	vals, shape, err := f.LoadF32("weight")
	if err != nil {
		t.Fatal(err)
	}
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("shape %v", shape)
	}
	if len(vals) != 6 {
		t.Fatalf("len %d", len(vals))
	}
	for i, v := range tensors["weight"] {
		if math.Abs(float64(vals[i]-v)) > 1e-6 {
			t.Fatalf("vals[%d]=%v want %v", i, vals[i], v)
		}
	}
}

func TestGGUFRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "W.gguf")
	tensors := map[string][]float32{
		"blk.0.weight": {1, 0, -1, 0, 1, 1},
	}
	shapes := map[string][]uint64{"blk.0.weight": {2, 3}}
	meta := map[string]string{"general.name": "toy"}
	if err := WriteGGUFF32(path, tensors, shapes, meta); err != nil {
		t.Fatal(err)
	}
	f, err := OpenGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	if f.Version != 3 {
		t.Fatalf("ver %d", f.Version)
	}
	if len(f.List()) != 1 {
		t.Fatalf("list %v", f.List())
	}
	vals, shape, err := f.LoadF32("blk.0.weight")
	if err != nil {
		t.Fatal(err)
	}
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("shape %v", shape)
	}
	for i, v := range tensors["blk.0.weight"] {
		if math.Abs(float64(vals[i]-v)) > 1e-6 {
			t.Fatalf("vals[%d]=%v want %v", i, vals[i], v)
		}
	}
	// metadata present
	if f.Meta["general.name"] != "toy" {
		t.Fatalf("meta %v", f.Meta)
	}
}

func TestAutoLoad(t *testing.T) {
	dir := t.TempDir()
	st := filepath.Join(dir, "a.safetensors")
	gg := filepath.Join(dir, "a.gguf")
	_ = WriteSafeTensors(st, map[string][]float32{"x": {1.5, 2.5}}, map[string][]int64{"x": {2}})
	_ = WriteGGUFF32(gg, map[string][]float32{"x": {3, 4}}, map[string][]uint64{"x": {2}}, nil)
	v1, _, err := AutoLoad(st, "x")
	if err != nil || len(v1) != 2 || math.Abs(float64(v1[0]-1.5)) > 1e-6 {
		t.Fatalf("st %v %v", v1, err)
	}
	v2, _, err := AutoLoad(gg, "x")
	if err != nil || len(v2) != 2 || math.Abs(float64(v2[0]-3)) > 1e-6 {
		t.Fatalf("gg %v %v", v2, err)
	}
}

func TestFloat16(t *testing.T) {
	// 1.0 in f16 is 0x3c00
	if v := float16ToFloat32(0x3c00); math.Abs(float64(v-1.0)) > 1e-5 {
		t.Fatalf("f16 1.0 got %v", v)
	}
	if v := float16ToFloat32(0x0000); v != 0 {
		t.Fatalf("zero %v", v)
	}
}

func TestWriteFilesExist(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "t.safetensors")
	if err := WriteSafeTensors(p, map[string][]float32{"a": {1}}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(p); err != nil {
		t.Fatal(err)
	}
}
