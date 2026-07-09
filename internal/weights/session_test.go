package weights

import (
	"math"
	"path/filepath"
	"testing"
)

func TestWeightSessionMultiTensor(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "multi.gguf")
	tensors := map[string][]float32{
		"weight": {1, 2, 3, 4, 5, 6},
		"bias":   {0.5, -0.5},
	}
	shapes := map[string][]uint64{"weight": {2, 3}, "bias": {2}}
	if err := WriteGGUFF32(path, tensors, shapes, map[string]string{"general.name": "sess"}); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	if s.ID <= 0 {
		t.Fatalf("id %d", s.ID)
	}
	if !s.Mapped {
		t.Log("Mapped=false fallback ok")
	}
	if SessionCount() != 1 {
		t.Fatalf("count %d", SessionCount())
	}
	names, err := s.List()
	if err != nil || len(names) != 2 {
		t.Fatalf("list %v %v", names, err)
	}
	// load two tensors from same open
	w, sh, err := s.LoadF32("weight")
	if err != nil || len(w) != 6 || len(sh) != 2 {
		t.Fatalf("w %v %v %v", w, sh, err)
	}
	b, _, err := s.LoadF32("bias")
	if err != nil || len(b) != 2 || math.Abs(float64(b[0]-0.5)) > 1e-6 {
		t.Fatalf("b %v %v", b, err)
	}
	if s.Meta("general.name") != "sess" {
		t.Fatalf("meta %q", s.Meta("general.name"))
	}
	// get by id
	s2, err := GetSession(s.ID)
	if err != nil || s2 != s {
		t.Fatal(err)
	}
	id := s.ID
	if err := CloseSession(id); err != nil {
		t.Fatal(err)
	}
	if SessionCount() != 0 {
		t.Fatalf("after close %d", SessionCount())
	}
	// idempotent close
	if err := CloseSession(id); err != nil {
		t.Fatal(err)
	}
	if _, err := GetSession(id); err == nil {
		t.Fatal("expected missing after close")
	}
	// Load after close via retained pointer should fail
	if _, _, err := s.LoadF32("weight"); err == nil {
		t.Fatal("expected closed load error")
	}
}

func TestWeightSessionSafeTensors(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.safetensors")
	if err := WriteSafeTensors(path, map[string][]float32{
		"a": {1, 2}, "b": {3},
	}, map[string][]int64{"a": {2}, "b": {1}}); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	if s.Kind != "safetensors" {
		t.Fatalf("kind %s", s.Kind)
	}
	av, _, err := s.LoadF32("a")
	if err != nil || len(av) != 2 {
		t.Fatal(err)
	}
	bv, _, err := s.LoadF32("b")
	if err != nil || bv[0] != 3 {
		t.Fatal(err)
	}
}
