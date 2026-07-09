package weights

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCharVocabEncodeDecode(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "v.txt")
	if err := os.WriteFile(p, []byte("<pad>\nA\nB\nC\nD\nE\nF\n<eos>\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	v, err := OpenVocab(p)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseVocab(v.ID)
	if v.Size() != 8 {
		t.Fatalf("size %d", v.Size())
	}
	ids := v.Encode("AB")
	if len(ids) != 2 || ids[0] != 1 || ids[1] != 2 {
		t.Fatalf("encode AB → %v", ids)
	}
	if s := v.Decode(ids); s != "AB" {
		t.Fatalf("decode %q", s)
	}
	// multi-char token
	ids2 := v.Encode("<eos>")
	if len(ids2) != 1 || ids2[0] != 7 {
		t.Fatalf("encode eos %v", ids2)
	}
	if v.Token(1) != "A" || v.IDOf("B") != 2 {
		t.Fatalf("lookup")
	}
	// longest match prefers <eos> over chars
	ids3 := v.Encode("A<eos>B")
	if len(ids3) != 3 || ids3[0] != 1 || ids3[1] != 7 || ids3[2] != 2 {
		t.Fatalf("mix %v", ids3)
	}
}

func TestLoadVocabTextAndClose(t *testing.T) {
	v, err := LoadVocabText("inline", "x\ny\n")
	if err != nil {
		t.Fatal(err)
	}
	id := v.ID
	if _, err := GetVocab(id); err != nil {
		t.Fatal(err)
	}
	_ = CloseVocab(id)
	if _, err := GetVocab(id); err == nil {
		t.Fatal("expected closed")
	}
}
