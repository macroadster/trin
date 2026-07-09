package weights

import (
	"os"
	"path/filepath"
	"testing"
)

func TestBPEEncodeMerges(t *testing.T) {
	// vocab: individual letters + merged "AB" + pad/eos
	vocab := "<pad>\nA\nB\nC\nAB\nBC\nABC\n<unk>\n"
	// merge A B → AB (rank 0), then AB C → ABC (rank 1)
	merges := "# toy merges\nA B\nAB C\n"
	tok, err := OpenBPEText(vocab, merges)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseBPE(tok.ID)
	if tok.MergeCount() != 2 {
		t.Fatalf("merges %d", tok.MergeCount())
	}
	ids := tok.Encode("ABC")
	// should become single "ABC" token id 6
	if len(ids) != 1 || ids[0] != 6 {
		t.Fatalf("encode ABC → %v", ids)
	}
	if tok.Decode(ids) != "ABC" {
		t.Fatalf("decode %q", tok.Decode(ids))
	}
	// AB only
	ids2 := tok.Encode("AB")
	if len(ids2) != 1 || ids2[0] != 4 {
		t.Fatalf("encode AB → %v", ids2)
	}
	// C alone
	ids3 := tok.Encode("C")
	if len(ids3) != 1 || ids3[0] != 3 {
		t.Fatalf("encode C → %v", ids3)
	}
}

func TestBPENoMergesFallsBack(t *testing.T) {
	tok, err := OpenBPEText("<pad>\nA\nB\n", "")
	if err != nil {
		t.Fatal(err)
	}
	defer CloseBPE(tok.ID)
	ids := tok.Encode("AB")
	if len(ids) != 2 || ids[0] != 1 || ids[1] != 2 {
		t.Fatalf("fallback %v", ids)
	}
}

func TestBPEFromFiles(t *testing.T) {
	dir := t.TempDir()
	vp := filepath.Join(dir, "v.txt")
	mp := filepath.Join(dir, "m.txt")
	if err := os.WriteFile(vp, []byte("<pad>\nA\nB\nAB\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(mp, []byte("A B\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	tok, err := OpenBPE(vp, mp)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseBPE(tok.ID)
	ids := tok.Encode("AB")
	if len(ids) != 1 || ids[0] != 3 {
		t.Fatalf("file bpe %v", ids)
	}
	if tok.VocabSize() != 4 {
		t.Fatalf("size %d", tok.VocabSize())
	}
}
