package interp

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func repoRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("caller")
	}
	// internal/interp → repo root
	return filepath.Clean(filepath.Join(filepath.Dir(file), "../.."))
}

func ensureTinyLLMFixture(t *testing.T) {
	t.Helper()
	root := repoRoot(t)
	gg := filepath.Join(root, "demos/weights/tiny_llm.gguf")
	vb := filepath.Join(root, "demos/weights/tiny_llm.vocab")
	if _, err := os.Stat(gg); err == nil {
		if _, err2 := os.Stat(vb); err2 == nil {
			return
		}
	}
	cmd := exec.Command("go", "run", "scripts/gen_tiny_llm_gguf.go", root)
	cmd.Dir = root
	cmd.Env = append(os.Environ(), "PATH="+os.Getenv("PATH")+":/home/nanobot/.local/go/bin")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("gen fixture: %v\n%s", err, out)
	}
}

func TestVocabTableDemo(t *testing.T) {
	ensureTinyLLMFixture(t)
	root := repoRoot(t)
	src, err := os.ReadFile(filepath.Join(root, "examples/vocab_table.trin"))
	if err != nil {
		t.Fatal(err)
	}
	// demos path is relative to cwd
	wd, _ := os.Getwd()
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(wd)
	out := runOK(t, string(src))
	for _, k := range []string{"vocab_table_ok", "vocab_ok=1", "ids_match=1", "dec_match=1", "file_match=1"} {
		if !strings.Contains(out, k) {
			t.Fatalf("missing %s in %q", k, out)
		}
	}
}

func TestTinyGGUFDecodeDemo(t *testing.T) {
	ensureTinyLLMFixture(t)
	root := repoRoot(t)
	src, err := os.ReadFile(filepath.Join(root, "examples/tiny_gguf_decode.trin"))
	if err != nil {
		t.Fatal(err)
	}
	wd, _ := os.Getwd()
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(wd)
	out := runOK(t, string(src))
	for _, k := range []string{"tiny_gguf_decode_ok", "prompt=AB", "prompt_ids=", "final_kv_toks=4", "decode_ok=1", "gen_text="} {
		if !strings.Contains(out, k) {
			t.Fatalf("missing %s in %q", k, out)
		}
	}
}

func TestTinyGgufDecodeTritDemo(t *testing.T) {
	root := repoRoot(t)
	if _, err := os.Stat(filepath.Join(root, "demos/weights/tiny_llm.gguf")); err != nil {
		t.Skip("missing demos/weights/tiny_llm.gguf")
	}
	src, err := os.ReadFile(filepath.Join(root, "examples/tiny_gguf_decode_trit.trin"))
	if err != nil {
		t.Fatal(err)
	}
	wd, _ := os.Getwd()
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(wd)
	out := runOK(t, string(src))
	for _, k := range []string{
		"tiny_gguf_decode_trit_ok",
		"cache_hit_ok=1",
		"parity_ok=1",
		"decode_ok=1",
		"prepack_q_built=1",
		"prepack_q_hit=0",
		"prepack_q_after_hit=0",
		"all_packs_present=1",
	} {
		if !strings.Contains(out, k) {
			t.Fatalf("missing %s in %q", k, out)
		}
	}
}
