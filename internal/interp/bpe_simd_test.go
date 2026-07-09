package interp

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLinearTritSimdDemo(t *testing.T) {
	root := repoRoot(t)
	src, err := os.ReadFile(filepath.Join(root, "examples/linear_trit_simd.trin"))
	if err != nil {
		t.Fatal(err)
	}
	wd, _ := os.Getwd()
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(wd)
	out := runOK(t, string(src))
	for _, k := range []string{"linear_trit_simd_ok", "parity_ok=1", "simd_lane_width=128"} {
		if !strings.Contains(out, k) {
			t.Fatalf("missing %s in %q", k, out)
		}
	}
}

func TestBPESidecarDemo(t *testing.T) {
	root := repoRoot(t)
	src, err := os.ReadFile(filepath.Join(root, "examples/bpe_sidecar.trin"))
	if err != nil {
		t.Fatal(err)
	}
	wd, _ := os.Getwd()
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(wd)
	out := runOK(t, string(src))
	for _, k := range []string{"bpe_sidecar_ok", "bpe_ok=1", "match_abc=1", "match_ab=1"} {
		if !strings.Contains(out, k) {
			t.Fatalf("missing %s in %q", k, out)
		}
	}
}

func TestWeightSessionLinearTritDemo(t *testing.T) {
	root := repoRoot(t)
	// ensure fixture exists
	if _, err := os.Stat(filepath.Join(root, "demos/weights/trit_session.gguf")); err != nil {
		t.Skip("missing demos/weights/trit_session.gguf — run scripts/gen_trit_session_fixture.go")
	}
	src, err := os.ReadFile(filepath.Join(root, "examples/weight_session_linear_trit.trin"))
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
		"weight_session_linear_trit_ok",
		"parity_ok=1",
		"tile_ok=1",
		"oneshot_ok=1",
	} {
		if !strings.Contains(out, k) {
			t.Fatalf("missing %s in %q", k, out)
		}
	}
}

func TestLinearTritBenchDemo(t *testing.T) {
	root := repoRoot(t)
	src, err := os.ReadFile(filepath.Join(root, "examples/linear_trit_bench.trin"))
	if err != nil {
		t.Fatal(err)
	}
	wd, _ := os.Getwd()
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(wd)
	out := runOK(t, string(src))
	for _, k := range []string{"linear_trit_bench_ok", "parity_ok=1", "sum_parity_ok=1"} {
		if !strings.Contains(out, k) {
			t.Fatalf("missing %s in %q", k, out)
		}
	}
}

func TestWeightSessionTritCacheDemo(t *testing.T) {
	root := repoRoot(t)
	if _, err := os.Stat(filepath.Join(root, "demos/weights/trit_session.gguf")); err != nil {
		t.Skip("missing demos/weights/trit_session.gguf")
	}
	src, err := os.ReadFile(filepath.Join(root, "examples/weight_session_trit_cache.trin"))
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
		"weight_session_trit_cache_ok",
		"cache_hit_ok=1",
		"parity_ok=1",
		"built1=1",
		"built2=0",
	} {
		if !strings.Contains(out, k) {
			t.Fatalf("missing %s in %q", k, out)
		}
	}
}

func TestWeightSessionLayerStackTritDemo(t *testing.T) {
	root := repoRoot(t)
	if _, err := os.Stat(filepath.Join(root, "demos/weights/layer_stack.gguf")); err != nil {
		t.Skip("missing demos/weights/layer_stack.gguf")
	}
	src, err := os.ReadFile(filepath.Join(root, "examples/weight_session_layer_stack_trit.trin"))
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
		"weight_session_layer_stack_trit_ok",
		"cache_hit_ok=1",
		"parity_ok=1",
		"stack_match=1",
		"plane_ok=1",
		"uncached_ok=1",
		"prepack_built=2",
		"prepack_hits2=2",
	} {
		if !strings.Contains(out, k) {
			t.Fatalf("missing %s in %q", k, out)
		}
	}
}

func TestWeightSessionAttnTritDemo(t *testing.T) {
	root := repoRoot(t)
	if _, err := os.Stat(filepath.Join(root, "demos/weights/session_attn.gguf")); err != nil {
		t.Skip("missing demos/weights/session_attn.gguf")
	}
	src, err := os.ReadFile(filepath.Join(root, "examples/weight_session_attn_trit.trin"))
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
		"weight_session_attn_trit_ok",
		"cache_hit_ok=1",
		"parity_ok=1",
		"proj_ok=1",
		"y1_ok=1",
		"plane_ok=1",
		"grow_ok=1",
		"prepack_built=4",
		"prepack_hits2=4",
	} {
		if !strings.Contains(out, k) {
			t.Fatalf("missing %s in %q", k, out)
		}
	}
}

func TestTinyGGUFDecodeAttnTritDemo(t *testing.T) {
	root := repoRoot(t)
	if _, err := os.Stat(filepath.Join(root, "demos/weights/tiny_llm.gguf")); err != nil {
		t.Skip("missing demos/weights/tiny_llm.gguf")
	}
	if _, err := os.Stat(filepath.Join(root, "demos/weights/tiny_llm.vocab")); err != nil {
		t.Skip("missing demos/weights/tiny_llm.vocab")
	}
	src, err := os.ReadFile(filepath.Join(root, "examples/tiny_gguf_decode_attn_trit.trin"))
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
		"tiny_gguf_decode_attn_trit_ok",
		"cache_hit_ok=1",
		"parity_ok=1",
		"decode_ok=1",
		"parity_attn0_ok=1",
		"match_trit_decode=1",
		"gen_text=EEE",
		"prepack_q_built=1",
		"prepack_q_hit=0",
		"final_kv_toks=4",
	} {
		if !strings.Contains(out, k) {
			t.Fatalf("missing %s in %q", k, out)
		}
	}
}

func TestWeightSessionAttnStackTritDemo(t *testing.T) {
	root := repoRoot(t)
	if _, err := os.Stat(filepath.Join(root, "demos/weights/attn_stack.gguf")); err != nil {
		t.Skip("missing demos/weights/attn_stack.gguf")
	}
	src, err := os.ReadFile(filepath.Join(root, "examples/weight_session_attn_stack_trit.trin"))
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
		"weight_session_attn_stack_trit_ok",
		"cache_hit_ok=1",
		"parity_ok=1",
		"decode_ok=1",
		"stack_match=1",
		"plane_ok=1",
		"grow_ok=1",
		"flat_ok=1",
		"prepack_built=12",
		"prepack_hits2=12",
	} {
		if !strings.Contains(out, k) {
			t.Fatalf("missing %s in %q", k, out)
		}
	}
}
