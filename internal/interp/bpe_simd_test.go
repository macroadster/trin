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
