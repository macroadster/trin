package codegen

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/macroadster/trin/internal/parser"
)

func root(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("no caller")
	}
	return filepath.Join(filepath.Dir(file), "..", "..")
}

func TestEmitAndCompileExamples(t *testing.T) {
	cc := os.Getenv("TRIN_CC")
	if cc == "" {
		cc = "cc"
	}
	if _, err := exec.LookPath(cc); err != nil {
		t.Skip("no C compiler")
	}
	dir := t.TempDir()
	files := []string{
		"hello.trin", "soft_mux.trin", "span_ops.trin", "trit_gemm.trin", "bitplane_gemm.trin", "bitplane_words.trin", "bitplane_simd.trin",
	}
	for _, name := range files {
		t.Run(name, func(t *testing.T) {
			src, err := os.ReadFile(filepath.Join(root(t), "examples", name))
			if err != nil {
				t.Fatal(err)
			}
			prog, err := parser.ParseSource(string(src))
			if err != nil {
				t.Fatal(err)
			}
			c, err := EmitC(prog)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(c, "trin_fn_main") {
				t.Fatal("missing main")
			}
			cpath := filepath.Join(dir, name+".c")
			bin := filepath.Join(dir, strings.TrimSuffix(name, ".trin"))
			if err := os.WriteFile(cpath, []byte(c), 0o644); err != nil {
				t.Fatal(err)
			}
			cmd := exec.Command(cc, "-O2", "-std=gnu11", "-o", bin, cpath, "-lm")
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("cc: %v\n%s", err, out)
			}
		})
	}
}

func TestEmitCReduceLambda(t *testing.T) {
	src, err := os.ReadFile(filepath.Join(root(t), "examples", "reduce.trin"))
	if err != nil {
		t.Fatal(err)
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		t.Fatal(err)
	}
	c, err := EmitC(prog)
	if err != nil {
		t.Fatalf("EmitC reduce: %v", err)
	}
	if !strings.Contains(c, "for(int") || !strings.Contains(c, "env_set") {
		t.Fatal("expected reduce fold loop")
	}
}

func TestEmitCZip3Scan(t *testing.T) {
	src, err := os.ReadFile(filepath.Join(root(t), "examples", "zip3_scan.trin"))
	if err != nil {
		t.Fatal(err)
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		t.Fatal(err)
	}
	c, err := EmitC(prog)
	if err != nil {
		t.Fatalf("EmitC zip3_scan: %v", err)
	}
	if !strings.Contains(c, "for(int") || !strings.Contains(c, "env_set") {
		t.Fatal("expected ternary zip / scan loops")
	}
}

func TestEmitCFilterNary(t *testing.T) {
	src, err := os.ReadFile(filepath.Join(root(t), "examples", "filter_nary.trin"))
	if err != nil {
		t.Fatal(err)
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		t.Fatal(err)
	}
	c, err := EmitC(prog)
	if err != nil {
		t.Fatalf("EmitC filter_nary: %v", err)
	}
	if !strings.Contains(c, "for(int") || !strings.Contains(c, "env_set") {
		t.Fatal("expected n-ary zip / filter loops")
	}
}

func TestEmitCZipWithLambda(t *testing.T) {
	src, err := os.ReadFile(filepath.Join(root(t), "examples", "zip_with.trin"))
	if err != nil {
		t.Fatal(err)
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		t.Fatal(err)
	}
	c, err := EmitC(prog)
	if err != nil {
		t.Fatalf("EmitC zip_with: %v", err)
	}
	if !strings.Contains(c, "for(int") || !strings.Contains(c, "env_set") {
		t.Fatal("expected zip_with element loop")
	}
}

func TestEmitCGeneralMapLambda(t *testing.T) {
	src, err := os.ReadFile(filepath.Join(root(t), "examples", "map_general.trin"))
	if err != nil {
		t.Fatal(err)
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		t.Fatal(err)
	}
	c, err := EmitC(prog)
	if err != nil {
		t.Fatalf("EmitC map_general: %v", err)
	}
	if !strings.Contains(c, "trin_fn_main") {
		t.Fatal("missing main")
	}
	// general unary map should expand to element loop + nested env, not only mul2 helper
	if !strings.Contains(c, "env_init") || !strings.Contains(c, "env_set") {
		t.Fatal("expected nested env for map lambda")
	}
	if !strings.Contains(c, "for(int") {
		t.Fatal("expected map element loop")
	}
	if !strings.Contains(c, "bi_soft_mux") && !strings.Contains(c, "soft_mux") {
		t.Fatal("expected soft_mux body in general map example")
	}
	cc := os.Getenv("TRIN_CC")
	if cc == "" {
		cc = "cc"
	}
	if _, err := exec.LookPath(cc); err != nil {
		t.Skip("no C compiler")
	}
	dir := t.TempDir()
	cpath := filepath.Join(dir, "map_general.c")
	bin := filepath.Join(dir, "map_general")
	if err := os.WriteFile(cpath, []byte(c), 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(cc, "-O2", "-std=gnu11", "-o", bin, cpath, "-lm")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("cc map_general: %v\n%s", err, out)
	}
}
