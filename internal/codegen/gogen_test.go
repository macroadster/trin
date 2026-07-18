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

func goRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("no caller")
	}
	return filepath.Join(filepath.Dir(file), "..", "..")
}

func TestEmitAndGoBuildExamples(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("no go")
	}
	dir := t.TempDir()
	files := []string{
		"hello.trin", "soft_mux.trin", "span_ops.trin", "trit_gemm.trin", "bitplane_gemm.trin", "bitplane_words.trin", "bitplane_simd.trin", "soft_mux_grad.trin", "grad_tape.trin", "grad_span.trin", "grad_map.trin", "grad_map_lambda.trin", "map_general.trin", "grad_nested_map.trin", "zip_with.trin", "grad_zip_with.trin", "reduce.trin", "grad_reduce.trin", "zip3_scan.trin", "grad_zip3_scan.trin", "filter_nary.trin", "grad_filter_nary.trin",
	}
	for _, name := range files {
		t.Run(name, func(t *testing.T) {
			src, err := os.ReadFile(filepath.Join(goRoot(t), "examples", name))
			if err != nil {
				t.Fatal(err)
			}
			prog, err := parser.ParseSource(string(src))
			if err != nil {
				t.Fatal(err)
			}
			code, err := EmitGo(prog)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(code, "trinFnMain") {
				t.Fatal("missing main")
			}
			ws := filepath.Join(dir, name)
			if err := os.MkdirAll(ws, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(ws, "main.go"), []byte(code), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(ws, "go.mod"), []byte("module t\n\ngo 1.22\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			bin := filepath.Join(ws, "out")
			cmd := exec.Command("go", "build", "-o", bin, ".")
			cmd.Dir = ws
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("go build: %v\n%s", err, out)
			}
		})
	}
}
