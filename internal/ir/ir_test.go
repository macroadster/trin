package ir

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/ericchien/trin/internal/parser"
)

func TestEmitSoftMux(t *testing.T) {
	src := `
fn main() {
    let g = unit_of(0.8);
    let y = soft_mux(g, 10.0, 0.0);
    println(to_str(y));
}
`
	prog, err := parser.ParseSource(src)
	if err != nil {
		t.Fatal(err)
	}
	m, err := Emit(prog)
	if err != nil {
		t.Fatal(err)
	}
	s := m.String()
	if !strings.Contains(s, "soft_mux") {
		t.Fatalf("missing soft_mux:\n%s", s)
	}
	if !strings.Contains(s, "func @main") {
		t.Fatal("missing main")
	}
}

func TestEmitExamples(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..", "..", "examples")
	for _, name := range []string{"soft_mux.trin", "trit_gemm.trin", "hello.trin"} {
		src, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			t.Fatal(err)
		}
		prog, err := parser.ParseSource(string(src))
		if err != nil {
			t.Fatalf("%s parse: %v", name, err)
		}
		m, err := Emit(prog)
		if err != nil {
			t.Fatalf("%s emit: %v", name, err)
		}
		if len(m.Funcs) == 0 {
			t.Fatalf("%s no funcs", name)
		}
	}
}
