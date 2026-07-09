package codegen

import (
	"strings"
	"testing"

	"github.com/ericchien/trin/internal/parser"
)

func TestEmitFullProgramLLVM(t *testing.T) {
	src := `
fn main() {
    let a = 3.0;
    let b = 4.0;
    let p = a * b;
    let m = soft_mux(0.25, 10.0, 2.0);
    let s = sum([1.0, 2.0, 3.0, 4.0]);
    let d = dot([1.0, 2.0], [3.0, 4.0]);
    let xs = span_add([1.0, 2.0], [3.0, 4.0]);
    let sx = sum(xs);
    println(str_concat("product = ", to_str(p)));
    println(str_concat("soft_mux = ", to_str(m)));
    println(str_concat("sum = ", to_str(s)));
    println(str_concat("dot = ", to_str(d)));
    println(str_concat("span_add sum = ", to_str(sx)));
    println(str_concat("simd_lane_width = ", to_str(simd_lane_width())));
}
`
	prog, err := parser.ParseSource(src)
	if err != nil {
		t.Fatal(err)
	}
	full := emitFullProgram(prog)
	if full == "" {
		t.Fatal("emitFullProgram returned empty")
	}
	for _, want := range []string{"define i32 @main(", "trin_println_str", "trin_soft_mux", "product = ", "@.str.1"} {
		if !strings.Contains(full, want) {
			t.Fatalf("missing %q in:\n%s", want, full)
		}
	}
	ll, err := EmitLLVM(prog)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(ll, "define i32 @main(") {
		t.Fatalf("EmitLLVM missing main:\n%s", ll)
	}
	if LLVMHostDriver("main", ll) != "" {
		t.Fatalf("host should be empty when main present")
	}
	// pure float kernel modules still get a host
	src2 := `
fn loss(w: f32, a: f32, x: f32, b: f32) {
    soft_mux(w, a * x, b)
}
fn main() {
    let y = soft_mux(0.5, 3.0, 1.0);
    println(to_str(y));
}
`
	prog2, err := parser.ParseSource(src2)
	if err != nil {
		t.Fatal(err)
	}
	ll2, err := EmitLLVM(prog2)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(ll2, "define i32 @main(") {
		t.Fatalf("loss/kernel modules should keep float-host path, not embed @main")
	}
	if !strings.Contains(ll2, "define float @loss(") {
		t.Fatal("loss missing")
	}
	host := LLVMHostDriver("loss", ll2)
	if !strings.Contains(host, "float loss(float") {
		t.Fatalf("host:\n%s", host)
	}
}
