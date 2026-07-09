package codegen

import (
	"strings"
	"testing"

	"github.com/ericchien/trin/internal/parser"
)

func TestEmitFullProgramMLIR(t *testing.T) {
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
	mlir, err := EmitMLIR(prog)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`trin.version = "0.81"`,
		`trin.dyn_tile = "runtime_32"`,
		"full-program MLIR",
		"func.func @main()",
		"func.func @trin_soft_mux",
		"func.call @trin_soft_mux",
		"arith.mulf",
		"func.call @trin_println_str",
		"simd_lane_width",
		"dyn tile",
		"full-program dyn_tile sum",
	} {
		if !strings.Contains(mlir, want) {
			t.Fatalf("missing %q in MLIR:\n%s", want, mlir)
		}
	}
	// Should not claim "future work" when main lowered.
	if strings.Contains(mlir, "full-program MLIR lowering is future work") {
		t.Fatal("stale future-work note")
	}

	ll, err := EmitMLIRLLVM(prog)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"define i32 @main(",
		"full-program MLIR→LLVM",
		"trin_println_str",
		"product = ",
		"define float @trin_soft_mux",
		"define float @trin_dyn_sum4",
		"call float @trin_dyn_sum4",
		"define i32 @trin_hw_vlen_for_n",
		"dyn tiles v0.32",
	} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q in lowered LL:\n%s", want, ll)
		}
	}
	if LLVMHostDriver("main", ll) != "" {
		t.Fatal("host should be empty when @main present")
	}
}

func TestEmitMLIRKeepsFloatHostForLoss(t *testing.T) {
	src := `
fn loss(w: f32, a: f32, x: f32, b: f32) {
    soft_mux(w, a * x, b)
}
fn main() {
    let y = soft_mux(0.5, 3.0, 1.0);
    println(to_str(y));
}
`
	prog, err := parser.ParseSource(src)
	if err != nil {
		t.Fatal(err)
	}
	mlir, err := EmitMLIR(prog)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(mlir, "func.func @main()") {
		t.Fatal("loss modules must not emit full-program MLIR @main")
	}
	if !strings.Contains(mlir, "loss/kernel") && !strings.Contains(mlir, "float-host") {
		// still ok if only outline notes
		if !strings.Contains(mlir, "user fn @loss") && !strings.Contains(mlir, "program has fn main") {
			t.Fatalf("expected note about host path:\n%s", mlir)
		}
	}
	ll, err := EmitMLIRLLVM(prog)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(ll, "define i32 @main(") {
		t.Fatal("EmitMLIRLLVM must not append @main for loss modules")
	}
}
