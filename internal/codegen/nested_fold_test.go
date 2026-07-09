package codegen

import (
	"strings"
	"testing"

	"github.com/ericchien/trin/internal/parser"
)

func TestRewriteNestedSoftMuxProductPureArm(t *testing.T) {
	mlir := `
module attributes {trin.version = "0.34"} {
  func.func @trin_soft_mux(%w: f32, %p: f32, %e: f32) -> f32 { return %p : f32 }
  func.func @probe() -> f32 {
    %w1 = arith.constant 5.000000e-01 : f32
    %w2 = arith.constant 5.000000e-01 : f32
    %p0 = arith.constant 1.000000e+01 : f32
    %e0 = arith.constant 2.000000e+00 : f32
    %z = arith.constant 0.000000e+00 : f32
    %p = arith.addf %p0, %z : f32
    %e = arith.addf %e0, %z : f32
    %inner = func.call @trin_soft_mux(%w2, %p, %e) : (f32, f32, f32) -> f32
    %prod = func.call @trin_soft_mux(%w1, %inner, %e) : (f32, f32, f32) -> f32
    return %prod : f32
  }
}
`
	out, err := RewriteMLIR(mlir)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`trin.version = "0.81"`,
		"nested=",
		"soft_mux(w1, soft_mux(w2,p,e), e) → soft_mux(w1*w2, p, e)",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "func.call @trin_soft_mux(%w1, %inner, %e)") {
		t.Fatal("product nest should rewrite outer call")
	}
}

func TestRewriteNestedSoftMuxProductEachArm(t *testing.T) {
	mlir := `
module attributes {trin.version = "0.34"} {
  func.func @trin_soft_mux(%w: f32, %p: f32, %e: f32) -> f32 { return %p : f32 }
  func.func @probe() -> f32 {
    %w1 = arith.constant 5.000000e-01 : f32
    %w2 = arith.constant 5.000000e-01 : f32
    %p0 = arith.constant 1.000000e+01 : f32
    %e0 = arith.constant 2.000000e+00 : f32
    %z = arith.constant 0.000000e+00 : f32
    %p = arith.addf %p0, %z : f32
    %e = arith.addf %e0, %z : f32
    %inner = func.call @trin_soft_mux(%w2, %p, %e) : (f32, f32, f32) -> f32
    %prod = func.call @trin_soft_mux(%w1, %p, %inner) : (f32, f32, f32) -> f32
    return %prod : f32
  }
}
`
	out, err := RewriteMLIR(mlir)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"soft_mux(w1, p, soft_mux(w2,p,e)) → soft_mux(w1+(1-w1)*w2, p, e)",
		"nested=",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "func.call @trin_soft_mux(%w1, %p, %inner)") {
		t.Fatal("each-arm product nest should rewrite outer call")
	}
}

func TestRewriteBitplaneProductAbsorb(t *testing.T) {
	mlir := `
module attributes {trin.version = "0.34"} {
  func.func @trin_plane_and(%a: i32, %b: i32) -> i32 { return %a : i32 }
  func.func @trin_plane_or(%a: i32, %b: i32) -> i32 { return %a : i32 }
  func.func @probe() -> i32 {
    %x0 = arith.constant 15 : i32
    %y0 = arith.constant 3 : i32
    %z = arith.constant 0 : i32
    %x = arith.addi %x0, %z : i32
    %y = arith.addi %y0, %z : i32
    %a = func.call @trin_plane_and(%x, %y) : (i32, i32) -> i32
    %o = func.call @trin_plane_or(%x, %y) : (i32, i32) -> i32
    %and_abs = func.call @trin_plane_and(%a, %x) : (i32, i32) -> i32
    %or_abs = func.call @trin_plane_or(%o, %x) : (i32, i32) -> i32
    %and_x_o = func.call @trin_plane_and(%x, %o) : (i32, i32) -> i32
    %and_prod = func.call @trin_plane_and(%a, %o) : (i32, i32) -> i32
    %s0 = arith.addi %and_abs, %or_abs : i32
    %s1 = arith.addi %s0, %and_x_o : i32
    %y2 = arith.addi %s1, %and_prod : i32
    return %y2 : i32
  }
}
`
	out, err := RewriteMLIR(mlir)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"plane_and(and(a,b), …) absorb → and(a,b)",
		"plane_or(or(a,b), …) absorb → or(a,b)",
		"plane_and(a, or(a,b)) → a",
		"nested=",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q:\n%s", want, out)
		}
	}
}

func TestEmitMLIRNestedProbe(t *testing.T) {
	src := `
fn soft_kernel(w, a, x, b) {
    soft_mux(w, a * x, b)
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
		"func.func @trin_mlir_nested_probe",
		"nested=",
		"soft_mux(w1, soft_mux(w2,p,e), e) → soft_mux(w1*w2, p, e)",
		"soft_mux(w1, p, soft_mux(w2,p,e)) → soft_mux(w1+(1-w1)*w2, p, e)",
		"plane_and(and(a,b), …) absorb → and(a,b)",
	} {
		if !strings.Contains(mlir, want) {
			t.Fatalf("missing %q in emit-mlir:\n%s", want, mlir)
		}
	}
	ll, err := EmitMLIRLLVM(prog)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"define float @trin_mlir_nested_probe",
		"ret float 5.200000e+01",
		"v0.36",
	} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q in lower:\n%s", want, ll)
		}
	}
}
