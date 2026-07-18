package codegen

import (
	"strings"
	"testing"

	"github.com/macroadster/trin/internal/parser"
)

func TestRewriteBitplaneAlgebraic(t *testing.T) {
	mlir := `
module attributes {trin.version = "0.32"} {
  func.func @trin_plane_and(%a: i32, %b: i32) -> i32 { return %a : i32 }
  func.func @trin_plane_or(%a: i32, %b: i32) -> i32 { return %a : i32 }
  func.func @trin_plane_xor(%a: i32, %b: i32) -> i32 { return %a : i32 }
  func.func @trin_plane_popcount(%w: i32) -> i32 { return %w : i32 }
  func.func @probe() -> i32 {
    %x = arith.constant 15 : i32
    %z = arith.constant 0 : i32
    %a1 = func.call @trin_plane_and(%x, %x) : (i32, i32) -> i32
    %a0 = func.call @trin_plane_and(%x, %z) : (i32, i32) -> i32
    %o0 = func.call @trin_plane_or(%x, %z) : (i32, i32) -> i32
    %xr = func.call @trin_plane_xor(%x, %x) : (i32, i32) -> i32
    %pc = func.call @trin_plane_popcount(%x) : (i32) -> i32
    %pz = func.call @trin_plane_popcount(%z) : (i32) -> i32
    return %pc : i32
  }
}
`
	out, err := RewriteMLIR(mlir)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`trin.version = "0.81"`,
		"bitplane=",
		"plane_and(x,x) → x",
		"plane_and(x,0) → 0",
		"plane_or(x,0) → x",
		"plane_xor(x,x) → 0",
		"plane_popcount(const) fold",
		"plane_popcount(0) → 0",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in rewritten MLIR:\n%s", want, out)
		}
	}
	if strings.Contains(out, "func.call @trin_plane_and(%x, %x)") ||
		strings.Contains(out, "func.call @trin_plane_xor(%x, %x)") {
		t.Fatalf("expected plane calls folded away:\n%s", out)
	}
}

func TestRewriteSoftMuxVStrength(t *testing.T) {
	mlir := `
module attributes {trin.version = "0.32"} {
  func.func @trin_soft_mux(%w: f32, %p: f32, %e: f32) -> f32 { return %p : f32 }
  func.func @trin_soft_mux_v4(%w: f32, %p: vector<4xf32>, %e: vector<4xf32>) -> vector<4xf32> {
    return %p : vector<4xf32>
  }
  func.func @probe() -> vector<4xf32> {
    %w = arith.constant 5.000000e-01 : f32
    %c2 = arith.constant 2.000000e+00 : f32
    %z0 = arith.constant 0.000000e+00 : f32
    %pure = vector.from_elements %c2, %c2, %c2, %c2 : vector<4xf32>
    %zv = vector.broadcast %z0 : f32 to vector<4xf32>
    %m0 = func.call @trin_soft_mux_v4(%w, %pure, %zv) : (f32, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %m1 = func.call @trin_soft_mux_v4(%w, %zv, %pure) : (f32, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    return %m0 : vector<4xf32>
  }
}
`
	out, err := RewriteMLIR(mlir)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"soft_mux_v4(w,pure,0) → broadcast(w)*pure",
		"soft_mux_v4(w,0,each) → broadcast(1-w)*each",
		"strength=",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "func.call @trin_soft_mux_v4(%w, %pure, %zv)") {
		t.Fatal("each=0 arm should be strength-reduced")
	}
}

func TestEmitMLIRBitplaneProbe(t *testing.T) {
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
		"func.func @trin_mlir_bitplane_probe",
		"bitplane=",
		"plane_and(x,x) → x",
		"soft_mux_v4(w,pure,0)",
		"soft_mux_v4(w,0,each)",
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
		"define float @trin_mlir_bitplane_probe",
		"ret float 4.200000e+01",
		"v0.33",
	} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q in lower:\n%s", want, ll)
		}
	}
}
