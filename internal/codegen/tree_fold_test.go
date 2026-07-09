package codegen

import (
	"strings"
	"testing"

	"github.com/ericchien/trin/internal/parser"
)

func TestRewriteMultiLevelSoftMuxBothArms(t *testing.T) {
	mlir := `
module attributes {trin.version = "0.37"} {
  func.func @trin_soft_mux(%w: f32, %p: f32, %e: f32) -> f32 { return %p : f32 }
  func.func @probe() -> f32 {
    %w = arith.constant 5.000000e-01 : f32
    %w2 = arith.constant 5.000000e-01 : f32
    %w3 = arith.constant 2.500000e-01 : f32
    %p0 = arith.constant 1.000000e+01 : f32
    %e0 = arith.constant 2.000000e+00 : f32
    %z = arith.constant 0.000000e+00 : f32
    %p = arith.addf %p0, %z : f32
    %e = arith.addf %e0, %z : f32
    %in_p = func.call @trin_soft_mux(%w2, %p, %e) : (f32, f32, f32) -> f32
    %in_e = func.call @trin_soft_mux(%w3, %p, %e) : (f32, f32, f32) -> f32
    %y = func.call @trin_soft_mux(%w, %in_p, %in_e) : (f32, f32, f32) -> f32
    return %y : f32
  }
}
`
	out, err := RewriteMLIR(mlir)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`trin.version = "0.81"`,
		"soft_mux(w, soft_mux(w2,p,e), soft_mux(w3,p,e))",
		"nested=",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q:\n%s", want, out)
		}
	}
	// arms are SSA (not pure consts), so weight product folds to 0.375 then soft_mux
	if !strings.Contains(out, "3.750000e-01") {
		t.Fatalf("expected combined weight 0.375:\n%s", out)
	}
	if strings.Contains(out, "func.call @trin_soft_mux(%w, %in_p, %in_e)") {
		t.Fatal("both-arms tree should rewrite outer call")
	}
}

func TestRewriteTriplePureNest(t *testing.T) {
	mlir := `
module attributes {trin.version = "0.37"} {
  func.func @trin_soft_mux(%w: f32, %p: f32, %e: f32) -> f32 { return %p : f32 }
  func.func @probe() -> f32 {
    %w1 = arith.constant 5.000000e-01 : f32
    %w2 = arith.constant 5.000000e-01 : f32
    %w3 = arith.constant 5.000000e-01 : f32
    %p0 = arith.constant 1.000000e+01 : f32
    %e0 = arith.constant 2.000000e+00 : f32
    %z = arith.constant 0.000000e+00 : f32
    %p = arith.addf %p0, %z : f32
    %e = arith.addf %e0, %z : f32
    %a = func.call @trin_soft_mux(%w3, %p, %e) : (f32, f32, f32) -> f32
    %b = func.call @trin_soft_mux(%w2, %a, %e) : (f32, f32, f32) -> f32
    %c = func.call @trin_soft_mux(%w1, %b, %e) : (f32, f32, f32) -> f32
    return %c : f32
  }
}
`
	out, err := RewriteMLIR(mlir)
	if err != nil {
		t.Fatal(err)
	}
	// Stepwise pure product: two product-weight rewrites (0.5*0.5=0.25, then *0.5=0.125)
	if !strings.Contains(out, "soft_mux(w1, soft_mux(w2,p,e), e) → soft_mux(w1*w2, p, e)") {
		t.Fatalf("missing pure product rewrite:\n%s", out)
	}
	if !strings.Contains(out, "2.500000e-01") {
		t.Fatalf("expected mid product weight 0.25:\n%s", out)
	}
	if !strings.Contains(out, "1.250000e-01") {
		t.Fatalf("expected triple product weight 0.125:\n%s", out)
	}
}

func TestRewriteMultiWordPlaneGEMM(t *testing.T) {
	mlir := `
module attributes {trin.version = "0.37"} {
  func.func @trin_trit_dot_planes_word(%ap: i32, %an: i32, %bp: i32, %bn: i32) -> i32 { return %ap : i32 }
  func.func @trin_trit_dot_planes_x4(
      %ap0: i32, %an0: i32, %bp0: i32, %bn0: i32,
      %ap1: i32, %an1: i32, %bp1: i32, %bn1: i32,
      %ap2: i32, %an2: i32, %bp2: i32, %bn2: i32,
      %ap3: i32, %an3: i32, %bp3: i32, %bn3: i32) -> i32 { return %ap0 : i32 }
  func.func @probe() -> i32 {
    %ap = arith.constant 16777215 : i32
    %an = arith.constant 0 : i32
    %bp = arith.constant 16777215 : i32
    %bn = arith.constant 0 : i32
    %z = arith.constant 0 : i32
    %d0 = func.call @trin_trit_dot_planes_word(%ap, %an, %bp, %bn) : (i32, i32, i32, i32) -> i32
    %d1 = func.call @trin_trit_dot_planes_word(%z, %z, %bp, %bn) : (i32, i32, i32, i32) -> i32
    %x4 = func.call @trin_trit_dot_planes_x4(%z, %z, %z, %z, %z, %z, %z, %z, %z, %z, %z, %z, %z, %z, %z, %z) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    %s0 = arith.addi %d0, %d1 : i32
    %y = arith.addi %s0, %x4 : i32
    return %y : i32
  }
}
`
	out, err := RewriteMLIR(mlir)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"trit_dot_planes_word(const) fold",
		"trit_dot_planes_word(zero) → 0",
		"trit_dot_planes_x4(zero) → 0",
		`trin.version = "0.81"`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q:\n%s", want, out)
		}
	}
	// 24 + 0 + 0
	if !strings.Contains(out, "arith.constant 24 : i32") {
		t.Fatalf("expected const 24 for word popcount:\n%s", out)
	}
}

func TestEmitMLIRTreeProbe(t *testing.T) {
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
		"func.func @trin_mlir_tree_probe",
		"soft_mux(w, soft_mux(w2,p,e), soft_mux(w3,p,e))",
		"trit_dot_planes_word",
		"nested=",
	} {
		if !strings.Contains(mlir, want) {
			t.Fatalf("missing %q in emit-mlir:\n%s", want, mlir[max(0, len(mlir)-4000):])
		}
	}
	ll, err := EmitMLIRLLVM(prog)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"define float @trin_mlir_tree_probe",
		"ret float 6.400000e+01",
		"v0.67",
	} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q in lower:\n%s", want, ll)
		}
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
