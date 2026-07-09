package codegen

import (
	"strings"
	"testing"

	"github.com/ericchien/trin/internal/parser"
)

func TestRewriteDialectOps(t *testing.T) {
	// pure/each via addf so full-const strength does not fold the soft_mux call.
	mlir := `
module attributes {trin.version = "0.24"} {
  func.func @trin_soft_mux(%w: f32, %pure: f32, %each: f32) -> f32 {
    return %w : f32
  }
  func.func @trin_sum_v4(%a: vector<4xf32>) -> f32 {
    return %a : f32
  }
  func.func @probe() -> f32 {
    %w = arith.constant 2.500000e-01 : f32
    %p0 = arith.constant 1.000000e+01 : f32
    %e0 = arith.constant 2.000000e+00 : f32
    %z = arith.constant 0.000000e+00 : f32
    %p = arith.addf %p0, %z : f32
    %e = arith.addf %e0, %z : f32
    %y = "trin.soft_mux"(%w, %p, %e) : (f32, f32, f32) -> f32
    return %y : f32
  }
  func.func @vsum() -> f32 {
    %c1 = arith.constant 1.000000e+00 : f32
    %a = vector.from_elements %c1, %c1, %c1, %c1 : vector<4xf32>
    %s = "trin.vsum"(%a) : (vector<4xf32>) -> f32
    return %s : f32
  }
}
`
	out, err := RewriteMLIR(mlir)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, `trin.version = "0.81"`) {
		t.Fatal("expected version bump to 0.37")
	}
	if !strings.Contains(out, "func.call @trin_soft_mux(%w, %p, %e)") {
		t.Fatalf("expected dialect soft_mux rewrite:\n%s", out)
	}
	if !strings.Contains(out, "func.call @trin_sum_v4(%a)") {
		t.Fatalf("expected vsum → sum_v4:\n%s", out)
	}
	if strings.Contains(out, `%y = "trin.soft_mux"`) || strings.Contains(out, `%s = "trin.vsum"`) {
		t.Fatalf("expected high-level ops expanded:\n%s", out)
	}
	if !strings.Contains(out, "rewrite: dialect_ops=2") {
		t.Fatalf("expected dialect_ops=2 note:\n%s", out)
	}
	if !strings.Contains(out, "vector_width=4") {
		t.Fatalf("expected vector-width pick note:\n%s", out)
	}
}

func TestRewriteSoftMuxFolds(t *testing.T) {
	mlir := `
module attributes {trin.version = "0.24"} {
  func.func @trin_soft_mux(%w: f32, %pure: f32, %each: f32) -> f32 {
    return %w : f32
  }
  func.func @fold1() -> f32 {
    %w1 = arith.constant 1.000000e+00 : f32
    %pure = arith.constant 7.000000e+00 : f32
    %each = arith.constant 9.000000e+00 : f32
    %y = "trin.soft_mux"(%w1, %pure, %each) : (f32, f32, f32) -> f32
    return %y : f32
  }
  func.func @fold0() -> f32 {
    %w0 = arith.constant 0.000000e+00 : f32
    %pure = arith.constant 7.000000e+00 : f32
    %each = arith.constant 9.000000e+00 : f32
    %z = "trin.soft_mux"(%w0, %pure, %each) : (f32, f32, f32) -> f32
    return %z : f32
  }
}
`
	out, err := RewriteMLIR(mlir)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "rewrite: soft_mux(1,pure,each) → pure") {
		t.Fatalf("expected w=1 fold:\n%s", out)
	}
	if !strings.Contains(out, "rewrite: soft_mux(0,pure,each) → each") {
		t.Fatalf("expected w=0 fold:\n%s", out)
	}
	if strings.Contains(out, "func.call @trin_soft_mux(%w1") ||
		strings.Contains(out, "func.call @trin_soft_mux(%w0") ||
		strings.Contains(out, `%y = func.call @trin_soft_mux`) ||
		strings.Contains(out, `%z = func.call @trin_soft_mux`) {
		t.Fatalf("expected soft_mux calls folded away:\n%s", out)
	}
	if !strings.Contains(out, "soft_mux_folds=2") {
		t.Fatalf("expected soft_mux_folds=2:\n%s", out)
	}
}

func TestRewriteCSE(t *testing.T) {
	mlir := `
module attributes {trin.version = "0.26"} {
  func.func @cse() -> f32 {
    %c1 = arith.constant 1.000000e+00 : f32
    %c2 = arith.constant 2.000000e+00 : f32
    %a = arith.addf %c1, %c2 : f32
    %b = arith.addf %c1, %c2 : f32
    %y = arith.addf %a, %b : f32
    return %y : f32
  }
}
`
	out, err := RewriteMLIR(mlir)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "rewrite: cse → %a") {
		t.Fatalf("expected CSE of duplicate addf:\n%s", out)
	}
	if strings.Contains(out, "%b = arith.addf") {
		t.Fatalf("duplicate bind should be dropped:\n%s", out)
	}
	if !strings.Contains(out, "arith.addf %a, %a : f32") {
		t.Fatalf("uses should be rewritten to first bind:\n%s", out)
	}
	if !strings.Contains(out, "cse=1") {
		t.Fatalf("expected cse=1:\n%s", out)
	}
}

func TestRewriteSoftMuxMultiUseCSE(t *testing.T) {
	// pure/each via addf so v0.29 full-const strength does not fold the call away
	// before multi-use CSE can common identical soft_mux ops.
	mlir := `
module attributes {trin.version = "0.26"} {
  func.func @trin_soft_mux(%w: f32, %pure: f32, %each: f32) -> f32 {
    return %w : f32
  }
  func.func @soft_cse() -> f32 {
    %w = arith.constant 5.000000e-01 : f32
    %p0 = arith.constant 1.000000e+01 : f32
    %e0 = arith.constant 2.000000e+00 : f32
    %z = arith.constant 0.000000e+00 : f32
    %p = arith.addf %p0, %z : f32
    %e = arith.addf %e0, %z : f32
    %y1 = "trin.soft_mux"(%w, %p, %e) : (f32, f32, f32) -> f32
    %y2 = "trin.soft_mux"(%w, %p, %e) : (f32, f32, f32) -> f32
    %y = arith.addf %y1, %y2 : f32
    return %y : f32
  }
}
`
	out, err := RewriteMLIR(mlir)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "func.call @trin_soft_mux(%w, %p, %e)") {
		t.Fatalf("expected soft_mux expansion:\n%s", out)
	}
	// Second identical call should be commoned.
	if strings.Contains(out, "%y2 = func.call @trin_soft_mux") {
		t.Fatalf("multi-use soft_mux should CSE second call:\n%s", out)
	}
	if !strings.Contains(out, "rewrite: cse → %y1") {
		t.Fatalf("expected soft_mux CSE note:\n%s", out)
	}
	if !strings.Contains(out, "arith.addf %y1, %y1") {
		t.Fatalf("expected y2 uses remapped to y1:\n%s", out)
	}
}

func TestRewriteSoftMuxStrength(t *testing.T) {
	mlir := `
module attributes {trin.version = "0.28"} {
  func.func @trin_soft_mux(%w: f32, %pure: f32, %each: f32) -> f32 {
    return %w : f32
  }
  func.func @ident() -> f32 {
    %w = arith.constant 3.000000e-01 : f32
    %x = arith.constant 5.000000e+00 : f32
    %y = "trin.soft_mux"(%w, %x, %x) : (f32, f32, f32) -> f32
    return %y : f32
  }
  func.func @mul_pure() -> f32 {
    %w = arith.constant 5.000000e-01 : f32
    %p = arith.constant 6.000000e+00 : f32
    %z = arith.constant 0.000000e+00 : f32
    %y = "trin.soft_mux"(%w, %p, %z) : (f32, f32, f32) -> f32
    return %y : f32
  }
  func.func @mul_each() -> f32 {
    %w = arith.constant 5.000000e-01 : f32
    %z = arith.constant 0.000000e+00 : f32
    %e = arith.constant 3.000000e+00 : f32
    %y = "trin.soft_mux"(%w, %z, %e) : (f32, f32, f32) -> f32
    return %y : f32
  }
  func.func @full() -> f32 {
    %w = arith.constant 2.500000e-01 : f32
    %p = arith.constant 1.000000e+01 : f32
    %e = arith.constant 2.000000e+00 : f32
    %y = "trin.soft_mux"(%w, %p, %e) : (f32, f32, f32) -> f32
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
		"soft_mux(w,x,x) → x",
		"soft_mux(w,pure,0) → w*pure",
		"soft_mux(w,0,each) → (1-w)*each",
		"soft_mux(const) → 4",
		"strength=",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in rewrite:\n%s", want, out)
		}
	}
	// No residual soft_mux calls in strength-reduced probes.
	for _, bad := range []string{
		`func.call @trin_soft_mux(%w, %x, %x)`,
		`func.call @trin_soft_mux(%w, %p, %z)`,
		`func.call @trin_soft_mux(%w, %z, %e)`,
		`func.call @trin_soft_mux(%w, %p, %e)`,
	} {
		if strings.Contains(out, bad) {
			t.Fatalf("expected soft_mux call strength-reduced away: %s\n%s", bad, out)
		}
	}
	if !strings.Contains(out, "arith.mulf %w, %p : f32") {
		t.Fatalf("expected each=0 → mulf:\n%s", out)
	}
	if !strings.Contains(out, "arith.subf") || !strings.Contains(out, "arith.mulf") {
		t.Fatalf("expected pure=0 → (1-w)*each:\n%s", out)
	}
}

func TestRewriteVectorWidthPick(t *testing.T) {
	mlir := `
module attributes {trin.version = "0.26"} {
  func.func @trin_sum_v16(%a: vector<16xf32>) -> f32 {
    %y = vector.reduction <add>, %a : vector<16xf32> into f32
    return %y : f32
  }
  func.func @vw() -> f32 {
    %c1 = arith.constant 1.000000e+00 : f32
    %a = vector.from_elements %c1, %c1, %c1, %c1, %c1, %c1, %c1, %c1, %c1, %c1, %c1, %c1, %c1, %c1, %c1, %c1 : vector<16xf32>
    %y = "trin.vsum"(%a) : (vector<16xf32>) -> f32
    return %y : f32
  }
}
`
	out, err := RewriteMLIR(mlir)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "func.call @trin_sum_v16") {
		t.Fatalf("expected vsum → sum_v16:\n%s", out)
	}
	if !strings.Contains(out, "vector_width=16") {
		t.Fatalf("expected width-pick note:\n%s", out)
	}
	if !strings.Contains(out, "vwidth=1") {
		t.Fatalf("expected vwidth=1:\n%s", out)
	}
}

func TestEmitMLIRAppliesRewrites(t *testing.T) {
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
		"rewrite: dialect_ops=",
		"strength=",
		"span_alg=",
		"bitplane=",
		"cse=",
		"vwidth=",
		"Patterns (v0.",
		"func.func @trin_mlir_dialect_probe",
		"func.func @trin_mlir_fold_probe",
		"func.func @trin_mlir_vsum_probe",
		"func.func @trin_mlir_cse_probe",
		"func.func @trin_mlir_soft_cse_probe",
		"func.func @trin_mlir_vwidth_probe",
		"func.func @trin_mlir_strength_probe",
		"func.func @trin_mlir_span_alg_probe",
		"func.func @trin_mlir_bitplane_probe",
		"func.func @trin_mlir_vec16_probe",
		"func.func @trin_vlen",
		"func.func @trin_hw_vlen",
		"func.func @trin_mlir_hw_vlen_probe",
		// rewritten forms
		"func.call @trin_soft_mux",
		"func.call @trin_sum_v4",
		"soft_mux(1,pure,each) → pure",
		"soft_mux(w,x,x) → x",
		"soft_mux(w,pure,0) → w*pure",
		"span_add(x,0) → x",
		"span_scale(1,x) → x",
		"span_scale(0,x) → 0",
		"sum(0) → 0",
		"dot(0,x)|dot(x,0) → 0",
		"plane_and(x,x) → x",
		"soft_mux_v4(w,pure,0)",
		"rewrite: cse →",
		"vector_width=16",
	} {
		if !strings.Contains(mlir, want) {
			t.Fatalf("missing %q in emit-mlir:\n%s", want, mlir)
		}
	}
	if strings.Contains(mlir, `%y = "trin.soft_mux"`) || strings.Contains(mlir, `%y = "trin.vsum"`) {
		t.Fatalf("high-level dialect ops should be rewritten:\n%s", mlir)
	}
}

func TestRewriteSpanVectorAlgebraic(t *testing.T) {
	mlir := `
module attributes {trin.version = "0.30"} {
  func.func @trin_span_add_v4(%a: vector<4xf32>, %b: vector<4xf32>) -> vector<4xf32> {
    return %a : vector<4xf32>
  }
  func.func @trin_span_scale_v4(%s: f32, %a: vector<4xf32>) -> vector<4xf32> {
    return %a : vector<4xf32>
  }
  func.func @trin_sum_v4(%a: vector<4xf32>) -> f32 {
    return %a : f32
  }
  func.func @trin_dot_v4(%a: vector<4xf32>, %b: vector<4xf32>) -> f32 {
    return %a : f32
  }
  func.func @probe() -> f32 {
    %c1 = arith.constant 1.000000e+00 : f32
    %c2 = arith.constant 2.000000e+00 : f32
    %c3 = arith.constant 3.000000e+00 : f32
    %c4 = arith.constant 4.000000e+00 : f32
    %z0 = arith.constant 0.000000e+00 : f32
    %one = arith.constant 1.000000e+00 : f32
    %a = vector.from_elements %c1, %c2, %c3, %c4 : vector<4xf32>
    %z = vector.broadcast %z0 : f32 to vector<4xf32>
    %add0 = "trin.vspan_add"(%a, %z) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %add1 = "trin.vspan_add"(%z, %a) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %sc1 = "trin.vspan_scale"(%one, %a) : (f32, vector<4xf32>) -> vector<4xf32>
    %sc0 = "trin.vspan_scale"(%z0, %a) : (f32, vector<4xf32>) -> vector<4xf32>
    %s0 = "trin.vsum"(%add0) : (vector<4xf32>) -> f32
    %s1 = "trin.vsum"(%add1) : (vector<4xf32>) -> f32
    %s2 = "trin.vsum"(%sc1) : (vector<4xf32>) -> f32
    %s3 = "trin.vsum"(%sc0) : (vector<4xf32>) -> f32
    %s4 = "trin.vsum"(%z) : (vector<4xf32>) -> f32
    %s5 = "trin.vdot"(%a, %z) : (vector<4xf32>, vector<4xf32>) -> f32
    %t0 = arith.addf %s0, %s1 : f32
    %t1 = arith.addf %t0, %s2 : f32
    %t2 = arith.addf %t1, %s3 : f32
    %t3 = arith.addf %t2, %s4 : f32
    %y = arith.addf %t3, %s5 : f32
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
		"span_add(x,0) → x",
		"span_add(0,x) → x",
		"span_scale(1,x) → x",
		"span_scale(0,x) → 0",
		"sum(0) → 0",
		"dot(0,x)|dot(x,0) → 0",
		"span_alg=",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in rewrite:\n%s", want, out)
		}
	}
	// Residual helper calls for these folds should be gone for the zero cases.
	if strings.Contains(out, "func.call @trin_span_add_v4(%a, %z)") ||
		strings.Contains(out, "func.call @trin_span_add_v4(%z, %a)") ||
		strings.Contains(out, "func.call @trin_span_scale_v4(%one, %a)") ||
		strings.Contains(out, "func.call @trin_span_scale_v4(%z0, %a)") ||
		strings.Contains(out, "func.call @trin_sum_v4(%z)") ||
		strings.Contains(out, "func.call @trin_dot_v4(%a, %z)") {
		t.Fatalf("expected span/vector folds to remove helper calls:\n%s", out)
	}
}

func TestLowerDialectProbes(t *testing.T) {
	src := `
fn main() {
    let s = sum([1.0, 2.0, 3.0, 4.0]);
    println(to_str(s));
}
`
	prog, err := parser.ParseSource(src)
	if err != nil {
		t.Fatal(err)
	}
	ll, err := EmitMLIRLLVM(prog)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"define float @trin_mlir_dialect_probe",
		"define float @trin_mlir_fold_probe",
		"define float @trin_mlir_vsum_probe",
		"define float @trin_mlir_cse_probe",
		"define float @trin_mlir_soft_cse_probe",
		"define float @trin_mlir_vwidth_probe",
		"define float @trin_mlir_strength_probe",
		"define float @trin_mlir_span_alg_probe",
		"define float @trin_mlir_bitplane_probe",
		"ret float 7.000000e+00",
		"ret float 1.350000e+01",
		"ret float 3.000000e+01",
		"ret float 4.200000e+01",
		"dialect pattern rewrites",
		"deeper rewrites: CSE",
		"soft_mux strength reduction",
		"span/vector algebraic rewrites",
		"soft_mux_v + bitplane algebraic",
	} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q in lowered IR:\n%s", want, ll)
		}
	}
}

func TestRewriteRejectsGarbage(t *testing.T) {
	if _, err := RewriteMLIR("not mlir"); err == nil {
		t.Fatal("expected error")
	}
}
