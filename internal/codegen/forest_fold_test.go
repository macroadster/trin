package codegen

import (
	"strings"
	"testing"

	"github.com/ericchien/trin/internal/parser"
)

func TestRewriteSoftMuxForestFold(t *testing.T) {
	// soft_mux(0.5, soft_mux(0.5, soft_mux(0.5,10,2), 2), soft_mux(0.25,10,2))
	// wn = 0.5*0.5*0.5 + 0.5*0.25 = 0.25 → y = 0.25*10 + 0.75*2 = 4
	mlir := `
module attributes {trin.version = "0.39"} {
  func.func @trin_soft_mux(%w: f32, %p: f32, %e: f32) -> f32 { return %p : f32 }
  func.func @probe() -> f32 {
    %w = arith.constant 5.000000e-01 : f32
    %w2 = arith.constant 5.000000e-01 : f32
    %w3 = arith.constant 5.000000e-01 : f32
    %w4 = arith.constant 2.500000e-01 : f32
    %p = arith.constant 1.000000e+01 : f32
    %e = arith.constant 2.000000e+00 : f32
    %leaf = func.call @trin_soft_mux(%w3, %p, %e) : (f32, f32, f32) -> f32
    %mid = func.call @trin_soft_mux(%w2, %leaf, %e) : (f32, f32, f32) -> f32
    %arm = func.call @trin_soft_mux(%w4, %p, %e) : (f32, f32, f32) -> f32
    %y = func.call @trin_soft_mux(%w, %mid, %arm) : (f32, f32, f32) -> f32
    return %y : f32
  }
}
`
	out, err := RewriteMLIR(mlir)
	if err != nil {
		t.Fatal(err)
	}
	// Forest algebra may fold via stepwise product + both-arms (same wn) or
	// the dedicated forest rule; either way the outer result is const 4.
	if !strings.Contains(out, `trin.version = "0.81"`) {
		t.Fatalf("missing version 0.50:\n%s", out)
	}
	if !strings.Contains(out, "arith.constant 4.000000e+00 : f32") {
		t.Fatalf("expected forest algebra → 4:\n%s", out)
	}
	if !strings.Contains(out, "soft_mux forest product both-arms") &&
		!strings.Contains(out, "soft_mux(w, soft_mux(w2,p,e), soft_mux(w3,p,e))") &&
		!strings.Contains(out, "soft_mux(w1, soft_mux(w2,p,e), e)") {
		t.Fatalf("expected some soft_mux nest fold note:\n%s", out)
	}
}

func TestRewriteGEMMScheduleEqualWords(t *testing.T) {
	// x4 of word with ap=bp=0xF → pop 4; equal schedule → 16
	mlir := `
module attributes {trin.version = "0.39"} {
  func.func @trin_soft_mux(%w: f32, %p: f32, %e: f32) -> f32 { return %p : f32 }
  func.func @trin_trit_dot_planes_word(%ap: i32, %an: i32, %bp: i32, %bn: i32) -> i32 { return %ap : i32 }
  func.func @trin_trit_dot_planes_x4(
      %ap0: i32, %an0: i32, %bp0: i32, %bn0: i32,
      %ap1: i32, %an1: i32, %bp1: i32, %bn1: i32,
      %ap2: i32, %an2: i32, %bp2: i32, %bn2: i32,
      %ap3: i32, %an3: i32, %bp3: i32, %bn3: i32) -> i32 { return %ap0 : i32 }
  func.func @probe() -> i32 {
    %ap = arith.constant 15 : i32
    %an = arith.constant 0 : i32
    %bp = arith.constant 15 : i32
    %bn = arith.constant 0 : i32
    %y = func.call @trin_trit_dot_planes_x4(%ap, %an, %bp, %bn, %ap, %an, %bp, %bn, %ap, %an, %bp, %bn, %ap, %an, %bp, %bn) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    return %y : i32
  }
}
`
	out, err := RewriteMLIR(mlir)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"trit_dot_planes_x4(schedule equal-words)",
		"arith.constant 16 : i32",
		`trin.version = "0.81"`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q:\n%s", want, out)
		}
	}
}

func TestEmitMLIRForestProbe(t *testing.T) {
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
		`trin.soft_forest = "deep_schedule"`,
		`trin.map_store_peel = "mul_axpy"`,
		"func.func @trin_mlir_forest_probe",
		"func.func @trin_span_mul_f32",
		"func.func @trin_span_axpy_f32",
		"soft_mux forest",
	} {
		if !strings.Contains(mlir, want) {
			t.Fatalf("missing %q in emit-mlir:\n%s", want, mlir[max(0, len(mlir)-5000):])
		}
	}
	ll, err := EmitMLIRLLVM(prog)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"define float @trin_mlir_forest_probe",
		"ret float 8.000000e+01",
		"define void @trin_span_mul_f32",
		"define void @trin_span_axpy_f32",
		"define float @trin_map_store_peel_probe",
		"v0.67",
	} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q in lower:\n%s", want, ll[max(0, len(ll)-4000):])
		}
	}
}

func TestMapStorePeelKernels(t *testing.T) {
	src := `
fn soft_kernel(w, a, x, b) {
    soft_mux(w, a * x, b)
}
`
	prog, err := parser.ParseSource(src)
	if err != nil {
		t.Fatal(err)
	}
	ll, err := EmitLLVM(prog)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"define void @trin_span_mul_f32",
		"define void @trin_span_axpy_f32",
		"map/store peel probe",
		"define float @trin_map_store_peel_probe",
		"fmul <16 x float>",
		"fadd <16 x float>",
		"peel16.cond",
	} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q in emit-llvm map peel:\n%s", want, ll[:min(3000, len(ll))])
		}
	}
	mulIdx := strings.Index(ll, "define void @trin_span_mul_f32")
	axpyIdx := strings.Index(ll, "define void @trin_span_axpy_f32")
	if mulIdx < 0 || axpyIdx < 0 {
		t.Fatal("missing mul/axpy kernels")
	}
	mulBody := ll[mulIdx:axpyIdx]
	for _, want := range []string{
		"call i32 @trin_adaptive_peel_tile",
		"fmul <16 x float>",
		"fmul <8 x float>",
		"fmul <4 x float>",
		"store <16 x float>",
	} {
		if !strings.Contains(mulBody, want) {
			t.Fatalf("span_mul missing %q", want)
		}
	}
}
