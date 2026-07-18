package codegen

import (
	"strings"
	"testing"

	"github.com/macroadster/trin/internal/parser"
)

func TestLowerMLIRToLLVMFromProg(t *testing.T) {
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
		"MLIR→LLVM lowering",
		"define float @trin_soft_mux",
		"define float @trin_sum_f32",
		"define float @trin_dot_f32",
		"define void @trin_span_add_f32",
		"define void @trin_span_scale_f32",
		"define <4 x float> @trin_soft_mux_v4",
		"define <4 x float> @trin_span_add_v4",
		"define <4 x float> @trin_span_scale_v4",
		"define float @trin_sum_v4",
		"define float @trin_dot_v4",
		"define <8 x float> @trin_soft_mux_v8",
		"define <8 x float> @trin_span_add_v8",
		"define <8 x float> @trin_span_scale_v8",
		"define float @trin_sum_v8",
		"define float @trin_dot_v8",
		"define <16 x float> @trin_soft_mux_v16",
		"define <16 x float> @trin_span_add_v16",
		"define <16 x float> @trin_span_scale_v16",
		"define float @trin_sum_v16",
		"define float @trin_dot_v16",
		"define i32 @trin_vlen",
		"define i32 @trin_vlen_for_n",
		"define i32 @trin_hw_vlen",
		"define i32 @trin_hw_vlen_for_n",
		"define i32 @trin_hw_isa_code",
		"define float @trin_mlir_vec_probe",
		"define float @trin_mlir_vec8_probe",
		"define float @trin_mlir_vec16_probe",
		"define float @trin_mlir_vlen_probe",
		"define float @trin_mlir_hw_vlen_probe",
		"define float @trin_mlir_soft_probe",
		"define float @trin_mlir_dialect_probe",
		"define float @trin_mlir_fold_probe",
		"define float @trin_mlir_vsum_probe",
		"define float @trin_mlir_cse_probe",
		"define float @trin_mlir_soft_cse_probe",
		"define float @trin_mlir_vwidth_probe",
		"define float @trin_mlir_strength_probe",
		"define float @trin_mlir_span_alg_probe",
		"define float @trin_mlir_bitplane_probe",
		"span/vector algebraic rewrites",
		"hardware VLEN selection",
		"runtime dynamic feature dispatch",
		"runtime CPU feature detect (v0.30)",
		"call void @trin_hw_detect_once",
		"llvm.loop.vectorize.enable",
	} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q in lowered LLVM:\n%s", want, ll)
		}
	}
}

func TestLowerMLIRVec8ProbeValue(t *testing.T) {
	src := `
fn main() {
    let s = sum([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
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
	if !strings.Contains(ll, "trin_mlir_vec8_probe → target 36.0") {
		t.Fatal("expected vec8 probe note")
	}
	if !strings.Contains(ll, "wider vector tiles") {
		t.Fatal("expected wider tiles note")
	}
}

func TestLowerMLIRToLLVMFromText(t *testing.T) {
	mlir := `
module attributes {trin.version = "0.21"} {
  func.func @trin_soft_mux(%w: f32, %pure: f32, %each: f32) -> f32 {
    return %w : f32
  }
  func.func @trin_sum_v4(%a: vector<4xf32>) -> f32 {
    return %a : f32
  }
}
`
	ll, err := LowerMLIRToLLVM(mlir)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(ll, "define float @trin_soft_mux") {
		t.Fatal("expected soft_mux lower")
	}
	if !strings.Contains(ll, "define float @trin_sum_v4") {
		t.Fatal("expected sum_v4 lower")
	}
	// sum_f32 not present in this minimal mlir → should not appear
	if strings.Contains(ll, "define float @trin_sum_f32") {
		t.Fatal("unexpected sum_f32 for minimal mlir")
	}
}

func TestLowerMLIRRejectsGarbage(t *testing.T) {
	_, err := LowerMLIRToLLVM("not mlir at all")
	if err == nil {
		t.Fatal("expected error")
	}
}
