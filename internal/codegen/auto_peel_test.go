package codegen

import (
	"strings"
	"testing"

	"github.com/macroadster/trin/internal/parser"
)

func TestAutoPeelSpanLoops(t *testing.T) {
	src := `
fn main() {
    let s = sum([1.0, 2.0, 3.0]);
    println(to_str(s));
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
		"auto residual peel span loops (v0.36–v0.67)",
		"define float @trin_sum_f32",
		"define float @trin_dot_f32",
		"define void @trin_span_add_f32",
		"define void @trin_span_scale_f32",
		"peel16.cond",
		"peel8.cond",
		"peel4.cond",
		"scalar.cond",
		"define float @trin_auto_peel_probe",
		"define float @trin_store_peel_probe",
		"call i32 @trin_adaptive_peel_tile",
		"define i32 @trin_adaptive_peel_tile",
		"define i32 @trin_hw_vlen_for_n",
		"!\"llvm.loop.vectorize.enable\"",
		// v0.37 store peels use vector fadd / fmul stores
		"store <16 x float>",
		"store <8 x float>",
		"store <4 x float>",
		"fmul <16 x float>",
	} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q in emit-llvm auto peel:\n%s", want, ll[:min(2500, len(ll))])
		}
	}
	// classic probe still present
	if !strings.Contains(ll, "define float @trin_vec_loop_probe") {
		t.Fatal("missing vec loop probe")
	}
}

func TestStorePeelKernels(t *testing.T) {
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
	// Extract span_add body slice roughly via markers
	if !strings.Contains(ll, "store peel probe") {
		t.Fatal("missing store peel probe comment")
	}
	// span_add / scale should call hw_vlen and vector store
	addIdx := strings.Index(ll, "define void @trin_span_add_f32")
	scaleIdx := strings.Index(ll, "define void @trin_span_scale_f32")
	if addIdx < 0 || scaleIdx < 0 {
		t.Fatal("missing store kernels")
	}
	addBody := ll[addIdx:scaleIdx]
	for _, want := range []string{
		"call i32 @trin_adaptive_peel_tile",
		"peel16.cond",
		"peel8.cond",
		"peel4.cond",
		"fadd <16 x float>",
		"fadd <8 x float>",
		"fadd <4 x float>",
		"store <16 x float>",
	} {
		if !strings.Contains(addBody, want) {
			t.Fatalf("span_add missing %q", want)
		}
	}
	// scale section until next define
	rest := ll[scaleIdx:]
	end := strings.Index(rest[10:], "\ndefine ")
	scaleBody := rest
	if end >= 0 {
		scaleBody = rest[:end+10]
	}
	for _, want := range []string{
		"call i32 @trin_adaptive_peel_tile",
		"fmul <16 x float>",
		"fmul <8 x float>",
		"fmul <4 x float>",
		"store <16 x float>",
		"shufflevector <16 x float>",
	} {
		if !strings.Contains(scaleBody, want) {
			t.Fatalf("span_scale missing %q", want)
		}
	}
}

func TestEmitMLIRAutoPeelAttr(t *testing.T) {
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
		`trin.residual_peel = "auto"`,
		`trin.auto_peel = "pointer_store_map_scan_mul_multi_tuple"`,
		"Auto residual peel",
		"func.func @trin_sum_f32",
	} {
		if !strings.Contains(mlir, want) {
			t.Fatalf("missing %q in MLIR:\n%s", want, mlir[:min(1500, len(mlir))])
		}
	}
	ll, err := EmitMLIRLLVM(prog)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"auto residual peel",
		"define float @trin_sum_f32",
		"peel16.cond",
		"define float @trin_auto_peel_probe",
	} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q in lower:\n%s", want, ll[:min(2000, len(ll))])
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
