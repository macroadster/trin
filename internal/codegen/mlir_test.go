package codegen

import (
	"strings"
	"testing"

	"github.com/macroadster/trin/internal/parser"
)

func TestEmitMLIRBasics(t *testing.T) {
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
		"trin.version = \"0.81\"",
		"trin.vlen = 16",
		"trin.hw_vlen",
		"trin.hw_isa",
		"trin.hw_dispatch = \"runtime\"",
		"func.func @trin_mlir_span_alg_probe",
		"func.func @trin_soft_mux",
		"func.func @trin_sum_f32",
		"func.func @trin_dot_f32",
		"func.func @trin_span_add_f32",
		"func.func @trin_span_scale_f32",
		"func.func @trin_soft_mux_v4",
		"func.func @trin_span_add_v4",
		"func.func @trin_span_scale_v4",
		"func.func @trin_sum_v4",
		"func.func @trin_dot_v4",
		"func.func @trin_soft_mux_v8",
		"func.func @trin_span_add_v8",
		"func.func @trin_span_scale_v8",
		"func.func @trin_sum_v8",
		"func.func @trin_dot_v8",
		"func.func @trin_soft_mux_v16",
		"func.func @trin_span_add_v16",
		"func.func @trin_span_scale_v16",
		"func.func @trin_sum_v16",
		"func.func @trin_dot_v16",
		"func.func @trin_vlen",
		"func.func @trin_vlen_for_n",
		"func.func @trin_hw_vlen",
		"func.func @trin_hw_vlen_for_n",
		"func.func @trin_hw_isa_code",
		"func.func @trin_mlir_vec_probe",
		"func.func @trin_mlir_vec8_probe",
		"func.func @trin_mlir_vec16_probe",
		"func.func @trin_mlir_vlen_probe",
		"func.func @trin_mlir_hw_vlen_probe",
		"func.func @trin_mlir_soft_probe",
		"func.func @trin_mlir_dialect_probe",
		"func.func @trin_mlir_fold_probe",
		"func.func @trin_mlir_vsum_probe",
		"func.func @trin_mlir_cse_probe",
		"func.func @trin_mlir_soft_cse_probe",
		"func.func @trin_mlir_vwidth_probe",
		"func.func @trin_mlir_strength_probe",
		"func.func @trin_mlir_span_alg_probe",
		"func.func @trin_mlir_bitplane_probe",
		"vector<4xf32>",
		"vector<8xf32>",
		"vector<16xf32>",
		"vector.reduction",
		"scf.for",
		"trin.soft_mux",
		"Patterns (v0.",
		"rewrite: dialect_ops=",
		"span_alg=",
		"bitplane=",
		"cse=",
		"vwidth=",
	} {
		if !strings.Contains(mlir, want) {
			t.Fatalf("missing %q in MLIR:\n%s", want, mlir)
		}
	}
}

func TestEmitMLIRVecLoopsExample(t *testing.T) {
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
	mlir, err := EmitMLIR(prog)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(mlir, "vector<4xf32>") {
		t.Fatal("expected vector IR")
	}
	// v0.23: programs with main (no loss/kernel) emit func.func @main
	if !strings.Contains(mlir, "func.func @main()") &&
		!strings.Contains(mlir, "program has fn main") {
		t.Fatal("expected full-program @main or main note")
	}
}
