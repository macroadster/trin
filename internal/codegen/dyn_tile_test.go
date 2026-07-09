package codegen

import (
	"strings"
	"testing"

	"github.com/ericchien/trin/internal/parser"
)

func TestDynTileHelpersPresent(t *testing.T) {
	s := emitDynTileLLVMHelpers()
	for _, want := range []string{
		"dynamic tile selection for full-program @main (v0.32)",
		"define float @trin_dyn_sum4",
		"define float @trin_dyn_sum8",
		"define float @trin_dyn_sum16",
		"define float @trin_dyn_sum32",
		"define float @trin_dyn_dot4",
		"define float @trin_dyn_dot8",
		"define i32 @trin_dyn_tile",
		"call i32 @trin_hw_vlen_for_n",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %q in dyn tile helpers:\n%s", want, s)
		}
	}
}

func TestFullProgramUsesDynTiles(t *testing.T) {
	src := `
fn main() {
    let s = sum([1.0, 2.0, 3.0, 4.0]);
    let d = dot([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]);
    let t = dyn_tile(8);
    println(str_concat("sum = ", to_str(s)));
    println(str_concat("dot = ", to_str(d)));
    println(str_concat("tile = ", to_str(t)));
}
`
	prog, err := parser.ParseSource(src)
	if err != nil {
		t.Fatal(err)
	}
	full := emitFullProgram(prog)
	if full == "" {
		t.Fatal("emitFullProgram empty")
	}
	for _, want := range []string{
		"define float @trin_dyn_sum4",
		"define i32 @trin_dyn_tile",
		"define i32 @trin_hw_vlen_for_n",
		"call float @trin_dyn_sum4",
		"call float @trin_dyn_dot4",
		"call i32 @trin_dyn_tile",
		"runtime CPU feature detect",
		"dynamic tile selection for full-program @main (v0.32)",
	} {
		if !strings.Contains(full, want) {
			t.Fatalf("missing %q in full-program IR:\n%s", want, full)
		}
	}
	// Must not fall back to pure scalar fadd chain for the 4-wide sum.
	// (scalar chain may still exist inside helpers as residual path.)
	if !strings.Contains(full, "call float @trin_dyn_sum4") {
		t.Fatal("expected dyn sum dispatch in @main")
	}
}

func TestEmitMLIRDynTileNotes(t *testing.T) {
	src := `
fn main() {
    let s = sum([1.0, 2.0, 3.0, 4.0]);
    let d = dot([1.0, 2.0], [3.0, 4.0]);
    println(to_str(s));
    println(to_str(d));
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
		"dyn tile",
		"full-program dyn_tile sum(n=4)",
		"func.call @trin_hw_vlen_for_n",
	} {
		if !strings.Contains(mlir, want) {
			t.Fatalf("missing %q in MLIR:\n%s", want, mlir)
		}
	}

	ll, err := EmitMLIRLLVM(prog)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"define i32 @main",
		"define float @trin_dyn_sum4",
		"call float @trin_dyn_sum4",
		"dyn tiles v0.32",
	} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q in MLIR→LLVM:\n%s", want, ll)
		}
	}
}

func TestPreferDynTileN(t *testing.T) {
	if !preferDynTileN(4) || !preferDynTileN(8) || !preferDynTileN(16) || !preferDynTileN(32) {
		t.Fatal("expected 4/8/16/32 preferred")
	}
	if preferDynTileN(2) || preferDynTileN(5) || preferDynTileN(0) {
		t.Fatal("unexpected prefer")
	}
}
