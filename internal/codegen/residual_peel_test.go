package codegen

import (
	"strings"
	"testing"

	"github.com/macroadster/trin/internal/parser"
)

func TestPeelDynTileChunks(t *testing.T) {
	// v0.42: peelDynTileChunks follows host AdaptivePeelChunks.
	hw := DetectHostCPU().HWVLEN
	if hw < 4 {
		hw = 16
	}
	cases := []int{0, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 20, 24, 30, 32, 40}
	for _, n := range cases {
		tiles, r := peelDynTileChunks(n)
		wantTiles, wantR := AdaptivePeelChunks(hw, n)
		if hw < 16 {
			wantTiles, wantR = AdaptivePeelChunks(hw, n)
		}
		if r != wantR {
			t.Fatalf("n=%d residual got %d want %d tiles=%v (hw=%d)", n, r, wantR, tiles, hw)
		}
		if len(tiles) != len(wantTiles) {
			t.Fatalf("n=%d tiles got %v want %v (hw=%d)", n, tiles, wantTiles, hw)
		}
		for i := range tiles {
			if tiles[i] != wantTiles[i] {
				t.Fatalf("n=%d tiles got %v want %v (hw=%d)", n, tiles, wantTiles, hw)
			}
		}
	}
	// Explicit AdaptivePeelChunks wide cases independent of host.
	tiles, r := AdaptivePeelChunks(32, 40)
	if len(tiles) != 2 || tiles[0] != 32 || tiles[1] != 8 || r != 0 {
		t.Fatalf("AdaptivePeelChunks(32,40)=%v/%d", tiles, r)
	}
	tiles, r = AdaptivePeelChunks(16, 20)
	if len(tiles) != 2 || tiles[0] != 16 || tiles[1] != 4 || r != 0 {
		t.Fatalf("AdaptivePeelChunks(16,20)=%v/%d", tiles, r)
	}
}

func TestPreferResidualPeel(t *testing.T) {
	if preferResidualPeel(4) || preferResidualPeel(8) || preferResidualPeel(16) || preferResidualPeel(32) {
		t.Fatal("exact dyn tiles should not use residual peel path")
	}
	if preferResidualPeel(3) || preferResidualPeel(0) {
		t.Fatal("n<4 should stay pure scalar")
	}
	if !preferResidualPeel(5) || !preferResidualPeel(10) || !preferResidualPeel(20) {
		t.Fatal("expected residual peel for 5/10/20")
	}
}

func TestFullProgramResidualPeel(t *testing.T) {
	src := `
fn main() {
    let s5 = sum([1.0, 2.0, 3.0, 4.0, 5.0]);
    let s10 = sum([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let d5 = dot([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 1.0, 1.0, 1.0, 1.0]);
    println(str_concat("sum5 = ", to_str(s5)));
    println(str_concat("sum10 = ", to_str(s10)));
    println(str_concat("dot5 = ", to_str(d5)));
}
`
	prog, err := parser.ParseSource(src)
	if err != nil {
		t.Fatal(err)
	}
	full := emitFullProgram(prog)
	if full == "" {
		t.Fatal("empty full program")
	}
	for _, want := range []string{
		"residual peel",
		"rewrite: residual peel sum(n=5)",
		"rewrite: residual peel sum(n=10)",
		"AdaptivePeelChunks",
		"call float @trin_dyn_sum4",
		"call float @trin_dyn_sum8",
		// sum5 = dyn_sum4 + residual 1
		// sum10 = dyn_sum8 + residual 2
	} {
		if !strings.Contains(full, want) {
			t.Fatalf("missing %q in full IR:\n%s", want, full)
		}
	}
	// should peel, not pure 10-long scalar-only fadd chain only
	if !strings.Contains(full, "tiles=[8]") && !strings.Contains(full, "tiles=[8] residual=2") {
		// comment format uses Go %v → [8]
		if !strings.Contains(full, "sum(n=10)") {
			t.Fatalf("missing sum10 peel note:\n%s", full)
		}
	}
}

func TestEmitMLIRResidualPeelNotes(t *testing.T) {
	src := `
fn main() {
    let s5 = sum([1.0, 2.0, 3.0, 4.0, 5.0]);
    let d6 = dot([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    println(to_str(s5));
    println(to_str(d6));
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
		"residual peel sum(n=5)",
		"residual peel dot(n=6)",
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
		"call float @trin_dyn_sum4",
		"residual peel",
	} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q in MLIR→LLVM:\n%s", want, ll)
		}
	}
}
