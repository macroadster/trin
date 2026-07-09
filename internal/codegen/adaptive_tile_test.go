package codegen

import (
	"strings"
	"testing"

	"github.com/ericchien/trin/internal/parser"
)

func TestAdaptiveTileForISA(t *testing.T) {
	cases := []struct {
		isa, want int
		name      string
	}{
		{ISAPortable, 16, "portable"},
		{ISASSE, 4, "sse"},
		{ISAAVX, 8, "avx"},
		{ISAAVX512, 16, "avx512"},
		{ISASVE, 32, "sve"},
		{ISANEON, 4, "neon"},
	}
	for _, c := range cases {
		if AdaptiveTileForISA(c.isa) != c.want {
			t.Fatalf("isa %d → tile %d want %d", c.isa, AdaptiveTileForISA(c.isa), c.want)
		}
		if ISANameOf(c.isa) != c.name {
			t.Fatalf("isa %d name %s want %s", c.isa, ISANameOf(c.isa), c.name)
		}
	}
}

func TestAdaptiveTileSequence(t *testing.T) {
	if got := AdaptiveTileSequence(64); len(got) != 5 || got[0] != 64 || got[1] != 32 || got[2] != 16 || got[3] != 8 || got[4] != 4 {
		t.Fatalf("hw64 seq: %v", got)
	}
	if got := AdaptiveTileSequence(32); len(got) != 4 || got[0] != 32 || got[1] != 16 || got[2] != 8 || got[3] != 4 {
		t.Fatalf("hw32 seq: %v", got)
	}
	if got := AdaptiveTileSequence(16); len(got) != 3 || got[0] != 16 || got[1] != 8 || got[2] != 4 {
		t.Fatalf("hw16 seq: %v", got)
	}
	if got := AdaptiveTileSequence(8); len(got) != 2 || got[0] != 8 || got[1] != 4 {
		t.Fatalf("hw8 seq: %v", got)
	}
	if got := AdaptiveTileSequence(4); len(got) != 1 || got[0] != 4 {
		t.Fatalf("hw4 seq: %v", got)
	}
	if got := AdaptiveTileSequence(2); len(got) != 0 {
		t.Fatalf("hw2 seq empty: %v", got)
	}
}

func TestAdaptivePeelChunks(t *testing.T) {
	// portable 16: n=20 → 16+4
	tiles, r := AdaptivePeelChunks(16, 20)
	if len(tiles) != 2 || tiles[0] != 16 || tiles[1] != 4 || r != 0 {
		t.Fatalf("hw16 n20: tiles=%v r=%d", tiles, r)
	}
	// AVX 8: n=20 → 8+8+4
	tiles, r = AdaptivePeelChunks(8, 20)
	if len(tiles) != 3 || tiles[0] != 8 || tiles[1] != 8 || tiles[2] != 4 || r != 0 {
		t.Fatalf("hw8 n20: tiles=%v r=%d", tiles, r)
	}
	// SSE 4: n=20 → five 4s
	tiles, r = AdaptivePeelChunks(4, 20)
	if len(tiles) != 5 || r != 0 {
		t.Fatalf("hw4 n20: tiles=%v r=%d", tiles, r)
	}
	// residual
	tiles, r = AdaptivePeelChunks(8, 10)
	if len(tiles) != 1 || tiles[0] != 8 || r != 2 {
		t.Fatalf("hw8 n10: tiles=%v r=%d", tiles, r)
	}
	// SVE-class 32: n=40 → 32+8
	tiles, r = AdaptivePeelChunks(32, 40)
	if len(tiles) != 2 || tiles[0] != 32 || tiles[1] != 8 || r != 0 {
		t.Fatalf("hw32 n40: tiles=%v r=%d", tiles, r)
	}
	// VL>32: n=80 → 64+16
	tiles, r = AdaptivePeelChunks(64, 80)
	if len(tiles) != 2 || tiles[0] != 64 || tiles[1] != 16 || r != 0 {
		t.Fatalf("hw64 n80: tiles=%v r=%d", tiles, r)
	}
	// peelDynTileChunks uses host AdaptivePeelChunks (portable default → 16-class)
	tiles, r = peelDynTileChunks(20)
	hw := DetectHostCPU().HWVLEN
	wantTiles, wantR := AdaptivePeelChunks(hw, 20)
	if hw < 16 {
		wantTiles, wantR = AdaptivePeelChunks(hw, 20)
	}
	if r != wantR || len(tiles) != len(wantTiles) {
		t.Fatalf("peelDyn 20 host hw=%d: tiles=%v r=%d want %v/%d", hw, tiles, r, wantTiles, wantR)
	}
	for i := range tiles {
		if tiles[i] != wantTiles[i] {
			t.Fatalf("peelDyn 20 host hw=%d: tiles=%v want %v", hw, tiles, wantTiles)
		}
	}
}

func TestEmitAdaptiveTileLLVM(t *testing.T) {
	s := EmitAdaptiveTileLLVM()
	for _, want := range []string{
		"adaptive tile from runtime ISA (v0.39)",
		"define i32 @trin_adaptive_tile",
		"define i32 @trin_adaptive_tile_max",
		"define i32 @trin_adaptive_isa_tile",
		"call i32 @trin_hw_vlen_for_n",
		"call i32 @trin_hw_vlen",
		"call i32 @trin_hw_isa_code",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %q:\n%s", want, s)
		}
	}
}

func TestEmitMLIRAdaptiveTileProbe(t *testing.T) {
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
		`trin.adaptive_tile = "runtime_isa"`,
		"func.func @trin_adaptive_tile",
		"func.func @trin_adaptive_tile_max",
		"func.func @trin_adaptive_isa_tile",
		"func.func @trin_mlir_adaptive_tile_probe",
		"Adaptive tile from runtime ISA",
	} {
		if !strings.Contains(mlir, want) {
			t.Fatalf("missing %q in emit-mlir", want)
		}
	}
	ll, err := EmitMLIRLLVM(prog)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"define float @trin_mlir_adaptive_tile_probe",
		"ret float 1.000000e+02",
		"define i32 @trin_adaptive_tile",
		"define i32 @trin_adaptive_isa_tile",
		"runtime feature dispatch (v0.30/v0.39 adaptive)",
		"v0.67",
	} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q in lower", want)
		}
	}
}
