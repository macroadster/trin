package codegen

import (
	"fmt"
	"strings"
)

// emitDynTileLLVMHelpers returns runtime dynamic tile helpers used by
// full-program @main span kernels (v0.32+). Depends on @trin_hw_vlen_for_n
// from EmitRuntimeCPUIDLLVM (callers must inject it first when missing).
//
// Fixed-N kernels pick among vector tiles {4,8,16} at runtime via
// @trin_hw_vlen_for_n(N) then packed <N x float> bodies.
//   trin_dyn_sum4 / trin_dyn_sum8 / trin_dyn_sum16
//   trin_dyn_dot4 / trin_dyn_dot8
//   trin_dyn_tile (alias of hw_vlen_for_n)
// v0.34 residual peel for non-fixed lengths is emit-side in full-program
// @main (see peelDynTileChunks / emitSumArg), not additional LLVM helpers.
func emitDynTileLLVMHelpers() string {
	var b strings.Builder
	b.WriteString("; --- dynamic tile selection for full-program @main (v0.32) + residual peel (v0.34) ---\n")
	b.WriteString("; prefers @trin_hw_vlen_for_n then packed <N x float> helpers\n")
	b.WriteString("; non-fixed n peels into max 16/8/4 tiles + scalar residual (emit-side v0.34)\n\n")

	// Fixed vector kernels used by dispatch.
	b.WriteString("define internal float @trin_dyn_sum_v4_body(<4 x float> %a) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %e0 = extractelement <4 x float> %a, i32 0\n")
	b.WriteString("  %e1 = extractelement <4 x float> %a, i32 1\n")
	b.WriteString("  %e2 = extractelement <4 x float> %a, i32 2\n")
	b.WriteString("  %e3 = extractelement <4 x float> %a, i32 3\n")
	b.WriteString("  %s0 = fadd float %e0, %e1\n")
	b.WriteString("  %s1 = fadd float %s0, %e2\n")
	b.WriteString("  %s2 = fadd float %s1, %e3\n")
	b.WriteString("  ret float %s2\n")
	b.WriteString("}\n\n")

	b.WriteString("define internal float @trin_dyn_sum_v8_body(<8 x float> %a) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %lo = shufflevector <8 x float> %a, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>\n")
	b.WriteString("  %hi = shufflevector <8 x float> %a, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>\n")
	b.WriteString("  %s0 = call float @trin_dyn_sum_v4_body(<4 x float> %lo)\n")
	b.WriteString("  %s1 = call float @trin_dyn_sum_v4_body(<4 x float> %hi)\n")
	b.WriteString("  %y = fadd float %s0, %s1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	b.WriteString("define internal float @trin_dyn_sum_v16_body(<16 x float> %a) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %lo = shufflevector <16 x float> %a, <16 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>\n")
	b.WriteString("  %hi = shufflevector <16 x float> %a, <16 x float> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>\n")
	b.WriteString("  %s0 = call float @trin_dyn_sum_v8_body(<8 x float> %lo)\n")
	b.WriteString("  %s1 = call float @trin_dyn_sum_v8_body(<8 x float> %hi)\n")
	b.WriteString("  %y = fadd float %s0, %s1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// sum4: query tile then pack+reduce or scalar fallback
	b.WriteString("define float @trin_dyn_sum4(float %a0, float %a1, float %a2, float %a3) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t = call i32 @trin_hw_vlen_for_n(i32 4)\n")
	b.WriteString("  %v0 = insertelement <4 x float> undef, float %a0, i32 0\n")
	b.WriteString("  %v1 = insertelement <4 x float> %v0, float %a1, i32 1\n")
	b.WriteString("  %v2 = insertelement <4 x float> %v1, float %a2, i32 2\n")
	b.WriteString("  %v3 = insertelement <4 x float> %v2, float %a3, i32 3\n")
	b.WriteString("  %ok = icmp sge i32 %t, 4\n")
	b.WriteString("  br i1 %ok, label %vec, label %scalar\n")
	b.WriteString("vec:\n")
	b.WriteString("  %ys = call float @trin_dyn_sum_v4_body(<4 x float> %v3)\n")
	b.WriteString("  ret float %ys\n")
	b.WriteString("scalar:\n")
	b.WriteString("  %s0 = fadd float %a0, %a1\n")
	b.WriteString("  %s1 = fadd float %s0, %a2\n")
	b.WriteString("  %s2 = fadd float %s1, %a3\n")
	b.WriteString("  ret float %s2\n")
	b.WriteString("}\n\n")

	// sum8: prefer 8-lane tile; peel into two sum4 when hw_vlen < 8
	b.WriteString("define float @trin_dyn_sum8(float %a0, float %a1, float %a2, float %a3, float %a4, float %a5, float %a6, float %a7) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t = call i32 @trin_hw_vlen_for_n(i32 8)\n")
	b.WriteString("  %ge8 = icmp sge i32 %t, 8\n")
	b.WriteString("  br i1 %ge8, label %v8, label %v4\n")
	b.WriteString("v8:\n")
	b.WriteString("  %u0 = insertelement <8 x float> undef, float %a0, i32 0\n")
	b.WriteString("  %u1 = insertelement <8 x float> %u0, float %a1, i32 1\n")
	b.WriteString("  %u2 = insertelement <8 x float> %u1, float %a2, i32 2\n")
	b.WriteString("  %u3 = insertelement <8 x float> %u2, float %a3, i32 3\n")
	b.WriteString("  %u4 = insertelement <8 x float> %u3, float %a4, i32 4\n")
	b.WriteString("  %u5 = insertelement <8 x float> %u4, float %a5, i32 5\n")
	b.WriteString("  %u6 = insertelement <8 x float> %u5, float %a6, i32 6\n")
	b.WriteString("  %u7 = insertelement <8 x float> %u6, float %a7, i32 7\n")
	b.WriteString("  %y8 = call float @trin_dyn_sum_v8_body(<8 x float> %u7)\n")
	b.WriteString("  ret float %y8\n")
	b.WriteString("v4:\n")
	b.WriteString("  %lo = call float @trin_dyn_sum4(float %a0, float %a1, float %a2, float %a3)\n")
	b.WriteString("  %hi = call float @trin_dyn_sum4(float %a4, float %a5, float %a6, float %a7)\n")
	b.WriteString("  %y4 = fadd float %lo, %hi\n")
	b.WriteString("  ret float %y4\n")
	b.WriteString("}\n\n")

	// sum16: prefer 16-lane; peel into two sum8 otherwise
	b.WriteString("define float @trin_dyn_sum16(\n")
	b.WriteString("    float %a0, float %a1, float %a2, float %a3, float %a4, float %a5, float %a6, float %a7,\n")
	b.WriteString("    float %a8, float %a9, float %a10, float %a11, float %a12, float %a13, float %a14, float %a15) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t = call i32 @trin_hw_vlen_for_n(i32 16)\n")
	b.WriteString("  %ge16 = icmp sge i32 %t, 16\n")
	b.WriteString("  br i1 %ge16, label %v16, label %chk8\n")
	b.WriteString("v16:\n")
	b.WriteString("  %v0 = insertelement <16 x float> undef, float %a0, i32 0\n")
	for i := 1; i < 16; i++ {
		fmt.Fprintf(&b, "  %%v%d = insertelement <16 x float> %%v%d, float %%a%d, i32 %d\n", i, i-1, i, i)
	}
	b.WriteString("  %y16 = call float @trin_dyn_sum_v16_body(<16 x float> %v15)\n")
	b.WriteString("  ret float %y16\n")
	b.WriteString("chk8:\n")
	b.WriteString("  %lo8 = call float @trin_dyn_sum8(float %a0, float %a1, float %a2, float %a3, float %a4, float %a5, float %a6, float %a7)\n")
	b.WriteString("  %hi8 = call float @trin_dyn_sum8(float %a8, float %a9, float %a10, float %a11, float %a12, float %a13, float %a14, float %a15)\n")
	b.WriteString("  %y8 = fadd float %lo8, %hi8\n")
	b.WriteString("  ret float %y8\n")
	b.WriteString("}\n\n")

	// sum32: exact-32 full-program dyn helper (v0.51). Two sum16 peels; consults
	// @trin_hw_vlen_for_n(32) so residual/wide paths observe runtime ISA.
	b.WriteString("define float @trin_dyn_sum32(\n")
	b.WriteString("    float %a0, float %a1, float %a2, float %a3, float %a4, float %a5, float %a6, float %a7,\n")
	b.WriteString("    float %a8, float %a9, float %a10, float %a11, float %a12, float %a13, float %a14, float %a15,\n")
	b.WriteString("    float %a16, float %a17, float %a18, float %a19, float %a20, float %a21, float %a22, float %a23,\n")
	b.WriteString("    float %a24, float %a25, float %a26, float %a27, float %a28, float %a29, float %a30, float %a31) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t = call i32 @trin_hw_vlen_for_n(i32 32)\n")
	b.WriteString("  %lo = call float @trin_dyn_sum16(float %a0, float %a1, float %a2, float %a3, float %a4, float %a5, float %a6, float %a7, float %a8, float %a9, float %a10, float %a11, float %a12, float %a13, float %a14, float %a15)\n")
	b.WriteString("  %hi = call float @trin_dyn_sum16(float %a16, float %a17, float %a18, float %a19, float %a20, float %a21, float %a22, float %a23, float %a24, float %a25, float %a26, float %a27, float %a28, float %a29, float %a30, float %a31)\n")
	b.WriteString("  %y = fadd float %lo, %hi\n")
	b.WriteString("  %keep = add i32 %t, 0\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// Dot = sum of products after pairwise muls; reuses sum dispatch.
	b.WriteString("define float @trin_dyn_dot4(float %a0, float %a1, float %a2, float %a3, float %b0, float %b1, float %b2, float %b3) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %p0 = fmul float %a0, %b0\n")
	b.WriteString("  %p1 = fmul float %a1, %b1\n")
	b.WriteString("  %p2 = fmul float %a2, %b2\n")
	b.WriteString("  %p3 = fmul float %a3, %b3\n")
	b.WriteString("  %y = call float @trin_dyn_sum4(float %p0, float %p1, float %p2, float %p3)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	b.WriteString("define float @trin_dyn_dot8(\n")
	b.WriteString("    float %a0, float %a1, float %a2, float %a3, float %a4, float %a5, float %a6, float %a7,\n")
	b.WriteString("    float %b0, float %b1, float %b2, float %b3, float %b4, float %b5, float %b6, float %b7) {\n")
	b.WriteString("entry:\n")
	for i := 0; i < 8; i++ {
		fmt.Fprintf(&b, "  %%p%d = fmul float %%a%d, %%b%d\n", i, i, i)
	}
	b.WriteString("  %y = call float @trin_dyn_sum8(float %p0, float %p1, float %p2, float %p3, float %p4, float %p5, float %p6, float %p7)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// Public tile query (language surface dyn_tile / hw_vlen_for_n).
	b.WriteString("define i32 @trin_dyn_tile(i32 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t = call i32 @trin_hw_vlen_for_n(i32 %n)\n")
	b.WriteString("  ret i32 %t\n")
	b.WriteString("}\n\n")

	return b.String()
}

// preferDynTileN reports whether a fixed span length should use a single
// exact dyn tile helper (no residual). Classic exact dyn tiles are 4/8/16;
// v0.51 adds exact-32 (@trin_dyn_sum32).
func preferDynTileN(n int) bool {
	return n == 4 || n == 8 || n == 16 || n == 32
}

// peelDynTileChunks decomposes n into descending adaptive tiles via host ISA
// HWVLEN (v0.51 AdaptivePeelChunks at emit time). Falls back to portable
// schedule when host tile is unknown / low.
// Example: portable/n=20 → [16,4] residual 0; SVE(hw=32)/n=40 → [32,8] residual 0;
// VL>32 host(hw=64)/n=80 → [64,16] residual 0.
// Runtime pointer peels still gate each tile with @trin_adaptive_peel_tile /
// @trin_hw_vlen_for_n so emit-time host notes never oversubscribe lanes.
func peelDynTileChunks(n int) (tiles []int, residual int) {
	hw := DetectHostCPU().HWVLEN
	if hw < 4 {
		hw = MaxPeelTile
	}
	// Portable default surface still prefers at least 16 so existing full-
	// program residual comments (tiles=[16,4]) stay stable when host is portable.
	if hw < 16 {
		// Keep host-gated sequence (e.g. AVX→8) when known narrower.
		return AdaptivePeelChunks(hw, n)
	}
	// Prefer host HWVLEN, but allow wide ladder (32) only when host reports it.
	return AdaptivePeelChunks(hw, n)
}

// AdaptivePeelChunks is the v0.39/v0.51 ISA-aware peel: only tiles <= hw are used.
// Example: hw=8, n=20 → [8,8,4], residual 0; hw=32, n=40 → [32,8], residual 0.
func AdaptivePeelChunks(hw, n int) (tiles []int, residual int) {
	if n < 0 {
		return nil, 0
	}
	r := n
	for _, w := range AdaptiveTileSequence(hw) {
		for r >= w {
			tiles = append(tiles, w)
			r -= w
		}
	}
	// If hw < 4, all residual.
	return tiles, r
}

// preferResidualPeel reports whether length n should use tile+residual peel
// rather than a pure scalar chain (v0.34). True when n >= 4 and n is not an
// exact single dyn tile (those use preferDynTileN path).
func preferResidualPeel(n int) bool {
	return n >= 4 && !preferDynTileN(n)
}

// EmitAdaptivePeelChunksNote returns a short IR comment documenting the
// AdaptivePeelChunks decomposition used by full-program residual peels (v0.51).
func EmitAdaptivePeelChunksNote(hw, n int) string {
	tiles, r := AdaptivePeelChunks(hw, n)
	return fmt.Sprintf("; rewrite: AdaptivePeelChunks(hw=%d,n=%d) tiles=%v residual=%d (v0.51)", hw, n, tiles, r)
}
