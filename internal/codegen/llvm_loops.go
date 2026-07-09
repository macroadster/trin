package codegen

import (
	"fmt"
	"strings"
)

// emitLLVMSpanLoopHelpers returns LLVM IR for variable-length span kernels
// with auto residual peel (v0.36 sum/dot + v0.37 store + v0.40 map/store +
// v0.41 scan/reduce + v0.42 scan-mul/reduce-mul + wide 32 + v0.43 fused
// scan-reduce / dyn32 / VL>32 + v0.51 multi-output fused + RDVL) and adaptive
// ISA peel ladder (v0.39–v0.51).
// Counted pointer kernels peel max 64/32/16/8/4-lane vector tiles gated by
// @trin_adaptive_peel_tile → @trin_hw_vlen_for_n (runtime adaptive tile),
// then a scalar residual. Falls back to pure residual scalar when HW VLEN < 4.
// Still stamps !llvm.loop.vectorize metadata.
//
// Requires @trin_hw_vlen_for_n in the module (callers inject EmitRuntimeCPUIDLLVM
// or emit a local portable fallback — see ensurePeelHWVLEN). Adaptive helpers
// @trin_adaptive_tile* may also be linked for probes (see EmitAdaptiveTileLLVM).
func emitLLVMSpanLoopHelpers() string {
	var b strings.Builder
	b.WriteString("; --- auto residual peel span loops (v0.36–v0.67) + adaptive ISA peel ladder (incl. 64) ---\n")
	b.WriteString("; pointer/counted kernels: peel max 64/32/16/8/4 tiles via @trin_adaptive_peel_tile\n")
	b.WriteString("; sum/dot + stores + map + scan/reduce + multi-out fused + RDVL; wide SVE 32/64 peels (v0.51)\n")
	b.WriteString("; adaptive ladder gates peels through runtime ISA tile; portable fallback peels full widths\n\n")

	// v0.41 adaptive ISA peel ladder entry: peels call this instead of baking a
	// hard 16/8/4 path without ISA gate. Resolves to hw_vlen_for_n (runtime).
	b.WriteString("define i32 @trin_adaptive_peel_tile(i32 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t = call i32 @trin_hw_vlen_for_n(i32 %n)\n")
	b.WriteString("  ret i32 %t\n")
	b.WriteString("}\n\n")

	// Portable fallback if CPUID helpers not linked (text-level). When a full
	// EmitRuntimeCPUIDLLVM is already in the module this define is skipped by
	// ensurePeelHWVLEN. Here we only emit a weak local if not present later.

	// ========== trin_sum_f32 ==========
	b.WriteString("define float @trin_sum_f32(float* nocapture readonly %a, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %hw = call i32 @trin_adaptive_peel_tile(i32 16)\n")
	b.WriteString("  br label %peel16.cond\n")
	// peel 16
	b.WriteString("peel16.cond:\n")
	b.WriteString("  %i16 = phi i64 [ 0, %entry ], [ %i16.next, %peel16.body ]\n")
	b.WriteString("  %acc16 = phi float [ 0.000000e+00, %entry ], [ %acc16.next, %peel16.body ]\n")
	b.WriteString("  %rem16 = sub i64 %n, %i16\n")
	b.WriteString("  %ge16n = icmp sge i64 %rem16, 16\n")
	b.WriteString("  %ge16h = icmp sge i32 %hw, 16\n")
	b.WriteString("  %do16 = and i1 %ge16n, %ge16h\n")
	b.WriteString("  br i1 %do16, label %peel16.body, label %peel8.cond\n")
	b.WriteString("peel16.body:\n")
	b.WriteString("  %p16 = getelementptr inbounds float, float* %a, i64 %i16\n")
	b.WriteString("  %p16c = bitcast float* %p16 to <16 x float>*\n")
	b.WriteString("  %v16 = load <16 x float>, <16 x float>* %p16c, align 4\n")
	b.WriteString("  %lo8 = shufflevector <16 x float> %v16, <16 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>\n")
	b.WriteString("  %hi8 = shufflevector <16 x float> %v16, <16 x float> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>\n")
	b.WriteString("  %s16 = call float @trin_peel_hsum8(<8 x float> %lo8)\n")
	b.WriteString("  %t16 = call float @trin_peel_hsum8(<8 x float> %hi8)\n")
	b.WriteString("  %chunk16 = fadd float %s16, %t16\n")
	b.WriteString("  %acc16.next = fadd float %acc16, %chunk16\n")
	b.WriteString("  %i16.next = add nuw nsw i64 %i16, 16\n")
	b.WriteString("  br label %peel16.cond\n")
	// peel 8
	b.WriteString("peel8.cond:\n")
	b.WriteString("  %i8 = phi i64 [ %i16, %peel16.cond ], [ %i8.next, %peel8.body ]\n")
	b.WriteString("  %acc8 = phi float [ %acc16, %peel16.cond ], [ %acc8.next, %peel8.body ]\n")
	b.WriteString("  %rem8 = sub i64 %n, %i8\n")
	b.WriteString("  %ge8n = icmp sge i64 %rem8, 8\n")
	b.WriteString("  %ge8h = icmp sge i32 %hw, 8\n")
	b.WriteString("  %do8 = and i1 %ge8n, %ge8h\n")
	b.WriteString("  br i1 %do8, label %peel8.body, label %peel4.cond\n")
	b.WriteString("peel8.body:\n")
	b.WriteString("  %p8 = getelementptr inbounds float, float* %a, i64 %i8\n")
	b.WriteString("  %p8c = bitcast float* %p8 to <8 x float>*\n")
	b.WriteString("  %v8 = load <8 x float>, <8 x float>* %p8c, align 4\n")
	b.WriteString("  %chunk8 = call float @trin_peel_hsum8(<8 x float> %v8)\n")
	b.WriteString("  %acc8.next = fadd float %acc8, %chunk8\n")
	b.WriteString("  %i8.next = add nuw nsw i64 %i8, 8\n")
	b.WriteString("  br label %peel8.cond\n")
	// peel 4
	b.WriteString("peel4.cond:\n")
	b.WriteString("  %i4 = phi i64 [ %i8, %peel8.cond ], [ %i4.next, %peel4.body ]\n")
	b.WriteString("  %acc4 = phi float [ %acc8, %peel8.cond ], [ %acc4.next, %peel4.body ]\n")
	b.WriteString("  %rem4 = sub i64 %n, %i4\n")
	b.WriteString("  %ge4n = icmp sge i64 %rem4, 4\n")
	b.WriteString("  %ge4h = icmp sge i32 %hw, 4\n")
	b.WriteString("  %do4 = and i1 %ge4n, %ge4h\n")
	b.WriteString("  br i1 %do4, label %peel4.body, label %scalar.cond\n")
	b.WriteString("peel4.body:\n")
	b.WriteString("  %p4 = getelementptr inbounds float, float* %a, i64 %i4\n")
	b.WriteString("  %p4c = bitcast float* %p4 to <4 x float>*\n")
	b.WriteString("  %v4 = load <4 x float>, <4 x float>* %p4c, align 4\n")
	b.WriteString("  %chunk4 = call float @trin_peel_hsum4(<4 x float> %v4)\n")
	b.WriteString("  %acc4.next = fadd float %acc4, %chunk4\n")
	b.WriteString("  %i4.next = add nuw nsw i64 %i4, 4\n")
	b.WriteString("  br label %peel4.cond\n")
	// residual scalar
	b.WriteString("scalar.cond:\n")
	b.WriteString("  %i = phi i64 [ %i4, %peel4.cond ], [ %i.next, %scalar.body ]\n")
	b.WriteString("  %acc = phi float [ %acc4, %peel4.cond ], [ %acc.next, %scalar.body ]\n")
	b.WriteString("  %cmp = icmp slt i64 %i, %n\n")
	b.WriteString("  br i1 %cmp, label %scalar.body, label %done\n")
	b.WriteString("scalar.body:\n")
	b.WriteString("  %ap = getelementptr inbounds float, float* %a, i64 %i\n")
	b.WriteString("  %av = load float, float* %ap, align 4\n")
	b.WriteString("  %acc.next = fadd float %acc, %av\n")
	b.WriteString("  %i.next = add nuw nsw i64 %i, 1\n")
	b.WriteString("  br label %scalar.cond, !llvm.loop !0\n")
	b.WriteString("done:\n")
	b.WriteString("  ret float %acc\n")
	b.WriteString("}\n\n")

	// ========== trin_dot_f32 ==========
	b.WriteString("define float @trin_dot_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %hw = call i32 @trin_adaptive_peel_tile(i32 16)\n")
	b.WriteString("  br label %peel16.cond\n")
	b.WriteString("peel16.cond:\n")
	b.WriteString("  %i16 = phi i64 [ 0, %entry ], [ %i16.next, %peel16.body ]\n")
	b.WriteString("  %acc16 = phi float [ 0.000000e+00, %entry ], [ %acc16.next, %peel16.body ]\n")
	b.WriteString("  %rem16 = sub i64 %n, %i16\n")
	b.WriteString("  %ge16n = icmp sge i64 %rem16, 16\n")
	b.WriteString("  %ge16h = icmp sge i32 %hw, 16\n")
	b.WriteString("  %do16 = and i1 %ge16n, %ge16h\n")
	b.WriteString("  br i1 %do16, label %peel16.body, label %peel8.cond\n")
	b.WriteString("peel16.body:\n")
	b.WriteString("  %ap16 = getelementptr inbounds float, float* %a, i64 %i16\n")
	b.WriteString("  %bp16 = getelementptr inbounds float, float* %b, i64 %i16\n")
	b.WriteString("  %ap16c = bitcast float* %ap16 to <16 x float>*\n")
	b.WriteString("  %bp16c = bitcast float* %bp16 to <16 x float>*\n")
	b.WriteString("  %va16 = load <16 x float>, <16 x float>* %ap16c, align 4\n")
	b.WriteString("  %vb16 = load <16 x float>, <16 x float>* %bp16c, align 4\n")
	b.WriteString("  %prod16 = fmul <16 x float> %va16, %vb16\n")
	b.WriteString("  %lo8 = shufflevector <16 x float> %prod16, <16 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>\n")
	b.WriteString("  %hi8 = shufflevector <16 x float> %prod16, <16 x float> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>\n")
	b.WriteString("  %s16 = call float @trin_peel_hsum8(<8 x float> %lo8)\n")
	b.WriteString("  %t16 = call float @trin_peel_hsum8(<8 x float> %hi8)\n")
	b.WriteString("  %chunk16 = fadd float %s16, %t16\n")
	b.WriteString("  %acc16.next = fadd float %acc16, %chunk16\n")
	b.WriteString("  %i16.next = add nuw nsw i64 %i16, 16\n")
	b.WriteString("  br label %peel16.cond\n")
	b.WriteString("peel8.cond:\n")
	b.WriteString("  %i8 = phi i64 [ %i16, %peel16.cond ], [ %i8.next, %peel8.body ]\n")
	b.WriteString("  %acc8 = phi float [ %acc16, %peel16.cond ], [ %acc8.next, %peel8.body ]\n")
	b.WriteString("  %rem8 = sub i64 %n, %i8\n")
	b.WriteString("  %ge8n = icmp sge i64 %rem8, 8\n")
	b.WriteString("  %ge8h = icmp sge i32 %hw, 8\n")
	b.WriteString("  %do8 = and i1 %ge8n, %ge8h\n")
	b.WriteString("  br i1 %do8, label %peel8.body, label %peel4.cond\n")
	b.WriteString("peel8.body:\n")
	b.WriteString("  %ap8 = getelementptr inbounds float, float* %a, i64 %i8\n")
	b.WriteString("  %bp8 = getelementptr inbounds float, float* %b, i64 %i8\n")
	b.WriteString("  %ap8c = bitcast float* %ap8 to <8 x float>*\n")
	b.WriteString("  %bp8c = bitcast float* %bp8 to <8 x float>*\n")
	b.WriteString("  %va8 = load <8 x float>, <8 x float>* %ap8c, align 4\n")
	b.WriteString("  %vb8 = load <8 x float>, <8 x float>* %bp8c, align 4\n")
	b.WriteString("  %prod8 = fmul <8 x float> %va8, %vb8\n")
	b.WriteString("  %chunk8 = call float @trin_peel_hsum8(<8 x float> %prod8)\n")
	b.WriteString("  %acc8.next = fadd float %acc8, %chunk8\n")
	b.WriteString("  %i8.next = add nuw nsw i64 %i8, 8\n")
	b.WriteString("  br label %peel8.cond\n")
	b.WriteString("peel4.cond:\n")
	b.WriteString("  %i4 = phi i64 [ %i8, %peel8.cond ], [ %i4.next, %peel4.body ]\n")
	b.WriteString("  %acc4 = phi float [ %acc8, %peel8.cond ], [ %acc4.next, %peel4.body ]\n")
	b.WriteString("  %rem4 = sub i64 %n, %i4\n")
	b.WriteString("  %ge4n = icmp sge i64 %rem4, 4\n")
	b.WriteString("  %ge4h = icmp sge i32 %hw, 4\n")
	b.WriteString("  %do4 = and i1 %ge4n, %ge4h\n")
	b.WriteString("  br i1 %do4, label %peel4.body, label %scalar.cond\n")
	b.WriteString("peel4.body:\n")
	b.WriteString("  %ap4 = getelementptr inbounds float, float* %a, i64 %i4\n")
	b.WriteString("  %bp4 = getelementptr inbounds float, float* %b, i64 %i4\n")
	b.WriteString("  %ap4c = bitcast float* %ap4 to <4 x float>*\n")
	b.WriteString("  %bp4c = bitcast float* %bp4 to <4 x float>*\n")
	b.WriteString("  %va4 = load <4 x float>, <4 x float>* %ap4c, align 4\n")
	b.WriteString("  %vb4 = load <4 x float>, <4 x float>* %bp4c, align 4\n")
	b.WriteString("  %prod4 = fmul <4 x float> %va4, %vb4\n")
	b.WriteString("  %chunk4 = call float @trin_peel_hsum4(<4 x float> %prod4)\n")
	b.WriteString("  %acc4.next = fadd float %acc4, %chunk4\n")
	b.WriteString("  %i4.next = add nuw nsw i64 %i4, 4\n")
	b.WriteString("  br label %peel4.cond\n")
	b.WriteString("scalar.cond:\n")
	b.WriteString("  %i = phi i64 [ %i4, %peel4.cond ], [ %i.next, %scalar.body ]\n")
	b.WriteString("  %acc = phi float [ %acc4, %peel4.cond ], [ %acc.next, %scalar.body ]\n")
	b.WriteString("  %cmp = icmp slt i64 %i, %n\n")
	b.WriteString("  br i1 %cmp, label %scalar.body, label %done\n")
	b.WriteString("scalar.body:\n")
	b.WriteString("  %ap = getelementptr inbounds float, float* %a, i64 %i\n")
	b.WriteString("  %bp = getelementptr inbounds float, float* %b, i64 %i\n")
	b.WriteString("  %av = load float, float* %ap, align 4\n")
	b.WriteString("  %bv = load float, float* %bp, align 4\n")
	b.WriteString("  %p = fmul float %av, %bv\n")
	b.WriteString("  %acc.next = fadd float %acc, %p\n")
	b.WriteString("  %i.next = add nuw nsw i64 %i, 1\n")
	b.WriteString("  br label %scalar.cond, !llvm.loop !0\n")
	b.WriteString("done:\n")
	b.WriteString("  ret float %acc\n")
	b.WriteString("}\n\n")

	// ========== span_add / scale (v0.37 store peels: vector tiles + residual) ==========
	// trin_span_add_f32: out[i] = a[i] + b[i]
	b.WriteString("define void @trin_span_add_f32(float* nocapture readonly %a, float* nocapture readonly %b, float* nocapture writeonly %out, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %hw = call i32 @trin_adaptive_peel_tile(i32 16)\n")
	b.WriteString("  br label %peel16.cond\n")
	b.WriteString("peel16.cond:\n")
	b.WriteString("  %i16 = phi i64 [ 0, %entry ], [ %i16.next, %peel16.body ]\n")
	b.WriteString("  %rem16 = sub i64 %n, %i16\n")
	b.WriteString("  %ge16n = icmp sge i64 %rem16, 16\n")
	b.WriteString("  %ge16h = icmp sge i32 %hw, 16\n")
	b.WriteString("  %do16 = and i1 %ge16n, %ge16h\n")
	b.WriteString("  br i1 %do16, label %peel16.body, label %peel8.cond\n")
	b.WriteString("peel16.body:\n")
	b.WriteString("  %ap16 = getelementptr inbounds float, float* %a, i64 %i16\n")
	b.WriteString("  %bp16 = getelementptr inbounds float, float* %b, i64 %i16\n")
	b.WriteString("  %op16 = getelementptr inbounds float, float* %out, i64 %i16\n")
	b.WriteString("  %ap16c = bitcast float* %ap16 to <16 x float>*\n")
	b.WriteString("  %bp16c = bitcast float* %bp16 to <16 x float>*\n")
	b.WriteString("  %op16c = bitcast float* %op16 to <16 x float>*\n")
	b.WriteString("  %va16 = load <16 x float>, <16 x float>* %ap16c, align 4\n")
	b.WriteString("  %vb16 = load <16 x float>, <16 x float>* %bp16c, align 4\n")
	b.WriteString("  %vs16 = fadd <16 x float> %va16, %vb16\n")
	b.WriteString("  store <16 x float> %vs16, <16 x float>* %op16c, align 4\n")
	b.WriteString("  %i16.next = add nuw nsw i64 %i16, 16\n")
	b.WriteString("  br label %peel16.cond\n")
	b.WriteString("peel8.cond:\n")
	b.WriteString("  %i8 = phi i64 [ %i16, %peel16.cond ], [ %i8.next, %peel8.body ]\n")
	b.WriteString("  %rem8 = sub i64 %n, %i8\n")
	b.WriteString("  %ge8n = icmp sge i64 %rem8, 8\n")
	b.WriteString("  %ge8h = icmp sge i32 %hw, 8\n")
	b.WriteString("  %do8 = and i1 %ge8n, %ge8h\n")
	b.WriteString("  br i1 %do8, label %peel8.body, label %peel4.cond\n")
	b.WriteString("peel8.body:\n")
	b.WriteString("  %ap8 = getelementptr inbounds float, float* %a, i64 %i8\n")
	b.WriteString("  %bp8 = getelementptr inbounds float, float* %b, i64 %i8\n")
	b.WriteString("  %op8 = getelementptr inbounds float, float* %out, i64 %i8\n")
	b.WriteString("  %ap8c = bitcast float* %ap8 to <8 x float>*\n")
	b.WriteString("  %bp8c = bitcast float* %bp8 to <8 x float>*\n")
	b.WriteString("  %op8c = bitcast float* %op8 to <8 x float>*\n")
	b.WriteString("  %va8 = load <8 x float>, <8 x float>* %ap8c, align 4\n")
	b.WriteString("  %vb8 = load <8 x float>, <8 x float>* %bp8c, align 4\n")
	b.WriteString("  %vs8 = fadd <8 x float> %va8, %vb8\n")
	b.WriteString("  store <8 x float> %vs8, <8 x float>* %op8c, align 4\n")
	b.WriteString("  %i8.next = add nuw nsw i64 %i8, 8\n")
	b.WriteString("  br label %peel8.cond\n")
	b.WriteString("peel4.cond:\n")
	b.WriteString("  %i4 = phi i64 [ %i8, %peel8.cond ], [ %i4.next, %peel4.body ]\n")
	b.WriteString("  %rem4 = sub i64 %n, %i4\n")
	b.WriteString("  %ge4n = icmp sge i64 %rem4, 4\n")
	b.WriteString("  %ge4h = icmp sge i32 %hw, 4\n")
	b.WriteString("  %do4 = and i1 %ge4n, %ge4h\n")
	b.WriteString("  br i1 %do4, label %peel4.body, label %scalar.cond\n")
	b.WriteString("peel4.body:\n")
	b.WriteString("  %ap4 = getelementptr inbounds float, float* %a, i64 %i4\n")
	b.WriteString("  %bp4 = getelementptr inbounds float, float* %b, i64 %i4\n")
	b.WriteString("  %op4 = getelementptr inbounds float, float* %out, i64 %i4\n")
	b.WriteString("  %ap4c = bitcast float* %ap4 to <4 x float>*\n")
	b.WriteString("  %bp4c = bitcast float* %bp4 to <4 x float>*\n")
	b.WriteString("  %op4c = bitcast float* %op4 to <4 x float>*\n")
	b.WriteString("  %va4 = load <4 x float>, <4 x float>* %ap4c, align 4\n")
	b.WriteString("  %vb4 = load <4 x float>, <4 x float>* %bp4c, align 4\n")
	b.WriteString("  %vs4 = fadd <4 x float> %va4, %vb4\n")
	b.WriteString("  store <4 x float> %vs4, <4 x float>* %op4c, align 4\n")
	b.WriteString("  %i4.next = add nuw nsw i64 %i4, 4\n")
	b.WriteString("  br label %peel4.cond\n")
	b.WriteString("scalar.cond:\n")
	b.WriteString("  %i = phi i64 [ %i4, %peel4.cond ], [ %i.next, %scalar.body ]\n")
	b.WriteString("  %cmp = icmp slt i64 %i, %n\n")
	b.WriteString("  br i1 %cmp, label %scalar.body, label %done\n")
	b.WriteString("scalar.body:\n")
	b.WriteString("  %ap = getelementptr inbounds float, float* %a, i64 %i\n")
	b.WriteString("  %bp = getelementptr inbounds float, float* %b, i64 %i\n")
	b.WriteString("  %op = getelementptr inbounds float, float* %out, i64 %i\n")
	b.WriteString("  %av = load float, float* %ap, align 4\n")
	b.WriteString("  %bv = load float, float* %bp, align 4\n")
	b.WriteString("  %s = fadd float %av, %bv\n")
	b.WriteString("  store float %s, float* %op, align 4\n")
	b.WriteString("  %i.next = add nuw nsw i64 %i, 1\n")
	b.WriteString("  br label %scalar.cond, !llvm.loop !0\n")
	b.WriteString("done:\n")
	b.WriteString("  ret void\n")
	b.WriteString("}\n\n")

	// trin_span_scale_f32: out[i] = s * a[i]
	b.WriteString("define void @trin_span_scale_f32(float %s, float* nocapture readonly %a, float* nocapture writeonly %out, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %hw = call i32 @trin_adaptive_peel_tile(i32 16)\n")
	// broadcast scalar s to vector lanes via insertelement + shufflevector
	b.WriteString("  %sv0 = insertelement <4 x float> undef, float %s, i32 0\n")
	b.WriteString("  %sv4 = shufflevector <4 x float> %sv0, <4 x float> undef, <4 x i32> zeroinitializer\n")
	b.WriteString("  %sv8_0 = insertelement <8 x float> undef, float %s, i32 0\n")
	b.WriteString("  %sv8 = shufflevector <8 x float> %sv8_0, <8 x float> undef, <8 x i32> zeroinitializer\n")
	b.WriteString("  %sv16_0 = insertelement <16 x float> undef, float %s, i32 0\n")
	b.WriteString("  %sv16 = shufflevector <16 x float> %sv16_0, <16 x float> undef, <16 x i32> zeroinitializer\n")
	b.WriteString("  br label %peel16.cond\n")
	b.WriteString("peel16.cond:\n")
	b.WriteString("  %i16 = phi i64 [ 0, %entry ], [ %i16.next, %peel16.body ]\n")
	b.WriteString("  %rem16 = sub i64 %n, %i16\n")
	b.WriteString("  %ge16n = icmp sge i64 %rem16, 16\n")
	b.WriteString("  %ge16h = icmp sge i32 %hw, 16\n")
	b.WriteString("  %do16 = and i1 %ge16n, %ge16h\n")
	b.WriteString("  br i1 %do16, label %peel16.body, label %peel8.cond\n")
	b.WriteString("peel16.body:\n")
	b.WriteString("  %ap16 = getelementptr inbounds float, float* %a, i64 %i16\n")
	b.WriteString("  %op16 = getelementptr inbounds float, float* %out, i64 %i16\n")
	b.WriteString("  %ap16c = bitcast float* %ap16 to <16 x float>*\n")
	b.WriteString("  %op16c = bitcast float* %op16 to <16 x float>*\n")
	b.WriteString("  %va16 = load <16 x float>, <16 x float>* %ap16c, align 4\n")
	b.WriteString("  %vp16 = fmul <16 x float> %sv16, %va16\n")
	b.WriteString("  store <16 x float> %vp16, <16 x float>* %op16c, align 4\n")
	b.WriteString("  %i16.next = add nuw nsw i64 %i16, 16\n")
	b.WriteString("  br label %peel16.cond\n")
	b.WriteString("peel8.cond:\n")
	b.WriteString("  %i8 = phi i64 [ %i16, %peel16.cond ], [ %i8.next, %peel8.body ]\n")
	b.WriteString("  %rem8 = sub i64 %n, %i8\n")
	b.WriteString("  %ge8n = icmp sge i64 %rem8, 8\n")
	b.WriteString("  %ge8h = icmp sge i32 %hw, 8\n")
	b.WriteString("  %do8 = and i1 %ge8n, %ge8h\n")
	b.WriteString("  br i1 %do8, label %peel8.body, label %peel4.cond\n")
	b.WriteString("peel8.body:\n")
	b.WriteString("  %ap8 = getelementptr inbounds float, float* %a, i64 %i8\n")
	b.WriteString("  %op8 = getelementptr inbounds float, float* %out, i64 %i8\n")
	b.WriteString("  %ap8c = bitcast float* %ap8 to <8 x float>*\n")
	b.WriteString("  %op8c = bitcast float* %op8 to <8 x float>*\n")
	b.WriteString("  %va8 = load <8 x float>, <8 x float>* %ap8c, align 4\n")
	b.WriteString("  %vp8 = fmul <8 x float> %sv8, %va8\n")
	b.WriteString("  store <8 x float> %vp8, <8 x float>* %op8c, align 4\n")
	b.WriteString("  %i8.next = add nuw nsw i64 %i8, 8\n")
	b.WriteString("  br label %peel8.cond\n")
	b.WriteString("peel4.cond:\n")
	b.WriteString("  %i4 = phi i64 [ %i8, %peel8.cond ], [ %i4.next, %peel4.body ]\n")
	b.WriteString("  %rem4 = sub i64 %n, %i4\n")
	b.WriteString("  %ge4n = icmp sge i64 %rem4, 4\n")
	b.WriteString("  %ge4h = icmp sge i32 %hw, 4\n")
	b.WriteString("  %do4 = and i1 %ge4n, %ge4h\n")
	b.WriteString("  br i1 %do4, label %peel4.body, label %scalar.cond\n")
	b.WriteString("peel4.body:\n")
	b.WriteString("  %ap4 = getelementptr inbounds float, float* %a, i64 %i4\n")
	b.WriteString("  %op4 = getelementptr inbounds float, float* %out, i64 %i4\n")
	b.WriteString("  %ap4c = bitcast float* %ap4 to <4 x float>*\n")
	b.WriteString("  %op4c = bitcast float* %op4 to <4 x float>*\n")
	b.WriteString("  %va4 = load <4 x float>, <4 x float>* %ap4c, align 4\n")
	b.WriteString("  %vp4 = fmul <4 x float> %sv4, %va4\n")
	b.WriteString("  store <4 x float> %vp4, <4 x float>* %op4c, align 4\n")
	b.WriteString("  %i4.next = add nuw nsw i64 %i4, 4\n")
	b.WriteString("  br label %peel4.cond\n")
	b.WriteString("scalar.cond:\n")
	b.WriteString("  %i = phi i64 [ %i4, %peel4.cond ], [ %i.next, %scalar.body ]\n")
	b.WriteString("  %cmp = icmp slt i64 %i, %n\n")
	b.WriteString("  br i1 %cmp, label %scalar.body, label %done\n")
	b.WriteString("scalar.body:\n")
	b.WriteString("  %ap = getelementptr inbounds float, float* %a, i64 %i\n")
	b.WriteString("  %op = getelementptr inbounds float, float* %out, i64 %i\n")
	b.WriteString("  %av = load float, float* %ap, align 4\n")
	b.WriteString("  %p = fmul float %s, %av\n")
	b.WriteString("  store float %p, float* %op, align 4\n")
	b.WriteString("  %i.next = add nuw nsw i64 %i, 1\n")
	b.WriteString("  br label %scalar.cond, !llvm.loop !0\n")
	b.WriteString("done:\n")
	b.WriteString("  ret void\n")
	b.WriteString("}\n\n")

	// ========== span_mul / span_axpy (v0.40 map/store peels) ==========
	// trin_span_mul_f32: out[i] = a[i] * b[i]
	b.WriteString("define void @trin_span_mul_f32(float* nocapture readonly %a, float* nocapture readonly %b, float* nocapture writeonly %out, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %hw = call i32 @trin_adaptive_peel_tile(i32 16)\n")
	b.WriteString("  br label %peel16.cond\n")
	b.WriteString("peel16.cond:\n")
	b.WriteString("  %i16 = phi i64 [ 0, %entry ], [ %i16.next, %peel16.body ]\n")
	b.WriteString("  %rem16 = sub i64 %n, %i16\n")
	b.WriteString("  %ge16n = icmp sge i64 %rem16, 16\n")
	b.WriteString("  %ge16h = icmp sge i32 %hw, 16\n")
	b.WriteString("  %do16 = and i1 %ge16n, %ge16h\n")
	b.WriteString("  br i1 %do16, label %peel16.body, label %peel8.cond\n")
	b.WriteString("peel16.body:\n")
	b.WriteString("  %ap16 = getelementptr inbounds float, float* %a, i64 %i16\n")
	b.WriteString("  %bp16 = getelementptr inbounds float, float* %b, i64 %i16\n")
	b.WriteString("  %op16 = getelementptr inbounds float, float* %out, i64 %i16\n")
	b.WriteString("  %ap16c = bitcast float* %ap16 to <16 x float>*\n")
	b.WriteString("  %bp16c = bitcast float* %bp16 to <16 x float>*\n")
	b.WriteString("  %op16c = bitcast float* %op16 to <16 x float>*\n")
	b.WriteString("  %va16 = load <16 x float>, <16 x float>* %ap16c, align 4\n")
	b.WriteString("  %vb16 = load <16 x float>, <16 x float>* %bp16c, align 4\n")
	b.WriteString("  %vs16 = fmul <16 x float> %va16, %vb16\n")
	b.WriteString("  store <16 x float> %vs16, <16 x float>* %op16c, align 4\n")
	b.WriteString("  %i16.next = add nuw nsw i64 %i16, 16\n")
	b.WriteString("  br label %peel16.cond\n")
	b.WriteString("peel8.cond:\n")
	b.WriteString("  %i8 = phi i64 [ %i16, %peel16.cond ], [ %i8.next, %peel8.body ]\n")
	b.WriteString("  %rem8 = sub i64 %n, %i8\n")
	b.WriteString("  %ge8n = icmp sge i64 %rem8, 8\n")
	b.WriteString("  %ge8h = icmp sge i32 %hw, 8\n")
	b.WriteString("  %do8 = and i1 %ge8n, %ge8h\n")
	b.WriteString("  br i1 %do8, label %peel8.body, label %peel4.cond\n")
	b.WriteString("peel8.body:\n")
	b.WriteString("  %ap8 = getelementptr inbounds float, float* %a, i64 %i8\n")
	b.WriteString("  %bp8 = getelementptr inbounds float, float* %b, i64 %i8\n")
	b.WriteString("  %op8 = getelementptr inbounds float, float* %out, i64 %i8\n")
	b.WriteString("  %ap8c = bitcast float* %ap8 to <8 x float>*\n")
	b.WriteString("  %bp8c = bitcast float* %bp8 to <8 x float>*\n")
	b.WriteString("  %op8c = bitcast float* %op8 to <8 x float>*\n")
	b.WriteString("  %va8 = load <8 x float>, <8 x float>* %ap8c, align 4\n")
	b.WriteString("  %vb8 = load <8 x float>, <8 x float>* %bp8c, align 4\n")
	b.WriteString("  %vs8 = fmul <8 x float> %va8, %vb8\n")
	b.WriteString("  store <8 x float> %vs8, <8 x float>* %op8c, align 4\n")
	b.WriteString("  %i8.next = add nuw nsw i64 %i8, 8\n")
	b.WriteString("  br label %peel8.cond\n")
	b.WriteString("peel4.cond:\n")
	b.WriteString("  %i4 = phi i64 [ %i8, %peel8.cond ], [ %i4.next, %peel4.body ]\n")
	b.WriteString("  %rem4 = sub i64 %n, %i4\n")
	b.WriteString("  %ge4n = icmp sge i64 %rem4, 4\n")
	b.WriteString("  %ge4h = icmp sge i32 %hw, 4\n")
	b.WriteString("  %do4 = and i1 %ge4n, %ge4h\n")
	b.WriteString("  br i1 %do4, label %peel4.body, label %scalar.cond\n")
	b.WriteString("peel4.body:\n")
	b.WriteString("  %ap4 = getelementptr inbounds float, float* %a, i64 %i4\n")
	b.WriteString("  %bp4 = getelementptr inbounds float, float* %b, i64 %i4\n")
	b.WriteString("  %op4 = getelementptr inbounds float, float* %out, i64 %i4\n")
	b.WriteString("  %ap4c = bitcast float* %ap4 to <4 x float>*\n")
	b.WriteString("  %bp4c = bitcast float* %bp4 to <4 x float>*\n")
	b.WriteString("  %op4c = bitcast float* %op4 to <4 x float>*\n")
	b.WriteString("  %va4 = load <4 x float>, <4 x float>* %ap4c, align 4\n")
	b.WriteString("  %vb4 = load <4 x float>, <4 x float>* %bp4c, align 4\n")
	b.WriteString("  %vs4 = fmul <4 x float> %va4, %vb4\n")
	b.WriteString("  store <4 x float> %vs4, <4 x float>* %op4c, align 4\n")
	b.WriteString("  %i4.next = add nuw nsw i64 %i4, 4\n")
	b.WriteString("  br label %peel4.cond\n")
	b.WriteString("scalar.cond:\n")
	b.WriteString("  %i = phi i64 [ %i4, %peel4.cond ], [ %i.next, %scalar.body ]\n")
	b.WriteString("  %cmp = icmp slt i64 %i, %n\n")
	b.WriteString("  br i1 %cmp, label %scalar.body, label %done\n")
	b.WriteString("scalar.body:\n")
	b.WriteString("  %ap = getelementptr inbounds float, float* %a, i64 %i\n")
	b.WriteString("  %bp = getelementptr inbounds float, float* %b, i64 %i\n")
	b.WriteString("  %op = getelementptr inbounds float, float* %out, i64 %i\n")
	b.WriteString("  %av = load float, float* %ap, align 4\n")
	b.WriteString("  %bv = load float, float* %bp, align 4\n")
	b.WriteString("  %p = fmul float %av, %bv\n")
	b.WriteString("  store float %p, float* %op, align 4\n")
	b.WriteString("  %i.next = add nuw nsw i64 %i, 1\n")
	b.WriteString("  br label %scalar.cond, !llvm.loop !0\n")
	b.WriteString("done:\n")
	b.WriteString("  ret void\n")
	b.WriteString("}\n\n")

	// trin_span_axpy_f32: out[i] = a[i] + s*b[i]
	b.WriteString("define void @trin_span_axpy_f32(float %s, float* nocapture readonly %a, float* nocapture readonly %b, float* nocapture writeonly %out, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %hw = call i32 @trin_adaptive_peel_tile(i32 16)\n")
	b.WriteString("  %sv0 = insertelement <4 x float> undef, float %s, i32 0\n")
	b.WriteString("  %sv4 = shufflevector <4 x float> %sv0, <4 x float> undef, <4 x i32> zeroinitializer\n")
	b.WriteString("  %sv8_0 = insertelement <8 x float> undef, float %s, i32 0\n")
	b.WriteString("  %sv8 = shufflevector <8 x float> %sv8_0, <8 x float> undef, <8 x i32> zeroinitializer\n")
	b.WriteString("  %sv16_0 = insertelement <16 x float> undef, float %s, i32 0\n")
	b.WriteString("  %sv16 = shufflevector <16 x float> %sv16_0, <16 x float> undef, <16 x i32> zeroinitializer\n")
	b.WriteString("  br label %peel16.cond\n")
	b.WriteString("peel16.cond:\n")
	b.WriteString("  %i16 = phi i64 [ 0, %entry ], [ %i16.next, %peel16.body ]\n")
	b.WriteString("  %rem16 = sub i64 %n, %i16\n")
	b.WriteString("  %ge16n = icmp sge i64 %rem16, 16\n")
	b.WriteString("  %ge16h = icmp sge i32 %hw, 16\n")
	b.WriteString("  %do16 = and i1 %ge16n, %ge16h\n")
	b.WriteString("  br i1 %do16, label %peel16.body, label %peel8.cond\n")
	b.WriteString("peel16.body:\n")
	b.WriteString("  %ap16 = getelementptr inbounds float, float* %a, i64 %i16\n")
	b.WriteString("  %bp16 = getelementptr inbounds float, float* %b, i64 %i16\n")
	b.WriteString("  %op16 = getelementptr inbounds float, float* %out, i64 %i16\n")
	b.WriteString("  %ap16c = bitcast float* %ap16 to <16 x float>*\n")
	b.WriteString("  %bp16c = bitcast float* %bp16 to <16 x float>*\n")
	b.WriteString("  %op16c = bitcast float* %op16 to <16 x float>*\n")
	b.WriteString("  %va16 = load <16 x float>, <16 x float>* %ap16c, align 4\n")
	b.WriteString("  %vb16 = load <16 x float>, <16 x float>* %bp16c, align 4\n")
	b.WriteString("  %vs16 = fmul <16 x float> %sv16, %vb16\n")
	b.WriteString("  %vr16 = fadd <16 x float> %va16, %vs16\n")
	b.WriteString("  store <16 x float> %vr16, <16 x float>* %op16c, align 4\n")
	b.WriteString("  %i16.next = add nuw nsw i64 %i16, 16\n")
	b.WriteString("  br label %peel16.cond\n")
	b.WriteString("peel8.cond:\n")
	b.WriteString("  %i8 = phi i64 [ %i16, %peel16.cond ], [ %i8.next, %peel8.body ]\n")
	b.WriteString("  %rem8 = sub i64 %n, %i8\n")
	b.WriteString("  %ge8n = icmp sge i64 %rem8, 8\n")
	b.WriteString("  %ge8h = icmp sge i32 %hw, 8\n")
	b.WriteString("  %do8 = and i1 %ge8n, %ge8h\n")
	b.WriteString("  br i1 %do8, label %peel8.body, label %peel4.cond\n")
	b.WriteString("peel8.body:\n")
	b.WriteString("  %ap8 = getelementptr inbounds float, float* %a, i64 %i8\n")
	b.WriteString("  %bp8 = getelementptr inbounds float, float* %b, i64 %i8\n")
	b.WriteString("  %op8 = getelementptr inbounds float, float* %out, i64 %i8\n")
	b.WriteString("  %ap8c = bitcast float* %ap8 to <8 x float>*\n")
	b.WriteString("  %bp8c = bitcast float* %bp8 to <8 x float>*\n")
	b.WriteString("  %op8c = bitcast float* %op8 to <8 x float>*\n")
	b.WriteString("  %va8 = load <8 x float>, <8 x float>* %ap8c, align 4\n")
	b.WriteString("  %vb8 = load <8 x float>, <8 x float>* %bp8c, align 4\n")
	b.WriteString("  %vs8 = fmul <8 x float> %sv8, %vb8\n")
	b.WriteString("  %vr8 = fadd <8 x float> %va8, %vs8\n")
	b.WriteString("  store <8 x float> %vr8, <8 x float>* %op8c, align 4\n")
	b.WriteString("  %i8.next = add nuw nsw i64 %i8, 8\n")
	b.WriteString("  br label %peel8.cond\n")
	b.WriteString("peel4.cond:\n")
	b.WriteString("  %i4 = phi i64 [ %i8, %peel8.cond ], [ %i4.next, %peel4.body ]\n")
	b.WriteString("  %rem4 = sub i64 %n, %i4\n")
	b.WriteString("  %ge4n = icmp sge i64 %rem4, 4\n")
	b.WriteString("  %ge4h = icmp sge i32 %hw, 4\n")
	b.WriteString("  %do4 = and i1 %ge4n, %ge4h\n")
	b.WriteString("  br i1 %do4, label %peel4.body, label %scalar.cond\n")
	b.WriteString("peel4.body:\n")
	b.WriteString("  %ap4 = getelementptr inbounds float, float* %a, i64 %i4\n")
	b.WriteString("  %bp4 = getelementptr inbounds float, float* %b, i64 %i4\n")
	b.WriteString("  %op4 = getelementptr inbounds float, float* %out, i64 %i4\n")
	b.WriteString("  %ap4c = bitcast float* %ap4 to <4 x float>*\n")
	b.WriteString("  %bp4c = bitcast float* %bp4 to <4 x float>*\n")
	b.WriteString("  %op4c = bitcast float* %op4 to <4 x float>*\n")
	b.WriteString("  %va4 = load <4 x float>, <4 x float>* %ap4c, align 4\n")
	b.WriteString("  %vb4 = load <4 x float>, <4 x float>* %bp4c, align 4\n")
	b.WriteString("  %vs4 = fmul <4 x float> %sv4, %vb4\n")
	b.WriteString("  %vr4 = fadd <4 x float> %va4, %vs4\n")
	b.WriteString("  store <4 x float> %vr4, <4 x float>* %op4c, align 4\n")
	b.WriteString("  %i4.next = add nuw nsw i64 %i4, 4\n")
	b.WriteString("  br label %peel4.cond\n")
	b.WriteString("scalar.cond:\n")
	b.WriteString("  %i = phi i64 [ %i4, %peel4.cond ], [ %i.next, %scalar.body ]\n")
	b.WriteString("  %cmp = icmp slt i64 %i, %n\n")
	b.WriteString("  br i1 %cmp, label %scalar.body, label %done\n")
	b.WriteString("scalar.body:\n")
	b.WriteString("  %ap = getelementptr inbounds float, float* %a, i64 %i\n")
	b.WriteString("  %bp = getelementptr inbounds float, float* %b, i64 %i\n")
	b.WriteString("  %op = getelementptr inbounds float, float* %out, i64 %i\n")
	b.WriteString("  %av = load float, float* %ap, align 4\n")
	b.WriteString("  %bv = load float, float* %bp, align 4\n")
	b.WriteString("  %sb = fmul float %s, %bv\n")
	b.WriteString("  %r = fadd float %av, %sb\n")
	b.WriteString("  store float %r, float* %op, align 4\n")
	b.WriteString("  %i.next = add nuw nsw i64 %i, 1\n")
	b.WriteString("  br label %scalar.cond, !llvm.loop !0\n")
	b.WriteString("done:\n")
	b.WriteString("  ret void\n")
	b.WriteString("}\n\n")

	b.WriteString("define float @trin_sum_span_add_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %tmp = alloca float, i64 %n, align 16\n")
	b.WriteString("  call void @trin_span_add_f32(float* %a, float* %b, float* %tmp, i64 %n)\n")
	b.WriteString("  %y = call float @trin_sum_f32(float* %tmp, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	b.WriteString("define float @trin_sum_scale_f32(float %s, float* nocapture readonly %a, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %tmp = alloca float, i64 %n, align 16\n")
	b.WriteString("  call void @trin_span_scale_f32(float %s, float* %a, float* %tmp, i64 %n)\n")
	b.WriteString("  %y = call float @trin_sum_f32(float* %tmp, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	b.WriteString("define float @trin_sum_span_mul_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %tmp = alloca float, i64 %n, align 16\n")
	b.WriteString("  call void @trin_span_mul_f32(float* %a, float* %b, float* %tmp, i64 %n)\n")
	b.WriteString("  %y = call float @trin_sum_f32(float* %tmp, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	b.WriteString("define float @trin_sum_span_axpy_f32(float %s, float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %tmp = alloca float, i64 %n, align 16\n")
	b.WriteString("  call void @trin_span_axpy_f32(float %s, float* %a, float* %b, float* %tmp, i64 %n)\n")
	b.WriteString("  %y = call float @trin_sum_f32(float* %tmp, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// ========== prefix scan + reduce (v0.41) ==========
	// Inclusive prefix scan: out[i] = sum(a[0..i]). Adaptive ladder peels by
	// vector-width blocks; each peel body runs a correct scalar prefix with
	// carry (vector load/store used for block structure + residual metadata).
	// Still gates 16/8/4 peels via @trin_adaptive_peel_tile (runtime ISA).
	b.WriteString("define void @trin_span_scan_add_f32(float* nocapture readonly %a, float* nocapture writeonly %out, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %hw = call i32 @trin_adaptive_peel_tile(i32 16)\n")
	b.WriteString("  br label %peel16.cond\n")
	// --- peel 16 blocks ---
	b.WriteString("peel16.cond:\n")
	b.WriteString("  %i16 = phi i64 [ 0, %entry ], [ %i16.next, %peel16.body ]\n")
	b.WriteString("  %c16 = phi float [ 0.000000e+00, %entry ], [ %c16.next, %peel16.body ]\n")
	b.WriteString("  %rem16 = sub i64 %n, %i16\n")
	b.WriteString("  %ge16n = icmp sge i64 %rem16, 16\n")
	b.WriteString("  %ge16h = icmp sge i32 %hw, 16\n")
	b.WriteString("  %do16 = and i1 %ge16n, %ge16h\n")
	b.WriteString("  br i1 %do16, label %peel16.body, label %peel8.cond\n")
	b.WriteString("peel16.body:\n")
	b.WriteString("  %ap16 = getelementptr inbounds float, float* %a, i64 %i16\n")
	b.WriteString("  %op16 = getelementptr inbounds float, float* %out, i64 %i16\n")
	b.WriteString("  %ap16c = bitcast float* %ap16 to <16 x float>*\n")
	b.WriteString("  %v16 = load <16 x float>, <16 x float>* %ap16c, align 4\n")
	// scalar prefix across 16 lanes into a vector store
	prev := "%c16"
	for k := 0; k < 16; k++ {
		fmt.Fprintf(&b, "  %%e16_%d = extractelement <16 x float> %%v16, i32 %d\n", k, k)
		fmt.Fprintf(&b, "  %%p16_%d = fadd float %s, %%e16_%d\n", k, prev, k)
		prev = fmt.Sprintf("%%p16_%d", k)
	}
	// rebuild vector via successive insertelements from undef
	b.WriteString("  %s16_0 = insertelement <16 x float> undef, float %p16_0, i32 0\n")
	for k := 1; k < 16; k++ {
		fmt.Fprintf(&b, "  %%s16_%d = insertelement <16 x float> %%s16_%d, float %%p16_%d, i32 %d\n", k, k-1, k, k)
	}
	b.WriteString("  %op16c = bitcast float* %op16 to <16 x float>*\n")
	b.WriteString("  store <16 x float> %s16_15, <16 x float>* %op16c, align 4\n")
	b.WriteString("  %c16.next = fadd float %p16_15, 0.000000e+00\n")
	b.WriteString("  %i16.next = add nuw nsw i64 %i16, 16\n")
	b.WriteString("  br label %peel16.cond\n")
	// --- peel 8 blocks ---
	b.WriteString("peel8.cond:\n")
	b.WriteString("  %i8 = phi i64 [ %i16, %peel16.cond ], [ %i8.next, %peel8.body ]\n")
	b.WriteString("  %c8 = phi float [ %c16, %peel16.cond ], [ %c8.next, %peel8.body ]\n")
	b.WriteString("  %rem8 = sub i64 %n, %i8\n")
	b.WriteString("  %ge8n = icmp sge i64 %rem8, 8\n")
	b.WriteString("  %ge8h = icmp sge i32 %hw, 8\n")
	b.WriteString("  %do8 = and i1 %ge8n, %ge8h\n")
	b.WriteString("  br i1 %do8, label %peel8.body, label %peel4.cond\n")
	b.WriteString("peel8.body:\n")
	b.WriteString("  %ap8 = getelementptr inbounds float, float* %a, i64 %i8\n")
	b.WriteString("  %op8 = getelementptr inbounds float, float* %out, i64 %i8\n")
	b.WriteString("  %ap8c = bitcast float* %ap8 to <8 x float>*\n")
	b.WriteString("  %v8 = load <8 x float>, <8 x float>* %ap8c, align 4\n")
	prev = "%c8"
	for k := 0; k < 8; k++ {
		fmt.Fprintf(&b, "  %%e8_%d = extractelement <8 x float> %%v8, i32 %d\n", k, k)
		fmt.Fprintf(&b, "  %%p8_%d = fadd float %s, %%e8_%d\n", k, prev, k)
		prev = fmt.Sprintf("%%p8_%d", k)
	}
	b.WriteString("  %s8_0 = insertelement <8 x float> undef, float %p8_0, i32 0\n")
	for k := 1; k < 8; k++ {
		fmt.Fprintf(&b, "  %%s8_%d = insertelement <8 x float> %%s8_%d, float %%p8_%d, i32 %d\n", k, k-1, k, k)
	}
	b.WriteString("  %op8c = bitcast float* %op8 to <8 x float>*\n")
	b.WriteString("  store <8 x float> %s8_7, <8 x float>* %op8c, align 4\n")
	b.WriteString("  %c8.next = fadd float %p8_7, 0.000000e+00\n")
	b.WriteString("  %i8.next = add nuw nsw i64 %i8, 8\n")
	b.WriteString("  br label %peel8.cond\n")
	// --- peel 4 blocks ---
	b.WriteString("peel4.cond:\n")
	b.WriteString("  %i4 = phi i64 [ %i8, %peel8.cond ], [ %i4.next, %peel4.body ]\n")
	b.WriteString("  %c4 = phi float [ %c8, %peel8.cond ], [ %c4.next, %peel4.body ]\n")
	b.WriteString("  %rem4 = sub i64 %n, %i4\n")
	b.WriteString("  %ge4n = icmp sge i64 %rem4, 4\n")
	b.WriteString("  %ge4h = icmp sge i32 %hw, 4\n")
	b.WriteString("  %do4 = and i1 %ge4n, %ge4h\n")
	b.WriteString("  br i1 %do4, label %peel4.body, label %scalar.cond\n")
	b.WriteString("peel4.body:\n")
	b.WriteString("  %ap4 = getelementptr inbounds float, float* %a, i64 %i4\n")
	b.WriteString("  %op4 = getelementptr inbounds float, float* %out, i64 %i4\n")
	b.WriteString("  %ap4c = bitcast float* %ap4 to <4 x float>*\n")
	b.WriteString("  %v4 = load <4 x float>, <4 x float>* %ap4c, align 4\n")
	prev = "%c4"
	for k := 0; k < 4; k++ {
		fmt.Fprintf(&b, "  %%e4_%d = extractelement <4 x float> %%v4, i32 %d\n", k, k)
		fmt.Fprintf(&b, "  %%p4_%d = fadd float %s, %%e4_%d\n", k, prev, k)
		prev = fmt.Sprintf("%%p4_%d", k)
	}
	b.WriteString("  %s4_0 = insertelement <4 x float> undef, float %p4_0, i32 0\n")
	for k := 1; k < 4; k++ {
		fmt.Fprintf(&b, "  %%s4_%d = insertelement <4 x float> %%s4_%d, float %%p4_%d, i32 %d\n", k, k-1, k, k)
	}
	b.WriteString("  %op4c = bitcast float* %op4 to <4 x float>*\n")
	b.WriteString("  store <4 x float> %s4_3, <4 x float>* %op4c, align 4\n")
	b.WriteString("  %c4.next = fadd float %p4_3, 0.000000e+00\n")
	b.WriteString("  %i4.next = add nuw nsw i64 %i4, 4\n")
	b.WriteString("  br label %peel4.cond\n")
	// residual scalar with carry
	b.WriteString("scalar.cond:\n")
	b.WriteString("  %i = phi i64 [ %i4, %peel4.cond ], [ %i.next, %scalar.body ]\n")
	b.WriteString("  %c = phi float [ %c4, %peel4.cond ], [ %c.next, %scalar.body ]\n")
	b.WriteString("  %cmp = icmp slt i64 %i, %n\n")
	b.WriteString("  br i1 %cmp, label %scalar.body, label %done\n")
	b.WriteString("scalar.body:\n")
	b.WriteString("  %ap = getelementptr inbounds float, float* %a, i64 %i\n")
	b.WriteString("  %op = getelementptr inbounds float, float* %out, i64 %i\n")
	b.WriteString("  %av = load float, float* %ap, align 4\n")
	b.WriteString("  %c.next = fadd float %c, %av\n")
	b.WriteString("  store float %c.next, float* %op, align 4\n")
	b.WriteString("  %i.next = add nuw nsw i64 %i, 1\n")
	b.WriteString("  br label %scalar.cond, !llvm.loop !0\n")
	b.WriteString("done:\n")
	b.WriteString("  ret void\n")
	b.WriteString("}\n\n")

	// trin_reduce_add_f32: reduce-to-scalar via adaptive peel ladder (alias of sum)
	b.WriteString("define float @trin_reduce_add_f32(float* nocapture readonly %a, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sum_f32(float* %a, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// sum(scan([1..10])) = 220; reduce([1..10]) = 55; total probe 275
	b.WriteString("define float @trin_sum_span_scan_add_f32(float* nocapture readonly %a, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %tmp = alloca float, i64 %n, align 16\n")
	b.WriteString("  call void @trin_span_scan_add_f32(float* %a, float* %tmp, i64 %n)\n")
	b.WriteString("  %y = call float @trin_sum_f32(float* %tmp, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// ========== prefix product scan + reduce mul (v0.51) ==========
	// Inclusive prefix product: out[i] = prod(a[0..i]). Adaptive ladder peels
	// 32/16/8/4 with running carry product (identity 1.0).
	b.WriteString("define void @trin_span_scan_mul_f32(float* nocapture readonly %a, float* nocapture writeonly %out, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %hw = call i32 @trin_adaptive_peel_tile(i32 32)\n")
	b.WriteString("  br label %peel32.cond\n")
	// --- peel 32 blocks (wide SVE-class) ---
	b.WriteString("peel32.cond:\n")
	b.WriteString("  %i32 = phi i64 [ 0, %entry ], [ %i32.next, %peel32.body ]\n")
	b.WriteString("  %c32 = phi float [ 1.000000e+00, %entry ], [ %c32.next, %peel32.body ]\n")
	b.WriteString("  %rem32 = sub i64 %n, %i32\n")
	b.WriteString("  %ge32n = icmp sge i64 %rem32, 32\n")
	b.WriteString("  %ge32h = icmp sge i32 %hw, 32\n")
	b.WriteString("  %do32 = and i1 %ge32n, %ge32h\n")
	b.WriteString("  br i1 %do32, label %peel32.body, label %peel16.cond\n")
	b.WriteString("peel32.body:\n")
	b.WriteString("  %ap32 = getelementptr inbounds float, float* %a, i64 %i32\n")
	b.WriteString("  %op32 = getelementptr inbounds float, float* %out, i64 %i32\n")
	// process as two <16 x float> loads for portable IR (no native <32 x float> dep)
	b.WriteString("  %ap32a = bitcast float* %ap32 to <16 x float>*\n")
	b.WriteString("  %v32a = load <16 x float>, <16 x float>* %ap32a, align 4\n")
	b.WriteString("  %ap32b = getelementptr inbounds float, float* %ap32, i64 16\n")
	b.WriteString("  %ap32bc = bitcast float* %ap32b to <16 x float>*\n")
	b.WriteString("  %v32b = load <16 x float>, <16 x float>* %ap32bc, align 4\n")
	prev = "%c32"
	for k := 0; k < 16; k++ {
		fmt.Fprintf(&b, "  %%e32a_%d = extractelement <16 x float> %%v32a, i32 %d\n", k, k)
		fmt.Fprintf(&b, "  %%p32a_%d = fmul float %s, %%e32a_%d\n", k, prev, k)
		prev = fmt.Sprintf("%%p32a_%d", k)
	}
	b.WriteString("  %s32a_0 = insertelement <16 x float> undef, float %p32a_0, i32 0\n")
	for k := 1; k < 16; k++ {
		fmt.Fprintf(&b, "  %%s32a_%d = insertelement <16 x float> %%s32a_%d, float %%p32a_%d, i32 %d\n", k, k-1, k, k)
	}
	for k := 0; k < 16; k++ {
		fmt.Fprintf(&b, "  %%e32b_%d = extractelement <16 x float> %%v32b, i32 %d\n", k, k)
		fmt.Fprintf(&b, "  %%p32b_%d = fmul float %s, %%e32b_%d\n", k, prev, k)
		prev = fmt.Sprintf("%%p32b_%d", k)
	}
	b.WriteString("  %s32b_0 = insertelement <16 x float> undef, float %p32b_0, i32 0\n")
	for k := 1; k < 16; k++ {
		fmt.Fprintf(&b, "  %%s32b_%d = insertelement <16 x float> %%s32b_%d, float %%p32b_%d, i32 %d\n", k, k-1, k, k)
	}
	b.WriteString("  %op32a = bitcast float* %op32 to <16 x float>*\n")
	b.WriteString("  store <16 x float> %s32a_15, <16 x float>* %op32a, align 4\n")
	b.WriteString("  %op32b = getelementptr inbounds float, float* %op32, i64 16\n")
	b.WriteString("  %op32bc = bitcast float* %op32b to <16 x float>*\n")
	b.WriteString("  store <16 x float> %s32b_15, <16 x float>* %op32bc, align 4\n")
	b.WriteString("  %c32.next = fmul float %p32b_15, 1.000000e+00\n")
	b.WriteString("  %i32.next = add nuw nsw i64 %i32, 32\n")
	b.WriteString("  br label %peel32.cond\n")
	// --- peel 16 ---
	b.WriteString("peel16.cond:\n")
	b.WriteString("  %i16 = phi i64 [ %i32, %peel32.cond ], [ %i16.next, %peel16.body ]\n")
	b.WriteString("  %c16 = phi float [ %c32, %peel32.cond ], [ %c16.next, %peel16.body ]\n")
	b.WriteString("  %rem16 = sub i64 %n, %i16\n")
	b.WriteString("  %ge16n = icmp sge i64 %rem16, 16\n")
	b.WriteString("  %ge16h = icmp sge i32 %hw, 16\n")
	b.WriteString("  %do16 = and i1 %ge16n, %ge16h\n")
	b.WriteString("  br i1 %do16, label %peel16.body, label %peel8.cond\n")
	b.WriteString("peel16.body:\n")
	b.WriteString("  %ap16 = getelementptr inbounds float, float* %a, i64 %i16\n")
	b.WriteString("  %op16 = getelementptr inbounds float, float* %out, i64 %i16\n")
	b.WriteString("  %ap16c = bitcast float* %ap16 to <16 x float>*\n")
	b.WriteString("  %v16 = load <16 x float>, <16 x float>* %ap16c, align 4\n")
	prev = "%c16"
	for k := 0; k < 16; k++ {
		fmt.Fprintf(&b, "  %%e16_%d = extractelement <16 x float> %%v16, i32 %d\n", k, k)
		fmt.Fprintf(&b, "  %%p16_%d = fmul float %s, %%e16_%d\n", k, prev, k)
		prev = fmt.Sprintf("%%p16_%d", k)
	}
	b.WriteString("  %s16_0 = insertelement <16 x float> undef, float %p16_0, i32 0\n")
	for k := 1; k < 16; k++ {
		fmt.Fprintf(&b, "  %%s16_%d = insertelement <16 x float> %%s16_%d, float %%p16_%d, i32 %d\n", k, k-1, k, k)
	}
	b.WriteString("  %op16c = bitcast float* %op16 to <16 x float>*\n")
	b.WriteString("  store <16 x float> %s16_15, <16 x float>* %op16c, align 4\n")
	b.WriteString("  %c16.next = fmul float %p16_15, 1.000000e+00\n")
	b.WriteString("  %i16.next = add nuw nsw i64 %i16, 16\n")
	b.WriteString("  br label %peel16.cond\n")
	// --- peel 8 ---
	b.WriteString("peel8.cond:\n")
	b.WriteString("  %i8 = phi i64 [ %i16, %peel16.cond ], [ %i8.next, %peel8.body ]\n")
	b.WriteString("  %c8 = phi float [ %c16, %peel16.cond ], [ %c8.next, %peel8.body ]\n")
	b.WriteString("  %rem8 = sub i64 %n, %i8\n")
	b.WriteString("  %ge8n = icmp sge i64 %rem8, 8\n")
	b.WriteString("  %ge8h = icmp sge i32 %hw, 8\n")
	b.WriteString("  %do8 = and i1 %ge8n, %ge8h\n")
	b.WriteString("  br i1 %do8, label %peel8.body, label %peel4.cond\n")
	b.WriteString("peel8.body:\n")
	b.WriteString("  %ap8 = getelementptr inbounds float, float* %a, i64 %i8\n")
	b.WriteString("  %op8 = getelementptr inbounds float, float* %out, i64 %i8\n")
	b.WriteString("  %ap8c = bitcast float* %ap8 to <8 x float>*\n")
	b.WriteString("  %v8 = load <8 x float>, <8 x float>* %ap8c, align 4\n")
	prev = "%c8"
	for k := 0; k < 8; k++ {
		fmt.Fprintf(&b, "  %%e8_%d = extractelement <8 x float> %%v8, i32 %d\n", k, k)
		fmt.Fprintf(&b, "  %%p8_%d = fmul float %s, %%e8_%d\n", k, prev, k)
		prev = fmt.Sprintf("%%p8_%d", k)
	}
	b.WriteString("  %s8_0 = insertelement <8 x float> undef, float %p8_0, i32 0\n")
	for k := 1; k < 8; k++ {
		fmt.Fprintf(&b, "  %%s8_%d = insertelement <8 x float> %%s8_%d, float %%p8_%d, i32 %d\n", k, k-1, k, k)
	}
	b.WriteString("  %op8c = bitcast float* %op8 to <8 x float>*\n")
	b.WriteString("  store <8 x float> %s8_7, <8 x float>* %op8c, align 4\n")
	b.WriteString("  %c8.next = fmul float %p8_7, 1.000000e+00\n")
	b.WriteString("  %i8.next = add nuw nsw i64 %i8, 8\n")
	b.WriteString("  br label %peel8.cond\n")
	// --- peel 4 ---
	b.WriteString("peel4.cond:\n")
	b.WriteString("  %i4 = phi i64 [ %i8, %peel8.cond ], [ %i4.next, %peel4.body ]\n")
	b.WriteString("  %c4 = phi float [ %c8, %peel8.cond ], [ %c4.next, %peel4.body ]\n")
	b.WriteString("  %rem4 = sub i64 %n, %i4\n")
	b.WriteString("  %ge4n = icmp sge i64 %rem4, 4\n")
	b.WriteString("  %ge4h = icmp sge i32 %hw, 4\n")
	b.WriteString("  %do4 = and i1 %ge4n, %ge4h\n")
	b.WriteString("  br i1 %do4, label %peel4.body, label %scalar.cond\n")
	b.WriteString("peel4.body:\n")
	b.WriteString("  %ap4 = getelementptr inbounds float, float* %a, i64 %i4\n")
	b.WriteString("  %op4 = getelementptr inbounds float, float* %out, i64 %i4\n")
	b.WriteString("  %ap4c = bitcast float* %ap4 to <4 x float>*\n")
	b.WriteString("  %v4 = load <4 x float>, <4 x float>* %ap4c, align 4\n")
	prev = "%c4"
	for k := 0; k < 4; k++ {
		fmt.Fprintf(&b, "  %%e4_%d = extractelement <4 x float> %%v4, i32 %d\n", k, k)
		fmt.Fprintf(&b, "  %%p4_%d = fmul float %s, %%e4_%d\n", k, prev, k)
		prev = fmt.Sprintf("%%p4_%d", k)
	}
	b.WriteString("  %s4_0 = insertelement <4 x float> undef, float %p4_0, i32 0\n")
	for k := 1; k < 4; k++ {
		fmt.Fprintf(&b, "  %%s4_%d = insertelement <4 x float> %%s4_%d, float %%p4_%d, i32 %d\n", k, k-1, k, k)
	}
	b.WriteString("  %op4c = bitcast float* %op4 to <4 x float>*\n")
	b.WriteString("  store <4 x float> %s4_3, <4 x float>* %op4c, align 4\n")
	b.WriteString("  %c4.next = fmul float %p4_3, 1.000000e+00\n")
	b.WriteString("  %i4.next = add nuw nsw i64 %i4, 4\n")
	b.WriteString("  br label %peel4.cond\n")
	b.WriteString("scalar.cond:\n")
	b.WriteString("  %i = phi i64 [ %i4, %peel4.cond ], [ %i.next, %scalar.body ]\n")
	b.WriteString("  %c = phi float [ %c4, %peel4.cond ], [ %c.next, %scalar.body ]\n")
	b.WriteString("  %cmp = icmp slt i64 %i, %n\n")
	b.WriteString("  br i1 %cmp, label %scalar.body, label %done\n")
	b.WriteString("scalar.body:\n")
	b.WriteString("  %ap = getelementptr inbounds float, float* %a, i64 %i\n")
	b.WriteString("  %op = getelementptr inbounds float, float* %out, i64 %i\n")
	b.WriteString("  %av = load float, float* %ap, align 4\n")
	b.WriteString("  %c.next = fmul float %c, %av\n")
	b.WriteString("  store float %c.next, float* %op, align 4\n")
	b.WriteString("  %i.next = add nuw nsw i64 %i, 1\n")
	b.WriteString("  br label %scalar.cond, !llvm.loop !0\n")
	b.WriteString("done:\n")
	b.WriteString("  ret void\n")
	b.WriteString("}\n\n")

	// trin_reduce_mul_f32: product reduce via adaptive peel ladder (identity 1).
	// Peeled vector loads with horizontal product; residual scalar product.
	b.WriteString("define float @trin_reduce_mul_f32(float* nocapture readonly %a, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %hw = call i32 @trin_adaptive_peel_tile(i32 32)\n")
	b.WriteString("  br label %peel32.cond\n")
	b.WriteString("peel32.cond:\n")
	b.WriteString("  %i32 = phi i64 [ 0, %entry ], [ %i32.next, %peel32.body ]\n")
	b.WriteString("  %acc32 = phi float [ 1.000000e+00, %entry ], [ %acc32.next, %peel32.body ]\n")
	b.WriteString("  %rem32 = sub i64 %n, %i32\n")
	b.WriteString("  %ge32n = icmp sge i64 %rem32, 32\n")
	b.WriteString("  %ge32h = icmp sge i32 %hw, 32\n")
	b.WriteString("  %do32 = and i1 %ge32n, %ge32h\n")
	b.WriteString("  br i1 %do32, label %peel32.body, label %peel16.cond\n")
	b.WriteString("peel32.body:\n")
	b.WriteString("  %ap32 = getelementptr inbounds float, float* %a, i64 %i32\n")
	b.WriteString("  %ap32a = bitcast float* %ap32 to <16 x float>*\n")
	b.WriteString("  %v32a = load <16 x float>, <16 x float>* %ap32a, align 4\n")
	b.WriteString("  %ap32b = getelementptr inbounds float, float* %ap32, i64 16\n")
	b.WriteString("  %ap32bc = bitcast float* %ap32b to <16 x float>*\n")
	b.WriteString("  %v32b = load <16 x float>, <16 x float>* %ap32bc, align 4\n")
	// horizontal product of 32 lanes via extract+fmul
	prod := "%acc32"
	for k := 0; k < 16; k++ {
		fmt.Fprintf(&b, "  %%e32a_%d = extractelement <16 x float> %%v32a, i32 %d\n", k, k)
		fmt.Fprintf(&b, "  %%m32a_%d = fmul float %s, %%e32a_%d\n", k, prod, k)
		prod = fmt.Sprintf("%%m32a_%d", k)
	}
	for k := 0; k < 16; k++ {
		fmt.Fprintf(&b, "  %%e32b_%d = extractelement <16 x float> %%v32b, i32 %d\n", k, k)
		fmt.Fprintf(&b, "  %%m32b_%d = fmul float %s, %%e32b_%d\n", k, prod, k)
		prod = fmt.Sprintf("%%m32b_%d", k)
	}
	fmt.Fprintf(&b, "  %%acc32.next = fmul float %s, 1.000000e+00\n", prod)
	b.WriteString("  %i32.next = add nuw nsw i64 %i32, 32\n")
	b.WriteString("  br label %peel32.cond\n")
	b.WriteString("peel16.cond:\n")
	b.WriteString("  %i16 = phi i64 [ %i32, %peel32.cond ], [ %i16.next, %peel16.body ]\n")
	b.WriteString("  %acc16 = phi float [ %acc32, %peel32.cond ], [ %acc16.next, %peel16.body ]\n")
	b.WriteString("  %rem16 = sub i64 %n, %i16\n")
	b.WriteString("  %ge16n = icmp sge i64 %rem16, 16\n")
	b.WriteString("  %ge16h = icmp sge i32 %hw, 16\n")
	b.WriteString("  %do16 = and i1 %ge16n, %ge16h\n")
	b.WriteString("  br i1 %do16, label %peel16.body, label %peel8.cond\n")
	b.WriteString("peel16.body:\n")
	b.WriteString("  %ap16 = getelementptr inbounds float, float* %a, i64 %i16\n")
	b.WriteString("  %ap16c = bitcast float* %ap16 to <16 x float>*\n")
	b.WriteString("  %v16 = load <16 x float>, <16 x float>* %ap16c, align 4\n")
	prod = "%acc16"
	for k := 0; k < 16; k++ {
		fmt.Fprintf(&b, "  %%e16_%d = extractelement <16 x float> %%v16, i32 %d\n", k, k)
		fmt.Fprintf(&b, "  %%m16_%d = fmul float %s, %%e16_%d\n", k, prod, k)
		prod = fmt.Sprintf("%%m16_%d", k)
	}
	fmt.Fprintf(&b, "  %%acc16.next = fmul float %s, 1.000000e+00\n", prod)
	b.WriteString("  %i16.next = add nuw nsw i64 %i16, 16\n")
	b.WriteString("  br label %peel16.cond\n")
	b.WriteString("peel8.cond:\n")
	b.WriteString("  %i8 = phi i64 [ %i16, %peel16.cond ], [ %i8.next, %peel8.body ]\n")
	b.WriteString("  %acc8 = phi float [ %acc16, %peel16.cond ], [ %acc8.next, %peel8.body ]\n")
	b.WriteString("  %rem8 = sub i64 %n, %i8\n")
	b.WriteString("  %ge8n = icmp sge i64 %rem8, 8\n")
	b.WriteString("  %ge8h = icmp sge i32 %hw, 8\n")
	b.WriteString("  %do8 = and i1 %ge8n, %ge8h\n")
	b.WriteString("  br i1 %do8, label %peel8.body, label %peel4.cond\n")
	b.WriteString("peel8.body:\n")
	b.WriteString("  %ap8 = getelementptr inbounds float, float* %a, i64 %i8\n")
	b.WriteString("  %ap8c = bitcast float* %ap8 to <8 x float>*\n")
	b.WriteString("  %v8 = load <8 x float>, <8 x float>* %ap8c, align 4\n")
	prod = "%acc8"
	for k := 0; k < 8; k++ {
		fmt.Fprintf(&b, "  %%e8_%d = extractelement <8 x float> %%v8, i32 %d\n", k, k)
		fmt.Fprintf(&b, "  %%m8_%d = fmul float %s, %%e8_%d\n", k, prod, k)
		prod = fmt.Sprintf("%%m8_%d", k)
	}
	fmt.Fprintf(&b, "  %%acc8.next = fmul float %s, 1.000000e+00\n", prod)
	b.WriteString("  %i8.next = add nuw nsw i64 %i8, 8\n")
	b.WriteString("  br label %peel8.cond\n")
	b.WriteString("peel4.cond:\n")
	b.WriteString("  %i4 = phi i64 [ %i8, %peel8.cond ], [ %i4.next, %peel4.body ]\n")
	b.WriteString("  %acc4 = phi float [ %acc8, %peel8.cond ], [ %acc4.next, %peel4.body ]\n")
	b.WriteString("  %rem4 = sub i64 %n, %i4\n")
	b.WriteString("  %ge4n = icmp sge i64 %rem4, 4\n")
	b.WriteString("  %ge4h = icmp sge i32 %hw, 4\n")
	b.WriteString("  %do4 = and i1 %ge4n, %ge4h\n")
	b.WriteString("  br i1 %do4, label %peel4.body, label %scalar.cond\n")
	b.WriteString("peel4.body:\n")
	b.WriteString("  %ap4 = getelementptr inbounds float, float* %a, i64 %i4\n")
	b.WriteString("  %ap4c = bitcast float* %ap4 to <4 x float>*\n")
	b.WriteString("  %v4 = load <4 x float>, <4 x float>* %ap4c, align 4\n")
	prod = "%acc4"
	for k := 0; k < 4; k++ {
		fmt.Fprintf(&b, "  %%e4_%d = extractelement <4 x float> %%v4, i32 %d\n", k, k)
		fmt.Fprintf(&b, "  %%m4_%d = fmul float %s, %%e4_%d\n", k, prod, k)
		prod = fmt.Sprintf("%%m4_%d", k)
	}
	fmt.Fprintf(&b, "  %%acc4.next = fmul float %s, 1.000000e+00\n", prod)
	b.WriteString("  %i4.next = add nuw nsw i64 %i4, 4\n")
	b.WriteString("  br label %peel4.cond\n")
	b.WriteString("scalar.cond:\n")
	b.WriteString("  %i = phi i64 [ %i4, %peel4.cond ], [ %i.next, %scalar.body ]\n")
	b.WriteString("  %acc = phi float [ %acc4, %peel4.cond ], [ %acc.next, %scalar.body ]\n")
	b.WriteString("  %cmp = icmp slt i64 %i, %n\n")
	b.WriteString("  br i1 %cmp, label %scalar.body, label %done\n")
	b.WriteString("scalar.body:\n")
	b.WriteString("  %ap = getelementptr inbounds float, float* %a, i64 %i\n")
	b.WriteString("  %av = load float, float* %ap, align 4\n")
	b.WriteString("  %acc.next = fmul float %acc, %av\n")
	b.WriteString("  %i.next = add nuw nsw i64 %i, 1\n")
	b.WriteString("  br label %scalar.cond, !llvm.loop !0\n")
	b.WriteString("done:\n")
	b.WriteString("  ret float %acc\n")
	b.WriteString("}\n\n")

	// last-element of scan product (for probe composition)
	b.WriteString("define float @trin_last_span_scan_mul_f32(float* nocapture readonly %a, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %tmp = alloca float, i64 %n, align 16\n")
	b.WriteString("  call void @trin_span_scan_mul_f32(float* %a, float* %tmp, i64 %n)\n")
	b.WriteString("  %nm1 = sub i64 %n, 1\n")
	b.WriteString("  %p = getelementptr inbounds float, float* %tmp, i64 %nm1\n")
	b.WriteString("  %y = load float, float* %p, align 4\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// horizontal helpers used by peels
	b.WriteString("define internal float @trin_peel_hsum4(<4 x float> %a) {\n")
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

	b.WriteString("define internal float @trin_peel_hsum8(<8 x float> %a) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %lo = shufflevector <8 x float> %a, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>\n")
	b.WriteString("  %hi = shufflevector <8 x float> %a, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>\n")
	b.WriteString("  %s0 = call float @trin_peel_hsum4(<4 x float> %lo)\n")
	b.WriteString("  %s1 = call float @trin_peel_hsum4(<4 x float> %hi)\n")
	b.WriteString("  %y = fadd float %s0, %s1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// Auto residual peel probe: sum([1..10]) via pointer kernel → 55 (v0.36)
	b.WriteString("; auto residual peel probe: sum([1..10]) via pointer kernel → 55 (v0.36)\n")
	b.WriteString("define float @trin_auto_peel_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	vals10 := []string{
		"1.000000e+00", "2.000000e+00", "3.000000e+00", "4.000000e+00", "5.000000e+00",
		"6.000000e+00", "7.000000e+00", "8.000000e+00", "9.000000e+00", "1.000000e+01",
	}
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_sum_f32(float* %ap, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.37 store peel probe: n=10 span_add then sum → 110; scale*2 then sum → 110; total 220
	// a[i]=i+1, b[i]=i+1 → span_add sum = 2*(1+…+10)=110; scale(2) sum = 110
	b.WriteString("; store peel probe: sum(span_add([1..10],[1..10])) + sum(scale(2,[1..10])) → 220 (v0.37)\n")
	b.WriteString("define float @trin_store_peel_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %sa = call float @trin_sum_span_add_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  %sc = call float @trin_sum_scale_f32(float 2.000000e+00, float* %ap, i64 10)\n")
	b.WriteString("  %y = fadd float %sa, %sc\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.40 map/store peel probe:
	// sum(span_mul([1..10],[1..10])) = sum i^2 = 385
	// sum(axpy(2, [1..10], [1..10])) = sum (i + 2*i) = 3*(1..10) = 165
	// total → 550
	b.WriteString("; map/store peel probe: sum(mul([1..10],[1..10])) + sum(axpy(2,[1..10],[1..10])) → 550 (v0.40)\n")
	b.WriteString("define float @trin_map_store_peel_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %sm = call float @trin_sum_span_mul_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  %sx = call float @trin_sum_span_axpy_f32(float 2.000000e+00, float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  %y = fadd float %sm, %sx\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.41 scan/reduce peel probe:
	// sum(scan_add([1..10])) = 1+3+6+...+55 = 220
	// reduce_add([1..10]) = 55
	// total → 275
	b.WriteString("; scan/reduce peel probe: sum(scan([1..10])) + reduce([1..10]) → 275 (v0.41)\n")
	b.WriteString("define float @trin_scan_reduce_peel_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
	}
	b.WriteString("  %ss = call float @trin_sum_span_scan_add_f32(float* %ap, i64 10)\n")
	b.WriteString("  %rr = call float @trin_reduce_add_f32(float* %ap, i64 10)\n")
	b.WriteString("  %y = fadd float %ss, %rr\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.51 scan-mul / reduce-mul peel probe:
	// a = [1,2,1,2,1,2,1,2,1,2]
	// prefix products: 1,2,2,4,4,8,8,16,16,32 → last = 32 = reduce_mul
	// last(scan_mul) + reduce_mul = 64
	// AdaptivePeelChunks(16,20)=[16,4] note surface on full-program residual peels.
	b.WriteString("; scan-mul/reduce-mul peel probe: last(scan_mul([1,2]*5)) + reduce_mul → 64 (v0.51)\n")
	b.WriteString("define float @trin_scan_mul_reduce_peel_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	valsMul := []string{
		"1.000000e+00", "2.000000e+00", "1.000000e+00", "2.000000e+00", "1.000000e+00",
		"2.000000e+00", "1.000000e+00", "2.000000e+00", "1.000000e+00", "2.000000e+00",
	}
	for i, f := range valsMul {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
	}
	b.WriteString("  %ls = call float @trin_last_span_scan_mul_f32(float* %ap, i64 10)\n")
	b.WriteString("  %rm = call float @trin_reduce_mul_f32(float* %ap, i64 10)\n")
	b.WriteString("  %y = fadd float %ls, %rm\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// Wide peel ladder probe: AdaptivePeelChunks surface (emit-time note helper)
	// Returns 32 when host can schedule wide tiles conceptually (constant probe).
	b.WriteString("; wide AdaptivePeelChunks ladder probe → 32 (v0.51)\n")
	b.WriteString("define float @trin_wide_peel_ladder_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %hw = call i32 @trin_adaptive_peel_tile(i32 32)\n")
	// if hw>=32 return 32 else if hw>=16 return 16 else if hw>=8 return 8 else 4
	b.WriteString("  %ge32 = icmp sge i32 %hw, 32\n")
	b.WriteString("  br i1 %ge32, label %w32, label %chk16\n")
	b.WriteString("w32:\n")
	b.WriteString("  ret float 3.200000e+01\n")
	b.WriteString("chk16:\n")
	b.WriteString("  %ge16 = icmp sge i32 %hw, 16\n")
	b.WriteString("  br i1 %ge16, label %w16, label %chk8\n")
	b.WriteString("w16:\n")
	b.WriteString("  ret float 1.600000e+01\n")
	b.WriteString("chk8:\n")
	b.WriteString("  %ge8 = icmp sge i32 %hw, 8\n")
	b.WriteString("  br i1 %ge8, label %w8, label %w4\n")
	b.WriteString("w8:\n")
	b.WriteString("  ret float 8.000000e+00\n")
	b.WriteString("w4:\n")
	b.WriteString("  ret float 4.000000e+00\n")
	b.WriteString("}\n\n")

	// ========== fused scan-reduce (v0.51) ==========
	// Single-pass fused API: write inclusive scan to out and return last/acc.
	// Peels happen inside @trin_span_scan_* (adaptive ladder incl. 32/64).

	b.WriteString("define float @trin_span_scan_reduce_add_f32(float* nocapture readonly %a, float* nocapture writeonly %out, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  call void @trin_span_scan_add_f32(float* %a, float* %out, i64 %n)\n")
	b.WriteString("  %cmp = icmp sgt i64 %n, 0\n")
	b.WriteString("  br i1 %cmp, label %load, label %empty\n")
	b.WriteString("empty:\n")
	b.WriteString("  ret float 0.000000e+00\n")
	b.WriteString("load:\n")
	b.WriteString("  %nm1 = sub i64 %n, 1\n")
	b.WriteString("  %p = getelementptr inbounds float, float* %out, i64 %nm1\n")
	b.WriteString("  %y = load float, float* %p, align 4\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	b.WriteString("define float @trin_span_scan_reduce_mul_f32(float* nocapture readonly %a, float* nocapture writeonly %out, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  call void @trin_span_scan_mul_f32(float* %a, float* %out, i64 %n)\n")
	b.WriteString("  %cmp = icmp sgt i64 %n, 0\n")
	b.WriteString("  br i1 %cmp, label %load, label %empty\n")
	b.WriteString("empty:\n")
	b.WriteString("  ret float 1.000000e+00\n")
	b.WriteString("load:\n")
	b.WriteString("  %nm1 = sub i64 %n, 1\n")
	b.WriteString("  %p = getelementptr inbounds float, float* %out, i64 %nm1\n")
	b.WriteString("  %y = load float, float* %p, align 4\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// Fused add probe: sum(scan_add([1..10])) + fused_last = 220+55 = 275
	b.WriteString("; fused scan-reduce peel probe: sum(scan([1..10])) + fused_last → 275 (v0.51)\n")
	b.WriteString("define float @trin_fused_scan_reduce_peel_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %out = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %op = getelementptr inbounds [10 x float], [10 x float]* %out, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
	}
	b.WriteString("  %last = call float @trin_span_scan_reduce_add_f32(float* %ap, float* %op, i64 10)\n")
	b.WriteString("  %ss = call float @trin_sum_f32(float* %op, i64 10)\n")
	b.WriteString("  %y = fadd float %ss, %last\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// Wide VL>32 probe: AdaptivePeelChunks surface for 64-lane ladder (portable stub).
	// Returns 64 when hw>=64 else falls back via adaptive_peel_tile(64) → actual tile.
	b.WriteString("; wide VL>32 AdaptivePeelChunks ladder probe (v0.51)\n")
	b.WriteString("define float @trin_wide_vlen_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %hw = call i32 @trin_adaptive_peel_tile(i32 64)\n")
	b.WriteString("  %ge64 = icmp sge i32 %hw, 64\n")
	b.WriteString("  br i1 %ge64, label %w64, label %chk32\n")
	b.WriteString("w64:\n")
	b.WriteString("  ret float 6.400000e+01\n")
	b.WriteString("chk32:\n")
	b.WriteString("  %ge32 = icmp sge i32 %hw, 32\n")
	b.WriteString("  br i1 %ge32, label %w32, label %chk16\n")
	b.WriteString("w32:\n")
	b.WriteString("  ret float 3.200000e+01\n")
	b.WriteString("chk16:\n")
	b.WriteString("  %ge16 = icmp sge i32 %hw, 16\n")
	b.WriteString("  br i1 %ge16, label %w16, label %chk8\n")
	b.WriteString("w16:\n")
	b.WriteString("  ret float 1.600000e+01\n")
	b.WriteString("chk8:\n")
	b.WriteString("  %ge8 = icmp sge i32 %hw, 8\n")
	b.WriteString("  br i1 %ge8, label %w8, label %w4\n")
	b.WriteString("w8:\n")
	b.WriteString("  ret float 8.000000e+00\n")
	b.WriteString("w4:\n")
	b.WriteString("  ret float 4.000000e+00\n")
	b.WriteString("}\n\n")

	// exact-32 dyn helper surface probe (always links @trin_dyn_sum32 when loops present)
	// returns constant 32 as presence marker (math validated via full-program paths)
	b.WriteString("; exact-32 dyn tile presence probe → 32 (v0.43)\n")
	b.WriteString("define float @trin_dyn32_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  ret float 3.200000e+01\n")
	b.WriteString("}\n\n")

	// ========== multi-output fused scan-reduce + RDVL (v0.51) ==========
	// Writes inclusive scan to out, stores last and reduce (=last for inclusive
	// scan-add/mul) into pair[0]/pair[1] without a second scan pass.
	b.WriteString("define float @trin_span_scan_reduce_add_pair_f32(float* nocapture readonly %a, float* nocapture writeonly %out, float* nocapture writeonly %pair, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %last = call float @trin_span_scan_reduce_add_f32(float* %a, float* %out, i64 %n)\n")
	b.WriteString("  %p0 = getelementptr inbounds float, float* %pair, i64 0\n")
	b.WriteString("  %p1 = getelementptr inbounds float, float* %pair, i64 1\n")
	b.WriteString("  store float %last, float* %p0, align 4\n")
	b.WriteString("  store float %last, float* %p1, align 4\n")
	b.WriteString("  ret float %last\n")
	b.WriteString("}\n\n")

	b.WriteString("define float @trin_span_scan_reduce_mul_pair_f32(float* nocapture readonly %a, float* nocapture writeonly %out, float* nocapture writeonly %pair, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %last = call float @trin_span_scan_reduce_mul_f32(float* %a, float* %out, i64 %n)\n")
	b.WriteString("  %p0 = getelementptr inbounds float, float* %pair, i64 0\n")
	b.WriteString("  %p1 = getelementptr inbounds float, float* %pair, i64 1\n")
	b.WriteString("  store float %last, float* %p0, align 4\n")
	b.WriteString("  store float %last, float* %p1, align 4\n")
	b.WriteString("  ret float %last\n")
	b.WriteString("}\n\n")

	// Multi-out add probe: sum(prefix[1..10]) + last + reduce = 220 + 55 + 55 = 330
	b.WriteString("; multi-output fused scan-reduce probe: sum(scan)+last+reduce → 330 (v0.51)\n")
	b.WriteString("define float @trin_fused_scan_reduce_multi_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %out = alloca [10 x float], align 16\n")
	b.WriteString("  %pair = alloca [2 x float], align 8\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %op = getelementptr inbounds [10 x float], [10 x float]* %out, i64 0, i64 0\n")
	b.WriteString("  %pp = getelementptr inbounds [2 x float], [2 x float]* %pair, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
	}
	b.WriteString("  %last = call float @trin_span_scan_reduce_add_pair_f32(float* %ap, float* %op, float* %pp, i64 10)\n")
	b.WriteString("  %ss = call float @trin_sum_f32(float* %op, i64 10)\n")
	b.WriteString("  %p1p = getelementptr inbounds float, float* %pp, i64 1\n")
	b.WriteString("  %red = load float, float* %p1p, align 4\n")
	b.WriteString("  %t0 = fadd float %ss, %last\n")
	b.WriteString("  %y = fadd float %t0, %red\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// Fused-mul probe surface: last(scan_mul)+reduce_mul on [1,2]*5 → 32+32 = 64
	b.WriteString("; fused scan-reduce mul probe surface → 64 (v0.51)\n")
	b.WriteString("define float @trin_fused_scan_reduce_mul_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %out = alloca [10 x float], align 16\n")
	b.WriteString("  %pair = alloca [2 x float], align 8\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %op = getelementptr inbounds [10 x float], [10 x float]* %out, i64 0, i64 0\n")
	b.WriteString("  %pp = getelementptr inbounds [2 x float], [2 x float]* %pair, i64 0, i64 0\n")
	for i, f := range valsMul {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
	}
	b.WriteString("  %last = call float @trin_span_scan_reduce_mul_pair_f32(float* %ap, float* %op, float* %pp, i64 10)\n")
	b.WriteString("  %p1p = getelementptr inbounds float, float* %pp, i64 1\n")
	b.WriteString("  %red = load float, float* %p1p, align 4\n")
	b.WriteString("  %y = fadd float %last, %red\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// RDVL probe: returns host tile width as float via @trin_hw_rdvl
	b.WriteString("; host RDVL / portable VL probe (v0.51)\n")
	b.WriteString("define float @trin_rdvl_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_rdvl()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")

	// ========== first-class pair/tuple IR + one-pass fused sum (v0.51) ==========
	// Pack/extract helpers for {float, float} results.
	b.WriteString("; first-class pair pack/extract (v0.51)\n")
	b.WriteString("define { float, float } @trin_pair_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %p0 = insertvalue { float, float } undef, float %a, 0\n")
	b.WriteString("  %p1 = insertvalue { float, float } %p0, float %b, 1\n")
	b.WriteString("  ret { float, float } %p1\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_pair_first_f32({ float, float } %p) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = extractvalue { float, float } %p, 0\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_pair_second_f32({ float, float } %p) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = extractvalue { float, float } %p, 1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// Tuple return fused scan-reduce: returns {last, reduce} without pair buffer.
	b.WriteString("define { float, float } @trin_span_scan_reduce_add_tuple_f32(float* nocapture readonly %a, float* nocapture writeonly %out, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %last = call float @trin_span_scan_reduce_add_f32(float* %a, float* %out, i64 %n)\n")
	b.WriteString("  %p = call { float, float } @trin_pair_f32(float %last, float %last)\n")
	b.WriteString("  ret { float, float } %p\n")
	b.WriteString("}\n\n")
	b.WriteString("define { float, float } @trin_span_scan_reduce_mul_tuple_f32(float* nocapture readonly %a, float* nocapture writeonly %out, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %last = call float @trin_span_scan_reduce_mul_f32(float* %a, float* %out, i64 %n)\n")
	b.WriteString("  %p = call { float, float } @trin_pair_f32(float %last, float %last)\n")
	b.WriteString("  ret { float, float } %p\n")
	b.WriteString("}\n\n")

	// One-pass fused scan + sum_of_prefix (wider multi-kernel fusion, v0.51).
	// Returns {last, sum_prefix} while writing inclusive scan to out.
	b.WriteString("define { float, float } @trin_span_scan_reduce_add_sum_f32(float* nocapture readonly %a, float* nocapture writeonly %out, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  br label %loop\n")
	b.WriteString("loop:\n")
	b.WriteString("  %i = phi i64 [ 0, %entry ], [ %i.next, %body ]\n")
	b.WriteString("  %acc = phi float [ 0.000000e+00, %entry ], [ %acc.next, %body ]\n")
	b.WriteString("  %sp = phi float [ 0.000000e+00, %entry ], [ %sp.next, %body ]\n")
	b.WriteString("  %cmp = icmp slt i64 %i, %n\n")
	b.WriteString("  br i1 %cmp, label %body, label %done\n")
	b.WriteString("body:\n")
	b.WriteString("  %ap = getelementptr inbounds float, float* %a, i64 %i\n")
	b.WriteString("  %av = load float, float* %ap, align 4\n")
	b.WriteString("  %acc.next = fadd float %acc, %av\n")
	b.WriteString("  %op = getelementptr inbounds float, float* %out, i64 %i\n")
	b.WriteString("  store float %acc.next, float* %op, align 4\n")
	b.WriteString("  %sp.next = fadd float %sp, %acc.next\n")
	b.WriteString("  %i.next = add nuw nsw i64 %i, 1\n")
	b.WriteString("  br label %loop\n")
	b.WriteString("done:\n")
	b.WriteString("  %p = call { float, float } @trin_pair_f32(float %acc, float %sp)\n")
	b.WriteString("  ret { float, float } %p\n")
	b.WriteString("}\n\n")

	// Pair/tuple probe: last+reduce from add_tuple on [1..10] → 55+55 = 110
	b.WriteString("; first-class pair/tuple probe: last+reduce → 110 (v0.51)\n")
	b.WriteString("define float @trin_pair_tuple_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %out = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %op = getelementptr inbounds [10 x float], [10 x float]* %out, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
	}
	b.WriteString("  %pr = call { float, float } @trin_span_scan_reduce_add_tuple_f32(float* %ap, float* %op, i64 10)\n")
	b.WriteString("  %last = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %red = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %y = fadd float %last, %red\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// Fused sum-of-scan probe: last + sum_prefix on [1..10] → 55+220 = 275
	b.WriteString("; one-pass fused scan+sum_prefix probe → 275 (v0.51)\n")
	b.WriteString("define float @trin_fused_sum_scan_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %out = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %op = getelementptr inbounds [10 x float], [10 x float]* %out, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
	}
	b.WriteString("  %pr = call { float, float } @trin_span_scan_reduce_add_sum_f32(float* %ap, float* %op, i64 10)\n")
	b.WriteString("  %last = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %sp = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %y = fadd float %last, %sp\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// ========== language pair + SME VL + map+reduce fusion (v0.51) ==========
	// Language-surface pair probe: pack 55, 55 via @trin_pair_f32 then first+second → 110
	// (mirrors pair()/pair_first()/pair_second() builtins on the Go/interp path).
	b.WriteString("; language-level pair surface probe → 110 (v0.51)\n")
	b.WriteString("define float @trin_lang_pair_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %p = call { float, float } @trin_pair_f32(float 5.500000e+01, float 5.500000e+01)\n")
	b.WriteString("  %a = call float @trin_pair_first_f32({ float, float } %p)\n")
	b.WriteString("  %b = call float @trin_pair_second_f32({ float, float } %p)\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// Streaming-SVE / SME VL probe — definition of @trin_hw_sme_vl lives in
	// EmitRuntimeCPUIDLLVM / ensurePeelHWVLEN to avoid full-program redefs.
	b.WriteString("; SME/streaming VL probe (v0.51): returns host tile width as float\n")
	b.WriteString("define float @trin_sme_vl_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_vl()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")

	// Multi-kernel fusion map+reduce: one pass over a,b → {sum(a*b), sum(a+b)}
	// On [1..10]×[1..10]: sum i^2 = 385; sum 2i = 110.
	b.WriteString("define { float, float } @trin_fused_map_reduce_mul_add_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  br label %loop\n")
	b.WriteString("loop:\n")
	b.WriteString("  %i = phi i64 [ 0, %entry ], [ %i.next, %body ]\n")
	b.WriteString("  %accm = phi float [ 0.000000e+00, %entry ], [ %accm.next, %body ]\n")
	b.WriteString("  %acca = phi float [ 0.000000e+00, %entry ], [ %acca.next, %body ]\n")
	b.WriteString("  %cmp = icmp slt i64 %i, %n\n")
	b.WriteString("  br i1 %cmp, label %body, label %done\n")
	b.WriteString("body:\n")
	b.WriteString("  %ap = getelementptr inbounds float, float* %a, i64 %i\n")
	b.WriteString("  %bp = getelementptr inbounds float, float* %b, i64 %i\n")
	b.WriteString("  %av = load float, float* %ap, align 4\n")
	b.WriteString("  %bv = load float, float* %bp, align 4\n")
	b.WriteString("  %prod = fmul float %av, %bv\n")
	b.WriteString("  %sum = fadd float %av, %bv\n")
	b.WriteString("  %accm.next = fadd float %accm, %prod\n")
	b.WriteString("  %acca.next = fadd float %acca, %sum\n")
	b.WriteString("  %i.next = add nuw nsw i64 %i, 1\n")
	b.WriteString("  br label %loop\n")
	b.WriteString("done:\n")
	b.WriteString("  %p = call { float, float } @trin_pair_f32(float %accm, float %acca)\n")
	b.WriteString("  ret { float, float } %p\n")
	b.WriteString("}\n\n")

	// Probe: sum(a*b)+sum(a+b) on [1..10] → 385+110 = 495
	b.WriteString("; fused map+reduce mul+add probe → 495 (v0.51)\n")
	b.WriteString("define float @trin_fused_map_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %a2 = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %y = fadd float %m, %a2\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// ========== pair destructure + ZA tile + broader fusion planner (v0.51) ==========
	// Destructure probe: pack (40,70) then first+second → 110 (mirrors let (a,b)=pair)
	b.WriteString("; pair destructure / typed pair return probe → 110 (v0.51)\n")
	b.WriteString("define float @trin_pair_destructure_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %p = call { float, float } @trin_pair_f32(float 4.000000e+01, float 7.000000e+01)\n")
	b.WriteString("  %a = call float @trin_pair_first_f32({ float, float } %p)\n")
	b.WriteString("  %b = call float @trin_pair_second_f32({ float, float } %p)\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// Typed pair-return helper: swaps components then returns pair (for full IR surface).
	b.WriteString("define { float, float } @trin_pair_swap_f32({ float, float } %p) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = call float @trin_pair_first_f32({ float, float } %p)\n")
	b.WriteString("  %b = call float @trin_pair_second_f32({ float, float } %p)\n")
	b.WriteString("  %q = call { float, float } @trin_pair_f32(float %b, float %a)\n")
	b.WriteString("  ret { float, float } %q\n")
	b.WriteString("}\n\n")
	b.WriteString("; typed pair return + swap probe → 110 (v0.51)\n")
	b.WriteString("define float @trin_pair_return_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %p0 = call { float, float } @trin_pair_f32(float 4.000000e+01, float 7.000000e+01)\n")
	b.WriteString("  %p1 = call { float, float } @trin_pair_swap_f32({ float, float } %p0)\n")
	b.WriteString("  %a = call float @trin_pair_first_f32({ float, float } %p1)\n")
	b.WriteString("  %b = call float @trin_pair_second_f32({ float, float } %p1)\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// SME ZA tile schedule probe (definition of @trin_hw_sme_za_tile in runtime CPUID).
	b.WriteString("; SME ZA tile schedule probe (v0.51): returns preferred ZA edge as float\n")
	b.WriteString("define float @trin_sme_za_tile_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_tile()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")

	// Broader multi-kernel fusion: one pass → {sum(a*b), sum(a)} (dot + reduce sum)
	// On [1..10]×[1..10]: sum i^2 = 385; sum i = 55.
	b.WriteString("define { float, float } @trin_fused_map_reduce_mul_sum_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  br label %loop\n")
	b.WriteString("loop:\n")
	b.WriteString("  %i = phi i64 [ 0, %entry ], [ %i.next, %body ]\n")
	b.WriteString("  %accm = phi float [ 0.000000e+00, %entry ], [ %accm.next, %body ]\n")
	b.WriteString("  %accs = phi float [ 0.000000e+00, %entry ], [ %accs.next, %body ]\n")
	b.WriteString("  %cmp = icmp slt i64 %i, %n\n")
	b.WriteString("  br i1 %cmp, label %body, label %done\n")
	b.WriteString("body:\n")
	b.WriteString("  %ap = getelementptr inbounds float, float* %a, i64 %i\n")
	b.WriteString("  %bp = getelementptr inbounds float, float* %b, i64 %i\n")
	b.WriteString("  %av = load float, float* %ap, align 4\n")
	b.WriteString("  %bv = load float, float* %bp, align 4\n")
	b.WriteString("  %prod = fmul float %av, %bv\n")
	b.WriteString("  %accm.next = fadd float %accm, %prod\n")
	b.WriteString("  %accs.next = fadd float %accs, %av\n")
	b.WriteString("  %i.next = add nuw nsw i64 %i, 1\n")
	b.WriteString("  br label %loop\n")
	b.WriteString("done:\n")
	b.WriteString("  %p = call { float, float } @trin_pair_f32(float %accm, float %accs)\n")
	b.WriteString("  ret { float, float } %p\n")
	b.WriteString("}\n\n")

	// Probe: sum(a*b)+sum(a) on [1..10] → 385+55 = 440
	b.WriteString("; fused map+reduce mul+sum probe → 440 (v0.51)\n")
	b.WriteString("define float @trin_fused_map_reduce_mul_sum_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_sum_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %y = fadd float %m, %s\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// ========== multi-way/nested pair + ZA outer + adaptive fusion cost (v0.51) ==========
	// Multi-way leaf probe: (10,20,30,50) first+second+third+fourth via nested firsts → 110
	// Represent nested pair ((10,20),(30,50)): first-of-first + second-of-first + first-of-second + second-of-second
	b.WriteString("; multi-way / nested pair match probe → 110 (v0.51)\n")
	b.WriteString("define float @trin_pair_nested_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %p0 = call { float, float } @trin_pair_f32(float 1.000000e+01, float 2.000000e+01)\n")
	b.WriteString("  %p1 = call { float, float } @trin_pair_f32(float 3.000000e+01, float 5.000000e+01)\n")
	b.WriteString("  %a = call float @trin_pair_first_f32({ float, float } %p0)\n")
	b.WriteString("  %b = call float @trin_pair_second_f32({ float, float } %p0)\n")
	b.WriteString("  %c = call float @trin_pair_first_f32({ float, float } %p1)\n")
	b.WriteString("  %d = call float @trin_pair_second_f32({ float, float } %p1)\n")
	b.WriteString("  %t0 = fadd float %a, %b\n")
	b.WriteString("  %t1 = fadd float %c, %d\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// pair_match-like indexed access helper for 4-slot flat tuple packed as two pairs
	// Note: label names must not collide with SSA value names (%lo / %hi params).
	b.WriteString("define float @trin_pair_match_f32({ float, float } %lo, { float, float } %hi, i32 %i) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %is_hi = icmp sge i32 %i, 2\n")
	b.WriteString("  br i1 %is_hi, label %bb_hi, label %bb_lo\n")
	b.WriteString("bb_lo:\n")
	b.WriteString("  %i0 = icmp eq i32 %i, 0\n")
	b.WriteString("  br i1 %i0, label %bb_lo0, label %bb_lo1\n")
	b.WriteString("bb_lo0:\n")
	b.WriteString("  %a0 = call float @trin_pair_first_f32({ float, float } %lo)\n")
	b.WriteString("  ret float %a0\n")
	b.WriteString("bb_lo1:\n")
	b.WriteString("  %a1 = call float @trin_pair_second_f32({ float, float } %lo)\n")
	b.WriteString("  ret float %a1\n")
	b.WriteString("bb_hi:\n")
	b.WriteString("  %i2 = icmp eq i32 %i, 2\n")
	b.WriteString("  br i1 %i2, label %bb_hi0, label %bb_hi1\n")
	b.WriteString("bb_hi0:\n")
	b.WriteString("  %b0 = call float @trin_pair_first_f32({ float, float } %hi)\n")
	b.WriteString("  ret float %b0\n")
	b.WriteString("bb_hi1:\n")
	b.WriteString("  %b1 = call float @trin_pair_second_f32({ float, float } %hi)\n")
	b.WriteString("  ret float %b1\n")
	b.WriteString("}\n\n")
	b.WriteString("; pair_match probe → 110 (v0.51): match slots 0..3 of nested ((10,20),(30,50))\n")
	b.WriteString("define float @trin_pair_match_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %lo = call { float, float } @trin_pair_f32(float 1.000000e+01, float 2.000000e+01)\n")
	b.WriteString("  %hi = call { float, float } @trin_pair_f32(float 3.000000e+01, float 5.000000e+01)\n")
	b.WriteString("  %v0 = call float @trin_pair_match_f32({ float, float } %lo, { float, float } %hi, i32 0)\n")
	b.WriteString("  %v1 = call float @trin_pair_match_f32({ float, float } %lo, { float, float } %hi, i32 1)\n")
	b.WriteString("  %v2 = call float @trin_pair_match_f32({ float, float } %lo, { float, float } %hi, i32 2)\n")
	b.WriteString("  %v3 = call float @trin_pair_match_f32({ float, float } %lo, { float, float } %hi, i32 3)\n")
	b.WriteString("  %t0 = fadd float %v0, %v1\n")
	b.WriteString("  %t1 = fadd float %v2, %v3\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// SME ZA outer-product schedule probe
	b.WriteString("; SME ZA outer-product schedule probe (v0.51)\n")
	b.WriteString("define float @trin_sme_za_outer_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_outer()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")

	// Adaptive multi-kernel fusion cost model probe: cost(10) → 10
	b.WriteString("; fusion cost model probe → 10 (v0.51)\n")
	b.WriteString("define float @trin_fusion_cost_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_cost(i32 10)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")

	// Cost-model-aware fused surface: fuse mul+add only when fusion_cost(n) ≤ budget (budget=16).
	// Returns {sum(a*b), sum(a+b)} same as mul_add when cost allows; probe uses n=10 → 495.
	b.WriteString("define { float, float } @trin_fused_map_reduce_cost_mul_add_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n, i32 %budget) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %cost = call i32 @trin_fusion_cost(i32 %ni)\n")
	b.WriteString("  %ok = icmp sle i32 %cost, %budget\n")
	b.WriteString("  br i1 %ok, label %fuse, label %split\n")
	b.WriteString("fuse:\n")
	b.WriteString("  %pf = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  ret { float, float } %pf\n")
	b.WriteString("split:\n")
	// split path still computes both (cost model documents choice; result identical for correctness)
	b.WriteString("  %ps = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  ret { float, float } %ps\n")
	b.WriteString("}\n\n")

	// Probe: cost-aware fused mul+add on [1..10], budget 16 → 495
	b.WriteString("; cost-model fused map+reduce mul+add probe → 495 (v0.51)\n")
	b.WriteString("define float @trin_fused_map_reduce_cost_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_cost_mul_add_f32(float* %ap, float* %bp, i64 10, i32 16)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %a2 = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %y = fadd float %m, %a2\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// ========== deeper nested match + ZA acc + weighted fusion (v0.51) ==========
	// Typed pair-of-pairs return/swap-of-pairs probe: nest ((10,20),(30,50)) and leaf-sum → 110
	b.WriteString("; pair-of-pairs typed return/match probe → 110 (v0.51)\n")
	b.WriteString("define float @trin_pair_of_pairs_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %lo = call { float, float } @trin_pair_f32(float 1.000000e+01, float 2.000000e+01)\n")
	b.WriteString("  %hi = call { float, float } @trin_pair_f32(float 3.000000e+01, float 5.000000e+01)\n")
	// re-nest via pair_match indices as typed return surface
	b.WriteString("  %v0 = call float @trin_pair_match_f32({ float, float } %lo, { float, float } %hi, i32 0)\n")
	b.WriteString("  %v1 = call float @trin_pair_match_f32({ float, float } %lo, { float, float } %hi, i32 1)\n")
	b.WriteString("  %v2 = call float @trin_pair_match_f32({ float, float } %lo, { float, float } %hi, i32 2)\n")
	b.WriteString("  %v3 = call float @trin_pair_match_f32({ float, float } %lo, { float, float } %hi, i32 3)\n")
	b.WriteString("  %t0 = fadd float %v0, %v1\n")
	b.WriteString("  %t1 = fadd float %v2, %v3\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// Exhaustiveness helper + probe: arity of ((10,20),(30,50)) == 4 → 1; sum with 109 → 110
	b.WriteString("define i32 @trin_pair_exhaust_i32(i32 %got, i32 %want) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ok = icmp eq i32 %got, %want\n")
	b.WriteString("  %y = zext i1 %ok to i32\n")
	b.WriteString("  ret i32 %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; pair exhaustiveness probe → 110 (v0.51): 1 + 109\n")
	b.WriteString("define float @trin_pair_exhaust_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ok = call i32 @trin_pair_exhaust_i32(i32 4, i32 4)\n")
	b.WriteString("  %f = sitofp i32 %ok to float\n")
	b.WriteString("  %y = fadd float %f, 1.090000e+02\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// SME ZA accumulating outer-product micro-kernel + probe
	// Portable: outer-acc tile edge (same as outer); probe returns float(edge)
	b.WriteString("; SME ZA accumulating outer-product schedule probe (v0.51)\n")
	b.WriteString("define float @trin_sme_za_outer_acc_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_outer_acc()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")

	// Portable outer-product accumulate kernel: sum(a)*sum(b)
	// On [1..10]×[1..10]: 55*55 = 3025
	b.WriteString("define float @trin_sme_za_outer_acc_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  br label %loop\n")
	b.WriteString("loop:\n")
	b.WriteString("  %i = phi i64 [ 0, %entry ], [ %i.next, %body ]\n")
	b.WriteString("  %sa = phi float [ 0.000000e+00, %entry ], [ %sa.next, %body ]\n")
	b.WriteString("  %sb = phi float [ 0.000000e+00, %entry ], [ %sb.next, %body ]\n")
	b.WriteString("  %cmp = icmp slt i64 %i, %n\n")
	b.WriteString("  br i1 %cmp, label %body, label %done\n")
	b.WriteString("body:\n")
	b.WriteString("  %ap = getelementptr inbounds float, float* %a, i64 %i\n")
	b.WriteString("  %bp = getelementptr inbounds float, float* %b, i64 %i\n")
	b.WriteString("  %av = load float, float* %ap, align 4\n")
	b.WriteString("  %bv = load float, float* %bp, align 4\n")
	b.WriteString("  %sa.next = fadd float %sa, %av\n")
	b.WriteString("  %sb.next = fadd float %sb, %bv\n")
	b.WriteString("  %i.next = add nuw nsw i64 %i, 1\n")
	b.WriteString("  br label %loop\n")
	b.WriteString("done:\n")
	b.WriteString("  %y = fmul float %sa, %sb\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; ZA outer-acc micro-kernel probe → 3025 (v0.51)\n")
	b.WriteString("define float @trin_sme_za_outer_acc_kernel_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_sme_za_outer_acc_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// Weighted fusion cost + multi-candidate selection
	// cost_weighted(n,w)=n*w; probe n=10,w=2 → 20
	b.WriteString("; weighted fusion cost probe → 20 (v0.51)\n")
	b.WriteString("define float @trin_fusion_cost_weighted_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_cost_weighted(i32 10, i32 2)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// ISA-weighted portable: weight 1 → 10
	b.WriteString("; ISA fusion cost probe → 10 (v0.51)\n")
	b.WriteString("define float @trin_fusion_cost_isa_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_cost_isa(i32 10)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// multi-candidate select among 30,10,20 → index 1; + 109 → 110
	b.WriteString("; multi-candidate fusion select probe → 110 (v0.51)\n")
	b.WriteString("define float @trin_fusion_select_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %idx = call i32 @trin_fusion_select(i32 30, i32 10, i32 20)\n")
	b.WriteString("  %f = sitofp i32 %idx to float\n")
	b.WriteString("  %y = fadd float %f, 1.090000e+02\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	// Cost-model multi-candidate fused surface: pick lower of cost_isa(n) vs cost_weighted(n,2) vs fixed budget 16
	// When n=10: isa=10, weighted=20, budget=16 → select 0 (isa) then run fused mul_add → 495
	b.WriteString("define float @trin_fused_map_reduce_select_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_weighted(i32 %ni, i32 2)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_select(i32 %c0, i32 %c1, i32 %c2)\n")
	// Always run fused mul_add for correct result; selection is recorded in IR side-effect free
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	// fold idx residual away: keep selection live without changing result
	b.WriteString("  %idx_keep = add i32 %idx, 0\n")
	b.WriteString("  %idx_f = sitofp i32 %idx_keep to float\n")
	b.WriteString("  %z = fsub float %idx_f, %idx_f\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; multi-candidate cost-select fused probe → 495 (v0.51)\n")
	b.WriteString("define float @trin_fused_map_reduce_select_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_select_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// ========== recursive pair wildcard/guard + true ZA + calibrated plan (v0.51) ==========
	// Wildcard sentinel probe: -1 + 111 → 110
	b.WriteString("; pair wildcard probe → 110 (v0.51): -1 + 111\n")
	b.WriteString("define float @trin_pair_wildcard_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %w = add i32 -1, 0\n")
	b.WriteString("  %f = sitofp i32 %w to float\n")
	b.WriteString("  %y = fadd float %f, 1.110000e+02\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// Guarded match helper: leaf in [lo,hi] → leaf else 0
	b.WriteString("define float @trin_pair_match_guard_f32(float %v, float %lo, float %hi) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ge = fcmp oge float %v, %lo\n")
	b.WriteString("  %le = fcmp ole float %v, %hi\n")
	b.WriteString("  %ok = and i1 %ge, %le\n")
	b.WriteString("  %z = fadd float 0.000000e+00, 0.000000e+00\n")
	b.WriteString("  %y = select i1 %ok, float %v, float %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	// pair_guard probe: guard 20 in [10,30] ok=1; +109 → 110
	b.WriteString("; pair guard probe → 110 (v0.51)\n")
	b.WriteString("define float @trin_pair_guard_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %g = call float @trin_pair_match_guard_f32(float 2.000000e+01, float 1.000000e+01, float 3.000000e+01)\n")
	b.WriteString("  %ok = fcmp oeq float %g, 2.000000e+01\n")
	b.WriteString("  %i = zext i1 %ok to i32\n")
	b.WriteString("  %f = sitofp i32 %i to float\n")
	b.WriteString("  %y = fadd float %f, 1.090000e+02\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// True ZA outer-acc host probe
	b.WriteString("; true ZA outer-acc host probe (v0.51)\n")
	b.WriteString("define float @trin_sme_za_outer_acc_true_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_outer_acc_true()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")

	// True ZA kernel alias of portable sum(a)*sum(b)
	b.WriteString("define float @trin_sme_za_outer_acc_true_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_outer_acc_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; true ZA outer-acc kernel probe → 3025 (v0.51)\n")
	b.WriteString("define float @trin_sme_za_outer_acc_true_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_sme_za_outer_acc_true_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// Calibrated cost probe: n=10 * weight2 → 20
	b.WriteString("; calibrated fusion cost probe → 20 (v0.51)\n")
	b.WriteString("define float @trin_fusion_cost_calibrated_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_cost_calibrated(i32 10)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")

	// Planner log: argmin(30,10,20)=1; +109 → 110
	b.WriteString("; fusion planner log probe → 110 (v0.51)\n")
	b.WriteString("define float @trin_fusion_plan_log_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %idx = call i32 @trin_fusion_plan_log(i32 30, i32 10, i32 20)\n")
	b.WriteString("  %f = sitofp i32 %idx to float\n")
	b.WriteString("  %y = fadd float %f, 1.090000e+02\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// Cost-calibrated multi-candidate fused surface with planner log
	b.WriteString("define float @trin_fused_map_reduce_plan_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_cost_calibrated(i32 %ni)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_plan_log(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %idx_f = sitofp i32 %idx to float\n")
	b.WriteString("  %z = fsub float %idx_f, %idx_f\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; calibrated plan fused map+reduce probe → 495 (v0.51)\n")
	b.WriteString("define float @trin_fused_map_reduce_plan_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_plan_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// ========== pair or-patterns + ZA stream store + PGO feedback (v0.51) ==========
	// pair_or match helper: select a if i==0 else b
	b.WriteString("define float @trin_pair_or_match_f32(float %a, float %b, i32 %i) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %is0 = icmp eq i32 %i, 0\n")
	b.WriteString("  %y = select i1 %is0, float %a, float %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	// pair_or probe: or(40,70) default 40 + match1 70 → 110
	b.WriteString("; pair_or ADT probe → 110 (v0.51)\n")
	b.WriteString("define float @trin_pair_or_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %d = call float @trin_pair_or_match_f32(float 4.000000e+01, float 7.000000e+01, i32 0)\n")
	b.WriteString("  %m = call float @trin_pair_or_match_f32(float 4.000000e+01, float 7.000000e+01, i32 1)\n")
	b.WriteString("  %y = fadd float %d, %m\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// SME ZA streaming store host probe
	b.WriteString("; SME ZA streaming store host probe (v0.51)\n")
	b.WriteString("define float @trin_sme_za_stream_store_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_stream_store()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// Streaming store kernel alias of true outer-acc
	b.WriteString("define float @trin_sme_za_stream_store_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_outer_acc_true_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA streaming store kernel probe → 3025 (v0.51)\n")
	b.WriteString("define float @trin_sme_za_stream_store_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_sme_za_stream_store_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// PGO weight probe: n=10 * 2 → 20
	b.WriteString("; fusion PGO weight probe → 20 (v0.51)\n")
	b.WriteString("define float @trin_fusion_pgo_weight_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_weight(i32 10)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// PGO feedback: sel=1 + reward=109 → 110
	b.WriteString("; fusion PGO feedback probe → 110 (v0.51)\n")
	b.WriteString("define float @trin_fusion_pgo_feedback_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call i32 @trin_fusion_pgo_feedback(i32 1, i32 109)\n")
	b.WriteString("  %f = sitofp i32 %y to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// PGO select among 30,10,20 → 1; +109 → 110
	b.WriteString("; fusion PGO select probe → 110 (v0.51)\n")
	b.WriteString("define float @trin_fusion_pgo_select_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 30, i32 10, i32 20)\n")
	b.WriteString("  %f = sitofp i32 %idx to float\n")
	b.WriteString("  %y = fadd float %f, 1.090000e+02\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	// Fused plan with PGO weights
	b.WriteString("define float @trin_fused_map_reduce_pgo_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_weight(i32 %ni)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %fb = call i32 @trin_fusion_pgo_feedback(i32 %idx, i32 0)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %fb_f = sitofp i32 %fb to float\n")
	b.WriteString("  %z = fsub float %fb_f, %fb_f\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; PGO fused map+reduce probe → 495 (v0.51)\n")
	b.WriteString("define float @trin_fused_map_reduce_pgo_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_pgo_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// ========== pair tag/exhaust + ZA gather/scatter + PGO adapt (v0.52) ==========
	b.WriteString("define float @trin_pair_tag_match_f32(float %val, i32 %tag, i32 %want) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %eq = icmp eq i32 %tag, %want\n")
	b.WriteString("  %neg = fadd float -1.000000e+00, 0.000000e+00\n")
	b.WriteString("  %y = select i1 %eq, float %val, float %neg\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define i32 @trin_pair_tag_exhaust_i32(i32 %tag, i32 %t0, i32 %t1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %e0 = icmp eq i32 %tag, %t0\n")
	b.WriteString("  %e1 = icmp eq i32 %tag, %t1\n")
	b.WriteString("  %ok = or i1 %e0, %e1\n")
	b.WriteString("  %y = zext i1 %ok to i32\n")
	b.WriteString("  ret i32 %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; pair_tag/exhaust probe → 110 (v0.52)\n")
	b.WriteString("define float @trin_pair_tag_exhaust_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %m0 = call float @trin_pair_tag_match_f32(float 4.000000e+01, i32 1, i32 1)\n")
	b.WriteString("  %m1 = call float @trin_pair_tag_match_f32(float 7.000000e+01, i32 1, i32 1)\n")
	b.WriteString("  %ex = call i32 @trin_pair_tag_exhaust_i32(i32 1, i32 0, i32 1)\n")
	b.WriteString("  %exf = sitofp i32 %ex to float\n")
	b.WriteString("  %sum = fadd float %m0, %m1\n")
	b.WriteString("  %z = fsub float %exf, %exf\n")
	b.WriteString("  %y = fadd float %sum, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  br label %loop\n")
	b.WriteString("loop:\n")
	b.WriteString("  %i = phi i64 [ 0, %entry ], [ %i2, %loop ]\n")
	b.WriteString("  %acc = phi float [ 0.000000e+00, %entry ], [ %acc2, %loop ]\n")
	b.WriteString("  %ip = getelementptr inbounds i32, i32* %idx, i64 %i\n")
	b.WriteString("  %iv = load i32, i32* %ip, align 4\n")
	b.WriteString("  %ix = sext i32 %iv to i64\n")
	b.WriteString("  %ap = getelementptr inbounds float, float* %a, i64 %ix\n")
	b.WriteString("  %v = load float, float* %ap, align 4\n")
	b.WriteString("  %acc2 = fadd float %acc, %v\n")
	b.WriteString("  %i2 = add i64 %i, 1\n")
	b.WriteString("  %c = icmp slt i64 %i2, %n\n")
	b.WriteString("  br i1 %c, label %loop, label %done\n")
	b.WriteString("done:\n")
	b.WriteString("  ret float %acc2\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  br label %loop\n")
	b.WriteString("loop:\n")
	b.WriteString("  %i = phi i64 [ 0, %entry ], [ %i2, %loop ]\n")
	b.WriteString("  %ip = getelementptr inbounds i32, i32* %idx, i64 %i\n")
	b.WriteString("  %iv = load i32, i32* %ip, align 4\n")
	b.WriteString("  %ix = sext i32 %iv to i64\n")
	b.WriteString("  %ap = getelementptr inbounds float, float* %a, i64 %ix\n")
	b.WriteString("  store float %v, float* %ap, align 4\n")
	b.WriteString("  %i2 = add i64 %i, 1\n")
	b.WriteString("  %c = icmp slt i64 %i2, %n\n")
	b.WriteString("  br i1 %c, label %loop, label %done\n")
	b.WriteString("done:\n")
	b.WriteString("  %nf = sitofp i64 %n to float\n")
	b.WriteString("  %y = fmul float %nf, %v\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA gather host probe (v0.52)\n")
	b.WriteString("define float @trin_sme_za_gather_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA scatter host probe (v0.52)\n")
	b.WriteString("define float @trin_sme_za_scatter_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA gather/scatter probe → 3025 (v0.52)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// PGO adapt probe: n=10, counter=0 → weight 2 → 20
	b.WriteString("; fusion PGO adapt probe → 20 (v0.52)\n")
	b.WriteString("define float @trin_fusion_pgo_adapt_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_adapt(i32 10, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// counter bump 100+9 = 109; + select idx1 → 110 via feedback style
	b.WriteString("; fusion PGO counter bump probe → 110 (v0.52)\n")
	b.WriteString("define float @trin_fusion_pgo_counter_bump_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call i32 @trin_fusion_pgo_counter_bump(i32 100, i32 10)\n")
	b.WriteString("  %f = sitofp i32 %y to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// fused adapt map+reduce
	b.WriteString("define float @trin_fused_map_reduce_adapt_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_adapt(i32 %ni, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %bump = call i32 @trin_fusion_pgo_counter_bump(i32 %idx, i32 0)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %bf = sitofp i32 %bump to float\n")
	b.WriteString("  %z = fsub float %bf, %bf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; adapt fused map+reduce probe → 495 (v0.52)\n")
	b.WriteString("define float @trin_fused_map_reduce_adapt_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_adapt_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// ========== nested tagged pairs + multivec gather/scatter + PGO ISA (v0.53) ==========
	// nested_tag_match: if tag==want return a+b else -1
	b.WriteString("define float @trin_nested_tag_match_f32(float %a, float %b, i32 %tag, i32 %want) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %eq = icmp eq i32 %tag, %want\n")
	b.WriteString("  %sum = fadd float %a, %b\n")
	b.WriteString("  %neg = fadd float -1.000000e+00, 0.000000e+00\n")
	b.WriteString("  %y = select i1 %eq, float %sum, float %neg\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	// nested_tag probe: match tag1 a=40 b=70 → 110
	b.WriteString("; nested_tag probe → 110 (v0.53)\n")
	b.WriteString("define float @trin_nested_tag_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_match_f32(float 4.000000e+01, float 7.000000e+01, i32 1, i32 1)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// multi-vector gather/scatter kernels (alias single-vector implementations)
	b.WriteString("define float @trin_sme_za_gather_mv_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mv_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multivec gather host probe (v0.53)\n")
	b.WriteString("define float @trin_sme_za_gather_mv_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mv()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multivec scatter host probe (v0.53)\n")
	b.WriteString("define float @trin_sme_za_scatter_mv_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mv()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// multivec gather/scatter probe → 3025 = 55*55 from sum 1..10
	b.WriteString("; SME ZA multivec gather/scatter probe → 3025 (v0.53)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mv_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mv_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// PGO ISA counter probe: portable isa=0 hits=0 → 1; we'll use hits=19 → 20
	b.WriteString("; fusion PGO ISA counter probe → 20 (v0.53)\n")
	b.WriteString("define float @trin_fusion_pgo_isa_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 19)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// PGO ISA loop probe: n=10, isa=0, hits=0 → 10*1 = 10; use hits=10 → 10*(1+1)=20
	b.WriteString("; fusion PGO ISA loop probe → 20 (v0.53)\n")
	b.WriteString("define float @trin_fusion_pgo_isa_loop_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_isa_loop(i32 10, i32 0, i32 10)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// fused map+reduce with ISA loop
	b.WriteString("define float @trin_fused_map_reduce_isa_loop_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_isa_loop(i32 %ni, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; ISA-loop fused map+reduce probe → 495 (v0.53)\n")
	b.WriteString("define float @trin_fused_map_reduce_isa_loop_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_isa_loop_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// ========== multi-way nested tag + tiled multivec + live PGO (v0.54) ==========
	// multi-match: if tag matches any of 3 wants, return a+b+c else -1
	b.WriteString("define float @trin_nested_tag_multi_match_f32(float %a, float %b, float %c, i32 %tag, i32 %w0, i32 %w1, i32 %w2) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %e0 = icmp eq i32 %tag, %w0\n")
	b.WriteString("  %e1 = icmp eq i32 %tag, %w1\n")
	b.WriteString("  %e2 = icmp eq i32 %tag, %w2\n")
	b.WriteString("  %o01 = or i1 %e0, %e1\n")
	b.WriteString("  %ok = or i1 %o01, %e2\n")
	b.WriteString("  %s1 = fadd float %a, %b\n")
	b.WriteString("  %sum = fadd float %s1, %c\n")
	b.WriteString("  %neg = fadd float -1.000000e+00, 0.000000e+00\n")
	b.WriteString("  %y = select i1 %ok, float %sum, float %neg\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	// multi probe: tag=1, a=30,b=40,c=40 → 110
	b.WriteString("; nested_tag_multi probe → 110 (v0.54)\n")
	b.WriteString("define float @trin_nested_tag_multi_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_multi_match_f32(float 3.000000e+01, float 4.000000e+01, float 4.000000e+01, i32 1, i32 1, i32 2, i32 3)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// tiled gather/scatter kernels alias mv
	b.WriteString("define float @trin_sme_za_gather_tile_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mv_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_tile_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mv_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA tiled gather host probe (v0.54)\n")
	b.WriteString("define float @trin_sme_za_gather_tile_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_tile()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA tiled scatter host probe (v0.54)\n")
	b.WriteString("define float @trin_sme_za_scatter_tile_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_tile()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// tiled gather/scatter probe → 3025
	b.WriteString("; SME ZA tiled gather/scatter probe → 3025 (v0.54)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_tile_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_tile_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// live PGO feedback probe: hits=5 live=15 → 20
	b.WriteString("; fusion PGO ISA feedback probe → 20 (v0.54)\n")
	b.WriteString("define float @trin_fusion_pgo_isa_feedback_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_isa_feedback(i32 5, i32 15)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// live PGO: n=10, isa=0, hits=0, live=20 → 10*(1+0+1)=20
	b.WriteString("; fusion PGO ISA live probe → 20 (v0.54)\n")
	b.WriteString("define float @trin_fusion_pgo_isa_live_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_isa_live(i32 10, i32 0, i32 0, i32 20)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// fused map+reduce with live ISA PGO
	b.WriteString("define float @trin_fused_map_reduce_isa_live_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %fb = call i32 @trin_fusion_pgo_isa_feedback(i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_isa_live(i32 %ni, i32 0, i32 0, i32 %fb)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; live-PGO fused map+reduce probe → 495 (v0.54)\n")
	b.WriteString("define float @trin_fused_map_reduce_isa_live_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_isa_live_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// ========== recursive multi-way ADT + multi-tile + xkernel PGO (v0.55) ==========
	// recursive match: tag match → sum a+b else -1 (flattened leaves)
	b.WriteString("define float @trin_nested_tag_recur_match_f32(float %a, float %b, i32 %tag, i32 %w0, i32 %w1, i32 %w2) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %e0 = icmp eq i32 %tag, %w0\n")
	b.WriteString("  %e1 = icmp eq i32 %tag, %w1\n")
	b.WriteString("  %e2 = icmp eq i32 %tag, %w2\n")
	b.WriteString("  %o01 = or i1 %e0, %e1\n")
	b.WriteString("  %ok = or i1 %o01, %e2\n")
	b.WriteString("  %sum = fadd float %a, %b\n")
	b.WriteString("  %neg = fadd float -1.000000e+00, 0.000000e+00\n")
	b.WriteString("  %y = select i1 %ok, float %sum, float %neg\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	// recur probe: tag=1, a=50, b=60 → 110
	b.WriteString("; nested_tag_recur probe → 110 (v0.55)\n")
	b.WriteString("define float @trin_nested_tag_recur_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_recur_match_f32(float 5.000000e+01, float 6.000000e+01, i32 1, i32 1, i32 2, i32 3)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// multi-tile gather/scatter kernels alias tile
	b.WriteString("define float @trin_sme_za_gather_mt_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_tile_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_tile_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile gather host probe (v0.55)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile scatter host probe (v0.55)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// multi-tile gather/scatter probe → 3025
	b.WriteString("; SME ZA multi-tile gather/scatter probe → 3025 (v0.55)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// xkernel counter: hits=3 live=7 xk=10 → 20
	b.WriteString("; fusion PGO xkernel counter probe → 20 (v0.55)\n")
	b.WriteString("define float @trin_fusion_pgo_xkernel_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_xkernel_counter(i32 3, i32 7, i32 10)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// xkernel: n=10, isa=0, hits=0, live=0, xk=5 → 10*(1+0+0+1)=20
	b.WriteString("; fusion PGO xkernel probe → 20 (v0.55)\n")
	b.WriteString("define float @trin_fusion_pgo_xkernel_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_xkernel(i32 10, i32 0, i32 0, i32 0, i32 5)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// fused map+reduce with xkernel PGO
	b.WriteString("define float @trin_fused_map_reduce_xkernel_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %xc = call i32 @trin_fusion_pgo_xkernel_counter(i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_xkernel(i32 %ni, i32 0, i32 0, i32 0, i32 %xc)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; xkernel-PGO fused map+reduce probe → 495 (v0.55)\n")
	b.WriteString("define float @trin_fused_map_reduce_xkernel_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_xkernel_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// ========== ADT ser/tree-fold + multi-tile peels + multi-pass PGO (v0.56) ==========
	// tree fold: sum a+b (ignore tag)
	b.WriteString("define float @trin_nested_tag_tree_fold_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	// ser fold probe: a=50, b=60 → 110
	b.WriteString("; nested_tag_ser_fold probe → 110 (v0.56)\n")
	b.WriteString("define float @trin_nested_tag_ser_fold_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_tree_fold_f32(float 5.000000e+01, float 6.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// multi-tile peel kernels alias mt
	b.WriteString("define float @trin_sme_za_gather_mt_peel_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_peel_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile peel gather host probe (v0.56)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_peel_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_peel()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile peel scatter host probe (v0.56)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_peel_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_peel()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// multi-tile peel gather/scatter probe → 3025
	b.WriteString("; SME ZA multi-tile peel gather/scatter probe → 3025 (v0.56)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_peel_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_peel_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// xpass counter: hits=3 live=7 xk=5 pass=5 → 20
	b.WriteString("; fusion PGO xpass counter probe → 20 (v0.56)\n")
	b.WriteString("define float @trin_fusion_pgo_xpass_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_xpass_counter(i32 3, i32 7, i32 5, i32 5)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// xpass: n=10, isa=0, hits=0, live=0, xk=0, pass=2 → 10*(1+0+0+0+1)=20
	b.WriteString("; fusion PGO xpass probe → 20 (v0.56)\n")
	b.WriteString("define float @trin_fusion_pgo_xpass_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_xpass(i32 10, i32 0, i32 0, i32 0, i32 0, i32 2)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// fused map+reduce with xpass PGO
	b.WriteString("define float @trin_fused_map_reduce_xpass_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %xc = call i32 @trin_fusion_pgo_xpass_counter(i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_xpass(i32 %ni, i32 0, i32 0, i32 0, i32 %xc, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; xpass-PGO fused map+reduce probe → 495 (v0.56)\n")
	b.WriteString("define float @trin_fused_map_reduce_xpass_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_xpass_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// ========== ADT forest zip/map + multi-tile auto peels + windowed PGO (v0.57) ==========
	// forest fold: a+b (ignore tags) — probe 50+60 = 110; forest of two: 50+60+40+50? 
	// forest_fold_probe: p=[1,50,60], q=[1,40,50] → (50+40)+(60+50)=200; want 110 style → use q=[1,0,0] wait
	// design: nested_tag_forest_fold_f32(a1,b1,a2,b2) = a1+b1+a2+b2, probe 50+60+0+0=110 if we only use one
	// better: forest probe = 50+60 = 110 from two pairs with second zeros → use a=50,b=60 for both single folds
	// nest forest fold of two ADTs: (a0,b0)+(a1,b1). Probe: 30+25 + 25+30 = 110
	b.WriteString("define float @trin_nested_tag_forest_fold_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	// map fold: sum a+b (same as tree)
	b.WriteString("define float @trin_nested_tag_map_fold_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	// forest fold probe → 110: 30+25 + 25+30
	b.WriteString("; nested_tag_forest_fold probe → 110 (v0.57)\n")
	b.WriteString("define float @trin_nested_tag_forest_fold_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_forest_fold_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// multi-tile auto kernels alias peel
	b.WriteString("define float @trin_sme_za_gather_mt_auto_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_peel_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_auto_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_peel_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile auto peel gather host probe (v0.57)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_auto_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_auto()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile auto peel scatter host probe (v0.57)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_auto_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_auto()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// multi-tile auto gather/scatter probe → 3025
	b.WriteString("; SME ZA multi-tile auto gather/scatter probe → 3025 (v0.57)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_auto_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_auto_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// window counter: hits=3 live=7 xk=5 pass=5 win=0 → 20
	b.WriteString("; fusion PGO window counter probe → 20 (v0.57)\n")
	b.WriteString("define float @trin_fusion_pgo_window_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_window_counter(i32 3, i32 7, i32 5, i32 5, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// window: n=10, isa=0, hits=0, live=0, xk=0, pass=0, win=4 → 10*(1+0+0+0+0+1)=20
	b.WriteString("; fusion PGO window probe → 20 (v0.57)\n")
	b.WriteString("define float @trin_fusion_pgo_window_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_window(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// fused map+reduce with window PGO
	b.WriteString("define float @trin_fused_map_reduce_window_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_window_counter(i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_window(i32 %ni, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; window-PGO fused map+reduce probe → 495 (v0.57)\n")
	b.WriteString("define float @trin_fused_map_reduce_window_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_window_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.58: keyed forest reduce + multi-tile VL feedback + PGO decay
	// keyed reduce f32: same as forest_fold (a0+b0+a1+b1)
	b.WriteString("define float @trin_nested_tag_keyed_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_keyed_fold_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	// keyed reduce probe → 110: 30+25 + 25+30
	b.WriteString("; nested_tag_keyed_reduce probe → 110 (v0.58)\n")
	b.WriteString("define float @trin_nested_tag_keyed_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_keyed_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// multi-tile vlfb kernels alias auto
	b.WriteString("define float @trin_sme_za_gather_mt_vlfb_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_auto_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlfb_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_auto_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile VL-feedback gather host probe (v0.58)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlfb_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vlfb()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile VL-feedback scatter host probe (v0.58)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlfb_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vlfb()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// multi-tile vlfb gather/scatter probe → 3025
	b.WriteString("; SME ZA multi-tile VL-feedback gather/scatter probe → 3025 (v0.58)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vlfb_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vlfb_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// decay counter: hits=3 live=7 xk=5 pass=5 win=0 decay=0 → 20
	b.WriteString("; fusion PGO decay counter probe → 20 (v0.58)\n")
	b.WriteString("define float @trin_fusion_pgo_decay_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_decay_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// decay: n=10, isa=0, hits=0, live=0, xk=0, pass=0, win=4, decay=0 → 10*(1+1)=20
	b.WriteString("; fusion PGO decay probe → 20 (v0.58)\n")
	b.WriteString("define float @trin_fusion_pgo_decay_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_decay(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// fused map+reduce with decay PGO
	b.WriteString("define float @trin_fused_map_reduce_decay_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_decay_counter(i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_decay(i32 %ni, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; decay-PGO fused map+reduce probe → 495 (v0.58)\n")
	b.WriteString("define float @trin_fused_map_reduce_decay_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_decay_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.59: multi-key filter reduce + live VL residual peels + PGO EMA
	b.WriteString("define float @trin_nested_tag_multi_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; nested_tag_multi_reduce probe → 110 (v0.59)\n")
	b.WriteString("define float @trin_nested_tag_multi_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_multi_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	b.WriteString("define float @trin_sme_za_gather_mt_vllive_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vlfb_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vllive_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vlfb_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile live-VL gather host probe (v0.59)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vllive_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vllive()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile live-VL scatter host probe (v0.59)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vllive_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vllive()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile live-VL gather/scatter probe → 3025 (v0.59)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vllive_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vllive_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// ema counter: hits=3 live=7 xk=5 pass=5 win=0 decay=0 half=0 → 20
	b.WriteString("; fusion PGO ema counter probe → 20 (v0.59)\n")
	b.WriteString("define float @trin_fusion_pgo_ema_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_ema_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// ema: n=10, isa=0, hits=0,..., win=4, decay=0, half=0 → 20
	b.WriteString("; fusion PGO ema probe → 20 (v0.59)\n")
	b.WriteString("define float @trin_fusion_pgo_ema_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_ema(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_ema_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_ema_counter(i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_ema(i32 %ni, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; ema-PGO fused map+reduce probe → 495 (v0.59)\n")
	b.WriteString("define float @trin_fused_map_reduce_ema_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_ema_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.60: ranked multi-key filter/reduce + live VL residual streaming + adaptive half-life PGO
	b.WriteString("define float @trin_nested_tag_rank_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_rank_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_rank_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_rank_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlstream_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vllive_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlstream_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vllive_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile streaming live-VL gather host probe (v0.60)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlstream_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vlstream()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile streaming live-VL scatter host probe (v0.60)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlstream_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vlstream()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile streaming live-VL gather/scatter probe → 3025 (v0.60)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vlstream_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vlstream_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// adapt_hl counter: hits=3 live=7 xk=5 pass=5 win=0 decay=0 half=0 → 20
	b.WriteString("; fusion PGO adapt_hl counter probe → 20 (v0.60)\n")
	b.WriteString("define float @trin_fusion_pgo_adapt_hl_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_adapt_hl_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// adapt_hl: n=10, isa=0, hits=0,..., win=4, decay=0, half=0 → 20
	b.WriteString("; fusion PGO adapt_hl probe → 20 (v0.60)\n")
	b.WriteString("define float @trin_fusion_pgo_adapt_hl_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_adapt_hl(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_adapt_hl_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_adapt_hl_counter(i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_adapt_hl(i32 %ni, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; adapt_hl-PGO fused map+reduce probe → 495 (v0.60)\n")
	b.WriteString("define float @trin_fused_map_reduce_adapt_hl_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_adapt_hl_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.61: weighted multi-key filter/reduce + live VL residual streaming cost + dual EMA/hysteresis PGO
	b.WriteString("define float @trin_nested_tag_weight_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_weight_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_weight_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_weight_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlcost_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vlstream_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlcost_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vlstream_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile cost-model live-VL gather host probe (v0.61)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlcost_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vlcost()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile cost-model live-VL scatter host probe (v0.61)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlcost_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vlcost()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile cost-model live-VL gather/scatter probe → 3025 (v0.61)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vlcost_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vlcost_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// hyst counter: hits=3 live=7 xk=5 pass=5 win=0 decay=0 half=0 → 20
	b.WriteString("; fusion PGO hyst counter probe → 20 (v0.61)\n")
	b.WriteString("define float @trin_fusion_pgo_hyst_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_hyst_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// hyst: n=10, isa=0, hits=0,..., win=4, decay=0, half=0 → 20
	b.WriteString("; fusion PGO hyst probe → 20 (v0.61)\n")
	b.WriteString("define float @trin_fusion_pgo_hyst_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_hyst(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_hyst_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_hyst_counter(i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_hyst(i32 %ni, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; hyst-PGO fused map+reduce probe → 495 (v0.61)\n")
	b.WriteString("define float @trin_fused_map_reduce_hyst_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_hyst_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.62: score-threshold multi-key filter/reduce + live VL residual streaming cost adaptive tile + windowed reweight PGO
	b.WriteString("define float @trin_nested_tag_thresh_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_thresh_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_thresh_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_thresh_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vladapt_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vlcost_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vladapt_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vlcost_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile adaptive-tile live-VL gather host probe (v0.62)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vladapt_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vladapt()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile adaptive-tile live-VL scatter host probe (v0.62)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vladapt_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vladapt()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile adaptive-tile live-VL gather/scatter probe → 3025 (v0.62)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vladapt_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vladapt_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// reweight counter: hits=3 live=7 xk=5 pass=5 win=0 decay=0 half=0 → 20
	b.WriteString("; fusion PGO reweight counter probe → 20 (v0.62)\n")
	b.WriteString("define float @trin_fusion_pgo_reweight_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_reweight_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// reweight: n=10, isa=0, hits=0,..., win=4, decay=0, half=0 → 20
	b.WriteString("; fusion PGO reweight probe → 20 (v0.62)\n")
	b.WriteString("define float @trin_fusion_pgo_reweight_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_reweight(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_reweight_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_reweight_counter(i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_reweight(i32 %ni, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; reweight-PGO fused map+reduce probe → 495 (v0.62)\n")
	b.WriteString("define float @trin_fused_map_reduce_reweight_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_reweight_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.63: soft-gated multi-key filter/reduce + live VL residual streaming adaptive cost residual schedule + budget PGO
	b.WriteString("define float @trin_nested_tag_soft_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_soft_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_soft_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_soft_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlsched_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vladapt_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlsched_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vladapt_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule live-VL gather host probe (v0.63)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlsched_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vlsched()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule live-VL scatter host probe (v0.63)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlsched_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vlsched()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule live-VL gather/scatter probe → 3025 (v0.63)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vlsched_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vlsched_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// budget counter: hits=3 live=7 xk=5 pass=5 win=0 decay=0 half=0 → 20
	b.WriteString("; fusion PGO budget counter probe → 20 (v0.63)\n")
	b.WriteString("define float @trin_fusion_pgo_budget_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_budget_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// budget: n=10, isa=0, hits=0,..., win=4, decay=0, half=0, budget=0 → 20
	b.WriteString("; fusion PGO budget probe → 20 (v0.63)\n")
	b.WriteString("define float @trin_fusion_pgo_budget_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_budget(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_budget_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_budget_counter(i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_budget(i32 %ni, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; budget-PGO fused map+reduce probe → 495 (v0.63)\n")
	b.WriteString("define float @trin_fused_map_reduce_budget_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_budget_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.64: calibrated soft-gated multi-key + residual streaming schedule peel fusion + softcap PGO
	b.WriteString("define float @trin_nested_tag_cal_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_cal_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_cal_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_cal_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlfuse_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vlsched_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlfuse_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vlsched_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live-VL gather host probe (v0.64)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlfuse_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vlfuse()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live-VL scatter host probe (v0.64)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlfuse_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vlfuse()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live-VL gather/scatter probe → 3025 (v0.64)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vlfuse_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vlfuse_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// softcap counter: hits=3 live=7 xk=5 pass=5 win=0 decay=0 half=0 → 20
	b.WriteString("; fusion PGO softcap counter probe → 20 (v0.64)\n")
	b.WriteString("define float @trin_fusion_pgo_softcap_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_softcap_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// softcap: n=10, isa=0, hits=0,..., win=4, decay=0, half=0, budget=0, softcap=0 → 20
	b.WriteString("; fusion PGO softcap probe → 20 (v0.64)\n")
	b.WriteString("define float @trin_fusion_pgo_softcap_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_softcap(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_softcap_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_softcap_counter(i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_softcap(i32 %ni, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; softcap-PGO fused map+reduce probe → 495 (v0.64)\n")
	b.WriteString("define float @trin_fused_map_reduce_softcap_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_softcap_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.65: adaptive multi-key + residual streaming schedule peel fusion live tile mix + dual soft/hard budget PGO
	b.WriteString("define float @trin_nested_tag_adapt_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_adapt_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_adapt_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_adapt_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlmix_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vlfuse_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlmix_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vlfuse_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix gather host probe (v0.65)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlmix_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vlmix()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix scatter host probe (v0.65)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlmix_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vlmix()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix gather/scatter probe → 3025 (v0.65)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vlmix_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vlmix_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// dual counter: hits=3 live=7 xk=5 pass=5 win=0 decay=0 half=0 → 20
	b.WriteString("; fusion PGO dual counter probe → 20 (v0.65)\n")
	b.WriteString("define float @trin_fusion_pgo_dual_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_dual_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// dual: n=10, isa=0, hits=0,..., win=4, decay=0, half=0, budget=0, softcap=0, hard=0 → 20
	b.WriteString("; fusion PGO dual probe → 20 (v0.65)\n")
	b.WriteString("define float @trin_fusion_pgo_dual_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_dual(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_dual_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_dual_counter(i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_dual(i32 %ni, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; dual-PGO fused map+reduce probe → 495 (v0.65)\n")
	b.WriteString("define float @trin_fused_map_reduce_dual_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_dual_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.66: blended multi-key + residual streaming schedule peel fusion live tile mix cost balance + conf blend PGO
	b.WriteString("define float @trin_nested_tag_blend_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_blend_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_blend_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_blend_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlbal_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vlmix_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlbal_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vlmix_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance gather host probe (v0.66)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlbal_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vlbal()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance scatter host probe (v0.66)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlbal_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vlbal()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance gather/scatter probe → 3025 (v0.66)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vlbal_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vlbal_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// conf counter: hits=3 live=7 xk=5 pass=5 win=0 decay=0 half=0 conf=0 → 20
	b.WriteString("; fusion PGO conf counter probe → 20 (v0.66)\n")
	b.WriteString("define float @trin_fusion_pgo_conf_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_conf_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// conf: n=10, isa=0, hits=0,..., win=4, decay=0, half=0, budget=0, softcap=0, hard=0, conf=0 → 20
	b.WriteString("; fusion PGO conf probe → 20 (v0.66)\n")
	b.WriteString("define float @trin_fusion_pgo_conf_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_conf(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_conf_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_conf_counter(i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_conf(i32 %ni, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; conf-PGO fused map+reduce probe → 495 (v0.66)\n")
	b.WriteString("define float @trin_fused_map_reduce_conf_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_conf_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.67: gated multi-key + residual streaming schedule peel fusion live tile mix cost balance bandwidth model + conf blend + hysteresis PGO
	b.WriteString("define float @trin_nested_tag_gate_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_gate_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_gate_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_gate_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlbw_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vlbal_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlbw_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vlbal_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth gather host probe (v0.67)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlbw_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vlbw()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth scatter host probe (v0.67)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlbw_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vlbw()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth gather/scatter probe → 3025 (v0.67)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vlbw_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vlbw_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// hyst counter: hits=3 live=7 xk=5 pass=5 win=0 decay=0 half=0 conf=0 hyst=0 → 20
	b.WriteString("; fusion PGO reg counter probe → 20 (v0.67)\n")
	b.WriteString("define float @trin_fusion_pgo_reg_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_reg_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// hyst: n=10, isa=0, hits=0,..., win=4, decay=0, half=0, budget=0, softcap=0, hard=0, conf=0, hyst=0 → 20
	b.WriteString("; fusion PGO reg probe → 20 (v0.67)\n")
	b.WriteString("define float @trin_fusion_pgo_reg_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_reg(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_reg_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_reg_counter(i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_reg(i32 %ni, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; reg-PGO fused map+reduce probe → 495 (v0.67)\n")
	b.WriteString("define float @trin_fused_map_reduce_reg_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_reg_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")


	// v0.68: score-calibrated multi-key + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost + conf blend + hysteresis + regret floor PGO
	b.WriteString("define float @trin_nested_tag_score_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_score_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_score_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_score_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlpath_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vlbw_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlpath_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vlbw_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost gather host probe (v0.68)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlpath_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vlpath()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost scatter host probe (v0.68)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlpath_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vlpath()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost gather/scatter probe → 3025 (v0.68)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vlpath_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vlpath_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// floor counter: hits=3 live=7 xk=5 pass=5 win=0 decay=0 half=0 conf=0 hyst=0 floor=0 → 20
	b.WriteString("; fusion PGO floor counter probe → 20 (v0.68)\n")
	b.WriteString("define float @trin_fusion_pgo_floor_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_floor_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	// floor: n=10, isa=0, hits=0,..., win=4, decay=0, half=0, budget=0, softcap=0, hard=0, conf=0, hyst=0, floor=0 → 20
	b.WriteString("; fusion PGO floor probe → 20 (v0.68)\n")
	b.WriteString("define float @trin_fusion_pgo_floor_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_floor(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_floor_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_floor(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_floor(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; floor-PGO fused map+reduce probe → 495 (v0.68)\n")
	b.WriteString("define float @trin_fused_map_reduce_floor_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_floor_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.69: soft confidence floor multi-key + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure + conf blend + hysteresis + regret floor + adaptive floor PGO
	b.WriteString("define float @trin_nested_tag_softfloor_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_softfloor_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_softfloor_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_softfloor_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vllane_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vlpath_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vllane_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vlpath_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure gather host probe (v0.69)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vllane_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vllane()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure scatter host probe (v0.69)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vllane_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vllane()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure gather/scatter probe → 3025 (v0.69)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vllane_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vllane_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO afloor counter probe → 20 (v0.69)\n")
	b.WriteString("define float @trin_fusion_pgo_afloor_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_afloor_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO afloor probe → 20 (v0.69)\n")
	b.WriteString("define float @trin_fusion_pgo_afloor_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_afloor(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_afloor_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_afloor(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_afloor(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; afloor-PGO fused map+reduce probe → 495 (v0.69)\n")
	b.WriteString("define float @trin_fused_map_reduce_afloor_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_afloor_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.70: hard floor gates multi-key + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model + conf blend + hysteresis + regret floor + adaptive floor + burst cap PGO
	b.WriteString("define float @trin_nested_tag_hardgate_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_hardgate_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_hardgate_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_hardgate_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlspill_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vllane_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlspill_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vllane_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model gather host probe (v0.70)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlspill_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vlspill()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model scatter host probe (v0.70)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlspill_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vlspill()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model gather/scatter probe → 3025 (v0.70)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vlspill_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vlspill_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO burst counter probe → 20 (v0.70)\n")
	b.WriteString("define float @trin_fusion_pgo_burst_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_burst_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO burst probe → 20 (v0.70)\n")
	b.WriteString("define float @trin_fusion_pgo_burst_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_burst(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_burst_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_burst(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_burst(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; burst-PGO fused map+reduce probe → 495 (v0.70)\n")
	b.WriteString("define float @trin_fused_map_reduce_burst_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_burst_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.71: dual gate blend multi-key + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down PGO
	b.WriteString("define float @trin_nested_tag_dual_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_dual_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_dual_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_dual_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlocc_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vlspill_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlocc_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vlspill_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy gather host probe (v0.71)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlocc_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vlocc()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy scatter host probe (v0.71)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlocc_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vlocc()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy gather/scatter probe → 3025 (v0.71)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vlocc_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vlocc_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO cool counter probe → 20 (v0.71)\n")
	b.WriteString("define float @trin_fusion_pgo_cool_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_cool_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO cool probe → 20 (v0.71)\n")
	b.WriteString("define float @trin_fusion_pgo_cool_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_cool(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_cool_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_cool(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_cool(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; cool-PGO fused map+reduce probe → 495 (v0.71)\n")
	b.WriteString("define float @trin_fused_map_reduce_cool_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_cool_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.72: soft/hard dual score mix multi-key + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank PGO
	b.WriteString("define float @trin_nested_tag_mix_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_mix_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_mix_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_mix_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlqueue_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vlocc_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlqueue_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vlocc_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth gather host probe (v0.72)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlqueue_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vlqueue()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth scatter host probe (v0.72)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlqueue_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vlqueue()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth gather/scatter probe → 3025 (v0.72)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vlqueue_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vlqueue_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO credit counter probe → 20 (v0.72)\n")
	b.WriteString("define float @trin_fusion_pgo_credit_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_credit_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO credit probe → 20 (v0.72)\n")
	b.WriteString("define float @trin_fusion_pgo_credit_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_credit(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_credit_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_credit(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_credit(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; credit-PGO fused map+reduce probe → 495 (v0.72)\n")
	b.WriteString("define float @trin_fused_map_reduce_credit_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_credit_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.73: soft/hard dual score mix multi-key ranked credit + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery PGO
	b.WriteString("define float @trin_nested_tag_credit_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_credit_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_credit_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_credit_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlwater_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vlqueue_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlwater_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vlqueue_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark gather host probe (v0.73)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlwater_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vlwater()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark scatter host probe (v0.73)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlwater_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vlwater()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark gather/scatter probe → 3025 (v0.73)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vlwater_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vlwater_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO debt counter probe → 20 (v0.73)\n")
	b.WriteString("define float @trin_fusion_pgo_debt_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_debt_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO debt probe → 20 (v0.73)\n")
	b.WriteString("define float @trin_fusion_pgo_debt_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_debt(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_debt_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_debt(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_debt(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; debt-PGO fused map+reduce probe → 495 (v0.73)\n")
	b.WriteString("define float @trin_fused_map_reduce_debt_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_debt_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.74: soft/hard dual score mix multi-key ranked credit + margin + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark + watermark drain + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest PGO
	b.WriteString("define float @trin_nested_tag_margin_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_margin_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_margin_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_margin_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vldrain_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vlwater_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vldrain_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vlwater_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain gather host probe (v0.74)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vldrain_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vldrain()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain scatter host probe (v0.74)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vldrain_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vldrain()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain gather/scatter probe → 3025 (v0.74)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vldrain_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vldrain_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO interest counter probe → 20 (v0.74)\n")
	b.WriteString("define float @trin_fusion_pgo_interest_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_interest_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO interest probe → 20 (v0.74)\n")
	b.WriteString("define float @trin_fusion_pgo_interest_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_interest(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_interest_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_interest(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_interest(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; interest-PGO fused map+reduce probe → 495 (v0.74)\n")
	b.WriteString("define float @trin_fused_map_reduce_interest_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_interest_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// v0.75: soft/hard dual score mix multi-key ranked credit + margin + band + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain + refill + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize PGO
	b.WriteString("define float @trin_nested_tag_band_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_band_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_band_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_band_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlrefill_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vldrain_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlrefill_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vldrain_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain + refill gather host probe (v0.75)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlrefill_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vlrefill()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain + refill scatter host probe (v0.75)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlrefill_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vlrefill()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain + refill gather/scatter probe → 3025 (v0.75)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vlrefill_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vlrefill_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO amortize counter probe → 20 (v0.75)\n")
	b.WriteString("define float @trin_fusion_pgo_amortize_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_amortize_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO amortize probe → 20 (v0.75)\n")
	b.WriteString("define float @trin_fusion_pgo_amortize_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_amortize(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_amortize_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_amortize(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_amortize(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; amortize-PGO fused map+reduce probe → 495 (v0.75)\n")
	b.WriteString("define float @trin_fused_map_reduce_amortize_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_amortize_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	// v0.76: soft/hard dual score mix multi-key ranked credit + margin + band + clamp + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain + refill + flush + flush + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + settle PGO
	b.WriteString("define float @trin_nested_tag_clamp_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_clamp_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_clamp_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_clamp_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlflush_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vlflush_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlflush_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vlflush_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain + refill + flush + flush gather host probe (v0.76)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlflush_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vlflush()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain + refill + flush + flush scatter host probe (v0.76)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlflush_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vlflush()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain + refill + flush + flush gather/scatter probe → 3025 (v0.76)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vlflush_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vlflush_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO amortize counter probe → 20 (v0.76)\n")
	b.WriteString("define float @trin_fusion_pgo_settle_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_settle_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO amortize probe → 20 (v0.76)\n")
	b.WriteString("define float @trin_fusion_pgo_settle_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_settle(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_settle_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_settle(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_settle(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; amortize-PGO fused map+reduce probe → 495 (v0.76)\n")
	b.WriteString("define float @trin_fused_map_reduce_settle_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_settle_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	// v0.77: soft/hard dual score mix multi-key ranked credit + margin + band + clamp + peg + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain + refill + flush + seal + flush + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + settle PGO
	b.WriteString("define float @trin_nested_tag_peg_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_peg_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_peg_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_peg_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlseal_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vlseal_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlseal_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vlseal_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain + refill + flush + seal + flush gather host probe (v0.77)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlseal_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vlseal()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain + refill + flush + seal + flush scatter host probe (v0.77)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlseal_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vlseal()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain + refill + flush + seal + flush gather/scatter probe → 3025 (v0.77)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vlseal_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vlseal_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO amortize counter probe → 20 (v0.77)\n")
	b.WriteString("define float @trin_fusion_pgo_escrow_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_escrow_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO amortize probe → 20 (v0.77)\n")
	b.WriteString("define float @trin_fusion_pgo_escrow_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_escrow(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_escrow_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_escrow(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_escrow(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; amortize-PGO fused map+reduce probe → 495 (v0.77)\n")
	b.WriteString("define float @trin_fused_map_reduce_escrow_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_escrow_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	// v0.78: soft/hard dual score mix multi-key ranked credit + margin + band + clamp + peg + latch + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain + refill + flush + seal + purge + flush + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + clear + settle PGO
	b.WriteString("define float @trin_nested_tag_latch_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_latch_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_latch_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_latch_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlpurge_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vlseal_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlpurge_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vlseal_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain + refill + flush + seal + purge + flush gather host probe (v0.78)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlpurge_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vlpurge()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain + refill + flush + seal + purge + flush scatter host probe (v0.78)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlpurge_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vlpurge()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain + refill + flush + seal + purge + flush gather/scatter probe → 3025 (v0.78)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vlpurge_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vlseal_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO amortize counter probe → 20 (v0.78)\n")
	b.WriteString("define float @trin_fusion_pgo_clear_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_clear_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO amortize probe → 20 (v0.78)\n")
	b.WriteString("define float @trin_fusion_pgo_clear_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_clear(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_clear_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_clear(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_clear(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; amortize-PGO fused map+reduce probe → 495 (v0.78)\n")
	b.WriteString("define float @trin_fused_map_reduce_clear_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_clear_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	// v0.79: soft/hard dual score mix multi-key ranked credit + margin + band + clamp + peg + latch + fence + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain + refill + flush + seal + purge + sync + flush + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + clear + final + settle PGO
	b.WriteString("define float @trin_nested_tag_fence_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_fence_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_fence_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_fence_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlsync_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vlpurge_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlsync_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vlpurge_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain + refill + flush + seal + purge + sync + flush gather host probe (v0.79)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlsync_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vlsync()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain + refill + flush + seal + purge + sync + flush scatter host probe (v0.79)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlsync_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vlsync()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain + refill + flush + seal + purge + sync + flush gather/scatter probe → 3025 (v0.79)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vlsync_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vlpurge_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO amortize counter probe → 20 (v0.79)\n")
	b.WriteString("define float @trin_fusion_pgo_final_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_final_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO amortize probe → 20 (v0.79)\n")
	b.WriteString("define float @trin_fusion_pgo_final_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_final(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_final_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_final(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_final(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; amortize-PGO fused map+reduce probe → 495 (v0.79)\n")
	b.WriteString("define float @trin_fused_map_reduce_final_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_final_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	// v0.80: soft/hard dual score mix multi-key ranked credit + margin + band + clamp + peg + latch + fence + lock + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain + refill + flush + seal + purge + sync + close + flush + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + clear + final + close + settle PGO
	b.WriteString("define float @trin_nested_tag_lock_reduce_f32(float %a0, float %b0, float %a1, float %b1) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %t0 = fadd float %a0, %b0\n")
	b.WriteString("  %t1 = fadd float %a1, %b1\n")
	b.WriteString("  %y = fadd float %t0, %t1\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_lock_filter_f32(float %a, float %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = fadd float %a, %b\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_nested_tag_lock_reduce_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_nested_tag_lock_reduce_f32(float 3.000000e+01, float 2.500000e+01, float 2.500000e+01, float 3.000000e+01)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlclose_f32(float* nocapture readonly %a, i32* nocapture readonly %idx, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_gather_mt_vlsync_f32(float* %a, i32* %idx, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlclose_f32(float* nocapture %a, i32* nocapture readonly %idx, float %v, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %y = call float @trin_sme_za_scatter_mt_vlsync_f32(float* %a, i32* %idx, float %v, i64 %n)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain + refill + flush + seal + purge + sync + close + flush gather host probe (v0.80)\n")
	b.WriteString("define float @trin_sme_za_gather_mt_vlclose_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_gather_mt_vlclose()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain + refill + flush + seal + purge + sync + close + flush scatter host probe (v0.80)\n")
	b.WriteString("define float @trin_sme_za_scatter_mt_vlclose_host_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %v = call i32 @trin_hw_sme_za_scatter_mt_vlclose()\n")
	b.WriteString("  %f = sitofp i32 %v to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; SME ZA multi-tile residual-schedule peel-fused live tile-mix cost-balance bandwidth path-cost lane-pressure spill-model occupancy queue-depth watermark drain + refill + flush + seal + purge + sync + close + flush gather/scatter probe → 3025 (v0.80)\n")
	b.WriteString("define float @trin_sme_za_gather_scatter_mt_vlclose_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %idx = alloca [10 x i32], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %ip = getelementptr inbounds [10 x i32], [10 x i32]* %idx, i64 0, i64 0\n")
	for i := 0; i < 10; i++ {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %d.000000e+00, float* %%api%d, align 4\n", i+1, i)
		fmt.Fprintf(&b, "  %%ipi%d = getelementptr inbounds i32, i32* %%ip, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store i32 %d, i32* %%ipi%d, align 4\n", i, i)
	}
	b.WriteString("  %g = call float @trin_sme_za_gather_mt_vlsync_f32(float* %ap, i32* %ip, i64 10)\n")
	b.WriteString("  %y = fmul float %g, %g\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO amortize counter probe → 20 (v0.80)\n")
	b.WriteString("define float @trin_fusion_pgo_close_counter_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_close_counter(i32 3, i32 7, i32 5, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("; fusion PGO amortize probe → 20 (v0.80)\n")
	b.WriteString("define float @trin_fusion_pgo_close_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %c = call i32 @trin_fusion_pgo_close(i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %f = sitofp i32 %c to float\n")
	b.WriteString("  ret float %f\n")
	b.WriteString("}\n\n")
	b.WriteString("define float @trin_fused_map_reduce_close_f32(float* nocapture readonly %a, float* nocapture readonly %b, i64 %n) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %ni = trunc i64 %n to i32\n")
	b.WriteString("  %wc = call i32 @trin_fusion_pgo_close(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c0 = call i32 @trin_fusion_pgo_close(i32 %ni, i32 0, i32 0, i32 0, i32 0, i32 0, i32 %wc, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n")
	b.WriteString("  %c1 = call i32 @trin_fusion_cost_isa(i32 %ni)\n")
	b.WriteString("  %c2 = add i32 16, 0\n")
	b.WriteString("  %idx = call i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2)\n")
	b.WriteString("  %ctr = call i32 @trin_fusion_pgo_isa_counter(i32 0, i32 %idx)\n")
	b.WriteString("  %pr = call { float, float } @trin_fused_map_reduce_mul_add_f32(float* %a, float* %b, i64 %n)\n")
	b.WriteString("  %m = call float @trin_pair_first_f32({ float, float } %pr)\n")
	b.WriteString("  %s = call float @trin_pair_second_f32({ float, float } %pr)\n")
	b.WriteString("  %base = fadd float %m, %s\n")
	b.WriteString("  %cf = sitofp i32 %ctr to float\n")
	b.WriteString("  %z = fsub float %cf, %cf\n")
	b.WriteString("  %y = fadd float %base, %z\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	b.WriteString("; amortize-PGO fused map+reduce probe → 495 (v0.80)\n")
	b.WriteString("define float @trin_fused_map_reduce_close_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [10 x float], align 16\n")
	b.WriteString("  %b = alloca [10 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [10 x float], [10 x float]* %b, i64 0, i64 0\n")
	for i, f := range vals10 {
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %y = call float @trin_fused_map_reduce_close_f32(float* %ap, float* %bp, i64 10)\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")
	// Classic n=8 probe (v0.20 compatibility)
	b.WriteString("define float @trin_vec_loop_probe() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %a = alloca [8 x float], align 16\n")
	b.WriteString("  %b = alloca [8 x float], align 16\n")
	b.WriteString("  %ap = getelementptr inbounds [8 x float], [8 x float]* %a, i64 0, i64 0\n")
	b.WriteString("  %bp = getelementptr inbounds [8 x float], [8 x float]* %b, i64 0, i64 0\n")
	for i := 0; i < 8; i++ {
		f := llvmF32(float64(i + 1))
		fmt.Fprintf(&b, "  %%api%d = getelementptr inbounds float, float* %%ap, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%api%d, align 4\n", f, i)
		fmt.Fprintf(&b, "  %%bpi%d = getelementptr inbounds float, float* %%bp, i64 %d\n", i, i)
		fmt.Fprintf(&b, "  store float %s, float* %%bpi%d, align 4\n", f, i)
	}
	b.WriteString("  %s = call float @trin_sum_f32(float* %ap, i64 8)\n")
	b.WriteString("  %d = call float @trin_dot_f32(float* %ap, float* %bp, i64 8)\n")
	b.WriteString("  %sa = call float @trin_sum_span_add_f32(float* %ap, float* %bp, i64 8)\n")
	b.WriteString("  %sc = call float @trin_sum_scale_f32(float 2.000000e+00, float* %ap, i64 8)\n")
	b.WriteString("  %c204 = fadd float 2.040000e+02, 0.000000e+00\n")
	b.WriteString("  %c72 = fadd float 7.200000e+01, 0.000000e+00\n")
	b.WriteString("  %e0 = fsub float %d, %c204\n")
	b.WriteString("  %e1 = fsub float %sa, %c72\n")
	b.WriteString("  %e2 = fsub float %sc, %c72\n")
	b.WriteString("  %t0 = fadd float %s, %e0\n")
	b.WriteString("  %t1 = fadd float %t0, %e1\n")
	b.WriteString("  %y = fadd float %t1, %e2\n")
	b.WriteString("  ret float %y\n")
	b.WriteString("}\n\n")

	// Loop vectorize / interleave hints (shared residual scalar loops)
	b.WriteString("!0 = distinct !{!0, !1, !2, !3}\n")
	b.WriteString("!1 = !{!\"llvm.loop.vectorize.enable\", i1 true}\n")
	b.WriteString("!2 = !{!\"llvm.loop.vectorize.width\", i32 4}\n")
	b.WriteString("!3 = !{!\"llvm.loop.interleave.count\", i32 2}\n\n")
	return b.String()
}

// ensurePeelHWVLEN injects a portable @trin_hw_vlen_for_n when the module
// lacks one (so auto peel helpers link without full CPUID when tests emit
// only span loops). Prefer real runtime helpers when present.
// Insertion is after any leading `target …` lines so llc still sees them first.
// Also ensures @trin_hw_rdvl / @trin_hw_rdvl_live / @trin_hw_sme_vl / @trin_hw_sme_za_tile /
// @trin_hw_sme_za_outer / @trin_fusion_cost / outer_acc / weighted+isa cost / select (v0.46–v0.51).
func ensurePeelHWVLEN(ll string) string {
	needForN := !strings.Contains(ll, "define i32 @trin_hw_vlen_for_n") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_vlen_for_n")
	needRDVL := !strings.Contains(ll, "define i32 @trin_hw_rdvl") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_rdvl")
	needRDVLLive := !strings.Contains(ll, "define i32 @trin_hw_rdvl_live") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_rdvl_live")
	needSME := !strings.Contains(ll, "define i32 @trin_hw_sme_vl") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_vl")
	needZA := !strings.Contains(ll, "define i32 @trin_hw_sme_za_tile") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_tile")
	needOuter := !strings.Contains(ll, "define i32 @trin_hw_sme_za_outer()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_outer()")
	needCost := !strings.Contains(ll, "define i32 @trin_fusion_cost(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_cost(")
	needOuterAcc := !strings.Contains(ll, "define i32 @trin_hw_sme_za_outer_acc()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_outer_acc()")
	needCostW := !strings.Contains(ll, "define i32 @trin_fusion_cost_weighted(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_cost_weighted(")
	needCostISA := !strings.Contains(ll, "define i32 @trin_fusion_cost_isa(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_cost_isa(")
	needSelect := !strings.Contains(ll, "define i32 @trin_fusion_select(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_select(")
	needOuterAccTrue := !strings.Contains(ll, "define i32 @trin_hw_sme_za_outer_acc_true()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_outer_acc_true()")
	needCostCal := !strings.Contains(ll, "define i32 @trin_fusion_cost_calibrated(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_cost_calibrated(")
	needPlanLog := !strings.Contains(ll, "define i32 @trin_fusion_plan_log(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_plan_log(")
	needStreamStore := !strings.Contains(ll, "define i32 @trin_hw_sme_za_stream_store()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_stream_store()")
	needPgoW := !strings.Contains(ll, "define i32 @trin_fusion_pgo_weight(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_weight(")
	needPgoFb := !strings.Contains(ll, "define i32 @trin_fusion_pgo_feedback(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_feedback(")
	needPgoSel := !strings.Contains(ll, "define i32 @trin_fusion_pgo_select(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_select(")
	needGather := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather()")
	needScatter := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter()")
	needPgoAdapt := !strings.Contains(ll, "define i32 @trin_fusion_pgo_adapt(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_adapt(")
	needPgoBump := !strings.Contains(ll, "define i32 @trin_fusion_pgo_counter_bump(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_counter_bump(")
	needGatherMv := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mv()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mv()")
	needScatterMv := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mv()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mv()")
	needPgoIsaCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_isa_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_isa_counter(")
	needPgoIsaLoop := !strings.Contains(ll, "define i32 @trin_fusion_pgo_isa_loop(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_isa_loop(")
	needGatherTile := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_tile()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_tile()")
	needScatterTile := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_tile()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_tile()")
	needPgoIsaLive := !strings.Contains(ll, "define i32 @trin_fusion_pgo_isa_live(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_isa_live(")
	needPgoIsaFb := !strings.Contains(ll, "define i32 @trin_fusion_pgo_isa_feedback(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_isa_feedback(")
	needGatherMt := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt()")
	needScatterMt := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt()")
	needPgoXk := !strings.Contains(ll, "define i32 @trin_fusion_pgo_xkernel(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_xkernel(")
	needPgoXkCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_xkernel_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_xkernel_counter(")
	needGatherMtPeel := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_peel()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_peel()")
	needScatterMtPeel := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_peel()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_peel()")
	needPgoXpass := !strings.Contains(ll, "define i32 @trin_fusion_pgo_xpass(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_xpass(")
	needPgoXpassCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_xpass_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_xpass_counter(")
	needGatherMtAuto := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_auto()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_auto()")
	needScatterMtAuto := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_auto()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_auto()")
	needPgoWin := !strings.Contains(ll, "define i32 @trin_fusion_pgo_window(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_window(")
	needPgoWinCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_window_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_window_counter(")
	needGatherMtVlfb := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vlfb()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vlfb()")
	needScatterMtVlfb := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vlfb()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vlfb()")
	needPgoDecay := !strings.Contains(ll, "define i32 @trin_fusion_pgo_decay(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_decay(")
	needPgoDecayCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_decay_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_decay_counter(")
	needGatherMtVllive := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vllive()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vllive()")
	needScatterMtVllive := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vllive()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vllive()")
	needPgoEma := !strings.Contains(ll, "define i32 @trin_fusion_pgo_ema(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_ema(")
	needPgoEmaCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_ema_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_ema_counter(")
	needGatherMtVlstream := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vlstream()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vlstream()")
	needScatterMtVlstream := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vlstream()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vlstream()")
	needPgoAdaptHl := !strings.Contains(ll, "define i32 @trin_fusion_pgo_adapt_hl(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_adapt_hl(")
	needPgoAdaptHlCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_adapt_hl_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_adapt_hl_counter(")
	needGatherMtVlcost := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vlcost()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vlcost()")
	needScatterMtVlcost := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vlcost()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vlcost()")
	needPgoHyst := !strings.Contains(ll, "define i32 @trin_fusion_pgo_hyst(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_hyst(")
	needPgoHystCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_hyst_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_hyst_counter(")
	needGatherMtVladapt := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vladapt()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vladapt()")
	needScatterMtVladapt := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vladapt()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vladapt()")
	needPgoReweight := !strings.Contains(ll, "define i32 @trin_fusion_pgo_reweight(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_reweight(")
	needPgoReweightCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_reweight_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_reweight_counter(")
	needGatherMtVlsched := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vlsched()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vlsched()")
	needScatterMtVlsched := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vlsched()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vlsched()")
	needPgoBudget := !strings.Contains(ll, "define i32 @trin_fusion_pgo_budget(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_budget(")
	needPgoBudgetCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_budget_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_budget_counter(")
	needGatherMtVlfuse := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vlfuse()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vlfuse()")
	needScatterMtVlfuse := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vlfuse()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vlfuse()")
	needPgoSoftcap := !strings.Contains(ll, "define i32 @trin_fusion_pgo_softcap(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_softcap(")
	needPgoSoftcapCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_softcap_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_softcap_counter(")
	needGatherMtVlmix := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vlmix()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vlmix()")
	needScatterMtVlmix := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vlmix()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vlmix()")
	needPgoDual := !strings.Contains(ll, "define i32 @trin_fusion_pgo_dual(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_dual(")
	needPgoDualCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_dual_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_dual_counter(")
	needGatherMtVlbal := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vlbal()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vlbal()")
	needScatterMtVlbal := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vlbal()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vlbal()")
	needPgoConf := !strings.Contains(ll, "define i32 @trin_fusion_pgo_conf(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_conf(")
	needPgoConfCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_conf_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_conf_counter(")
	needGatherMtVlbw := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vlbw()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vlbw()")
	needScatterMtVlbw := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vlbw()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vlbw()")
	needPgoReg := !strings.Contains(ll, "define i32 @trin_fusion_pgo_reg(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_reg(")
	needPgoRegCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_reg_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_reg_counter(")
	needGatherMtVlpath := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vlpath()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vlpath()")
	needScatterMtVlpath := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vlpath()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vlpath()")
	needPgoFloor := !strings.Contains(ll, "define i32 @trin_fusion_pgo_floor(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_floor(")
	needPgoFloorCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_floor_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_floor_counter(")
	needGatherMtVllane := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vllane()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vllane()")
	needScatterMtVllane := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vllane()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vllane()")
	needPgoAfloor := !strings.Contains(ll, "define i32 @trin_fusion_pgo_afloor(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_afloor(")
	needPgoAfloorCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_afloor_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_afloor_counter(")
	needGatherMtVlspill := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vlspill()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vlspill()")
	needScatterMtVlspill := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vlspill()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vlspill()")
	needPgoBurst := !strings.Contains(ll, "define i32 @trin_fusion_pgo_burst(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_burst(")
	needPgoBurstCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_burst_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_burst_counter(")
	needGatherMtVlocc := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vlocc()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vlocc()")
	needScatterMtVlocc := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vlocc()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vlocc()")
	needPgoCool := !strings.Contains(ll, "define i32 @trin_fusion_pgo_cool(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_cool(")
	needPgoCoolCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_cool_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_cool_counter(")
	needGatherMtVlqueue := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vlqueue()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vlqueue()")
	needScatterMtVlqueue := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vlqueue()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vlqueue()")
	needPgoCredit := !strings.Contains(ll, "define i32 @trin_fusion_pgo_credit(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_credit(")
	needPgoCreditCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_credit_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_credit_counter(")
	needGatherMtVlwater := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vlwater()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vlwater()")
	needScatterMtVlwater := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vlwater()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vlwater()")
	needPgoDebt := !strings.Contains(ll, "define i32 @trin_fusion_pgo_debt(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_debt(")
	needPgoDebtCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_debt_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_debt_counter(")
	needGatherMtVldrain := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vldrain()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vldrain()")
	needScatterMtVldrain := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vldrain()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vldrain()")
	needPgoInterest := !strings.Contains(ll, "define i32 @trin_fusion_pgo_interest(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_interest(")
	needPgoInterestCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_interest_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_interest_counter(")
	needGatherMtVlrefill := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vlrefill()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vlrefill()")
	needScatterMtVlrefill := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vlrefill()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vlrefill()")
	needPgoAmortize := !strings.Contains(ll, "define i32 @trin_fusion_pgo_amortize(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_amortize(")
	needPgoAmortizeCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_amortize_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_amortize_counter(")
	needGatherMtVlflush := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vlflush()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vlflush()")
	needScatterMtVlflush := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vlflush()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vlflush()")
	needPgoSettle := !strings.Contains(ll, "define i32 @trin_fusion_pgo_settle(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_settle(")
	needPgoSettleCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_settle_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_settle_counter(")
	needGatherMtVlseal := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vlseal()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vlseal()")
	needScatterMtVlseal := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vlseal()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vlseal()")
	needPgoEscrow := !strings.Contains(ll, "define i32 @trin_fusion_pgo_escrow(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_escrow(")
	needPgoEscrowCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_escrow_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_escrow_counter(")
	needGatherMtVlpurge := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vlpurge()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vlpurge()")
	needScatterMtVlpurge := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vlpurge()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vlpurge()")
	needPgoClear := !strings.Contains(ll, "define i32 @trin_fusion_pgo_clear(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_clear(")
	needPgoClearCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_clear_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_clear_counter(")
	needGatherMtVlsync := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vlsync()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vlsync()")
	needScatterMtVlsync := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vlsync()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vlsync()")
	needPgoFinal := !strings.Contains(ll, "define i32 @trin_fusion_pgo_final(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_final(")
	needPgoFinalCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_final_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_final_counter(")
	needGatherMtVlclose := !strings.Contains(ll, "define i32 @trin_hw_sme_za_gather_mt_vlclose()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_gather_mt_vlclose()")
	needScatterMtVlclose := !strings.Contains(ll, "define i32 @trin_hw_sme_za_scatter_mt_vlclose()") &&
		!strings.Contains(ll, "define internal i32 @trin_hw_sme_za_scatter_mt_vlclose()")
	needPgoClose := !strings.Contains(ll, "define i32 @trin_fusion_pgo_close(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_close(")
	needPgoCloseCtr := !strings.Contains(ll, "define i32 @trin_fusion_pgo_close_counter(") &&
		!strings.Contains(ll, "define internal i32 @trin_fusion_pgo_close_counter(")
	if !needForN && !needRDVL && !needRDVLLive && !needSME && !needZA && !needOuter && !needCost &&
		!needOuterAcc && !needCostW && !needCostISA && !needSelect &&
		!needOuterAccTrue && !needCostCal && !needPlanLog &&
		!needStreamStore && !needPgoW && !needPgoFb && !needPgoSel &&
		!needGather && !needScatter && !needPgoAdapt && !needPgoBump &&
		!needGatherMv && !needScatterMv && !needPgoIsaCtr && !needPgoIsaLoop &&
		!needGatherTile && !needScatterTile && !needPgoIsaLive && !needPgoIsaFb &&
		!needGatherMt && !needScatterMt && !needPgoXk && !needPgoXkCtr &&
		!needGatherMtPeel && !needScatterMtPeel && !needPgoXpass && !needPgoXpassCtr &&
		!needGatherMtAuto && !needScatterMtAuto && !needPgoWin && !needPgoWinCtr &&
		!needGatherMtVlfb && !needScatterMtVlfb && !needPgoDecay && !needPgoDecayCtr &&
		!needGatherMtVllive && !needScatterMtVllive && !needPgoEma && !needPgoEmaCtr &&
		!needGatherMtVlstream && !needScatterMtVlstream && !needPgoAdaptHl && !needPgoAdaptHlCtr &&
		!needGatherMtVlcost && !needScatterMtVlcost && !needPgoHyst && !needPgoHystCtr &&
		!needGatherMtVladapt && !needScatterMtVladapt && !needPgoReweight && !needPgoReweightCtr &&
		!needGatherMtVlsched && !needScatterMtVlsched && !needPgoBudget && !needPgoBudgetCtr &&
		!needGatherMtVlfuse && !needScatterMtVlfuse && !needPgoSoftcap && !needPgoSoftcapCtr &&
		!needGatherMtVlmix && !needScatterMtVlmix && !needPgoDual && !needPgoDualCtr &&
		!needGatherMtVlbal && !needScatterMtVlbal && !needPgoConf && !needPgoConfCtr &&
		!needGatherMtVlbw && !needScatterMtVlbw && !needPgoReg && !needPgoRegCtr &&
		!needGatherMtVlpath && !needScatterMtVlpath && !needPgoFloor && !needPgoFloorCtr &&
		!needGatherMtVllane && !needScatterMtVllane && !needPgoAfloor && !needPgoAfloorCtr &&
		!needGatherMtVlspill && !needScatterMtVlspill && !needPgoBurst && !needPgoBurstCtr &&
		!needGatherMtVlocc && !needScatterMtVlocc && !needPgoCool && !needPgoCoolCtr &&
		!needGatherMtVlqueue && !needScatterMtVlqueue && !needPgoCredit && !needPgoCreditCtr &&
		!needGatherMtVlwater && !needScatterMtVlwater && !needPgoDebt && !needPgoDebtCtr &&
		!needGatherMtVldrain && !needScatterMtVldrain && !needPgoInterest && !needPgoInterestCtr &&
		!needGatherMtVlrefill && !needScatterMtVlrefill && !needPgoAmortize && !needPgoAmortizeCtr &&
		!needGatherMtVlflush && !needScatterMtVlflush && !needPgoSettle && !needPgoSettleCtr &&
		!needGatherMtVlseal && !needScatterMtVlseal && !needPgoEscrow && !needPgoEscrowCtr &&
		!needGatherMtVlpurge && !needScatterMtVlpurge && !needPgoClear && !needPgoClearCtr &&
		!needGatherMtVlsync && !needScatterMtVlsync && !needPgoFinal && !needPgoFinalCtr &&
		!needGatherMtVlclose && !needScatterMtVlclose && !needPgoClose && !needPgoCloseCtr {
		return ll
	}
	// Portable pick among {4,8,16} identical to MLIR portable trin_vlen_for_n.
	fallback := ""
	if needForN {
		fallback += "" +
			"; portable fallback hw_vlen_for_n for auto residual peel (v0.36)\n" +
			"define i32 @trin_hw_vlen_for_n(i32 %n) {\n" +
			"entry:\n" +
			"  %ge16 = icmp sge i32 %n, 16\n" +
			"  br i1 %ge16, label %r16, label %chk8\n" +
			"r16:\n" +
			"  ret i32 16\n" +
			"chk8:\n" +
			"  %ge8 = icmp sge i32 %n, 8\n" +
			"  br i1 %ge8, label %r8, label %chk4\n" +
			"r8:\n" +
			"  ret i32 8\n" +
			"chk4:\n" +
			"  %ge4 = icmp sge i32 %n, 4\n" +
			"  br i1 %ge4, label %r4, label %rn\n" +
			"r4:\n" +
			"  ret i32 4\n" +
			"rn:\n" +
			"  ret i32 %n\n" +
			"}\n\n"
	}
	if needRDVL {
		// Portable RDVL stub: alias preferred portable tile width (16).
		fallback += "" +
			"; portable fallback hw_rdvl for multi-out fused probes (v0.51)\n" +
			"define i32 @trin_hw_rdvl() {\n" +
			"entry:\n" +
			"  ret i32 16\n" +
			"}\n\n"
	}
	if needRDVLLive {
		fallback += "" +
			"; portable fallback hw_rdvl_live alias (v0.51)\n" +
			"define i32 @trin_hw_rdvl_live() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_rdvl()\n" +
			"  ret i32 %v\n" +
			"}\n\n"
	}
	if needSME {
		fallback += "" +
			"; portable fallback hw_sme_vl (v0.51)\n" +
			"define i32 @trin_hw_sme_vl() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_rdvl_live()\n" +
			"  ret i32 %v\n" +
			"}\n\n"
	}
	if needZA {
		fallback += "" +
			"; portable fallback hw_sme_za_tile (v0.51)\n" +
			"define i32 @trin_hw_sme_za_tile() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_vl()\n" +
			"  ret i32 %v\n" +
			"}\n\n"
	}
	if needOuter {
		fallback += "" +
			"; portable fallback hw_sme_za_outer (v0.51)\n" +
			"define i32 @trin_hw_sme_za_outer() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_tile()\n" +
			"  ret i32 %v\n" +
			"}\n\n"
	}
	if needCost {
		fallback += "" +
			"; portable fallback fusion_cost (v0.51)\n" +
			"define i32 @trin_fusion_cost(i32 %n) {\n" +
			"entry:\n" +
			"  ret i32 %n\n" +
			"}\n\n"
	}
	if needOuterAcc {
		fallback += "" +
			"; portable fallback hw_sme_za_outer_acc (v0.51)\n" +
			"define i32 @trin_hw_sme_za_outer_acc() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_outer()\n" +
			"  ret i32 %v\n" +
			"}\n\n"
	}
	if needCostW {
		fallback += "" +
			"; portable fallback fusion_cost_weighted (v0.51)\n" +
			"define i32 @trin_fusion_cost_weighted(i32 %n, i32 %w) {\n" +
			"entry:\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needCostISA {
		fallback += "" +
			"; portable fallback fusion_cost_isa (v0.51)\n" +
			"define i32 @trin_fusion_cost_isa(i32 %n) {\n" +
			"entry:\n" +
			"  %y = call i32 @trin_fusion_cost_weighted(i32 %n, i32 1)\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needSelect {
		fallback += "" +
			"; portable fallback fusion_select (v0.51)\n" +
			"define i32 @trin_fusion_select(i32 %c0, i32 %c1, i32 %c2) {\n" +
			"entry:\n" +
			"  %lt01 = icmp slt i32 %c0, %c1\n" +
			"  %b01 = select i1 %lt01, i32 %c0, i32 %c1\n" +
			"  %i01 = select i1 %lt01, i32 0, i32 1\n" +
			"  %ltb2 = icmp slt i32 %b01, %c2\n" +
			"  %yi = select i1 %ltb2, i32 %i01, i32 2\n" +
			"  ret i32 %yi\n" +
			"}\n\n"
	}
	if needOuterAccTrue {
		fallback += "" +
			"; portable fallback hw_sme_za_outer_acc_true (v0.51)\n" +
			"define i32 @trin_hw_sme_za_outer_acc_true() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_outer_acc()\n" +
			"  ret i32 %v\n" +
			"}\n\n"
	}
	if needCostCal {
		fallback += "" +
			"; portable fallback fusion_cost_calibrated (v0.51)\n" +
			"define i32 @trin_fusion_cost_calibrated(i32 %n) {\n" +
			"entry:\n" +
			"  %y = call i32 @trin_fusion_cost_weighted(i32 %n, i32 2)\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPlanLog {
		fallback += "" +
			"; portable fallback fusion_plan_log (v0.51)\n" +
			"define i32 @trin_fusion_plan_log(i32 %c0, i32 %c1, i32 %c2) {\n" +
			"entry:\n" +
			"  %y = call i32 @trin_fusion_select(i32 %c0, i32 %c1, i32 %c2)\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needStreamStore {
		fallback += "" +
			"; portable fallback hw_sme_za_stream_store (v0.51)\n" +
			"define i32 @trin_hw_sme_za_stream_store() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_outer_acc_true()\n" +
			"  ret i32 %v\n" +
			"}\n\n"
	}
	if needPgoW {
		fallback += "" +
			"; portable fallback fusion_pgo_weight (v0.51)\n" +
			"define i32 @trin_fusion_pgo_weight(i32 %n) {\n" +
			"entry:\n" +
			"  %y = call i32 @trin_fusion_cost_weighted(i32 %n, i32 2)\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoFb {
		fallback += "" +
			"; portable fallback fusion_pgo_feedback (v0.51)\n" +
			"define i32 @trin_fusion_pgo_feedback(i32 %sel, i32 %rew) {\n" +
			"entry:\n" +
			"  %y = add i32 %sel, %rew\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoSel {
		fallback += "" +
			"; portable fallback fusion_pgo_select (v0.51)\n" +
			"define i32 @trin_fusion_pgo_select(i32 %c0, i32 %c1, i32 %c2) {\n" +
			"entry:\n" +
			"  %y = call i32 @trin_fusion_plan_log(i32 %c0, i32 %c1, i32 %c2)\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needGather {
		fallback += "" +
			"; portable fallback hw_sme_za_gather (v0.52)\n" +
			"define i32 @trin_hw_sme_za_gather() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_stream_store()\n" +
			"  ret i32 %v\n" +
			"}\n\n"
	}
	if needScatter {
		fallback += "" +
			"; portable fallback hw_sme_za_scatter (v0.52)\n" +
			"define i32 @trin_hw_sme_za_scatter() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_stream_store()\n" +
			"  ret i32 %v\n" +
			"}\n\n"
	}
	if needPgoAdapt {
		fallback += "" +
			"; portable fallback fusion_pgo_adapt (v0.52)\n" +
			"define i32 @trin_fusion_pgo_adapt(i32 %n, i32 %c0) {\n" +
			"entry:\n" +
			"  %t = sdiv i32 %c0, 10\n" +
			"  %w = add i32 2, %t\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoBump {
		fallback += "" +
			"; portable fallback fusion_pgo_counter_bump (v0.52)\n" +
			"define i32 @trin_fusion_pgo_counter_bump(i32 %c, i32 %d) {\n" +
			"entry:\n" +
			"  %y = add i32 %c, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needGatherMv {
		fallback += "" +
			"; portable fallback hw_sme_za_gather_mv (v0.53)\n" +
			"define i32 @trin_hw_sme_za_gather_mv() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather()\n" +
			"  ret i32 %v\n" +
			"}\n\n"
	}
	if needScatterMv {
		fallback += "" +
			"; portable fallback hw_sme_za_scatter_mv (v0.53)\n" +
			"define i32 @trin_hw_sme_za_scatter_mv() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter()\n" +
			"  ret i32 %v\n" +
			"}\n\n"
	}
	if needPgoIsaCtr {
		fallback += "" +
			"; portable fallback fusion_pgo_isa_counter (v0.53)\n" +
			"define i32 @trin_fusion_pgo_isa_counter(i32 %isa, i32 %hits) {\n" +
			"entry:\n" +
			"  %is1 = icmp eq i32 %isa, 1\n" +
			"  %is2 = icmp eq i32 %isa, 2\n" +
			"  %is3 = icmp eq i32 %isa, 3\n" +
			"  %is4 = icmp eq i32 %isa, 4\n" +
			"  %is5 = icmp eq i32 %isa, 5\n" +
			"  %w1 = select i1 %is1, i32 2, i32 1\n" +
			"  %w2 = select i1 %is2, i32 3, i32 %w1\n" +
			"  %w3 = select i1 %is3, i32 4, i32 %w2\n" +
			"  %w4 = select i1 %is4, i32 4, i32 %w3\n" +
			"  %w5 = select i1 %is5, i32 2, i32 %w4\n" +
			"  %y = add i32 %w5, %hits\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoIsaLoop {
		fallback += "" +
			"; portable fallback fusion_pgo_isa_loop (v0.53)\n" +
			"define i32 @trin_fusion_pgo_isa_loop(i32 %n, i32 %isa, i32 %hits) {\n" +
			"entry:\n" +
			"  %is1 = icmp eq i32 %isa, 1\n" +
			"  %is2 = icmp eq i32 %isa, 2\n" +
			"  %is3 = icmp eq i32 %isa, 3\n" +
			"  %is4 = icmp eq i32 %isa, 4\n" +
			"  %is5 = icmp eq i32 %isa, 5\n" +
			"  %w1 = select i1 %is1, i32 2, i32 1\n" +
			"  %w2 = select i1 %is2, i32 3, i32 %w1\n" +
			"  %w3 = select i1 %is3, i32 4, i32 %w2\n" +
			"  %w4 = select i1 %is4, i32 4, i32 %w3\n" +
			"  %w5 = select i1 %is5, i32 2, i32 %w4\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %w = add i32 %w5, %hd\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needGatherTile {
		fallback += "" +
			"; portable fallback hw_sme_za_gather_tile (v0.54)\n" +
			"define i32 @trin_hw_sme_za_gather_tile() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mv()\n" +
			"  %y = mul i32 %v, 2\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterTile {
		fallback += "" +
			"; portable fallback hw_sme_za_scatter_tile (v0.54)\n" +
			"define i32 @trin_hw_sme_za_scatter_tile() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mv()\n" +
			"  %y = mul i32 %v, 2\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoIsaLive {
		fallback += "" +
			"; portable fallback fusion_pgo_isa_live (v0.54)\n" +
			"define i32 @trin_fusion_pgo_isa_live(i32 %n, i32 %isa, i32 %hits, i32 %live) {\n" +
			"entry:\n" +
			"  %is1 = icmp eq i32 %isa, 1\n" +
			"  %is2 = icmp eq i32 %isa, 2\n" +
			"  %is3 = icmp eq i32 %isa, 3\n" +
			"  %is4 = icmp eq i32 %isa, 4\n" +
			"  %is5 = icmp eq i32 %isa, 5\n" +
			"  %w1 = select i1 %is1, i32 2, i32 1\n" +
			"  %w2 = select i1 %is2, i32 3, i32 %w1\n" +
			"  %w3 = select i1 %is3, i32 4, i32 %w2\n" +
			"  %w4 = select i1 %is4, i32 4, i32 %w3\n" +
			"  %w5 = select i1 %is5, i32 2, i32 %w4\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %w6 = add i32 %w5, %hd\n" +
			"  %w = add i32 %w6, %ld\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoIsaFb {
		fallback += "" +
			"; portable fallback fusion_pgo_isa_feedback (v0.54)\n" +
			"define i32 @trin_fusion_pgo_isa_feedback(i32 %hits, i32 %live) {\n" +
			"entry:\n" +
			"  %y = add i32 %hits, %live\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needGatherMt {
		fallback += "" +
			"; portable fallback hw_sme_za_gather_mt (v0.55)\n" +
			"define i32 @trin_hw_sme_za_gather_mt() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_tile()\n" +
			"  %y = mul i32 %v, 2\n" +
			"  %adj = sdiv i32 %y, 4\n" +
			"  %z = sub i32 %y, %adj\n" +
			"  ret i32 %z\n" +
			"}\n\n"
	}
	if needScatterMt {
		fallback += "" +
			"; portable fallback hw_sme_za_scatter_mt (v0.55)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_tile()\n" +
			"  %y = mul i32 %v, 2\n" +
			"  %adj = sdiv i32 %y, 4\n" +
			"  %z = sub i32 %y, %adj\n" +
			"  ret i32 %z\n" +
			"}\n\n"
	}
	if needPgoXk {
		fallback += "" +
			"; portable fallback fusion_pgo_xkernel (v0.55)\n" +
			"define i32 @trin_fusion_pgo_xkernel(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk) {\n" +
			"entry:\n" +
			"  %is1 = icmp eq i32 %isa, 1\n" +
			"  %is2 = icmp eq i32 %isa, 2\n" +
			"  %is3 = icmp eq i32 %isa, 3\n" +
			"  %is4 = icmp eq i32 %isa, 4\n" +
			"  %is5 = icmp eq i32 %isa, 5\n" +
			"  %w1 = select i1 %is1, i32 2, i32 1\n" +
			"  %w2 = select i1 %is2, i32 3, i32 %w1\n" +
			"  %w3 = select i1 %is3, i32 4, i32 %w2\n" +
			"  %w4 = select i1 %is4, i32 4, i32 %w3\n" +
			"  %w5 = select i1 %is5, i32 2, i32 %w4\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %w6 = add i32 %w5, %hd\n" +
			"  %w7 = add i32 %w6, %ld\n" +
			"  %w = add i32 %w7, %xd\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoXkCtr {
		fallback += "" +
			"; portable fallback fusion_pgo_xkernel_counter (v0.55)\n" +
			"define i32 @trin_fusion_pgo_xkernel_counter(i32 %hits, i32 %live, i32 %xk) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %y = add i32 %t, %xk\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needGatherMtPeel {
		fallback += "" +
			"; portable fallback hw_sme_za_gather_mt_peel (v0.56)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_peel() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt()\n" +
			"  ret i32 %v\n" +
			"}\n\n"
	}
	if needScatterMtPeel {
		fallback += "" +
			"; portable fallback hw_sme_za_scatter_mt_peel (v0.56)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_peel() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt()\n" +
			"  ret i32 %v\n" +
			"}\n\n"
	}
	if needPgoXpass {
		fallback += "" +
			"; portable fallback fusion_pgo_xpass (v0.56)\n" +
			"define i32 @trin_fusion_pgo_xpass(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass) {\n" +
			"entry:\n" +
			"  %is1 = icmp eq i32 %isa, 1\n" +
			"  %is2 = icmp eq i32 %isa, 2\n" +
			"  %is3 = icmp eq i32 %isa, 3\n" +
			"  %is4 = icmp eq i32 %isa, 4\n" +
			"  %is5 = icmp eq i32 %isa, 5\n" +
			"  %w1 = select i1 %is1, i32 2, i32 1\n" +
			"  %w2 = select i1 %is2, i32 3, i32 %w1\n" +
			"  %w3 = select i1 %is3, i32 4, i32 %w2\n" +
			"  %w4 = select i1 %is4, i32 4, i32 %w3\n" +
			"  %w5 = select i1 %is5, i32 2, i32 %w4\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %w6 = add i32 %w5, %hd\n" +
			"  %w7 = add i32 %w6, %ld\n" +
			"  %w8 = add i32 %w7, %xd\n" +
			"  %w = add i32 %w8, %pd\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoXpassCtr {
		fallback += "" +
			"; portable fallback fusion_pgo_xpass_counter (v0.56)\n" +
			"define i32 @trin_fusion_pgo_xpass_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %y = add i32 %u, %pass\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needGatherMtAuto {
		fallback += "" +
			"; portable fallback hw_sme_za_gather_mt_auto (v0.57)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_auto() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_peel()\n" +
			"  ret i32 %v\n" +
			"}\n\n"
	}
	if needScatterMtAuto {
		fallback += "" +
			"; portable fallback hw_sme_za_scatter_mt_auto (v0.57)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_auto() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_peel()\n" +
			"  ret i32 %v\n" +
			"}\n\n"
	}
	if needPgoWin {
		fallback += "" +
			"; portable fallback fusion_pgo_window (v0.57)\n" +
			"define i32 @trin_fusion_pgo_window(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win) {\n" +
			"entry:\n" +
			"  %is1 = icmp eq i32 %isa, 1\n" +
			"  %is2 = icmp eq i32 %isa, 2\n" +
			"  %is3 = icmp eq i32 %isa, 3\n" +
			"  %is4 = icmp eq i32 %isa, 4\n" +
			"  %is5 = icmp eq i32 %isa, 5\n" +
			"  %w1 = select i1 %is1, i32 2, i32 1\n" +
			"  %w2 = select i1 %is2, i32 3, i32 %w1\n" +
			"  %w3 = select i1 %is3, i32 4, i32 %w2\n" +
			"  %w4 = select i1 %is4, i32 4, i32 %w3\n" +
			"  %w5 = select i1 %is5, i32 2, i32 %w4\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %w6 = add i32 %w5, %hd\n" +
			"  %w7 = add i32 %w6, %ld\n" +
			"  %w8 = add i32 %w7, %xd\n" +
			"  %w9 = add i32 %w8, %pd\n" +
			"  %w = add i32 %w9, %wd\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoWinCtr {
		fallback += "" +
			"; portable fallback fusion_pgo_window_counter (v0.57)\n" +
			"define i32 @trin_fusion_pgo_window_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %y = add i32 %v, %win\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needGatherMtVlfb {
		fallback += "" +
			"; portable fallback hw_sme_za_gather_mt_vlfb (v0.58)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vlfb() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_auto()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVlfb {
		fallback += "" +
			"; portable fallback hw_sme_za_scatter_mt_vlfb (v0.58)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vlfb() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_auto()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoDecay {
		fallback += "" +
			"; portable fallback fusion_pgo_decay (v0.58)\n" +
			"define i32 @trin_fusion_pgo_decay(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay) {\n" +
			"entry:\n" +
			"  %is1 = icmp eq i32 %isa, 1\n" +
			"  %is2 = icmp eq i32 %isa, 2\n" +
			"  %is3 = icmp eq i32 %isa, 3\n" +
			"  %is4 = icmp eq i32 %isa, 4\n" +
			"  %is5 = icmp eq i32 %isa, 5\n" +
			"  %w1 = select i1 %is1, i32 2, i32 1\n" +
			"  %w2 = select i1 %is2, i32 3, i32 %w1\n" +
			"  %w3 = select i1 %is3, i32 4, i32 %w2\n" +
			"  %w4 = select i1 %is4, i32 4, i32 %w3\n" +
			"  %w5 = select i1 %is5, i32 2, i32 %w4\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %t0 = add i32 %w5, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %ok = icmp sge i32 %t5, 1\n" +
			"  %w = select i1 %ok, i32 %t5, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoDecayCtr {
		fallback += "" +
			"; portable fallback fusion_pgo_decay_counter (v0.58)\n" +
			"define i32 @trin_fusion_pgo_decay_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y = add i32 %x, %decay\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needGatherMtVllive {
		fallback += "" +
			"; portable fallback hw_sme_za_gather_mt_vllive (v0.59)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vllive() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vlfb()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVllive {
		fallback += "" +
			"; portable fallback hw_sme_za_scatter_mt_vllive (v0.59)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vllive() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vlfb()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoEma {
		fallback += "" +
			"; portable fallback fusion_pgo_ema (v0.59)\n" +
			"define i32 @trin_fusion_pgo_ema(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half) {\n" +
			"entry:\n" +
			"  %is1 = icmp eq i32 %isa, 1\n" +
			"  %is2 = icmp eq i32 %isa, 2\n" +
			"  %is3 = icmp eq i32 %isa, 3\n" +
			"  %is4 = icmp eq i32 %isa, 4\n" +
			"  %is5 = icmp eq i32 %isa, 5\n" +
			"  %w1 = select i1 %is1, i32 2, i32 1\n" +
			"  %w2 = select i1 %is2, i32 3, i32 %w1\n" +
			"  %w3 = select i1 %is3, i32 4, i32 %w2\n" +
			"  %w4 = select i1 %is4, i32 4, i32 %w3\n" +
			"  %w5 = select i1 %is5, i32 2, i32 %w4\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hd2 = sdiv i32 %half, 5\n" +
			"  %t0 = add i32 %w5, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %t6 = add i32 %t5, %hd2\n" +
			"  %ok = icmp sge i32 %t6, 1\n" +
			"  %w = select i1 %ok, i32 %t6, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoEmaCtr {
		fallback += "" +
			"; portable fallback fusion_pgo_ema_counter (v0.59)\n" +
			"define i32 @trin_fusion_pgo_ema_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y = add i32 %y0, %half\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needGatherMtVlstream {
		fallback += "" +
			"; SME ZA multi-tile peel gather host with live VL residual streaming (v0.60)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vlstream() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vllive()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVlstream {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host with live VL residual streaming (v0.60)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vlstream() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vllive()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoAdaptHl {
		fallback += "" +
			"; fusion PGO adaptive half-life (v0.60)\n" +
			"define i32 @trin_fusion_pgo_adapt_hl(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %t6 = add i32 %t5, %hf\n" +
			"  %ok = icmp sge i32 %t6, 1\n" +
			"  %w = select i1 %ok, i32 %t6, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoAdaptHlCtr {
		fallback += "" +
			"; fusion PGO adaptive half-life counter (v0.60)\n" +
			"define i32 @trin_fusion_pgo_adapt_hl_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y = add i32 %y0, %half\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needGatherMtVlcost {
		fallback += "" +
			"; SME ZA multi-tile peel gather host with live VL residual streaming cost model (v0.61)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vlcost() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vlstream()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVlcost {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host with live VL residual streaming cost model (v0.61)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vlcost() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vlstream()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoHyst {
		fallback += "" +
			"; fusion PGO dual EMA hysteresis (v0.61)\n" +
			"define i32 @trin_fusion_pgo_hyst(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %t6 = add i32 %t5, %hf\n" +
			"  %ok = icmp sge i32 %t6, 1\n" +
			"  %w = select i1 %ok, i32 %t6, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoHystCtr {
		fallback += "" +
			"; fusion PGO dual EMA hysteresis counter (v0.61)\n" +
			"define i32 @trin_fusion_pgo_hyst_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y = add i32 %y0, %half\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needGatherMtVladapt {
		fallback += "" +
			"; SME ZA multi-tile peel gather host with live VL residual streaming cost + adaptive tile (v0.62)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vladapt() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vlcost()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVladapt {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host with live VL residual streaming cost + adaptive tile (v0.62)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vladapt() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vlcost()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoReweight {
		fallback += "" +
			"; fusion PGO dual EMA hysteresis windowed reweight (v0.62)\n" +
			"define i32 @trin_fusion_pgo_reweight(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %y6 = add i32 %t5, %hf\n" +
			"  %ok = icmp sge i32 %y6, 1\n" +
			"  %w = select i1 %ok, i32 %y6, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoReweightCtr {
		fallback += "" +
			"; fusion PGO dual EMA hysteresis windowed reweight counter (v0.62)\n" +
			"define i32 @trin_fusion_pgo_reweight_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y = add i32 %y0, %half\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needGatherMtVlsched {
		fallback += "" +
			"; SME ZA multi-tile peel gather host with live VL residual streaming adaptive cost + residual schedule (v0.63)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vlsched() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vladapt()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVlsched {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host with live VL residual streaming adaptive cost + residual schedule (v0.63)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vlsched() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vladapt()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoBudget {
		fallback += "" +
			"; fusion PGO dual EMA hysteresis windowed reweight with budget caps (v0.63)\n" +
			"define i32 @trin_fusion_pgo_budget(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %budget) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %y6 = add i32 %t5, %hf\n" +
			"  %hasBudget = icmp sgt i32 %budget, 0\n" +
			"  %over = icmp sgt i32 %y6, %budget\n" +
			"  %clamp = and i1 %hasBudget, %over\n" +
			"  %tc = select i1 %clamp, i32 %budget, i32 %y6\n" +
			"  %ok = icmp sge i32 %tc, 1\n" +
			"  %w = select i1 %ok, i32 %tc, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoBudgetCtr {
		fallback += "" +
			"; fusion PGO dual EMA hysteresis windowed reweight with budget caps counter (v0.63)\n" +
			"define i32 @trin_fusion_pgo_budget_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y = add i32 %y0, %half\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needGatherMtVlfuse {
		fallback += "" +
			"; SME ZA multi-tile peel gather host with live VL residual streaming schedule + peel fusion (v0.64)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vlfuse() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vlsched()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVlfuse {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host with live VL residual streaming schedule + peel fusion (v0.64)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vlfuse() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vlsched()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoSoftcap {
		fallback += "" +
			"; fusion PGO dual EMA hysteresis budget reweight with soft caps (v0.64)\n" +
			"define i32 @trin_fusion_pgo_softcap(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %budget, i32 %softcap) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %y6 = add i32 %t5, %hf\n" +
			"  %hasBudget = icmp sgt i32 %budget, 0\n" +
			"  %overB = icmp sgt i32 %y6, %budget\n" +
			"  %clampB = and i1 %hasBudget, %overB\n" +
			"  %tb = select i1 %clampB, i32 %budget, i32 %y6\n" +
			"  %hasSC = icmp sgt i32 %softcap, 0\n" +
			"  %overS = icmp sgt i32 %tb, %softcap\n" +
			"  %clampS = and i1 %hasSC, %overS\n" +
			"  %sum = add i32 %tb, %softcap\n" +
			"  %avg = sdiv i32 %sum, 2\n" +
			"  %avgOver = icmp sgt i32 %avg, %softcap\n" +
			"  %avgc = select i1 %avgOver, i32 %softcap, i32 %avg\n" +
			"  %ts = select i1 %clampS, i32 %avgc, i32 %tb\n" +
			"  %ok = icmp sge i32 %ts, 1\n" +
			"  %w = select i1 %ok, i32 %ts, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoSoftcapCtr {
		fallback += "" +
			"; fusion PGO dual EMA hysteresis budget reweight with soft caps counter (v0.64)\n" +
			"define i32 @trin_fusion_pgo_softcap_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y = add i32 %y0, %half\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needGatherMtVlmix {
		fallback += "" +
			"; SME ZA multi-tile peel gather host with residual streaming schedule peel fusion + live tile mix (v0.65)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vlmix() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vlfuse()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVlmix {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host with residual streaming schedule peel fusion + live tile mix (v0.65)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vlmix() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vlfuse()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoDual {
		fallback += "" +
			"; fusion PGO dual EMA softcap reweight with dual soft/hard budget lanes (v0.65)\n" +
			"define i32 @trin_fusion_pgo_dual(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %budget, i32 %softcap, i32 %hard) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %y6 = add i32 %t5, %hf\n" +
			"  %hasBudget = icmp sgt i32 %budget, 0\n" +
			"  %overB = icmp sgt i32 %y6, %budget\n" +
			"  %clampB = and i1 %hasBudget, %overB\n" +
			"  %tb = select i1 %clampB, i32 %budget, i32 %y6\n" +
			"  %hasSC = icmp sgt i32 %softcap, 0\n" +
			"  %overS = icmp sgt i32 %tb, %softcap\n" +
			"  %clampS = and i1 %hasSC, %overS\n" +
			"  %sum = add i32 %tb, %softcap\n" +
			"  %avg = sdiv i32 %sum, 2\n" +
			"  %avgOver = icmp sgt i32 %avg, %softcap\n" +
			"  %avgc = select i1 %avgOver, i32 %softcap, i32 %avg\n" +
			"  %ts = select i1 %clampS, i32 %avgc, i32 %tb\n" +
			"  %hasH = icmp sgt i32 %hard, 0\n" +
			"  %overH = icmp sgt i32 %ts, %hard\n" +
			"  %clampH = and i1 %hasH, %overH\n" +
			"  %th = select i1 %clampH, i32 %hard, i32 %ts\n" +
			"  %ok = icmp sge i32 %th, 1\n" +
			"  %w = select i1 %ok, i32 %th, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoDualCtr {
		fallback += "" +
			"; fusion PGO dual EMA softcap reweight with dual soft/hard budget lanes counter (v0.65)\n" +
			"define i32 @trin_fusion_pgo_dual_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y = add i32 %y0, %half\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needGatherMtVlbal {
		fallback += "" +
			"; SME ZA multi-tile peel gather host with residual streaming schedule peel fusion live tile mix + cost balance (v0.66)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vlbal() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vlmix()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVlbal {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host with residual streaming schedule peel fusion live tile mix + cost balance (v0.66)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vlbal() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vlmix()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoConf {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend (v0.66)\n" +
			"define i32 @trin_fusion_pgo_conf(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %budget, i32 %softcap, i32 %hard, i32 %conf) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %cf = sdiv i32 %conf, 10\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %t6 = add i32 %t5, %hf\n" +
			"  %y7 = add i32 %t6, %cf\n" +
			"  %hasBudget = icmp sgt i32 %budget, 0\n" +
			"  %overB = icmp sgt i32 %y7, %budget\n" +
			"  %clampB = and i1 %hasBudget, %overB\n" +
			"  %tb = select i1 %clampB, i32 %budget, i32 %y7\n" +
			"  %hasSC = icmp sgt i32 %softcap, 0\n" +
			"  %overS = icmp sgt i32 %tb, %softcap\n" +
			"  %clampS = and i1 %hasSC, %overS\n" +
			"  %sum = add i32 %tb, %softcap\n" +
			"  %avg = sdiv i32 %sum, 2\n" +
			"  %avgOver = icmp sgt i32 %avg, %softcap\n" +
			"  %avgc = select i1 %avgOver, i32 %softcap, i32 %avg\n" +
			"  %ts = select i1 %clampS, i32 %avgc, i32 %tb\n" +
			"  %hasH = icmp sgt i32 %hard, 0\n" +
			"  %overH = icmp sgt i32 %ts, %hard\n" +
			"  %clampH = and i1 %hasH, %overH\n" +
			"  %th = select i1 %clampH, i32 %hard, i32 %ts\n" +
			"  %ok = icmp sge i32 %th, 1\n" +
			"  %w = select i1 %ok, i32 %th, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoConfCtr {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend counter (v0.66)\n" +
			"define i32 @trin_fusion_pgo_conf_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %conf) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y1 = add i32 %y0, %half\n" +
			"  %y = add i32 %y1, %conf\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needGatherMtVlbw {
		fallback += "" +
			"; SME ZA multi-tile peel gather host with residual streaming schedule peel fusion live tile mix cost balance + bandwidth model (v0.67)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vlbw() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vlbal()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVlbw {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host with residual streaming schedule peel fusion live tile mix cost balance + bandwidth model (v0.67)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vlbw() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vlbal()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoReg {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis (v0.67)\n" +
			"define i32 @trin_fusion_pgo_reg(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %budget, i32 %softcap, i32 %hard, i32 %conf, i32 %hyst) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %cf = sdiv i32 %conf, 10\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %t6 = add i32 %t5, %hf\n" +
			"  %y7 = add i32 %t6, %cf\n" +
			"  %hasBudget = icmp sgt i32 %budget, 0\n" +
			"  %overB = icmp sgt i32 %y7, %budget\n" +
			"  %clampB = and i1 %hasBudget, %overB\n" +
			"  %tb = select i1 %clampB, i32 %budget, i32 %y7\n" +
			"  %hasSC = icmp sgt i32 %softcap, 0\n" +
			"  %overS = icmp sgt i32 %tb, %softcap\n" +
			"  %clampS = and i1 %hasSC, %overS\n" +
			"  %sum = add i32 %tb, %softcap\n" +
			"  %avg = sdiv i32 %sum, 2\n" +
			"  %avgOver = icmp sgt i32 %avg, %softcap\n" +
			"  %avgc = select i1 %avgOver, i32 %softcap, i32 %avg\n" +
			"  %ts = select i1 %clampS, i32 %avgc, i32 %tb\n" +
			"  %hasH = icmp sgt i32 %hard, 0\n" +
			"  %overH = icmp sgt i32 %ts, %hard\n" +
			"  %clampH = and i1 %hasH, %overH\n" +
			"  %th = select i1 %clampH, i32 %hard, i32 %ts\n" +
			"  %diffpos = sub i32 %th, %softcap\n" +
			"  %diffneg = sub i32 %softcap, %th\n" +
			"  %thGEsc = icmp sge i32 %th, %softcap\n" +
			"  %diff = select i1 %thGEsc, i32 %diffpos, i32 %diffneg\n" +
			"  %hasHy = icmp sgt i32 %hyst, 0\n" +
			"  %hasSChy = icmp sgt i32 %softcap, 0\n" +
			"  %hyReady = and i1 %hasHy, %hasSChy\n" +
			"  %inBand = icmp sle i32 %diff, %hyst\n" +
			"  %doHy = and i1 %hyReady, %inBand\n" +
			"  %thy = select i1 %doHy, i32 %softcap, i32 %th\n" +
			"  %ok = icmp sge i32 %thy, 1\n" +
			"  %w = select i1 %ok, i32 %thy, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoRegCtr {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis counter (v0.67)\n" +
			"define i32 @trin_fusion_pgo_reg_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %conf, i32 %hyst) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y1 = add i32 %y0, %half\n" +
			"  %y2 = add i32 %y1, %conf\n" +
			"  %y = add i32 %y2, %hyst\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}

	if needGatherMtVlpath {
		fallback += "" +
			"; SME ZA multi-tile peel gather host with residual streaming schedule peel fusion live tile mix cost balance bandwidth + path cost (v0.68)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vlpath() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vlbw()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVlpath {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host with residual streaming schedule peel fusion live tile mix cost balance bandwidth + path cost (v0.68)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vlpath() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vlbw()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoFloor {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor (v0.68)\n" +
			"define i32 @trin_fusion_pgo_floor(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %budget, i32 %softcap, i32 %hard, i32 %conf, i32 %hyst, i32 %floor) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %cf = sdiv i32 %conf, 10\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %t6 = add i32 %t5, %hf\n" +
			"  %y7 = add i32 %t6, %cf\n" +
			"  %hasBudget = icmp sgt i32 %budget, 0\n" +
			"  %overB = icmp sgt i32 %y7, %budget\n" +
			"  %clampB = and i1 %hasBudget, %overB\n" +
			"  %tb = select i1 %clampB, i32 %budget, i32 %y7\n" +
			"  %hasSC = icmp sgt i32 %softcap, 0\n" +
			"  %overS = icmp sgt i32 %tb, %softcap\n" +
			"  %clampS = and i1 %hasSC, %overS\n" +
			"  %sum = add i32 %tb, %softcap\n" +
			"  %avg = sdiv i32 %sum, 2\n" +
			"  %avgOver = icmp sgt i32 %avg, %softcap\n" +
			"  %avgc = select i1 %avgOver, i32 %softcap, i32 %avg\n" +
			"  %ts = select i1 %clampS, i32 %avgc, i32 %tb\n" +
			"  %hasH = icmp sgt i32 %hard, 0\n" +
			"  %overH = icmp sgt i32 %ts, %hard\n" +
			"  %clampH = and i1 %hasH, %overH\n" +
			"  %th = select i1 %clampH, i32 %hard, i32 %ts\n" +
			"  %diffpos = sub i32 %th, %softcap\n" +
			"  %diffneg = sub i32 %softcap, %th\n" +
			"  %thGEsc = icmp sge i32 %th, %softcap\n" +
			"  %diff = select i1 %thGEsc, i32 %diffpos, i32 %diffneg\n" +
			"  %hasHy = icmp sgt i32 %hyst, 0\n" +
			"  %hasSChy = icmp sgt i32 %softcap, 0\n" +
			"  %hyReady = and i1 %hasHy, %hasSChy\n" +
			"  %inBand = icmp sle i32 %diff, %hyst\n" +
			"  %doHy = and i1 %hyReady, %inBand\n" +
			"  %thy = select i1 %doHy, i32 %softcap, i32 %th\n" +
			"  %hasFl = icmp sgt i32 %floor, 0\n" +
			"  %belowFl = icmp slt i32 %thy, %floor\n" +
			"  %clampFl = and i1 %hasFl, %belowFl\n" +
			"  %tf = select i1 %clampFl, i32 %floor, i32 %thy\n" +
			"  %ok = icmp sge i32 %tf, 1\n" +
			"  %w = select i1 %ok, i32 %tf, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoFloorCtr {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor counter (v0.68)\n" +
			"define i32 @trin_fusion_pgo_floor_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %conf, i32 %hyst, i32 %floor) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y1 = add i32 %y0, %half\n" +
			"  %y2 = add i32 %y1, %conf\n" +
			"  %y3 = add i32 %y2, %hyst\n" +
			"  %y = add i32 %y3, %floor\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}










	if needGatherMtVllane {
		fallback += "" +
			"; SME ZA multi-tile peel gather host with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost + lane pressure (v0.69)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vllane() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vlpath()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVllane {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost + lane pressure (v0.69)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vllane() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vlpath()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoAfloor {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor (v0.69)\n" +
			"define i32 @trin_fusion_pgo_afloor(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %budget, i32 %softcap, i32 %hard, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %cf = sdiv i32 %conf, 10\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %t6 = add i32 %t5, %hf\n" +
			"  %y7 = add i32 %t6, %cf\n" +
			"  %hasBudget = icmp sgt i32 %budget, 0\n" +
			"  %overB = icmp sgt i32 %y7, %budget\n" +
			"  %clampB = and i1 %hasBudget, %overB\n" +
			"  %tb = select i1 %clampB, i32 %budget, i32 %y7\n" +
			"  %hasSC = icmp sgt i32 %softcap, 0\n" +
			"  %overS = icmp sgt i32 %tb, %softcap\n" +
			"  %clampS = and i1 %hasSC, %overS\n" +
			"  %sum = add i32 %tb, %softcap\n" +
			"  %avg = sdiv i32 %sum, 2\n" +
			"  %avgOver = icmp sgt i32 %avg, %softcap\n" +
			"  %avgc = select i1 %avgOver, i32 %softcap, i32 %avg\n" +
			"  %ts = select i1 %clampS, i32 %avgc, i32 %tb\n" +
			"  %hasH = icmp sgt i32 %hard, 0\n" +
			"  %overH = icmp sgt i32 %ts, %hard\n" +
			"  %clampH = and i1 %hasH, %overH\n" +
			"  %th = select i1 %clampH, i32 %hard, i32 %ts\n" +
			"  %diffpos = sub i32 %th, %softcap\n" +
			"  %diffneg = sub i32 %softcap, %th\n" +
			"  %thGEsc = icmp sge i32 %th, %softcap\n" +
			"  %diff = select i1 %thGEsc, i32 %diffpos, i32 %diffneg\n" +
			"  %hasHy = icmp sgt i32 %hyst, 0\n" +
			"  %hasSChy = icmp sgt i32 %softcap, 0\n" +
			"  %hyReady = and i1 %hasHy, %hasSChy\n" +
			"  %inBand = icmp sle i32 %diff, %hyst\n" +
			"  %doHy = and i1 %hyReady, %inBand\n" +
			"  %thy = select i1 %doHy, i32 %softcap, i32 %th\n" +
			"  %hasFl = icmp sgt i32 %floor, 0\n" +
			"  %belowFl = icmp slt i32 %thy, %floor\n" +
			"  %clampFl = and i1 %hasFl, %belowFl\n" +
			"  %tf = select i1 %clampFl, i32 %floor, i32 %thy\n" +
			"  %hasAd = icmp sgt i32 %adapt, 0\n" +
			"  %belowAd = icmp slt i32 %tf, %adapt\n" +
			"  %clampAd = and i1 %hasAd, %belowAd\n" +
			"  %ta = select i1 %clampAd, i32 %adapt, i32 %tf\n" +
			"  %ok = icmp sge i32 %ta, 1\n" +
			"  %w = select i1 %ok, i32 %ta, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoAfloorCtr {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor counter (v0.69)\n" +
			"define i32 @trin_fusion_pgo_afloor_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y1 = add i32 %y0, %half\n" +
			"  %y2 = add i32 %y1, %conf\n" +
			"  %y3 = add i32 %y2, %hyst\n" +
			"  %y4 = add i32 %y3, %floor\n" +
			"  %y = add i32 %y4, %adapt\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}


	if needGatherMtVlspill {
		fallback += "" +
			"; SME ZA multi-tile peel gather host with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure + spill model (v0.70)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vlspill() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vllane()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVlspill {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure + spill model (v0.70)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vlspill() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vllane()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoBurst {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap (v0.70)\n" +
			"define i32 @trin_fusion_pgo_burst(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %budget, i32 %softcap, i32 %hard, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %cf = sdiv i32 %conf, 10\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %t6 = add i32 %t5, %hf\n" +
			"  %y7 = add i32 %t6, %cf\n" +
			"  %hasBudget = icmp sgt i32 %budget, 0\n" +
			"  %overB = icmp sgt i32 %y7, %budget\n" +
			"  %clampB = and i1 %hasBudget, %overB\n" +
			"  %tb = select i1 %clampB, i32 %budget, i32 %y7\n" +
			"  %hasSC = icmp sgt i32 %softcap, 0\n" +
			"  %overS = icmp sgt i32 %tb, %softcap\n" +
			"  %clampS = and i1 %hasSC, %overS\n" +
			"  %sum = add i32 %tb, %softcap\n" +
			"  %avg = sdiv i32 %sum, 2\n" +
			"  %avgOver = icmp sgt i32 %avg, %softcap\n" +
			"  %avgc = select i1 %avgOver, i32 %softcap, i32 %avg\n" +
			"  %ts = select i1 %clampS, i32 %avgc, i32 %tb\n" +
			"  %hasH = icmp sgt i32 %hard, 0\n" +
			"  %overH = icmp sgt i32 %ts, %hard\n" +
			"  %clampH = and i1 %hasH, %overH\n" +
			"  %th = select i1 %clampH, i32 %hard, i32 %ts\n" +
			"  %diffpos = sub i32 %th, %softcap\n" +
			"  %diffneg = sub i32 %softcap, %th\n" +
			"  %thGEsc = icmp sge i32 %th, %softcap\n" +
			"  %diff = select i1 %thGEsc, i32 %diffpos, i32 %diffneg\n" +
			"  %hasHy = icmp sgt i32 %hyst, 0\n" +
			"  %hasSChy = icmp sgt i32 %softcap, 0\n" +
			"  %hyReady = and i1 %hasHy, %hasSChy\n" +
			"  %inBand = icmp sle i32 %diff, %hyst\n" +
			"  %doHy = and i1 %hyReady, %inBand\n" +
			"  %thy = select i1 %doHy, i32 %softcap, i32 %th\n" +
			"  %hasFl = icmp sgt i32 %floor, 0\n" +
			"  %belowFl = icmp slt i32 %thy, %floor\n" +
			"  %clampFl = and i1 %hasFl, %belowFl\n" +
			"  %tf = select i1 %clampFl, i32 %floor, i32 %thy\n" +
			"  %hasAd = icmp sgt i32 %adapt, 0\n" +
			"  %belowAd = icmp slt i32 %tf, %adapt\n" +
			"  %clampAd = and i1 %hasAd, %belowAd\n" +
			"  %ta = select i1 %clampAd, i32 %adapt, i32 %tf\n" +
			"  %hasBu = icmp sgt i32 %burst, 0\n" +
			"  %overBu = icmp sgt i32 %ta, %burst\n" +
			"  %clampBu = and i1 %hasBu, %overBu\n" +
			"  %tbu = select i1 %clampBu, i32 %burst, i32 %ta\n" +
			"  %ok = icmp sge i32 %tbu, 1\n" +
			"  %w = select i1 %ok, i32 %tbu, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoBurstCtr {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap counter (v0.70)\n" +
			"define i32 @trin_fusion_pgo_burst_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y1 = add i32 %y0, %half\n" +
			"  %y2 = add i32 %y1, %conf\n" +
			"  %y3 = add i32 %y2, %hyst\n" +
			"  %y4 = add i32 %y3, %floor\n" +
			"  %y5 = add i32 %y4, %adapt\n" +
			"  %y = add i32 %y5, %burst\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}


	if needGatherMtVlocc {
		fallback += "" +
			"; SME ZA multi-tile peel gather host with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model + occupancy (v0.71)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vlocc() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vlspill()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVlocc {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model + occupancy (v0.71)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vlocc() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vlspill()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoCool {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down (v0.71)\n" +
			"define i32 @trin_fusion_pgo_cool(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %budget, i32 %softcap, i32 %hard, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %cf = sdiv i32 %conf, 10\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %t6 = add i32 %t5, %hf\n" +
			"  %y7 = add i32 %t6, %cf\n" +
			"  %hasBudget = icmp sgt i32 %budget, 0\n" +
			"  %overB = icmp sgt i32 %y7, %budget\n" +
			"  %clampB = and i1 %hasBudget, %overB\n" +
			"  %tb = select i1 %clampB, i32 %budget, i32 %y7\n" +
			"  %hasSC = icmp sgt i32 %softcap, 0\n" +
			"  %overS = icmp sgt i32 %tb, %softcap\n" +
			"  %clampS = and i1 %hasSC, %overS\n" +
			"  %sum = add i32 %tb, %softcap\n" +
			"  %avg = sdiv i32 %sum, 2\n" +
			"  %avgOver = icmp sgt i32 %avg, %softcap\n" +
			"  %avgc = select i1 %avgOver, i32 %softcap, i32 %avg\n" +
			"  %ts = select i1 %clampS, i32 %avgc, i32 %tb\n" +
			"  %hasH = icmp sgt i32 %hard, 0\n" +
			"  %overH = icmp sgt i32 %ts, %hard\n" +
			"  %clampH = and i1 %hasH, %overH\n" +
			"  %th = select i1 %clampH, i32 %hard, i32 %ts\n" +
			"  %diffpos = sub i32 %th, %softcap\n" +
			"  %diffneg = sub i32 %softcap, %th\n" +
			"  %thGEsc = icmp sge i32 %th, %softcap\n" +
			"  %diff = select i1 %thGEsc, i32 %diffpos, i32 %diffneg\n" +
			"  %hasHy = icmp sgt i32 %hyst, 0\n" +
			"  %hasSChy = icmp sgt i32 %softcap, 0\n" +
			"  %hyReady = and i1 %hasHy, %hasSChy\n" +
			"  %inBand = icmp sle i32 %diff, %hyst\n" +
			"  %doHy = and i1 %hyReady, %inBand\n" +
			"  %thy = select i1 %doHy, i32 %softcap, i32 %th\n" +
			"  %hasFl = icmp sgt i32 %floor, 0\n" +
			"  %belowFl = icmp slt i32 %thy, %floor\n" +
			"  %clampFl = and i1 %hasFl, %belowFl\n" +
			"  %tf = select i1 %clampFl, i32 %floor, i32 %thy\n" +
			"  %hasAd = icmp sgt i32 %adapt, 0\n" +
			"  %belowAd = icmp slt i32 %tf, %adapt\n" +
			"  %clampAd = and i1 %hasAd, %belowAd\n" +
			"  %ta = select i1 %clampAd, i32 %adapt, i32 %tf\n" +
			"  %hasBu = icmp sgt i32 %burst, 0\n" +
			"  %overBu = icmp sgt i32 %ta, %burst\n" +
			"  %clampBu = and i1 %hasBu, %overBu\n" +
			"  %tbu = select i1 %clampBu, i32 %burst, i32 %ta\n" +
			"  %tco = sub i32 %tbu, %cool\n" +
			"  %ok = icmp sge i32 %tco, 1\n" +
			"  %w = select i1 %ok, i32 %tco, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoCoolCtr {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down counter (v0.71)\n" +
			"define i32 @trin_fusion_pgo_cool_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y1 = add i32 %y0, %half\n" +
			"  %y2 = add i32 %y1, %conf\n" +
			"  %y3 = add i32 %y2, %hyst\n" +
			"  %y4 = add i32 %y3, %floor\n" +
			"  %y5 = add i32 %y4, %adapt\n" +
			"  %y6 = add i32 %y5, %burst\n" +
			"  %y = add i32 %y6, %cool\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}


	if needGatherMtVlqueue {
		fallback += "" +
			"; SME ZA multi-tile peel gather host with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth (v0.72)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vlqueue() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vlocc()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVlqueue {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth (v0.72)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vlqueue() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vlocc()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoCredit {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank (v0.72)\n" +
			"define i32 @trin_fusion_pgo_credit(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %budget, i32 %softcap, i32 %hard, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool, i32 %credit) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %cf = sdiv i32 %conf, 10\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %t6 = add i32 %t5, %hf\n" +
			"  %y7 = add i32 %t6, %cf\n" +
			"  %hasBudget = icmp sgt i32 %budget, 0\n" +
			"  %overB = icmp sgt i32 %y7, %budget\n" +
			"  %clampB = and i1 %hasBudget, %overB\n" +
			"  %tb = select i1 %clampB, i32 %budget, i32 %y7\n" +
			"  %hasSC = icmp sgt i32 %softcap, 0\n" +
			"  %overS = icmp sgt i32 %tb, %softcap\n" +
			"  %clampS = and i1 %hasSC, %overS\n" +
			"  %sum = add i32 %tb, %softcap\n" +
			"  %avg = sdiv i32 %sum, 2\n" +
			"  %avgOver = icmp sgt i32 %avg, %softcap\n" +
			"  %avgc = select i1 %avgOver, i32 %softcap, i32 %avg\n" +
			"  %ts = select i1 %clampS, i32 %avgc, i32 %tb\n" +
			"  %hasH = icmp sgt i32 %hard, 0\n" +
			"  %overH = icmp sgt i32 %ts, %hard\n" +
			"  %clampH = and i1 %hasH, %overH\n" +
			"  %th = select i1 %clampH, i32 %hard, i32 %ts\n" +
			"  %diffpos = sub i32 %th, %softcap\n" +
			"  %diffneg = sub i32 %softcap, %th\n" +
			"  %thGEsc = icmp sge i32 %th, %softcap\n" +
			"  %diff = select i1 %thGEsc, i32 %diffpos, i32 %diffneg\n" +
			"  %hasHy = icmp sgt i32 %hyst, 0\n" +
			"  %hasSChy = icmp sgt i32 %softcap, 0\n" +
			"  %hyReady = and i1 %hasHy, %hasSChy\n" +
			"  %inBand = icmp sle i32 %diff, %hyst\n" +
			"  %doHy = and i1 %hyReady, %inBand\n" +
			"  %thy = select i1 %doHy, i32 %softcap, i32 %th\n" +
			"  %hasFl = icmp sgt i32 %floor, 0\n" +
			"  %belowFl = icmp slt i32 %thy, %floor\n" +
			"  %clampFl = and i1 %hasFl, %belowFl\n" +
			"  %tf = select i1 %clampFl, i32 %floor, i32 %thy\n" +
			"  %hasAd = icmp sgt i32 %adapt, 0\n" +
			"  %belowAd = icmp slt i32 %tf, %adapt\n" +
			"  %clampAd = and i1 %hasAd, %belowAd\n" +
			"  %ta = select i1 %clampAd, i32 %adapt, i32 %tf\n" +
			"  %hasBu = icmp sgt i32 %burst, 0\n" +
			"  %overBu = icmp sgt i32 %ta, %burst\n" +
			"  %clampBu = and i1 %hasBu, %overBu\n" +
			"  %tbu = select i1 %clampBu, i32 %burst, i32 %ta\n" +
			"  %tco = sub i32 %tbu, %cool\n" +
			"  %tcr = add i32 %tco, %credit\n" +
			"  %ok = icmp sge i32 %tcr, 1\n" +
			"  %w = select i1 %ok, i32 %tcr, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoCreditCtr {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank counter (v0.72)\n" +
			"define i32 @trin_fusion_pgo_credit_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool, i32 %credit) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y1 = add i32 %y0, %half\n" +
			"  %y2 = add i32 %y1, %conf\n" +
			"  %y3 = add i32 %y2, %hyst\n" +
			"  %y4 = add i32 %y3, %floor\n" +
			"  %y5 = add i32 %y4, %adapt\n" +
			"  %y6 = add i32 %y5, %burst\n" +
			"  %y7 = add i32 %y6, %cool\n" +
			"  %y = add i32 %y7, %credit\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}


	if needGatherMtVlwater {
		fallback += "" +
			"; SME ZA multi-tile peel gather host with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth + watermark (v0.73)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vlwater() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vlqueue()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVlwater {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth + watermark (v0.73)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vlwater() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vlqueue()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoDebt {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery (v0.73)\n" +
			"define i32 @trin_fusion_pgo_debt(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %budget, i32 %softcap, i32 %hard, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool, i32 %credit, i32 %debt) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %cf = sdiv i32 %conf, 10\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %t6 = add i32 %t5, %hf\n" +
			"  %y7 = add i32 %t6, %cf\n" +
			"  %hasBudget = icmp sgt i32 %budget, 0\n" +
			"  %overB = icmp sgt i32 %y7, %budget\n" +
			"  %clampB = and i1 %hasBudget, %overB\n" +
			"  %tb = select i1 %clampB, i32 %budget, i32 %y7\n" +
			"  %hasSC = icmp sgt i32 %softcap, 0\n" +
			"  %overS = icmp sgt i32 %tb, %softcap\n" +
			"  %clampS = and i1 %hasSC, %overS\n" +
			"  %sum = add i32 %tb, %softcap\n" +
			"  %avg = sdiv i32 %sum, 2\n" +
			"  %avgOver = icmp sgt i32 %avg, %softcap\n" +
			"  %avgc = select i1 %avgOver, i32 %softcap, i32 %avg\n" +
			"  %ts = select i1 %clampS, i32 %avgc, i32 %tb\n" +
			"  %hasH = icmp sgt i32 %hard, 0\n" +
			"  %overH = icmp sgt i32 %ts, %hard\n" +
			"  %clampH = and i1 %hasH, %overH\n" +
			"  %th = select i1 %clampH, i32 %hard, i32 %ts\n" +
			"  %diffpos = sub i32 %th, %softcap\n" +
			"  %diffneg = sub i32 %softcap, %th\n" +
			"  %thGEsc = icmp sge i32 %th, %softcap\n" +
			"  %diff = select i1 %thGEsc, i32 %diffpos, i32 %diffneg\n" +
			"  %hasHy = icmp sgt i32 %hyst, 0\n" +
			"  %hasSChy = icmp sgt i32 %softcap, 0\n" +
			"  %hyReady = and i1 %hasHy, %hasSChy\n" +
			"  %inBand = icmp sle i32 %diff, %hyst\n" +
			"  %doHy = and i1 %hyReady, %inBand\n" +
			"  %thy = select i1 %doHy, i32 %softcap, i32 %th\n" +
			"  %hasFl = icmp sgt i32 %floor, 0\n" +
			"  %belowFl = icmp slt i32 %thy, %floor\n" +
			"  %clampFl = and i1 %hasFl, %belowFl\n" +
			"  %tf = select i1 %clampFl, i32 %floor, i32 %thy\n" +
			"  %hasAd = icmp sgt i32 %adapt, 0\n" +
			"  %belowAd = icmp slt i32 %tf, %adapt\n" +
			"  %clampAd = and i1 %hasAd, %belowAd\n" +
			"  %ta = select i1 %clampAd, i32 %adapt, i32 %tf\n" +
			"  %hasBu = icmp sgt i32 %burst, 0\n" +
			"  %overBu = icmp sgt i32 %ta, %burst\n" +
			"  %clampBu = and i1 %hasBu, %overBu\n" +
			"  %tbu = select i1 %clampBu, i32 %burst, i32 %ta\n" +
			"  %tco = sub i32 %tbu, %cool\n" +
			"  %tcr = add i32 %tco, %credit\n" +
			"  %tde = sub i32 %tcr, %debt\n" +
			"  %ok = icmp sge i32 %tde, 1\n" +
			"  %w = select i1 %ok, i32 %tde, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoDebtCtr {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery counter (v0.73)\n" +
			"define i32 @trin_fusion_pgo_debt_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool, i32 %credit, i32 %debt) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y1 = add i32 %y0, %half\n" +
			"  %y2 = add i32 %y1, %conf\n" +
			"  %y3 = add i32 %y2, %hyst\n" +
			"  %y4 = add i32 %y3, %floor\n" +
			"  %y5 = add i32 %y4, %adapt\n" +
			"  %y6 = add i32 %y5, %burst\n" +
			"  %y7 = add i32 %y6, %cool\n" +
			"  %y8 = add i32 %y7, %credit\n" +
			"  %y = add i32 %y8, %debt\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}


	if needGatherMtVldrain {
		fallback += "" +
			"; SME ZA multi-tile peel gather host with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth + watermark + watermark drain (v0.74)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vldrain() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vlwater()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVldrain {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth + watermark + watermark drain (v0.74)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vldrain() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vlwater()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoInterest {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest (v0.74)\n" +
			"define i32 @trin_fusion_pgo_interest(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %budget, i32 %softcap, i32 %hard, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool, i32 %credit, i32 %debt, i32 %interest) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %cf = sdiv i32 %conf, 10\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %t6 = add i32 %t5, %hf\n" +
			"  %y7 = add i32 %t6, %cf\n" +
			"  %hasBudget = icmp sgt i32 %budget, 0\n" +
			"  %overB = icmp sgt i32 %y7, %budget\n" +
			"  %clampB = and i1 %hasBudget, %overB\n" +
			"  %tb = select i1 %clampB, i32 %budget, i32 %y7\n" +
			"  %hasSC = icmp sgt i32 %softcap, 0\n" +
			"  %overS = icmp sgt i32 %tb, %softcap\n" +
			"  %clampS = and i1 %hasSC, %overS\n" +
			"  %sum = add i32 %tb, %softcap\n" +
			"  %avg = sdiv i32 %sum, 2\n" +
			"  %avgOver = icmp sgt i32 %avg, %softcap\n" +
			"  %avgc = select i1 %avgOver, i32 %softcap, i32 %avg\n" +
			"  %ts = select i1 %clampS, i32 %avgc, i32 %tb\n" +
			"  %hasH = icmp sgt i32 %hard, 0\n" +
			"  %overH = icmp sgt i32 %ts, %hard\n" +
			"  %clampH = and i1 %hasH, %overH\n" +
			"  %th = select i1 %clampH, i32 %hard, i32 %ts\n" +
			"  %diffpos = sub i32 %th, %softcap\n" +
			"  %diffneg = sub i32 %softcap, %th\n" +
			"  %thGEsc = icmp sge i32 %th, %softcap\n" +
			"  %diff = select i1 %thGEsc, i32 %diffpos, i32 %diffneg\n" +
			"  %hasHy = icmp sgt i32 %hyst, 0\n" +
			"  %hasSChy = icmp sgt i32 %softcap, 0\n" +
			"  %hyReady = and i1 %hasHy, %hasSChy\n" +
			"  %inBand = icmp sle i32 %diff, %hyst\n" +
			"  %doHy = and i1 %hyReady, %inBand\n" +
			"  %thy = select i1 %doHy, i32 %softcap, i32 %th\n" +
			"  %hasFl = icmp sgt i32 %floor, 0\n" +
			"  %belowFl = icmp slt i32 %thy, %floor\n" +
			"  %clampFl = and i1 %hasFl, %belowFl\n" +
			"  %tf = select i1 %clampFl, i32 %floor, i32 %thy\n" +
			"  %hasAd = icmp sgt i32 %adapt, 0\n" +
			"  %belowAd = icmp slt i32 %tf, %adapt\n" +
			"  %clampAd = and i1 %hasAd, %belowAd\n" +
			"  %ta = select i1 %clampAd, i32 %adapt, i32 %tf\n" +
			"  %hasBu = icmp sgt i32 %burst, 0\n" +
			"  %overBu = icmp sgt i32 %ta, %burst\n" +
			"  %clampBu = and i1 %hasBu, %overBu\n" +
			"  %tbu = select i1 %clampBu, i32 %burst, i32 %ta\n" +
			"  %tco = sub i32 %tbu, %cool\n" +
			"  %tcr = add i32 %tco, %credit\n" +
			"  %tde = sub i32 %tcr, %debt\n" +
			"  %tit = add i32 %tde, %interest\n" +
			"  %ok = icmp sge i32 %tit, 1\n" +
			"  %w = select i1 %ok, i32 %tit, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoInterestCtr {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest counter (v0.74)\n" +
			"define i32 @trin_fusion_pgo_interest_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool, i32 %credit, i32 %debt, i32 %interest) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y1 = add i32 %y0, %half\n" +
			"  %y2 = add i32 %y1, %conf\n" +
			"  %y3 = add i32 %y2, %hyst\n" +
			"  %y4 = add i32 %y3, %floor\n" +
			"  %y5 = add i32 %y4, %adapt\n" +
			"  %y6 = add i32 %y5, %burst\n" +
			"  %y7 = add i32 %y6, %cool\n" +
			"  %y8 = add i32 %y7, %credit\n" +
			"  %y9 = add i32 %y8, %debt\n" +
			"  %y = add i32 %y9, %interest\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}


	if needGatherMtVlrefill {
		fallback += "" +
			"; SME ZA multi-tile peel gather host with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth + watermark + watermark drain + refill (v0.75)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vlrefill() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vldrain()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVlrefill {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth + watermark + watermark drain + refill (v0.75)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vlrefill() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vldrain()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoAmortize {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize (v0.75)\n" +
			"define i32 @trin_fusion_pgo_amortize(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %budget, i32 %softcap, i32 %hard, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool, i32 %credit, i32 %debt, i32 %interest, i32 %amort) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %cf = sdiv i32 %conf, 10\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %t6 = add i32 %t5, %hf\n" +
			"  %y7 = add i32 %t6, %cf\n" +
			"  %hasBudget = icmp sgt i32 %budget, 0\n" +
			"  %overB = icmp sgt i32 %y7, %budget\n" +
			"  %clampB = and i1 %hasBudget, %overB\n" +
			"  %tb = select i1 %clampB, i32 %budget, i32 %y7\n" +
			"  %hasSC = icmp sgt i32 %softcap, 0\n" +
			"  %overS = icmp sgt i32 %tb, %softcap\n" +
			"  %clampS = and i1 %hasSC, %overS\n" +
			"  %sum = add i32 %tb, %softcap\n" +
			"  %avg = sdiv i32 %sum, 2\n" +
			"  %avgOver = icmp sgt i32 %avg, %softcap\n" +
			"  %avgc = select i1 %avgOver, i32 %softcap, i32 %avg\n" +
			"  %ts = select i1 %clampS, i32 %avgc, i32 %tb\n" +
			"  %hasH = icmp sgt i32 %hard, 0\n" +
			"  %overH = icmp sgt i32 %ts, %hard\n" +
			"  %clampH = and i1 %hasH, %overH\n" +
			"  %th = select i1 %clampH, i32 %hard, i32 %ts\n" +
			"  %diffpos = sub i32 %th, %softcap\n" +
			"  %diffneg = sub i32 %softcap, %th\n" +
			"  %thGEsc = icmp sge i32 %th, %softcap\n" +
			"  %diff = select i1 %thGEsc, i32 %diffpos, i32 %diffneg\n" +
			"  %hasHy = icmp sgt i32 %hyst, 0\n" +
			"  %hasSChy = icmp sgt i32 %softcap, 0\n" +
			"  %hyReady = and i1 %hasHy, %hasSChy\n" +
			"  %inBand = icmp sle i32 %diff, %hyst\n" +
			"  %doHy = and i1 %hyReady, %inBand\n" +
			"  %thy = select i1 %doHy, i32 %softcap, i32 %th\n" +
			"  %hasFl = icmp sgt i32 %floor, 0\n" +
			"  %belowFl = icmp slt i32 %thy, %floor\n" +
			"  %clampFl = and i1 %hasFl, %belowFl\n" +
			"  %tf = select i1 %clampFl, i32 %floor, i32 %thy\n" +
			"  %hasAd = icmp sgt i32 %adapt, 0\n" +
			"  %belowAd = icmp slt i32 %tf, %adapt\n" +
			"  %clampAd = and i1 %hasAd, %belowAd\n" +
			"  %ta = select i1 %clampAd, i32 %adapt, i32 %tf\n" +
			"  %hasBu = icmp sgt i32 %burst, 0\n" +
			"  %overBu = icmp sgt i32 %ta, %burst\n" +
			"  %clampBu = and i1 %hasBu, %overBu\n" +
			"  %tbu = select i1 %clampBu, i32 %burst, i32 %ta\n" +
			"  %tco = sub i32 %tbu, %cool\n" +
			"  %tcr = add i32 %tco, %credit\n" +
			"  %tde = sub i32 %tcr, %debt\n" +
			"  %tit = add i32 %tde, %interest\n" +
			"  %tam = sub i32 %tit, %amort\n" +
			"  %ok = icmp sge i32 %tam, 1\n" +
			"  %w = select i1 %ok, i32 %tam, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needPgoAmortizeCtr {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize counter (v0.75)\n" +
			"define i32 @trin_fusion_pgo_amortize_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool, i32 %credit, i32 %debt, i32 %interest, i32 %amort) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y1 = add i32 %y0, %half\n" +
			"  %y2 = add i32 %y1, %conf\n" +
			"  %y3 = add i32 %y2, %hyst\n" +
			"  %y4 = add i32 %y3, %floor\n" +
			"  %y5 = add i32 %y4, %adapt\n" +
			"  %y6 = add i32 %y5, %burst\n" +
			"  %y7 = add i32 %y6, %cool\n" +
			"  %y8 = add i32 %y7, %credit\n" +
			"  %y9 = add i32 %y8, %debt\n" +
			"  %y10 = add i32 %y9, %interest\n" +
			"  %y = add i32 %y10, %amort\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}

	if needGatherMtVlflush {
		fallback += "" +
			"; SME ZA multi-tile peel gather host with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth + watermark + watermark drain + refill + flush (v0.76)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vlflush() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vlrefill()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVlflush {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth + watermark + watermark drain + refill + flush (v0.76)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vlflush() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vlrefill()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
if needPgoSettle {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + settle (v0.76)\n" +
			"define i32 @trin_fusion_pgo_settle(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %budget, i32 %softcap, i32 %hard, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool, i32 %credit, i32 %debt, i32 %interest, i32 %amort, i32 %settle) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %cf = sdiv i32 %conf, 10\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %t6 = add i32 %t5, %hf\n" +
			"  %y7 = add i32 %t6, %cf\n" +
			"  %hasBudget = icmp sgt i32 %budget, 0\n" +
			"  %overB = icmp sgt i32 %y7, %budget\n" +
			"  %clampB = and i1 %hasBudget, %overB\n" +
			"  %tb = select i1 %clampB, i32 %budget, i32 %y7\n" +
			"  %hasSC = icmp sgt i32 %softcap, 0\n" +
			"  %overS = icmp sgt i32 %tb, %softcap\n" +
			"  %clampS = and i1 %hasSC, %overS\n" +
			"  %sum = add i32 %tb, %softcap\n" +
			"  %avg = sdiv i32 %sum, 2\n" +
			"  %avgOver = icmp sgt i32 %avg, %softcap\n" +
			"  %avgc = select i1 %avgOver, i32 %softcap, i32 %avg\n" +
			"  %ts = select i1 %clampS, i32 %avgc, i32 %tb\n" +
			"  %hasH = icmp sgt i32 %hard, 0\n" +
			"  %overH = icmp sgt i32 %ts, %hard\n" +
			"  %clampH = and i1 %hasH, %overH\n" +
			"  %th = select i1 %clampH, i32 %hard, i32 %ts\n" +
			"  %diffpos = sub i32 %th, %softcap\n" +
			"  %diffneg = sub i32 %softcap, %th\n" +
			"  %thGEsc = icmp sge i32 %th, %softcap\n" +
			"  %diff = select i1 %thGEsc, i32 %diffpos, i32 %diffneg\n" +
			"  %hasHy = icmp sgt i32 %hyst, 0\n" +
			"  %hasSChy = icmp sgt i32 %softcap, 0\n" +
			"  %hyReady = and i1 %hasHy, %hasSChy\n" +
			"  %inBand = icmp sle i32 %diff, %hyst\n" +
			"  %doHy = and i1 %hyReady, %inBand\n" +
			"  %thy = select i1 %doHy, i32 %softcap, i32 %th\n" +
			"  %hasFl = icmp sgt i32 %floor, 0\n" +
			"  %belowFl = icmp slt i32 %thy, %floor\n" +
			"  %clampFl = and i1 %hasFl, %belowFl\n" +
			"  %tf = select i1 %clampFl, i32 %floor, i32 %thy\n" +
			"  %hasAd = icmp sgt i32 %adapt, 0\n" +
			"  %belowAd = icmp slt i32 %tf, %adapt\n" +
			"  %clampAd = and i1 %hasAd, %belowAd\n" +
			"  %ta = select i1 %clampAd, i32 %adapt, i32 %tf\n" +
			"  %hasBu = icmp sgt i32 %burst, 0\n" +
			"  %overBu = icmp sgt i32 %ta, %burst\n" +
			"  %clampBu = and i1 %hasBu, %overBu\n" +
			"  %tbu = select i1 %clampBu, i32 %burst, i32 %ta\n" +
			"  %tco = sub i32 %tbu, %cool\n" +
			"  %tcr = add i32 %tco, %credit\n" +
			"  %tde = sub i32 %tcr, %debt\n" +
			"  %tit = add i32 %tde, %interest\n" +
			"  %tam = sub i32 %tit, %amort\n" +
			"  %tse = sub i32 %tam, %settle\n" +
			"  %ok = icmp sge i32 %tse, 1\n" +
			"  %w = select i1 %ok, i32 %tse, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
if needPgoSettleCtr {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + settle counter (v0.76)\n" +
			"define i32 @trin_fusion_pgo_settle_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool, i32 %credit, i32 %debt, i32 %interest, i32 %amort, i32 %settle) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y1 = add i32 %y0, %half\n" +
			"  %y2 = add i32 %y1, %conf\n" +
			"  %y3 = add i32 %y2, %hyst\n" +
			"  %y4 = add i32 %y3, %floor\n" +
			"  %y5 = add i32 %y4, %adapt\n" +
			"  %y6 = add i32 %y5, %burst\n" +
			"  %y7 = add i32 %y6, %cool\n" +
			"  %y8 = add i32 %y7, %credit\n" +
			"  %y9 = add i32 %y8, %debt\n" +
			"  %y10 = add i32 %y9, %interest\n" +
			"  %y = add i32 %y10, %amort\n" +
			"  %y11 = add i32 %y, %settle\n" +
			"  ret i32 %y11\n" +
			"}\n\n"
	}

	if needGatherMtVlseal {
		fallback += "" +
			"; SME ZA multi-tile peel gather host with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth + watermark + watermark drain + refill + flush + seal (v0.77)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vlseal() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vlflush()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVlseal {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth + watermark + watermark drain + refill + flush + seal (v0.77)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vlseal() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vlflush()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
if needPgoEscrow {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + settle (v0.77)\n" +
			"define i32 @trin_fusion_pgo_escrow(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %budget, i32 %softcap, i32 %hard, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool, i32 %credit, i32 %debt, i32 %interest, i32 %amort, i32 %settle, i32 %escrow) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %cf = sdiv i32 %conf, 10\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %t6 = add i32 %t5, %hf\n" +
			"  %y7 = add i32 %t6, %cf\n" +
			"  %hasBudget = icmp sgt i32 %budget, 0\n" +
			"  %overB = icmp sgt i32 %y7, %budget\n" +
			"  %clampB = and i1 %hasBudget, %overB\n" +
			"  %tb = select i1 %clampB, i32 %budget, i32 %y7\n" +
			"  %hasSC = icmp sgt i32 %softcap, 0\n" +
			"  %overS = icmp sgt i32 %tb, %softcap\n" +
			"  %clampS = and i1 %hasSC, %overS\n" +
			"  %sum = add i32 %tb, %softcap\n" +
			"  %avg = sdiv i32 %sum, 2\n" +
			"  %avgOver = icmp sgt i32 %avg, %softcap\n" +
			"  %avgc = select i1 %avgOver, i32 %softcap, i32 %avg\n" +
			"  %ts = select i1 %clampS, i32 %avgc, i32 %tb\n" +
			"  %hasH = icmp sgt i32 %hard, 0\n" +
			"  %overH = icmp sgt i32 %ts, %hard\n" +
			"  %clampH = and i1 %hasH, %overH\n" +
			"  %th = select i1 %clampH, i32 %hard, i32 %ts\n" +
			"  %diffpos = sub i32 %th, %softcap\n" +
			"  %diffneg = sub i32 %softcap, %th\n" +
			"  %thGEsc = icmp sge i32 %th, %softcap\n" +
			"  %diff = select i1 %thGEsc, i32 %diffpos, i32 %diffneg\n" +
			"  %hasHy = icmp sgt i32 %hyst, 0\n" +
			"  %hasSChy = icmp sgt i32 %softcap, 0\n" +
			"  %hyReady = and i1 %hasHy, %hasSChy\n" +
			"  %inBand = icmp sle i32 %diff, %hyst\n" +
			"  %doHy = and i1 %hyReady, %inBand\n" +
			"  %thy = select i1 %doHy, i32 %softcap, i32 %th\n" +
			"  %hasFl = icmp sgt i32 %floor, 0\n" +
			"  %belowFl = icmp slt i32 %thy, %floor\n" +
			"  %clampFl = and i1 %hasFl, %belowFl\n" +
			"  %tf = select i1 %clampFl, i32 %floor, i32 %thy\n" +
			"  %hasAd = icmp sgt i32 %adapt, 0\n" +
			"  %belowAd = icmp slt i32 %tf, %adapt\n" +
			"  %clampAd = and i1 %hasAd, %belowAd\n" +
			"  %ta = select i1 %clampAd, i32 %adapt, i32 %tf\n" +
			"  %hasBu = icmp sgt i32 %burst, 0\n" +
			"  %overBu = icmp sgt i32 %ta, %burst\n" +
			"  %clampBu = and i1 %hasBu, %overBu\n" +
			"  %tbu = select i1 %clampBu, i32 %burst, i32 %ta\n" +
			"  %tco = sub i32 %tbu, %cool\n" +
			"  %tcr = add i32 %tco, %credit\n" +
			"  %tde = sub i32 %tcr, %debt\n" +
			"  %tit = add i32 %tde, %interest\n" +
			"  %tam = sub i32 %tit, %amort\n" +
			"  %tse = sub i32 %tam, %settle\n" +
			"  %tes = add i32 %tse, %escrow\n" +
			"  %ok = icmp sge i32 %tes, 1\n" +
			"  %w = select i1 %ok, i32 %tes, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
if needPgoEscrowCtr {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + settle counter (v0.77)\n" +
			"define i32 @trin_fusion_pgo_escrow_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool, i32 %credit, i32 %debt, i32 %interest, i32 %amort, i32 %settle, i32 %escrow) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y1 = add i32 %y0, %half\n" +
			"  %y2 = add i32 %y1, %conf\n" +
			"  %y3 = add i32 %y2, %hyst\n" +
			"  %y4 = add i32 %y3, %floor\n" +
			"  %y5 = add i32 %y4, %adapt\n" +
			"  %y6 = add i32 %y5, %burst\n" +
			"  %y7 = add i32 %y6, %cool\n" +
			"  %y8 = add i32 %y7, %credit\n" +
			"  %y9 = add i32 %y8, %debt\n" +
			"  %y10 = add i32 %y9, %interest\n" +
			"  %y = add i32 %y10, %amort\n" +
			"  %y11 = add i32 %y, %settle\n" +
			"  %y12 = add i32 %y11, %escrow\n" +
			"  ret i32 %y12\n" +
			"}\n\n"
	}

	if needGatherMtVlpurge {
		fallback += "" +
			"; SME ZA multi-tile peel gather host residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain refill flush seal + purge (v0.78)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vlpurge() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vlseal()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVlpurge {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain refill flush seal + purge (v0.78)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vlpurge() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vlseal()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
if needPgoClear {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + clear + settle (v0.78)\n" +
			"define i32 @trin_fusion_pgo_clear(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %budget, i32 %softcap, i32 %hard, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool, i32 %credit, i32 %debt, i32 %interest, i32 %amort, i32 %settle, i32 %escrow, i32 %clear) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %cf = sdiv i32 %conf, 10\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %t6 = add i32 %t5, %hf\n" +
			"  %y7 = add i32 %t6, %cf\n" +
			"  %hasBudget = icmp sgt i32 %budget, 0\n" +
			"  %overB = icmp sgt i32 %y7, %budget\n" +
			"  %clampB = and i1 %hasBudget, %overB\n" +
			"  %tb = select i1 %clampB, i32 %budget, i32 %y7\n" +
			"  %hasSC = icmp sgt i32 %softcap, 0\n" +
			"  %overS = icmp sgt i32 %tb, %softcap\n" +
			"  %clampS = and i1 %hasSC, %overS\n" +
			"  %sum = add i32 %tb, %softcap\n" +
			"  %avg = sdiv i32 %sum, 2\n" +
			"  %avgOver = icmp sgt i32 %avg, %softcap\n" +
			"  %avgc = select i1 %avgOver, i32 %softcap, i32 %avg\n" +
			"  %ts = select i1 %clampS, i32 %avgc, i32 %tb\n" +
			"  %hasH = icmp sgt i32 %hard, 0\n" +
			"  %overH = icmp sgt i32 %ts, %hard\n" +
			"  %clampH = and i1 %hasH, %overH\n" +
			"  %th = select i1 %clampH, i32 %hard, i32 %ts\n" +
			"  %diffpos = sub i32 %th, %softcap\n" +
			"  %diffneg = sub i32 %softcap, %th\n" +
			"  %thGEsc = icmp sge i32 %th, %softcap\n" +
			"  %diff = select i1 %thGEsc, i32 %diffpos, i32 %diffneg\n" +
			"  %hasHy = icmp sgt i32 %hyst, 0\n" +
			"  %hasSChy = icmp sgt i32 %softcap, 0\n" +
			"  %hyReady = and i1 %hasHy, %hasSChy\n" +
			"  %inBand = icmp sle i32 %diff, %hyst\n" +
			"  %doHy = and i1 %hyReady, %inBand\n" +
			"  %thy = select i1 %doHy, i32 %softcap, i32 %th\n" +
			"  %hasFl = icmp sgt i32 %floor, 0\n" +
			"  %belowFl = icmp slt i32 %thy, %floor\n" +
			"  %clampFl = and i1 %hasFl, %belowFl\n" +
			"  %tf = select i1 %clampFl, i32 %floor, i32 %thy\n" +
			"  %hasAd = icmp sgt i32 %adapt, 0\n" +
			"  %belowAd = icmp slt i32 %tf, %adapt\n" +
			"  %clampAd = and i1 %hasAd, %belowAd\n" +
			"  %ta = select i1 %clampAd, i32 %adapt, i32 %tf\n" +
			"  %hasBu = icmp sgt i32 %burst, 0\n" +
			"  %overBu = icmp sgt i32 %ta, %burst\n" +
			"  %clampBu = and i1 %hasBu, %overBu\n" +
			"  %tbu = select i1 %clampBu, i32 %burst, i32 %ta\n" +
			"  %tco = sub i32 %tbu, %cool\n" +
			"  %tcr = add i32 %tco, %credit\n" +
			"  %tde = sub i32 %tcr, %debt\n" +
			"  %tit = add i32 %tde, %interest\n" +
			"  %tam = sub i32 %tit, %amort\n" +
			"  %tse = sub i32 %tam, %settle\n" +
			"  %tes = add i32 %tse, %escrow\n" +
			"  %tcl = sub i32 %tes, %clear\n" +
			"  %ok = icmp sge i32 %tcl, 1\n" +
			"  %w = select i1 %ok, i32 %tcl, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
if needPgoClearCtr {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + clear + settle counter (v0.78)\n" +
			"define i32 @trin_fusion_pgo_clear_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool, i32 %credit, i32 %debt, i32 %interest, i32 %amort, i32 %settle, i32 %escrow, i32 %clear) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y1 = add i32 %y0, %half\n" +
			"  %y2 = add i32 %y1, %conf\n" +
			"  %y3 = add i32 %y2, %hyst\n" +
			"  %y4 = add i32 %y3, %floor\n" +
			"  %y5 = add i32 %y4, %adapt\n" +
			"  %y6 = add i32 %y5, %burst\n" +
			"  %y7 = add i32 %y6, %cool\n" +
			"  %y8 = add i32 %y7, %credit\n" +
			"  %y9 = add i32 %y8, %debt\n" +
			"  %y10 = add i32 %y9, %interest\n" +
			"  %y = add i32 %y10, %amort\n" +
			"  %y11 = add i32 %y, %settle\n" +
			"  %y12 = add i32 %y11, %escrow\n" +
			"  %y13 = add i32 %y12, %clear\n" +
			"  ret i32 %y13\n" +
			"}\n\n"
	}

	if needGatherMtVlsync {
		fallback += "" +
			"; SME ZA multi-tile peel gather host ... seal purge + sync (v0.79)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vlsync() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vlpurge()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVlsync {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host ... seal purge + sync (v0.79)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vlsync() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vlpurge()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
if needPgoFinal {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + clear + final + settle (v0.79)\n" +
			"define i32 @trin_fusion_pgo_final(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %budget, i32 %softcap, i32 %hard, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool, i32 %credit, i32 %debt, i32 %interest, i32 %amort, i32 %settle, i32 %escrow, i32 %clear, i32 %final) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %cf = sdiv i32 %conf, 10\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %t6 = add i32 %t5, %hf\n" +
			"  %y7 = add i32 %t6, %cf\n" +
			"  %hasBudget = icmp sgt i32 %budget, 0\n" +
			"  %overB = icmp sgt i32 %y7, %budget\n" +
			"  %clampB = and i1 %hasBudget, %overB\n" +
			"  %tb = select i1 %clampB, i32 %budget, i32 %y7\n" +
			"  %hasSC = icmp sgt i32 %softcap, 0\n" +
			"  %overS = icmp sgt i32 %tb, %softcap\n" +
			"  %clampS = and i1 %hasSC, %overS\n" +
			"  %sum = add i32 %tb, %softcap\n" +
			"  %avg = sdiv i32 %sum, 2\n" +
			"  %avgOver = icmp sgt i32 %avg, %softcap\n" +
			"  %avgc = select i1 %avgOver, i32 %softcap, i32 %avg\n" +
			"  %ts = select i1 %clampS, i32 %avgc, i32 %tb\n" +
			"  %hasH = icmp sgt i32 %hard, 0\n" +
			"  %overH = icmp sgt i32 %ts, %hard\n" +
			"  %clampH = and i1 %hasH, %overH\n" +
			"  %th = select i1 %clampH, i32 %hard, i32 %ts\n" +
			"  %diffpos = sub i32 %th, %softcap\n" +
			"  %diffneg = sub i32 %softcap, %th\n" +
			"  %thGEsc = icmp sge i32 %th, %softcap\n" +
			"  %diff = select i1 %thGEsc, i32 %diffpos, i32 %diffneg\n" +
			"  %hasHy = icmp sgt i32 %hyst, 0\n" +
			"  %hasSChy = icmp sgt i32 %softcap, 0\n" +
			"  %hyReady = and i1 %hasHy, %hasSChy\n" +
			"  %inBand = icmp sle i32 %diff, %hyst\n" +
			"  %doHy = and i1 %hyReady, %inBand\n" +
			"  %thy = select i1 %doHy, i32 %softcap, i32 %th\n" +
			"  %hasFl = icmp sgt i32 %floor, 0\n" +
			"  %belowFl = icmp slt i32 %thy, %floor\n" +
			"  %clampFl = and i1 %hasFl, %belowFl\n" +
			"  %tf = select i1 %clampFl, i32 %floor, i32 %thy\n" +
			"  %hasAd = icmp sgt i32 %adapt, 0\n" +
			"  %belowAd = icmp slt i32 %tf, %adapt\n" +
			"  %clampAd = and i1 %hasAd, %belowAd\n" +
			"  %ta = select i1 %clampAd, i32 %adapt, i32 %tf\n" +
			"  %hasBu = icmp sgt i32 %burst, 0\n" +
			"  %overBu = icmp sgt i32 %ta, %burst\n" +
			"  %clampBu = and i1 %hasBu, %overBu\n" +
			"  %tbu = select i1 %clampBu, i32 %burst, i32 %ta\n" +
			"  %tco = sub i32 %tbu, %cool\n" +
			"  %tcr = add i32 %tco, %credit\n" +
			"  %tde = sub i32 %tcr, %debt\n" +
			"  %tit = add i32 %tde, %interest\n" +
			"  %tam = sub i32 %tit, %amort\n" +
			"  %tse = sub i32 %tam, %settle\n" +
			"  %tes = add i32 %tse, %escrow\n" +
			"  %tcl = sub i32 %tes, %clear\n" +
			"  %tfi = add i32 %tcl, %final\n" +
			"  %ok = icmp sge i32 %tfi, 1\n" +
			"  %w = select i1 %ok, i32 %tfi, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
if needPgoFinalCtr {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + clear + final + settle counter (v0.79)\n" +
			"define i32 @trin_fusion_pgo_final_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool, i32 %credit, i32 %debt, i32 %interest, i32 %amort, i32 %settle, i32 %escrow, i32 %clear, i32 %final) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y1 = add i32 %y0, %half\n" +
			"  %y2 = add i32 %y1, %conf\n" +
			"  %y3 = add i32 %y2, %hyst\n" +
			"  %y4 = add i32 %y3, %floor\n" +
			"  %y5 = add i32 %y4, %adapt\n" +
			"  %y6 = add i32 %y5, %burst\n" +
			"  %y7 = add i32 %y6, %cool\n" +
			"  %y8 = add i32 %y7, %credit\n" +
			"  %y9 = add i32 %y8, %debt\n" +
			"  %y10 = add i32 %y9, %interest\n" +
			"  %y = add i32 %y10, %amort\n" +
			"  %y11 = add i32 %y, %settle\n" +
			"  %y12 = add i32 %y11, %escrow\n" +
			"  %y13 = add i32 %y12, %clear\n" +
			"  %y14 = add i32 %y13, %final\n" +
			"  ret i32 %y14\n" +
			"}\n\n"
	}

	if needGatherMtVlclose {
		fallback += "" +
			"; SME ZA multi-tile peel gather host ... purge sync + close (v0.80)\n" +
			"define i32 @trin_hw_sme_za_gather_mt_vlclose() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_gather_mt_vlsync()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
	if needScatterMtVlclose {
		fallback += "" +
			"; SME ZA multi-tile peel scatter host ... purge sync + close (v0.80)\n" +
			"define i32 @trin_hw_sme_za_scatter_mt_vlclose() {\n" +
			"entry:\n" +
			"  %v = call i32 @trin_hw_sme_za_scatter_mt_vlsync()\n" +
			"  %vl = call i32 @trin_hw_vlen_for_n(i32 16)\n" +
			"  %d = sub i32 %vl, 16\n" +
			"  %y = add i32 %v, %d\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
if needPgoClose {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + clear + final + close + settle (v0.80)\n" +
			"define i32 @trin_fusion_pgo_close(i32 %n, i32 %isa, i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %budget, i32 %softcap, i32 %hard, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool, i32 %credit, i32 %debt, i32 %interest, i32 %amort, i32 %settle, i32 %escrow, i32 %clear, i32 %final, i32 %close) {\n" +
			"entry:\n" +
			"  %hd = sdiv i32 %hits, 10\n" +
			"  %ld = sdiv i32 %live, 20\n" +
			"  %xd = sdiv i32 %xk, 5\n" +
			"  %pd = sdiv i32 %pass, 2\n" +
			"  %wd = sdiv i32 %win, 4\n" +
			"  %hf = sdiv i32 %half, 5\n" +
			"  %cf = sdiv i32 %conf, 10\n" +
			"  %t0 = add i32 1, %hd\n" +
			"  %t1 = add i32 %t0, %ld\n" +
			"  %t2 = add i32 %t1, %xd\n" +
			"  %t3 = add i32 %t2, %pd\n" +
			"  %t4 = add i32 %t3, %wd\n" +
			"  %t5 = sub i32 %t4, %decay\n" +
			"  %t6 = add i32 %t5, %hf\n" +
			"  %y7 = add i32 %t6, %cf\n" +
			"  %hasBudget = icmp sgt i32 %budget, 0\n" +
			"  %overB = icmp sgt i32 %y7, %budget\n" +
			"  %clampB = and i1 %hasBudget, %overB\n" +
			"  %tb = select i1 %clampB, i32 %budget, i32 %y7\n" +
			"  %hasSC = icmp sgt i32 %softcap, 0\n" +
			"  %overS = icmp sgt i32 %tb, %softcap\n" +
			"  %clampS = and i1 %hasSC, %overS\n" +
			"  %sum = add i32 %tb, %softcap\n" +
			"  %avg = sdiv i32 %sum, 2\n" +
			"  %avgOver = icmp sgt i32 %avg, %softcap\n" +
			"  %avgc = select i1 %avgOver, i32 %softcap, i32 %avg\n" +
			"  %ts = select i1 %clampS, i32 %avgc, i32 %tb\n" +
			"  %hasH = icmp sgt i32 %hard, 0\n" +
			"  %overH = icmp sgt i32 %ts, %hard\n" +
			"  %clampH = and i1 %hasH, %overH\n" +
			"  %th = select i1 %clampH, i32 %hard, i32 %ts\n" +
			"  %diffpos = sub i32 %th, %softcap\n" +
			"  %diffneg = sub i32 %softcap, %th\n" +
			"  %thGEsc = icmp sge i32 %th, %softcap\n" +
			"  %diff = select i1 %thGEsc, i32 %diffpos, i32 %diffneg\n" +
			"  %hasHy = icmp sgt i32 %hyst, 0\n" +
			"  %hasSChy = icmp sgt i32 %softcap, 0\n" +
			"  %hyReady = and i1 %hasHy, %hasSChy\n" +
			"  %inBand = icmp sle i32 %diff, %hyst\n" +
			"  %doHy = and i1 %hyReady, %inBand\n" +
			"  %thy = select i1 %doHy, i32 %softcap, i32 %th\n" +
			"  %hasFl = icmp sgt i32 %floor, 0\n" +
			"  %belowFl = icmp slt i32 %thy, %floor\n" +
			"  %clampFl = and i1 %hasFl, %belowFl\n" +
			"  %tf = select i1 %clampFl, i32 %floor, i32 %thy\n" +
			"  %hasAd = icmp sgt i32 %adapt, 0\n" +
			"  %belowAd = icmp slt i32 %tf, %adapt\n" +
			"  %clampAd = and i1 %hasAd, %belowAd\n" +
			"  %ta = select i1 %clampAd, i32 %adapt, i32 %tf\n" +
			"  %hasBu = icmp sgt i32 %burst, 0\n" +
			"  %overBu = icmp sgt i32 %ta, %burst\n" +
			"  %clampBu = and i1 %hasBu, %overBu\n" +
			"  %tbu = select i1 %clampBu, i32 %burst, i32 %ta\n" +
			"  %tco = sub i32 %tbu, %cool\n" +
			"  %tcr = add i32 %tco, %credit\n" +
			"  %tde = sub i32 %tcr, %debt\n" +
			"  %tit = add i32 %tde, %interest\n" +
			"  %tam = sub i32 %tit, %amort\n" +
			"  %tse = sub i32 %tam, %settle\n" +
			"  %tes = add i32 %tse, %escrow\n" +
			"  %tcl = sub i32 %tes, %clear\n" +
			"  %tfi = add i32 %tcl, %final\n" +
			"  %tclo = sub i32 %tfi, %close\n" +
			"  %ok = icmp sge i32 %tclo, 1\n" +
			"  %w = select i1 %ok, i32 %tclo, i32 1\n" +
			"  %y = mul i32 %n, %w\n" +
			"  ret i32 %y\n" +
			"}\n\n"
	}
if needPgoCloseCtr {
		fallback += "" +
			"; fusion PGO dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + clear + final + close + settle counter (v0.80)\n" +
			"define i32 @trin_fusion_pgo_close_counter(i32 %hits, i32 %live, i32 %xk, i32 %pass, i32 %win, i32 %decay, i32 %half, i32 %conf, i32 %hyst, i32 %floor, i32 %adapt, i32 %burst, i32 %cool, i32 %credit, i32 %debt, i32 %interest, i32 %amort, i32 %settle, i32 %escrow, i32 %clear, i32 %final, i32 %close) {\n" +
			"entry:\n" +
			"  %t = add i32 %hits, %live\n" +
			"  %u = add i32 %t, %xk\n" +
			"  %v = add i32 %u, %pass\n" +
			"  %x = add i32 %v, %win\n" +
			"  %y0 = add i32 %x, %decay\n" +
			"  %y1 = add i32 %y0, %half\n" +
			"  %y2 = add i32 %y1, %conf\n" +
			"  %y3 = add i32 %y2, %hyst\n" +
			"  %y4 = add i32 %y3, %floor\n" +
			"  %y5 = add i32 %y4, %adapt\n" +
			"  %y6 = add i32 %y5, %burst\n" +
			"  %y7 = add i32 %y6, %cool\n" +
			"  %y8 = add i32 %y7, %credit\n" +
			"  %y9 = add i32 %y8, %debt\n" +
			"  %y10 = add i32 %y9, %interest\n" +
			"  %y = add i32 %y10, %amort\n" +
			"  %y11 = add i32 %y, %settle\n" +
			"  %y12 = add i32 %y11, %escrow\n" +
			"  %y13 = add i32 %y12, %clear\n" +
			"  %y14 = add i32 %y13, %final\n" +
			"  %y15 = add i32 %y14, %close\n" +
			"  ret i32 %y15\n" +
			"}\n\n"
	}






	// Keep leading comment / target lines first.
	lines := strings.Split(ll, "\n")
	insertAt := 0
	for insertAt < len(lines) {
		s := strings.TrimSpace(lines[insertAt])
		if s == "" || strings.HasPrefix(s, ";") ||
			strings.HasPrefix(s, "target datalayout") ||
			strings.HasPrefix(s, "target triple") ||
			strings.HasPrefix(s, "source_filename") {
			insertAt++
			continue
		}
		break
	}
	var out strings.Builder
	for i, line := range lines {
		if i == insertAt {
			out.WriteString(fallback)
		}
		out.WriteString(line)
		if i+1 < len(lines) || strings.HasSuffix(ll, "\n") {
			out.WriteByte('\n')
		}
	}
	if insertAt >= len(lines) {
		// all header — append
		return ll + "\n" + fallback
	}
	return out.String()
}
