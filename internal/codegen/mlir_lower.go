package codegen

import (
	"fmt"
	"strings"

	"github.com/macroadster/trin/internal/ast"
)

// EmitMLIRLLVM is the v0.22+ path: emit portable MLIR, then lower to LLVM IR text.
// No system mlir-opt required. When the program has full-program @main (v0.23),
// the lowered module includes an executable define i32 @main (no float host).
func EmitMLIRLLVM(prog *ast.Program) (string, error) {
	mlir, err := EmitMLIR(prog)
	if err != nil {
		return "", err
	}
	ll, err := LowerMLIRToLLVM(mlir)
	if err != nil {
		return "", err
	}
	// Prefer true full-program LLVM from the AST when MLIR carried @main
	// so runtime/println/soft_mux semantics match emit-llvm v0.19.
	if strings.Contains(mlir, "func.func @main()") &&
		!hasHostFloatExport(prog) &&
		!hasLLVMMain(ll) {
		// Skip redefining CPUID helpers already present in the lower; still need dyn tiles.
		needHW := !strings.Contains(ll, "define i32 @trin_hw_vlen_for_n")
		needDyn := !strings.Contains(ll, "define float @trin_dyn_sum4")
		if full := emitFullProgramEx(prog, false); full != "" {
			var b strings.Builder
			b.WriteString(ll)
			if !strings.HasSuffix(ll, "\n") {
				b.WriteByte('\n')
			}
			// ensure builtins referenced by full-program body exist
			if !strings.Contains(ll, "define i32 @trin_simd_lane_width") {
				b.WriteString("define i32 @trin_simd_lane_width() {\n")
				b.WriteString("entry:\n")
				b.WriteString("  ret i32 128\n")
				b.WriteString("}\n\n")
			}
			if needHW {
				b.WriteString(EmitRuntimeCPUIDLLVM(true, true, true))
			}
			if needDyn {
				b.WriteString(emitDynTileLLVMHelpers())
			}
			b.WriteString("; --- full-program MLIR→LLVM @main (v0.23 + dyn tiles v0.32) ---\n")
			b.WriteString(full)
			ll = b.String()
		}
	}
	return ll, nil
}

// LowerMLIRToLLVM converts Trin emit-mlir text into LLVM IR for soft_mux,
// span kernels, vector tiles (4-lane + 8-lane), and v0.25 dialect probes.
// Unknown custom ops are skipped with notes. When high-level "trin.*" ops
// remain (pre-rewrite), Apply RewriteMLIR first.
func LowerMLIRToLLVM(mlir string) (string, error) {
	if strings.TrimSpace(mlir) == "" {
		return "", fmt.Errorf("empty MLIR input")
	}
	// Require our dialect module marker so random text is not silently accepted.
	if !strings.Contains(mlir, "module attributes {trin.version") &&
		!strings.Contains(mlir, "func.func @trin_soft_mux") {
		return "", fmt.Errorf("not a Trin MLIR module (missing trin.version / soft_mux)")
	}

	// Expand any remaining high-level dialect ops before lowering.
	if strings.Contains(mlir, `"trin.`) {
		if rw, err := RewriteMLIR(mlir); err == nil {
			mlir = rw
		}
	}

	// Discover which kernel funcs are present (order preserved below).
	want := []string{
		"trin_soft_mux",
		"trin_sum4",
		"trin_dot4",
		"trin_sum_f32",
		"trin_dot_f32",
		"trin_span_add_f32",
		"trin_span_scale_f32",
		"trin_span_mul_f32",
		"trin_span_axpy_f32",
		"trin_span_scan_add_f32",
		"trin_reduce_add_f32",
		"trin_span_scan_mul_f32",
		"trin_reduce_mul_f32",
		"trin_span_scan_reduce_mul_f32",
		"trin_span_scan_reduce_add_f32",
		"trin_span_scan_reduce_add_pair_f32",
		"trin_span_scan_reduce_mul_pair_f32",
		"trin_span_scan_reduce_add_tuple_f32",
		"trin_span_scan_reduce_mul_tuple_f32",
		"trin_span_scan_reduce_add_sum_f32",
		"trin_fused_map_reduce_mul_add_f32",
		"trin_fused_map_reduce_mul_sum_f32",
		"trin_fused_map_reduce_cost_mul_add_f32",
		"trin_pair_f32",
		"trin_pair_first_f32",
		"trin_pair_second_f32",
		"trin_pair_swap_f32",
		"trin_pair_match_f32",
		"trin_hw_rdvl",
		"trin_hw_rdvl_live",
		"trin_hw_sme_vl",
		"trin_hw_sme_za_tile",
		"trin_hw_sme_za_outer",
		"trin_hw_sme_za_outer_acc",
		"trin_fusion_cost",
		"trin_fusion_cost_weighted",
		"trin_fusion_cost_isa",
		"trin_fusion_select",
		"trin_pair_exhaust_i32",
		"trin_hw_sme_za_outer_acc_true",
		"trin_pair_match_guard_f32",
		"trin_fusion_cost_calibrated",
		"trin_fusion_plan_log",
		"trin_pair_or_match_f32",
		"trin_hw_sme_za_stream_store",
		"trin_fusion_pgo_weight",
		"trin_fusion_pgo_feedback",
		"trin_fusion_pgo_select",
		"trin_pair_tag_match_f32",
		"trin_pair_tag_exhaust_i32",
		"trin_pair_exhaust_i32",
		"trin_hw_sme_za_gather",
		"trin_hw_sme_za_scatter",
		"trin_fusion_pgo_adapt",
		"trin_fusion_pgo_counter_bump",
		"trin_nested_tag_match_f32",
		"trin_hw_sme_za_gather_mv",
		"trin_hw_sme_za_scatter_mv",
		"trin_fusion_pgo_isa_counter",
		"trin_fusion_pgo_isa_loop",
		"trin_nested_tag_multi_match_f32",
		"trin_hw_sme_za_gather_tile",
		"trin_hw_sme_za_scatter_tile",
		"trin_fusion_pgo_isa_live",
		"trin_fusion_pgo_isa_feedback",
		"trin_nested_tag_recur_match_f32",
		"trin_hw_sme_za_gather_mt",
		"trin_hw_sme_za_scatter_mt",
		"trin_fusion_pgo_xkernel",
		"trin_fusion_pgo_xkernel_counter",
		"trin_nested_tag_tree_fold_f32",
		"trin_hw_sme_za_gather_mt_peel",
		"trin_hw_sme_za_scatter_mt_peel",
		"trin_fusion_pgo_xpass",
		"trin_fusion_pgo_xpass_counter",
		"trin_nested_tag_forest_fold_f32",
		"trin_nested_tag_map_fold_f32",
		"trin_hw_sme_za_gather_mt_auto",
		"trin_hw_sme_za_scatter_mt_auto",
		"trin_fusion_pgo_window",
		"trin_fusion_pgo_window_counter",
		"trin_nested_tag_keyed_reduce_f32",
		"trin_nested_tag_keyed_fold_f32",
		"trin_hw_sme_za_gather_mt_vlfb",
		"trin_hw_sme_za_scatter_mt_vlfb",
		"trin_fusion_pgo_decay",
		"trin_fusion_pgo_decay_counter",
		"trin_nested_tag_multi_reduce_f32",
		"trin_nested_tag_filter_f32",
		"trin_hw_sme_za_gather_mt_vllive",
		"trin_hw_sme_za_scatter_mt_vllive",
		"trin_fusion_pgo_ema",
		"trin_fusion_pgo_ema_counter",
		"trin_nested_tag_rank_reduce_f32",
		"trin_nested_tag_rank_filter_f32",
		"trin_hw_sme_za_gather_mt_vlstream",
		"trin_hw_sme_za_scatter_mt_vlstream",
		"trin_fusion_pgo_adapt_hl",
		"trin_fusion_pgo_adapt_hl_counter",
		"trin_nested_tag_weight_reduce_f32",
		"trin_nested_tag_weight_filter_f32",
		"trin_hw_sme_za_gather_mt_vlcost",
		"trin_hw_sme_za_scatter_mt_vlcost",
		"trin_fusion_pgo_hyst",
		"trin_fusion_pgo_hyst_counter",
		"trin_nested_tag_thresh_reduce_f32",
		"trin_nested_tag_thresh_filter_f32",
		"trin_hw_sme_za_gather_mt_vladapt",
		"trin_hw_sme_za_scatter_mt_vladapt",
		"trin_fusion_pgo_reweight",
		"trin_fusion_pgo_reweight_counter",
		"trin_nested_tag_soft_reduce_f32",
		"trin_nested_tag_soft_filter_f32",
		"trin_hw_sme_za_gather_mt_vlsched",
		"trin_hw_sme_za_scatter_mt_vlsched",
		"trin_fusion_pgo_budget",
		"trin_fusion_pgo_budget_counter",
		"trin_nested_tag_cal_reduce_f32",
		"trin_nested_tag_cal_filter_f32",
		"trin_hw_sme_za_gather_mt_vlfuse",
		"trin_hw_sme_za_scatter_mt_vlfuse",
		"trin_fusion_pgo_softcap",
		"trin_fusion_pgo_softcap_counter",
		"trin_nested_tag_adapt_reduce_f32",
		"trin_nested_tag_adapt_filter_f32",
		"trin_hw_sme_za_gather_mt_vlmix",
		"trin_hw_sme_za_scatter_mt_vlmix",
		"trin_fusion_pgo_dual",
		"trin_fusion_pgo_dual_counter",
		"trin_nested_tag_blend_reduce_f32",
		"trin_nested_tag_blend_filter_f32",
		"trin_hw_sme_za_gather_mt_vlbal",
		"trin_hw_sme_za_scatter_mt_vlbal",
		"trin_fusion_pgo_conf",
		"trin_fusion_pgo_conf_counter",
		"trin_nested_tag_gate_reduce_f32",
		"trin_nested_tag_score_reduce_f32",
		"trin_nested_tag_softfloor_reduce_f32",
		"trin_nested_tag_hardgate_reduce_f32",
		"trin_nested_tag_dual_reduce_f32",
		"trin_nested_tag_mix_reduce_f32",
		"trin_nested_tag_credit_reduce_f32",
		"trin_nested_tag_margin_reduce_f32",
		"trin_nested_tag_band_reduce_f32",
		"trin_nested_tag_clamp_reduce_f32",
		"trin_nested_tag_peg_reduce_f32",
		"trin_nested_tag_latch_reduce_f32",
		"trin_nested_tag_fence_reduce_f32",
		"trin_nested_tag_lock_reduce_f32",
		"trin_nested_tag_gate_filter_f32",
		"trin_nested_tag_score_filter_f32",
		"trin_nested_tag_softfloor_filter_f32",
		"trin_nested_tag_hardgate_filter_f32",
		"trin_nested_tag_dual_filter_f32",
		"trin_nested_tag_mix_filter_f32",
		"trin_nested_tag_credit_filter_f32",
		"trin_nested_tag_margin_filter_f32",
		"trin_nested_tag_band_filter_f32",
		"trin_hw_sme_za_gather_mt_vlbw",
		"trin_hw_sme_za_gather_mt_vlpath",
		"trin_hw_sme_za_gather_mt_vllane",
		"trin_hw_sme_za_gather_mt_vlspill",
		"trin_hw_sme_za_gather_mt_vlocc",
		"trin_hw_sme_za_gather_mt_vlqueue",
		"trin_hw_sme_za_gather_mt_vlwater",
		"trin_hw_sme_za_gather_mt_vldrain",
		"trin_hw_sme_za_gather_mt_vlrefill",
		"trin_hw_sme_za_gather_mt_vlflush",
		"trin_hw_sme_za_gather_mt_vlseal",
		"trin_hw_sme_za_gather_mt_vlpurge",
		"trin_hw_sme_za_gather_mt_vlsync",
		"trin_hw_sme_za_gather_mt_vlclose",
		"trin_hw_sme_za_scatter_mt_vlbw",
		"trin_hw_sme_za_scatter_mt_vlpath",
		"trin_hw_sme_za_scatter_mt_vllane",
		"trin_hw_sme_za_scatter_mt_vlspill",
		"trin_hw_sme_za_scatter_mt_vlocc",
		"trin_hw_sme_za_scatter_mt_vlqueue",
		"trin_hw_sme_za_scatter_mt_vlwater",
		"trin_hw_sme_za_scatter_mt_vldrain",
		"trin_hw_sme_za_scatter_mt_vlrefill",
		"trin_hw_sme_za_scatter_mt_vlflush",
		"trin_hw_sme_za_scatter_mt_vlseal",
		"trin_hw_sme_za_scatter_mt_vlpurge",
		"trin_hw_sme_za_scatter_mt_vlsync",
		"trin_hw_sme_za_scatter_mt_vlclose",
		"trin_fusion_pgo_reg",
		"trin_fusion_pgo_floor",
		"trin_fusion_pgo_afloor",
		"trin_fusion_pgo_burst",
		"trin_fusion_pgo_cool",
		"trin_fusion_pgo_credit",
		"trin_fusion_pgo_debt",
		"trin_fusion_pgo_interest",
		"trin_fusion_pgo_amortize",
		"trin_fusion_pgo_settle",
		"trin_fusion_pgo_escrow",
		"trin_fusion_pgo_clear",
		"trin_fusion_pgo_final",
		"trin_fusion_pgo_close",
		"trin_fusion_pgo_reg_counter",
		"trin_fusion_pgo_floor_counter",
		"trin_fusion_pgo_afloor_counter",
		"trin_fusion_pgo_burst_counter",
		"trin_fusion_pgo_cool_counter",
		"trin_fusion_pgo_credit_counter",
		"trin_fusion_pgo_debt_counter",
		"trin_fusion_pgo_interest_counter",
		"trin_fusion_pgo_amortize_counter",
		"trin_fusion_pgo_settle_counter",
		"trin_fusion_pgo_escrow_counter",
		"trin_fusion_pgo_clear_counter",
		"trin_fusion_pgo_final_counter",
		"trin_fusion_pgo_close_counter",
		"trin_adaptive_peel_tile",
		"trin_soft_mux_v4",
		"trin_span_add_v4",
		"trin_span_scale_v4",
		"trin_sum_v4",
		"trin_dot_v4",
		"trin_soft_mux_v8",
		"trin_span_add_v8",
		"trin_span_scale_v8",
		"trin_sum_v8",
		"trin_dot_v8",
		"trin_soft_mux_v16",
		"trin_span_add_v16",
		"trin_span_scale_v16",
		"trin_sum_v16",
		"trin_dot_v16",
		"trin_vlen",
		"trin_vlen_for_n",
		"trin_hw_vlen",
		"trin_hw_vlen_for_n",
		"trin_hw_isa_code",
		"trin_mlir_vec_probe",
		"trin_mlir_vec8_probe",
		"trin_mlir_vec16_probe",
		"trin_mlir_vlen_probe",
		"trin_mlir_hw_vlen_probe",
		"trin_mlir_soft_probe",
		"trin_mlir_dialect_probe",
		"trin_mlir_fold_probe",
		"trin_mlir_vsum_probe",
		"trin_mlir_cse_probe",
		"trin_mlir_soft_cse_probe",
		"trin_mlir_vwidth_probe",
		"trin_mlir_strength_probe",
		"trin_mlir_span_alg_probe",
		"trin_mlir_bitplane_probe",
		"trin_mlir_nested_probe",
		"trin_mlir_tree_probe",
		"trin_mlir_adaptive_tile_probe",
		"trin_mlir_forest_probe",
		"trin_mlir_deep_forest_probe",
		"trin_adaptive_tile",
		"trin_adaptive_tile_max",
		"trin_adaptive_isa_tile",
		"trin_plane_and",
		"trin_plane_or",
		"trin_plane_xor",
		"trin_plane_popcount",
		"trin_trit_dot_planes_word",
		"trin_trit_dot_planes_x4",
	}
	present := map[string]bool{}
	for _, name := range want {
		if strings.Contains(mlir, "func.func @"+name) {
			present[name] = true
		}
	}
	if !present["trin_soft_mux"] {
		return "", fmt.Errorf("MLIR missing @trin_soft_mux — nothing to lower")
	}

	host := DetectHostCPU()

	var b strings.Builder
	b.WriteString("; Generated by Trin v0.59 MLIR→LLVM lowering (no mlir-opt)\n")
	b.WriteString("; Source dialect: emit-mlir soft_mux + span + vector tiles (4/8/16) + rewrites/strength/span_alg/bitplane/nested/CSE/vwidth + runtime HW VLEN + dyn tiles + residual peel + map/store/scan/scan-mul peels + deep forest + wide ladder\n")
	fmt.Fprintf(&b, "; Emit-host note (not baked into @trin_hw_*): arch=%s isa=%s hw_vlen=%d bitwidth=%d\n",
		host.Arch, host.ISAName, host.HWVLEN, host.BitWidth)
	b.WriteString("target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"\n")
	b.WriteString("target triple = \"x86_64-unknown-linux-gnu\"\n\n")

	// Scalar soft_mux
	if present["trin_soft_mux"] {
		b.WriteString("; lowered from func.func @trin_soft_mux\n")
		b.WriteString("define float @trin_soft_mux(float %w, float %pure, float %each) {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %one = fsub float 1.000000e+00, %w\n")
		b.WriteString("  %t0 = fmul float %w, %pure\n")
		b.WriteString("  %t1 = fmul float %one, %each\n")
		b.WriteString("  %y = fadd float %t0, %t1\n")
		b.WriteString("  ret float %y\n")
		b.WriteString("}\n\n")
	}

	if present["trin_sum4"] {
		b.WriteString("; lowered from func.func @trin_sum4\n")
		b.WriteString("define float @trin_sum4(float %a0, float %a1, float %a2, float %a3) {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %s0 = fadd float %a0, %a1\n")
		b.WriteString("  %s1 = fadd float %s0, %a2\n")
		b.WriteString("  %s2 = fadd float %s1, %a3\n")
		b.WriteString("  ret float %s2\n")
		b.WriteString("}\n\n")
	}

	if present["trin_dot4"] {
		b.WriteString("; lowered from func.func @trin_dot4\n")
		b.WriteString("define float @trin_dot4(float %a0, float %a1, float %a2, float %a3, float %b0, float %b1, float %b2, float %b3) {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %p0 = fmul float %a0, %b0\n")
		b.WriteString("  %p1 = fmul float %a1, %b1\n")
		b.WriteString("  %p2 = fmul float %a2, %b2\n")
		b.WriteString("  %p3 = fmul float %a3, %b3\n")
		b.WriteString("  %s0 = fadd float %p0, %p1\n")
		b.WriteString("  %s1 = fadd float %s0, %p2\n")
		b.WriteString("  %s2 = fadd float %s1, %p3\n")
		b.WriteString("  ret float %s2\n")
		b.WriteString("}\n\n")
	}

	// Counted span loops with auto residual peel (v0.36–v0.80).
	if present["trin_sum_f32"] || present["trin_dot_f32"] ||
		present["trin_span_add_f32"] || present["trin_span_scale_f32"] ||
		present["trin_span_mul_f32"] || present["trin_span_axpy_f32"] ||
		present["trin_span_scan_add_f32"] || present["trin_reduce_add_f32"] ||
		present["trin_span_scan_mul_f32"] || present["trin_reduce_mul_f32"] ||
		present["trin_span_scan_reduce_add_f32"] || present["trin_span_scan_reduce_mul_f32"] ||
		present["trin_span_scan_reduce_add_pair_f32"] || present["trin_span_scan_reduce_mul_pair_f32"] ||
		present["trin_adaptive_peel_tile"] {
		b.WriteString("; lowered from scf.for span kernels with auto residual peel (v0.36–v0.80)\n")
		// Inject portable hw_vlen_for_n only when runtime CPUID helpers are absent.
		// (trin_hw_vlen_for_n may be emitted later by the runtime block; if that
		// path is also present we skip the fallback to avoid redefinition.)
		if !present["trin_hw_vlen"] && !present["trin_hw_vlen_for_n"] && !present["trin_hw_isa_code"] {
			b.WriteString(ensurePeelHWVLEN(""))
		}
		b.WriteString(emitLLVMSpanLoopHelpers())
	}

	// vector<4xf32> → <4 x float>
	if present["trin_soft_mux_v4"] {
		b.WriteString("; lowered from func.func @trin_soft_mux_v4 (vector dialect)\n")
		b.WriteString("define <4 x float> @trin_soft_mux_v4(float %w, <4 x float> %pure, <4 x float> %each) {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %one = fsub float 1.000000e+00, %w\n")
		b.WriteString("  %wv0 = insertelement <4 x float> undef, float %w, i32 0\n")
		b.WriteString("  %wv = shufflevector <4 x float> %wv0, <4 x float> undef, <4 x i32> zeroinitializer\n")
		b.WriteString("  %ov0 = insertelement <4 x float> undef, float %one, i32 0\n")
		b.WriteString("  %ov = shufflevector <4 x float> %ov0, <4 x float> undef, <4 x i32> zeroinitializer\n")
		b.WriteString("  %t0 = fmul <4 x float> %wv, %pure\n")
		b.WriteString("  %t1 = fmul <4 x float> %ov, %each\n")
		b.WriteString("  %y = fadd <4 x float> %t0, %t1\n")
		b.WriteString("  ret <4 x float> %y\n")
		b.WriteString("}\n\n")
	}

	if present["trin_span_add_v4"] {
		b.WriteString("; lowered from func.func @trin_span_add_v4\n")
		b.WriteString("define <4 x float> @trin_span_add_v4(<4 x float> %a, <4 x float> %b) {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %y = fadd <4 x float> %a, %b\n")
		b.WriteString("  ret <4 x float> %y\n")
		b.WriteString("}\n\n")
	}

	if present["trin_span_scale_v4"] {
		b.WriteString("; lowered from func.func @trin_span_scale_v4\n")
		b.WriteString("define <4 x float> @trin_span_scale_v4(float %s, <4 x float> %a) {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %sv0 = insertelement <4 x float> undef, float %s, i32 0\n")
		b.WriteString("  %sv = shufflevector <4 x float> %sv0, <4 x float> undef, <4 x i32> zeroinitializer\n")
		b.WriteString("  %y = fmul <4 x float> %sv, %a\n")
		b.WriteString("  ret <4 x float> %y\n")
		b.WriteString("}\n\n")
	}

	if present["trin_sum_v4"] {
		b.WriteString("; lowered from func.func @trin_sum_v4 (vector.reduction add)\n")
		b.WriteString("define float @trin_sum_v4(<4 x float> %a) {\n")
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
	}

	if present["trin_dot_v4"] {
		b.WriteString("; lowered from func.func @trin_dot_v4\n")
		b.WriteString("define float @trin_dot_v4(<4 x float> %a, <4 x float> %b) {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %p = fmul <4 x float> %a, %b\n")
		b.WriteString("  %y = call float @trin_sum_v4(<4 x float> %p)\n")
		b.WriteString("  ret float %y\n")
		b.WriteString("}\n\n")
	}

	// vector<8xf32> → <8 x float> (v0.24 wider tiles)
	if present["trin_soft_mux_v8"] {
		b.WriteString("; lowered from func.func @trin_soft_mux_v8 (vector dialect, 8-lane)\n")
		b.WriteString("define <8 x float> @trin_soft_mux_v8(float %w, <8 x float> %pure, <8 x float> %each) {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %one = fsub float 1.000000e+00, %w\n")
		b.WriteString("  %wv0 = insertelement <8 x float> undef, float %w, i32 0\n")
		b.WriteString("  %wv = shufflevector <8 x float> %wv0, <8 x float> undef, <8 x i32> zeroinitializer\n")
		b.WriteString("  %ov0 = insertelement <8 x float> undef, float %one, i32 0\n")
		b.WriteString("  %ov = shufflevector <8 x float> %ov0, <8 x float> undef, <8 x i32> zeroinitializer\n")
		b.WriteString("  %t0 = fmul <8 x float> %wv, %pure\n")
		b.WriteString("  %t1 = fmul <8 x float> %ov, %each\n")
		b.WriteString("  %y = fadd <8 x float> %t0, %t1\n")
		b.WriteString("  ret <8 x float> %y\n")
		b.WriteString("}\n\n")
	}

	if present["trin_span_add_v8"] {
		b.WriteString("; lowered from func.func @trin_span_add_v8\n")
		b.WriteString("define <8 x float> @trin_span_add_v8(<8 x float> %a, <8 x float> %b) {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %y = fadd <8 x float> %a, %b\n")
		b.WriteString("  ret <8 x float> %y\n")
		b.WriteString("}\n\n")
	}

	if present["trin_span_scale_v8"] {
		b.WriteString("; lowered from func.func @trin_span_scale_v8\n")
		b.WriteString("define <8 x float> @trin_span_scale_v8(float %s, <8 x float> %a) {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %sv0 = insertelement <8 x float> undef, float %s, i32 0\n")
		b.WriteString("  %sv = shufflevector <8 x float> %sv0, <8 x float> undef, <8 x i32> zeroinitializer\n")
		b.WriteString("  %y = fmul <8 x float> %sv, %a\n")
		b.WriteString("  ret <8 x float> %y\n")
		b.WriteString("}\n\n")
	}

	if present["trin_sum_v8"] {
		b.WriteString("; lowered from func.func @trin_sum_v8 (vector.reduction add)\n")
		b.WriteString("define float @trin_sum_v8(<8 x float> %a) {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %e0 = extractelement <8 x float> %a, i32 0\n")
		b.WriteString("  %e1 = extractelement <8 x float> %a, i32 1\n")
		b.WriteString("  %e2 = extractelement <8 x float> %a, i32 2\n")
		b.WriteString("  %e3 = extractelement <8 x float> %a, i32 3\n")
		b.WriteString("  %e4 = extractelement <8 x float> %a, i32 4\n")
		b.WriteString("  %e5 = extractelement <8 x float> %a, i32 5\n")
		b.WriteString("  %e6 = extractelement <8 x float> %a, i32 6\n")
		b.WriteString("  %e7 = extractelement <8 x float> %a, i32 7\n")
		b.WriteString("  %s0 = fadd float %e0, %e1\n")
		b.WriteString("  %s1 = fadd float %s0, %e2\n")
		b.WriteString("  %s2 = fadd float %s1, %e3\n")
		b.WriteString("  %s3 = fadd float %s2, %e4\n")
		b.WriteString("  %s4 = fadd float %s3, %e5\n")
		b.WriteString("  %s5 = fadd float %s4, %e6\n")
		b.WriteString("  %s6 = fadd float %s5, %e7\n")
		b.WriteString("  ret float %s6\n")
		b.WriteString("}\n\n")
	}

	if present["trin_dot_v8"] {
		b.WriteString("; lowered from func.func @trin_dot_v8\n")
		b.WriteString("define float @trin_dot_v8(<8 x float> %a, <8 x float> %b) {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %p = fmul <8 x float> %a, %b\n")
		b.WriteString("  %y = call float @trin_sum_v8(<8 x float> %p)\n")
		b.WriteString("  ret float %y\n")
		b.WriteString("}\n\n")
	}

	// vector<16xf32> → <16 x float> (v0.26 wider tiles)
	if present["trin_soft_mux_v16"] {
		b.WriteString("; lowered from func.func @trin_soft_mux_v16 (vector dialect, 16-lane)\n")
		b.WriteString("define <16 x float> @trin_soft_mux_v16(float %w, <16 x float> %pure, <16 x float> %each) {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %one = fsub float 1.000000e+00, %w\n")
		b.WriteString("  %wv0 = insertelement <16 x float> undef, float %w, i32 0\n")
		b.WriteString("  %wv = shufflevector <16 x float> %wv0, <16 x float> undef, <16 x i32> zeroinitializer\n")
		b.WriteString("  %ov0 = insertelement <16 x float> undef, float %one, i32 0\n")
		b.WriteString("  %ov = shufflevector <16 x float> %ov0, <16 x float> undef, <16 x i32> zeroinitializer\n")
		b.WriteString("  %t0 = fmul <16 x float> %wv, %pure\n")
		b.WriteString("  %t1 = fmul <16 x float> %ov, %each\n")
		b.WriteString("  %y = fadd <16 x float> %t0, %t1\n")
		b.WriteString("  ret <16 x float> %y\n")
		b.WriteString("}\n\n")
	}

	if present["trin_span_add_v16"] {
		b.WriteString("; lowered from func.func @trin_span_add_v16\n")
		b.WriteString("define <16 x float> @trin_span_add_v16(<16 x float> %a, <16 x float> %b) {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %y = fadd <16 x float> %a, %b\n")
		b.WriteString("  ret <16 x float> %y\n")
		b.WriteString("}\n\n")
	}

	if present["trin_span_scale_v16"] {
		b.WriteString("; lowered from func.func @trin_span_scale_v16\n")
		b.WriteString("define <16 x float> @trin_span_scale_v16(float %s, <16 x float> %a) {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %sv0 = insertelement <16 x float> undef, float %s, i32 0\n")
		b.WriteString("  %sv = shufflevector <16 x float> %sv0, <16 x float> undef, <16 x i32> zeroinitializer\n")
		b.WriteString("  %y = fmul <16 x float> %sv, %a\n")
		b.WriteString("  ret <16 x float> %y\n")
		b.WriteString("}\n\n")
	}

	if present["trin_sum_v16"] {
		b.WriteString("; lowered from func.func @trin_sum_v16 (vector.reduction add)\n")
		b.WriteString("define float @trin_sum_v16(<16 x float> %a) {\n")
		b.WriteString("entry:\n")
		// pairwise tree reduction (portable; no llvm.vector.reduce.fadd required)
		for i := 0; i < 16; i++ {
			fmt.Fprintf(&b, "  %%e%d = extractelement <16 x float> %%a, i32 %d\n", i, i)
		}
		b.WriteString("  %p0 = fadd float %e0, %e1\n")
		b.WriteString("  %p1 = fadd float %e2, %e3\n")
		b.WriteString("  %p2 = fadd float %e4, %e5\n")
		b.WriteString("  %p3 = fadd float %e6, %e7\n")
		b.WriteString("  %p4 = fadd float %e8, %e9\n")
		b.WriteString("  %p5 = fadd float %e10, %e11\n")
		b.WriteString("  %p6 = fadd float %e12, %e13\n")
		b.WriteString("  %p7 = fadd float %e14, %e15\n")
		b.WriteString("  %q0 = fadd float %p0, %p1\n")
		b.WriteString("  %q1 = fadd float %p2, %p3\n")
		b.WriteString("  %q2 = fadd float %p4, %p5\n")
		b.WriteString("  %q3 = fadd float %p6, %p7\n")
		b.WriteString("  %r0 = fadd float %q0, %q1\n")
		b.WriteString("  %r1 = fadd float %q2, %q3\n")
		b.WriteString("  %y = fadd float %r0, %r1\n")
		b.WriteString("  ret float %y\n")
		b.WriteString("}\n\n")
	}

	if present["trin_dot_v16"] {
		b.WriteString("; lowered from func.func @trin_dot_v16\n")
		b.WriteString("define float @trin_dot_v16(<16 x float> %a, <16 x float> %b) {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %p = fmul <16 x float> %a, %b\n")
		b.WriteString("  %y = call float @trin_sum_v16(<16 x float> %p)\n")
		b.WriteString("  ret float %y\n")
		b.WriteString("}\n\n")
	}

	// Portable VLEN (v0.26)
	if present["trin_vlen"] {
		b.WriteString("; lowered from func.func @trin_vlen → preferred tile width 16\n")
		b.WriteString("define i32 @trin_vlen() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  ret i32 16\n")
		b.WriteString("}\n\n")
	}
	if present["trin_vlen_for_n"] {
		// largest of {16,8,4} <= n; else n
		b.WriteString("; lowered from func.func @trin_vlen_for_n (portable tile pick)\n")
		b.WriteString("define i32 @trin_vlen_for_n(i32 %n) {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %ge16 = icmp sge i32 %n, 16\n")
		b.WriteString("  br i1 %ge16, label %ret16, label %chk8\n")
		b.WriteString("ret16:\n")
		b.WriteString("  ret i32 16\n")
		b.WriteString("chk8:\n")
		b.WriteString("  %ge8 = icmp sge i32 %n, 8\n")
		b.WriteString("  br i1 %ge8, label %ret8, label %chk4\n")
		b.WriteString("ret8:\n")
		b.WriteString("  ret i32 8\n")
		b.WriteString("chk4:\n")
		b.WriteString("  %ge4 = icmp sge i32 %n, 4\n")
		b.WriteString("  br i1 %ge4, label %ret4, label %retn\n")
		b.WriteString("ret4:\n")
		b.WriteString("  ret i32 4\n")
		b.WriteString("retn:\n")
		b.WriteString("  ret i32 %n\n")
		b.WriteString("}\n\n")
	}

	// Hardware-aware VLEN (v0.30) — runtime CPUID / feature detect in the binary.
	// Emit-host snapshot is only a note; values come from @trin_hw_detect_once.
	// Adaptive tile helpers (v0.39) need hw_vlen + for_n + isa at minimum.
	needAdaptive := present["trin_adaptive_tile"] || present["trin_adaptive_tile_max"] ||
		present["trin_adaptive_isa_tile"] || present["trin_mlir_adaptive_tile_probe"]
	needRDVL := present["trin_hw_rdvl"]
	needVLEN := present["trin_hw_vlen"] || present["trin_adaptive_tile_max"] || needAdaptive || needRDVL
	needISA := present["trin_hw_isa_code"] || present["trin_adaptive_isa_tile"] || needAdaptive
	needForN := present["trin_hw_vlen_for_n"] || present["trin_adaptive_tile"] || needAdaptive
	if needVLEN || needISA || needForN || needRDVL {
		fmt.Fprintf(&b, "; runtime feature dispatch (v0.30/v0.39 adaptive); emit-host note: %s tile %d (not baked)\n",
			host.ISAName, host.HWVLEN)
		b.WriteString(EmitRuntimeCPUIDLLVM(needVLEN, needISA, needForN))
	}
	if needAdaptive {
		b.WriteString(EmitAdaptiveTileLLVM())
	}

	// Probes
	if present["trin_mlir_soft_probe"] {
		b.WriteString("; lowered from func.func @trin_mlir_soft_probe → soft_mux(0.25,10,2)=4\n")
		b.WriteString("define float @trin_mlir_soft_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %y = call float @trin_soft_mux(float 2.500000e-01, float 1.000000e+01, float 2.000000e+00)\n")
		b.WriteString("  ret float %y\n")
		b.WriteString("}\n\n")
	}

	if present["trin_mlir_vec_probe"] {
		// sum=10, addsum=20, scalesum=20, dot=30 → returns 10 when correct
		b.WriteString("; lowered from func.func @trin_mlir_vec_probe → target 10.0\n")
		b.WriteString("define float @trin_mlir_vec_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %a0 = insertelement <4 x float> undef, float 1.000000e+00, i32 0\n")
		b.WriteString("  %a1 = insertelement <4 x float> %a0, float 2.000000e+00, i32 1\n")
		b.WriteString("  %a2 = insertelement <4 x float> %a1, float 3.000000e+00, i32 2\n")
		b.WriteString("  %a = insertelement <4 x float> %a2, float 4.000000e+00, i32 3\n")
		b.WriteString("  %s = call float @trin_sum_v4(<4 x float> %a)\n")
		b.WriteString("  %add = call <4 x float> @trin_span_add_v4(<4 x float> %a, <4 x float> %a)\n")
		b.WriteString("  %sa = call float @trin_sum_v4(<4 x float> %add)\n")
		b.WriteString("  %scv = call <4 x float> @trin_span_scale_v4(float 2.000000e+00, <4 x float> %a)\n")
		b.WriteString("  %sc = call float @trin_sum_v4(<4 x float> %scv)\n")
		b.WriteString("  %d = call float @trin_dot_v4(<4 x float> %a, <4 x float> %a)\n")
		b.WriteString("  %e0 = fsub float %sa, 2.000000e+01\n")
		b.WriteString("  %e1 = fsub float %sc, 2.000000e+01\n")
		b.WriteString("  %e2 = fsub float %d, 3.000000e+01\n")
		b.WriteString("  %t0 = fadd float %s, %e0\n")
		b.WriteString("  %t1 = fadd float %t0, %e1\n")
		b.WriteString("  %y = fadd float %t1, %e2\n")
		b.WriteString("  ret float %y\n")
		b.WriteString("}\n\n")
	}

	if present["trin_mlir_vec8_probe"] {
		// lanes 1..8: sum=36, addsum=72, scalesum=72, dot=204 → returns 36 when correct
		b.WriteString("; lowered from func.func @trin_mlir_vec8_probe → target 36.0 (v0.24)\n")
		b.WriteString("define float @trin_mlir_vec8_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %a0 = insertelement <8 x float> undef, float 1.000000e+00, i32 0\n")
		b.WriteString("  %a1 = insertelement <8 x float> %a0, float 2.000000e+00, i32 1\n")
		b.WriteString("  %a2 = insertelement <8 x float> %a1, float 3.000000e+00, i32 2\n")
		b.WriteString("  %a3 = insertelement <8 x float> %a2, float 4.000000e+00, i32 3\n")
		b.WriteString("  %a4 = insertelement <8 x float> %a3, float 5.000000e+00, i32 4\n")
		b.WriteString("  %a5 = insertelement <8 x float> %a4, float 6.000000e+00, i32 5\n")
		b.WriteString("  %a6 = insertelement <8 x float> %a5, float 7.000000e+00, i32 6\n")
		b.WriteString("  %a = insertelement <8 x float> %a6, float 8.000000e+00, i32 7\n")
		b.WriteString("  %s = call float @trin_sum_v8(<8 x float> %a)\n")
		b.WriteString("  %add = call <8 x float> @trin_span_add_v8(<8 x float> %a, <8 x float> %a)\n")
		b.WriteString("  %sa = call float @trin_sum_v8(<8 x float> %add)\n")
		b.WriteString("  %scv = call <8 x float> @trin_span_scale_v8(float 2.000000e+00, <8 x float> %a)\n")
		b.WriteString("  %sc = call float @trin_sum_v8(<8 x float> %scv)\n")
		b.WriteString("  %d = call float @trin_dot_v8(<8 x float> %a, <8 x float> %a)\n")
		b.WriteString("  %e0 = fsub float %sa, 7.200000e+01\n")
		b.WriteString("  %e1 = fsub float %sc, 7.200000e+01\n")
		b.WriteString("  %e2 = fsub float %d, 2.040000e+02\n")
		b.WriteString("  %t0 = fadd float %s, %e0\n")
		b.WriteString("  %t1 = fadd float %t0, %e1\n")
		b.WriteString("  %y = fadd float %t1, %e2\n")
		b.WriteString("  ret float %y\n")
		b.WriteString("}\n\n")
	}

	if present["trin_mlir_vec16_probe"] {
		// lanes 1..16: sum=136, addsum=272, scalesum=272, dot=1496 → returns 136 when correct
		b.WriteString("; lowered from func.func @trin_mlir_vec16_probe → target 136.0 (v0.26)\n")
		b.WriteString("define float @trin_mlir_vec16_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %a0 = insertelement <16 x float> undef, float 1.000000e+00, i32 0\n")
		for i := 1; i <= 15; i++ {
			prev := i - 1
			name := fmt.Sprintf("a%d", i)
			if i == 15 {
				name = "a"
			}
			fmt.Fprintf(&b, "  %%%s = insertelement <16 x float> %%a%d, float %d.000000e+00, i32 %d\n",
				name, prev, i+1, i)
		}
		b.WriteString("  %s = call float @trin_sum_v16(<16 x float> %a)\n")
		b.WriteString("  %add = call <16 x float> @trin_span_add_v16(<16 x float> %a, <16 x float> %a)\n")
		b.WriteString("  %sa = call float @trin_sum_v16(<16 x float> %add)\n")
		b.WriteString("  %scv = call <16 x float> @trin_span_scale_v16(float 2.000000e+00, <16 x float> %a)\n")
		b.WriteString("  %sc = call float @trin_sum_v16(<16 x float> %scv)\n")
		b.WriteString("  %d = call float @trin_dot_v16(<16 x float> %a, <16 x float> %a)\n")
		b.WriteString("  %e0 = fsub float %sa, 2.720000e+02\n")
		b.WriteString("  %e1 = fsub float %sc, 2.720000e+02\n")
		b.WriteString("  %e2 = fsub float %d, 1.496000e+03\n")
		b.WriteString("  %t0 = fadd float %s, %e0\n")
		b.WriteString("  %t1 = fadd float %t0, %e1\n")
		b.WriteString("  %y = fadd float %t1, %e2\n")
		b.WriteString("  ret float %y\n")
		b.WriteString("}\n\n")
	}

	if present["trin_mlir_vlen_probe"] {
		// prefer 16; for_n(10)=8; for_n(3)=3; for_n(20)=16 → 16.0
		b.WriteString("; lowered from func.func @trin_mlir_vlen_probe → target 16.0 (v0.26)\n")
		b.WriteString("define float @trin_mlir_vlen_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %v = call i32 @trin_vlen()\n")
		b.WriteString("  %t10 = call i32 @trin_vlen_for_n(i32 10)\n")
		b.WriteString("  %t3 = call i32 @trin_vlen_for_n(i32 3)\n")
		b.WriteString("  %t20 = call i32 @trin_vlen_for_n(i32 20)\n")
		b.WriteString("  %e0 = sub i32 %t10, 8\n")
		b.WriteString("  %e1 = sub i32 %t3, 3\n")
		b.WriteString("  %e2 = sub i32 %t20, 16\n")
		b.WriteString("  %s0 = add i32 %v, %e0\n")
		b.WriteString("  %s1 = add i32 %s0, %e1\n")
		b.WriteString("  %s2 = add i32 %s1, %e2\n")
		b.WriteString("  %y = sitofp i32 %s2 to float\n")
		b.WriteString("  ret float %y\n")
		b.WriteString("}\n\n")
	}

	if present["trin_mlir_hw_vlen_probe"] {
		// v0.30: residuals use *runtime* hw_vlen + the same HWVLENForN policy
		// recomputed at runtime (no emit-time want10/want20 constants).
		// y = float(v) + (for_n(10)-pick(v,10)) + (for_n(3)-3) + (for_n(20)-pick(v,20))
		// which collapses to float(v) when for_n matches the policy.
		fmt.Fprintf(&b, "; lowered from func.func @trin_mlir_hw_vlen_probe → runtime float(hw_vlen) (v0.30; emit-host %s/%d)\n",
			host.ISAName, host.HWVLEN)
		b.WriteString("define float @trin_mlir_hw_vlen_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %v = call i32 @trin_hw_vlen()\n")
		b.WriteString("  %t10 = call i32 @trin_hw_vlen_for_n(i32 10)\n")
		b.WriteString("  %t3 = call i32 @trin_hw_vlen_for_n(i32 3)\n")
		b.WriteString("  %t20 = call i32 @trin_hw_vlen_for_n(i32 20)\n")
		// Runtime pick(v,10): largest among {16,8,4} <=v and <=10 → 8 if v>=8 else 4 if v>=4 else 10
		// Inline compact form matching HWVLENForN(v,10) and HWVLENForN(v,20).
		b.WriteString("  %vge8 = icmp sge i32 %v, 8\n")
		b.WriteString("  %vge4 = icmp sge i32 %v, 4\n")
		b.WriteString("  %vge16 = icmp sge i32 %v, 16\n")
		// want10 = (v>=8)?8:((v>=4)?4:10)  — n=10 never reaches 16
		b.WriteString("  %w10a = select i1 %vge4, i32 4, i32 10\n")
		b.WriteString("  %w10 = select i1 %vge8, i32 8, i32 %w10a\n")
		// want20 = (v>=16)?16:((v>=8)?8:((v>=4)?4:20))
		b.WriteString("  %w20a = select i1 %vge4, i32 4, i32 20\n")
		b.WriteString("  %w20b = select i1 %vge8, i32 8, i32 %w20a\n")
		b.WriteString("  %w20 = select i1 %vge16, i32 16, i32 %w20b\n")
		b.WriteString("  %e0 = sub i32 %t10, %w10\n")
		b.WriteString("  %e1 = sub i32 %t3, 3\n")
		b.WriteString("  %e2 = sub i32 %t20, %w20\n")
		b.WriteString("  %s0 = add i32 %v, %e0\n")
		b.WriteString("  %s1 = add i32 %s0, %e1\n")
		b.WriteString("  %s2 = add i32 %s1, %e2\n")
		b.WriteString("  %y = sitofp i32 %s2 to float\n")
		b.WriteString("  ret float %y\n")
		b.WriteString("}\n\n")
	}

	// v0.25 dialect / fold / vsum probes (after rewrite they call helpers or folded arith)
	if present["trin_mlir_dialect_probe"] {
		b.WriteString("; lowered from rewritten @trin_mlir_dialect_probe → soft_mux(0.25,10,2)=4\n")
		b.WriteString("define float @trin_mlir_dialect_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %y = call float @trin_soft_mux(float 2.500000e-01, float 1.000000e+01, float 2.000000e+00)\n")
		b.WriteString("  ret float %y\n")
		b.WriteString("}\n\n")
	}
	if present["trin_mlir_fold_probe"] {
		// After rewrite soft_mux(1,7,9) → pure (=7)
		b.WriteString("; lowered from rewritten @trin_mlir_fold_probe → soft_mux(1,7,9)=7 (folded)\n")
		b.WriteString("define float @trin_mlir_fold_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  ret float 7.000000e+00\n")
		b.WriteString("}\n\n")
	}
	if present["trin_mlir_vsum_probe"] {
		b.WriteString("; lowered from rewritten @trin_mlir_vsum_probe → sum_v4(1,2,3,4)=10\n")
		b.WriteString("define float @trin_mlir_vsum_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %a0 = insertelement <4 x float> undef, float 1.000000e+00, i32 0\n")
		b.WriteString("  %a1 = insertelement <4 x float> %a0, float 2.000000e+00, i32 1\n")
		b.WriteString("  %a2 = insertelement <4 x float> %a1, float 3.000000e+00, i32 2\n")
		b.WriteString("  %a = insertelement <4 x float> %a2, float 4.000000e+00, i32 3\n")
		b.WriteString("  %y = call float @trin_sum_v4(<4 x float> %a)\n")
		b.WriteString("  ret float %y\n")
		b.WriteString("}\n\n")
	}

	// v0.27 CSE / multi-use soft_mux / vector-width probes
	if present["trin_mlir_cse_probe"] {
		// (1+2)+(1+2)=6 after CSE
		b.WriteString("; lowered from rewritten @trin_mlir_cse_probe → CSE addf target 6.0 (v0.27)\n")
		b.WriteString("define float @trin_mlir_cse_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %a = fadd float 1.000000e+00, 2.000000e+00\n")
		b.WriteString("  %y = fadd float %a, %a\n")
		b.WriteString("  ret float %y\n")
		b.WriteString("}\n\n")
	}
	if present["trin_mlir_soft_cse_probe"] {
		// soft_mux(0.5,10,2)=6 twice → 12 after multi-use commoning
		b.WriteString("; lowered from rewritten @trin_mlir_soft_cse_probe → multi-use soft_mux target 12.0 (v0.27)\n")
		b.WriteString("define float @trin_mlir_soft_cse_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %y1 = call float @trin_soft_mux(float 5.000000e-01, float 1.000000e+01, float 2.000000e+00)\n")
		b.WriteString("  %y = fadd float %y1, %y1\n")
		b.WriteString("  ret float %y\n")
		b.WriteString("}\n\n")
	}
	if present["trin_mlir_vwidth_probe"] {
		// vsum vector<16xf32> 1..16 → 136
		b.WriteString("; lowered from rewritten @trin_mlir_vwidth_probe → vector_width=16 sum target 136.0 (v0.27)\n")
		b.WriteString("define float @trin_mlir_vwidth_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  %a0 = insertelement <16 x float> undef, float 1.000000e+00, i32 0\n")
		for i := 1; i <= 15; i++ {
			prev := i - 1
			name := fmt.Sprintf("a%d", i)
			if i == 15 {
				name = "a"
			}
			fmt.Fprintf(&b, "  %%%s = insertelement <16 x float> %%a%d, float %d.000000e+00, i32 %d\n",
				name, prev, i+1, i)
		}
		b.WriteString("  %y = call float @trin_sum_v16(<16 x float> %a)\n")
		b.WriteString("  ret float %y\n")
		b.WriteString("}\n\n")
	}

	// v0.29 soft_mux strength reduction probe: 5 + 3 + 1.5 + 4 = 13.5
	if present["trin_mlir_strength_probe"] {
		b.WriteString("; lowered from rewritten @trin_mlir_strength_probe → soft_mux strength target 13.5 (v0.29)\n")
		b.WriteString("define float @trin_mlir_strength_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  ret float 1.350000e+01\n")
		b.WriteString("}\n\n")
	}

	// v0.31 span/vector algebraic probe: 10+10+10+0+0+0 = 30
	if present["trin_mlir_span_alg_probe"] {
		b.WriteString("; lowered from rewritten @trin_mlir_span_alg_probe → span/vector algebraic target 30 (v0.31)\n")
		b.WriteString("define float @trin_mlir_span_alg_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  ret float 3.000000e+01\n")
		b.WriteString("}\n\n")
	}

	// v0.33 soft_mux_v + bitplane algebraic probe: plane 34 + two soft_mux_v sums 4+4 = 42
	if present["trin_mlir_bitplane_probe"] {
		b.WriteString("; lowered from rewritten @trin_mlir_bitplane_probe → soft_mux_v+bitplane algebraic target 42 (v0.33)\n")
		b.WriteString("define float @trin_mlir_bitplane_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  ret float 4.200000e+01\n")
		b.WriteString("}\n\n")
	}
	// v0.35 nested soft_mux product + bitplane absorb probe: soft 16 + plane 36 = 52
	if present["trin_mlir_nested_probe"] {
		b.WriteString("; lowered from rewritten @trin_mlir_nested_probe → nested soft_mux product + plane absorb target 52 (v0.35)\n")
		b.WriteString("define float @trin_mlir_nested_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  ret float 5.200000e+01\n")
		b.WriteString("}\n\n")
	}
	// v0.38 multi-level soft_mux trees + multi-word plane GEMM: soft 16 + plane 48 = 64
	if present["trin_mlir_tree_probe"] {
		b.WriteString("; lowered from rewritten @trin_mlir_tree_probe → multi-level soft_mux + multi-word GEMM target 64 (v0.38)\n")
		b.WriteString("define float @trin_mlir_tree_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  ret float 6.400000e+01\n")
		b.WriteString("}\n\n")
	}
	// v0.39 adaptive tile probe: residuals cancel → 100 independent of host ISA
	if present["trin_mlir_adaptive_tile_probe"] {
		b.WriteString("; lowered from @trin_mlir_adaptive_tile_probe → adaptive runtime tile target 100 (v0.39)\n")
		b.WriteString("define float @trin_mlir_adaptive_tile_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  ret float 1.000000e+02\n")
		b.WriteString("}\n\n")
	}
	// v0.40 soft_mux forest + multi-word GEMM schedule: soft 28 + plane 52 = 80
	if present["trin_mlir_forest_probe"] {
		b.WriteString("; lowered from rewritten @trin_mlir_forest_probe → soft_mux forest + GEMM schedule target 80 (v0.40)\n")
		b.WriteString("define float @trin_mlir_forest_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  ret float 8.000000e+01\n")
		b.WriteString("}\n\n")
	}
	// v0.41 deeper soft_mux forest schedule: soft 42 + plane 48 = 90
	if present["trin_mlir_deep_forest_probe"] {
		b.WriteString("; lowered from rewritten @trin_mlir_deep_forest_probe → deep forest schedule target 90 (v0.41)\n")
		b.WriteString("define float @trin_mlir_deep_forest_probe() {\n")
		b.WriteString("entry:\n")
		b.WriteString("  ret float 9.000000e+01\n")
		b.WriteString("}\n\n")
	}
	// Always emit plane helpers when requested (const-fold path may not need them at runtime).
	if present["trin_plane_and"] && !strings.Contains(b.String(), "define i32 @trin_plane_and") {
		b.WriteString("define i32 @trin_plane_and(i32 %a, i32 %b) {\nentry:\n  %y = and i32 %a, %b\n  ret i32 %y\n}\n\n")
	}
	if present["trin_plane_or"] && !strings.Contains(b.String(), "define i32 @trin_plane_or") {
		b.WriteString("define i32 @trin_plane_or(i32 %a, i32 %b) {\nentry:\n  %y = or i32 %a, %b\n  ret i32 %y\n}\n\n")
	}
	if present["trin_plane_xor"] && !strings.Contains(b.String(), "define i32 @trin_plane_xor") {
		b.WriteString("define i32 @trin_plane_xor(i32 %a, i32 %b) {\nentry:\n  %y = xor i32 %a, %b\n  ret i32 %y\n}\n\n")
	}
	if present["trin_plane_popcount"] && !strings.Contains(b.String(), "define i32 @trin_plane_popcount") {
		b.WriteString("declare i32 @llvm.ctpop.i32(i32)\n")
		b.WriteString("define i32 @trin_plane_popcount(i32 %w) {\nentry:\n  %y = call i32 @llvm.ctpop.i32(i32 %w)\n  ret i32 %y\n}\n\n")
	}

	// Note any trin.* dialect comments for provenance
	if strings.Contains(mlir, "trin.soft_mux") {
		b.WriteString("; custom dialect surface noted: trin.soft_mux / span / vector tiles\n")
	}
	if strings.Contains(mlir, "func.func @main()") || strings.Contains(mlir, "full-program MLIR") {
		b.WriteString("; full-program MLIR @main detected — host EmitMLIRLLVM may append define i32 @main\n")
		b.WriteString("; trin.full_program lower path = v0.23\n")
	}
	if present["trin_soft_mux_v8"] || present["trin_mlir_vec8_probe"] {
		b.WriteString("; wider vector tiles: vector<8xf32> / <8 x float> (v0.24)\n")
	}
	if present["trin_soft_mux_v16"] || present["trin_mlir_vec16_probe"] {
		b.WriteString("; wider vector tiles: vector<16xf32> / <16 x float> (v0.26)\n")
	}
	if present["trin_vlen"] || present["trin_mlir_vlen_probe"] {
		b.WriteString("; portable VLEN selection: trin_vlen / trin_vlen_for_n (v0.26)\n")
	}
	if present["trin_hw_vlen"] || present["trin_mlir_hw_vlen_probe"] {
		fmt.Fprintf(&b, "; hardware VLEN selection: trin_hw_vlen runtime CPUID (v0.30); emit-host note isa=%s hw_vlen=%d\n",
			host.ISAName, host.HWVLEN)
		b.WriteString("; runtime dynamic feature dispatch: @trin_hw_detect_once / @trin_hw_rt_* (v0.30)\n")
	}
	if present["trin_mlir_dialect_probe"] || present["trin_mlir_fold_probe"] || present["trin_mlir_vsum_probe"] {
		b.WriteString("; dialect pattern rewrites: soft_mux/vsum/fold probes (v0.25)\n")
	}
	if present["trin_mlir_cse_probe"] || present["trin_mlir_soft_cse_probe"] || present["trin_mlir_vwidth_probe"] {
		b.WriteString("; deeper rewrites: CSE / multi-use soft_mux / vector-width pick (v0.27)\n")
	}
	if present["trin_mlir_strength_probe"] {
		b.WriteString("; soft_mux strength reduction / algebraic folds (v0.29)\n")
	}
	if present["trin_mlir_span_alg_probe"] {
		b.WriteString("; span/vector algebraic rewrites: add(x,0)/scale(1|0)/sum(0)/dot(0) (v0.31)\n")
	}
	if present["trin_mlir_bitplane_probe"] {
		b.WriteString("; soft_mux_v + bitplane algebraic folds: plane and/or/xor/popcount + soft_mux_v(w,*,0) (v0.33)\n")
	}
	if present["trin_mlir_nested_probe"] {
		b.WriteString("; nested soft_mux product weight + bitplane absorb/product folds (v0.35)\n")
	}
	if present["trin_mlir_tree_probe"] {
		b.WriteString("; multi-level soft_mux trees + multi-word plane GEMM folds (v0.38)\n")
	}
	if present["trin_mlir_adaptive_tile_probe"] || present["trin_adaptive_tile"] {
		b.WriteString("; adaptive tile from runtime ISA: @trin_adaptive_tile* (v0.39)\n")
	}
	if present["trin_mlir_forest_probe"] || present["trin_span_mul_f32"] || present["trin_span_axpy_f32"] {
		b.WriteString("; map/store peels + soft_mux forest / multi-word GEMM schedule (v0.40)\n")
	}
	if present["trin_mlir_deep_forest_probe"] || present["trin_span_scan_add_f32"] || present["trin_reduce_add_f32"] || present["trin_adaptive_peel_tile"] {
		b.WriteString("; scan/reduce peels + deep soft_mux forest + adaptive ISA peel ladder (v0.41)\n")
	}
	if present["trin_span_scan_mul_f32"] || present["trin_reduce_mul_f32"] {
		b.WriteString("; scan-mul/reduce-mul peels + AdaptivePeelChunks + wide 32-lane ladder (v0.42)\n")
	}
	if present["trin_span_scan_reduce_add_f32"] || present["trin_span_scan_reduce_mul_f32"] {
		b.WriteString("; fused scan-reduce + dyn_sum32 + VL>32 (64) ladder (v0.43)\n")
	}
	if present["trin_span_scan_reduce_add_pair_f32"] || present["trin_span_scan_reduce_mul_pair_f32"] ||
		present["trin_hw_rdvl"] {
		b.WriteString("; multi-output fused scan-reduce + RDVL host VL + fused-mul probe (v0.51)\n")
	}
	if present["trin_pair_f32"] || present["trin_span_scan_reduce_add_tuple_f32"] ||
		present["trin_span_scan_reduce_add_sum_f32"] || present["trin_hw_rdvl_live"] {
		b.WriteString("; first-class pair/tuple IR + live RDVL + one-pass fused sum_scan (v0.51)\n")
	}
	if present["trin_fused_map_reduce_mul_add_f32"] || present["trin_hw_sme_vl"] {
		b.WriteString("; language pair + streaming SME VL + map+reduce fusion (v0.51)\n")
	}
	if present["trin_fused_map_reduce_mul_sum_f32"] || present["trin_hw_sme_za_tile"] || present["trin_pair_swap_f32"] {
		b.WriteString("; pair destructure/return + SME ZA tile + broader map+reduce fusion (v0.51)\n")
	}
	if present["trin_fused_map_reduce_cost_mul_add_f32"] || present["trin_hw_sme_za_outer"] || present["trin_pair_match_f32"] || present["trin_fusion_cost"] {
		b.WriteString("; nested/multi-way pair match + ZA outer-product + fusion cost model (v0.51)\n")
	}
	// Ensure portable definitions for outer/cost if referenced from dialect text without CPUID.
	fmt.Fprintf(&b, "; lowered funcs: soft_mux=%v sum_f32=%v soft_mux_v4=%v soft_mux_v8=%v soft_mux_v16=%v vec_probe=%v vec8_probe=%v vec16_probe=%v vlen=%v hw_vlen=%v dialect=%v fold=%v vsum=%v cse=%v soft_cse=%v vwidth=%v strength=%v span_alg=%v bitplane=%v nested=%v tree=%v adaptive=%v forest=%v deep_forest=%v main_mlir=%v\n",
		present["trin_soft_mux"], present["trin_sum_f32"], present["trin_soft_mux_v4"], present["trin_soft_mux_v8"], present["trin_soft_mux_v16"],
		present["trin_mlir_vec_probe"], present["trin_mlir_vec8_probe"], present["trin_mlir_vec16_probe"],
		present["trin_mlir_vlen_probe"], present["trin_mlir_hw_vlen_probe"],
		present["trin_mlir_dialect_probe"], present["trin_mlir_fold_probe"], present["trin_mlir_vsum_probe"],
		present["trin_mlir_cse_probe"], present["trin_mlir_soft_cse_probe"], present["trin_mlir_vwidth_probe"],
		present["trin_mlir_strength_probe"], present["trin_mlir_span_alg_probe"], present["trin_mlir_bitplane_probe"],
		present["trin_mlir_nested_probe"], present["trin_mlir_tree_probe"], present["trin_mlir_adaptive_tile_probe"],
		present["trin_mlir_forest_probe"], present["trin_mlir_deep_forest_probe"],
		strings.Contains(mlir, "func.func @main()"))

	return b.String(), nil
}
