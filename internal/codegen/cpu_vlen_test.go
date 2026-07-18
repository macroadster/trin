package codegen

import (
	"fmt"
	"runtime"
	"strings"
	"testing"

	"github.com/macroadster/trin/internal/parser"
)

func TestDetectHostCPU(t *testing.T) {
	h := DetectHostCPU()
	if h.Arch != runtime.GOARCH {
		t.Fatalf("arch: got %s want %s", h.Arch, runtime.GOARCH)
	}
	switch h.HWVLEN {
	case 4, 8, 16, 32:
	default:
		t.Fatalf("HWVLEN must be 4/8/16/32, got %d", h.HWVLEN)
	}
	if h.ISAName == "" {
		t.Fatal("empty ISAName")
	}
	// Idempotent snapshot
	h2 := DetectHostCPU()
	if h2.HWVLEN != h.HWVLEN || h2.ISA != h.ISA {
		t.Fatalf("DetectHostCPU not stable: %+v vs %+v", h, h2)
	}
}

func TestDetectHostVLENX86(t *testing.T) {
	if runtime.GOARCH != "amd64" && runtime.GOARCH != "386" {
		t.Skip("x86-only check")
	}
	h := DetectHostCPU()
	// This CI host historically has AVX without AVX-512 → 8.
	// Soft assert: if flags readable, mapping must match helpers.
	if len(h.Flags) == 0 {
		t.Log("no /proc/cpuinfo flags; skip flag mapping assert")
		return
	}
	want := 16
	wantISA := ISAAVX512
	wantName := "avx512"
	switch {
	case h.Flags["avx512f"] || h.Flags["avx512"]:
		want, wantISA, wantName = 16, ISAAVX512, "avx512"
	case h.Flags["avx"] || h.Flags["avx2"]:
		want, wantISA, wantName = 8, ISAAVX, "avx"
	case h.Flags["sse2"] || h.Flags["sse"]:
		want, wantISA, wantName = 4, ISASSE, "sse"
	}
	if h.HWVLEN != want || h.ISA != wantISA || h.ISAName != wantName {
		t.Fatalf("got vlen=%d isa=%d name=%s; want %d/%d/%s (flags has avx=%v avx512f=%v)",
			h.HWVLEN, h.ISA, h.ISAName, want, wantISA, wantName, h.Flags["avx"], h.Flags["avx512f"])
	}
}

func TestHWVLENForN(t *testing.T) {
	// hw=8: cap at 8
	if HWVLENForN(8, 20) != 8 {
		t.Fatal("hw=8 n=20 → 8")
	}
	if HWVLENForN(8, 10) != 8 {
		t.Fatal("hw=8 n=10 → 8")
	}
	if HWVLENForN(8, 5) != 4 {
		t.Fatal("hw=8 n=5 → 4")
	}
	if HWVLENForN(8, 3) != 3 {
		t.Fatal("hw=8 n=3 → 3")
	}
	// hw=16 portable
	if HWVLENForN(16, 20) != 16 {
		t.Fatal("hw=16 n=20 → 16")
	}
	if HWVLENForN(16, 10) != 8 {
		t.Fatal("hw=16 n=10 → 8")
	}
	// hw=4
	if HWVLENForN(4, 20) != 4 {
		t.Fatal("hw=4 n=20 → 4")
	}
	// v0.42/v0.52 wide ladder
	if HWVLENForN(32, 40) != 32 {
		t.Fatal("hw=32 n=40 → 32")
	}
	if HWVLENForN(32, 20) != 16 {
		t.Fatal("hw=32 n=20 → 16")
	}
	if HWVLENForN(64, 80) != 64 {
		t.Fatal("hw=64 n=80 → 64")
	}
	if HWVLENForN(64, 40) != 32 {
		t.Fatal("hw=64 n=40 → 32")
	}
}

func TestEmitRuntimeCPUIDLLVM(t *testing.T) {
	s := EmitRuntimeCPUIDLLVM(true, true, true)
	for _, want := range []string{
		"runtime CPU feature detect (v0.30)",
		"@trin_hw_rt_init",
		"@trin_hw_rt_isa",
		"@trin_hw_rt_vlen",
		"define internal void @trin_hw_detect_once",
		"define i32 @trin_hw_vlen",
		"define i32 @trin_hw_isa_code",
		"define i32 @trin_hw_vlen_for_n",
		"define i32 @trin_hw_rdvl",
		"define i32 @trin_hw_rdvl_live",
		"define i32 @trin_hw_sme_vl",
		"define i32 @trin_hw_sme_za_tile",
		"define i32 @trin_fusion_cost",
		"define i32 @trin_fusion_select",
		"define i32 @trin_fusion_cost_isa",
		"define i32 @trin_fusion_cost_weighted",
		"define i32 @trin_hw_sme_za_outer_acc",
		"define i32 @trin_hw_sme_za_outer",
		"define i32 @trin_hw_sme_za_outer_acc_true",
		"define i32 @trin_fusion_cost_calibrated",
		"define i32 @trin_fusion_plan_log",
		"define i32 @trin_hw_sme_za_stream_store",
		"define i32 @trin_fusion_pgo_weight",
		"define i32 @trin_fusion_pgo_feedback",
		"define i32 @trin_fusion_pgo_select",
		"define i32 @trin_hw_sme_za_gather",
		"define i32 @trin_hw_sme_za_scatter",
		"define i32 @trin_fusion_pgo_adapt",
		"define i32 @trin_fusion_pgo_counter_bump",
		"define i32 @trin_hw_sme_za_gather_mv",
		"define i32 @trin_hw_sme_za_scatter_mv",
		"define i32 @trin_fusion_pgo_isa_counter",
		"define i32 @trin_fusion_pgo_isa_loop",
		"define i32 @trin_hw_sme_za_gather_tile",
		"define i32 @trin_hw_sme_za_scatter_tile",
		"define i32 @trin_fusion_pgo_isa_live",
		"define i32 @trin_fusion_pgo_isa_feedback",
		"define i32 @trin_hw_sme_za_gather_mt",
		"define i32 @trin_hw_sme_za_scatter_mt",
		"define i32 @trin_fusion_pgo_xkernel",
		"define i32 @trin_fusion_pgo_xkernel_counter",
		"define i32 @trin_hw_sme_za_gather_mt_peel",
		"define i32 @trin_hw_sme_za_scatter_mt_peel",
		"define i32 @trin_fusion_pgo_xpass",
		"define i32 @trin_fusion_pgo_xpass_counter",
		"define i32 @trin_hw_sme_za_gather_mt_auto",
		"define i32 @trin_hw_sme_za_scatter_mt_auto",
		"define i32 @trin_fusion_pgo_window",
		"define i32 @trin_fusion_pgo_window_counter",
		"define i32 @trin_hw_sme_za_gather_mt_vlfb",
		"define i32 @trin_hw_sme_za_scatter_mt_vlfb",
		"define i32 @trin_fusion_pgo_decay",
		"define i32 @trin_fusion_pgo_decay_counter",
		"define i32 @trin_hw_sme_za_gather_mt_vllive",
		"define i32 @trin_hw_sme_za_scatter_mt_vllive",
		"define i32 @trin_hw_sme_za_gather_mt_vlstream",
		"define i32 @trin_hw_sme_za_scatter_mt_vlstream",
		"define i32 @trin_hw_sme_za_gather_mt_vlcost",
		"define i32 @trin_hw_sme_za_scatter_mt_vlcost",
		"define i32 @trin_hw_sme_za_gather_mt_vladapt",
		"define i32 @trin_hw_sme_za_scatter_mt_vladapt",
		"define i32 @trin_hw_sme_za_gather_mt_vlsched",
		"define i32 @trin_hw_sme_za_scatter_mt_vlsched",
		"define i32 @trin_hw_sme_za_gather_mt_vlfuse",
		"define i32 @trin_hw_sme_za_scatter_mt_vlfuse",
		"define i32 @trin_fusion_pgo_reweight",
		"define i32 @trin_fusion_pgo_reweight_counter",
		"define i32 @trin_fusion_pgo_budget",
		"define i32 @trin_fusion_pgo_budget_counter",
		"define i32 @trin_fusion_pgo_softcap",
		"define i32 @trin_fusion_pgo_softcap_counter",
		"define i32 @trin_hw_sme_za_gather_mt_vlmix",
		"define i32 @trin_hw_sme_za_scatter_mt_vlmix",
		"define i32 @trin_fusion_pgo_dual",
		"define i32 @trin_fusion_pgo_dual_counter",
		"define i32 @trin_hw_sme_za_gather_mt_vlbal",
		"define i32 @trin_hw_sme_za_scatter_mt_vlbal",
		"define i32 @trin_fusion_pgo_conf",
		"define i32 @trin_fusion_pgo_conf_counter",
		"define i32 @trin_hw_sme_za_gather_mt_vlbw",
		"define i32 @trin_hw_sme_za_gather_mt_vlpath",
		"define i32 @trin_hw_sme_za_gather_mt_vllane",
		"define i32 @trin_hw_sme_za_gather_mt_vlspill",
		"define i32 @trin_hw_sme_za_gather_mt_vlocc",
		"define i32 @trin_hw_sme_za_gather_mt_vlqueue",
		"define i32 @trin_hw_sme_za_gather_mt_vlwater",
		"define i32 @trin_hw_sme_za_gather_mt_vldrain",
		"define i32 @trin_hw_sme_za_gather_mt_vlrefill",
"define i32 @trin_hw_sme_za_gather_mt_vlflush",
"define i32 @trin_hw_sme_za_gather_mt_vlseal",
"define i32 @trin_hw_sme_za_gather_mt_vlpurge",
"define i32 @trin_hw_sme_za_gather_mt_vlsync",
"define i32 @trin_hw_sme_za_gather_mt_vlclose",
		"define i32 @trin_hw_sme_za_scatter_mt_vlbw",
		"define i32 @trin_hw_sme_za_scatter_mt_vlpath",
		"define i32 @trin_hw_sme_za_scatter_mt_vllane",
		"define i32 @trin_hw_sme_za_scatter_mt_vlspill",
		"define i32 @trin_hw_sme_za_scatter_mt_vlocc",
		"define i32 @trin_hw_sme_za_scatter_mt_vlqueue",
		"define i32 @trin_hw_sme_za_scatter_mt_vlwater",
		"define i32 @trin_hw_sme_za_scatter_mt_vldrain",
		"define i32 @trin_hw_sme_za_scatter_mt_vlrefill",
"define i32 @trin_hw_sme_za_scatter_mt_vlflush",
"define i32 @trin_hw_sme_za_scatter_mt_vlseal",
"define i32 @trin_hw_sme_za_scatter_mt_vlpurge",
"define i32 @trin_hw_sme_za_scatter_mt_vlsync",
"define i32 @trin_hw_sme_za_scatter_mt_vlclose",
		"define i32 @trin_fusion_pgo_reg",
		"define i32 @trin_fusion_pgo_floor",
		"define i32 @trin_fusion_pgo_afloor",
		"define i32 @trin_fusion_pgo_burst",
		"define i32 @trin_fusion_pgo_cool",
		"define i32 @trin_fusion_pgo_credit",
		"define i32 @trin_fusion_pgo_debt",
		"define i32 @trin_fusion_pgo_interest",
		"define i32 @trin_fusion_pgo_amortize",
"define i32 @trin_fusion_pgo_settle",
"define i32 @trin_fusion_pgo_escrow",
"define i32 @trin_fusion_pgo_clear",
"define i32 @trin_fusion_pgo_final",
"define i32 @trin_fusion_pgo_close",
		"define i32 @trin_fusion_pgo_reg_counter",
		"define i32 @trin_fusion_pgo_floor_counter",
		"define i32 @trin_fusion_pgo_afloor_counter",
		"define i32 @trin_fusion_pgo_burst_counter",
		"define i32 @trin_fusion_pgo_cool_counter",
		"define i32 @trin_fusion_pgo_credit_counter",
		"define i32 @trin_fusion_pgo_debt_counter",
		"define i32 @trin_fusion_pgo_interest_counter",
		"define i32 @trin_fusion_pgo_amortize_counter",
"define i32 @trin_fusion_pgo_settle_counter",
"define i32 @trin_fusion_pgo_escrow_counter",
"define i32 @trin_fusion_pgo_clear_counter",
"define i32 @trin_fusion_pgo_final_counter",
"define i32 @trin_fusion_pgo_close_counter",
		"define i32 @trin_fusion_pgo_hyst",
		"define i32 @trin_fusion_pgo_hyst_counter",
		"define i32 @trin_fusion_pgo_ema",
		"define i32 @trin_fusion_pgo_ema_counter",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %q in runtime dispatch IR:\n%s", want, s)
		}
	}
	// Should not bake a single ret constant for vlen/isa when dispatch is used.
	if strings.Contains(s, "define i32 @trin_hw_vlen() {\nentry:\n  ret i32 ") {
		t.Fatal("trin_hw_vlen should not be a pure constant ret under runtime dispatch")
	}
	if runtime.GOARCH == "amd64" || runtime.GOARCH == "386" {
		if !strings.Contains(s, "cpuid") {
			t.Fatal("x86 should emit cpuid asm")
		}
	}
	// Selective emission
	onlyV := EmitRuntimeCPUIDLLVM(true, false, false)
	if !strings.Contains(onlyV, "define i32 @trin_hw_vlen") {
		t.Fatal("expected vlen")
	}
	if strings.Contains(onlyV, "define i32 @trin_hw_isa_code") {
		t.Fatal("did not request isa")
	}
	if EmitRuntimeCPUIDLLVM(false, false, false) != "" {
		t.Fatal("expected empty when nothing requested")
	}
}

func TestLowerUsesRuntimeDispatch(t *testing.T) {
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
		"runtime CPU feature detect (v0.30)",
		"runtime dynamic feature dispatch",
		"define i32 @trin_hw_vlen",
		"call void @trin_hw_detect_once",
		"@trin_hw_rt_vlen",
		"Generated by Trin v0.59",
	} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q in lowered LLVM:\n%s", want, ll)
		}
	}
	// Must not bake emit-host vlen as sole body of trin_hw_vlen.
	if strings.Contains(ll, "define i32 @trin_hw_vlen() {\nentry:\n  ret i32 "+itoa(DetectHostCPU().HWVLEN)+"\n}") {
		t.Fatal("trin_hw_vlen still baked to emit-host constant")
	}
}

func itoa(n int) string {
	return strings.TrimSpace(strings.ReplaceAll(fmt.Sprintf("%d", n), " ", ""))
}
