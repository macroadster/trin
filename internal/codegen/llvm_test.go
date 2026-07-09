package codegen

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/ericchien/trin/internal/parser"
)

func TestEmitLLVMSoftMux(t *testing.T) {
	src := `
fn loss(w: f32, a: f32, x: f32, b: f32) {
    soft_mux(w, a * x, b)
}
fn main() {
    let y = soft_mux(0.5, 3.0, 1.0);
    println(to_str(y));
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
		"@trin_soft_mux", "define float @loss", "fmul", "ret float",
		"@trin_trit_dot_planes_word", "@llvm.ctpop.i32", "@trin_plane_popcount",
		"@trin_trit_dot_planes_x4", "@trin_simd_lane_width",
		"@trin_sum_f32", "@trin_dot_f32", "@trin_span_add_f32", "@trin_span_scale_f32",
		"@trin_vec_loop_probe", "llvm.loop.vectorize.enable", "!llvm.loop !0",
	} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q in:\n%s", want, ll)
		}
	}
	host := LLVMHostDriver("loss", ll)
	for _, want := range []string{"float loss(float", "0.5f", "printf"} {
		if !strings.Contains(host, want) {
			t.Fatalf("host missing %q:\n%s", want, host)
		}
	}
	if llvmFnArity(ll, "loss") != 4 {
		t.Fatalf("loss arity=%d", llvmFnArity(ll, "loss"))
	}
	if llvmFnArity(ll, "trin_soft_mux") != 3 {
		t.Fatalf("soft_mux arity")
	}
}

func TestEmitLLVMExample(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..", "..", "examples")
	src, err := os.ReadFile(filepath.Join(root, "soft_mux_grad.trin"))
	if err != nil {
		t.Fatal(err)
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		t.Fatal(err)
	}
	ll, err := EmitLLVM(prog)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(ll, "@trin_soft_mux") {
		t.Fatal(ll)
	}
}

func TestEmitLLVMSpanKernel(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..", "..", "examples")
	src, err := os.ReadFile(filepath.Join(root, "span_kernel.trin"))
	if err != nil {
		t.Fatal(err)
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		t.Fatal(err)
	}
	ll, err := EmitLLVM(prog)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"@trin_soft_mux", "@trin_sum4", "@trin_dot4", "define float @loss", "fadd", "fmul"} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q in:\n%s", want, ll)
		}
	}
}

func TestEmitLLVMMapSpan(t *testing.T) {
	src := `
fn loss(w: f32, s: f32, a0: f32, a1: f32, b0: f32, b1: f32, x0: f32, x1: f32) {
    soft_mux(w, sum(span_add([a0, a1], [b0, b1])), sum(map_mul(s, [x0, x1])))
}
fn main() {
    let y = soft_mux(0.25, 4.0, 12.0);
    println(to_str(y));
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
	for _, want := range []string{"@trin_soft_mux", "@trin_sum_span_add4", "@trin_sum_map_mul4", "define float @loss", "fadd", "fmul"} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q in:\n%s", want, ll)
		}
	}
}

func TestEmitLLVMMapLambda(t *testing.T) {
	src := `
fn loss(w: f32, a: f32, p: f32, q: f32, b: f32, r: f32) {
    soft_mux(w, sum(map((x) => x * a, [p, q])), sum(map((x) => x + b, [r])))
}
fn main() {
    let y = soft_mux(0.5, 14.0, 6.0);
    println(to_str(y));
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
	for _, want := range []string{"@trin_soft_mux", "define float @loss", "fmul", "fadd"} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q in:\n%s", want, ll)
		}
	}
}

func TestEmitLLVMSpanLoops(t *testing.T) {
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
		"define float @trin_sum_f32",
		"define float @trin_dot_f32",
		"define void @trin_span_add_f32",
		"define void @trin_span_scale_f32",
		"define float @trin_sum_span_add_f32",
		"define float @trin_sum_scale_f32",
		"define float @trin_vec_loop_probe",
		"!\"llvm.loop.vectorize.enable\"",
		"!\"llvm.loop.vectorize.width\"",
		"!\"llvm.loop.interleave.count\"",
		// v0.36/v0.37 residual peels use scalar.cond / peels with !llvm.loop !0
		"br label %scalar.cond, !llvm.loop !0",
		"peel16.cond",
		"define float @trin_store_peel_probe",
	} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q", want)
		}
	}
	if llvmFnArity(ll, "trin_vec_loop_probe") != 0 {
		t.Fatalf("probe arity")
	}
	// Nullary host when module has no @main (float-host path).
	// Full-program modules that already define @main correctly return "" host.
	llNoMain := strings.ReplaceAll(ll, "define i32 @main()", "define i32 @trin_unused_main()")
	// Strip after first replacement only if present; also ensure hasLLVMMain is false.
	if hasLLVMMain(llNoMain) {
		// comment-out residual
		llNoMain = strings.ReplaceAll(llNoMain, "define i32 @main(", "define i32 @trin_unused_main(")
	}
	host := LLVMHostDriver("trin_vec_loop_probe", llNoMain)
	if hasLLVMMain(ll) {
		// full-program path: empty host is correct
		if h := LLVMHostDriver("trin_vec_loop_probe", ll); h != "" {
			t.Fatalf("expected empty host with @main, got:\n%s", h)
		}
	}
	for _, want := range []string{"trin_vec_loop_probe", "printf"} {
		if !strings.Contains(host, want) {
			t.Fatalf("host missing %q:\n%s", want, host)
		}
	}
}

func TestEmitLLVMNestedMap(t *testing.T) {
	src := `
fn loss(s: f32, a: f32, u: f32, v: f32) {
    sum(map((x) => x * s, map((y) => y + a, [u, v])))
}
fn main() {
    let y = 36.0;
    println(to_str(y));
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
	for _, want := range []string{"define float @loss", "fadd", "fmul"} {
		if !strings.Contains(ll, want) {
			t.Fatalf("missing %q in:\n%s", want, ll)
		}
	}
}

