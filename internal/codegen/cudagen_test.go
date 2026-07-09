package codegen

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ericchien/trin/internal/parser"
)

func cudaRoot(t *testing.T) string {
	t.Helper()
	return goRoot(t)
}

func TestEmitCUDAHello(t *testing.T) {
	src, err := os.ReadFile(filepath.Join(cudaRoot(t), "examples", "hello.trin"))
	if err != nil {
		t.Fatal(err)
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		t.Fatal(err)
	}
	cu, err := EmitCUDA(prog)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(cu, "trin_fn_main") {
		t.Fatal("missing main")
	}
	if !strings.Contains(cu, "cuda_runtime.h") {
		t.Fatal("missing CUDA runtime include")
	}
	if !strings.Contains(cu, "__global__") {
		t.Fatal("missing __global__ CUDA kernel")
	}
	if !strings.Contains(cu, "cudaMalloc") {
		t.Fatal("missing cudaMalloc")
	}
}

func TestEmitCUDASoftMux(t *testing.T) {
	src, err := os.ReadFile(filepath.Join(cudaRoot(t), "examples", "soft_mux.trin"))
	if err != nil {
		t.Fatal(err)
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		t.Fatal(err)
	}
	cu, err := EmitCUDA(prog)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(cu, "bi_soft_mux") {
		t.Fatal("missing soft_mux builtin")
	}
}

func TestEmitCUDALLMKernels(t *testing.T) {
	src, err := os.ReadFile(filepath.Join(cudaRoot(t), "examples", "llm_kernels.trin"))
	if err != nil {
		t.Fatal(err)
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		t.Fatal(err)
	}
	cu, err := EmitCUDA(prog)
	if err != nil {
		t.Fatal(err)
	}
	// Should contain CUDA kernels for the LLM ops
	for _, marker := range []string{
		"trin_silu_kernel",
		"trin_gelu_kernel",
		"trin_softmax_max_kernel",
		"trin_rmsnorm_sq_kernel",
		"trin_rope_kernel",
		"trin_linear_trit_kernel",
		"trin_attn_scores_kernel",
		"trin_sum_kernel",
		"trin_dot_kernel",
		"trin_span_add_kernel",
		"trin_span_scale_kernel",
		"CUDA_CHECK",
		"<<<",
	} {
		if !strings.Contains(cu, marker) {
			t.Fatalf("missing CUDA marker %q", marker)
		}
	}
}

func TestEmitCUDASpanOps(t *testing.T) {
	src, err := os.ReadFile(filepath.Join(cudaRoot(t), "examples", "span_ops.trin"))
	if err != nil {
		t.Fatal(err)
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		t.Fatal(err)
	}
	cu, err := EmitCUDA(prog)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(cu, "bi_sum") {
		t.Fatal("missing sum builtin")
	}
	if !strings.Contains(cu, "bi_dot") {
		t.Fatal("missing dot builtin")
	}
}

func TestEmitCUDABitplane(t *testing.T) {
	src, err := os.ReadFile(filepath.Join(cudaRoot(t), "examples", "bitplane_gemm.trin"))
	if err != nil {
		t.Fatal(err)
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		t.Fatal(err)
	}
	cu, err := EmitCUDA(prog)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(cu, "bi_trit_gemm") {
		t.Fatal("missing trit_gemm builtin")
	}
}

func TestEmitCUDAExamples(t *testing.T) {
	files := []string{
		"hello.trin", "soft_mux.trin", "span_ops.trin",
		"trit_gemm.trin", "bitplane_gemm.trin",
	}
	for _, name := range files {
		t.Run(name, func(t *testing.T) {
			src, err := os.ReadFile(filepath.Join(cudaRoot(t), "examples", name))
			if err != nil {
				t.Fatal(err)
			}
			prog, err := parser.ParseSource(string(src))
			if err != nil {
				t.Fatal(err)
			}
			cu, err := EmitCUDA(prog)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(cu, "trin_fn_main") {
				t.Fatal("missing main")
			}
			if !strings.Contains(cu, "__global__") {
				t.Fatal("missing __global__ kernel")
			}
		})
	}
}

func TestEmitCUDACompileWithNVCC(t *testing.T) {
	nvcc := os.Getenv("TRIN_NVCC")
	if nvcc == "" {
		nvcc = "nvcc"
	}
	if _, err := exec.LookPath(nvcc); err != nil {
		t.Skip("nvcc not found — skipping CUDA compilation test")
	}
	dir := t.TempDir()
	files := []string{"hello.trin", "soft_mux.trin", "span_ops.trin"}
	for _, name := range files {
		t.Run(name, func(t *testing.T) {
			src, err := os.ReadFile(filepath.Join(cudaRoot(t), "examples", name))
			if err != nil {
				t.Fatal(err)
			}
			prog, err := parser.ParseSource(string(src))
			if err != nil {
				t.Fatal(err)
			}
			cu, err := EmitCUDA(prog)
			if err != nil {
				t.Fatal(err)
			}
			cupath := filepath.Join(dir, name+".cu")
			bin := filepath.Join(dir, strings.TrimSuffix(name, ".trin"))
			if err := os.WriteFile(cupath, []byte(cu), 0o644); err != nil {
				t.Fatal(err)
			}
			cmd := exec.Command(nvcc, "-O2", "-o", bin, cupath)
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("nvcc: %v\n%s", err, out)
			}
		})
	}
}
