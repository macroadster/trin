package interp

import (
	"os"
	"strings"
	"testing"
)

func TestCudaPTXLayer(t *testing.T) {
	os.Setenv("TRIN_CUDA", "0")
	// Prefer venv ptxas if present for cubin path.
	os.Setenv("PATH", "/home/nanobot/.venv/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:"+os.Getenv("PATH"))

	out := runOK(t, `
fn main() {
    println(str_concat("arch=", cuda_ptx_arch()));
    println(str_concat("ptxas=", to_str(cuda_ptxas_available())));
    let s = cuda_ptx_source("linear_f32");
    println(str_concat("src_len=", to_str(str_len(s))));
    let pb = cuda_ptx_bytes("linear_f32");
    println(str_concat("ptx_bytes=", to_str(pb)));
    let c = cuda_compile_ptx("linear_f32");
    println(str_concat("cubin=", to_str(c)));
    let plan = cuda_plan_launch("gemm_f32");
    println(str_concat("would=", to_str(plan[0])));
    println(str_concat("be=", plan[1]));
    println(str_concat("entry=", plan[3]));
    let id = cuda_alloc_f32(3);
    cuda_copy_hto_d(id, [1.5, 2.5, 3.5]);
    let v = cuda_copy_dto_h(id, 3);
    println(str_concat("v=", to_str(v)));
    cuda_free(id);
}
`)
	for _, k := range []string{
		"arch=sm_75",
		"would=0",
		"be=cpu-fallback",
		"entry=trin_gemm_f32",
		"v=[1.5, 2.5, 3.5]",
	} {
		if !strings.Contains(out, k) {
			t.Fatalf("missing %q in %q", k, out)
		}
	}
	if !strings.Contains(out, "ptx_bytes=") {
		t.Fatalf("no ptx_bytes: %q", out)
	}
	// src should be substantial
	if !strings.Contains(out, "src_len=") {
		t.Fatalf("no src_len")
	}
}
