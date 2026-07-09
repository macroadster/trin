package interp

import (
	"os"
	"strings"
	"testing"
)

func TestCudaPathDemo(t *testing.T) {
	// Deterministic fallback on this host / CI.
	os.Setenv("TRIN_CUDA", "0")
	out := runOK(t, `
fn main() {
    let avail = cuda_available();
    let n = cuda_device_count();
    let be = cuda_backend();
    println(str_concat("cuda_available=", to_str(avail)));
    println(str_concat("cuda_device_count=", to_str(n)));
    println(str_concat("cuda_backend=", be));
    let W = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = [1.0, 0.0, -1.0];
    let y = cuda_linear_f32(W, x, 2, 3);
    println(str_concat("y=", to_str(y)));
    let A = [1.0, 0.0, 0.0, 1.0];
    let B = [1.0, 2.0, 3.0, 4.0];
    let C = cuda_gemm_f32(A, B, 2, 2, 2);
    println(str_concat("C=", to_str(C)));
    let Wt = [trit(1), trit(0), trit(-1), trit(0), trit(1), trit(1)];
    let xt = [1.0, 2.0, 3.0];
    let yt = cuda_linear_trit(Wt, xt, 2, 3, 2.0);
    println(str_concat("yt=", to_str(yt)));
    let ycpu = linear_trit(Wt, xt, 2, 3, 2.0);
    println(str_concat("parity=", to_str((yt[0] == ycpu[0]) * (yt[1] == ycpu[1]))));
}
`)
	for _, k := range []string{
		"cuda_available=0",
		"cuda_device_count=0",
		"cuda_backend=cpu-fallback",
		"y=[-2.0, -2.0]",
		"C=[1.0, 2.0, 3.0, 4.0]",
		"yt=[-4.0, 10.0]",
		"parity=1",
	} {
		if !strings.Contains(out, k) {
			t.Fatalf("missing %q in %q", k, out)
		}
	}
}
