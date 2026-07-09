package interp

import (
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func root(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("no caller")
	}
	return filepath.Join(filepath.Dir(file), "..", "..")
}

func TestExamples(t *testing.T) {
	cases := []struct {
		file  string
		stdin string
		want  []string
	}{
		{"examples/hello.trin", "Ada\n", []string{"Hello world, Ada"}},
		{"examples/hello.trin", "\n", []string{"Hello world, stranger"}},
		{"examples/soft_mux.trin", "", []string{"soft number ~ 8", "high confidence"}},
		{"examples/span_ops.trin", "", []string{"doubled: [2.0, 4.0, 6.0, 8.0]", "dot: 10.0", "non-empty"}},
		{"examples/trit_gemm.trin", "", []string{"C = [1, -2, 1, 1]", "C00 positive"}},
		{"examples/bitplane_gemm.trin", "", []string{"scalar   C = [1, -2, 1, 1]", "bitplane C = [1, -2, 1, 1]", "roundtrip A = [1, 0, -1, 0, 1, 1]"}},
		{"examples/bitplane_words.trin", "", []string{"n = 36", "nwords = 2", "round sum A = 4", "round sum back = 4", "dot planes = -9", "dot scalar = -9", "gemm scalar  = [-9]", "gemm bitplane= [-9]", "pos pop = 15", "xor self pop = 0"}},
		{"examples/bitplane_simd.trin", "", []string{"simd_lane_width = 128", "n = 140", "nwords = 5", "dot scalar-plane = -4", "dot simd-plane = -4", "gemm scalar  = [-4]", "gemm bitplane= [-4]", "gemm simd    = [-4]"}},
		{"examples/llvm_full_prog.trin", "", []string{"product = 12", "soft_mux = 4", "sum = 10", "dot = 11", "span_add sum = 10", "simd_lane_width = 128"}},
		{"examples/llvm_vec_loops.trin", "", []string{"sum8 = 36", "dot8 = 204", "span_add_sum8 = 72", "scale_sum8 = 72", "llvm_probe_target = 36"}},
		{"examples/mlir_vec.trin", "", []string{"sum4 = 10", "dot4 = 30", "span_add_sum4 = 20", "scale_sum4 = 20", "soft_mux = 4", "mlir_vec_probe_target = 10"}},
		{"examples/soft_mux_grad.trin", "", []string{"y = 6", "dy/dw = 8", "dy/dpure = 0.5", "dz/dw = 10", "dz/da = 2", "dz/dx = 1.5", "dz/db = 0.5"}},
		{"examples/grad_tape.trin", "", []string{"loss = 11.6", "d/dw = 4", "d/da = 8.2", "d/dx = 1.6", "y = 2", "dy/du = 4", "dy/dw = 4", "dy/da = 0.125"}},
		{"examples/grad_span.trin", "", []string{"sum = 10", "dot = 32", "y = 8.5", "dy/dw = -5", "dy/da = 0.5", "dy/dx = 1.5", "loss = 24", "dloss/dx = 6", "dloss/da = 4"}},
		{"examples/zip_with.trin", "", []string{"ys = [10.5, 21.0, 31.5]", "sum = 63", "mixed = "}},
		{"examples/grad_zip_with.trin", "", []string{"v = 22", "du = 5", "ds = 3", "q = 8.5", "da = 1", "dw = 7"}},
		{"examples/reduce.trin", "", []string{"sum = 10", "prod = 24", "wk = 20", "nested = 32"}},
		{"examples/grad_reduce.trin", "", []string{"v = 24", "du = 4", "ds = 6", "q = 29", "da = 12", "m = 21", "dms = 7"}},
		{"examples/zip3_scan.trin", "", []string{"ys = [12.5, 25.0, 37.5]", "sum = 75", "pref = [1.0, 3.0, 6.0, 10.0]", "sk = [2.0, 6.0, 12.0]", "nested = 7"}},
		{"examples/grad_zip3_scan.trin", "", []string{"v = 39", "du = 7", "ds = 3", "q = 40", "da = 12", "db = 8", "dc = 4", "ds = 10", "m = 7.5", "dmp = 1", "dmw = 3"}},
		{"examples/filter_nary.trin", "", []string{"ys = [12.5, 27.5]", "sum = 40", "nz = [-1.0, 2.0, 3.0, -4.0]", "skipped = [1.0, 3.0]", "nested = 4"}},
		{"examples/loop_sugar.trin", "", []string{"r0=[0, 1, 2, 3, 4]", "when_t=42", "when_f=0", "over=[2.0, 4.0, 6.0]", "for=[0, 1, 4, 9]", "sum_for=10.0", "loop_sugar_ok=1"}},
		{"examples/grad_filter_nary.trin", "", []string{"v = 40", "da = 2", "db = 3", "dg = 1", "dh = 2", "q = 4", "dp = 1", "dq = 0", "dr = 1", "dk = 0", "m = 15", "dmr = 4", "dms = 3"}},
	}
	for _, tc := range cases {
		t.Run(tc.file+"/"+strings.ReplaceAll(tc.stdin, "\n", "_"), func(t *testing.T) {
			src, err := os.ReadFile(filepath.Join(root(t), tc.file))
			if err != nil {
				t.Fatal(err)
			}
			var out bytes.Buffer
			if err := RunSource(string(src), strings.NewReader(tc.stdin), &out); err != nil {
				t.Fatal(err)
			}
			got := out.String()
			for _, w := range tc.want {
				if !strings.Contains(got, w) {
					t.Fatalf("missing %q in:\n%s", w, got)
				}
			}
		})
	}
}
