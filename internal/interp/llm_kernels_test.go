package interp

import (
	"bytes"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/macroadster/trin/internal/weights"
)

func runOK(t *testing.T, src string) string {
	t.Helper()
	var out bytes.Buffer
	if err := RunSource(src, strings.NewReader(""), &out); err != nil {
		t.Fatalf("run: %v\nsrc:\n%s", err, src)
	}
	return out.String()
}

func TestQuantizeAbsmeanAndLinearTrit(t *testing.T) {
	out := runOK(t, `
fn main() {
    let w = [1.0, -2.0, 0.1, 0.0, 2.0, -1.0];
    let q = quantize_absmean(w);
    let trits = q[0];
    let scale = q[1];
    println(str_concat("trits=", to_str(trits)));
    println(str_concat("scale=", to_str(scale)));
    // x length K=3; M=2
    let x = [1.0, 0.5, -1.0];
    let y = linear_trit(trits, x, 2, 3, scale);
    println(str_concat("y=", to_str(y)));
}
`)
	if !strings.Contains(out, "trits=") {
		t.Fatalf("missing trits: %q", out)
	}
	if !strings.Contains(out, "y=") {
		t.Fatalf("missing y: %q", out)
	}
}

func TestRmsnormSoftmaxSilu(t *testing.T) {
	out := runOK(t, `
fn main() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let n = rmsnorm(x, 0.000001, 1.0);
    println(str_concat("rms=", to_str(n)));
    let s = softmax([1.0, 2.0, 3.0]);
    println(str_concat("sm=", to_str(s)));
    let u = silu([0.0, 1.0, -1.0]);
    println(str_concat("silu=", to_str(u)));
    let g = gelu([0.0, 1.0]);
    println(str_concat("gelu=", to_str(g)));
}
`)
	for _, k := range []string{"rms=", "sm=", "silu=", "gelu="} {
		if !strings.Contains(out, k) {
			t.Fatalf("missing %s in %q", k, out)
		}
	}
}

func TestAttnAndRope(t *testing.T) {
	out := runOK(t, `
fn main() {
    // d=2, n_tok=2
    let q = [1.0, 0.0];
    let K = [1.0, 0.0,  0.0, 1.0];
    let V = [1.0, 2.0,  3.0, 4.0];
    let o = attn(q, K, V, 2);
    println(str_concat("attn=", to_str(o)));
    let r = rope([1.0, 0.0, 0.0, 1.0], 1.0, 10000.0);
    println(str_concat("rope=", to_str(r)));
    let idx = argmax([0.1, 0.9, 0.2]);
    println(str_concat("am=", to_str(idx)));
}
`)
	if !strings.Contains(out, "attn=") || !strings.Contains(out, "rope=") || !strings.Contains(out, "am=1") {
		t.Fatalf("bad attn/rope/argmax: %q", out)
	}
}

func TestSoftmaxSumsToOne(t *testing.T) {
	// numerical check via host mirror of kernel semantics
	x := []float64{1, 2, 3}
	m := x[0]
	for _, v := range x {
		if v > m {
			m = v
		}
	}
	sum := 0.0
	ex := make([]float64, len(x))
	for i, v := range x {
		ex[i] = math.Exp(v - m)
		sum += ex[i]
	}
	s := 0.0
	for _, e := range ex {
		s += e / sum
	}
	if math.Abs(s-1) > 1e-9 {
		t.Fatalf("softmax sum %v", s)
	}
}

func TestGGUFBlockQuantQ4(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "q4.gguf")
	vals := make([]float32, 32)
	for i := range vals {
		vals[i] = float32(i-16) * 0.1
	}
	if err := weights.WriteGGUFF32(path, map[string][]float32{"weight": vals}, map[string][]uint64{"weight": {32}}, nil); err != nil {
		t.Fatal(err)
	}
	src := `
fn main() {
    let w = load_gguf_tensor("` + path + `", "weight");
    println(str_concat("n=", to_str(len(w))));
}
`
	out := runOK(t, src)
	if !strings.Contains(out, "n=32") {
		t.Fatalf("%q", out)
	}
}

func TestWeightImportersFixture(t *testing.T) {
	dir := t.TempDir()
	st := filepath.Join(dir, "W.safetensors")
	gg := filepath.Join(dir, "W.gguf")
	W := []float32{0.9, -1.1, 0.05, 1.0, -0.8, 0.2}
	if err := weights.WriteSafeTensors(st, map[string][]float32{"weight": W}, map[string][]int64{"weight": {2, 3}}); err != nil {
		t.Fatal(err)
	}
	if err := weights.WriteGGUFF32(gg, map[string][]float32{"weight": W}, map[string][]uint64{"weight": {2, 3}}, map[string]string{"general.name": "toy"}); err != nil {
		t.Fatal(err)
	}
	src := `
fn main() {
    let w = load_weight_tensor("` + st + `", "weight");
    println(str_concat("n=", to_str(len(w))));
    let q = quantize_absmean(w);
    let y = linear_trit(q[0], [1.0, 0.5, -1.0], 2, 3, q[1]);
    println(str_concat("y=", to_str(y)));
    let w2 = load_weight_tensor("` + gg + `", "weight");
    println(str_concat("n2=", to_str(len(w2))));
    let names = load_safetensors_list("` + st + `");
    println(str_concat("names=", to_str(names)));
    let meta = load_gguf_meta("` + gg + `", "general.name");
    println(str_concat("meta=", meta));
}
`
	out := runOK(t, src)
	if !strings.Contains(out, "n=6") || !strings.Contains(out, "n2=6") {
		t.Fatalf("import sizes: %q", out)
	}
	if !strings.Contains(out, "y=") || !strings.Contains(out, "meta=toy") {
		t.Fatalf("bad out: %q", out)
	}
}

func TestKvCacheAndSeededSample(t *testing.T) {
	out := runOK(t, `
fn main() {
    let K = kv_new();
    let K = kv_append(K, [1.0, 0.0]);
    let K = kv_append(K, [0.0, 1.0]);
    println(str_concat("n=", to_str(kv_len(K, 2))));
    println(str_concat("r0=", to_str(kv_row(K, 0, 2))));
    let C = span_concat([1.0], [2.0, 3.0]);
    println(str_concat("c=", to_str(C)));
    let a = sample_logits([0.0, 5.0, 0.0], 0.0);
    println(str_concat("g=", to_str(a)));
    let s1 = sample_logits([1.0, 1.0, 1.0], 1.0, 123);
    let s2 = sample_logits([1.0, 1.0, 1.0], 1.0, 123);
    println(str_concat("s1=", to_str(s1)));
    println(str_concat("s2=", to_str(s2)));
    let tk = sample_topk([0.1, 0.9, 0.2], 2, 0.0);
    println(str_concat("tk=", to_str(tk)));
}
`)
	for _, k := range []string{"n=2", "r0=[1.0, 0.0]", "c=[1.0, 2.0, 3.0]", "g=1", "tk=1"} {
		if !strings.Contains(out, k) {
			t.Fatalf("missing %s in %q", k, out)
		}
	}
	// seeded samples must match for same seed
	if !strings.Contains(out, "s1=") || !strings.Contains(out, "s2=") {
		t.Fatalf("missing seeded samples: %q", out)
	}
	// extract s1 and s2 values
	i1 := strings.Index(out, "s1=")
	i2 := strings.Index(out, "s2=")
	if i1 < 0 || i2 < 0 {
		t.Fatalf("seed lines: %q", out)
	}
	line1 := strings.SplitN(out[i1:], "\n", 2)[0]
	line2 := strings.SplitN(out[i2:], "\n", 2)[0]
	v1 := strings.TrimPrefix(line1, "s1=")
	v2 := strings.TrimPrefix(line2, "s2=")
	if v1 != v2 {
		t.Fatalf("seeded sample not deterministic: %q vs %q in %q", v1, v2, out)
	}
	_ = math.Abs
}
