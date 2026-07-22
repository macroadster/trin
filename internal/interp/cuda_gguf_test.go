package interp

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/macroadster/trin/internal/cuda"
	"github.com/macroadster/trin/internal/weights"
)

// makeQ4_0Block encodes 32 float32 values as one Q4_0 block (18 bytes).
// This is a minimal encoder for test fixtures — not a production quantiser.
func makeQ4_0Block(vals []float32) []byte {
	if len(vals) != 32 {
		panic("Q4_0 block must be 32 elements")
	}
	// Find max absolute value to compute d
	maxAbs := float32(0)
	for _, v := range vals {
		a := v
		if a < 0 {
			a = -a
		}
		if a > maxAbs {
			maxAbs = a
		}
	}
	d := maxAbs / 7
	if d == 0 {
		d = 1e-10 // avoid division by zero
	}
	// Quantize: q = clamp(round(v/d) + 8, 0, 15)
	var qs [16]byte
	for j := 0; j < 16; j++ {
		q0 := int(math.Round(float64(vals[j]/d))) + 8
		if q0 < 0 {
			q0 = 0
		}
		if q0 > 15 {
			q0 = 15
		}
		q1 := int(math.Round(float64(vals[j+16]/d))) + 8
		if q1 < 0 {
			q1 = 0
		}
		if q1 > 15 {
			q1 = 15
		}
		qs[j] = byte(q0) | (byte(q1) << 4)
	}
	// Encode d as float16
	var blk [18]byte
	dBits := float32ToFloat16(d)
	binary.LittleEndian.PutUint16(blk[0:2], dBits)
	copy(blk[2:], qs[:])
	return blk[:]
}

// makeQ8_0Block encodes 32 float32 values as one Q8_0 block (34 bytes).
func makeQ8_0Block(vals []float32) []byte {
	if len(vals) != 32 {
		panic("Q8_0 block must be 32 elements")
	}
	maxAbs := float32(0)
	for _, v := range vals {
		a := v
		if a < 0 {
			a = -a
		}
		if a > maxAbs {
			maxAbs = a
		}
	}
	d := maxAbs / 127
	if d == 0 {
		d = 1e-10
	}
	var blk [34]byte
	dBits := float32ToFloat16(d)
	binary.LittleEndian.PutUint16(blk[0:2], dBits)
	for j := 0; j < 32; j++ {
		q := int(math.Round(float64(vals[j] / d)))
		if q > 127 {
			q = 127
		}
		if q < -128 {
			q = -128
		}
		blk[2+j] = byte(int8(q))
	}
	return blk[:]
}

// float32ToFloat16 converts f32 to IEEE-754 f16 bit pattern (truncation, not rounding).
func float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xFF) - 127 + 15
	man := uint16((bits >> 13) & 0x3FF)
	if exp <= 0 {
		return sign // flush to zero
	}
	if exp >= 31 {
		return sign | 0x7C00 // inf
	}
	return sign | uint16(exp)<<10 | man
}

func TestCUDADequantLinearQ4_0(t *testing.T) {
	// 2x32 matrix: row 0 = all 1.0, row 1 = all -1.0
	// x = all 1.0 → y[0] ≈ 32, y[1] ≈ -32 (within quant error)
	M, K := 2, 32
	row0 := make([]float32, 32)
	row1 := make([]float32, 32)
	for i := range row0 {
		row0[i] = 1.0
		row1[i] = -1.0
	}
	blk0 := makeQ4_0Block(row0)
	blk1 := makeQ4_0Block(row1)
	W := append(blk0, blk1...)

	x := make([]float32, 32)
	for i := range x {
		x[i] = 1.0
	}
	y, err := cuda.DequantLinearQ4_0(W, x, M, K)
	if err != nil {
		t.Fatal(err)
	}
	if len(y) != 2 {
		t.Fatalf("y len=%d", len(y))
	}
	// Q4_0 encodes 1.0 with d ≈ 1/7 ≈ 0.143; q = round(1.0/d)+8 = round(7)+8 = 15
	// dequant: (15-8)*d ≈ 1.0; sum of 32 ≈ 32
	if math.Abs(float64(y[0])-32) > 4 {
		t.Errorf("y[0]=%v want ~32", y[0])
	}
	if math.Abs(float64(y[1])+32) > 4 {
		t.Errorf("y[1]=%v want ~-32", y[1])
	}
}

func TestCUDADequantLinearQ8_0(t *testing.T) {
	M, K := 2, 32
	row0 := make([]float32, 32)
	row1 := make([]float32, 32)
	for i := range row0 {
		row0[i] = 0.5
		row1[i] = -0.5
	}
	blk0 := makeQ8_0Block(row0)
	blk1 := makeQ8_0Block(row1)
	W := append(blk0, blk1...)

	x := make([]float32, 32)
	for i := range x {
		x[i] = 1.0
	}
	y, err := cuda.DequantLinearQ8_0(W, x, M, K)
	if err != nil {
		t.Fatal(err)
	}
	if len(y) != 2 {
		t.Fatalf("y len=%d", len(y))
	}
	// Q8_0 with 0.5 and d=0.5/127 → high fidelity
	if math.Abs(float64(y[0])-16) > 1 {
		t.Errorf("y[0]=%v want ~16", y[0])
	}
	if math.Abs(float64(y[1])+16) > 1 {
		t.Errorf("y[1]=%v want ~-16", y[1])
	}
}

func TestCUDAGGUFSessionLinear(t *testing.T) {
	os.Setenv("TRIN_CUDA", "0")
	cuda.ResetProbeForTest()

	// Write a small Q4_0 GGUF fixture: 2×32 tensor "weight"
	dir := t.TempDir()
	path := filepath.Join(dir, "q4.gguf")

	row0 := make([]float32, 32)
	row1 := make([]float32, 32)
	for i := range row0 {
		row0[i] = float32(i-16) * 0.1
		row1[i] = float32(16-i) * 0.1
	}
	blk0 := makeQ4_0Block(row0)
	blk1 := makeQ4_0Block(row1)
	data := append(blk0, blk1...)

	if err := weights.WriteGGUFTyped(path, "weight", weights.GGML_Q4_0, []uint64{32, 2}, data, nil); err != nil {
		t.Fatal(err)
	}

	src := `
fn main() {
    let sid = open_weight_session("` + path + `");
    let x = [` + genOnes(32) + `];
    let qt = cuda_session_tensor_qtype(sid, "weight");
    println(str_concat("qtype=", to_str(qt[0])));
    println(str_concat("nelem=", to_str(qt[1])));
    let y = cuda_session_linear(sid, "weight", x, 2, 32);
    println(str_concat("y_len=", to_str(len(y))));
    println(str_concat("y0_sign=", to_str(y[0] < 0.0)));
    close_weight_session(sid);
}
`
	out := runOK(t, src)
	for _, k := range []string{
		"qtype=2",   // GGML_Q4_0 = 2
		"nelem=64",  // 2×32
		"y_len=2",
	} {
		if !strings.Contains(out, k) {
			t.Fatalf("missing %q in %q", k, out)
		}
	}
}

func TestCUDAGGUFDequantPTX(t *testing.T) {
	os.Setenv("TRIN_CUDA", "0")
	cuda.ResetProbeForTest()

	out := runOK(t, `
fn main() {
    let src_q4 = cuda_ptx_source("dequant_linear_q4_0");
    println(str_concat("q4_len=", to_str(str_len(src_q4))));
    let src_q8 = cuda_ptx_source("dequant_linear_q8_0");
    println(str_concat("q8_len=", to_str(str_len(src_q8))));
    let plan = cuda_plan_launch("dequant_linear_q4_0");
    println(str_concat("backend=", plan[1]));
    println(str_concat("entry=", plan[3]));
}
`)
	for _, k := range []string{
		"q4_len=",
		"q8_len=",
		"backend=cpu-fallback",
		"entry=trin_dequant_linear_q4_0",
	} {
		if !strings.Contains(out, k) {
			t.Fatalf("missing %q in %q", k, out)
		}
	}
}

func TestDequantPTXContainsEntry(t *testing.T) {
	for _, mod := range cuda.AllDequantPTXModules("sm_75") {
		if !strings.Contains(mod.Source, ".entry "+mod.Name) {
			t.Fatalf("missing entry in %s", mod.Name)
		}
		if !strings.Contains(mod.Source, ".target sm_75") {
			t.Fatalf("bad target in %s", mod.Name)
		}
		if len(mod.Source) < 200 {
			t.Fatalf("PTX too short for %s (%d bytes)", mod.Name, len(mod.Source))
		}
	}
}

func TestCompileDequantPTXIfAvailable(t *testing.T) {
	if !cuda.PTXASAvailable() {
		t.Skip("ptxas not on host")
	}
	for _, mod := range cuda.AllDequantPTXModules("") {
		cub, err := cuda.CompilePTX(mod)
		if err != nil {
			t.Fatalf("%s: %v\n%s", mod.Name, err, cub.Stderr)
		}
		if len(cub.Bytes) < 64 {
			t.Fatalf("%s cubin too small: %d", mod.Name, len(cub.Bytes))
		}
		t.Logf("%s cubin=%dB arch=%s", mod.Name, len(cub.Bytes), cub.Arch)
	}
}

// genOnes generates a comma-separated string of n "1.0" values for Trin source embedding.
func genOnes(n int) string {
	var b strings.Builder
	for i := 0; i < n; i++ {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString("1.0")
	}
	return b.String()
}
