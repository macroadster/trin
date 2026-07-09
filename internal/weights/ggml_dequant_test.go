package weights

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func fp16(f float32) uint16 {
	// convert via float32 bits → float16 (simple IEEE round-to-nearest enough for tests)
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xff) - 127 + 15
	mant := (bits >> 13) & 0x3ff
	if exp <= 0 {
		return sign
	}
	if exp >= 31 {
		return sign | 0x7c00
	}
	return sign | uint16(exp<<10) | uint16(mant)
}

func packQ4_0(vals []float32) []byte {
	// quantize_row_q4_0_ref simplified for full blocks
	if len(vals)%QK4_0 != 0 {
		panic("pad")
	}
	nb := len(vals) / QK4_0
	out := make([]byte, 0, nb*bsQ4_0)
	for i := 0; i < nb; i++ {
		block := vals[i*QK4_0 : (i+1)*QK4_0]
		amax, maxv := float32(0), float32(0)
		for _, v := range block {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
				maxv = v
			}
		}
		d := maxv / -8
		id := float32(0)
		if d != 0 {
			id = 1 / d
		}
		var buf [bsQ4_0]byte
		binary.LittleEndian.PutUint16(buf[0:2], fp16(d))
		for j := 0; j < QK4_0/2; j++ {
			x0 := block[j] * id
			x1 := block[j+QK4_0/2] * id
			xi0 := int8(x0 + 8.5)
			if xi0 > 15 {
				xi0 = 15
			}
			if xi0 < 0 {
				xi0 = 0
			}
			xi1 := int8(x1 + 8.5)
			if xi1 > 15 {
				xi1 = 15
			}
			if xi1 < 0 {
				xi1 = 0
			}
			buf[2+j] = byte(xi0) | byte(xi1)<<4
		}
		out = append(out, buf[:]...)
	}
	return out
}

func packQ8_0(vals []float32) []byte {
	nb := len(vals) / QK8_0
	out := make([]byte, 0, nb*bsQ8_0)
	for i := 0; i < nb; i++ {
		block := vals[i*QK8_0 : (i+1)*QK8_0]
		amax := float32(0)
		for _, v := range block {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}
		d := amax / 127
		id := float32(0)
		if d != 0 {
			id = 1 / d
		}
		var buf [bsQ8_0]byte
		binary.LittleEndian.PutUint16(buf[0:2], fp16(d))
		for j := 0; j < QK8_0; j++ {
			q := int(math.Round(float64(block[j] * id)))
			if q < -128 {
				q = -128
			}
			if q > 127 {
				q = 127
			}
			buf[2+j] = byte(int8(q))
		}
		out = append(out, buf[:]...)
	}
	return out
}

func maxAbsDiff(a, b []float32) float64 {
	m := 0.0
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if d > m {
			m = d
		}
	}
	return m
}

func TestDequantQ4_0RoundTrip(t *testing.T) {
	// values that land cleanly on Q4_0 grid (multiples of d, with d from max/-8)
	src := make([]float32, 32)
	for i := range src {
		src[i] = float32(i-16) * 0.1
	}
	raw := packQ4_0(src)
	got, err := DecodeGGMLBlock(GGML_Q4_0, raw, 32)
	if err != nil {
		t.Fatal(err)
	}
	// re-dequant of packed should match d*(q-8) reconstruct from raw
	var recon []float32
	tmp := make([]float32, 32)
	dequantBlockQ4_0(raw, tmp)
	recon = tmp
	if maxAbsDiff(got, recon) > 1e-6 {
		t.Fatalf("decode mismatch")
	}
	// sanity: dequant roughly near original
	if maxAbsDiff(got, src) > 0.2 {
		t.Fatalf("too far from src maxdiff=%v\ngot=%v\nsrc=%v", maxAbsDiff(got, src), got, src)
	}
}

func TestDequantQ8_0RoundTrip(t *testing.T) {
	src := make([]float32, 64)
	for i := range src {
		src[i] = float32(i-32) * 0.05
	}
	raw := packQ8_0(src)
	got, err := DecodeGGMLBlock(GGML_Q8_0, raw, 64)
	if err != nil {
		t.Fatal(err)
	}
	if maxAbsDiff(got, src) > 0.05 {
		t.Fatalf("Q8_0 maxdiff=%v", maxAbsDiff(got, src))
	}
}

func TestDequantQ4_1AndQ5(t *testing.T) {
	// hand-built Q4_1 block: d=1, m=-2, qs nibble j for both halves
	blk := make([]byte, bsQ4_1)
	binary.LittleEndian.PutUint16(blk[0:2], fp16(1))
	binary.LittleEndian.PutUint16(blk[2:4], fp16(-2))
	for j := 0; j < 16; j++ {
		blk[4+j] = byte(j) | byte(j)<<4
	}
	out := make([]float32, 32)
	dequantBlockQ4_1(blk, out)
	if math.Abs(float64(out[0]-(-2))) > 1e-3 { // 0*1 + (-2)
		t.Fatalf("q4_1[0]=%v", out[0])
	}
	if math.Abs(float64(out[5]-(5-2))) > 1e-3 {
		t.Fatalf("q4_1[5]=%v", out[5])
	}

	// Q5_0: d=1, all zero quants → -16
	blk5 := make([]byte, bsQ5_0)
	binary.LittleEndian.PutUint16(blk5[0:2], fp16(1))
	// qh=0, qs=0
	out5 := make([]float32, 32)
	dequantBlockQ5_0(blk5, out5)
	for i, v := range out5 {
		if math.Abs(float64(v-(-16))) > 1e-3 {
			t.Fatalf("q5_0[%d]=%v", i, v)
		}
	}
}

func TestDequantQ4_KHand(t *testing.T) {
	// one super-block: d=1, dmin=0, all scales sc=1 m=0 for j=0..7 via simple packing for j<4
	blk := make([]byte, bsQ4_K)
	binary.LittleEndian.PutUint16(blk[0:2], fp16(1))
	binary.LittleEndian.PutUint16(blk[2:4], fp16(0))
	// scales: for j=0..3, q[j]=1 (sc), q[j+4]=0 (m)
	for j := 0; j < 4; j++ {
		blk[4+j] = 1
		blk[4+4+j] = 0
	}
	// remaining scale bytes 0
	// qs: all zeros → dequant 0
	out := make([]float32, 256)
	dequantBlockQ4_K(blk, out)
	for i, v := range out {
		if v != 0 {
			t.Fatalf("expected 0 got %v at %d", v, i)
		}
	}
	// set first 32 low nibbles to 3 → 3*d1 - 0 with d1=1*1=1
	for l := 0; l < 32; l++ {
		blk[4+12+l] = 3
	}
	dequantBlockQ4_K(blk, out)
	for l := 0; l < 32; l++ {
		if math.Abs(float64(out[l]-3)) > 1e-3 {
			t.Fatalf("out[%d]=%v", l, out[l])
		}
	}
}

func TestDequantQ6_KAndQ8_K(t *testing.T) {
	// Q8_K: d=0.5, qs=2 → 1.0
	blk := make([]byte, bsQ8_K)
	binary.LittleEndian.PutUint32(blk[0:4], math.Float32bits(0.5))
	for j := 0; j < 256; j++ {
		blk[4+j] = 2 // int8 2
	}
	out := make([]float32, 256)
	dequantBlockQ8_K(blk, out)
	if math.Abs(float64(out[10]-1.0)) > 1e-5 {
		t.Fatalf("q8k %v", out[10])
	}

	// Q6_K zero block with d=1 and scale=1 → all (0-32)*1*1 = -32
	b6 := make([]byte, bsQ6_K)
	// scales at offset 128+64 = 192, 16 bytes of 1
	for i := 0; i < 16; i++ {
		b6[192+i] = 1
	}
	binary.LittleEndian.PutUint16(b6[208:210], fp16(1))
	o6 := make([]float32, 256)
	dequantBlockQ6_K(b6, o6)
	if math.Abs(float64(o6[0]-(-32))) > 1e-2 {
		t.Fatalf("q6k[0]=%v", o6[0])
	}
}

func TestGGUFWithQ4_0Tensor(t *testing.T) {
	// Build a synthetic Q4_0 GGUF: cannot use WriteGGUFF32; write minimal manually
	dir := t.TempDir()
	path := filepath.Join(dir, "q4.gguf")
	vals := make([]float32, 32)
	for i := range vals {
		vals[i] = float32(i-16) * 0.25
	}
	raw := packQ4_0(vals)

	// minimal GGUF v3 with one Q4_0 tensor
	var buf []byte
	putU32 := func(v uint32) {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], v)
		buf = append(buf, b[:]...)
	}
	putU64 := func(v uint64) {
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], v)
		buf = append(buf, b[:]...)
	}
	putStr := func(s string) {
		putU64(uint64(len(s)))
		buf = append(buf, []byte(s)...)
	}
	putU32(0x46554747)
	putU32(3)
	putU64(1) // n_tensors
	putU64(1) // n_kv
	putStr("general.alignment")
	putU32(4)
	putU32(32)
	putStr("weight")
	putU32(1) // n_dims
	putU64(32)
	putU32(GGML_Q4_0)
	putU64(0)
	// pad header
	for len(buf)%32 != 0 {
		buf = append(buf, 0)
	}
	buf = append(buf, raw...)
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := OpenGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	got, shape, err := f.LoadF32("weight")
	if err != nil {
		t.Fatal(err)
	}
	if len(shape) != 1 || shape[0] != 32 || len(got) != 32 {
		t.Fatalf("shape %v len %d", shape, len(got))
	}
	if maxAbsDiff(got, vals) > 0.3 {
		t.Fatalf("diff %v", maxAbsDiff(got, vals))
	}
}

func TestUnsupportedIQErrors(t *testing.T) {
	// Remaining after v0.99 MXFP4/NVFP4: other exotic types only.
	_, err := DecodeGGMLBlock(100, make([]byte, 100), 32)
	if err == nil {
		t.Fatal("expected error for unsupported type 100")
	}
}
