package cuda

import (
	"math"
	"testing"
)

// ---- float16ToFloat32 ----

func TestFloat16ToFloat32_Normals(t *testing.T) {
	cases := []struct {
		bits uint16
		want float32
	}{
		{0x3C00, 1.0},          // 1.0
		{0xBC00, -1.0},         // -1.0
		{0x4000, 2.0},          // 2.0
		{0x3800, 0.5},          // 0.5
		{0x4248, 3.140625},     // ~π (3.140625 in f16)
		{0x4170, 2.71875},      // ~e (2.71875 in f16)
		{0x7BFF, 65504.0},      // max normal f16
	}
	for _, c := range cases {
		got := float16ToFloat32(c.bits)
		if got != c.want {
			t.Errorf("float16ToFloat32(0x%04X) = %v, want %v", c.bits, got, c.want)
		}
	}
}

func TestFloat16ToFloat32_Zeros(t *testing.T) {
	pz := float16ToFloat32(0x0000) // +0
	nz := float16ToFloat32(0x8000) // -0
	if pz != 0 || math.Signbit(float64(pz)) {
		t.Errorf("+0: got %v", pz)
	}
	if nz != 0 || !math.Signbit(float64(nz)) {
		t.Errorf("-0: got %v", nz)
	}
}

func TestFloat16ToFloat32_Subnormals(t *testing.T) {
	// Smallest subnormal: 0x0001 = 2^(-14) * (1/1024) = 2^(-24) ≈ 5.96e-8
	got := float16ToFloat32(0x0001)
	want := float32(math.Ldexp(1, -24))
	if math.Abs(float64(got-want)) > 1e-15 {
		t.Errorf("smallest subnormal: got %e want %e", got, want)
	}
	// Largest subnormal: 0x03FF = 2^(-14) * (1023/1024) ≈ 6.098e-5
	got2 := float16ToFloat32(0x03FF)
	want2 := float32(math.Ldexp(1023, -24))
	if math.Abs(float64(got2-want2)) > 1e-12 {
		t.Errorf("largest subnormal: got %e want %e", got2, want2)
	}
}

func TestFloat16ToFloat32_InfNaN(t *testing.T) {
	pinf := float16ToFloat32(0x7C00) // +inf
	ninf := float16ToFloat32(0xFC00) // -inf
	nan := float16ToFloat32(0x7E00)  // quiet NaN
	if !math.IsInf(float64(pinf), 1) {
		t.Errorf("+inf: got %v", pinf)
	}
	if !math.IsInf(float64(ninf), -1) {
		t.Errorf("-inf: got %v", ninf)
	}
	if !math.IsNaN(float64(nan)) {
		t.Errorf("NaN: got %v", nan)
	}
}

// ---- DequantLinearQ4_0 ----

// encodeQ4_0Block builds one 18-byte Q4_0 block from 32 float32 values.
// Layout: 2 bytes f16 scale (d) + 16 nibble bytes.
// d = max(abs(vals)) / 7; each element quantised as round(v/d) + 8, packed
// low nibble = elem j, high nibble = elem j+16.
func encodeQ4_0Block(vals [32]float32) [18]byte {
	var blk [18]byte
	// compute d = max(abs) / 7
	amax := float32(0)
	for _, v := range vals {
		a := v
		if a < 0 {
			a = -a
		}
		if a > amax {
			amax = a
		}
	}
	d := amax / 7.0
	if d == 0 {
		return blk // all zeros
	}
	// f32→f16 (round to nearest)
	f16bits := f32ToF16(d)
	blk[0] = byte(f16bits)
	blk[1] = byte(f16bits >> 8)
	// quantise
	for j := 0; j < 16; j++ {
		q0 := int(math.Round(float64(vals[j]/d))) + 8
		q1 := int(math.Round(float64(vals[j+16]/d))) + 8
		if q0 < 0 {
			q0 = 0
		} else if q0 > 15 {
			q0 = 15
		}
		if q1 < 0 {
			q1 = 0
		} else if q1 > 15 {
			q1 = 15
		}
		blk[2+j] = byte(q0) | byte(q1)<<4
	}
	return blk
}

// f32ToF16 converts f32 to f16 bits (round to nearest, ties to even — simplified).
func f32ToF16(f float32) uint16 {
	bits := float32bits(f)
	sign := (bits >> 16) & 0x8000
	exp := int((bits>>23)&0xFF) - 127 + 15
	man := bits & 0x7FFFFF
	if exp <= 0 {
		return uint16(sign)
	}
	if exp >= 31 {
		return uint16(sign | 0x7C00) // inf
	}
	return uint16(sign | uint32(exp)<<10 | (man >> 13))
}

func TestDequantLinearQ4_0_KnownAnswer(t *testing.T) {
	// 1 row, K=32: one Q4_0 block with known values.
	var vals [32]float32
	for i := range vals {
		vals[i] = float32(i) - 15.5
	}
	blk := encodeQ4_0Block(vals)
	W := blk[:]

	x := make([]float32, 32)
	for i := range x {
		x[i] = 1.0 // sum of dequanted values
	}

	y, err := DequantLinearQ4_0(W, x, 1, 32)
	if err != nil {
		t.Fatal(err)
	}

	// Compute expected: dequant each element, sum with x=1.0
	d := float16ToFloat32(uint16(W[0]) | uint16(W[1])<<8)
	expected := float32(0)
	for j := 0; j < 16; j++ {
		nib := W[2+j]
		q0 := float32(int(nib&0x0F)-8) * d
		q1 := float32(int(nib>>4)-8) * d
		expected += q0 + q1
	}
	if math.Abs(float64(y[0]-expected)) > 1e-4 {
		t.Fatalf("Q4_0 matvec: got %v want %v", y[0], expected)
	}
}

func TestDequantLinearQ4_0_MultiRow(t *testing.T) {
	// 3 rows, K=64 (2 blocks per row)
	M, K := 3, 64
	nbk := K / 32
	W := make([]byte, M*nbk*18)

	// Fill each block with a constant per row, to make the expected answer easy.
	for row := 0; row < M; row++ {
		for b := 0; b < nbk; b++ {
			var vals [32]float32
			for i := range vals {
				vals[i] = float32(row + 1)
			}
			blk := encodeQ4_0Block(vals)
			off := (row*nbk + b) * 18
			copy(W[off:off+18], blk[:])
		}
	}

	x := make([]float32, K)
	for i := range x {
		x[i] = 1.0
	}

	y, err := DequantLinearQ4_0(W, x, M, K)
	if err != nil {
		t.Fatal(err)
	}

	// Each row's dequanted value should be close to (row+1) for every element,
	// so the dot product ≈ (row+1) * K.
	for row := 0; row < M; row++ {
		want := float32(row+1) * float32(K)
		if math.Abs(float64(y[row]-want))/float64(want) > 0.15 {
			t.Errorf("row %d: got %v want ~%v", row, y[row], want)
		}
	}
}

func TestDequantLinearQ4_0_ZeroBlock(t *testing.T) {
	// All-zero W should produce all-zero output.
	W := make([]byte, 18)
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	y, err := DequantLinearQ4_0(W, x, 1, 32)
	if err != nil {
		t.Fatal(err)
	}
	if y[0] != 0 {
		t.Fatalf("zero block: got %v want 0", y[0])
	}
}

func TestDequantLinearQ4_0_Errors(t *testing.T) {
	if _, err := DequantLinearQ4_0(nil, nil, -1, 32); err == nil {
		t.Fatal("expected error for negative M")
	}
	if _, err := DequantLinearQ4_0(nil, nil, 1, 17); err == nil {
		t.Fatal("expected error for K not multiple of 32")
	}
	if _, err := DequantLinearQ4_0([]byte{0}, []float32{0}, 1, 32); err == nil {
		t.Fatal("expected error for W too short")
	}
	if _, err := DequantLinearQ4_0(make([]byte, 18), make([]float32, 16), 1, 32); err == nil {
		t.Fatal("expected error for x too short")
	}
}

// ---- DequantLinearQ8_0 ----

// encodeQ8_0Block builds one 34-byte Q8_0 block from 32 float32 values.
// Layout: 2 bytes f16 scale (d) + 32 int8 values.
// d = max(abs(vals)) / 127; each element quantised as round(v/d).
func encodeQ8_0Block(vals [32]float32) [34]byte {
	var blk [34]byte
	amax := float32(0)
	for _, v := range vals {
		a := v
		if a < 0 {
			a = -a
		}
		if a > amax {
			amax = a
		}
	}
	d := amax / 127.0
	if d == 0 {
		return blk
	}
	f16bits := f32ToF16(d)
	blk[0] = byte(f16bits)
	blk[1] = byte(f16bits >> 8)
	for j := 0; j < 32; j++ {
		q := int(math.Round(float64(vals[j] / d)))
		if q < -128 {
			q = -128
		} else if q > 127 {
			q = 127
		}
		blk[2+j] = byte(int8(q))
	}
	return blk
}

func TestDequantLinearQ8_0_KnownAnswer(t *testing.T) {
	var vals [32]float32
	for i := range vals {
		vals[i] = float32(i) - 15.5
	}
	blk := encodeQ8_0Block(vals)
	W := blk[:]

	x := make([]float32, 32)
	for i := range x {
		x[i] = 1.0
	}

	y, err := DequantLinearQ8_0(W, x, 1, 32)
	if err != nil {
		t.Fatal(err)
	}

	// Expected: dequant each element (int8 * d), sum.
	d := float16ToFloat32(uint16(W[0]) | uint16(W[1])<<8)
	expected := float32(0)
	for j := 0; j < 32; j++ {
		q := int8(W[2+j])
		expected += float32(q) * d
	}
	if math.Abs(float64(y[0]-expected)) > 1e-4 {
		t.Fatalf("Q8_0 matvec: got %v want %v", y[0], expected)
	}
}

func TestDequantLinearQ8_0_MultiRow(t *testing.T) {
	M, K := 3, 64
	nbk := K / 32
	W := make([]byte, M*nbk*34)

	for row := 0; row < M; row++ {
		for b := 0; b < nbk; b++ {
			var vals [32]float32
			for i := range vals {
				vals[i] = float32(row + 1)
			}
			blk := encodeQ8_0Block(vals)
			off := (row*nbk + b) * 34
			copy(W[off:off+34], blk[:])
		}
	}

	x := make([]float32, K)
	for i := range x {
		x[i] = 1.0
	}

	y, err := DequantLinearQ8_0(W, x, M, K)
	if err != nil {
		t.Fatal(err)
	}

	for row := 0; row < M; row++ {
		want := float32(row+1) * float32(K)
		// Q8_0 has much higher precision than Q4_0
		if math.Abs(float64(y[row]-want))/float64(want) > 0.02 {
			t.Errorf("row %d: got %v want ~%v", row, y[row], want)
		}
	}
}

func TestDequantLinearQ8_0_ZeroBlock(t *testing.T) {
	W := make([]byte, 34)
	x := make([]float32, 32)
	for i := range x {
		x[i] = float32(i + 1)
	}
	y, err := DequantLinearQ8_0(W, x, 1, 32)
	if err != nil {
		t.Fatal(err)
	}
	if y[0] != 0 {
		t.Fatalf("zero block: got %v want 0", y[0])
	}
}

func TestDequantLinearQ8_0_Errors(t *testing.T) {
	if _, err := DequantLinearQ8_0(nil, nil, -1, 32); err == nil {
		t.Fatal("expected error for negative M")
	}
	if _, err := DequantLinearQ8_0(nil, nil, 1, 17); err == nil {
		t.Fatal("expected error for K not multiple of 32")
	}
	if _, err := DequantLinearQ8_0([]byte{0}, []float32{0}, 1, 32); err == nil {
		t.Fatal("expected error for W too short")
	}
	if _, err := DequantLinearQ8_0(make([]byte, 34), make([]float32, 16), 1, 32); err == nil {
		t.Fatal("expected error for x too short")
	}
}

// ---- PTX dequant emission ----

func TestDequantPTXContainsEntries(t *testing.T) {
	mods := AllDequantPTXModules("sm_75")
	if len(mods) != 2 {
		t.Fatalf("expected 2 dequant PTX modules, got %d", len(mods))
	}
	for _, mod := range mods {
		if mod.Source == "" {
			t.Fatalf("empty source for %s", mod.Name)
		}
		if mod.Arch != "sm_75" {
			t.Fatalf("arch=%s for %s", mod.Arch, mod.Name)
		}
	}
}

func TestPlanLaunchDequantEntries(t *testing.T) {
	// Verify PlanLaunch recognises the dequant entries.
	for _, entry := range []string{
		PTXDequantLinearQ4_0Entry,
		"dequant_linear_q4_0",
		"cuda_dequant_linear_q4_0",
		PTXDequantLinearQ8_0Entry,
		"dequant_linear_q8_0",
		"cuda_dequant_linear_q8_0",
	} {
		st := PlanLaunch(entry)
		if st.Entry == "" {
			t.Errorf("PlanLaunch(%q): empty Entry", entry)
		}
		if st.PTXBytes == 0 {
			t.Errorf("PlanLaunch(%q): no PTX bytes", entry)
		}
	}
}
