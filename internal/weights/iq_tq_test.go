package weights

import (
	"encoding/binary"
	"math"
	"path/filepath"
	"testing"
)

func packIQ4_NL(vals []float32) []byte {
	// simplest constructive pack: choose scale so codebook index maps cleanly.
	// For tests we reverse from known indices using d=1/127.
	if len(vals)%QK4_NL != 0 {
		panic("pad")
	}
	nb := len(vals) / QK4_NL
	out := make([]byte, 0, nb*bsIQ4_NL)
	// We don't re-quantize arbitrary floats here; helpers build synthetic blocks.
	_ = vals
	return out
}

func makeIQ4NLBlock(d float32, idx [32]uint8) []byte {
	var buf [bsIQ4_NL]byte
	binary.LittleEndian.PutUint16(buf[0:2], fp16(d))
	for j := 0; j < 16; j++ {
		lo := idx[j] & 0x0F
		hi := idx[j+16] & 0x0F
		buf[2+j] = lo | (hi << 4)
	}
	return buf[:]
}

func TestDequantIQ4_NL(t *testing.T) {
	// one block: indices 0 and 15 → -127d and 113d
	var idx [32]uint8
	for i := 0; i < 16; i++ {
		idx[i] = 0
		idx[i+16] = 15
	}
	d := float32(0.01)
	blk := makeIQ4NLBlock(d, idx)
	raw := blk
	out, err := DecodeGGMLBlock(GGML_IQ4_NL, raw, 32)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 32 {
		t.Fatalf("len %d", len(out))
	}
	// layout: first half uses lo nibbles of qs[j] → index 0 → -127*d
	// second half uses hi → index 15 → 113*d
	// recompute scale through the same fp16 roundtrip used by the block
	dRT := f16bits(blk, 0)
	wantLo := dRT * float32(kvaluesIQ4NL[0])
	wantHi := dRT * float32(kvaluesIQ4NL[15])
	for j := 0; j < 16; j++ {
		if math.Abs(float64(out[j]-wantLo)) > 1e-5 {
			t.Fatalf("lo[%d]=%v want %v", j, out[j], wantLo)
		}
		if math.Abs(float64(out[j+16]-wantHi)) > 1e-5 {
			t.Fatalf("hi[%d]=%v want %v", j, out[j+16], wantHi)
		}
	}
	// all indices sweep
	for i := 0; i < 16; i++ {
		idx[i] = uint8(i)
		idx[i+16] = uint8(15 - i)
	}
	blk = makeIQ4NLBlock(1.0, idx)
	out, err = DecodeGGMLBlock(GGML_IQ4_NL, blk, 32)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 16; i++ {
		if math.Abs(float64(out[i]-float32(kvaluesIQ4NL[i]))) > 1e-4 {
			t.Fatalf("idx %d got %v", i, out[i])
		}
		if math.Abs(float64(out[i+16]-float32(kvaluesIQ4NL[15-i]))) > 1e-4 {
			t.Fatalf("idx2 %d got %v", i, out[i+16])
		}
	}
	// window partial
	w, err := DecodeGGMLWindow(GGML_IQ4_NL, blk, 32, 8, 8)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 8; i++ {
		if math.Abs(float64(w[i]-out[8+i])) > 1e-4 {
			t.Fatalf("window %v vs %v", w, out[8:16])
		}
	}
	// raw bytes
	if GGMLRawBytes(GGML_IQ4_NL, 32) != bsIQ4_NL {
		t.Fatal(GGMLRawBytes(GGML_IQ4_NL, 32))
	}
	if GGMLRawBytes(GGML_IQ4_NL, 64) != 2*bsIQ4_NL {
		t.Fatal("2 blocks")
	}
}

func makeTQ2_0Block(d float32, levels []int8) []byte {
	// levels must be 256 values in {-1,0,+1}; map q = level+1 ∈ {0,1,2}
	if len(levels) != QK_K {
		panic("need 256")
	}
	buf := make([]byte, bsTQ2_0)
	// inverse of dequant layout
	// for j in 0..64 step 32; for l 0..3; for m 0..31: oi advances
	// so level index = j_chunk*128 + l*32 + m, with j_chunk = j/32, j runs 0,32
	// qs[j+m] bits l*2 hold q for that level
	for oi, lev := range levels {
		q := lev + 1
		if q < 0 {
			q = 0
		}
		if q > 2 {
			q = 2
		}
		// invert oi → (j,l,m)
		// oi = j*4 + l*32 + m wait: j steps 32 bytes, each j-chunk produces 4*32=128 elems
		chunk := oi / 128
		rem := oi % 128
		l := rem / 32
		m := rem % 32
		j := chunk * 32
		pos := j + m
		buf[pos] |= byte(q&3) << uint(l*2)
	}
	binary.LittleEndian.PutUint16(buf[QK_K/4:], fp16(d))
	return buf
}

func TestDequantTQ2_0(t *testing.T) {
	// pattern: first 32 are -1, next 32 are 0, next 32 are +1, rest 0
	levels := make([]int8, QK_K)
	for i := 0; i < 32; i++ {
		levels[i] = -1
		levels[32+i] = 0
		levels[64+i] = 1
	}
	d := float32(0.5)
	blk := makeTQ2_0Block(d, levels)
	out, err := DecodeGGMLBlock(GGML_TQ2_0, blk, QK_K)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 256 {
		t.Fatal(len(out))
	}
	for i := 0; i < 32; i++ {
		if math.Abs(float64(out[i]-(-1)*d)) > 1e-4 {
			t.Fatalf("i%d=%v", i, out[i])
		}
		if math.Abs(float64(out[32+i]-0)) > 1e-4 {
			t.Fatalf("z%d=%v", i, out[32+i])
		}
		if math.Abs(float64(out[64+i]-1*d)) > 1e-4 {
			t.Fatalf("p%d=%v", i, out[64+i])
		}
	}
	if GGMLRawBytes(GGML_TQ2_0, 256) != bsTQ2_0 {
		t.Fatal(GGMLRawBytes(GGML_TQ2_0, 256))
	}
	// window
	w, err := DecodeGGMLWindow(GGML_TQ2_0, blk, 256, 64, 16)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 16; i++ {
		if math.Abs(float64(w[i]-out[64+i])) > 1e-4 {
			t.Fatalf("win %v", w)
		}
	}
}

func TestSessionIQ4_NL(t *testing.T) {
	// 2×2 matrix IQ4_NL: build 4 elems in one partial block (need full block of 32)
	// pad to 32 with zeros indices mid (index 8 ~ 1)
	dir := t.TempDir()
	path := filepath.Join(dir, "iq4.gguf")
	var idx [32]uint8
	// pack rows [1,0; 0,1] identity approx using codebook
	// use d chosen so kvalues[i]*d ≈ target. With d=1/127: idx0→-1, idx15→113/127
	// For simplicity store exact f32-like by dequanting known and checking LoadF32.
	// We store identity rows as [k0,0,0,k0] won't fit; store 2x2 as first 4 of 32:
	// Want mat [[2,0],[0,3]] at positions 0..3; rest 0 → index 8 ~1 * d with d chosen?
	// indices: use 0→-127, hard. Just use positive: index 11 → 38
	// Better: manual d and index for exact 1.0 values:
	// want value 1 → pick idx with kvalues=1 which is index 8, d=1 → 1.0
	idx[0] = 8 // 1
	idx[1] = 8 // wait row0: [1,0]
	// actually identity:
	// elem0=1 (r0c0), elem1=0 (r0c1), elem2=0 (r1c0), elem3=1 (r1c1)
	// 0 maps to kvalues[?] none is 0. Closest small: use d scale after?
	// For matvec test: linear 2x2 with W row-major [1,0,0,1] after dequant.
	// set d=1, use idx8=1 for diagonals; for zeros use idx7=-10? no.
	// offset: store scaled? Easier: use two equal positive weights.
	// W = [[k[15], 0], [0, k[15]]] with zeros from a zero-scale second block? Only one block.
	// Use LoadF32 alone for type path; fused uses full M*K.
	// Make M=1 K=32 row of 1's (idx8) * d=1 → all 1's; x all 1 → dot 32
	for i := 0; i < 32; i++ {
		idx[i] = 8 // kvalues=1
	}
	blk := makeIQ4NLBlock(1.0, idx)
	if err := WriteGGUFTyped(path, "W", GGML_IQ4_NL, []uint64{32, 1}, blk, map[string]string{"general.name": "iq4"}); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	// full load
	vals, shape, err := s.LoadF32("W")
	if err != nil {
		t.Fatal(err)
	}
	if len(vals) != 32 {
		t.Fatal(len(vals), shape)
	}
	for i, v := range vals {
		if math.Abs(float64(v-1)) > 1e-3 {
			t.Fatalf("load[%d]=%v", i, v)
		}
	}
	// windowed
	w, err := s.LoadF32Window("W", 0, 8)
	if err != nil {
		t.Fatal(err)
	}
	if len(w) != 8 {
		t.Fatal(len(w))
	}
	// fused matvec: M=1 K=32, x ones → y≈32
	x := make([]float32, 32)
	for i := range x {
		x[i] = 1
	}
	y, err := s.LinearFused("W", x, 1, 32, 1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(y[0]-32)) > 1e-2 {
		t.Fatalf("y=%v", y)
	}
}

func TestSessionTQ2_0(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tq2.gguf")
	levels := make([]int8, QK_K)
	// one row of 256: all +1
	for i := range levels {
		levels[i] = 1
	}
	blk := makeTQ2_0Block(1.0, levels)
	if err := WriteGGUFTyped(path, "W", GGML_TQ2_0, []uint64{256, 1}, blk, map[string]string{"general.name": "tq2"}); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	vals, _, err := s.LoadF32("W")
	if err != nil {
		t.Fatal(err)
	}
	if len(vals) != 256 {
		t.Fatal(len(vals))
	}
	for i, v := range vals {
		if math.Abs(float64(v-1)) > 1e-3 {
			t.Fatalf("v[%d]=%v", i, v)
		}
	}
	x := make([]float32, 256)
	for i := range x {
		x[i] = 1
	}
	y, err := s.LinearFused("W", x, 1, 256, 1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(y[0]-256)) > 0.1 {
		t.Fatalf("y=%v", y)
	}
	// unsupported IQ2_XXS still errors
	if _, err := DecodeGGMLBlock(GGML_IQ2_XXS, []byte{0}, 256); err == nil {
		t.Fatal("expected unsupported")
	}
}


func makeIQ4XSBlock(d float32, ls [8]int, idx [256]uint8) []byte {
	// ls values in 0..63; maps to scale (ls-32)*d
	buf := make([]byte, bsIQ4_XS)
	binary.LittleEndian.PutUint16(buf[0:2], fp16(d))
	var scalesH uint16
	scalesL := make([]byte, QK_K/64)
	for ib := 0; ib < 8; ib++ {
		v := ls[ib] & 63
		// low 4 bits in scales_l, high 2 in scales_h
		lo := v & 0x0F
		hi := (v >> 4) & 3
		if ib%2 == 0 {
			scalesL[ib/2] = byte(lo)
		} else {
			scalesL[ib/2] |= byte(lo << 4)
		}
		scalesH |= uint16(hi) << (2 * uint(ib))
	}
	binary.LittleEndian.PutUint16(buf[2:4], scalesH)
	copy(buf[4:8], scalesL)
	qs := buf[8:]
	for ib := 0; ib < 8; ib++ {
		base := ib * 32
		for j := 0; j < 16; j++ {
			lo := idx[base+j] & 0x0F
			hi := idx[base+16+j] & 0x0F
			qs[ib*16+j] = lo | (hi << 4)
		}
	}
	return buf
}

func TestDequantIQ4_XS(t *testing.T) {
	var ls [8]int
	// all mid scale ls=32 → dl=0; then ls=33 → dl=d
	for i := range ls {
		ls[i] = 33
	}
	var idx [256]uint8
	for i := 0; i < 256; i++ {
		idx[i] = 8 // kvalues=1 → y = d*1
	}
	d := float32(2.0)
	blk := makeIQ4XSBlock(d, ls, idx)
	out, err := DecodeGGMLBlock(GGML_IQ4_XS, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT := f16bits(blk, 0)
	// dl = dRT * (33-32) = dRT
	want := dRT * 1.0
	for i, v := range out {
		if math.Abs(float64(v-want)) > 1e-3 {
			t.Fatalf("[%d]=%v want %v", i, v, want)
		}
	}
	// different scale on sub-block 0: ls=34 → 2*dRT
	ls[0] = 34
	blk = makeIQ4XSBlock(d, ls, idx)
	out, err = DecodeGGMLBlock(GGML_IQ4_XS, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT = f16bits(blk, 0)
	if math.Abs(float64(out[0]-2*dRT)) > 1e-3 || math.Abs(float64(out[32]-dRT)) > 1e-3 {
		t.Fatalf("scale mismatch %v %v", out[0], out[32])
	}
	if GGMLRawBytes(GGML_IQ4_XS, 256) != bsIQ4_XS {
		t.Fatal(GGMLRawBytes(GGML_IQ4_XS, 256), bsIQ4_XS)
	}
	w, err := DecodeGGMLWindow(GGML_IQ4_XS, blk, 256, 16, 8)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 8; i++ {
		if math.Abs(float64(w[i]-out[16+i])) > 1e-4 {
			t.Fatal(w, out[16:24])
		}
	}
}

// makeTQ1_0Block packs desired dequant outputs {-1,0,+1}*d for 256 elems.
// builds an encode-ordered trit stream so llama dequant recovers levels[]
// in its (n-outer, m-inner) walk for qs and qh.
func makeTQ1_0Block(d float32, levels []int8) []byte {
	if len(levels) != QK_K {
		panic("256")
	}
	const qsN = (QK_K - 4*(QK_K/64)) / 5 // 48
	const qhN = QK_K / 64                 // 4
	buf := make([]byte, bsTQ1_0)

	// Convert desired dequant stream → encode spatial layout (same as quantize_row walks x).
	// qs region 0 (160): dequant emits for n,m at ei=n*32+m value for encode pos m+n*32
	// so enc[m+n*32] = digit(levels[n*32+m])
	enc := make([]int, 256)
	for n := 0; n < 5; n++ {
		for m := 0; m < 32; m++ {
			enc[m+n*32] = clampTritDigit(levels[n*32+m])
		}
	}
	for n := 0; n < 5; n++ {
		for m := 0; m < 16; m++ {
			enc[160+m+n*16] = clampTritDigit(levels[160+n*16+m])
		}
	}
	// qh: dequant for n,j: ei=240+n*4+j from x[j+n*4] so enc[240+j+n*4]=digit(levels[ei])
	for n := 0; n < 4; n++ {
		for j := 0; j < 4; j++ {
			enc[240+j+n*4] = clampTritDigit(levels[240+n*4+j])
		}
	}

	// qs first 32 (one j=0 chunk)
	for m := 0; m < 32; m++ {
		q := 0
		for n := 0; n < 5; n++ {
			q = q*3 + enc[m+n*32]
		}
		q = (q*256 + 242) / 243
		buf[m] = byte(q)
	}
	// qs rem 16
	for m := 0; m < 16; m++ {
		q := 0
		for n := 0; n < 5; n++ {
			q = q*3 + enc[160+m+n*16]
		}
		q = (q*256 + 242) / 243
		buf[32+m] = byte(q)
	}
	// qh
	for j := 0; j < qhN; j++ {
		q := 0
		for m := 0; m < 4; m++ {
			q = q*3 + enc[240+j+m*qhN]
		}
		q *= 3
		q = (q*256 + 242) / 243
		buf[qsN+j] = byte(q)
	}
	binary.LittleEndian.PutUint16(buf[qsN+qhN:], fp16(d))
	return buf
}

func clampTritDigit(lev int8) int {
	dgt := int(lev) + 1
	if dgt < 0 {
		return 0
	}
	if dgt > 2 {
		return 2
	}
	return dgt
}

func TestDequantTQ1_0(t *testing.T) {
	levels := make([]int8, 256)
	// first 32: -1, next 32: 0, next 32: +1, rest 0
	for i := 0; i < 32; i++ {
		levels[i] = -1
		levels[32+i] = 0
		levels[64+i] = 1
	}
	d := float32(0.5)
	blk := makeTQ1_0Block(d, levels)
	if len(blk) != bsTQ1_0 {
		t.Fatal(len(blk), bsTQ1_0)
	}
	out, err := DecodeGGMLBlock(GGML_TQ1_0, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT := f16bits(blk, (QK_K-4*(QK_K/64))/5+QK_K/64)
	for i := 0; i < 32; i++ {
		if math.Abs(float64(out[i]-(-1)*dRT)) > 1e-4 {
			t.Fatalf("m1 %d %v", i, out[i])
		}
		if math.Abs(float64(out[32+i]-0)) > 1e-4 {
			t.Fatalf("z %d %v", i, out[32+i])
		}
		if math.Abs(float64(out[64+i]-1*dRT)) > 1e-4 {
			t.Fatalf("p %d %v", i, out[64+i])
		}
	}
	if GGMLRawBytes(GGML_TQ1_0, 256) != bsTQ1_0 {
		t.Fatal(GGMLRawBytes(GGML_TQ1_0, 256))
	}
	w, err := DecodeGGMLWindow(GGML_TQ1_0, blk, 256, 60, 8)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 8; i++ {
		if math.Abs(float64(w[i]-out[60+i])) > 1e-4 {
			t.Fatal(w)
		}
	}
}

func TestSessionIQ4_XS_TQ1(t *testing.T) {
	dir := t.TempDir()
	// IQ4_XS: 1x256 all ≈ dRT (ls=33, idx=8)
	var ls [8]int
	for i := range ls {
		ls[i] = 33
	}
	var idx [256]uint8
	for i := range idx {
		idx[i] = 8
	}
	blk := makeIQ4XSBlock(1.0, ls, idx)
	path := filepath.Join(dir, "iq4xs.gguf")
	if err := WriteGGUFTyped(path, "W", GGML_IQ4_XS, []uint64{256, 1}, blk, nil); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	vals, _, err := s.LoadF32("W")
	if err != nil {
		t.Fatal(err)
	}
	dRT := f16bits(blk, 0)
	// y = dRT * 1 for each; linear with x=ones → 256*dRT
	x := make([]float32, 256)
	for i := range x {
		x[i] = 1
	}
	y, err := s.LinearFused("W", x, 1, 256, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := float64(256) * float64(dRT)
	if math.Abs(float64(y[0])-want) > 0.5 {
		t.Fatalf("iq4xs y=%v want %v vals0=%v", y[0], want, vals[0])
	}

	// TQ1 all +1
	levels := make([]int8, 256)
	for i := range levels {
		levels[i] = 1
	}
	tblk := makeTQ1_0Block(1.0, levels)
	path2 := filepath.Join(dir, "tq1.gguf")
	if err := WriteGGUFTyped(path2, "W", GGML_TQ1_0, []uint64{256, 1}, tblk, nil); err != nil {
		t.Fatal(err)
	}
	s2, err := OpenSession(path2)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s2.ID)
	x2 := make([]float32, 256)
	for i := range x2 {
		x2[i] = 1
	}
	y2, err := s2.LinearFused("W", x2, 1, 256, 1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(y2[0]-256)) > 0.5 {
		t.Fatalf("tq1 y=%v", y2)
	}
}


func makeIQ2XXSBlock(d float32, gridIdx byte, scaleNibble uint8, signIdx uint8) []byte {
	// scaleNibble 0..15 in high 4 bits of each aux1; signIdx 0..127 packs into 7-bit slots.
	buf := make([]byte, bsIQ2_XXS)
	binary.LittleEndian.PutUint16(buf[0:2], fp16(d))
	qs := buf[2:]
	for ib32 := 0; ib32 < QK_K/32; ib32++ {
		// aux0: 4 grid indices
		off := 8 * ib32
		qs[off+0] = gridIdx
		qs[off+1] = gridIdx
		qs[off+2] = gridIdx
		qs[off+3] = gridIdx
		// aux1: signs in 7-bit slots l=0..3, scale in top nibble
		var aux1 uint32
		for l := 0; l < 4; l++ {
			aux1 |= uint32(signIdx&127) << (7 * uint(l))
		}
		aux1 |= uint32(scaleNibble&0x0F) << 28
		binary.LittleEndian.PutUint32(qs[off+4:off+8], aux1)
	}
	return buf
}

func TestDequantIQ2_XXS(t *testing.T) {
	// grid[0]=all 8s, signs=0, scale=0 → y = d*(0.5)*0.25*8 = d
	d := float32(2.0)
	blk := makeIQ2XXSBlock(d, 0, 0, 0)
	out, err := DecodeGGMLBlock(GGML_IQ2_XXS, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT := f16bits(blk, 0)
	for i, v := range out {
		if math.Abs(float64(v-dRT)) > 1e-3 {
			t.Fatalf("[%d]=%v want %v", i, v, dRT)
		}
	}
	// scale nibble=1 → db = d*(0.5+1)*0.25 = d*1.5/4; y = db*8 = d*3
	blk = makeIQ2XXSBlock(d, 0, 1, 0)
	out, err = DecodeGGMLBlock(GGML_IQ2_XXS, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT = f16bits(blk, 0)
	want := dRT * 3
	if math.Abs(float64(out[0]-want)) > 1e-2 {
		t.Fatalf("scale %v want %v", out[0], want)
	}
	// signIdx=1 → ksigns[1]=129 has bit 0 set → first of each 8-group negative
	blk = makeIQ2XXSBlock(d, 0, 0, 1)
	out, err = DecodeGGMLBlock(GGML_IQ2_XXS, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT = f16bits(blk, 0)
	if out[0] >= 0 || math.Abs(float64(out[0]+dRT)) > 1e-2 {
		t.Fatalf("signed0 %v", out[0])
	}
	if out[1] <= 0 || math.Abs(float64(out[1]-dRT)) > 1e-2 {
		t.Fatalf("signed1 %v", out[1])
	}
	if GGMLRawBytes(GGML_IQ2_XXS, 256) != bsIQ2_XXS {
		t.Fatal(GGMLRawBytes(GGML_IQ2_XXS, 256), bsIQ2_XXS)
	}
	w, err := DecodeGGMLWindow(GGML_IQ2_XXS, blk, 256, 10, 6)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 6; i++ {
		if math.Abs(float64(w[i]-out[10+i])) > 1e-4 {
			t.Fatal(w, out[10:16])
		}
	}
}

func makeIQ3XXSBlock(d float32, gridIdx byte, scaleNibble uint8, signIdx uint8) []byte {
	buf := make([]byte, bsIQ3_XXS)
	binary.LittleEndian.PutUint16(buf[0:2], fp16(d))
	qs := buf[2 : 2+QK_K/4] // 64 grid index bytes
	for i := range qs {
		qs[i] = gridIdx
	}
	ss := buf[2+QK_K/4:] // 32 bytes scales_and_signs
	for ib32 := 0; ib32 < QK_K/32; ib32++ {
		var aux uint32
		for l := 0; l < 4; l++ {
			aux |= uint32(signIdx&127) << (7 * uint(l))
		}
		aux |= uint32(scaleNibble&0x0F) << 28
		binary.LittleEndian.PutUint32(ss[4*ib32:4*ib32+4], aux)
	}
	return buf
}

func TestDequantIQ3_XXS(t *testing.T) {
	// grid[0]=all 4s, signs=0, scale=0 → y = d*0.5*0.5*4 = d
	d := float32(1.5)
	blk := makeIQ3XXSBlock(d, 0, 0, 0)
	out, err := DecodeGGMLBlock(GGML_IQ3_XXS, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT := f16bits(blk, 0)
	for i, v := range out {
		if math.Abs(float64(v-dRT)) > 1e-3 {
			t.Fatalf("[%d]=%v want %v", i, v, dRT)
		}
	}
	// scale=1 → db=d*(1.5)*0.5=0.75d; y=0.75d*4=3d
	blk = makeIQ3XXSBlock(d, 0, 1, 0)
	out, err = DecodeGGMLBlock(GGML_IQ3_XXS, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT = f16bits(blk, 0)
	if math.Abs(float64(out[0]-3*dRT)) > 1e-2 {
		t.Fatalf("scale %v want %v", out[0], 3*dRT)
	}
	if GGMLRawBytes(GGML_IQ3_XXS, 256) != bsIQ3_XXS {
		t.Fatal(GGMLRawBytes(GGML_IQ3_XXS, 256))
	}
	w, err := DecodeGGMLWindow(GGML_IQ3_XXS, blk, 256, 200, 8)
	if err != nil {
		t.Fatal(err)
	}
	full, _ := DecodeGGMLBlock(GGML_IQ3_XXS, blk, 256)
	for i := 0; i < 8; i++ {
		if math.Abs(float64(w[i]-full[200+i])) > 1e-4 {
			t.Fatal(w)
		}
	}
}

func TestSessionIQ2_IQ3_XXS(t *testing.T) {
	dir := t.TempDir()
	// IQ2 all ≈ dRT
	blk := makeIQ2XXSBlock(1.0, 0, 0, 0)
	path := filepath.Join(dir, "iq2.gguf")
	if err := WriteGGUFTyped(path, "W", GGML_IQ2_XXS, []uint64{256, 1}, blk, nil); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	dRT := f16bits(blk, 0)
	x := make([]float32, 256)
	for i := range x {
		x[i] = 1
	}
	y, err := s.LinearFused("W", x, 1, 256, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := float64(256) * float64(dRT)
	if math.Abs(float64(y[0])-want) > 1.0 {
		t.Fatalf("iq2 y=%v want %v", y[0], want)
	}

	blk3 := makeIQ3XXSBlock(1.0, 0, 0, 0)
	path3 := filepath.Join(dir, "iq3.gguf")
	if err := WriteGGUFTyped(path3, "W", GGML_IQ3_XXS, []uint64{256, 1}, blk3, nil); err != nil {
		t.Fatal(err)
	}
	s3, err := OpenSession(path3)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s3.ID)
	y3, err := s3.LinearFused("W", x, 1, 256, 1)
	if err != nil {
		t.Fatal(err)
	}
	d3 := f16bits(blk3, 0)
	want3 := float64(256) * float64(d3)
	if math.Abs(float64(y3[0])-want3) > 1.0 {
		t.Fatalf("iq3 y=%v want %v", y3[0], want3)
	}
}


func makeIQ2XSBlock(d float32, gridIdx uint16, scaleLo, scaleHi uint8, signIdx uint8) []byte {
	buf := make([]byte, bsIQ2_XS)
	binary.LittleEndian.PutUint16(buf[0:2], fp16(d))
	qs := buf[2 : 2+(QK_K/8)*2]
	scales := buf[2+(QK_K/8)*2:]
	for ib32 := 0; ib32 < QK_K/32; ib32++ {
		scales[ib32] = (scaleHi << 4) | (scaleLo & 0xf)
		for l := 0; l < 4; l++ {
			q := (uint16(signIdx&127) << 9) | (gridIdx & 511)
			binary.LittleEndian.PutUint16(qs[2*(4*ib32+l):2*(4*ib32+l)+2], q)
		}
	}
	return buf
}

func TestDequantIQ2_XS(t *testing.T) {
	// grid[0] values all 8 (0x08...), signs=0, scale=0 → y = d*(0.5)*0.25*8 = d
	d := float32(2.0)
	blk := makeIQ2XSBlock(d, 0, 0, 0, 0)
	out, err := DecodeGGMLBlock(GGML_IQ2_XS, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT := f16bits(blk, 0)
	for i, v := range out {
		if math.Abs(float64(v-dRT)) > 1e-3 {
			t.Fatalf("[%d]=%v want %v", i, v, dRT)
		}
	}
	// scaleLo=1 → db = d*1.5*0.25; y = db*8 = d*3
	blk = makeIQ2XSBlock(d, 0, 1, 0, 0)
	out, err = DecodeGGMLBlock(GGML_IQ2_XS, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT = f16bits(blk, 0)
	if math.Abs(float64(out[0]-3*dRT)) > 1e-2 {
		t.Fatalf("scale %v want %v", out[0], 3*dRT)
	}
	// signIdx=1 → ksigns[1]=129 has bit0 set
	blk = makeIQ2XSBlock(d, 0, 0, 0, 1)
	out, err = DecodeGGMLBlock(GGML_IQ2_XS, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT = f16bits(blk, 0)
	if out[0] >= 0 || math.Abs(float64(out[0]+dRT)) > 1e-2 {
		t.Fatalf("signed0 %v", out[0])
	}
	if out[1] <= 0 || math.Abs(float64(out[1]-dRT)) > 1e-2 {
		t.Fatalf("signed1 %v", out[1])
	}
	if GGMLRawBytes(GGML_IQ2_XS, 256) != bsIQ2_XS {
		t.Fatal(GGMLRawBytes(GGML_IQ2_XS, 256))
	}
	w, err := DecodeGGMLWindow(GGML_IQ2_XS, blk, 256, 20, 8)
	if err != nil {
		t.Fatal(err)
	}
	full, _ := DecodeGGMLBlock(GGML_IQ2_XS, blk, 256)
	for i := 0; i < 8; i++ {
		if math.Abs(float64(w[i]-full[20+i])) > 1e-4 {
			t.Fatal(w)
		}
	}
}

func makeIQ2SBlock(d float32, gridLow byte, qhBits byte, scaleLo, scaleHi uint8, signByte byte) []byte {
	// qs layout: qs[0:32] low grid, qs[32:64] signs, then qh[8], scales[8]
	buf := make([]byte, bsIQ2_S)
	binary.LittleEndian.PutUint16(buf[0:2], fp16(d))
	qsAll := buf[2 : 2+QK_K/4]
	qh := buf[2+QK_K/4 : 2+QK_K/4+QK_K/32]
	scales := buf[2+QK_K/4+QK_K/32:]
	for i := 0; i < QK_K/8; i++ {
		qsAll[i] = gridLow
	}
	for i := QK_K / 8; i < QK_K/4; i++ {
		qsAll[i] = signByte
	}
	for i := range qh {
		qh[i] = qhBits
	}
	for ib32 := 0; ib32 < QK_K/32; ib32++ {
		scales[ib32] = (scaleHi << 4) | (scaleLo & 0xf)
	}
	return buf
}

func TestDequantIQ2_S(t *testing.T) {
	// grid index 0 → all 8s; scale 0; signs 0 → y=d
	d := float32(1.5)
	blk := makeIQ2SBlock(d, 0, 0, 0, 0, 0)
	out, err := DecodeGGMLBlock(GGML_IQ2_S, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT := f16bits(blk, 0)
	for i, v := range out {
		if math.Abs(float64(v-dRT)) > 1e-3 {
			t.Fatalf("[%d]=%v want %v", i, v, dRT)
		}
	}
	// scaleLo=1 → 3d
	blk = makeIQ2SBlock(d, 0, 0, 1, 0, 0)
	out, err = DecodeGGMLBlock(GGML_IQ2_S, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT = f16bits(blk, 0)
	if math.Abs(float64(out[0]-3*dRT)) > 1e-2 {
		t.Fatalf("scale %v want %v", out[0], 3*dRT)
	}
	// sign byte 1 → bit0 set → first of each 8-group negative
	blk = makeIQ2SBlock(d, 0, 0, 0, 0, 1)
	out, err = DecodeGGMLBlock(GGML_IQ2_S, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT = f16bits(blk, 0)
	if out[0] >= 0 || math.Abs(float64(out[0]+dRT)) > 1e-2 {
		t.Fatalf("signed0 %v", out[0])
	}
	if out[1] <= 0 {
		t.Fatalf("signed1 %v", out[1])
	}
	if GGMLRawBytes(GGML_IQ2_S, 256) != bsIQ2_S {
		t.Fatal(GGMLRawBytes(GGML_IQ2_S, 256))
	}
	w, err := DecodeGGMLWindow(GGML_IQ2_S, blk, 256, 100, 5)
	if err != nil {
		t.Fatal(err)
	}
	full, _ := DecodeGGMLBlock(GGML_IQ2_S, blk, 256)
	for i := 0; i < 5; i++ {
		if math.Abs(float64(w[i]-full[100+i])) > 1e-4 {
			t.Fatal(w)
		}
	}
}

func TestSessionIQ2_XS_S(t *testing.T) {
	dir := t.TempDir()
	x := make([]float32, 256)
	for i := range x {
		x[i] = 1
	}
	blk := makeIQ2XSBlock(1.0, 0, 0, 0, 0)
	path := filepath.Join(dir, "iq2xs.gguf")
	if err := WriteGGUFTyped(path, "W", GGML_IQ2_XS, []uint64{256, 1}, blk, nil); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	dRT := f16bits(blk, 0)
	y, err := s.LinearFused("W", x, 1, 256, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := float64(256) * float64(dRT)
	if math.Abs(float64(y[0])-want) > 1.0 {
		t.Fatalf("xs y=%v want %v", y[0], want)
	}

	blkS := makeIQ2SBlock(1.0, 0, 0, 0, 0, 0)
	pathS := filepath.Join(dir, "iq2s.gguf")
	if err := WriteGGUFTyped(pathS, "W", GGML_IQ2_S, []uint64{256, 1}, blkS, nil); err != nil {
		t.Fatal(err)
	}
	s2, err := OpenSession(pathS)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s2.ID)
	y2, err := s2.LinearFused("W", x, 1, 256, 1)
	if err != nil {
		t.Fatal(err)
	}
	d2 := f16bits(blkS, 0)
	want2 := float64(256) * float64(d2)
	if math.Abs(float64(y2[0])-want2) > 1.0 {
		t.Fatalf("s y=%v want %v", y2[0], want2)
	}
}


func makeIQ3SBlock(d float32, gridLow byte, qhBits byte, scaleLo, scaleHi uint8, signByte byte) []byte {
	// d + qs[64] + qh[8] + signs[32] + scales[4]
	buf := make([]byte, bsIQ3_S)
	binary.LittleEndian.PutUint16(buf[0:2], fp16(d))
	qs := buf[2 : 2+QK_K/4]
	qh := buf[2+QK_K/4 : 2+QK_K/4+QK_K/32]
	signs := buf[2+QK_K/4+QK_K/32 : 2+QK_K/4+QK_K/32+QK_K/8]
	scales := buf[2+QK_K/4+QK_K/32+QK_K/8:]
	for i := range qs {
		qs[i] = gridLow
	}
	for i := range qh {
		qh[i] = qhBits
	}
	for i := range signs {
		signs[i] = signByte
	}
	for i := range scales {
		scales[i] = (scaleHi << 4) | (scaleLo & 0xf)
	}
	return buf
}

func TestDequantIQ3_S(t *testing.T) {
	// grid[0]=0x01010101 → all ones; scale 0 → y=d
	d := float32(2.0)
	blk := makeIQ3SBlock(d, 0, 0, 0, 0, 0)
	out, err := DecodeGGMLBlock(GGML_IQ3_S, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT := f16bits(blk, 0)
	for i, v := range out {
		if math.Abs(float64(v-dRT)) > 1e-3 {
			t.Fatalf("[%d]=%v want %v", i, v, dRT)
		}
	}
	// scaleLo=1 → db = d*(1+2)=3d
	blk = makeIQ3SBlock(d, 0, 0, 1, 0, 0)
	out, err = DecodeGGMLBlock(GGML_IQ3_S, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT = f16bits(blk, 0)
	// first 32 of each 64 use scaleLo; next 32 use scaleHi=0
	if math.Abs(float64(out[0]-3*dRT)) > 1e-2 {
		t.Fatalf("scale0 %v want %v", out[0], 3*dRT)
	}
	if math.Abs(float64(out[32]-dRT)) > 1e-2 {
		t.Fatalf("scale1 half %v want %v", out[32], dRT)
	}
	// sign bit0 → first of each 8-group negative
	blk = makeIQ3SBlock(d, 0, 0, 0, 0, 1)
	out, err = DecodeGGMLBlock(GGML_IQ3_S, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT = f16bits(blk, 0)
	if out[0] >= 0 || math.Abs(float64(out[0]+dRT)) > 1e-2 {
		t.Fatalf("signed0 %v", out[0])
	}
	if out[1] <= 0 || math.Abs(float64(out[1]-dRT)) > 1e-2 {
		t.Fatalf("signed1 %v", out[1])
	}
	if GGMLRawBytes(GGML_IQ3_S, 256) != bsIQ3_S {
		t.Fatal(GGMLRawBytes(GGML_IQ3_S, 256), bsIQ3_S)
	}
	w, err := DecodeGGMLWindow(GGML_IQ3_S, blk, 256, 40, 6)
	if err != nil {
		t.Fatal(err)
	}
	full, _ := DecodeGGMLBlock(GGML_IQ3_S, blk, 256)
	for i := 0; i < 6; i++ {
		if math.Abs(float64(w[i]-full[40+i])) > 1e-4 {
			t.Fatal(w)
		}
	}
}

func TestSessionIQ3_S(t *testing.T) {
	dir := t.TempDir()
	x := make([]float32, 256)
	for i := range x {
		x[i] = 1
	}
	blk := makeIQ3SBlock(1.0, 0, 0, 0, 0, 0)
	path := filepath.Join(dir, "iq3s.gguf")
	if err := WriteGGUFTyped(path, "W", GGML_IQ3_S, []uint64{256, 1}, blk, nil); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	dRT := f16bits(blk, 0)
	y, err := s.LinearFused("W", x, 1, 256, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := float64(256) * float64(dRT)
	if math.Abs(float64(y[0])-want) > 1.0 {
		t.Fatalf("y=%v want %v", y[0], want)
	}
}


func makeIQ1SBlock(d float32, gridIdxLo byte, scale3 uint16, negDelta bool) []byte {
	// d + qs[32] + qh[8]uint16; use same grid for all 4 slots in each 32-group
	buf := make([]byte, bsIQ1_S)
	binary.LittleEndian.PutUint16(buf[0:2], fp16(d))
	qs := buf[2 : 2+QK_K/8]
	qhBytes := buf[2+QK_K/8:]
	for i := range qs {
		qs[i] = gridIdxLo
	}
	// qh: bits 0-2,3-5,6-8,9-11 = hi3 for 4 grids (0); bits12-14 = scale3; bit15 = delta sign
	// hi3=0 so index = gridIdxLo; scale3; optional 0x8000
	base := (scale3 & 7) << 12
	if negDelta {
		base |= 0x8000
	}
	for ib := 0; ib < QK_K/32; ib++ {
		binary.LittleEndian.PutUint16(qhBytes[2*ib:2*ib+2], base)
	}
	return buf
}

func TestDequantIQ1_S(t *testing.T) {
	// grid[2047]=0x0101010101010101 → all +1; scale=0 → dl=d; delta=+0.125
	// y = d * (1 + 0.125) = 1.125 d
	d := float32(2.0)
	// gridIdx 2047 needs hi bits = 2047>>8 = 7 in each 3-bit field
	buf := make([]byte, bsIQ1_S)
	binary.LittleEndian.PutUint16(buf[0:2], fp16(d))
	qs := buf[2 : 2+QK_K/8]
	qhB := buf[2+QK_K/8:]
	for i := range qs {
		qs[i] = byte(2047 & 0xff) // 0xff
	}
	// hi3=7 for each of 4 groups: bits 0-2,3-5,6-8,9-11 all 7; scale=0
	qh := uint16(7 | (7 << 3) | (7 << 6) | (7 << 9))
	for ib := 0; ib < 8; ib++ {
		binary.LittleEndian.PutUint16(qhB[2*ib:2*ib+2], qh)
	}
	out, err := DecodeGGMLBlock(GGML_IQ1_S, buf, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT := f16bits(buf, 0)
	want := dRT * (1 + 0.125)
	for i, v := range out {
		if math.Abs(float64(v-want)) > 1e-3 {
			t.Fatalf("[%d]=%v want %v", i, v, want)
		}
	}
	// scale=1 → dl=d*(2*1+1)=3d; y=3d*(1.125)
	qh = uint16(7 | (7 << 3) | (7 << 6) | (7 << 9) | (1 << 12))
	for ib := 0; ib < 8; ib++ {
		binary.LittleEndian.PutUint16(qhB[2*ib:2*ib+2], qh)
	}
	out, err = DecodeGGMLBlock(GGML_IQ1_S, buf, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT = f16bits(buf, 0)
	want = 3 * dRT * (1 + 0.125)
	if math.Abs(float64(out[0]-want)) > 1e-2 {
		t.Fatalf("scale %v want %v", out[0], want)
	}
	// neg delta
	qh |= 0x8000
	for ib := 0; ib < 8; ib++ {
		binary.LittleEndian.PutUint16(qhB[2*ib:2*ib+2], qh)
	}
	out, err = DecodeGGMLBlock(GGML_IQ1_S, buf, 256)
	if err != nil {
		t.Fatal(err)
	}
	want = 3 * dRT * (1 - 0.125)
	if math.Abs(float64(out[0]-want)) > 1e-2 {
		t.Fatalf("neg %v want %v", out[0], want)
	}
	// grid[0]=all -1 (0xff bytes as int8)
	for i := range qs {
		qs[i] = 0
	}
	qh = 0 // hi3=0 scale=0 delta+
	for ib := 0; ib < 8; ib++ {
		binary.LittleEndian.PutUint16(qhB[2*ib:2*ib+2], qh)
	}
	out, err = DecodeGGMLBlock(GGML_IQ1_S, buf, 256)
	if err != nil {
		t.Fatal(err)
	}
	dRT = f16bits(buf, 0)
	want = dRT * (-1 + 0.125)
	if math.Abs(float64(out[0]-want)) > 1e-2 {
		t.Fatalf("grid0 %v want %v", out[0], want)
	}
	if GGMLRawBytes(GGML_IQ1_S, 256) != bsIQ1_S {
		t.Fatal(GGMLRawBytes(GGML_IQ1_S, 256), bsIQ1_S)
	}
	w, err := DecodeGGMLWindow(GGML_IQ1_S, buf, 256, 40, 6)
	if err != nil {
		t.Fatal(err)
	}
	full, _ := DecodeGGMLBlock(GGML_IQ1_S, buf, 256)
	for i := 0; i < 6; i++ {
		if math.Abs(float64(w[i]-full[40+i])) > 1e-4 {
			t.Fatal(w)
		}
	}
}

func makeIQ1MBlockAllOnes(d float32) []byte {
	// qs/qh point to grid 2047; scales: packed d in hi nibbles + per-group scale=0 (dl=d)
	buf := make([]byte, bsIQ1_M)
	qs := buf[0 : QK_K/8]
	qh := buf[QK_K/8 : QK_K/8+QK_K/16]
	scales := buf[QK_K/8+QK_K/16:]
	for i := range qs {
		qs[i] = 0xff // low of 2047
	}
	// hi3 of 2047 is 7; for qh byte layout:
	// idx0: (qh0<<8)&0x700 → needs bits of qh0 that shift into 0x700: bits 0-2 of qh0 after <<8 → bits of qh0
	// (qh0 << 8) & 0x700 = (qh0 & 0x7) << 8  so qh0 low 3 bits = 7
	// idx1: (qh0 << 4) & 0x700 = (qh0 & 0x70) << 4 → qh0 bits 4-6 = 7 → need 0x70
	// delta0 uses bit 3 (0x08), delta1 bit 7 (0x80)
	// so qh0 = 0x07 | 0x70 = 0x77 (no delta flips)
	// same for qh1
	for i := range qh {
		qh[i] = 0x77
	}
	// pack d fp16 into high nibble positions of sc[0..3] as uint16 words
	// scale.u16 = (sc0>>12) | ((sc1>>8)&0xf0) | ((sc2>>4)&0xf00) | (sc3&0xf000)
	dBits := fp16(d)
	// sc words also hold 3-bit scales: two groups per uint16 at shifts 0 and 6 for ib%2
	// we want all 3-bit scales = 0 so dl = d*(0*2+1)=d
	// low 12 bits free for scales (actually used bits 0-2,3-5,6-8,9-11 roughly in pairs)
	// for ib even: bits 0-2 and 3-5 of sc[ib/2]; for ib odd: bits 6-8 and 9-11
	// put dBits parts only in high regions as C expects
	sc := [4]uint16{}
	// reconstruct: sc[0] high 4 bits = dBits bits 0-3
	sc[0] = (dBits & 0x000f) << 12
	sc[1] = (dBits & 0x00f0) << 8
	sc[2] = (dBits & 0x0f00) << 4
	sc[3] = (dBits & 0xf000)
	// leave lower scale fields 0
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint16(scales[2*i:2*i+2], sc[i])
	}
	return buf
}

func TestDequantIQ1_M(t *testing.T) {
	d := float32(2.0)
	buf := makeIQ1MBlockAllOnes(d)
	out, err := DecodeGGMLBlock(GGML_IQ1_M, buf, 256)
	if err != nil {
		t.Fatal(err)
	}
	// recover d from packing
	scales := buf[QK_K/8+QK_K/16:]
	sc0 := binary.LittleEndian.Uint16(scales[0:2])
	sc1 := binary.LittleEndian.Uint16(scales[2:4])
	sc2 := binary.LittleEndian.Uint16(scales[4:6])
	sc3 := binary.LittleEndian.Uint16(scales[6:8])
	dBits := (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000)
	var dbuf [2]byte
	binary.LittleEndian.PutUint16(dbuf[:], dBits)
	dRT := f16bits(dbuf[:], 0)
	want := dRT * (1 + 0.125)
	for i, v := range out {
		if math.Abs(float64(v-want)) > 1e-2 {
			t.Fatalf("[%d]=%v want %v (dRT=%v)", i, v, want, dRT)
		}
	}
	if GGMLRawBytes(GGML_IQ1_M, 256) != bsIQ1_M {
		t.Fatal(GGMLRawBytes(GGML_IQ1_M, 256), bsIQ1_M)
	}
	w, err := DecodeGGMLWindow(GGML_IQ1_M, buf, 256, 40, 6)
	if err != nil {
		t.Fatal(err)
	}
	full, _ := DecodeGGMLBlock(GGML_IQ1_M, buf, 256)
	for i := 0; i < 6; i++ {
		if math.Abs(float64(w[i]-full[40+i])) > 1e-4 {
			t.Fatal(w)
		}
	}
}

func TestSessionIQ1_S(t *testing.T) {
	dir := t.TempDir()
	x := make([]float32, 256)
	for i := range x {
		x[i] = 1
	}
	// use all-ones grid 2047 block
	buf := make([]byte, bsIQ1_S)
	binary.LittleEndian.PutUint16(buf[0:2], fp16(1.0))
	qs := buf[2 : 2+QK_K/8]
	qhB := buf[2+QK_K/8:]
	for i := range qs {
		qs[i] = 0xff
	}
	qh := uint16(7 | (7 << 3) | (7 << 6) | (7 << 9))
	for ib := 0; ib < 8; ib++ {
		binary.LittleEndian.PutUint16(qhB[2*ib:2*ib+2], qh)
	}
	path := filepath.Join(dir, "iq1s.gguf")
	if err := WriteGGUFTyped(path, "W", GGML_IQ1_S, []uint64{256, 1}, buf, nil); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	dRT := f16bits(buf, 0)
	// each weight ≈ 1.125 * d
	y, err := s.LinearFused("W", x, 1, 256, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := float64(256) * float64(dRT) * 1.125
	if math.Abs(float64(y[0])-want) > 2.0 {
		t.Fatalf("y=%v want %v", y[0], want)
	}
}

func TestSessionIQ1_M(t *testing.T) {
	dir := t.TempDir()
	x := make([]float32, 256)
	for i := range x {
		x[i] = 1
	}
	buf := makeIQ1MBlockAllOnes(1.0)
	path := filepath.Join(dir, "iq1m.gguf")
	if err := WriteGGUFTyped(path, "W", GGML_IQ1_M, []uint64{256, 1}, buf, nil); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	scales := buf[QK_K/8+QK_K/16:]
	sc0 := binary.LittleEndian.Uint16(scales[0:2])
	sc1 := binary.LittleEndian.Uint16(scales[2:4])
	sc2 := binary.LittleEndian.Uint16(scales[4:6])
	sc3 := binary.LittleEndian.Uint16(scales[6:8])
	dBits := (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000)
	var dbuf [2]byte
	binary.LittleEndian.PutUint16(dbuf[:], dBits)
	dRT := f16bits(dbuf[:], 0)
	y, err := s.LinearFused("W", x, 1, 256, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := float64(256) * float64(dRT) * 1.125
	if math.Abs(float64(y[0])-want) > 2.0 {
		t.Fatalf("y=%v want %v", y[0], want)
	}
}

// makeMXFP4BlockAll fills one 32-elem MXFP4 block with e + nibble qs.
// qs layout matches dequant: lo nibble → out[j], hi nibble → out[j+16].
func makeMXFP4BlockAll(e uint8, lo, hi byte) []byte {
	buf := make([]byte, bsMXFP4)
	buf[0] = e
	for j := 0; j < QK_MXFP4/2; j++ {
		buf[1+j] = (lo & 0x0F) | ((hi & 0x0F) << 4)
	}
	return buf
}

func TestDequantMXFP4(t *testing.T) {
	// e=128 → d=1.0; fill lo=code7 (12), hi=code15 (-12)
	buf := makeMXFP4BlockAll(128, 7, 15)
	out, err := DecodeGGMLBlock(GGML_MXFP4, buf, 32)
	if err != nil {
		t.Fatal(err)
	}
	d := e8m0ToFP32Half(128)
	if math.Abs(float64(d-1.0)) > 1e-6 {
		t.Fatalf("d=%v", d)
	}
	for j := 0; j < 16; j++ {
		if math.Abs(float64(out[j]-12.0*d)) > 1e-4 {
			t.Fatalf("lo[%d]=%v", j, out[j])
		}
		if math.Abs(float64(out[j+16]-(-12.0)*d)) > 1e-4 {
			t.Fatalf("hi[%d]=%v", j, out[j+16])
		}
	}
	// codebook sweep: qs[j] = j | ((15-j)<<4)
	buf = make([]byte, bsMXFP4)
	buf[0] = 128
	for j := 0; j < 16; j++ {
		buf[1+j] = byte(j) | byte(15-j)<<4
	}
	out, err = DecodeGGMLBlock(GGML_MXFP4, buf, 32)
	if err != nil {
		t.Fatal(err)
	}
	for j := 0; j < 16; j++ {
		wantLo := float32(kvaluesFP4[j]) * d
		wantHi := float32(kvaluesFP4[15-j]) * d
		if math.Abs(float64(out[j]-wantLo)) > 1e-4 || math.Abs(float64(out[j+16]-wantHi)) > 1e-4 {
			t.Fatalf("j=%d out=%v/%v want %v/%v", j, out[j], out[j+16], wantLo, wantHi)
		}
	}
	// e=0 denorm half-scale path still returns finite values
	buf[0] = 0
	out, err = DecodeGGMLBlock(GGML_MXFP4, buf, 32)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(float64(out[7])) || math.IsInf(float64(out[7]), 0) {
		t.Fatal(out[7])
	}
	if GGMLRawBytes(GGML_MXFP4, 32) != bsMXFP4 {
		t.Fatal(GGMLRawBytes(GGML_MXFP4, 32), bsMXFP4)
	}
	if GGMLRawBytes(GGML_MXFP4, 64) != 2*bsMXFP4 {
		t.Fatal(GGMLRawBytes(GGML_MXFP4, 64))
	}
	w, err := DecodeGGMLWindow(GGML_MXFP4, buf, 32, 10, 6)
	if err != nil {
		t.Fatal(err)
	}
	full, _ := DecodeGGMLBlock(GGML_MXFP4, buf, 32)
	for i := 0; i < 6; i++ {
		if math.Abs(float64(w[i]-full[10+i])) > 1e-6 {
			t.Fatal(w, full[10:16])
		}
	}
}

func makeNVFP4BlockConst(scaleUE uint8, lo, hi byte) []byte {
	buf := make([]byte, bsNVFP4)
	for s := 0; s < QK_NVFP4/QK_NVFP4_SUB; s++ {
		buf[s] = scaleUE
	}
	qs := buf[QK_NVFP4/QK_NVFP4_SUB:]
	for i := range qs {
		qs[i] = (lo & 0x0F) | ((hi & 0x0F) << 4)
	}
	return buf
}

func TestDequantNVFP4(t *testing.T) {
	// UE4M3 0x40 → d=1.0; lo=code7 (12), hi=code15 (-12) in each sub-block
	buf := makeNVFP4BlockConst(0x40, 7, 15)
	out, err := DecodeGGMLBlock(GGML_NVFP4, buf, 64)
	if err != nil {
		t.Fatal(err)
	}
	d := ue4m3ToFP32(0x40)
	if math.Abs(float64(d-1.0)) > 1e-5 {
		t.Fatalf("d=%v", d)
	}
	for i := 0; i < 64; i++ {
		sub := i / QK_NVFP4_SUB
		local := i % QK_NVFP4_SUB
		want := 12.0 * d
		if local >= QK_NVFP4_SUB/2 {
			want = -12.0 * d
		}
		if math.Abs(float64(out[i]-want)) > 1e-4 {
			t.Fatalf("[%d] (sub %d loc %d)=%v want %v", i, sub, local, out[i], want)
		}
	}
	// per-sub scale: sub0 d=1 (0x40), sub1 d=0.5 (0x38), sub2 d=2 (0x48), sub3 d=0 (0)
	buf = make([]byte, bsNVFP4)
	buf[0], buf[1], buf[2], buf[3] = 0x40, 0x38, 0x48, 0x00
	qs := buf[4:]
	for i := range qs {
		// code 1 → kvalue 1 in both halves
		qs[i] = 0x11
	}
	out, err = DecodeGGMLBlock(GGML_NVFP4, buf, 64)
	if err != nil {
		t.Fatal(err)
	}
	scales := []float32{ue4m3ToFP32(0x40), ue4m3ToFP32(0x38), ue4m3ToFP32(0x48), ue4m3ToFP32(0)}
	for s := 0; s < 4; s++ {
		for j := 0; j < QK_NVFP4_SUB; j++ {
			want := scales[s] * 1.0
			if math.Abs(float64(out[s*QK_NVFP4_SUB+j]-want)) > 1e-4 {
				t.Fatalf("s=%d j=%d got %v want %v", s, j, out[s*QK_NVFP4_SUB+j], want)
			}
		}
	}
	if GGMLRawBytes(GGML_NVFP4, 64) != bsNVFP4 {
		t.Fatal(GGMLRawBytes(GGML_NVFP4, 64), bsNVFP4)
	}
	w, err := DecodeGGMLWindow(GGML_NVFP4, buf, 64, 20, 8)
	if err != nil {
		t.Fatal(err)
	}
	full, _ := DecodeGGMLBlock(GGML_NVFP4, buf, 64)
	for i := 0; i < 8; i++ {
		if math.Abs(float64(w[i]-full[20+i])) > 1e-6 {
			t.Fatal(w)
		}
	}
}

func TestSessionMXFP4(t *testing.T) {
	dir := t.TempDir()
	// 256 elems = 8 MXFP4 blocks; constant code7, e=128 → weight 12
	n := 256
	raw := make([]byte, 0, (n/QK_MXFP4)*bsMXFP4)
	for b := 0; b < n/QK_MXFP4; b++ {
		raw = append(raw, makeMXFP4BlockAll(128, 7, 7)...)
	}
	path := filepath.Join(dir, "mxfp4.gguf")
	if err := WriteGGUFTyped(path, "W", GGML_MXFP4, []uint64{uint64(n), 1}, raw, nil); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	x := make([]float32, n)
	for i := range x {
		x[i] = 1
	}
	y, err := s.LinearFused("W", x, 1, n, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := float64(n) * 12.0
	if math.Abs(float64(y[0])-want) > 1e-2 {
		t.Fatalf("y=%v want %v", y[0], want)
	}
	// tensor load path
	w, _, err := s.LoadF32("W")
	if err != nil {
		t.Fatal(err)
	}
	if len(w) != n || math.Abs(float64(w[0]-12)) > 1e-4 {
		t.Fatal(w[0], len(w))
	}
}

func TestSessionNVFP4(t *testing.T) {
	dir := t.TempDir()
	// 256 elems = 4 NVFP4 blocks; UE 0x40, code7 → weight 12
	n := 256
	raw := make([]byte, 0, (n/QK_NVFP4)*bsNVFP4)
	for b := 0; b < n/QK_NVFP4; b++ {
		raw = append(raw, makeNVFP4BlockConst(0x40, 7, 7)...)
	}
	path := filepath.Join(dir, "nvfp4.gguf")
	if err := WriteGGUFTyped(path, "W", GGML_NVFP4, []uint64{uint64(n), 1}, raw, nil); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	x := make([]float32, n)
	for i := range x {
		x[i] = 1
	}
	y, err := s.LinearFused("W", x, 1, n, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := float64(n) * 12.0
	if math.Abs(float64(y[0])-want) > 1e-2 {
		t.Fatalf("y=%v want %v", y[0], want)
	}
}
