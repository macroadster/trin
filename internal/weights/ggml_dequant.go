package weights

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Block sizes (ggml-common.h).
const (
	QK4_0  = 32
	QK4_1  = 32
	QK5_0  = 32
	QK5_1  = 32
	QK8_0  = 32
	QK8_1  = 32
	QK4_NL = 32
	QK_K   = 256
)

// Extra GGML type codes (additive to existing F32/F16/Q8_0/I*/BF16).
const (
	GGML_Q4_0 = 2
	GGML_Q4_1 = 3
	GGML_Q5_0 = 6
	GGML_Q5_1 = 7
	GGML_Q8_1 = 9
	GGML_Q2_K = 10
	GGML_Q3_K = 11
	GGML_Q4_K = 12
	GGML_Q5_K = 13
	GGML_Q6_K = 14
	GGML_Q8_K = 15
	// IQ* / TQ* (ggml type ids; v0.93 ships IQ4_NL + TQ2_0 dequant)
	GGML_IQ2_XXS = 16
	GGML_IQ2_XS  = 17
	GGML_IQ3_XXS = 18
	GGML_IQ1_S   = 19
	GGML_IQ4_NL  = 20
	GGML_IQ3_S   = 21
	GGML_IQ2_S   = 22
	GGML_IQ4_XS  = 23
	// I8..BF16 already defined in gguf.go (24–30)
	GGML_IQ1_M   = 29
	GGML_TQ1_0   = 34
	GGML_TQ2_0   = 35
	// MXFP*/NVFP* OCP microscaling FP4 (ggml type ids)
	GGML_MXFP4   = 39
	GGML_NVFP4   = 40
)

// Block byte sizes.
const (
	bsQ4_0 = 2 + QK4_0/2           // 18
	bsQ4_1 = 2 + 2 + QK4_1/2       // 20
	bsQ5_0 = 2 + 4 + QK5_0/2       // 22
	bsQ5_1 = 2 + 2 + 4 + QK5_1/2   // 24
	bsQ8_0 = 2 + QK8_0             // 34
	bsQ8_1 = 2 + 2 + QK8_1         // 36
	bsQ2_K = 2*2 + QK_K/16 + QK_K/4 // 84
	bsQ3_K = 2 + QK_K/4 + QK_K/8 + 12 // 110
	bsQ4_K = 2*2 + 12 + QK_K/2     // 144
	bsQ5_K = 2*2 + 12 + QK_K/2 + QK_K/8 // 176
	bsQ6_K = 2 + QK_K/16 + 3*QK_K/4 // 210
	bsQ8_K = 4 + QK_K + (QK_K/16)*2 // 292
	// IQ4_NL: half d + 16 nibble bytes (same footprint as Q4_0)
	bsIQ4_NL = 2 + QK4_NL/2 // 18
	// IQ2_XXS: half d + qs[QK_K/8] as uint16
	bsIQ2_XXS = 2 + (QK_K/8)*2 // 66
	// IQ2_XS: half d + qs[QK_K/8]uint16 + scales[QK_K/32]
	bsIQ2_XS = 2 + (QK_K/8)*2 + QK_K/32 // 74
	// IQ2_S: half d + qs[QK_K/4] + qh[QK_K/32] + scales[QK_K/32]
	bsIQ2_S = 2 + QK_K/4 + QK_K/32 + QK_K/32 // 82
	// IQ3_XXS: half d + qs[3*QK_K/8]
	bsIQ3_XXS = 2 + 3*(QK_K/8) // 98
	// IQ3_S: half d + qs[QK_K/4] + qh[QK_K/32] + signs[QK_K/8] + scales[QK_K/64]
	bsIQ3_S = 2 + QK_K/4 + QK_K/32 + QK_K/8 + QK_K/64 // 110
	// IQ1_S: half d + qs[QK_K/8] + qh[QK_K/32]uint16
	bsIQ1_S = 2 + QK_K/8 + (QK_K/32)*2 // 50
	// IQ1_M: qs[QK_K/8] + qh[QK_K/16] + scales[QK_K/32] (d packed in scale hi bits)
	bsIQ1_M = QK_K/8 + QK_K/16 + QK_K/32 // 56
	// IQ4_XS: half d + scales_h + scales_l[4] + qs[128]
	bsIQ4_XS = 2 + 2 + QK_K/64 + QK_K/2 // 136
	// TQ1_0: qs[48] + qh[4] + half d
	bsTQ1_0 = (QK_K-4*(QK_K/64))/5 + QK_K/64 + 2 // 54
	// TQ2_0: 64 qs bytes (2b/elem) + half d
	bsTQ2_0 = QK_K/4 + 2 // 66
	// MXFP4: E8M0 scale byte + 16 nibble qs (32 elems)
	QK_MXFP4 = 32
	bsMXFP4  = 1 + QK_MXFP4/2 // 17
	// NVFP4: 4×UE4M3 scales + 32 nibble qs (64 elems, 4×16 sub-blocks)
	QK_NVFP4     = 64
	QK_NVFP4_SUB = 16
	bsNVFP4      = QK_NVFP4/QK_NVFP4_SUB + QK_NVFP4/2 // 36
)

func f16bits(raw []byte, off int) float32 {
	return float16ToFloat32(binary.LittleEndian.Uint16(raw[off:]))
}

// dequantBlockQ4_0: block = half d + 16 nibble bytes; y = (q-8)*d
func dequantBlockQ4_0(blk []byte, out []float32) {
	d := f16bits(blk, 0)
	qs := blk[2:]
	for j := 0; j < QK4_0/2; j++ {
		x0 := int(qs[j]&0x0F) - 8
		x1 := int(qs[j]>>4) - 8
		out[j] = float32(x0) * d
		out[j+QK4_0/2] = float32(x1) * d
	}
}

// dequantBlockQ4_1: half d, half m, 16 nibbles; y = q*d + m
func dequantBlockQ4_1(blk []byte, out []float32) {
	d := f16bits(blk, 0)
	m := f16bits(blk, 2)
	qs := blk[4:]
	for j := 0; j < QK4_1/2; j++ {
		x0 := int(qs[j] & 0x0F)
		x1 := int(qs[j] >> 4)
		out[j] = float32(x0)*d + m
		out[j+QK4_1/2] = float32(x1)*d + m
	}
}

// dequantBlockQ5_0: half d, 4-byte qh, 16 nibbles
func dequantBlockQ5_0(blk []byte, out []float32) {
	d := f16bits(blk, 0)
	qh := binary.LittleEndian.Uint32(blk[2:6])
	qs := blk[6:]
	for j := 0; j < QK5_0/2; j++ {
		xh0 := uint8(((qh >> uint(j+0)) << 4) & 0x10)
		xh1 := uint8((qh >> uint(j+12)) & 0x10)
		x0 := int((qs[j]&0x0F)|xh0) - 16
		x1 := int((qs[j]>>4)|xh1) - 16
		out[j] = float32(x0) * d
		out[j+QK5_0/2] = float32(x1) * d
	}
}

// dequantBlockQ5_1: half d, half m, 4-byte qh, 16 nibbles
func dequantBlockQ5_1(blk []byte, out []float32) {
	d := f16bits(blk, 0)
	m := f16bits(blk, 2)
	qh := binary.LittleEndian.Uint32(blk[4:8])
	qs := blk[8:]
	for j := 0; j < QK5_1/2; j++ {
		xh0 := uint8(((qh >> uint(j+0)) << 4) & 0x10)
		xh1 := uint8((qh >> uint(j+12)) & 0x10)
		x0 := int((qs[j] & 0x0F) | xh0)
		x1 := int((qs[j] >> 4) | xh1)
		out[j] = float32(x0)*d + m
		out[j+QK5_1/2] = float32(x1)*d + m
	}
}

func dequantBlockQ8_0(blk []byte, out []float32) {
	d := f16bits(blk, 0)
	qs := blk[2:]
	for j := 0; j < QK8_0; j++ {
		out[j] = float32(int8(qs[j])) * d
	}
}

func dequantBlockQ8_1(blk []byte, out []float32) {
	// d,s halves then int8 qs; dequant uses d*qs only (s is for dots)
	d := f16bits(blk, 0)
	qs := blk[4:]
	for j := 0; j < QK8_1; j++ {
		out[j] = float32(int8(qs[j])) * d
	}
}

// ---- K-quants (super-block 256) ----

func getScaleMinK4(j int, q []byte) (d, m uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
	m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4)
	return d, m
}

// block_q2_K layout: scales[16] | qs[64] | d,dmin halves  — wait:
// From ggml-common: scales[QK_K/16]=16, qs[QK_K/4]=64, then d,dmin
// sizeof = 2*half + 16 + 64 = 84. Order in struct: scales, qs, dm.
func dequantBlockQ2_K(blk []byte, out []float32) {
	scales := blk[0:16]
	q := blk[16 : 16+64]
	d := f16bits(blk, 16+64)
	minv := f16bits(blk, 16+64+2)
	y := 0
	is := 0
	qi := 0
	for n := 0; n < QK_K; n += 128 {
		shift := 0
		for j := 0; j < 4; j++ {
			sc := scales[is]
			is++
			dl := d * float32(sc&0xF)
			ml := minv * float32(sc>>4)
			for l := 0; l < 16; l++ {
				out[y] = dl*float32((q[qi+l]>>shift)&3) - ml
				y++
			}
			sc = scales[is]
			is++
			dl = d * float32(sc&0xF)
			ml = minv * float32(sc>>4)
			for l := 0; l < 16; l++ {
				out[y] = dl*float32((q[qi+16+l]>>shift)&3) - ml
				y++
			}
			shift += 2
		}
		qi += 32
	}
}

// block_q3_K: hmask[32] | qs[64] | scales[12] | d
func dequantBlockQ3_K(blk []byte, out []float32) {
	hm := blk[0:32]
	q := blk[32 : 32+64]
	scRaw := blk[32+64 : 32+64+12]
	dAll := f16bits(blk, 32+64+12)

	// unpack 6-bit scales into 16 int8 values (mirrors C)
	var aux [16]byte
	// memcpy first 12 then rearrange per dequantize_row_q3_K
	var a0, a1, a2, a3 uint32
	a0 = binary.LittleEndian.Uint32(scRaw[0:4])
	a1 = binary.LittleEndian.Uint32(scRaw[4:8])
	a2 = binary.LittleEndian.Uint32(scRaw[8:12])
	tmp := a2
	const kmask1 = 0x03030303
	const kmask2 = 0x0f0f0f0f
	a2 = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
	a3 = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
	a0 = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4)
	a1 = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4)
	binary.LittleEndian.PutUint32(aux[0:4], a0)
	binary.LittleEndian.PutUint32(aux[4:8], a1)
	binary.LittleEndian.PutUint32(aux[8:12], a2)
	binary.LittleEndian.PutUint32(aux[12:16], a3)
	scales := make([]int8, 16)
	for i := 0; i < 16; i++ {
		scales[i] = int8(aux[i])
	}

	y := 0
	qi := 0
	m := uint8(1)
	is := 0
	for n := 0; n < QK_K; n += 128 {
		shift := 0
		for j := 0; j < 4; j++ {
			dl := dAll * float32(int(scales[is])-32)
			is++
			for l := 0; l < 16; l++ {
				sub := int8(0)
				if hm[l]&m == 0 {
					sub = 4
				}
				out[y] = dl * float32(int8((q[qi+l]>>shift)&3)-sub)
				y++
			}
			dl = dAll * float32(int(scales[is])-32)
			is++
			for l := 0; l < 16; l++ {
				sub := int8(0)
				if hm[16+l]&m == 0 {
					sub = 4
				}
				out[y] = dl * float32(int8((q[qi+16+l]>>shift)&3)-sub)
				y++
			}
			shift += 2
			m <<= 1
		}
		qi += 32
	}
}

// block_q4_K: d,dmin | scales[12] | qs[128]
func dequantBlockQ4_K(blk []byte, out []float32) {
	d := f16bits(blk, 0)
	minv := f16bits(blk, 2)
	scales := blk[4 : 4+12]
	q := blk[4+12:]
	y := 0
	is := 0
	for j := 0; j < QK_K; j += 64 {
		sc, m := getScaleMinK4(is+0, scales)
		d1 := d * float32(sc)
		m1 := minv * float32(m)
		sc, m = getScaleMinK4(is+1, scales)
		d2 := d * float32(sc)
		m2 := minv * float32(m)
		for l := 0; l < 32; l++ {
			out[y] = d1*float32(q[l]&0xF) - m1
			y++
		}
		for l := 0; l < 32; l++ {
			out[y] = d2*float32(q[l]>>4) - m2
			y++
		}
		q = q[32:]
		is += 2
	}
}

// block_q5_K: d,dmin | scales[12] | qh[32] | qs[128]
func dequantBlockQ5_K(blk []byte, out []float32) {
	d := f16bits(blk, 0)
	minv := f16bits(blk, 2)
	scales := blk[4 : 4+12]
	qh := blk[4+12 : 4+12+32]
	ql := blk[4+12+32:]
	y := 0
	is := 0
	u1, u2 := uint8(1), uint8(2)
	for j := 0; j < QK_K; j += 64 {
		sc, m := getScaleMinK4(is+0, scales)
		d1 := d * float32(sc)
		m1 := minv * float32(m)
		sc, m = getScaleMinK4(is+1, scales)
		d2 := d * float32(sc)
		m2 := minv * float32(m)
		for l := 0; l < 32; l++ {
			hi := 0
			if qh[l]&u1 != 0 {
				hi = 16
			}
			out[y] = d1*float32(int(ql[l]&0xF)+hi) - m1
			y++
		}
		for l := 0; l < 32; l++ {
			hi := 0
			if qh[l]&u2 != 0 {
				hi = 16
			}
			out[y] = d2*float32(int(ql[l]>>4)+hi) - m2
			y++
		}
		ql = ql[32:]
		is += 2
		u1 <<= 2
		u2 <<= 2
	}
}

// block_q6_K: ql[128] | qh[64] | scales[16] int8 | d
func dequantBlockQ6_K(blk []byte, out []float32) {
	ql := blk[0:128]
	qh := blk[128 : 128+64]
	sc := blk[128+64 : 128+64+16] // int8
	d := f16bits(blk, 128+64+16)
	y := 0
	qli, qhi, sci := 0, 0, 0
	for n := 0; n < QK_K; n += 128 {
		for l := 0; l < 32; l++ {
			is := l / 16
			q1 := int8((ql[qli+l]&0xF)|(((qh[qhi+l]>>0)&3)<<4)) - 32
			q2 := int8((ql[qli+32+l]&0xF)|(((qh[qhi+l]>>2)&3)<<4)) - 32
			q3 := int8((ql[qli+l]>>4)|(((qh[qhi+l]>>4)&3)<<4)) - 32
			q4 := int8((ql[qli+32+l]>>4)|(((qh[qhi+l]>>6)&3)<<4)) - 32
			s0 := float32(int8(sc[sci+is+0]))
			s2 := float32(int8(sc[sci+is+2]))
			s4 := float32(int8(sc[sci+is+4]))
			s6 := float32(int8(sc[sci+is+6]))
			out[y+l+0] = d * s0 * float32(q1)
			out[y+l+32] = d * s2 * float32(q2)
			out[y+l+64] = d * s4 * float32(q3)
			out[y+l+96] = d * s6 * float32(q4)
		}
		y += 128
		qli += 64
		qhi += 32
		sci += 8
	}
}

// block_q8_K: float d | qs[256] int8 | bsums[16] int16 (ignored for dequant)
func dequantBlockQ8_K(blk []byte, out []float32) {
	d := math.Float32frombits(binary.LittleEndian.Uint32(blk[0:4]))
	qs := blk[4 : 4+256]
	for j := 0; j < QK_K; j++ {
		out[j] = d * float32(int8(qs[j]))
	}
}

// dequantBlocks runs a fixed block dequantizer over raw tensor bytes.

// kvalues_iq4nl from ggml-common.h (non-linear 4-bit codebook).
var kvaluesIQ4NL = [16]int8{
	-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
}

// dequantBlockIQ4_NL: half d + 16 nibble bytes; y = d * kvalues[q]
func dequantBlockIQ4_NL(blk []byte, out []float32) {
	d := f16bits(blk, 0)
	qs := blk[2:]
	for j := 0; j < QK4_NL/2; j++ {
		out[j] = d * float32(kvaluesIQ4NL[qs[j]&0x0F])
		out[j+QK4_NL/2] = d * float32(kvaluesIQ4NL[qs[j]>>4])
	}
}

// dequantBlockTQ2_0: qs[64] (2 bits/elem packed), half d at end.
// Layout matches llama.cpp dequantize_row_tq2_0: for each 32-byte qs chunk,
// emit 4×32 ternary values in bitplane order.
func dequantBlockTQ2_0(blk []byte, out []float32) {
	// blk layout: qs[0:64], d at 64
	d := f16bits(blk, QK_K/4)
	qs := blk[:QK_K/4]
	oi := 0
	for j := 0; j < len(qs); j += 32 {
		for l := 0; l < 4; l++ {
			shift := uint(l * 2)
			for m := 0; m < 32; m++ {
				q := int8((qs[j+m] >> shift) & 3)
				out[oi] = float32(q-1) * d
				oi++
			}
		}
	}
}


// dequantBlockIQ4_XS: d, scales_h, scales_l[4], qs[128]
// 8 sub-blocks of 32; scale ls in 0..63 with dl = d*(ls-32)
func dequantBlockIQ4_XS(blk []byte, out []float32) {
	d := f16bits(blk, 0)
	scalesH := binary.LittleEndian.Uint16(blk[2:4])
	scalesL := blk[4 : 4+QK_K/64]
	qs := blk[4+QK_K/64:]
	oi := 0
	for ib := 0; ib < QK_K/32; ib++ {
		ls := int((scalesL[ib/2] >> (4 * (ib % 2))) & 0x0F)
		ls |= int((scalesH >> (2 * uint(ib))) & 3) << 4
		dl := d * float32(ls-32)
		for j := 0; j < 16; j++ {
			out[oi+j] = dl * float32(kvaluesIQ4NL[qs[j]&0x0F])
			out[oi+j+16] = dl * float32(kvaluesIQ4NL[qs[j]>>4])
		}
		oi += 32
		qs = qs[16:]
	}
}

// dequantBlockTQ1_0: qs[48] + qh[4] + half d  (llama.cpp dequantize_row_tq1_0)
// Intermediate multiply is truncated to uint8, then (uint16(q)*3)>>8 extracts the trit.
func dequantBlockTQ1_0(blk []byte, out []float32) {
	const qsN = (QK_K - 4*(QK_K/64)) / 5 // 48
	const qhN = QK_K / 64                 // 4
	qs := blk[:qsN]
	qh := blk[qsN : qsN+qhN]
	d := f16bits(blk, qsN+qhN)
	pow3 := [6]uint8{1, 3, 9, 27, 81, 243}
	oi := 0
	j := 0
	for ; j+32 <= qsN; j += 32 {
		for n := 0; n < 5; n++ {
			p := pow3[n]
			for m := 0; m < 32; m++ {
				q := uint8(qs[j+m] * p) // match C uint8 truncate
				xi := int16((uint16(q) * 3) >> 8)
				out[oi] = float32(xi-1) * d
				oi++
			}
		}
	}
	for ; j < qsN; j += 16 {
		for n := 0; n < 5; n++ {
			p := pow3[n]
			for m := 0; m < 16; m++ {
				q := uint8(qs[j+m] * p)
				xi := int16((uint16(q) * 3) >> 8)
				out[oi] = float32(xi-1) * d
				oi++
			}
		}
	}
	for n := 0; n < 4; n++ {
		p := pow3[n]
		for j := 0; j < qhN; j++ {
			q := uint8(qh[j] * p)
			xi := int16((uint16(q) * 3) >> 8)
			out[oi] = float32(xi-1) * d
			oi++
		}
	}
}



// dequantBlockIQ2_XXS: d + qs[32]uint16 (as 64 bytes little-endian)
// 8 groups of 32 elems; each group packs 2×uint32 in qs[4*ib32 ..]
func dequantBlockIQ2_XXS(blk []byte, out []float32) {
	d := f16bits(blk, 0)
	qs := blk[2:]
	oi := 0
	for ib32 := 0; ib32 < QK_K/32; ib32++ {
		// 8 bytes = 2×uint32
		off := 8 * ib32
		aux0 := binary.LittleEndian.Uint32(qs[off : off+4])
		aux1 := binary.LittleEndian.Uint32(qs[off+4 : off+8])
		db := d * (0.5 + float32(aux1>>28)) * 0.25
		// treat aux0||low28 of aux1 as 8 grid indices + signs
		auxBytes := [8]byte{
			byte(aux0), byte(aux0 >> 8), byte(aux0 >> 16), byte(aux0 >> 24),
			byte(aux1), byte(aux1 >> 8), byte(aux1 >> 16), byte(aux1 >> 24),
		}
		for l := 0; l < 4; l++ {
			g := iq2xxsGrid[auxBytes[l]]
			signs := ksignsIQ2xs[(aux1>>(7*uint(l)))&127]
			for j := 0; j < 8; j++ {
				gv := float32((g >> (8 * uint(j))) & 0xFF)
				sign := float32(1)
				if signs&kmaskIQ2xs[j] != 0 {
					sign = -1
				}
				out[oi] = db * gv * sign
				oi++
			}
		}
	}
}

// dequantBlockIQ3_XXS: d + qs[3*QK_K/8]=96 bytes
// qs[0:64] grid indices for 8×ib32×8, then scales_and_signs uint32[8]
func dequantBlockIQ3_XXS(blk []byte, out []float32) {
	d := f16bits(blk, 0)
	qsAll := blk[2:]
	qs := qsAll[:QK_K/4]                 // 64 bytes of grid indices
	scalesAndSigns := qsAll[QK_K/4:]     // 32 bytes = 8×uint32
	oi := 0
	qsi := 0
	for ib32 := 0; ib32 < QK_K/32; ib32++ {
		aux32 := binary.LittleEndian.Uint32(scalesAndSigns[4*ib32 : 4*ib32+4])
		db := d * (0.5 + float32(aux32>>28)) * 0.5
		for l := 0; l < 4; l++ {
			signs := ksignsIQ2xs[(aux32>>(7*uint(l)))&127]
			g1 := iq3xxsGrid[qs[qsi+2*l+0]]
			g2 := iq3xxsGrid[qs[qsi+2*l+1]]
			for j := 0; j < 4; j++ {
				v1 := float32((g1 >> (8 * uint(j))) & 0xFF)
				v2 := float32((g2 >> (8 * uint(j))) & 0xFF)
				s1 := float32(1)
				if signs&kmaskIQ2xs[j] != 0 {
					s1 = -1
				}
				s2 := float32(1)
				if signs&kmaskIQ2xs[j+4] != 0 {
					s2 = -1
				}
				out[oi+j] = db * v1 * s1
				out[oi+j+4] = db * v2 * s2
			}
			oi += 8
		}
		qsi += 8
	}
}



// dequantBlockIQ2_XS: d + qs[32]uint16 + scales[8]
// qs word: low 9 bits = grid idx, high 7 = sign table idx
func dequantBlockIQ2_XS(blk []byte, out []float32) {
	d := f16bits(blk, 0)
	qs := blk[2 : 2+(QK_K/8)*2]
	scales := blk[2+(QK_K/8)*2:]
	oi := 0
	for ib32 := 0; ib32 < QK_K/32; ib32++ {
		sc := scales[ib32]
		db0 := d * (0.5 + float32(sc&0xf)) * 0.25
		db1 := d * (0.5 + float32(sc>>4)) * 0.25
		for l := 0; l < 4; l++ {
			q := binary.LittleEndian.Uint16(qs[2*(4*ib32+l) : 2*(4*ib32+l)+2])
			g := iq2xsGrid[q&511]
			signs := ksignsIQ2xs[q>>9]
			db := db0
			if l >= 2 {
				db = db1
			}
			for j := 0; j < 8; j++ {
				gv := float32((g >> (8 * uint(j))) & 0xFF)
				sign := float32(1)
				if signs&kmaskIQ2xs[j] != 0 {
					sign = -1
				}
				out[oi] = db * gv * sign
				oi++
			}
		}
	}
}

// dequantBlockIQ2_S: d + qs[64] + qh[8] + scales[8]
// qs[0:32] low grid bytes; qs[32:64] signs; qh high 2 bits per group
func dequantBlockIQ2_S(blk []byte, out []float32) {
	d := f16bits(blk, 0)
	qsAll := blk[2 : 2+QK_K/4]
	qh := blk[2+QK_K/4 : 2+QK_K/4+QK_K/32]
	scales := blk[2+QK_K/4+QK_K/32:]
	qs := qsAll[:QK_K/8]
	signs := qsAll[QK_K/8:]
	oi := 0
	qsi := 0
	sgi := 0
	for ib32 := 0; ib32 < QK_K/32; ib32++ {
		sc := scales[ib32]
		db0 := d * (0.5 + float32(sc&0xf)) * 0.25
		db1 := d * (0.5 + float32(sc>>4)) * 0.25
		for l := 0; l < 4; l++ {
			db := db0
			if l >= 2 {
				db = db1
			}
			idx := uint16(qs[qsi+l]) | ((uint16(qh[ib32])<<(8-2*uint(l)))&0x300)
			g := iq2sGrid[idx]
			sg := signs[sgi+l]
			for j := 0; j < 8; j++ {
				gv := float32((g >> (8 * uint(j))) & 0xFF)
				sign := float32(1)
				if sg&kmaskIQ2xs[j] != 0 {
					sign = -1
				}
				out[oi] = db * gv * sign
				oi++
			}
		}
		qsi += 4
		sgi += 4
	}
}



// dequantBlockIQ3_S: d + qs[64] + qh[8] + signs[32] + scales[4]
// processes 64-weight superblocks (two 32-weight halves) with shared scale nibble pairs.
func dequantBlockIQ3_S(blk []byte, out []float32) {
	d := f16bits(blk, 0)
	qs := blk[2 : 2+QK_K/4]
	qh := blk[2+QK_K/4 : 2+QK_K/4+QK_K/32]
	signs := blk[2+QK_K/4+QK_K/32 : 2+QK_K/4+QK_K/32+QK_K/8]
	scales := blk[2+QK_K/4+QK_K/32+QK_K/8:]
	oi := 0
	qsi := 0
	sgi := 0
	qhi := 0
	// ib32 steps of 2: first 32 use scale lo, next 32 use scale hi; qh advances by 2
	for ib64 := 0; ib64 < QK_K/64; ib64++ {
		sc := scales[ib64]
		db1 := d * (1 + 2*float32(sc&0xf))
		db2 := d * (1 + 2*float32(sc>>4))
		// first 32 weights (qh[qhi])
		for l := 0; l < 4; l++ {
			idx1 := uint16(qs[qsi+2*l+0]) | ((uint16(qh[qhi])<<(8-2*uint(l)))&256)
			idx2 := uint16(qs[qsi+2*l+1]) | ((uint16(qh[qhi])<<(7-2*uint(l)))&256)
			g1 := iq3sGrid[idx1]
			g2 := iq3sGrid[idx2]
			sg := signs[sgi+l]
			for j := 0; j < 4; j++ {
				v1 := float32((g1 >> (8 * uint(j))) & 0xFF)
				v2 := float32((g2 >> (8 * uint(j))) & 0xFF)
				s1 := float32(1)
				if sg&kmaskIQ2xs[j] != 0 {
					s1 = -1
				}
				s2 := float32(1)
				if sg&kmaskIQ2xs[j+4] != 0 {
					s2 = -1
				}
				out[oi+j] = db1 * v1 * s1
				out[oi+j+4] = db1 * v2 * s2
			}
			oi += 8
		}
		qsi += 8
		sgi += 4
		// second 32 weights (qh[qhi+1])
		for l := 0; l < 4; l++ {
			idx1 := uint16(qs[qsi+2*l+0]) | ((uint16(qh[qhi+1])<<(8-2*uint(l)))&256)
			idx2 := uint16(qs[qsi+2*l+1]) | ((uint16(qh[qhi+1])<<(7-2*uint(l)))&256)
			g1 := iq3sGrid[idx1]
			g2 := iq3sGrid[idx2]
			sg := signs[sgi+l]
			for j := 0; j < 4; j++ {
				v1 := float32((g1 >> (8 * uint(j))) & 0xFF)
				v2 := float32((g2 >> (8 * uint(j))) & 0xFF)
				s1 := float32(1)
				if sg&kmaskIQ2xs[j] != 0 {
					s1 = -1
				}
				s2 := float32(1)
				if sg&kmaskIQ2xs[j+4] != 0 {
					s2 = -1
				}
				out[oi+j] = db2 * v1 * s1
				out[oi+j+4] = db2 * v2 * s2
			}
			oi += 8
		}
		qsi += 8
		sgi += 4
		qhi += 2
	}
}



// dequantBlockIQ1_S: d + qs[32] + qh[8]uint16
// 32-weight groups; qh holds 3-bit grid hi + scale (bits12-14) + delta sign (bit15).
func dequantBlockIQ1_S(blk []byte, out []float32) {
	d := f16bits(blk, 0)
	qs := blk[2 : 2+QK_K/8]
	qhBytes := blk[2+QK_K/8:]
	oi := 0
	qsi := 0
	for ib := 0; ib < QK_K/32; ib++ {
		qh := binary.LittleEndian.Uint16(qhBytes[2*ib : 2*ib+2])
		dl := d * float32(2*int((qh>>12)&7)+1)
		delta := iq1sDelta
		if qh&0x8000 != 0 {
			delta = -iq1sDelta
		}
		for l := 0; l < 4; l++ {
			idx := uint16(qs[qsi+l]) | (uint16((qh>>(3*uint(l)))&7) << 8)
			g := iq1sGrid[idx]
			for j := 0; j < 8; j++ {
				// each packed byte is an int8 grid value
				gv := int8(byte(g >> (8 * uint(j))))
				out[oi+j] = dl * (float32(gv) + delta)
			}
			oi += 8
		}
		qsi += 4
	}
}

// kvalues_fp4 / kvalues_mxfp4 from ggml-common.h (E2M1 values doubled).
// Shared by MXFP4 and NVFP4; dequant multiplies by E8M0_half / UE4M3_half scale.
var kvaluesFP4 = [16]int8{
	0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
}

// e8m0ToFP32Half: 0.5 * 2^(e-127) = 2^(e-128); matches GGML_E8M0_TO_FP32_HALF.
func e8m0ToFP32Half(e uint8) float32 {
	var bits uint32
	if e < 2 {
		// 0x00200000 = 2^-128, 0x00400000 = 2^-127
		bits = uint32(0x00200000) << e
	} else {
		bits = uint32(e-1) << 23
	}
	return math.Float32frombits(bits)
}

// ue4m3ToFP32: unsigned E4M3 → fp32, then ×0.5 so it pairs with doubled kvalues_fp4.
func ue4m3ToFP32(x uint8) float32 {
	if x == 0 || x == 0x7F {
		return 0
	}
	exp := int((x >> 3) & 0xF)
	man := int(x & 0x7)
	var raw float64
	if exp == 0 {
		raw = math.Ldexp(float64(man), -9)
	} else {
		raw = math.Ldexp(1.0+float64(man)/8.0, exp-7)
	}
	return float32(raw * 0.5)
}

// dequantBlockMXFP4: e (E8M0) + qs[16]; y = kvalues_fp4[q] * e8m0_half(e)
func dequantBlockMXFP4(blk []byte, out []float32) {
	d := e8m0ToFP32Half(blk[0])
	qs := blk[1:]
	for j := 0; j < QK_MXFP4/2; j++ {
		out[j] = float32(kvaluesFP4[qs[j]&0x0F]) * d
		out[j+QK_MXFP4/2] = float32(kvaluesFP4[qs[j]>>4]) * d
	}
}

// dequantBlockNVFP4: d[4] UE4M3 + qs[32]; 4 sub-blocks of 16 elems.
func dequantBlockNVFP4(blk []byte, out []float32) {
	scales := blk[0 : QK_NVFP4/QK_NVFP4_SUB]
	qs := blk[QK_NVFP4/QK_NVFP4_SUB:]
	nSub := QK_NVFP4 / QK_NVFP4_SUB
	for s := 0; s < nSub; s++ {
		d := ue4m3ToFP32(scales[s])
		base := s * QK_NVFP4_SUB
		qbase := s * (QK_NVFP4_SUB / 2)
		for j := 0; j < QK_NVFP4_SUB/2; j++ {
			out[base+j] = float32(kvaluesFP4[qs[qbase+j]&0x0F]) * d
			out[base+j+QK_NVFP4_SUB/2] = float32(kvaluesFP4[qs[qbase+j]>>4]) * d
		}
	}
}

// dequantBlockIQ1_M: qs[32] + qh[16] + scales[8]; global d packed in scale high nibbles
func dequantBlockIQ1_M(blk []byte, out []float32) {
	qs := blk[0 : QK_K/8]
	qh := blk[QK_K/8 : QK_K/8+QK_K/16]
	scales := blk[QK_K/8+QK_K/16:]
	// reconstruct packed fp16 d from high bits of the four uint16 scale words
	sc0 := binary.LittleEndian.Uint16(scales[0:2])
	sc1 := binary.LittleEndian.Uint16(scales[2:4])
	sc2 := binary.LittleEndian.Uint16(scales[4:6])
	sc3 := binary.LittleEndian.Uint16(scales[6:8])
	dBits := (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000)
	// fp16 bits in little-endian order via f16bits helper on a temp
	var dbuf [2]byte
	binary.LittleEndian.PutUint16(dbuf[:], dBits)
	d := f16bits(dbuf[:], 0)
	oi := 0
	qsi := 0
	qhi := 0
	for ib := 0; ib < QK_K/32; ib++ {
		sc := binary.LittleEndian.Uint16(scales[(ib/2)*2 : (ib/2)*2+2])
		shiftBase := 6 * (ib % 2)
		dl1 := d * float32(2*int((sc>>uint(shiftBase+0))&0x7)+1)
		dl2 := d * float32(2*int((sc>>uint(shiftBase+3))&0x7)+1)
		qh0 := qh[qhi]
		qh1 := qh[qhi+1]
		idx0 := uint16(qs[qsi+0]) | ((uint16(qh0) << 8) & 0x700)
		idx1 := uint16(qs[qsi+1]) | ((uint16(qh0) << 4) & 0x700)
		idx2 := uint16(qs[qsi+2]) | ((uint16(qh1) << 8) & 0x700)
		idx3 := uint16(qs[qsi+3]) | ((uint16(qh1) << 4) & 0x700)
		deltas := [4]float32{iq1sDelta, iq1sDelta, iq1sDelta, iq1sDelta}
		if qh0&0x08 != 0 {
			deltas[0] = -iq1sDelta
		}
		if qh0&0x80 != 0 {
			deltas[1] = -iq1sDelta
		}
		if qh1&0x08 != 0 {
			deltas[2] = -iq1sDelta
		}
		if qh1&0x80 != 0 {
			deltas[3] = -iq1sDelta
		}
		idxs := [4]uint16{idx0, idx1, idx2, idx3}
		// first 16 weights use dl1, next 16 use dl2
		for l := 0; l < 2; l++ {
			g := iq1sGrid[idxs[l]]
			for j := 0; j < 8; j++ {
				gv := int8(byte(g >> (8 * uint(j))))
				out[oi+j] = dl1 * (float32(gv) + deltas[l])
			}
			oi += 8
		}
		for l := 2; l < 4; l++ {
			g := iq1sGrid[idxs[l]]
			for j := 0; j < 8; j++ {
				gv := int8(byte(g >> (8 * uint(j))))
				out[oi+j] = dl2 * (float32(gv) + deltas[l])
			}
			oi += 8
		}
		qsi += 4
		qhi += 2
	}
}


func dequantBlocks(raw []byte, nElem, blkSize, qk int, fn func([]byte, []float32)) ([]float32, error) {
	nBlocks := (nElem + qk - 1) / qk
	need := nBlocks * blkSize
	if len(raw) < need {
		return nil, fmt.Errorf("gguf quant short: need %d have %d", need, len(raw))
	}
	out := make([]float32, nElem)
	tmp := make([]float32, qk)
	for b := 0; b < nBlocks; b++ {
		base := b * blkSize
		fn(raw[base:base+blkSize], tmp)
		off := b * qk
		n := qk
		if off+n > nElem {
			n = nElem - off
		}
		copy(out[off:off+n], tmp[:n])
	}
	return out, nil
}

// DecodeGGMLBlock expands a full tensor of the given type to f32.
func DecodeGGMLBlock(typ uint32, raw []byte, nElem int) ([]float32, error) {
	switch typ {
	case GGML_Q4_0:
		return dequantBlocks(raw, nElem, bsQ4_0, QK4_0, dequantBlockQ4_0)
	case GGML_Q4_1:
		return dequantBlocks(raw, nElem, bsQ4_1, QK4_1, dequantBlockQ4_1)
	case GGML_Q5_0:
		return dequantBlocks(raw, nElem, bsQ5_0, QK5_0, dequantBlockQ5_0)
	case GGML_Q5_1:
		return dequantBlocks(raw, nElem, bsQ5_1, QK5_1, dequantBlockQ5_1)
	case GGML_Q8_0:
		return dequantBlocks(raw, nElem, bsQ8_0, QK8_0, dequantBlockQ8_0)
	case GGML_Q8_1:
		return dequantBlocks(raw, nElem, bsQ8_1, QK8_1, dequantBlockQ8_1)
	case GGML_Q2_K:
		return dequantBlocks(raw, nElem, bsQ2_K, QK_K, dequantBlockQ2_K)
	case GGML_Q3_K:
		return dequantBlocks(raw, nElem, bsQ3_K, QK_K, dequantBlockQ3_K)
	case GGML_Q4_K:
		return dequantBlocks(raw, nElem, bsQ4_K, QK_K, dequantBlockQ4_K)
	case GGML_Q5_K:
		return dequantBlocks(raw, nElem, bsQ5_K, QK_K, dequantBlockQ5_K)
	case GGML_Q6_K:
		return dequantBlocks(raw, nElem, bsQ6_K, QK_K, dequantBlockQ6_K)
	case GGML_Q8_K:
		return dequantBlocks(raw, nElem, bsQ8_K, QK_K, dequantBlockQ8_K)
	case GGML_IQ4_NL:
		return dequantBlocks(raw, nElem, bsIQ4_NL, QK4_NL, dequantBlockIQ4_NL)
	case GGML_TQ2_0:
		return dequantBlocks(raw, nElem, bsTQ2_0, QK_K, dequantBlockTQ2_0)
	case GGML_IQ4_XS:
		return dequantBlocks(raw, nElem, bsIQ4_XS, QK_K, dequantBlockIQ4_XS)
	case GGML_TQ1_0:
		return dequantBlocks(raw, nElem, bsTQ1_0, QK_K, dequantBlockTQ1_0)
	case GGML_IQ2_XXS:
		return dequantBlocks(raw, nElem, bsIQ2_XXS, QK_K, dequantBlockIQ2_XXS)
	case GGML_IQ2_XS:
		return dequantBlocks(raw, nElem, bsIQ2_XS, QK_K, dequantBlockIQ2_XS)
	case GGML_IQ2_S:
		return dequantBlocks(raw, nElem, bsIQ2_S, QK_K, dequantBlockIQ2_S)
	case GGML_IQ3_XXS:
		return dequantBlocks(raw, nElem, bsIQ3_XXS, QK_K, dequantBlockIQ3_XXS)
	case GGML_IQ3_S:
		return dequantBlocks(raw, nElem, bsIQ3_S, QK_K, dequantBlockIQ3_S)
	case GGML_IQ1_S:
		return dequantBlocks(raw, nElem, bsIQ1_S, QK_K, dequantBlockIQ1_S)
	case GGML_IQ1_M:
		return dequantBlocks(raw, nElem, bsIQ1_M, QK_K, dequantBlockIQ1_M)
	case GGML_MXFP4:
		return dequantBlocks(raw, nElem, bsMXFP4, QK_MXFP4, dequantBlockMXFP4)
	case GGML_NVFP4:
		return dequantBlocks(raw, nElem, bsNVFP4, QK_NVFP4, dequantBlockNVFP4)
	default:
		return nil, fmt.Errorf("gguf: unsupported block type %d (block-quant zoo partial; remaining exotic types only)", typ)
	}
}


// blockLayout returns (blockByteSize, qk) for supported block quants.
func blockLayout(typ uint32) (blkSize, qk int, ok bool) {
	switch typ {
	case GGML_Q4_0:
		return bsQ4_0, QK4_0, true
	case GGML_Q4_1:
		return bsQ4_1, QK4_1, true
	case GGML_Q5_0:
		return bsQ5_0, QK5_0, true
	case GGML_Q5_1:
		return bsQ5_1, QK5_1, true
	case GGML_Q8_0:
		return bsQ8_0, QK8_0, true
	case GGML_Q8_1:
		return bsQ8_1, QK8_1, true
	case GGML_Q2_K:
		return bsQ2_K, QK_K, true
	case GGML_Q3_K:
		return bsQ3_K, QK_K, true
	case GGML_Q4_K:
		return bsQ4_K, QK_K, true
	case GGML_Q5_K:
		return bsQ5_K, QK_K, true
	case GGML_Q6_K:
		return bsQ6_K, QK_K, true
	case GGML_Q8_K:
		return bsQ8_K, QK_K, true
	case GGML_IQ4_NL:
		return bsIQ4_NL, QK4_NL, true
	case GGML_TQ2_0:
		return bsTQ2_0, QK_K, true
	case GGML_IQ4_XS:
		return bsIQ4_XS, QK_K, true
	case GGML_TQ1_0:
		return bsTQ1_0, QK_K, true
	case GGML_IQ2_XXS:
		return bsIQ2_XXS, QK_K, true
	case GGML_IQ2_XS:
		return bsIQ2_XS, QK_K, true
	case GGML_IQ2_S:
		return bsIQ2_S, QK_K, true
	case GGML_IQ3_XXS:
		return bsIQ3_XXS, QK_K, true
	case GGML_IQ3_S:
		return bsIQ3_S, QK_K, true
	case GGML_IQ1_S:
		return bsIQ1_S, QK_K, true
	case GGML_IQ1_M:
		return bsIQ1_M, QK_K, true
	case GGML_MXFP4:
		return bsMXFP4, QK_MXFP4, true
	case GGML_NVFP4:
		return bsNVFP4, QK_NVFP4, true
	default:
		return 0, 0, false
	}
}

func dequantFn(typ uint32) (func([]byte, []float32), bool) {
	switch typ {
	case GGML_Q4_0:
		return dequantBlockQ4_0, true
	case GGML_Q4_1:
		return dequantBlockQ4_1, true
	case GGML_Q5_0:
		return dequantBlockQ5_0, true
	case GGML_Q5_1:
		return dequantBlockQ5_1, true
	case GGML_Q8_0:
		return dequantBlockQ8_0, true
	case GGML_Q8_1:
		return dequantBlockQ8_1, true
	case GGML_Q2_K:
		return dequantBlockQ2_K, true
	case GGML_Q3_K:
		return dequantBlockQ3_K, true
	case GGML_Q4_K:
		return dequantBlockQ4_K, true
	case GGML_Q5_K:
		return dequantBlockQ5_K, true
	case GGML_Q6_K:
		return dequantBlockQ6_K, true
	case GGML_Q8_K:
		return dequantBlockQ8_K, true
	case GGML_IQ4_NL:
		return dequantBlockIQ4_NL, true
	case GGML_TQ2_0:
		return dequantBlockTQ2_0, true
	case GGML_IQ4_XS:
		return dequantBlockIQ4_XS, true
	case GGML_TQ1_0:
		return dequantBlockTQ1_0, true
	case GGML_IQ2_XXS:
		return dequantBlockIQ2_XXS, true
	case GGML_IQ2_XS:
		return dequantBlockIQ2_XS, true
	case GGML_IQ2_S:
		return dequantBlockIQ2_S, true
	case GGML_IQ3_XXS:
		return dequantBlockIQ3_XXS, true
	case GGML_IQ3_S:
		return dequantBlockIQ3_S, true
	case GGML_IQ1_S:
		return dequantBlockIQ1_S, true
	case GGML_IQ1_M:
		return dequantBlockIQ1_M, true
	case GGML_MXFP4:
		return dequantBlockMXFP4, true
	case GGML_NVFP4:
		return dequantBlockNVFP4, true
	default:
		return nil, false
	}
}

// DecodeGGMLWindow decodes [offset, offset+count) elements only.
// For block quants, only intersecting blocks are dequantized.
func DecodeGGMLWindow(typ uint32, raw []byte, nElem, offset, count int) ([]float32, error) {
	if offset < 0 || count < 0 || offset+count > nElem {
		return nil, fmt.Errorf("gguf window OOB: offset=%d count=%d nElem=%d", offset, count, nElem)
	}
	if count == 0 {
		return []float32{}, nil
	}
	// dense numeric types — slice raw then decode
	switch typ {
	case GGML_F32:
		need := nElem * 4
		if len(raw) < need {
			return nil, fmt.Errorf("gguf F32 short")
		}
		return decodeGGML(typ, raw[offset*4:(offset+count)*4], count)
	case GGML_F64:
		need := nElem * 8
		if len(raw) < need {
			return nil, fmt.Errorf("gguf F64 short")
		}
		return decodeGGML(typ, raw[offset*8:(offset+count)*8], count)
	case GGML_F16, GGML_BF16, GGML_I16:
		need := nElem * 2
		if len(raw) < need {
			return nil, fmt.Errorf("gguf half/i16 short")
		}
		return decodeGGML(typ, raw[offset*2:(offset+count)*2], count)
	case GGML_I8:
		if len(raw) < nElem {
			return nil, fmt.Errorf("gguf I8 short")
		}
		return decodeGGML(typ, raw[offset:offset+count], count)
	case GGML_I32:
		need := nElem * 4
		if len(raw) < need {
			return nil, fmt.Errorf("gguf I32 short")
		}
		return decodeGGML(typ, raw[offset*4:(offset+count)*4], count)
	case GGML_I64:
		need := nElem * 8
		if len(raw) < need {
			return nil, fmt.Errorf("gguf I64 short")
		}
		return decodeGGML(typ, raw[offset*8:(offset+count)*8], count)
	}
	blkSize, qk, ok := blockLayout(typ)
	fn, ok2 := dequantFn(typ)
	if !ok || !ok2 {
		return nil, fmt.Errorf("gguf: unsupported window type %d", typ)
	}
	// blocks covering [offset, offset+count)
	b0 := offset / qk
	b1 := (offset + count + qk - 1) / qk // exclusive
	nBlocksTotal := (nElem + qk - 1) / qk
	if b1 > nBlocksTotal {
		b1 = nBlocksTotal
	}
	need := nBlocksTotal * blkSize
	if len(raw) < need {
		return nil, fmt.Errorf("gguf quant short: need %d have %d", need, len(raw))
	}
	out := make([]float32, count)
	tmp := make([]float32, qk)
	for b := b0; b < b1; b++ {
		base := b * blkSize
		fn(raw[base:base+blkSize], tmp)
		blkOff := b * qk
		// intersection of block range [blkOff, blkOff+qk) with [offset, offset+count)
		src0 := 0
		dst0 := blkOff - offset
		if dst0 < 0 {
			src0 = -dst0
			dst0 = 0
		}
		src1 := qk
		if blkOff+qk > offset+count {
			src1 = offset + count - blkOff
		}
		if src0 >= src1 {
			continue
		}
		// also clip residual last block against nElem
		if blkOff+src1 > nElem {
			src1 = nElem - blkOff
		}
		if src0 >= src1 {
			continue
		}
		copy(out[dst0:dst0+(src1-src0)], tmp[src0:src1])
	}
	return out, nil
}
