package weights

import "math/bits"

// QuantizeAbsMeanTrits maps f32 weights to BitNet-style absmean trits + scale.
// dequant ≈ scale * trit with trit ∈ {-1,0,+1}.
func QuantizeAbsMeanTrits(w []float32) (trits []int8, scale float32) {
	n := len(w)
	if n == 0 {
		return []int8{}, 0
	}
	var sumAbs float64
	for _, v := range w {
		if v < 0 {
			sumAbs -= float64(v)
		} else {
			sumAbs += float64(v)
		}
	}
	scale64 := sumAbs / float64(n)
	scale = float32(scale64)
	trits = make([]int8, n)
	if scale64 == 0 {
		return trits, 0
	}
	for i, v := range w {
		r := float64(v) / scale64
		if r >= 0.5 {
			trits[i] = 1
		} else if r <= -0.5 {
			trits[i] = -1
		}
	}
	return trits, scale
}

// LinearTritScalar is the naive row-major mixed matvec:
// y[i] = scale * sum_t W[i*K+t] * x[t]  with W in {-1,0,+1}.
func LinearTritScalar(W []int8, x []float32, M, K int, scale float32) []float32 {
	out := make([]float32, M)
	if M <= 0 || K <= 0 {
		return out
	}
	for i := 0; i < M; i++ {
		base := i * K
		var acc float64
		t := 0
		// 4-way unroll for scalar path throughput.
		for ; t+4 <= K; t += 4 {
			acc += float64(W[base+t]) * float64(x[t])
			acc += float64(W[base+t+1]) * float64(x[t+1])
			acc += float64(W[base+t+2]) * float64(x[t+2])
			acc += float64(W[base+t+3]) * float64(x[t+3])
		}
		for ; t < K; t++ {
			acc += float64(W[base+t]) * float64(x[t])
		}
		out[i] = scale * float32(acc)
	}
	return out
}

// LinearTritPlanes is a packed bitplane mixed matvec.
// Each row of W is packed as 32 trits/word (pos/neg planes). For each set bit
// in pos add x[lane], in neg subtract. Words are walked in 4-wide (128-lane)
// primary chunks then residual pairs — same schedule as tritDotPlanesSIMD.
func LinearTritPlanes(W []int8, x []float32, M, K int, scale float32) []float32 {
	out := make([]float32, M)
	if M <= 0 || K <= 0 {
		return out
	}
	nw := (K + 31) / 32
	for i := 0; i < M; i++ {
		pos, neg := packInt8Words(W[i*K:i*K+K])
		acc := mixedDotPlanes(pos, neg, x, K, nw)
		out[i] = scale * float32(acc)
	}
	return out
}

// packInt8Words packs trits ∈ {-1,0,+1} into pos/neg uint32 words (bit 0 = lowest index).
func packInt8Words(row []int8) (pos, neg []uint32) {
	n := len(row)
	nw := (n + 31) / 32
	if nw == 0 {
		return []uint32{}, []uint32{}
	}
	pos = make([]uint32, nw)
	neg = make([]uint32, nw)
	for i, t := range row {
		w, b := i/32, uint(i%32)
		if t > 0 {
			pos[w] |= 1 << b
		} else if t < 0 {
			neg[w] |= 1 << b
		}
	}
	return pos, neg
}

func mixedDotPlanes(pos, neg []uint32, x []float32, n, nw int) float64 {
	if n <= 0 || nw <= 0 {
		return 0
	}
	var acc float64
	w := 0
	// Primary: 4×u32 = 128 trit lanes.
	for ; w+4 <= nw; w += 4 {
		acc += mixedWord(pos, neg, x, w, n, nw)
		acc += mixedWord(pos, neg, x, w+1, n, nw)
		acc += mixedWord(pos, neg, x, w+2, n, nw)
		acc += mixedWord(pos, neg, x, w+3, n, nw)
	}
	for ; w+2 <= nw; w += 2 {
		acc += mixedWord(pos, neg, x, w, n, nw)
		acc += mixedWord(pos, neg, x, w+1, n, nw)
	}
	for ; w < nw; w++ {
		acc += mixedWord(pos, neg, x, w, n, nw)
	}
	return acc
}

func mixedWord(pos, neg []uint32, x []float32, w, n, nw int) float64 {
	var mask uint32 = 0xffffffff
	if w == nw-1 && n%32 != 0 {
		mask = (uint32(1) << uint(n%32)) - 1
	}
	var p, g uint32
	if w < len(pos) {
		p = pos[w] & mask
	}
	if w < len(neg) {
		g = neg[w] & mask
	}
	base := w * 32
	var acc float64
	// Scan set bits only — sparse/dense both win vs full 32 mul when many zeros.
	for p != 0 {
		b := bits.TrailingZeros32(p)
		idx := base + b
		if idx < len(x) {
			acc += float64(x[idx])
		}
		p &^= 1 << uint(b)
	}
	for g != 0 {
		b := bits.TrailingZeros32(g)
		idx := base + b
		if idx < len(x) {
			acc -= float64(x[idx])
		}
		g &^= 1 << uint(b)
	}
	return acc
}

// ClampTritInt clamps an int to {-1,0,+1}.
func ClampTritInt(t int) int8 {
	if t > 1 {
		return 1
	}
	if t < -1 {
		return -1
	}
	return int8(t)
}
