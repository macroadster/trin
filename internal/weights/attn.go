package weights

import (
	"fmt"
	"math"
)

// RopeF32 applies interleaved pair rotary embedding (matches interp rope builtin).
func RopeF32(x []float32, pos, base float64) []float32 {
	n := len(x)
	out := make([]float32, n)
	if n == 0 {
		return out
	}
	if base <= 0 {
		base = 10000
	}
	half := n / 2
	d := float64(n)
	if d < 2 {
		d = 2
	}
	for i := 0; i < half; i++ {
		expo := float64(2*i) / d
		theta := pos / math.Pow(base, expo)
		c, s := math.Cos(theta), math.Sin(theta)
		xi := float64(x[2*i])
		xj := float64(x[2*i+1])
		out[2*i] = float32(xi*c - xj*s)
		out[2*i+1] = float32(xi*s + xj*c)
	}
	if n%2 == 1 {
		out[n-1] = x[n-1]
	}
	return out
}

// AttnF32 is single-head attention: y = softmax(q K^T / sqrt(d)) V.
// K, V layout: nTok * d contiguous rows.
func AttnF32(q, K, V []float32, d int) ([]float32, error) {
	if d <= 0 {
		return nil, fmt.Errorf("attn: d must be > 0")
	}
	if len(q) < d {
		return nil, fmt.Errorf("attn: q short")
	}
	nTok := len(K) / d
	if nTok*d != len(K) || len(V) < nTok*d {
		return nil, fmt.Errorf("attn: K/V dimension mismatch")
	}
	if nTok == 0 {
		return make([]float32, d), nil
	}
	scale := 1.0 / math.Sqrt(float64(d))
	logits := make([]float64, nTok)
	m := -1e30
	for t := 0; t < nTok; t++ {
		acc := 0.0
		base := t * d
		for i := 0; i < d; i++ {
			acc += float64(q[i]) * float64(K[base+i])
		}
		logits[t] = acc * scale
		if logits[t] > m {
			m = logits[t]
		}
	}
	sum := 0.0
	for t := 0; t < nTok; t++ {
		logits[t] = math.Exp(logits[t] - m)
		sum += logits[t]
	}
	out := make([]float32, d)
	if sum == 0 {
		return out, nil
	}
	inv := 1.0 / sum
	for t := 0; t < nTok; t++ {
		p := logits[t] * inv
		base := t * d
		for i := 0; i < d; i++ {
			out[i] += float32(p * float64(V[base+i]))
		}
	}
	return out, nil
}

// MHAF32 multi-head attention with contiguous head packing (same as interp mha).
func MHAF32(q, K, V []float32, d, h int) ([]float32, error) {
	if h <= 0 || d%h != 0 {
		return nil, fmt.Errorf("mha: d must be divisible by h")
	}
	if h == 1 {
		return AttnF32(q, K, V, d)
	}
	dh := d / h
	nTok := len(K) / d
	if nTok*d != len(K) || len(V) < nTok*d || len(q) < d {
		return nil, fmt.Errorf("mha: dimension mismatch")
	}
	out := make([]float32, d)
	for head := 0; head < h; head++ {
		qs := make([]float32, dh)
		for i := 0; i < dh; i++ {
			qs[i] = q[head*dh+i]
		}
		Ks := make([]float32, nTok*dh)
		Vs := make([]float32, nTok*dh)
		for t := 0; t < nTok; t++ {
			for i := 0; i < dh; i++ {
				Ks[t*dh+i] = K[t*d+head*dh+i]
				Vs[t*dh+i] = V[t*d+head*dh+i]
			}
		}
		yh, err := AttnF32(qs, Ks, Vs, dh)
		if err != nil {
			return nil, err
		}
		for i := 0; i < dh; i++ {
			out[head*dh+i] = yh[i]
		}
	}
	return out, nil
}

// AttnProj fuses Q/K/V projections: each weight is dOut×dIn row-major.
func (s *WeightSession) AttnProj(qName, kName, vName string, x []float32, dIn, dOut, tileRows int) (q, k, v []float32, err error) {
	if s == nil {
		return nil, nil, nil, fmt.Errorf("nil session")
	}
	if dIn <= 0 || dOut <= 0 || len(x) < dIn {
		return nil, nil, nil, fmt.Errorf("attn_proj: bad dims")
	}
	xIn := x[:dIn]
	q, err = s.LinearFused(qName, xIn, dOut, dIn, tileRows)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("attn_proj Q %q: %w", qName, err)
	}
	k, err = s.LinearFused(kName, xIn, dOut, dIn, tileRows)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("attn_proj K %q: %w", kName, err)
	}
	v, err = s.LinearFused(vName, xIn, dOut, dIn, tileRows)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("attn_proj V %q: %w", vName, err)
	}
	return q, k, v, nil
}

// SessionAttn projects Q/K/V, optional RoPE (base>0), appends to past KV, runs
// single-head or multi-head attention, optional out projection oName.
// Returns y (dOut), Knew, Vnew (each (nPast+1)*dOut).
// pos used only when base > 0.
func (s *WeightSession) SessionAttn(qName, kName, vName, oName string, x, Kpast, Vpast []float32, dIn, dOut, h int, pos, base float64, tileRows int) (y, Knew, Vnew []float32, err error) {
	q, k, v, err := s.AttnProj(qName, kName, vName, x, dIn, dOut, tileRows)
	if err != nil {
		return nil, nil, nil, err
	}
	if base > 0 {
		q = RopeF32(q, pos, base)
		k = RopeF32(k, pos, base)
	}
	if len(Kpast)%dOut != 0 || len(Vpast)%dOut != 0 {
		return nil, nil, nil, fmt.Errorf("session_attn: past KV not multiple of d")
	}
	if len(Kpast) != len(Vpast) {
		return nil, nil, nil, fmt.Errorf("session_attn: past K/V length mismatch")
	}
	Knew = make([]float32, 0, len(Kpast)+dOut)
	Vnew = make([]float32, 0, len(Vpast)+dOut)
	Knew = append(Knew, Kpast...)
	Knew = append(Knew, k...)
	Vnew = append(Vnew, Vpast...)
	Vnew = append(Vnew, v...)
	if h <= 1 {
		y, err = AttnF32(q, Knew, Vnew, dOut)
	} else {
		y, err = MHAF32(q, Knew, Vnew, dOut, h)
	}
	if err != nil {
		return nil, nil, nil, err
	}
	if oName != "" {
		y, err = s.LinearFused(oName, y, dOut, dOut, tileRows)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("session_attn O %q: %w", oName, err)
		}
	}
	return y, Knew, Vnew, nil
}

// GQAExpandKV expands packed K/V (nTok × nKVHeads × dh) to full MHA layout
// (nTok × nHeads × dh) by repeating each KV head group times (= nHeads/nKVHeads).
// Layout per token: contiguous heads [h0|h1|...].
func GQAExpandKV(kv []float32, nTok, nKVHeads, nHeads, dh int) ([]float32, error) {
	if nTok < 0 || nKVHeads <= 0 || nHeads <= 0 || dh <= 0 {
		return nil, fmt.Errorf("gqa_expand_kv: bad dims")
	}
	if nHeads%nKVHeads != 0 {
		return nil, fmt.Errorf("gqa_expand_kv: nHeads must be multiple of nKVHeads")
	}
	dKV := nKVHeads * dh
	dFull := nHeads * dh
	if nTok*dKV != len(kv) {
		return nil, fmt.Errorf("gqa_expand_kv: kv len %d want %d", len(kv), nTok*dKV)
	}
	if nKVHeads == nHeads {
		out := make([]float32, len(kv))
		copy(out, kv)
		return out, nil
	}
	group := nHeads / nKVHeads
	out := make([]float32, nTok*dFull)
	for t := 0; t < nTok; t++ {
		srcBase := t * dKV
		dstBase := t * dFull
		for h := 0; h < nHeads; h++ {
			kvh := h / group
			for i := 0; i < dh; i++ {
				out[dstBase+h*dh+i] = kv[srcBase+kvh*dh+i]
			}
		}
	}
	return out, nil
}

// GQARepeatKVRow expands one token's KV row (nKVHeads×dh) → (nHeads×dh).
func GQARepeatKVRow(row []float32, nKVHeads, nHeads, dh int) ([]float32, error) {
	return GQAExpandKV(row, 1, nKVHeads, nHeads, dh)
}

// AttnProjGQA projects Q at nHeads*dh and K/V at nKVHeads*dh (fused).
func (s *WeightSession) AttnProjGQA(qName, kName, vName string, x []float32, dIn, nHeads, nKVHeads, dh, tileRows int) (q, k, v []float32, err error) {
	if s == nil {
		return nil, nil, nil, fmt.Errorf("nil session")
	}
	if dIn <= 0 || nHeads <= 0 || nKVHeads <= 0 || dh <= 0 || len(x) < dIn {
		return nil, nil, nil, fmt.Errorf("attn_proj_gqa: bad dims")
	}
	if nHeads%nKVHeads != 0 {
		return nil, nil, nil, fmt.Errorf("attn_proj_gqa: nHeads must be multiple of nKVHeads")
	}
	dQ := nHeads * dh
	dKV := nKVHeads * dh
	xIn := x[:dIn]
	q, err = s.LinearFused(qName, xIn, dQ, dIn, tileRows)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("attn_proj_gqa Q %q: %w", qName, err)
	}
	k, err = s.LinearFused(kName, xIn, dKV, dIn, tileRows)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("attn_proj_gqa K %q: %w", kName, err)
	}
	v, err = s.LinearFused(vName, xIn, dKV, dIn, tileRows)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("attn_proj_gqa V %q: %w", vName, err)
	}
	return q, k, v, nil
}

// ropeHeads applies RoPE independently to each head slice of width dh.
func ropeHeads(x []float32, nHeads, dh int, pos, base float64) []float32 {
	out := make([]float32, len(x))
	for h := 0; h < nHeads; h++ {
		off := h * dh
		if off+dh > len(x) {
			break
		}
		rot := RopeF32(x[off:off+dh], pos, base)
		copy(out[off:off+dh], rot)
	}
	return out
}

// SessionAttnGQA: fused GQA project, optional per-head RoPE, compact KV cache,
// expand K/V for multi-head attention, optional out projection at dQ×dQ.
// Kpast/Vpast store compact nTok×(nKVHeads*dh). Returns y (nHeads*dh), Knew, Vnew compact.
func (s *WeightSession) SessionAttnGQA(qName, kName, vName, oName string, x, Kpast, Vpast []float32, dIn, nHeads, nKVHeads, dh int, pos, base float64, tileRows int) (y, Knew, Vnew []float32, err error) {
	q, k, v, err := s.AttnProjGQA(qName, kName, vName, x, dIn, nHeads, nKVHeads, dh, tileRows)
	if err != nil {
		return nil, nil, nil, err
	}
	dQ := nHeads * dh
	dKV := nKVHeads * dh
	if base > 0 {
		q = ropeHeads(q, nHeads, dh, pos, base)
		k = ropeHeads(k, nKVHeads, dh, pos, base)
	}
	if len(Kpast)%dKV != 0 || len(Vpast)%dKV != 0 {
		return nil, nil, nil, fmt.Errorf("session_gqa: past KV not multiple of dKV=%d", dKV)
	}
	if len(Kpast) != len(Vpast) {
		return nil, nil, nil, fmt.Errorf("session_gqa: past K/V length mismatch")
	}
	Knew = make([]float32, 0, len(Kpast)+dKV)
	Vnew = make([]float32, 0, len(Vpast)+dKV)
	Knew = append(Knew, Kpast...)
	Knew = append(Knew, k...)
	Vnew = append(Vnew, Vpast...)
	Vnew = append(Vnew, v...)
	nTok := len(Knew) / dKV
	Kexp, err := GQAExpandKV(Knew, nTok, nKVHeads, nHeads, dh)
	if err != nil {
		return nil, nil, nil, err
	}
	Vexp, err := GQAExpandKV(Vnew, nTok, nKVHeads, nHeads, dh)
	if err != nil {
		return nil, nil, nil, err
	}
	y, err = MHAF32(q, Kexp, Vexp, dQ, nHeads)
	if err != nil {
		return nil, nil, nil, err
	}
	if oName != "" {
		y, err = s.LinearFused(oName, y, dQ, dQ, tileRows)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("session_gqa O %q: %w", oName, err)
		}
	}
	return y, Knew, Vnew, nil
}

// --- v1.10 trit-cached session attention ---

// AttnProjTrit projects Q/K/V via LinearTritCached (auto-prepack).
// Weights: Q/K/V are dOut×dIn. bitplane selects packed plane matvec.
func (s *WeightSession) AttnProjTrit(qName, kName, vName string, x []float32, dIn, dOut int, bitplane bool) (q, k, v []float32, err error) {
	if s == nil {
		return nil, nil, nil, fmt.Errorf("nil session")
	}
	if dIn <= 0 || dOut <= 0 || len(x) < dIn {
		return nil, nil, nil, fmt.Errorf("attn_proj_trit: bad dims")
	}
	xIn := x[:dIn]
	q, _, err = s.LinearTritCached(qName, xIn, dOut, dIn, bitplane)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("attn_proj_trit Q %q: %w", qName, err)
	}
	k, _, err = s.LinearTritCached(kName, xIn, dOut, dIn, bitplane)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("attn_proj_trit K %q: %w", kName, err)
	}
	v, _, err = s.LinearTritCached(vName, xIn, dOut, dIn, bitplane)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("attn_proj_trit V %q: %w", vName, err)
	}
	return q, k, v, nil
}

// SessionAttnTrit is SessionAttn with trit-cached projections (and optional O).
// Bitplane only affects the linear projections; attention math stays f32.
func (s *WeightSession) SessionAttnTrit(qName, kName, vName, oName string, x, Kpast, Vpast []float32, dIn, dOut, h int, pos, base float64, bitplane bool) (y, Knew, Vnew []float32, err error) {
	q, k, v, err := s.AttnProjTrit(qName, kName, vName, x, dIn, dOut, bitplane)
	if err != nil {
		return nil, nil, nil, err
	}
	if base > 0 {
		q = RopeF32(q, pos, base)
		k = RopeF32(k, pos, base)
	}
	if len(Kpast)%dOut != 0 || len(Vpast)%dOut != 0 {
		return nil, nil, nil, fmt.Errorf("session_attn_trit: past KV not multiple of d")
	}
	if len(Kpast) != len(Vpast) {
		return nil, nil, nil, fmt.Errorf("session_attn_trit: past K/V length mismatch")
	}
	Knew = make([]float32, 0, len(Kpast)+dOut)
	Vnew = make([]float32, 0, len(Vpast)+dOut)
	Knew = append(Knew, Kpast...)
	Knew = append(Knew, k...)
	Vnew = append(Vnew, Vpast...)
	Vnew = append(Vnew, v...)
	if h <= 1 {
		y, err = AttnF32(q, Knew, Vnew, dOut)
	} else {
		y, err = MHAF32(q, Knew, Vnew, dOut, h)
	}
	if err != nil {
		return nil, nil, nil, err
	}
	if oName != "" {
		y, _, err = s.LinearTritCached(oName, y, dOut, dOut, bitplane)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("session_attn_trit O %q: %w", oName, err)
		}
	}
	return y, Knew, Vnew, nil
}

// AttnProjGQATrit is AttnProjGQA with trit-cached Q/K/V.
func (s *WeightSession) AttnProjGQATrit(qName, kName, vName string, x []float32, dIn, nHeads, nKVHeads, dh int, bitplane bool) (q, k, v []float32, err error) {
	if s == nil {
		return nil, nil, nil, fmt.Errorf("nil session")
	}
	if dIn <= 0 || nHeads <= 0 || nKVHeads <= 0 || dh <= 0 || len(x) < dIn {
		return nil, nil, nil, fmt.Errorf("attn_proj_gqa_trit: bad dims")
	}
	if nHeads%nKVHeads != 0 {
		return nil, nil, nil, fmt.Errorf("attn_proj_gqa_trit: nHeads must be multiple of nKVHeads")
	}
	dQ := nHeads * dh
	dKV := nKVHeads * dh
	xIn := x[:dIn]
	q, _, err = s.LinearTritCached(qName, xIn, dQ, dIn, bitplane)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("attn_proj_gqa_trit Q %q: %w", qName, err)
	}
	k, _, err = s.LinearTritCached(kName, xIn, dKV, dIn, bitplane)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("attn_proj_gqa_trit K %q: %w", kName, err)
	}
	v, _, err = s.LinearTritCached(vName, xIn, dKV, dIn, bitplane)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("attn_proj_gqa_trit V %q: %w", vName, err)
	}
	return q, k, v, nil
}

// SessionAttnGQATrit is SessionAttnGQA with trit-cached projections (+ optional O).
func (s *WeightSession) SessionAttnGQATrit(qName, kName, vName, oName string, x, Kpast, Vpast []float32, dIn, nHeads, nKVHeads, dh int, pos, base float64, bitplane bool) (y, Knew, Vnew []float32, err error) {
	q, k, v, err := s.AttnProjGQATrit(qName, kName, vName, x, dIn, nHeads, nKVHeads, dh, bitplane)
	if err != nil {
		return nil, nil, nil, err
	}
	dQ := nHeads * dh
	dKV := nKVHeads * dh
	if base > 0 {
		q = ropeHeads(q, nHeads, dh, pos, base)
		k = ropeHeads(k, nKVHeads, dh, pos, base)
	}
	if len(Kpast)%dKV != 0 || len(Vpast)%dKV != 0 {
		return nil, nil, nil, fmt.Errorf("session_gqa_trit: past KV not multiple of dKV=%d", dKV)
	}
	if len(Kpast) != len(Vpast) {
		return nil, nil, nil, fmt.Errorf("session_gqa_trit: past K/V length mismatch")
	}
	Knew = make([]float32, 0, len(Kpast)+dKV)
	Vnew = make([]float32, 0, len(Vpast)+dKV)
	Knew = append(Knew, Kpast...)
	Knew = append(Knew, k...)
	Vnew = append(Vnew, Vpast...)
	Vnew = append(Vnew, v...)
	nTok := len(Knew) / dKV
	Kexp, err := GQAExpandKV(Knew, nTok, nKVHeads, nHeads, dh)
	if err != nil {
		return nil, nil, nil, err
	}
	Vexp, err := GQAExpandKV(Vnew, nTok, nKVHeads, nHeads, dh)
	if err != nil {
		return nil, nil, nil, err
	}
	y, err = MHAF32(q, Kexp, Vexp, dQ, nHeads)
	if err != nil {
		return nil, nil, nil, err
	}
	if oName != "" {
		y, _, err = s.LinearTritCached(oName, y, dQ, dQ, bitplane)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("session_gqa_trit O %q: %w", oName, err)
		}
	}
	return y, Knew, Vnew, nil
}
