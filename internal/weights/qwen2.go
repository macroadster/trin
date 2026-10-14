package weights

import (
	"context"
	"fmt"
	"math"
	"strconv"
)

// Qwen2 is a Qwen2 GGUF session (Q/K/V bias, RoPE-Neox, GQA, SwiGLU).
// Matmul stays on the weight session, so Q8_0 blocks are dequantized per row.
type Qwen2 struct {
	s        *WeightSession
	tok      *qwenTokenizer
	layers   []qwenLayer
	dim      int
	hidden   int
	nHeads   int
	nKV      int
	dh       int
	vocab    int
	eps      float64
	ropeBase float64
	finalN   []float32
	eos      int
}

type qwenLayer struct {
	qName, kName, vName, oName string
	gateName, upName, downName string
	qB, kB, vB                 []float32
	attnN, ffnN                []float32
}

type qwenKV struct {
	K []float32
	V []float32
}

// LoadQwen2 reads hparams and tokenizer from an open GGUF session.
// The session must stay open for Generate.
func LoadQwen2(s *WeightSession) (*Qwen2, error) {
	if s == nil {
		return nil, fmt.Errorf("qwen2: nil session")
	}
	if s.Meta("general.architecture") != "qwen2" {
		return nil, fmt.Errorf("qwen2: architecture %q", s.Meta("general.architecture"))
	}
	dim := metaAtoi(s, "qwen2.embedding_length")
	hidden := metaAtoi(s, "qwen2.feed_forward_length")
	nHeads := metaAtoi(s, "qwen2.attention.head_count")
	nKV := metaAtoi(s, "qwen2.attention.head_count_kv")
	nLayers := metaAtoi(s, "qwen2.block_count")
	if dim <= 0 || hidden <= 0 || nHeads <= 0 || nKV <= 0 || nLayers <= 0 {
		return nil, fmt.Errorf("qwen2: bad hparams dim=%d hidden=%d heads=%d kv=%d layers=%d", dim, hidden, nHeads, nKV, nLayers)
	}
	if nHeads%nKV != 0 || dim%nHeads != 0 {
		return nil, fmt.Errorf("qwen2: heads do not divide dim")
	}
	dh := dim / nHeads
	if v := metaAtoi(s, "qwen2.attention.key_length"); v > 0 {
		dh = v
	}
	eps := 1e-6
	if raw := s.Meta("qwen2.attention.layer_norm_rms_epsilon"); raw != "" {
		if f, err := strconv.ParseFloat(raw, 64); err == nil && f > 0 {
			eps = f
		}
	}
	ropeBase := 1000000.0
	if raw := s.Meta("qwen2.rope.freq_base"); raw != "" {
		if f, err := strconv.ParseFloat(raw, 64); err == nil && f > 0 {
			ropeBase = f
		}
	}
	tok, err := loadQwenTokenizer(s)
	if err != nil {
		return nil, err
	}
	m := &Qwen2{
		s: s, tok: tok,
		dim: dim, hidden: hidden, nHeads: nHeads, nKV: nKV, dh: dh,
		vocab:    len(tok.tokens),
		eps:      eps,
		ropeBase: ropeBase,
		eos:      tok.eos,
	}
	m.finalN, err = s.LoadF32Window("output_norm.weight", 0, dim)
	if err != nil {
		return nil, err
	}
	dKV := nKV * dh
	m.layers = make([]qwenLayer, nLayers)
	for i := 0; i < nLayers; i++ {
		p := fmt.Sprintf("blk.%d.", i)
		lay := qwenLayer{
			qName: p + "attn_q.weight", kName: p + "attn_k.weight", vName: p + "attn_v.weight",
			oName:    p + "attn_output.weight",
			gateName: p + "ffn_gate.weight", upName: p + "ffn_up.weight", downName: p + "ffn_down.weight",
		}
		if lay.qB, err = s.LoadF32Window(p+"attn_q.bias", 0, nHeads*dh); err != nil {
			return nil, err
		}
		if lay.kB, err = s.LoadF32Window(p+"attn_k.bias", 0, dKV); err != nil {
			return nil, err
		}
		if lay.vB, err = s.LoadF32Window(p+"attn_v.bias", 0, dKV); err != nil {
			return nil, err
		}
		if lay.attnN, err = s.LoadF32Window(p+"attn_norm.weight", 0, dim); err != nil {
			return nil, err
		}
		if lay.ffnN, err = s.LoadF32Window(p+"ffn_norm.weight", 0, dim); err != nil {
			return nil, err
		}
		m.layers[i] = lay
	}
	return m, nil
}

// Generate encodes prompt and samples greedily (temperature 0) for n new tokens.
func (m *Qwen2) Generate(ctx context.Context, prompt string, n int) (string, error) {
	if m == nil || m.tok == nil {
		return "", fmt.Errorf("qwen2: model not loaded")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if n <= 0 {
		n = 1
	}
	ids := m.tok.Encode(prompt)
	if len(ids) == 0 {
		return "", fmt.Errorf("qwen2: prompt encoded to no tokens")
	}
	cache := make([]qwenKV, len(m.layers))
	var logits []float32
	var err error
	for pos, id := range ids {
		if err = ctx.Err(); err != nil {
			return "", err
		}
		logits, err = m.step(id, pos, cache)
		if err != nil {
			return "", err
		}
	}
	out := make([]int, 0, n)
	pos := len(ids)
	for i := 0; i < n; i++ {
		if err = ctx.Err(); err != nil {
			return "", err
		}
		next := argmax(logits)
		if next == m.eos || m.tok.isControl(next) {
			break
		}
		out = append(out, next)
		logits, err = m.step(next, pos, cache)
		if err != nil {
			return "", err
		}
		pos++
	}
	return m.tok.Decode(out), nil
}

func (m *Qwen2) step(id, pos int, cache []qwenKV) ([]float32, error) {
	x, err := m.s.EmbedRow("token_embd.weight", id, m.dim)
	if err != nil {
		return nil, err
	}
	dKV := m.nKV * m.dh
	for i, lay := range m.layers {
		h := RMSNormF32(x, m.eps, lay.attnN)
		q, err := m.s.LinearFused(lay.qName, h, m.nHeads*m.dh, m.dim, 64)
		if err != nil {
			return nil, fmt.Errorf("qwen2 layer %d q: %w", i, err)
		}
		k, err := m.s.LinearFused(lay.kName, h, dKV, m.dim, 64)
		if err != nil {
			return nil, fmt.Errorf("qwen2 layer %d k: %w", i, err)
		}
		v, err := m.s.LinearFused(lay.vName, h, dKV, m.dim, 64)
		if err != nil {
			return nil, fmt.Errorf("qwen2 layer %d v: %w", i, err)
		}
		addInto(q, lay.qB)
		addInto(k, lay.kB)
		addInto(v, lay.vB)
		q = ropeNeoXHeads(q, m.nHeads, m.dh, float64(pos), m.ropeBase)
		k = ropeNeoXHeads(k, m.nKV, m.dh, float64(pos), m.ropeBase)
		cache[i].K = append(cache[i].K, k...)
		cache[i].V = append(cache[i].V, v...)
		nTok := len(cache[i].K) / dKV
		kExp, err := GQAExpandKV(cache[i].K, nTok, m.nKV, m.nHeads, m.dh)
		if err != nil {
			return nil, err
		}
		vExp, err := GQAExpandKV(cache[i].V, nTok, m.nKV, m.nHeads, m.dh)
		if err != nil {
			return nil, err
		}
		attn, err := MHAF32(q, kExp, vExp, m.nHeads*m.dh, m.nHeads)
		if err != nil {
			return nil, err
		}
		o, err := m.s.LinearFused(lay.oName, attn, m.dim, m.dim, 64)
		if err != nil {
			return nil, fmt.Errorf("qwen2 layer %d o: %w", i, err)
		}
		addInto(x, o)
		h = RMSNormF32(x, m.eps, lay.ffnN)
		gate, err := m.s.LinearFused(lay.gateName, h, m.hidden, m.dim, 64)
		if err != nil {
			return nil, fmt.Errorf("qwen2 layer %d gate: %w", i, err)
		}
		up, err := m.s.LinearFused(lay.upName, h, m.hidden, m.dim, 64)
		if err != nil {
			return nil, fmt.Errorf("qwen2 layer %d up: %w", i, err)
		}
		mid := mulVec(silu(gate), up)
		down, err := m.s.LinearFused(lay.downName, mid, m.dim, m.hidden, 64)
		if err != nil {
			return nil, fmt.Errorf("qwen2 layer %d down: %w", i, err)
		}
		addInto(x, down)
	}
	h := RMSNormF32(x, m.eps, m.finalN)
	logits, err := m.s.LinearFused("output.weight", h, m.vocab, m.dim, 64)
	if err != nil {
		return nil, fmt.Errorf("qwen2 lm_head: %w", err)
	}
	return logits, nil
}

// ropeNeoXHeads applies Llama/Qwen rotate-half RoPE to each head.
func ropeNeoXHeads(x []float32, nHeads, dh int, pos, base float64) []float32 {
	out := make([]float32, len(x))
	if base <= 0 {
		base = 10000
	}
	half := dh / 2
	for h := 0; h < nHeads; h++ {
		off := h * dh
		if off+dh > len(x) || half <= 0 {
			break
		}
		for i := 0; i < half; i++ {
			expo := float64(2*i) / float64(dh)
			theta := pos / math.Pow(base, expo)
			c, s := math.Cos(theta), math.Sin(theta)
			x0 := float64(x[off+i])
			x1 := float64(x[off+half+i])
			out[off+i] = float32(x0*c - x1*s)
			out[off+half+i] = float32(x0*s + x1*c)
		}
	}
	return out
}

func silu(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v / float32(1+math.Exp(-float64(v)))
	}
	return out
}

func mulVec(a, b []float32) []float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = a[i] * b[i]
	}
	return out
}

func addInto(dst, src []float32) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	for i := 0; i < n; i++ {
		dst[i] += src[i]
	}
}

func argmax(v []float32) int {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

func metaAtoi(s *WeightSession, key string) int {
	n, _ := strconv.Atoi(s.Meta(key))
	return n
}
