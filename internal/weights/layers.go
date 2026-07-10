package weights

import (
	"fmt"
	"math"
	"strings"
)

// LayerSpec describes one fused residual block on a retained session.
// WName / OutName are weight tensors (MxK and dim×dim or dim×hidden).
// GainName optional RMSNorm gain tensor (dim F32 vector); empty → unit gain.
// Mode: "residual" (y = rms(x + W x)) or "plain" (y = rms(W x)).
type LayerSpec struct {
	WName    string
	GainName string
	Mode     string // "residual" | "plain" | ""
}

// RMSNormF32 computes y_i = x_i / sqrt(mean(x^2)+eps) * gain_i (gain nil → 1).
func RMSNormF32(x []float32, eps float64, gain []float32) []float32 {
	n := len(x)
	out := make([]float32, n)
	if n == 0 {
		return out
	}
	ss := 0.0
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := 1.0 / math.Sqrt(ss/float64(n)+eps)
	for i := 0; i < n; i++ {
		g := float32(1)
		if gain != nil && i < len(gain) {
			g = gain[i]
		}
		out[i] = float32(float64(x[i])*inv) * g
	}
	return out
}

// loadGainWindow: fused-safe gain as a 1-d vector via window decode (no full multi-GB map).
func (s *WeightSession) loadGain(name string, dim int) ([]float32, error) {
	if name == "" {
		return nil, nil
	}
	return s.LoadF32Window(name, 0, dim)
}

// ForwardBlock runs one layer: optional fused linear + residual + RMSNorm.
// dim is activation size; weight is dim×dim (square) or M=dim, K=dim.
func (s *WeightSession) ForwardBlock(spec LayerSpec, x []float32, dim, tileRows int, eps float64) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("nil session")
	}
	if dim <= 0 || len(x) < dim {
		return nil, fmt.Errorf("forward_block: bad dim")
	}
	mode := strings.ToLower(spec.Mode)
	if mode == "" {
		mode = "residual"
	}
	xIn := x[:dim]
	y, err := s.LinearFused(spec.WName, xIn, dim, dim, tileRows)
	if err != nil {
		return nil, fmt.Errorf("forward_block W %q: %w", spec.WName, err)
	}
	// residual
	h := make([]float32, dim)
	switch mode {
	case "plain":
		copy(h, y)
	case "residual":
		for i := 0; i < dim; i++ {
			h[i] = xIn[i] + y[i]
		}
	default:
		return nil, fmt.Errorf("forward_block: unknown mode %q", mode)
	}
	gain, err := s.loadGain(spec.GainName, dim)
	if err != nil {
		return nil, fmt.Errorf("forward_block gain %q: %w", spec.GainName, err)
	}
	return RMSNormF32(h, eps, gain), nil
}

// LayerStack runs N residual blocks chained: x0 → L0 → … → LN-1 → y
// names = weight names (required, length N); gains may be shorter/empty.
// modes string is comma-separated residual|plain (empty → residual for all).
func (s *WeightSession) LayerStack(wNames []string, gainNames []string, modes []string, x []float32, dim, tileRows int, eps float64) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("nil session")
	}
	if len(wNames) == 0 {
		return nil, fmt.Errorf("layer_stack: empty layers")
	}
	if dim <= 0 || len(x) < dim {
		return nil, fmt.Errorf("layer_stack: bad dim")
	}
	cur := make([]float32, dim)
	copy(cur, x[:dim])
	for i, w := range wNames {
		spec := LayerSpec{WName: w}
		if i < len(gainNames) {
			spec.GainName = gainNames[i]
		}
		if i < len(modes) && modes[i] != "" {
			spec.Mode = modes[i]
		}
		next, err := s.ForwardBlock(spec, cur, dim, tileRows, eps)
		if err != nil {
			return nil, fmt.Errorf("layer_stack layer %d: %w", i, err)
		}
		cur = next
	}
	return cur, nil
}

// LayerStackCSV is a language-friendly surface:
// wCSV / gainCSV / modeCSV are comma-separated names (gainCSV/modeCSV may be empty).
func (s *WeightSession) LayerStackCSV(wCSV, gainCSV, modeCSV string, x []float32, dim, tileRows int, eps float64) ([]float32, error) {
	return s.LayerStack(splitCSV(wCSV), splitCSV(gainCSV), splitCSV(modeCSV), x, dim, tileRows, eps)
}

func splitCSV(s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// ForwardBlockTrit is the trit-cached residual block (v1.09):
// y = LinearTritCached(W, x) then residual/plain then RMSNorm(+gain).
// Weights are dim×dim; packs are auto-built on first call (or via PrepackTrit).
// bitplane selects packed plane matvec vs scalar trit path.
func (s *WeightSession) ForwardBlockTrit(spec LayerSpec, x []float32, dim int, bitplane bool, eps float64) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("nil session")
	}
	if dim <= 0 || len(x) < dim {
		return nil, fmt.Errorf("forward_block_trit: bad dim")
	}
	mode := strings.ToLower(spec.Mode)
	if mode == "" {
		mode = "residual"
	}
	xIn := x[:dim]
	y, _, err := s.LinearTritCached(spec.WName, xIn, dim, dim, bitplane)
	if err != nil {
		return nil, fmt.Errorf("forward_block_trit W %q: %w", spec.WName, err)
	}
	h := make([]float32, dim)
	switch mode {
	case "plain":
		copy(h, y)
	case "residual":
		for i := 0; i < dim; i++ {
			h[i] = xIn[i] + y[i]
		}
	default:
		return nil, fmt.Errorf("forward_block_trit: unknown mode %q", mode)
	}
	gain, err := s.loadGain(spec.GainName, dim)
	if err != nil {
		return nil, fmt.Errorf("forward_block_trit gain %q: %w", spec.GainName, err)
	}
	return RMSNormF32(h, eps, gain), nil
}

// LayerStackTrit chains ForwardBlockTrit layers (shared trit prepack cache).
func (s *WeightSession) LayerStackTrit(wNames []string, gainNames []string, modes []string, x []float32, dim int, bitplane bool, eps float64) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("nil session")
	}
	if len(wNames) == 0 {
		return nil, fmt.Errorf("layer_stack_trit: empty layers")
	}
	if dim <= 0 || len(x) < dim {
		return nil, fmt.Errorf("layer_stack_trit: bad dim")
	}
	cur := make([]float32, dim)
	copy(cur, x[:dim])
	for i, w := range wNames {
		spec := LayerSpec{WName: w}
		if i < len(gainNames) {
			spec.GainName = gainNames[i]
		}
		if i < len(modes) && modes[i] != "" {
			spec.Mode = modes[i]
		}
		next, err := s.ForwardBlockTrit(spec, cur, dim, bitplane, eps)
		if err != nil {
			return nil, fmt.Errorf("layer_stack_trit layer %d: %w", i, err)
		}
		cur = next
	}
	return cur, nil
}

// LayerStackTritCSV is the language-friendly trit stack:
// weight_session_layer_stack_trit(id, wCSV, x, dim [, gainCSV] [, modeCSV] [, bitplane] [, eps]).
func (s *WeightSession) LayerStackTritCSV(wCSV, gainCSV, modeCSV string, x []float32, dim int, bitplane bool, eps float64) ([]float32, error) {
	return s.LayerStackTrit(splitCSV(wCSV), splitCSV(gainCSV), splitCSV(modeCSV), x, dim, bitplane, eps)
}

// PrepackTritCSV prepacks each weight name in wCSV with K in-features.
// Returns (builtCount, hitCount).
func (s *WeightSession) PrepackTritCSV(wCSV string, K int) (built, hits int, err error) {
	names := splitCSV(wCSV)
	if len(names) == 0 {
		return 0, 0, fmt.Errorf("prepack_trit_csv: empty names")
	}
	for _, n := range names {
		b, e := s.PrepackTrit(n, K)
		if e != nil {
			return built, hits, fmt.Errorf("prepack_trit_csv %q: %w", n, e)
		}
		if b {
			built++
		} else {
			hits++
		}
	}
	return built, hits, nil
}

// --- v1.12 multi-layer trit-attn decoder stack ---

// AttnLayerSpec describes one transformer residual layer:
//   xn = RMSNorm(x, AttnGain)
//   a, K, V = SessionAttnTrit(Q/K/V[/O], xn, past, …)
//   h = x + a
//   hn = RMSNorm(h, FFNGain)
//   f = LinearTritCached(Down, SiLU(LinearTritCached(Up, hn)))        // SiLU FFN
//   or SwiGLU: Down(SiLU(Gate(hn)) ⊙ Up(hn)) when Gate is non-empty
//   y = h + f
// Q/K/V are dOut×dIn (dOut==dim); O is dim×dim when non-empty; Gate/Up are
// hidden×dim; Down is dim×hidden. Gains are length-dim vectors (empty → unit).
// Gate is unused by pre-v1.15 SiLU paths (kept empty).
type AttnLayerSpec struct {
	Q, K, V, O   string
	AttnGain     string
	FFNGain      string
	Gate         string // optional SwiGLU gate proj (v1.15+)
	Up, Down     string
	Hidden       int
}

// SiLUF32 is the elementwise sigmoid-linear unit (matches interp silu).
func SiLUF32(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		f := float64(v)
		out[i] = float32(f / (1.0 + math.Exp(-f)))
	}
	return out
}

// MulF32 is elementwise product (Hadamard); both must share length.
func MulF32(a, b []float32) ([]float32, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("mul_f32: len %d vs %d", len(a), len(b))
	}
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] * b[i]
	}
	return out, nil
}

// SwiGLUF32 computes SiLU(gate) ⊙ up (elementwise). lengths must match.
func SwiGLUF32(gate, up []float32) ([]float32, error) {
	sg := SiLUF32(gate)
	return MulF32(sg, up)
}

// ForwardAttnBlockTrit runs one residual attn+FFN block on the trit prepack cache.
// dIn/dOut are activation dims (square residual); h is MHA heads (≤1 → single-head).
// past K/V grow by one token; attention math stays f32, projections use trit packs.
func (s *WeightSession) ForwardAttnBlockTrit(
	spec AttnLayerSpec,
	x, Kpast, Vpast []float32,
	dim, h int,
	pos, base float64,
	bitplane bool,
	eps float64,
) (y, Knew, Vnew []float32, err error) {
	if s == nil {
		return nil, nil, nil, fmt.Errorf("nil session")
	}
	if dim <= 0 || len(x) < dim {
		return nil, nil, nil, fmt.Errorf("forward_attn_block_trit: bad dim")
	}
	if spec.Q == "" || spec.K == "" || spec.V == "" {
		return nil, nil, nil, fmt.Errorf("forward_attn_block_trit: Q/K/V required")
	}
	if spec.Up == "" || spec.Down == "" {
		return nil, nil, nil, fmt.Errorf("forward_attn_block_trit: FFN Up/Down required")
	}
	if spec.Hidden <= 0 {
		return nil, nil, nil, fmt.Errorf("forward_attn_block_trit: hidden must be > 0")
	}
	if h <= 0 {
		h = 1
	}
	xIn := x[:dim]
	attnGain, err := s.loadGain(spec.AttnGain, dim)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_attn_block_trit attn gain %q: %w", spec.AttnGain, err)
	}
	xn := RMSNormF32(xIn, eps, attnGain)
	a, Knew, Vnew, err := s.SessionAttnTrit(
		spec.Q, spec.K, spec.V, spec.O,
		xn, Kpast, Vpast,
		dim, dim, h, pos, base, bitplane,
	)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_attn_block_trit attn: %w", err)
	}
	hRes := make([]float32, dim)
	for i := 0; i < dim; i++ {
		hRes[i] = xIn[i] + a[i]
	}
	ffnGain, err := s.loadGain(spec.FFNGain, dim)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_attn_block_trit ffn gain %q: %w", spec.FFNGain, err)
	}
	hn := RMSNormF32(hRes, eps, ffnGain)
	up, _, err := s.LinearTritCached(spec.Up, hn, spec.Hidden, dim, bitplane)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_attn_block_trit up %q: %w", spec.Up, err)
	}
	act := SiLUF32(up)
	down, _, err := s.LinearTritCached(spec.Down, act, dim, spec.Hidden, bitplane)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_attn_block_trit down %q: %w", spec.Down, err)
	}
	y = make([]float32, dim)
	for i := 0; i < dim; i++ {
		y[i] = hRes[i] + down[i]
	}
	return y, Knew, Vnew, nil
}

// AttnStackTrit chains ForwardAttnBlockTrit over layers with per-layer KV caches.
// Kpasts/Vpasts length must match layers (nil or short entries treated as empty).
func (s *WeightSession) AttnStackTrit(
	layers []AttnLayerSpec,
	x []float32,
	Kpasts, Vpasts [][]float32,
	dim, h int,
	pos, base float64,
	bitplane bool,
	eps float64,
) (y []float32, Knews, Vnews [][]float32, err error) {
	if s == nil {
		return nil, nil, nil, fmt.Errorf("nil session")
	}
	if len(layers) == 0 {
		return nil, nil, nil, fmt.Errorf("attn_stack_trit: empty layers")
	}
	if dim <= 0 || len(x) < dim {
		return nil, nil, nil, fmt.Errorf("attn_stack_trit: bad dim")
	}
	cur := make([]float32, dim)
	copy(cur, x[:dim])
	Knews = make([][]float32, len(layers))
	Vnews = make([][]float32, len(layers))
	for i, spec := range layers {
		var kp, vp []float32
		if i < len(Kpasts) {
			kp = Kpasts[i]
		}
		if i < len(Vpasts) {
			vp = Vpasts[i]
		}
		next, kn, vn, e := s.ForwardAttnBlockTrit(spec, cur, kp, vp, dim, h, pos, base, bitplane, eps)
		if e != nil {
			return nil, nil, nil, fmt.Errorf("attn_stack_trit layer %d: %w", i, e)
		}
		cur = next
		Knews[i] = kn
		Vnews[i] = vn
	}
	return cur, Knews, Vnews, nil
}

// FlattenLayerKV concatenates per-layer KV caches (layer-major): [L0 | L1 | …].
func FlattenLayerKV(parts [][]float32) []float32 {
	n := 0
	for _, p := range parts {
		n += len(p)
	}
	out := make([]float32, 0, n)
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

// SplitLayerKV splits a layer-major flat KV into nLayers slices of equal token length.
// Empty flat → nLayers empty slices. Each layer length must be multiple of d.
func SplitLayerKV(flat []float32, nLayers, d int) ([][]float32, error) {
	if nLayers <= 0 {
		return nil, fmt.Errorf("split_layer_kv: bad nLayers")
	}
	if d <= 0 {
		return nil, fmt.Errorf("split_layer_kv: bad d")
	}
	out := make([][]float32, nLayers)
	if len(flat) == 0 {
		return out, nil
	}
	if len(flat)%(nLayers*d) != 0 {
		return nil, fmt.Errorf("split_layer_kv: flat len %d not multiple of nLayers*d=%d", len(flat), nLayers*d)
	}
	nTok := len(flat) / (nLayers * d)
	step := nTok * d
	for i := 0; i < nLayers; i++ {
		base := i * step
		out[i] = flat[base : base+step]
	}
	return out, nil
}

// AttnStackTritCSV is the language-friendly multi-layer trit-attn surface.
// q/k/v/o/up/down CSVs must share the same layer count; o/gains may be shorter/empty.
// KpastFlat/VpastFlat are layer-major flats (empty = no past). hidden applies to every layer.
// Returns y, KnewFlat, VnewFlat.
func (s *WeightSession) AttnStackTritCSV(
	qCSV, kCSV, vCSV, oCSV, upCSV, downCSV, attnGainCSV, ffnGainCSV string,
	x, KpastFlat, VpastFlat []float32,
	dim, hidden, heads int,
	pos, base float64,
	bitplane bool,
	eps float64,
) (y, KnewFlat, VnewFlat []float32, err error) {
	qs := splitCSV(qCSV)
	ks := splitCSV(kCSV)
	vs := splitCSV(vCSV)
	os_ := splitCSV(oCSV)
	ups := splitCSV(upCSV)
	downs := splitCSV(downCSV)
	ag := splitCSV(attnGainCSV)
	fg := splitCSV(ffnGainCSV)
	n := len(qs)
	if n == 0 || len(ks) != n || len(vs) != n || len(ups) != n || len(downs) != n {
		return nil, nil, nil, fmt.Errorf("attn_stack_trit: Q/K/V/Up/Down CSVs must share non-zero length")
	}
	layers := make([]AttnLayerSpec, n)
	for i := 0; i < n; i++ {
		spec := AttnLayerSpec{
			Q: qs[i], K: ks[i], V: vs[i],
			Up: ups[i], Down: downs[i],
			Hidden: hidden,
		}
		if i < len(os_) {
			spec.O = os_[i]
		}
		if i < len(ag) {
			spec.AttnGain = ag[i]
		}
		if i < len(fg) {
			spec.FFNGain = fg[i]
		}
		layers[i] = spec
	}
	Kpasts, err := SplitLayerKV(KpastFlat, n, dim)
	if err != nil {
		return nil, nil, nil, err
	}
	Vpasts, err := SplitLayerKV(VpastFlat, n, dim)
	if err != nil {
		return nil, nil, nil, err
	}
	y, Knews, Vnews, err := s.AttnStackTrit(layers, x, Kpasts, Vpasts, dim, heads, pos, base, bitplane, eps)
	if err != nil {
		return nil, nil, nil, err
	}
	return y, FlattenLayerKV(Knews), FlattenLayerKV(Vnews), nil
}

// --- v1.14 multi-layer residual GQA stack + final RMSNorm ---

// ForwardGQABlockTrit runs one residual GQA+FFN block on the trit prepack cache.
// dim must equal nHeads*dh for residual add; compact past K/V grow by one token
// of length nKVHeads*dh. Attention uses SessionAttnGQATrit (expand for MHA).
func (s *WeightSession) ForwardGQABlockTrit(
	spec AttnLayerSpec,
	x, Kpast, Vpast []float32,
	dim, nHeads, nKVHeads, dh int,
	pos, base float64,
	bitplane bool,
	eps float64,
) (y, Knew, Vnew []float32, err error) {
	if s == nil {
		return nil, nil, nil, fmt.Errorf("nil session")
	}
	if dim <= 0 || len(x) < dim {
		return nil, nil, nil, fmt.Errorf("forward_gqa_block_trit: bad dim")
	}
	if nHeads <= 0 || nKVHeads <= 0 || dh <= 0 {
		return nil, nil, nil, fmt.Errorf("forward_gqa_block_trit: bad heads")
	}
	if nHeads%nKVHeads != 0 {
		return nil, nil, nil, fmt.Errorf("forward_gqa_block_trit: nHeads must be multiple of nKVHeads")
	}
	if nHeads*dh != dim {
		return nil, nil, nil, fmt.Errorf("forward_gqa_block_trit: dim=%d != nHeads*dh=%d", dim, nHeads*dh)
	}
	if spec.Q == "" || spec.K == "" || spec.V == "" {
		return nil, nil, nil, fmt.Errorf("forward_gqa_block_trit: Q/K/V required")
	}
	if spec.Up == "" || spec.Down == "" {
		return nil, nil, nil, fmt.Errorf("forward_gqa_block_trit: FFN Up/Down required")
	}
	if spec.Hidden <= 0 {
		return nil, nil, nil, fmt.Errorf("forward_gqa_block_trit: hidden must be > 0")
	}
	xIn := x[:dim]
	attnGain, err := s.loadGain(spec.AttnGain, dim)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_gqa_block_trit attn gain %q: %w", spec.AttnGain, err)
	}
	xn := RMSNormF32(xIn, eps, attnGain)
	a, Knew, Vnew, err := s.SessionAttnGQATrit(
		spec.Q, spec.K, spec.V, spec.O,
		xn, Kpast, Vpast,
		dim, nHeads, nKVHeads, dh, pos, base, bitplane,
	)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_gqa_block_trit attn: %w", err)
	}
	if len(a) < dim {
		return nil, nil, nil, fmt.Errorf("forward_gqa_block_trit: attn out len %d < dim %d", len(a), dim)
	}
	hRes := make([]float32, dim)
	for i := 0; i < dim; i++ {
		hRes[i] = xIn[i] + a[i]
	}
	ffnGain, err := s.loadGain(spec.FFNGain, dim)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_gqa_block_trit ffn gain %q: %w", spec.FFNGain, err)
	}
	hn := RMSNormF32(hRes, eps, ffnGain)
	up, _, err := s.LinearTritCached(spec.Up, hn, spec.Hidden, dim, bitplane)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_gqa_block_trit up %q: %w", spec.Up, err)
	}
	act := SiLUF32(up)
	down, _, err := s.LinearTritCached(spec.Down, act, dim, spec.Hidden, bitplane)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_gqa_block_trit down %q: %w", spec.Down, err)
	}
	y = make([]float32, dim)
	for i := 0; i < dim; i++ {
		y[i] = hRes[i] + down[i]
	}
	return y, Knew, Vnew, nil
}

// GQAStackTrit chains ForwardGQABlockTrit over layers with per-layer compact KV caches.
func (s *WeightSession) GQAStackTrit(
	layers []AttnLayerSpec,
	x []float32,
	Kpasts, Vpasts [][]float32,
	dim, nHeads, nKVHeads, dh int,
	pos, base float64,
	bitplane bool,
	eps float64,
) (y []float32, Knews, Vnews [][]float32, err error) {
	if s == nil {
		return nil, nil, nil, fmt.Errorf("nil session")
	}
	if len(layers) == 0 {
		return nil, nil, nil, fmt.Errorf("gqa_stack_trit: empty layers")
	}
	if dim <= 0 || len(x) < dim {
		return nil, nil, nil, fmt.Errorf("gqa_stack_trit: bad dim")
	}
	cur := make([]float32, dim)
	copy(cur, x[:dim])
	Knews = make([][]float32, len(layers))
	Vnews = make([][]float32, len(layers))
	for i, spec := range layers {
		var kp, vp []float32
		if i < len(Kpasts) {
			kp = Kpasts[i]
		}
		if i < len(Vpasts) {
			vp = Vpasts[i]
		}
		next, kn, vn, e := s.ForwardGQABlockTrit(spec, cur, kp, vp, dim, nHeads, nKVHeads, dh, pos, base, bitplane, eps)
		if e != nil {
			return nil, nil, nil, fmt.Errorf("gqa_stack_trit layer %d: %w", i, e)
		}
		cur = next
		Knews[i] = kn
		Vnews[i] = vn
	}
	return cur, Knews, Vnews, nil
}

// GQAStackTritCSV is the language-friendly multi-layer GQA surface.
// KpastFlat/VpastFlat are layer-major flats of compact dKV = nKVHeads*dh per token.
// Returns y, KnewFlat, VnewFlat (still compact).
func (s *WeightSession) GQAStackTritCSV(
	qCSV, kCSV, vCSV, oCSV, upCSV, downCSV, attnGainCSV, ffnGainCSV string,
	x, KpastFlat, VpastFlat []float32,
	dim, hidden, nHeads, nKVHeads, dh int,
	pos, base float64,
	bitplane bool,
	eps float64,
) (y, KnewFlat, VnewFlat []float32, err error) {
	qs := splitCSV(qCSV)
	ks := splitCSV(kCSV)
	vs := splitCSV(vCSV)
	os_ := splitCSV(oCSV)
	ups := splitCSV(upCSV)
	downs := splitCSV(downCSV)
	ag := splitCSV(attnGainCSV)
	fg := splitCSV(ffnGainCSV)
	n := len(qs)
	if n == 0 || len(ks) != n || len(vs) != n || len(ups) != n || len(downs) != n {
		return nil, nil, nil, fmt.Errorf("gqa_stack_trit: Q/K/V/Up/Down CSVs must share non-zero length")
	}
	if nHeads <= 0 || nKVHeads <= 0 || dh <= 0 {
		return nil, nil, nil, fmt.Errorf("gqa_stack_trit: bad heads")
	}
	dKV := nKVHeads * dh
	layers := make([]AttnLayerSpec, n)
	for i := 0; i < n; i++ {
		spec := AttnLayerSpec{
			Q: qs[i], K: ks[i], V: vs[i],
			Up: ups[i], Down: downs[i],
			Hidden: hidden,
		}
		if i < len(os_) {
			spec.O = os_[i]
		}
		if i < len(ag) {
			spec.AttnGain = ag[i]
		}
		if i < len(fg) {
			spec.FFNGain = fg[i]
		}
		layers[i] = spec
	}
	Kpasts, err := SplitLayerKV(KpastFlat, n, dKV)
	if err != nil {
		return nil, nil, nil, err
	}
	Vpasts, err := SplitLayerKV(VpastFlat, n, dKV)
	if err != nil {
		return nil, nil, nil, err
	}
	y, Knews, Vnews, err := s.GQAStackTrit(layers, x, Kpasts, Vpasts, dim, nHeads, nKVHeads, dh, pos, base, bitplane, eps)
	if err != nil {
		return nil, nil, nil, err
	}
	return y, FlattenLayerKV(Knews), FlattenLayerKV(Vnews), nil
}

// FinalNorm applies RMSNorm with an optional named gain tensor (empty → unit gain).
func (s *WeightSession) FinalNorm(gainName string, x []float32, dim int, eps float64) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("nil session")
	}
	if dim <= 0 || len(x) < dim {
		return nil, fmt.Errorf("final_norm: bad dim")
	}
	gain, err := s.loadGain(gainName, dim)
	if err != nil {
		return nil, fmt.Errorf("final_norm gain %q: %w", gainName, err)
	}
	return RMSNormF32(x[:dim], eps, gain), nil
}

// --- v1.15 SwiGLU residual GQA + multi-token prefill ---

// ForwardGQASwiGLUBlockTrit is ForwardGQABlockTrit with SwiGLU FFN:
//   Down(SiLU(Gate(hn)) ⊙ Up(hn)). Gate/Up/Down required.
func (s *WeightSession) ForwardGQASwiGLUBlockTrit(
	spec AttnLayerSpec,
	x, Kpast, Vpast []float32,
	dim, nHeads, nKVHeads, dh int,
	pos, base float64,
	bitplane bool,
	eps float64,
) (y, Knew, Vnew []float32, err error) {
	if s == nil {
		return nil, nil, nil, fmt.Errorf("nil session")
	}
	if dim <= 0 || len(x) < dim {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_block_trit: bad dim")
	}
	if nHeads <= 0 || nKVHeads <= 0 || dh <= 0 {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_block_trit: bad heads")
	}
	if nHeads%nKVHeads != 0 {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_block_trit: nHeads must be multiple of nKVHeads")
	}
	if nHeads*dh != dim {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_block_trit: dim=%d != nHeads*dh=%d", dim, nHeads*dh)
	}
	if spec.Q == "" || spec.K == "" || spec.V == "" {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_block_trit: Q/K/V required")
	}
	if spec.Gate == "" || spec.Up == "" || spec.Down == "" {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_block_trit: Gate/Up/Down required")
	}
	if spec.Hidden <= 0 {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_block_trit: hidden must be > 0")
	}
	xIn := x[:dim]
	attnGain, err := s.loadGain(spec.AttnGain, dim)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_block_trit attn gain %q: %w", spec.AttnGain, err)
	}
	xn := RMSNormF32(xIn, eps, attnGain)
	a, Knew, Vnew, err := s.SessionAttnGQATrit(
		spec.Q, spec.K, spec.V, spec.O,
		xn, Kpast, Vpast,
		dim, nHeads, nKVHeads, dh, pos, base, bitplane,
	)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_block_trit attn: %w", err)
	}
	if len(a) < dim {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_block_trit: attn out len %d < dim %d", len(a), dim)
	}
	hRes := make([]float32, dim)
	for i := 0; i < dim; i++ {
		hRes[i] = xIn[i] + a[i]
	}
	ffnGain, err := s.loadGain(spec.FFNGain, dim)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_block_trit ffn gain %q: %w", spec.FFNGain, err)
	}
	hn := RMSNormF32(hRes, eps, ffnGain)
	gate, _, err := s.LinearTritCached(spec.Gate, hn, spec.Hidden, dim, bitplane)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_block_trit gate %q: %w", spec.Gate, err)
	}
	up, _, err := s.LinearTritCached(spec.Up, hn, spec.Hidden, dim, bitplane)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_block_trit up %q: %w", spec.Up, err)
	}
	act, err := SwiGLUF32(gate, up)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_block_trit swiglu: %w", err)
	}
	down, _, err := s.LinearTritCached(spec.Down, act, dim, spec.Hidden, bitplane)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_block_trit down %q: %w", spec.Down, err)
	}
	y = make([]float32, dim)
	for i := 0; i < dim; i++ {
		y[i] = hRes[i] + down[i]
	}
	return y, Knew, Vnew, nil
}

// GQASwiGLUStackTrit chains ForwardGQASwiGLUBlockTrit over layers.
func (s *WeightSession) GQASwiGLUStackTrit(
	layers []AttnLayerSpec,
	x []float32,
	Kpasts, Vpasts [][]float32,
	dim, nHeads, nKVHeads, dh int,
	pos, base float64,
	bitplane bool,
	eps float64,
) (y []float32, Knews, Vnews [][]float32, err error) {
	if s == nil {
		return nil, nil, nil, fmt.Errorf("nil session")
	}
	if len(layers) == 0 {
		return nil, nil, nil, fmt.Errorf("gqa_swiglu_stack_trit: empty layers")
	}
	if dim <= 0 || len(x) < dim {
		return nil, nil, nil, fmt.Errorf("gqa_swiglu_stack_trit: bad dim")
	}
	cur := make([]float32, dim)
	copy(cur, x[:dim])
	Knews = make([][]float32, len(layers))
	Vnews = make([][]float32, len(layers))
	for i, spec := range layers {
		var kp, vp []float32
		if i < len(Kpasts) {
			kp = Kpasts[i]
		}
		if i < len(Vpasts) {
			vp = Vpasts[i]
		}
		next, kn, vn, e := s.ForwardGQASwiGLUBlockTrit(spec, cur, kp, vp, dim, nHeads, nKVHeads, dh, pos, base, bitplane, eps)
		if e != nil {
			return nil, nil, nil, fmt.Errorf("gqa_swiglu_stack_trit layer %d: %w", i, e)
		}
		cur = next
		Knews[i] = kn
		Vnews[i] = vn
	}
	return cur, Knews, Vnews, nil
}

// parseGQASwiGLULayers builds layer specs from CSV name lists (gate required).
func parseGQASwiGLULayers(
	qCSV, kCSV, vCSV, oCSV, gateCSV, upCSV, downCSV, attnGainCSV, ffnGainCSV string,
	hidden int,
) ([]AttnLayerSpec, error) {
	qs := splitCSV(qCSV)
	ks := splitCSV(kCSV)
	vs := splitCSV(vCSV)
	os_ := splitCSV(oCSV)
	gates := splitCSV(gateCSV)
	ups := splitCSV(upCSV)
	downs := splitCSV(downCSV)
	ag := splitCSV(attnGainCSV)
	fg := splitCSV(ffnGainCSV)
	n := len(qs)
	if n == 0 || len(ks) != n || len(vs) != n || len(gates) != n || len(ups) != n || len(downs) != n {
		return nil, fmt.Errorf("gqa_swiglu_stack_trit: Q/K/V/Gate/Up/Down CSVs must share non-zero length")
	}
	layers := make([]AttnLayerSpec, n)
	for i := 0; i < n; i++ {
		spec := AttnLayerSpec{
			Q: qs[i], K: ks[i], V: vs[i],
			Gate: gates[i], Up: ups[i], Down: downs[i],
			Hidden: hidden,
		}
		if i < len(os_) {
			spec.O = os_[i]
		}
		if i < len(ag) {
			spec.AttnGain = ag[i]
		}
		if i < len(fg) {
			spec.FFNGain = fg[i]
		}
		layers[i] = spec
	}
	return layers, nil
}

// GQASwiGLUStackTritCSV is the language-friendly multi-layer SwiGLU GQA surface.
// gateCSV required; compact layer-major KV flats (dKV = nKVHeads*dh).
func (s *WeightSession) GQASwiGLUStackTritCSV(
	qCSV, kCSV, vCSV, oCSV, gateCSV, upCSV, downCSV, attnGainCSV, ffnGainCSV string,
	x, KpastFlat, VpastFlat []float32,
	dim, hidden, nHeads, nKVHeads, dh int,
	pos, base float64,
	bitplane bool,
	eps float64,
) (y, KnewFlat, VnewFlat []float32, err error) {
	if nHeads <= 0 || nKVHeads <= 0 || dh <= 0 {
		return nil, nil, nil, fmt.Errorf("gqa_swiglu_stack_trit: bad heads")
	}
	layers, err := parseGQASwiGLULayers(qCSV, kCSV, vCSV, oCSV, gateCSV, upCSV, downCSV, attnGainCSV, ffnGainCSV, hidden)
	if err != nil {
		return nil, nil, nil, err
	}
	dKV := nKVHeads * dh
	n := len(layers)
	Kpasts, err := SplitLayerKV(KpastFlat, n, dKV)
	if err != nil {
		return nil, nil, nil, err
	}
	Vpasts, err := SplitLayerKV(VpastFlat, n, dKV)
	if err != nil {
		return nil, nil, nil, err
	}
	y, Knews, Vnews, err := s.GQASwiGLUStackTrit(layers, x, Kpasts, Vpasts, dim, nHeads, nKVHeads, dh, pos, base, bitplane, eps)
	if err != nil {
		return nil, nil, nil, err
	}
	return y, FlattenLayerKV(Knews), FlattenLayerKV(Vnews), nil
}

// PrefillGQASwiGLUStackTritCSV runs multi-token sequential prefill through the
// SwiGLU GQA stack. xs is nTok*dim row-major; startPos is rope pos of xs[0].
// Each token advances the compact layer-major KV (causal by construction).
// Returns Yflat (nTok*dim), KnewFlat, VnewFlat, yLast (= Yflat[(nTok-1)*dim:]).
func (s *WeightSession) PrefillGQASwiGLUStackTritCSV(
	qCSV, kCSV, vCSV, oCSV, gateCSV, upCSV, downCSV, attnGainCSV, ffnGainCSV string,
	xs, KpastFlat, VpastFlat []float32,
	dim, hidden, nHeads, nKVHeads, dh int,
	startPos, base float64,
	bitplane bool,
	eps float64,
) (Yflat, KnewFlat, VnewFlat, yLast []float32, err error) {
	if s == nil {
		return nil, nil, nil, nil, fmt.Errorf("nil session")
	}
	if dim <= 0 {
		return nil, nil, nil, nil, fmt.Errorf("gqa_swiglu_prefill: bad dim")
	}
	if len(xs) == 0 || len(xs)%dim != 0 {
		return nil, nil, nil, nil, fmt.Errorf("gqa_swiglu_prefill: xs len %d not multiple of dim %d", len(xs), dim)
	}
	nTok := len(xs) / dim
	layers, err := parseGQASwiGLULayers(qCSV, kCSV, vCSV, oCSV, gateCSV, upCSV, downCSV, attnGainCSV, ffnGainCSV, hidden)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	if nHeads <= 0 || nKVHeads <= 0 || dh <= 0 {
		return nil, nil, nil, nil, fmt.Errorf("gqa_swiglu_prefill: bad heads")
	}
	dKV := nKVHeads * dh
	n := len(layers)
	Kpasts, err := SplitLayerKV(KpastFlat, n, dKV)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	Vpasts, err := SplitLayerKV(VpastFlat, n, dKV)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	Yflat = make([]float32, 0, nTok*dim)
	kCur, vCur := Kpasts, Vpasts
	var y []float32
	for t := 0; t < nTok; t++ {
		x := xs[t*dim : (t+1)*dim]
		pos := startPos + float64(t)
		y, kCur, vCur, err = s.GQASwiGLUStackTrit(layers, x, kCur, vCur, dim, nHeads, nKVHeads, dh, pos, base, bitplane, eps)
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("gqa_swiglu_prefill token %d: %w", t, err)
		}
		Yflat = append(Yflat, y...)
	}
	KnewFlat = FlattenLayerKV(kCur)
	VnewFlat = FlattenLayerKV(vCur)
	yLast = make([]float32, dim)
	copy(yLast, y)
	return Yflat, KnewFlat, VnewFlat, yLast, nil
}

// --- v1.16 tied embeddings + ALiBi residual decode ---

// EmbedRow loads one token row from a V×dim row-major emb tensor.
func (s *WeightSession) EmbedRow(name string, idx, dim int) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("nil session")
	}
	if dim <= 0 {
		return nil, fmt.Errorf("embed_row: bad dim")
	}
	if idx < 0 {
		return nil, fmt.Errorf("embed_row: negative idx")
	}
	nElem, _, err := s.tensorNElem(name)
	if err != nil {
		return nil, err
	}
	if nElem%dim != 0 {
		return nil, fmt.Errorf("embed_row: nElem=%d not divisible by dim=%d", nElem, dim)
	}
	V := nElem / dim
	if idx >= V {
		return nil, fmt.Errorf("embed_row: idx %d OOB vocab %d", idx, V)
	}
	return s.LoadF32Window(name, idx*dim, dim)
}

// LogitsTied computes logits = E @ h for tied embedding table E (vocab×dim
// row-major). Equivalent to lm_head = E^T when E is stored as rows = tokens.
func (s *WeightSession) LogitsTied(embName string, h []float32, dim int) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("nil session")
	}
	if dim <= 0 || len(h) < dim {
		return nil, fmt.Errorf("logits_tied: bad dim")
	}
	nElem, _, err := s.tensorNElem(embName)
	if err != nil {
		return nil, err
	}
	if nElem%dim != 0 {
		return nil, fmt.Errorf("logits_tied: nElem=%d not divisible by dim=%d", nElem, dim)
	}
	V := nElem / dim
	// y[v] = sum_d E[v,d] * h[d] — use Linear with M=V K=dim
	return s.Linear(embName, h[:dim], V, dim, 0)
}

// LogitsTiedTrit is LogitsTied via trit-cached linear (auto-prepack emb rows).
func (s *WeightSession) LogitsTiedTrit(embName string, h []float32, dim int, bitplane bool) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("nil session")
	}
	if dim <= 0 || len(h) < dim {
		return nil, fmt.Errorf("logits_tied_trit: bad dim")
	}
	nElem, _, err := s.tensorNElem(embName)
	if err != nil {
		return nil, err
	}
	if nElem%dim != 0 {
		return nil, fmt.Errorf("logits_tied_trit: nElem=%d not divisible by dim=%d", nElem, dim)
	}
	V := nElem / dim
	y, _, err := s.LinearTritCached(embName, h[:dim], V, dim, bitplane)
	return y, err
}

// ForwardGQASwiGLUBlockTritAlibi is ForwardGQASwiGLUBlockTrit with ALiBi (no RoPE).
// Full history (window=0).
func (s *WeightSession) ForwardGQASwiGLUBlockTritAlibi(
	spec AttnLayerSpec,
	x, Kpast, Vpast []float32,
	dim, nHeads, nKVHeads, dh int,
	pos float64,
	bitplane bool,
	eps float64,
	slopes []float32,
) (y, Knew, Vnew []float32, err error) {
	return s.ForwardGQASwiGLUBlockTritAlibiWindow(spec, x, Kpast, Vpast, dim, nHeads, nKVHeads, dh, pos, bitplane, eps, slopes, 0)
}

// ForwardGQASwiGLUBlockTritAlibiWindow is ALiBi SwiGLU block with sliding window (v1.18).
// window<=0 → full history.
func (s *WeightSession) ForwardGQASwiGLUBlockTritAlibiWindow(
	spec AttnLayerSpec,
	x, Kpast, Vpast []float32,
	dim, nHeads, nKVHeads, dh int,
	pos float64,
	bitplane bool,
	eps float64,
	slopes []float32,
	window int,
) (y, Knew, Vnew []float32, err error) {
	if s == nil {
		return nil, nil, nil, fmt.Errorf("nil session")
	}
	if dim <= 0 || len(x) < dim {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_alibi: bad dim")
	}
	if nHeads <= 0 || nKVHeads <= 0 || dh <= 0 {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_alibi: bad heads")
	}
	if nHeads%nKVHeads != 0 {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_alibi: nHeads must be multiple of nKVHeads")
	}
	if nHeads*dh != dim {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_alibi: dim=%d != nHeads*dh=%d", dim, nHeads*dh)
	}
	if spec.Q == "" || spec.K == "" || spec.V == "" {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_alibi: Q/K/V required")
	}
	if spec.Gate == "" || spec.Up == "" || spec.Down == "" {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_alibi: Gate/Up/Down required")
	}
	if spec.Hidden <= 0 {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_alibi: hidden must be > 0")
	}
	xIn := x[:dim]
	attnGain, err := s.loadGain(spec.AttnGain, dim)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_alibi attn gain %q: %w", spec.AttnGain, err)
	}
	xn := RMSNormF32(xIn, eps, attnGain)
	a, Knew, Vnew, err := s.SessionAttnGQATritAlibiWindow(
		spec.Q, spec.K, spec.V, spec.O,
		xn, Kpast, Vpast,
		dim, nHeads, nKVHeads, dh, pos, bitplane, slopes, window,
	)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_alibi attn: %w", err)
	}
	if len(a) < dim {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_alibi: attn out len %d < dim %d", len(a), dim)
	}
	hRes := make([]float32, dim)
	for i := 0; i < dim; i++ {
		hRes[i] = xIn[i] + a[i]
	}
	ffnGain, err := s.loadGain(spec.FFNGain, dim)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_alibi ffn gain %q: %w", spec.FFNGain, err)
	}
	hn := RMSNormF32(hRes, eps, ffnGain)
	gate, _, err := s.LinearTritCached(spec.Gate, hn, spec.Hidden, dim, bitplane)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_alibi gate %q: %w", spec.Gate, err)
	}
	up, _, err := s.LinearTritCached(spec.Up, hn, spec.Hidden, dim, bitplane)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_alibi up %q: %w", spec.Up, err)
	}
	act, err := SwiGLUF32(gate, up)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_alibi swiglu: %w", err)
	}
	down, _, err := s.LinearTritCached(spec.Down, act, dim, spec.Hidden, bitplane)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("forward_gqa_swiglu_alibi down %q: %w", spec.Down, err)
	}
	y = make([]float32, dim)
	for i := 0; i < dim; i++ {
		y[i] = hRes[i] + down[i]
	}
	return y, Knew, Vnew, nil
}

// GQASwiGLUStackTritAlibi chains ALiBi SwiGLU GQA blocks (full history).
func (s *WeightSession) GQASwiGLUStackTritAlibi(
	layers []AttnLayerSpec,
	x []float32,
	Kpasts, Vpasts [][]float32,
	dim, nHeads, nKVHeads, dh int,
	pos float64,
	bitplane bool,
	eps float64,
	slopes []float32,
) (y []float32, Knews, Vnews [][]float32, err error) {
	return s.GQASwiGLUStackTritAlibiWindow(layers, x, Kpasts, Vpasts, dim, nHeads, nKVHeads, dh, pos, bitplane, eps, slopes, 0)
}

// GQASwiGLUStackTritAlibiWindow chains ALiBi SwiGLU GQA with sliding window (v1.18).
// window<=0 → full history.
func (s *WeightSession) GQASwiGLUStackTritAlibiWindow(
	layers []AttnLayerSpec,
	x []float32,
	Kpasts, Vpasts [][]float32,
	dim, nHeads, nKVHeads, dh int,
	pos float64,
	bitplane bool,
	eps float64,
	slopes []float32,
	window int,
) (y []float32, Knews, Vnews [][]float32, err error) {
	if s == nil {
		return nil, nil, nil, fmt.Errorf("nil session")
	}
	if len(layers) == 0 {
		return nil, nil, nil, fmt.Errorf("gqa_swiglu_stack_alibi: empty layers")
	}
	if dim <= 0 || len(x) < dim {
		return nil, nil, nil, fmt.Errorf("gqa_swiglu_stack_alibi: bad dim")
	}
	cur := make([]float32, dim)
	copy(cur, x[:dim])
	Knews = make([][]float32, len(layers))
	Vnews = make([][]float32, len(layers))
	for i, spec := range layers {
		var kp, vp []float32
		if i < len(Kpasts) {
			kp = Kpasts[i]
		}
		if i < len(Vpasts) {
			vp = Vpasts[i]
		}
		next, kn, vn, e := s.ForwardGQASwiGLUBlockTritAlibiWindow(spec, cur, kp, vp, dim, nHeads, nKVHeads, dh, pos, bitplane, eps, slopes, window)
		if e != nil {
			return nil, nil, nil, fmt.Errorf("gqa_swiglu_stack_alibi layer %d: %w", i, e)
		}
		cur = next
		Knews[i] = kn
		Vnews[i] = vn
	}
	return cur, Knews, Vnews, nil
}

// GQASwiGLUStackTritAlibiCSV is the language surface for ALiBi SwiGLU GQA.
func (s *WeightSession) GQASwiGLUStackTritAlibiCSV(
	qCSV, kCSV, vCSV, oCSV, gateCSV, upCSV, downCSV, attnGainCSV, ffnGainCSV string,
	x, KpastFlat, VpastFlat []float32,
	dim, hidden, nHeads, nKVHeads, dh int,
	pos float64,
	bitplane bool,
	eps float64,
) (y, KnewFlat, VnewFlat []float32, err error) {
	return s.GQASwiGLUStackTritAlibiWindowCSV(
		qCSV, kCSV, vCSV, oCSV, gateCSV, upCSV, downCSV, attnGainCSV, ffnGainCSV,
		x, KpastFlat, VpastFlat, dim, hidden, nHeads, nKVHeads, dh, pos, bitplane, eps, 0,
	)
}

// GQASwiGLUStackTritAlibiWindowCSV is ALiBi stack with window (v1.18).
func (s *WeightSession) GQASwiGLUStackTritAlibiWindowCSV(
	qCSV, kCSV, vCSV, oCSV, gateCSV, upCSV, downCSV, attnGainCSV, ffnGainCSV string,
	x, KpastFlat, VpastFlat []float32,
	dim, hidden, nHeads, nKVHeads, dh int,
	pos float64,
	bitplane bool,
	eps float64,
	window int,
) (y, KnewFlat, VnewFlat []float32, err error) {
	if nHeads <= 0 || nKVHeads <= 0 || dh <= 0 {
		return nil, nil, nil, fmt.Errorf("gqa_swiglu_stack_alibi: bad heads")
	}
	layers, err := parseGQASwiGLULayers(qCSV, kCSV, vCSV, oCSV, gateCSV, upCSV, downCSV, attnGainCSV, ffnGainCSV, hidden)
	if err != nil {
		return nil, nil, nil, err
	}
	dKV := nKVHeads * dh
	n := len(layers)
	Kpasts, err := SplitLayerKV(KpastFlat, n, dKV)
	if err != nil {
		return nil, nil, nil, err
	}
	Vpasts, err := SplitLayerKV(VpastFlat, n, dKV)
	if err != nil {
		return nil, nil, nil, err
	}
	y, Knews, Vnews, err := s.GQASwiGLUStackTritAlibiWindow(layers, x, Kpasts, Vpasts, dim, nHeads, nKVHeads, dh, pos, bitplane, eps, nil, window)
	if err != nil {
		return nil, nil, nil, err
	}
	return y, FlattenLayerKV(Knews), FlattenLayerKV(Vnews), nil
}

// PrefillGQASwiGLUStackTritAlibiCSV multi-token sequential ALiBi prefill.
func (s *WeightSession) PrefillGQASwiGLUStackTritAlibiCSV(
	qCSV, kCSV, vCSV, oCSV, gateCSV, upCSV, downCSV, attnGainCSV, ffnGainCSV string,
	xs, KpastFlat, VpastFlat []float32,
	dim, hidden, nHeads, nKVHeads, dh int,
	startPos float64,
	bitplane bool,
	eps float64,
) (Yflat, KnewFlat, VnewFlat, yLast []float32, err error) {
	return s.PrefillGQASwiGLUStackTritAlibiWindowCSV(
		qCSV, kCSV, vCSV, oCSV, gateCSV, upCSV, downCSV, attnGainCSV, ffnGainCSV,
		xs, KpastFlat, VpastFlat, dim, hidden, nHeads, nKVHeads, dh, startPos, bitplane, eps, 0,
	)
}

// PrefillGQASwiGLUStackTritAlibiWindowCSV multi-token ALiBi prefill with window (v1.18).
func (s *WeightSession) PrefillGQASwiGLUStackTritAlibiWindowCSV(
	qCSV, kCSV, vCSV, oCSV, gateCSV, upCSV, downCSV, attnGainCSV, ffnGainCSV string,
	xs, KpastFlat, VpastFlat []float32,
	dim, hidden, nHeads, nKVHeads, dh int,
	startPos float64,
	bitplane bool,
	eps float64,
	window int,
) (Yflat, KnewFlat, VnewFlat, yLast []float32, err error) {
	if s == nil {
		return nil, nil, nil, nil, fmt.Errorf("nil session")
	}
	if dim <= 0 {
		return nil, nil, nil, nil, fmt.Errorf("gqa_swiglu_prefill_alibi: bad dim")
	}
	if len(xs) == 0 || len(xs)%dim != 0 {
		return nil, nil, nil, nil, fmt.Errorf("gqa_swiglu_prefill_alibi: xs len %d not multiple of dim %d", len(xs), dim)
	}
	nTok := len(xs) / dim
	layers, err := parseGQASwiGLULayers(qCSV, kCSV, vCSV, oCSV, gateCSV, upCSV, downCSV, attnGainCSV, ffnGainCSV, hidden)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	if nHeads <= 0 || nKVHeads <= 0 || dh <= 0 {
		return nil, nil, nil, nil, fmt.Errorf("gqa_swiglu_prefill_alibi: bad heads")
	}
	dKV := nKVHeads * dh
	n := len(layers)
	Kpasts, err := SplitLayerKV(KpastFlat, n, dKV)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	Vpasts, err := SplitLayerKV(VpastFlat, n, dKV)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	Yflat = make([]float32, 0, nTok*dim)
	kCur, vCur := Kpasts, Vpasts
	var y []float32
	for t := 0; t < nTok; t++ {
		x := xs[t*dim : (t+1)*dim]
		pos := startPos + float64(t)
		y, kCur, vCur, err = s.GQASwiGLUStackTritAlibiWindow(layers, x, kCur, vCur, dim, nHeads, nKVHeads, dh, pos, bitplane, eps, nil, window)
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("gqa_swiglu_prefill_alibi token %d: %w", t, err)
		}
		Yflat = append(Yflat, y...)
	}
	KnewFlat = FlattenLayerKV(kCur)
	VnewFlat = FlattenLayerKV(vCur)
	yLast = make([]float32, dim)
	copy(yLast, y)
	return Yflat, KnewFlat, VnewFlat, yLast, nil
}
