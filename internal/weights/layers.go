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
//   f = LinearTritCached(Down, SiLU(LinearTritCached(Up, hn)))
//   y = h + f
// Q/K/V are dOut×dIn (dOut==dim); O is dim×dim when non-empty; Up is hidden×dim;
// Down is dim×hidden. Gains are length-dim vectors (empty → unit gain).
type AttnLayerSpec struct {
	Q, K, V, O   string
	AttnGain     string
	FFNGain      string
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
