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
	split := func(s string) []string {
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
	return s.LayerStack(split(wCSV), split(gainCSV), split(modeCSV), x, dim, tileRows, eps)
}
