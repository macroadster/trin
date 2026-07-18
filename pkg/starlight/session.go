// Package starlight is the public Trin API for BalancedStarlightDetector
// inference over a GGUF weight session.
//
// Stargate integration — replace ForwardStarlight body:
//
//	import "github.com/macroadster/trin/pkg/starlight"
//
//	sess, err := starlight.Open(ggufPath) // or reuse a process-wide session
//	if err != nil { return err }
//	defer sess.Close()
//
//	out, err := sess.Forward(starlight.Inputs{
//	    Pixel: u.PixelFlat, Meta: u.MetaFlat, Alpha: u.AlphaFlat,
//	    LSB: u.LSBFlat, Palette: u.PaletteFlat, PaletteLSB: u.PaletteLSBFlat,
//	    FormatFeatures: u.FormatFeatures, ContentFeatures: u.ContentFeatures,
//	    H: u.H, W: u.W, MetaWeight: 0.3, // 0 defaults to 0.3
//	})
//	// out.Stego, out.Method, out.BitOrder, out.Embedding
//
// Do NOT import github.com/macroadster/trin/internal/* from Stargate.
// Keep MinimalGGUF open for other heads or switch fully to starlight.Open.
package starlight

import (
	"fmt"

	"github.com/macroadster/trin/internal/weights"
)

// Session wraps a GGUF weight session for BalancedStarlightDetector inference.
type Session struct {
	ws *weights.WeightSession
}

// Open opens a Starlight GGUF (or any weight session path) and retains mmap.
func Open(path string) (*Session, error) {
	if path == "" {
		return nil, fmt.Errorf("starlight.Open: empty path")
	}
	s, err := weights.OpenSession(path)
	if err != nil {
		return nil, err
	}
	return &Session{ws: s}, nil
}

// Close releases the weight session (idempotent if already closed via id).
func (s *Session) Close() error {
	if s == nil || s.ws == nil {
		return nil
	}
	id := s.ws.ID
	s.ws = nil
	return weights.CloseSession(id)
}

// List returns on-disk tensor names.
func (s *Session) List() ([]string, error) {
	if s == nil || s.ws == nil {
		return nil, fmt.Errorf("starlight: session closed")
	}
	return s.ws.List()
}

// Meta returns a GGUF metadata string value (empty if missing).
func (s *Session) Meta(key string) string {
	if s == nil || s.ws == nil {
		return ""
	}
	return s.ws.Meta(key)
}

// Inputs holds NCHW/flat inputs for batch=1 (matches weights.StarlightInputs).
type Inputs struct {
	Pixel, Meta, Alpha, LSB, Palette, PaletteLSB, FormatFeatures, ContentFeatures []float32
	H, W                                                                          int
	MetaWeight                                                                    float32 // 0 => 0.3
}

// Outputs holds detector head outputs (batch=1).
type Outputs struct {
	Stego, Method, BitOrder, Embedding []float32
}

// Forward runs BalancedStarlightDetector eval (dropout=identity, BN eps=1e-5).
func (s *Session) Forward(in Inputs) (*Outputs, error) {
	if s == nil || s.ws == nil {
		return nil, fmt.Errorf("starlight: session closed")
	}
	out, err := weights.StarlightForwardSession(s.ws, weights.StarlightInputs{
		Pixel:           in.Pixel,
		Meta:            in.Meta,
		Alpha:           in.Alpha,
		LSB:             in.LSB,
		Palette:         in.Palette,
		PaletteLSB:      in.PaletteLSB,
		FormatFeatures:  in.FormatFeatures,
		ContentFeatures: in.ContentFeatures,
		H:               in.H,
		W:               in.W,
		MetaWeight:      in.MetaWeight,
	})
	if err != nil {
		return nil, err
	}
	return &Outputs{
		Stego:     append([]float32(nil), out.Stego...),
		Method:    append([]float32(nil), out.Method...),
		BitOrder:  append([]float32(nil), out.BitOrder...),
		Embedding: append([]float32(nil), out.Embedding...),
	}, nil
}

// Concat flattens heads to [stego(1)|method(5)|bit_order(3)|embedding(64)] = 73.
func (o *Outputs) Concat() []float32 {
	if o == nil {
		return nil
	}
	out := make([]float32, 0, 73)
	out = append(out, o.Stego...)
	out = append(out, o.Method...)
	out = append(out, o.BitOrder...)
	out = append(out, o.Embedding...)
	return out
}
