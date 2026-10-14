// Package qwen runs a Qwen2 GGUF with Trin's weight session.
// Stargate uses it when the scan model is the Opus Qwen2.5-0.5B file
// instead of the Starlight CNN.
package qwen

import (
	"context"
	"fmt"

	"github.com/macroadster/trin/internal/weights"
)

// Session is one open Qwen2 GGUF.
type Session struct {
	ws *weights.WeightSession
	m  *weights.Qwen2
}

// Open loads a Qwen2 GGUF. The file stays mapped until Close.
func Open(path string) (*Session, error) {
	if path == "" {
		return nil, fmt.Errorf("qwen.Open: empty path")
	}
	ws, err := weights.OpenSession(path)
	if err != nil {
		return nil, err
	}
	m, err := weights.LoadQwen2(ws)
	if err != nil {
		_ = weights.CloseSession(ws.ID)
		return nil, err
	}
	return &Session{ws: ws, m: m}, nil
}

// Close releases the weight file.
func (s *Session) Close() error {
	if s == nil || s.ws == nil {
		return nil
	}
	id := s.ws.ID
	s.ws = nil
	s.m = nil
	return weights.CloseSession(id)
}

// Generate runs a greedy completion of prompt for at most n new tokens.
func (s *Session) Generate(ctx context.Context, prompt string, n int) (string, error) {
	if s == nil || s.m == nil {
		return "", fmt.Errorf("qwen: session closed")
	}
	return s.m.Generate(ctx, prompt, n)
}
