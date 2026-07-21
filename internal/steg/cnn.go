package steg

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/macroadster/trin/internal/weights"
)

// TinyStegCNNDims is the default export shape for demos/weights/tiny_steg_cnn.gguf.
const (
	DefaultCNNInK     = 64
	DefaultCNNHidden0 = 32
	DefaultCNNHidden1 = 16
	DefaultCNNNClass  = 2
)

// CNNConfig describes a GGUF-exported Starlight scanner CNN lowered to linears.
// This is the stargate-side contract: open GGUF → prepack trit → forward, no Python.
type CNNConfig struct {
	InK     int
	Hidden0 int
	Hidden1 int
	NClass  int
	L0      string
	L0b     string
	L1      string
	L1b     string
	Head    string
	Headb   string
	// Threshold on (stego - clean) for HasPayload decision.
	Margin float64
}

// DefaultCNNConfig matches scripts/gen_tiny_steg_cnn_gguf.go.
func DefaultCNNConfig() CNNConfig {
	return CNNConfig{
		InK: DefaultCNNInK, Hidden0: DefaultCNNHidden0, Hidden1: DefaultCNNHidden1,
		NClass: DefaultCNNNClass,
		L0: "L0", L0b: "L0_b", L1: "L1", L1b: "L1_b", Head: "Head", Headb: "Head_b",
		Margin: 0,
	}
}

// CNNConfigFromSession reads starlight.steg.* meta when present; else defaults.
func CNNConfigFromSession(s *weights.WeightSession) CNNConfig {
	cfg := DefaultCNNConfig()
	if s == nil {
		return cfg
	}
	if v := s.Meta("starlight.steg.in_k"); v != "" {
		if n, e := strconv.Atoi(strings.TrimSpace(v)); e == nil && n > 0 {
			cfg.InK = n
		}
	}
	if v := s.Meta("starlight.steg.hidden0"); v != "" {
		if n, e := strconv.Atoi(strings.TrimSpace(v)); e == nil && n > 0 {
			cfg.Hidden0 = n
		}
	}
	if v := s.Meta("starlight.steg.hidden1"); v != "" {
		if n, e := strconv.Atoi(strings.TrimSpace(v)); e == nil && n > 0 {
			cfg.Hidden1 = n
		}
	}
	if v := s.Meta("starlight.steg.n_class"); v != "" {
		if n, e := strconv.Atoi(strings.TrimSpace(v)); e == nil && n > 0 {
			cfg.NClass = n
		}
	}
	return cfg
}

// CNNResult is a pure-Go CNN scan decision (stargate inference output).
type CNNResult struct {
	CleanScore float64
	StegoScore float64
	Margin     float64
	HasPayload bool
	Decision   string // "clean" | "stego"
	InK        int
	Bitplane   bool
	Notes      string
}

// PrepackCNNTrit prepacks L0/L1/Head for repeated scans (call once per process).
func PrepackCNNTrit(s *weights.WeightSession, cfg CNNConfig) (built, hits int, err error) {
	if s == nil {
		return 0, 0, fmt.Errorf("steg: nil session")
	}
	if cfg.InK == 0 {
		cfg = DefaultCNNConfig()
	}
	specs := []struct {
		name string
		K    int
	}{
		{cfg.L0, cfg.InK},
		{cfg.L1, cfg.Hidden0},
		{cfg.Head, cfg.Hidden1},
	}
	for _, sp := range specs {
		b, e := s.PrepackTrit(sp.name, sp.K)
		if e != nil {
			return built, hits, e
		}
		if b {
			built++
		} else {
			hits++
		}
	}
	return built, hits, nil
}

// addBias adds b into y in-place (broadcast length min).
func addBias(y, b []float32) {
	n := len(y)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		y[i] += b[i]
	}
}

// loadBias loads optional bias vector; missing/empty → zeros of size n.
func loadBias(s *weights.WeightSession, name string, n int) ([]float32, error) {
	if name == "" || n <= 0 {
		return make([]float32, n), nil
	}
	// Try window load; if tensor missing, zeros.
	b, err := s.LoadF32Window(name, 0, n)
	if err != nil {
		// soft-missing for optional biases
		if strings.Contains(err.Error(), "unknown") || strings.Contains(err.Error(), "not found") ||
			strings.Contains(err.Error(), "missing") {
			return make([]float32, n), nil
		}
		// still return zeros for toy robustness
		return make([]float32, n), nil
	}
	if len(b) < n {
		out := make([]float32, n)
		copy(out, b)
		return out, nil
	}
	return b[:n], nil
}

// ForwardStegCNN runs the GGUF CNN graph with trit-prepacked linears + SiLU.
// bitplane selects packed bitplane matvec (preferred in stargate) vs scalar int8.
func ForwardStegCNN(s *weights.WeightSession, x []float32, cfg CNNConfig, bitplane bool) (CNNResult, error) {
	res := CNNResult{Bitplane: bitplane, InK: cfg.InK}
	if s == nil {
		return res, fmt.Errorf("steg: nil session")
	}
	if cfg.InK == 0 {
		cfg = DefaultCNNConfig()
		res.InK = cfg.InK
	}
	if len(x) < cfg.InK {
		return res, fmt.Errorf("steg: feature short need %d have %d", cfg.InK, len(x))
	}
	xIn := x[:cfg.InK]

	h0, _, err := s.LinearTritCached(cfg.L0, xIn, cfg.Hidden0, cfg.InK, bitplane)
	if err != nil {
		return res, fmt.Errorf("steg L0: %w", err)
	}
	b0, _ := loadBias(s, cfg.L0b, cfg.Hidden0)
	addBias(h0, b0)
	h0 = weights.SiLUF32(h0)

	h1, _, err := s.LinearTritCached(cfg.L1, h0, cfg.Hidden1, cfg.Hidden0, bitplane)
	if err != nil {
		return res, fmt.Errorf("steg L1: %w", err)
	}
	b1, _ := loadBias(s, cfg.L1b, cfg.Hidden1)
	addBias(h1, b1)
	h1 = weights.SiLUF32(h1)

	logits, _, err := s.LinearTritCached(cfg.Head, h1, cfg.NClass, cfg.Hidden1, bitplane)
	if err != nil {
		return res, fmt.Errorf("steg Head: %w", err)
	}
	hb, _ := loadBias(s, cfg.Headb, cfg.NClass)
	addBias(logits, hb)

	if len(logits) < 2 {
		return res, fmt.Errorf("steg: expected 2 logits got %d", len(logits))
	}
	res.CleanScore = float64(logits[0])
	res.StegoScore = float64(logits[1])
	res.Margin = res.StegoScore - res.CleanScore
	if res.Margin > cfg.Margin {
		res.HasPayload = true
		res.Decision = "stego"
	} else {
		res.HasPayload = false
		res.Decision = "clean"
	}
	return res, nil
}

// FeaturesFromImage builds the K-length feature vector the CNN expects:
// blue-channel residual (scan-order first-diff) sampled into K lanes +
// normalized raw blue energy. Matches the im2col export story: residual patch.
func FeaturesFromImage(img []RGB, K int) []float32 {
	if K < 1 {
		return nil
	}
	x := make([]float32, K)
	if len(img) == 0 {
		return x
	}
	// half residual, half level
	half := K / 2
	if half < 1 {
		half = K
	}
	var prev float32 = float32(img[0].B)
	for i := 0; i < half; i++ {
		c := float32(img[i%len(img)].B)
		x[i] = (c - prev) / 127.5
		prev = c
	}
	for i := half; i < K; i++ {
		c := float32(img[i%len(img)].B)
		x[i] = (c - 127.5) / 127.5
	}
	return x
}

// ScanImageCNN is the stargate entry: residual/level features → GGUF CNN → decision.
// Optionally also recovers LSB payload when nPayloadBytes > 0 (bit codec path).
func ScanImageCNN(s *weights.WeightSession, img []RGB, width, height, nPayloadBytes int, bitplane bool) (ScanResult, CNNResult, error) {
	cfg := CNNConfigFromSession(s)
	base := ScanImageAssist(img, width, height, nPayloadBytes)
	feat := FeaturesFromImage(img, cfg.InK)
	cnn, err := ForwardStegCNN(s, feat, cfg, bitplane)
	if err != nil {
		return base, cnn, err
	}
	// Fuse classical LSB extract with CNN decision when extract present.
	if nPayloadBytes > 0 && base.HasPayload {
		// keep extract truth; CNN still reported
		return base, cnn, nil
	}
	if nPayloadBytes == 0 {
		base.HasPayload = cnn.HasPayload
		base.Notes = strings.TrimSpace(base.Notes + " cnn=" + cnn.Decision)
		base.FilterScore = cnn.StegoScore
	}
	return base, cnn, nil
}

// DenseForwardStegCNN is a dense-f32 reference of the same graph (export parity /
// microbench baseline). Loads full f32 tiles via session linear, no trit quant.
func DenseForwardStegCNN(s *weights.WeightSession, x []float32, cfg CNNConfig) (CNNResult, error) {
	res := CNNResult{Bitplane: false, InK: cfg.InK, Notes: "dense_f32"}
	if s == nil {
		return res, fmt.Errorf("steg: nil session")
	}
	if cfg.InK == 0 {
		cfg = DefaultCNNConfig()
		res.InK = cfg.InK
	}
	if len(x) < cfg.InK {
		return res, fmt.Errorf("steg: feature short")
	}
	h0, err := s.Linear(cfg.L0, x[:cfg.InK], cfg.Hidden0, cfg.InK, 0)
	if err != nil {
		return res, err
	}
	b0, _ := loadBias(s, cfg.L0b, cfg.Hidden0)
	addBias(h0, b0)
	h0 = weights.SiLUF32(h0)

	h1, err := s.Linear(cfg.L1, h0, cfg.Hidden1, cfg.Hidden0, 0)
	if err != nil {
		return res, err
	}
	b1, _ := loadBias(s, cfg.L1b, cfg.Hidden1)
	addBias(h1, b1)
	h1 = weights.SiLUF32(h1)

	logits, err := s.Linear(cfg.Head, h1, cfg.NClass, cfg.Hidden1, 0)
	if err != nil {
		return res, err
	}
	hb, _ := loadBias(s, cfg.Headb, cfg.NClass)
	addBias(logits, hb)
	if len(logits) < 2 {
		return res, fmt.Errorf("steg: logits")
	}
	res.CleanScore = float64(logits[0])
	res.StegoScore = float64(logits[1])
	res.Margin = res.StegoScore - res.CleanScore
	if res.Margin > cfg.Margin {
		res.HasPayload = true
		res.Decision = "stego"
	} else {
		res.Decision = "clean"
	}
	return res, nil
}
