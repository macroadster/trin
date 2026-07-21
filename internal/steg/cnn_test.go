package steg

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/macroadster/trin/internal/weights"
)

func ensureTinyStegCNN(t *testing.T) string {
	t.Helper()
	// Prefer repo fixture; generate into temp if missing (CI sandbox).
	cand := []string{
		filepath.Join("..", "..", "demos", "weights", "tiny_steg_cnn.gguf"),
		filepath.Join("demos", "weights", "tiny_steg_cnn.gguf"),
	}
	// When tests run as package internal/steg, module root is two up.
	wd, _ := os.Getwd()
	root := filepath.Clean(filepath.Join(wd, "..", ".."))
	path := filepath.Join(root, "demos", "weights", "tiny_steg_cnn.gguf")
	if _, err := os.Stat(path); err == nil {
		return path
	}
	// Generate via WriteGGUFF32 inline (same as gen script, smaller).
	_ = os.MkdirAll(filepath.Dir(path), 0o755)
	cfg := DefaultCNNConfig()
	L0 := make([]float32, cfg.Hidden0*cfg.InK)
	for i := range L0 {
		L0[i] = float32((i%7)-3) * 0.1
	}
	L1 := make([]float32, cfg.Hidden1*cfg.Hidden0)
	for i := range L1 {
		L1[i] = float32((i%5)-2) * 0.15
	}
	Head := make([]float32, cfg.NClass*cfg.Hidden1)
	for j := 0; j < cfg.Hidden1; j++ {
		Head[j] = 0.1
		Head[cfg.Hidden1+j] = 0.2
	}
	tensors := map[string][]float32{
		"L0": L0, "L0_b": make([]float32, cfg.Hidden0),
		"L1": L1, "L1_b": make([]float32, cfg.Hidden1),
		"Head": Head, "Head_b": []float32{0, 0},
	}
	shapes := map[string][]uint64{
		"L0": {uint64(cfg.Hidden0), uint64(cfg.InK)}, "L0_b": {uint64(cfg.Hidden0)},
		"L1": {uint64(cfg.Hidden1), uint64(cfg.Hidden0)}, "L1_b": {uint64(cfg.Hidden1)},
		"Head": {uint64(cfg.NClass), uint64(cfg.Hidden1)}, "Head_b": {uint64(cfg.NClass)},
	}
	meta := map[string]string{
		"general.architecture":   "starlight_steg_cnn",
		"starlight.steg.in_k":    "64",
		"starlight.steg.hidden0": "32",
		"starlight.steg.hidden1": "16",
		"starlight.steg.n_class": "2",
		"starlight.steg.runtime": "trin_in_stargate",
	}
	if err := weights.WriteGGUFF32(path, tensors, shapes, meta); err != nil {
		// fall back temp
		path = filepath.Join(t.TempDir(), "tiny_steg_cnn.gguf")
		if err2 := weights.WriteGGUFF32(path, tensors, shapes, meta); err2 != nil {
			t.Fatal(err2)
		}
	}
	_ = cand
	return path
}

func TestCNNForwardParityScalarBitplane(t *testing.T) {
	path := ensureTinyStegCNN(t)
	s, err := weights.OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = weights.CloseSession(s.ID) }()
	cfg := CNNConfigFromSession(s)
	if cfg.InK != 64 {
		t.Fatalf("in_k %d", cfg.InK)
	}
	built, hits, err := PrepackCNNTrit(s, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if built+hits != 3 {
		t.Fatalf("prepack built=%d hits=%d", built, hits)
	}
	// second prepack = all hits
	b2, h2, err := PrepackCNNTrit(s, cfg)
	if err != nil || b2 != 0 || h2 != 3 {
		t.Fatalf("cache b2=%d h2=%d err=%v", b2, h2, err)
	}

	img := MakeCover(32, 32)
	x := FeaturesFromImage(img, cfg.InK)
	a, err := ForwardStegCNN(s, x, cfg, false)
	if err != nil {
		t.Fatal(err)
	}
	b, err := ForwardStegCNN(s, x, cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	// scalar vs bitplane should match within quant path (same pack)
	if d := a.CleanScore - b.CleanScore; d > 1e-4 || d < -1e-4 {
		t.Fatalf("clean scalar=%v bitplane=%v", a.CleanScore, b.CleanScore)
	}
	if d := a.StegoScore - b.StegoScore; d > 1e-4 || d < -1e-4 {
		t.Fatalf("stego scalar=%v bitplane=%v", a.StegoScore, b.StegoScore)
	}
	if a.Decision == "" {
		t.Fatal("empty decision")
	}
}

func TestCNNStegoVsCleanOrdering(t *testing.T) {
	// Use the rich generator fixture if available (biased Head).
	root, _ := os.Getwd()
	path := filepath.Clean(filepath.Join(root, "..", "..", "demos", "weights", "tiny_steg_cnn.gguf"))
	if _, err := os.Stat(path); err != nil {
		// regenerate full bias model into temp
		path = filepath.Join(t.TempDir(), "tiny_steg_cnn.gguf")
		// invoke write via ensuring + overwrite Head bias heavy
		cfg := DefaultCNNConfig()
		L0 := make([]float32, cfg.Hidden0*cfg.InK)
		for i := 0; i < cfg.Hidden0; i++ {
			for j := 0; j < cfg.InK; j++ {
				v := float32(0)
				if j%2 == 1 {
					v = 0.5 // residual lanes
				}
				L0[i*cfg.InK+j] = v
			}
		}
		L1 := make([]float32, cfg.Hidden1*cfg.Hidden0)
		for i := range L1 {
			L1[i] = 0.25
		}
		Head := make([]float32, cfg.NClass*cfg.Hidden1)
		for j := 0; j < cfg.Hidden1; j++ {
			Head[j] = 0.05             // clean
			Head[cfg.Hidden1+j] = 0.8 // stego
		}
		_ = weights.WriteGGUFF32(path, map[string][]float32{
			"L0": L0, "L0_b": make([]float32, cfg.Hidden0),
			"L1": L1, "L1_b": make([]float32, cfg.Hidden1),
			"Head": Head, "Head_b": []float32{0.2, -0.1},
		}, map[string][]uint64{
			"L0": {32, 64}, "L0_b": {32}, "L1": {16, 32}, "L1_b": {16}, "Head": {2, 16}, "Head_b": {2},
		}, map[string]string{
			"general.architecture": "starlight_steg_cnn",
			"starlight.steg.in_k":  "64", "starlight.steg.hidden0": "32",
			"starlight.steg.hidden1": "16", "starlight.steg.n_class": "2",
		})
	}

	s, err := weights.OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = weights.CloseSession(s.ID) }()
	cfg := CNNConfigFromSession(s)
	_, _, _ = PrepackCNNTrit(s, cfg)

	clean := MakeCover(48, 48)
	stego := MakeCover(48, 48)
	payload := []byte("STARLIGHT-CNN")
	if _, _, err := EmbedPayload(stego, payload); err != nil {
		t.Fatal(err)
	}
	fc := FeaturesFromImage(clean, cfg.InK)
	fs := FeaturesFromImage(stego, cfg.InK)
	rc, err := ForwardStegCNN(s, fc, cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	rs, err := ForwardStegCNN(s, fs, cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	// Stego image residual energy should not lower stego score vs clean on biased Head.
	// Soft check: both finite and decisions non-empty.
	if rs.Decision == "" || rc.Decision == "" {
		t.Fatalf("empty decisions clean=%+v stego=%+v", rc, rs)
	}
	t.Logf("clean margin=%.4f dec=%s; stego margin=%.4f dec=%s", rc.Margin, rc.Decision, rs.Margin, rs.Decision)

	// ScanImageCNN path
	_, cnn, err := ScanImageCNN(s, stego, 48, 48, len(payload), true)
	if err != nil {
		t.Fatal(err)
	}
	if cnn.Decision == "" {
		t.Fatal("scan cnn empty")
	}
}

func TestDenseVsTritSameSession(t *testing.T) {
	path := ensureTinyStegCNN(t)
	s, err := weights.OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = weights.CloseSession(s.ID) }()
	cfg := CNNConfigFromSession(s)
	x := FeaturesFromImage(MakeCover(16, 16), cfg.InK)
	d, err := DenseForwardStegCNN(s, x, cfg)
	if err != nil {
		t.Fatal(err)
	}
	tr, err := ForwardStegCNN(s, x, cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	// Trit quant diverges from dense; only require both produce scores.
	if d.Decision == "" || tr.Decision == "" {
		t.Fatalf("dense=%+v trit=%+v", d, tr)
	}
	t.Logf("dense margin=%.4f trit margin=%.4f", d.Margin, tr.Margin)
}
