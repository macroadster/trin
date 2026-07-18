package starlight_test

import (
	"math"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/ericchien/trin/internal/weights"
	"github.com/ericchien/trin/pkg/starlight"
)

func repoRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("caller")
	}
	// pkg/starlight -> repo root
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
}

func TestOpenForwardFixture(t *testing.T) {
	path := filepath.Join(repoRoot(t), "demos/weights/starlight_fixture.gguf")
	s, err := starlight.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	names, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(names) < 50 {
		t.Fatalf("n_tensors=%d", len(names))
	}
	const H, W = 32, 32
	g := &weights.StarlightLCG{S: 99}
	out, err := s.Forward(starlight.Inputs{
		Pixel: g.Fill(3 * H * W), Meta: g.Fill(2048), Alpha: g.Fill(2 * H * W),
		LSB: g.Fill(3 * H * W), Palette: g.Fill(768), PaletteLSB: g.Fill(1 * H * W),
		FormatFeatures: g.Fill(6), ContentFeatures: g.Fill(6),
		H: H, W: W, MetaWeight: 0.3,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Stego) != 1 || len(out.Method) != 5 || len(out.BitOrder) != 3 || len(out.Embedding) != 64 {
		t.Fatalf("bad shapes stego=%d method=%d bit=%d emb=%d",
			len(out.Stego), len(out.Method), len(out.BitOrder), len(out.Embedding))
	}
	cat := out.Concat()
	if len(cat) != 73 {
		t.Fatalf("concat %d", len(cat))
	}
	// Fixture expected stego ~0.0057309084
	if math.Abs(float64(out.Stego[0])-0.0057309084) > 1e-4 {
		t.Fatalf("stego=%v", out.Stego[0])
	}
}
