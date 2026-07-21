package steg

import (
	"bytes"
	"testing"

	"github.com/macroadster/trin/internal/weights"
)

func TestPayloadRoundTrip(t *testing.T) {
	payload := []byte("starlight-wish-skill:v1|hello")
	tr := TritsFromPayload(payload)
	back := PayloadFromTrits(tr)
	if !bytes.Equal(payload, back) {
		t.Fatalf("payload roundtrip %q vs %q", payload, back)
	}
	h1 := PixelCommitmentHash(payload)
	if len(h1) != 64 {
		t.Fatalf("hash len %d", len(h1))
	}
	if PixelCommitmentHash(payload) != h1 {
		t.Fatal("hash unstable")
	}
}

func TestEmbedExtract(t *testing.T) {
	payload := []byte("pixel-commitment-demo")
	img := MakeCover(64, 32) // 2048 px ≥ 8*21
	hash, n, err := EmbedPayload(img, payload)
	if err != nil {
		t.Fatal(err)
	}
	if n != 8*len(payload) {
		t.Fatalf("n=%d", n)
	}
	got, h2, err := ExtractPayload(img, len(payload))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) || h2 != hash {
		t.Fatalf("extract %q hash %s want %q %s", got, h2, payload, hash)
	}
}

func TestLSBTritRoundTrip(t *testing.T) {
	img := MakeCover(8, 8)
	tr := []int8{1, 0, -1, 1, -1, 0, 1, 1, 0, -1}
	EncodeTritLSB(img, tr)
	back := DecodeTritLSB(img, len(tr))
	for i := range tr {
		if back[i] != tr[i] {
			t.Fatalf("i=%d got %d want %d", i, back[i], tr[i])
		}
	}
}

func TestScanDetectsEmbed(t *testing.T) {
	payload := []byte("STEG")
	cover := MakeCover(32, 32)
	clean := ScanImageAssist(cover, 32, 32, 0)

	stego := MakeCover(32, 32)
	if _, _, err := EmbedPayload(stego, payload); err != nil {
		t.Fatal(err)
	}
	s := ScanImageAssist(stego, 32, 32, len(payload))
	if !s.HasPayload || s.PayloadHash == "" {
		t.Fatalf("expected payload detect %+v", s)
	}
	if s.PayloadHex == "" {
		t.Fatal("missing hex")
	}
	// filter score should be finite
	if s.FilterScore < 0 {
		t.Fatalf("filter %v", s.FilterScore)
	}
	_ = clean
}

func TestFilterPackParity(t *testing.T) {
	M, K := 8, 64
	Wt, scale := SyntheticFilterWeights(M, K, 42)
	pos, neg, nw := weights.PackTritMatrix(Wt, M, K)
	x := make([]float32, K)
	for i := range x {
		x[i] = float32(i%5-2) * 0.25
	}
	a := weights.LinearTritScalar(Wt, x, M, K, scale)
	b := weights.LinearTritPlanesPacked(pos, neg, x, M, K, nw, scale)
	for i := 0; i < M; i++ {
		d := a[i] - b[i]
		if d < -1e-4 || d > 1e-4 {
			t.Fatalf("row %d a=%v b=%v", i, a[i], b[i])
		}
	}
	sc := FilterScore(pos, neg, x, M, K, nw, scale)
	if sc < 0 {
		t.Fatal(sc)
	}
}

func TestMemFootprint(t *testing.T) {
	mf := MemFootprintF32(64, 1024)
	mt := MemFootprintTritPack(64, 1024)
	if !(mf > mt*10) {
		t.Fatalf("expected ~13x mem win f32=%d pack=%d", mf, mt)
	}
	note := EfficiencyNote(64, 1024, 100, 180)
	if note == "" {
		t.Fatal("empty note")
	}
}
