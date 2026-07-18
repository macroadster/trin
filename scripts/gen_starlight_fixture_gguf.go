//go:build ignore

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/macroadster/trin/internal/weights"
)

// Writes demos/weights/starlight_fixture.gguf (deterministic BalancedStarlightDetector weights)
// plus input CSVs and expected head logits for H=W=32 (fast path) and notes for 256.
//
// Usage: go run scripts/gen_starlight_fixture_gguf.go [repo_root]

func main() {
	root := "."
	if len(os.Args) > 1 {
		root = os.Args[1]
	}
	outGGUF := filepath.Join(root, "demos/weights/starlight_fixture.gguf")
	inDir := filepath.Join(root, "demos/weights/starlight_inputs")
	_ = os.MkdirAll(filepath.Join(root, "demos/weights"), 0o755)
	_ = os.MkdirAll(inDir, 0o755)

	const seed uint64 = 42
	tensors, shapes := weights.GenStarlightFixtureTensors(seed)
	meta := map[string]string{
		"general.name":         "starlight-fixture",
		"general.architecture": "starlight",
		"trin.meta_weight":     "0.3",
		"trin.input_h":         "32",
		"trin.input_w":         "32",
		"trin.proj_dim":        "64",
		"trin.fusion_dim":      "448",
		"trin.seed":            "42",
	}
	if err := weights.WriteGGUFF32(outGGUF, tensors, shapes, meta); err != nil {
		panic(err)
	}
	fmt.Println("wrote", outGGUF, "tensors=", len(tensors))

	// Deterministic inputs H=W=32
	const H, W = 32, 32
	g := &weights.StarlightLCG{S: 99}
	pixel := g.Fill(3 * H * W)
	metaV := g.Fill(2048)
	alpha := g.Fill(2 * H * W)
	lsb := g.Fill(3 * H * W)
	palette := g.Fill(768)
	plsb := g.Fill(1 * H * W)
	fmtF := g.Fill(6)
	content := g.Fill(6)

	writeCSV := func(name string, v []float32) {
		path := filepath.Join(inDir, name)
		var b strings.Builder
		for i, x := range v {
			if i > 0 {
				b.WriteByte(',')
			}
			b.WriteString(strconv.FormatFloat(float64(x), 'g', -1, 32))
		}
		b.WriteByte('\n')
		if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
			panic(err)
		}
		fmt.Println("wrote", path, "n=", len(v))
	}
	writeCSV("pixel.csv", pixel)
	writeCSV("meta.csv", metaV)
	writeCSV("alpha.csv", alpha)
	writeCSV("lsb.csv", lsb)
	writeCSV("palette.csv", palette)
	writeCSV("palette_lsb.csv", plsb)
	writeCSV("format_features.csv", fmtF)
	writeCSV("content_features.csv", content)

	// Expected via Go forward
	s, err := weights.OpenSession(outGGUF)
	if err != nil {
		panic(err)
	}
	defer weights.CloseSession(s.ID)
	out, err := weights.StarlightForwardSession(s, weights.StarlightInputs{
		Pixel: pixel, Meta: metaV, Alpha: alpha, LSB: lsb,
		Palette: palette, PaletteLSB: plsb,
		FormatFeatures: fmtF, ContentFeatures: content,
		H: H, W: W, MetaWeight: 0.3,
	})
	if err != nil {
		panic(err)
	}
	expPath := filepath.Join(root, "demos/weights/starlight_expected.csv")
	// format: stego(1), method(5), bit_order(3), embedding first 4 for smoke, then full concat optional
	all := out.ConcatOutputs()
	var b strings.Builder
	b.WriteString("# stego(1),method(5),bit_order(3),embedding(64) = 73 floats\n")
	for i, x := range all {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatFloat(float64(x), 'g', -1, 32))
	}
	b.WriteByte('\n')
	if err := os.WriteFile(expPath, []byte(b.String()), 0o644); err != nil {
		panic(err)
	}
	fmt.Println("wrote", expPath)
	fmt.Printf("stego=%v method=%v bit_order=%v\n", out.Stego, out.Method, out.BitOrder)
}
