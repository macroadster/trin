//go:build ignore

package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/macroadster/trin/internal/weights"
)

// Writes demos/weights/tiny_gqa.gguf — 2-layer GQA toy for greedy decode.
// vocab=8 dim=4 nHeads=2 nKVHeads=1 dh=2 hidden=8; L0/L1 + tok_emb + final_norm + lm_head.
// Sidecar vocab reuses tiny_llm.vocab.

func main() {
	root := "."
	if len(os.Args) > 1 {
		root = os.Args[1]
	}
	out := filepath.Join(root, "demos/weights/tiny_gqa.gguf")
	vocabPath := filepath.Join(root, "demos/weights/tiny_llm.vocab")

	tokEmb := []float32{
		0.1, 0.0, 0.0, 0.0,
		1.0, 0.2, 0.0, 0.1,
		0.0, 1.0, 0.1, 0.0,
		0.2, 0.0, 1.0, 0.0,
		0.0, 0.1, 0.0, 1.0,
		0.5, 0.5, 0.0, 0.0,
		0.0, 0.5, 0.5, 0.0,
		0.0, 0.0, 0.0, 0.5,
	}
	// Q: 4×4 identity; K/V: 2×4 compact
	q0 := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	k0 := []float32{
		1.0, 0.1, 0.0, 0.0,
		0.0, 1.0, 0.1, 0.0,
	}
	v0 := []float32{
		0.8, 0.0, 0.1, 0.0,
		0.0, 0.8, 0.0, 0.1,
	}
	o0 := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	up0 := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		0.5, 0.5, 0.0, 0.0,
		0.0, 0.5, 0.5, 0.0,
		0.0, 0.0, 0.5, 0.5,
		0.5, 0.0, 0.0, 0.5,
	}
	down0 := []float32{
		1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5,
		0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0,
		0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5,
	}
	q1 := []float32{
		0.9, 0, 0, 0,
		0, 0.9, 0, 0,
		0, 0, 0.9, 0,
		0, 0, 0, 0.9,
	}
	k1 := []float32{
		1.0, 0.0, 0.0, 0.05,
		0.0, 1.0, 0.05, 0.0,
	}
	v1 := []float32{
		0.75, 0, 0.05, 0,
		0, 0.75, 0, 0.05,
	}
	o1 := o0
	up1 := []float32{
		0.5, 0, 0, 0,
		0, 0.5, 0, 0,
		0, 0, 0.5, 0,
		0, 0, 0, 0.5,
		0.25, 0.25, 0, 0,
		0, 0.25, 0.25, 0,
		0, 0, 0.25, 0.25,
		0.25, 0, 0, 0.25,
	}
	down1 := []float32{
		0.5, 0, 0, 0, 0.25, 0, 0, 0.25,
		0, 0.5, 0, 0, 0, 0.25, 0, 0,
		0, 0, 0.5, 0, 0, 0, 0.25, 0,
		0, 0, 0, 0.5, 0.25, 0, 0, 0.25,
	}
	lmHead := []float32{
		0.2, 0.0, 0.0, 0.0,
		1.2, 0.1, 0.0, 0.0,
		0.1, 1.2, 0.0, 0.0,
		0.0, 0.1, 1.2, 0.0,
		0.0, 0.0, 0.1, 1.2,
		0.6, 0.6, 0.0, 0.0,
		0.0, 0.6, 0.6, 0.0,
		-0.5, -0.5, -0.5, -0.5,
	}
	g := []float32{1, 1, 1, 1}

	tensors := map[string][]float32{
		"tok_emb": tokEmb,
		"L0_q":    q0, "L0_k": k0, "L0_v": v0, "L0_o": o0,
		"L0_up": up0, "L0_down": down0, "L0_an": g, "L0_fn": g,
		"L1_q": q1, "L1_k": k1, "L1_v": v1, "L1_o": o1,
		"L1_up": up1, "L1_down": down1, "L1_an": g, "L1_fn": g,
		"final_norm": g,
		"lm_head":    lmHead,
	}
	shapes := map[string][]uint64{
		"tok_emb": {8, 4},
		"L0_q":    {4, 4}, "L0_k": {2, 4}, "L0_v": {2, 4}, "L0_o": {4, 4},
		"L0_up": {8, 4}, "L0_down": {4, 8}, "L0_an": {4}, "L0_fn": {4},
		"L1_q": {4, 4}, "L1_k": {2, 4}, "L1_v": {2, 4}, "L1_o": {4, 4},
		"L1_up": {8, 4}, "L1_down": {4, 8}, "L1_an": {4}, "L1_fn": {4},
		"final_norm": {4},
		"lm_head":    {8, 4},
	}
	meta := map[string]string{
		"general.name":         "trin-tiny-gqa",
		"general.architecture": "trin_tiny_gqa",
		"trin.vocab_size":      "8",
		"trin.dim":             "4",
		"trin.hidden":          "8",
		"trin.n_heads":         "2",
		"trin.n_kv":            "1",
		"trin.dh":              "2",
		"trin.n_layers":        "2",
		"tokenizer.ggml.model": "trin_char",
	}
	if err := weights.WriteGGUFF32(out, tensors, shapes, meta); err != nil {
		panic(err)
	}
	if _, err := os.Stat(vocabPath); err != nil {
		vocab := "<pad>\nA\nB\nC\nD\nE\nF\n<eos>\n"
		if err := os.WriteFile(vocabPath, []byte(vocab), 0o644); err != nil {
			panic(err)
		}
		fmt.Println("wrote", vocabPath)
	}
	fmt.Println("wrote", out)
}
