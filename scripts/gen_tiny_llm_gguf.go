//go:build ignore

package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/macroadster/trin/internal/weights"
)

// Writes demos/weights/tiny_llm.gguf — one transformer block toy model:
// vocab=8 dim=4 hidden=8, with tok_emb + Q/K/V/O + ffn_up/down + lm_head + norms.
// Weights match examples/tiny_ternary_llm*.trin so greedy decode is deterministic.

func main() {
	root := "."
	if len(os.Args) > 1 {
		root = os.Args[1]
	}
	out := filepath.Join(root, "demos/weights/tiny_llm.gguf")
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
	attnQ := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	attnK := []float32{
		1.0, 0.1, 0.0, 0.0,
		0.0, 1.0, 0.1, 0.0,
		0.0, 0.0, 1.0, 0.1,
		0.1, 0.0, 0.0, 1.0,
	}
	attnV := []float32{
		0.8, 0.0, 0.1, 0.0,
		0.0, 0.8, 0.0, 0.1,
		0.1, 0.0, 0.8, 0.0,
		0.0, 0.1, 0.0, 0.8,
	}
	attnO := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	ffnUp := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		0.5, 0.5, 0.0, 0.0,
		0.0, 0.5, 0.5, 0.0,
		0.0, 0.0, 0.5, 0.5,
		0.5, 0.0, 0.0, 0.5,
	}
	ffnDown := []float32{
		1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5,
		0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0,
		0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5,
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
	attnNorm := []float32{1, 1, 1, 1}
	ffnNorm := []float32{1, 1, 1, 1}

	tensors := map[string][]float32{
		"tok_emb":   tokEmb,
		"attn_q":    attnQ,
		"attn_k":    attnK,
		"attn_v":    attnV,
		"attn_o":    attnO,
		"ffn_up":    ffnUp,
		"ffn_down":  ffnDown,
		"lm_head":   lmHead,
		"attn_norm": attnNorm,
		"ffn_norm":  ffnNorm,
	}
	shapes := map[string][]uint64{
		"tok_emb":   {8, 4},
		"attn_q":    {4, 4},
		"attn_k":    {4, 4},
		"attn_v":    {4, 4},
		"attn_o":    {4, 4},
		"ffn_up":    {8, 4},
		"ffn_down":  {4, 8},
		"lm_head":   {8, 4},
		"attn_norm": {4},
		"ffn_norm":  {4},
	}
	meta := map[string]string{
		"general.name":               "trin-tiny-llm",
		"general.architecture":       "trin_tiny",
		"trin.vocab_size":            "8",
		"trin.dim":                   "4",
		"trin.hidden":                "8",
		"trin.n_heads":               "1",
		"trin.n_layers":              "1",
		"tokenizer.ggml.model":       "trin_char",
		"tokenizer.ggml.tokens":      "<pad>\nA\nB\nC\nD\nE\nF\n<eos>",
		"tokenizer.ggml.bos_token_id": "1",
		"tokenizer.ggml.eos_token_id": "7",
	}
	if err := weights.WriteGGUFF32(out, tensors, shapes, meta); err != nil {
		panic(err)
	}
	// sidecar vocab file (one token per line) for load_vocab
	vocab := "<pad>\nA\nB\nC\nD\nE\nF\n<eos>\n"
	if err := os.WriteFile(vocabPath, []byte(vocab), 0o644); err != nil {
		panic(err)
	}
	fmt.Println("wrote", out)
	fmt.Println("wrote", vocabPath)
}
