//go:build ignore

package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/ericchien/trin/internal/weights"
)

// Writes demos/weights/tiny_llama_named.gguf — same content as tiny_tied_alibi
// but with llama.cpp-style tensor names (token_embd / blk.N.* / output_norm).

func main() {
	root := "."
	if len(os.Args) > 1 {
		root = os.Args[1]
	}
	out := filepath.Join(root, "demos/weights/tiny_llama_named.gguf")

	tokEmb := []float32{
		0.1, 0.0, 0.0, 0.0,
		1.0, 0.2, 0.0, 0.1,
		0.0, 1.0, 0.1, 0.0,
		0.2, 0.0, 1.0, 0.0,
		0.0, 0.1, 0.0, 1.0,
		0.5, 0.5, 0.0, 0.0,
		2.0, 2.0, 2.0, 2.0,
		0.0, 0.0, 0.0, 0.5,
	}
	I4 := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	k0 := []float32{1.0, 0.1, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0}
	v0 := []float32{0.8, 0.0, 0.1, 0.0, 0.0, 0.8, 0.0, 0.1}
	gate0 := []float32{
		1, 0, 0, 0.05, 0, 1, 0, 0, 0, 0, 1, 0, 0.05, 0, 0, 1,
		0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5,
	}
	down0 := []float32{
		1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5,
		0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0,
		0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5,
	}
	q1 := []float32{0.9, 0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0.9}
	k1 := []float32{1.0, 0.0, 0.0, 0.05, 0.0, 1.0, 0.05, 0.0}
	v1 := []float32{0.75, 0, 0.05, 0, 0, 0.75, 0, 0.05}
	gate1 := []float32{
		0.9, 0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0.9,
		0.25, 0.25, 0, 0, 0, 0.25, 0.25, 0, 0, 0, 0.25, 0.25, 0.25, 0, 0, 0.25,
	}
	down1 := []float32{
		0.5, 0, 0, 0, 0.25, 0, 0, 0.25,
		0, 0.5, 0, 0, 0, 0.25, 0, 0,
		0, 0, 0.5, 0, 0, 0, 0.25, 0,
		0, 0, 0, 0.5, 0.25, 0, 0, 0.25,
	}
	g := []float32{1, 1, 1, 1}

	tensors := map[string][]float32{
		"token_embd.weight":        tokEmb,
		"output_norm.weight":       g,
		"blk.0.attn_q.weight":      I4,
		"blk.0.attn_k.weight":      k0,
		"blk.0.attn_v.weight":      v0,
		"blk.0.attn_output.weight": I4,
		"blk.0.ffn_gate.weight":    gate0,
		"blk.0.ffn_up.weight":      gate0,
		"blk.0.ffn_down.weight":    down0,
		"blk.0.attn_norm.weight":   g,
		"blk.0.ffn_norm.weight":    g,
		"blk.1.attn_q.weight":      q1,
		"blk.1.attn_k.weight":      k1,
		"blk.1.attn_v.weight":      v1,
		"blk.1.attn_output.weight": I4,
		"blk.1.ffn_gate.weight":    gate1,
		"blk.1.ffn_up.weight":      gate1,
		"blk.1.ffn_down.weight":    down1,
		"blk.1.attn_norm.weight":   g,
		"blk.1.ffn_norm.weight":    g,
	}
	shapes := map[string][]uint64{
		"token_embd.weight": {8, 4}, "output_norm.weight": {4},
		"blk.0.attn_q.weight": {4, 4}, "blk.0.attn_k.weight": {2, 4}, "blk.0.attn_v.weight": {2, 4},
		"blk.0.attn_output.weight": {4, 4},
		"blk.0.ffn_gate.weight": {8, 4}, "blk.0.ffn_up.weight": {8, 4}, "blk.0.ffn_down.weight": {4, 8},
		"blk.0.attn_norm.weight": {4}, "blk.0.ffn_norm.weight": {4},
		"blk.1.attn_q.weight": {4, 4}, "blk.1.attn_k.weight": {2, 4}, "blk.1.attn_v.weight": {2, 4},
		"blk.1.attn_output.weight": {4, 4},
		"blk.1.ffn_gate.weight": {8, 4}, "blk.1.ffn_up.weight": {8, 4}, "blk.1.ffn_down.weight": {4, 8},
		"blk.1.attn_norm.weight": {4}, "blk.1.ffn_norm.weight": {4},
	}
	meta := map[string]string{
		"general.name":         "trin-tiny-llama-named",
		"general.architecture": "llama",
		"trin.vocab_size":      "8",
		"trin.dim":             "4",
		"trin.hidden":          "8",
		"trin.n_heads":         "2",
		"trin.n_kv":            "1",
		"trin.dh":              "2",
		"trin.n_layers":        "2",
		"trin.ffn":             "swiglu",
		"trin.pos":             "alibi",
		"trin.tied_emb":        "1",
		"tokenizer.ggml.model": "trin_char",
	}
	if err := weights.WriteGGUFF32(out, tensors, shapes, meta); err != nil {
		panic(err)
	}
	fmt.Println("wrote", out)
}
