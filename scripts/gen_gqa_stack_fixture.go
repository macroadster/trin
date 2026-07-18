//go:build ignore

package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/macroadster/trin/internal/weights"
)

// Writes demos/weights/gqa_stack.gguf — 2 residual GQA layers.
// dim=4 nHeads=2 nKVHeads=1 dh=2 hidden=4; L{0,1}_{q,k,v,o,up,down,an,fn} + final_norm.

func main() {
	root := "."
	if len(os.Args) > 1 {
		root = os.Args[1]
	}
	out := filepath.Join(root, "demos/weights/gqa_stack.gguf")

	Q0 := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	K0 := []float32{
		1, 0.1, 0, 0,
		0, 1, 0.1, 0,
	}
	V0 := []float32{
		0.8, 0, 0.1, 0,
		0, 0.8, 0, 0.1,
	}
	O0 := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	Up0 := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	Down0 := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	Q1 := []float32{
		0.9, 0, 0, 0,
		0, 0.9, 0, 0,
		0, 0, 0.9, 0,
		0, 0, 0, 0.9,
	}
	K1 := []float32{
		1, 0, 0, 0.05,
		0, 1, 0.05, 0,
	}
	V1 := []float32{
		0.75, 0, 0.05, 0,
		0, 0.75, 0, 0.05,
	}
	O1 := O0
	Up1 := []float32{
		0.5, 0, 0, 0,
		0, 0.5, 0, 0,
		0, 0, 0.5, 0,
		0, 0, 0, 0.5,
	}
	Down1 := []float32{
		0.5, 0, 0, 0,
		0, 0.5, 0, 0,
		0, 0, 0.5, 0,
		0, 0, 0, 0.5,
	}
	g := []float32{1, 1, 1, 1}

	tensors := map[string][]float32{
		"L0_q": Q0, "L0_k": K0, "L0_v": V0, "L0_o": O0,
		"L0_up": Up0, "L0_down": Down0, "L0_an": g, "L0_fn": g,
		"L1_q": Q1, "L1_k": K1, "L1_v": V1, "L1_o": O1,
		"L1_up": Up1, "L1_down": Down1, "L1_an": g, "L1_fn": g,
		"final_norm": g,
	}
	shapes := map[string][]uint64{
		"L0_q": {4, 4}, "L0_k": {2, 4}, "L0_v": {2, 4}, "L0_o": {4, 4},
		"L0_up": {4, 4}, "L0_down": {4, 4}, "L0_an": {4}, "L0_fn": {4},
		"L1_q": {4, 4}, "L1_k": {2, 4}, "L1_v": {2, 4}, "L1_o": {4, 4},
		"L1_up": {4, 4}, "L1_down": {4, 4}, "L1_an": {4}, "L1_fn": {4},
		"final_norm": {4},
	}
	meta := map[string]string{
		"general.name":  "trin-gqa-stack",
		"trin.dim":      "4",
		"trin.hidden":   "4",
		"trin.n_layers": "2",
		"trin.n_heads":  "2",
		"trin.n_kv":     "1",
		"trin.dh":       "2",
	}
	if err := weights.WriteGGUFF32(out, tensors, shapes, meta); err != nil {
		panic(err)
	}
	fmt.Println("wrote", out)
}
