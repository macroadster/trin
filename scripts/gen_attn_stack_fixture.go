//go:build ignore

package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/macroadster/trin/internal/weights"
)

// Writes demos/weights/attn_stack.gguf — two residual transformer layers.
// dim=2 hidden=4; tensors L{0,1}_{q,k,v,o,up,down,an,fn}.

func main() {
	root := "."
	if len(os.Args) > 1 {
		root = os.Args[1]
	}
	out := filepath.Join(root, "demos/weights/attn_stack.gguf")

	I2 := []float32{1, 0, 0, 1}
	K0 := []float32{1, 0.1, 0, 1}
	V0 := []float32{0.8, 0, 0, 0.8}
	Up0 := []float32{
		1, 0,
		0, 1,
		0.5, 0.5,
		0, 0.5,
	}
	Down0 := []float32{
		1, 0, 0.5, 0,
		0, 1, 0, 0.5,
	}
	Q1 := []float32{0.9, 0, 0, 0.9}
	K1 := []float32{1, 0, 0.1, 1}
	V1 := []float32{0.7, 0, 0, 0.7}
	O1 := []float32{1, 0, 0, 1}
	Up1 := []float32{
		0.8, 0,
		0, 0.8,
		0.4, 0.4,
		0, 0.4,
	}
	Down1 := []float32{
		0.9, 0, 0.4, 0,
		0, 0.9, 0, 0.4,
	}
	g := []float32{1, 1}

	tensors := map[string][]float32{
		"L0_q": I2, "L0_k": K0, "L0_v": V0, "L0_o": I2,
		"L0_up": Up0, "L0_down": Down0, "L0_an": g, "L0_fn": g,
		"L1_q": Q1, "L1_k": K1, "L1_v": V1, "L1_o": O1,
		"L1_up": Up1, "L1_down": Down1, "L1_an": g, "L1_fn": g,
	}
	shapes := map[string][]uint64{
		"L0_q": {2, 2}, "L0_k": {2, 2}, "L0_v": {2, 2}, "L0_o": {2, 2},
		"L0_up": {4, 2}, "L0_down": {2, 4}, "L0_an": {2}, "L0_fn": {2},
		"L1_q": {2, 2}, "L1_k": {2, 2}, "L1_v": {2, 2}, "L1_o": {2, 2},
		"L1_up": {4, 2}, "L1_down": {2, 4}, "L1_an": {2}, "L1_fn": {2},
	}
	meta := map[string]string{
		"general.name":    "trin-attn-stack",
		"trin.dim":        "2",
		"trin.hidden":     "4",
		"trin.n_layers":   "2",
		"trin.n_heads":    "1",
	}
	if err := weights.WriteGGUFF32(out, tensors, shapes, meta); err != nil {
		panic(err)
	}
	fmt.Println("wrote", out)
}
