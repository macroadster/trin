//go:build ignore

package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/macroadster/trin/internal/weights"
)

// Writes demos/weights/tiny_steg_cnn.gguf — Starlight scanner-shaped toy CNN
// lowered to im2col/linear stacks (what a PyTorch/TF export would emit after
// flattening conv kernels). Graph:
//
//	x[K=64] → Linear(L0:32×64) → SiLU → Linear(L1:16×32) → SiLU → Linear(Head:2×16)
//	→ [clean_logit, stego_logit]
//
// Pure-Go Trin/stargate inference: open session → prepack trit → forward.
// No Python runtime on the scan path.

func main() {
	root := "."
	if len(os.Args) > 1 {
		root = os.Args[1]
	}
	out := filepath.Join(root, "demos/weights/tiny_steg_cnn.gguf")

	const (
		K0 = 64
		H0 = 32
		H1 = 16
		C  = 2
	)

	// Deterministic kernels with a mild "LSB residual" bias on Head→stego so
	// synthetic payload images (high blue residual energy) score higher.
	L0 := make([]float32, H0*K0)
	for i := 0; i < H0; i++ {
		for j := 0; j < K0; j++ {
			// alternating edge / residual filters
			v := math.Sin(float64(i*3+j)*0.17) * 0.8
			if j%2 == 1 {
				v += 0.15 // odd lanes = "diff" channels
			}
			L0[i*K0+j] = float32(v)
		}
	}
	L0b := make([]float32, H0)
	for i := range L0b {
		L0b[i] = float32(0.01 * float64(i%5-2))
	}

	L1 := make([]float32, H1*H0)
	for i := 0; i < H1; i++ {
		for j := 0; j < H0; j++ {
			L1[i*H0+j] = float32(math.Cos(float64(i+j)*0.21) * 0.6)
		}
	}
	L1b := make([]float32, H1)

	// Head: row0 = clean, row1 = stego. Stego row weights odd residual lanes.
	Head := make([]float32, C*H1)
	for j := 0; j < H1; j++ {
		Head[0*H1+j] = float32(0.15 - 0.02*float64(j)) // clean prefers smooth
		Head[1*H1+j] = float32(0.25 + 0.03*float64(j%4)) // stego energy
	}
	Headb := []float32{0.05, -0.02}

	tensors := map[string][]float32{
		"L0":     L0,
		"L0_b":   L0b,
		"L1":     L1,
		"L1_b":   L1b,
		"Head":   Head,
		"Head_b": Headb,
	}
	shapes := map[string][]uint64{
		"L0":     {H0, K0}, // row-major out×in (export convention)
		"L0_b":   {H0},
		"L1":     {H1, H0},
		"L1_b":   {H1},
		"Head":   {C, H1},
		"Head_b": {C},
	}
	meta := map[string]string{
		"general.name":               "starlight-tiny-steg-cnn",
		"general.architecture":       "starlight_steg_cnn",
		"starlight.steg.model":       "tiny_cnn_v1",
		"starlight.steg.in_k":        fmt.Sprintf("%d", K0),
		"starlight.steg.hidden0":     fmt.Sprintf("%d", H0),
		"starlight.steg.hidden1":     fmt.Sprintf("%d", H1),
		"starlight.steg.n_class":     fmt.Sprintf("%d", C),
		"starlight.steg.act":         "silu",
		"starlight.steg.export":      "im2col_linear",
		"starlight.steg.runtime":     "trin_in_stargate",
		"starlight.steg.no_python":   "1",
		"starlight.steg.class.0":     "clean",
		"starlight.steg.class.1":     "stego",
		"starlight.steg.tensor.L0":   "L0",
		"starlight.steg.tensor.L1":   "L1",
		"starlight.steg.tensor.Head": "Head",
	}
	if err := weights.WriteGGUFF32(out, tensors, shapes, meta); err != nil {
		panic(err)
	}
	fmt.Println("wrote", out)
}
