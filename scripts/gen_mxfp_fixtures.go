//go:build ignore

package main

import (
	"fmt"

	"github.com/macroadster/trin/internal/weights"
)

func main() {
	// MXFP4: 256 × code7, e=128 → 12.0
	const n = 256
	const qkM = 32
	const bsM = 17
	rawM := make([]byte, 0, (n/qkM)*bsM)
	for b := 0; b < n/qkM; b++ {
		blk := make([]byte, bsM)
		blk[0] = 128 // e8m0 → d=1
		for j := 0; j < qkM/2; j++ {
			blk[1+j] = 0x77 // code 7 both halves
		}
		rawM = append(rawM, blk...)
	}
	if err := weights.WriteGGUFTyped("demos/weights/W_mxfp4.gguf", "W", weights.GGML_MXFP4, []uint64{n, 1}, rawM, map[string]string{"general.name": "trin-mxfp4"}); err != nil {
		panic(err)
	}

	// NVFP4: 256 elems, UE 0x40, code7 → 12.0
	const qkN = 64
	const bsN = 36
	rawN := make([]byte, 0, (n/qkN)*bsN)
	for b := 0; b < n/qkN; b++ {
		blk := make([]byte, bsN)
		for s := 0; s < 4; s++ {
			blk[s] = 0x40
		}
		for j := 4; j < bsN; j++ {
			blk[j] = 0x77
		}
		rawN = append(rawN, blk...)
	}
	if err := weights.WriteGGUFTyped("demos/weights/W_nvfp4.gguf", "W", weights.GGML_NVFP4, []uint64{n, 1}, rawN, map[string]string{"general.name": "trin-nvfp4"}); err != nil {
		panic(err)
	}
	fmt.Println("wrote demos/weights/W_mxfp4.gguf and W_nvfp4.gguf")
}
