//go:build ignore

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"path/filepath"

	"github.com/macroadster/trin/internal/weights"
)

func f32raw(vals []float32) []byte {
	out := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

func main() {
	dir := "demos/weights"
	// 2-shard toy: bias on 0 (zeros), W 8x8 identity-ish all ones? use simple 4-vector + 4x4 ones for linear
	// For demo: tensor "x" 4 elems all 1.0 on shard0, tensor "W" 4x4 with 1.0 on shard1
	// linear: y = W @ x → each row sum = 4
	const n = 4
	x := make([]float32, n)
	for i := range x {
		x[i] = 1
	}
	W := make([]float32, n*n)
	for i := range W {
		W[i] = 1
	}
	p0 := filepath.Join(dir, "toy_shard-00000-of-00002.gguf")
	p1 := filepath.Join(dir, "toy_shard-00001-of-00002.gguf")
	meta0 := map[string]weights.TypedMeta{
		"general.name":                {Kind: "string", Str: "trin-multi-shard"},
		weights.MetaSplitNo:           {Kind: "u16", U64: 0},
		weights.MetaSplitCount:        {Kind: "u16", U64: 2},
		weights.MetaSplitTensorsCount: {Kind: "u16", U64: 2},
	}
	meta1 := map[string]weights.TypedMeta{
		weights.MetaSplitNo:           {Kind: "u16", U64: 1},
		weights.MetaSplitCount:        {Kind: "u16", U64: 2},
		weights.MetaSplitTensorsCount: {Kind: "u16", U64: 2},
	}
	if err := weights.WriteGGUFTypedMulti(p0, []string{"x"},
		map[string]uint32{"x": weights.GGML_F32},
		map[string][]uint64{"x": {uint64(n)}},
		map[string][]byte{"x": f32raw(x)}, meta0); err != nil {
		panic(err)
	}
	if err := weights.WriteGGUFTypedMulti(p1, []string{"W"},
		map[string]uint32{"W": weights.GGML_F32},
		map[string][]uint64{"W": {uint64(n), uint64(n)}},
		map[string][]byte{"W": f32raw(W)}, meta1); err != nil {
		panic(err)
	}
	fmt.Println("wrote", p0, "and", p1)
}
