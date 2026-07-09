//go:build ignore

package main

import (
	"fmt"

	"github.com/ericchien/trin/internal/weights"
)

func main() {
	W := []float32{0.9, -1.1, 0.05, 1.0, -0.8, 0.2}
	st := "demos/weights/W.safetensors"
	gg := "demos/weights/W.gguf"
	if err := weights.WriteSafeTensors(st, map[string][]float32{"weight": W}, map[string][]int64{"weight": {2, 3}}); err != nil {
		panic(err)
	}
	if err := weights.WriteGGUFF32(gg, map[string][]float32{"weight": W}, map[string][]uint64{"weight": {2, 3}}, map[string]string{"general.name": "trin-toy"}); err != nil {
		panic(err)
	}
	if err := weights.WriteSafeTensors("demos/weights/toy.safetensors", map[string][]float32{
		"weight": W,
		"bias":   {0.1, -0.1},
	}, map[string][]int64{"weight": {2, 3}, "bias": {2}}); err != nil {
		panic(err)
	}
	if err := weights.WriteGGUFF32("demos/weights/toy.gguf", map[string][]float32{
		"weight": W,
		"bias":   {0.1, -0.1},
	}, map[string][]uint64{"weight": {2, 3}, "bias": {2}}, map[string]string{"general.name": "trin-toy"}); err != nil {
		panic(err)
	}
	fmt.Println("wrote fixtures")
	f, err := weights.OpenSafeTensors(st)
	if err != nil {
		panic(err)
	}
	v, s, err := f.LoadF32("weight")
	if err != nil {
		panic(err)
	}
	fmt.Println("st", v, s)
	g, err := weights.OpenGGUF(gg)
	if err != nil {
		panic(err)
	}
	v2, s2, err := g.LoadF32("weight")
	if err != nil {
		panic(err)
	}
	fmt.Println("gg", v2, s2, g.Meta["general.name"])
}
