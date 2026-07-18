//go:build ignore

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/macroadster/trin/internal/weights"
)

func fp16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xff) - 127 + 15
	mant := (bits >> 13) & 0x3ff
	if exp <= 0 {
		return sign
	}
	if exp >= 31 {
		return sign | 0x7c00
	}
	return sign | uint16(exp<<10) | uint16(mant)
}

func packQ4_0(vals []float32) []byte {
	nb := len(vals) / 32
	out := make([]byte, 0, nb*18)
	for i := 0; i < nb; i++ {
		block := vals[i*32 : (i+1)*32]
		amax, maxv := float32(0), float32(0)
		for _, v := range block {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax, maxv = a, v
			}
		}
		d := maxv / -8
		id := float32(0)
		if d != 0 {
			id = 1 / d
		}
		var buf [18]byte
		binary.LittleEndian.PutUint16(buf[0:2], fp16(d))
		for j := 0; j < 16; j++ {
			x0 := block[j] * id
			x1 := block[j+16] * id
			xi0 := int8(x0 + 8.5)
			if xi0 > 15 {
				xi0 = 15
			}
			if xi0 < 0 {
				xi0 = 0
			}
			xi1 := int8(x1 + 8.5)
			if xi1 > 15 {
				xi1 = 15
			}
			if xi1 < 0 {
				xi1 = 0
			}
			buf[2+j] = byte(xi0) | byte(xi1)<<4
		}
		out = append(out, buf[:]...)
	}
	return out
}

func packQ8_0(vals []float32) []byte {
	nb := len(vals) / 32
	out := make([]byte, 0, nb*34)
	for i := 0; i < nb; i++ {
		block := vals[i*32 : (i+1)*32]
		amax := float32(0)
		for _, v := range block {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}
		d := amax / 127
		id := float32(0)
		if d != 0 {
			id = 1 / d
		}
		var buf [34]byte
		binary.LittleEndian.PutUint16(buf[0:2], fp16(d))
		for j := 0; j < 32; j++ {
			q := int(math.Round(float64(block[j] * id)))
			if q < -128 {
				q = -128
			}
			if q > 127 {
				q = 127
			}
			buf[2+j] = byte(int8(q))
		}
		out = append(out, buf[:]...)
	}
	return out
}

func main() {
	// 64 f32 values → two Q4_0/Q8_0 blocks; reshape as 2x32 for linear later
	vals := make([]float32, 64)
	for i := range vals {
		vals[i] = float32(i-32) * 0.05
	}
	_ = os.MkdirAll("demos/weights", 0o755)
	raw4 := packQ4_0(vals)
	if err := weights.WriteGGUFTyped("demos/weights/W_q4_0.gguf", "weight", weights.GGML_Q4_0, []uint64{2, 32}, raw4, map[string]string{"general.name": "trin-q4"}); err != nil {
		panic(err)
	}
	raw8 := packQ8_0(vals)
	if err := weights.WriteGGUFTyped("demos/weights/W_q8_0.gguf", "weight", weights.GGML_Q8_0, []uint64{2, 32}, raw8, map[string]string{"general.name": "trin-q8"}); err != nil {
		panic(err)
	}
	// verify
	f, err := weights.OpenGGUF("demos/weights/W_q4_0.gguf")
	if err != nil {
		panic(err)
	}
	v, s, err := f.LoadF32("weight")
	if err != nil {
		panic(err)
	}
	fmt.Println("q4 n", len(v), "shape", s, "first", v[0], "meta", f.Meta["general.name"])
	fmt.Println("wrote demos/weights/W_q4_0.gguf demos/weights/W_q8_0.gguf")
}
