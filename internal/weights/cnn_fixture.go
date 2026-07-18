package weights

import (
	"math"
)

// StarlightLCG is a deterministic PRNG matching fixture generators / tests.
type StarlightLCG struct{ S uint64 }

// Next returns a float in (-0.05, 0.05).
func (g *StarlightLCG) Next() float32 {
	g.S = g.S*6364136223846793005 + 1
	u := float64((g.S>>33)&0xffffffff) / float64(1<<32)
	return float32((u - 0.5) * 0.1)
}

// Fill returns n draws.
func (g *StarlightLCG) Fill(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = g.Next()
	}
	return out
}

func starlightItoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [16]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

func starlightNElem(sh []uint64) int {
	n := 1
	for _, d := range sh {
		n *= int(d)
	}
	return n
}

func starlightContains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// StarlightTensorShapes returns BalancedStarlightDetector state_dict shapes (GGUF layout).
func StarlightTensorShapes() map[string][]uint64 {
	sh := map[string][]uint64{}
	add2d := func(prefix string, inC int) {
		chans := [][2]int{{inC, 16}, {16, 32}, {32, 64}}
		convIdx := []int{0, 3, 6}
		bnIdx := []int{1, 4, 7}
		for i, ch := range chans {
			ci, bi := convIdx[i], bnIdx[i]
			cout, cin := ch[1], ch[0]
			sh[prefix+"."+starlightItoa(ci)+".weight"] = []uint64{uint64(cout), uint64(cin), 3, 3}
			sh[prefix+"."+starlightItoa(ci)+".bias"] = []uint64{uint64(cout)}
			sh[prefix+"."+starlightItoa(bi)+".weight"] = []uint64{uint64(cout)}
			sh[prefix+"."+starlightItoa(bi)+".bias"] = []uint64{uint64(cout)}
			sh[prefix+"."+starlightItoa(bi)+".running_mean"] = []uint64{uint64(cout)}
			sh[prefix+"."+starlightItoa(bi)+".running_var"] = []uint64{uint64(cout)}
		}
	}
	add2d("pixel_conv", 3)
	add2d("alpha_conv", 2)
	add2d("lsb_conv", 3)
	add2d("palette_lsb_conv", 1)

	metaCh := []struct {
		cin, cout, k, idx int
	}{
		{1, 32, 7, 0}, {32, 64, 5, 3}, {64, 128, 3, 6},
	}
	bnM := []int{1, 4, 7}
	for i, m := range metaCh {
		ci, bi := m.idx, bnM[i]
		sh["meta_conv."+starlightItoa(ci)+".weight"] = []uint64{uint64(m.cout), uint64(m.cin), uint64(m.k)}
		sh["meta_conv."+starlightItoa(ci)+".bias"] = []uint64{uint64(m.cout)}
		sh["meta_conv."+starlightItoa(bi)+".weight"] = []uint64{uint64(m.cout)}
		sh["meta_conv."+starlightItoa(bi)+".bias"] = []uint64{uint64(m.cout)}
		sh["meta_conv."+starlightItoa(bi)+".running_mean"] = []uint64{uint64(m.cout)}
		sh["meta_conv."+starlightItoa(bi)+".running_var"] = []uint64{uint64(m.cout)}
	}

	for _, p := range []string{"pixel_proj", "meta_proj", "alpha_proj", "lsb_proj", "palette_lsb_proj"} {
		in := uint64(1024)
		if p == "meta_proj" {
			in = 2048
		}
		sh[p+".weight"] = []uint64{64, in}
		sh[p+".bias"] = []uint64{64}
	}
	sh["palette_fc.0.weight"] = []uint64{128, 768}
	sh["palette_fc.0.bias"] = []uint64{128}
	sh["palette_fc.3.weight"] = []uint64{64, 128}
	sh["palette_fc.3.bias"] = []uint64{64}

	for _, p := range []string{"format_features_fc", "content_features_fc"} {
		sh[p+".0.weight"] = []uint64{32, 6}
		sh[p+".0.bias"] = []uint64{32}
		sh[p+".2.weight"] = []uint64{32, 32}
		sh[p+".2.bias"] = []uint64{32}
	}
	sh["stream_gate.0.weight"] = []uint64{8, 448}
	sh["stream_gate.0.bias"] = []uint64{8}
	sh["fusion.0.weight"] = []uint64{256, 448}
	sh["fusion.0.bias"] = []uint64{256}
	sh["fusion.3.weight"] = []uint64{128, 256}
	sh["fusion.3.bias"] = []uint64{128}
	sh["fusion.6.weight"] = []uint64{192, 128}
	sh["fusion.6.bias"] = []uint64{192}

	sh["stego_head.weight"] = []uint64{1, 64}
	sh["stego_head.bias"] = []uint64{1}
	sh["method_head.weight"] = []uint64{5, 64}
	sh["method_head.bias"] = []uint64{5}
	sh["bit_order_head.weight"] = []uint64{3, 64}
	sh["bit_order_head.bias"] = []uint64{3}
	sh["embedding_head.weight"] = []uint64{64, 64}
	sh["embedding_head.bias"] = []uint64{64}
	return sh
}

// GenStarlightFixtureTensors builds deterministic weights + BN stats (seeded LCG).
func GenStarlightFixtureTensors(seed uint64) (map[string][]float32, map[string][]uint64) {
	shapes := StarlightTensorShapes()
	// Stable iteration order for reproducible LCG assignment across maps.
	names := make([]string, 0, len(shapes))
	for n := range shapes {
		names = append(names, n)
	}
	// insertion sort
	for i := 0; i < len(names); i++ {
		for j := i + 1; j < len(names); j++ {
			if names[j] < names[i] {
				names[i], names[j] = names[j], names[i]
			}
		}
	}
	g := &StarlightLCG{S: seed}
	tensors := make(map[string][]float32, len(shapes))
	for _, name := range names {
		sh := shapes[name]
		n := starlightNElem(sh)
		vals := g.Fill(n)
		if starlightContains(name, "running_var") {
			for i := range vals {
				vals[i] = float32(math.Abs(float64(vals[i]))) + 0.5
			}
		}
		// BN gamma: *.1.weight / *.4.weight / *.7.weight under *_conv
		if starlightContains(name, "_conv.") &&
			(starlightContains(name, ".1.weight") || starlightContains(name, ".4.weight") || starlightContains(name, ".7.weight")) {
			for i := range vals {
				vals[i] = 1.0 + vals[i]*0.1
			}
		}
		tensors[name] = vals
	}
	return tensors, shapes
}
