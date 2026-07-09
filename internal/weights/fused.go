package weights

import (
	"encoding/binary"
	"fmt"
	"math"
)

// DotGGMLWindow computes sum_{i=0..count-1} W[offset+i] * x[i]
// streaming through quant blocks with a single-block scratch (no count-sized decode).
func DotGGMLWindow(typ uint32, raw []byte, nElem, offset, count int, x []float32) (float32, error) {
	if offset < 0 || count < 0 || offset+count > nElem {
		return 0, fmt.Errorf("fused OOB: offset=%d count=%d nElem=%d", offset, count, nElem)
	}
	if count == 0 {
		return 0, nil
	}
	if len(x) < count {
		return 0, fmt.Errorf("fused: x short need %d have %d", count, len(x))
	}
	// dense numeric — stream elements (scratch ≤1 elem at a time conceptually)
	switch typ {
	case GGML_F32:
		need := (offset + count) * 4
		if len(raw) < need {
			return 0, fmt.Errorf("gguf F32 short")
		}
		acc := float32(0)
		base := offset * 4
		for i := 0; i < count; i++ {
			acc += math.Float32frombits(binary.LittleEndian.Uint32(raw[base+i*4:])) * x[i]
		}
		return acc, nil
	case GGML_F16:
		need := (offset + count) * 2
		if len(raw) < need {
			return 0, fmt.Errorf("gguf F16 short")
		}
		acc := float32(0)
		base := offset * 2
		for i := 0; i < count; i++ {
			acc += float16ToFloat32(binary.LittleEndian.Uint16(raw[base+i*2:])) * x[i]
		}
		return acc, nil
	case GGML_BF16:
		need := (offset + count) * 2
		if len(raw) < need {
			return 0, fmt.Errorf("gguf BF16 short")
		}
		acc := float32(0)
		base := offset * 2
		for i := 0; i < count; i++ {
			bits := uint32(binary.LittleEndian.Uint16(raw[base+i*2:])) << 16
			acc += math.Float32frombits(bits) * x[i]
		}
		return acc, nil
	case GGML_I8:
		if len(raw) < offset+count {
			return 0, fmt.Errorf("gguf I8 short")
		}
		acc := float32(0)
		for i := 0; i < count; i++ {
			acc += float32(int8(raw[offset+i])) * x[i]
		}
		return acc, nil
	case GGML_I16:
		need := (offset + count) * 2
		if len(raw) < need {
			return 0, fmt.Errorf("gguf I16 short")
		}
		acc := float32(0)
		base := offset * 2
		for i := 0; i < count; i++ {
			acc += float32(int16(binary.LittleEndian.Uint16(raw[base+i*2:]))) * x[i]
		}
		return acc, nil
	case GGML_I32:
		need := (offset + count) * 4
		if len(raw) < need {
			return 0, fmt.Errorf("gguf I32 short")
		}
		acc := float32(0)
		base := offset * 4
		for i := 0; i < count; i++ {
			acc += float32(int32(binary.LittleEndian.Uint32(raw[base+i*4:]))) * x[i]
		}
		return acc, nil
	case GGML_F64:
		need := (offset + count) * 8
		if len(raw) < need {
			return 0, fmt.Errorf("gguf F64 short")
		}
		acc := float32(0)
		base := offset * 8
		for i := 0; i < count; i++ {
			acc += float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[base+i*8:]))) * x[i]
		}
		return acc, nil
	case GGML_I64:
		need := (offset + count) * 8
		if len(raw) < need {
			return 0, fmt.Errorf("gguf I64 short")
		}
		acc := float32(0)
		base := offset * 8
		for i := 0; i < count; i++ {
			acc += float32(int64(binary.LittleEndian.Uint64(raw[base+i*8:]))) * x[i]
		}
		return acc, nil
	}

	blkSize, qk, ok := blockLayout(typ)
	fn, ok2 := dequantFn(typ)
	if !ok || !ok2 {
		return 0, fmt.Errorf("fused: unsupported type %d", typ)
	}
	b0 := offset / qk
	b1 := (offset + count + qk - 1) / qk
	nBlocksTotal := (nElem + qk - 1) / qk
	if b1 > nBlocksTotal {
		b1 = nBlocksTotal
	}
	need := nBlocksTotal * blkSize
	if len(raw) < need {
		return 0, fmt.Errorf("gguf quant short: need %d have %d", need, len(raw))
	}
	tmp := make([]float32, qk) // ≤ one block scratch
	acc := float32(0)
	end := offset + count
	for b := b0; b < b1; b++ {
		base := b * blkSize
		fn(raw[base:base+blkSize], tmp)
		blkOff := b * qk
		// intersection of block [blkOff, blkOff+qk) with [offset, end)
		lo := blkOff
		if lo < offset {
			lo = offset
		}
		hi := blkOff + qk
		if hi > end {
			hi = end
		}
		for g := lo; g < hi; g++ {
			acc += tmp[g-blkOff] * x[g-offset]
		}
	}
	return acc, nil
}

// tensorRaw returns type, raw payload, and nElem for a GGUF tensor.
func (f *GGUFFile) tensorRaw(name string) (typ uint32, raw []byte, nElem int, err error) {
	ti, ok := f.ByName[name]
	if !ok {
		return 0, nil, 0, fmt.Errorf("gguf: tensor %q not found", name)
	}
	sf := f.shardFile(ti)
	nElem = 1
	for _, d := range ti.Dims {
		if d == 0 {
			return 0, nil, 0, fmt.Errorf("gguf: zero dim")
		}
		nElem *= int(d)
	}
	base := sf.DataBase + int64(ti.Offset)
	if base < 0 || base > int64(len(sf.Data)) {
		return 0, nil, 0, fmt.Errorf("gguf: bad data offset")
	}
	raw = sf.Data[base:]
	if need := GGMLRawBytes(ti.Type, nElem); need > 0 {
		if int64(len(sf.Data))-base < int64(need) {
			return 0, nil, 0, fmt.Errorf("gguf: tensor %q short", name)
		}
		raw = sf.Data[base : base+int64(need)]
	}
	return ti.Type, raw, nElem, nil
}

// LinearRowsFused: y = W[row0:row0+nRows,:] @ x using block-stream fused dots.
// F32/dense streams elems; Q* uses ≤1-block scratch per block.
func (s *WeightSession) LinearRowsFused(name string, x []float32, K, row0, nRows int) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("nil session")
	}
	if K <= 0 || nRows <= 0 {
		return nil, fmt.Errorf("linear_fused: bad K/nRows")
	}
	if len(x) < K {
		return nil, fmt.Errorf("linear_fused: x short")
	}
	if row0 < 0 {
		return nil, fmt.Errorf("linear_fused: negative row0")
	}
	nElem, _, err := s.tensorNElem(name)
	if err != nil {
		return nil, err
	}
	if nElem%K != 0 {
		return nil, fmt.Errorf("linear_fused: nElem=%d not divisible by K=%d", nElem, K)
	}
	M := nElem / K
	if row0+nRows > M {
		return nil, fmt.Errorf("linear_fused OOB: row0=%d nRows=%d M=%d", row0, nRows, M)
	}
	out := make([]float32, nRows)

	switch s.Kind {
	case "gguf":
		if s.gguf == nil {
			return nil, fmt.Errorf("session closed")
		}
		typ, raw, n, err := s.gguf.tensorRaw(name)
		if err != nil {
			return nil, err
		}
		for i := 0; i < nRows; i++ {
			off := (row0 + i) * K
			v, err := DotGGMLWindow(typ, raw, n, off, K, x)
			if err != nil {
				return nil, err
			}
			out[i] = v
		}
		return out, nil
	case "safetensors":
		// dense F32/F16 stream via window decode for fallback path (small row)
		// Prefer LoadF32Window which already slice-decodes without full tensor.
		// For fused guarantee: stream F32 if possible.
		if s.st == nil {
			return nil, fmt.Errorf("session closed")
		}
		// Use window per row only — never full tensor. Still allocates K, not nRows*K in one shot
		// for multi-row we can stream F32 from the raw if dtype is F32.
		for i := 0; i < nRows; i++ {
			off := (row0 + i) * K
			w, err := s.st.LoadF32Window(name, off, K)
			if err != nil {
				return nil, err
			}
			acc := float32(0)
			for t := 0; t < K; t++ {
				acc += w[t] * x[t]
			}
			out[i] = acc
		}
		return out, nil
	default:
		return nil, fmt.Errorf("linear_fused: unknown kind %s", s.Kind)
	}
}

// LinearFused y=W@x with row tiling; each tile uses fused row dots (no full-tile f32).
func (s *WeightSession) LinearFused(name string, x []float32, M, K, tileRows int) ([]float32, error) {
	if M <= 0 || K <= 0 {
		return nil, fmt.Errorf("linear_fused: bad M/K")
	}
	if len(x) < K {
		return nil, fmt.Errorf("linear_fused: x short")
	}
	nElem, _, err := s.tensorNElem(name)
	if err != nil {
		return nil, err
	}
	if nElem < M*K {
		return nil, fmt.Errorf("linear_fused: tensor too small")
	}
	if tileRows <= 0 {
		tileRows = 8
	}
	if tileRows > M {
		tileRows = M
	}
	out := make([]float32, M)
	for r := 0; r < M; r += tileRows {
		n := tileRows
		if r+n > M {
			n = M - r
		}
		y, err := s.LinearRowsFused(name, x, K, r, n)
		if err != nil {
			return nil, err
		}
		copy(out[r:r+n], y)
	}
	return out, nil
}

// LinearGGUFFused is a one-shot open + fused Linear.
func LinearGGUFFused(path, name string, x []float32, M, K, tileRows int) ([]float32, error) {
	f, err := OpenGGUF(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	s := &WeightSession{Kind: "gguf", Path: path, gguf: f, Mapped: f.Mapped}
	return s.LinearFused(name, x, M, K, tileRows)
}
