// Package weights provides pure-Go GGUF and safetensors readers for Trin.
// Supported dtypes are upcast to float64 then used as f32 spans by the language runtime.
package weights

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
)

// STInfo describes one tensor in a safetensors file.
type STInfo struct {
	Name   string
	Dtype  string
	Shape  []int64
	Offset int64 // absolute file offset of first byte
	Length int64 // byte length
}

// SafeTensorsFile is a parsed header plus mapped bytes (mmap when available, v0.85).
type SafeTensorsFile struct {
	Path    string
	Data    []byte
	Tensors map[string]STInfo
	Order   []string // stable insertion order from JSON map walk (unspecified) — use Order if set
	Mapped  bool
	mf      *mappedFile
}

// OpenSafeTensors mmaps the file (heap fallback) and parses only the JSON header.
// Tensor bodies page-fault on LoadF32.
func OpenSafeTensors(path string) (*SafeTensorsFile, error) {
	mf, err := openMapped(path)
	if err != nil {
		return nil, err
	}
	b := mf.Data
	if len(b) < 8 {
		_ = mf.Close()
		return nil, fmt.Errorf("safetensors: file too short")
	}
	hlen := binary.LittleEndian.Uint64(b[0:8])
	if hlen == 0 || 8+hlen > uint64(len(b)) {
		_ = mf.Close()
		return nil, fmt.Errorf("safetensors: bad header length %d", hlen)
	}
	var header map[string]json.RawMessage
	if err := json.Unmarshal(b[8:8+hlen], &header); err != nil {
		_ = mf.Close()
		return nil, fmt.Errorf("safetensors: header json: %w", err)
	}
	dataBase := int64(8 + hlen)
	f := &SafeTensorsFile{
		Path:    path,
		Data:    b,
		Tensors: make(map[string]STInfo),
		Mapped:  mf.Mapped,
		mf:      mf,
	}
	// preserve a stable order: sort keys later if needed
	for name, raw := range header {
		if name == "__metadata__" {
			continue
		}
		var ti struct {
			Dtype       string  `json:"dtype"`
			Shape       []int64 `json:"shape"`
			DataOffsets [2]int64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(raw, &ti); err != nil {
			return nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}
		start, end := ti.DataOffsets[0], ti.DataOffsets[1]
		if start < 0 || end < start {
			return nil, fmt.Errorf("safetensors: tensor %q bad offsets", name)
		}
		info := STInfo{
			Name:   name,
			Dtype:  strings.ToUpper(ti.Dtype),
			Shape:  ti.Shape,
			Offset: dataBase + start,
			Length: end - start,
		}
		f.Tensors[name] = info
		f.Order = append(f.Order, name)
	}
	return f, nil
}

// Close releases the underlying mmap (idempotent).
func (f *SafeTensorsFile) Close() error {
	if f == nil {
		return nil
	}
	var err error
	if f.mf != nil {
		err = f.mf.Close()
		f.mf = nil
	}
	f.Data = nil
	f.Mapped = false
	return err
}

// IsMapped reports whether the file data is OS-mapped.
func (f *SafeTensorsFile) IsMapped() bool {
	return f != nil && f.Mapped
}

// List returns tensor names.
func (f *SafeTensorsFile) List() []string {
	if len(f.Order) > 0 {
		return append([]string(nil), f.Order...)
	}
	out := make([]string, 0, len(f.Tensors))
	for k := range f.Tensors {
		out = append(out, k)
	}
	return out
}

// LoadF32 decodes a named tensor as a flat f32 span (row-major).
func (f *SafeTensorsFile) LoadF32(name string) ([]float32, []int64, error) {
	info, ok := f.Tensors[name]
	if !ok {
		return nil, nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}
	if info.Offset+info.Length > int64(len(f.Data)) {
		return nil, nil, fmt.Errorf("safetensors: tensor %q OOB", name)
	}
	raw := f.Data[info.Offset : info.Offset+info.Length]
	vals, err := decodeNumeric(info.Dtype, raw)
	if err != nil {
		return nil, nil, err
	}
	return vals, info.Shape, nil
}

// LoadF32Window decodes only [offset, offset+count) elements (row-major flat).
func (f *SafeTensorsFile) LoadF32Window(name string, offset, count int) ([]float32, error) {
	info, ok := f.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}
	if info.Offset+info.Length > int64(len(f.Data)) {
		return nil, fmt.Errorf("safetensors: tensor %q OOB", name)
	}
	raw := f.Data[info.Offset : info.Offset+info.Length]
	return decodeNumericWindow(info.Dtype, raw, offset, count)
}

func decodeNumericWindow(dtype string, raw []byte, offset, count int) ([]float32, error) {
	if offset < 0 || count < 0 {
		return nil, fmt.Errorf("safetensors window: bad range")
	}
	if count == 0 {
		// still validate offset against nElem for dense types
		n, err := numericNElem(dtype, raw)
		if err != nil {
			return nil, err
		}
		if offset > n {
			return nil, fmt.Errorf("safetensors window OOB: offset=%d count=%d nElem=%d", offset, count, n)
		}
		return []float32{}, nil
	}
	switch strings.ToUpper(dtype) {
	case "F32", "FLOAT32":
		nElem := len(raw) / 4
		if offset+count > nElem {
			return nil, fmt.Errorf("safetensors window OOB: offset=%d count=%d nElem=%d", offset, count, nElem)
		}
		return decodeNumeric(dtype, raw[offset*4:(offset+count)*4])
	case "F64", "FLOAT64":
		nElem := len(raw) / 8
		if offset+count > nElem {
			return nil, fmt.Errorf("safetensors window OOB: offset=%d count=%d nElem=%d", offset, count, nElem)
		}
		return decodeNumeric(dtype, raw[offset*8:(offset+count)*8])
	case "F16", "FLOAT16", "BF16", "BFLOAT16", "I16", "INT16":
		nElem := len(raw) / 2
		if offset+count > nElem {
			return nil, fmt.Errorf("safetensors window OOB: offset=%d count=%d nElem=%d", offset, count, nElem)
		}
		return decodeNumeric(dtype, raw[offset*2:(offset+count)*2])
	case "I8", "INT8", "U8", "UINT8":
		nElem := len(raw)
		if offset+count > nElem {
			return nil, fmt.Errorf("safetensors window OOB: offset=%d count=%d nElem=%d", offset, count, nElem)
		}
		return decodeNumeric(dtype, raw[offset:offset+count])
	case "I32", "INT32":
		nElem := len(raw) / 4
		if offset+count > nElem {
			return nil, fmt.Errorf("safetensors window OOB: offset=%d count=%d nElem=%d", offset, count, nElem)
		}
		return decodeNumeric(dtype, raw[offset*4:(offset+count)*4])
	case "I64", "INT64":
		nElem := len(raw) / 8
		if offset+count > nElem {
			return nil, fmt.Errorf("safetensors window OOB: offset=%d count=%d nElem=%d", offset, count, nElem)
		}
		return decodeNumeric(dtype, raw[offset*8:(offset+count)*8])
	default:
		return nil, fmt.Errorf("safetensors: unsupported dtype %s for window", dtype)
	}
}

func numericNElem(dtype string, raw []byte) (int, error) {
	switch strings.ToUpper(dtype) {
	case "F32", "FLOAT32", "I32", "INT32":
		return len(raw) / 4, nil
	case "F64", "FLOAT64", "I64", "INT64":
		return len(raw) / 8, nil
	case "F16", "FLOAT16", "BF16", "BFLOAT16", "I16", "INT16":
		return len(raw) / 2, nil
	case "I8", "INT8", "U8", "UINT8":
		return len(raw), nil
	default:
		return 0, fmt.Errorf("safetensors: unsupported dtype %s", dtype)
	}
}

func decodeNumeric(dtype string, raw []byte) ([]float32, error) {
	switch strings.ToUpper(dtype) {
	case "F32", "FLOAT32":
		if len(raw)%4 != 0 {
			return nil, fmt.Errorf("F32 length")
		}
		out := make([]float32, len(raw)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out, nil
	case "F64", "FLOAT64":
		if len(raw)%8 != 0 {
			return nil, fmt.Errorf("F64 length")
		}
		out := make([]float32, len(raw)/8)
		for i := range out {
			out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:])))
		}
		return out, nil
	case "F16", "FLOAT16":
		if len(raw)%2 != 0 {
			return nil, fmt.Errorf("F16 length")
		}
		out := make([]float32, len(raw)/2)
		for i := range out {
			out[i] = float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return out, nil
	case "BF16", "BFLOAT16":
		if len(raw)%2 != 0 {
			return nil, fmt.Errorf("BF16 length")
		}
		out := make([]float32, len(raw)/2)
		for i := range out {
			bits := uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16
			out[i] = math.Float32frombits(bits)
		}
		return out, nil
	case "I8", "INT8":
		out := make([]float32, len(raw))
		for i, b := range raw {
			out[i] = float32(int8(b))
		}
		return out, nil
	case "U8", "UINT8":
		out := make([]float32, len(raw))
		for i, b := range raw {
			out[i] = float32(b)
		}
		return out, nil
	case "I16", "INT16":
		if len(raw)%2 != 0 {
			return nil, fmt.Errorf("I16 length")
		}
		out := make([]float32, len(raw)/2)
		for i := range out {
			out[i] = float32(int16(binary.LittleEndian.Uint16(raw[i*2:])))
		}
		return out, nil
	case "I32", "INT32":
		if len(raw)%4 != 0 {
			return nil, fmt.Errorf("I32 length")
		}
		out := make([]float32, len(raw)/4)
		for i := range out {
			out[i] = float32(int32(binary.LittleEndian.Uint32(raw[i*4:])))
		}
		return out, nil
	case "I64", "INT64":
		if len(raw)%8 != 0 {
			return nil, fmt.Errorf("I64 length")
		}
		out := make([]float32, len(raw)/8)
		for i := range out {
			out[i] = float32(int64(binary.LittleEndian.Uint64(raw[i*8:])))
		}
		return out, nil
	default:
		return nil, fmt.Errorf("safetensors: unsupported dtype %s", dtype)
	}
}

// WriteSafeTensors writes a minimal single/multi tensor F32 file (little-endian).
func WriteSafeTensors(path string, tensors map[string][]float32, shapes map[string][]int64) error {
	type thdr struct {
		Dtype       string  `json:"dtype"`
		Shape       []int64 `json:"shape"`
		DataOffsets [2]int64 `json:"data_offsets"`
	}
	// fixed order by iterating sorted? preserve map insertion is random — use given map iteration order is fine for tests
	header := map[string]thdr{}
	var data []byte
	var off int64
	// stable: collect names
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	// simple insertion-order sort lexicographically for determinism
	for i := 0; i < len(names); i++ {
		for j := i + 1; j < len(names); j++ {
			if names[j] < names[i] {
				names[i], names[j] = names[j], names[i]
			}
		}
	}
	for _, name := range names {
		vals := tensors[name]
		shape := shapes[name]
		if shape == nil {
			shape = []int64{int64(len(vals))}
		}
		start := off
		for _, v := range vals {
			var buf [4]byte
			binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
			data = append(data, buf[:]...)
		}
		end := off + int64(len(vals)*4)
		header[name] = thdr{Dtype: "F32", Shape: shape, DataOffsets: [2]int64{start, end}}
		off = end
	}
	hb, err := json.Marshal(header)
	if err != nil {
		return err
	}
	// pad header to 8-byte multiple (common convention)
	for len(hb)%8 != 0 {
		hb = append(hb, ' ')
	}
	out := make([]byte, 0, 8+len(hb)+len(data))
	var lenbuf [8]byte
	binary.LittleEndian.PutUint64(lenbuf[:], uint64(len(hb)))
	out = append(out, lenbuf[:]...)
	out = append(out, hb...)
	out = append(out, data...)
	return os.WriteFile(path, out, 0o644)
}

func float16ToFloat32(h uint16) float32 {
	// IEEE half → float32
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	var fbits uint32
	switch exp {
	case 0:
		if mant == 0 {
			fbits = sign << 31
		} else {
			// subnormal
			exp32 := int32(-14 + 127)
			for mant&0x400 == 0 {
				mant <<= 1
				exp32--
			}
			mant &= 0x3ff
			fbits = (sign << 31) | (uint32(exp32) << 23) | (mant << 13)
		}
	case 31:
		fbits = (sign << 31) | (0xff << 23) | (mant << 13)
	default:
		fbits = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13)
	}
	return math.Float32frombits(fbits)
}
