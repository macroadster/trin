package weights

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

// GGML type codes we decode to f32 (block quants in ggml_dequant.go).
const (
	GGML_F32  = 0
	GGML_F16  = 1
	// Q4_0..Q6_K defined in ggml_dequant.go
	GGML_Q8_0 = 8
	GGML_I8   = 24
	GGML_I16  = 25
	GGML_I32  = 26
	GGML_I64  = 27
	GGML_F64  = 28
	GGML_BF16 = 30
)

// GGUFTensorInfo is one tensor descriptor.
type GGUFTensorInfo struct {
	Name   string
	NDims  uint32
	Dims   []uint64 // length NDims, dim[0] is typically the fastest-changing in ggml
	Type   uint32
	Offset uint64 // relative to data section start
	Shard  int    // multi-shard file index (0 = this GGUFFile); set by OpenGGUFSharded
}

// GGUFFile is a parsed GGUF container (LE only).
// Data is mmap'd when possible (v0.85); Close releases the mapping.
// Multi-shard sets (v1.00): OpenGGUFSharded fills Shards[1:] and tags
// GGUFTensorInfo.Shard; Load* route payload reads through shardFile().
type GGUFFile struct {
	Path       string
	Version    uint32
	Alignment  uint64
	Data       []byte
	DataBase   int64 // absolute offset of tensor data region
	Tensors    []GGUFTensorInfo
	ByName     map[string]GGUFTensorInfo
	Meta       map[string]string // string metadata only (summary)
	Mapped     bool
	mf         *mappedFile
	IsSharded  bool
	SplitCount uint16
	// Shards[0] is always nil (self); Shards[i] holds open file for split.no==i.
	Shards []*GGUFFile
}

// OpenGGUF parses a little-endian GGUF header via mmap (or heap fallback).
// Tensor payloads are not decoded until LoadF32; page faults touch only needed bytes.
func OpenGGUF(path string) (*GGUFFile, error) {
	mf, err := openMapped(path)
	if err != nil {
		return nil, err
	}
	b := mf.Data
	r := &byteReader{b: b, i: 0}
	fail := func(err error) (*GGUFFile, error) {
		_ = mf.Close()
		return nil, err
	}
	magic, err := r.u32()
	if err != nil {
		return fail(err)
	}
	if magic != 0x46554747 { // "GGUF" LE
		return fail(fmt.Errorf("gguf: bad magic %08x", magic))
	}
	ver, err := r.u32()
	if err != nil {
		return fail(err)
	}
	if ver < 2 || ver > 3 {
		// v2/v3 layout we implement; v1 exists but rare
		return fail(fmt.Errorf("gguf: unsupported version %d (need 2 or 3)", ver))
	}
	nTensors, err := r.u64()
	if err != nil {
		return fail(err)
	}
	nKV, err := r.u64()
	if err != nil {
		return fail(err)
	}
	f := &GGUFFile{
		Path:      path,
		Version:   ver,
		Alignment: 32,
		Data:      b,
		ByName:    make(map[string]GGUFTensorInfo),
		Meta:      make(map[string]string),
		Mapped:    mf.Mapped,
		mf:        mf,
	}
	for i := uint64(0); i < nKV; i++ {
		key, err := r.str()
		if err != nil {
			return fail(fmt.Errorf("gguf kv key: %w", err))
		}
		vt, err := r.u32()
		if err != nil {
			return fail(err)
		}
		val, err := r.readValue(vt)
		if err != nil {
			return fail(fmt.Errorf("gguf kv %q: %w", key, err))
		}
		if key == "general.alignment" {
			if u, ok := val.(uint64); ok && u > 0 {
				f.Alignment = u
			} else if u32, ok := val.(uint32); ok && u32 > 0 {
				f.Alignment = uint64(u32)
			}
		}
		if s, ok := val.(string); ok {
			f.Meta[key] = s
		} else {
			f.Meta[key] = fmt.Sprint(val)
		}
	}
	for i := uint64(0); i < nTensors; i++ {
		name, err := r.str()
		if err != nil {
			return fail(err)
		}
		nDims, err := r.u32()
		if err != nil {
			return fail(err)
		}
		if nDims > 4 {
			return fail(fmt.Errorf("gguf: tensor %q n_dims=%d", name, nDims))
		}
		dims := make([]uint64, nDims)
		for d := uint32(0); d < nDims; d++ {
			dims[d], err = r.u64()
			if err != nil {
				return fail(err)
			}
		}
		typ, err := r.u32()
		if err != nil {
			return fail(err)
		}
		off, err := r.u64()
		if err != nil {
			return fail(err)
		}
		ti := GGUFTensorInfo{Name: name, NDims: nDims, Dims: dims, Type: typ, Offset: off}
		f.Tensors = append(f.Tensors, ti)
		f.ByName[name] = ti
	}
	// data section starts at next alignment boundary after tensor info
	pos := r.i
	align := int64(f.Alignment)
	if align <= 0 {
		align = 32
	}
	if pos%align != 0 {
		pos = ((pos / align) + 1) * align
	}
	f.DataBase = pos
	return f, nil
}

// Close releases the underlying mmap (idempotent). Also closes multi-shards.
func (f *GGUFFile) Close() error {
	if f == nil {
		return nil
	}
	var err error
	for i := 1; i < len(f.Shards); i++ {
		if f.Shards[i] != nil {
			if e := f.Shards[i].Close(); e != nil && err == nil {
				err = e
			}
			f.Shards[i] = nil
		}
	}
	f.Shards = nil
	if f.mf != nil {
		if e := f.mf.Close(); e != nil && err == nil {
			err = e
		}
		f.mf = nil
	}
	f.Data = nil
	f.Mapped = false
	f.IsSharded = false
	return err
}

// IsMapped reports whether the file data is OS-mapped.
func (f *GGUFFile) IsMapped() bool {
	return f != nil && f.Mapped
}

// List returns tensor names in file order.
func (f *GGUFFile) List() []string {
	out := make([]string, len(f.Tensors))
	for i, t := range f.Tensors {
		out[i] = t.Name
	}
	return out
}

// LoadF32 decodes a named tensor to flat f32 (GGML dimension order preserved as stored).
func (f *GGUFFile) LoadF32(name string) ([]float32, []int64, error) {
	ti, ok := f.ByName[name]
	if !ok {
		return nil, nil, fmt.Errorf("gguf: tensor %q not found", name)
	}
	sf := f.shardFile(ti)
	nElem := uint64(1)
	shape := make([]int64, len(ti.Dims))
	for i, d := range ti.Dims {
		if d == 0 {
			return nil, nil, fmt.Errorf("gguf: zero dim")
		}
		nElem *= d
		shape[i] = int64(d)
	}
	base := sf.DataBase + int64(ti.Offset)
	if base < 0 || base > int64(len(sf.Data)) {
		return nil, nil, fmt.Errorf("gguf: bad data offset")
	}
	// Slice to tensor payload only when size is known — avoids retaining a
	// multi-GB file tail view after decode; OS only faults touched pages.
	raw := sf.Data[base:]
	if need := GGMLRawBytes(ti.Type, int(nElem)); need > 0 {
		if int64(len(sf.Data))-base < int64(need) {
			return nil, nil, fmt.Errorf("gguf: tensor %q short (%d need %d)", name, int64(len(sf.Data))-base, need)
		}
		raw = sf.Data[base : base+int64(need)]
	}
	vals, err := decodeGGML(ti.Type, raw, int(nElem))
	if err != nil {
		return nil, nil, err
	}
	return vals, shape, nil
}

// LoadF32Window decodes only [offset, offset+count) elements of a named tensor.
func (f *GGUFFile) LoadF32Window(name string, offset, count int) ([]float32, error) {
	ti, ok := f.ByName[name]
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q not found", name)
	}
	sf := f.shardFile(ti)
	nElem := 1
	for _, d := range ti.Dims {
		if d == 0 {
			return nil, fmt.Errorf("gguf: zero dim")
		}
		nElem *= int(d)
	}
	if offset < 0 || count < 0 || offset+count > nElem {
		return nil, fmt.Errorf("gguf window OOB: offset=%d count=%d nElem=%d", offset, count, nElem)
	}
	base := sf.DataBase + int64(ti.Offset)
	if base < 0 || base > int64(len(sf.Data)) {
		return nil, fmt.Errorf("gguf: bad data offset")
	}
	raw := sf.Data[base:]
	if need := GGMLRawBytes(ti.Type, nElem); need > 0 {
		if int64(len(sf.Data))-base < int64(need) {
			return nil, fmt.Errorf("gguf: tensor %q short (%d need %d)", name, int64(len(sf.Data))-base, need)
		}
		raw = sf.Data[base : base+int64(need)]
	}
	return DecodeGGMLWindow(ti.Type, raw, nElem, offset, count)
}

func decodeGGML(typ uint32, raw []byte, nElem int) ([]float32, error) {
	switch typ {
	case GGML_F32:
		need := nElem * 4
		if len(raw) < need {
			return nil, fmt.Errorf("gguf F32 short")
		}
		out := make([]float32, nElem)
		for i := 0; i < nElem; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out, nil
	case GGML_F64:
		need := nElem * 8
		if len(raw) < need {
			return nil, fmt.Errorf("gguf F64 short")
		}
		out := make([]float32, nElem)
		for i := 0; i < nElem; i++ {
			out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:])))
		}
		return out, nil
	case GGML_F16:
		need := nElem * 2
		if len(raw) < need {
			return nil, fmt.Errorf("gguf F16 short")
		}
		out := make([]float32, nElem)
		for i := 0; i < nElem; i++ {
			out[i] = float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return out, nil
	case GGML_BF16:
		need := nElem * 2
		if len(raw) < need {
			return nil, fmt.Errorf("gguf BF16 short")
		}
		out := make([]float32, nElem)
		for i := 0; i < nElem; i++ {
			bits := uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16
			out[i] = math.Float32frombits(bits)
		}
		return out, nil
	case GGML_I8:
		if len(raw) < nElem {
			return nil, fmt.Errorf("gguf I8 short")
		}
		out := make([]float32, nElem)
		for i := 0; i < nElem; i++ {
			out[i] = float32(int8(raw[i]))
		}
		return out, nil
	case GGML_I16:
		need := nElem * 2
		if len(raw) < need {
			return nil, fmt.Errorf("gguf I16 short")
		}
		out := make([]float32, nElem)
		for i := 0; i < nElem; i++ {
			out[i] = float32(int16(binary.LittleEndian.Uint16(raw[i*2:])))
		}
		return out, nil
	case GGML_I32:
		need := nElem * 4
		if len(raw) < need {
			return nil, fmt.Errorf("gguf I32 short")
		}
		out := make([]float32, nElem)
		for i := 0; i < nElem; i++ {
			out[i] = float32(int32(binary.LittleEndian.Uint32(raw[i*4:])))
		}
		return out, nil
	case GGML_I64:
		need := nElem * 8
		if len(raw) < need {
			return nil, fmt.Errorf("gguf I64 short")
		}
		out := make([]float32, nElem)
		for i := 0; i < nElem; i++ {
			out[i] = float32(int64(binary.LittleEndian.Uint64(raw[i*8:])))
		}
		return out, nil
	case GGML_Q4_0, GGML_Q4_1, GGML_Q5_0, GGML_Q5_1,
		GGML_Q8_0, GGML_Q8_1,
		GGML_Q2_K, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_K,
		GGML_IQ4_NL, GGML_TQ2_0, GGML_IQ4_XS, GGML_TQ1_0, GGML_IQ2_XXS, GGML_IQ2_XS, GGML_IQ2_S, GGML_IQ3_XXS, GGML_IQ3_S, GGML_IQ1_S, GGML_IQ1_M, GGML_MXFP4, GGML_NVFP4:
		return DecodeGGMLBlock(typ, raw, nElem)
	default:
		return nil, fmt.Errorf("gguf: unsupported ggml type %d (%s)", typ, GGMLTypeName(typ))
	}
}

// WriteGGUFTyped writes a minimal GGUF v3 with pre-encoded tensor bytes of a given type.
func WriteGGUFTyped(path string, name string, typ uint32, shape []uint64, data []byte, meta map[string]string) error {
	var buf []byte
	putU32 := func(v uint32) {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], v)
		buf = append(buf, b[:]...)
	}
	putU64 := func(v uint64) {
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], v)
		buf = append(buf, b[:]...)
	}
	putStr := func(s string) {
		putU64(uint64(len(s)))
		buf = append(buf, []byte(s)...)
	}
	if meta == nil {
		meta = map[string]string{}
	}
	putU32(0x46554747)
	putU32(3)
	putU64(1)
	putU64(uint64(1 + len(meta)))
	putStr("general.alignment")
	putU32(4)
	putU32(32)
	for k, v := range meta {
		putStr(k)
		putU32(8)
		putStr(v)
	}
	putStr(name)
	putU32(uint32(len(shape)))
	for _, d := range shape {
		putU64(d)
	}
	putU32(typ)
	putU64(0)
	align := uint64(32)
	for uint64(len(buf))%align != 0 {
		buf = append(buf, 0)
	}
	buf = append(buf, data...)
	return os.WriteFile(path, buf, 0o644)
}

// WriteGGUFF32 writes a minimal GGUF v3 file with F32 tensors only (LE).
func WriteGGUFF32(path string, tensors map[string][]float32, shapes map[string][]uint64, meta map[string]string) error {
	var buf []byte
	putU32 := func(v uint32) {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], v)
		buf = append(buf, b[:]...)
	}
	putU64 := func(v uint64) {
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], v)
		buf = append(buf, b[:]...)
	}
	putStr := func(s string) {
		putU64(uint64(len(s)))
		buf = append(buf, []byte(s)...)
	}
	// magic + version 3
	putU32(0x46554747)
	putU32(3)
	// stable name list
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	for i := 0; i < len(names); i++ {
		for j := i + 1; j < len(names); j++ {
			if names[j] < names[i] {
				names[i], names[j] = names[j], names[i]
			}
		}
	}
	// metadata: general.alignment + user strings
	if meta == nil {
		meta = map[string]string{}
	}
	// n_tensors, n_kv
	nKV := uint64(1 + len(meta)) // alignment + strings
	putU64(uint64(len(names)))
	putU64(nKV)
	// general.alignment as uint32 value type 4
	putStr("general.alignment")
	putU32(4) // UINT32
	putU32(32)
	for k, v := range meta {
		putStr(k)
		putU32(8) // STRING
		putStr(v)
	}
	// tensor infos; offsets sequential, padded per tensor to 32
	var data []byte
	var off uint64
	align := uint64(32)
	for _, name := range names {
		vals := tensors[name]
		shape := shapes[name]
		if shape == nil {
			shape = []uint64{uint64(len(vals))}
		}
		putStr(name)
		putU32(uint32(len(shape)))
		for _, d := range shape {
			putU64(d)
		}
		putU32(GGML_F32)
		// align offset
		if off%align != 0 {
			pad := align - (off % align)
			data = append(data, make([]byte, pad)...)
			off += pad
		}
		putU64(off)
		for _, v := range vals {
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
			data = append(data, b[:]...)
		}
		off += uint64(len(vals) * 4)
	}
	// pad header to alignment
	for uint64(len(buf))%align != 0 {
		buf = append(buf, 0)
	}
	buf = append(buf, data...)
	return os.WriteFile(path, buf, 0o644)
}

type byteReader struct {
	b []byte
	i int64
}

func (r *byteReader) need(n int) error {
	if r.i+int64(n) > int64(len(r.b)) {
		return io.ErrUnexpectedEOF
	}
	return nil
}

func (r *byteReader) u32() (uint32, error) {
	if err := r.need(4); err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint32(r.b[r.i:])
	r.i += 4
	return v, nil
}

func (r *byteReader) u64() (uint64, error) {
	if err := r.need(8); err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint64(r.b[r.i:])
	r.i += 8
	return v, nil
}

func (r *byteReader) str() (string, error) {
	n, err := r.u64()
	if err != nil {
		return "", err
	}
	if n > 1<<20 {
		return "", fmt.Errorf("string too long")
	}
	if err := r.need(int(n)); err != nil {
		return "", err
	}
	s := string(r.b[r.i : r.i+int64(n)])
	r.i += int64(n)
	return s, nil
}

func (r *byteReader) readValue(vt uint32) (any, error) {
	switch vt {
	case 0: // u8
		if err := r.need(1); err != nil {
			return nil, err
		}
		v := r.b[r.i]
		r.i++
		return uint64(v), nil
	case 1: // i8
		if err := r.need(1); err != nil {
			return nil, err
		}
		v := int8(r.b[r.i])
		r.i++
		return int64(v), nil
	case 2: // u16
		if err := r.need(2); err != nil {
			return nil, err
		}
		v := binary.LittleEndian.Uint16(r.b[r.i:])
		r.i += 2
		return uint64(v), nil
	case 3: // i16
		if err := r.need(2); err != nil {
			return nil, err
		}
		v := int16(binary.LittleEndian.Uint16(r.b[r.i:]))
		r.i += 2
		return int64(v), nil
	case 4: // u32
		v, err := r.u32()
		return v, err
	case 5: // i32
		v, err := r.u32()
		return int32(v), err
	case 6: // f32
		v, err := r.u32()
		return math.Float32frombits(v), err
	case 7: // bool
		if err := r.need(1); err != nil {
			return nil, err
		}
		v := r.b[r.i] != 0
		r.i++
		return v, nil
	case 8: // string
		return r.str()
	case 9: // array
		et, err := r.u32()
		if err != nil {
			return nil, err
		}
		n, err := r.u64()
		if err != nil {
			return nil, err
		}
		if n > 1<<24 {
			return nil, fmt.Errorf("array too large")
		}
		// skip elements (or collect)
		arr := make([]any, 0, n)
		for i := uint64(0); i < n; i++ {
			v, err := r.readValue(et)
			if err != nil {
				return nil, err
			}
			arr = append(arr, v)
		}
		return arr, nil
	case 10: // u64
		return r.u64()
	case 11: // i64
		v, err := r.u64()
		return int64(v), err
	case 12: // f64
		v, err := r.u64()
		return math.Float64frombits(v), err
	default:
		return nil, fmt.Errorf("unknown value type %d", vt)
	}
}

// TypeName maps ggml type code to string for listing.
func GGMLTypeName(t uint32) string {
	switch t {
	case GGML_F32:
		return "F32"
	case GGML_F16:
		return "F16"
	case GGML_Q4_0:
		return "Q4_0"
	case GGML_Q4_1:
		return "Q4_1"
	case GGML_Q5_0:
		return "Q5_0"
	case GGML_Q5_1:
		return "Q5_1"
	case GGML_Q8_0:
		return "Q8_0"
	case GGML_Q8_1:
		return "Q8_1"
	case GGML_Q2_K:
		return "Q2_K"
	case GGML_Q3_K:
		return "Q3_K"
	case GGML_Q4_K:
		return "Q4_K"
	case GGML_Q5_K:
		return "Q5_K"
	case GGML_Q6_K:
		return "Q6_K"
	case GGML_Q8_K:
		return "Q8_K"
	case GGML_IQ4_NL:
		return "IQ4_NL"
	case GGML_IQ4_XS:
		return "IQ4_XS"
	case GGML_TQ2_0:
		return "TQ2_0"
	case GGML_TQ1_0:
		return "TQ1_0"
	case GGML_IQ2_XXS:
		return "IQ2_XXS"
	case GGML_IQ2_XS:
		return "IQ2_XS"
	case GGML_IQ2_S:
		return "IQ2_S"
	case GGML_IQ3_XXS:
		return "IQ3_XXS"
	case GGML_IQ3_S:
		return "IQ3_S"
	case GGML_IQ1_S:
		return "IQ1_S"
	case GGML_IQ1_M:
		return "IQ1_M"
	case GGML_MXFP4:
		return "MXFP4"
	case GGML_NVFP4:
		return "NVFP4"
	case GGML_I8:
		return "I8"
	case GGML_I16:
		return "I16"
	case GGML_I32:
		return "I32"
	case GGML_I64:
		return "I64"
	case GGML_F64:
		return "F64"
	case GGML_BF16:
		return "BF16"
	default:
		return fmt.Sprintf("T%d", t)
	}
}

// AutoLoad detects format by extension or magic and loads named tensor.
// Uses mmap open paths; closes the handle after the single tensor load.
func AutoLoad(path, name string) ([]float32, []int64, error) {
	lower := strings.ToLower(path)
	switch {
	case strings.HasSuffix(lower, ".safetensors"):
		f, err := OpenSafeTensors(path)
		if err != nil {
			return nil, nil, err
		}
		defer f.Close()
		return f.LoadF32(name)
	case strings.HasSuffix(lower, ".gguf"):
		f, err := OpenGGUF(path)
		if err != nil {
			return nil, nil, err
		}
		defer f.Close()
		return f.LoadF32(name)
	default:
		// magic sniff — only first 8 bytes
		fd, err := os.Open(path)
		if err != nil {
			return nil, nil, err
		}
		var hdr [8]byte
		n, _ := fd.Read(hdr[:])
		_ = fd.Close()
		if n >= 4 && string(hdr[0:4]) == "GGUF" {
			f, err := OpenGGUF(path)
			if err != nil {
				return nil, nil, err
			}
			defer f.Close()
			return f.LoadF32(name)
		}
		f, err := OpenSafeTensors(path)
		if err != nil {
			return nil, nil, fmt.Errorf("unknown weight format for %s: %w", path, err)
		}
		defer f.Close()
		return f.LoadF32(name)
	}
}
