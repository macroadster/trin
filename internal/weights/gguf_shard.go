package weights

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// llama.cpp multi-shard GGUF metadata keys (gguf-py Keys.Split / LLM_KV_SPLIT_*).
const (
	MetaSplitNo           = "split.no"
	MetaSplitCount        = "split.count"
	MetaSplitTensorsCount = "split.tensors.count"
)

// GGUFTensorInfo.Shard is set by OpenGGUFSharded (file index into GGUFFile.Shards).
// Zero for single-file OpenGGUF (and for tensors living on shard 0).

// OpenGGUFSharded opens a llama.cpp multi-shard GGUF set.
// path may be any shard filename matching "<prefix>-%05d-of-%05d.gguf"; the
// implementation always starts from split.no==0 and loads all split.count shards.
// Non-sharded files (missing split.count or count<=1) fall through to OpenGGUF.
//
// Compatibility: OpenGGUF still opens a single file without walking siblings.
// Sessions use OpenGGUFSharded so .gguf paths with split.* metadata auto-expand.
func OpenGGUFSharded(path string) (*GGUFFile, error) {
	first, err := OpenGGUF(path)
	if err != nil {
		return nil, err
	}
	nSplit, hasCount := metaU16(first.Meta, MetaSplitCount)
	if !hasCount || nSplit <= 1 {
		// single file
		return first, nil
	}
	idx, hasIdx := metaU16(first.Meta, MetaSplitNo)
	if !hasIdx {
		_ = first.Close()
		return nil, fmt.Errorf("gguf: multi-shard missing %s in %s", MetaSplitNo, path)
	}
	// Always load from shard 0; if caller passed a non-zero shard, resolve siblings.
	paths, err := listShardPaths(path, int(idx), int(nSplit))
	if err != nil {
		_ = first.Close()
		return nil, err
	}
	if idx != 0 {
		_ = first.Close()
		first, err = OpenGGUF(paths[0])
		if err != nil {
			return nil, err
		}
		// re-read idx from first file
		idx, hasIdx = metaU16(first.Meta, MetaSplitNo)
		if !hasIdx || idx != 0 {
			_ = first.Close()
			return nil, fmt.Errorf("gguf: first shard %s must have %s=0", paths[0], MetaSplitNo)
		}
	}

	// Promote first into multi-shard holder.
	// first.Shards[0] remains nil (payload lives on first itself).
	// Additional shards append under first.Shards with indices 1..n-1.
	// Tensor infos get Shard file index.
	// Tag tensors already on first as shard 0.
	for i := range first.Tensors {
		first.Tensors[i].Shard = 0
		first.ByName[first.Tensors[i].Name] = first.Tensors[i]
	}
	first.Shards = make([]*GGUFFile, nSplit)
	first.Shards[0] = nil // self
	first.IsSharded = true
	first.SplitCount = nSplit

	for si := 1; si < int(nSplit); si++ {
		sf, err := OpenGGUF(paths[si])
		if err != nil {
			_ = first.Close()
			return nil, fmt.Errorf("gguf: open shard %d (%s): %w", si, paths[si], err)
		}
		got, ok := metaU16(sf.Meta, MetaSplitNo)
		if !ok || int(got) != si {
			_ = sf.Close()
			_ = first.Close()
			return nil, fmt.Errorf("gguf: shard %s has split.no=%v want %d", paths[si], sf.Meta[MetaSplitNo], si)
		}
		for _, ti := range sf.Tensors {
			if _, dup := first.ByName[ti.Name]; dup {
				_ = sf.Close()
				_ = first.Close()
				return nil, fmt.Errorf("gguf: duplicate tensor %q across shards", ti.Name)
			}
			ti.Shard = si
			first.Tensors = append(first.Tensors, ti)
			first.ByName[ti.Name] = ti
		}
		// Merge string meta (prefer first); keep split keys from first.
		for k, v := range sf.Meta {
			if _, exists := first.Meta[k]; !exists {
				first.Meta[k] = v
			}
		}
		first.Shards[si] = sf
		if !sf.Mapped {
			first.Mapped = false
		}
	}

	// Optional integrity vs split.tensors.count
	if want, ok := metaU16(first.Meta, MetaSplitTensorsCount); ok {
		if int(want) != len(first.Tensors) {
			_ = first.Close()
			return nil, fmt.Errorf("gguf: split.tensors.count=%d but loaded %d tensors", want, len(first.Tensors))
		}
	}
	return first, nil
}

// shardFile returns the GGUFFile that owns the tensor payload.
func (f *GGUFFile) shardFile(ti GGUFTensorInfo) *GGUFFile {
	if f == nil {
		return nil
	}
	if ti.Shard <= 0 || ti.Shard >= len(f.Shards) || f.Shards[ti.Shard] == nil {
		return f
	}
	return f.Shards[ti.Shard]
}

func metaU16(meta map[string]string, key string) (uint16, bool) {
	if meta == nil {
		return 0, false
	}
	s, ok := meta[key]
	if !ok || s == "" {
		return 0, false
	}
	// fmt.Sprint of uint64 etc. — accept base-10
	n, err := strconv.ParseUint(s, 10, 16)
	if err != nil {
		return 0, false
	}
	return uint16(n), true
}

// listShardPaths builds "<prefix>-%05d-of-%05d.gguf" for idx 0..nSplit-1.
// Accepts any current shard path whose basename ends with -IIIII-of-NNNNN.gguf.
func listShardPaths(path string, idx, nSplit int) ([]string, error) {
	if nSplit <= 0 {
		return nil, fmt.Errorf("gguf: invalid split count %d", nSplit)
	}
	dir := filepath.Dir(path)
	base := filepath.Base(path)
	// strip .gguf
	lower := strings.ToLower(base)
	if !strings.HasSuffix(lower, ".gguf") {
		return nil, fmt.Errorf("gguf: shard path must end with .gguf: %s", path)
	}
	stem := base[:len(base)-len(".gguf")]
	// find "-%05d-of-%05d" suffix
	// expect ...-0000I-of-0000N
	ofTag := fmt.Sprintf("-of-%05d", nSplit)
	// also allow unpadded via known idx
	candidates := []string{
		fmt.Sprintf("-%05d-of-%05d", idx, nSplit),
		fmt.Sprintf("-%d-of-%d", idx, nSplit),
	}
	var prefix string
	for _, suf := range candidates {
		if strings.HasSuffix(stem, suf) {
			prefix = stem[:len(stem)-len(suf)]
			break
		}
	}
	if prefix == "" {
		// try generic parse: last "-of-"
		i := strings.LastIndex(stem, "-of-")
		if i <= 0 {
			return nil, fmt.Errorf("gguf: cannot parse split filename %s (need <name>-%%05d-of-%%05d.gguf)", base)
		}
		// back up over digits and dash of idx
		j := i - 1
		for j >= 0 && stem[j] >= '0' && stem[j] <= '9' {
			j--
		}
		if j < 0 || stem[j] != '-' {
			return nil, fmt.Errorf("gguf: cannot parse split idx in %s", base)
		}
		prefix = stem[:j]
		// verify of count
		if !strings.HasSuffix(stem, ofTag) && !strings.HasSuffix(stem, fmt.Sprintf("-of-%d", nSplit)) {
			// still accept if numeric of-part matches nSplit
			ofPart := stem[i+4:]
			got, err := strconv.Atoi(ofPart)
			if err != nil || got != nSplit {
				return nil, fmt.Errorf("gguf: filename of-count %q != split.count %d", ofPart, nSplit)
			}
		}
	}
	out := make([]string, nSplit)
	for i := 0; i < nSplit; i++ {
		name := fmt.Sprintf("%s-%05d-of-%05d.gguf", prefix, i, nSplit)
		out[i] = filepath.Join(dir, name)
	}
	return out, nil
}

// TypedMeta is a GGUF KV entry for writers (v1.00).
// Kind: "u16"|"u32"|"u64"|"string" (default string).
type TypedMeta struct {
	Kind string
	U64  uint64
	Str  string
}

// WriteGGUFTypedMulti writes one GGUF with multiple pre-encoded tensors + typed KV.
// Tensor order is the order of the names slice.
func WriteGGUFTypedMulti(path string, names []string, types map[string]uint32, shapes map[string][]uint64, data map[string][]byte, meta map[string]TypedMeta) error {
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
	putU16 := func(v uint16) {
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], v)
		buf = append(buf, b[:]...)
	}
	putStr := func(s string) {
		putU64(uint64(len(s)))
		buf = append(buf, []byte(s)...)
	}
	if meta == nil {
		meta = map[string]TypedMeta{}
	}
	// always include alignment if missing
	if _, ok := meta["general.alignment"]; !ok {
		meta["general.alignment"] = TypedMeta{Kind: "u32", U64: 32}
	}
	// stable meta key order: alignment first, then sorted
	keys := make([]string, 0, len(meta))
	for k := range meta {
		if k != "general.alignment" {
			keys = append(keys, k)
		}
	}
	// sort
	for i := 0; i < len(keys); i++ {
		for j := i + 1; j < len(keys); j++ {
			if keys[j] < keys[i] {
				keys[i], keys[j] = keys[j], keys[i]
			}
		}
	}
	metaOrder := append([]string{"general.alignment"}, keys...)

	putU32(0x46554747)
	putU32(3)
	putU64(uint64(len(names)))
	putU64(uint64(len(metaOrder)))
	for _, k := range metaOrder {
		m := meta[k]
		putStr(k)
		switch m.Kind {
		case "u16":
			putU32(2) // UINT16
			putU16(uint16(m.U64))
		case "u32":
			putU32(4)
			putU32(uint32(m.U64))
		case "u64":
			putU32(10)
			putU64(m.U64)
		default:
			putU32(8)
			putStr(m.Str)
		}
	}
	// tensor infos with running data offset
	var payload []byte
	var off uint64
	align := uint64(32)
	for _, name := range names {
		shape := shapes[name]
		if shape == nil {
			return fmt.Errorf("WriteGGUFTypedMulti: missing shape for %q", name)
		}
		raw := data[name]
		typ := types[name]
		putStr(name)
		putU32(uint32(len(shape)))
		for _, d := range shape {
			putU64(d)
		}
		putU32(typ)
		if off%align != 0 {
			pad := align - (off % align)
			payload = append(payload, make([]byte, pad)...)
			off += pad
		}
		putU64(off)
		payload = append(payload, raw...)
		off += uint64(len(raw))
	}
	for uint64(len(buf))%align != 0 {
		buf = append(buf, 0)
	}
	buf = append(buf, payload...)
	return os.WriteFile(path, buf, 0o644)
}
