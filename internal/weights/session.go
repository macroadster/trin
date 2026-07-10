package weights

import (
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
)

// WeightSession is a retained open GGUF or safetensors file (v0.86).
// Identified by a positive int32 id for the Trin language surface.
type WeightSession struct {
	ID     int32
	Kind   string // "gguf" | "safetensors"
	Path   string
	Mapped bool
	gguf   *GGUFFile
	st     *SafeTensorsFile

	// tritPack caches absmean-quantized trit planes per tensor name (v1.07).
	// Lazily filled by PrepackTrit / LinearTritCached; cleared on session close.
	tritMu   sync.Mutex
	tritPack map[string]*TritPack

	// nameMap maps short/canonical names → on-disk tensor names (v1.17).
	// nil = identity. Lookups resolve before List misses.
	nameMu  sync.Mutex
	nameMap *NameMap
}

// TritPack is a whole-tensor absmean trit packing retained on a session.
// One global scale over the flat M*K body (matches LinearTrit with tileRows=M).
type TritPack struct {
	Name  string
	M     int
	K     int
	Scale float32
	W     []int8    // row-major M*K trits
	Pos   []uint32  // M * nw packed positive planes
	Neg   []uint32  // M * nw packed negative planes
	NW    int       // words per row: (K+31)/32
}

var (
	sessMu   sync.Mutex
	sessMap  = map[int32]*WeightSession{}
	sessNext int32 = 1 // atomic-ish via mutex + add
)

// OpenSession opens path as GGUF or safetensors (ext/magic) and retains mmap until CloseSession.
func OpenSession(path string) (*WeightSession, error) {
	lower := strings.ToLower(path)
	var s *WeightSession
	switch {
	case strings.HasSuffix(lower, ".safetensors"):
		f, err := OpenSafeTensors(path)
		if err != nil {
			return nil, err
		}
		s = &WeightSession{Kind: "safetensors", Path: path, Mapped: f.Mapped, st: f}
	case strings.HasSuffix(lower, ".gguf"):
		// OpenGGUFSharded: single-file when no split.* meta; multi-shard otherwise.
		f, err := OpenGGUFSharded(path)
		if err != nil {
			return nil, err
		}
		s = &WeightSession{Kind: "gguf", Path: path, Mapped: f.Mapped, gguf: f}
	default:
		// magic sniff first 8 bytes
		f, err := OpenGGUFSharded(path)
		if err == nil {
			s = &WeightSession{Kind: "gguf", Path: path, Mapped: f.Mapped, gguf: f}
		} else {
			st, err2 := OpenSafeTensors(path)
			if err2 != nil {
				return nil, fmt.Errorf("open_weight_session: %v / %v", err, err2)
			}
			s = &WeightSession{Kind: "safetensors", Path: path, Mapped: st.Mapped, st: st}
		}
	}
	sessMu.Lock()
	id := atomic.AddInt32(&sessNext, 1) - 1
	// ensure positive unique
	if id <= 0 {
		id = 1
		sessNext = 2
	}
	s.ID = id
	sessMap[id] = s
	sessMu.Unlock()
	return s, nil
}

// GetSession returns a live session or error.
func GetSession(id int32) (*WeightSession, error) {
	sessMu.Lock()
	defer sessMu.Unlock()
	s, ok := sessMap[id]
	if !ok || s == nil {
		return nil, fmt.Errorf("weight session %d not found (closed?)", id)
	}
	return s, nil
}

// CloseSession releases mmap and removes the id (idempotent: already-closed is ok).
func CloseSession(id int32) error {
	sessMu.Lock()
	s, ok := sessMap[id]
	if ok {
		delete(sessMap, id)
	}
	sessMu.Unlock()
	if !ok || s == nil {
		return nil
	}
	s.ClearTritPacks()
	var err error
	if s.gguf != nil {
		err = s.gguf.Close()
		s.gguf = nil
	}
	if s.st != nil {
		if e := s.st.Close(); e != nil && err == nil {
			err = e
		}
		s.st = nil
	}
	s.Mapped = false
	return err
}

// ClearTritPacks drops all cached trit packs for this session.
func (s *WeightSession) ClearTritPacks() {
	if s == nil {
		return
	}
	s.tritMu.Lock()
	s.tritPack = nil
	s.tritMu.Unlock()
}

// List returns tensor names without decoding bodies.
func (s *WeightSession) List() ([]string, error) {
	if s == nil {
		return nil, fmt.Errorf("nil session")
	}
	switch s.Kind {
	case "gguf":
		if s.gguf == nil {
			return nil, fmt.Errorf("session closed")
		}
		return s.gguf.List(), nil
	case "safetensors":
		if s.st == nil {
			return nil, fmt.Errorf("session closed")
		}
		return s.st.List(), nil
	default:
		return nil, fmt.Errorf("unknown session kind")
	}
}

// resolveOnDisk applies the session NameMap (identity if unset).
func (s *WeightSession) resolveOnDisk(name string) string {
	if s == nil {
		return name
	}
	return s.ResolveName(name)
}

// Shape returns dims for a tensor name (accepts short or on-disk names).
func (s *WeightSession) Shape(name string) ([]int64, error) {
	if s == nil {
		return nil, fmt.Errorf("nil session")
	}
	name = s.resolveOnDisk(name)
	switch s.Kind {
	case "gguf":
		if s.gguf == nil {
			return nil, fmt.Errorf("session closed")
		}
		ti, ok := s.gguf.ByName[name]
		if !ok {
			return nil, fmt.Errorf("tensor %q not found", name)
		}
		out := make([]int64, len(ti.Dims))
		for i, d := range ti.Dims {
			out[i] = int64(d)
		}
		return out, nil
	case "safetensors":
		if s.st == nil {
			return nil, fmt.Errorf("session closed")
		}
		info, ok := s.st.Tensors[name]
		if !ok {
			return nil, fmt.Errorf("tensor %q not found", name)
		}
		return append([]int64(nil), info.Shape...), nil
	default:
		return nil, fmt.Errorf("unknown session kind")
	}
}

// Meta returns GGUF string metadata value (empty if missing / safetensors).
func (s *WeightSession) Meta(key string) string {
	if s == nil || s.gguf == nil {
		return ""
	}
	return s.gguf.Meta[key]
}

// LoadF32 decodes one named tensor using the retained mapping.
// Name is resolved through the session NameMap when set (v1.17).
func (s *WeightSession) LoadF32(name string) ([]float32, []int64, error) {
	if s == nil {
		return nil, nil, fmt.Errorf("nil session")
	}
	name = s.resolveOnDisk(name)
	switch s.Kind {
	case "gguf":
		if s.gguf == nil {
			return nil, nil, fmt.Errorf("session closed")
		}
		return s.gguf.LoadF32(name)
	case "safetensors":
		if s.st == nil {
			return nil, nil, fmt.Errorf("session closed")
		}
		return s.st.LoadF32(name)
	default:
		return nil, nil, fmt.Errorf("unknown session kind")
	}
}

// SessionCount is for tests.
func SessionCount() int {
	sessMu.Lock()
	defer sessMu.Unlock()
	return len(sessMap)
}


// LoadF32Window loads a contiguous element window from a retained session.
// Name is resolved through the session NameMap when set (v1.17).
func (s *WeightSession) LoadF32Window(name string, offset, count int) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("nil session")
	}
	name = s.resolveOnDisk(name)
	switch s.Kind {
	case "gguf":
		if s.gguf == nil {
			return nil, fmt.Errorf("session closed")
		}
		return s.gguf.LoadF32Window(name, offset, count)
	case "safetensors":
		if s.st == nil {
			return nil, fmt.Errorf("session closed")
		}
		return s.st.LoadF32Window(name, offset, count)
	default:
		return nil, fmt.Errorf("unknown session kind")
	}
}
