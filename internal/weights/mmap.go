package weights

import (
	"fmt"
	"os"
	"syscall"
)

// mappedFile holds an mmap'd region (or a heap fallback) plus cleanup.
type mappedFile struct {
	Data   []byte
	Path   string
	Mapped bool
	fd     *os.File // kept for lifetime clarity; closed on Close
}

func openMapped(path string) (*mappedFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	size := st.Size()
	if size < 0 {
		_ = f.Close()
		return nil, fmt.Errorf("mmap: negative size")
	}
	if size == 0 {
		// empty mapping: keep empty slice, close fd
		_ = f.Close()
		return &mappedFile{Data: nil, Path: path, Mapped: false}, nil
	}
	// Cap absurd maps (defensive; OS still page-faults on demand)
	if size > 1<<40 {
		_ = f.Close()
		return nil, fmt.Errorf("mmap: file too large (%d)", size)
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		// fallback: heap read (small files / restricted environments)
		b, rerr := os.ReadFile(path)
		_ = f.Close()
		if rerr != nil {
			return nil, fmt.Errorf("mmap: %v (fallback read: %v)", err, rerr)
		}
		return &mappedFile{Data: b, Path: path, Mapped: false}, nil
	}
	return &mappedFile{Data: data, Path: path, Mapped: true, fd: f}, nil
}

func (m *mappedFile) Close() error {
	if m == nil {
		return nil
	}
	var err error
	if m.Mapped && m.Data != nil {
		err = syscall.Munmap(m.Data)
		m.Data = nil
		m.Mapped = false
	} else {
		m.Data = nil
	}
	if m.fd != nil {
		if cerr := m.fd.Close(); cerr != nil && err == nil {
			err = cerr
		}
		m.fd = nil
	}
	return err
}

// GGMLRawBytes returns the encoded byte length for nElem of the given type.
// Returns -1 for unknown/unsupported types.
func GGMLRawBytes(typ uint32, nElem int) int {
	if nElem < 0 {
		return -1
	}
	switch typ {
	case GGML_F32, GGML_I32:
		return nElem * 4
	case GGML_F16, GGML_BF16, GGML_I16:
		return nElem * 2
	case GGML_F64, GGML_I64:
		return nElem * 8
	case GGML_I8:
		return nElem
	case GGML_Q4_0:
		return nBlocks(nElem, QK4_0) * bsQ4_0
	case GGML_Q4_1:
		return nBlocks(nElem, QK4_1) * bsQ4_1
	case GGML_Q5_0:
		return nBlocks(nElem, QK5_0) * bsQ5_0
	case GGML_Q5_1:
		return nBlocks(nElem, QK5_1) * bsQ5_1
	case GGML_Q8_0:
		return nBlocks(nElem, QK8_0) * bsQ8_0
	case GGML_Q8_1:
		return nBlocks(nElem, QK8_1) * bsQ8_1
	case GGML_Q2_K:
		return nBlocks(nElem, QK_K) * bsQ2_K
	case GGML_Q3_K:
		return nBlocks(nElem, QK_K) * bsQ3_K
	case GGML_Q4_K:
		return nBlocks(nElem, QK_K) * bsQ4_K
	case GGML_Q5_K:
		return nBlocks(nElem, QK_K) * bsQ5_K
	case GGML_Q6_K:
		return nBlocks(nElem, QK_K) * bsQ6_K
	case GGML_Q8_K:
		return nBlocks(nElem, QK_K) * bsQ8_K
	case GGML_IQ4_NL:
		return nBlocks(nElem, QK4_NL) * bsIQ4_NL
	case GGML_TQ2_0:
		return nBlocks(nElem, QK_K) * bsTQ2_0
	case GGML_IQ4_XS:
		return nBlocks(nElem, QK_K) * bsIQ4_XS
	case GGML_TQ1_0:
		return nBlocks(nElem, QK_K) * bsTQ1_0
	case GGML_IQ2_XXS:
		return nBlocks(nElem, QK_K) * bsIQ2_XXS
	case GGML_IQ2_XS:
		return nBlocks(nElem, QK_K) * bsIQ2_XS
	case GGML_IQ2_S:
		return nBlocks(nElem, QK_K) * bsIQ2_S
	case GGML_IQ3_XXS:
		return nBlocks(nElem, QK_K) * bsIQ3_XXS
	case GGML_IQ3_S:
		return nBlocks(nElem, QK_K) * bsIQ3_S
	case GGML_IQ1_S:
		return nBlocks(nElem, QK_K) * bsIQ1_S
	case GGML_IQ1_M:
		return nBlocks(nElem, QK_K) * bsIQ1_M
	case GGML_MXFP4:
		return nBlocks(nElem, QK_MXFP4) * bsMXFP4
	case GGML_NVFP4:
		return nBlocks(nElem, QK_NVFP4) * bsNVFP4
	default:
		return -1
	}
}

func nBlocks(nElem, qk int) int {
	if qk <= 0 {
		return 0
	}
	return (nElem + qk - 1) / qk
}
