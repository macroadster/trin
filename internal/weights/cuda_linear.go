// cuda_linear.go — CUDA-backed GGUF linear algebra on weight sessions.
//
// When a GPU is available the dequant-fused PTX kernels run on device;
// otherwise the cuda package's CPU fallback dequant matvec is used, which
// mirrors the PTX numerical contract exactly.
package weights

import (
	"fmt"

	"github.com/macroadster/trin/internal/cuda"
)

// TensorQType returns the GGML quant type for a named tensor in a GGUF session.
// Returns (type, nElem, error). Non-GGUF sessions return an error.
func (s *WeightSession) TensorQType(name string) (uint32, int, error) {
	if s == nil {
		return 0, 0, fmt.Errorf("nil session")
	}
	name = s.resolveOnDisk(name)
	if s.Kind != "gguf" || s.gguf == nil {
		return 0, 0, fmt.Errorf("tensor_qtype: requires GGUF session")
	}
	ti, ok := s.gguf.ByName[name]
	if !ok {
		return 0, 0, fmt.Errorf("tensor_qtype: %q not found", name)
	}
	nElem := 1
	for _, d := range ti.Dims {
		nElem *= int(d)
	}
	return ti.Type, nElem, nil
}

// LoadRawBlocks returns the raw encoded bytes for a named GGUF tensor.
// The bytes can be fed directly to the CUDA dequant kernels.
func (s *WeightSession) LoadRawBlocks(name string) ([]byte, uint32, int, error) {
	if s == nil {
		return nil, 0, 0, fmt.Errorf("nil session")
	}
	name = s.resolveOnDisk(name)
	if s.Kind != "gguf" || s.gguf == nil {
		return nil, 0, 0, fmt.Errorf("load_raw_blocks: requires GGUF session")
	}
	ti, ok := s.gguf.ByName[name]
	if !ok {
		return nil, 0, 0, fmt.Errorf("load_raw_blocks: %q not found", name)
	}
	nElem := 1
	for _, d := range ti.Dims {
		nElem *= int(d)
	}
	sf := s.gguf.shardFile(ti)
	base := sf.DataBase + int64(ti.Offset)
	if base < 0 || base > int64(len(sf.Data)) {
		return nil, 0, 0, fmt.Errorf("load_raw_blocks: bad offset")
	}
	need := GGMLRawBytes(ti.Type, nElem)
	if need < 0 {
		return nil, 0, 0, fmt.Errorf("load_raw_blocks: unknown type %d", ti.Type)
	}
	if int64(len(sf.Data))-base < int64(need) {
		return nil, 0, 0, fmt.Errorf("load_raw_blocks: data short")
	}
	raw := sf.Data[base : base+int64(need)]
	return raw, ti.Type, nElem, nil
}

// CUDALinear computes y = W @ x using the CUDA dequant-fused path when W is
// stored as Q4_0 or Q8_0. For F32/F16/other types it falls back to the regular
// decode-then-matvec path.
//
// M = output rows, K = inner dimension (columns of W / length of x).
// K must be a multiple of 32 for block quant types.
func (s *WeightSession) CUDALinear(name string, x []float32, M, K int) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("nil session")
	}
	if M <= 0 || K <= 0 {
		return nil, fmt.Errorf("cuda_session_linear: bad M/K")
	}
	if len(x) < K {
		return nil, fmt.Errorf("cuda_session_linear: x short")
	}

	name = s.resolveOnDisk(name)

	// For GGUF sessions, try the fused dequant path.
	if s.Kind == "gguf" && s.gguf != nil {
		ti, ok := s.gguf.ByName[name]
		if !ok {
			return nil, fmt.Errorf("cuda_session_linear: tensor %q not found", name)
		}
		switch ti.Type {
		case GGML_Q4_0:
			raw, _, _, err := s.LoadRawBlocks(name)
			if err != nil {
				return nil, err
			}
			return cuda.DequantLinearQ4_0(raw, x, M, K)
		case GGML_Q8_0:
			raw, _, _, err := s.LoadRawBlocks(name)
			if err != nil {
				return nil, err
			}
			return cuda.DequantLinearQ8_0(raw, x, M, K)
		}
	}

	// Fallback: decode to f32, then use cuda.LinearF32.
	return s.Linear(name, x, M, K, 0)
}

// CUDALinearTrit computes y = scale * (trit(W) @ x) using the CUDA trit path.
// First decodes W from GGUF, absmean-quantizes to trits, then uses the
// cuda.LinearTrit CPU fallback (or device kernel when available).
func (s *WeightSession) CUDALinearTrit(name string, x []float32, M, K int) ([]float32, float32, error) {
	if s == nil {
		return nil, 0, fmt.Errorf("nil session")
	}
	if M <= 0 || K <= 0 {
		return nil, 0, fmt.Errorf("cuda_session_linear_trit: bad M/K")
	}
	if len(x) < K {
		return nil, 0, fmt.Errorf("cuda_session_linear_trit: x short")
	}

	// Load + dequant to f32, then trit-quantize + matvec via CUDA path.
	wF, err := s.LoadF32Window(name, 0, M*K)
	if err != nil {
		return nil, 0, err
	}
	W, scale := QuantizeAbsMeanTrits(wF)
	// Convert int8 trits to int32 for cuda.LinearTrit
	W32 := make([]int32, len(W))
	for i, t := range W {
		W32[i] = int32(t)
	}
	y, err := cuda.LinearTrit(W32, x, M, K, scale)
	if err != nil {
		return nil, 0, err
	}
	return y, scale, nil
}

// CUDABackend returns the active CUDA backend string (for status reporting).
func CUDABackend() string {
	return cuda.Backend()
}

// CUDAAvailable reports whether a CUDA device is visible.
func CUDAAvailable() bool {
	return cuda.Available()
}
