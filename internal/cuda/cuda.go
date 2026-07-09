// Package cuda is Trin's additive CUDA path (v1.02).
//
// Real device execution needs an NVIDIA driver + GPU. On hosts without either
// (typical CI/containers), DeviceCount returns 0 and the f32 kernels still run
// via a tiled CPU fallback that matches the CUDA-shaped row-major contract.
//
// Future real PTX/CUBLAS backends plug in behind the same API without renames.
package cuda

import (
	"os"
	"strings"
	"sync"
)

// Backend names reported by Backend().
const (
	BackendCUDA        = "cuda"
	BackendCPUFallback = "cpu-fallback"
)

// Device describes one logical CUDA device (when present).
type Device struct {
	Index int
	Name  string
}

var (
	probeOnce sync.Once
	probeSnap probeState
)

type probeState struct {
	count   int
	names   []string
	backend string
	reason  string
}

// probe inspects the host for NVIDIA driver nodes. No cgo / no libcuda required.
func probe() probeState {
	probeOnce.Do(func() {
		// Explicit force-off for deterministic tests.
		if v := strings.TrimSpace(os.Getenv("TRIN_CUDA")); v == "0" || strings.EqualFold(v, "off") || strings.EqualFold(v, "false") {
			probeSnap = probeState{count: 0, backend: BackendCPUFallback, reason: "TRIN_CUDA=0"}
			return
		}

		names := []string{}
		// Driver character devices (docker --gpus all typically mounts these).
		for i := 0; i < 8; i++ {
			p := "/dev/nvidia" + itoa(i)
			if fileExists(p) {
				names = append(names, "nvidia"+itoa(i))
			}
		}
		// Driver version file is a strong signal even if device nodes are restricted.
		if fileExists("/proc/driver/nvidia/version") && len(names) == 0 {
			// Count unknown — report 1 generic device when version is present.
			names = append(names, "nvidia")
		}
		// libcuda hints (userspace driver).
		for _, p := range []string{
			"/usr/lib/x86_64-linux-gnu/libcuda.so.1",
			"/usr/lib64/libcuda.so.1",
			"/usr/local/cuda/lib64/libcuda.so.1",
		} {
			if fileExists(p) && len(names) == 0 {
				names = append(names, "libcuda")
				break
			}
		}

		if len(names) == 0 {
			probeSnap = probeState{
				count:   0,
				backend: BackendCPUFallback,
				reason:  "no NVIDIA driver/device nodes",
			}
			return
		}
		probeSnap = probeState{
			count:   len(names),
			names:   names,
			backend: BackendCUDA,
			reason:  "nvidia device/driver visible",
		}
	})
	return probeSnap
}

// Available reports whether at least one CUDA device/driver signal is present.
func Available() bool { return probe().count > 0 }

// DeviceCount is the probed device count (0 on this CPU-only host).
func DeviceCount() int { return probe().count }

// Backend is "cuda" when a device is visible, else "cpu-fallback".
func Backend() string { return probe().backend }

// Reason is a short human probe note (debug/status).
func Reason() string { return probe().reason }

// DeviceName returns a short name for device i, or "" if out of range.
func DeviceName(i int) string {
	p := probe()
	if i < 0 || i >= len(p.names) {
		if i == 0 && p.count == 0 {
			return ""
		}
		return ""
	}
	return p.names[i]
}

// LinearF32 computes y[M] = W[M,K] @ x[K] (row-major W). Always defined;
// uses tiled CPU math that mirrors the CUDA kernel contract.
func LinearF32(W, x []float32, M, K int) ([]float32, error) {
	if M < 0 || K < 0 {
		return nil, errMsg("cuda_linear_f32: negative dims")
	}
	if len(W) < M*K || len(x) < K {
		return nil, errMsg("cuda_linear_f32: dimension mismatch")
	}
	out := make([]float32, M)
	// 16-wide tiles match portable hw_vlen.
	const tile = 16
	for i0 := 0; i0 < M; i0 += tile {
		i1 := i0 + tile
		if i1 > M {
			i1 = M
		}
		for i := i0; i < i1; i++ {
			acc := float32(0)
			row := W[i*K : i*K+K]
			// K peel by 4 for modest ILP; pure Go, no SIMD deps.
			t := 0
			for ; t+3 < K; t += 4 {
				acc += row[t]*x[t] + row[t+1]*x[t+1] + row[t+2]*x[t+2] + row[t+3]*x[t+3]
			}
			for ; t < K; t++ {
				acc += row[t] * x[t]
			}
			out[i] = acc
		}
	}
	return out, nil
}

// GemmF32 computes C[M,N] = A[M,K] @ B[K,N] (all row-major).
func GemmF32(A, B []float32, M, K, N int) ([]float32, error) {
	if M < 0 || K < 0 || N < 0 {
		return nil, errMsg("cuda_gemm_f32: negative dims")
	}
	if len(A) < M*K || len(B) < K*N {
		return nil, errMsg("cuda_gemm_f32: dimension mismatch")
	}
	C := make([]float32, M*N)
	const tile = 16
	for i0 := 0; i0 < M; i0 += tile {
		i1 := i0 + tile
		if i1 > M {
			i1 = M
		}
		for j0 := 0; j0 < N; j0 += tile {
			j1 := j0 + tile
			if j1 > N {
				j1 = N
			}
			for k0 := 0; k0 < K; k0 += tile {
				k1 := k0 + tile
				if k1 > K {
					k1 = K
				}
				for i := i0; i < i1; i++ {
					ai := A[i*K : i*K+K]
					ci := C[i*N : i*N+N]
					for k := k0; k < k1; k++ {
						aik := ai[k]
						bk := B[k*N : k*N+N]
						for j := j0; j < j1; j++ {
							ci[j] += aik * bk[j]
						}
					}
				}
			}
		}
	}
	return C, nil
}

// LinearTrit is y = scale * (W_trit[M,K] @ x[K]) with W entries in {-1,0,+1}.
// Same math as the CPU linear_trit path; exposed on the CUDA surface for
// device/fallback dispatch symmetry.
func LinearTrit(Wtrit []int32, x []float32, M, K int, scale float32) ([]float32, error) {
	if M < 0 || K < 0 {
		return nil, errMsg("cuda_linear_trit: negative dims")
	}
	if len(Wtrit) < M*K || len(x) < K {
		return nil, errMsg("cuda_linear_trit: dimension mismatch")
	}
	out := make([]float32, M)
	for i := 0; i < M; i++ {
		acc := float32(0)
		base := i * K
		for t := 0; t < K; t++ {
			w := Wtrit[base+t]
			if w > 1 {
				w = 1
			} else if w < -1 {
				w = -1
			}
			acc += float32(w) * x[t]
		}
		out[i] = scale * acc
	}
	return out, nil
}

// ResetProbeForTest clears the once-probe (tests only).
func ResetProbeForTest() {
	probeOnce = sync.Once{}
	probeSnap = probeState{}
}

type cudaError string

func (e cudaError) Error() string { return string(e) }

func errMsg(s string) error { return cudaError(s) }

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st != nil
}

func itoa(n int) string {
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
