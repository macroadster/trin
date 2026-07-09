package cuda

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
)

// PTX kernel entry names (stable; additive surface).
const (
	PTXLinearF32Entry = "trin_linear_f32"
	PTXGemmF32Entry   = "trin_gemm_f32"
	PTXLinearTritEntry = "trin_linear_trit"
)

// DefaultPTXTarget is the ISA target string for generated .ptx.
// sm_75 (Turing) is a conservative middle; override with TRIN_CUDA_ARCH.
const DefaultPTXTarget = "sm_75"

// PTXModule is a named blob of PTX source for one kernel family.
type PTXModule struct {
	Name   string // entry point
	Source string // full .ptx text
	Arch   string // e.g. sm_75
}

// CubinResult is offline ptxas output (when available).
type CubinResult struct {
	Arch   string
	Bytes  []byte
	Path   string // temp path if written; may be empty after cleanup
	Tool   string // ptxas path used
	Stderr string
}

var (
	ptxasOnce sync.Once
	ptxasPath string
	ptxasOK   bool
)

// PTXArch returns the active arch (TRIN_CUDA_ARCH or DefaultPTXTarget).
func PTXArch() string {
	if v := strings.TrimSpace(os.Getenv("TRIN_CUDA_ARCH")); v != "" {
		if !strings.HasPrefix(v, "sm_") {
			return "sm_" + v
		}
		return v
	}
	return DefaultPTXTarget
}

// PTXVersionHeader is the .version/.target/.address_size prefix for PTX 7.8.
func PTXVersionHeader(arch string) string {
	if arch == "" {
		arch = PTXArch()
	}
	return fmt.Sprintf(".version 7.8\n.target %s\n.address_size 64\n\n", arch)
}

// EmitLinearF32PTX generates a 1D matvec kernel:
//
//	y[row] = sum_k W[row,k] * x[k]
//
// Grid: one thread per output row. Pointers are u64 device addresses.
func EmitLinearF32PTX(arch string) PTXModule {
	if arch == "" {
		arch = PTXArch()
	}
	src := PTXVersionHeader(arch) + `.visible .entry ` + PTXLinearF32Entry + `(
    .param .u64 _W,
    .param .u64 _x,
    .param .u64 _y,
    .param .u32 _M,
    .param .u32 _K
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [_W];
    ld.param.u64 %rd2, [_x];
    ld.param.u64 %rd3, [_y];
    ld.param.u32 %r1, [_M];
    ld.param.u32 %r2, [_K];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;   // row
    setp.ge.s32 %p1, %r6, %r1;
    @%p1 bra $L_done;

    mov.f32 %f1, 0f00000000;         // acc
    mov.u32 %r7, 0;                  // k
$L_k:
    setp.ge.s32 %p1, %r7, %r2;
    @%p1 bra $L_store;

    // W[row*K + k]
    mul.lo.s32 %r8, %r6, %r2;
    add.s32 %r9, %r8, %r7;
    mul.wide.s32 %rd4, %r9, 4;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];

    // x[k]
    mul.wide.s32 %rd6, %r7, 4;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.f32 %f3, [%rd7];

    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.s32 %r7, %r7, 1;
    bra $L_k;

$L_store:
    mul.wide.s32 %rd8, %r6, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;
$L_done:
    ret;
}
`
	return PTXModule{Name: PTXLinearF32Entry, Source: src, Arch: arch}
}

// EmitGemmF32PTX generates C[i,j] = sum_k A[i,k]*B[k,j] with 2D grid (i,j).
func EmitGemmF32PTX(arch string) PTXModule {
	if arch == "" {
		arch = PTXArch()
	}
	src := PTXVersionHeader(arch) + `.visible .entry ` + PTXGemmF32Entry + `(
    .param .u64 _A,
    .param .u64 _B,
    .param .u64 _C,
    .param .u32 _M,
    .param .u32 _K,
    .param .u32 _N
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [_A];
    ld.param.u64 %rd2, [_B];
    ld.param.u64 %rd3, [_C];
    ld.param.u32 %r1, [_M];
    ld.param.u32 %r2, [_K];
    ld.param.u32 %r3, [_N];

    // i = blockIdx.y * blockDim.y + threadIdx.y
    mov.u32 %r4, %ctaid.y;
    mov.u32 %r5, %ntid.y;
    mov.u32 %r6, %tid.y;
    mad.lo.s32 %r7, %r4, %r5, %r6;
    // j = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %r8, %ctaid.x;
    mov.u32 %r9, %ntid.x;
    mov.u32 %r10, %tid.x;
    mad.lo.s32 %r11, %r8, %r9, %r10;

    setp.ge.s32 %p1, %r7, %r1;
    @%p1 bra $L_done;
    setp.ge.s32 %p1, %r11, %r3;
    @%p1 bra $L_done;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r12, 0; // k
$L_k:
    setp.ge.s32 %p1, %r12, %r2;
    @%p1 bra $L_store;

    // A[i*K+k]
    mul.lo.s32 %r13, %r7, %r2;
    add.s32 %r14, %r13, %r12;
    mul.wide.s32 %rd4, %r14, 4;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];

    // B[k*N+j]
    mul.lo.s32 %r15, %r12, %r3;
    add.s32 %r16, %r15, %r11;
    mul.wide.s32 %rd6, %r16, 4;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.f32 %f3, [%rd7];

    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.s32 %r12, %r12, 1;
    bra $L_k;

$L_store:
    mul.lo.s32 %r17, %r7, %r3;
    add.s32 %r18, %r17, %r11;
    mul.wide.s32 %rd8, %r18, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;
$L_done:
    ret;
}
`
	return PTXModule{Name: PTXGemmF32Entry, Source: src, Arch: arch}
}

// EmitLinearTritPTX generates y[row] = scale * sum_k clamp(W[row,k]) * x[k]
// with W stored as s32 trits.
func EmitLinearTritPTX(arch string) PTXModule {
	if arch == "" {
		arch = PTXArch()
	}
	src := PTXVersionHeader(arch) + `.visible .entry ` + PTXLinearTritEntry + `(
    .param .u64 _W,
    .param .u64 _x,
    .param .u64 _y,
    .param .u32 _M,
    .param .u32 _K,
    .param .f32 _scale
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<10>;
    .reg .s32 %s<4>;

    ld.param.u64 %rd1, [_W];
    ld.param.u64 %rd2, [_x];
    ld.param.u64 %rd3, [_y];
    ld.param.u32 %r1, [_M];
    ld.param.u32 %r2, [_K];
    ld.param.f32 %f4, [_scale];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    setp.ge.s32 %p1, %r6, %r1;
    @%p1 bra $L_done;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r7, 0;
$L_k:
    setp.ge.s32 %p1, %r7, %r2;
    @%p1 bra $L_store;

    mul.lo.s32 %r8, %r6, %r2;
    add.s32 %r9, %r8, %r7;
    mul.wide.s32 %rd4, %r9, 4;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.s32 %s1, [%rd5];

    // clamp trit to {-1,0,1}
    setp.gt.s32 %p2, %s1, 1;
    @%p2 mov.s32 %s1, 1;
    setp.lt.s32 %p3, %s1, -1;
    @%p3 mov.s32 %s1, -1;
    cvt.rn.f32.s32 %f2, %s1;

    mul.wide.s32 %rd6, %r7, 4;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.f32 %f3, [%rd7];

    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.s32 %r7, %r7, 1;
    bra $L_k;

$L_store:
    mul.f32 %f5, %f1, %f4;
    mul.wide.s32 %rd8, %r6, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f5;
$L_done:
    ret;
}
`
	return PTXModule{Name: PTXLinearTritEntry, Source: src, Arch: arch}
}

// AllPTXModules returns the three primary kernels for arch.
func AllPTXModules(arch string) []PTXModule {
	if arch == "" {
		arch = PTXArch()
	}
	return []PTXModule{
		EmitLinearF32PTX(arch),
		EmitGemmF32PTX(arch),
		EmitLinearTritPTX(arch),
	}
}

// FindPTXAS locates ptxas (PATH, then common venv / CUDA roots).
func FindPTXAS() (string, bool) {
	ptxasOnce.Do(func() {
		if p, err := exec.LookPath("ptxas"); err == nil {
			ptxasPath, ptxasOK = p, true
			return
		}
		cands := []string{
			"/home/nanobot/.venv/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas",
			filepath.Join(os.Getenv("HOME"), ".venv/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas"),
			"/usr/local/cuda/bin/ptxas",
			"/usr/bin/ptxas",
		}
		// triton / nvidia pip nests
		if home := os.Getenv("HOME"); home != "" {
			cands = append(cands,
				filepath.Join(home, ".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas"),
			)
		}
		for _, c := range cands {
			if st, err := os.Stat(c); err == nil && !st.IsDir() {
				ptxasPath, ptxasOK = c, true
				return
			}
		}
	})
	return ptxasPath, ptxasOK
}

// PTXASAvailable reports whether offline cubin assembly is possible.
func PTXASAvailable() bool {
	_, ok := FindPTXAS()
	return ok
}

// CompilePTX runs ptxas on src and returns cubin bytes.
// Returns a soft error if ptxas is missing (so callers can fall back).
func CompilePTX(mod PTXModule) (*CubinResult, error) {
	tool, ok := FindPTXAS()
	if !ok {
		return nil, errMsg("ptxas not found (install CUDA toolkit or nvidia-cuda-nvcc wheel)")
	}
	arch := mod.Arch
	if arch == "" {
		arch = PTXArch()
	}
	dir, err := os.MkdirTemp("", "trin-ptx-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	ptxPath := filepath.Join(dir, mod.Name+".ptx")
	cubinPath := filepath.Join(dir, mod.Name+".cubin")
	if err := os.WriteFile(ptxPath, []byte(mod.Source), 0o644); err != nil {
		return nil, err
	}
	cmd := exec.Command(tool, "-arch="+arch, "-o", cubinPath, ptxPath)
	out, err := cmd.CombinedOutput()
	res := &CubinResult{Arch: arch, Tool: tool, Stderr: string(out)}
	if err != nil {
		return res, fmt.Errorf("ptxas: %w: %s", err, strings.TrimSpace(string(out)))
	}
	b, err := os.ReadFile(cubinPath)
	if err != nil {
		return res, err
	}
	res.Bytes = b
	return res, nil
}

// LaunchStatus describes what a device launch would do on this host.
type LaunchStatus struct {
	WouldLaunch bool
	Backend     string // cuda | cpu-fallback
	Reason      string
	Entry       string
	PTXBytes    int
	CubinBytes  int
	PTXAS       bool
}

// PlanLaunch reports launch readiness for a kernel without requiring a GPU.
// On this CPU-only host: WouldLaunch=false, Backend=cpu-fallback, but PTX/cubin
// sizes are filled when assembly works.
func PlanLaunch(entry string) LaunchStatus {
	st := LaunchStatus{
		Entry:   entry,
		Backend: Backend(),
		PTXAS:   PTXASAvailable(),
	}
	// Locate module source
	var mod PTXModule
	switch entry {
	case PTXLinearF32Entry, "linear_f32", "cuda_linear_f32":
		mod = EmitLinearF32PTX("")
		st.Entry = PTXLinearF32Entry
	case PTXGemmF32Entry, "gemm_f32", "cuda_gemm_f32":
		mod = EmitGemmF32PTX("")
		st.Entry = PTXGemmF32Entry
	case PTXLinearTritEntry, "linear_trit", "cuda_linear_trit":
		mod = EmitLinearTritPTX("")
		st.Entry = PTXLinearTritEntry
	default:
		st.Reason = "unknown entry " + entry
		return st
	}
	st.PTXBytes = len(mod.Source)
	if st.PTXAS {
		if cub, err := CompilePTX(mod); err == nil {
			st.CubinBytes = len(cub.Bytes)
		} else {
			st.Reason = "ptxas compile: " + err.Error()
		}
	}
	if Available() && DeviceCount() > 0 {
		// Device visible — still no real driver launch wired in v1.03.
		st.WouldLaunch = false
		st.Backend = BackendCUDA
		if st.Reason == "" {
			st.Reason = "device visible; PTX ready; driver launch not wired (v1.03)"
		}
		return st
	}
	st.WouldLaunch = false
	st.Backend = BackendCPUFallback
	if st.Reason == "" {
		st.Reason = "no device; use cpu-fallback kernels (PTX still emitted)"
	}
	return st
}

// DeviceBuffer is a host-side stand-in for a CUDA allocation.
// On CPU-only hosts, Bytes live in process memory and Copy* are memcopies.
// On a future driver-backed path the same type can hold device pointers.
type DeviceBuffer struct {
	N     int    // element count (f32)
	Bytes []byte // host-backed when no device
	Dev   bool   // true when on device (always false in v1.03 no-GPU)
	Tag   string
}

// AllocF32 allocates an f32 buffer (host-backed on this host).
func AllocF32(n int) (*DeviceBuffer, error) {
	if n < 0 {
		return nil, errMsg("cuda_alloc_f32: negative size")
	}
	return &DeviceBuffer{N: n, Bytes: make([]byte, n*4), Dev: false, Tag: "host"}, nil
}

// Free releases a buffer (idempotent).
func (b *DeviceBuffer) Free() {
	if b == nil {
		return
	}
	b.Bytes = nil
	b.N = 0
}

// CopyHtoD copies float32 host slice into the buffer.
func (b *DeviceBuffer) CopyHtoD(src []float32) error {
	if b == nil {
		return errMsg("nil buffer")
	}
	if len(src) > b.N {
		return errMsg("CopyHtoD: overflow")
	}
	for i, v := range src {
		u := mathFloat32bits(v)
		off := i * 4
		b.Bytes[off] = byte(u)
		b.Bytes[off+1] = byte(u >> 8)
		b.Bytes[off+2] = byte(u >> 16)
		b.Bytes[off+3] = byte(u >> 24)
	}
	return nil
}

// CopyDtoH copies buffer contents into dst (truncated to min lengths).
func (b *DeviceBuffer) CopyDtoH(dst []float32) error {
	if b == nil {
		return errMsg("nil buffer")
	}
	n := len(dst)
	if n > b.N {
		n = b.N
	}
	for i := 0; i < n; i++ {
		off := i * 4
		u := uint32(b.Bytes[off]) | uint32(b.Bytes[off+1])<<8 |
			uint32(b.Bytes[off+2])<<16 | uint32(b.Bytes[off+3])<<24
		dst[i] = mathFloat32frombits(u)
	}
	return nil
}

// WritePTXFile writes module source to path (for inspection / offline tools).
func WritePTXFile(mod PTXModule, path string) error {
	return os.WriteFile(path, []byte(mod.Source), 0o644)
}

// ---- float32 bit casts without importing math (small helpers) ----

func mathFloat32bits(f float32) uint32 {
	return float32bits(f)
}
func mathFloat32frombits(u uint32) float32 {
	return float32frombits(u)
}
