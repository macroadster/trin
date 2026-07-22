// ptx_dequant.go — dequant-fused PTX kernels for GGUF block types.
//
// These generate PTX that folds the dequant step into the matvec loop so that
// quantised blocks are loaded directly from device memory and multiplied with
// the activation vector without a separate expansion pass.
//
// On hosts without a GPU the kernels are still emitted (and optionally assembled
// by ptxas); the CPU fallback path in cuda.go handles actual compute.
//
// NOTE: these kernels use cvt.f32.f16 which requires sm_53 or higher.
// Assembling with ptxas for an earlier target (e.g. sm_50) will fail.
package cuda

// PTX entry names for dequant-fused matvec kernels.
const (
	PTXDequantLinearQ4_0Entry = "trin_dequant_linear_q4_0"
	PTXDequantLinearQ8_0Entry = "trin_dequant_linear_q8_0"
)

// EmitDequantLinearQ4_0PTX generates a 1D matvec kernel that loads Q4_0 blocks
// directly: each block is 18 bytes (half d + 16 nibble bytes, QK=32).
//
// y[row] = sum over K-dim of  dequant(W_q4_0[row, :]) * x[:]
//
// Grid: one thread per output row (scalar accumulation per row).
// TODO: use warp-parallel reduction or shared-memory tiling for higher
// GPU utilisation on wide K dimensions (e.g. K=4096 → 128 blocks/row).
//
// Requires sm_53+ for cvt.f32.f16 instruction.
// Parameters: _W (u64 ptr to row-major Q4_0 blocks), _x (u64 ptr to f32 x),
//             _y (u64 ptr to f32 output), _M (u32 rows), _K (u32 columns, multiple of 32).
func EmitDequantLinearQ4_0PTX(arch string) PTXModule {
	if arch == "" {
		arch = PTXArch()
	}
	src := PTXVersionHeader(arch) + `.visible .entry ` + PTXDequantLinearQ4_0Entry + `(
    .param .u64 _W,
    .param .u64 _x,
    .param .u64 _y,
    .param .u32 _M,
    .param .u32 _K
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<12>;

    ld.param.u64 %rd1, [_W];
    ld.param.u64 %rd2, [_x];
    ld.param.u64 %rd3, [_y];
    ld.param.u32 %r1, [_M];
    ld.param.u32 %r2, [_K];

    // row = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    setp.ge.s32 %p1, %r6, %r1;
    @%p1 bra $L_done;

    // nblocks_per_row = K / 32
    shr.u32 %r10, %r2, 5;

    // row_block_offset = row * nblocks_per_row * 18
    mul.lo.s32 %r11, %r6, %r10;
    mul.lo.s32 %r12, %r11, 18;
    cvt.u64.u32 %rd4, %r12;
    add.s64 %rd5, %rd1, %rd4;    // ptr to row start in Q4_0 data

    mov.f32 %f1, 0f00000000;     // accumulator
    mov.u32 %r7, 0;              // block index within row

$L_blk:
    setp.ge.u32 %p2, %r7, %r10;
    @%p2 bra $L_store;

    // block byte offset from row start = r7 * 18
    mul.lo.s32 %r13, %r7, 18;
    cvt.u64.u32 %rd6, %r13;
    add.s64 %rd7, %rd5, %rd6;    // ptr to this block

    // Load d (f16 at offset 0) — use ld.global.u16 + cvt
    ld.global.u16 %r14, [%rd7];
    // Convert f16 bits to f32 via integer math (PTX 7.x cvt.f32.f16 needs sm_53+)
    // Fallback: store as u32, use mov to f32 reg
    // For simplicity, use f16→f32 conversion instruction:
    {
        .reg .b16 %h1;
        mov.b16 %h1, %r14;
        cvt.f32.f16 %f2, %h1;      // f2 = d (scale)
    }

    // Load 16 nibble bytes (offset 2..17), expand to f32, multiply by x and d
    // Inner loop over 16 bytes → 32 elements
    mov.u32 %r15, 0;     // j = 0..15
$L_nib:
    setp.ge.u32 %p3, %r15, 16;
    @%p3 bra $L_blk_next;

    // load nibble byte
    cvt.u64.u32 %rd8, %r15;
    add.s64 %rd9, %rd7, %rd8;
    add.s64 %rd9, %rd9, 2;        // skip 2-byte header
    ld.global.u8 %r16, [%rd9];

    // low nibble: q0 = (byte & 0x0F) - 8
    and.b32 %r17, %r16, 15;
    sub.s32 %r17, %r17, 8;
    cvt.rn.f32.s32 %f3, %r17;     // f3 = (q0 - 8) as f32
    mul.f32 %f4, %f3, %f2;        // f4 = q0 * d

    // x[blk*32 + j]
    shl.b32 %r18, %r7, 5;         // blk * 32
    add.s32 %r18, %r18, %r15;     // + j
    mul.wide.s32 %rd10, %r18, 4;
    add.s64 %rd11, %rd2, %rd10;
    ld.global.f32 %f5, [%rd11];
    fma.rn.f32 %f1, %f4, %f5, %f1;

    // high nibble: q1 = (byte >> 4) - 8
    shr.u32 %r17, %r16, 4;
    sub.s32 %r17, %r17, 8;
    cvt.rn.f32.s32 %f3, %r17;
    mul.f32 %f4, %f3, %f2;        // q1 * d

    // x[blk*32 + j + 16]
    add.s32 %r19, %r18, 16;
    mul.wide.s32 %rd12, %r19, 4;
    add.s64 %rd13, %rd2, %rd12;
    ld.global.f32 %f5, [%rd13];
    fma.rn.f32 %f1, %f4, %f5, %f1;

    add.u32 %r15, %r15, 1;
    bra $L_nib;

$L_blk_next:
    add.u32 %r7, %r7, 1;
    bra $L_blk;

$L_store:
    mul.wide.s32 %rd14, %r6, 4;
    add.s64 %rd15, %rd3, %rd14;
    st.global.f32 [%rd15], %f1;

$L_done:
    ret;
}
`
	return PTXModule{Name: PTXDequantLinearQ4_0Entry, Source: src, Arch: arch}
}

// EmitDequantLinearQ8_0PTX generates a 1D matvec kernel that loads Q8_0 blocks
// directly: each block is 34 bytes (half d + 32 int8 quantised values, QK=32).
//
// y[row] = sum over K-dim of  dequant(W_q8_0[row, :]) * x[:]
//
// Grid: one thread per output row (scalar accumulation per row).
// TODO: use warp-parallel reduction for better GPU utilisation on wide K.
//
// Requires sm_53+ for cvt.f32.f16 instruction.
func EmitDequantLinearQ8_0PTX(arch string) PTXModule {
	if arch == "" {
		arch = PTXArch()
	}
	src := PTXVersionHeader(arch) + `.visible .entry ` + PTXDequantLinearQ8_0Entry + `(
    .param .u64 _W,
    .param .u64 _x,
    .param .u64 _y,
    .param .u32 _M,
    .param .u32 _K
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<10>;

    ld.param.u64 %rd1, [_W];
    ld.param.u64 %rd2, [_x];
    ld.param.u64 %rd3, [_y];
    ld.param.u32 %r1, [_M];
    ld.param.u32 %r2, [_K];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    setp.ge.s32 %p1, %r6, %r1;
    @%p1 bra $L_done;

    // nblocks_per_row = K / 32
    shr.u32 %r10, %r2, 5;

    // row byte offset = row * nblocks_per_row * 34
    mul.lo.s32 %r11, %r6, %r10;
    mul.lo.s32 %r12, %r11, 34;
    cvt.u64.u32 %rd4, %r12;
    add.s64 %rd5, %rd1, %rd4;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r7, 0;

$L_blk:
    setp.ge.u32 %p2, %r7, %r10;
    @%p2 bra $L_store;

    mul.lo.s32 %r13, %r7, 34;
    cvt.u64.u32 %rd6, %r13;
    add.s64 %rd7, %rd5, %rd6;

    // Load d (f16 at offset 0)
    ld.global.u16 %r14, [%rd7];
    {
        .reg .b16 %h1;
        mov.b16 %h1, %r14;
        cvt.f32.f16 %f2, %h1;
    }

    // Inner loop: 32 int8 values at offset 2..33
    mov.u32 %r15, 0;
$L_qs:
    setp.ge.u32 %p3, %r15, 32;
    @%p3 bra $L_blk_next;

    cvt.u64.u32 %rd8, %r15;
    add.s64 %rd9, %rd7, %rd8;
    add.s64 %rd9, %rd9, 2;
    ld.global.s8 %r16, [%rd9];
    cvt.rn.f32.s32 %f3, %r16;
    mul.f32 %f4, %f3, %f2;

    // x[blk*32 + j]
    shl.b32 %r18, %r7, 5;
    add.s32 %r18, %r18, %r15;
    mul.wide.s32 %rd10, %r18, 4;
    add.s64 %rd11, %rd2, %rd10;
    ld.global.f32 %f5, [%rd11];
    fma.rn.f32 %f1, %f4, %f5, %f1;

    add.u32 %r15, %r15, 1;
    bra $L_qs;

$L_blk_next:
    add.u32 %r7, %r7, 1;
    bra $L_blk;

$L_store:
    mul.wide.s32 %rd14, %r6, 4;
    add.s64 %rd15, %rd3, %rd14;
    st.global.f32 [%rd15], %f1;

$L_done:
    ret;
}
`
	return PTXModule{Name: PTXDequantLinearQ8_0Entry, Source: src, Arch: arch}
}

// AllDequantPTXModules returns the dequant-fused kernels for arch.
func AllDequantPTXModules(arch string) []PTXModule {
	if arch == "" {
		arch = PTXArch()
	}
	return []PTXModule{
		EmitDequantLinearQ4_0PTX(arch),
		EmitDequantLinearQ8_0PTX(arch),
	}
}
