# Trin production readiness — virtual ternary LLM compiler

**Version:** 1.01.0-go  
**Status:** Production-ready for CPU virtual-ternary LLM *inference kernels*,
KV-cache helpers, deterministic sampling, CSV weight I/O, pure-Go **GGUF /
safetensors** importers with **block-quant dequant** (incl. **full IQ1 family
S/M**, **full IQ2 family XXS/XS/S**, **IQ3 XXS/S**, **IQ4_NL/XS**, **TQ1_0/TQ2_0**,
**MXFP4 + NVFP4**), **multi-shard GGUF** (`split.*` / `-NNNNN-of-MMMMM.gguf`),
**mmap opens**, **retained weight sessions**, **windowed partial decode**,
**session-backed tiled linear**, **fused quant linear**, **multi-layer residual
session runner**, **session-wired multi-head attention**, **GQA packing
helpers**, and **loop/for surface sugar** (`when` / `over` / `for-in` / `range`).

## What is production-ready

1–18. Through v1.00 stack (sessions, fused linear, MHA, GQA, full IQ*/TQ/MXFP, multi-shard)
19. **Loop/for sugar (v1.01)** — `when g do e`, `over xs as x do e`, `for x in xs do e` desugar to ternary/map; `range(n|[lo,hi]|[lo,hi,step])` span builder
20–22. Linear trit, transformer cores, KV/sample, toolchain, verification

## Loop/for sugar (v1.01)

| Surface | Desugars to |
|---------|-------------|
| `when g do e` | `g \| e \| 0` |
| `over xs as x do e` | `map((x)=>e, xs)` |
| `for x in xs do e` | `map((x)=>e, xs)` |
| `range(n)` | `[0..n)` i32 span |
| `range(lo, hi)` | `[lo..hi)` |
| `range(lo, hi, step)` | stepped half-open |

No new control combinators — only desugar onto `gate \| pure \| each` + `map`.

## Multi-shard GGUF (v1.00)

- Metadata: `split.no`, `split.count`, `split.tensors.count` (UINT16)
- Filenames: `<prefix>-%05d-of-%05d.gguf`
- `OpenGGUF` remains single-file; `OpenGGUFSharded` / `open_weight_session` auto-walk siblings

## IQ/TQ/MXFP support (v1.01)

| Type | id | block | layout |
|------|----|-------|--------|
| IQ1_S   | 19 | 256 / 50 B | fp16 d + qs + qh(u16 scale/delta) |
| IQ1_M   | 29 | 256 / 56 B | qs + qh + scales (d packed in hi bits) |
| IQ2_XXS | 16 | 256 / 66 B | fp16 d + qs uint16 grid/signs |
| IQ2_XS  | 17 | 256 / 74 B | fp16 d + qs uint16 + scales[8] |
| IQ2_S   | 22 | 256 / 82 B | fp16 d + qs + qh + scales |
| IQ3_XXS | 18 | 256 / 98 B | fp16 d + qs grid + scales/signs |
| IQ3_S   | 21 | 256 / 110 B | fp16 d + qs + qh + signs + scales |
| IQ4_NL | 20 | 32 / 18 B | fp16 d + 16 nibble qs |
| IQ4_XS | 23 | 256 / 136 B | fp16 d + scales_h/l + qs[128] |
| TQ1_0  | 34 | 256 / 54 B | qs[48]+qh[4]+d; 5 trits/byte |
| TQ2_0  | 35 | 256 / 66 B | qs[64]+d; 2-bit ternary |
| MXFP4  | 39 | 32 / 17 B | E8M0 e + 16 nibble E2M1 qs |
| NVFP4  | 40 | 64 / 36 B | 4×UE4M3 scales + 32 nibble E2M1 qs |

Still unsupported (clear error for IQ/TQ types not listed above).

## CUDA backend (v1.02)

GPU-accelerated code generation via `emit-cuda` / `--backend cuda`. Emits
standalone `.cu` files with `__global__` kernels for:

- **Elementwise**: `silu`, `gelu`, `span_add`, `span_scale`, `span_mul`
- **Reductions**: `sum`, `dot` (shared-memory tree)
- **Normalization**: `rmsnorm` (two-pass), `softmax` (three-pass)
- **Rotation**: `rope` (pairwise parallel)
- **Linear**: `linear_trit` (one thread per output row)
- **Attention**: `attn_scores` (one thread per query-key pair)

CPU-only: I/O, weight loading, sampling, KV cache, trit packing, argmax.

Requires `nvcc` on PATH (or `TRIN_NVCC`). No external library needed.

## Limits
- CUDA backend is correctness-first; competitive tok/s and multi-GPU are follow-ons.
- Go-backend ternary still evaluates both arms then soft-selects (side-effect
  short-circuit only in interpreter hard paths).

## Acceptance checklist

- [x] `when` / `over` / `for-in` parse → Ternary/map desugar
- [x] `range` interp + Go native (`biRange`)
- [x] Demo `examples/loop_sugar.trin` interpret + native
- [x] Version `1.01.0-go`
- [x] No bulk renames

## How to run

```bash
export PATH="$PATH:$HOME/.local/go/bin"
cd trin
go test ./internal/parser ./internal/interp ./internal/weights -count=1
./trin run examples/loop_sugar.trin
bash scripts/e2e_native.sh
```
