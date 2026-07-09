# Trin production readiness — virtual ternary LLM compiler

**Version:** 1.11.0-go  
**Status:** Production-ready for CPU virtual-ternary LLM *inference kernels*,
KV-cache helpers, deterministic sampling, CSV weight I/O, pure-Go **GGUF /
safetensors** importers with **block-quant dequant** (incl. **full IQ1 family
S/M**, **full IQ2 family XXS/XS/S**, **IQ3 XXS/S**, **IQ4_NL/XS**, **TQ1_0/TQ2_0**,
**MXFP4 + NVFP4**), **multi-shard GGUF** (`split.*` / `-NNNNN-of-MMMMM.gguf`),
**mmap opens**, **retained weight sessions**, **windowed partial decode**,
**session-backed tiled linear**, **fused quant linear**, **multi-layer residual
session runner**, **session-wired multi-head attention**, **GQA packing
helpers**, and **loop/for surface sugar** (`when` / `over` / `for-in` / `range`), and an additive **CUDA path**
(device probe + portable `cuda_linear_f32` / `cuda_gemm_f32` / `cuda_linear_trit` with
cpu-fallback when no NVIDIA driver is present),
and a **PTX layer** (emitters + optional offline `ptxas` cubin + host buffer stubs), plus a **char/token table tokenizer** and an end-to-end **GGUF session greedy decode** path (v1.04), plus **packed bitplane `linear_trit`** and a **local BPE sidecar** (v1.05), plus **session-backed packed trit linear** (`weight_session_linear_trit`, v1.06), plus a **session trit prepack cache** (`weight_session_prepack_trit` / `weight_session_linear_trit_cached`, v1.07), plus **tiny GGUF decode wired through the trit cache** (v1.08), plus a **trit residual layer stack** (`weight_session_layer_stack_trit` / `forward_block_trit`, v1.09), plus **session attn/MHA/GQA on trit-cached projections** (v1.10), plus **tiny GGUF greedy decode on session trit-attn** (v1.11).

## What is production-ready

1–18. Through v1.00 stack (sessions, fused linear, MHA, GQA, full IQ*/TQ/MXFP, multi-shard)
19. **Loop/for sugar (v1.01)** — `when g do e`, `over xs as x do e`, `for x in xs do e` desugar to ternary/map; `range(n|[lo,hi]|[lo,hi,step])` span builder
20. **CUDA path (v1.02)** — `cuda_available` / `cuda_device_count` / `cuda_backend` / `cuda_reason` / `cuda_device_name` + `cuda_linear_f32` / `cuda_gemm_f32` / `cuda_linear_trit` (cpu-fallback without driver)
21. **CUDA PTX layer (v1.03)** — PTX emitters for linear/GEMM/trit, optional `ptxas` cubin, `cuda_plan_launch`, host-backed `cuda_alloc_f32` buffers
22. **Tokenizer + GGUF decode (v1.04)** — `open_vocab` / `vocab_encode` / `vocab_decode` / `load_text` + multi-tensor `tiny_llm.gguf` session greedy decode with KV
23. **SIMD trit matvec + BPE sidecar (v1.05)** — packed set-bit `linear_trit` bitplane path + `open_bpe` / `bpe_encode` local merges
24. **Session packed trit linear (v1.06)** — `weight_session_linear_trit` / `_tile` / `load_gguf_linear_trit`: GGUF/safetensors row tiles → absmean trit → scalar or bitplane matvec; microbench scripts report both paths
25. **Session trit prepack cache (v1.07)** — `weight_session_prepack_trit` / `weight_session_linear_trit_cached`: whole-tensor absmean + packed planes retained on the session; hit path skips re-quant/re-pack
26. **Tiny GGUF decode via trit cache (v1.08)** — `examples/tiny_gguf_decode_trit.trin` prepacks Q/K/V/O/FFN/lm_head once and reuses packs across tokens
27. **Trit residual layer stack (v1.09)** — `weight_session_forward_block_trit` / `weight_session_layer_stack_trit` / `weight_session_prepack_trit_csv`: multi-layer residual or plain blocks on prepacked trit linears + RMSNorm
28. **Session attn on trit projections (v1.10)** — `weight_session_attn_proj_trit` / `weight_session_attn_trit` / `weight_session_mha_trit` / `weight_session_gqa_trit` (+ GQA proj): prepacked Q/K/V/O linears, f32 attn math, growing KV
29. **Tiny GGUF decode via trit-attn (v1.11)** — `examples/tiny_gguf_decode_attn_trit.trin` prepacks once, runs prompt+generate on `weight_session_attn_trit` + cached FFN/lm_head
30–32. Linear trit, transformer cores, KV/sample, toolchain, verification

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

Still unsupported (clear error): live driver kernel launch (cuModuleLoad/cuLaunch), CUBLAS, multi-GPU.

## CUDA path (v1.02)

| Builtin | Role |
|---------|------|
| `cuda_available()` | 1 if NVIDIA driver/device nodes visible |
| `cuda_device_count()` | probed device count (0 here) |
| `cuda_backend()` | `"cuda"` or `"cpu-fallback"` |
| `cuda_reason()` | short probe note |
| `cuda_device_name(i)` | device label or `""` |
| `cuda_linear_f32(W,x,M,K)` | row-major matvec |
| `cuda_gemm_f32(A,B,M,K,N)` | row-major GEMM |
| `cuda_linear_trit(W,x,M,K,scale)` | scaled trit matvec |

Force fallback: `TRIN_CUDA=0`. Real kernels need host driver + `/dev/nvidia*`.

### PTX layer (v1.03)

| Builtin | Role |
|---------|------|
| `cuda_ptxas_available()` | 1 if offline `ptxas` found |
| `cuda_ptx_arch()` | `sm_*` (default `sm_75`, override `TRIN_CUDA_ARCH`) |
| `cuda_ptx_source(kind)` | full PTX text (`linear_f32`/`gemm_f32`/`linear_trit`) |
| `cuda_ptx_bytes(kind)` / `cuda_cubin_bytes(kind)` | sizes |
| `cuda_compile_ptx(kind)` | offline cubin size (0 if no ptxas) |
| `cuda_plan_launch(kind)` | `[would, backend, reason, entry, ptx, cubin, ptxas]` |
| `cuda_alloc_f32` / `cuda_copy_hto_d` / `cuda_copy_dto_h` / `cuda_free` | host-backed device buffer stubs |

Entries: `trin_linear_f32`, `trin_gemm_f32`, `trin_linear_trit`. Driver kernel launch still deferred.


## Tokenizer + GGUF decode (v1.04)

| Builtin | Role |
|---------|------|
| `open_vocab(path)` / `close_vocab(id)` | retained one-token-per-line vocab session |
| `load_vocab_text(text)` | inline newline-separated vocab |
| `vocab_size(id)` | number of tokens |
| `vocab_encode(id, prompt)` | greedy longest-match → `span[i32]` |
| `vocab_decode(id, ids)` | join ids → string |
| `vocab_token(id, i)` / `vocab_id(id, tok)` | single lookups |
| `load_text(path)` | raw file → string |

Fixture `demos/weights/tiny_llm.gguf` is a 10-tensor one-block toy (vocab=8, dim=4, hidden=8)
with sidecar `tiny_llm.vocab`. Demo `examples/tiny_gguf_decode.trin` opens the session,
tokenizes prompt `"AB"`, runs session attn + fused FFN/lm_head, grows KV, greedily emits 3 tokens.

CUDA device launch remains deferred; this release is the CPU end-to-end path.


## SIMD trit matvec + BPE (v1.05)

| Builtin / API | Role |
|---|---|
| `linear_trit(W,x,M,K,scale)` | scalar 4-unroll mixed matvec |
| `linear_trit(..., 1)` 6th arg | packed pos/neg word set-bit accumulate (128-lane primary) |
| `open_bpe(vocab, merges)` / `close_bpe` | retained BPE session (local files) |
| `bpe_encode` / `bpe_decode` / `bpe_token` | rank-ordered merges encode |
| `bpe_vocab_size` / `bpe_merge_count` | session metadata |

Fixtures: `demos/weights/tiny_bpe.vocab` + `tiny_bpe.merges`. Demos: `linear_trit_simd.trin`, `bpe_sidecar.trin`.

## Session packed trit linear (v1.06)

| Builtin / API | Role |
|---|---|
| `weight_session_linear_trit(id,name,x,M,K [,tile [,bitplane]])` | session load → absmean trit → matvec; returns `[y, scale]` |
| `weight_session_linear_trit_tile(id,name,x,K,row0,nRows [,bitplane])` | row-window trit matvec |
| `load_gguf_linear_trit(path,name,x,M,K [,tile [,bitplane]])` | one-shot GGUF path |

Fixture: `demos/weights/trit_session.gguf` (via `scripts/gen_trit_session_fixture.go`).
Demos: `weight_session_linear_trit.trin`, `linear_trit_bench.trin`.
Bench: `scripts/bench_session_linear_trit.sh` (unit path + native wall time).
Note: per-tile absmean scale is independent when `tileRows < M`; compare free-span full-tensor quant only with `tileRows=M`.

## Session trit prepack cache (v1.07)

| Builtin / API | Role |
|---|---|
| `weight_session_prepack_trit(id,name,K)` | build absmean trit + pos/neg planes once; returns `1` built / `0` hit |
| `weight_session_has_trit_pack(id,name)` | `1` if pack retained |
| `weight_session_trit_pack_info(id,name)` | `[M,K,scale,hit]` |
| `weight_session_drop_trit_pack(id,name)` | drop one pack |
| `weight_session_linear_trit_cached(id,name,x,M,K [,bitplane])` | matvec from pack (auto-prepack); returns `[y,scale]` |

Demo: `examples/weight_session_trit_cache.trin` (`cache_hit_ok=1`, `parity_ok=1`).
Cache cleared on `close_weight_session`. Full-tensor scale matches `weight_session_linear_trit(..., tileRows=M)`.

## Tiny GGUF decode via trit cache (v1.08)

`examples/tiny_gguf_decode_trit.trin` opens `demos/weights/tiny_llm.gguf`, prepacks each linear tensor once (`attn_q/k/v/o`, `ffn_up/down`, `lm_head`), then runs prompt `"AB"` + 3 greedy tokens with `weight_session_linear_trit_cached` + KV append. Checks:
- first prepack build / second hit per layer
- uncached full-tile parity vs cached packed
- packs retained across multi-token reuse
- decode completes with growing KV

Note: greedy tokens may differ from fused-f32 `tiny_gguf_decode.trin` because absmean trit quantizes projections — this path proves cache reuse + self-parity, not token identity with f32 fused decode.

## Trit residual layer stack (v1.09)

| Builtin / API | Role |
|---|---|
| `weight_session_forward_block_trit(id,w,x,dim [,gain] [,mode] [,bitplane] [,eps])` | one residual/plain RMSNorm block via `LinearTritCached` |
| `weight_session_layer_stack_trit(id,wCSV,x,dim [,gainCSV] [,modeCSV] [,bitplane] [,eps])` | chain N residual/plain trit blocks |
| `weight_session_prepack_trit_csv(id,wCSV,K)` | batch prepack; returns `[built,hits]` |

Demo: `examples/weight_session_layer_stack_trit.trin` on `layer_stack.gguf` (`cache_hit_ok=1`, `parity_ok=1`, `stack_match=1`, `uncached_ok=1`).
Parity target is uncached full-tile `weight_session_linear_trit(..., tileRows=M)` residual composition (not f32 fused stack).

## Session attn on trit projections (v1.10)

| Builtin / API | Role |
|---|---|
| `weight_session_attn_proj_trit(id,q,k,v,x,dIn,dOut [,bitplane])` | Q/K/V via `LinearTritCached` |
| `weight_session_attn_trit(...)` / `weight_session_mha_trit(...)` | growing-KV attn/mha; optional O/rope; last arg bitplane |
| `weight_session_attn_proj_gqa_trit(...)` / `weight_session_gqa_trit(...)` | compact-KV GQA path on same packs |

Demo: `examples/weight_session_attn_trit.trin` on `session_attn.gguf` (`cache_hit_ok=1`, `parity_ok=1`, `proj_ok=1`, `y1_ok=1`, `grow_ok=1`).
Projections reuse session trit packs; attention scores/values stay f32.

## Tiny GGUF decode on trit-attn (v1.11)

| Piece | Role |
|---|---|
| prepack Q/K/V/O + FFN + lm_head | one-shot absmean pack build |
| `weight_session_attn_trit` | growing-KV step with RoPE + optional O on trit packs |
| `weight_session_linear_trit_cached` | FFN up/down + lm_head on same packs |

Demo: `examples/tiny_gguf_decode_attn_trit.trin` on `tiny_llm.gguf` (`cache_hit_ok=1`, `parity_ok=1`, `match_trit_decode=1`, `gen_text=EEE`).
Step-0 attn parity vs manual `linear_trit_cached` + `rope` + `attn` + O; tokens match v1.08 trit decode (not f32 fused).

## Limits

- CUDA **device execution** still needs a GPU host; this release ships the API + portable kernels.
- Multi-GPU out of scope.
- Go-backend ternary still evaluates both arms then soft-selects (side-effect
  short-circuit only in interpreter hard paths).

## Acceptance checklist

- [x] `when` / `over` / `for-in` parse → Ternary/map desugar
- [x] `range` interp + Go native (`biRange`)
- [x] Demo `examples/loop_sugar.trin` interpret + native
- [x] CUDA package probe + f32/trit kernels (`internal/cuda`)
- [x] Interpreter + Go/C backend `cuda_*` builtins
- [x] Demo `examples/cuda_path.trin` interpret + native
- [x] PTX emitters + ptxas cubin (`internal/cuda`)
- [x] Demo `examples/cuda_ptx.trin` interpret + native
- [x] Version `1.03.0-go`
- [x] Char vocab package (`internal/weights/tokenizer.go`)
- [x] Interpreter + Go backend `vocab_*` / `load_text`
- [x] Fixture `demos/weights/tiny_llm.gguf` + `.vocab`
- [x] Demo `examples/vocab_table.trin` interpret + native
- [x] Demo `examples/tiny_gguf_decode.trin` interpret + native
- [x] Version `1.04.0-go`
- [x] Packed bitplane `linear_trit` + unit tests
- [x] BPE sidecar package + `examples/bpe_sidecar.trin`
- [x] Version `1.05.0-go`
- [x] Session `weight_session_linear_trit` + tile/oneshot
- [x] Microbench scalar vs planes + session demo
- [x] Version `1.06.0-go`
- [x] Session trit prepack cache + cached matvec
- [x] Demo `weight_session_trit_cache` (hit + parity)
- [x] Version `1.07.0-go`
- [x] Tiny GGUF decode wired through trit cache
- [x] Demo `tiny_gguf_decode_trit` (hit + parity + decode)
- [x] Version `1.08.0-go`
- [x] Trit residual layer stack + prepack CSV
- [x] Demo `weight_session_layer_stack_trit` (hit + parity + uncached)
- [x] Version `1.09.0-go`
- [x] Session attn/MHA/GQA on trit-cached projections
- [x] Demo `weight_session_attn_trit` (hit + proj/y parity + grow)
- [x] Version `1.10.0-go`
- [x] Tiny GGUF decode on session trit-attn
- [x] Demo `tiny_gguf_decode_attn_trit` (hit + step0 parity + match v1.08)
- [x] Version `1.11.0-go`
- [x] No bulk renames

## How to run

```bash
export PATH="$PATH:$HOME/.local/go/bin"
cd trin
go test ./internal/cuda ./internal/parser ./internal/interp ./internal/weights -count=1
./trin run examples/cuda_path.trin
./trin run examples/cuda_ptx.trin
go run scripts/gen_tiny_llm_gguf.go .
./trin run examples/vocab_table.trin
./trin run examples/tiny_gguf_decode.trin
./trin run examples/tiny_gguf_decode_trit.trin
./trin run examples/tiny_gguf_decode_attn_trit.trin
./trin run examples/weight_session_layer_stack_trit.trin
./trin run examples/weight_session_attn_trit.trin
./trin run examples/linear_trit_simd.trin
./trin run examples/bpe_sidecar.trin
go run scripts/gen_trit_session_fixture.go .
./trin run examples/weight_session_linear_trit.trin
./trin run examples/linear_trit_bench.trin
./trin run examples/weight_session_trit_cache.trin
bash scripts/bench_session_linear_trit.sh
bash scripts/e2e_native.sh
```
