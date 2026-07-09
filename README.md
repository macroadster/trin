# Trin — virtual ternary language (native compiler)

One control combinator only:

```
gate | pure | each
```

Binary silicon runs this as select/FMA + bitplane boolean ops.

## Native toolchain (no Python, no host cc)

Single Go binary `trin` compiles examples to native Linux executables.
Default backend is **Go** (AST → Go → `go build`); optional **C** backend remains.

```bash
export PATH="$PATH:$HOME/.local/go/bin"
cd trin
go build -o trin ./cmd/trin

# interpret
printf 'Ada\n' | ./trin run examples/hello.trin
./trin run examples/trit_gemm.trin

# compile to native (host-cc free)
./trin build examples/hello.trin -o build/hello
./trin build examples/bitplane_gemm.trin -o build/bitplane_gemm --backend go
printf 'Ada\n' | ./build/hello
./build/bitplane_gemm

# full e2e (no python3)
bash scripts/e2e_native.sh
```

CLI:

```
trin run <file.trin>                         # interpreter
trin build <file.trin> [-o x] [--backend go|c]
trin emit-go <file.trin>                     # Go only
trin emit-c <file.trin>                      # C only
trin emit-ir <file.trin>                     # TrinIR text
trin emit-llvm <file.trin>                   # LLVM IR text
trin emit-obj <file.trin> [-o x.o]           # LLVM IR → object via llc (when present)
trin link-llvm <file.trin> [-o x]            # llc .o + cc host driver → binary
trin build <file.trin> --backend llvm -o x   # same full link path
trin version
```

Env: `TRIN_BACKEND` (go|c|llvm), `TRIN_CC`, `TRIN_KEEP_C=1`, `TRIN_KEEP_GO=1`,
`TRIN_KEEP_LLVM=1`, `TRIN_GO`, `TRIN_LLC`, `TRIN_LLVM_ENTRY`.

Soft-mux / span AD surface (v0.6–0.18):

```
soft_mux(w, pure, each)           # w*pure + (1-w)*each
soft_mux_grad(w, pure, each, dy)  # [dw, dpure, deach]
soft_mux_grad_w / _pure / _each
grad_expr(expr, names, values)    # reverse tape AD → [y, ∂n0, …]
                                  # + sum/dot/span_add/scale/map_mul
grad_sum(xs)                      # [sum, ∂/∂x0, …]
grad_dot(a, b)                    # [dot, ∂a..., ∂b...]
grad_span_add(a, b)               # [sum(a+b), ∂a..., ∂b...]
grad_scale(s, xs) / grad_map_mul  # [sum(s*x), ∂s, ∂x...]  (const-mul map subset)
grad_expr(... sum(map((x)=>body, span)) ...)  # true unary map lambda AD (v0.10)
grad_expr(... nested map(... map(...)) ...)  # nested map reverse AD (v0.11)
map((x)=>x*a+1, xs)                          # general unary map Go/C codegen (v0.11)
zip_with((x,y)=>x*s+y, a, b)                 # binary zip_with eval + AD (v0.12)
grad_expr(... sum(zip_with((x,y)=>...,a,b))) # zip_with reverse AD (v0.12)
reduce((acc,x)=>acc+x*s, 0, xs)              # left-fold reduce/fold eval + AD (v0.13)
fold((acc,x)=>acc*x+b, a, xs)                # fold alias (v0.13)
zip_with((x,y,z)=>x*s+y+z, a, b, c)          # ternary zip_with eval + AD (v0.14)
scan((acc,x)=>acc+x*s, 0, xs)                # left-scan prefixes span (v0.14)
grad_expr(... sum(scan(...)) ...)            # scan reverse AD (v0.14)
zip_with((x,y,z,w)=>..., a,b,c,d)            # n-ary zip_with N>=2 (v0.15)
filter((x)=>pred, xs)                        # hard keep where pred!=0 (v0.15)
grad_expr(... sum(filter(...)) ...)          # filter reverse AD, keep-set frozen (v0.15)
                                             # bodies: scalar + - * / soft_mux vars/lits
```

C backend notes: host `cc` optional; general unary `map`, n-ary `zip_with`,
`reduce`/`fold`/`scan`/`filter` match Go (element / multi-span / fold loops +
nested env). Use `--backend go` (default) when `cc1` is missing.

## Layout

```
trin/
  cmd/trin/           # CLI entry
  internal/
    lexer parser ast  # frontend
    interp            # Go interpreter
    codegen           # AST → standalone Go or C
    ad                # reverse-mode f32 + soft_mux tape
    ir                # TrinIR SSA dump
    codegen           # Go/C/LLVM text backends
  examples/*.trin
  trin_lang/          # optional Python reference (v0.1–0.3)
  backend/            # C kernel headers + demos
  scripts/e2e_native.sh
```

## Bitplane trit (dense packing v0.16 + SIMD multiword v0.18)

```
t=+1 → pos   t=0 → zero   t=-1 → neg
product via AND/OR; accumulate popcount(pos)-popcount(neg)
dense pack: 32 trits → i32 pos/neg words (bit 0 = index 0)

trit_pack_words(xs) → [pos_w, neg_w, n]
trit_unpack_words(pos_w, neg_w, n)
trit_plane_and / or / xor / popcount
trit_mul_planes(ap,an,bp,bn) → [p_pos, p_neg]
trit_dot_planes(ap,an,bp,bn,n)        # masked popcount product
trit_dot_planes_simd(...)             # 128-lane multiword chunks (v0.18)
trit_gemm_bitplane / trit_gemm_bitplane_simd
simd_lane_width() → 128
```

## Versions

| Ver | Delivered |
|-----|-----------|
| 0.1 | Python spike + hello |
| 0.2 | typechecker, span, trit_gemm |
| 0.3 | bitplane path |
| 0.4 | Go compiler + C codegen |
| 0.5 | Go backend — build without host cc |
| 0.6 | soft-mux reverse AD + TrinIR (`emit-ir`) |
| 0.7 | full f32 tape AD (`grad_expr`) + `emit-llvm` text |
| 0.8 | span-kernel AD (`sum`/`dot`/`grad_sum`/`grad_dot`) + LLVM sum4/dot4 |
| 0.9 | `span_add` / const-mul `map` AD (`grad_span_add`/`grad_scale`/`map_mul`) + LLVM helpers |
| 0.10 | true `map((x)=>body)` reverse AD + optional `emit-obj` via `llc` |
| 0.11 | nested map reverse AD + general unary map Go/C codegen |
| 0.12 | binary `zip_with` eval + reverse AD + Go/C codegen |
| 0.13 | `reduce`/`fold` left-fold eval + reverse AD + Go/C codegen |
| 0.14 | ternary `zip_with` + `scan` eval + reverse AD + Go/C codegen |
| 0.15 | n-ary `zip_with` + `filter` eval + reverse AD + Go/C codegen |
| 0.16 | dense 32-trit word packing + plane AND/OR/XOR/popcount product + real bitplane GEMM |
| 0.17 | hosted `llc`+`cc` link into full binaries (`link-llvm` / `--backend llvm`) + bitplane LLVM helpers |
| 0.18 | multiword SIMD-friendly bitplane kernels (128-lane primary): `trit_dot_planes_simd`, `trit_gemm_bitplane_simd`, LLVM `x4` helper |
| 0.19 | full-program LLVM lowering: `@main` with println/to_str/spans/i32 + PIC `link-llvm` (float-kernel host preserved for `loss`/`kernel`) |
| 0.20 | auto-vectorizable LLVM span loops (`trin_sum_f32`/`trin_dot_f32`/`trin_span_add_f32`/`trin_span_scale_f32` + vectorize metadata) + `trin_vec_loop_probe` |
| 0.21 | `emit-mlir`: portable MLIR dialect text + `vector<4xf32>` span/soft_mux tiles (no system mlir-opt required) |
| 0.22 | MLIR→LLVM lowering (`emit-mlir-llvm` / `lower-mlir`): soft_mux + span loops + `<4 x float>` vector tiles/probes |
| 0.23 | full-program MLIR `@main` + executable MLIR→LLVM (println/soft_mux/spans; no loss/kernel host path) |
| 0.24 | wider auto-vector tiles: `vector<8xf32>` / `<8 x float>` soft_mux/span/sum/dot + `trin_mlir_vec8_probe` → 36 |
| 0.25 | richer MLIR dialect ops + portable pattern rewrites (`rewrite-mlir`): `"trin.*"` → helpers, soft_mux(0/1) folds; probes 4/7/10 |
| 0.26 | 16-lane tiles `vector<16xf32>` / `<16 x float>` + portable VLEN (`trin_vlen` / `trin_vlen_for_n`); probes 136 / 16 |
| 0.27 | deeper rewrites: pure-SSA CSE, multi-use soft_mux commoning, vector-width pick; probes 6 / 12 / 136 |
| 0.28 | hardware-aware VLEN (`trin_hw_vlen` / `_for_n` / `trin_hw_isa_code`) over portable default; AVX-512 / AVX / SSE / SVE / NEON |
| 0.29 | soft_mux strength / algebraic folds: `pure==each → pure`, full-const fold, `each=0 → w*pure`, `pure=0 → (1-w)*each`; probe 13.5 |
| 0.30 | runtime dynamic feature dispatch: `@trin_hw_*` use CPUID/`@trin_hw_detect_once` in lowered binaries (emit-host is note only) |
| 0.31 | span/vector algebraic rewrites: `span_add(x,0)`, `span_scale(1|0,x)`, `sum(0)`, `dot(0,x)`; probe 30 |
| 0.32 | full-program `@main` dynamic tile selection: fixed 4/8/16 `sum`/`dot` → `@trin_dyn_*` via `@trin_hw_vlen_for_n` |
| 0.33 | more soft_mux/bitplane algebraic folds: `soft_mux_vN` zero arms + plane and/or/xor/popcount identities; probe 42 |
| 0.34 | residual peel for non-fixed-length spans: n∉{4,8,16} → max 16/8/4 dyn tiles + scalar residual in full-program kernels |
| 0.35 | nested soft_mux / bitplane product folds: weight product `w1*w2` / `w1+(1-w1)*w2`; plane absorb `and(a,or)→a`; probe 52 |
| 0.36 | auto residual peel for pointer/counted kernels: `@trin_sum_f32`/`@trin_dot_f32` peel 16/8/4 + residual; probe 55 |
| 0.37 | store peels for `@trin_span_add_f32`/`@trin_span_scale_f32`: vector 16/8/4 + residual; probe 220 |
| 0.38 | multi-level soft_mux trees + multi-word plane GEMM folds; probe 64 |
| 0.39 | adaptive tile from runtime ISA (`@trin_adaptive_tile*`); probe 100 |
| 0.40 | map/store peels (`span_mul`/`span_axpy`) + soft_mux forest / GEMM schedule; probes 550 / 80 |
| 0.41 | scan/reduce peels + deep soft_mux forest + adaptive ISA peel ladder; probes 275 / 90 |
| 0.42 | scan-mul/reduce-mul peels + AdaptivePeelChunks emit + wide 32 ladder; probes 64 / host |
| 0.43 | fused scan-reduce + dyn_sum32 + VL>32 (64) ladder; probes 275 / 32 / host |
| 0.44 | multi-output fused scan-reduce (sum+last) + RDVL VL path + fused-mul probe; probes 330 / 64 / host |
| 0.45 | first-class pair/tuple IR + live aarch64 RDVL + one-pass fused sum_scan; probes 110 / 275 / host |
| 0.46 | language pair surface + streaming-SVE/SME VL + fused map+reduce mul/add; probes 110 / 495 / host |
| 0.47 | pair destructure/return + SME ZA tile schedule + fused map+reduce mul/sum; probes 110 / 440 / host |
| 0.48 | nested/multi-way pair match + SME ZA outer-product + fusion cost model; probes 110 / 495 / 10 / host |
| 0.49 | pair-of-pairs exhaust + ZA outer-acc + weighted multi-candidate fusion; probes 110 / 3025 / 20 / 495 / host |
| 0.50 | pair wildcard/guard + true ZA outer-acc + calibrated fusion plan log; probes 110 / 3025 / 20 / 495 / host |
| 0.51 | pair or-patterns + ZA stream store + fusion PGO feedback; probes 110 / 3025 / 20 / 495 / host |
| 0.52 | pair tag/exhaust + ZA gather/scatter + PGO adapt; probes 110 / 3025 / 20 / 110 / 495 / host |
| 0.53 | nested tag + multivec gather/scatter + PGO ISA loop; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.54 | multi-way nested tag + tiled multivec stream + PGO ISA live; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.55 | recursive multi-way ADT + multi-tile stream + PGO xkernel; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.56 | ADT ser/tree-fold + multi-tile residual peels + PGO xpass; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.57 | ADT forest zip/map folds + multi-tile auto peels + PGO window; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.58 | ADT keyed forest reduce + multi-tile VL-fb peels + PGO decay; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.59 | ADT multi-key filter/reduce + live VL residual peels + PGO EMA; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.60 | ADT ranked multi-key filter/reduce + live VL residual streaming peels + PGO adapt half-life; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.61 | ADT weighted multi-key scores + VL residual streaming cost peels + dual EMA/hysteresis PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.62 | ADT score-threshold multi-key + VL residual streaming cost adaptive tile peels + windowed reweight PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.63 | ADT soft-gated score-threshold multi-key + VL residual streaming adaptive cost residual schedule + budget-cap PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.64 | ADT calibrated soft-gated multi-key + residual streaming schedule peel fusion + softcap PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.65 | ADT adaptive multi-key + residual streaming schedule peel fusion live tile mix + dual soft/hard budget PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.66 | ADT blended multi-key + residual streaming schedule peel fusion live tile mix cost balance + conf blend PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.67 | ADT gated multi-key + residual streaming schedule peel fusion live tile mix cost balance bandwidth + conf blend + hysteresis + regret floor PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.68 | ADT score-calibrated multi-key + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost + conf blend + hysteresis + regret floor PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.69 | ADT soft-confidence-floor multi-key + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure + conf blend + hysteresis + regret floor + adaptive floor PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.70 | ADT hard-floor-gate multi-key + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model + conf blend + hysteresis + regret floor + adaptive floor + burst cap PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.71 | ADT dual-gate-blend multi-key + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.72 | ADT soft/hard dual score mix multi-key + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.73 | ADT soft/hard dual score mix multi-key ranked credit gates + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.74 | ADT soft/hard dual score mix multi-key ranked credit + margin gates + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.75 | ADT soft/hard dual score mix multi-key ranked credit + margin + band gates + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain + refill + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.76 | ADT soft/hard dual score mix multi-key ranked credit + margin + band + clamp gates + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain + refill + flush + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.77 | ADT soft/hard dual score mix multi-key ranked credit + margin + band + clamp + peg gates + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain + refill + flush + seal + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.78 | ADT soft/hard dual score mix multi-key ranked credit + margin + band + clamp + peg + latch gates + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain + refill + flush + seal + purge + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + clear PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.79 | ADT soft/hard dual score mix multi-key ranked credit + margin + band + clamp + peg + latch + fence gates + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain + refill + flush + seal + purge + sync + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + clear + final PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.80 | ADT soft/hard dual score mix multi-key ranked credit + margin + band + clamp + peg + latch + fence + lock gates + residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain + refill + flush + seal + purge + sync + close + conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + clear + final + close PGO; probes 110 / 3025 / 20 / 20 / 495 / host |
| 0.81 | **Production ternary LLM surface**: absmean quantize, trit Linear/GEMM, RMSNorm, Softmax, SiLU/GELU, RoPE, attention+KV helpers, sampling, embed_row, CSV weight I/O; tiny ternary LLM demo compile+run |
| 0.82 | **Production hardening**: first-class KV-cache helpers (`kv_new`/`kv_append`/`kv_len`/`kv_row`/`span_concat`), deterministic LCG sampling (`rng_seed`/`rng_uniform`, seeded `sample_logits`, `sample_topk`), quantize_file sidecars on Go backend, KV tiny-LLM demo |
| 0.83 | **GGUF + safetensors importers** (pure Go): list/load tensors, shapes, GGUF string meta; F32/F16/BF16/I*/Q8_0 decode → f32 span → quantize/linear path |
| 0.84 | **GGUF block-quant dequant**: Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 + K-quants Q2_K–Q6_K/Q8_K → f32; fixtures + load_gguf_q4 demo |


## Production ternary LLM runtime (v0.81–0.84)

Trin ships a **production-ready virtual-ternary inference surface** for mapping LLM
parameters onto trit bitplanes and running transformer cores on binary silicon via the
native compiler (default Go backend).

### Weight path

```
quantize_absmean(w)            → [trits, scale]   // BitNet-style absmean
trit_pack_quant(w)             → [pos,neg,n,scale]
linear_trit(W,x,M,K,scale)     → f32 Mx1          // W trit row-major
linear_trit_gemm(A,B,M,K,N,s)  → f32              // scale * bitplane GEMM
load_f32_csv / load_trit_csv / save_f32_csv
quantize_file_absmean(path)    → writes .trit.csv + .scale.txt sidecars
// v0.83 GGUF / safetensors
load_safetensors_list(path) / load_safetensors_tensor(path,name) / load_safetensors_shape
load_gguf_list(path) / load_gguf_tensor / load_gguf_shape / load_gguf_meta(path,key)
load_weight_tensor(path,name)  // auto-detect .gguf / .safetensors (or magic)
// v0.85 mmap open — list/shape/meta do not decode tensor bodies; Load page-faults only needed bytes
weight_file_mapped(path)       // 1 if OS mmap succeeded (0 = heap fallback)
// v0.86 retained session — one open, multi-tensor load, explicit close
open_weight_session(path) → id
weight_session_list / _shape / _tensor / _meta / _mapped(id)
close_weight_session(id)       // idempotent; munmaps
// v0.87 windowed / partial decode (no full-tensor alloc for dense/block types)
weight_session_window(id, name, offset, count)
load_gguf_window / load_safetensors_window(path, name, offset, count)
// v0.88 session-backed tiled matvec (row tiles via window decode)
weight_session_linear(id, name, x, M, K [, tileRows])
weight_session_linear_tile(id, name, x, K, row0, nRows)
load_gguf_linear(path, name, x, M, K [, tileRows])
// v0.89 fused quant linear (block-stream, ≤1-block scratch)
weight_session_linear_fused(id, name, x, M, K [, tileRows])
weight_session_linear_fused_tile(id, name, x, K, row0, nRows)
load_gguf_linear_fused(path, name, x, M, K [, tileRows])
// v0.90 multi-layer residual fused linear + RMSNorm
weight_session_forward_block(id, wName, x, dim [, gainName] [, mode] [, tileRows] [, eps])
weight_session_layer_stack(id, wCSV, x, dim [, gainCSV] [, modeCSV] [, tileRows] [, eps])
// v1.09 multi-layer residual trit-cached linear + RMSNorm
weight_session_forward_block_trit(id, wName, x, dim [, gainName] [, mode] [, bitplane] [, eps])
weight_session_layer_stack_trit(id, wCSV, x, dim [, gainCSV] [, modeCSV] [, bitplane] [, eps])
weight_session_prepack_trit_csv(id, wCSV, K) → [built, hits]
// v0.91 session-wired QKV + attn/mha (growing KV)
weight_session_attn_proj(id, q,k,v, x, dIn, dOut [, tileRows]) → [q,k,v]
weight_session_attn(id, q,k,v, x, Kpast, Vpast, dIn, dOut [, o] [, h] [, pos] [, base] [, tile]) → [y,K,V]
weight_session_mha(id, q,k,v, x, Kpast, Vpast, dIn, dOut, h [, o] [, pos] [, base] [, tile]) → [y,K,V]
// v1.10 session-wired QKV + attn/mha on trit prepack cache
weight_session_attn_proj_trit(id, q,k,v, x, dIn, dOut [, bitplane]) → [q,k,v]
weight_session_attn_trit(id, q,k,v, x, Kpast, Vpast, dIn, dOut [, o] [, h] [, pos] [, base] [, bitplane]) → [y,K,V]
weight_session_forward_attn_block_trit(id, q,k,v,o, up,down, x, Kpast,Vpast, dim, hidden [, attnGain] [, ffnGain] [, h] [, pos] [, base] [, bitplane] [, eps]) → [y,K,V]
weight_session_attn_stack_trit(id, qCSV,kCSV,vCSV,oCSV, upCSV,downCSV, x, KpastFlat,VpastFlat, dim, hidden [, attnGainCSV] [, ffnGainCSV] [, h] [, pos] [, base] [, bitplane] [, eps]) → [y,Kflat,Vflat]
weight_session_mha_trit(id, q,k,v, x, Kpast, Vpast, dIn, dOut, h [, o] [, pos] [, base] [, bitplane]) → [y,K,V]
// v0.92 GQA packing + session-wired grouped-query attention
gqa_expand_kv(kv, nTok, nKVHeads, nHeads, dh)
gqa_repeat_kv_row(row, nKVHeads, nHeads, dh)
weight_session_attn_proj_gqa(id, q,k,v, x, dIn, nHeads, nKVHeads, dh [, tile]) → [q,k,v]
weight_session_gqa(id, q,k,v, x, Kpast, Vpast, dIn, nHeads, nKVHeads, dh
                   [, o] [, pos] [, base] [, tile]) → [y,Knew,Vnew]  // compact KV
// v1.10 GQA on trit prepack cache
weight_session_attn_proj_gqa_trit(id, q,k,v, x, dIn, nHeads, nKVHeads, dh [, bitplane]) → [q,k,v]
weight_session_gqa_trit(id, q,k,v, x, Kpast, Vpast, dIn, nHeads, nKVHeads, dh
                        [, o] [, pos] [, base] [, bitplane]) → [y,Knew,Vnew]
```

### Transformer ops

```
rmsnorm(x [,eps] [,gain])
softmax(x)  silu(x)  gelu(x)
rope(x, pos [,base])
residual_add(a,b)
attn_scores(q,K,d)  attn_values(p,V,d)  attn(q,K,V,d)
mha(q,K,V,d,h)                 // multi-head convenience
embed_row(table, idx, dim)
argmax(x)
sample_logits(logits [,temp] [,seed])  // temp<=0 → greedy; seed reseeds LCG
sample_topk(logits, k [,temp] [,seed])
rng_seed(s)  rng_uniform()
kv_new()  kv_append(cache,row)  kv_len(cache,d)  kv_row(cache,idx,d)
span_concat(a,b)
```

### Demos

```bash
./trin run examples/llm_kernels.trin
./trin run examples/tiny_ternary_llm.trin
./trin run examples/tiny_ternary_llm_kv.trin
./trin run examples/kv_sample.trin
./trin run examples/load_quant_infer.trin
./trin run examples/load_weights_gguf_st.trin
./trin run examples/load_gguf_mmap.trin
./trin run examples/weight_session.trin
./trin run examples/weight_window.trin
./trin run examples/weight_session_linear.trin
./trin run examples/weight_session_linear_fused.trin
./trin run examples/weight_session_layer_stack.trin
./trin run examples/weight_session_attn.trin
./trin run examples/weight_session_gqa.trin
./trin run examples/weight_iq_tq.trin
./trin run examples/weight_iq_tq2.trin
./trin run examples/weight_iq_grid.trin
./trin run examples/weight_iq2_family.trin
./trin run examples/weight_iq3_s.trin
./trin run examples/weight_iq1_family.trin
./trin build examples/tiny_ternary_llm.trin -o build/tiny_ternary_llm
./build/tiny_ternary_llm
```

`examples/tiny_ternary_llm.trin` is an end-to-end **tokenizer table → embed →
RMSNorm → RoPE → ternary Linear attention/FFN → lm_head → greedy decode** program
that both interprets and compiles. `tiny_ternary_llm_kv.trin` is the same path with
growing KV via `kv_append`.
`examples/vocab_table.trin` exercises `open_vocab` / `vocab_encode` / `vocab_decode` / `load_text`.
`examples/tiny_gguf_decode.trin` loads multi-tensor `demos/weights/tiny_llm.gguf` via a weight
session, tokenizes prompt `"AB"`, runs session attn + fused FFN/lm_head with growing KV, and
greedily decodes three tokens on CPU (no CUDA).
`examples/linear_trit_simd.trin` checks scalar vs bitplane `linear_trit` parity on K=140.
`examples/bpe_sidecar.trin` loads local `tiny_bpe.vocab` + `tiny_bpe.merges` and encodes `"ABC"`.
`examples/weight_session_linear_trit.trin` loads `trit_session.gguf` and runs session-backed scalar vs bitplane trit matvec.
`examples/linear_trit_bench.trin` hammers free-span scalar vs planes for wall-time measurement.
`examples/weight_session_trit_cache.trin` prepacks once, hits cache, and checks cached scalar/planes parity.
`examples/tiny_gguf_decode_trit.trin` prepacks tiny_llm linears once and runs multi-token decode on the hit path.
`examples/weight_session_layer_stack_trit.trin` prepacks W0,W1 once and chains residual/plain RMSNorm blocks on the trit cache.
`examples/weight_session_attn_trit.trin` prepacks Q/K/V/O once and runs growing-KV attn/mha on trit-cached projections.
`examples/tiny_gguf_decode_attn_trit.trin` runs multi-token greedy decode with `weight_session_attn_trit` (+ FFN/lm_head on the same packs).
`examples/weight_session_attn_stack_trit.trin` chains two residual trit-attn+FFN layers with growing layer-major KV flats.
`examples/tiny_gguf_decode_stack_trit.trin` runs multi-token greedy decode on a 2-layer GGUF via `weight_session_attn_stack_trit`.
`examples/weight_session_gqa_stack_trit.trin` / `examples/tiny_gguf_decode_gqa_trit.trin` add multi-layer residual GQA + final RMSNorm.

`examples/load_weights_gguf_st.trin` loads the toy fixtures under `demos/weights/`
(`.safetensors` + `.gguf`), quantizes, and runs `linear_trit`.
`examples/load_gguf_mmap.trin` probes `weight_file_mapped` and loads one Q4_0 tensor.
`examples/weight_session.trin` keeps one mmap open and loads weight+bias.
`examples/weight_window.trin` decodes contiguous element windows (F32 + Q4).
`examples/weight_session_linear.trin` matsvec from row tiles on retained sessions.
`examples/weight_session_linear_fused.trin` block-streams Q dequant into dots (no full-tile f32).
`examples/weight_session_layer_stack.trin` chains residual fused-linear + RMSNorm layers on one session.
`examples/weight_session_attn.trin` projects Q/K/V fused, runs attn/mha, grows KV.
`examples/weight_session_gqa.trin` expands compact KV heads and runs session GQA.
`examples/weight_iq_tq.trin` loads IQ4_NL + TQ2_0 tensors through sessions/linear.
`examples/weight_iq_tq2.trin` loads IQ4_XS + TQ1_0 tensors through sessions/linear.
`examples/weight_iq_grid.trin` loads IQ2_XXS + IQ3_XXS grid tensors through sessions/linear.
`examples/weight_iq2_family.trin` loads IQ2_XS + IQ2_S tensors through sessions/linear.
`examples/weight_iq3_s.trin` loads IQ3_S tensors through sessions/linear.
`examples/weight_iq1_family.trin` loads IQ1_S + IQ1_M tensors through sessions/linear.

### Guarantees and limits (production)

| Guaranteed | Limit |
|---|---|
| Deterministic interpret == Go-backend native for LLM builtins | C backend incomplete for new I/O on some hosts |
| Loop/for sugar: `when`/`over`/`for-in` + `range` desugar to ternary/map | Go ternary evaluates both arms before soft-select |
| Absmean quantize + scaled trit Linear match bitplane GEMM math | Competitive CUDA tok/s still out of scope |
| GGUF v2/v3 LE + common block quants + **IQ1 (S/M) + IQ2 family + IQ3 (XXS/S) + IQ4_NL/XS** + **TQ1_0/TQ2_0** + **MXFP4/NVFP4** + **multi-shard** → f32 | Network fetch out of scope |
| mmap (syscall) open for GGUF + safetensors; multi-shard auto-walk via sessions | Dequant still allocates the f32 output span |
| Retained weight sessions: multi-tensor load without re-open | Sessions are process-local int32 handles |
| Windowed loads: decode [offset,count) only (block-local for Q*) | Element window is flat row-major |
| Session tiled linear: matvec without full-tensor decode | Default tileRows=8 |
| Fused quant linear: stream Q blocks with ≤1-block scratch | Parity with window decode path |
| Multi-layer residual stack on one session (RMSNorm + fused linear) | CSV layer/gain name lists |
| Session-wired QKV fused proj + attn/mha with growing KV (+ optional RoPE/O) | Returns `[y, Knew, Vnew]` |
| GQA expand/repeat + session GQA (compact KV cache, expand into mha) | `nHeads` multiple of `nKVHeads` |
| Attention + multi-head (`mha`) + RoPE + RMSNorm + SiLU block composable | `gqa_expand_kv` / `weight_session_gqa` for GQA |
| Greedy / argmax / seeded categorical & top-k sampling (LCG) | Hardware CSPRNG not used — intentional for reproducible demos |
| KV cache as row-major span + append/len/row helpers | Not a distinct GC'd tensor type; user owns layout |
| Additive CUDA path: probe + f32/trit kernels with cpu-fallback | Real PTX/CUBLAS multi-GPU still out of scope |
| Char/token table tokenizer + load_text; multi-tensor GGUF session greedy decode demo | Network model fetch out of scope |
| Packed bitplane `linear_trit` set-bit accumulate (v1.05) + scalar 4-unroll | Competitive tok/s still out of scope |
| BPE sidecar `open_bpe` / merges file (local ranks, v1.05) | Full HF sentencepiece proto & network fetch out of scope |
| Session `weight_session_linear_trit` GGUF→absmean→scalar/bitplane (v1.06) | Per-tile absmean ≠ full-tensor quant when tileRows < M |
| Session trit prepack cache + `linear_trit_cached` (v1.07) | Cache is process-local; dropped on close |
| Tiny GGUF decode via trit cache (v1.08) | Trit quant tokens may differ from fused-f32 decode |
| Trit residual stack on prepack cache (v1.09) | Square dim×dim layers; not full attn blocks |
| Session attn on trit projections (v1.10) | Attention math f32; only projs use trit packs |
| Tiny GGUF decode via trit-attn (v1.11) | Growing-KV session attn_trit + FFN/lm_head packs |
| Multi-layer trit-attn stack (v1.12) | Residual attn+FFN layers, layer-major KV flats |
| Multi-layer GGUF decode via stack (v1.13) | 2-layer greedy decode, growing layer-major KV |
| Multi-layer GQA stack + final-norm (v1.14) | Compact KV flats, residual GQA, final RMSNorm |

### Acceptance (v1.14)

- `go test` green (weights GQA stack + interp gqa demos)
- `bash scripts/e2e_native.sh` green including `weight_session_gqa_stack_trit` / `tiny_gguf_decode_gqa_trit`
- `./trin run examples/tiny_gguf_decode_gqa_trit.trin` shows `tiny_gguf_decode_gqa_trit_ok` / `cache_hit_ok=1` / `parity_ok=1` / `grow_final_ok=1` / `gen_text=FFF`
- Prior v1.13 demos still pass (`tiny_gguf_decode_stack_trit_ok`)

## Next

- v1.14: multi-layer residual GQA stack + final RMSNorm decode
- v1.13: multi-layer GGUF greedy decode on trit-attn stack
- v1.12: multi-layer residual trit-attn decoder stack
- v1.11: multi-token GGUF greedy decode on session trit-attn
- v1.10: session-wired attn/MHA/GQA on trit-cached projections
- v1.09: multi-layer residual stack on trit-cached linears
- v1.08: wire cached trit linear into tiny GGUF decode
- v1.07: session trit prepack cache (skip re-quant/re-pack on repeated matvec)
- v1.06: session-backed packed trit linear + microbench
- v1.05: packed bitplane `linear_trit` throughput path + local BPE sidecar merges loader
- v1.04: char/token table tokenizer + multi-tensor GGUF session greedy decode (CPU)
- v1.03: PTX emitters + optional ptxas cubin + buffer/plan_launch surface
- v1.02: additive CUDA path (`cuda_*` probe + portable kernels; cpu-fallback without GPU)
- v1.01 production baseline: + loop/for surface sugar (`when`/`over`/`for-in`/`range`)
- Follow-ons (tabled): driver kernel launch (cuModule/cuLaunch), CUBLAS, multi-GPU; HF sentencepiece proto
