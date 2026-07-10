# Trin Language Specification (v0.3)

Trin is a kernel-oriented language whose **only control combinator** is the ternary form:

```
gate | pure | each
```

Read as:

- **if gate is false (0)** → return / short-circuit (monoid identity or void)
- **if gate is true (1)** → evaluate `pure` once
- **else (soft or residual)** → evaluate `each` over a collection / residual stream

The design goal is **virtual ternary on binary metal**: every control edge is a
weight in `{0,1}` (hard) or `[0,1]` (soft), so backends lower to select/blend/FMA
rather than unpredictable branches. Long-term: bare-metal AI kernels as a C path.

---

## 1. Design principles

1. **One combinator.** No `if`, `else`, `for`, `while`, `match` keywords in kernel mode.
2. **Unit spine.** Soft gates are `unit ∈ [0,1]`. Hard gates are `bit ∈ {0,1}` or `trit ∈ {-1,0,+1}`.
3. **Branchless default.** Soft paths compile to arithmetic multiplex; hard paths to masks/select.
4. **Spans, not hidden loops.** The third arm walks an explicit collection (or is a scalar residual).
5. **C-adjacent memory.** Arenas + length-carrying spans; no GC in v0.1 kernels.
6. **Interop.** Emit C / LLVM; call `extern "C"` freely.

This document is normative for the reference interpreter and the C-lowering backend in this tree.

---

## 2. Lexical structure

### 2.1 Comments

```
// line comment
/* block comment */
```

### 2.2 Tokens

| Class | Pattern / literals |
|-------|--------------------|
| Ident | `[A-Za-z_][A-Za-z0-9_]*` |
| Int | `0` \| `[1-9][0-9]*` |
| Float | digits with `.` (e.g. `0.5`, `1.0`) |
| String | `"…"` with `\\`, `\"`, `\n`, `\t` |
| Operators | `\|` `=` `=>` `->` `.` `,` `(` `)` `{` `}` `[` `]` `:` `;` `+` `-` `*` `/` `<` `>` `<=` `>=` `==` `!=` `!` |
| Keywords | `fn` `let` `type` `extern` `return` `when` `do` `over` `as` `for` `in` `span` `unit` `bit` `trit` `str` `i32` `f32` `void` `true` `false` `import` |

`|` is both the ternary separator and must appear **exactly twice** at each ternary level (gate, pure, each). Nested ternaries are allowed; parentheses disambiguate.

### 2.3 Layout

Expressions may span lines. A ternary may be written:

```
gate
  | pure
  | each
```

---

## 3. Types

### 3.1 Primitive types

| Type | Domain | Notes |
|------|--------|-------|
| `void` | `{}` | Only as identity / no value |
| `bit` | `{0,1}` | Hard gate |
| `trit` | `{-1,0,+1}` | Discrete ternary / ternary-net mode |
| `unit` | `[0,1] ⊂ f32` | Soft gate; debug-checked |
| `i32` | 32-bit int | Indexing, lengths |
| `f32` | IEEE-754 binary32 | Numeric tensors |
| `str` | UTF-8 byte string | Host I/O; not a gate |

### 3.2 Spans and functions

```
span[T]          // contiguous sequence of T
fn (T1, …) -> R  // function type
```

### 3.3 Gate kinds

A value is a **gate** if its type is `bit`, `trit`, or `unit`.

Coercions (explicit in surface syntax via builtins; implicit only where noted):

| From | To | Rule |
|------|-----|------|
| `bit` | `unit` | `0→0.0`, `1→1.0` |
| `unit` | `bit` | `harden(u, τ=0.5)`: `u≥τ → 1 else 0` |
| `trit` | `unit` | `-1→0`, `0→0.5`, `+1→1` |
| `unit` | `trit` | thresholds `τ±` (default 1/3, 2/3) |
| `i32` / `str` | gate | **illegal** as gate |

### 3.4 Unit algebra (closed in soft mode)

```
and(a,b)  = a * b
or(a,b)   = 1 - (1-a)*(1-b)
not(a)    = 1 - a
mux(g,a,b)= g | a | b
```

Debug builds trap if a `unit` leaves `[0,1]` by more than `ε=1e-5` after ops that claim unit closure.

---

## 4. Grammar (core)

```
program     ::= item*
item        ::= fn_def | let_stmt | extern_decl

fn_def      ::= 'fn' Ident '(' params? ')' ('->' type)? block
params      ::= param (',' param)*
param       ::= Ident ':' type
extern_decl ::= 'extern' 'fn' Ident '(' params? ')' ('->' type)? ';'

block       ::= '{' stmt* expr? '}'
stmt        ::= let_stmt | expr ';' | return_stmt
let_stmt    ::= 'let' Ident (':' type)? '=' expr ';'
return_stmt ::= 'return' expr? ';'

type        ::= 'void' | 'bit' | 'trit' | 'unit' | 'i32' | 'f32' | 'str'
              | 'span' '[' type ']'
              | 'fn' '(' type_list? ')' '->' type

expr        ::= ternary
ternary     ::= pipe_or_atom ('|' pipe_or_atom '|' pipe_or_atom)?
pipe_or_atom::= or_expr
or_expr     ::= and_expr (('||' | 'or') and_expr)*   // desugar: unit or
and_expr    ::= cmp_expr (('&&' | 'and') cmp_expr)*
cmp_expr    ::= add_expr (cmp_op add_expr)?
add_expr    ::= mul_expr (('+'|'-') mul_expr)*
mul_expr    ::= unary (('*'|'/') unary)*
unary       ::= ('!'|'-'|'+') unary | postfix
postfix     ::= primary (call | index | field)*
call        ::= '(' args? ')'
index       ::= '[' expr ']'
field       ::= '.' Ident
primary     ::= literal | Ident | '(' expr ')' | block | lambda | span_lit
lambda      ::= '(' params? ')' '=>' expr
span_lit    ::= '[' args? ']'

literal     ::= Int | Float | String | 'true' | 'false'
args        ::= expr (',' expr)*
```

**Precedence:** ternary `| |` binds **looser** than all arithmetic and comparisons.
A bare `a | b | c` is always ternary, never bitwise-or (Trin has no bitwise `|` in v0.1).

### 4.1 Surface sugar (desugars to ternary)

| Sugar | Desugars to |
|-------|-------------|
| `when g do e` | `g \| e \| 0` (numeric identity each-arm; v1.01) |
| `return? g` | `g \| () \| ()` with void id |
| `over xs as x do e` | `map((x)=>e, xs)` (v1.01 practical foreach; empty → empty) |
| `for x in xs do e` | `map((x)=>e, xs)` (v1.01 alias of over) |
| `range(n)` / `range(lo,hi)` / `range(lo,hi,step)` | half-open i32 span builder (v1.01 builtin) |
| `g && e` / `g and e` | value-level bit AND (also sugar target `g \| e \| 0`) |
| `g \|\| e` / `g or e` | value-level bit OR (also sugar target `g \| 1 \| e`) |

Keywords added in v1.01: `when` `do` `over` `as` `for` `in`. Still no real
branch/loop control combinators — only desugar onto `gate \| pure \| each` + `map`.

Reference interpreter implements the v1.01 sugar set above.

---

## 5. Operational semantics

We define a big-step relation `σ ⊢ e ⇓ v` under store/env `σ`.

### 5.1 Values

```
v ::= void
    | bit(b) | trit(t) | unit(u) | i32(n) | f32(x) | str(s)
    | span(T, [v0,…,vn-1])
    | closure(params, body, env)
    | hostfn(name)
```

### 5.2 Ternary evaluation (core rule)

```
σ ⊢ g ⇓ vg
gate_weight(vg) = w ∈ [0,1]
```

**Case A — hard false:** `w = 0` (bit 0, or unit ≤ ε)

Classic if/else form used by the reference interpreter and hello example:

```
σ ⊢ e ⇓ ve
σ ⊢ (g | p | e) ⇓ ve
```

So **false takes the third arm** (return-default / else / for-collection body when that arm is a map).  
Kernel mode may instead reduce false to monoid `identity` when `e` is omitted or marked `!abort`; v0.1 always evaluates `e`.

**Case B — hard true:** `w = 1` (bit 1, unit ≥ 1−ε, trit +1)

```
σ ⊢ p ⇓ vp
σ ⊢ (g | p | e) ⇓ vp
```

**Case C — soft / residual:** `0 < w < 1` (unit mid-range or trit 0 → weight 0.5)

```
σ ⊢ p ⇓ vp
σ ⊢ e ⇓ ve
σ ⊢ (g | p | e) ⇓ blend(w, vp, ve)
```

`blend` by type:

| Type | blend(w, a, b) |
|------|----------------|
| `unit`/`f32`/`i32`/`bit` | `w*a + (1-w)*b` (bit rounded in hard mode) |
| `str` | if `w ≥ 0.5` then `a` else `b` (soft string select) |
| `span[T]` | elementwise blend if same len; else prefer `a` if `w≥0.5` else `b` |
| `void` | `void` |

**Collection-each convention:** if `e` is a `span` of values that should be *mapped*,
the surface form uses `map` / lambda. Evaluation of `map(f, xs)` is:

```
σ ⊢ xs ⇓ span([x0,…,xn-1])
σ ⊢ f(xi) ⇓ yi   for each i
result = span([y0,…,yn-1])
```

Optional reduce: `sum(span)` folds with `+` and identity `0`.

### 5.3 Gate weight

```
gate_weight(bit(b))  = float(b)
gate_weight(unit(u)) = clamp(u, 0, 1)
gate_weight(trit(t)) = { -1: 0.0, 0: 0.5, +1: 1.0 }
gate_weight(other)   = type error
```

### 5.4 Side effects

- **Pure arm** may perform host I/O (read line, print) when called via hostfns.
- **Each arm** in kernel mode **should** be pure (enforced as a warning in v0.1 interpreter; hard error planned for AI kernel mode).
- Order: evaluate `g` fully; then if hard-true only `p`; if soft both `p` then `e`; if hard-false neither (identity).

This preserves the “if false return” early-out for hard gates while allowing soft blend when training / MoE routing needs gradients through gates.

---

## 6. Built-in host functions (v0.1)

| Name | Type | Meaning |
|------|------|---------|
| `print(s: str) -> void` | write `s` to stdout (no forced newline) |
| `println(s: str) -> void` | write `s` + newline |
| `input(prompt: str) -> str` | print prompt, read line, strip trailing newline |
| `str_concat(a: str, b: str) -> str` | concatenate |
| `str_len(s: str) -> i32` | byte length |
| `dens(xs: span[T]) -> unit` | `0` if empty else `1` (hard presence) |
| `len(xs: span[T]) -> i32` | length |
| `sum(xs: span[f32\|i32\|unit])` | fold `+` |
| `map(f, xs)` | elementwise apply (interpreter) |
| `harden(u: unit) -> bit` | `u ≥ 0.5` |
| `soften(b: bit) -> unit` | cast |
| `unit_of(x: f32) -> unit` | clamp to `[0,1]` |

---

## 7. Compilation model

```
Source .trin
   → Parse (AST)
   → Typecheck (optional in spike; interpreter is dynamic)
   → TrinIR
   → Backend: Interpreter | C lower | (future MLIR/LLVM)
```

### 7.1 TrinIR node

```
Ternary { gate: ValueId, pure: Region, each: Region, blend: BlendOp }
BlendOp = SoftMux | HardSelect | MapReduce
```

### 7.2 C lowering (branchless soft)

```c
// g | p | e  for f32/unit
float out = g * p + (1.0f - g) * e;
```

Hard bit:

```c
// select without data-dependent branch when p,e cheap:
float out = b ? p : e;   // may become CMOV / blendv
```

Map-each over span length `n`:

```c
for (int i = 0; i < n; i++) ys[i] = f(xs[i]); // vectorizable
```

### 7.3 Why this is “virtual ternary”

Binary wires stay base-2. Ternary appears as:

1. **Control arity-3** in the AST (abort / pure / each)
2. **Optional trit values** for discrete nets `{-1,0,+1}`
3. **Soft unit gates** for probability routing

Efficiency comes from **one CFG family → SIMD/GPU-friendly code**, not from ternary transistors.

---

## 8. Hello-world program (normative example)

File: `examples/hello.trin`

Semantics:

1. Prompt the user for a name (`input`).
2. Non-empty name → pure path builds `Hello world, $username`.
3. Empty name → each/false path falls back to `Hello world, stranger`.

Using only the ternary combinator for the name choice:

```
let name = input("Enter your name: ");
let msg =
  (str_len(name) > 0)
    | str_concat("Hello world, ", name)
    | "Hello world, stranger";
println(msg);
```

Hard `bit` gate from comparison; pure arm personalizes; false arm returns default string (identity-as-default, not void).

---

## 9. Error model (v0.1)

| Condition | Result |
|-----------|--------|
| Gate not bit/trit/unit | TypeError |
| Arity wrong in call | RuntimeError |
| Division by zero | RuntimeError |
| Index OOB | RuntimeError |
| Unit out of range (debug) | UnitRangeError |

---

## 10. Out of scope (v0.1)

- Full borrow checker / lifetimes
- Generics beyond `span[T]` dynamic
- Modules beyond single file
- Async, threads
- True balanced-ternary ISA backend
- Full ML / tensor autograd package (partial reverse tape via `soft_mux`/`grad_expr`/`map`/`zip_with`/`reduce`/`scan`/`filter` already ships through v0.15)

---

## 11. Acceptance for this spike

1. This SPEC exists and defines `g \| pure \| each`.
2. `examples/hello.trin` prompts and prints `Hello world, $username`.
3. Python interpreter runs hello end-to-end.
4. C backend emits compilable C for the same program (I/O via stdio).

---

## 12. Versioning

- **0.1** — spike: interpreter + C string/IO hello + soft mux patterns
- **0.2** — typechecker (`python -m trin_lang --check`), span numeric kernels, trit GEMM, C `kernels.h`
- **0.3** — bitplane trit (pos/neg), popcount GEMM, kernel C emission (`emit_kernels.py`)
- **0.4** — native Go toolchain (`cmd/trin`): lex/parse/interp + C codegen; builds examples without python3
- **0.5** — Go codegen backend (`--backend go`, default): host-cc free native builds via `go build`
- **0.6** — soft-mux reverse-mode AD (`soft_mux`/`soft_mux_grad*`) + TrinIR (`emit-ir`); Go tape in `internal/ad`
- **0.7** — full f32 reverse tape (`grad_expr` over `+ - * / soft_mux`) + `emit-llvm` text emission (no llc)
- **0.8** — span-kernel reverse AD (`sum`/`dot` on tape + `grad_sum`/`grad_dot`) + LLVM helpers (`trin_sum4`/`trin_dot4`)
- **0.9** — `span_add` / const-mul `map` AD (`grad_span_add`/`grad_scale`/`map_mul`) + LLVM span helpers
- **0.10** — true unary `map((x)=>body)` reverse AD on the tape + optional `emit-obj` via hosted `llc`
- **0.11** — nested map reverse AD + general unary map codegen (Go default + C element-loop parity for scalar `+ - * / soft_mux` bodies)
- **0.12** — binary `zip_with((x,y)=>body, a, b)` interpreter + reverse AD (`grad_expr`) + Go/C pairwise codegen; free vars and nested map spans OK
- **0.13** — `reduce`/`fold((acc,x)=>body, init, span)` left-fold interpreter + reverse AD + Go/C fold-loop codegen; composes with map/zip_with spans
- **0.14** — ternary `zip_with((x,y,z)=>body, a, b, c)` + `scan((acc,x)=>body, init, span)` interpreter + reverse AD + Go/C multi-span / scan-loop codegen; composes with map/reduce
- **0.15** — general n-ary `zip_with((p0..pN-1)=>body, s0..sN-1)` (N≥2) + `filter((x)=>pred, span)` (hard keep when pred≠0) interpreter + reverse AD + Go/C codegen; filter keep-set frozen at tape-build
- **0.16** — denser trit bitplane packing: 32 trits/i32 word planes, boolean product combinators, real `trit_gemm_bitplane` via popcount planes (Go/C parity)
- **0.17** — hosted `llc` link into full binaries: `emit-obj` + `link-llvm` / `build --backend llvm` (cc host driver) with graceful miss when tools absent; LLVM IR bitplane word helpers (`trin_trit_dot_planes_word`, plane and/or/xor/popcount)
- **0.18** — wider SIMD-friendly multiword bitplane surface: `trit_dot_planes_simd` / `trit_gemm_bitplane_simd` (128-lane primary chunks + residual), `simd_lane_width`, C/`kernels.h` u64 multiword path, LLVM `trin_trit_dot_planes_x4`
- **0.19** — full-program LLVM lowering beyond pure f32 kernels: programs without `loss`/`kernel` export emit runnable `define i32 @main` with `println`/`to_str`/`str_concat`/`sum`/`dot`/`span_add`/`soft_mux`/`i32`; PIC `llc` + optional empty host driver; pure float `loss` modules still use the float host probe
- **0.20** — wider auto-vectorized LLVM span loops: counted `trin_sum_f32` / `trin_dot_f32` / `trin_span_add_f32` / `trin_span_scale_f32` (+ sum wrappers) with `llvm.loop.vectorize` metadata; `trin_vec_loop_probe` (n=8 → 36); fixed-4 unrolls kept; C `kernels.h` remains auto-vectorizable
- **0.21** — portable MLIR dialect / vector IR via `emit-mlir`: `func.func` soft_mux + span kernels (`scf.for` + `memref`), `vector<4xf32>` tiles (`trin_soft_mux_v4` / `trin_span_add_v4` / `trin_span_scale_v4` / `trin_sum_v4` / `trin_dot_v4`), probes (`trin_mlir_vec_probe` → 10, `trin_mlir_soft_probe` → 4); no system `mlir-opt` required; LLVM path unchanged
- **0.22** — MLIR→LLVM lowering without `mlir-opt`: `emit-mlir-llvm` (AST → MLIR → LLVM) and `lower-mlir` (MLIR text → LLVM); lowers soft_mux, sum4/dot4, span `scf.for` kernels to counted LLVM loops, and `vector<4xf32>` ops to `<4 x float>` (broadcast/shuffles, extractelement reductions); probes `trin_mlir_vec_probe` / `trin_mlir_soft_probe` become linkable float symbols (targets 10 / 4); `emit-mlir` / `emit-llvm` preserved
- **0.23** — full-program MLIR + executable MLIR→LLVM for `@main`: programs without `loss`/`kernel`/`soft_kernel` export emit `func.func @main()` in `emit-mlir` (arith + soft_mux/sum/dot/span_add/println/to_str/str_concat/simd_lane_width); `emit-mlir-llvm` appends `define i32 @main` with the same runtime as v0.19 full-program LLVM; loss/kernel modules keep the float-host path (no `@main` MLIR); no system `mlir-opt`
- **0.24** — wider auto-vector tiles beyond 4-lane: `emit-mlir` adds `vector<8xf32>` helpers (`trin_soft_mux_v8` / `trin_span_add_v8` / `trin_span_scale_v8` / `trin_sum_v8` / `trin_dot_v8`) plus `trin_mlir_vec8_probe` (lanes 1..8 → 36); `emit-mlir-llvm` / `lower-mlir` lower to `<8 x float>`; existing `vector<4xf32>` tiles preserved; no system `mlir-opt`
- **0.25** — richer MLIR dialect ops + portable pattern-match rewrites (no `mlir-opt`): high-level `"trin.soft_mux"` / `"trin.vsum"` / `"trin.vdot"` / `"trin.vspan_*"` / span ops expand to `func.call @trin_*` (width picked by type); constant soft_mux folds `w=1→pure`, `w=0→each`; CLI `rewrite-mlir`; `emit-mlir` embeds dialect/fold/vsum probes (targets 4 / 7 / 10) and applies rewrites; `lower-mlir` auto-rewrites residual dialect ops before LLVM lower; v4/v8 tiles preserved
- **0.26** — even wider auto-vector tiles (16-lane) + portable VLEN selection: `emit-mlir` adds `vector<16xf32>` helpers (`trin_soft_mux_v16` / `trin_span_add_v16` / `trin_span_scale_v16` / `trin_sum_v16` / `trin_dot_v16`), module attr `trin.vlen = 16`, `trin_vlen` / `trin_vlen_for_n` (pick among {4,8,16}), probes `trin_mlir_vec16_probe` → 136 and `trin_mlir_vlen_probe` → 16; `emit-mlir-llvm` / `lower-mlir` lower to `<16 x float>`; dialect rewrite width pick includes v16; v4/v8 tiles and rewrites preserved; no system `mlir-opt`
- **0.27** — deeper portable pattern-match rewrites (no `mlir-opt`): pure-SSA CSE within `func.func` bodies (arith constants / addf-subf-mulf / `vector.from_elements` / multi-use soft_mux commoning via identical `func.call @trin_soft_mux*`); vector-width pick notes when dialect ops expand to v4/v8/v16 helpers (`rewrite: vector_width=N`); rewrite summary carries `dialect_ops` / `soft_mux_folds` / `cse` / `vwidth`; probes `trin_mlir_cse_probe` → 6, `trin_mlir_soft_cse_probe` → 12, `trin_mlir_vwidth_probe` → 136; lower path emits matching LLVM probes; prior dialect folds + tiles preserved
- **0.28** — hardware-aware VLEN over portable default: host detect via `/proc/cpuinfo` flags (Linux) + GOARCH heuristics — AVX-512 → 16, AVX/AVX2 → 8, SSE-class → 4, aarch64 SVE → 16, NEON → 4, else portable 16; `emit-mlir` module attrs `trin.hw_vlen` / `trin.hw_isa`; helpers `@trin_hw_vlen` / `@trin_hw_vlen_for_n` / `@trin_hw_isa_code` (codes: portable=0,sse=1,avx=2,avx512=3,sve=4,neon=5); probe `@trin_mlir_hw_vlen_probe` → float(hw_vlen); lower path bakes host snapshot into LLVM constants; portable `@trin_vlen*` (always 16-class) preserved
- **0.29** — soft_mux strength reduction / algebraic folds on the portable rewrite path (no `mlir-opt`): after dialect expand, `soft_mux(w,x,x) → x` (SSA identity or equal constants), `soft_mux(w,p,0) → w*p`, `soft_mux(w,0,e) → (1-w)*e`, full three-constant fold `soft_mux(wc,pc,ec) → arith.constant`; classic `w=0/1` folds retained; rewrite summary carries `strength=N` alongside `soft_mux_folds` / `cse` / `vwidth`; probe `@trin_mlir_strength_probe` → 13.5; lower path emits the matching float constant; CSE multi-use soft_mux preserved by feeding non-literal pure/each through addf(+0)
- **0.30** — runtime dynamic feature dispatch for hardware VLEN in generated binaries: `lower-mlir` / `emit-mlir-llvm` lower `@trin_hw_vlen`, `@trin_hw_isa_code`, and `@trin_hw_vlen_for_n` via `@trin_hw_detect_once` (lazy CPUID on x86 via inline `cpuid` asm; aarch64 defaults to NEON 4; other arches portable 16) rather than baking emit-host constants; portable `@trin_vlen*` unchanged (always 16-class); MLIR module attrs keep emit-host `trin.hw_vlen` / `trin.hw_isa` as notes plus `trin.hw_dispatch = "runtime"`; `@trin_mlir_hw_vlen_probe` recomputes residual wants from runtime vlen; linked probes still report float(hw_vlen) matching the *running* CPU
- **0.31** — span/vector algebraic rewrites on the portable rewrite path (no `mlir-opt`): after dialect expand + soft_mux strength, fold `span_add_vN(x,0)` / `span_add_vN(0,x) → x`, `span_scale_vN(1,x) → x`, `span_scale_vN(0,x)` / `span_scale_vN(s,0) → 0`, `sum_vN(0) → 0`, `dot_vN(0,x)` / `dot_vN(x,0) → 0` using tracked zero vectors (broadcast / all-zero from_elements); rewrite summary carries `span_alg=N`; probe `@trin_mlir_span_alg_probe` → 30; lower path emits the matching float constant; soft_mux strength/CSE/vwidth and runtime HW VLEN unchanged
- **0.32** — optional dynamic tile selection inside full-program `@main` kernels: fixed-length `sum`/`dot` of n∈{4,8,16} lower through `@trin_dyn_sum4|8|16` / `@trin_dyn_dot4|8`, which call `@trin_hw_vlen_for_n(n)` (runtime CPUID path from 0.30) then pack into `<N x float>` bodies, peeling to smaller tiles when hw width is narrower; language surface adds `hw_vlen`, `hw_vlen_for_n`, `hw_isa_code`, `dyn_tile`; full-program MLIR notes dyn-tile markers + `func.call @trin_hw_vlen_for_n`; module attr `trin.dyn_tile = "runtime"`; emit-llvm and emit-mlir-llvm both embed runtime CPUID + dyn helpers with the `@main` fragment
- **0.33** — more soft_mux / bitplane algebraic folds on the portable rewrite path (no `mlir-opt`): vector soft_mux strength `soft_mux_vN(w,pure,0) → broadcast(w)*pure` and `soft_mux_vN(w,0,each) → broadcast(1-w)*each` when the zero arm is a tracked zero vector; trit plane folds `plane_and(x,x)→x`, `and(x,0)|and(0,x)→0`, `or(x,x)→x`, `or(x,0)|or(0,x)→x`, `xor(x,x)→0`, `xor(x,0)|xor(0,x)→x`, `popcount(0)→0` plus full const fold for and/or/xor/popcount; rewrite summary carries `bitplane=N`; probe `@trin_mlir_bitplane_probe` → 42; lower path emits the matching float constant; prior strength/span_alg/CSE/vwidth preserved
- **0.34** — residual peel for non-fixed-length spans in full-program kernels: lengths n∉{4,8,16} with n≥4 decompose via `peelDynTileChunks` into descending tiles {16,8,4} plus residual r∈{0,1,2,3}; full-program LLVM `@main` emits chained `@trin_dyn_sum16|8|4` calls plus scalar `fadd` residual (dots mul first, then peel sum); exact n∈{4,8,16} still use single dyn helpers; full-program MLIR notes residual-peel markers + `func.call @trin_hw_vlen_for_n`; module attr `trin.residual_peel = "emit"`
- **0.35** — nested soft_mux / bitplane product folds on the portable rewrite path (no `mlir-opt`): product weight `soft_mux(w1, soft_mux(w2,p,e), e) → soft_mux(w1*w2, p, e)` and each-arm dual `soft_mux(w1, p, soft_mux(w2,p,e)) → soft_mux(w1+(1-w1)*w2, p, e)` (same-weight pure re-nest is the special case w², not identity); plane absorb/product `and(and(a,b), a|b) → and`, `or(or(a,b), a|b) → or`, `and(a, or(a,b)) → a`, `or(a, and(a,b)) → a`; rewrite summary carries `nested=N`; probe `@trin_mlir_nested_probe` → 52; lower path emits the matching float constant

- **0.36** — auto residual peel for pointer/counted loop kernels (not only fixed span literals): `@trin_sum_f32` / `@trin_dot_f32` load peels max 16/8/4 via `@trin_hw_vlen_for_n` then scalar residual 0..3; portable fallback `hw_vlen_for_n` when CPUID helpers absent; probe `@trin_auto_peel_probe` → 55; module attrs `trin.residual_peel = "auto"`, `trin.auto_peel = "pointer"`; fixed-span residual peel (v0.34) preserved

- **0.37** — auto residual peel for span_add/scale store loops: `@trin_span_add_f32` / `@trin_span_scale_f32` peel max 16/8/4 vector loads/stores via `@trin_hw_vlen_for_n` then residual scalar; probe `@trin_store_peel_probe` → 220; module attr `trin.auto_peel = "pointer_store"`; sum/dot peels (v0.36) preserved
- **0.38** — multi-level soft_mux tree folds: `soft_mux(w, soft_mux(w2,p,e), soft_mux(w3,p,e)) → soft_mux(w*w2+(1-w)*w3,p,e)` plus stepwise pure-nest product; multi-word plane GEMM folds on `@trin_trit_dot_planes_word` / `_x4` (zero-arm / const / all-zero); probe `@trin_mlir_tree_probe` → 64
- **0.39** — adaptive tile width from runtime ISA: `@trin_adaptive_tile(n)` / `@trin_adaptive_tile_max` / `@trin_adaptive_isa_tile` consult live `@trin_hw_*` (sse→4, avx→8, avx512/sve→16, neon→4, portable→16); `AdaptivePeelChunks` gates peel schedules; peels/dyn already use `@trin_hw_vlen_for_n`; module attr `trin.adaptive_tile = "runtime_isa"`; probe `@trin_mlir_adaptive_tile_probe` → 100
- **0.40** — residual peels for more map/store kernels: `@trin_span_mul_f32` (out[i]=a[i]*b[i]) and `@trin_span_axpy_f32` (out[i]=a[i]+s*b[i]) peel 16/8/4 via `@trin_hw_vlen_for_n` + residual; probe `@trin_map_store_peel_probe` → 550; soft_mux forest fold `soft_mux(w, soft_mux(w2, soft_mux(w3,p,e), e), soft_mux(w4,p,e)) → soft_mux(w*w2*w3+(1-w)*w4,p,e)`; multi-word GEMM schedule folds on `@trin_trit_dot_planes_x4` (equal-words / 2+2); probe `@trin_mlir_forest_probe` → 80; module attrs `trin.auto_peel = "pointer_store_map"`, `trin.map_store_peel = "mul_axpy"`, `trin.soft_forest = "schedule"`
- **0.41** — residual peels for reduce/scan store maps: `@trin_span_scan_add_f32` inclusive prefix scan + `@trin_reduce_add_f32` (peeled sum alias); peels gate via `@trin_adaptive_peel_tile` (adaptive ISA peel ladder, not a hard-coded 16/8/4-only path); probe `@trin_scan_reduce_peel_probe` → 275; deeper soft_mux forest schedule `w*w2*w3*w4+(1-w)*w5` and each-arm product `w*w2+(1-w)*w3*w4`; probe `@trin_mlir_deep_forest_probe` → 90; module attrs `trin.auto_peel = "pointer_store_map_scan"`, `trin.scan_reduce_peel = "prefix_add"`, `trin.adaptive_peel = "isa_ladder"`, `trin.soft_forest = "deep_schedule"`
- **0.42** — scan-mul/reduce-mul peels: `@trin_span_scan_mul_f32` inclusive prefix product + `@trin_reduce_mul_f32`; probe `@trin_scan_mul_reduce_peel_probe` → 64; full-program residual peels emit `AdaptivePeelChunks` notes at host-ISA schedule (beyond fixed portable ladder-only comments); wide ladder includes 32-lane SVE-class peels (`AdaptiveTileSequence` / `@trin_adaptive_peel_tile(32)`); probe `@trin_wide_peel_ladder_probe` → host tile width; module attrs `trin.auto_peel = "pointer_store_map_scan_mul"`, `trin.scan_reduce_peel = "prefix_add_mul"`, `trin.adaptive_peel = "isa_ladder_32"`, `trin.wide_peel = "sve32"`
- **0.43** — fused scan-reduce: `@trin_span_scan_reduce_add_f32` / `@trin_span_scan_reduce_mul_f32` single-pass prefix+reduce peels via adaptive ISA ladder; probe `@trin_fused_scan_reduce_peel_probe` → 275; full-program exact-32 dyn helper `@trin_dyn_sum32` (and residual peels call it for w=32 tiles); host-reported / portable stub SVE VL>32 peels (`MaxPeelTile=64`, `AdaptiveTileSequence` 64/32/16/8/4, `@trin_wide_vlen_probe`); module attrs `trin.scan_reduce_peel = "prefix_add_mul_fused"`, `trin.dyn_tile = "runtime_32"`, `trin.adaptive_peel = "isa_ladder_64"`, `trin.wide_peel = "sve64"`, `trin.auto_peel = "pointer_store_map_scan_mul_fused"`
- **0.44** — multi-output fused scan-reduce: `@trin_span_scan_reduce_add_pair_f32` / `@trin_span_scan_reduce_mul_pair_f32` write inclusive prefix and store both last and reduce (sum+last / prod+last) without a second scan; fused-mul probe surface `@trin_fused_scan_reduce_mul_probe` → 64; multi-out add probe `@trin_fused_scan_reduce_multi_probe` → 330 (sum_of_prefixes+last+reduce = 220+55+55); host VL path via `@trin_hw_rdvl` (SVE RDVL / HWCAP when available, else `TRIN_HW_VLEN` / portable `MaxPeelTile` stub) with probe `@trin_rdvl_probe`; module attrs `trin.scan_reduce_peel = "prefix_add_mul_multi"`, `trin.rdvl = "host_sve"`, `trin.auto_peel = "pointer_store_map_scan_mul_multi"`
- **0.45** — first-class pair/tuple IR: `@trin_pair_f32` / `@trin_pair_first_f32` / `@trin_pair_second_f32` plus tuple-return fused kernels `@trin_span_scan_reduce_add_tuple_f32` / `_mul_tuple_f32`; one-pass multi-kernel fusion `@trin_span_scan_reduce_add_sum_f32` returns `{last, sum_prefix}`; probes `@trin_pair_tuple_probe` → 110 and `@trin_fused_sum_scan_probe` → 275; live aarch64 RDVL via `getauxval(AT_HWCAP)` + `rdvl` insn (snap to {4,8,16,32,64}) with `@trin_hw_rdvl_live` alias (non-SVE portable fallback preserved); module attrs `trin.scan_reduce_peel = "prefix_add_mul_multi_tuple"`, `trin.rdvl = "live_sve"`, `trin.pair = "tuple_f32"`, `trin.auto_peel = "pointer_store_map_scan_mul_multi_tuple"`
- **0.46** — language-level pair surface: type form `pair[T]`, builtins `pair` / `pair_first` / `pair_second` (interp + Go backend) lowering to `{float,float}` helpers; probe `@trin_lang_pair_probe` → 110; streaming-SVE/SME VL host surface `@trin_hw_sme_vl` / builtin `hw_sme_vl` (+ `hw_rdvl`) with probe `@trin_sme_vl_probe` (portable fallback = preferred tile); multi-kernel map+reduce fusion `@trin_fused_map_reduce_mul_add_f32` returns `{sum(a*b), sum(a+b)}` with probe `@trin_fused_map_reduce_probe` → 495; module attrs `trin.pair = "lang_tuple_f32"`, `trin.sme_vl = "streaming"`, `trin.map_reduce = "mul_add_fused"`

- **0.47** — deeper pair destructuring / typed pair returns: syntax `let (a, b) = pair_expr;` (parser `Names` + interp/Go/C/LLVM full bind of first/second); user fns may annotate `-> pair[f32]` and return `pair(...)`; IR helpers `@trin_pair_swap_f32` and probes `@trin_pair_destructure_probe` / `@trin_pair_return_probe` → 110; SME ZA tile schedule host surface `@trin_hw_sme_za_tile` / builtin `hw_sme_za_tile` with probe `@trin_sme_za_tile_probe` (portable = preferred tile edge); broader multi-kernel fusion `@trin_fused_map_reduce_mul_sum_f32` returns `{sum(a*b), sum(a)}` with probe `@trin_fused_map_reduce_mul_sum_probe` → 440; module attrs `trin.pair = "destructure_return_f32"`, `trin.sme_za = "tile_schedule"`, `trin.map_reduce = "mul_add_mul_sum_fused"`

- **0.48** — nested pair / multi-way destructure and pair pattern match: `let (a,b,c,...) = ...;` and nested `let (a, (b, c)) = ...;` flatten to leaf order; builtins `pair(a,b,...)` multi-way, `pair_match(p,i)`; IR `@trin_pair_match_f32` / probes `@trin_pair_nested_probe` / `@trin_pair_match_probe` → 110; SME ZA outer-product schedule host surface `@trin_hw_sme_za_outer` / `hw_sme_za_outer` with probe `@trin_sme_za_outer_probe` (portable = ZA tile edge); adaptive multi-kernel fusion cost model `@trin_fusion_cost` / `fusion_cost(n)` (portable cost=n) and cost-aware fused surface `@trin_fused_map_reduce_cost_mul_add_f32` with probes `@trin_fusion_cost_probe` → 10 and `@trin_fused_map_reduce_cost_probe` → 495; module attrs `trin.pair = "nested_match_f32"`, `trin.sme_za = "outer_product"`, `trin.map_reduce = "cost_mul_add_fused"`

- **0.49** — deeper nested pattern match (pair-of-pairs typed returns) and basic exhaustiveness: `pair_of_pairs(a,b,c,d)`, `pair_arity` / `pair_exhaust(p,n)`, let-destructure requires leaf arity match; probes `@trin_pair_of_pairs_probe` / `@trin_pair_exhaust_probe` → 110; SME ZA accumulating outer-product micro-kernel `@trin_hw_sme_za_outer_acc` / `hw_sme_za_outer_acc` / `sme_za_outer_acc(a,b)=sum(a)*sum(b)` with probes `@trin_sme_za_outer_acc_probe` (host) and `@trin_sme_za_outer_acc_kernel_probe` → 3025; fusion cost model with ISA/traffic weights `@trin_fusion_cost_weighted` / `@trin_fusion_cost_isa` / `@trin_fusion_select` and multi-candidate fused surface `@trin_fused_map_reduce_select_f32` with probes 20 / 10 / 110 / 495; module attrs `trin.pair = "nested_exhaust_f32"`, `trin.sme_za = "outer_acc"`, `trin.map_reduce = "cost_select_weighted"`

- **0.50** — recursive pair pattern match with wildcards / guards: `pair_wildcard` / `pair_guard(p,i,lo,hi)` / `pair_match_guard(p,i,lo,hi)` (guarded leaf extract; wildcard bind-any); probes `@trin_pair_wildcard_probe` / `@trin_pair_guard_probe` → 110; true ZA outer-product accumulate host path `@trin_hw_sme_za_outer_acc_true` / `hw_sme_za_outer_acc_true` with portable fallback to outer_acc, plus kernel `@trin_sme_za_outer_acc_true_f32` / probe `@trin_sme_za_outer_acc_true_probe` → 3025; calibrated ISA traffic weights `@trin_fusion_cost_calibrated` / `fusion_cost_calibrated(n)` (portable weight table) and fusion planner log surface `@trin_fusion_plan_log` / `fusion_plan_log(c0,c1,c2)` with multi-candidate fused surface `@trin_fused_map_reduce_plan_f32` and probes 20 / 110 / 495; module attrs `trin.pair = "pair_guard_wildcard_f32"`, `trin.sme_za = "outer_acc_true"`, `trin.map_reduce = "cost_calibrated_log"`

- **0.51** — richer pair ADTs with or-patterns: `pair_or` / `pair_or_match` / `pair_or_default`; probes `@trin_pair_or_probe` → 110; SME ZA streaming store host `@trin_hw_sme_za_stream_store` / `hw_sme_za_stream_store` / `sme_za_stream_store` with probes host + kernel → 3025; fusion planner feedback / PGO weights `@trin_fusion_pgo_weight` / `@trin_fusion_pgo_feedback` / `@trin_fusion_pgo_select` and fused surface `@trin_fused_map_reduce_pgo_f32` with probes 20 / 110 / 495; module attrs `trin.pair = "pair_or_f32"`, `trin.sme_za = "stream_store"`, `trin.map_reduce = "pgo_feedback"`

- **0.52** — pair ADT tagging / exhaustiveness: `pair_tag` / `pair_tag_match` / `pair_tag_exhaust`; probes `@trin_pair_tag_exhaust_probe` → 110; SME ZA streaming gather/scatter host `@trin_hw_sme_za_gather` / `@trin_hw_sme_za_scatter` and kernels `@trin_sme_za_gather_f32` / `@trin_sme_za_scatter_f32` with probe `@trin_sme_za_gather_scatter_probe` → 3025; online PGO adapt `@trin_fusion_pgo_adapt` / `@trin_fusion_pgo_counter_bump` and fused `@trin_fused_map_reduce_adapt_f32` with probes 20 / 110 / 495; module attrs `trin.pair = "pair_tag_exhaust_f32"`, `trin.sme_za = "gather_scatter"`, `trin.map_reduce = "pgo_adapt"`

- **0.53** — nested tagged pair ADTs / pattern sugar: `nested_tag` / `nested_tag_match` / `nested_tag_sugar`; probes `@trin_nested_tag_probe` → 110; SME ZA multi-vector gather/scatter host `@trin_hw_sme_za_gather_mv` / `@trin_hw_sme_za_scatter_mv` and kernels `@trin_sme_za_gather_mv_f32` / `@trin_sme_za_scatter_mv_f32` with probe `@trin_sme_za_gather_scatter_mv_probe` → 3025; closed-loop PGO with per-ISA counters `@trin_fusion_pgo_isa_counter` / `@trin_fusion_pgo_isa_loop` and fused `@trin_fused_map_reduce_isa_loop_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_tag_f32"`, `trin.sme_za = "multivec_gather_scatter"`, `trin.map_reduce = "pgo_isa_loop"`

- **0.54** — nested tag multi-way pattern sugar: `nested_tag_multi` / `nested_tag_multi_match` / `nested_tag_or_sugar`; probes `@trin_nested_tag_multi_probe` → 110; SME ZA tiled multivec stream host `@trin_hw_sme_za_gather_tile` / `@trin_hw_sme_za_scatter_tile` and kernels `@trin_sme_za_gather_tile_f32` / `@trin_sme_za_scatter_tile_f32` with probe `@trin_sme_za_gather_scatter_tile_probe` → 3025; online PGO ISA live `@trin_fusion_pgo_isa_live` / `@trin_fusion_pgo_isa_feedback` and fused `@trin_fused_map_reduce_isa_live_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_multiway_f32"`, `trin.sme_za = "tiled_multivec_stream"`, `trin.map_reduce = "pgo_isa_live"`

- **0.55** — nested tag recursive multi-way ADTs: `nested_tag_recur` / `nested_tag_recur_match` / `nested_tag_recur_sugar`; probes `@trin_nested_tag_recur_probe` → 110; SME ZA multi-tile stream host `@trin_hw_sme_za_gather_mt` / `@trin_hw_sme_za_scatter_mt` and kernels `@trin_sme_za_gather_mt_f32` / `@trin_sme_za_scatter_mt_f32` with probe `@trin_sme_za_gather_scatter_mt_probe` → 3025; online PGO ISA xkernel `@trin_fusion_pgo_xkernel` / `@trin_fusion_pgo_xkernel_counter` and fused `@trin_fused_map_reduce_xkernel_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_recur_f32"`, `trin.sme_za = "multi_tile_stream"`, `trin.map_reduce = "pgo_xkernel"`

- **0.56** — nested tag ADT serialization / tree folds: `nested_tag_ser` / `nested_tag_tree_fold` / `nested_tag_ser_fold`; probes `@trin_nested_tag_ser_fold_probe` → 110; SME ZA multi-tile residual peels host `@trin_hw_sme_za_gather_mt_peel` / `@trin_hw_sme_za_scatter_mt_peel` and kernels `@trin_sme_za_gather_mt_peel_f32` / `@trin_sme_za_scatter_mt_peel_f32` with probe `@trin_sme_za_gather_scatter_mt_peel_probe` → 3025; online multi-pass PGO `@trin_fusion_pgo_xpass` / `@trin_fusion_pgo_xpass_counter` and fused `@trin_fused_map_reduce_xpass_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_ser_f32"`, `trin.sme_za = "multi_tile_peel"`, `trin.map_reduce = "pgo_xpass"`

- **0.57** — nested tag ADT zip/map forest folds: `nested_tag_zip` / `nested_tag_map_fold` / `nested_tag_forest_fold`; probes `@trin_nested_tag_forest_fold_probe` → 110; SME ZA multi-tile auto peels host `@trin_hw_sme_za_gather_mt_auto` / `@trin_hw_sme_za_scatter_mt_auto` and kernels `@trin_sme_za_gather_mt_auto_f32` / `@trin_sme_za_scatter_mt_auto_f32` with probe `@trin_sme_za_gather_scatter_mt_auto_probe` → 3025; online windowed multi-pass PGO `@trin_fusion_pgo_window` / `@trin_fusion_pgo_window_counter` and fused `@trin_fused_map_reduce_window_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_forest_f32"`, `trin.sme_za = "multi_tile_auto"`, `trin.map_reduce = "pgo_window"`

- **0.58** — nested tag ADT forest reduce with keyed tags: `nested_tag_keyed` / `nested_tag_keyed_fold` / `nested_tag_keyed_reduce`; probes `@trin_nested_tag_keyed_reduce_probe` → 110; SME ZA multi-tile auto peels with host VL feedback host `@trin_hw_sme_za_gather_mt_vlfb` / `@trin_hw_sme_za_scatter_mt_vlfb` and kernels `@trin_sme_za_gather_mt_vlfb_f32` / `@trin_sme_za_scatter_mt_vlfb_f32` with probe `@trin_sme_za_gather_scatter_mt_vlfb_probe` → 3025; online windowed PGO with decay `@trin_fusion_pgo_decay` / `@trin_fusion_pgo_decay_counter` and fused `@trin_fused_map_reduce_decay_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_keyed_f32"`, `trin.sme_za = "multi_tile_vlfb"`, `trin.map_reduce = "pgo_decay"`

- **0.59** — nested tag ADT multi-key filter/reduce: `nested_tag_filter` / `nested_tag_multi_key` / `nested_tag_multi_reduce`; probes `@trin_nested_tag_multi_reduce_probe` → 110; SME ZA multi-tile peels with live host VL residual schedule host `@trin_hw_sme_za_gather_mt_vllive` / `@trin_hw_sme_za_scatter_mt_vllive` and kernels `@trin_sme_za_gather_mt_vllive_f32` / `@trin_sme_za_scatter_mt_vllive_f32` with probe `@trin_sme_za_gather_scatter_mt_vllive_probe` → 3025; online PGO EMA/half-life `@trin_fusion_pgo_ema` / `@trin_fusion_pgo_ema_counter` and fused `@trin_fused_map_reduce_ema_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_multi_f32"`, `trin.sme_za = "multi_tile_vllive"`, `trin.map_reduce = "pgo_ema"`
- **0.60** — nested tag ADT multi-key forest with ranked tag filters: `nested_tag_rank_filter` / `nested_tag_rank_key` / `nested_tag_rank_reduce`; probes `@trin_nested_tag_rank_reduce_probe` → 110; SME ZA multi-tile peels with live VL residual streaming schedule host `@trin_hw_sme_za_gather_mt_vlstream` / `@trin_hw_sme_za_scatter_mt_vlstream` and kernels `@trin_sme_za_gather_mt_vlstream_f32` / `@trin_sme_za_scatter_mt_vlstream_f32` with probe `@trin_sme_za_gather_scatter_mt_vlstream_probe` → 3025; online PGO ISA EMA counters with adaptive half-life `@trin_fusion_pgo_adapt_hl` / `@trin_fusion_pgo_adapt_hl_counter` and fused `@trin_fused_map_reduce_adapt_hl_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_rank_f32"`, `trin.sme_za = "multi_tile_vlstream"`, `trin.map_reduce = "pgo_adapt_hl"`
- **0.61** — nested tag ADT ranked multi-key forest with weighted tag scores: `nested_tag_weight_filter` / `nested_tag_weight_key` / `nested_tag_weight_reduce`; probes `@trin_nested_tag_weight_reduce_probe` → 110; SME ZA multi-tile peels with live VL residual streaming + cost model host `@trin_hw_sme_za_gather_mt_vlcost` / `@trin_hw_sme_za_scatter_mt_vlcost` and kernels `@trin_sme_za_gather_mt_vlcost_f32` / `@trin_sme_za_scatter_mt_vlcost_f32` with probe `@trin_sme_za_gather_scatter_mt_vlcost_probe` → 3025; online PGO ISA adaptive half-life with dual EMA / hysteresis `@trin_fusion_pgo_hyst` / `@trin_fusion_pgo_hyst_counter` and fused `@trin_fused_map_reduce_hyst_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_weight_f32"`, `trin.sme_za = "multi_tile_vlcost"`, `trin.map_reduce = "pgo_hyst"`
- **0.62** — nested tag ADT weighted multi-key forest with score thresholds: `nested_tag_thresh_filter` / `nested_tag_thresh_key` / `nested_tag_thresh_reduce`; probes `@trin_nested_tag_thresh_reduce_probe` → 110; SME ZA multi-tile peels with live VL residual streaming cost + adaptive tile pick host `@trin_hw_sme_za_gather_mt_vladapt` / `@trin_hw_sme_za_scatter_mt_vladapt` and kernels `@trin_sme_za_gather_mt_vladapt_f32` / `@trin_sme_za_scatter_mt_vladapt_f32` with probe `@trin_sme_za_gather_scatter_mt_vladapt_probe` → 3025; online PGO ISA dual EMA hysteresis with windowed reweight `@trin_fusion_pgo_reweight` / `@trin_fusion_pgo_reweight_counter` and fused `@trin_fused_map_reduce_reweight_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_thresh_f32"`, `trin.sme_za = "multi_tile_vladapt"`, `trin.map_reduce = "pgo_reweight"`
- **0.63** — nested tag ADT score-threshold multi-key forest with soft gates: `nested_tag_soft_filter` / `nested_tag_soft_key` / `nested_tag_soft_reduce`; probes `@trin_nested_tag_soft_reduce_probe` → 110; SME ZA multi-tile peels with live VL residual streaming adaptive cost + residual schedule host `@trin_hw_sme_za_gather_mt_vlsched` / `@trin_hw_sme_za_scatter_mt_vlsched` and kernels `@trin_sme_za_gather_mt_vlsched_f32` / `@trin_sme_za_scatter_mt_vlsched_f32` with probe `@trin_sme_za_gather_scatter_mt_vlsched_probe` → 3025; online PGO ISA dual EMA hysteresis windowed reweight with budget caps `@trin_fusion_pgo_budget` / `@trin_fusion_pgo_budget_counter` and fused `@trin_fused_map_reduce_budget_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_soft_f32"`, `trin.sme_za = "multi_tile_vlsched"`, `trin.map_reduce = "pgo_budget"`
- **0.64** — nested tag ADT soft-gated multi-key forest with calibrated gates: `nested_tag_cal_filter` / `nested_tag_cal_key` / `nested_tag_cal_reduce`; probes `@trin_nested_tag_cal_reduce_probe` → 110; SME ZA multi-tile peels with live VL residual streaming schedule + peel fusion host `@trin_hw_sme_za_gather_mt_vlfuse` / `@trin_hw_sme_za_scatter_mt_vlfuse` and kernels `@trin_sme_za_gather_mt_vlfuse_f32` / `@trin_sme_za_scatter_mt_vlfuse_f32` with probe `@trin_sme_za_gather_scatter_mt_vlfuse_probe` → 3025; online PGO ISA dual EMA hysteresis budget reweight with soft caps `@trin_fusion_pgo_softcap` / `@trin_fusion_pgo_softcap_counter` and fused `@trin_fused_map_reduce_softcap_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_cal_f32"`, `trin.sme_za = "multi_tile_vlfuse"`, `trin.map_reduce = "pgo_softcap"`
- **0.65** — nested tag ADT calibrated multi-key forest with adaptive gates: `nested_tag_adapt_filter` / `nested_tag_adapt_key` / `nested_tag_adapt_reduce`; probes `@trin_nested_tag_adapt_reduce_probe` → 110; SME ZA multi-tile peels with residual streaming schedule peel fusion + live tile mix host `@trin_hw_sme_za_gather_mt_vlmix` / `@trin_hw_sme_za_scatter_mt_vlmix` and kernels `@trin_sme_za_gather_mt_vlmix_f32` / `@trin_sme_za_scatter_mt_vlmix_f32` with probe `@trin_sme_za_gather_scatter_mt_vlmix_probe` → 3025; online PGO ISA dual EMA softcap reweight with dual soft/hard budget lanes `@trin_fusion_pgo_dual` / `@trin_fusion_pgo_dual_counter` and fused `@trin_fused_map_reduce_dual_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_adapt_f32"`, `trin.sme_za = "multi_tile_vlmix"`, `trin.map_reduce = "pgo_dual"`
- **0.66** — nested tag ADT adaptive multi-key forest with blended gates: `nested_tag_blend_filter` / `nested_tag_blend_key` / `nested_tag_blend_reduce`; probes `@trin_nested_tag_blend_reduce_probe` → 110; SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix + cost balance host `@trin_hw_sme_za_gather_mt_vlbal` / `@trin_hw_sme_za_scatter_mt_vlbal` and kernels `@trin_sme_za_gather_mt_vlbal_f32` / `@trin_sme_za_scatter_mt_vlbal_f32` with probe `@trin_sme_za_gather_scatter_mt_vlbal_probe` → 3025; online PGO ISA dual-lane soft/hard budget reweight with confidence blend `@trin_fusion_pgo_conf` / `@trin_fusion_pgo_conf_counter` and fused `@trin_fused_map_reduce_conf_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_blend_f32"`, `trin.sme_za = "multi_tile_vlbal"`, `trin.map_reduce = "pgo_conf"`
- **0.67** — nested tag ADT blended multi-key forest with gated confidence: `nested_tag_gate_filter` / `nested_tag_gate_key` / `nested_tag_gate_reduce`; probes `@trin_nested_tag_gate_reduce_probe` → 110; SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance + bandwidth model host `@trin_hw_sme_za_gather_mt_vlbw` / `@trin_hw_sme_za_scatter_mt_vlbw` and kernels `@trin_sme_za_gather_mt_vlbw_f32` / `@trin_sme_za_scatter_mt_vlbw_f32` with probe `@trin_sme_za_gather_scatter_mt_vlbw_probe` → 3025; online PGO ISA dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor `@trin_fusion_pgo_reg` / `@trin_fusion_pgo_reg_counter` and fused `@trin_fused_map_reduce_reg_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_gate_f32"`, `trin.sme_za = "multi_tile_vlbw"`, `trin.map_reduce = "pgo_reg"`
- **0.68** — nested tag ADT gated multi-key forest with score-calibrated confidence: `nested_tag_score_filter` / `nested_tag_score_key` / `nested_tag_score_reduce`; probes `@trin_nested_tag_score_reduce_probe` → 110; SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth + path cost host `@trin_hw_sme_za_gather_mt_vlpath` / `@trin_hw_sme_za_scatter_mt_vlpath` and kernels `@trin_sme_za_gather_mt_vlpath_f32` / `@trin_sme_za_scatter_mt_vlpath_f32` with probe `@trin_sme_za_gather_scatter_mt_vlpath_probe` → 3025; online PGO ISA dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor `@trin_fusion_pgo_floor` / `@trin_fusion_pgo_floor_counter` and fused `@trin_fused_map_reduce_floor_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_score_f32"`, `trin.sme_za = "multi_tile_vlpath"`, `trin.map_reduce = "pgo_floor"`
- **0.69** — nested tag ADT score-calibrated multi-key forest with soft confidence floor: `nested_tag_softfloor_filter` / `nested_tag_softfloor_key` / `nested_tag_softfloor_reduce`; probes `@trin_nested_tag_softfloor_reduce_probe` → 110; SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost + lane pressure host `@trin_hw_sme_za_gather_mt_vllane` / `@trin_hw_sme_za_scatter_mt_vllane` and kernels `@trin_sme_za_gather_mt_vllane_f32` / `@trin_sme_za_scatter_mt_vllane_f32` with probe `@trin_sme_za_gather_scatter_mt_vllane_probe` → 3025; online PGO ISA dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor `@trin_fusion_pgo_afloor` / `@trin_fusion_pgo_afloor_counter` and fused `@trin_fused_map_reduce_afloor_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_softfloor_f32"`, `trin.sme_za = "multi_tile_vllane"`, `trin.map_reduce = "pgo_afloor"`
- **0.70** — nested tag ADT soft-confidence-floor multi-key forest with hard floor gates: `nested_tag_hardgate_filter` / `nested_tag_hardgate_key` / `nested_tag_hardgate_reduce`; probes `@trin_nested_tag_hardgate_reduce_probe` → 110; SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure + spill model host `@trin_hw_sme_za_gather_mt_vlspill` / `@trin_hw_sme_za_scatter_mt_vlspill` and kernels `@trin_sme_za_gather_mt_vlspill_f32` / `@trin_sme_za_scatter_mt_vlspill_f32` with probe `@trin_sme_za_gather_scatter_mt_vlspill_probe` → 3025; online PGO ISA dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap `@trin_fusion_pgo_burst` / `@trin_fusion_pgo_burst_counter` and fused `@trin_fused_map_reduce_burst_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_hardgate_f32"`, `trin.sme_za = "multi_tile_vlspill"`, `trin.map_reduce = "pgo_burst"`
- **0.71** — nested tag ADT hard-floor-gate multi-key forest with dual gate blend: `nested_tag_dual_filter` / `nested_tag_dual_key` / `nested_tag_dual_reduce`; probes `@trin_nested_tag_dual_reduce_probe` → 110; SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model + occupancy host `@trin_hw_sme_za_gather_mt_vlocc` / `@trin_hw_sme_za_scatter_mt_vlocc` and kernels `@trin_sme_za_gather_mt_vlocc_f32` / `@trin_sme_za_scatter_mt_vlocc_f32` with probe `@trin_sme_za_gather_scatter_mt_vlocc_probe` → 3025; online PGO ISA dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down `@trin_fusion_pgo_cool` / `@trin_fusion_pgo_cool_counter` and fused `@trin_fused_map_reduce_cool_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_dual_f32"`, `trin.sme_za = "multi_tile_vlocc"`, `trin.map_reduce = "pgo_cool"`
- **0.72** — nested tag ADT dual-gate-blend multi-key forest with soft/hard dual score mix: `nested_tag_mix_filter` / `nested_tag_mix_key` / `nested_tag_mix_reduce`; probes `@trin_nested_tag_mix_reduce_probe` → 110; SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth host `@trin_hw_sme_za_gather_mt_vlqueue` / `@trin_hw_sme_za_scatter_mt_vlqueue` and kernels `@trin_sme_za_gather_mt_vlqueue_f32` / `@trin_sme_za_scatter_mt_vlqueue_f32` with probe `@trin_sme_za_gather_scatter_mt_vlqueue_probe` → 3025; online PGO ISA dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank `@trin_fusion_pgo_credit` / `@trin_fusion_pgo_credit_counter` and fused `@trin_fused_map_reduce_credit_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_mix_f32"`, `trin.sme_za = "multi_tile_vlqueue"`, `trin.map_reduce = "pgo_credit"`
- **0.73** — nested tag ADT soft/hard dual score mix multi-key forest with ranked credit gates: `nested_tag_credit_filter` / `nested_tag_credit_key` / `nested_tag_credit_reduce`; probes `@trin_nested_tag_credit_reduce_probe` → 110; SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth + watermark host `@trin_hw_sme_za_gather_mt_vlwater` / `@trin_hw_sme_za_scatter_mt_vlwater` and kernels `@trin_sme_za_gather_mt_vlwater_f32` / `@trin_sme_za_scatter_mt_vlwater_f32` with probe `@trin_sme_za_gather_scatter_mt_vlwater_probe` → 3025; online PGO ISA dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery `@trin_fusion_pgo_debt` / `@trin_fusion_pgo_debt_counter` and fused `@trin_fused_map_reduce_debt_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_credit_f32"`, `trin.sme_za = "multi_tile_vlwater"`, `trin.map_reduce = "pgo_debt"`
- **0.74** — nested tag ADT soft/hard dual score mix multi-key forest with ranked credit + margin gates: `nested_tag_margin_filter` / `nested_tag_margin_key` / `nested_tag_margin_reduce`; probes `@trin_nested_tag_margin_reduce_probe` → 110; SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark + watermark drain host `@trin_hw_sme_za_gather_mt_vldrain` / `@trin_hw_sme_za_scatter_mt_vldrain` and kernels `@trin_sme_za_gather_mt_vldrain_f32` / `@trin_sme_za_scatter_mt_vldrain_f32` with probe `@trin_sme_za_gather_scatter_mt_vldrain_probe` → 3025; online PGO ISA dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest `@trin_fusion_pgo_interest` / `@trin_fusion_pgo_interest_counter` and fused `@trin_fused_map_reduce_interest_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_margin_f32"`, `trin.sme_za = "multi_tile_vldrain"`, `trin.map_reduce = "pgo_interest"`
- **0.75** — nested tag ADT soft/hard dual score mix multi-key forest with ranked credit + margin + band gates: `nested_tag_band_filter` / `nested_tag_band_key` / `nested_tag_band_reduce`; probes `@trin_nested_tag_band_reduce_probe` → 110; SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain + refill host `@trin_hw_sme_za_gather_mt_vlrefill` / `@trin_hw_sme_za_scatter_mt_vlrefill` and kernels `@trin_sme_za_gather_mt_vlrefill_f32` / `@trin_sme_za_scatter_mt_vlrefill_f32` with probe `@trin_sme_za_gather_scatter_mt_vlrefill_probe` → 3025; online PGO ISA dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize `@trin_fusion_pgo_amortize` / `@trin_fusion_pgo_amortize_counter` and fused `@trin_fused_map_reduce_amortize_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_band_f32"`, `trin.sme_za = "multi_tile_vlrefill"`, `trin.map_reduce = "pgo_amortize"`

- **0.76** — nested tag ADT soft/hard dual score mix multi-key forest with ranked credit + margin + band + clamp gates: `nested_tag_clamp_filter` / `nested_tag_clamp_key` / `nested_tag_clamp_reduce`; probes `@trin_nested_tag_clamp_reduce_probe` → 110; SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain + refill + flush host `@trin_hw_sme_za_gather_mt_vlflush` / `@trin_hw_sme_za_scatter_mt_vlflush` and kernels `@trin_sme_za_gather_mt_vlflush_f32` / `@trin_sme_za_scatter_mt_vlflush_f32` with probe `@trin_sme_za_gather_scatter_mt_vlflush_probe` → 3025; online PGO ISA dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle `@trin_fusion_pgo_settle` / `@trin_fusion_pgo_settle_counter` and fused `@trin_fused_map_reduce_settle_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_clamp_f32"`, `trin.sme_za = "multi_tile_vlflush"`, `trin.map_reduce = "pgo_settle"`

- **0.77** — nested tag ADT soft/hard dual score mix multi-key forest with ranked credit + margin + band + clamp + peg gates: `nested_tag_peg_filter` / `nested_tag_peg_key` / `nested_tag_peg_reduce`; probes `@trin_nested_tag_peg_reduce_probe` → 110; SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain + refill + flush + seal host `@trin_hw_sme_za_gather_mt_vlseal` / `@trin_hw_sme_za_scatter_mt_vlseal` and kernels `@trin_sme_za_gather_mt_vlseal_f32` / `@trin_sme_za_scatter_mt_vlseal_f32` with probe `@trin_sme_za_gather_scatter_mt_vlseal_probe` → 3025; online PGO ISA dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow `@trin_fusion_pgo_escrow` / `@trin_fusion_pgo_escrow_counter` and fused `@trin_fused_map_reduce_escrow_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_peg_f32"`, `trin.sme_za = "multi_tile_vlseal"`, `trin.map_reduce = "pgo_escrow"`

- **0.78** — nested tag ADT soft/hard dual score mix multi-key forest with ranked credit + margin + band + clamp + peg + latch gates: `nested_tag_latch_filter` / `nested_tag_latch_key` / `nested_tag_latch_reduce`; probes `@trin_nested_tag_latch_reduce_probe` → 110; SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain + refill + flush + seal + purge host `@trin_hw_sme_za_gather_mt_vlpurge` / `@trin_hw_sme_za_scatter_mt_vlpurge` and kernels `@trin_sme_za_gather_mt_vlpurge_f32` / `@trin_sme_za_scatter_mt_vlpurge_f32` with probe `@trin_sme_za_gather_scatter_mt_vlpurge_probe` → 3025; online PGO ISA dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + clear `@trin_fusion_pgo_clear` / `@trin_fusion_pgo_clear_counter` and fused `@trin_fused_map_reduce_clear_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_latch_f32"`, `trin.sme_za = "multi_tile_vlpurge"`, `trin.map_reduce = "pgo_clear"`

- **0.79** — nested tag ADT soft/hard dual score mix multi-key forest with ranked credit + margin + band + clamp + peg + latch + fence gates: `nested_tag_fence_filter` / `nested_tag_fence_key` / `nested_tag_fence_reduce`; probes `@trin_nested_tag_fence_reduce_probe` → 110; SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain + refill + flush + seal + purge + sync host `@trin_hw_sme_za_gather_mt_vlsync` / `@trin_hw_sme_za_scatter_mt_vlsync` and kernels `@trin_sme_za_gather_mt_vlsync_f32` / `@trin_sme_za_scatter_mt_vlsync_f32` with probe `@trin_sme_za_gather_scatter_mt_vlsync_probe` → 3025; online PGO ISA dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + clear + final `@trin_fusion_pgo_final` / `@trin_fusion_pgo_final_counter` and fused `@trin_fused_map_reduce_final_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_fence_f32"`, `trin.sme_za = "multi_tile_vlsync"`, `trin.map_reduce = "pgo_final"`

- **0.80** — nested tag ADT soft/hard dual score mix multi-key forest with ranked credit + margin + band + clamp + peg + latch + fence + lock gates: `nested_tag_lock_filter` / `nested_tag_lock_key` / `nested_tag_lock_reduce`; probes `@trin_nested_tag_lock_reduce_probe` → 110; SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy queue depth watermark drain + refill + flush + seal + purge + sync + close host `@trin_hw_sme_za_gather_mt_vlclose` / `@trin_hw_sme_za_scatter_mt_vlclose` and kernels `@trin_sme_za_gather_mt_vlclose_f32` / `@trin_sme_za_scatter_mt_vlclose_f32` with probe `@trin_sme_za_gather_scatter_mt_vlclose_probe` → 3025; online PGO ISA dual-lane soft/hard budget reweight with confidence blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + clear + final + close `@trin_fusion_pgo_close` / `@trin_fusion_pgo_close_counter` and fused `@trin_fused_map_reduce_close_f32` with probes 20 / 20 / 495; module attrs `trin.pair = "nested_lock_f32"`, `trin.sme_za = "multi_tile_vlclose"`, `trin.map_reduce = "pgo_close"`

Point-release series v0.76–v0.80 complete on the ADT / SME peel / PGO triple surface.

---

## 13. v0.2 extensions

### 13.1 Typechecker

- Structural types: `void|bit|trit|unit|i32|f32|str|span[T]|fn(...) -> R`
- Gate requirement on ternary first arm: `bit|unit|trit`
- Pure/each arms unified via lattice (`unify`)
- Builtins registered with signatures; `map` / `sum` / `trit_gemm` specialized

### 13.2 Span kernels (interpreter + C)

| Builtin | Meaning |
|---------|---------|
| `map(f, xs)` | elementwise |
| `sum(xs)` | fold `+` |
| `zip` / `zip_with` | pair / n-ary combine (N≥2) |
| `filter(f, xs)` | keep elements where `f(x)` is nonzero |
| `reduce` / `fold` | left fold to scalar |
| `scan` | left fold collecting intermediate accs as a span |
| `scale(s, xs)` | `s * xs[i]` |
| `span_add` / `span_mul` | elementwise |
| `dot(a,b)` | Σ aᵢbᵢ |

C: `backend/kernels.h` — `trin_span_*`, auto-vectorizable loops (`-O3 -march=native`).

### 13.3 Trit GEMM

```
trit_gemm(A, B, M, K, N) -> span[i32]
```

Row-major flat spans; entries `trit ∈ {-1,0,+1}`; product closed; accumulate `i32`.

```
C[i,j] = Σ_t A[i,t] * B[t,j]
```

C: `trin_trit_gemm(const int8_t *A, const int8_t *B, int32_t *C, int M, int K, int N)`.

---

## 14. v0.3 bitplane trit

### 14.1 Encoding

Each trit `t ∈ {-1,0,+1}` uses two bits:

| t  | pos | neg |
|----|-----|-----|
| +1 | 1   | 0   |
|  0 | 0   | 0   |
| -1 | 0   | 1   |

Word packing: 64 trits → `uint64_t pos_w`, `uint64_t neg_w` (bit 0 = index 0).

### 14.2 Boolean product

```
p_pos = (Ap ∧ Bp) ∨ (An ∧ Bn)
p_neg = (Ap ∧ Bn) ∨ (An ∧ Bp)
a*b   = p_pos - p_neg          ∈ {-1,0,+1}
```

Dot over K: `popcount(p_pos) - popcount(p_neg)` on word-masked planes.

### 14.3 Language surface

| Builtin | Meaning |
|---------|---------|
| `trit_pack(xs)` | `span[trit] → [pos_bits, neg_bits]` (1 bit/element) |
| `trit_unpack(planes)` | reverse |
| `trit_pack_words(xs)` | dense pack → `[pos_w, neg_w, n]` with 32 trits/word |
| `trit_unpack_words(pos_w,neg_w,n)` | reverse dense pack |
| `trit_plane_and` / `or` / `xor` | bitwise over word spans |
| `trit_plane_popcount(ws)` | sum of set bits across words |
| `trit_mul_planes(ap,an,bp,bn)` | boolean product planes |
| `trit_dot_planes(...,n)` | masked `popcount(p_pos)-popcount(p_neg)` |
| `trit_gemm_bitplane(A,B,M,K,N)` | same contract as `trit_gemm` via dense word planes |

Python: `trin_lang/bitplane.py`. C: `trin_pack_*`, `trin_trit_dot_bitplane_u64`, `trin_trit_gemm_bitplane` in `kernels.h`. Go/C runtimes pack 32-wide words (v0.16).

### 14.4 Kernel emission

```
python3 backend/emit_kernels.py examples/trit_gemm.trin -o /tmp/tg.c
```

Pattern-matches constant `trit_gemm` / span ops and emits a standalone C main calling scalar + bitplane kernels.


---

## Appendix P — Production ternary LLM surface (v0.81–0.82)

Normative builtins for virtual-ternary LLM inference. Semantics must match between
the reference interpreter and the Go backend runtime embedded by `EmitGo`.

### P.1 Quantization

- `quantize_absmean(w: span[f32]) → [span[trit], f32]`
  - `scale = mean(|w_i|)`; `t_i = clamp(round(w_i/scale), {-1,0,+1})` with
    thresholds at ±0.5 on `w_i/scale`.
- `trit_pack_quant(w)` → `[pos_words, neg_words, n, scale]`.
- `quantize_file_absmean(path)` loads CSV floats, quantizes, writes
  `path.trit.csv` and `path.scale.txt`.

### P.2 Linear

- `linear_trit(W, x, M, K, scale [, bitplane_flag]) → span[f32]` length M
  - `y_i = scale * Σ_k W[i,k] * x[k]` with `W` trit, `x` f32.
- `linear_trit_gemm(A,B,M,K,N,scale)` → scaled bitplane GEMM as f32 span.

### P.3 Transformer cores

- `rmsnorm(x [, eps=1e-6] [, gain])`
- `softmax(x)`, `silu(x)`, `gelu(x)` (tanh approx)
- `rope(x, pos [, base=10000])` pair-wise interleaved rotations
- `residual_add` ≡ `span_add`
- `attn(q,K,V,d)`, `attn_scores`, `attn_values`, `mha(q,K,V,d,h)`
- `embed_row(table, idx, dim)`, `argmax`
- `sample_logits(logits [, temp] [, seed])` — greedy if temp≤0; else
  categorical from softmax(logits/temp) via deterministic LCG; optional seed
- `sample_topk(logits, k [, temp] [, seed])` — top-k then same path

### P.4 I/O

- `load_f32_csv`, `save_f32_csv`, `load_trit_csv` — comma/whitespace separated.

### P.5 KV-cache helpers (v0.82)

- `kv_new()` → empty span cache
- `kv_append(cache, row)` / `span_concat(a,b)` → grow cache
- `kv_len(cache, d)` → token count (`len/d`)
- `kv_row(cache, idx, d)` → token row
- Layout: row-major f32 span of length `n_tok * d`

### P.6 Deterministic RNG (v0.82)

- `rng_seed(s)`, `rng_uniform()` → xorshift64* LCG shared by sample path
- Same algorithm in interpreter and Go backend (reproducible binaries)

### P.7 GGUF / safetensors (v0.83–0.98)

Pure-Go readers in `internal/weights` (embedded into Go-backend builds):

- **safetensors** LE header: `load_safetensors_list`, `load_safetensors_tensor`,
  `load_safetensors_shape`. Dtypes: F32, F16, BF16, F64, I8/U8/I16/I32/I64 → f32.
- **GGUF** v2/v3 LE: `load_gguf_list`, `load_gguf_tensor`, `load_gguf_shape`,
  `load_gguf_meta`.
- `load_weight_tensor(path, name)` auto-detects by extension/magic.
- **mmap (v0.85):** opens use `syscall.Mmap` (heap fallback). List/shape/meta
  parse headers only. `LoadF32` slices encoded tensor bytes (`GGMLRawBytes` when
  known). `weight_file_mapped(path) → i32` reports whether the OS map succeeded.
  One-shot builtins close after each call.
- **Sessions (v0.86):** `open_weight_session(path) → id`,
  `weight_session_list/shape/tensor/meta/mapped(id…)`,
  `close_weight_session(id)` (idempotent). One mmap retained across multi-tensor
  loads; process-local int32 handles.
- **Windows (v0.87):** `weight_session_window(id, name, offset, count)` and
  one-shot `load_gguf_window` / `load_safetensors_window` decode only
  `[offset, offset+count)` flat elements. Block quants dequant intersecting
  blocks only; F32/F16 slice raw bytes. OOB ranges error cleanly.
- **Tiled linear (v0.88):** `weight_session_linear(id, name, x, M, K [, tileRows])`
  and `weight_session_linear_tile(id, name, x, K, row0, nRows)` compute
  `y = W @ x` for row-major MxK weights by decoding only each row tile window.
  One-shot: `load_gguf_linear`. Default tileRows=8. Q* remains block-local.
- **Fused linear (v0.89):** `weight_session_linear_fused` / `_fused_tile` /
  `load_gguf_linear_fused` stream quant blocks through a ≤1-block scratch,
  accumulating dots without materializing a full `nRows*K` f32 tile. F32/dense
  streams elements; Q4/Q8/QK use `dequantFn` block-by-block. Parity with window path.
- **Layer stack (v0.90):** `weight_session_forward_block` and
  `weight_session_layer_stack` chain residual (or plain) fused linears with
  RMSNorm on one retained session. Layer/gain names are CSV strings; gains are
  window-loaded (dim vectors). No full-matrix `LoadF32` of W tensors.
- **Session attention (v0.91):** `weight_session_attn_proj`, `weight_session_attn`,
  `weight_session_mha` run fused Q/K/V (and optional O) projections then existing
  attn/mha over growing KV; optional RoPE when `base>0`. No full proj LoadF32.
- **GQA packing (v0.92):** `gqa_expand_kv` / `gqa_repeat_kv_row` expand compact
  K/V heads for MHA; `weight_session_attn_proj_gqa` / `weight_session_gqa` project
  Q at full heads and K/V at nKV heads, keep a compact KV cache, expand into mha.
- **IQ/TQ dequant (v0.93):** `IQ4_NL` (non-linear 4-bit, QK=32) and `TQ2_0`
  (ternary 2-bit, QK=256) decode through `DecodeGGML` / window / session linear.
- **IQ4_XS + TQ1_0 (v0.94):** super-block IQ4 with sub-scales; ternary TQ1_0
  (5 trits/byte packing, QK=256).
- **IQ2_XXS + IQ3_XXS (v0.95):** importance-matrix grid quants with shared
  sign/kmask tables and iq2xxs/iq3xxs grids.
- **IQ2_XS + IQ2_S (v0.96):** extended IQ2 family with per-group scales,
  iq2xs (512) / iq2s (1024) grids, and per-group signs.
- **IQ3_S (v0.97):** IQ3 family completion — qs/qh/signs/scales layout with
  iq3s (512) grid, two-scale superblocks, shared kmask signs.
- **IQ1_S + IQ1_M (v0.98):** 1.5–1.75 bpw grids — shared iq1s (2048×uint64/int8)
  with IQ1S_DELTA=0.125; IQ1_M packs global d into scale high bits.

**GGUF tensor types decoded to f32 (v0.84):**

- Full precision: F32, F16, BF16, F64, I8/I16/I32/I64
- Block-32: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1
- Super-block-256 (K-quants): Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K

Supported additive IQ/TQ slice (v0.93–0.98): **IQ1_S** (19), **IQ1_M** (29),
**IQ2_XXS** (16), **IQ2_XS** (17), **IQ2_S** (22), **IQ3_XXS** (18), **IQ3_S** (21),
**IQ4_NL** (20), **IQ4_XS** (23), **TQ1_0** (34), **TQ2_0** (35). Other IQ*/MXFP*
still return a clear error.

### P.8 Production status

v0.98 marks this surface **production-ready** for CPU virtual-ternary inference
kernels, KV growth, deterministic sampling, CSV + GGUF/safetensors (incl. common
block quants, full IQ1/IQ2/IQ3 families + IQ4_NL/XS + TQ1_0/TQ2_0, mmap open,
retained sessions, windowed partial decode, tiled session matvec, fused quant
linear, multi-layer residual stack, session-wired multi-head attention, GQA
packing helpers), and compiled demos. CUDA path (v1.02–v1.03): device probe + portable kernels + PTX emitters + offline ptxas cubin + host buffer stubs. Tokenizer + GGUF session decode (v1.04). SIMD trit matvec + local BPE sidecar (v1.05). Session packed trit linear (v1.06). Session trit prepack cache (v1.07). Tiny GGUF decode via trit cache (v1.08). Trit residual layer stack (v1.09). Session attn/MHA/GQA on trit projections (v1.10). Tiny GGUF decode via session trit-attn (v1.11). Multi-layer residual trit-attn decoder stack (v1.12). Multi-layer GGUF greedy decode on that stack (v1.13). Multi-layer residual GQA stack + final RMSNorm decode (v1.14). Multi-token GQA SwiGLU prefill + residual decode (v1.15). Tied emb + ALiBi residual decode (v1.16). Llama GGUF name adapters (v1.17). Out of scope: multi-GPU, competitive tok/s, live driver launch, HF sentencepiece protobuf, network fetch.

- **1.01** — loop/for surface sugar: `when g do e`, `over xs as x do e`, `for x in xs do e` desugar to ternary/map; `range` half-open i32 span builtin; Go + interpreter parity demo `examples/loop_sugar.trin`
- **1.04** — char/token table tokenizer (`open_vocab`/`vocab_encode`/`vocab_decode`/`load_text`) + multi-tensor `tiny_llm.gguf` session greedy decode demo; CUDA device launch still deferred
- **1.05** — packed bitplane `linear_trit` set-bit accumulate + local BPE sidecar (`open_bpe`/`bpe_encode` merges ranks); no network
- **1.06** — session-backed packed trit linear: `weight_session_linear_trit` / `_tile` / `load_gguf_linear_trit` (GGUF→absmean→scalar/bitplane matvec) + microbench
- **1.07** — session trit prepack cache: `weight_session_prepack_trit` / `weight_session_linear_trit_cached` retain absmean trit + packed planes across matvecs
- **1.08** — tiny GGUF decode on trit cache: prepack Q/K/V/O/FFN/lm_head once, reuse `weight_session_linear_trit_cached` across tokens (`examples/tiny_gguf_decode_trit.trin`)
- **1.09** — multi-layer residual trit stack: `weight_session_forward_block_trit` / `weight_session_layer_stack_trit` / `weight_session_prepack_trit_csv` on shared prepack cache
- **1.10** — session attn/MHA/GQA on trit packs: `weight_session_attn_trit` / `mha_trit` / `gqa_trit` + proj variants; projections cached, attention f32
- **1.11** — multi-token GGUF greedy decode on trit-attn: prepack QKV/O/FFN/lm_head once, `weight_session_attn_trit` growing KV + RoPE, tokens match v1.08 trit decode
- **1.12** — multi-layer residual trit-attn stack: `forward_attn_block_trit` / `attn_stack_trit` with layer-major KV flats, SiLU FFN, shared prepack cache
- **1.13** — multi-layer GGUF greedy decode: `tiny_llm2.gguf` 2-layer fixture + `tiny_gguf_decode_stack_trit` via `attn_stack_trit` growing KV + cached lm_head
- **1.14** — multi-layer residual GQA stack: `forward_gqa_block_trit` / `gqa_stack_trit` with compact layer-major KV; `final_norm`; `tiny_gqa.gguf` decode → FFF
- **1.15** — SwiGLU FFN on GQA residual (`gate/up/down`); multi-token `gqa_swiglu_prefill_trit` causal fill; `tiny_gqa_swiglu.gguf` → FFF
- **1.16** — tied `embed_row` / `logits_tied[_trit]`; ALiBi GQA (`*_alibi`); `tiny_tied_alibi.gguf` → FFF
- **1.17** — `NameMap` / `AutoNameMap`; short↔`token_embd`/`blk.N.*`; `tiny_llama_named.gguf` → FFF
