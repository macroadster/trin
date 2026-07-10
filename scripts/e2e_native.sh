#!/usr/bin/env bash
# Native toolchain e2e — no python3; default backend is Go (no host cc required)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${PATH}:/home/nanobot/.local/go/bin:/home/nanobot/.local/bin:/home/nanobot/.local/llvm/bin"
cd "$ROOT"

echo "[1] go build trin"
go build -o trin ./cmd/trin
./trin version

echo "[2] interpret examples"
printf 'Ada\n' | ./trin run examples/hello.trin | tee /tmp/trin_hello_i.txt
grep -q 'Hello world, Ada' /tmp/trin_hello_i.txt
./trin run examples/soft_mux.trin | tee /tmp/trin_soft_i.txt
grep -q 'soft number ~ 8' /tmp/trin_soft_i.txt
./trin run examples/span_ops.trin | tee /tmp/trin_span_i.txt
grep -q 'dot: 10.0' /tmp/trin_span_i.txt
./trin run examples/trit_gemm.trin | tee /tmp/trin_tg_i.txt
grep -q 'C = \[1, -2, 1, 1\]' /tmp/trin_tg_i.txt
./trin run examples/bitplane_gemm.trin | tee /tmp/trin_bp_i.txt
grep -q 'bitplane C = \[1, -2, 1, 1\]' /tmp/trin_bp_i.txt
./trin run examples/bitplane_words.trin | tee /tmp/trin_bpw_i.txt
grep -q 'n = 36' /tmp/trin_bpw_i.txt
grep -q 'nwords = 2' /tmp/trin_bpw_i.txt
grep -q 'dot planes = -9' /tmp/trin_bpw_i.txt
grep -q 'dot scalar = -9' /tmp/trin_bpw_i.txt
grep -q 'gemm bitplane= \[-9\]' /tmp/trin_bpw_i.txt
grep -q 'xor self pop = 0' /tmp/trin_bpw_i.txt
./trin run examples/bitplane_simd.trin | tee /tmp/trin_bpsimd_i.txt
grep -q 'simd_lane_width = 128' /tmp/trin_bpsimd_i.txt
grep -q 'nwords = 5' /tmp/trin_bpsimd_i.txt
grep -q 'dot scalar-plane = -4' /tmp/trin_bpsimd_i.txt
grep -q 'dot simd-plane = -4' /tmp/trin_bpsimd_i.txt
grep -q 'gemm simd    = \[-4\]' /tmp/trin_bpsimd_i.txt
./trin run examples/soft_mux_grad.trin | tee /tmp/trin_grad_i.txt
grep -q 'y = 6' /tmp/trin_grad_i.txt
grep -q 'dy/dw = 8' /tmp/trin_grad_i.txt
grep -q 'dz/da = 2' /tmp/trin_grad_i.txt
./trin run examples/grad_tape.trin | tee /tmp/trin_tape_i.txt
grep -q 'loss = 11.6' /tmp/trin_tape_i.txt
grep -q 'd/da = 8.2' /tmp/trin_tape_i.txt
grep -q 'dy/du = 4' /tmp/trin_tape_i.txt
./trin run examples/grad_span.trin | tee /tmp/trin_spanad_i.txt
grep -q 'sum = 10' /tmp/trin_spanad_i.txt
grep -q 'dot = 32' /tmp/trin_spanad_i.txt
grep -q 'y = 8.5' /tmp/trin_spanad_i.txt
grep -q 'dy/dw = -5' /tmp/trin_spanad_i.txt
grep -q 'dloss/dx = 6' /tmp/trin_spanad_i.txt
./trin run examples/grad_map.trin | tee /tmp/trin_mapad_i.txt
grep -q 'sum_add = 10' /tmp/trin_mapad_i.txt
grep -q 'sum_scale = 12' /tmp/trin_mapad_i.txt
grep -q 'sum_map = 12' /tmp/trin_mapad_i.txt
grep -q 'y = 10' /tmp/trin_mapad_i.txt
grep -q 'dy/dw = -8' /tmp/trin_mapad_i.txt
grep -q 'dy/ds = 3' /tmp/trin_mapad_i.txt
grep -q 'dloss/ds = 6' /tmp/trin_mapad_i.txt
./trin run examples/grad_map_lambda.trin | tee /tmp/trin_maplam_i.txt
grep -q 'v = 15' /tmp/trin_maplam_i.txt
grep -q 'da = 6' /tmp/trin_maplam_i.txt
grep -q 'q = 27' /tmp/trin_maplam_i.txt
grep -q 'du = 6' /tmp/trin_maplam_i.txt
grep -q 'y = 10' /tmp/trin_maplam_i.txt
grep -q 'dy/dw = 8' /tmp/trin_maplam_i.txt
grep -q 'dy/da = 3.5' /tmp/trin_maplam_i.txt
./trin run examples/grad_nested_map.trin | tee /tmp/trin_nest_i.txt
grep -q 'v = 36' /tmp/trin_nest_i.txt
grep -q 'du = 4' /tmp/trin_nest_i.txt
grep -q 'da = 8' /tmp/trin_nest_i.txt
grep -q 'ds = 9' /tmp/trin_nest_i.txt
grep -q 'q = 9' /tmp/trin_nest_i.txt
grep -q 'dp = 2' /tmp/trin_nest_i.txt
grep -q 'dw = -2' /tmp/trin_nest_i.txt
./trin run examples/map_general.trin | tee /tmp/trin_mapgen_i.txt
grep -q 'ys = \[11' /tmp/trin_mapgen_i.txt
grep -q 'nested = \[6' /tmp/trin_mapgen_i.txt
grep -q 'sum = 27' /tmp/trin_mapgen_i.txt
./trin run examples/zip_with.trin | tee /tmp/trin_zip_i.txt
grep -q 'ys = \[10.5' /tmp/trin_zip_i.txt
grep -q 'sum = 63' /tmp/trin_zip_i.txt
./trin run examples/grad_zip_with.trin | tee /tmp/trin_gzip_i.txt
grep -q 'v = 22' /tmp/trin_gzip_i.txt
grep -q 'du = 5' /tmp/trin_gzip_i.txt
grep -q 'ds = 3' /tmp/trin_gzip_i.txt
grep -q 'q = 8.5' /tmp/trin_gzip_i.txt
grep -q 'dw = 7' /tmp/trin_gzip_i.txt
./trin run examples/reduce.trin | tee /tmp/trin_red_i.txt
grep -q 'sum = 10' /tmp/trin_red_i.txt
grep -q 'prod = 24' /tmp/trin_red_i.txt
grep -q 'wk = 20' /tmp/trin_red_i.txt
./trin run examples/grad_reduce.trin | tee /tmp/trin_gred_i.txt
grep -q 'v = 24' /tmp/trin_gred_i.txt
grep -q 'ds = 6' /tmp/trin_gred_i.txt
grep -q 'q = 29' /tmp/trin_gred_i.txt
grep -q 'm = 21' /tmp/trin_gred_i.txt
./trin run examples/zip3_scan.trin | tee /tmp/trin_z3_i.txt
grep -q 'ys = \[12.5' /tmp/trin_z3_i.txt
grep -q 'sum = 75' /tmp/trin_z3_i.txt
grep -q 'pref = \[1.0, 3.0, 6.0, 10.0\]' /tmp/trin_z3_i.txt
grep -q 'sk = \[2.0, 6.0, 12.0\]' /tmp/trin_z3_i.txt
grep -q 'nested = 7' /tmp/trin_z3_i.txt
./trin run examples/grad_zip3_scan.trin | tee /tmp/trin_gz3_i.txt
grep -q 'v = 39' /tmp/trin_gz3_i.txt
grep -q 'du = 7' /tmp/trin_gz3_i.txt
grep -q 'ds = 3' /tmp/trin_gz3_i.txt
grep -q 'q = 40' /tmp/trin_gz3_i.txt
grep -q 'da = 12' /tmp/trin_gz3_i.txt
grep -q 'db = 8' /tmp/trin_gz3_i.txt
grep -q 'm = 7.5' /tmp/trin_gz3_i.txt
grep -q 'dmw = 3' /tmp/trin_gz3_i.txt
./trin run examples/filter_nary.trin | tee /tmp/trin_fn_i.txt
grep -q 'ys = \[12.5, 27.5\]' /tmp/trin_fn_i.txt
grep -q 'sum = 40' /tmp/trin_fn_i.txt
grep -q 'skipped = \[1.0, 3.0\]' /tmp/trin_fn_i.txt
grep -q 'nested = 4' /tmp/trin_fn_i.txt
./trin run examples/grad_filter_nary.trin | tee /tmp/trin_gfn_i.txt
grep -q 'v = 40' /tmp/trin_gfn_i.txt
grep -q 'da = 2' /tmp/trin_gfn_i.txt
grep -q 'dh = 2' /tmp/trin_gfn_i.txt
grep -q 'q = 4' /tmp/trin_gfn_i.txt
grep -q 'dq = 0' /tmp/trin_gfn_i.txt
grep -q 'dk = 0' /tmp/trin_gfn_i.txt
grep -q 'm = 15' /tmp/trin_gfn_i.txt
grep -q 'dmr = 4' /tmp/trin_gfn_i.txt

echo "[3] emit-ir + emit-llvm smoke"
./trin emit-ir examples/soft_mux.trin -o /tmp/soft_mux.ir
grep -q 'soft_mux' /tmp/soft_mux.ir
./trin emit-llvm examples/soft_kernel.trin -o /tmp/soft_kernel.ll
grep -q 'trin_soft_mux' /tmp/soft_kernel.ll
grep -q 'define float @loss' /tmp/soft_kernel.ll
./trin emit-llvm examples/span_kernel.trin -o /tmp/span_kernel.ll
grep -q 'trin_sum4' /tmp/span_kernel.ll
grep -q 'trin_dot4' /tmp/span_kernel.ll
grep -q 'define float @loss' /tmp/span_kernel.ll
./trin emit-llvm examples/grad_map.trin -o /tmp/grad_map.ll
grep -q 'trin_sum_span_add4' /tmp/grad_map.ll
grep -q 'trin_sum_map_mul4' /tmp/grad_map.ll
./trin emit-llvm examples/map_lambda_kernel.trin -o /tmp/map_lambda_kernel.ll
grep -q 'define float @loss' /tmp/map_lambda_kernel.ll
grep -q 'fmul' /tmp/map_lambda_kernel.ll
./trin emit-llvm examples/nested_map_kernel.trin -o /tmp/nested_map_kernel.ll
grep -q 'define float @loss' /tmp/nested_map_kernel.ll
grep -q 'fadd' /tmp/nested_map_kernel.ll
grep -q 'fmul' /tmp/nested_map_kernel.ll
# emit-obj: graceful miss when llc absent; succeed if present
set +e
./trin emit-obj examples/soft_kernel.trin -o /tmp/soft_kernel.o 2>/tmp/trin_emit_obj.err
EC=$?
set -e
if command -v llc >/dev/null 2>&1; then
  test $EC -eq 0
  test -f /tmp/soft_kernel.o
  # full llvm link path when llc + cc both available
  if command -v cc >/dev/null 2>&1; then
    ./trin link-llvm examples/soft_kernel.trin -o /tmp/soft_kernel_llvm
    /tmp/soft_kernel_llvm | tee /tmp/trin_llvm_link.txt
    grep -q 'llvm entry loss = 7.0' /tmp/trin_llvm_link.txt
    ./trin build examples/llvm_link_probe.trin -o /tmp/llvm_link_probe --backend llvm
    /tmp/llvm_link_probe | grep -q 'llvm entry loss = 7.0'
    # full-program @main path (v0.19) — no float host driver
    ./trin emit-llvm examples/llvm_full_prog.trin -o /tmp/llvm_full_prog.ll
    grep -q 'define i32 @main' /tmp/llvm_full_prog.ll
    grep -q 'full-program LLVM lowering' /tmp/llvm_full_prog.ll
    ./trin link-llvm examples/llvm_full_prog.trin -o /tmp/llvm_full_prog
    /tmp/llvm_full_prog | tee /tmp/trin_llvm_full.txt
    grep -q 'product = 12' /tmp/trin_llvm_full.txt
    grep -q 'soft_mux = 4' /tmp/trin_llvm_full.txt
    grep -q 'sum = 10' /tmp/trin_llvm_full.txt
    grep -q 'dot = 11' /tmp/trin_llvm_full.txt
    grep -q 'span_add sum = 10' /tmp/trin_llvm_full.txt
    grep -q 'simd_lane_width = 128' /tmp/trin_llvm_full.txt
    ./trin build examples/llvm_full_prog.trin -o /tmp/llvm_full_prog_build --backend llvm
    /tmp/llvm_full_prog_build | grep -q 'product = 12'
    # bitplane helpers present in IR
    ./trin emit-llvm examples/soft_kernel.trin -o /tmp/soft_kernel_v17.ll
    grep -q 'trin_trit_dot_planes_word' /tmp/soft_kernel_v17.ll
    grep -q 'llvm.ctpop.i32' /tmp/soft_kernel_v17.ll
  fi
else
  test $EC -ne 0
  grep -q 'llc not found' /tmp/trin_emit_obj.err
  # link-llvm should also fail gracefully without llc
  set +e
  ./trin link-llvm examples/soft_kernel.trin -o /tmp/soft_kernel_llvm 2>/tmp/trin_link_llvm.err
  EC2=$?
  set -e
  test $EC2 -ne 0
  grep -q 'llc not found' /tmp/trin_link_llvm.err
fi

echo "[4] native compile (backend=go, host-cc free)"
mkdir -p build
./trin build examples/hello.trin -o build/hello --backend go
./trin build examples/soft_mux.trin -o build/soft_mux --backend go
./trin build examples/span_ops.trin -o build/span_ops --backend go
./trin build examples/trit_gemm.trin -o build/trit_gemm --backend go
./trin build examples/bitplane_gemm.trin -o build/bitplane_gemm --backend go
./trin build examples/bitplane_words.trin -o build/bitplane_words --backend go
./trin build examples/bitplane_simd.trin -o build/bitplane_simd --backend go
./trin build examples/soft_mux_grad.trin -o build/soft_mux_grad --backend go
./trin build examples/grad_tape.trin -o build/grad_tape --backend go
./trin build examples/grad_span.trin -o build/grad_span --backend go
./trin build examples/grad_map.trin -o build/grad_map --backend go
./trin build examples/grad_map_lambda.trin -o build/grad_map_lambda --backend go
./trin build examples/grad_nested_map.trin -o build/grad_nested_map --backend go
./trin build examples/map_general.trin -o build/map_general --backend go
./trin build examples/nested_map_kernel.trin -o build/nested_map_kernel --backend go
./trin build examples/zip_with.trin -o build/zip_with --backend go
./trin build examples/grad_zip_with.trin -o build/grad_zip_with --backend go
./trin build examples/reduce.trin -o build/reduce --backend go
./trin build examples/grad_reduce.trin -o build/grad_reduce --backend go
./trin build examples/zip3_scan.trin -o build/zip3_scan --backend go
./trin build examples/grad_zip3_scan.trin -o build/grad_zip3_scan --backend go
./trin build examples/filter_nary.trin -o build/filter_nary --backend go
./trin build examples/grad_filter_nary.trin -o build/grad_filter_nary --backend go

echo "[5] native run (go-backend bins)"
printf 'Ada\n' | ./build/hello | tee /tmp/trin_hello_n.txt
grep -q 'Hello world, Ada' /tmp/trin_hello_n.txt
printf '\n' | ./build/hello | tee /tmp/trin_hello_empty.txt
grep -q 'Hello world, stranger' /tmp/trin_hello_empty.txt
./build/soft_mux | tee /tmp/trin_soft_n.txt
grep -q 'high confidence' /tmp/trin_soft_n.txt
./build/span_ops | tee /tmp/trin_span_n.txt
grep -q 'non-empty' /tmp/trin_span_n.txt
./build/trit_gemm | tee /tmp/trin_tg_n.txt
grep -q 'C00 positive' /tmp/trin_tg_n.txt
./build/bitplane_gemm | tee /tmp/trin_bp_n.txt
grep -q 'roundtrip A = \[1, 0, -1, 0, 1, 1\]' /tmp/trin_bp_n.txt
./build/bitplane_words | tee /tmp/trin_bpw_n.txt
grep -q 'n = 36' /tmp/trin_bpw_n.txt
grep -q 'dot planes = -9' /tmp/trin_bpw_n.txt
grep -q 'gemm bitplane= \[-9\]' /tmp/trin_bpw_n.txt
grep -q 'xor self pop = 0' /tmp/trin_bpw_n.txt
./build/bitplane_simd | tee /tmp/trin_bpsimd_n.txt
grep -q 'simd_lane_width = 128' /tmp/trin_bpsimd_n.txt
grep -q 'dot simd-plane = -4' /tmp/trin_bpsimd_n.txt
grep -q 'gemm simd    = \[-4\]' /tmp/trin_bpsimd_n.txt
./build/soft_mux_grad | tee /tmp/trin_grad_n.txt
grep -q 'dz/dw = 10' /tmp/trin_grad_n.txt
grep -q 'dz/dx = 1.5' /tmp/trin_grad_n.txt
./build/grad_tape | tee /tmp/trin_tape_n.txt
grep -q 'loss = 11.6' /tmp/trin_tape_n.txt
grep -q 'd/dx = 1.6' /tmp/trin_tape_n.txt
./build/grad_span | tee /tmp/trin_spanad_n.txt
grep -q 'sum = 10' /tmp/trin_spanad_n.txt
grep -q 'y = 8.5' /tmp/trin_spanad_n.txt
grep -q 'dloss/da = 4' /tmp/trin_spanad_n.txt
./build/grad_map | tee /tmp/trin_mapad_n.txt
grep -q 'sum_add = 10' /tmp/trin_mapad_n.txt
grep -q 'y = 10' /tmp/trin_mapad_n.txt
grep -q 'dy/dw = -8' /tmp/trin_mapad_n.txt
grep -q 'dloss/dx = 2' /tmp/trin_mapad_n.txt
./build/grad_map_lambda | tee /tmp/trin_maplam_n.txt
grep -q 'v = 15' /tmp/trin_maplam_n.txt
grep -q 'q = 27' /tmp/trin_maplam_n.txt
grep -q 'dy/dw = 8' /tmp/trin_maplam_n.txt
grep -q 'dy/da = 3.5' /tmp/trin_maplam_n.txt
./build/grad_nested_map | tee /tmp/trin_nest_n.txt
grep -q 'v = 36' /tmp/trin_nest_n.txt
grep -q 'da = 8' /tmp/trin_nest_n.txt
grep -q 'ds = 9' /tmp/trin_nest_n.txt
grep -q 'q = 9' /tmp/trin_nest_n.txt
./build/map_general | tee /tmp/trin_mapgen_n.txt
grep -q 'ys = \[11' /tmp/trin_mapgen_n.txt
grep -q 'sum = 27' /tmp/trin_mapgen_n.txt
./build/nested_map_kernel | tee /tmp/trin_nestk_n.txt
grep -q 'y = 36' /tmp/trin_nestk_n.txt
./build/zip_with | tee /tmp/trin_zip_n.txt
grep -q 'ys = \[10.5' /tmp/trin_zip_n.txt
grep -q 'sum = 63' /tmp/trin_zip_n.txt
./build/grad_zip_with | tee /tmp/trin_gzip_n.txt
grep -q 'v = 22' /tmp/trin_gzip_n.txt
grep -q 'ds = 3' /tmp/trin_gzip_n.txt
grep -q 'q = 8.5' /tmp/trin_gzip_n.txt
grep -q 'dw = 7' /tmp/trin_gzip_n.txt
./build/reduce | tee /tmp/trin_red_n.txt
grep -q 'sum = 10' /tmp/trin_red_n.txt
grep -q 'prod = 24' /tmp/trin_red_n.txt
./build/grad_reduce | tee /tmp/trin_gred_n.txt
grep -q 'v = 24' /tmp/trin_gred_n.txt
grep -q 'q = 29' /tmp/trin_gred_n.txt
grep -q 'm = 21' /tmp/trin_gred_n.txt
./build/zip3_scan | tee /tmp/trin_z3_n.txt
grep -q 'ys = \[12.5' /tmp/trin_z3_n.txt
grep -q 'sum = 75' /tmp/trin_z3_n.txt
grep -q 'pref = \[1.0, 3.0, 6.0, 10.0\]' /tmp/trin_z3_n.txt
grep -q 'nested = 7' /tmp/trin_z3_n.txt
./build/grad_zip3_scan | tee /tmp/trin_gz3_n.txt
grep -q 'v = 39' /tmp/trin_gz3_n.txt
grep -q 'q = 40' /tmp/trin_gz3_n.txt
grep -q 'm = 7.5' /tmp/trin_gz3_n.txt
grep -q 'dmw = 3' /tmp/trin_gz3_n.txt
./build/filter_nary | tee /tmp/trin_fn_n.txt
grep -q 'ys = \[12.5, 27.5\]' /tmp/trin_fn_n.txt
grep -q 'sum = 40' /tmp/trin_fn_n.txt
grep -q 'nested = 4' /tmp/trin_fn_n.txt
./build/grad_filter_nary | tee /tmp/trin_gfn_n.txt
grep -q 'v = 40' /tmp/trin_gfn_n.txt
grep -q 'q = 4' /tmp/trin_gfn_n.txt
grep -q 'm = 15' /tmp/trin_gfn_n.txt
grep -q 'dms = 3' /tmp/trin_gfn_n.txt

# Optional C backend if cc present
if command -v cc >/dev/null 2>&1; then
  echo "[6] optional C backend smoke (hello + general map/zip/reduce/scan/filter emit)"
  ./trin build examples/hello.trin -o build/hello_c --backend c
  printf 'Ada\n' | ./build/hello_c | grep -q 'Hello world, Ada'
  # emit-only: proves general unary map / zip_with / reduce / zip3+scan / filter C codegen
  ./trin emit-c examples/map_general.trin -o /tmp/map_general_c.c
  grep -q 'for(int' /tmp/map_general_c.c
  grep -q 'env_set' /tmp/map_general_c.c
  ./trin emit-c examples/zip_with.trin -o /tmp/zip_with_c.c
  grep -q 'for(int' /tmp/zip_with_c.c
  grep -q 'env_set' /tmp/zip_with_c.c
  ./trin emit-c examples/reduce.trin -o /tmp/reduce_c.c
  grep -q 'for(int' /tmp/reduce_c.c
  grep -q 'env_set' /tmp/reduce_c.c
  ./trin emit-c examples/zip3_scan.trin -o /tmp/zip3_scan_c.c
  grep -q 'for(int' /tmp/zip3_scan_c.c
  grep -q 'env_set' /tmp/zip3_scan_c.c
  ./trin emit-c examples/filter_nary.trin -o /tmp/filter_nary_c.c
  grep -q 'for(int' /tmp/filter_nary_c.c
  grep -q 'env_set' /tmp/filter_nary_c.c
  ./trin emit-c examples/bitplane_words.trin -o /tmp/bitplane_words_c.c
  grep -q 'bi_trit_pack_words' /tmp/bitplane_words_c.c
  grep -q 'bi_trit_dot_planes' /tmp/bitplane_words_c.c
  ./trin emit-c examples/bitplane_simd.trin -o /tmp/bitplane_simd_c.c
  grep -q 'bi_trit_dot_planes_simd' /tmp/bitplane_simd_c.c
  grep -q 'bi_trit_gemm_bitplane_simd' /tmp/bitplane_simd_c.c
  grep -q 'bi_simd_lane_width' /tmp/bitplane_simd_c.c
fi

# LLVM IR multiword helpers always available (no llc needed)
./trin emit-llvm examples/soft_kernel.trin -o /tmp/soft_kernel_v18.ll
grep -q 'trin_trit_dot_planes_x4' /tmp/soft_kernel_v18.ll
grep -q 'trin_simd_lane_width' /tmp/soft_kernel_v18.ll
# full-program IR present even without llc
./trin emit-llvm examples/llvm_full_prog.trin -o /tmp/llvm_full_prog_v19.ll
grep -q 'define i32 @main' /tmp/llvm_full_prog_v19.ll
grep -q 'trin_println_str' /tmp/llvm_full_prog_v19.ll
# vectorizable span-loop helpers (v0.20)
./trin emit-llvm examples/llvm_vec_loops.trin -o /tmp/llvm_vec_loops.ll
grep -q 'define float @trin_sum_f32' /tmp/llvm_vec_loops.ll
grep -q 'define float @trin_dot_f32' /tmp/llvm_vec_loops.ll
grep -q 'define void @trin_span_add_f32' /tmp/llvm_vec_loops.ll
grep -q 'define void @trin_span_scale_f32' /tmp/llvm_vec_loops.ll
grep -q 'llvm.loop.vectorize.enable' /tmp/llvm_vec_loops.ll
grep -q 'trin_vec_loop_probe' /tmp/llvm_vec_loops.ll
if command -v llc >/dev/null 2>&1 && command -v cc >/dev/null 2>&1; then
  # Force float-host entry: renames @main so host can call the loop probe.
  ./trin link-llvm examples/llvm_vec_loops.trin -o /tmp/llvm_vec_loops --entry trin_vec_loop_probe
  /tmp/llvm_vec_loops | tee /tmp/trin_llvm_vec.txt
  grep -q 'llvm entry trin_vec_loop_probe = 36.0' /tmp/trin_llvm_vec.txt
  ./trin run examples/llvm_vec_loops.trin | tee /tmp/trin_vec_interp.txt
  grep -q 'sum8 = 36' /tmp/trin_vec_interp.txt
  grep -q 'dot8 = 204' /tmp/trin_vec_interp.txt
fi

# MLIR dialect / vector IR (v0.21+) — text only, no mlir-opt required
./trin emit-mlir examples/mlir_vec.trin -o /tmp/mlir_vec.mlir
grep -q 'trin.version = "0.81"' /tmp/mlir_vec.mlir
grep -q 'trin.vlen = 16' /tmp/mlir_vec.mlir
grep -q 'trin.hw_vlen' /tmp/mlir_vec.mlir
grep -q 'trin.hw_isa' /tmp/mlir_vec.mlir
grep -q 'trin.hw_dispatch = "runtime"' /tmp/mlir_vec.mlir
grep -q 'trin.dyn_tile = "runtime_32"' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_soft_mux' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_sum_f32' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_dot_f32' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_span_add_f32' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_span_scale_f32' /tmp/mlir_vec.mlir
grep -q 'vector<4xf32>' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_soft_mux_v4' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_span_add_v4' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_mlir_vec_probe' /tmp/mlir_vec.mlir
grep -q 'vector.reduction' /tmp/mlir_vec.mlir
# v0.24 wider 8-lane tiles
grep -q 'vector<8xf32>' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_soft_mux_v8' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_sum_v8' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_mlir_vec8_probe' /tmp/mlir_vec.mlir
# v0.26 16-lane tiles + VLEN
grep -q 'vector<16xf32>' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_soft_mux_v16' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_sum_v16' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_mlir_vec16_probe' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_vlen' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_vlen_for_n' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_mlir_vlen_probe' /tmp/mlir_vec.mlir
# v0.28 hardware-aware VLEN
grep -q 'func.func @trin_hw_vlen' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_hw_vlen_for_n' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_hw_isa_code' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_mlir_hw_vlen_probe' /tmp/mlir_vec.mlir
# v0.25/v0.27/v0.29 dialect rewrites applied on emit
grep -q 'Patterns (v0.' /tmp/mlir_vec.mlir
grep -q 'rewrite: dialect_ops=' /tmp/mlir_vec.mlir
grep -q 'cse=' /tmp/mlir_vec.mlir
grep -q 'vwidth=' /tmp/mlir_vec.mlir
grep -q 'strength=' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_mlir_dialect_probe' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_mlir_fold_probe' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_mlir_vsum_probe' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_mlir_cse_probe' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_mlir_soft_cse_probe' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_mlir_vwidth_probe' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_mlir_strength_probe' /tmp/mlir_vec.mlir
grep -q 'func.func @trin_mlir_span_alg_probe' /tmp/mlir_vec.mlir
grep -q 'span_alg=' /tmp/mlir_vec.mlir
grep -q 'span_add(x,0) → x\|span_scale(1,x) → x\|sum(0) → 0' /tmp/mlir_vec.mlir
grep -q 'soft_mux(1,pure,each) → pure' /tmp/mlir_vec.mlir
grep -q 'soft_mux(w,x,x) → x' /tmp/mlir_vec.mlir
grep -F -q 'soft_mux(w,pure,0) → w*pure' /tmp/mlir_vec.mlir
grep -q 'rewrite: cse →' /tmp/mlir_vec.mlir
grep -q 'vector_width=16' /tmp/mlir_vec.mlir
# v0.23 full-program main surface on non-host demos
grep -q 'func.func @main()' /tmp/mlir_vec.mlir
./trin run examples/mlir_vec.trin | tee /tmp/trin_mlir_interp.txt
grep -q 'sum4 = 10' /tmp/trin_mlir_interp.txt
grep -q 'dot4 = 30' /tmp/trin_mlir_interp.txt
grep -q 'soft_mux = 4' /tmp/trin_mlir_interp.txt

# MLIR→LLVM lowering (v0.22+) — no mlir-opt
./trin emit-mlir-llvm examples/mlir_vec.trin -o /tmp/mlir_vec_lower.ll
grep -q 'MLIR→LLVM lowering' /tmp/mlir_vec_lower.ll
grep -q 'define float @trin_soft_mux' /tmp/mlir_vec_lower.ll
grep -q 'define float @trin_sum_f32' /tmp/mlir_vec_lower.ll
grep -q 'define <4 x float> @trin_soft_mux_v4' /tmp/mlir_vec_lower.ll
grep -q 'define float @trin_mlir_vec_probe' /tmp/mlir_vec_lower.ll
grep -q 'define float @trin_mlir_soft_probe' /tmp/mlir_vec_lower.ll
# v0.24: <8 x float> tiles + probe
grep -q 'define <8 x float> @trin_soft_mux_v8' /tmp/mlir_vec_lower.ll
grep -q 'define <8 x float> @trin_span_add_v8' /tmp/mlir_vec_lower.ll
grep -q 'define float @trin_sum_v8' /tmp/mlir_vec_lower.ll
grep -q 'define float @trin_mlir_vec8_probe' /tmp/mlir_vec_lower.ll
grep -q 'wider vector tiles' /tmp/mlir_vec_lower.ll
# v0.26: <16 x float> tiles + VLEN
grep -q 'define <16 x float> @trin_soft_mux_v16' /tmp/mlir_vec_lower.ll
grep -q 'define <16 x float> @trin_span_add_v16' /tmp/mlir_vec_lower.ll
grep -q 'define float @trin_sum_v16' /tmp/mlir_vec_lower.ll
grep -q 'define float @trin_mlir_vec16_probe' /tmp/mlir_vec_lower.ll
grep -q 'define i32 @trin_vlen' /tmp/mlir_vec_lower.ll
grep -q 'define float @trin_mlir_vlen_probe' /tmp/mlir_vec_lower.ll
grep -q 'portable VLEN selection' /tmp/mlir_vec_lower.ll
# v0.28/v0.30 hardware VLEN lowers (runtime CPUID dispatch)
grep -q 'define i32 @trin_hw_vlen' /tmp/mlir_vec_lower.ll
grep -q 'define i32 @trin_hw_vlen_for_n' /tmp/mlir_vec_lower.ll
grep -q 'define i32 @trin_hw_isa_code' /tmp/mlir_vec_lower.ll
grep -q 'define float @trin_mlir_hw_vlen_probe' /tmp/mlir_vec_lower.ll
grep -q 'hardware VLEN selection' /tmp/mlir_vec_lower.ll
grep -q 'runtime CPU feature detect (v0.30)' /tmp/mlir_vec_lower.ll
grep -q 'runtime dynamic feature dispatch' /tmp/mlir_vec_lower.ll
grep -q 'call void @trin_hw_detect_once' /tmp/mlir_vec_lower.ll
grep -q '@trin_hw_rt_vlen' /tmp/mlir_vec_lower.ll
grep -q 'cpuid\|@trin_hw_detect_once' /tmp/mlir_vec_lower.ll
# v0.25 dialect probe lowers
grep -q 'define float @trin_mlir_dialect_probe' /tmp/mlir_vec_lower.ll
grep -q 'define float @trin_mlir_fold_probe' /tmp/mlir_vec_lower.ll
grep -q 'define float @trin_mlir_vsum_probe' /tmp/mlir_vec_lower.ll
grep -q 'dialect pattern rewrites' /tmp/mlir_vec_lower.ll
# v0.27 deeper rewrite probe lowers
grep -q 'define float @trin_mlir_cse_probe' /tmp/mlir_vec_lower.ll
grep -q 'define float @trin_mlir_soft_cse_probe' /tmp/mlir_vec_lower.ll
grep -q 'define float @trin_mlir_vwidth_probe' /tmp/mlir_vec_lower.ll
grep -q 'deeper rewrites: CSE' /tmp/mlir_vec_lower.ll
# v0.29 soft_mux strength + v0.32 span/vector algebraic
grep -q 'define float @trin_mlir_strength_probe' /tmp/mlir_vec_lower.ll
grep -q 'define float @trin_mlir_span_alg_probe' /tmp/mlir_vec_lower.ll
grep -q 'span/vector algebraic rewrites' /tmp/mlir_vec_lower.ll
grep -q 'soft_mux strength reduction' /tmp/mlir_vec_lower.ll
grep -q 'ret float 1.350000e+01' /tmp/mlir_vec_lower.ll
# v0.23: executable @main from MLIR path
grep -q 'define i32 @main' /tmp/mlir_vec_lower.ll
grep -q 'full-program MLIR→LLVM' /tmp/mlir_vec_lower.ll
./trin lower-mlir /tmp/mlir_vec.mlir -o /tmp/mlir_vec_from_text.ll
grep -q 'define float @trin_soft_mux' /tmp/mlir_vec_from_text.ll
grep -q 'define <4 x float> @trin_span_add_v4' /tmp/mlir_vec_from_text.ll
grep -q 'define <8 x float> @trin_span_add_v8' /tmp/mlir_vec_from_text.ll
grep -q 'define <16 x float> @trin_span_add_v16' /tmp/mlir_vec_from_text.ll
grep -q 'full-program MLIR @main detected' /tmp/mlir_vec_from_text.ll
# rewrite-mlir CLI on a pre-rewrite dialect snippet
cat > /tmp/mlir_dialect_raw.mlir <<'EOF'
module attributes {trin.version = "0.24"} {
  func.func @trin_soft_mux(%w: f32, %pure: f32, %each: f32) -> f32 {
    %c1 = arith.constant 1.000000e+00 : f32
    %one = arith.subf %c1, %w : f32
    %t0 = arith.mulf %w, %pure : f32
    %t1 = arith.mulf %one, %each : f32
    %y = arith.addf %t0, %t1 : f32
    return %y : f32
  }
  func.func @trin_sum_v4(%a: vector<4xf32>) -> f32 {
    %y = vector.reduction <add>, %a : vector<4xf32> into f32
    return %y : f32
  }
  func.func @p() -> f32 {
    %w = arith.constant 1.000000e+00 : f32
    %p = arith.constant 7.000000e+00 : f32
    %e = arith.constant 9.000000e+00 : f32
    %y = "trin.soft_mux"(%w, %p, %e) : (f32, f32, f32) -> f32
    return %y : f32
  }
}
EOF
./trin rewrite-mlir /tmp/mlir_dialect_raw.mlir -o /tmp/mlir_dialect_rw.mlir
grep -q 'func.call @trin_soft_mux\|rewrite: soft_mux(1,pure,each)' /tmp/mlir_dialect_rw.mlir
grep -q 'rewrite: soft_mux(1,pure,each) → pure' /tmp/mlir_dialect_rw.mlir
grep -q 'trin.version = "0.81"' /tmp/mlir_dialect_rw.mlir
# dedicated full-program MLIR demo
./trin emit-mlir examples/mlir_full_prog.trin -o /tmp/mlir_full_prog.mlir
grep -q 'func.func @main()' /tmp/mlir_full_prog.mlir
grep -q 'full-program MLIR' /tmp/mlir_full_prog.mlir
grep -q 'dyn tile' /tmp/mlir_full_prog.mlir
grep -q 'full-program dyn_tile sum' /tmp/mlir_full_prog.mlir
./trin emit-mlir-llvm examples/mlir_full_prog.trin -o /tmp/mlir_full_prog.ll
grep -q 'define i32 @main' /tmp/mlir_full_prog.ll
grep -q 'trin_println_str' /tmp/mlir_full_prog.ll
grep -q 'define float @trin_dyn_sum4' /tmp/mlir_full_prog.ll
grep -q 'call float @trin_dyn_sum4' /tmp/mlir_full_prog.ll
grep -q 'define i32 @trin_hw_vlen_for_n' /tmp/mlir_full_prog.ll
# v0.32 dyn-tile dedicated demo
./trin emit-llvm examples/mlir_dyn_tile.trin -o /tmp/mlir_dyn_tile.ll
grep -q 'define float @trin_dyn_sum4' /tmp/mlir_dyn_tile.ll
grep -q 'define float @trin_dyn_sum8' /tmp/mlir_dyn_tile.ll
grep -q 'call float @trin_dyn_sum4' /tmp/mlir_dyn_tile.ll
grep -q 'call float @trin_dyn_sum8' /tmp/mlir_dyn_tile.ll
grep -q 'call float @trin_dyn_dot4' /tmp/mlir_dyn_tile.ll
grep -q 'call i32 @trin_dyn_tile' /tmp/mlir_dyn_tile.ll
grep -q 'runtime CPU feature detect' /tmp/mlir_dyn_tile.ll
# 8-lane interpret demo
./trin run examples/mlir_vec8.trin | tee /tmp/trin_mlir_vec8_interp.txt
grep -q 'sum8 = 36' /tmp/trin_mlir_vec8_interp.txt
grep -q 'dot8 = 204' /tmp/trin_mlir_vec8_interp.txt
grep -q 'span_add_sum8 = 72' /tmp/trin_mlir_vec8_interp.txt
grep -q 'scale_sum8 = 72' /tmp/trin_mlir_vec8_interp.txt
# 16-lane interpret demo
./trin run examples/mlir_vec16.trin | tee /tmp/trin_mlir_vec16_interp.txt
grep -q 'sum16 = 136' /tmp/trin_mlir_vec16_interp.txt
grep -q 'dot16 = 1496' /tmp/trin_mlir_vec16_interp.txt
grep -q 'span_add_sum16 = 272' /tmp/trin_mlir_vec16_interp.txt
grep -q 'scale_sum16 = 272' /tmp/trin_mlir_vec16_interp.txt
# rewrite demo interpret
./trin run examples/mlir_rewrite.trin | tee /tmp/trin_mlir_rewrite_interp.txt
grep -q 'soft_mux = 4' /tmp/trin_mlir_rewrite_interp.txt
grep -q 'dialect_probe_target = 4' /tmp/trin_mlir_rewrite_interp.txt
# deeper rewrite demo interpret
./trin run examples/mlir_deep_rewrite.trin | tee /tmp/trin_mlir_deep_rewrite_interp.txt
grep -q 'cse = 6' /tmp/trin_mlir_deep_rewrite_interp.txt
grep -q 'soft_cse = 12' /tmp/trin_mlir_deep_rewrite_interp.txt
grep -q 'vwidth_sum16 = 136' /tmp/trin_mlir_deep_rewrite_interp.txt
# v0.29 soft_mux strength demo interpret
./trin run examples/mlir_strength.trin | tee /tmp/trin_mlir_strength_interp.txt
grep -q 'ident = 5' /tmp/trin_mlir_strength_interp.txt
grep -q 'mul_pure = 3' /tmp/trin_mlir_strength_interp.txt
grep -q 'mul_each = 1.5' /tmp/trin_mlir_strength_interp.txt
grep -q 'full_const = 4' /tmp/trin_mlir_strength_interp.txt
grep -q 'strength_sum = 13.5' /tmp/trin_mlir_strength_interp.txt
if command -v llc >/dev/null 2>&1 && command -v cc >/dev/null 2>&1; then
  # Probe host path: strip program @main from lowered IR for float probes
  # (full-program binary uses the emitted @main directly)
  grep -v 'define i32 @main' /tmp/mlir_vec_lower.ll | \
    sed '/full-program MLIR→LLVM/,/^define i32 @main/d' > /tmp/mlir_vec_probes_only.ll 2>/dev/null || true
  # Rebuild probe-only module via emit of soft_kernel style — simpler: host on soft probes without main
  # Link full-program MLIR→LLVM binary:
  llc -relocation-model=pic -filetype=obj -o /tmp/mlir_full_prog.o /tmp/mlir_full_prog.ll
  cc -O2 -o /tmp/mlir_full_prog /tmp/mlir_full_prog.o -lm
  /tmp/mlir_full_prog | tee /tmp/mlir_full_prog.txt
  grep -q 'product = 12' /tmp/mlir_full_prog.txt
  grep -q 'soft_mux = 4' /tmp/mlir_full_prog.txt
  grep -q 'sum = 10' /tmp/mlir_full_prog.txt
  grep -q 'dot = 11' /tmp/mlir_full_prog.txt
  grep -q 'span_add sum = 10' /tmp/mlir_full_prog.txt
  grep -q 'simd_lane_width = 128' /tmp/mlir_full_prog.txt
  # v0.32 dyn-tile full-program link
  llc -relocation-model=pic -filetype=obj -o /tmp/mlir_dyn_tile.o /tmp/mlir_dyn_tile.ll
  cc -O2 -o /tmp/mlir_dyn_tile /tmp/mlir_dyn_tile.o -lm
  /tmp/mlir_dyn_tile | tee /tmp/mlir_dyn_tile_run.txt
  grep -q 'sum4 = 10' /tmp/mlir_dyn_tile_run.txt
  grep -q 'sum8 = 36' /tmp/mlir_dyn_tile_run.txt
  grep -q 'dot4 = 10' /tmp/mlir_dyn_tile_run.txt
  grep -E -q 'dyn_tile4 = 4' /tmp/mlir_dyn_tile_run.txt
  grep -E -q 'dyn_tile8 = (4|8)' /tmp/mlir_dyn_tile_run.txt
  grep -E -q 'dyn_tile10 = (4|8)' /tmp/mlir_dyn_tile_run.txt
  grep -E -q 'dyn_tile20 = (4|8|16)' /tmp/mlir_dyn_tile_run.txt
  grep -E -q 'hw_vlen = (4|8|16)' /tmp/mlir_dyn_tile_run.txt
  # Also run mlir_vec lowered binary via its @main
  llc -relocation-model=pic -filetype=obj -o /tmp/mlir_vec_lower.o /tmp/mlir_vec_lower.ll
  cc -O2 -o /tmp/mlir_vec_main /tmp/mlir_vec_lower.o -lm
  /tmp/mlir_vec_main | tee /tmp/mlir_vec_main.txt
  grep -q 'sum4 = 10' /tmp/mlir_vec_main.txt
  grep -q 'soft_mux = 4' /tmp/mlir_vec_main.txt
  # float probe symbols still present in probe-only lower from text (no AST main append)
  # Re-lower soft-only via a tiny host object built from lower-mlir text without @main body
  # Keep soft-probe check via interpret + text lower:
  grep -q 'define float @trin_mlir_soft_probe' /tmp/mlir_vec_from_text.ll
  # Run 8/16-lane + dialect + VLEN + CSE MLIR-lowered probes via tiny host (no @main)
  cat > /tmp/mlir_vec8_host.c <<'EOF'
#include <stdio.h>
float trin_mlir_vec8_probe(void);
float trin_mlir_vec_probe(void);
float trin_mlir_vec16_probe(void);
float trin_mlir_vlen_probe(void);
float trin_mlir_hw_vlen_probe(void);
int trin_hw_vlen(void);
int trin_hw_isa_code(void);
float trin_mlir_dialect_probe(void);
float trin_mlir_fold_probe(void);
float trin_mlir_vsum_probe(void);
float trin_mlir_cse_probe(void);
float trin_mlir_soft_cse_probe(void);
float trin_mlir_vwidth_probe(void);
float trin_mlir_strength_probe(void);
float trin_mlir_span_alg_probe(void);
float trin_mlir_bitplane_probe(void);
int main(void) {
  printf("mlir_vec_probe = %.1f\n", trin_mlir_vec_probe());
  printf("mlir_vec8_probe = %.1f\n", trin_mlir_vec8_probe());
  printf("mlir_vec16_probe = %.1f\n", trin_mlir_vec16_probe());
  printf("mlir_vlen_probe = %.1f\n", trin_mlir_vlen_probe());
  printf("mlir_hw_vlen_probe = %.1f\n", trin_mlir_hw_vlen_probe());
  printf("hw_vlen = %d\n", trin_hw_vlen());
  printf("hw_isa_code = %d\n", trin_hw_isa_code());
  printf("mlir_dialect_probe = %.1f\n", trin_mlir_dialect_probe());
  printf("mlir_fold_probe = %.1f\n", trin_mlir_fold_probe());
  printf("mlir_vsum_probe = %.1f\n", trin_mlir_vsum_probe());
  printf("mlir_cse_probe = %.1f\n", trin_mlir_cse_probe());
  printf("mlir_soft_cse_probe = %.1f\n", trin_mlir_soft_cse_probe());
  printf("mlir_vwidth_probe = %.1f\n", trin_mlir_vwidth_probe());
  printf("mlir_strength_probe = %.1f\n", trin_mlir_strength_probe());
  printf("mlir_span_alg_probe = %.1f\n", trin_mlir_span_alg_probe());
  printf("mlir_bitplane_probe = %.1f\n", trin_mlir_bitplane_probe());
  return 0;
}
EOF
  # strip full-program main from lowered IR for probe host
  awk '/; --- full-program MLIR→LLVM @main/{exit} {print}' /tmp/mlir_vec_lower.ll > /tmp/mlir_vec_nomenu.ll
  # also drop any leftover define i32 @main block if present without marker
  if grep -q 'define i32 @main' /tmp/mlir_vec_nomenu.ll; then
    awk '/define i32 @main/{flag=1} flag && /^}$/{flag=0; next} !flag' /tmp/mlir_vec_nomenu.ll > /tmp/mlir_vec_probes.ll
  else
    cp /tmp/mlir_vec_nomenu.ll /tmp/mlir_vec_probes.ll
  fi
  llc -relocation-model=pic -filetype=obj -o /tmp/mlir_vec_probes.o /tmp/mlir_vec_probes.ll
  cc -O2 -o /tmp/mlir_vec8_probes /tmp/mlir_vec8_host.c /tmp/mlir_vec_probes.o -lm
  /tmp/mlir_vec8_probes | tee /tmp/mlir_vec8_probes.txt
  grep -q 'mlir_vec_probe = 10.0' /tmp/mlir_vec8_probes.txt
  grep -q 'mlir_vec8_probe = 36.0' /tmp/mlir_vec8_probes.txt
  grep -q 'mlir_vec16_probe = 136.0' /tmp/mlir_vec8_probes.txt
  grep -q 'mlir_vlen_probe = 16.0' /tmp/mlir_vec8_probes.txt
  # hardware probe returns host hw_vlen as float (AVX host → 8.0)
  grep -E -q 'mlir_hw_vlen_probe = (4|8|16)\.0' /tmp/mlir_vec8_probes.txt
  grep -E -q 'hw_vlen = (4|8|16)' /tmp/mlir_vec8_probes.txt
  grep -E -q 'hw_isa_code = [0-5]' /tmp/mlir_vec8_probes.txt
  # consistency: probe float equals integer hw_vlen
  python3 - <<'PY'
import re
t=open("/tmp/mlir_vec8_probes.txt").read()
m1=re.search(r"mlir_hw_vlen_probe = ([0-9.]+)", t)
m2=re.search(r"hw_vlen = (\d+)", t)
assert m1 and m2, t
assert float(m1.group(1)) == float(m2.group(1)), (m1.group(1), m2.group(1))
print("hw_vlen consistency OK")
PY
  grep -q 'mlir_dialect_probe = 4.0' /tmp/mlir_vec8_probes.txt
  grep -q 'mlir_fold_probe = 7.0' /tmp/mlir_vec8_probes.txt
  grep -q 'mlir_vsum_probe = 10.0' /tmp/mlir_vec8_probes.txt
  grep -q 'mlir_cse_probe = 6.0' /tmp/mlir_vec8_probes.txt
  grep -q 'mlir_soft_cse_probe = 12.0' /tmp/mlir_vec8_probes.txt
  grep -q 'mlir_vwidth_probe = 136.0' /tmp/mlir_vec8_probes.txt
  grep -q 'mlir_strength_probe = 13.5' /tmp/mlir_vec8_probes.txt
  grep -q 'mlir_span_alg_probe = 30.0' /tmp/mlir_vec8_probes.txt
  grep -q 'mlir_bitplane_probe = 42.0' /tmp/mlir_vec8_probes.txt
fi

# hardware VLEN demo interpret
./trin run examples/mlir_hw_vlen.trin | tee /tmp/trin_mlir_hw_vlen_interp.txt
grep -q 'sum8 = 36' /tmp/trin_mlir_hw_vlen_interp.txt
grep -q 'mlir_vlen_probe_target = 16' /tmp/trin_mlir_hw_vlen_interp.txt
grep -q 'runtime_cpuid_dispatch' /tmp/trin_mlir_hw_vlen_interp.txt
# runtime dispatch demo interpret
./trin run examples/mlir_runtime_dispatch.trin | tee /tmp/trin_mlir_rt_dispatch_interp.txt
grep -q 'sum8 = 36' /tmp/trin_mlir_rt_dispatch_interp.txt
grep -q 'mlir_hw_dispatch = runtime' /tmp/trin_mlir_rt_dispatch_interp.txt
# span/vector algebraic demo interpret
./trin run examples/mlir_span_alg.trin | tee /tmp/trin_mlir_span_alg_interp.txt
grep -q 'span_alg_sum = 30' /tmp/trin_mlir_span_alg_interp.txt
grep -q 'span_alg_probe_target = 30' /tmp/trin_mlir_span_alg_interp.txt
# dyn-tile demo interpret
./trin run examples/mlir_dyn_tile.trin | tee /tmp/trin_mlir_dyn_tile_interp.txt
grep -q 'sum4 = 10' /tmp/trin_mlir_dyn_tile_interp.txt
grep -q 'sum8 = 36' /tmp/trin_mlir_dyn_tile_interp.txt
grep -q 'dot4 = 10' /tmp/trin_mlir_dyn_tile_interp.txt
grep -q 'dyn_tile4 = 4' /tmp/trin_mlir_dyn_tile_interp.txt
grep -q 'dyn_tile8 = 8' /tmp/trin_mlir_dyn_tile_interp.txt
grep -q 'dyn_tile10 = 8' /tmp/trin_mlir_dyn_tile_interp.txt
grep -q 'dyn_tile20 = 16' /tmp/trin_mlir_dyn_tile_interp.txt
grep -q 'mlir_dyn_tile_note = runtime_hw_vlen_for_n' /tmp/trin_mlir_dyn_tile_interp.txt
# bitplane + soft_mux_v algebraic demo interpret
./trin run examples/mlir_bitplane_alg.trin | tee /tmp/trin_mlir_bitplane_alg_interp.txt
grep -q 'plane_sum = 34' /tmp/trin_mlir_bitplane_alg_interp.txt
grep -q 'bitplane_alg_sum = 42' /tmp/trin_mlir_bitplane_alg_interp.txt
grep -q 'bitplane_alg_probe_target = 42' /tmp/trin_mlir_bitplane_alg_interp.txt
grep -q 'mlir_bitplane_alg_note = rewrite_soft_mux_v_and_plane_identities' /tmp/trin_mlir_bitplane_alg_interp.txt
# rewrite surface carries bitplane folds
./trin emit-mlir examples/mlir_vec.trin -o /tmp/mlir_bitplane_check.mlir
grep -q 'bitplane=' /tmp/mlir_bitplane_check.mlir
grep -q 'plane_and(x,x)' /tmp/mlir_bitplane_check.mlir
grep -q 'func.func @trin_mlir_bitplane_probe' /tmp/mlir_bitplane_check.mlir

# residual peel demo interpret + llvm path (v0.34)
./trin run examples/mlir_residual_peel.trin | tee /tmp/trin_mlir_residual_peel_interp.txt
grep -q 'sum5 = 15' /tmp/trin_mlir_residual_peel_interp.txt
grep -q 'sum6 = 21' /tmp/trin_mlir_residual_peel_interp.txt
grep -q 'sum10 = 55' /tmp/trin_mlir_residual_peel_interp.txt
grep -q 'sum20 = 210' /tmp/trin_mlir_residual_peel_interp.txt
grep -q 'dot5 = 15' /tmp/trin_mlir_residual_peel_interp.txt
grep -q 'residual_peel_probe_target = sum5:15' /tmp/trin_mlir_residual_peel_interp.txt
grep -q 'mlir_residual_peel_note = tile_plus_scalar_residual' /tmp/trin_mlir_residual_peel_interp.txt
# full-program llvm residual peel present
./trin emit-llvm examples/mlir_residual_peel.trin -o /tmp/mlir_residual_peel.ll
grep -q 'residual peel sum(n=5)' /tmp/mlir_residual_peel.ll
grep -q 'residual peel sum(n=10)' /tmp/mlir_residual_peel.ll
grep -q 'AdaptivePeelChunks' /tmp/mlir_residual_peel.ll
grep -q 'call float @trin_dyn_sum4' /tmp/mlir_residual_peel.ll
# host ISA may schedule only 4/8 (AVX) or include 16/32 — accept any dyn tile peel helper
grep -E -q 'call float @trin_dyn_sum(8|16)' /tmp/mlir_residual_peel.ll
# run residual peel via llvm link if tools present
if command -v llc >/dev/null 2>&1; then
  llc -relocation-model=pic -filetype=obj -o /tmp/mlir_residual_peel.o /tmp/mlir_residual_peel.ll
  cc -O2 -o /tmp/mlir_residual_peel_bin /tmp/mlir_residual_peel.o -lm
  /tmp/mlir_residual_peel_bin | tee /tmp/mlir_residual_peel_run.txt
  grep -q 'sum5 = 15' /tmp/mlir_residual_peel_run.txt
  grep -q 'sum10 = 55' /tmp/mlir_residual_peel_run.txt
  grep -q 'sum20 = 210' /tmp/mlir_residual_peel_run.txt
  grep -q 'dot5 = 15' /tmp/mlir_residual_peel_run.txt
fi
# mlir notes residual peel
./trin emit-mlir examples/mlir_residual_peel.trin -o /tmp/mlir_residual_peel.mlir
grep -q 'residual peel sum(n=5)' /tmp/mlir_residual_peel.mlir
grep -q 'trin.residual_peel = "auto"' /tmp/mlir_residual_peel.mlir

# nested soft_mux / bitplane product folds (v0.36)
./trin run examples/mlir_nested_fold.trin | tee /tmp/trin_mlir_nested_interp.txt
grep -q 'soft_product_pure = 4' /tmp/trin_mlir_nested_interp.txt
grep -q 'soft_product_each = 8' /tmp/trin_mlir_nested_interp.txt
grep -q 'soft_product_pure2 = 4' /tmp/trin_mlir_nested_interp.txt
grep -q 'soft_sum = 16' /tmp/trin_mlir_nested_interp.txt
grep -q 'plane_sum = 36' /tmp/trin_mlir_nested_interp.txt
grep -q 'nested_fold_sum = 52' /tmp/trin_mlir_nested_interp.txt
grep -q 'nested_fold_probe_target = 52' /tmp/trin_mlir_nested_interp.txt
grep -q 'mlir_nested_fold_note = soft_mux_product_and_plane_absorb' /tmp/trin_mlir_nested_interp.txt
./trin emit-mlir examples/soft_kernel_stub.trin -o /tmp/nested_probe_check.mlir 2>/dev/null || true
# soft_kernel probe emission via soft_kernel source
cat > /tmp/nested_soft_kernel.trin <<'EOF'
fn soft_kernel(w, a, x, b) {
    soft_mux(w, a * x, b)
}
EOF
./trin emit-mlir /tmp/nested_soft_kernel.trin -o /tmp/nested_probe_check.mlir
grep -q 'func.func @trin_mlir_nested_probe' /tmp/nested_probe_check.mlir
grep -q 'nested=' /tmp/nested_probe_check.mlir
grep -q 'soft_mux(w1, soft_mux(w2,p,e), e)' /tmp/nested_probe_check.mlir
grep -q 'soft_mux(w1, p, soft_mux(w2,p,e))' /tmp/nested_probe_check.mlir
grep -q 'plane_and(and(a,b)' /tmp/nested_probe_check.mlir
./trin emit-mlir-llvm /tmp/nested_soft_kernel.trin -o /tmp/nested_probe_check.ll
grep -q 'define float @trin_mlir_nested_probe' /tmp/nested_probe_check.ll
grep -q 'ret float 5.200000e+01' /tmp/nested_probe_check.ll

# auto residual peel pointer kernels (v0.36)
./trin run examples/mlir_auto_peel.trin | tee /tmp/trin_mlir_auto_peel_interp.txt
grep -q 'sum5 = 15' /tmp/trin_mlir_auto_peel_interp.txt
grep -q 'sum10 = 55' /tmp/trin_mlir_auto_peel_interp.txt
grep -q 'sum20 = 210' /tmp/trin_mlir_auto_peel_interp.txt
grep -q 'dot10 = 55' /tmp/trin_mlir_auto_peel_interp.txt
grep -q 'auto_peel_probe_target = sum5:15 sum10:55 sum20:210 dot10:55' /tmp/trin_mlir_auto_peel_interp.txt
grep -q 'mlir_auto_peel_note = pointer_counted_16_8_4_residual' /tmp/trin_mlir_auto_peel_interp.txt
./trin emit-llvm examples/llvm_vec_loops.trin -o /tmp/auto_peel_check.ll
grep -q 'auto residual peel span loops (v0.36–v0.67)' /tmp/auto_peel_check.ll
grep -q 'define float @trin_auto_peel_probe' /tmp/auto_peel_check.ll
grep -q 'peel16.cond' /tmp/auto_peel_check.ll
grep -q 'peel8.cond' /tmp/auto_peel_check.ll
grep -q 'peel4.cond' /tmp/auto_peel_check.ll
# soft_kernel path also lowers peeled loops
cat > /tmp/auto_peel_soft.trin <<'EOF'
fn soft_kernel(w, a, x, b) {
    soft_mux(w, a * x, b)
}
EOF
./trin emit-mlir /tmp/auto_peel_soft.trin -o /tmp/auto_peel_soft.mlir
grep -q 'trin.auto_peel = "pointer_store_map_scan_mul_multi_tuple"' /tmp/auto_peel_soft.mlir
grep -q 'trin.residual_peel = "auto"' /tmp/auto_peel_soft.mlir
./trin emit-mlir-llvm /tmp/auto_peel_soft.trin -o /tmp/auto_peel_soft.ll
grep -q 'auto residual peel' /tmp/auto_peel_soft.ll
grep -q 'define float @trin_sum_f32' /tmp/auto_peel_soft.ll
grep -q 'peel16.cond' /tmp/auto_peel_soft.ll

# store peels for span_add/scale (v0.37)
./trin run examples/mlir_store_peel.trin | tee /tmp/trin_mlir_store_peel_interp.txt
grep -q 'sa10 = 110' /tmp/trin_mlir_store_peel_interp.txt
grep -q 'sc10 = 110' /tmp/trin_mlir_store_peel_interp.txt
grep -q 'sa5 = 30' /tmp/trin_mlir_store_peel_interp.txt
grep -q 'sc20 = 630' /tmp/trin_mlir_store_peel_interp.txt
grep -q 'store_peel_probe_target = sa10:110 sc10:110 sa5:30 sc20:630 sum220:220' /tmp/trin_mlir_store_peel_interp.txt
grep -q 'mlir_store_peel_note = span_add_scale_16_8_4_residual' /tmp/trin_mlir_store_peel_interp.txt
./trin emit-llvm examples/llvm_vec_loops.trin -o /tmp/store_peel_check.ll
grep -q 'auto residual peel span loops (v0.36–v0.67)' /tmp/store_peel_check.ll
grep -q 'define float @trin_store_peel_probe' /tmp/store_peel_check.ll
grep -q 'define void @trin_span_add_f32' /tmp/store_peel_check.ll
grep -q 'define void @trin_span_scale_f32' /tmp/store_peel_check.ll
# span_add / scale bodies have peel labels + vector stores
grep -q 'store <16 x float>' /tmp/store_peel_check.ll
grep -q 'fmul <16 x float>' /tmp/store_peel_check.ll
./trin emit-mlir examples/soft_mux.trin -o /tmp/store_peel_soft.mlir 2>/dev/null || true
cat > /tmp/store_peel_soft.trin <<'EOF'
fn soft_kernel(w, a, x, b) {
    soft_mux(w, a * x, b)
}
EOF
./trin emit-mlir /tmp/store_peel_soft.trin -o /tmp/store_peel_soft.mlir
grep -q 'trin.auto_peel = "pointer_store_map_scan_mul_multi_tuple"' /tmp/store_peel_soft.mlir
grep -q 'Store peels' /tmp/store_peel_soft.mlir
./trin emit-mlir-llvm /tmp/store_peel_soft.trin -o /tmp/store_peel_soft.ll
grep -q 'define float @trin_store_peel_probe' /tmp/store_peel_soft.ll
grep -q 'store <8 x float>' /tmp/store_peel_soft.ll
# link store peel probe when llc available
if command -v llc >/dev/null 2>&1 && command -v cc >/dev/null 2>&1; then
  ./trin link-llvm examples/llvm_vec_loops.trin -o /tmp/store_peel_bin --entry trin_store_peel_probe
  /tmp/store_peel_bin | tee /tmp/store_peel_run.txt
  grep -q 'llvm entry trin_store_peel_probe = 220.0' /tmp/store_peel_run.txt
fi

# multi-level soft_mux trees + multi-word plane GEMM (v0.38)
./trin run examples/mlir_tree_fold.trin | tee /tmp/trin_mlir_tree_fold_interp.txt
grep -q 'tree = 5' /tmp/trin_mlir_tree_fold_interp.txt
grep -q 'triple = 3' /tmp/trin_mlir_tree_fold_interp.txt
grep -q 'tree2 = 8' /tmp/trin_mlir_tree_fold_interp.txt
grep -q 'soft = 16' /tmp/trin_mlir_tree_fold_interp.txt
grep -q 'tree_probe_target = soft:16 plane:48 total:64' /tmp/trin_mlir_tree_fold_interp.txt
grep -q 'mlir_tree_fold_note = multi_level_soft_mux_multi_word_gemm' /tmp/trin_mlir_tree_fold_interp.txt
cat > /tmp/tree_soft_kernel.trin <<'EOF'
fn soft_kernel(w, a, x, b) {
    soft_mux(w, a * x, b)
}
EOF
./trin emit-mlir /tmp/tree_soft_kernel.trin -o /tmp/tree_probe_check.mlir
grep -q 'func.func @trin_mlir_tree_probe' /tmp/tree_probe_check.mlir
grep -q 'soft_mux(w, soft_mux(w2,p,e), soft_mux(w3,p,e))' /tmp/tree_probe_check.mlir
grep -q 'trit_dot_planes_word' /tmp/tree_probe_check.mlir
grep -q 'nested=' /tmp/tree_probe_check.mlir
./trin emit-mlir-llvm /tmp/tree_soft_kernel.trin -o /tmp/tree_probe_check.ll
grep -q 'define float @trin_mlir_tree_probe' /tmp/tree_probe_check.ll
grep -q 'ret float 6.400000e+01' /tmp/tree_probe_check.ll
# rewrite isolated both-arms / gemm folds
cat > /tmp/tree_rewrite_snippet.mlir <<'EOF'
module attributes {trin.version = "0.37"} {
  func.func @trin_soft_mux(%w: f32, %p: f32, %e: f32) -> f32 { return %p : f32 }
  func.func @trin_trit_dot_planes_word(%ap: i32, %an: i32, %bp: i32, %bn: i32) -> i32 { return %ap : i32 }
  func.func @probe() -> f32 {
    %w = arith.constant 5.000000e-01 : f32
    %w2 = arith.constant 5.000000e-01 : f32
    %w3 = arith.constant 2.500000e-01 : f32
    %p0 = arith.constant 1.000000e+01 : f32
    %e0 = arith.constant 2.000000e+00 : f32
    %z = arith.constant 0.000000e+00 : f32
    %p = arith.addf %p0, %z : f32
    %e = arith.addf %e0, %z : f32
    %in_p = func.call @trin_soft_mux(%w2, %p, %e) : (f32, f32, f32) -> f32
    %in_e = func.call @trin_soft_mux(%w3, %p, %e) : (f32, f32, f32) -> f32
    %y = func.call @trin_soft_mux(%w, %in_p, %in_e) : (f32, f32, f32) -> f32
    %ap = arith.constant 16777215 : i32
    %an = arith.constant 0 : i32
    %bp = arith.constant 16777215 : i32
    %bn = arith.constant 0 : i32
    %zi = arith.constant 0 : i32
    %d0 = func.call @trin_trit_dot_planes_word(%ap, %an, %bp, %bn) : (i32, i32, i32, i32) -> i32
    %d1 = func.call @trin_trit_dot_planes_word(%zi, %zi, %bp, %bn) : (i32, i32, i32, i32) -> i32
    return %y : f32
  }
}
EOF
./trin rewrite-mlir /tmp/tree_rewrite_snippet.mlir -o /tmp/tree_rewrite_out.mlir
grep -q 'soft_mux(w, soft_mux(w2,p,e), soft_mux(w3,p,e))' /tmp/tree_rewrite_out.mlir
grep -q 'trit_dot_planes_word(const) fold' /tmp/tree_rewrite_out.mlir
grep -q 'trit_dot_planes_word(zero)' /tmp/tree_rewrite_out.mlir

# adaptive tile from runtime ISA (v0.39)
./trin run examples/mlir_adaptive_tile.trin | tee /tmp/trin_adaptive_tile_interp.txt
grep -q 'adaptive_tile_note = runtime_isa' /tmp/trin_adaptive_tile_interp.txt
grep -q 'adaptive_tile_probe_target = 100' /tmp/trin_adaptive_tile_interp.txt
grep -q 'adaptive_tile_table = portable:16 sse:4 avx:8 avx512:16 sve:32 neon:4' /tmp/trin_adaptive_tile_interp.txt
cat > /tmp/adaptive_soft_kernel.trin <<'EOF'
fn soft_kernel(w, a, x, b) {
    soft_mux(w, a * x, b)
}
EOF
./trin emit-mlir /tmp/adaptive_soft_kernel.trin -o /tmp/adaptive_tile_check.mlir
grep -q 'func.func @trin_mlir_adaptive_tile_probe' /tmp/adaptive_tile_check.mlir
grep -q 'func.func @trin_adaptive_tile' /tmp/adaptive_tile_check.mlir
grep -q 'func.func @trin_adaptive_tile_max' /tmp/adaptive_tile_check.mlir
grep -q 'func.func @trin_adaptive_isa_tile' /tmp/adaptive_tile_check.mlir
grep -q 'trin.adaptive_tile = "runtime_isa"' /tmp/adaptive_tile_check.mlir
./trin emit-mlir-llvm /tmp/adaptive_soft_kernel.trin -o /tmp/adaptive_tile_check.ll
grep -q 'define float @trin_mlir_adaptive_tile_probe' /tmp/adaptive_tile_check.ll
grep -q 'ret float 1.000000e+02' /tmp/adaptive_tile_check.ll
grep -q 'define i32 @trin_adaptive_tile' /tmp/adaptive_tile_check.ll
grep -q 'define i32 @trin_adaptive_isa_tile' /tmp/adaptive_tile_check.ll
grep -q 'runtime feature dispatch (v0.30/v0.39 adaptive)' /tmp/adaptive_tile_check.ll


# soft_mux forest + map/store peels (v0.40)
./trin run examples/mlir_forest_fold.trin | tee /tmp/trin_forest_interp.txt
grep -q 'forest_fold_note = soft_mux_forest_gemm_schedule' /tmp/trin_forest_interp.txt
grep -q 'forest_probe_target = 80' /tmp/trin_forest_interp.txt
grep -q 'map_store_peel_probe_target = 550' /tmp/trin_forest_interp.txt
cat > /tmp/forest_soft_kernel.trin <<'EOF'
fn soft_kernel(w, a, x, b) {
    soft_mux(w, a * x, b)
}
EOF
./trin emit-mlir /tmp/forest_soft_kernel.trin -o /tmp/forest_check.mlir
grep -q 'func.func @trin_mlir_forest_probe' /tmp/forest_check.mlir
grep -q 'func.func @trin_span_mul_f32' /tmp/forest_check.mlir
grep -q 'func.func @trin_span_axpy_f32' /tmp/forest_check.mlir
grep -q 'trin.map_store_peel = "mul_axpy"' /tmp/forest_check.mlir
grep -q 'trin.soft_forest = "deep_schedule"' /tmp/forest_check.mlir
grep -q 'trin.auto_peel = "pointer_store_map_scan_mul_multi_tuple"' /tmp/forest_check.mlir
./trin emit-mlir-llvm /tmp/forest_soft_kernel.trin -o /tmp/forest_check.ll
grep -q 'define float @trin_mlir_forest_probe' /tmp/forest_check.ll
grep -q 'ret float 8.000000e+01' /tmp/forest_check.ll
grep -q 'define void @trin_span_mul_f32' /tmp/forest_check.ll
grep -q 'define void @trin_span_axpy_f32' /tmp/forest_check.ll
grep -q 'define float @trin_map_store_peel_probe' /tmp/forest_check.ll
grep -q 'map/store peel probe' /tmp/forest_check.ll
# rewrite forest fold
cat > /tmp/forest_rewrite_snippet.mlir <<'EOF'
module attributes {trin.version = "0.39"} {
  func.func @trin_soft_mux(%w: f32, %p: f32, %e: f32) -> f32 { return %p : f32 }
  func.func @trin_trit_dot_planes_word(%ap: i32, %an: i32, %bp: i32, %bn: i32) -> i32 { return %ap : i32 }
  func.func @trin_trit_dot_planes_x4(%a0: i32, %a1: i32, %a2: i32, %a3: i32, %a4: i32, %a5: i32, %a6: i32, %a7: i32, %a8: i32, %a9: i32, %a10: i32, %a11: i32, %a12: i32, %a13: i32, %a14: i32, %a15: i32) -> i32 { return %a0 : i32 }
  func.func @probe() -> f32 {
    %w = arith.constant 5.000000e-01 : f32
    %w2 = arith.constant 5.000000e-01 : f32
    %w3 = arith.constant 5.000000e-01 : f32
    %w4 = arith.constant 2.500000e-01 : f32
    %p = arith.constant 1.000000e+01 : f32
    %e = arith.constant 2.000000e+00 : f32
    %leaf = func.call @trin_soft_mux(%w3, %p, %e) : (f32, f32, f32) -> f32
    %mid = func.call @trin_soft_mux(%w2, %leaf, %e) : (f32, f32, f32) -> f32
    %arm = func.call @trin_soft_mux(%w4, %p, %e) : (f32, f32, f32) -> f32
    %y = func.call @trin_soft_mux(%w, %mid, %arm) : (f32, f32, f32) -> f32
    return %y : f32
  }
}
EOF
./trin rewrite-mlir /tmp/forest_rewrite_snippet.mlir -o /tmp/forest_rewrite_out.mlir
# Forest may fold via dedicated forest note or stepwise product/tree const-fold to 4.0
grep -q '4.000000e+00' /tmp/forest_rewrite_out.mlir
grep -Eq 'soft_mux forest product both-arms|soft_mux\(w, soft_mux\(w2,p,e\), soft_mux\(w3,p,e\)\)|soft_mux\(w1, soft_mux\(w2,p,e\), e\)' /tmp/forest_rewrite_out.mlir


# scan/reduce peels + adaptive ISA ladder + deep forest (v0.41)
./trin run examples/mlir_scan_reduce.trin | tee /tmp/trin_scan_reduce_interp.txt
grep -q 'scan_reduce_note = prefix_scan_add_reduce_adaptive_ladder' /tmp/trin_scan_reduce_interp.txt
grep -q 'scan_reduce_probe_target = 275' /tmp/trin_scan_reduce_interp.txt
grep -q 'deep_forest_probe_target = 90' /tmp/trin_scan_reduce_interp.txt
cat > /tmp/scan_soft_kernel.trin <<'EOF'
fn soft_kernel(w, a, x, b) {
    soft_mux(w, a * x, b)
}
EOF
./trin emit-mlir /tmp/scan_soft_kernel.trin -o /tmp/scan_check.mlir
grep -q 'func.func @trin_span_scan_add_f32' /tmp/scan_check.mlir
grep -q 'func.func @trin_reduce_add_f32' /tmp/scan_check.mlir
grep -q 'func.func @trin_adaptive_peel_tile' /tmp/scan_check.mlir
grep -q 'func.func @trin_mlir_deep_forest_probe' /tmp/scan_check.mlir
grep -q 'trin.scan_reduce_peel = "prefix_add_mul_multi_tuple"' /tmp/scan_check.mlir
grep -q 'trin.adaptive_peel = "isa_ladder_64"' /tmp/scan_check.mlir
grep -q 'trin.soft_forest = "deep_schedule"' /tmp/scan_check.mlir
./trin emit-mlir-llvm /tmp/scan_soft_kernel.trin -o /tmp/scan_check.ll
grep -q 'define float @trin_mlir_deep_forest_probe' /tmp/scan_check.ll
grep -q 'ret float 9.000000e+01' /tmp/scan_check.ll
grep -q 'define void @trin_span_scan_add_f32' /tmp/scan_check.ll
grep -q 'define float @trin_reduce_add_f32' /tmp/scan_check.ll
grep -q 'define float @trin_scan_reduce_peel_probe' /tmp/scan_check.ll
grep -q 'define i32 @trin_adaptive_peel_tile' /tmp/scan_check.ll
# deep forest rewrite
cat > /tmp/deep_forest_rewrite.mlir <<'EOF'
module attributes {trin.version = "0.40"} {
  func.func @trin_soft_mux(%w: f32, %p: f32, %e: f32) -> f32 { return %p : f32 }
  func.func @probe() -> f32 {
    %w = arith.constant 5.000000e-01 : f32
    %w2 = arith.constant 5.000000e-01 : f32
    %w3 = arith.constant 5.000000e-01 : f32
    %w4 = arith.constant 5.000000e-01 : f32
    %w5 = arith.constant 2.500000e-01 : f32
    %p = arith.constant 2.000000e+01 : f32
    %e = arith.constant 4.000000e+00 : f32
    %leaf = func.call @trin_soft_mux(%w4, %p, %e) : (f32, f32, f32) -> f32
    %m1 = func.call @trin_soft_mux(%w3, %leaf, %e) : (f32, f32, f32) -> f32
    %m2 = func.call @trin_soft_mux(%w2, %m1, %e) : (f32, f32, f32) -> f32
    %arm = func.call @trin_soft_mux(%w5, %p, %e) : (f32, f32, f32) -> f32
    %y = func.call @trin_soft_mux(%w, %m2, %arm) : (f32, f32, f32) -> f32
    return %y : f32
  }
}
EOF
./trin rewrite-mlir /tmp/deep_forest_rewrite.mlir -o /tmp/deep_forest_out.mlir
grep -q '7.000000e+00' /tmp/deep_forest_out.mlir
grep -q 'trin.version = "0.81"' /tmp/deep_forest_out.mlir


# scan-mul/reduce-mul + AdaptivePeelChunks + wide ladder (v0.49)
./trin run examples/mlir_scan_mul.trin | tee /tmp/trin_scan_mul_interp.txt
grep -q 'scan_mul_note = prefix_product_scan_reduce_mul' /tmp/trin_scan_mul_interp.txt
grep -q 'scan_mul_probe_target = 64' /tmp/trin_scan_mul_interp.txt
grep -q 'wide_peel_ladder = sve64' /tmp/trin_scan_mul_interp.txt
cat > /tmp/scan_mul_soft_kernel.trin <<'EOF'
fn soft_kernel(w, a, x, b) {
    soft_mux(w, a * x, b)
}
EOF
./trin emit-mlir /tmp/scan_mul_soft_kernel.trin -o /tmp/scan_mul_check.mlir
grep -q 'func.func @trin_span_scan_mul_f32' /tmp/scan_mul_check.mlir
grep -q 'func.func @trin_reduce_mul_f32' /tmp/scan_mul_check.mlir
grep -q 'trin.scan_reduce_peel = "prefix_add_mul_multi_tuple"' /tmp/scan_mul_check.mlir
grep -q 'trin.adaptive_peel = "isa_ladder_64"' /tmp/scan_mul_check.mlir
grep -q 'trin.wide_peel = "sve64"' /tmp/scan_mul_check.mlir
grep -q 'trin.auto_peel = "pointer_store_map_scan_mul_multi_tuple"' /tmp/scan_mul_check.mlir
./trin emit-mlir-llvm /tmp/scan_mul_soft_kernel.trin -o /tmp/scan_mul_check.ll
grep -q 'define void @trin_span_scan_mul_f32' /tmp/scan_mul_check.ll
grep -q 'define float @trin_reduce_mul_f32' /tmp/scan_mul_check.ll
grep -q 'define float @trin_scan_mul_reduce_peel_probe' /tmp/scan_mul_check.ll
grep -q 'define float @trin_wide_peel_ladder_probe' /tmp/scan_mul_check.ll
grep -q 'peel32.cond' /tmp/scan_mul_check.ll
# full-program AdaptivePeelChunks residual notes
cat > /tmp/peel_chunks_main.trin <<'EOF'
fn main() {
    let s10 = sum([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    println(str_concat("sum10 = ", to_str(s10)));
}
EOF
./trin emit-llvm /tmp/peel_chunks_main.trin -o /tmp/peel_chunks.ll
grep -q 'AdaptivePeelChunks' /tmp/peel_chunks.ll
grep -q 'residual peel sum(n=10)' /tmp/peel_chunks.ll


# fused scan-reduce + dyn_sum32 + VL>32 (v0.49)
./trin run examples/mlir_fused_scan_reduce.trin | tee /tmp/trin_fused_scan_interp.txt
grep -q 'fused_scan_reduce_note = scan_reduce_add_mul_multi_tuple' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_scan_reduce_probe_target = 275' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_scan_reduce_multi_probe_target = 330' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_scan_reduce_mul_probe_target = 64' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_tuple_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_sum_scan_probe_target = 275' /tmp/trin_fused_scan_interp.txt
grep -q 'pair = nested_band_f32' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'lang_pair_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_destructure_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_return_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_mul_sum_probe_target = 440' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_nested_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_match_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_cost_probe_target = 10' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_cost_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_of_pairs_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_exhaust_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_cost_weighted_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_select_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_outer_acc_kernel_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_select_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_wildcard_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_guard_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_cost_calibrated_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_plan_log_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_outer_acc_true_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_plan_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_or_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_stream_store_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_weight_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_feedback_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_select_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_pgo_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_tag_exhaust_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_adapt_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_counter_bump_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_adapt_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mv_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_isa_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_isa_loop_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_isa_loop_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_multi_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_tile_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_isa_feedback_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_isa_live_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_isa_live_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_recur_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_xkernel_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_xkernel_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_xkernel_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_ser_fold_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_peel_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_xpass_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_xpass_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_xpass_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_forest_fold_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_auto_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_window_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_window_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_window_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_keyed_reduce_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_vlfb_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_decay_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_decay_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_decay_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_multi_reduce_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_rank_reduce_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_weight_reduce_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_thresh_reduce_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_soft_reduce_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_cal_reduce_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_adapt_reduce_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_blend_reduce_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_gate_reduce_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_score_reduce_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_softfloor_reduce_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_hardgate_reduce_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_dual_reduce_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_mix_reduce_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_credit_reduce_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_margin_reduce_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_band_reduce_probe_target = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_vllive_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_vlstream_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_vlcost_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_vladapt_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_vlsched_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_vlfuse_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_vlmix_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_vlbal_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_vlbw_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_vlpath_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_vllane_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_vlspill_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_vlocc_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_vlqueue_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_vlwater_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_vldrain_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_scatter_mt_vlrefill_probe_target = 3025' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_ema_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_adapt_hl_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_reg_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_floor_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_afloor_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_burst_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_cool_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_credit_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_debt_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_interest_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_amortize_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_reweight_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_budget_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_softcap_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_dual_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_conf_counter_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_ema_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_adapt_hl_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_reg_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_floor_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_afloor_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_burst_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_cool_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_credit_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_debt_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_interest_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_amortize_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_reweight_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_budget_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_softcap_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_dual_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_conf_probe_target = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_ema_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_adapt_hl_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_reg_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_reweight_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_budget_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_softcap_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_dual_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'fused_map_reduce_conf_probe_target = 495' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_keyed_fold_val = 55' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_keyed_reduce_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_vlfb = 48' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_decay_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_decay_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_multi_reduce_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_rank_reduce_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_weight_reduce_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_thresh_reduce_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_soft_reduce_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_cal_reduce_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_adapt_reduce_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_blend_reduce_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_gate_reduce_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_score_reduce_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_softfloor_reduce_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_hardgate_reduce_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_dual_reduce_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_mix_reduce_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_credit_reduce_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_margin_reduce_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_band_reduce_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_vllive = 48' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_vlstream = 48' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_vlcost = 48' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_vladapt = 48' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_vlsched = 48' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_vlfuse = 48' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_vlmix = 48' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_vlbal = 48' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_vlbw = 48' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_vlpath = 48' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_vllane = 48' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_vlspill = 48' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_vlocc = 48' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_vlqueue = 48' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_vlwater = 48' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_vldrain = 48' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_vlrefill = 48' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_ema_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_adapt_hl_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_reg_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_floor_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_afloor_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_burst_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_cool_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_credit_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_debt_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_interest_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_amortize_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_reweight_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_budget_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_softcap_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_dual_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_conf_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_ema_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_adapt_hl_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_reg_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_floor_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_afloor_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_burst_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_cool_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_credit_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_debt_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_interest_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_amortize_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_reweight_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_budget_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_softcap_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_dual_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_conf_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_cost_10 = 10' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_cost_weighted_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_cost_isa_10 = 10' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_select_idx = 1' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_outer_acc_val = 225' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_cost_calibrated_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_plan_log_idx = 1' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_guard_ok = 1' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_wildcard = -1' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_outer_acc_true =' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_or_sum = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_stream_store =' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_stream_store_val = 225' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_weight_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_select_idx = 1' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_feedback = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_tag_match_val = 40' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_exhaust_ok = 1' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_tag_exhaust_ok = 1' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather =' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_scatter =' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_val = 45' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_scatter_val = 550' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_counter_bump = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_adapt_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_match_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_sugar_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mv =' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_scatter_mv =' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_mv_val = 45' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_scatter_mv_val = 550' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_isa_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_isa_loop_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_multi_match_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_or_sugar_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_tile =' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_scatter_tile =' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_tile_val = 45' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_scatter_tile_val = 550' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_isa_feedback_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_isa_live_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_recur_match_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_recur_sugar_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt =' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_scatter_mt =' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_mt_val = 45' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_scatter_mt_val = 550' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_xkernel_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_xkernel_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_tree_fold_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_ser_fold_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_peel =' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_scatter_mt_peel =' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_mt_peel_val = 45' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_scatter_mt_peel_val = 550' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_xpass_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_xpass_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_map_fold_val = 55' /tmp/trin_fused_scan_interp.txt
grep -q 'nested_tag_forest_fold_val = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_gather_mt_auto =' /tmp/trin_fused_scan_interp.txt
grep -q 'hw_sme_za_scatter_mt_auto =' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_gather_mt_auto_val = 45' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za_scatter_mt_auto_val = 550' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_window_counter_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'fusion_pgo_window_20 = 20' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_destructure_sum = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_return_sum = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_multiway_sum = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_nested_sum = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_match_sum = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'pair_of_pairs_sum = 110' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_vl = streaming' /tmp/trin_fused_scan_interp.txt
grep -q 'sme_za = multi_tile_vlrefill' /tmp/trin_fused_scan_interp.txt
grep -q 'map_reduce = pgo_amortize' /tmp/trin_fused_scan_interp.txt
grep -q 'dyn32_helper = trin_dyn_sum32' /tmp/trin_fused_scan_interp.txt
grep -q 'wide_peel_ladder = sve64' /tmp/trin_fused_scan_interp.txt
grep -q 'rdvl = live_sve' /tmp/trin_fused_scan_interp.txt
cat > /tmp/fused_soft_kernel.trin <<'EOF'
fn soft_kernel(w, a, x, b) {
    soft_mux(w, a * x, b)
}
EOF
./trin emit-mlir /tmp/fused_soft_kernel.trin -o /tmp/fused_check.mlir
grep -q 'func.func @trin_span_scan_reduce_add_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_span_scan_reduce_mul_f32' /tmp/fused_check.mlir
grep -q 'trin.scan_reduce_peel = "prefix_add_mul_multi_tuple"' /tmp/fused_check.mlir
grep -q 'trin.adaptive_peel = "isa_ladder_64"' /tmp/fused_check.mlir
grep -q 'trin.wide_peel = "sve64"' /tmp/fused_check.mlir
grep -q 'trin.dyn_tile = "runtime_32"' /tmp/fused_check.mlir
grep -q 'trin.auto_peel = "pointer_store_map_scan_mul_multi_tuple"' /tmp/fused_check.mlir
grep -q 'trin.rdvl = "live_sve"' /tmp/fused_check.mlir
grep -q 'func.func @trin_span_scan_reduce_add_pair_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_span_scan_reduce_mul_pair_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_rdvl' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_vl' /tmp/fused_check.mlir
grep -q 'trin.sme_za = "multi_tile_vlclose"' /tmp/fused_check.mlir
grep -q 'func.func @trin_pair_swap_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_pair_match_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_pair_exhaust_i32' /tmp/fused_check.mlir
grep -q 'func.func @trin_fused_map_reduce_mul_sum_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_tile' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_outer' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_outer_acc' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_cost' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_cost_weighted' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_cost_isa' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_select' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_outer_acc_true' /tmp/fused_check.mlir
grep -q 'func.func @trin_pair_match_guard_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_cost_calibrated' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_plan_log' /tmp/fused_check.mlir
grep -q 'func.func @trin_pair_or_match_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_stream_store' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_weight' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_feedback' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_select' /tmp/fused_check.mlir
grep -q 'func.func @trin_pair_tag_match_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_pair_exhaust_i32' /tmp/fused_check.mlir
grep -q 'func.func @trin_pair_tag_exhaust_i32' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_adapt' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_counter_bump' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_match_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mv' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mv' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_isa_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_isa_loop' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_multi_match_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_tile' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_tile' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_isa_live' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_isa_feedback' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_recur_match_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_xkernel' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_xkernel_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_tree_fold_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_peel' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_peel' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_xpass' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_xpass_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_forest_fold_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_map_fold_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_auto' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_auto' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_window' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_window_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_keyed_reduce_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_keyed_fold_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_vlfb' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_vlfb' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_decay' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_decay_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_multi_reduce_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_rank_reduce_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_weight_reduce_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_thresh_reduce_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_soft_reduce_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_cal_reduce_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_adapt_reduce_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_blend_reduce_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_gate_reduce_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_score_reduce_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_softfloor_reduce_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_hardgate_reduce_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_dual_reduce_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_mix_reduce_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_credit_reduce_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_margin_reduce_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_band_reduce_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_gate_filter_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_score_filter_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_softfloor_filter_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_hardgate_filter_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_dual_filter_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_mix_filter_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_credit_filter_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_margin_filter_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_band_filter_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_blend_filter_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_adapt_filter_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_cal_filter_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_soft_filter_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_thresh_filter_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_weight_filter_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_rank_filter_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_nested_tag_filter_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_vllive' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_vlstream' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_vlcost' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_vladapt' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_vlsched' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_vlfuse' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_vlmix' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_vlbal' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_vlbw' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_vlpath' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_vllane' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_vlspill' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_vlocc' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_vlqueue' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_vlwater' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_vldrain' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_gather_mt_vlrefill' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_vlbw' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_vlpath' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_vllane' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_vlspill' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_vlocc' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_vlqueue' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_vlwater' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_vldrain' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_vlrefill' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_vlbal' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_vlmix' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_vlfuse' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_vlsched' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_vladapt' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_vlcost' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_vlstream' /tmp/fused_check.mlir
grep -q 'func.func @trin_hw_sme_za_scatter_mt_vllive' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_ema' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_adapt_hl' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_reg' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_floor' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_afloor' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_burst' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_cool' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_credit' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_debt' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_interest' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_amortize' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_reweight' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_budget' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_softcap' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_dual' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_conf' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_ema_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_adapt_hl_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_reg_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_floor_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_afloor_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_burst_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_cool_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_credit_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_debt_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_interest_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_amortize_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_reweight_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_budget_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_softcap_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_dual_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_fusion_pgo_conf_counter' /tmp/fused_check.mlir
grep -q 'func.func @trin_fused_map_reduce_cost_mul_add_f32' /tmp/fused_check.mlir
grep -q 'func.func @trin_fused_map_reduce_mul_add_f32' /tmp/fused_check.mlir
grep -q 'trin.pair = "nested_lock_f32"' /tmp/fused_check.mlir
grep -q 'trin.sme_vl = "streaming"' /tmp/fused_check.mlir
grep -q 'trin.map_reduce = "pgo_close"' /tmp/fused_check.mlir
./trin emit-mlir-llvm /tmp/fused_soft_kernel.trin -o /tmp/fused_check.ll
grep -q 'define float @trin_span_scan_reduce_add_f32' /tmp/fused_check.ll
grep -q 'define float @trin_span_scan_reduce_mul_f32' /tmp/fused_check.ll
grep -q 'define float @trin_fused_scan_reduce_peel_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fused_scan_reduce_multi_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fused_scan_reduce_mul_probe' /tmp/fused_check.ll
grep -q 'define float @trin_rdvl_probe' /tmp/fused_check.ll
grep -q 'define float @trin_span_scan_reduce_add_pair_f32' /tmp/fused_check.ll
grep -q 'define float @trin_span_scan_reduce_mul_pair_f32' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_rdvl' /tmp/fused_check.ll
grep -q 'define { float, float } @trin_pair_f32' /tmp/fused_check.ll
grep -q 'define { float, float } @trin_span_scan_reduce_add_tuple_f32' /tmp/fused_check.ll
grep -q 'define { float, float } @trin_span_scan_reduce_add_sum_f32' /tmp/fused_check.ll
grep -q 'define float @trin_pair_tuple_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fused_sum_scan_probe' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_rdvl_live' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_vl' /tmp/fused_check.ll
grep -q 'define { float, float } @trin_fused_map_reduce_mul_add_f32' /tmp/fused_check.ll
grep -q 'define float @trin_fused_map_reduce_probe' /tmp/fused_check.ll
grep -q 'define float @trin_sme_vl_probe' /tmp/fused_check.ll
grep -q 'define float @trin_lang_pair_probe' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_tile' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_outer' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_outer_acc' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_cost' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_cost_weighted' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_select' /tmp/fused_check.ll
grep -q 'define { float, float } @trin_pair_swap_f32' /tmp/fused_check.ll
grep -q 'define { float, float } @trin_fused_map_reduce_mul_sum_f32' /tmp/fused_check.ll
grep -q 'define float @trin_fused_map_reduce_mul_sum_probe' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_tile_probe' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_outer_probe' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_outer_acc_probe' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_outer_acc_kernel_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_cost_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_cost_weighted_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_select_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fused_map_reduce_select_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fused_map_reduce_cost_probe' /tmp/fused_check.ll
grep -q 'define { float, float } @trin_fused_map_reduce_cost_mul_add_f32' /tmp/fused_check.ll
grep -q 'define float @trin_pair_return_probe' /tmp/fused_check.ll
grep -q 'define float @trin_pair_destructure_probe' /tmp/fused_check.ll
grep -q 'define float @trin_pair_nested_probe' /tmp/fused_check.ll
grep -q 'define float @trin_pair_match_probe' /tmp/fused_check.ll
grep -q 'define float @trin_pair_of_pairs_probe' /tmp/fused_check.ll
grep -q 'define float @trin_pair_exhaust_probe' /tmp/fused_check.ll
grep -q 'define float @trin_pair_wildcard_probe' /tmp/fused_check.ll
grep -q 'define float @trin_pair_guard_probe' /tmp/fused_check.ll
grep -q 'define float @trin_pair_match_guard_f32' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_outer_acc_true' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_outer_acc_true_probe' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_cost_calibrated' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_plan_log' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_cost_calibrated_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_plan_log_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fused_map_reduce_plan_probe' /tmp/fused_check.ll
grep -q 'define float @trin_pair_or_probe' /tmp/fused_check.ll
grep -q 'define float @trin_pair_or_match_f32' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_stream_store' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_stream_store_probe' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_weight' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_feedback' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_select' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_weight_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_feedback_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_select_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fused_map_reduce_pgo_probe' /tmp/fused_check.ll
grep -q 'define float @trin_pair_tag_exhaust_probe' /tmp/fused_check.ll
grep -q 'define float @trin_pair_tag_match_f32' /tmp/fused_check.ll
grep -q 'define i32 @trin_pair_exhaust_i32' /tmp/fused_check.ll
grep -q 'define i32 @trin_pair_tag_exhaust_i32' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_gather' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_scatter' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_gather_scatter_probe' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_adapt' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_counter_bump' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_adapt_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_counter_bump_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fused_map_reduce_adapt_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_match_f32' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_gather_mv' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_scatter_mv' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_gather_scatter_mv_probe' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_isa_counter' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_isa_loop' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_isa_counter_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_isa_loop_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fused_map_reduce_isa_loop_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_multi_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_multi_match_f32' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_gather_tile' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_scatter_tile' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_gather_scatter_tile_probe' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_isa_live' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_isa_feedback' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_isa_live_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_isa_feedback_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fused_map_reduce_isa_live_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_recur_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_recur_match_f32' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_gather_mt' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_scatter_mt' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_gather_scatter_mt_probe' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_xkernel' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_xkernel_counter' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_xkernel_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_xkernel_counter_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fused_map_reduce_xkernel_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_ser_fold_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_tree_fold_f32' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_gather_mt_peel' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_scatter_mt_peel' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_gather_scatter_mt_peel_probe' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_xpass' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_xpass_counter' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_xpass_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_xpass_counter_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fused_map_reduce_xpass_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_forest_fold_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_forest_fold_f32' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_gather_mt_auto' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_scatter_mt_auto' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_gather_scatter_mt_auto_probe' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_window' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_window_counter' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_window_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_window_counter_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fused_map_reduce_window_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_keyed_reduce_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_keyed_reduce_f32' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_gather_mt_vlfb' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_scatter_mt_vlfb' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_gather_scatter_mt_vlfb_probe' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_decay' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_decay_counter' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_decay_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_decay_counter_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fused_map_reduce_decay_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_multi_reduce_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_rank_reduce_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_weight_reduce_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_thresh_reduce_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_soft_reduce_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_cal_reduce_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_adapt_reduce_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_blend_reduce_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_gate_reduce_probe' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_gate_reduce_f32' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_blend_reduce_f32' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_adapt_reduce_f32' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_cal_reduce_f32' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_soft_reduce_f32' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_thresh_reduce_f32' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_weight_reduce_f32' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_rank_reduce_f32' /tmp/fused_check.ll
grep -q 'define float @trin_nested_tag_multi_reduce_f32' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_gather_mt_vllive' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_gather_mt_vlstream' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_gather_mt_vlcost' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_gather_mt_vladapt' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_gather_mt_vlsched' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_gather_mt_vlfuse' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_gather_mt_vlmix' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_gather_mt_vlbal' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_gather_mt_vlbw' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_scatter_mt_vlbw' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_scatter_mt_vlbal' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_scatter_mt_vlmix' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_scatter_mt_vlfuse' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_scatter_mt_vlsched' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_scatter_mt_vladapt' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_scatter_mt_vlcost' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_scatter_mt_vlstream' /tmp/fused_check.ll
grep -q 'define i32 @trin_hw_sme_za_scatter_mt_vllive' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_gather_scatter_mt_vllive_probe' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_gather_scatter_mt_vlstream_probe' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_gather_scatter_mt_vlcost_probe' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_gather_scatter_mt_vladapt_probe' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_gather_scatter_mt_vlsched_probe' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_gather_scatter_mt_vlfuse_probe' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_gather_scatter_mt_vlmix_probe' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_gather_scatter_mt_vlbal_probe' /tmp/fused_check.ll
grep -q 'define float @trin_sme_za_gather_scatter_mt_vlbw_probe' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_ema' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_adapt_hl' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_reg' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_reweight' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_budget' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_softcap' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_dual' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_conf' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_ema_counter' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_adapt_hl_counter' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_reg_counter' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_reweight_counter' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_budget_counter' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_softcap_counter' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_dual_counter' /tmp/fused_check.ll
grep -q 'define i32 @trin_fusion_pgo_conf_counter' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_ema_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_adapt_hl_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_reg_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_reweight_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_budget_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_softcap_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_dual_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_conf_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_ema_counter_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_adapt_hl_counter_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_reg_counter_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_reweight_counter_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_budget_counter_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_softcap_counter_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_dual_counter_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fusion_pgo_conf_counter_probe' /tmp/fused_check.ll
grep -q 'define float @trin_fused_map_reduce_ema_probe' /tmp/fused_check.ll
grep -q 'define float @trin_pair_match_f32' /tmp/fused_check.ll
grep -q 'define float @trin_wide_vlen_probe' /tmp/fused_check.ll
grep -q 'define float @trin_dyn32_probe' /tmp/fused_check.ll
cat > /tmp/dyn32_main.trin <<'EOF'
fn main() {
    let s = sum([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0]);
    println(str_concat("sum32 = ", to_str(s)));
}
EOF
./trin emit-llvm /tmp/dyn32_main.trin -o /tmp/dyn32.ll
grep -q 'define float @trin_dyn_sum32' /tmp/dyn32.ll
grep -q 'call float @trin_dyn_sum32' /tmp/dyn32.ll
cat > /tmp/wide64_main.trin <<'EOF'
fn main() {
    let s = sum([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0]);
    println(str_concat("sum40 = ", to_str(s)));
}
EOF
TRIN_HW_VLEN=64 ./trin emit-llvm /tmp/wide64_main.trin -o /tmp/wide64.ll
grep -E -q 'AdaptivePeelChunks\(hw=64|call float @trin_dyn_sum32' /tmp/wide64.ll


echo "[llm] ternary LLM production kernels + tiny model"
./trin run examples/llm_kernels.trin | tee /tmp/trin_llm_k_i.txt
grep -q 'linear_y =' /tmp/trin_llm_k_i.txt
grep -q 'softmax =' /tmp/trin_llm_k_i.txt
grep -q 'attn =' /tmp/trin_llm_k_i.txt
grep -q 'tok = 1' /tmp/trin_llm_k_i.txt
./trin run examples/tiny_ternary_llm.trin | tee /tmp/trin_tiny_llm_i.txt
grep -q 'tiny_ternary_llm_ok' /tmp/trin_tiny_llm_i.txt
grep -q 'prompt=AB gen=' /tmp/trin_tiny_llm_i.txt
./trin run examples/load_quant_infer.trin | tee /tmp/trin_loadq_i.txt
grep -q 'load_quant_infer_ok' /tmp/trin_loadq_i.txt
./trin build examples/llm_kernels.trin -o build/llm_kernels --backend go
./build/llm_kernels | tee /tmp/trin_llm_k_n.txt
grep -q 'linear_y =' /tmp/trin_llm_k_n.txt
./trin build examples/tiny_ternary_llm.trin -o build/tiny_ternary_llm --backend go
./build/tiny_ternary_llm | tee /tmp/trin_tiny_llm_n.txt
grep -q 'tiny_ternary_llm_ok' /tmp/trin_tiny_llm_n.txt
./trin build examples/load_quant_infer.trin -o build/load_quant_infer --backend go
./build/load_quant_infer | tee /tmp/trin_loadq_n.txt
grep -q 'load_quant_infer_ok' /tmp/trin_loadq_n.txt

echo "[llm] v0.84 GGUF block-quant (Q4_0/Q8_0) dequant"
go run scripts/gen_q4_fixture.go >/tmp/trin_q4_gen.txt
./trin run examples/load_gguf_q4.trin | tee /tmp/trin_q4_i.txt
grep -q 'load_gguf_q4_ok' /tmp/trin_q4_i.txt
grep -q 'y_q4=' /tmp/trin_q4_i.txt
grep -q 'y_q8=' /tmp/trin_q4_i.txt
./trin build examples/load_gguf_q4.trin -o build/load_gguf_q4 --backend go
./build/load_gguf_q4 | tee /tmp/trin_q4_n.txt
grep -q 'load_gguf_q4_ok' /tmp/trin_q4_n.txt

echo "[llm] v0.85 GGUF/safetensors mmap open"
./trin run examples/load_gguf_mmap.trin | tee /tmp/trin_mmap_i.txt
grep -q 'load_gguf_mmap_ok' /tmp/trin_mmap_i.txt
grep -q 'mapped_q4=1' /tmp/trin_mmap_i.txt
./trin build examples/load_gguf_mmap.trin -o build/load_gguf_mmap --backend go
./build/load_gguf_mmap | tee /tmp/trin_mmap_n.txt
grep -q 'load_gguf_mmap_ok' /tmp/trin_mmap_n.txt
grep -q 'mapped_q4=1' /tmp/trin_mmap_n.txt

echo "[llm] v0.86 weight sessions (retained mmap)"
./trin run examples/weight_session.trin | tee /tmp/trin_sess_i.txt
grep -q 'weight_session_ok' /tmp/trin_sess_i.txt
grep -q 'mapped=1' /tmp/trin_sess_i.txt
grep -q 'n_b=' /tmp/trin_sess_i.txt
./trin build examples/weight_session.trin -o build/weight_session --backend go
./build/weight_session | tee /tmp/trin_sess_n.txt
grep -q 'weight_session_ok' /tmp/trin_sess_n.txt
grep -q 'mapped=1' /tmp/trin_sess_n.txt

echo "[llm] v0.87 windowed tensor loads"
./trin run examples/weight_window.trin | tee /tmp/trin_win_i.txt
grep -q 'weight_window_ok' /tmp/trin_win_i.txt
grep -q 'q4_match=1' /tmp/trin_win_i.txt
./trin build examples/weight_window.trin -o build/weight_window --backend go
./build/weight_window | tee /tmp/trin_win_n.txt
grep -q 'weight_window_ok' /tmp/trin_win_n.txt
grep -q 'q4_match=1' /tmp/trin_win_n.txt

echo "[llm] v0.88 session-backed tiled linear"
./trin run examples/weight_session_linear.trin | tee /tmp/trin_lin_i.txt
grep -q 'weight_session_linear_ok' /tmp/trin_lin_i.txt
grep -q 'tile_match=1' /tmp/trin_lin_i.txt
grep -q 'q4_tile_match=1' /tmp/trin_lin_i.txt
./trin build examples/weight_session_linear.trin -o build/weight_session_linear --backend go
./build/weight_session_linear | tee /tmp/trin_lin_n.txt
grep -q 'weight_session_linear_ok' /tmp/trin_lin_n.txt
grep -q 'q4_tile_match=1' /tmp/trin_lin_n.txt

echo "[llm] v0.89 fused session linear (block-stream)"
./trin run examples/weight_session_linear_fused.trin | tee /tmp/trin_fuse_i.txt
grep -q 'weight_session_linear_fused_ok' /tmp/trin_fuse_i.txt
grep -q 'q4_match=1' /tmp/trin_fuse_i.txt
./trin build examples/weight_session_linear_fused.trin -o build/weight_session_linear_fused --backend go
./build/weight_session_linear_fused | tee /tmp/trin_fuse_n.txt
grep -q 'weight_session_linear_fused_ok' /tmp/trin_fuse_n.txt
grep -q 'q4_match=1' /tmp/trin_fuse_n.txt

echo "[llm] v0.90 multi-layer residual session stack"
./trin run examples/weight_session_layer_stack.trin | tee /tmp/trin_stack_i.txt
grep -q 'weight_session_layer_stack_ok' /tmp/trin_stack_i.txt
grep -q 'stack_match=1' /tmp/trin_stack_i.txt
./trin build examples/weight_session_layer_stack.trin -o build/weight_session_layer_stack --backend go
./build/weight_session_layer_stack | tee /tmp/trin_stack_n.txt
grep -q 'weight_session_layer_stack_ok' /tmp/trin_stack_n.txt
grep -q 'stack_match=1' /tmp/trin_stack_n.txt

echo "[llm] v0.91 session-wired multi-head attention"
./trin run examples/weight_session_attn.trin | tee /tmp/trin_attn_i.txt
grep -q 'weight_session_attn_ok' /tmp/trin_attn_i.txt
grep -q 'attn_match=1' /tmp/trin_attn_i.txt
./trin build examples/weight_session_attn.trin -o build/weight_session_attn --backend go
./build/weight_session_attn | tee /tmp/trin_attn_n.txt
grep -q 'weight_session_attn_ok' /tmp/trin_attn_n.txt
grep -q 'attn_match=1' /tmp/trin_attn_n.txt

echo "[llm] v0.92 GQA packing helpers + session-wired GQA"
./trin run examples/weight_session_gqa.trin | tee /tmp/trin_gqa_i.txt
grep -q 'weight_session_gqa_ok' /tmp/trin_gqa_i.txt
grep -q 'gqa_match=1' /tmp/trin_gqa_i.txt
./trin build examples/weight_session_gqa.trin -o build/weight_session_gqa --backend go
./build/weight_session_gqa | tee /tmp/trin_gqa_n.txt
grep -q 'weight_session_gqa_ok' /tmp/trin_gqa_n.txt
grep -q 'gqa_match=1' /tmp/trin_gqa_n.txt

echo "[llm] v0.93 IQ4_NL + TQ2_0 GGUF dequant"
./trin run examples/weight_iq_tq.trin | tee /tmp/trin_iqtq_i.txt
grep -q 'weight_iq_tq_ok' /tmp/trin_iqtq_i.txt
grep -q 'iq_match=1' /tmp/trin_iqtq_i.txt
grep -q 'tq_match=1' /tmp/trin_iqtq_i.txt
./trin build examples/weight_iq_tq.trin -o build/weight_iq_tq --backend go
./build/weight_iq_tq | tee /tmp/trin_iqtq_n.txt
grep -q 'weight_iq_tq_ok' /tmp/trin_iqtq_n.txt
grep -q 'iq_tq_ok=1' /tmp/trin_iqtq_n.txt

echo "[llm] v0.94 IQ4_XS + TQ1_0 GGUF dequant"
./trin run examples/weight_iq_tq2.trin | tee /tmp/trin_iqtq2_i.txt
grep -q 'weight_iq_tq2_ok' /tmp/trin_iqtq2_i.txt
grep -q 'xs_match=1' /tmp/trin_iqtq2_i.txt
grep -q 'tq1_match=1' /tmp/trin_iqtq2_i.txt
./trin build examples/weight_iq_tq2.trin -o build/weight_iq_tq2 --backend go
./build/weight_iq_tq2 | tee /tmp/trin_iqtq2_n.txt
grep -q 'weight_iq_tq2_ok' /tmp/trin_iqtq2_n.txt
grep -q 'iq_tq2_ok=1' /tmp/trin_iqtq2_n.txt

echo "[llm] v0.95 IQ2_XXS + IQ3_XXS GGUF grid dequant"
./trin run examples/weight_iq_grid.trin | tee /tmp/trin_iqgrid_i.txt
grep -q 'weight_iq_grid_ok' /tmp/trin_iqgrid_i.txt
grep -q 'iq2_match=1' /tmp/trin_iqgrid_i.txt
grep -q 'iq3_match=1' /tmp/trin_iqgrid_i.txt
./trin build examples/weight_iq_grid.trin -o build/weight_iq_grid --backend go
./build/weight_iq_grid | tee /tmp/trin_iqgrid_n.txt
grep -q 'weight_iq_grid_ok' /tmp/trin_iqgrid_n.txt
grep -q 'iq_grid_ok=1' /tmp/trin_iqgrid_n.txt

echo "[llm] v0.96 IQ2_XS + IQ2_S GGUF family dequant"
./trin run examples/weight_iq2_family.trin | tee /tmp/trin_iq2fam_i.txt
grep -q 'weight_iq2_family_ok' /tmp/trin_iq2fam_i.txt
grep -q 'xs_match=1' /tmp/trin_iq2fam_i.txt
grep -q 's_match=1' /tmp/trin_iq2fam_i.txt
./trin build examples/weight_iq2_family.trin -o build/weight_iq2_family --backend go
./build/weight_iq2_family | tee /tmp/trin_iq2fam_n.txt
grep -q 'weight_iq2_family_ok' /tmp/trin_iq2fam_n.txt
grep -q 'iq2_family_ok=1' /tmp/trin_iq2fam_n.txt

echo "[llm] v0.97 IQ3_S GGUF dequant"
./trin run examples/weight_iq3_s.trin | tee /tmp/trin_iq3s_i.txt
grep -q 'weight_iq3_s_ok' /tmp/trin_iq3s_i.txt
grep -q 'iq3_s_ok=1' /tmp/trin_iq3s_i.txt
./trin build examples/weight_iq3_s.trin -o build/weight_iq3_s --backend go
./build/weight_iq3_s | tee /tmp/trin_iq3s_n.txt
grep -q 'weight_iq3_s_ok' /tmp/trin_iq3s_n.txt
grep -q 'iq3_s_ok=1' /tmp/trin_iq3s_n.txt

echo "[llm] v0.98 IQ1 family GGUF dequant"
./trin run examples/weight_iq1_family.trin | tee /tmp/trin_iq1fam_i.txt
grep -q 'weight_iq1_family_ok' /tmp/trin_iq1fam_i.txt
grep -q 'iq1_family_ok=1' /tmp/trin_iq1fam_i.txt
./trin build examples/weight_iq1_family.trin -o build/weight_iq1_family --backend go
./build/weight_iq1_family | tee /tmp/trin_iq1fam_n.txt
grep -q 'weight_iq1_family_ok' /tmp/trin_iq1fam_n.txt
grep -q 'iq1_family_ok=1' /tmp/trin_iq1fam_n.txt

echo "[llm] v0.99 MXFP4 + NVFP4 GGUF dequant"
./trin run examples/weight_mxfp_family.trin | tee /tmp/trin_mxfp_i.txt
grep -q 'weight_mxfp_family_ok' /tmp/trin_mxfp_i.txt
grep -q 'mxfp_family_ok=1' /tmp/trin_mxfp_i.txt
./trin build examples/weight_mxfp_family.trin -o build/weight_mxfp_family --backend go
./build/weight_mxfp_family | tee /tmp/trin_mxfp_n.txt
grep -q 'weight_mxfp_family_ok' /tmp/trin_mxfp_n.txt
grep -q 'mxfp_family_ok=1' /tmp/trin_mxfp_n.txt

echo "[llm] v1.00 multi-shard GGUF"
./trin run examples/weight_multishard.trin | tee /tmp/trin_ms_i.txt
grep -q 'weight_multishard_ok' /tmp/trin_ms_i.txt
grep -q 'multishard_ok=1' /tmp/trin_ms_i.txt
./trin build examples/weight_multishard.trin -o build/weight_multishard --backend go
./build/weight_multishard | tee /tmp/trin_ms_n.txt
grep -q 'weight_multishard_ok' /tmp/trin_ms_n.txt
grep -q 'multishard_ok=1' /tmp/trin_ms_n.txt

echo "[lang] v1.01 loop/for sugar"
./trin run examples/loop_sugar.trin | tee /tmp/trin_loop_i.txt
grep -q 'loop_sugar_ok' /tmp/trin_loop_i.txt
grep -q 'loop_sugar_ok=1' /tmp/trin_loop_i.txt
./trin build examples/loop_sugar.trin -o build/loop_sugar --backend go
./build/loop_sugar | tee /tmp/trin_loop_n.txt
grep -q 'loop_sugar_ok' /tmp/trin_loop_n.txt
grep -q 'loop_sugar_ok=1' /tmp/trin_loop_n.txt

echo "[llm] v0.83 GGUF + safetensors weight importers"
# ensure fixtures exist
go run scripts/gen_weight_fixtures.go >/tmp/trin_gen_fix.txt
./trin run examples/load_weights_gguf_st.trin | tee /tmp/trin_w_i.txt
grep -q 'load_weights_gguf_st_ok' /tmp/trin_w_i.txt
grep -q 'y_st=' /tmp/trin_w_i.txt
grep -q 'y_gg=' /tmp/trin_w_i.txt
grep -q 'gg_meta=trin-toy' /tmp/trin_w_i.txt
./trin build examples/load_weights_gguf_st.trin -o build/load_weights_gguf_st --backend go
./build/load_weights_gguf_st | tee /tmp/trin_w_n.txt
grep -q 'load_weights_gguf_st_ok' /tmp/trin_w_n.txt
grep -q 'y_st=' /tmp/trin_w_n.txt

echo "[llm] v0.82 KV-cache + deterministic sampling"
./trin run examples/kv_sample.trin | tee /tmp/trin_kv_i.txt
grep -q 'kv_sample_ok' /tmp/trin_kv_i.txt
grep -q 'kv_len=2' /tmp/trin_kv_i.txt
grep -q 'greedy=1' /tmp/trin_kv_i.txt
# seed42a and seed42b must match
a=$(grep 'seed42a=' /tmp/trin_kv_i.txt | head -1)
b=$(grep 'seed42b=' /tmp/trin_kv_i.txt | head -1)
test "${a#seed42a=}" = "${b#seed42b=}"
./trin run examples/tiny_ternary_llm_kv.trin | tee /tmp/trin_tiny_kv_i.txt
grep -q 'tiny_ternary_llm_kv_ok' /tmp/trin_tiny_kv_i.txt
grep -q 'final_kv_toks=4' /tmp/trin_tiny_kv_i.txt
./trin build examples/kv_sample.trin -o build/kv_sample --backend go
./build/kv_sample | tee /tmp/trin_kv_n.txt
grep -q 'kv_sample_ok' /tmp/trin_kv_n.txt
./trin build examples/tiny_ternary_llm_kv.trin -o build/tiny_ternary_llm_kv --backend go
./build/tiny_ternary_llm_kv | tee /tmp/trin_tiny_kv_n.txt
grep -q 'tiny_ternary_llm_kv_ok' /tmp/trin_tiny_kv_n.txt


echo "[cuda] v1.02 CUDA path surface"
./trin run examples/cuda_path.trin | tee /tmp/trin_cuda_i.txt
grep -q 'cuda_path_ok' /tmp/trin_cuda_i.txt
grep -q 'cuda_ok=1' /tmp/trin_cuda_i.txt
./trin build examples/cuda_path.trin -o build/cuda_path --backend go
./build/cuda_path | tee /tmp/trin_cuda_n.txt
grep -q 'cuda_path_ok' /tmp/trin_cuda_n.txt
grep -q 'cuda_ok=1' /tmp/trin_cuda_n.txt


echo "[cuda] v1.03 CUDA PTX layer"
./trin run examples/cuda_ptx.trin | tee /tmp/trin_cudaptx_i.txt
grep -q 'cuda_ptx_path_ok' /tmp/trin_cudaptx_i.txt
grep -q 'cuda_ptx_ok=1' /tmp/trin_cudaptx_i.txt
./trin build examples/cuda_ptx.trin -o build/cuda_ptx --backend go
./build/cuda_ptx | tee /tmp/trin_cudaptx_n.txt
grep -q 'cuda_ptx_path_ok' /tmp/trin_cudaptx_n.txt
grep -q 'cuda_ptx_ok=1' /tmp/trin_cudaptx_n.txt


echo "[llm] v1.04 tokenizer + GGUF session greedy decode"
./trin run examples/vocab_table.trin | tee /tmp/trin_vocab_i.txt
grep -q 'vocab_table_ok' /tmp/trin_vocab_i.txt
grep -q 'vocab_ok=1' /tmp/trin_vocab_i.txt
./trin build examples/vocab_table.trin -o build/vocab_table --backend go
./build/vocab_table | tee /tmp/trin_vocab_n.txt
grep -q 'vocab_table_ok' /tmp/trin_vocab_n.txt
grep -q 'vocab_ok=1' /tmp/trin_vocab_n.txt
go run scripts/gen_tiny_llm_gguf.go . >/tmp/trin_gen_tiny.txt
./trin run examples/tiny_gguf_decode.trin | tee /tmp/trin_ggufdec_i.txt
grep -q 'tiny_gguf_decode_ok' /tmp/trin_ggufdec_i.txt
grep -q 'decode_ok=1' /tmp/trin_ggufdec_i.txt
grep -q 'final_kv_toks=4' /tmp/trin_ggufdec_i.txt
./trin build examples/tiny_gguf_decode.trin -o build/tiny_gguf_decode --backend go
./build/tiny_gguf_decode | tee /tmp/trin_ggufdec_n.txt
grep -q 'tiny_gguf_decode_ok' /tmp/trin_ggufdec_n.txt
grep -q 'decode_ok=1' /tmp/trin_ggufdec_n.txt

echo "[llm] v1.05 linear_trit simd + BPE sidecar"
./trin run examples/linear_trit_simd.trin | tee /tmp/trin_ltsimd_i.txt
grep -q 'linear_trit_simd_ok' /tmp/trin_ltsimd_i.txt
grep -q 'parity_ok=1' /tmp/trin_ltsimd_i.txt
./trin build examples/linear_trit_simd.trin -o build/linear_trit_simd --backend go
./build/linear_trit_simd | tee /tmp/trin_ltsimd_n.txt
grep -q 'linear_trit_simd_ok' /tmp/trin_ltsimd_n.txt
grep -q 'parity_ok=1' /tmp/trin_ltsimd_n.txt
./trin run examples/bpe_sidecar.trin | tee /tmp/trin_bpe_i.txt
grep -q 'bpe_sidecar_ok' /tmp/trin_bpe_i.txt
grep -q 'match_abc=1' /tmp/trin_bpe_i.txt
./trin build examples/bpe_sidecar.trin -o build/bpe_sidecar --backend go
./build/bpe_sidecar | tee /tmp/trin_bpe_n.txt
grep -q 'bpe_sidecar_ok' /tmp/trin_bpe_n.txt
grep -q 'match_abc=1' /tmp/trin_bpe_n.txt

echo "[llm] v1.06 session linear_trit + microbench"
go run scripts/gen_trit_session_fixture.go . >/tmp/trin_gen_trit_sess.txt
./trin run examples/weight_session_linear_trit.trin | tee /tmp/trin_sess_lt_i.txt
grep -q 'weight_session_linear_trit_ok' /tmp/trin_sess_lt_i.txt
grep -q 'parity_ok=1' /tmp/trin_sess_lt_i.txt
grep -q 'tile_ok=1' /tmp/trin_sess_lt_i.txt
./trin build examples/weight_session_linear_trit.trin -o build/weight_session_linear_trit --backend go
./build/weight_session_linear_trit | tee /tmp/trin_sess_lt_n.txt
grep -q 'weight_session_linear_trit_ok' /tmp/trin_sess_lt_n.txt
grep -q 'parity_ok=1' /tmp/trin_sess_lt_n.txt
./trin run examples/linear_trit_bench.trin | tee /tmp/trin_lt_bench_i.txt
grep -q 'linear_trit_bench_ok' /tmp/trin_lt_bench_i.txt
grep -q 'parity_ok=1' /tmp/trin_lt_bench_i.txt
./trin build examples/linear_trit_bench.trin -o build/linear_trit_bench --backend go
./build/linear_trit_bench | tee /tmp/trin_lt_bench_n.txt
grep -q 'linear_trit_bench_ok' /tmp/trin_lt_bench_n.txt
grep -q 'parity_ok=1' /tmp/trin_lt_bench_n.txt

echo "[llm] v1.07 session trit prepack cache"
./trin run examples/weight_session_trit_cache.trin | tee /tmp/trin_trit_cache_i.txt
grep -q 'weight_session_trit_cache_ok' /tmp/trin_trit_cache_i.txt
grep -q 'cache_hit_ok=1' /tmp/trin_trit_cache_i.txt
grep -q 'parity_ok=1' /tmp/trin_trit_cache_i.txt
grep -q 'built2=0' /tmp/trin_trit_cache_i.txt
./trin build examples/weight_session_trit_cache.trin -o build/weight_session_trit_cache --backend go
./build/weight_session_trit_cache | tee /tmp/trin_trit_cache_n.txt
grep -q 'weight_session_trit_cache_ok' /tmp/trin_trit_cache_n.txt
grep -q 'cache_hit_ok=1' /tmp/trin_trit_cache_n.txt
grep -q 'parity_ok=1' /tmp/trin_trit_cache_n.txt

echo "[llm] v1.08 tiny GGUF decode with trit cache"
go run scripts/gen_tiny_llm_gguf.go . >/tmp/trin_gen_tiny_llm.txt
./trin run examples/tiny_gguf_decode_trit.trin | tee /tmp/trin_ggufdec_trit_i.txt
grep -q 'tiny_gguf_decode_trit_ok' /tmp/trin_ggufdec_trit_i.txt
grep -q 'cache_hit_ok=1' /tmp/trin_ggufdec_trit_i.txt
grep -q 'parity_ok=1' /tmp/trin_ggufdec_trit_i.txt
grep -q 'decode_ok=1' /tmp/trin_ggufdec_trit_i.txt
grep -q 'prepack_q_hit=0' /tmp/trin_ggufdec_trit_i.txt
./trin build examples/tiny_gguf_decode_trit.trin -o build/tiny_gguf_decode_trit --backend go
./build/tiny_gguf_decode_trit | tee /tmp/trin_ggufdec_trit_n.txt
grep -q 'tiny_gguf_decode_trit_ok' /tmp/trin_ggufdec_trit_n.txt
grep -q 'cache_hit_ok=1' /tmp/trin_ggufdec_trit_n.txt
grep -q 'parity_ok=1' /tmp/trin_ggufdec_trit_n.txt

echo "[llm] v1.09 layer stack trit residual"
./trin run examples/weight_session_layer_stack_trit.trin | tee /tmp/trin_stack_trit_i.txt
grep -q 'weight_session_layer_stack_trit_ok' /tmp/trin_stack_trit_i.txt
grep -q 'cache_hit_ok=1' /tmp/trin_stack_trit_i.txt
grep -q 'parity_ok=1' /tmp/trin_stack_trit_i.txt
grep -q 'stack_match=1' /tmp/trin_stack_trit_i.txt
grep -q 'uncached_ok=1' /tmp/trin_stack_trit_i.txt
./trin build examples/weight_session_layer_stack_trit.trin -o build/weight_session_layer_stack_trit --backend go
./build/weight_session_layer_stack_trit | tee /tmp/trin_stack_trit_n.txt
grep -q 'weight_session_layer_stack_trit_ok' /tmp/trin_stack_trit_n.txt
grep -q 'cache_hit_ok=1' /tmp/trin_stack_trit_n.txt
grep -q 'parity_ok=1' /tmp/trin_stack_trit_n.txt

echo "[llm] v1.10 session attn trit projections"
./trin run examples/weight_session_attn_trit.trin | tee /tmp/trin_attn_trit_i.txt
grep -q 'weight_session_attn_trit_ok' /tmp/trin_attn_trit_i.txt
grep -q 'cache_hit_ok=1' /tmp/trin_attn_trit_i.txt
grep -q 'parity_ok=1' /tmp/trin_attn_trit_i.txt
grep -q 'proj_ok=1' /tmp/trin_attn_trit_i.txt
grep -q 'y1_ok=1' /tmp/trin_attn_trit_i.txt
./trin build examples/weight_session_attn_trit.trin -o build/weight_session_attn_trit --backend go
./build/weight_session_attn_trit | tee /tmp/trin_attn_trit_n.txt
grep -q 'weight_session_attn_trit_ok' /tmp/trin_attn_trit_n.txt
grep -q 'cache_hit_ok=1' /tmp/trin_attn_trit_n.txt
grep -q 'parity_ok=1' /tmp/trin_attn_trit_n.txt

echo "[llm] v1.11 tiny GGUF decode on trit-attn"
./trin run examples/tiny_gguf_decode_attn_trit.trin | tee /tmp/trin_ggufdec_attn_trit_i.txt
grep -q 'tiny_gguf_decode_attn_trit_ok' /tmp/trin_ggufdec_attn_trit_i.txt
grep -q 'cache_hit_ok=1' /tmp/trin_ggufdec_attn_trit_i.txt
grep -q 'parity_ok=1' /tmp/trin_ggufdec_attn_trit_i.txt
grep -q 'decode_ok=1' /tmp/trin_ggufdec_attn_trit_i.txt
grep -q 'match_trit_decode=1' /tmp/trin_ggufdec_attn_trit_i.txt
./trin build examples/tiny_gguf_decode_attn_trit.trin -o build/tiny_gguf_decode_attn_trit --backend go
./build/tiny_gguf_decode_attn_trit | tee /tmp/trin_ggufdec_attn_trit_n.txt
grep -q 'tiny_gguf_decode_attn_trit_ok' /tmp/trin_ggufdec_attn_trit_n.txt
grep -q 'cache_hit_ok=1' /tmp/trin_ggufdec_attn_trit_n.txt
grep -q 'parity_ok=1' /tmp/trin_ggufdec_attn_trit_n.txt

echo "[llm] v1.12 multi-layer trit-attn stack"
./trin run examples/weight_session_attn_stack_trit.trin | tee /tmp/trin_attn_stack_trit_i.txt
grep -q 'weight_session_attn_stack_trit_ok' /tmp/trin_attn_stack_trit_i.txt
grep -q 'cache_hit_ok=1' /tmp/trin_attn_stack_trit_i.txt
grep -q 'parity_ok=1' /tmp/trin_attn_stack_trit_i.txt
grep -q 'stack_match=1' /tmp/trin_attn_stack_trit_i.txt
grep -q 'grow_ok=1' /tmp/trin_attn_stack_trit_i.txt
./trin build examples/weight_session_attn_stack_trit.trin -o build/weight_session_attn_stack_trit --backend go
./build/weight_session_attn_stack_trit | tee /tmp/trin_attn_stack_trit_n.txt
grep -q 'weight_session_attn_stack_trit_ok' /tmp/trin_attn_stack_trit_n.txt
grep -q 'cache_hit_ok=1' /tmp/trin_attn_stack_trit_n.txt
grep -q 'parity_ok=1' /tmp/trin_attn_stack_trit_n.txt

echo "[llm] v1.13 multi-layer GGUF decode on trit-attn stack"
./trin run examples/tiny_gguf_decode_stack_trit.trin | tee /tmp/trin_ggufdec_stack_trit_i.txt
grep -q 'tiny_gguf_decode_stack_trit_ok' /tmp/trin_ggufdec_stack_trit_i.txt
grep -q 'cache_hit_ok=1' /tmp/trin_ggufdec_stack_trit_i.txt
grep -q 'parity_ok=1' /tmp/trin_ggufdec_stack_trit_i.txt
grep -q 'decode_ok=1' /tmp/trin_ggufdec_stack_trit_i.txt
grep -q 'grow_final_ok=1' /tmp/trin_ggufdec_stack_trit_i.txt
grep -q 'gen_text=EEE' /tmp/trin_ggufdec_stack_trit_i.txt
./trin build examples/tiny_gguf_decode_stack_trit.trin -o build/tiny_gguf_decode_stack_trit --backend go
./build/tiny_gguf_decode_stack_trit | tee /tmp/trin_ggufdec_stack_trit_n.txt
grep -q 'tiny_gguf_decode_stack_trit_ok' /tmp/trin_ggufdec_stack_trit_n.txt
grep -q 'cache_hit_ok=1' /tmp/trin_ggufdec_stack_trit_n.txt
grep -q 'parity_ok=1' /tmp/trin_ggufdec_stack_trit_n.txt

echo "[llm] v1.14 multi-layer GQA stack + final-norm"
./trin run examples/weight_session_gqa_stack_trit.trin | tee /tmp/trin_gqa_stack_trit_i.txt
grep -q 'weight_session_gqa_stack_trit_ok' /tmp/trin_gqa_stack_trit_i.txt
grep -q 'cache_hit_ok=1' /tmp/trin_gqa_stack_trit_i.txt
grep -q 'parity_ok=1' /tmp/trin_gqa_stack_trit_i.txt
grep -q 'stack_match=1' /tmp/trin_gqa_stack_trit_i.txt
grep -q 'grow_ok=1' /tmp/trin_gqa_stack_trit_i.txt
grep -q 'final_ok=1' /tmp/trin_gqa_stack_trit_i.txt
./trin build examples/weight_session_gqa_stack_trit.trin -o build/weight_session_gqa_stack_trit --backend go
./build/weight_session_gqa_stack_trit | tee /tmp/trin_gqa_stack_trit_n.txt
grep -q 'weight_session_gqa_stack_trit_ok' /tmp/trin_gqa_stack_trit_n.txt
grep -q 'cache_hit_ok=1' /tmp/trin_gqa_stack_trit_n.txt
grep -q 'parity_ok=1' /tmp/trin_gqa_stack_trit_n.txt

echo "[llm] v1.14 multi-layer GQA GGUF decode + final-norm"
./trin run examples/tiny_gguf_decode_gqa_trit.trin | tee /tmp/trin_ggufdec_gqa_trit_i.txt
grep -q 'tiny_gguf_decode_gqa_trit_ok' /tmp/trin_ggufdec_gqa_trit_i.txt
grep -q 'cache_hit_ok=1' /tmp/trin_ggufdec_gqa_trit_i.txt
grep -q 'parity_ok=1' /tmp/trin_ggufdec_gqa_trit_i.txt
grep -q 'decode_ok=1' /tmp/trin_ggufdec_gqa_trit_i.txt
grep -q 'grow_final_ok=1' /tmp/trin_ggufdec_gqa_trit_i.txt
grep -q 'gen_text=FFF' /tmp/trin_ggufdec_gqa_trit_i.txt
./trin build examples/tiny_gguf_decode_gqa_trit.trin -o build/tiny_gguf_decode_gqa_trit --backend go
./build/tiny_gguf_decode_gqa_trit | tee /tmp/trin_ggufdec_gqa_trit_n.txt
grep -q 'tiny_gguf_decode_gqa_trit_ok' /tmp/trin_ggufdec_gqa_trit_n.txt
grep -q 'cache_hit_ok=1' /tmp/trin_ggufdec_gqa_trit_n.txt
grep -q 'parity_ok=1' /tmp/trin_ggufdec_gqa_trit_n.txt

echo "[llm] v1.15 multi-token GQA SwiGLU prefill + decode"
./trin run examples/weight_session_gqa_swiglu_prefill.trin | tee /tmp/trin_gqa_swiglu_prefill_i.txt
grep -q 'weight_session_gqa_swiglu_prefill_ok' /tmp/trin_gqa_swiglu_prefill_i.txt
grep -q 'cache_hit_ok=1' /tmp/trin_gqa_swiglu_prefill_i.txt
grep -q 'parity_ok=1' /tmp/trin_gqa_swiglu_prefill_i.txt
grep -q 'parity_prefill_ok=1' /tmp/trin_gqa_swiglu_prefill_i.txt
grep -q 'grow_final_ok=1' /tmp/trin_gqa_swiglu_prefill_i.txt
grep -q 'gen_text=FFF' /tmp/trin_gqa_swiglu_prefill_i.txt
./trin build examples/weight_session_gqa_swiglu_prefill.trin -o build/weight_session_gqa_swiglu_prefill --backend go
./build/weight_session_gqa_swiglu_prefill | tee /tmp/trin_gqa_swiglu_prefill_n.txt
grep -q 'weight_session_gqa_swiglu_prefill_ok' /tmp/trin_gqa_swiglu_prefill_n.txt
grep -q 'cache_hit_ok=1' /tmp/trin_gqa_swiglu_prefill_n.txt
grep -q 'parity_ok=1' /tmp/trin_gqa_swiglu_prefill_n.txt

echo "[llm] v1.16 tied emb + ALiBi GQA SwiGLU prefill"
./trin run examples/weight_session_tied_alibi.trin | tee /tmp/trin_tied_alibi_i.txt
grep -q 'weight_session_tied_alibi_ok' /tmp/trin_tied_alibi_i.txt
grep -q 'cache_hit_ok=1' /tmp/trin_tied_alibi_i.txt
grep -q 'parity_ok=1' /tmp/trin_tied_alibi_i.txt
grep -q 'emb_ok=1' /tmp/trin_tied_alibi_i.txt
grep -q 'tied_match=1' /tmp/trin_tied_alibi_i.txt
grep -q 'gen_text=FFF' /tmp/trin_tied_alibi_i.txt
./trin build examples/weight_session_tied_alibi.trin -o build/weight_session_tied_alibi --backend go
./build/weight_session_tied_alibi | tee /tmp/trin_tied_alibi_n.txt
grep -q 'weight_session_tied_alibi_ok' /tmp/trin_tied_alibi_n.txt
grep -q 'cache_hit_ok=1' /tmp/trin_tied_alibi_n.txt
grep -q 'tied_match=1' /tmp/trin_tied_alibi_n.txt

echo "[llm] v1.17 llama name adapters"
./trin run examples/weight_session_llama_named.trin | tee /tmp/trin_llama_named_i.txt
grep -q 'weight_session_llama_named_ok' /tmp/trin_llama_named_i.txt
grep -q 'cache_hit_ok=1' /tmp/trin_llama_named_i.txt
grep -q 'name_ok=1' /tmp/trin_llama_named_i.txt
grep -q 'style=llama' /tmp/trin_llama_named_i.txt
grep -q 'gen_text=FFF' /tmp/trin_llama_named_i.txt
./trin build examples/weight_session_llama_named.trin -o build/weight_session_llama_named --backend go
./build/weight_session_llama_named | tee /tmp/trin_llama_named_n.txt
grep -q 'weight_session_llama_named_ok' /tmp/trin_llama_named_n.txt
grep -q 'name_ok=1' /tmp/trin_llama_named_n.txt
grep -q 'cache_hit_ok=1' /tmp/trin_llama_named_n.txt

echo "OK: all examples interpret + native compile/run without python3 (backend=go + optional llvm link + full-program llvm + vec loops + mlir dialect + mlir→llvm + full-program mlir + vec8/vec16 tiles + portable/HW VLEN + runtime feature dispatch + dialect rewrites/CSE/vwidth/strength/span_alg/bitplane/nested + dyn tiles + residual/auto peel + store peels + tree/gemm folds + adaptive tiles + map/store peels + soft_mux forest + scan/reduce peels + deep forest + scan-mul + AdaptivePeelChunks + wide ladder + fused scan-reduce + dyn32 + VL>32 + multi-out fused + RDVL + pair/tuple + live RDVL + lang pair + SME VL + pair-of-pairs exhaust + ZA outer-acc + weighted multi-candidate fusion + ADT forest zip/map folds + multi-tile auto peels + fusion PGO window + keyed/multi reduce + VL-fb/live peels + PGO decay/EMA + map+reduce + simd bitplanes + kv cache + deterministic sample + cuda path + ptx layer + vocab/gguf decode + linear_trit simd + bpe sidecar + session linear_trit + trit prepack cache + decode trit cache + layer stack trit + attn trit + decode attn trit + attn stack trit + decode stack trit + gqa stack trit + decode gqa trit + gqa swiglu prefill + tied alibi + llama namemap)"
