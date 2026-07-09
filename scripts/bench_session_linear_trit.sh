#!/usr/bin/env bash
# Microbench: free-span + session-backed scalar vs packed linear_trit
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${PATH}:/home/nanobot/.local/go/bin"
cd "$ROOT"

echo "[0] build trin + fixture"
go build -o trin ./cmd/trin
go run scripts/gen_trit_session_fixture.go .

echo "[1] go unit microbench (scalar vs planes)"
go test ./internal/weights -run TestLinearTritBenchPaths -count=1 -v | tee /tmp/trin_lt_bench_unit.txt

echo "[2] interpret session path (parity)"
./trin run examples/weight_session_linear_trit.trin | tee /tmp/trin_sess_lt_i.txt
grep -q 'weight_session_linear_trit_ok' /tmp/trin_sess_lt_i.txt
grep -q 'parity_ok=1' /tmp/trin_sess_lt_i.txt

echo "[3] native free-span bench binary wall time"
./trin build examples/linear_trit_bench.trin -o build/linear_trit_bench --backend go
START=$(date +%s%N)
./build/linear_trit_bench | tee /tmp/trin_lt_bench_n.txt
END=$(date +%s%N)
python3 - <<PY
start=int("$START"); end=int("$END")
print(f"native_linear_trit_bench_elapsed_sec={(end-start)/1e9:.6f}")
PY
grep -q 'linear_trit_bench_ok' /tmp/trin_lt_bench_n.txt
grep -q 'parity_ok=1' /tmp/trin_lt_bench_n.txt

echo "[4] native session path wall time"
./trin build examples/weight_session_linear_trit.trin -o build/weight_session_linear_trit --backend go
START=$(date +%s%N)
./build/weight_session_linear_trit | tee /tmp/trin_sess_lt_n.txt
END=$(date +%s%N)
python3 - <<PY
start=int("$START"); end=int("$END")
print(f"native_session_linear_trit_elapsed_sec={(end-start)/1e9:.6f}")
PY
grep -q 'weight_session_linear_trit_ok' /tmp/trin_sess_lt_n.txt
grep -q 'parity_ok=1' /tmp/trin_sess_lt_n.txt

echo "OK: session linear_trit + microbench"
