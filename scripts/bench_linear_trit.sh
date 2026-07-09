#!/usr/bin/env bash
# Micro-bench: interpret linear_trit on growing K
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${PATH}:/home/nanobot/.local/go/bin"
cd "$ROOT"
go build -o trin ./cmd/trin
cat > /tmp/bench_lt.trin <<'T'
fn main() {
    let w = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    let q = quantize_absmean(w);
    let tw = q[0];
    let s = q[1];
    let x = [0.5, -0.5, 0.25, -0.25, 0.5, -0.5, 0.25, -0.25, 0.5, -0.5, 0.25, -0.25, 0.5, -0.5, 0.25, -0.25];
    let y = linear_trit(tw, x, 1, 16, s);
    println(str_concat("y0=", to_str(y[0])));
}
T
./trin build /tmp/bench_lt.trin -o /tmp/bench_lt --backend go
/usr/bin/time -f 'elapsed_sec=%e' /tmp/bench_lt
