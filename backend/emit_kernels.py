"""
Emit C demos from Trin programs that use span / trit_gemm builtins (v0.3).

Not a full compiler: pattern-matches Call to known kernels and writes a
standalone C main that includes kernels.h with concrete constant operands
when available, else a generic kernel harness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trin_lang.ast_nodes import (  # noqa: E402
    Call,
    Expr,
    FnDef,
    LetStmt,
    Literal,
    Program,
    SpanLit,
    Ternary,
    Var,
)
from trin_lang.parser import parse_source  # noqa: E402


def _lit_int(e: Expr) -> Optional[int]:
    if isinstance(e, Literal) and e.kind == "int":
        return int(e.value)
    from trin_lang.ast_nodes import Unary

    if isinstance(e, Unary) and e.op == "-" and isinstance(e.expr, Literal):
        if e.expr.kind == "int":
            return -int(e.expr.value)
    return None


def _span_trits(e: Expr) -> Optional[List[int]]:
    """Extract flat trit list from [trit(1), trit(0), ...] or [1, -1, ...]."""
    if not isinstance(e, SpanLit):
        return None
    out: List[int] = []
    for el in e.elems:
        if isinstance(el, Call) and isinstance(el.callee, Var) and el.callee.name == "trit":
            if el.args:
                v = _lit_int(el.args[0])
                if v is None:
                    return None
                out.append(max(-1, min(1, v)))
                continue
        v = _lit_int(el)
        if v is None:
            return None
        out.append(max(-1, min(1, int(v))))
    return out


def _span_f32(e: Expr) -> Optional[List[float]]:
    if not isinstance(e, SpanLit):
        return None
    out: List[float] = []
    for el in e.elems:
        if isinstance(el, Literal):
            if el.kind in ("int", "float"):
                out.append(float(el.value))
            else:
                return None
        else:
            v = _lit_int(el)
            if v is None:
                return None
            out.append(float(v))
    return out


def _find_lets(prog: Program) -> dict:
    lets = {}
    for item in prog.items:
        if isinstance(item, FnDef) and item.body:
            for s in item.body.stmts:
                if isinstance(s, LetStmt) and s.value is not None:
                    lets[s.name] = s.value
        if isinstance(item, LetStmt) and item.value is not None:
            lets[item.name] = item.value
    return lets


def _resolve_span(expr: Expr, lets: dict) -> Optional[Expr]:
    if isinstance(expr, Var) and expr.name in lets:
        return lets[expr.name]
    return expr


def detect_kernel_calls(prog: Program) -> List[Tuple[str, Any]]:
    """Return list of (kind, payload) for emission."""
    lets = _find_lets(prog)
    found: List[Tuple[str, Any]] = []

    def walk(e: Optional[Expr]) -> None:
        if e is None:
            return
        if isinstance(e, Call) and isinstance(e.callee, Var):
            name = e.callee.name
            if name == "trit_gemm" and len(e.args) == 5:
                A = _span_trits(_resolve_span(e.args[0], lets) or e.args[0])
                B = _span_trits(_resolve_span(e.args[1], lets) or e.args[1])
                M = _lit_int(e.args[2])
                K = _lit_int(e.args[3])
                N = _lit_int(e.args[4])
                if A is not None and B is not None and None not in (M, K, N):
                    found.append(
                        (
                            "trit_gemm",
                            {"A": A, "B": B, "M": M, "K": K, "N": N},
                        )
                    )
            if name in ("span_add", "span_mul", "dot") and len(e.args) >= 2:
                a = _span_f32(_resolve_span(e.args[0], lets) or e.args[0])
                b = _span_f32(_resolve_span(e.args[1], lets) or e.args[1])
                if a is not None and b is not None:
                    found.append((name, {"a": a, "b": b}))
            if name == "scale" and len(e.args) == 2:
                s = None
                if isinstance(e.args[0], Literal):
                    s = float(e.args[0].value)
                xs = _span_f32(_resolve_span(e.args[1], lets) or e.args[1])
                if s is not None and xs is not None:
                    found.append(("scale", {"s": s, "xs": xs}))
            for a in e.args:
                walk(a)
            walk(e.callee)
            return
        if isinstance(e, SpanLit):
            for el in e.elems:
                walk(el)
            return
        if isinstance(e, Ternary):
            walk(e.gate)
            walk(e.pure)
            walk(e.each)
            return
        # generic attrs
        for attr in ("left", "right", "expr", "body", "value", "index", "target"):
            if hasattr(e, attr):
                walk(getattr(e, attr))
        if hasattr(e, "args") and isinstance(getattr(e, "args"), list):
            for a in e.args:
                walk(a)
        if hasattr(e, "elems"):
            for el in e.elems:
                walk(el)

    for item in prog.items:
        if isinstance(item, FnDef) and item.body:
            for s in item.body.stmts:
                if isinstance(s, LetStmt):
                    walk(s.value)
                elif hasattr(s, "expr"):
                    walk(getattr(s, "expr"))
            walk(item.body.expr)
        elif isinstance(item, LetStmt):
            walk(item.value)
    return found


def emit_c_from_kernels(kernels: List[Tuple[str, Any]], title: str = "trin_kernel_emit") -> str:
    lines: List[str] = []
    lines.append(f"/* Generated by Trin emit_kernels v0.3 — {title} */")
    lines.append('#include <stdio.h>')
    lines.append('#include <stdint.h>')
    lines.append('#include "kernels.h"')
    lines.append("")
    lines.append("int main(void) {")
    if not kernels:
        lines.append("    /* no recognized span/trit_gemm constants; emit harness */")
        lines.append("    float a[4] = {1.f, 2.f, 3.f, 4.f};")
        lines.append("    float b[4] = {1.f, 1.f, 1.f, 1.f};")
        lines.append("    float out[4];")
        lines.append("    trin_span_add_f32(a, b, out, 4);")
        lines.append('    printf("span_add harness: %.0f\\n", out[0]);')
        lines.append("    int8_t A[4] = {1,0,0,1};")
        lines.append("    int8_t B[4] = {1,-1,0,1};")
        lines.append("    int32_t C[4];")
        lines.append("    trin_trit_gemm_bitplane(A, B, C, 2, 2, 2);")
        lines.append('    printf("bitplane gemm: %d %d %d %d\\n", C[0], C[1], C[2], C[3]);')
        lines.append("    return 0;")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    idx = 0
    for kind, payload in kernels:
        if kind == "trit_gemm":
            A, B = payload["A"], payload["B"]
            M, K, N = payload["M"], payload["K"], payload["N"]
            lines.append(f"    /* trit_gemm bitplane path {M}x{K}x{N} */")
            lines.append(
                "    int8_t A{i}[{n}] = {{{vals}}};".format(
                    i=idx,
                    n=len(A),
                    vals=", ".join(str(x) for x in A),
                )
            )
            lines.append(
                "    int8_t B{i}[{n}] = {{{vals}}};".format(
                    i=idx,
                    n=len(B),
                    vals=", ".join(str(x) for x in B),
                )
            )
            lines.append(f"    int32_t C{idx}[{M * N}];")
            lines.append(
                f"    trin_trit_gemm(A{idx}, B{idx}, C{idx}, {M}, {K}, {N});"
            )
            lines.append(
                f"    trin_trit_gemm_bitplane(A{idx}, B{idx}, C{idx}, {M}, {K}, {N});"
            )
            lines.append(
                f'    printf("C{idx}:");'
            )
            lines.append(f"    for (int k = 0; k < {M * N}; k++) printf(\" %d\", C{idx}[k]);")
            lines.append('    printf("\\n");')
            idx += 1
        elif kind in ("span_add", "span_mul", "dot"):
            a, b = payload["a"], payload["b"]
            n = min(len(a), len(b))
            lines.append(f"    /* {kind} */")
            lines.append(
                "    float a{i}[{n}] = {{{vals}}};".format(
                    i=idx, n=n, vals=", ".join(f"{x}f" for x in a[:n])
                )
            )
            lines.append(
                "    float b{i}[{n}] = {{{vals}}};".format(
                    i=idx, n=n, vals=", ".join(f"{x}f" for x in b[:n])
                )
            )
            if kind == "dot":
                lines.append(f"    float d{idx} = trin_dot_f32(a{idx}, b{idx}, {n});")
                lines.append(f'    printf("dot{idx}: %f\\n", d{idx});')
            else:
                lines.append(f"    float o{idx}[{n}];")
                fn = "trin_span_add_f32" if kind == "span_add" else "trin_span_mul_f32"
                lines.append(f"    {fn}(a{idx}, b{idx}, o{idx}, {n});")
                lines.append(f'    printf("{kind}{idx}:");')
                lines.append(
                    f"    for (size_t k = 0; k < {n}; k++) printf(\" %f\", o{idx}[k]);"
                )
                lines.append('    printf("\\n");')
            idx += 1
        elif kind == "scale":
            s, xs = payload["s"], payload["xs"]
            n = len(xs)
            lines.append(f"    float x{idx}[{n}] = {{{', '.join(f'{v}f' for v in xs)}}};")
            lines.append(f"    float y{idx}[{n}];")
            lines.append(f"    trin_span_scale_f32({s}f, x{idx}, y{idx}, {n});")
            lines.append(f'    printf("scale{idx}: %f\\n", y{idx}[0]);')
            idx += 1

    lines.append("    return 0;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def emit_file(path: Path, out: Optional[Path] = None) -> Path:
    src = path.read_text(encoding="utf-8")
    prog = parse_source(src)
    kernels = detect_kernel_calls(prog)
    c_src = emit_c_from_kernels(kernels, title=path.name)
    if out is None:
        out = path.with_name(path.stem + "_kernels.c")
    out.write_text(c_src, encoding="utf-8")
    # ensure kernels.h nearby
    kh = Path(__file__).with_name("kernels.h")
    if kh.exists():
        (out.parent / "kernels.h").write_text(kh.read_text(encoding="utf-8"), encoding="utf-8")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Emit C kernel demos from Trin sources")
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)
    out = emit_file(args.input, args.output)
    print(f"wrote {out}")
    if (out.parent / "kernels.h").exists():
        print(f"wrote {out.parent / 'kernels.h'}")
    print(f"compile: cc -O3 -march=native -o kernel_out {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
