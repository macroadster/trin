"""Trin v0.3 interpreter — evaluates gate | pure | each."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, IO, List, Optional, Union

from .ast_nodes import (
    Binary,
    Block,
    Call,
    Expr,
    ExprStmt,
    ExternDecl,
    FnDef,
    Index,
    Lambda,
    LetStmt,
    Literal,
    Program,
    ReturnStmt,
    SpanLit,
    Ternary,
    Unary,
    Var,
)
from .parser import ParseError, parse_source


class TrinError(Exception):
    pass


class ReturnSignal(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


@dataclass
class Bit:
    v: int  # 0 or 1

    def __repr__(self) -> str:
        return f"bit({self.v})"


@dataclass
class Trit:
    v: int  # -1, 0, 1

    def __repr__(self) -> str:
        return f"trit({self.v})"


@dataclass
class Unit:
    v: float

    def __post_init__(self) -> None:
        # soft clamp for numerical safety in spike
        if self.v < -1e-5 or self.v > 1 + 1e-5:
            pass  # allow; harden uses clamp
        self.v = float(self.v)

    def __repr__(self) -> str:
        return f"unit({self.v})"


@dataclass
class Span:
    items: List[Any]

    def __repr__(self) -> str:
        return f"span{self.items!r}"


@dataclass
class Closure:
    params: List[str]
    body: Expr
    env: "Env"

    def __repr__(self) -> str:
        return f"<fn/{len(self.params)}>"


@dataclass
class Env:
    parent: Optional["Env"] = None
    bindings: Dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> Any:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.get(name)
        raise TrinError(f"undefined name: {name}")

    def set(self, name: str, value: Any) -> None:
        self.bindings[name] = value

    def child(self) -> "Env":
        return Env(parent=self)


def gate_weight(v: Any) -> float:
    if isinstance(v, Bit):
        return float(v.v)
    if isinstance(v, Unit):
        return max(0.0, min(1.0, float(v.v)))
    if isinstance(v, Trit):
        return { -1: 0.0, 0: 0.5, 1: 1.0 }[int(v.v)]
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, int) and v in (0, 1):
        # allow raw bool-like ints from comparisons stored as Bit only — reject plain int gates
        raise TrinError(f"value of type int cannot be used as gate: {v!r} (use bit/unit/trit)")
    raise TrinError(f"value cannot be used as gate: {type(v).__name__} {v!r}")


def identity_for(v: Any) -> Any:
    """Monoid-ish identity when hard-false short-circuits; for strings we prefer default via each arm."""
    if isinstance(v, Unit):
        return Unit(0.0)
    if isinstance(v, Bit):
        return Bit(0)
    if isinstance(v, Trit):
        return Trit(0)
    if isinstance(v, (int, float)):
        return type(v)(0)
    if isinstance(v, str):
        return ""
    if isinstance(v, Span):
        return Span([])
    if v is None:
        return None
    return None


def blend(w: float, a: Any, b: Any) -> Any:
    if isinstance(a, Unit) or isinstance(b, Unit):
        av = a.v if isinstance(a, Unit) else float(a)
        bv = b.v if isinstance(b, Unit) else float(b)
        return Unit(w * av + (1.0 - w) * bv)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return w * float(a) + (1.0 - w) * float(b)
    if isinstance(a, Bit) and isinstance(b, Bit):
        return Bit(1 if (w * a.v + (1 - w) * b.v) >= 0.5 else 0)
    if isinstance(a, str) and isinstance(b, str):
        return a if w >= 0.5 else b
    if isinstance(a, Span) and isinstance(b, Span):
        if len(a.items) == len(b.items):
            return Span([blend(w, x, y) for x, y in zip(a.items, b.items)])
        return a if w >= 0.5 else b
    return a if w >= 0.5 else b


def is_truthy_bit(v: Any) -> Bit:
    if isinstance(v, Bit):
        return v
    if isinstance(v, bool):
        return Bit(1 if v else 0)
    if isinstance(v, Unit):
        return Bit(1 if v.v >= 0.5 else 0)
    if isinstance(v, (int, float)):
        return Bit(1 if v else 0)
    if isinstance(v, str):
        return Bit(1 if len(v) > 0 else 0)
    if isinstance(v, Span):
        return Bit(1 if len(v.items) > 0 else 0)
    if v is None:
        return Bit(0)
    return Bit(1)


class Interpreter:
    def __init__(
        self,
        stdin: Optional[IO[str]] = None,
        stdout: Optional[IO[str]] = None,
        stderr: Optional[IO[str]] = None,
    ) -> None:
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self.global_env = Env()
        self._install_builtins()

    def _install_builtins(self) -> None:
        g = self.global_env

        def _print(s: Any) -> None:
            self.stdout.write(str(s))
            self.stdout.flush()
            return None

        def _println(s: Any = "") -> None:
            self.stdout.write(str(s) + "\n")
            self.stdout.flush()
            return None

        def _input(prompt: Any = "") -> str:
            self.stdout.write(str(prompt))
            self.stdout.flush()
            line = self.stdin.readline()
            if line == "":
                return ""
            return line.rstrip("\r\n")

        def _str_concat(a: Any, b: Any) -> str:
            def _s(x: Any) -> str:
                if isinstance(x, (Bit, Unit, Trit)):
                    return str(x.v)
                return str(x)

            return _s(a) + _s(b)

        def _str_len(s: Any) -> int:
            return len(str(s))

        def _len(xs: Any) -> int:
            if isinstance(xs, Span):
                return len(xs.items)
            if isinstance(xs, str):
                return len(xs)
            raise TrinError("len expects span or str")

        def _dens(xs: Any) -> Unit:
            if isinstance(xs, Span):
                return Unit(0.0 if not xs.items else 1.0)
            raise TrinError("dens expects span")

        def _sum(xs: Any) -> Any:
            if not isinstance(xs, Span):
                raise TrinError("sum expects span")
            if not xs.items:
                return 0
            acc = xs.items[0]
            for x in xs.items[1:]:
                acc = self._add(acc, x)
            return acc

        def _harden(u: Any) -> Bit:
            if isinstance(u, Unit):
                return Bit(1 if u.v >= 0.5 else 0)
            if isinstance(u, Bit):
                return u
            return Bit(1 if float(u) >= 0.5 else 0)

        def _soften(b: Any) -> Unit:
            if isinstance(b, Bit):
                return Unit(float(b.v))
            if isinstance(b, Unit):
                return b
            return Unit(1.0 if b else 0.0)

        def _unit_of(x: Any) -> Unit:
            return Unit(max(0.0, min(1.0, float(x))))

        def _map(f: Any, xs: Any) -> Span:
            if not isinstance(xs, Span):
                raise TrinError("map expects span")
            return Span([self._apply(f, [x]) for x in xs.items])

        def _bit(x: Any) -> Bit:
            return is_truthy_bit(x)

        def _trit(x: Any) -> Trit:
            n = int(x.v if isinstance(x, (Bit, Trit, Unit)) else x)
            if n < -1:
                n = -1
            if n > 1:
                n = 1
            return Trit(n)

        def _to_str(x: Any) -> str:
            if isinstance(x, Span):
                inner = ", ".join(_to_str(i) for i in x.items)
                return f"[{inner}]"
            if isinstance(x, (Bit, Unit, Trit)):
                return str(x.v)
            if x is None:
                return ""
            return str(x)

        def _zip(a: Any, b: Any) -> Span:
            if not isinstance(a, Span) or not isinstance(b, Span):
                raise TrinError("zip expects two spans")
            n = min(len(a.items), len(b.items))
            return Span([Span([a.items[i], b.items[i]]) for i in range(n)])

        def _zip_with(f: Any, a: Any, b: Any) -> Span:
            if not isinstance(a, Span) or not isinstance(b, Span):
                raise TrinError("zip_with expects two spans")
            n = min(len(a.items), len(b.items))
            return Span([self._apply(f, [a.items[i], b.items[i]]) for i in range(n)])

        def _as_float_list(xs: Any) -> List[float]:
            if not isinstance(xs, Span):
                raise TrinError("expected span")
            out: List[float] = []
            for x in xs.items:
                if isinstance(x, (Bit, Unit, Trit)):
                    out.append(float(x.v))
                else:
                    out.append(float(x))
            return out

        def _scale(s: Any, xs: Any) -> Span:
            sf = float(s.v if isinstance(s, (Bit, Unit, Trit)) else s)
            vals = _as_float_list(xs)
            return Span([sf * v for v in vals])

        def _span_add(a: Any, b: Any) -> Span:
            av, bv = _as_float_list(a), _as_float_list(b)
            n = min(len(av), len(bv))
            return Span([av[i] + bv[i] for i in range(n)])

        def _span_mul(a: Any, b: Any) -> Span:
            av, bv = _as_float_list(a), _as_float_list(b)
            n = min(len(av), len(bv))
            return Span([av[i] * bv[i] for i in range(n)])

        def _dot(a: Any, b: Any) -> float:
            av, bv = _as_float_list(a), _as_float_list(b)
            n = min(len(av), len(bv))
            acc = 0.0
            for i in range(n):
                acc += av[i] * bv[i]
            return acc

        def _trit_val(x: Any) -> int:
            if isinstance(x, Trit):
                return int(x.v)
            if isinstance(x, Bit):
                return int(x.v)
            if isinstance(x, Unit):
                # soft quantize to trit
                u = x.v
                if u < 1.0 / 3.0:
                    return -1
                if u > 2.0 / 3.0:
                    return 1
                return 0
            n = int(x)
            if n < -1:
                return -1
            if n > 1:
                return 1
            return n

        def _trit_gemm(A: Any, B: Any, M: Any, K: Any, N: Any) -> Span:
            """Row-major trit GEMM: C[M,N] = A[M,K] @ B[K,N], products in {-1,0,1}, accumulate i32."""
            if not isinstance(A, Span) or not isinstance(B, Span):
                raise TrinError("trit_gemm expects span[trit] A and B (row-major flat)")
            m = int(M.v if isinstance(M, (Bit, Unit, Trit)) else M)
            k = int(K.v if isinstance(K, (Bit, Unit, Trit)) else K)
            n = int(N.v if isinstance(N, (Bit, Unit, Trit)) else N)
            if m < 0 or k < 0 or n < 0:
                raise TrinError("trit_gemm dims must be non-negative")
            if len(A.items) < m * k or len(B.items) < k * n:
                raise TrinError(
                    f"trit_gemm shape mismatch: need {m*k}+{k*n} elems, got {len(A.items)}+{len(B.items)}"
                )
            out: List[int] = []
            for i in range(m):
                for j in range(n):
                    acc = 0
                    for t in range(k):
                        av = _trit_val(A.items[i * k + t])
                        bv = _trit_val(B.items[t * n + j])
                        # trit multiply: -1/0/+1 closed; matches sign multiply
                        acc += av * bv
                    out.append(acc)
            return Span(out)

        def _span_to_trits(xs: Any) -> List[int]:
            if not isinstance(xs, Span):
                raise TrinError("expected span of trits")
            return [_trit_val(x) for x in xs.items]

        def _trit_pack(xs: Any) -> Span:
            """Return span[span[bit]] = [pos_bits, neg_bits] elementwise planes."""
            from .bitplane import pack_trits

            pos, neg = pack_trits(_span_to_trits(xs))
            return Span([Span([Bit(p) for p in pos]), Span([Bit(n) for n in neg])])

        def _trit_unpack(planes: Any) -> Span:
            """planes = [pos_span, neg_span] → span[trit]."""
            from .bitplane import unpack_trits

            if not isinstance(planes, Span) or len(planes.items) != 2:
                raise TrinError("trit_unpack expects span of two bit spans [pos, neg]")
            pos_s, neg_s = planes.items
            if not isinstance(pos_s, Span) or not isinstance(neg_s, Span):
                raise TrinError("trit_unpack plane elements must be spans")
            pos = [int(x.v if isinstance(x, (Bit, Trit, Unit)) else x) for x in pos_s.items]
            neg = [int(x.v if isinstance(x, (Bit, Trit, Unit)) else x) for x in neg_s.items]
            return Span([Trit(t) for t in unpack_trits(pos, neg)])

        def _trit_gemm_bitplane(A: Any, B: Any, M: Any, K: Any, N: Any) -> Span:
            from .bitplane import trit_gemm_bitplane

            if not isinstance(A, Span) or not isinstance(B, Span):
                raise TrinError("trit_gemm_bitplane expects span A/B")
            m = int(M.v if isinstance(M, (Bit, Unit, Trit)) else M)
            k = int(K.v if isinstance(K, (Bit, Unit, Trit)) else K)
            n = int(N.v if isinstance(N, (Bit, Unit, Trit)) else N)
            av = _span_to_trits(A)
            bv = _span_to_trits(B)
            return Span(trit_gemm_bitplane(av, bv, m, k, n))

        builtins: Dict[str, Callable[..., Any]] = {
            "print": _print,
            "println": _println,
            "input": _input,
            "str_concat": _str_concat,
            "str_len": _str_len,
            "len": _len,
            "dens": _dens,
            "sum": _sum,
            "harden": _harden,
            "soften": _soften,
            "unit_of": _unit_of,
            "map": _map,
            "bit": _bit,
            "trit": _trit,
            "to_str": _to_str,
            "zip": _zip,
            "zip_with": _zip_with,
            "scale": _scale,
            "span_add": _span_add,
            "span_mul": _span_mul,
            "dot": _dot,
            "trit_gemm": _trit_gemm,
            "trit_pack": _trit_pack,
            "trit_unpack": _trit_unpack,
            "trit_gemm_bitplane": _trit_gemm_bitplane,
        }
        for name, fn in builtins.items():
            g.set(name, fn)

    def run_program(self, prog: Program) -> Any:
        # load top-level fns and lets
        for item in prog.items:
            if isinstance(item, FnDef):
                assert item.body is not None
                params = [p.name for p in item.params]
                self.global_env.set(
                    item.name,
                    Closure(params=params, body=item.body, env=self.global_env),
                )
            elif isinstance(item, LetStmt):
                assert item.value is not None
                self.global_env.set(item.name, self.eval_expr(item.value, self.global_env))
            elif isinstance(item, ExternDecl):
                # externs are expected to already exist as builtins/host
                if item.name not in self.global_env.bindings and (
                    self.global_env.parent is None
                ):
                    # allow missing until call
                    pass
            else:
                raise TrinError(f"unknown top-level item: {item}")

        # if main exists, call it
        try:
            main = self.global_env.get("main")
        except TrinError:
            main = None
        if main is not None:
            return self._apply(main, [])
        return None

    def eval_block(self, block: Block, env: Env) -> Any:
        local = env.child()
        try:
            for stmt in block.stmts:
                if isinstance(stmt, LetStmt):
                    assert stmt.value is not None
                    local.set(stmt.name, self.eval_expr(stmt.value, local))
                elif isinstance(stmt, ExprStmt):
                    assert stmt.expr is not None
                    self.eval_expr(stmt.expr, local)
                elif isinstance(stmt, ReturnStmt):
                    val = None if stmt.value is None else self.eval_expr(stmt.value, local)
                    raise ReturnSignal(val)
                else:
                    raise TrinError(f"bad stmt {stmt}")
            if block.expr is not None:
                return self.eval_expr(block.expr, local)
            return None
        except ReturnSignal as rs:
            return rs.value

    def eval_expr(self, expr: Expr, env: Env) -> Any:
        if isinstance(expr, Literal):
            if expr.kind == "bool":
                return Bit(1 if expr.value else 0)
            if expr.kind == "int":
                return int(expr.value)  # type: ignore[arg-type]
            if expr.kind == "float":
                return float(expr.value)  # type: ignore[arg-type]
            if expr.kind == "str":
                return str(expr.value)
            return expr.value

        if isinstance(expr, Var):
            return env.get(expr.name)

        if isinstance(expr, Unary):
            v = self.eval_expr(expr.expr, env)  # type: ignore[arg-type]
            if expr.op == "!":
                if isinstance(v, Unit):
                    return Unit(1.0 - max(0.0, min(1.0, v.v)))
                b = is_truthy_bit(v)
                return Bit(1 - b.v)
            if expr.op == "-":
                if isinstance(v, Unit):
                    return Unit(-v.v)
                return -v
            if expr.op == "+":
                return v
            raise TrinError(f"bad unary {expr.op}")

        if isinstance(expr, Binary):
            # short-circuit ternary sugar for and/or on gates
            if expr.op == "and":
                left = self.eval_expr(expr.left, env)  # type: ignore[arg-type]
                w = gate_weight(is_truthy_bit(left) if not isinstance(left, (Bit, Unit, Trit)) else left)
                if w <= 1e-12:
                    if isinstance(left, Unit):
                        return Unit(0.0)
                    return Bit(0)
                right = self.eval_expr(expr.right, env)  # type: ignore[arg-type]
                if isinstance(left, Unit) or isinstance(right, Unit):
                    return Unit(gate_weight(left) * gate_weight(right))
                return Bit(1 if gate_weight(left) >= 0.5 and gate_weight(right) >= 0.5 else 0)
            if expr.op == "or":
                left = self.eval_expr(expr.left, env)  # type: ignore[arg-type]
                w = gate_weight(is_truthy_bit(left) if not isinstance(left, (Bit, Unit, Trit)) else left)
                if w >= 1.0 - 1e-12:
                    if isinstance(left, Unit):
                        return Unit(1.0)
                    return Bit(1)
                right = self.eval_expr(expr.right, env)  # type: ignore[arg-type]
                if isinstance(left, Unit) or isinstance(right, Unit):
                    a, b = gate_weight(left), gate_weight(right)
                    return Unit(1.0 - (1.0 - a) * (1.0 - b))
                return Bit(1 if gate_weight(left) >= 0.5 or gate_weight(right) >= 0.5 else 0)

            left = self.eval_expr(expr.left, env)  # type: ignore[arg-type]
            right = self.eval_expr(expr.right, env)  # type: ignore[arg-type]
            return self._binop(expr.op, left, right)

        if isinstance(expr, Ternary):
            return self.eval_ternary(expr, env)

        if isinstance(expr, Call):
            callee = self.eval_expr(expr.callee, env)  # type: ignore[arg-type]
            args = [self.eval_expr(a, env) for a in expr.args]
            return self._apply(callee, args)

        if isinstance(expr, Index):
            target = self.eval_expr(expr.target, env)  # type: ignore[arg-type]
            idx = self.eval_expr(expr.index, env)  # type: ignore[arg-type]
            i = int(idx)
            if isinstance(target, Span):
                if i < 0 or i >= len(target.items):
                    raise TrinError(f"index {i} out of bounds")
                return target.items[i]
            if isinstance(target, str):
                return target[i]
            raise TrinError("index on non-span")

        if isinstance(expr, Lambda):
            params = [p.name for p in expr.params]
            assert expr.body is not None
            return Closure(params=params, body=expr.body, env=env)

        if isinstance(expr, SpanLit):
            return Span([self.eval_expr(e, env) for e in expr.elems])

        if isinstance(expr, Block):
            return self.eval_block(expr, env)

        raise TrinError(f"cannot evaluate {type(expr).__name__}")

    def eval_ternary(self, expr: Ternary, env: Env) -> Any:
        """Core rule: gate | pure | each."""
        assert expr.gate and expr.pure and expr.each
        gv = self.eval_expr(expr.gate, env)
        # comparisons return Bit; units/trits ok
        if not isinstance(gv, (Bit, Unit, Trit)):
            gv = is_truthy_bit(gv)
        w = gate_weight(gv)

        # hard false — if false return (identity); still: if each is default branch for strings
        # Spec: hard false → identity; hard true → pure; soft → blend.
        # For hello-world ergonomics, bit false takes the *each* arm as the else branch.
        # SPEC §5 Case A says identity; §8 example uses false → stranger via each.
        # Resolution: hard bit/trit uses classic if/else (pure vs each); soft unit blends.
        if isinstance(gv, (Bit, Trit)) or (isinstance(gv, Unit) and (w <= 1e-12 or w >= 1 - 1e-12)):
            if w >= 1.0 - 1e-12:
                return self.eval_expr(expr.pure, env)
            if w <= 1e-12:
                # "if false return / else path": evaluate each as the false branch default
                return self.eval_expr(expr.each, env)
        # soft residual blend
        pure_v = self.eval_expr(expr.pure, env)
        each_v = self.eval_expr(expr.each, env)
        return blend(w, pure_v, each_v)

    def _binop(self, op: str, left: Any, right: Any) -> Any:
        if op in ("<", ">", "<=", ">=", "==", "!="):
            # compare unwrapped
            lv = left.v if isinstance(left, (Bit, Unit, Trit)) else left
            rv = right.v if isinstance(right, (Bit, Unit, Trit)) else right
            ops = {
                "<": lv < rv,
                ">": lv > rv,
                "<=": lv <= rv,
                ">=": lv >= rv,
                "==": lv == rv,
                "!=": lv != rv,
            }
            return Bit(1 if ops[op] else 0)
        if op == "+":
            return self._add(left, right)
        if op == "-":
            if isinstance(left, Unit) or isinstance(right, Unit):
                return Unit(float(getattr(left, "v", left)) - float(getattr(right, "v", right)))
            return left - right
        if op == "*":
            if isinstance(left, Unit) or isinstance(right, Unit):
                return Unit(float(getattr(left, "v", left)) * float(getattr(right, "v", right)))
            return left * right
        if op == "/":
            rv = float(getattr(right, "v", right))
            if rv == 0:
                raise TrinError("division by zero")
            if isinstance(left, Unit) or isinstance(right, Unit):
                return Unit(float(getattr(left, "v", left)) / rv)
            return left / right
        raise TrinError(f"bad op {op}")

    def _add(self, left: Any, right: Any) -> Any:
        if isinstance(left, str) or isinstance(right, str):
            return str(left) + str(right)
        if isinstance(left, Unit) or isinstance(right, Unit):
            return Unit(float(getattr(left, "v", left)) + float(getattr(right, "v", right)))
        return left + right

    def _apply(self, callee: Any, args: List[Any]) -> Any:
        if callable(callee) and not isinstance(callee, Closure):
            return callee(*args)
        if isinstance(callee, Closure):
            if len(args) != len(callee.params):
                raise TrinError(
                    f"arity mismatch: expected {len(callee.params)} args, got {len(args)}"
                )
            local = callee.env.child()
            for name, val in zip(callee.params, args):
                local.set(name, val)
            body = callee.body
            if isinstance(body, Block):
                return self.eval_block(body, local)
            return self.eval_expr(body, local)
        raise TrinError(f"not callable: {callee!r}")


def run_source(
    source: str,
    stdin: Optional[IO[str]] = None,
    stdout: Optional[IO[str]] = None,
) -> Any:
    prog = parse_source(source)
    interp = Interpreter(stdin=stdin, stdout=stdout)
    return interp.run_program(prog)


def run_file(
    path: Union[str, Path],
    stdin: Optional[IO[str]] = None,
    stdout: Optional[IO[str]] = None,
) -> Any:
    p = Path(path)
    return run_source(p.read_text(encoding="utf-8"), stdin=stdin, stdout=stdout)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(
            "usage: python -m trin_lang [--check] <file.trin>\n"
            "  --check   run static typechecker only (no execute)",
            file=sys.stderr,
        )
        return 2
    check_only = False
    if argv[0] == "--check":
        check_only = True
        argv = argv[1:]
        if not argv:
            print("usage: python -m trin_lang [--check] <file.trin>", file=sys.stderr)
            return 2
    path = argv[0]
    try:
        if check_only:
            from .typecheck import typecheck_source

            src = Path(path).read_text(encoding="utf-8")
            errs = typecheck_source(src)
            if errs:
                for e in errs:
                    print(f"error: {e}", file=sys.stderr)
                return 1
            print(f"ok: {path} typechecks")
            return 0
        run_file(path)
        return 0
    except (ParseError, TrinError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        # include LexError without walrus portability issues
        from .lexer import LexError

        if isinstance(e, LexError):
            print(f"error: {e}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
