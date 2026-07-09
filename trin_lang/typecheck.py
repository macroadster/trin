"""Static typechecker for Trin v0.2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    TypeAnn,
    Unary,
    Var,
)
from .typesys import (
    ANY,
    BIT,
    F32,
    I32,
    NEVER,
    STR,
    TRIT,
    UNIT,
    VOID,
    FnT,
    Prim,
    SpanT,
    Type,
    assignable,
    is_gate,
    is_numeric,
    unify,
)


@dataclass
class TypeError(Exception):
    message: str
    line: int = 0
    col: int = 0

    def __str__(self) -> str:
        if self.line:
            return f"{self.line}:{self.col}: {self.message}"
        return self.message


@dataclass
class TypeEnv:
    parent: Optional["TypeEnv"] = None
    bindings: Dict[str, Type] = field(default_factory=dict)

    def get(self, name: str) -> Optional[Type]:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.get(name)
        return None

    def set(self, name: str, t: Type) -> None:
        self.bindings[name] = t

    def child(self) -> "TypeEnv":
        return TypeEnv(parent=self)


def ann_to_type(ann: Optional[TypeAnn]) -> Type:
    if ann is None:
        return ANY
    name = ann.name
    if name in ("void", "bit", "trit", "unit", "i32", "f32", "str"):
        return Prim(name)
    if name == "span":
        return SpanT(ann_to_type(ann.inner) if ann.inner else ANY)
    if name == "fn":
        params = tuple(ann_to_type(p) for p in ann.params)
        ret = ann_to_type(ann.ret) if ann.ret else VOID
        return FnT(params, ret)
    return ANY


class Typechecker:
    def __init__(self) -> None:
        self.errors: List[TypeError] = []
        self.global_env = TypeEnv()
        self._install_builtins()

    def _install_builtins(self) -> None:
        g = self.global_env
        g.set("print", FnT((ANY,), VOID))
        g.set("println", FnT((ANY,), VOID))
        g.set("input", FnT((STR,), STR))
        g.set("str_concat", FnT((STR, STR), STR))
        g.set("str_len", FnT((STR,), I32))
        g.set("len", FnT((ANY,), I32))
        g.set("dens", FnT((SpanT(ANY),), UNIT))
        g.set("sum", FnT((SpanT(ANY),), ANY))
        g.set("harden", FnT((UNIT,), BIT))
        g.set("soften", FnT((BIT,), UNIT))
        g.set("unit_of", FnT((ANY,), UNIT))  # accepts i32/f32; clamps at runtime
        g.set("map", FnT((FnT((ANY,), ANY), SpanT(ANY)), SpanT(ANY)))
        g.set("bit", FnT((ANY,), BIT))
        g.set("trit", FnT((I32,), TRIT))
        g.set("zip", FnT((SpanT(ANY), SpanT(ANY)), SpanT(ANY)))
        g.set("zip_with", FnT((FnT((ANY, ANY), ANY), SpanT(ANY), SpanT(ANY)), SpanT(ANY)))
        g.set("scale", FnT((F32, SpanT(F32)), SpanT(F32)))
        g.set("dot", FnT((SpanT(F32), SpanT(F32)), F32))
        g.set("span_add", FnT((SpanT(F32), SpanT(F32)), SpanT(F32)))
        g.set("span_mul", FnT((SpanT(F32), SpanT(F32)), SpanT(F32)))
        g.set("trit_gemm", FnT((SpanT(TRIT), SpanT(TRIT), I32, I32, I32), SpanT(I32)))
        g.set("trit_gemm_bitplane", FnT((SpanT(TRIT), SpanT(TRIT), I32, I32, I32), SpanT(I32)))
        g.set("trit_pack", FnT((SpanT(TRIT),), SpanT(SpanT(BIT))))
        g.set("trit_unpack", FnT((SpanT(SpanT(BIT)),), SpanT(TRIT)))
        g.set("to_str", FnT((ANY,), STR))

    def error(self, msg: str, node=None) -> Type:
        line = getattr(node, "line", 0) or 0
        col = getattr(node, "col", 0) or 0
        self.errors.append(TypeError(msg, line, col))
        return NEVER

    def check_program(self, prog: Program) -> List[TypeError]:
        for item in prog.items:
            if isinstance(item, FnDef):
                params = tuple(ann_to_type(p.typ) for p in item.params)
                ret = ann_to_type(item.ret) if item.ret else ANY
                self.global_env.set(item.name, FnT(params, ret))
            elif isinstance(item, ExternDecl):
                params = tuple(ann_to_type(p.typ) for p in item.params)
                ret = ann_to_type(item.ret) if item.ret else ANY
                self.global_env.set(item.name, FnT(params, ret))
            elif isinstance(item, LetStmt):
                assert item.value is not None
                vt = self.infer(item.value, self.global_env)
                if item.typ is not None:
                    at = ann_to_type(item.typ)
                    if not assignable(vt, at):
                        self.error(f"let {item.name}: expected {at}, got {vt}", item)
                    self.global_env.set(item.name, at)
                else:
                    self.global_env.set(item.name, vt)

        for item in prog.items:
            if isinstance(item, FnDef) and item.body is not None:
                self.check_fn(item)
        return self.errors

    def check_fn(self, fn: FnDef) -> None:
        env = self.global_env.child()
        for p in fn.params:
            env.set(p.name, ann_to_type(p.typ))
        expected = ann_to_type(fn.ret) if fn.ret else ANY
        got = self.check_block(fn.body, env, expected)  # type: ignore[arg-type]
        if fn.ret is not None and got is not NEVER and not assignable(got, expected):
            self.error(f"function {fn.name}: body has type {got}, expected {expected}", fn)

    def check_block(self, block: Block, env: TypeEnv, expected_ret: Type) -> Type:
        local = env.child()
        last: Type = VOID
        for stmt in block.stmts:
            if isinstance(stmt, LetStmt):
                assert stmt.value is not None
                vt = self.infer(stmt.value, local)
                if stmt.typ is not None:
                    at = ann_to_type(stmt.typ)
                    if not assignable(vt, at) and vt is not NEVER:
                        self.error(f"let {stmt.name}: expected {at}, got {vt}", stmt)
                    local.set(stmt.name, at)
                else:
                    local.set(stmt.name, vt)
                last = VOID
            elif isinstance(stmt, ExprStmt):
                assert stmt.expr is not None
                self.infer(stmt.expr, local)
                last = VOID
            elif isinstance(stmt, ReturnStmt):
                if stmt.value is None:
                    if expected_ret not in (VOID, ANY) and expected_ret is not NEVER:
                        self.error(f"return void but expected {expected_ret}", stmt)
                    last = VOID
                else:
                    rt = self.infer(stmt.value, local)
                    if (
                        expected_ret is not ANY
                        and not assignable(rt, expected_ret)
                        and rt is not NEVER
                    ):
                        self.error(
                            f"return type {rt} not assignable to {expected_ret}", stmt
                        )
                    last = rt
        if block.expr is not None:
            last = self.infer(block.expr, local)
        return last

    def infer(self, expr: Expr, env: TypeEnv) -> Type:
        if isinstance(expr, Literal):
            if expr.kind == "int":
                return I32
            if expr.kind == "float":
                return F32
            if expr.kind == "str":
                return STR
            if expr.kind == "bool":
                return BIT
            return ANY

        if isinstance(expr, Var):
            t = env.get(expr.name)
            if t is None:
                return self.error(f"undefined name '{expr.name}'", expr)
            return t

        if isinstance(expr, Unary):
            t = self.infer(expr.expr, env)  # type: ignore[arg-type]
            if expr.op == "!":
                if (
                    not is_gate(t)
                    and t not in (BIT, UNIT, ANY, NEVER)
                    and not (isinstance(t, Prim) and t.name in ("bit", "unit", "trit"))
                ):
                    if t not in (I32,):
                        self.error(f"unary ! expects gate, got {t}", expr)
                return UNIT if t is UNIT else BIT
            if expr.op in ("-", "+"):
                if not is_numeric(t) and t is not NEVER:
                    self.error(f"unary {expr.op} expects numeric, got {t}", expr)
                return t
            return ANY

        if isinstance(expr, Binary):
            if expr.op in ("and", "or"):
                lt = self.infer(expr.left, env)  # type: ignore[arg-type]
                rt = self.infer(expr.right, env)  # type: ignore[arg-type]
                if not is_gate(lt) and lt is not NEVER:
                    self.error(f"{expr.op} left operand not a gate: {lt}", expr)
                if not is_gate(rt) and rt is not NEVER:
                    self.error(f"{expr.op} right operand not a gate: {rt}", expr)
                if lt is UNIT or rt is UNIT:
                    return UNIT
                return BIT
            if expr.op in ("<", ">", "<=", ">=", "==", "!="):
                self.infer(expr.left, env)  # type: ignore[arg-type]
                self.infer(expr.right, env)  # type: ignore[arg-type]
                return BIT
            lt = self.infer(expr.left, env)  # type: ignore[arg-type]
            rt = self.infer(expr.right, env)  # type: ignore[arg-type]
            if expr.op == "+":
                if lt is STR or rt is STR:
                    return STR
                u = unify(lt, rt)
                if u is None and lt is not NEVER and rt is not NEVER:
                    self.error(f"cannot add {lt} and {rt}", expr)
                    return NEVER
                return u or ANY
            if expr.op in ("-", "*", "/"):
                u = unify(lt, rt)
                if u is None and lt is not NEVER and rt is not NEVER:
                    if not (is_numeric(lt) and is_numeric(rt)):
                        self.error(f"cannot apply {expr.op} to {lt} and {rt}", expr)
                        return NEVER
                    return F32
                return u or F32
            return ANY

        if isinstance(expr, Ternary):
            gt = self.infer(expr.gate, env)  # type: ignore[arg-type]
            if gt is STR or (
                isinstance(gt, Prim)
                and gt.name not in ("bit", "unit", "trit", "any", "never")
                and gt not in (BIT, UNIT, TRIT, ANY, NEVER)
            ):
                # reject obvious non-gates; allow BIT from comparisons
                if not is_gate(gt) and gt is not NEVER and gt is not BIT:
                    if isinstance(gt, Prim) and gt.name == "i32":
                        self.error(
                            f"ternary gate must be bit|unit|trit, got {gt}", expr
                        )
                    elif gt is STR:
                        self.error(
                            f"ternary gate must be bit|unit|trit, got {gt}", expr
                        )
            pt = self.infer(expr.pure, env)  # type: ignore[arg-type]
            et = self.infer(expr.each, env)  # type: ignore[arg-type]
            u = unify(pt, et)
            if u is None:
                if pt is NEVER:
                    return et
                if et is NEVER:
                    return pt
                self.error(f"ternary arms disagree: pure={pt}, each={et}", expr)
                return NEVER
            return u

        if isinstance(expr, Call):
            ct = self.infer(expr.callee, env)  # type: ignore[arg-type]
            args = [self.infer(a, env) for a in expr.args]
            if isinstance(expr.callee, Var):
                return self._check_builtin_call(expr.callee.name, args, ct, expr)
            if isinstance(ct, FnT):
                if len(args) != len(ct.params):
                    self.error(
                        f"arity mismatch: expected {len(ct.params)} args, got {len(args)}",
                        expr,
                    )
                    return ct.ret
                for i, (a, p) in enumerate(zip(args, ct.params)):
                    if not assignable(a, p) and a is not NEVER and p is not ANY:
                        self.error(f"arg {i}: expected {p}, got {a}", expr)
                return ct.ret
            if ct is ANY or ct is NEVER:
                return ANY
            self.error(f"cannot call non-function type {ct}", expr)
            return NEVER

        if isinstance(expr, Index):
            tt = self.infer(expr.target, env)  # type: ignore[arg-type]
            it = self.infer(expr.index, env)  # type: ignore[arg-type]
            if (
                it not in (I32, ANY, NEVER)
                and not (isinstance(it, Prim) and it.name == "i32")
            ):
                self.error(f"index must be i32, got {it}", expr)
            if isinstance(tt, SpanT):
                return tt.elem
            if tt is STR:
                return STR
            if tt is ANY:
                return ANY
            self.error(f"cannot index type {tt}", expr)
            return NEVER

        if isinstance(expr, Lambda):
            local = env.child()
            pts = []
            for p in expr.params:
                pt = ann_to_type(p.typ)
                local.set(p.name, pt)
                pts.append(pt)
            body_t = self.infer(expr.body, local)  # type: ignore[arg-type]
            return FnT(tuple(pts), body_t)

        if isinstance(expr, SpanLit):
            if not expr.elems:
                return SpanT(ANY)
            ts = [self.infer(e, env) for e in expr.elems]
            acc = ts[0]
            for t in ts[1:]:
                u = unify(acc, t)
                if u is None:
                    self.error(f"span elements disagree: {acc} vs {t}", expr)
                    return SpanT(NEVER)
                acc = u
            return SpanT(acc)

        if isinstance(expr, Block):
            return self.check_block(expr, env, ANY)

        return self.error(f"cannot type expression {type(expr).__name__}", expr)

    def _check_builtin_call(
        self, name: str, args: List[Type], ct: Type, expr: Call
    ) -> Type:
        if name == "map":
            if len(args) != 2:
                self.error("map expects 2 args", expr)
                return SpanT(ANY)
            f, xs = args
            if isinstance(xs, SpanT) and isinstance(f, FnT) and f.params:
                return SpanT(f.ret)
            if isinstance(xs, SpanT):
                return SpanT(ANY)
            if xs is not ANY and xs is not NEVER:
                self.error(f"map expects span, got {xs}", expr)
            return SpanT(ANY)
        if name == "sum":
            if len(args) == 1 and isinstance(args[0], SpanT):
                return args[0].elem if args[0].elem is not ANY else I32
            if len(args) == 1 and args[0] is not ANY and not isinstance(args[0], SpanT):
                self.error(f"sum expects span, got {args[0]}", expr)
            return ANY
        if name == "zip":
            return SpanT(ANY)
        if name == "zip_with":
            if len(args) == 3 and isinstance(args[0], FnT):
                return SpanT(args[0].ret)
            return SpanT(ANY)
        if name == "scale":
            return SpanT(F32)
        if name in ("span_add", "span_mul"):
            return SpanT(F32)
        if name == "dot":
            return F32
        if name in ("trit_gemm", "trit_gemm_bitplane"):
            if len(args) != 5:
                self.error(f"{name}(A, B, M, K, N) expects 5 args", expr)
            return SpanT(I32)
        if name == "trit_pack":
            return SpanT(SpanT(BIT))
        if name == "trit_unpack":
            return SpanT(TRIT)
        if name == "trit":
            return TRIT
        if name == "to_str":
            return STR
        if name == "dens":
            return UNIT
        if name in ("len", "str_len"):
            return I32
        if name == "harden":
            return BIT
        if name in ("soften", "unit_of"):
            return UNIT
        if name == "bit":
            return BIT
        if name in ("print", "println"):
            return VOID
        if name == "input":
            return STR
        if name == "str_concat":
            return STR
        if isinstance(ct, FnT):
            if len(args) != len(ct.params):
                if name == "println" and len(args) == 0:
                    return VOID
                self.error(
                    f"{name}: expected {len(ct.params)} args, got {len(args)}",
                    expr,
                )
            return ct.ret
        return ANY


def typecheck_source(source: str) -> List[TypeError]:
    from .parser import parse_source

    prog = parse_source(source)
    return Typechecker().check_program(prog)


def typecheck_program(prog: Program) -> List[TypeError]:
    return Typechecker().check_program(prog)
