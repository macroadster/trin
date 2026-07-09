"""Trin type system — structural types for v0.2 typechecker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Type:
    def __str__(self) -> str:
        return self.render()

    def render(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Prim(Type):
    name: str  # void, bit, trit, unit, i32, f32, str, any, never

    def render(self) -> str:
        return self.name


@dataclass(frozen=True)
class SpanT(Type):
    elem: Type

    def render(self) -> str:
        return f"span[{self.elem}]"


@dataclass(frozen=True)
class FnT(Type):
    params: tuple  # Tuple[Type, ...]
    ret: Type

    def render(self) -> str:
        ps = ", ".join(str(p) for p in self.params)
        return f"fn({ps}) -> {self.ret}"


VOID = Prim("void")
BIT = Prim("bit")
TRIT = Prim("trit")
UNIT = Prim("unit")
I32 = Prim("i32")
F32 = Prim("f32")
STR = Prim("str")
ANY = Prim("any")  # under-constrained / builtin poly
NEVER = Prim("never")


GATE_TYPES = {BIT, TRIT, UNIT}


def is_gate(t: Type) -> bool:
    return t in GATE_TYPES or t is ANY


def is_numeric(t: Type) -> bool:
    return t in (I32, F32, UNIT, BIT, TRIT) or t is ANY


def is_addable(t: Type) -> bool:
    return is_numeric(t) or t is STR or t is ANY


def unify(a: Type, b: Type) -> Optional[Type]:
    """Return a common type if a and b are compatible, else None."""
    if a is ANY:
        return b
    if b is ANY:
        return a
    if a is NEVER:
        return b
    if b is NEVER:
        return a
    if a == b:
        return a
    # numeric lattice promotions
    if {a, b} <= {I32, F32}:
        return F32 if F32 in (a, b) else I32
    if {a, b} <= {I32, UNIT, F32}:
        if UNIT in (a, b):
            return UNIT if a is UNIT and b is UNIT else F32
        return F32
    if isinstance(a, SpanT) and isinstance(b, SpanT):
        inner = unify(a.elem, b.elem)
        return SpanT(inner) if inner else None
    if isinstance(a, FnT) and isinstance(b, FnT):
        if len(a.params) != len(b.params):
            return None
        ps = []
        for p, q in zip(a.params, b.params):
            u = unify(p, q)
            if u is None:
                return None
            ps.append(u)
        r = unify(a.ret, b.ret)
        return FnT(tuple(ps), r) if r else None
    return None


def assignable(src: Type, dst: Type) -> bool:
    if dst is ANY or src is ANY or src is NEVER:
        return True
    if src == dst:
        return True
    if dst is F32 and src is I32:
        return True
    if dst is I32 and src is BIT:
        return True
    if dst is UNIT and src in (BIT, I32, F32):
        return True  # via unit_of semantics at runtime; static soft allow for annotated lets
    if isinstance(dst, SpanT) and isinstance(src, SpanT):
        return assignable(src.elem, dst.elem)
    if isinstance(dst, FnT) and isinstance(src, FnT):
        if len(dst.params) != len(src.params):
            return False
        # contravariant params, covariant ret (simplified: invariant params)
        for p, q in zip(src.params, dst.params):
            if not assignable(q, p):
                return False
        return assignable(src.ret, dst.ret)
    return False
