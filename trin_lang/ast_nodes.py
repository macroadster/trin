"""AST nodes for Trin v0.1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class Node:
    line: int = 0
    col: int = 0


@dataclass
class Program(Node):
    items: List[Union["FnDef", "LetStmt", "ExternDecl"]] = field(default_factory=list)


@dataclass
class ExternDecl(Node):
    name: str = ""
    params: List["Param"] = field(default_factory=list)
    ret: Optional["TypeAnn"] = None


@dataclass
class FnDef(Node):
    name: str = ""
    params: List["Param"] = field(default_factory=list)
    ret: Optional["TypeAnn"] = None
    body: Optional["Block"] = None


@dataclass
class Param(Node):
    name: str = ""
    typ: Optional["TypeAnn"] = None


@dataclass
class TypeAnn(Node):
    name: str = ""  # void, bit, unit, str, i32, f32, span, fn
    inner: Optional["TypeAnn"] = None
    params: List["TypeAnn"] = field(default_factory=list)
    ret: Optional["TypeAnn"] = None


@dataclass
class Block(Node):
    stmts: List[Union["LetStmt", "ExprStmt", "ReturnStmt"]] = field(default_factory=list)
    expr: Optional["Expr"] = None


@dataclass
class LetStmt(Node):
    name: str = ""
    typ: Optional[TypeAnn] = None
    value: Optional["Expr"] = None


@dataclass
class ExprStmt(Node):
    expr: Optional["Expr"] = None


@dataclass
class ReturnStmt(Node):
    value: Optional["Expr"] = None


# --- expressions ---


@dataclass
class Expr(Node):
    pass


@dataclass
class Ternary(Expr):
    gate: Optional[Expr] = None
    pure: Optional[Expr] = None
    each: Optional[Expr] = None


@dataclass
class Binary(Expr):
    op: str = ""
    left: Optional[Expr] = None
    right: Optional[Expr] = None


@dataclass
class Unary(Expr):
    op: str = ""
    expr: Optional[Expr] = None


@dataclass
class Call(Expr):
    callee: Optional[Expr] = None
    args: List[Expr] = field(default_factory=list)


@dataclass
class Index(Expr):
    target: Optional[Expr] = None
    index: Optional[Expr] = None


@dataclass
class Var(Expr):
    name: str = ""


@dataclass
class Literal(Expr):
    value: object = None
    kind: str = ""  # int, float, str, bool


@dataclass
class Lambda(Expr):
    params: List[Param] = field(default_factory=list)
    body: Optional[Expr] = None


@dataclass
class SpanLit(Expr):
    elems: List[Expr] = field(default_factory=list)


@dataclass
class AssignLike:
    """Not used — lets are statements."""
