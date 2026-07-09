"""Recursive-descent parser for Trin v0.1."""

from __future__ import annotations

from typing import List, Optional

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
    Param,
    Program,
    ReturnStmt,
    SpanLit,
    Ternary,
    TypeAnn,
    Unary,
    Var,
)
from .lexer import LexError, Tok, Token, tokenize


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.i = 0

    @property
    def cur(self) -> Token:
        return self.tokens[self.i]

    def check(self, *kinds: Tok) -> bool:
        return self.cur.kind in kinds

    def advance(self) -> Token:
        tok = self.cur
        if tok.kind is not Tok.EOF:
            self.i += 1
        return tok

    def expect(self, kind: Tok, msg: str = "") -> Token:
        if self.cur.kind is not kind:
            t = self.cur
            raise ParseError(
                msg or f"expected {kind.name}, got {t.kind.name} ({t.value!r}) at {t.line}:{t.col}"
            )
        return self.advance()

    def parse(self) -> Program:
        items = []
        while not self.check(Tok.EOF):
            items.append(self.parse_item())
        return Program(items=items, line=1, col=1)

    def parse_item(self):
        if self.check(Tok.FN):
            return self.parse_fn()
        if self.check(Tok.EXTERN):
            return self.parse_extern()
        if self.check(Tok.LET):
            return self.parse_let()
        t = self.cur
        raise ParseError(f"expected item at {t.line}:{t.col}")

    def parse_extern(self) -> ExternDecl:
        tok = self.expect(Tok.EXTERN)
        self.expect(Tok.FN)
        name = self.expect(Tok.IDENT).value
        self.expect(Tok.LPAREN)
        params = self.parse_params()
        self.expect(Tok.RPAREN)
        ret = None
        if self.check(Tok.ARROW):
            self.advance()
            ret = self.parse_type()
        self.expect(Tok.SEMI)
        return ExternDecl(name=name, params=params, ret=ret, line=tok.line, col=tok.col)

    def parse_fn(self) -> FnDef:
        tok = self.expect(Tok.FN)
        name = self.expect(Tok.IDENT).value
        self.expect(Tok.LPAREN)
        params = self.parse_params()
        self.expect(Tok.RPAREN)
        ret = None
        if self.check(Tok.ARROW):
            self.advance()
            ret = self.parse_type()
        body = self.parse_block()
        return FnDef(name=name, params=params, ret=ret, body=body, line=tok.line, col=tok.col)

    def parse_params(self) -> List[Param]:
        if self.check(Tok.RPAREN):
            return []
        params = [self.parse_param()]
        while self.check(Tok.COMMA):
            self.advance()
            params.append(self.parse_param())
        return params

    def parse_param(self) -> Param:
        name_tok = self.expect(Tok.IDENT)
        typ = None
        if self.check(Tok.COLON):
            self.advance()
            typ = self.parse_type()
        return Param(name=name_tok.value, typ=typ, line=name_tok.line, col=name_tok.col)

    def parse_type(self) -> TypeAnn:
        t = self.cur
        if self.check(Tok.VOID, Tok.BIT, Tok.TRIT, Tok.UNIT, Tok.STR, Tok.I32, Tok.F32):
            name = self.advance().value
            return TypeAnn(name=name, line=t.line, col=t.col)
        if self.check(Tok.SPAN):
            self.advance()
            self.expect(Tok.LBRACK)
            inner = self.parse_type()
            self.expect(Tok.RBRACK)
            return TypeAnn(name="span", inner=inner, line=t.line, col=t.col)
        if self.check(Tok.FN):
            self.advance()
            self.expect(Tok.LPAREN)
            params: List[TypeAnn] = []
            if not self.check(Tok.RPAREN):
                params.append(self.parse_type())
                while self.check(Tok.COMMA):
                    self.advance()
                    params.append(self.parse_type())
            self.expect(Tok.RPAREN)
            self.expect(Tok.ARROW)
            ret = self.parse_type()
            return TypeAnn(name="fn", params=params, ret=ret, line=t.line, col=t.col)
        if self.check(Tok.IDENT):
            name = self.advance().value
            return TypeAnn(name=name, line=t.line, col=t.col)
        raise ParseError(f"expected type at {t.line}:{t.col}")

    def parse_block(self) -> Block:
        tok = self.expect(Tok.LBRACE)
        stmts = []
        expr = None
        while not self.check(Tok.RBRACE, Tok.EOF):
            if self.check(Tok.LET):
                stmts.append(self.parse_let())
            elif self.check(Tok.RETURN):
                stmts.append(self.parse_return())
            else:
                # Lookahead: expression, maybe trailing semicolon (stmt) or final expr
                e = self.parse_expr()
                if self.check(Tok.SEMI):
                    self.advance()
                    stmts.append(ExprStmt(expr=e, line=e.line, col=e.col))
                elif self.check(Tok.RBRACE):
                    expr = e
                    break
                else:
                    # treat as stmt without semi (forgiving) only if next is clearly new stmt/end
                    if self.check(Tok.LET, Tok.RETURN, Tok.RBRACE):
                        stmts.append(ExprStmt(expr=e, line=e.line, col=e.col))
                    else:
                        # force semi
                        raise ParseError(
                            f"expected ';' or end of block after expression at {self.cur.line}:{self.cur.col}"
                        )
        self.expect(Tok.RBRACE)
        return Block(stmts=stmts, expr=expr, line=tok.line, col=tok.col)

    def parse_let(self) -> LetStmt:
        tok = self.expect(Tok.LET)
        name = self.expect(Tok.IDENT).value
        typ = None
        if self.check(Tok.COLON):
            self.advance()
            typ = self.parse_type()
        self.expect(Tok.EQ)
        value = self.parse_expr()
        self.expect(Tok.SEMI)
        return LetStmt(name=name, typ=typ, value=value, line=tok.line, col=tok.col)

    def parse_return(self) -> ReturnStmt:
        tok = self.expect(Tok.RETURN)
        value = None
        if not self.check(Tok.SEMI):
            value = self.parse_expr()
        self.expect(Tok.SEMI)
        return ReturnStmt(value=value, line=tok.line, col=tok.col)

    # --- expressions: ternary is loosest ---

    def parse_expr(self) -> Expr:
        return self.parse_ternary()

    def parse_ternary(self) -> Expr:
        left = self.parse_or()
        if self.check(Tok.PIPE):
            self.advance()
            pure = self.parse_or()
            self.expect(Tok.PIPE, "ternary requires gate | pure | each")
            each = self.parse_or()
            # allow chained nesting on the right via recursion already inside pure/each
            return Ternary(gate=left, pure=pure, each=each, line=left.line, col=left.col)
        return left

    def parse_or(self) -> Expr:
        left = self.parse_and()
        while self.check(Tok.OROR):
            op_tok = self.advance()
            right = self.parse_and()
            # sugar: g || e  =>  g | 1 | e  (when used as gate-ish); general: binary or for units
            left = Binary(op="or", left=left, right=right, line=op_tok.line, col=op_tok.col)
        return left

    def parse_and(self) -> Expr:
        left = self.parse_cmp()
        while self.check(Tok.ANDAND):
            op_tok = self.advance()
            right = self.parse_cmp()
            left = Binary(op="and", left=left, right=right, line=op_tok.line, col=op_tok.col)
        return left

    def parse_cmp(self) -> Expr:
        left = self.parse_add()
        if self.check(Tok.LT, Tok.GT, Tok.LE, Tok.GE, Tok.EQEQ, Tok.NE):
            op_map = {
                Tok.LT: "<",
                Tok.GT: ">",
                Tok.LE: "<=",
                Tok.GE: ">=",
                Tok.EQEQ: "==",
                Tok.NE: "!=",
            }
            op_tok = self.advance()
            right = self.parse_add()
            left = Binary(op=op_map[op_tok.kind], left=left, right=right, line=op_tok.line, col=op_tok.col)
        return left

    def parse_add(self) -> Expr:
        left = self.parse_mul()
        while self.check(Tok.PLUS, Tok.MINUS):
            op_tok = self.advance()
            right = self.parse_mul()
            left = Binary(
                op="+" if op_tok.kind is Tok.PLUS else "-",
                left=left,
                right=right,
                line=op_tok.line,
                col=op_tok.col,
            )
        return left

    def parse_mul(self) -> Expr:
        left = self.parse_unary()
        while self.check(Tok.STAR, Tok.SLASH):
            op_tok = self.advance()
            right = self.parse_unary()
            left = Binary(
                op="*" if op_tok.kind is Tok.STAR else "/",
                left=left,
                right=right,
                line=op_tok.line,
                col=op_tok.col,
            )
        return left

    def parse_unary(self) -> Expr:
        if self.check(Tok.BANG, Tok.MINUS, Tok.PLUS):
            op_tok = self.advance()
            expr = self.parse_unary()
            op = "!" if op_tok.kind is Tok.BANG else ("-" if op_tok.kind is Tok.MINUS else "+")
            return Unary(op=op, expr=expr, line=op_tok.line, col=op_tok.col)
        return self.parse_postfix()

    def parse_postfix(self) -> Expr:
        expr = self.parse_primary()
        while True:
            if self.check(Tok.LPAREN):
                lparen = self.advance()
                args: List[Expr] = []
                if not self.check(Tok.RPAREN):
                    args.append(self.parse_expr())
                    while self.check(Tok.COMMA):
                        self.advance()
                        args.append(self.parse_expr())
                self.expect(Tok.RPAREN)
                expr = Call(callee=expr, args=args, line=lparen.line, col=lparen.col)
            elif self.check(Tok.LBRACK):
                lb = self.advance()
                idx = self.parse_expr()
                self.expect(Tok.RBRACK)
                expr = Index(target=expr, index=idx, line=lb.line, col=lb.col)
            else:
                break
        return expr

    def parse_primary(self) -> Expr:
        t = self.cur
        if self.check(Tok.INT):
            self.advance()
            return Literal(value=t.value, kind="int", line=t.line, col=t.col)
        if self.check(Tok.FLOAT):
            self.advance()
            return Literal(value=t.value, kind="float", line=t.line, col=t.col)
        if self.check(Tok.STRING):
            self.advance()
            return Literal(value=t.value, kind="str", line=t.line, col=t.col)
        if self.check(Tok.TRUE, Tok.FALSE):
            self.advance()
            return Literal(value=bool(t.value), kind="bool", line=t.line, col=t.col)
        if self.check(Tok.IDENT):
            self.advance()
            return Var(name=t.value, line=t.line, col=t.col)
        # Type-name keywords may appear as value-level builtin constructors (trit/bit/unit/…)
        if self.check(
            Tok.TRIT, Tok.BIT, Tok.UNIT, Tok.STR, Tok.I32, Tok.F32, Tok.VOID, Tok.SPAN
        ):
            self.advance()
            return Var(name=str(t.value), line=t.line, col=t.col)
        if self.check(Tok.LPAREN):
            # (params) => expr   OR   (expr)
            self.advance()
            # lambda detection: non-destructive lookahead, then reparse
            save = self.i
            is_lambda = self._try_lambda_after_lparen()
            self.i = save
            if is_lambda:
                return self._parse_lambda_after_lparen(t.line, t.col)
            if self.check(Tok.RPAREN):
                # empty parens only valid as () => expr (handled above) or error
                self.advance()
                if self.check(Tok.FATARROW):
                    self.advance()
                    body = self.parse_expr()
                    return Lambda(params=[], body=body, line=t.line, col=t.col)
                raise ParseError(f"empty parentheses at {t.line}:{t.col}")
            expr = self.parse_expr()
            self.expect(Tok.RPAREN)
            return expr
        if self.check(Tok.LBRACK):
            self.advance()
            elems: List[Expr] = []
            if not self.check(Tok.RBRACK):
                elems.append(self.parse_expr())
                while self.check(Tok.COMMA):
                    self.advance()
                    elems.append(self.parse_expr())
            self.expect(Tok.RBRACK)
            return SpanLit(elems=elems, line=t.line, col=t.col)
        if self.check(Tok.LBRACE):
            return self.parse_block()  # type: ignore[return-value]
        raise ParseError(f"unexpected token {t.kind.name} ({t.value!r}) at {t.line}:{t.col}")

    def _try_lambda_after_lparen(self) -> bool:
        # already past '('
        if self.check(Tok.RPAREN):
            return False
        if not self.check(Tok.IDENT):
            return False
        self.advance()
        if self.check(Tok.COLON):
            return True  # typed param starts lambda
        if self.check(Tok.COMMA):
            return True
        if self.check(Tok.RPAREN):
            self.advance()
            ok = self.check(Tok.FATARROW)
            return ok
        return False

    def _parse_lambda_after_lparen(self, line: int, col: int) -> Lambda:
        params: List[Param] = []
        if not self.check(Tok.RPAREN):
            params.append(self.parse_param())
            while self.check(Tok.COMMA):
                self.advance()
                params.append(self.parse_param())
        self.expect(Tok.RPAREN)
        self.expect(Tok.FATARROW)
        body = self.parse_expr()
        return Lambda(params=params, body=body, line=line, col=col)


def parse_source(source: str) -> Program:
    try:
        tokens = tokenize(source)
    except LexError as e:
        raise ParseError(str(e)) from e
    return Parser(tokens).parse()
