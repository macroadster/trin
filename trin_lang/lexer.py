"""Lexer for Trin v0.1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator, List, Optional


class Tok(Enum):
    EOF = auto()
    IDENT = auto()
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    # keywords
    FN = auto()
    LET = auto()
    EXTERN = auto()
    RETURN = auto()
    SPAN = auto()
    UNIT = auto()
    BIT = auto()
    TRIT = auto()
    STR = auto()
    I32 = auto()
    F32 = auto()
    VOID = auto()
    TRUE = auto()
    FALSE = auto()
    # symbols
    PIPE = auto()  # |
    EQ = auto()
    ARROW = auto()  # ->
    FATARROW = auto()  # =>
    DOT = auto()
    COMMA = auto()
    COLON = auto()
    SEMI = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACK = auto()
    RBRACK = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    EQEQ = auto()
    NE = auto()
    BANG = auto()
    ANDAND = auto()
    OROR = auto()


KEYWORDS = {
    "fn": Tok.FN,
    "let": Tok.LET,
    "extern": Tok.EXTERN,
    "return": Tok.RETURN,
    "span": Tok.SPAN,
    "unit": Tok.UNIT,
    "bit": Tok.BIT,
    "trit": Tok.TRIT,
    "str": Tok.STR,
    "i32": Tok.I32,
    "f32": Tok.F32,
    "void": Tok.VOID,
    "true": Tok.TRUE,
    "false": Tok.FALSE,
    "and": Tok.ANDAND,
    "or": Tok.OROR,
}


@dataclass
class Token:
    kind: Tok
    value: object
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.kind.name}, {self.value!r}, @{self.line}:{self.col})"


class LexError(Exception):
    pass


def tokenize(source: str) -> List[Token]:
    return list(Lexer(source))


class Lexer:
    def __init__(self, source: str) -> None:
        self.src = source
        self.i = 0
        self.line = 1
        self.col = 1
        self.n = len(source)

    def __iter__(self) -> Iterator[Token]:
        while True:
            tok = self.next_token()
            yield tok
            if tok.kind is Tok.EOF:
                break

    def peek(self) -> str:
        return self.src[self.i] if self.i < self.n else "\0"

    def peek2(self) -> str:
        j = self.i + 1
        return self.src[j] if j < self.n else "\0"

    def advance(self) -> str:
        ch = self.peek()
        self.i += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def next_token(self) -> Token:
        self.skip_ws_and_comments()
        line, col = self.line, self.col
        ch = self.peek()
        if ch == "\0":
            return Token(Tok.EOF, None, line, col)

        # two-char ops
        two = ch + self.peek2()
        if two == "->":
            self.advance()
            self.advance()
            return Token(Tok.ARROW, two, line, col)
        if two == "=>":
            self.advance()
            self.advance()
            return Token(Tok.FATARROW, two, line, col)
        if two == "==":
            self.advance()
            self.advance()
            return Token(Tok.EQEQ, two, line, col)
        if two == "!=":
            self.advance()
            self.advance()
            return Token(Tok.NE, two, line, col)
        if two == "<=":
            self.advance()
            self.advance()
            return Token(Tok.LE, two, line, col)
        if two == ">=":
            self.advance()
            self.advance()
            return Token(Tok.GE, two, line, col)
        if two == "&&":
            self.advance()
            self.advance()
            return Token(Tok.ANDAND, two, line, col)
        if two == "||":
            self.advance()
            self.advance()
            return Token(Tok.OROR, two, line, col)

        single = {
            "|": Tok.PIPE,
            "=": Tok.EQ,
            ".": Tok.DOT,
            ",": Tok.COMMA,
            ":": Tok.COLON,
            ";": Tok.SEMI,
            "(": Tok.LPAREN,
            ")": Tok.RPAREN,
            "{": Tok.LBRACE,
            "}": Tok.RBRACE,
            "[": Tok.LBRACK,
            "]": Tok.RBRACK,
            "+": Tok.PLUS,
            "-": Tok.MINUS,
            "*": Tok.STAR,
            "/": Tok.SLASH,
            "<": Tok.LT,
            ">": Tok.GT,
            "!": Tok.BANG,
        }
        if ch in single:
            self.advance()
            return Token(single[ch], ch, line, col)

        if ch == '"':
            return self.string(line, col)

        if ch.isdigit() or (ch == "." and self.peek2().isdigit()):
            return self.number(line, col)

        if ch.isalpha() or ch == "_":
            return self.ident(line, col)

        raise LexError(f"unexpected character {ch!r} at {line}:{col}")

    def skip_ws_and_comments(self) -> None:
        while True:
            ch = self.peek()
            if ch in " \t\r\n":
                self.advance()
                continue
            if ch == "/" and self.peek2() == "/":
                while self.peek() not in ("\n", "\0"):
                    self.advance()
                continue
            if ch == "/" and self.peek2() == "*":
                self.advance()
                self.advance()
                while True:
                    if self.peek() == "\0":
                        raise LexError("unterminated block comment")
                    if self.peek() == "*" and self.peek2() == "/":
                        self.advance()
                        self.advance()
                        break
                    self.advance()
                continue
            break

    def string(self, line: int, col: int) -> Token:
        self.advance()  # "
        buf: List[str] = []
        while True:
            ch = self.peek()
            if ch == "\0":
                raise LexError(f"unterminated string at {line}:{col}")
            if ch == '"':
                self.advance()
                break
            if ch == "\\":
                self.advance()
                esc = self.advance()
                mapping = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"'}
                if esc not in mapping:
                    raise LexError(f"bad escape \\{esc} at {line}:{col}")
                buf.append(mapping[esc])
            else:
                buf.append(self.advance())
        return Token(Tok.STRING, "".join(buf), line, col)

    def number(self, line: int, col: int) -> Token:
        start = self.i
        is_float = False
        while self.peek().isdigit():
            self.advance()
        if self.peek() == "." and self.peek2().isdigit():
            is_float = True
            self.advance()
            while self.peek().isdigit():
                self.advance()
        text = self.src[start : self.i]
        if is_float:
            return Token(Tok.FLOAT, float(text), line, col)
        return Token(Tok.INT, int(text), line, col)

    def ident(self, line: int, col: int) -> Token:
        start = self.i
        while self.peek().isalnum() or self.peek() == "_":
            self.advance()
        text = self.src[start : self.i]
        kind = KEYWORDS.get(text, Tok.IDENT)
        if kind is Tok.TRUE:
            return Token(Tok.TRUE, True, line, col)
        if kind is Tok.FALSE:
            return Token(Tok.FALSE, False, line, col)
        if kind is Tok.IDENT:
            return Token(Tok.IDENT, text, line, col)
        return Token(kind, text, line, col)
