package token

type Kind int

const (
	EOF Kind = iota
	IDENT
	INT
	FLOAT
	STRING
	FN
	LET
	EXTERN
	RETURN
	WHEN
	DO
	OVER
	AS
	FOR
	IN
	SPAN
	UNIT
	BIT
	TRIT
	STR
	I32
	F32
	VOID
	TRUE
	FALSE
	PIPE
	EQ
	ARROW
	FATARROW
	DOT
	COMMA
	COLON
	SEMI
	LPAREN
	RPAREN
	LBRACE
	RBRACE
	LBRACK
	RBRACK
	PLUS
	MINUS
	STAR
	SLASH
	LT
	GT
	LE
	GE
	EQEQ
	NE
	BANG
	ANDAND
	OROR
)

func (k Kind) String() string {
	names := [...]string{
		"EOF", "IDENT", "INT", "FLOAT", "STRING",
		"FN", "LET", "EXTERN", "RETURN", "WHEN", "DO", "OVER", "AS", "FOR", "IN", "SPAN", "UNIT", "BIT", "TRIT", "STR", "I32", "F32", "VOID",
		"TRUE", "FALSE",
		"PIPE", "EQ", "ARROW", "FATARROW", "DOT", "COMMA", "COLON", "SEMI",
		"LPAREN", "RPAREN", "LBRACE", "RBRACE", "LBRACK", "RBRACK",
		"PLUS", "MINUS", "STAR", "SLASH", "LT", "GT", "LE", "GE", "EQEQ", "NE", "BANG", "ANDAND", "OROR",
	}
	if int(k) < len(names) {
		return names[k]
	}
	return "?"
}

type Token struct {
	Kind  Kind
	Value any
	Line  int
	Col   int
}

var Keywords = map[string]Kind{
	"fn": FN, "let": LET, "extern": EXTERN, "return": RETURN,
	"when": WHEN, "do": DO, "over": OVER, "as": AS, "for": FOR, "in": IN,
	"span": SPAN, "unit": UNIT, "bit": BIT, "trit": TRIT,
	"str": STR, "i32": I32, "f32": F32, "void": VOID,
	"true": TRUE, "false": FALSE,
	"and": ANDAND, "or": OROR,
}
