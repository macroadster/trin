package lexer

import (
	"fmt"
	"unicode"

	"github.com/macroadster/trin/internal/token"
)

type Error struct {
	Msg  string
	Line int
	Col  int
}

func (e *Error) Error() string {
	return fmt.Sprintf("lex error at %d:%d: %s", e.Line, e.Col, e.Msg)
}

type Lexer struct {
	src  []rune
	i    int
	line int
	col  int
}

func New(src string) *Lexer {
	return &Lexer{src: []rune(src), line: 1, col: 1}
}

func Tokenize(src string) ([]token.Token, error) {
	l := New(src)
	var out []token.Token
	for {
		tok, err := l.Next()
		if err != nil {
			return nil, err
		}
		out = append(out, tok)
		if tok.Kind == token.EOF {
			return out, nil
		}
	}
}

func (l *Lexer) peek() rune {
	if l.i >= len(l.src) {
		return 0
	}
	return l.src[l.i]
}

func (l *Lexer) peek2() rune {
	if l.i+1 >= len(l.src) {
		return 0
	}
	return l.src[l.i+1]
}

func (l *Lexer) adv() rune {
	ch := l.peek()
	if ch == 0 {
		return 0
	}
	l.i++
	if ch == '\n' {
		l.line++
		l.col = 1
	} else {
		l.col++
	}
	return ch
}

func (l *Lexer) Next() (token.Token, error) {
	if err := l.skipWSComments(); err != nil {
		return token.Token{}, err
	}
	line, col := l.line, l.col
	ch := l.peek()
	if ch == 0 {
		return token.Token{Kind: token.EOF, Line: line, Col: col}, nil
	}
	// two-char
	two := string([]rune{ch, l.peek2()})
	switch two {
	case "->":
		l.adv()
		l.adv()
		return token.Token{Kind: token.ARROW, Value: two, Line: line, Col: col}, nil
	case "=>":
		l.adv()
		l.adv()
		return token.Token{Kind: token.FATARROW, Value: two, Line: line, Col: col}, nil
	case "==":
		l.adv()
		l.adv()
		return token.Token{Kind: token.EQEQ, Value: two, Line: line, Col: col}, nil
	case "!=":
		l.adv()
		l.adv()
		return token.Token{Kind: token.NE, Value: two, Line: line, Col: col}, nil
	case "<=":
		l.adv()
		l.adv()
		return token.Token{Kind: token.LE, Value: two, Line: line, Col: col}, nil
	case ">=":
		l.adv()
		l.adv()
		return token.Token{Kind: token.GE, Value: two, Line: line, Col: col}, nil
	case "&&":
		l.adv()
		l.adv()
		return token.Token{Kind: token.ANDAND, Value: two, Line: line, Col: col}, nil
	case "||":
		l.adv()
		l.adv()
		return token.Token{Kind: token.OROR, Value: two, Line: line, Col: col}, nil
	}
	singles := map[rune]token.Kind{
		'|': token.PIPE, '=': token.EQ, '.': token.DOT, ',': token.COMMA,
		':': token.COLON, ';': token.SEMI, '(': token.LPAREN, ')': token.RPAREN,
		'{': token.LBRACE, '}': token.RBRACE, '[': token.LBRACK, ']': token.RBRACK,
		'+': token.PLUS, '-': token.MINUS, '*': token.STAR, '/': token.SLASH,
		'<': token.LT, '>': token.GT, '!': token.BANG,
	}
	if k, ok := singles[ch]; ok {
		l.adv()
		return token.Token{Kind: k, Value: string(ch), Line: line, Col: col}, nil
	}
	if ch == '"' {
		return l.stringLit(line, col)
	}
	if unicode.IsDigit(ch) || (ch == '.' && unicode.IsDigit(l.peek2())) {
		return l.number(line, col)
	}
	if unicode.IsLetter(ch) || ch == '_' {
		return l.ident(line, col), nil
	}
	return token.Token{}, &Error{Msg: fmt.Sprintf("unexpected character %q", ch), Line: line, Col: col}
}

func (l *Lexer) skipWSComments() error {
	for {
		ch := l.peek()
		if ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' {
			l.adv()
			continue
		}
		if ch == '/' && l.peek2() == '/' {
			for l.peek() != '\n' && l.peek() != 0 {
				l.adv()
			}
			continue
		}
		if ch == '/' && l.peek2() == '*' {
			l.adv()
			l.adv()
			for {
				if l.peek() == 0 {
					return &Error{Msg: "unterminated block comment", Line: l.line, Col: l.col}
				}
				if l.peek() == '*' && l.peek2() == '/' {
					l.adv()
					l.adv()
					break
				}
				l.adv()
			}
			continue
		}
		return nil
	}
}

func (l *Lexer) stringLit(line, col int) (token.Token, error) {
	l.adv() // "
	var buf []rune
	for {
		ch := l.peek()
		if ch == 0 {
			return token.Token{}, &Error{Msg: "unterminated string", Line: line, Col: col}
		}
		if ch == '"' {
			l.adv()
			break
		}
		if ch == '\\' {
			l.adv()
			esc := l.adv()
			switch esc {
			case 'n':
				buf = append(buf, '\n')
			case 't':
				buf = append(buf, '\t')
			case 'r':
				buf = append(buf, '\r')
			case '\\':
				buf = append(buf, '\\')
			case '"':
				buf = append(buf, '"')
			default:
				return token.Token{}, &Error{Msg: fmt.Sprintf("bad escape \\%c", esc), Line: line, Col: col}
			}
		} else {
			buf = append(buf, l.adv())
		}
	}
	return token.Token{Kind: token.STRING, Value: string(buf), Line: line, Col: col}, nil
}

func (l *Lexer) number(line, col int) (token.Token, error) {
	start := l.i
	isFloat := false
	for unicode.IsDigit(l.peek()) {
		l.adv()
	}
	if l.peek() == '.' && unicode.IsDigit(l.peek2()) {
		isFloat = true
		l.adv()
		for unicode.IsDigit(l.peek()) {
			l.adv()
		}
	}
	text := string(l.src[start:l.i])
	if isFloat {
		var f float64
		fmt.Sscanf(text, "%f", &f)
		return token.Token{Kind: token.FLOAT, Value: f, Line: line, Col: col}, nil
	}
	var n int64
	fmt.Sscanf(text, "%d", &n)
	return token.Token{Kind: token.INT, Value: int(n), Line: line, Col: col}, nil
}

func (l *Lexer) ident(line, col int) token.Token {
	start := l.i
	for {
		ch := l.peek()
		if unicode.IsLetter(ch) || unicode.IsDigit(ch) || ch == '_' {
			l.adv()
			continue
		}
		break
	}
	text := string(l.src[start:l.i])
	if k, ok := token.Keywords[text]; ok {
		if k == token.TRUE {
			return token.Token{Kind: token.TRUE, Value: true, Line: line, Col: col}
		}
		if k == token.FALSE {
			return token.Token{Kind: token.FALSE, Value: false, Line: line, Col: col}
		}
		return token.Token{Kind: k, Value: text, Line: line, Col: col}
	}
	return token.Token{Kind: token.IDENT, Value: text, Line: line, Col: col}
}
