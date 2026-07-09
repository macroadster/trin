package parser

import (
	"fmt"

	"github.com/ericchien/trin/internal/ast"
	"github.com/ericchien/trin/internal/lexer"
	"github.com/ericchien/trin/internal/token"
)

type Error struct {
	Msg  string
	Line int
	Col  int
}

func (e *Error) Error() string {
	return fmt.Sprintf("parse error at %d:%d: %s", e.Line, e.Col, e.Msg)
}

type Parser struct {
	toks []token.Token
	i    int
}

func ParseSource(src string) (*ast.Program, error) {
	toks, err := lexer.Tokenize(src)
	if err != nil {
		return nil, err
	}
	return New(toks).Parse()
}

func New(toks []token.Token) *Parser { return &Parser{toks: toks} }

func (p *Parser) cur() token.Token { return p.toks[p.i] }

func (p *Parser) check(ks ...token.Kind) bool {
	for _, k := range ks {
		if p.cur().Kind == k {
			return true
		}
	}
	return false
}

func (p *Parser) adv() token.Token {
	t := p.cur()
	if t.Kind != token.EOF {
		p.i++
	}
	return t
}

func (p *Parser) expect(k token.Kind, msg string) (token.Token, error) {
	if p.cur().Kind != k {
		t := p.cur()
		if msg == "" {
			msg = fmt.Sprintf("expected %s, got %s (%v)", k, t.Kind, t.Value)
		}
		return token.Token{}, &Error{Msg: msg, Line: t.Line, Col: t.Col}
	}
	return p.adv(), nil
}

func (p *Parser) Parse() (*ast.Program, error) {
	var items []ast.Item
	for !p.check(token.EOF) {
		it, err := p.parseItem()
		if err != nil {
			return nil, err
		}
		items = append(items, it)
	}
	return &ast.Program{Line: 1, Col: 1, Items: items}, nil
}

func (p *Parser) parseItem() (ast.Item, error) {
	if p.check(token.FN) {
		return p.parseFn()
	}
	if p.check(token.EXTERN) {
		return p.parseExtern()
	}
	if p.check(token.LET) {
		return p.parseLet()
	}
	t := p.cur()
	return nil, &Error{Msg: "expected item", Line: t.Line, Col: t.Col}
}

func (p *Parser) parseExtern() (*ast.ExternDecl, error) {
	tok := p.adv()
	if _, err := p.expect(token.FN, ""); err != nil {
		return nil, err
	}
	nameTok, err := p.expect(token.IDENT, "")
	if err != nil {
		return nil, err
	}
	if _, err := p.expect(token.LPAREN, ""); err != nil {
		return nil, err
	}
	params, err := p.parseParams()
	if err != nil {
		return nil, err
	}
	if _, err := p.expect(token.RPAREN, ""); err != nil {
		return nil, err
	}
	var ret *ast.TypeAnn
	if p.check(token.ARROW) {
		p.adv()
		ret, err = p.parseType()
		if err != nil {
			return nil, err
		}
	}
	if _, err := p.expect(token.SEMI, ""); err != nil {
		return nil, err
	}
	return &ast.ExternDecl{
		Line: tok.Line, Col: tok.Col,
		Name: nameTok.Value.(string), Params: params, Ret: ret,
	}, nil
}

func (p *Parser) parseFn() (*ast.FnDef, error) {
	tok := p.adv()
	nameTok, err := p.expect(token.IDENT, "")
	if err != nil {
		return nil, err
	}
	if _, err := p.expect(token.LPAREN, ""); err != nil {
		return nil, err
	}
	params, err := p.parseParams()
	if err != nil {
		return nil, err
	}
	if _, err := p.expect(token.RPAREN, ""); err != nil {
		return nil, err
	}
	var ret *ast.TypeAnn
	if p.check(token.ARROW) {
		p.adv()
		ret, err = p.parseType()
		if err != nil {
			return nil, err
		}
	}
	body, err := p.parseBlock()
	if err != nil {
		return nil, err
	}
	return &ast.FnDef{
		Line: tok.Line, Col: tok.Col,
		Name: nameTok.Value.(string), Params: params, Ret: ret, Body: body,
	}, nil
}

func (p *Parser) parseParams() ([]ast.Param, error) {
	if p.check(token.RPAREN) {
		return nil, nil
	}
	pm, err := p.parseParam()
	if err != nil {
		return nil, err
	}
	params := []ast.Param{pm}
	for p.check(token.COMMA) {
		p.adv()
		pm, err = p.parseParam()
		if err != nil {
			return nil, err
		}
		params = append(params, pm)
	}
	return params, nil
}

func (p *Parser) parseParam() (ast.Param, error) {
	nameTok, err := p.expect(token.IDENT, "")
	if err != nil {
		return ast.Param{}, err
	}
	var typ *ast.TypeAnn
	if p.check(token.COLON) {
		p.adv()
		typ, err = p.parseType()
		if err != nil {
			return ast.Param{}, err
		}
	}
	return ast.Param{
		Line: nameTok.Line, Col: nameTok.Col,
		Name: nameTok.Value.(string), Typ: typ,
	}, nil
}

func (p *Parser) parseType() (*ast.TypeAnn, error) {
	t := p.cur()
	if p.check(token.VOID, token.BIT, token.TRIT, token.UNIT, token.STR, token.I32, token.F32) {
		p.adv()
		return &ast.TypeAnn{Line: t.Line, Col: t.Col, Name: t.Value.(string)}, nil
	}
	if p.check(token.SPAN) {
		p.adv()
		if _, err := p.expect(token.LBRACK, ""); err != nil {
			return nil, err
		}
		inner, err := p.parseType()
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(token.RBRACK, ""); err != nil {
			return nil, err
		}
		return &ast.TypeAnn{Line: t.Line, Col: t.Col, Name: "span", Inner: inner}, nil
	}
	if p.check(token.FN) {
		p.adv()
		if _, err := p.expect(token.LPAREN, ""); err != nil {
			return nil, err
		}
		var params []*ast.TypeAnn
		if !p.check(token.RPAREN) {
			pt, err := p.parseType()
			if err != nil {
				return nil, err
			}
			params = append(params, pt)
			for p.check(token.COMMA) {
				p.adv()
				pt, err = p.parseType()
				if err != nil {
					return nil, err
				}
				params = append(params, pt)
			}
		}
		if _, err := p.expect(token.RPAREN, ""); err != nil {
			return nil, err
		}
		if _, err := p.expect(token.ARROW, ""); err != nil {
			return nil, err
		}
		ret, err := p.parseType()
		if err != nil {
			return nil, err
		}
		return &ast.TypeAnn{Line: t.Line, Col: t.Col, Name: "fn", Params: params, Ret: ret}, nil
	}
	if p.check(token.IDENT) {
		p.adv()
		name := t.Value.(string)
		// Language-level pair[T] (v0.46): pair[f32] → TypeAnn{Name:"pair", Inner:f32}
		if name == "pair" && p.check(token.LBRACK) {
			p.adv()
			inner, err := p.parseType()
			if err != nil {
				return nil, err
			}
			if _, err := p.expect(token.RBRACK, ""); err != nil {
				return nil, err
			}
			return &ast.TypeAnn{Line: t.Line, Col: t.Col, Name: "pair", Inner: inner}, nil
		}
		return &ast.TypeAnn{Line: t.Line, Col: t.Col, Name: name}, nil
	}
	return nil, &Error{Msg: "expected type", Line: t.Line, Col: t.Col}
}

func (p *Parser) parseBlock() (*ast.Block, error) {
	tok, err := p.expect(token.LBRACE, "")
	if err != nil {
		return nil, err
	}
	var stmts []ast.Stmt
	var finalExpr ast.Expr
	for !p.check(token.RBRACE, token.EOF) {
		if p.check(token.LET) {
			ls, err := p.parseLet()
			if err != nil {
				return nil, err
			}
			stmts = append(stmts, ls)
			continue
		}
		if p.check(token.RETURN) {
			rs, err := p.parseReturn()
			if err != nil {
				return nil, err
			}
			stmts = append(stmts, rs)
			continue
		}
		e, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		if p.check(token.SEMI) {
			p.adv()
			stmts = append(stmts, &ast.ExprStmt{Line: lineOf(e), Col: colOf(e), Expr: e})
		} else if p.check(token.RBRACE) {
			finalExpr = e
			break
		} else if p.check(token.LET, token.RETURN) {
			stmts = append(stmts, &ast.ExprStmt{Line: lineOf(e), Col: colOf(e), Expr: e})
		} else {
			t := p.cur()
			return nil, &Error{Msg: "expected ';' or end of block after expression", Line: t.Line, Col: t.Col}
		}
	}
	if _, err := p.expect(token.RBRACE, ""); err != nil {
		return nil, err
	}
	return &ast.Block{Line: tok.Line, Col: tok.Col, Stmts: stmts, Expr: finalExpr}, nil
}

func lineOf(e ast.Expr) int { l, _ := e.Pos(); return l }
func colOf(e ast.Expr) int  { _, c := e.Pos(); return c }

func (p *Parser) parseLet() (*ast.LetStmt, error) {
	tok := p.adv()
	// v0.47/v0.48: let (a, b) = ... and multi-way let (a, b, c, ...) = span;
	// also nested pair pattern: let (a, (b, c)) = nested_pair;
	if p.check(token.LPAREN) {
		names, err := p.parseDestructureNames()
		if err != nil {
			return nil, err
		}
		var typ *ast.TypeAnn
		if p.check(token.COLON) {
			p.adv()
			typ, err = p.parseType()
			if err != nil {
				return nil, err
			}
		}
		if _, err := p.expect(token.EQ, ""); err != nil {
			return nil, err
		}
		val, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(token.SEMI, ""); err != nil {
			return nil, err
		}
		if len(names) == 0 {
			return nil, &Error{Msg: "empty destructure", Line: tok.Line, Col: tok.Col}
		}
		return &ast.LetStmt{
			Line: tok.Line, Col: tok.Col,
			Name: names[0], Names: names, Typ: typ, Value: val,
		}, nil
	}
	nameTok, err := p.expect(token.IDENT, "")
	if err != nil {
		return nil, err
	}
	var typ *ast.TypeAnn
	if p.check(token.COLON) {
		p.adv()
		typ, err = p.parseType()
		if err != nil {
			return nil, err
		}
	}
	if _, err := p.expect(token.EQ, ""); err != nil {
		return nil, err
	}
	val, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	if _, err := p.expect(token.SEMI, ""); err != nil {
		return nil, err
	}
	return &ast.LetStmt{
		Line: tok.Line, Col: tok.Col,
		Name: nameTok.Value.(string), Typ: typ, Value: val,
	}, nil
}

func (p *Parser) parseReturn() (*ast.ReturnStmt, error) {
	tok := p.adv()
	var val ast.Expr
	if !p.check(token.SEMI) {
		var err error
		val, err = p.parseExpr()
		if err != nil {
			return nil, err
		}
	}
	if _, err := p.expect(token.SEMI, ""); err != nil {
		return nil, err
	}
	return &ast.ReturnStmt{Line: tok.Line, Col: tok.Col, Value: val}, nil
}

// parseDestructureNames parses (a, b, ...) and nested (a, (b, c)) patterns into
// a flat name list (left-to-right leaf order). Used by multi-way / nested pair
// destructure (v0.48).
func (p *Parser) parseDestructureNames() ([]string, error) {
	if _, err := p.expect(token.LPAREN, ""); err != nil {
		return nil, err
	}
	var names []string
	for {
		if p.check(token.LPAREN) {
			inner, err := p.parseDestructureNames()
			if err != nil {
				return nil, err
			}
			names = append(names, inner...)
		} else {
			id, err := p.expect(token.IDENT, "")
			if err != nil {
				return nil, err
			}
			names = append(names, id.Value.(string))
		}
		if p.check(token.COMMA) {
			p.adv()
			// allow trailing comma before )
			if p.check(token.RPAREN) {
				break
			}
			continue
		}
		break
	}
	if _, err := p.expect(token.RPAREN, ""); err != nil {
		return nil, err
	}
	return names, nil
}

func (p *Parser) parseExpr() (ast.Expr, error) { return p.parseTernary() }

func (p *Parser) parseTernary() (ast.Expr, error) {
	left, err := p.parseOr()
	if err != nil {
		return nil, err
	}
	if p.check(token.PIPE) {
		p.adv()
		pure, err := p.parseOr()
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(token.PIPE, "ternary requires gate | pure | each"); err != nil {
			return nil, err
		}
		each, err := p.parseOr()
		if err != nil {
			return nil, err
		}
		return &ast.Ternary{Line: lineOf(left), Col: colOf(left), Gate: left, Pure: pure, Each: each}, nil
	}
	return left, nil
}

func (p *Parser) parseOr() (ast.Expr, error) {
	left, err := p.parseAnd()
	if err != nil {
		return nil, err
	}
	for p.check(token.OROR) {
		op := p.adv()
		right, err := p.parseAnd()
		if err != nil {
			return nil, err
		}
		left = &ast.Binary{Line: op.Line, Col: op.Col, Op: "or", Left: left, Right: right}
	}
	return left, nil
}

func (p *Parser) parseAnd() (ast.Expr, error) {
	left, err := p.parseCmp()
	if err != nil {
		return nil, err
	}
	for p.check(token.ANDAND) {
		op := p.adv()
		right, err := p.parseCmp()
		if err != nil {
			return nil, err
		}
		left = &ast.Binary{Line: op.Line, Col: op.Col, Op: "and", Left: left, Right: right}
	}
	return left, nil
}

func (p *Parser) parseCmp() (ast.Expr, error) {
	left, err := p.parseAdd()
	if err != nil {
		return nil, err
	}
	if p.check(token.LT, token.GT, token.LE, token.GE, token.EQEQ, token.NE) {
		opTok := p.adv()
		right, err := p.parseAdd()
		if err != nil {
			return nil, err
		}
		opMap := map[token.Kind]string{
			token.LT: "<", token.GT: ">", token.LE: "<=", token.GE: ">=",
			token.EQEQ: "==", token.NE: "!=",
		}
		return &ast.Binary{Line: opTok.Line, Col: opTok.Col, Op: opMap[opTok.Kind], Left: left, Right: right}, nil
	}
	return left, nil
}

func (p *Parser) parseAdd() (ast.Expr, error) {
	left, err := p.parseMul()
	if err != nil {
		return nil, err
	}
	for p.check(token.PLUS, token.MINUS) {
		opTok := p.adv()
		right, err := p.parseMul()
		if err != nil {
			return nil, err
		}
		op := "+"
		if opTok.Kind == token.MINUS {
			op = "-"
		}
		left = &ast.Binary{Line: opTok.Line, Col: opTok.Col, Op: op, Left: left, Right: right}
	}
	return left, nil
}

func (p *Parser) parseMul() (ast.Expr, error) {
	left, err := p.parseUnary()
	if err != nil {
		return nil, err
	}
	for p.check(token.STAR, token.SLASH) {
		opTok := p.adv()
		right, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		op := "*"
		if opTok.Kind == token.SLASH {
			op = "/"
		}
		left = &ast.Binary{Line: opTok.Line, Col: opTok.Col, Op: op, Left: left, Right: right}
	}
	return left, nil
}

func (p *Parser) parseUnary() (ast.Expr, error) {
	if p.check(token.BANG, token.MINUS, token.PLUS) {
		opTok := p.adv()
		expr, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		op := "!"
		if opTok.Kind == token.MINUS {
			op = "-"
		} else if opTok.Kind == token.PLUS {
			op = "+"
		}
		return &ast.Unary{Line: opTok.Line, Col: opTok.Col, Op: op, Expr: expr}, nil
	}
	return p.parsePostfix()
}

func (p *Parser) parsePostfix() (ast.Expr, error) {
	expr, err := p.parsePrimary()
	if err != nil {
		return nil, err
	}
	for {
		if p.check(token.LPAREN) {
			lp := p.adv()
			var args []ast.Expr
			if !p.check(token.RPAREN) {
				a, err := p.parseExpr()
				if err != nil {
					return nil, err
				}
				args = append(args, a)
				for p.check(token.COMMA) {
					p.adv()
					a, err = p.parseExpr()
					if err != nil {
						return nil, err
					}
					args = append(args, a)
				}
			}
			if _, err := p.expect(token.RPAREN, ""); err != nil {
				return nil, err
			}
			expr = &ast.Call{Line: lp.Line, Col: lp.Col, Callee: expr, Args: args}
			continue
		}
		if p.check(token.LBRACK) {
			lb := p.adv()
			idx, err := p.parseExpr()
			if err != nil {
				return nil, err
			}
			if _, err := p.expect(token.RBRACK, ""); err != nil {
				return nil, err
			}
			expr = &ast.Index{Line: lb.Line, Col: lb.Col, Target: expr, Index: idx}
			continue
		}
		break
	}
	return expr, nil
}


// desugarWhen: when g do e  ⇒  g | e | 0  (numeric identity for each-arm)
func desugarWhen(line, col int, g, e ast.Expr) *ast.Ternary {
	return &ast.Ternary{
		Line: line, Col: col,
		Gate: g,
		Pure: e,
		Each: &ast.Literal{Line: line, Col: col, Kind: "int", Value: 0},
	}
}

// desugarOver: over xs as x do e  ⇒  map((x)=>e, xs)
// Practical foreach (map already handles empty spans). SPEC "collection-each"
// path is map; dens short-circuit is optional and unnecessary for empty.
func desugarOver(line, col int, xs ast.Expr, name string, body ast.Expr) *ast.Call {
	lam := &ast.Lambda{
		Line: line, Col: col,
		Params: []ast.Param{{Line: line, Col: col, Name: name}},
		Body:   body,
	}
	return &ast.Call{
		Line: line, Col: col,
		Callee: &ast.Var{Line: line, Col: col, Name: "map"},
		Args:   []ast.Expr{lam, xs},
	}
}

func (p *Parser) parsePrimary() (ast.Expr, error) {
	t := p.cur()
	switch {
	case p.check(token.WHEN):
		// when g do e  ⇒  g | e | 0
		wtok := p.adv()
		g, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(token.DO, "when requires do"); err != nil {
			return nil, err
		}
		e, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		return desugarWhen(wtok.Line, wtok.Col, g, e), nil
	case p.check(token.OVER):
		// over xs as x do e  ⇒  map((x)=>e, xs)
		otok := p.adv()
		xs, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(token.AS, "over requires as"); err != nil {
			return nil, err
		}
		nameTok, err := p.expect(token.IDENT, "over binder")
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(token.DO, "over requires do"); err != nil {
			return nil, err
		}
		body, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		return desugarOver(otok.Line, otok.Col, xs, nameTok.Value.(string), body), nil
	case p.check(token.FOR):
		// for x in xs do e  ⇒  map((x)=>e, xs)   (same as over)
		ftok := p.adv()
		nameTok, err := p.expect(token.IDENT, "for binder")
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(token.IN, "for requires in"); err != nil {
			return nil, err
		}
		xs, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(token.DO, "for requires do"); err != nil {
			return nil, err
		}
		body, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		return desugarOver(ftok.Line, ftok.Col, xs, nameTok.Value.(string), body), nil
	case p.check(token.INT):
		p.adv()
		return &ast.Literal{Line: t.Line, Col: t.Col, Kind: "int", Value: t.Value}, nil
	case p.check(token.FLOAT):
		p.adv()
		return &ast.Literal{Line: t.Line, Col: t.Col, Kind: "float", Value: t.Value}, nil
	case p.check(token.STRING):
		p.adv()
		return &ast.Literal{Line: t.Line, Col: t.Col, Kind: "str", Value: t.Value}, nil
	case p.check(token.TRUE, token.FALSE):
		p.adv()
		return &ast.Literal{Line: t.Line, Col: t.Col, Kind: "bool", Value: t.Value}, nil
	case p.check(token.IDENT):
		p.adv()
		return &ast.Var{Line: t.Line, Col: t.Col, Name: t.Value.(string)}, nil
	case p.check(token.TRIT, token.BIT, token.UNIT, token.STR, token.I32, token.F32, token.VOID, token.SPAN):
		p.adv()
		return &ast.Var{Line: t.Line, Col: t.Col, Name: t.Value.(string)}, nil
	case p.check(token.LPAREN):
		p.adv()
		save := p.i
		isLam := p.tryLambdaAfterLParen()
		p.i = save
		if isLam {
			return p.parseLambdaAfterLParen(t.Line, t.Col)
		}
		if p.check(token.RPAREN) {
			p.adv()
			if p.check(token.FATARROW) {
				p.adv()
				body, err := p.parseExpr()
				if err != nil {
					return nil, err
				}
				return &ast.Lambda{Line: t.Line, Col: t.Col, Body: body}, nil
			}
			return nil, &Error{Msg: "empty parentheses", Line: t.Line, Col: t.Col}
		}
		expr, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(token.RPAREN, ""); err != nil {
			return nil, err
		}
		return expr, nil
	case p.check(token.LBRACK):
		p.adv()
		var elems []ast.Expr
		if !p.check(token.RBRACK) {
			e, err := p.parseExpr()
			if err != nil {
				return nil, err
			}
			elems = append(elems, e)
			for p.check(token.COMMA) {
				p.adv()
				e, err = p.parseExpr()
				if err != nil {
					return nil, err
				}
				elems = append(elems, e)
			}
		}
		if _, err := p.expect(token.RBRACK, ""); err != nil {
			return nil, err
		}
		return &ast.SpanLit{Line: t.Line, Col: t.Col, Elems: elems}, nil
	case p.check(token.LBRACE):
		return p.parseBlock()
	default:
		return nil, &Error{Msg: fmt.Sprintf("unexpected token %s (%v)", t.Kind, t.Value), Line: t.Line, Col: t.Col}
	}
}

func (p *Parser) tryLambdaAfterLParen() bool {
	if p.check(token.RPAREN) {
		return false
	}
	if !p.check(token.IDENT) {
		return false
	}
	p.adv()
	if p.check(token.COLON) || p.check(token.COMMA) {
		return true
	}
	if p.check(token.RPAREN) {
		p.adv()
		return p.check(token.FATARROW)
	}
	return false
}

func (p *Parser) parseLambdaAfterLParen(line, col int) (*ast.Lambda, error) {
	var params []ast.Param
	if !p.check(token.RPAREN) {
		pm, err := p.parseParam()
		if err != nil {
			return nil, err
		}
		params = append(params, pm)
		for p.check(token.COMMA) {
			p.adv()
			pm, err = p.parseParam()
			if err != nil {
				return nil, err
			}
			params = append(params, pm)
		}
	}
	if _, err := p.expect(token.RPAREN, ""); err != nil {
		return nil, err
	}
	if _, err := p.expect(token.FATARROW, ""); err != nil {
		return nil, err
	}
	body, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	return &ast.Lambda{Line: line, Col: col, Params: params, Body: body}, nil
}
