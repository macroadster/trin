package interp

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"math/bits"
	"os"
	"strconv"
	"strings"

	"github.com/macroadster/trin/internal/ad"
	"github.com/macroadster/trin/internal/ast"
	"github.com/macroadster/trin/internal/cuda"
	"github.com/macroadster/trin/internal/parser"
	"github.com/macroadster/trin/internal/weights"
)

type Error struct {
	Msg string
}

func (e *Error) Error() string { return e.Msg }

type Interpreter struct {
	In  *bufio.Reader
	Out io.Writer
	Env *Env
}

func New(in io.Reader, out io.Writer) *Interpreter {
	if in == nil {
		in = strings.NewReader("")
	}
	if out == nil {
		out = io.Discard
	}
	it := &Interpreter{
		In:  bufio.NewReader(in),
		Out: out,
		Env: NewEnv(nil),
	}
	it.installBuiltins()
	return it
}

func RunSource(src string, in io.Reader, out io.Writer) error {
	prog, err := parser.ParseSource(src)
	if err != nil {
		return err
	}
	return New(in, out).Run(prog)
}

func RunFile(path string, in io.Reader, out io.Writer) error {
	// caller loads file
	return fmt.Errorf("use RunSource")
}

func (it *Interpreter) Run(prog *ast.Program) error {
	for _, item := range prog.Items {
		switch n := item.(type) {
		case *ast.FnDef:
			params := make([]string, len(n.Params))
			for i, p := range n.Params {
				params[i] = p.Name
			}
			// capture body as block expression
			var body ast.Expr = n.Body
			it.Env.Set(n.Name, Value{Kind: VFn, Fn: &Closure{Params: params, Body: body, Env: it.Env}})
		case *ast.LetStmt:
			v, err := it.evalExpr(n.Value, it.Env)
			if err != nil {
				return err
			}
			if names := n.BindingNames(); len(names) >= 2 {
				leaves := flattenPairLeaves(v)
				if len(leaves) != len(names) {
					return &Error{Msg: "pair destructure: not exhaustive (arity mismatch)"}
				}
				for i, name := range names {
					it.Env.Set(name, leaves[i])
				}
			} else {
				it.Env.Set(n.Name, v)
			}
		case *ast.ExternDecl:
			// no-op for interpreter
		}
	}
	main, ok := it.Env.Get("main")
	if !ok || main.Kind != VFn {
		return &Error{Msg: "no main function"}
	}
	_, err := it.apply(main, nil)
	return err
}

func (it *Interpreter) apply(fn Value, args []Value) (Value, error) {
	if fn.Kind == VNative {
		return fn.Nat(args)
	}
	if fn.Kind != VFn {
		return Void(), &Error{Msg: "not a function"}
	}
	c := fn.Fn
	if len(args) != len(c.Params) {
		return Void(), &Error{Msg: fmt.Sprintf("arity mismatch: want %d got %d", len(c.Params), len(args))}
	}
	env := NewEnv(c.Env)
	for i, p := range c.Params {
		env.Set(p, args[i])
	}
	switch b := c.Body.(type) {
	case *ast.Block:
		return it.evalBlock(b, env)
	case ast.Expr:
		return it.evalExpr(b, env)
	default:
		return Void(), &Error{Msg: "bad closure body"}
	}
}

// reduceFold implements reduce((acc,x)=>body, init, span) and fold(...).
func (it *Interpreter) reduceFold(args []Value) (Value, error) {
	if len(args) < 3 || args[2].Kind != VSpan {
		return Void(), &Error{Msg: "reduce/fold expects (f, init, span)"}
	}
	acc := args[1]
	for _, x := range args[2].Span {
		v, err := it.apply(args[0], []Value{acc, x})
		if err != nil {
			return Void(), err
		}
		acc = v
	}
	return acc, nil
}

// scanFold implements scan((acc,x)=>body, init, span) — left fold collecting
// each intermediate accumulator after applying body (prefix values, no init).
func (it *Interpreter) scanFold(args []Value) (Value, error) {
	if len(args) < 3 || args[2].Kind != VSpan {
		return Void(), &Error{Msg: "scan expects (f, init, span)"}
	}
	acc := args[1]
	out := make([]Value, 0, len(args[2].Span))
	for _, x := range args[2].Span {
		v, err := it.apply(args[0], []Value{acc, x})
		if err != nil {
			return Void(), err
		}
		acc = v
		out = append(out, acc)
	}
	return Span(out), nil
}

// flattenPairLeaves flattens nested pair/spans into leaf values (v0.48 multi-way/nested).
// Non-span values are leaves; spans recurse into elements.
func flattenPairLeaves(v Value) []Value {
	if v.Kind != VSpan {
		return []Value{v}
	}
	out := make([]Value, 0, len(v.Span))
	for _, e := range v.Span {
		out = append(out, flattenPairLeaves(e)...)
	}
	return out
}

func (it *Interpreter) evalBlock(b *ast.Block, env *Env) (Value, error) {
	local := NewEnv(env)
	for _, st := range b.Stmts {
		switch s := st.(type) {
		case *ast.LetStmt:
			v, err := it.evalExpr(s.Value, local)
			if err != nil {
				return Void(), err
			}
			// v0.47–v0.51: multi-way / nested pair destructure — leaf order + exhaustive arity check
			if names := s.BindingNames(); len(names) >= 2 {
				leaves := flattenPairLeaves(v)
				if len(leaves) != len(names) {
					return Void(), &Error{Msg: "pair destructure: not exhaustive (arity mismatch)"}
				}
				for i, name := range names {
					local.Set(name, leaves[i])
				}
			} else {
				local.Set(s.Name, v)
			}
		case *ast.ExprStmt:
			if _, err := it.evalExpr(s.Expr, local); err != nil {
				return Void(), err
			}
		case *ast.ReturnStmt:
			if s.Value == nil {
				return Void(), nil
			}
			return it.evalExpr(s.Value, local)
		}
	}
	if b.Expr != nil {
		return it.evalExpr(b.Expr, local)
	}
	return Void(), nil
}

func (it *Interpreter) evalExpr(e ast.Expr, env *Env) (Value, error) {
	switch n := e.(type) {
	case *ast.Literal:
		switch n.Kind {
		case "int":
			return I32(n.Value.(int)), nil
		case "float":
			return F32(n.Value.(float64)), nil
		case "str":
			return Str(n.Value.(string)), nil
		case "bool":
			if n.Value.(bool) {
				return Bit(1), nil
			}
			return Bit(0), nil
		}
	case *ast.Var:
		v, ok := env.Get(n.Name)
		if !ok {
			return Void(), &Error{Msg: "undefined name: " + n.Name}
		}
		return v, nil
	case *ast.Unary:
		x, err := it.evalExpr(n.Expr, env)
		if err != nil {
			return Void(), err
		}
		switch n.Op {
		case "-":
			if x.Kind == VF32 {
				return F32(-x.F32), nil
			}
			return I32(-x.AsInt()), nil
		case "+":
			return x, nil
		case "!":
			b := TruthyBit(x)
			return Bit(1 - b.Bit), nil
		}
	case *ast.Binary:
		// short-circuit not needed for and/or of values
		l, err := it.evalExpr(n.Left, env)
		if err != nil {
			return Void(), err
		}
		r, err := it.evalExpr(n.Right, env)
		if err != nil {
			return Void(), err
		}
		return it.bin(n.Op, l, r)
	case *ast.Ternary:
		g, err := it.evalExpr(n.Gate, env)
		if err != nil {
			return Void(), err
		}
		w, err := GateWeight(g)
		if err != nil {
			return Void(), err
		}
		if w >= 1.0-1e-9 {
			return it.evalExpr(n.Pure, env)
		}
		if w <= 1e-9 {
			return it.evalExpr(n.Each, env)
		}
		// soft blend: evaluate both
		pure, err := it.evalExpr(n.Pure, env)
		if err != nil {
			return Void(), err
		}
		each, err := it.evalExpr(n.Each, env)
		if err != nil {
			return Void(), err
		}
		return Blend(w, pure, each), nil
	case *ast.Call:
		cal, err := it.evalExpr(n.Callee, env)
		if err != nil {
			return Void(), err
		}
		args := make([]Value, len(n.Args))
		for i, a := range n.Args {
			args[i], err = it.evalExpr(a, env)
			if err != nil {
				return Void(), err
			}
		}
		return it.apply(cal, args)
	case *ast.Index:
		t, err := it.evalExpr(n.Target, env)
		if err != nil {
			return Void(), err
		}
		idx, err := it.evalExpr(n.Index, env)
		if err != nil {
			return Void(), err
		}
		i := idx.AsInt()
		if t.Kind != VSpan {
			return Void(), &Error{Msg: "index on non-span"}
		}
		if i < 0 || i >= len(t.Span) {
			return Void(), &Error{Msg: "index out of range"}
		}
		return t.Span[i], nil
	case *ast.SpanLit:
		items := make([]Value, len(n.Elems))
		for i, el := range n.Elems {
			var err error
			items[i], err = it.evalExpr(el, env)
			if err != nil {
				return Void(), err
			}
		}
		return Span(items), nil
	case *ast.Lambda:
		params := make([]string, len(n.Params))
		for i, p := range n.Params {
			params[i] = p.Name
		}
		return Value{Kind: VFn, Fn: &Closure{Params: params, Body: n.Body, Env: env}}, nil
	case *ast.Block:
		return it.evalBlock(n, env)
	}
	return Void(), &Error{Msg: fmt.Sprintf("unknown expr %T", e)}
}

func (it *Interpreter) bin(op string, l, r Value) (Value, error) {
	switch op {
	case "+":
		if l.Kind == VStr || r.Kind == VStr {
			return Str(l.String() + r.String()), nil
		}
		if l.Kind == VF32 || r.Kind == VF32 {
			return F32(l.AsFloat() + r.AsFloat()), nil
		}
		return I32(l.AsInt() + r.AsInt()), nil
	case "-":
		if l.Kind == VF32 || r.Kind == VF32 {
			return F32(l.AsFloat() - r.AsFloat()), nil
		}
		return I32(l.AsInt() - r.AsInt()), nil
	case "*":
		if l.Kind == VF32 || r.Kind == VF32 {
			return F32(l.AsFloat() * r.AsFloat()), nil
		}
		return I32(l.AsInt() * r.AsInt()), nil
	case "/":
		return F32(l.AsFloat() / r.AsFloat()), nil
	case "<", ">", "<=", ">=", "==", "!=":
		return cmp(op, l, r), nil
	case "and":
		return Bit(TruthyBit(l).Bit & TruthyBit(r).Bit), nil
	case "or":
		return Bit(TruthyBit(l).Bit | TruthyBit(r).Bit), nil
	}
	return Void(), &Error{Msg: "bad op " + op}
}

func cmp(op string, l, r Value) Value {
	// numeric or string
	if l.Kind == VStr && r.Kind == VStr {
		switch op {
		case "==":
			return bit(l.Str == r.Str)
		case "!=":
			return bit(l.Str != r.Str)
		case "<":
			return bit(l.Str < r.Str)
		case ">":
			return bit(l.Str > r.Str)
		case "<=":
			return bit(l.Str <= r.Str)
		case ">=":
			return bit(l.Str >= r.Str)
		}
	}
	lf, rf := l.AsFloat(), r.AsFloat()
	switch op {
	case "<":
		return bit(lf < rf)
	case ">":
		return bit(lf > rf)
	case "<=":
		return bit(lf <= rf)
	case ">=":
		return bit(lf >= rf)
	case "==":
		return bit(lf == rf)
	case "!=":
		return bit(lf != rf)
	}
	return Bit(0)
}

func bit(b bool) Value {
	if b {
		return Bit(1)
	}
	return Bit(0)
}

func (it *Interpreter) installBuiltins() {
	nat := func(name string, fn NativeFn) {
		it.Env.Set(name, Value{Kind: VNative, Nat: fn})
	}
	nat("print", func(args []Value) (Value, error) {
		if len(args) > 0 {
			fmt.Fprint(it.Out, args[0].String())
		}
		return Void(), nil
	})
	nat("println", func(args []Value) (Value, error) {
		if len(args) == 0 {
			fmt.Fprintln(it.Out)
		} else {
			fmt.Fprintln(it.Out, args[0].String())
		}
		return Void(), nil
	})
	nat("input", func(args []Value) (Value, error) {
		prompt := ""
		if len(args) > 0 {
			prompt = args[0].String()
		}
		fmt.Fprint(it.Out, prompt)
		line, err := it.In.ReadString('\n')
		if err != nil && err != io.EOF {
			return Str(""), nil
		}
		return Str(strings.TrimRight(line, "\r\n")), nil
	})
	nat("str_concat", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Str(""), nil
		}
		return Str(args[0].String() + args[1].String()), nil
	})
	nat("str_len", func(args []Value) (Value, error) {
		if len(args) == 0 {
			return I32(0), nil
		}
		return I32(len(args[0].String())), nil
	})
	nat("len", func(args []Value) (Value, error) {
		if len(args) == 0 {
			return I32(0), nil
		}
		if args[0].Kind == VSpan {
			return I32(len(args[0].Span)), nil
		}
		return I32(len(args[0].String())), nil
	})
	nat("dens", func(args []Value) (Value, error) {
		if len(args) == 0 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "dens expects span"}
		}
		if len(args[0].Span) == 0 {
			return Unit(0), nil
		}
		return Unit(1), nil
	})
	// range(n) → [0..n); range(lo,hi) → [lo..hi); range(lo,hi,step) with step≠0
	nat("range", func(args []Value) (Value, error) {
		if len(args) == 0 || len(args) > 3 {
			return Void(), &Error{Msg: "range expects (n) or (lo,hi) or (lo,hi,step)"}
		}
		var lo, hi, step int
		if len(args) == 1 {
			lo, hi, step = 0, args[0].AsInt(), 1
		} else if len(args) == 2 {
			lo, hi, step = args[0].AsInt(), args[1].AsInt(), 1
		} else {
			lo, hi, step = args[0].AsInt(), args[1].AsInt(), args[2].AsInt()
		}
		if step == 0 {
			return Void(), &Error{Msg: "range step must be non-zero"}
		}
		var out []Value
		if step > 0 {
			for i := lo; i < hi; i += step {
				out = append(out, I32(i))
			}
		} else {
			for i := lo; i > hi; i += step {
				out = append(out, I32(i))
			}
		}
		return Span(out), nil
	})
	nat("sum", func(args []Value) (Value, error) {
		if len(args) == 0 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "sum expects span"}
		}
		items := args[0].Span
		if len(items) == 0 {
			return I32(0), nil
		}
		acc := items[0]
		for _, x := range items[1:] {
			v, err := it.bin("+", acc, x)
			if err != nil {
				return Void(), err
			}
			acc = v
		}
		return acc, nil
	})
	nat("harden", func(args []Value) (Value, error) {
		if len(args) == 0 {
			return Bit(0), nil
		}
		if args[0].Kind == VUnit {
			if args[0].Unit >= 0.5 {
				return Bit(1), nil
			}
			return Bit(0), nil
		}
		return TruthyBit(args[0]), nil
	})
	nat("soften", func(args []Value) (Value, error) {
		if len(args) == 0 {
			return Unit(0), nil
		}
		if args[0].Kind == VBit {
			return Unit(float64(args[0].Bit)), nil
		}
		return Unit(args[0].AsFloat()), nil
	})
	nat("unit_of", func(args []Value) (Value, error) {
		if len(args) == 0 {
			return Unit(0), nil
		}
		u := args[0].AsFloat()
		if u < 0 {
			u = 0
		}
		if u > 1 {
			u = 1
		}
		return Unit(u), nil
	})
	nat("bit", func(args []Value) (Value, error) {
		if len(args) == 0 {
			return Bit(0), nil
		}
		return TruthyBit(args[0]), nil
	})
	nat("trit", func(args []Value) (Value, error) {
		n := 0
		if len(args) > 0 {
			n = args[0].AsInt()
		}
		if n < -1 {
			n = -1
		}
		if n > 1 {
			n = 1
		}
		return Trit(n), nil
	})
	nat("to_str", func(args []Value) (Value, error) {
		if len(args) == 0 {
			return Str(""), nil
		}
		return Str(args[0].String()), nil
	})
	nat("map", func(args []Value) (Value, error) {
		if len(args) < 2 || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "map expects (f, span)"}
		}
		out := make([]Value, len(args[1].Span))
		for i, x := range args[1].Span {
			v, err := it.apply(args[0], []Value{x})
			if err != nil {
				return Void(), err
			}
			out[i] = v
		}
		return Span(out), nil
	})
	nat("zip", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "zip expects two spans"}
		}
		n := min(len(args[0].Span), len(args[1].Span))
		out := make([]Value, n)
		for i := 0; i < n; i++ {
			out[i] = Span([]Value{args[0].Span[i], args[1].Span[i]})
		}
		return Span(out), nil
	})
	// zip_with((p0,...,pN-1)=>body, s0, ..., sN-1) for N>=2
	nat("zip_with", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Void(), &Error{Msg: "zip_with expects (f, s0, ..., sN-1) with N>=2"}
		}
		spans := args[1:]
		for _, s := range spans {
			if s.Kind != VSpan {
				return Void(), &Error{Msg: "zip_with expects all span args after f"}
			}
		}
		n := len(spans[0].Span)
		for _, s := range spans[1:] {
			if len(s.Span) < n {
				n = len(s.Span)
			}
		}
		out := make([]Value, n)
		for i := 0; i < n; i++ {
			callArgs := make([]Value, len(spans))
			for j, s := range spans {
				callArgs[j] = s.Span[i]
			}
			v, err := it.apply(args[0], callArgs)
			if err != nil {
				return Void(), err
			}
			out[i] = v
		}
		return Span(out), nil
	})
	// reduce((acc,x)=>body, init, span) / fold(...) — left fold to scalar
	nat("reduce", func(args []Value) (Value, error) {
		return it.reduceFold(args)
	})
	nat("fold", func(args []Value) (Value, error) {
		return it.reduceFold(args)
	})
	// scan((acc,x)=>body, init, span) — left fold collecting intermediate accs
	nat("scan", func(args []Value) (Value, error) {
		return it.scanFold(args)
	})
	// filter((x)=>pred, span) — keep elements where pred is nonzero (f32)
	nat("filter", func(args []Value) (Value, error) {
		if len(args) < 2 || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "filter expects (f, span)"}
		}
		out := make([]Value, 0, len(args[1].Span))
		for _, x := range args[1].Span {
			v, err := it.apply(args[0], []Value{x})
			if err != nil {
				return Void(), err
			}
			if v.AsFloat() != 0 {
				out = append(out, x)
			}
		}
		return Span(out), nil
	})
	nat("scale", func(args []Value) (Value, error) {
		if len(args) < 2 || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "scale expects (s, span)"}
		}
		s := args[0].AsFloat()
		out := make([]Value, len(args[1].Span))
		for i, x := range args[1].Span {
			out[i] = F32(s * x.AsFloat())
		}
		return Span(out), nil
	})
	nat("span_add", func(args []Value) (Value, error) {
		return spanOp(args, func(a, b float64) float64 { return a + b })
	})
	nat("span_mul", func(args []Value) (Value, error) {
		return spanOp(args, func(a, b float64) float64 { return a * b })
	})
	nat("dot", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "dot expects two spans"}
		}
		n := min(len(args[0].Span), len(args[1].Span))
		acc := 0.0
		for i := 0; i < n; i++ {
			acc += args[0].Span[i].AsFloat() * args[1].Span[i].AsFloat()
		}
		return F32(acc), nil
	})
	nat("trit_gemm", func(args []Value) (Value, error) {
		return tritGemm(args, false, false)
	})
	nat("trit_gemm_bitplane", func(args []Value) (Value, error) {
		return tritGemm(args, true, false)
	})
	nat("trit_pack", func(args []Value) (Value, error) {
		if len(args) < 1 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "trit_pack expects span"}
		}
		var pos, neg []Value
		for _, x := range args[0].Span {
			t := clampTrit(x.AsInt())
			p, n := 0, 0
			if t > 0 {
				p = 1
			} else if t < 0 {
				n = 1
			}
			pos = append(pos, Bit(p))
			neg = append(neg, Bit(n))
		}
		return Span([]Value{Span(pos), Span(neg)}), nil
	})
	nat("trit_unpack", func(args []Value) (Value, error) {
		if len(args) < 1 || args[0].Kind != VSpan || len(args[0].Span) != 2 {
			return Void(), &Error{Msg: "trit_unpack expects [pos,neg]"}
		}
		posS, negS := args[0].Span[0], args[0].Span[1]
		if posS.Kind != VSpan || negS.Kind != VSpan {
			return Void(), &Error{Msg: "planes must be spans"}
		}
		n := min(len(posS.Span), len(negS.Span))
		out := make([]Value, n)
		for i := 0; i < n; i++ {
			p := posS.Span[i].AsInt()
			ng := negS.Span[i].AsInt()
			t := 0
			if p != 0 && ng == 0 {
				t = 1
			} else if ng != 0 && p == 0 {
				t = -1
			}
			out[i] = Trit(t)
		}
		return Span(out), nil
	})
	// Dense word packing: 32 trits → one i32 pos/neg word (bit 0 = index 0).
	nat("trit_pack_words", func(args []Value) (Value, error) {
		if len(args) < 1 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "trit_pack_words expects span"}
		}
		posW, negW, n := packTritWords(args[0].Span)
		return Span([]Value{i32Span(posW), i32Span(negW), I32(n)}), nil
	})
	nat("trit_unpack_words", func(args []Value) (Value, error) {
		if len(args) < 3 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "trit_unpack_words expects (pos_w, neg_w, n)"}
		}
		n := args[2].AsInt()
		out := unpackTritWords(toU32s(args[0].Span), toU32s(args[1].Span), n)
		return tritSpan(out), nil
	})
	// Plane boolean combinators (word spans, bitwise).
	nat("trit_plane_and", func(args []Value) (Value, error) {
		return planeBinop(args, func(a, b uint32) uint32 { return a & b })
	})
	nat("trit_plane_or", func(args []Value) (Value, error) {
		return planeBinop(args, func(a, b uint32) uint32 { return a | b })
	})
	nat("trit_plane_xor", func(args []Value) (Value, error) {
		return planeBinop(args, func(a, b uint32) uint32 { return a ^ b })
	})
	nat("trit_plane_popcount", func(args []Value) (Value, error) {
		if len(args) < 1 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "trit_plane_popcount expects word span"}
		}
		acc := 0
		for _, w := range args[0].Span {
			acc += bits.OnesCount32(uint32(w.AsInt()))
		}
		return I32(acc), nil
	})
	// Boolean product of two trit vectors from packed planes → product planes.
	nat("trit_mul_planes", func(args []Value) (Value, error) {
		if len(args) < 4 || args[0].Kind != VSpan || args[1].Kind != VSpan ||
			args[2].Kind != VSpan || args[3].Kind != VSpan {
			return Void(), &Error{Msg: "trit_mul_planes expects (ap,an,bp,bn) word spans"}
		}
		ap, an := toU32s(args[0].Span), toU32s(args[1].Span)
		bp, bn := toU32s(args[2].Span), toU32s(args[3].Span)
		nw := min(len(ap), min(len(an), min(len(bp), len(bn))))
		pp, pn := make([]uint32, nw), make([]uint32, nw)
		for i := 0; i < nw; i++ {
			pp[i] = (ap[i] & bp[i]) | (an[i] & bn[i])
			pn[i] = (ap[i] & bn[i]) | (an[i] & bp[i])
		}
		return Span([]Value{u32Span(pp), u32Span(pn)}), nil
	})
	// Packed bitplane dot: popcount(p_pos)-popcount(p_neg) over n lanes.
	nat("trit_dot_planes", func(args []Value) (Value, error) {
		if len(args) < 5 || args[0].Kind != VSpan || args[1].Kind != VSpan ||
			args[2].Kind != VSpan || args[3].Kind != VSpan {
			return Void(), &Error{Msg: "trit_dot_planes expects (ap,an,bp,bn,n)"}
		}
		n := args[4].AsInt()
		return I32(tritDotPlanes(toU32s(args[0].Span), toU32s(args[1].Span),
			toU32s(args[2].Span), toU32s(args[3].Span), n)), nil
	})
	// SIMD-friendly multiword dot: same math, 4×32-lane (128) primary chunks + residual.
	nat("trit_dot_planes_simd", func(args []Value) (Value, error) {
		if len(args) < 5 || args[0].Kind != VSpan || args[1].Kind != VSpan ||
			args[2].Kind != VSpan || args[3].Kind != VSpan {
			return Void(), &Error{Msg: "trit_dot_planes_simd expects (ap,an,bp,bn,n)"}
		}
		n := args[4].AsInt()
		return I32(tritDotPlanesSIMD(toU32s(args[0].Span), toU32s(args[1].Span),
			toU32s(args[2].Span), toU32s(args[3].Span), n)), nil
	})
	nat("trit_gemm_bitplane_simd", func(args []Value) (Value, error) {
		return tritGemm(args, true /*bitplane*/, true /*simd*/)
	})
	// Preferred SIMD lane width for multiword kernels (4×32 = 128).
	nat("simd_lane_width", func(args []Value) (Value, error) {
		return I32(SimdLaneWidth), nil
	})
	// Hardware-aware / dynamic tile queries (v0.28–v0.32 language surface).
	// Interpreter uses a portable default tile of 16 (matches @trin_vlen);
	// LLVM full-program path dispatches via runtime CPUID.
	nat("hw_vlen", func(args []Value) (Value, error) {
		return I32(16), nil
	})
	nat("hw_isa_code", func(args []Value) (Value, error) {
		return I32(0), nil // portable
	})
	// Host RDVL / streaming-SVE SME VL surface (v0.45–v0.46); portable default 16.
	nat("hw_rdvl", func(args []Value) (Value, error) {
		return I32(16), nil
	})
	nat("hw_sme_vl", func(args []Value) (Value, error) {
		return I32(16), nil
	})
	// SME ZA tile schedule host surface (v0.47): returns preferred ZA tile edge
	// (portable: same as streaming VL / preferred f32 tile).
	nat("hw_sme_za_tile", func(args []Value) (Value, error) {
		return I32(16), nil
	})
	// SME ZA outer-product schedule (v0.48): preferred outer-product tile edge.
	nat("hw_sme_za_outer", func(args []Value) (Value, error) {
		return I32(16), nil
	})
	// SME ZA accumulating outer-product micro-kernel tile edge (v0.51); portable = outer.
	nat("hw_sme_za_outer_acc", func(args []Value) (Value, error) {
		return I32(16), nil
	})
	// Adaptive multi-kernel fusion cost model (v0.48): returns a portable cost score.
	// cost(n) ≈ n (element visits); probe uses n=10 → 10.
	nat("fusion_cost", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return I32(0), nil
		}
		switch args[0].Kind {
		case VI32:
			return I32(args[0].I32), nil
		case VF32:
			return I32(int(args[0].F32)), nil
		default:
			return I32(0), nil
		}
	})
	// Weighted fusion cost (v0.51): cost = n * traffic_weight (portable ISA weight=1).
	nat("fusion_cost_weighted", func(args []Value) (Value, error) {
		n, w := 0, 1
		if len(args) >= 1 {
			switch args[0].Kind {
			case VI32:
				n = args[0].I32
			case VF32:
				n = int(args[0].F32)
			}
		}
		if len(args) >= 2 {
			switch args[1].Kind {
			case VI32:
				w = args[1].I32
			case VF32:
				w = int(args[1].F32)
			}
		}
		return I32(n * w), nil
	})
	// ISA-weighted fusion cost (v0.51): portable ISA weight = 1 → same as fusion_cost(n).
	nat("fusion_cost_isa", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return I32(0), nil
		}
		n := 0
		switch args[0].Kind {
		case VI32:
			n = args[0].I32
		case VF32:
			n = int(args[0].F32)
		}
		return I32(n * 1), nil // portable weight
	})
	// Multi-candidate fusion selection (v0.51): argmin of up to 3 costs → index 0/1/2.
	nat("fusion_select", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return I32(0), nil
		}
		bestI, bestC := 0, 1<<30
		for i := 0; i < len(args) && i < 3; i++ {
			c := 0
			switch args[i].Kind {
			case VI32:
				c = args[i].I32
			case VF32:
				c = int(args[i].F32)
			}
			if c < bestC {
				bestC = c
				bestI = i
			}
		}
		return I32(bestI), nil
	})
	// Language-level pair surface (v0.46): pair is a 2-span; first/second extract slots.
	// v0.48: multi-way pair(a,b,c,...) packs as a flat span of slots.
	nat("pair", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "pair expects (a, b[, ...])"}
		}
		return Span(args), nil
	})
	// pair_match(p, i): extract leaf i from a (possibly nested) pair/span (v0.48).
	nat("pair_match", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "pair_match expects (pair, index)"}
		}
		leaves := flattenPairLeaves(args[0])
		idx := 0
		switch args[1].Kind {
		case VI32:
			idx = args[1].I32
		case VF32:
			idx = int(args[1].F32)
		}
		if idx < 0 || idx >= len(leaves) {
			return Void(), &Error{Msg: "pair_match index OOB"}
		}
		return leaves[idx], nil
	})
	// pair_arity(p): leaf count of nested pair (v0.51 exhaustiveness surface).
	nat("pair_arity", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return I32(0), nil
		}
		return I32(len(flattenPairLeaves(args[0]))), nil
	})
	// pair_exhaust(p, n): 1 if leaf arity == n else 0 (v0.51 basic exhaustiveness check).
	nat("pair_exhaust", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return I32(0), nil
		}
		n := 0
		switch args[1].Kind {
		case VI32:
			n = args[1].I32
		case VF32:
			n = int(args[1].F32)
		}
		if len(flattenPairLeaves(args[0])) == n {
			return I32(1), nil
		}
		return I32(0), nil
	})
	// pair_of_pairs(a,b,c,d): nest two pairs (v0.51 typed pair-of-pairs surface).
	nat("pair_of_pairs", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return Void(), &Error{Msg: "pair_of_pairs expects (a,b,c,d)"}
		}
		return Span([]Value{
			Span([]Value{args[0], args[1]}),
			Span([]Value{args[2], args[3]}),
		}), nil
	})
	// sme_za_outer_acc(a,b): portable outer-product accumulate = sum(a)*sum(b) (v0.51).
	// Accepts two spans; falls back to 0 on bad args.
	nat("sme_za_outer_acc", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		sa, sb := 0.0, 0.0
		for _, e := range args[0].Span {
			switch e.Kind {
			case VF32:
				sa += e.F32
			case VI32:
				sa += float64(e.I32)
			}
		}
		for _, e := range args[1].Span {
			switch e.Kind {
			case VF32:
				sb += e.F32
			case VI32:
				sb += float64(e.I32)
			}
		}
		return F32(sa * sb), nil
	})
	// True ZA outer-product accumulate host edge (v0.51); portable = outer_acc.
	nat("hw_sme_za_outer_acc_true", func(args []Value) (Value, error) {
		return I32(16), nil
	})
	// pair_wildcard(): bind-any sentinel (span of one void-like 0 slot flag as i32 -1).
	nat("pair_wildcard", func(args []Value) (Value, error) {
		return I32(-1), nil // wildcard index sentinel
	})
	// pair_guard(p, i, lo, hi): 1 if leaf i is in [lo,hi] else 0.
	nat("pair_guard", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return I32(0), nil
		}
		leaves := flattenPairLeaves(args[0])
		idx := 0
		switch args[1].Kind {
		case VI32:
			idx = args[1].I32
		case VF32:
			idx = int(args[1].F32)
		}
		if idx < 0 || idx >= len(leaves) {
			return I32(0), nil
		}
		v := leaves[idx].AsFloat()
		lo, hi := args[2].AsFloat(), args[3].AsFloat()
		if v >= lo && v <= hi {
			return I32(1), nil
		}
		return I32(0), nil
	})
	// pair_match_guard(p, i, lo, hi): leaf i if guard holds else 0 (recursive match surface).
	nat("pair_match_guard", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		leaves := flattenPairLeaves(args[0])
		idx := 0
		switch args[1].Kind {
		case VI32:
			idx = args[1].I32
		case VF32:
			idx = int(args[1].F32)
		}
		if idx < 0 || idx >= len(leaves) {
			return F32(0), nil
		}
		v := leaves[idx]
		x := v.AsFloat()
		lo, hi := args[2].AsFloat(), args[3].AsFloat()
		if x < lo || x > hi {
			return F32(0), nil
		}
		return v, nil
	})
	// fusion_cost_calibrated(n): n * portable calibrated ISA traffic weight (weight=2).
	nat("fusion_cost_calibrated", func(args []Value) (Value, error) {
		n := 0
		if len(args) >= 1 {
			switch args[0].Kind {
			case VI32:
				n = args[0].I32
			case VF32:
				n = int(args[0].F32)
			}
		}
		return I32(n * 2), nil // portable calibrated weight table entry
	})
	// fusion_plan_log(c0,c1,c2): planner log surface = selected index (0/1/2) via argmin.
	nat("fusion_plan_log", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return I32(0), nil
		}
		bestI, bestC := 0, 1<<30
		for i := 0; i < len(args) && i < 3; i++ {
			c := 0
			switch args[i].Kind {
			case VI32:
				c = args[i].I32
			case VF32:
				c = int(args[i].F32)
			}
			if c < bestC {
				bestC = c
				bestI = i
			}
		}
		return I32(bestI), nil
	})
	// pair_or(a, b[, ...]): ADT or-pattern — pack alternatives as a span (v0.51).
	nat("pair_or", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "pair_or expects (a, b[, ...])"}
		}
		return Span(args), nil
	})
	// pair_or_match(p, i): select alternative branch i from or-pattern (v0.51).
	nat("pair_or_match", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "pair_or_match expects (pair_or, index)"}
		}
		leaves := flattenPairLeaves(args[0])
		// for or-patterns, prefer direct span items if top-level span of alternatives
		choices := leaves
		if args[0].Kind == VSpan {
			choices = args[0].Span
		}
		idx := 0
		switch args[1].Kind {
		case VI32:
			idx = args[1].I32
		case VF32:
			idx = int(args[1].F32)
		}
		if idx < 0 || idx >= len(choices) {
			return Void(), &Error{Msg: "pair_or_match index OOB"}
		}
		return choices[idx], nil
	})
	// pair_or_default(p): first alternative (or first leaf) (v0.51).
	nat("pair_or_default", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "pair_or_default expects pair_or"}
		}
		if args[0].Kind == VSpan && len(args[0].Span) > 0 {
			return args[0].Span[0], nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) == 0 {
			return F32(0), nil
		}
		return leaves[0], nil
	})
	// SME ZA streaming store host edge (v0.51); portable = outer_acc_true / preferred tile.
	nat("hw_sme_za_stream_store", func(args []Value) (Value, error) {
		return I32(16), nil
	})
	// sme_za_stream_store(a,b): portable streaming store path → sum(a)*sum(b) (same as outer-acc math).
	nat("sme_za_stream_store", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		sa, sb := 0.0, 0.0
		for _, e := range args[0].Span {
			switch e.Kind {
			case VF32:
				sa += e.F32
			case VI32:
				sa += float64(e.I32)
			}
		}
		for _, e := range args[1].Span {
			switch e.Kind {
			case VF32:
				sb += e.F32
			case VI32:
				sb += float64(e.I32)
			}
		}
		return F32(sa * sb), nil
	})
	// fusion_pgo_weight(n): n * portable PGO weight (weight=2).
	nat("fusion_pgo_weight", func(args []Value) (Value, error) {
		n := 0
		if len(args) >= 1 {
			switch args[0].Kind {
			case VI32:
				n = args[0].I32
			case VF32:
				n = int(args[0].F32)
			}
		}
		return I32(n * 2), nil
	})
	// fusion_pgo_feedback(selected, reward): feedback loop surface; returns selected+reward (clipped demo).
	nat("fusion_pgo_feedback", func(args []Value) (Value, error) {
		sel, rew := 0, 0
		if len(args) >= 1 {
			switch args[0].Kind {
			case VI32:
				sel = args[0].I32
			case VF32:
				sel = int(args[0].F32)
			}
		}
		if len(args) >= 2 {
			switch args[1].Kind {
			case VI32:
				rew = args[1].I32
			case VF32:
				rew = int(args[1].F32)
			}
		}
		return I32(sel + rew), nil
	})
	// fusion_pgo_select(c0,c1,c2): PGO-biased argmin (= fusion_plan_log for portable demo).
	nat("fusion_pgo_select", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return I32(0), nil
		}
		bestI, bestC := 0, 1<<30
		for i := 0; i < len(args) && i < 3; i++ {
			c := 0
			switch args[i].Kind {
			case VI32:
				c = args[i].I32
			case VF32:
				c = int(args[i].F32)
			}
			if c < bestC {
				bestC = c
				bestI = i
			}
		}
		return I32(bestI), nil
	})
	// pair_tag(tag, val): tagged ADT alternative (span [tag, val]) (v0.52).
	nat("pair_tag", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "pair_tag expects (tag, val)"}
		}
		return Span([]Value{args[0], args[1]}), nil
	})
	// pair_tag_match(p, want): if p.tag==want return val else -1 sentinel (v0.52).
	nat("pair_tag_match", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return F32(-1), nil
		}
		var tag, val Value
		if args[0].Kind == VSpan && len(args[0].Span) >= 2 {
			tag, val = args[0].Span[0], args[0].Span[1]
		} else {
			leaves := flattenPairLeaves(args[0])
			if len(leaves) < 2 {
				return F32(-1), nil
			}
			tag, val = leaves[0], leaves[1]
		}
		want := args[1]
		tagI, wantI := 0, 0
		switch tag.Kind {
		case VI32:
			tagI = tag.I32
		case VF32:
			tagI = int(tag.F32)
		}
		switch want.Kind {
		case VI32:
			wantI = want.I32
		case VF32:
			wantI = int(want.F32)
		}
		if tagI != wantI {
			return F32(-1), nil
		}
		return val, nil
	})
	// pair_tag_exhaust(p, tags...): 1 if p.tag is one of tags (v0.52).
	nat("pair_tag_exhaust", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return I32(0), nil
		}
		var tagI int
		if args[0].Kind == VSpan && len(args[0].Span) >= 1 {
			switch args[0].Span[0].Kind {
			case VI32:
				tagI = args[0].Span[0].I32
			case VF32:
				tagI = int(args[0].Span[0].F32)
			}
		} else {
			leaves := flattenPairLeaves(args[0])
			if len(leaves) >= 1 {
				switch leaves[0].Kind {
				case VI32:
					tagI = leaves[0].I32
				case VF32:
					tagI = int(leaves[0].F32)
				}
			}
		}
		for _, a := range args[1:] {
			w := 0
			switch a.Kind {
			case VI32:
				w = a.I32
			case VF32:
				w = int(a.F32)
			case VSpan:
				// allow span of tags
				for _, e := range a.Span {
					w2 := 0
					switch e.Kind {
					case VI32:
						w2 = e.I32
					case VF32:
						w2 = int(e.F32)
					}
					if tagI == w2 {
						return I32(1), nil
					}
				}
				continue
			}
			if tagI == w {
				return I32(1), nil
			}
		}
		return I32(0), nil
	})
	// SME ZA streaming gather host edge (v0.52); portable = 16
	nat("hw_sme_za_gather", func(args []Value) (Value, error) {
		return I32(16), nil
	})
	// SME ZA streaming scatter host edge (v0.52); portable = 16
	nat("hw_sme_za_scatter", func(args []Value) (Value, error) {
		return I32(16), nil
	})
	// sme_za_gather(a, idx): sum a[i] for i in idx (portable) (v0.52).
	nat("sme_za_gather", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter(a, idx, v): sum v written at idx positions count * v (portable math probe).
	nat("sme_za_scatter", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_adapt(n, counters...): online weight = 2 + mean(counters)/scale; cost = n*weight (v0.52).
	nat("fusion_pgo_adapt", func(args []Value) (Value, error) {
		n := 0
		if len(args) >= 1 {
			switch args[0].Kind {
			case VI32:
				n = args[0].I32
			case VF32:
				n = int(args[0].F32)
			}
		}
		w := 2
		if len(args) >= 2 {
			sum, cnt := 0, 0
			for _, a := range args[1:] {
				switch a.Kind {
				case VI32:
					sum += a.I32
					cnt++
				case VF32:
					sum += int(a.F32)
					cnt++
				}
			}
			if cnt > 0 {
				// portable demo: weight = 2 + sum/cnt/10  (integer)
				w = 2 + (sum/cnt)/10
				if w < 1 {
					w = 1
				}
			}
		}
		return I32(n * w), nil
	})
	// fusion_pgo_counter_bump(c, delta): c+delta (runtime counter) (v0.52).
	nat("fusion_pgo_counter_bump", func(args []Value) (Value, error) {
		c, d := 0, 1
		if len(args) >= 1 {
			switch args[0].Kind {
			case VI32:
				c = args[0].I32
			case VF32:
				c = int(args[0].F32)
			}
		}
		if len(args) >= 2 {
			switch args[1].Kind {
			case VI32:
				d = args[1].I32
			case VF32:
				d = int(args[1].F32)
			}
		}
		return I32(c + d), nil
	})
	// nested_tag(tag, a, b): nested tagged pair ADT — span [tag, a, b] (v0.53).
	nat("nested_tag", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Void(), &Error{Msg: "nested_tag expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_match(p, want): if tag==want return sum(a,b) else -1 (v0.53 pattern sugar).
	nat("nested_tag_match", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return F32(-1), nil
		}
		var tag, a, b Value
		if args[0].Kind == VSpan && len(args[0].Span) >= 3 {
			tag, a, b = args[0].Span[0], args[0].Span[1], args[0].Span[2]
		} else {
			leaves := flattenPairLeaves(args[0])
			if len(leaves) < 3 {
				return F32(-1), nil
			}
			tag, a, b = leaves[0], leaves[1], leaves[2]
		}
		tagI, wantI := 0, 0
		switch tag.Kind {
		case VI32:
			tagI = tag.I32
		case VF32:
			tagI = int(tag.F32)
		}
		switch args[1].Kind {
		case VI32:
			wantI = args[1].I32
		case VF32:
			wantI = int(args[1].F32)
		}
		if tagI != wantI {
			return F32(-1), nil
		}
		af, bf := 0.0, 0.0
		switch a.Kind {
		case VF32:
			af = a.F32
		case VI32:
			af = float64(a.I32)
		}
		switch b.Kind {
		case VF32:
			bf = b.F32
		case VI32:
			bf = float64(b.I32)
		}
		return F32(af + bf), nil
	})
	// nested_tag_sugar(tag, a, b): sugar alias of nested_tag (v0.53).
	nat("nested_tag_sugar", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Void(), &Error{Msg: "nested_tag_sugar expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// SME ZA multi-vector gather host (v0.53); portable = 16
	nat("hw_sme_za_gather_mv", func(args []Value) (Value, error) {
		return I32(16), nil
	})
	// SME ZA multi-vector scatter host (v0.53); portable = 16
	nat("hw_sme_za_scatter_mv", func(args []Value) (Value, error) {
		return I32(16), nil
	})
	// sme_za_gather_mv(a, idx): multi-vector gather = sum a[i] for i in idx (v0.53).
	nat("sme_za_gather_mv", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mv(a, idx, v): n * v (portable probe math) (v0.53).
	nat("sme_za_scatter_mv", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_isa_counter(isa, hits): portable ISA weight ladder + hits (v0.53).
	// portable→1, sse→2, avx→3, avx512/sve→4, neon→2; returns weight+hits (demo).
	nat("fusion_pgo_isa_counter", func(args []Value) (Value, error) {
		isa, hits := 0, 0
		if len(args) >= 1 {
			switch args[0].Kind {
			case VI32:
				isa = args[0].I32
			case VF32:
				isa = int(args[0].F32)
			}
		}
		if len(args) >= 2 {
			switch args[1].Kind {
			case VI32:
				hits = args[1].I32
			case VF32:
				hits = int(args[1].F32)
			}
		}
		w := 1
		switch isa {
		case 1: // sse
			w = 2
		case 2: // avx
			w = 3
		case 3, 4: // avx512 / sve
			w = 4
		case 5: // neon
			w = 2
		default:
			w = 1
		}
		return I32(w + hits), nil
	})
	// fusion_pgo_isa_loop(n, isa, hits): closed-loop cost = n * (w + hits/10) (v0.53).
	nat("fusion_pgo_isa_loop", func(args []Value) (Value, error) {
		n, isa, hits := 0, 0, 0
		if len(args) >= 1 {
			switch args[0].Kind {
			case VI32:
				n = args[0].I32
			case VF32:
				n = int(args[0].F32)
			}
		}
		if len(args) >= 2 {
			switch args[1].Kind {
			case VI32:
				isa = args[1].I32
			case VF32:
				isa = int(args[1].F32)
			}
		}
		if len(args) >= 3 {
			switch args[2].Kind {
			case VI32:
				hits = args[2].I32
			case VF32:
				hits = int(args[2].F32)
			}
		}
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		default:
			w = 1
		}
		w = w + hits/10
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// nested_tag_multi(tag, a, b, c): multi-way nested tagged ADT — span [tag,a,b,c] (v0.54).
	nat("nested_tag_multi", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return Void(), &Error{Msg: "nested_tag_multi expects (tag, a, b, c)"}
		}
		return Span([]Value{args[0], args[1], args[2], args[3]}), nil
	})
	// nested_tag_multi_match(p, tags...): if tag in tags return sum(a,b,c) else -1 (v0.54 multi-way sugar).
	nat("nested_tag_multi_match", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return F32(-1), nil
		}
		var tag, a, b, c Value
		if args[0].Kind == VSpan && len(args[0].Span) >= 4 {
			tag, a, b, c = args[0].Span[0], args[0].Span[1], args[0].Span[2], args[0].Span[3]
		} else {
			leaves := flattenPairLeaves(args[0])
			if len(leaves) < 4 {
				return F32(-1), nil
			}
			tag, a, b, c = leaves[0], leaves[1], leaves[2], leaves[3]
		}
		tagI := 0
		switch tag.Kind {
		case VI32:
			tagI = tag.I32
		case VF32:
			tagI = int(tag.F32)
		}
		ok := false
		for _, w := range args[1:] {
			wI := 0
			switch w.Kind {
			case VI32:
				wI = w.I32
			case VF32:
				wI = int(w.F32)
			}
			if tagI == wI {
				ok = true
				break
			}
		}
		if !ok {
			return F32(-1), nil
		}
		af, bf, cf := 0.0, 0.0, 0.0
		switch a.Kind {
		case VF32:
			af = a.F32
		case VI32:
			af = float64(a.I32)
		}
		switch b.Kind {
		case VF32:
			bf = b.F32
		case VI32:
			bf = float64(b.I32)
		}
		switch c.Kind {
		case VF32:
			cf = c.F32
		case VI32:
			cf = float64(c.I32)
		}
		return F32(af + bf + cf), nil
	})
	// nested_tag_or_sugar(tag, a, b, c, tags...): pack then multi-match (v0.54).
	nat("nested_tag_or_sugar", func(args []Value) (Value, error) {
		if len(args) < 5 {
			return F32(-1), nil
		}
		p := Span([]Value{args[0], args[1], args[2], args[3]})
		margs := append([]Value{p}, args[4:]...)
		// inline multi match
		tagI := 0
		switch args[0].Kind {
		case VI32:
			tagI = args[0].I32
		case VF32:
			tagI = int(args[0].F32)
		}
		ok := false
		for _, w := range args[4:] {
			wI := 0
			switch w.Kind {
			case VI32:
				wI = w.I32
			case VF32:
				wI = int(w.F32)
			}
			if tagI == wI {
				ok = true
				break
			}
		}
		if !ok {
			return F32(-1), nil
		}
		af, bf, cf := 0.0, 0.0, 0.0
		switch args[1].Kind {
		case VF32:
			af = args[1].F32
		case VI32:
			af = float64(args[1].I32)
		}
		switch args[2].Kind {
		case VF32:
			bf = args[2].F32
		case VI32:
			bf = float64(args[2].I32)
		}
		switch args[3].Kind {
		case VF32:
			cf = args[3].F32
		case VI32:
			cf = float64(args[3].I32)
		}
		_ = margs
		return F32(af + bf + cf), nil
	})
	// SME ZA tiled multivec stream gather host (v0.54); portable = 32
	nat("hw_sme_za_gather_tile", func(args []Value) (Value, error) {
		return I32(32), nil
	})
	// SME ZA tiled multivec stream scatter host (v0.54); portable = 32
	nat("hw_sme_za_scatter_tile", func(args []Value) (Value, error) {
		return I32(32), nil
	})
	// sme_za_gather_tile(a, idx): tiled stream gather = sum a[i] for i in idx (v0.54).
	nat("sme_za_gather_tile", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_tile(a, idx, v): n * v (portable) (v0.54).
	nat("sme_za_scatter_tile", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_isa_live(n, isa, hits, live): online feedback cost = n * (w + hits/10 + live/20) (v0.54).
	nat("fusion_pgo_isa_live", func(args []Value) (Value, error) {
		n, isa, hits, live := 0, 0, 0, 0
		if len(args) >= 1 {
			switch args[0].Kind {
			case VI32:
				n = args[0].I32
			case VF32:
				n = int(args[0].F32)
			}
		}
		if len(args) >= 2 {
			switch args[1].Kind {
			case VI32:
				isa = args[1].I32
			case VF32:
				isa = int(args[1].F32)
			}
		}
		if len(args) >= 3 {
			switch args[2].Kind {
			case VI32:
				hits = args[2].I32
			case VF32:
				hits = int(args[2].F32)
			}
		}
		if len(args) >= 4 {
			switch args[3].Kind {
			case VI32:
				live = args[3].I32
			case VF32:
				live = int(args[3].F32)
			}
		}
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		default:
			w = 1
		}
		w = w + hits/10 + live/20
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_isa_feedback(hits, live): live counter feedback = hits + live (v0.54).
	nat("fusion_pgo_isa_feedback", func(args []Value) (Value, error) {
		hits, live := 0, 0
		if len(args) >= 1 {
			switch args[0].Kind {
			case VI32:
				hits = args[0].I32
			case VF32:
				hits = int(args[0].F32)
			}
		}
		if len(args) >= 2 {
			switch args[1].Kind {
			case VI32:
				live = args[1].I32
			case VF32:
				live = int(args[1].F32)
			}
		}
		return I32(hits + live), nil
	})
	// nested_tag_recur(tag, a, b): recursive multi-way ADT leaf/node span [tag,a,b] (v0.55).
	nat("nested_tag_recur", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Void(), &Error{Msg: "nested_tag_recur expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_recur_match(p, tags...): flatten recursive ADT, if root tag matches any, sum leaves else -1 (v0.55).
	nat("nested_tag_recur_match", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return F32(-1), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 2 {
			return F32(-1), nil
		}
		tagI := 0
		switch leaves[0].Kind {
		case VI32:
			tagI = leaves[0].I32
		case VF32:
			tagI = int(leaves[0].F32)
		}
		ok := false
		for _, w := range args[1:] {
			wI := 0
			switch w.Kind {
			case VI32:
				wI = w.I32
			case VF32:
				wI = int(w.F32)
			}
			if tagI == wI {
				ok = true
				break
			}
		}
		if !ok {
			return F32(-1), nil
		}
		s := 0.0
		for _, e := range leaves[1:] {
			switch e.Kind {
			case VF32:
				s += e.F32
			case VI32:
				s += float64(e.I32)
			}
		}
		return F32(s), nil
	})
	// nested_tag_recur_sugar(tag, a, b, tags...): pack then recursive multi-match (v0.55).
	nat("nested_tag_recur_sugar", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(-1), nil
		}
		p := Span([]Value{args[0], args[1], args[2]})
		margs := append([]Value{p}, args[3:]...)
		// reuse multi match logic
		leaves := flattenPairLeaves(p)
		tagI := 0
		switch args[0].Kind {
		case VI32:
			tagI = args[0].I32
		case VF32:
			tagI = int(args[0].F32)
		}
		ok := false
		for _, w := range args[3:] {
			wI := 0
			switch w.Kind {
			case VI32:
				wI = w.I32
			case VF32:
				wI = int(w.F32)
			}
			if tagI == wI {
				ok = true
				break
			}
		}
		if !ok {
			return F32(-1), nil
		}
		s := 0.0
		for _, e := range leaves[1:] {
			switch e.Kind {
			case VF32:
				s += e.F32
			case VI32:
				s += float64(e.I32)
			}
		}
		_ = margs
		return F32(s), nil
	})
	// SME ZA multi-tile gather host (v0.55); portable = 48
	nat("hw_sme_za_gather_mt", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// SME ZA multi-tile scatter host (v0.55); portable = 48
	nat("hw_sme_za_scatter_mt", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt(a, idx): multi-tile stream gather = sum a[i] for i in idx (v0.55).
	nat("sme_za_gather_mt", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt(a, idx, v): n * v (portable) (v0.55).
	nat("sme_za_scatter_mt", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_xkernel(n, isa, hits, live, xk): cross-kernel cost = n * (w + hits/10 + live/20 + xk/5) (v0.55).
	nat("fusion_pgo_xkernel", func(args []Value) (Value, error) {
		n, isa, hits, live, xk := 0, 0, 0, 0, 0
		if len(args) >= 1 {
			switch args[0].Kind {
			case VI32:
				n = args[0].I32
			case VF32:
				n = int(args[0].F32)
			}
		}
		if len(args) >= 2 {
			switch args[1].Kind {
			case VI32:
				isa = args[1].I32
			case VF32:
				isa = int(args[1].F32)
			}
		}
		if len(args) >= 3 {
			switch args[2].Kind {
			case VI32:
				hits = args[2].I32
			case VF32:
				hits = int(args[2].F32)
			}
		}
		if len(args) >= 4 {
			switch args[3].Kind {
			case VI32:
				live = args[3].I32
			case VF32:
				live = int(args[3].F32)
			}
		}
		if len(args) >= 5 {
			switch args[4].Kind {
			case VI32:
				xk = args[4].I32
			case VF32:
				xk = int(args[4].F32)
			}
		}
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		default:
			w = 1
		}
		w = w + hits/10 + live/20 + xk/5
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_xkernel_counter(hits, live, xk): cross-kernel counter = hits + live + xk (v0.55).
	nat("fusion_pgo_xkernel_counter", func(args []Value) (Value, error) {
		hits, live, xk := 0, 0, 0
		if len(args) >= 1 {
			switch args[0].Kind {
			case VI32:
				hits = args[0].I32
			case VF32:
				hits = int(args[0].F32)
			}
		}
		if len(args) >= 2 {
			switch args[1].Kind {
			case VI32:
				live = args[1].I32
			case VF32:
				live = int(args[1].F32)
			}
		}
		if len(args) >= 3 {
			switch args[2].Kind {
			case VI32:
				xk = args[2].I32
			case VF32:
				xk = int(args[2].F32)
			}
		}
		return I32(hits + live + xk), nil
	})
	// nested_tag_ser(tag, a, b): serialize ADT as span [tag,a,b] (v0.56).
	nat("nested_tag_ser", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Void(), &Error{Msg: "nested_tag_ser expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_tree_fold(p): serialize then fold = sum leaves after tag (v0.56).
	nat("nested_tag_tree_fold", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return F32(0), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 2 {
			return F32(0), nil
		}
		s := 0.0
		for _, e := range leaves[1:] {
			switch e.Kind {
			case VF32:
				s += e.F32
			case VI32:
				s += float64(e.I32)
			}
		}
		return F32(s), nil
	})
	// nested_tag_ser_fold(tag, a, b): pack then tree-fold (v0.56).
	nat("nested_tag_ser_fold", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		p := Span([]Value{args[0], args[1], args[2]})
		leaves := flattenPairLeaves(p)
		s := 0.0
		for _, e := range leaves[1:] {
			switch e.Kind {
			case VF32:
				s += e.F32
			case VI32:
				s += float64(e.I32)
			}
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peel gather host (v0.56); portable = 48
	nat("hw_sme_za_gather_mt_peel", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// SME ZA multi-tile peel scatter host (v0.56); portable = 48
	nat("hw_sme_za_scatter_mt_peel", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_peel(a, idx): residual-peel multi-tile gather = sum a[i] (v0.56).
	nat("sme_za_gather_mt_peel", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_peel(a, idx, v): n * v (portable) (v0.56).
	nat("sme_za_scatter_mt_peel", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_xpass(n, isa, hits, live, xk, pass): multi-pass cost = n * (w + hits/10 + live/20 + xk/5 + pass/2) (v0.56).
	nat("fusion_pgo_xpass", func(args []Value) (Value, error) {
		n, isa, hits, live, xk, pass := 0, 0, 0, 0, 0, 0
		if len(args) >= 1 {
			switch args[0].Kind {
			case VI32:
				n = args[0].I32
			case VF32:
				n = int(args[0].F32)
			}
		}
		if len(args) >= 2 {
			switch args[1].Kind {
			case VI32:
				isa = args[1].I32
			case VF32:
				isa = int(args[1].F32)
			}
		}
		if len(args) >= 3 {
			switch args[2].Kind {
			case VI32:
				hits = args[2].I32
			case VF32:
				hits = int(args[2].F32)
			}
		}
		if len(args) >= 4 {
			switch args[3].Kind {
			case VI32:
				live = args[3].I32
			case VF32:
				live = int(args[3].F32)
			}
		}
		if len(args) >= 5 {
			switch args[4].Kind {
			case VI32:
				xk = args[4].I32
			case VF32:
				xk = int(args[4].F32)
			}
		}
		if len(args) >= 6 {
			switch args[5].Kind {
			case VI32:
				pass = args[5].I32
			case VF32:
				pass = int(args[5].F32)
			}
		}
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		default:
			w = 1
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_xpass_counter(hits, live, xk, pass): multi-pass counter = hits + live + xk + pass (v0.56).
	nat("fusion_pgo_xpass_counter", func(args []Value) (Value, error) {
		hits, live, xk, pass := 0, 0, 0, 0
		if len(args) >= 1 {
			switch args[0].Kind {
			case VI32:
				hits = args[0].I32
			case VF32:
				hits = int(args[0].F32)
			}
		}
		if len(args) >= 2 {
			switch args[1].Kind {
			case VI32:
				live = args[1].I32
			case VF32:
				live = int(args[1].F32)
			}
		}
		if len(args) >= 3 {
			switch args[2].Kind {
			case VI32:
				xk = args[2].I32
			case VF32:
				xk = int(args[2].F32)
			}
		}
		if len(args) >= 4 {
			switch args[3].Kind {
			case VI32:
				pass = args[3].I32
			case VF32:
				pass = int(args[3].F32)
			}
		}
		return I32(hits + live + xk + pass), nil
	})
	// nested_tag_zip(p, q): zip flattened pair forests elementwise sum (v0.57).
	nat("nested_tag_zip", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), &Error{Msg: "nested_tag_zip expects (p, q)"}
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		out := make([]Value, n)
		for i := 0; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			out[i] = F32(av + bv)
		}
		return Span(out), nil
	})
	// nested_tag_map_fold(p): sum flattened forest after first tag (v0.57).
	nat("nested_tag_map_fold", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return F32(0), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 2 {
			return F32(0), nil
		}
		s := 0.0
		for _, e := range leaves[1:] {
			switch e.Kind {
			case VF32:
				s += e.F32
			case VI32:
				s += float64(e.I32)
			}
		}
		return F32(s), nil
	})
	// nested_tag_forest_fold(p, q): zip then map-fold (v0.57).
	nat("nested_tag_forest_fold", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		// skip first (tag) positions: fold zipped from index 1
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile auto peel gather host (v0.57); portable = 48
	nat("hw_sme_za_gather_mt_auto", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// SME ZA multi-tile auto peel scatter host (v0.57); portable = 48
	nat("hw_sme_za_scatter_mt_auto", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_auto(a, idx): auto-tile multi-tile gather = sum a[i] (v0.57).
	nat("sme_za_gather_mt_auto", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_auto(a, idx, v): n * v (portable) (v0.57).
	nat("sme_za_scatter_mt_auto", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_window(n, isa, hits, live, xk, pass, win): multi-pass windowed cost (v0.57).
	// = n * (w + hits/10 + live/20 + xk/5 + pass/2 + win/4)
	nat("fusion_pgo_window", func(args []Value) (Value, error) {
		n, isa, hits, live, xk, pass, win := 0, 0, 0, 0, 0, 0, 0
		if len(args) >= 1 {
			switch args[0].Kind {
			case VI32:
				n = args[0].I32
			case VF32:
				n = int(args[0].F32)
			}
		}
		if len(args) >= 2 {
			switch args[1].Kind {
			case VI32:
				isa = args[1].I32
			case VF32:
				isa = int(args[1].F32)
			}
		}
		if len(args) >= 3 {
			switch args[2].Kind {
			case VI32:
				hits = args[2].I32
			case VF32:
				hits = int(args[2].F32)
			}
		}
		if len(args) >= 4 {
			switch args[3].Kind {
			case VI32:
				live = args[3].I32
			case VF32:
				live = int(args[3].F32)
			}
		}
		if len(args) >= 5 {
			switch args[4].Kind {
			case VI32:
				xk = args[4].I32
			case VF32:
				xk = int(args[4].F32)
			}
		}
		if len(args) >= 6 {
			switch args[5].Kind {
			case VI32:
				pass = args[5].I32
			case VF32:
				pass = int(args[5].F32)
			}
		}
		if len(args) >= 7 {
			switch args[6].Kind {
			case VI32:
				win = args[6].I32
			case VF32:
				win = int(args[6].F32)
			}
		}
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		default:
			w = 1
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_window_counter(hits, live, xk, pass, win): = hits+live+xk+pass+win (v0.57).
	nat("fusion_pgo_window_counter", func(args []Value) (Value, error) {
		hits, live, xk, pass, win := 0, 0, 0, 0, 0
		if len(args) >= 1 {
			switch args[0].Kind {
			case VI32:
				hits = args[0].I32
			case VF32:
				hits = int(args[0].F32)
			}
		}
		if len(args) >= 2 {
			switch args[1].Kind {
			case VI32:
				live = args[1].I32
			case VF32:
				live = int(args[1].F32)
			}
		}
		if len(args) >= 3 {
			switch args[2].Kind {
			case VI32:
				xk = args[2].I32
			case VF32:
				xk = int(args[2].F32)
			}
		}
		if len(args) >= 4 {
			switch args[3].Kind {
			case VI32:
				pass = args[3].I32
			case VF32:
				pass = int(args[3].F32)
			}
		}
		if len(args) >= 5 {
			switch args[4].Kind {
			case VI32:
				win = args[4].I32
			case VF32:
				win = int(args[4].F32)
			}
		}
		return I32(hits + live + xk + pass + win), nil
	})
	// nested_tag_keyed(tag, a, b): tagged pair/forest node with key (v0.58).
	nat("nested_tag_keyed", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_keyed expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_keyed_fold(p, key): map-fold only when head tag matches key (v0.58).
	nat("nested_tag_keyed_fold", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return F32(0), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 2 {
			return F32(0), nil
		}
		key := 0
		switch args[1].Kind {
		case VI32:
			key = args[1].I32
		case VF32:
			key = int(args[1].F32)
		}
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key {
			return F32(0), nil
		}
		s := 0.0
		for _, e := range leaves[1:] {
			switch e.Kind {
			case VF32:
				s += e.F32
			case VI32:
				s += float64(e.I32)
			}
		}
		return F32(s), nil
	})
	// nested_tag_keyed_reduce(p, q, key): forest zip-reduce when both tags match key (v0.58).
	nat("nested_tag_keyed_reduce", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		key := 0
		switch args[2].Kind {
		case VI32:
			key = args[2].I32
		case VF32:
			key = int(args[2].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		if ta != key || tb != key {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile auto peels with host VL feedback (v0.58); portable = 48
	// = gather_mt_auto + (hw_vlen_for_n(16)-16) residual feedback (=0 portable)
	nat("hw_sme_za_gather_mt_vlfb", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vlfb", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vlfb(a, idx): VL-feedback multi-tile gather = sum a[i] (v0.58).
	nat("sme_za_gather_mt_vlfb", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vlfb(a, idx, v): n * v (portable) (v0.58).
	nat("sme_za_scatter_mt_vlfb", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_decay(n, isa, hits, live, xk, pass, win, decay): windowed cost with decay (v0.58).
	// = n * max(1, w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay)
	nat("fusion_pgo_decay", func(args []Value) (Value, error) {
		n, isa, hits, live, xk, pass, win, decay := 0, 0, 0, 0, 0, 0, 0, 0
		if len(args) >= 1 {
			switch args[0].Kind {
			case VI32:
				n = args[0].I32
			case VF32:
				n = int(args[0].F32)
			}
		}
		if len(args) >= 2 {
			switch args[1].Kind {
			case VI32:
				isa = args[1].I32
			case VF32:
				isa = int(args[1].F32)
			}
		}
		if len(args) >= 3 {
			switch args[2].Kind {
			case VI32:
				hits = args[2].I32
			case VF32:
				hits = int(args[2].F32)
			}
		}
		if len(args) >= 4 {
			switch args[3].Kind {
			case VI32:
				live = args[3].I32
			case VF32:
				live = int(args[3].F32)
			}
		}
		if len(args) >= 5 {
			switch args[4].Kind {
			case VI32:
				xk = args[4].I32
			case VF32:
				xk = int(args[4].F32)
			}
		}
		if len(args) >= 6 {
			switch args[5].Kind {
			case VI32:
				pass = args[5].I32
			case VF32:
				pass = int(args[5].F32)
			}
		}
		if len(args) >= 7 {
			switch args[6].Kind {
			case VI32:
				win = args[6].I32
			case VF32:
				win = int(args[6].F32)
			}
		}
		if len(args) >= 8 {
			switch args[7].Kind {
			case VI32:
				decay = args[7].I32
			case VF32:
				decay = int(args[7].F32)
			}
		}
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_decay_counter(hits, live, xk, pass, win, decay) (v0.58).
	nat("fusion_pgo_decay_counter", func(args []Value) (Value, error) {
		hits, live, xk, pass, win, decay := 0, 0, 0, 0, 0, 0
		if len(args) >= 1 {
			switch args[0].Kind {
			case VI32:
				hits = args[0].I32
			case VF32:
				hits = int(args[0].F32)
			}
		}
		if len(args) >= 2 {
			switch args[1].Kind {
			case VI32:
				live = args[1].I32
			case VF32:
				live = int(args[1].F32)
			}
		}
		if len(args) >= 3 {
			switch args[2].Kind {
			case VI32:
				xk = args[2].I32
			case VF32:
				xk = int(args[2].F32)
			}
		}
		if len(args) >= 4 {
			switch args[3].Kind {
			case VI32:
				pass = args[3].I32
			case VF32:
				pass = int(args[3].F32)
			}
		}
		if len(args) >= 5 {
			switch args[4].Kind {
			case VI32:
				win = args[4].I32
			case VF32:
				win = int(args[4].F32)
			}
		}
		if len(args) >= 6 {
			switch args[5].Kind {
			case VI32:
				decay = args[5].I32
			case VF32:
				decay = int(args[5].F32)
			}
		}
		return I32(hits + live + xk + pass + win + decay), nil
	})
	// nested_tag_filter(p, key): return payload span if tag matches key else empty (v0.59).
	nat("nested_tag_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		key := 0
		switch args[1].Kind {
		case VI32:
			key = args[1].I32
		case VF32:
			key = int(args[1].F32)
		}
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_multi_key(tag, a, b): same pack as keyed (v0.59 multi-key surface).
	nat("nested_tag_multi_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_multi_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_multi_reduce(p, q, k0, k1): forest zip-reduce when both tags in {k0,k1} (v0.59).
	nat("nested_tag_multi_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with live host VL + residual schedule (v0.59); portable = 48
	nat("hw_sme_za_gather_mt_vllive", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vllive", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vllive(a, idx): live VL residual multi-tile gather = sum a[i] (v0.59).
	nat("sme_za_gather_mt_vllive", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vllive(a, idx, v): n * v (portable) (v0.59).
	nat("sme_za_scatter_mt_vllive", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_ema(n, isa, hits, live, xk, pass, win, decay, half): EMA/half-life windowed cost (v0.59).
	// = n * max(1, w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5)
	// with half=0 → same as decay for probe (win=4 → 20)
	nat("fusion_pgo_ema", func(args []Value) (Value, error) {
		n, isa, hits, live, xk, pass, win, decay, half := 0, 0, 0, 0, 0, 0, 0, 0, 0
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half = get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_ema_counter(hits, live, xk, pass, win, decay, half) (v0.59).
	nat("fusion_pgo_ema_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6)), nil
	})

	// nested_tag_rank_filter(p, key, rank): filter when tag==key and rank>=0 (v0.60 ranked).
	nat("nested_tag_rank_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		key := 0
		switch args[1].Kind {
		case VI32:
			key = args[1].I32
		case VF32:
			key = int(args[1].F32)
		}
		rank := 0
		if len(args) >= 3 {
			switch args[2].Kind {
			case VI32:
				rank = args[2].I32
			case VF32:
				rank = int(args[2].F32)
			}
		}
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_rank_key(tag, a, b): pack with ranked multi-key surface (v0.60).
	nat("nested_tag_rank_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_rank_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_rank_reduce(p, q, k0, k1): ranked multi-key forest zip-reduce (v0.60).
	nat("nested_tag_rank_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with live VL residual streaming schedule (v0.60); portable = 48
	nat("hw_sme_za_gather_mt_vlstream", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vlstream", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vlstream(a, idx): live VL residual streaming multi-tile gather = sum a[i] (v0.60).
	nat("sme_za_gather_mt_vlstream", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vlstream(a, idx, v): n * v (portable) (v0.60).
	nat("sme_za_scatter_mt_vlstream", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_adapt_hl(n, isa, hits, live, xk, pass, win, decay, half): adaptive half-life EMA cost (v0.60).
	// = n * max(1, w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5)
	// with half=0 → same as decay for probe (win=4 → 20)
	nat("fusion_pgo_adapt_hl", func(args []Value) (Value, error) {
		n, isa, hits, live, xk, pass, win, decay, half := 0, 0, 0, 0, 0, 0, 0, 0, 0
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half = get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		// adaptive half-life term (half/5); half=0 keeps same probe as ema (win=4 → 20)
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_adapt_hl_counter(hits, live, xk, pass, win, decay, half) (v0.60).
	nat("fusion_pgo_adapt_hl_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6)), nil
	})

	// nested_tag_weight_filter(p, key, rank, weight): filter when tag==key, rank>=0, weight>=0 (v0.61 weighted).
	nat("nested_tag_weight_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		key := 0
		switch args[1].Kind {
		case VI32:
			key = args[1].I32
		case VF32:
			key = int(args[1].F32)
		}
		rank := 0
		if len(args) >= 3 {
			switch args[2].Kind {
			case VI32:
				rank = args[2].I32
			case VF32:
				rank = int(args[2].F32)
			}
		}
		weight := 0
		if len(args) >= 4 {
			switch args[3].Kind {
			case VI32:
				weight = args[3].I32
			case VF32:
				weight = int(args[3].F32)
			}
		}
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_weight_key(tag, a, b): pack with weighted multi-key surface (v0.61).
	nat("nested_tag_weight_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_weight_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_weight_reduce(p, q, k0, k1): weighted multi-key forest zip-reduce (v0.61).
	nat("nested_tag_weight_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with live VL residual streaming + cost model (v0.61); portable = 48
	nat("hw_sme_za_gather_mt_vlcost", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vlcost", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vlcost(a, idx): VL residual streaming + cost multi-tile gather = sum a[i] (v0.61).
	nat("sme_za_gather_mt_vlcost", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vlcost(a, idx, v): n * v (portable) (v0.61).
	nat("sme_za_scatter_mt_vlcost", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_hyst(n, isa, hits, live, xk, pass, win, decay, half): dual EMA / hysteresis adaptive cost (v0.61).
	// = n * max(1, w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5)
	// with half=0 → same probe path (win=4 → 20)
	nat("fusion_pgo_hyst", func(args []Value) (Value, error) {
		n, isa, hits, live, xk, pass, win, decay, half := 0, 0, 0, 0, 0, 0, 0, 0, 0
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half = get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		// dual EMA / hysteresis: half/5 term; half=0 keeps probe
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_hyst_counter(hits, live, xk, pass, win, decay, half) (v0.61).
	nat("fusion_pgo_hyst_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6)), nil
	})

	// nested_tag_thresh_filter(p, key, rank, weight, thresh): filter when tag==key, rank>=0, weight>=thresh (v0.62).
	nat("nested_tag_thresh_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		key := 0
		switch args[1].Kind {
		case VI32:
			key = args[1].I32
		case VF32:
			key = int(args[1].F32)
		}
		rank := 0
		if len(args) >= 3 {
			switch args[2].Kind {
			case VI32:
				rank = args[2].I32
			case VF32:
				rank = int(args[2].F32)
			}
		}
		weight := 0
		if len(args) >= 4 {
			switch args[3].Kind {
			case VI32:
				weight = args[3].I32
			case VF32:
				weight = int(args[3].F32)
			}
		}
		thresh := 0
		if len(args) >= 5 {
			switch args[4].Kind {
			case VI32:
				thresh = args[4].I32
			case VF32:
				thresh = int(args[4].F32)
			}
		}
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < thresh {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_thresh_key(tag, a, b): pack with score-threshold multi-key surface (v0.62).
	nat("nested_tag_thresh_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_thresh_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_thresh_reduce(p, q, k0, k1): score-threshold multi-key forest zip-reduce (v0.62).
	nat("nested_tag_thresh_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with live VL residual streaming cost + adaptive tile pick (v0.62); portable = 48
	nat("hw_sme_za_gather_mt_vladapt", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vladapt", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vladapt(a, idx): VL residual streaming cost + adaptive tile gather = sum a[i] (v0.62).
	nat("sme_za_gather_mt_vladapt", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vladapt(a, idx, v): n * v (portable) (v0.62).
	nat("sme_za_scatter_mt_vladapt", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_reweight(n, isa, hits, live, xk, pass, win, decay, half): dual EMA hysteresis with windowed reweight (v0.62).
	// = n * max(1, w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5)
	// with half=0 → same probe path (win=4 → 20)
	nat("fusion_pgo_reweight", func(args []Value) (Value, error) {
		n, isa, hits, live, xk, pass, win, decay, half := 0, 0, 0, 0, 0, 0, 0, 0, 0
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half = get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		// dual EMA hysteresis + windowed reweight; half=0 keeps probe
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_reweight_counter(hits, live, xk, pass, win, decay, half) (v0.62).
	nat("fusion_pgo_reweight_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6)), nil
	})

	// nested_tag_soft_filter(p, key, rank, weight, thresh, soft): soft-gated score-threshold filter (v0.63).
	// Passes when tag==key, rank>=0, weight>=thresh, and soft>=0.
	nat("nested_tag_soft_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		key := 0
		switch args[1].Kind {
		case VI32:
			key = args[1].I32
		case VF32:
			key = int(args[1].F32)
		}
		rank := 0
		if len(args) >= 3 {
			switch args[2].Kind {
			case VI32:
				rank = args[2].I32
			case VF32:
				rank = int(args[2].F32)
			}
		}
		weight := 0
		if len(args) >= 4 {
			switch args[3].Kind {
			case VI32:
				weight = args[3].I32
			case VF32:
				weight = int(args[3].F32)
			}
		}
		thresh := 0
		if len(args) >= 5 {
			switch args[4].Kind {
			case VI32:
				thresh = args[4].I32
			case VF32:
				thresh = int(args[4].F32)
			}
		}
		soft := 0
		if len(args) >= 6 {
			switch args[5].Kind {
			case VI32:
				soft = args[5].I32
			case VF32:
				soft = int(args[5].F32)
			}
		}
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < thresh || soft < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_soft_key(tag, a, b): pack with soft-gated multi-key surface (v0.63).
	nat("nested_tag_soft_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_soft_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_soft_reduce(p, q, k0, k1): soft-gated multi-key forest zip-reduce (v0.63).
	nat("nested_tag_soft_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with live VL residual streaming adaptive cost + residual schedule (v0.63); portable = 48
	nat("hw_sme_za_gather_mt_vlsched", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vlsched", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vlsched(a, idx): residual-schedule gather = sum a[i] (v0.63).
	nat("sme_za_gather_mt_vlsched", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vlsched(a, idx, v): n * v (portable) (v0.63).
	nat("sme_za_scatter_mt_vlsched", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_budget(n, isa, hits, live, xk, pass, win, decay, half, budget): dual EMA hysteresis windowed reweight with budget caps (v0.63).
	// = n * max(1, min(cap, w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5))
	// with half=0, budget=0 → same probe path (win=4 → 20)
	nat("fusion_pgo_budget", func(args []Value) (Value, error) {
		n, isa, hits, live, xk, pass, win, decay, half, budget := 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half, budget = get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5
		if budget > 0 && w > budget {
			w = budget
		}
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_budget_counter(hits, live, xk, pass, win, decay, half) (v0.63).
	nat("fusion_pgo_budget_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6)), nil
	})

	// nested_tag_cal_filter(p, key, rank, weight, thresh, soft, cal): calibrated soft-gated score-threshold filter (v0.64).
	// Passes when tag==key, rank>=0, weight>=thresh, soft>=0, cal>=0.
	nat("nested_tag_cal_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		key := 0
		switch args[1].Kind {
		case VI32:
			key = args[1].I32
		case VF32:
			key = int(args[1].F32)
		}
		rank := 0
		if len(args) >= 3 {
			switch args[2].Kind {
			case VI32:
				rank = args[2].I32
			case VF32:
				rank = int(args[2].F32)
			}
		}
		weight := 0
		if len(args) >= 4 {
			switch args[3].Kind {
			case VI32:
				weight = args[3].I32
			case VF32:
				weight = int(args[3].F32)
			}
		}
		thresh := 0
		if len(args) >= 5 {
			switch args[4].Kind {
			case VI32:
				thresh = args[4].I32
			case VF32:
				thresh = int(args[4].F32)
			}
		}
		soft := 0
		if len(args) >= 6 {
			switch args[5].Kind {
			case VI32:
				soft = args[5].I32
			case VF32:
				soft = int(args[5].F32)
			}
		}
		cal := 0
		if len(args) >= 7 {
			switch args[6].Kind {
			case VI32:
				cal = args[6].I32
			case VF32:
				cal = int(args[6].F32)
			}
		}
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < thresh || soft < 0 || cal < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_cal_key(tag, a, b): pack with calibrated soft-gated multi-key surface (v0.64).
	nat("nested_tag_cal_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_cal_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_cal_reduce(p, q, k0, k1): calibrated soft-gated multi-key forest zip-reduce (v0.64).
	nat("nested_tag_cal_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with live VL residual streaming schedule + peel fusion (v0.64); portable = 48
	nat("hw_sme_za_gather_mt_vlfuse", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vlfuse", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vlfuse(a, idx): peel-fused residual-schedule gather = sum a[i] (v0.64).
	nat("sme_za_gather_mt_vlfuse", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vlfuse(a, idx, v): n * v (portable) (v0.64).
	nat("sme_za_scatter_mt_vlfuse", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_softcap(n, isa, hits, live, xk, pass, win, decay, half, budget, softcap): dual EMA hysteresis budget reweight with soft caps (v0.64).
	// = n * max(1, min(cap, w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5))
	// with half=0, budget=0, softcap=0 → same probe path (win=4 → 20)
	nat("fusion_pgo_softcap", func(args []Value) (Value, error) {
		n, isa, hits, live, xk, pass, win, decay, half, budget, softcap := 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half, budget, softcap = get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5
		// hard budget cap
		if budget > 0 && w > budget {
			w = budget
		}
		// soft cap: when softcap > 0 and w > softcap, gently pull toward softcap by averaging
		if softcap > 0 && w > softcap {
			w = (w + softcap) / 2
			if w > softcap {
				// if still over, hard clamp to softcap+1 slack then clamp to softcap
				if w > softcap {
					w = softcap
				}
			}
		}
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_softcap_counter(hits, live, xk, pass, win, decay, half) (v0.64).
	nat("fusion_pgo_softcap_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6)), nil
	})

	// nested_tag_adapt_filter(p, key, rank, weight, thresh, soft, cal, adapt): adaptive-gated calibrated filter (v0.65).
	// Passes when tag==key, rank>=0, weight>=thresh, soft>=0, cal>=0, adapt>=0.
	nat("nested_tag_adapt_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		key, rank, weight, thresh, soft, cal, adapt := get(1), get(2), get(3), get(4), get(5), get(6), get(7)
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < thresh || soft < 0 || cal < 0 || adapt < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_adapt_key(tag, a, b): pack with adaptive multi-key surface (v0.65).
	nat("nested_tag_adapt_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_adapt_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_adapt_reduce(p, q, k0, k1): adaptive multi-key forest zip-reduce (v0.65).
	nat("nested_tag_adapt_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with residual streaming schedule peel fusion + live tile mix (v0.65); portable = 48
	nat("hw_sme_za_gather_mt_vlmix", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vlmix", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vlmix(a, idx): live tile-mix residual-schedule gather = sum a[i] (v0.65).
	nat("sme_za_gather_mt_vlmix", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vlmix(a, idx, v): n * v (portable) (v0.65).
	nat("sme_za_scatter_mt_vlmix", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_dual(n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard): dual soft/hard budget lanes (v0.65).
	// base path with zeros + win=4 → 20
	nat("fusion_pgo_dual", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard := get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5
		// hard budget cap (lane)
		if budget > 0 && w > budget {
			w = budget
		}
		// soft cap: average pull toward softcap
		if softcap > 0 && w > softcap {
			w = (w + softcap) / 2
			if w > softcap {
				w = softcap
			}
		}
		// dual hard lane: additional hard clamp
		if hard > 0 && w > hard {
			w = hard
		}
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_dual_counter(hits, live, xk, pass, win, decay, half) (v0.65).
	nat("fusion_pgo_dual_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6)), nil
	})

	// nested_tag_blend_filter(p, key, rank, weight, thresh, soft, cal, adapt, blend): blended adaptive filter (v0.66).
	// Passes when tag==key, rank>=0, weight>=thresh, soft>=0, cal>=0, adapt>=0, blend>=0.
	nat("nested_tag_blend_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		key, rank, weight, thresh, soft, cal, adapt, blend := get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8)
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < thresh || soft < 0 || cal < 0 || adapt < 0 || blend < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_blend_key(tag, a, b): pack with blended multi-key surface (v0.66).
	nat("nested_tag_blend_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_blend_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_blend_reduce(p, q, k0, k1): blended multi-key forest zip-reduce (v0.66).
	nat("nested_tag_blend_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix + cost balance (v0.66); portable = 48
	nat("hw_sme_za_gather_mt_vlbal", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vlbal", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vlbal(a, idx): cost-balanced live tile-mix residual-schedule gather = sum a[i] (v0.66).
	nat("sme_za_gather_mt_vlbal", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vlbal(a, idx, v): n * v (portable) (v0.66).
	nat("sme_za_scatter_mt_vlbal", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_conf(n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf): dual-lane soft/hard + confidence blend (v0.66).
	// base path with zeros + win=4 → 20
	nat("fusion_pgo_conf", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf := get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5 + conf/10
		// hard budget cap (lane)
		if budget > 0 && w > budget {
			w = budget
		}
		// soft cap: average pull toward softcap
		if softcap > 0 && w > softcap {
			w = (w + softcap) / 2
			if w > softcap {
				w = softcap
			}
		}
		// dual hard lane: additional hard clamp
		if hard > 0 && w > hard {
			w = hard
		}
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_conf_counter(hits, live, xk, pass, win, decay, half, conf) (v0.66).
	nat("fusion_pgo_conf_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6) + get(7)), nil
	})

	// nested_tag_gate_filter(p, key, rank, weight, thresh, soft, cal, adapt, blend, conf): gated confidence filter (v0.67).
	// Passes when tag==key, rank>=0, weight>=thresh, soft>=0, cal>=0, adapt>=0, blend>=0, conf>=0.
	nat("nested_tag_gate_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		key, rank, weight, thresh, soft, cal, adapt, blend, conf := get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9)
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < thresh || soft < 0 || cal < 0 || adapt < 0 || blend < 0 || conf < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_gate_key(tag, a, b): pack with gated multi-key surface (v0.67).
	nat("nested_tag_gate_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_gate_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_gate_reduce(p, q, k0, k1): gated multi-key forest zip-reduce (v0.67).
	nat("nested_tag_gate_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance + bandwidth model (v0.67); portable = 48
	nat("hw_sme_za_gather_mt_vlbw", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vlbw", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vlbw(a, idx): bandwidth-modeled live tile-mix residual-schedule gather = sum a[i] (v0.67).
	nat("sme_za_gather_mt_vlbw", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vlbw(a, idx, v): n * v (portable) (v0.67).
	nat("sme_za_scatter_mt_vlbw", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_hyst(n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst): dual-lane conf blend + hysteresis (v0.67).
	// base path with zeros + win=4 → 20
	nat("fusion_pgo_reg", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst := get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5 + conf/10
		// hard budget cap (lane)
		if budget > 0 && w > budget {
			w = budget
		}
		// soft cap: average pull toward softcap
		if softcap > 0 && w > softcap {
			w = (w + softcap) / 2
			if w > softcap {
				w = softcap
			}
		}
		// dual hard lane: additional hard clamp
		if hard > 0 && w > hard {
			w = hard
		}
		// hysteresis band: pull toward previous if within hyst of softcap
		if hyst > 0 && softcap > 0 {
			diff := w - softcap
			if diff < 0 {
				diff = -diff
			}
			if diff <= hyst {
				w = softcap
			}
		}
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_reg_counter(hits, live, xk, pass, win, decay, half, conf, hyst) (v0.67).
	nat("fusion_pgo_reg_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6) + get(7) + get(8)), nil
	})

	// nested_tag_score_filter(p, key, rank, weight, thresh, soft, cal, adapt, blend, conf, score):
	// score-calibrated gated confidence filter (v0.68).
	// Passes when tag==key, rank>=0, weight>=thresh, soft>=0, cal>=0, adapt>=0, blend>=0, conf>=0, score>=0.
	nat("nested_tag_score_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		key, rank, weight, thresh, soft, cal, adapt, blend, conf, score := get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10)
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < thresh || soft < 0 || cal < 0 || adapt < 0 || blend < 0 || conf < 0 || score < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_score_key(tag, a, b): pack with score-calibrated multi-key surface (v0.68).
	nat("nested_tag_score_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_score_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_score_reduce(p, q, k0, k1): score-calibrated multi-key forest zip-reduce (v0.68).
	nat("nested_tag_score_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth + path cost (v0.68); portable = 48
	nat("hw_sme_za_gather_mt_vlpath", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vlpath", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vlpath(a, idx): path-cost bandwidth live tile-mix residual-schedule gather = sum a[i] (v0.68).
	nat("sme_za_gather_mt_vlpath", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vlpath(a, idx, v): n * v (portable) (v0.68).
	nat("sme_za_scatter_mt_vlpath", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_floor(n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor):
	// dual-lane conf blend + hysteresis + regret floor (v0.68). base path with zeros + win=4 → 20
	nat("fusion_pgo_floor", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor := get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5 + conf/10
		// hard budget cap (lane)
		if budget > 0 && w > budget {
			w = budget
		}
		// soft cap: average pull toward softcap
		if softcap > 0 && w > softcap {
			w = (w + softcap) / 2
			if w > softcap {
				w = softcap
			}
		}
		// dual hard lane: additional hard clamp
		if hard > 0 && w > hard {
			w = hard
		}
		// hysteresis band: pull toward previous if within hyst of softcap
		if hyst > 0 && softcap > 0 {
			diff := w - softcap
			if diff < 0 {
				diff = -diff
			}
			if diff <= hyst {
				w = softcap
			}
		}
		// regret floor: never drop below floor when set
		if floor > 0 && w < floor {
			w = floor
		}
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_floor_counter(hits, live, xk, pass, win, decay, half, conf, hyst, floor) (v0.68).
	nat("fusion_pgo_floor_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6) + get(7) + get(8) + get(9)), nil
	})


	// nested_tag_softfloor_filter(p, key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor):
	// score-calibrated soft confidence floor filter (v0.69).
	// Passes when tag==key, rank>=0, weight>=thresh, soft/cal/adapt/blend/conf/score/softfloor all >=0.
	nat("nested_tag_softfloor_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor := get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11)
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < thresh || soft < 0 || cal < 0 || adapt < 0 || blend < 0 || conf < 0 || score < 0 || softfloor < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_softfloor_key(tag, a, b): pack with soft confidence floor multi-key surface (v0.69).
	nat("nested_tag_softfloor_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_softfloor_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_softfloor_reduce(p, q, k0, k1): soft confidence floor multi-key forest zip-reduce (v0.69).
	nat("nested_tag_softfloor_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost + lane pressure (v0.69); portable = 48
	nat("hw_sme_za_gather_mt_vllane", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vllane", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vllane(a, idx): lane-pressure path-cost gather = sum a[i] (v0.69).
	nat("sme_za_gather_mt_vllane", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vllane(a, idx, v): n * v (portable) (v0.69).
	nat("sme_za_scatter_mt_vllane", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_afloor(n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt):
	// dual-lane conf blend + hysteresis + regret floor + adaptive floor (v0.69). base zeros + win=4 → 20
	nat("fusion_pgo_afloor", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt := get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5 + conf/10
		if budget > 0 && w > budget {
			w = budget
		}
		if softcap > 0 && w > softcap {
			w = (w + softcap) / 2
			if w > softcap {
				w = softcap
			}
		}
		if hard > 0 && w > hard {
			w = hard
		}
		if hyst > 0 && softcap > 0 {
			diff := w - softcap
			if diff < 0 {
				diff = -diff
			}
			if diff <= hyst {
				w = softcap
			}
		}
		// regret floor
		if floor > 0 && w < floor {
			w = floor
		}
		// adaptive floor: raise min to adapt when set
		if adapt > 0 && w < adapt {
			w = adapt
		}
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_afloor_counter(hits, live, xk, pass, win, decay, half, conf, hyst, floor, adapt) (v0.69).
	nat("fusion_pgo_afloor_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6) + get(7) + get(8) + get(9) + get(10)), nil
	})


	// nested_tag_hardgate_filter(p, key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard):
	// soft confidence floor + hard floor gates filter (v0.70).
	// Passes when tag==key, rank>=0, weight>=thresh, soft/cal/adapt/blend/conf/score/softfloor/hard all >=0.
	nat("nested_tag_hardgate_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard := get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12)
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < thresh || soft < 0 || cal < 0 || adapt < 0 || blend < 0 || conf < 0 || score < 0 || softfloor < 0 || hard < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_hardgate_key(tag, a, b): pack with hard floor gate multi-key surface (v0.70).
	nat("nested_tag_hardgate_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_hardgate_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_hardgate_reduce(p, q, k0, k1): hard floor gate multi-key forest zip-reduce (v0.70).
	nat("nested_tag_hardgate_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure + spill model (v0.70); portable = 48
	nat("hw_sme_za_gather_mt_vlspill", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vlspill", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vlspill(a, idx): spill-model path-cost gather = sum a[i] (v0.70).
	nat("sme_za_gather_mt_vlspill", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vlspill(a, idx, v): n * v (portable) (v0.70).
	nat("sme_za_scatter_mt_vlspill", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_burst(n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst):
	// dual-lane conf blend + hysteresis + regret floor + adaptive floor + burst cap (v0.70). base zeros + win=4 → 20
	nat("fusion_pgo_burst", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst := get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15), get(16)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5 + conf/10
		if budget > 0 && w > budget {
			w = budget
		}
		if softcap > 0 && w > softcap {
			w = (w + softcap) / 2
			if w > softcap {
				w = softcap
			}
		}
		if hard > 0 && w > hard {
			w = hard
		}
		if hyst > 0 && softcap > 0 {
			diff := w - softcap
			if diff < 0 {
				diff = -diff
			}
			if diff <= hyst {
				w = softcap
			}
		}
		// regret floor
		if floor > 0 && w < floor {
			w = floor
		}
		// adaptive floor
		if adapt > 0 && w < adapt {
			w = adapt
		}
		// burst cap: raise upper limit when burst allows short spike then clamp
		if burst > 0 && w > burst {
			w = burst
		}
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_burst_counter(hits, live, xk, pass, win, decay, half, conf, hyst, floor, adapt, burst) (v0.70).
	nat("fusion_pgo_burst_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6) + get(7) + get(8) + get(9) + get(10) + get(11)), nil
	})


	// nested_tag_dual_filter(p, key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual):
	// hard floor gate + dual gate blend filter (v0.71).
	// Passes when tag==key, rank>=0, weight>=thresh, soft/cal/adapt/blend/conf/score/softfloor/hard/dual all >=0.
	nat("nested_tag_dual_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual := get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13)
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < thresh || soft < 0 || cal < 0 || adapt < 0 || blend < 0 || conf < 0 || score < 0 || softfloor < 0 || hard < 0 || dual < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_dual_key(tag, a, b): pack with dual gate blend multi-key surface (v0.71).
	nat("nested_tag_dual_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_dual_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_dual_reduce(p, q, k0, k1): dual gate blend multi-key forest zip-reduce (v0.71).
	nat("nested_tag_dual_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model + occupancy (v0.71); portable = 48
	nat("hw_sme_za_gather_mt_vlocc", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vlocc", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vlocc(a, idx): occupancy path-cost gather = sum a[i] (v0.71).
	nat("sme_za_gather_mt_vlocc", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vlocc(a, idx, v): n * v (portable) (v0.71).
	nat("sme_za_scatter_mt_vlocc", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_cool(n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool):
	// dual-lane conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down (v0.71). base zeros + win=4 → 20
	nat("fusion_pgo_cool", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool := get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15), get(16), get(17)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5 + conf/10
		if budget > 0 && w > budget {
			w = budget
		}
		if softcap > 0 && w > softcap {
			w = (w + softcap) / 2
			if w > softcap {
				w = softcap
			}
		}
		if hard > 0 && w > hard {
			w = hard
		}
		if hyst > 0 && softcap > 0 {
			diff := w - softcap
			if diff < 0 {
				diff = -diff
			}
			if diff <= hyst {
				w = softcap
			}
		}
		// regret floor
		if floor > 0 && w < floor {
			w = floor
		}
		// adaptive floor
		if adapt > 0 && w < adapt {
			w = adapt
		}
		// burst cap: raise upper limit when burst allows short spike then clamp
		if burst > 0 && w > burst {
			w = burst
		}
		// cool-down: reduce weight after burst spike
		if cool > 0 {
			w = w - cool
		}
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_cool_counter(hits, live, xk, pass, win, decay, half, conf, hyst, floor, adapt, burst, cool) (v0.71).
	nat("fusion_pgo_cool_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6) + get(7) + get(8) + get(9) + get(10) + get(11) + get(12)), nil
	})


	// nested_tag_mix_filter(p, key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual, mix):
	// dual gate blend + soft/hard dual score mix filter (v0.72).
	// Passes when tag==key, rank>=0, weight>=thresh, soft/cal/adapt/blend/conf/score/softfloor/hard/dual/mix all >=0.
	nat("nested_tag_mix_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual, mix := get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14)
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < thresh || soft < 0 || cal < 0 || adapt < 0 || blend < 0 || conf < 0 || score < 0 || softfloor < 0 || hard < 0 || dual < 0 || mix < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_mix_key(tag, a, b): pack with soft/hard dual score mix multi-key surface (v0.72).
	nat("nested_tag_mix_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_mix_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_mix_reduce(p, q, k0, k1): soft/hard dual score mix multi-key forest zip-reduce (v0.72).
	nat("nested_tag_mix_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth (v0.72); portable = 48
	nat("hw_sme_za_gather_mt_vlqueue", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vlqueue", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vlqueue(a, idx): queue-depth path-cost gather = sum a[i] (v0.72).
	nat("sme_za_gather_mt_vlqueue", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vlqueue(a, idx, v): n * v (portable) (v0.72).
	nat("sme_za_scatter_mt_vlqueue", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_credit(n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool, credit):
	// dual-lane conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank (v0.72). base zeros + win=4 → 20
	nat("fusion_pgo_credit", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool, credit := get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15), get(16), get(17), get(18)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5 + conf/10
		if budget > 0 && w > budget {
			w = budget
		}
		if softcap > 0 && w > softcap {
			w = (w + softcap) / 2
			if w > softcap {
				w = softcap
			}
		}
		if hard > 0 && w > hard {
			w = hard
		}
		if hyst > 0 && softcap > 0 {
			diff := w - softcap
			if diff < 0 {
				diff = -diff
			}
			if diff <= hyst {
				w = softcap
			}
		}
		// regret floor
		if floor > 0 && w < floor {
			w = floor
		}
		// adaptive floor
		if adapt > 0 && w < adapt {
			w = adapt
		}
		// burst cap
		if burst > 0 && w > burst {
			w = burst
		}
		// cool-down
		if cool > 0 {
			w = w - cool
		}
		// credit bank: restore weight from accumulated credit
		if credit > 0 {
			w = w + credit
		}
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_credit_counter(hits, live, xk, pass, win, decay, half, conf, hyst, floor, adapt, burst, cool, credit) (v0.72).
	nat("fusion_pgo_credit_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6) + get(7) + get(8) + get(9) + get(10) + get(11) + get(12) + get(13)), nil
	})


	// nested_tag_credit_filter(p, key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual, mix, credit):
	// soft/hard dual score mix + ranked credit gate filter (v0.73).
	// Passes when tag==key, rank>=0, weight>=thresh, soft/cal/adapt/blend/conf/score/softfloor/hard/dual/mix/credit all >=0.
	nat("nested_tag_credit_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual, mix, credit := get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15)
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < thresh || soft < 0 || cal < 0 || adapt < 0 || blend < 0 || conf < 0 || score < 0 || softfloor < 0 || hard < 0 || dual < 0 || mix < 0 || credit < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_credit_key(tag, a, b): pack with ranked credit gate multi-key surface (v0.73).
	nat("nested_tag_credit_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_credit_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_credit_reduce(p, q, k0, k1): ranked credit multi-key forest zip-reduce (v0.73).
	nat("nested_tag_credit_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth + watermark (v0.73); portable = 48
	nat("hw_sme_za_gather_mt_vlwater", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vlwater", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vlwater(a, idx): watermark path-cost gather = sum a[i] (v0.73).
	nat("sme_za_gather_mt_vlwater", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vlwater(a, idx, v): n * v (portable) (v0.73).
	nat("sme_za_scatter_mt_vlwater", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_debt(n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool, credit, debt):
	// dual-lane conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery (v0.73). base zeros + win=4 → 20
	nat("fusion_pgo_debt", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool, credit, debt := get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15), get(16), get(17), get(18), get(19)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5 + conf/10
		if budget > 0 && w > budget {
			w = budget
		}
		if softcap > 0 && w > softcap {
			w = (w + softcap) / 2
			if w > softcap {
				w = softcap
			}
		}
		if hard > 0 && w > hard {
			w = hard
		}
		if hyst > 0 && softcap > 0 {
			diff := w - softcap
			if diff < 0 {
				diff = -diff
			}
			if diff <= hyst {
				w = softcap
			}
		}
		// regret floor
		if floor > 0 && w < floor {
			w = floor
		}
		// adaptive floor
		if adapt > 0 && w < adapt {
			w = adapt
		}
		// burst cap
		if burst > 0 && w > burst {
			w = burst
		}
		// cool-down
		if cool > 0 {
			w = w - cool
		}
		// credit bank: restore weight from accumulated credit
		if credit > 0 {
			w = w + credit
		}
		// debt recovery: repay debt from weight (floored at 1 later)
		if debt > 0 {
			w = w - debt
		}
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_debt_counter(hits, live, xk, pass, win, decay, half, conf, hyst, floor, adapt, burst, cool, credit, debt) (v0.73).
	nat("fusion_pgo_debt_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6) + get(7) + get(8) + get(9) + get(10) + get(11) + get(12) + get(13) + get(14)), nil
	})


	// nested_tag_margin_filter(p, key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual, mix, credit, margin):
	// soft/hard dual score mix + ranked credit + margin gate filter (v0.74).
	// Passes when tag==key, rank>=0, weight>=thresh, soft/cal/adapt/blend/conf/score/softfloor/hard/dual/mix/credit/margin all >=0.
	nat("nested_tag_margin_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual, mix, credit, margin := get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15), get(16)
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < thresh || soft < 0 || cal < 0 || adapt < 0 || blend < 0 || conf < 0 || score < 0 || softfloor < 0 || hard < 0 || dual < 0 || mix < 0 || credit < 0 || margin < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_margin_key(tag, a, b): pack with ranked credit + margin gate multi-key surface (v0.74).
	nat("nested_tag_margin_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_margin_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_margin_reduce(p, q, k0, k1): ranked credit + margin multi-key forest zip-reduce (v0.74).
	nat("nested_tag_margin_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth + watermark + watermark drain (v0.74); portable = 48
	nat("hw_sme_za_gather_mt_vldrain", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vldrain", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vldrain(a, idx): watermark-drain path-cost gather = sum a[i] (v0.74).
	nat("sme_za_gather_mt_vldrain", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vldrain(a, idx, v): n * v (portable) (v0.74).
	nat("sme_za_scatter_mt_vldrain", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_interest(n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool, credit, debt, interest):
	// dual-lane conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest (v0.74). base zeros + win=4 → 20
	nat("fusion_pgo_interest", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool, credit, debt, interest := get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15), get(16), get(17), get(18), get(19), get(20)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5 + conf/10
		if budget > 0 && w > budget {
			w = budget
		}
		if softcap > 0 && w > softcap {
			w = (w + softcap) / 2
			if w > softcap {
				w = softcap
			}
		}
		if hard > 0 && w > hard {
			w = hard
		}
		if hyst > 0 && softcap > 0 {
			diff := w - softcap
			if diff < 0 {
				diff = -diff
			}
			if diff <= hyst {
				w = softcap
			}
		}
		// regret floor
		if floor > 0 && w < floor {
			w = floor
		}
		// adaptive floor
		if adapt > 0 && w < adapt {
			w = adapt
		}
		// burst cap
		if burst > 0 && w > burst {
			w = burst
		}
		// cool-down
		if cool > 0 {
			w = w - cool
		}
		// credit bank
		if credit > 0 {
			w = w + credit
		}
		// debt recovery
		if debt > 0 {
			w = w - debt
		}
		// interest: recover accrued interest back into weight
		if interest > 0 {
			w = w + interest
		}
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_interest_counter(hits, live, xk, pass, win, decay, half, conf, hyst, floor, adapt, burst, cool, credit, debt, interest) (v0.74).
	nat("fusion_pgo_interest_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6) + get(7) + get(8) + get(9) + get(10) + get(11) + get(12) + get(13) + get(14) + get(15)), nil
	})


	// nested_tag_band_filter(p, key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual, mix, credit, margin, band):
	// soft/hard dual score mix + ranked credit + margin + band gate filter (v0.75).
	// Passes when tag==key, rank>=0, weight>=thresh, soft/cal/adapt/blend/conf/score/softfloor/hard/dual/mix/credit/margin/band all >=0.
	nat("nested_tag_band_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual, mix, credit, margin, band := get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15), get(16), get(17)
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < thresh || soft < 0 || cal < 0 || adapt < 0 || blend < 0 || conf < 0 || score < 0 || softfloor < 0 || hard < 0 || dual < 0 || mix < 0 || credit < 0 || margin < 0 || band < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_band_key(tag, a, b): pack with ranked credit + margin + band gate multi-key surface (v0.75).
	nat("nested_tag_band_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_band_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_band_reduce(p, q, k0, k1): ranked credit + margin + band multi-key forest zip-reduce (v0.75).
	nat("nested_tag_band_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth + watermark + watermark drain + refill (v0.75); portable = 48
	nat("hw_sme_za_gather_mt_vlrefill", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vlrefill", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vlrefill(a, idx): watermark-drain-refill path-cost gather = sum a[i] (v0.75).
	nat("sme_za_gather_mt_vlrefill", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vlrefill(a, idx, v): n * v (portable) (v0.75).
	nat("sme_za_scatter_mt_vlrefill", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_amortize(n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool, credit, debt, interest, amort):
	// dual-lane conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize (v0.75). base zeros + win=4 → 20
	nat("fusion_pgo_amortize", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool, credit, debt, interest, amort := get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15), get(16), get(17), get(18), get(19), get(20), get(21)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5 + conf/10
		if budget > 0 && w > budget {
			w = budget
		}
		if softcap > 0 && w > softcap {
			w = (w + softcap) / 2
			if w > softcap {
				w = softcap
			}
		}
		if hard > 0 && w > hard {
			w = hard
		}
		if hyst > 0 && softcap > 0 {
			diff := w - softcap
			if diff < 0 {
				diff = -diff
			}
			if diff <= hyst {
				w = softcap
			}
		}
		// regret floor
		if floor > 0 && w < floor {
			w = floor
		}
		// adaptive floor
		if adapt > 0 && w < adapt {
			w = adapt
		}
		// burst cap
		if burst > 0 && w > burst {
			w = burst
		}
		// cool-down
		if cool > 0 {
			w = w - cool
		}
		// credit bank
		if credit > 0 {
			w = w + credit
		}
		// debt recovery
		if debt > 0 {
			w = w - debt
		}
		// interest
		if interest > 0 {
			w = w + interest
		}
		// amortize: spread load by reducing weight by amortization factor
		if amort > 0 {
			w = w - amort
		}
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_amortize_counter(hits, live, xk, pass, win, decay, half, conf, hyst, floor, adapt, burst, cool, credit, debt, interest, amort) (v0.75).
	nat("fusion_pgo_amortize_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6) + get(7) + get(8) + get(9) + get(10) + get(11) + get(12) + get(13) + get(14) + get(15) + get(16)), nil
	})

	// nested_tag_clamp_filter(p, key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual, mix, credit, margin, band, clamp):
	// soft/hard dual score mix + ranked credit + margin + clamp gate filter (v0.76).
	// Passes when tag==key, rank>=0, weight>=thresh, soft/cal/adapt/blend/conf/score/softfloor/hard/dual/mix/credit/margin/band/clamp all >=0.
	nat("nested_tag_clamp_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual, mix, credit, margin, band, clamp := get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15), get(16), get(17), get(18)
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < thresh || soft < 0 || cal < 0 || adapt < 0 || blend < 0 || conf < 0 || score < 0 || softfloor < 0 || hard < 0 || dual < 0 || mix < 0 || credit < 0 || margin < 0 || band < 0 || clamp < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_clamp_key(tag, a, b): pack with ranked credit + margin + clamp gate multi-key surface (v0.76).
	nat("nested_tag_clamp_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_clamp_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_clamp_reduce(p, q, k0, k1): ranked credit + margin + band + clamp + clamp multi-key forest zip-reduce (v0.76).
	nat("nested_tag_clamp_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth + watermark + watermark drain + refill + flush + flush (v0.76); portable = 48
	nat("hw_sme_za_gather_mt_vlflush", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vlflush", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vlflush(a, idx): watermark-drain-refill path-cost gather = sum a[i] (v0.76).
	nat("sme_za_gather_mt_vlflush", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vlflush(a, idx, v): n * v (portable) (v0.76).
	nat("sme_za_scatter_mt_vlflush", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_settle(n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool, credit, debt, interest, amort, settle):
	// dual-lane conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + settle (v0.76). base zeros + win=4 → 20
	nat("fusion_pgo_settle", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool, credit, debt, interest, amort, settle := get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15), get(16), get(17), get(18), get(19), get(20), get(21), get(22)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5 + conf/10
		if budget > 0 && w > budget {
			w = budget
		}
		if softcap > 0 && w > softcap {
			w = (w + softcap) / 2
			if w > softcap {
				w = softcap
			}
		}
		if hard > 0 && w > hard {
			w = hard
		}
		if hyst > 0 && softcap > 0 {
			diff := w - softcap
			if diff < 0 {
				diff = -diff
			}
			if diff <= hyst {
				w = softcap
			}
		}
		// regret floor
		if floor > 0 && w < floor {
			w = floor
		}
		// adaptive floor
		if adapt > 0 && w < adapt {
			w = adapt
		}
		// burst cap
		if burst > 0 && w > burst {
			w = burst
		}
		// cool-down
		if cool > 0 {
			w = w - cool
		}
		// credit bank
		if credit > 0 {
			w = w + credit
		}
		// debt recovery
		if debt > 0 {
			w = w - debt
		}
		// interest
		if interest > 0 {
			w = w + interest
		}
		// amortize: spread load by reducing weight by amortization factor
		if amort > 0 {
			w = w - amort
		}
		// settle: final settlement reduction
		if settle > 0 {
			w = w - settle
		}
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_settle_counter(hits, live, xk, pass, win, decay, half, conf, hyst, floor, adapt, burst, cool, credit, debt, interest, amort, settle) (v0.76).
	nat("fusion_pgo_settle_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6) + get(7) + get(8) + get(9) + get(10) + get(11) + get(12) + get(13) + get(14) + get(15) + get(16) + get(17)), nil
	})

	// nested_tag_peg_filter(p, key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual, mix, credit, margin, band, clamp, peg):
	// soft/hard dual score mix + ranked credit + margin + peg gate filter (v0.77).
	// Passes when tag==key, rank>=0, weight>=thresh, soft/cal/adapt/blend/conf/score/softfloor/hard/dual/mix/credit/margin/band/clamp/peg all >=0.
	nat("nested_tag_peg_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual, mix, credit, margin, band, clamp, peg := get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15), get(16), get(17), get(18), get(19)
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < thresh || soft < 0 || cal < 0 || adapt < 0 || blend < 0 || conf < 0 || score < 0 || softfloor < 0 || hard < 0 || dual < 0 || mix < 0 || credit < 0 || margin < 0 || band < 0 || clamp < 0 || peg < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_peg_key(tag, a, b): pack with ranked credit + margin + peg gate multi-key surface (v0.77).
	nat("nested_tag_peg_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_peg_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_peg_reduce(p, q, k0, k1): ranked credit + margin + band + clamp + peg + peg + clamp multi-key forest zip-reduce (v0.77).
	nat("nested_tag_peg_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth + watermark + watermark drain + refill + flush + seal + flush (v0.77); portable = 48
	nat("hw_sme_za_gather_mt_vlseal", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vlseal", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vlseal(a, idx): watermark-drain-refill path-cost gather = sum a[i] (v0.77).
	nat("sme_za_gather_mt_vlseal", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vlseal(a, idx, v): n * v (portable) (v0.77).
	nat("sme_za_scatter_mt_vlseal", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_escrow(n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool, credit, debt, interest, amort, settle, escrow):
	// dual-lane conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + settle (v0.77). base zeros + win=4 → 20
	nat("fusion_pgo_escrow", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool, credit, debt, interest, amort, settle, escrow := get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15), get(16), get(17), get(18), get(19), get(20), get(21), get(22), get(23)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5 + conf/10
		if budget > 0 && w > budget {
			w = budget
		}
		if softcap > 0 && w > softcap {
			w = (w + softcap) / 2
			if w > softcap {
				w = softcap
			}
		}
		if hard > 0 && w > hard {
			w = hard
		}
		if hyst > 0 && softcap > 0 {
			diff := w - softcap
			if diff < 0 {
				diff = -diff
			}
			if diff <= hyst {
				w = softcap
			}
		}
		// regret floor
		if floor > 0 && w < floor {
			w = floor
		}
		// adaptive floor
		if adapt > 0 && w < adapt {
			w = adapt
		}
		// burst cap
		if burst > 0 && w > burst {
			w = burst
		}
		// cool-down
		if cool > 0 {
			w = w - cool
		}
		// credit bank
		if credit > 0 {
			w = w + credit
		}
		// debt recovery
		if debt > 0 {
			w = w - debt
		}
		// interest
		if interest > 0 {
			w = w + interest
		}
		// amortize: spread load by reducing weight by amortization factor
		if amort > 0 {
			w = w - amort
		}
		// settle: final settlement reduction
		if settle > 0 {
			w = w - settle
		}
		// escrow: hold capacity by increasing weight
		if escrow > 0 {
			w = w + escrow
		}
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_escrow_counter(hits, live, xk, pass, win, decay, half, conf, hyst, floor, adapt, burst, cool, credit, debt, interest, amort, settle, escrow) (v0.77).
	nat("fusion_pgo_escrow_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6) + get(7) + get(8) + get(9) + get(10) + get(11) + get(12) + get(13) + get(14) + get(15) + get(16) + get(17) + get(18)), nil
	})

	// nested_tag_latch_filter(p, key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual, mix, credit, margin, band, clamp, peg, latch):
	// soft/hard dual score mix + ranked credit + margin + latch gate filter (v0.78).
	// Passes when tag==key, rank>=0, weight>=thresh, soft/cal/adapt/blend/conf/score/softfloor/hard/dual/mix/credit/margin/band/clamp/peg/latch all >=0.
	nat("nested_tag_latch_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual, mix, credit, margin, band, clamp, peg, latch := get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15), get(16), get(17), get(18), get(19), get(20)
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < thresh || soft < 0 || cal < 0 || adapt < 0 || blend < 0 || conf < 0 || score < 0 || softfloor < 0 || hard < 0 || dual < 0 || mix < 0 || credit < 0 || margin < 0 || band < 0 || clamp < 0 || peg < 0 || latch < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_latch_key(tag, a, b): pack with ranked credit + margin + latch gate multi-key surface (v0.78).
	nat("nested_tag_latch_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_latch_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_latch_reduce(p, q, k0, k1): ranked credit + margin + band + clamp + peg + latch + latch + peg + clamp multi-key forest zip-reduce (v0.78).
	nat("nested_tag_latch_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth + watermark + watermark drain + refill + flush + seal + purge + flush (v0.78); portable = 48
	nat("hw_sme_za_gather_mt_vlpurge", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vlpurge", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vlpurge(a, idx): watermark-drain-refill path-cost gather = sum a[i] (v0.78).
	nat("sme_za_gather_mt_vlpurge", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vlpurge(a, idx, v): n * v (portable) (v0.78).
	nat("sme_za_scatter_mt_vlpurge", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_clear(n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool, credit, debt, interest, amort, settle, escrow, clear):
	// dual-lane conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + clear + settle (v0.78). base zeros + win=4 → 20
	nat("fusion_pgo_clear", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool, credit, debt, interest, amort, settle, escrow, clear := get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15), get(16), get(17), get(18), get(19), get(20), get(21), get(22), get(23), get(24)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5 + conf/10
		if budget > 0 && w > budget {
			w = budget
		}
		if softcap > 0 && w > softcap {
			w = (w + softcap) / 2
			if w > softcap {
				w = softcap
			}
		}
		if hard > 0 && w > hard {
			w = hard
		}
		if hyst > 0 && softcap > 0 {
			diff := w - softcap
			if diff < 0 {
				diff = -diff
			}
			if diff <= hyst {
				w = softcap
			}
		}
		// regret floor
		if floor > 0 && w < floor {
			w = floor
		}
		// adaptive floor
		if adapt > 0 && w < adapt {
			w = adapt
		}
		// burst cap
		if burst > 0 && w > burst {
			w = burst
		}
		// cool-down
		if cool > 0 {
			w = w - cool
		}
		// credit bank
		if credit > 0 {
			w = w + credit
		}
		// debt recovery
		if debt > 0 {
			w = w - debt
		}
		// interest
		if interest > 0 {
			w = w + interest
		}
		// amortize: spread load by reducing weight by amortization factor
		if amort > 0 {
			w = w - amort
		}
		// settle: final settlement reduction
		if settle > 0 {
			w = w - settle
		}
		// escrow: hold capacity by increasing weight
		if escrow > 0 {
			w = w + escrow
		}
		// clear: drain residual weight
		if clear > 0 {
			w = w - clear
		}
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_clear_counter(hits, live, xk, pass, win, decay, half, conf, hyst, floor, adapt, burst, cool, credit, debt, interest, amort, settle, escrow, clear) (v0.78).
	nat("fusion_pgo_clear_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6) + get(7) + get(8) + get(9) + get(10) + get(11) + get(12) + get(13) + get(14) + get(15) + get(16) + get(17) + get(18) + get(19)), nil
	})

	// nested_tag_fence_filter(p, key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual, mix, credit, margin, band, clamp, peg, latch, fence):
	// soft/hard dual score mix + ranked credit + margin + fence gate filter (v0.79).
	// Passes when tag==key, rank>=0, weight>=thresh, soft/cal/adapt/blend/conf/score/softfloor/hard/dual/mix/credit/margin/band/clamp/peg/latch/fence all >=0.
	nat("nested_tag_fence_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual, mix, credit, margin, band, clamp, peg, latch, fence := get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15), get(16), get(17), get(18), get(19), get(20), get(21)
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < thresh || soft < 0 || cal < 0 || adapt < 0 || blend < 0 || conf < 0 || score < 0 || softfloor < 0 || hard < 0 || dual < 0 || mix < 0 || credit < 0 || margin < 0 || band < 0 || clamp < 0 || peg < 0 || latch < 0 || fence < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_fence_key(tag, a, b): pack with ranked credit + margin + fence gate multi-key surface (v0.79).
	nat("nested_tag_fence_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_fence_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_fence_reduce(p, q, k0, k1): ranked credit + margin + band + clamp + peg + latch + fence + latch + peg + clamp multi-key forest zip-reduce (v0.79).
	nat("nested_tag_fence_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth + watermark + watermark drain + refill + flush + seal + purge + sync + flush (v0.79); portable = 48
	nat("hw_sme_za_gather_mt_vlsync", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vlsync", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vlsync(a, idx): watermark-drain-refill path-cost gather = sum a[i] (v0.79).
	nat("sme_za_gather_mt_vlsync", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vlsync(a, idx, v): n * v (portable) (v0.79).
	nat("sme_za_scatter_mt_vlsync", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_final(n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool, credit, debt, interest, amort, settle, escrow, clear, final):
	// dual-lane conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + clear + final + settle (v0.79). base zeros + win=4 → 20
	nat("fusion_pgo_final", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool, credit, debt, interest, amort, settle, escrow, clear, final := get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15), get(16), get(17), get(18), get(19), get(20), get(21), get(22), get(23), get(24), get(25)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5 + conf/10
		if budget > 0 && w > budget {
			w = budget
		}
		if softcap > 0 && w > softcap {
			w = (w + softcap) / 2
			if w > softcap {
				w = softcap
			}
		}
		if hard > 0 && w > hard {
			w = hard
		}
		if hyst > 0 && softcap > 0 {
			diff := w - softcap
			if diff < 0 {
				diff = -diff
			}
			if diff <= hyst {
				w = softcap
			}
		}
		// regret floor
		if floor > 0 && w < floor {
			w = floor
		}
		// adaptive floor
		if adapt > 0 && w < adapt {
			w = adapt
		}
		// burst cap
		if burst > 0 && w > burst {
			w = burst
		}
		// cool-down
		if cool > 0 {
			w = w - cool
		}
		// credit bank
		if credit > 0 {
			w = w + credit
		}
		// debt recovery
		if debt > 0 {
			w = w - debt
		}
		// interest
		if interest > 0 {
			w = w + interest
		}
		// amortize: spread load by reducing weight by amortization factor
		if amort > 0 {
			w = w - amort
		}
		// settle: final settlement reduction
		if settle > 0 {
			w = w - settle
		}
		// escrow: hold capacity by increasing weight
		if escrow > 0 {
			w = w + escrow
		}
		// clear: drain residual weight
		if clear > 0 {
			w = w - clear
		}
		// final: terminal boost
		if final > 0 {
			w = w + final
		}
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_final_counter(hits, live, xk, pass, win, decay, half, conf, hyst, floor, adapt, burst, cool, credit, debt, interest, amort, settle, escrow, clear, final) (v0.79).
	nat("fusion_pgo_final_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6) + get(7) + get(8) + get(9) + get(10) + get(11) + get(12) + get(13) + get(14) + get(15) + get(16) + get(17) + get(18) + get(19) + get(20)), nil
	})

	// nested_tag_lock_filter(p, key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual, mix, credit, margin, band, clamp, peg, latch, fence, lock):
	// soft/hard dual score mix + ranked credit + margin + lock gate filter (v0.80).
	// Passes when tag==key, rank>=0, weight>=thresh, soft/cal/adapt/blend/conf/score/softfloor/hard/dual/mix/credit/margin/band/clamp/peg/latch/fence/lock all >=0.
	nat("nested_tag_lock_filter", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Span(nil), nil
		}
		leaves := flattenPairLeaves(args[0])
		if len(leaves) < 1 {
			return Span(nil), nil
		}
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		key, rank, weight, thresh, soft, cal, adapt, blend, conf, score, softfloor, hard, dual, mix, credit, margin, band, clamp, peg, latch, fence, lock := get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15), get(16), get(17), get(18), get(19), get(20), get(21), get(22)
		tag := 0
		switch leaves[0].Kind {
		case VI32:
			tag = leaves[0].I32
		case VF32:
			tag = int(leaves[0].F32)
		}
		if tag != key || rank < 0 || weight < thresh || soft < 0 || cal < 0 || adapt < 0 || blend < 0 || conf < 0 || score < 0 || softfloor < 0 || hard < 0 || dual < 0 || mix < 0 || credit < 0 || margin < 0 || band < 0 || clamp < 0 || peg < 0 || latch < 0 || fence < 0 || lock < 0 {
			return Span(nil), nil
		}
		return Span(leaves), nil
	})
	// nested_tag_lock_key(tag, a, b): pack with ranked credit + margin + lock gate multi-key surface (v0.80).
	nat("nested_tag_lock_key", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Span(nil), &Error{Msg: "nested_tag_lock_key expects (tag, a, b)"}
		}
		return Span([]Value{args[0], args[1], args[2]}), nil
	})
	// nested_tag_lock_reduce(p, q, k0, k1): ranked credit + margin + band + clamp + peg + latch + fence + lock + latch + peg + clamp multi-key forest zip-reduce (v0.80).
	nat("nested_tag_lock_reduce", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return F32(0), nil
		}
		a := flattenPairLeaves(args[0])
		b := flattenPairLeaves(args[1])
		k0, k1 := 0, 0
		switch args[2].Kind {
		case VI32:
			k0 = args[2].I32
		case VF32:
			k0 = int(args[2].F32)
		}
		switch args[3].Kind {
		case VI32:
			k1 = args[3].I32
		case VF32:
			k1 = int(args[3].F32)
		}
		if len(a) < 1 || len(b) < 1 {
			return F32(0), nil
		}
		ta, tb := 0, 0
		switch a[0].Kind {
		case VI32:
			ta = a[0].I32
		case VF32:
			ta = int(a[0].F32)
		}
		switch b[0].Kind {
		case VI32:
			tb = b[0].I32
		case VF32:
			tb = int(b[0].F32)
		}
		okA := ta == k0 || ta == k1
		okB := tb == k0 || tb == k1
		if !okA || !okB {
			return F32(0), nil
		}
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		s := 0.0
		for i := 1; i < n; i++ {
			av, bv := 0.0, 0.0
			switch a[i].Kind {
			case VF32:
				av = a[i].F32
			case VI32:
				av = float64(a[i].I32)
			}
			switch b[i].Kind {
			case VF32:
				bv = b[i].F32
			case VI32:
				bv = float64(b[i].I32)
			}
			s += av + bv
		}
		return F32(s), nil
	})
	// SME ZA multi-tile peels with residual streaming schedule peel fusion live tile mix cost balance bandwidth path cost lane pressure spill model occupancy + queue depth + watermark + watermark drain + refill + flush + seal + purge + sync + close + flush (v0.80); portable = 48
	nat("hw_sme_za_gather_mt_vlclose", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	nat("hw_sme_za_scatter_mt_vlclose", func(args []Value) (Value, error) {
		return I32(48), nil
	})
	// sme_za_gather_mt_vlclose(a, idx): watermark-drain-refill path-cost gather = sum a[i] (v0.80).
	nat("sme_za_gather_mt_vlclose", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return F32(0), nil
		}
		s := 0.0
		for _, ix := range args[1].Span {
			i := 0
			switch ix.Kind {
			case VI32:
				i = ix.I32
			case VF32:
				i = int(ix.F32)
			}
			if i >= 0 && i < len(args[0].Span) {
				e := args[0].Span[i]
				switch e.Kind {
				case VF32:
					s += e.F32
				case VI32:
					s += float64(e.I32)
				}
			}
		}
		return F32(s), nil
	})
	// sme_za_scatter_mt_vlclose(a, idx, v): n * v (portable) (v0.80).
	nat("sme_za_scatter_mt_vlclose", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return F32(0), nil
		}
		n := 0
		if args[1].Kind == VSpan {
			n = len(args[1].Span)
		}
		v := 0.0
		switch args[2].Kind {
		case VF32:
			v = args[2].F32
		case VI32:
			v = float64(args[2].I32)
		}
		return F32(float64(n) * v), nil
	})
	// fusion_pgo_close(n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool, credit, debt, interest, amort, settle, escrow, clear, final, close):
	// dual-lane conf blend + hysteresis + regret floor + adaptive floor + burst cap + cool-down + credit bank + debt recovery + interest + amortize + settle + escrow + clear + final + close + settle (v0.80). base zeros + win=4 → 20
	nat("fusion_pgo_close", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		n, isa, hits, live, xk, pass, win, decay, half, budget, softcap, hard, conf, hyst, floor, adapt, burst, cool, credit, debt, interest, amort, settle, escrow, clear, final, close := get(0), get(1), get(2), get(3), get(4), get(5), get(6), get(7), get(8), get(9), get(10), get(11), get(12), get(13), get(14), get(15), get(16), get(17), get(18), get(19), get(20), get(21), get(22), get(23), get(24), get(25), get(26)
		w := 1
		switch isa {
		case 1:
			w = 2
		case 2:
			w = 3
		case 3, 4:
			w = 4
		case 5:
			w = 2
		}
		w = w + hits/10 + live/20 + xk/5 + pass/2 + win/4 - decay + half/5 + conf/10
		if budget > 0 && w > budget {
			w = budget
		}
		if softcap > 0 && w > softcap {
			w = (w + softcap) / 2
			if w > softcap {
				w = softcap
			}
		}
		if hard > 0 && w > hard {
			w = hard
		}
		if hyst > 0 && softcap > 0 {
			diff := w - softcap
			if diff < 0 {
				diff = -diff
			}
			if diff <= hyst {
				w = softcap
			}
		}
		// regret floor
		if floor > 0 && w < floor {
			w = floor
		}
		// adaptive floor
		if adapt > 0 && w < adapt {
			w = adapt
		}
		// burst cap
		if burst > 0 && w > burst {
			w = burst
		}
		// cool-down
		if cool > 0 {
			w = w - cool
		}
		// credit bank
		if credit > 0 {
			w = w + credit
		}
		// debt recovery
		if debt > 0 {
			w = w - debt
		}
		// interest
		if interest > 0 {
			w = w + interest
		}
		// amortize: spread load by reducing weight by amortization factor
		if amort > 0 {
			w = w - amort
		}
		// settle: final settlement reduction
		if settle > 0 {
			w = w - settle
		}
		// escrow: hold capacity by increasing weight
		if escrow > 0 {
			w = w + escrow
		}
		// clear: drain residual weight
		if clear > 0 {
			w = w - clear
		}
		// final: terminal boost
		if final > 0 {
			w = w + final
		}
		// close: terminal drain
		if close > 0 {
			w = w - close
		}
		if w < 1 {
			w = 1
		}
		return I32(n * w), nil
	})
	// fusion_pgo_close_counter(hits, live, xk, pass, win, decay, half, conf, hyst, floor, adapt, burst, cool, credit, debt, interest, amort, settle, escrow, clear, final, close) (v0.80).
	nat("fusion_pgo_close_counter", func(args []Value) (Value, error) {
		get := func(i int) int {
			if len(args) <= i {
				return 0
			}
			switch args[i].Kind {
			case VI32:
				return args[i].I32
			case VF32:
				return int(args[i].F32)
			}
			return 0
		}
		return I32(get(0) + get(1) + get(2) + get(3) + get(4) + get(5) + get(6) + get(7) + get(8) + get(9) + get(10) + get(11) + get(12) + get(13) + get(14) + get(15) + get(16) + get(17) + get(18) + get(19) + get(20) + get(21)), nil
	})

	nat("pair_first", func(args []Value) (Value, error) {
		if len(args) < 1 || args[0].Kind != VSpan || len(args[0].Span) < 1 {
			return Void(), &Error{Msg: "pair_first expects pair/span"}
		}
		return args[0].Span[0], nil
	})
	nat("pair_second", func(args []Value) (Value, error) {
		if len(args) < 1 || args[0].Kind != VSpan || len(args[0].Span) < 2 {
			return Void(), &Error{Msg: "pair_second expects pair/span"}
		}
		return args[0].Span[1], nil
	})
	nat("hw_vlen_for_n", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "hw_vlen_for_n expects n"}
		}
		n := int(args[0].AsFloat())
		if n >= 16 {
			return I32(16), nil
		}
		if n >= 8 {
			return I32(8), nil
		}
		if n >= 4 {
			return I32(4), nil
		}
		if n < 0 {
			return I32(0), nil
		}
		return I32(n), nil
	})
	nat("dyn_tile", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "dyn_tile expects n"}
		}
		// Alias of portable for_n among {4,8,16}.
		n := int(args[0].AsFloat())
		if n >= 16 {
			return I32(16), nil
		}
		if n >= 8 {
			return I32(8), nil
		}
		if n >= 4 {
			return I32(4), nil
		}
		if n < 0 {
			return I32(0), nil
		}
		return I32(n), nil
	})
	// soft_mux / reverse-mode AD helpers (v0.6)
	nat("soft_mux", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Void(), &Error{Msg: "soft_mux expects (w, pure, each)"}
		}
		w := args[0].AsFloat()
		if w < 0 {
			w = 0
		}
		if w > 1 {
			w = 1
		}
		return F32(w*args[1].AsFloat() + (1-w)*args[2].AsFloat()), nil
	})
	nat("soft_mux_grad", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Void(), &Error{Msg: "soft_mux_grad expects (w, pure, each [, dy])"}
		}
		w := args[0].AsFloat()
		pure := args[1].AsFloat()
		each := args[2].AsFloat()
		dy := 1.0
		if len(args) >= 4 {
			dy = args[3].AsFloat()
		}
		if w < 0 {
			w = 0
		}
		if w > 1 {
			w = 1
		}
		return Span([]Value{
			F32(dy * (pure - each)),
			F32(dy * w),
			F32(dy * (1 - w)),
		}), nil
	})
	nat("soft_mux_grad_w", func(args []Value) (Value, error) {
		_, pure, each, dy := softMuxArgs(args)
		return F32(dy * (pure - each)), nil
	})
	nat("soft_mux_grad_pure", func(args []Value) (Value, error) {
		w, _, _, dy := softMuxArgs(args)
		return F32(dy * w), nil
	})
	nat("soft_mux_grad_each", func(args []Value) (Value, error) {
		w, _, _, dy := softMuxArgs(args)
		return F32(dy * (1 - w)), nil
	})
	// grad_expr(expr, names_span, values_span) → [value, g0, g1, ...]
	// Full reverse-mode AD over f32 +, -, *, /, soft_mux, sum, dot, span_add, scale, map_mul, map, zip_with, reduce/fold (v0.7–0.13).
	nat("grad_expr", func(args []Value) (Value, error) {
		if len(args) < 3 || args[1].Kind != VSpan || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "grad_expr expects (expr_str, names_span, values_span)"}
		}
		expr := args[0].String()
		names, vals := args[1].Span, args[2].Span
		n := min(len(names), len(vals))
		env := make(map[string]float64, n)
		order := make([]string, 0, n)
		for i := 0; i < n; i++ {
			name := names[i].String()
			env[name] = vals[i].AsFloat()
			order = append(order, name)
		}
		r, err := ad.GradExpr(expr, env)
		if err != nil {
			return Void(), &Error{Msg: "grad_expr: " + err.Error()}
		}
		out := make([]Value, 0, 1+len(order))
		out = append(out, F32(r.Value))
		for _, name := range order {
			out = append(out, F32(r.Grads[name]))
		}
		return Span(out), nil
	})
	// grad_sum(xs) → [sum, ∂/∂x0, ...]
	nat("grad_sum", func(args []Value) (Value, error) {
		if len(args) < 1 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "grad_sum expects span"}
		}
		xs := make([]float64, len(args[0].Span))
		for i, x := range args[0].Span {
			xs[i] = x.AsFloat()
		}
		s, gs := ad.GradSum(xs)
		out := make([]Value, 0, 1+len(gs))
		out = append(out, F32(s))
		for _, g := range gs {
			out = append(out, F32(g))
		}
		return Span(out), nil
	})
	// grad_dot(a, b) → [dot, ∂a..., ∂b...]
	nat("grad_dot", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "grad_dot expects two spans"}
		}
		n := min(len(args[0].Span), len(args[1].Span))
		a := make([]float64, n)
		b := make([]float64, n)
		for i := 0; i < n; i++ {
			a[i] = args[0].Span[i].AsFloat()
			b[i] = args[1].Span[i].AsFloat()
		}
		d, ga, gb := ad.GradDot(a, b)
		out := make([]Value, 0, 1+2*n)
		out = append(out, F32(d))
		for _, g := range ga {
			out = append(out, F32(g))
		}
		for _, g := range gb {
			out = append(out, F32(g))
		}
		return Span(out), nil
	})

	// grad_span_add(a,b) → [sum(a+b), ∂a..., ∂b...]
	nat("grad_span_add", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "grad_span_add expects two spans"}
		}
		n := min(len(args[0].Span), len(args[1].Span))
		a := make([]float64, n)
		b := make([]float64, n)
		for i := 0; i < n; i++ {
			a[i] = args[0].Span[i].AsFloat()
			b[i] = args[1].Span[i].AsFloat()
		}
		s, ga, gb := ad.GradSpanAdd(a, b)
		out := make([]Value, 0, 1+2*n)
		out = append(out, F32(s))
		for _, g := range ga {
			out = append(out, F32(g))
		}
		for _, g := range gb {
			out = append(out, F32(g))
		}
		return Span(out), nil
	})
	// grad_scale(s, xs) / grad_map_mul(s, xs) → [sum, ∂s, ∂x0, ...]
	nat("grad_scale", func(args []Value) (Value, error) {
		if len(args) < 2 || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "grad_scale expects (s, span)"}
		}
		xs := make([]float64, len(args[1].Span))
		for i, x := range args[1].Span {
			xs[i] = x.AsFloat()
		}
		v, ds, gxs := ad.GradScale(args[0].AsFloat(), xs)
		out := make([]Value, 0, 2+len(gxs))
		out = append(out, F32(v), F32(ds))
		for _, g := range gxs {
			out = append(out, F32(g))
		}
		return Span(out), nil
	})
	nat("grad_map_mul", func(args []Value) (Value, error) {
		if len(args) < 2 || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "grad_map_mul expects (s, span)"}
		}
		xs := make([]float64, len(args[1].Span))
		for i, x := range args[1].Span {
			xs[i] = x.AsFloat()
		}
		v, ds, gxs := ad.GradScale(args[0].AsFloat(), xs)
		out := make([]Value, 0, 2+len(gxs))
		out = append(out, F32(v), F32(ds))
		for _, g := range gxs {
			out = append(out, F32(g))
		}
		return Span(out), nil
	})

	// ---- Production ternary LLM kernel surface (v0.81+) ----
	// Quantize f32 span → trit span via absmean scale (BitNet-style).
	// Returns [trits, scale] where dequant ≈ scale * trit.
	nat("quantize_absmean", func(args []Value) (Value, error) {
		if len(args) < 1 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "quantize_absmean expects (weights span)"}
		}
		xs := args[0].Span
		n := len(xs)
		if n == 0 {
			return Span([]Value{Span(nil), F32(0)}), nil
		}
		sumAbs := 0.0
		for _, x := range xs {
			v := x.AsFloat()
			if v < 0 {
				v = -v
			}
			sumAbs += v
		}
		scale := sumAbs / float64(n)
		if scale == 0 {
			out := make([]Value, n)
			for i := range out {
				out[i] = Trit(0)
			}
			return Span([]Value{Span(out), F32(0)}), nil
		}
		out := make([]Value, n)
		for i, x := range xs {
			// round(w/scale) then clamp to {-1,0,+1}
			r := x.AsFloat() / scale
			t := 0
			if r >= 0.5 {
				t = 1
			} else if r <= -0.5 {
				t = -1
			}
			out[i] = Trit(t)
		}
		return Span([]Value{Span(out), F32(scale)}), nil
	})
	// Pack f32 weights into trit bitplanes + scale: [pos_words, neg_words, n, scale]
	nat("trit_pack_quant", func(args []Value) (Value, error) {
		if len(args) < 1 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "trit_pack_quant expects (weights span)"}
		}
		xs := args[0].Span
		n := len(xs)
		if n == 0 {
			return Span([]Value{Span(nil), Span(nil), I32(0), F32(0)}), nil
		}
		sumAbs := 0.0
		for _, x := range xs {
			v := x.AsFloat()
			if v < 0 {
				v = -v
			}
			sumAbs += v
		}
		scale := sumAbs / float64(n)
		trits := make([]Value, n)
		if scale == 0 {
			for i := range trits {
				trits[i] = Trit(0)
			}
		} else {
			for i, x := range xs {
				r := x.AsFloat() / scale
				t := 0
				if r >= 0.5 {
					t = 1
				} else if r <= -0.5 {
					t = -1
				}
				trits[i] = Trit(t)
			}
		}
		posW, negW, nn := packTritWords(trits)
		return Span([]Value{i32Span(posW), i32Span(negW), I32(nn), F32(scale)}), nil
	})
	// y = scale * (W_trit @ x)  with W row-major M x K trit, x length K f32/i32 → f32 span M
	// Also accepts bitplane mode when W is already trit span.
	nat("linear_trit", func(args []Value) (Value, error) {
		// linear_trit(W, x, M, K, scale) or linear_trit(W, x, M, K, scale, use_bitplane)
		// bitplane path uses packed pos/neg words + set-bit mixed accumulate (v1.05).
		if len(args) < 5 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "linear_trit(W, x, M, K, scale [, bitplane])"}
		}
		W, x := args[0].Span, args[1].Span
		M, K := args[2].AsInt(), args[3].AsInt()
		scale := float32(args[4].AsFloat())
		bitplane := false
		if len(args) >= 6 {
			bitplane = args[5].AsFloat() != 0 || args[5].AsInt() != 0
		}
		if M < 0 || K < 0 || len(W) < M*K || len(x) < K {
			return Void(), &Error{Msg: "linear_trit: dimension mismatch"}
		}
		Wi := make([]int8, M*K)
		for i := 0; i < M*K; i++ {
			Wi[i] = weights.ClampTritInt(W[i].AsInt())
		}
		xf := make([]float32, K)
		for t := 0; t < K; t++ {
			xf[t] = float32(x[t].AsFloat())
		}
		var y []float32
		if bitplane {
			y = weights.LinearTritPlanes(Wi, xf, M, K, scale)
		} else {
			y = weights.LinearTritScalar(Wi, xf, M, K, scale)
		}
		out := make([]Value, M)
		for i := 0; i < M; i++ {
			out[i] = F32(float64(y[i]))
		}
		return Span(out), nil
	})
	// Scaled bitplane GEMM: C_f32 = scale * (A_trit @ B_trit)  A:MxK B:KxN
	nat("linear_trit_gemm", func(args []Value) (Value, error) {
		if len(args) < 6 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "linear_trit_gemm(A, B, M, K, N, scale)"}
		}
		M, _, N := args[2].AsInt(), args[3].AsInt(), args[4].AsInt()
		scale := args[5].AsFloat()
		C, err := tritGemm(args[:5], true, true)
		if err != nil {
			return Void(), err
		}
		out := make([]Value, M*N)
		for i, v := range C.Span {
			out[i] = F32(scale * float64(v.AsInt()))
		}
		return Span(out), nil
	})
	// scale(s, xs) already exists; ensure codegen later.
	// RMSNorm over last-dim vector: y = x / rms * g (g optional scale span or scalar)
	nat("rmsnorm", func(args []Value) (Value, error) {
		// rmsnorm(x [, eps] [, gain_span_or_scalar])
		if len(args) < 1 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "rmsnorm(x [, eps] [, gain])"}
		}
		x := args[0].Span
		n := len(x)
		if n == 0 {
			return Span(nil), nil
		}
		eps := 1e-6
		if len(args) >= 2 {
			eps = args[1].AsFloat()
		}
		ss := 0.0
		for _, v := range x {
			f := v.AsFloat()
			ss += f * f
		}
		inv := 1.0 / math.Sqrt(ss/float64(n)+eps)
		out := make([]Value, n)
		if len(args) >= 3 {
			g := args[2]
			if g.Kind == VSpan {
				for i := 0; i < n; i++ {
					gi := 1.0
					if i < len(g.Span) {
						gi = g.Span[i].AsFloat()
					}
					out[i] = F32(x[i].AsFloat() * inv * gi)
				}
			} else {
				gi := g.AsFloat()
				for i := 0; i < n; i++ {
					out[i] = F32(x[i].AsFloat() * inv * gi)
				}
			}
		} else {
			for i := 0; i < n; i++ {
				out[i] = F32(x[i].AsFloat() * inv)
			}
		}
		return Span(out), nil
	})
	// Softmax over span → span of probs
	nat("softmax", func(args []Value) (Value, error) {
		if len(args) < 1 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "softmax(x)"}
		}
		x := args[0].Span
		n := len(x)
		if n == 0 {
			return Span(nil), nil
		}
		// max for stability
		m := x[0].AsFloat()
		for i := 1; i < n; i++ {
			if v := x[i].AsFloat(); v > m {
				m = v
			}
		}
		exps := make([]float64, n)
		sum := 0.0
		for i := 0; i < n; i++ {
			e := math.Exp(x[i].AsFloat() - m)
			exps[i] = e
			sum += e
		}
		out := make([]Value, n)
		if sum == 0 {
			for i := range out {
				out[i] = F32(0)
			}
			return Span(out), nil
		}
		inv := 1.0 / sum
		for i := 0; i < n; i++ {
			out[i] = F32(exps[i] * inv)
		}
		return Span(out), nil
	})
	// SiLU(x) = x * sigmoid(x); elementwise on span or scalar
	nat("silu", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "silu(x)"}
		}
		if args[0].Kind == VSpan {
			out := make([]Value, len(args[0].Span))
			for i, v := range args[0].Span {
				f := v.AsFloat()
				out[i] = F32(f / (1.0 + math.Exp(-f)))
			}
			return Span(out), nil
		}
		f := args[0].AsFloat()
		return F32(f / (1.0 + math.Exp(-f))), nil
	})
	// GELU tanh approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
	nat("gelu", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "gelu(x)"}
		}
		gelu1 := func(f float64) float64 {
			return 0.5 * f * (1.0 + math.Tanh(0.7978845608028654*(f+0.044715*f*f*f)))
		}
		if args[0].Kind == VSpan {
			out := make([]Value, len(args[0].Span))
			for i, v := range args[0].Span {
				out[i] = F32(gelu1(v.AsFloat()))
			}
			return Span(out), nil
		}
		return F32(gelu1(args[0].AsFloat())), nil
	})
	// RoPE: rotate pairs in x (even/odd) by position * theta base.
	// rope(x, pos [, base=10000] [, n_rot optional use half dims])
	nat("rope", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "rope(x, pos [, base])"}
		}
		x := args[0].Span
		pos := args[1].AsFloat()
		base := 10000.0
		if len(args) >= 3 {
			base = args[2].AsFloat()
			if base <= 0 {
				base = 10000.0
			}
		}
		n := len(x)
		out := make([]Value, n)
		// process pairs (i,i+1)
		half := n / 2
		for i := 0; i < half; i++ {
			// freq for dim i
			// standard: theta = pos / base^(2i/d)
			d := float64(n)
			if d < 2 {
				d = 2
			}
			expo := float64(2*i) / d
			theta := pos / math.Pow(base, expo)
			c, s := math.Cos(theta), math.Sin(theta)
			// if even-odd interleaved layout
			xi := x[2*i].AsFloat()
			xj := x[2*i+1].AsFloat()
			out[2*i] = F32(xi*c - xj*s)
			out[2*i+1] = F32(xi*s + xj*c)
		}
		if n%2 == 1 {
			out[n-1] = F32(x[n-1].AsFloat())
		}
		return Span(out), nil
	})
	// residual add: residual_add(a,b) == span_add (alias)
	nat("residual_add", func(args []Value) (Value, error) {
		return spanOp(args, func(a, b float64) float64 { return a + b })
	})
	// attention scores: attn_scores(q, K, d) → softmax(q·K^T / sqrt(d)) where
	// q: span d, K: flat n_tok x d row-major → probs span n_tok
	nat("attn_scores", func(args []Value) (Value, error) {
		if len(args) < 3 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "attn_scores(q, K, d)"}
		}
		q := args[0].Span
		K := args[1].Span
		d := args[2].AsInt()
		if d <= 0 {
			return Void(), &Error{Msg: "attn_scores: d must be > 0"}
		}
		nTok := len(K) / d
		if nTok*d != len(K) || len(q) < d {
			return Void(), &Error{Msg: "attn_scores: dimension mismatch"}
		}
		scale := 1.0 / math.Sqrt(float64(d))
		logits := make([]Value, nTok)
		for t := 0; t < nTok; t++ {
			acc := 0.0
			for i := 0; i < d; i++ {
				acc += q[i].AsFloat() * K[t*d+i].AsFloat()
			}
			logits[t] = F32(acc * scale)
		}
		// inline softmax
		m := logits[0].AsFloat()
		for i := 1; i < nTok; i++ {
			if v := logits[i].AsFloat(); v > m {
				m = v
			}
		}
		exps := make([]float64, nTok)
		sum := 0.0
		for i := 0; i < nTok; i++ {
			e := math.Exp(logits[i].AsFloat() - m)
			exps[i] = e
			sum += e
		}
		out := make([]Value, nTok)
		inv := 0.0
		if sum != 0 {
			inv = 1.0 / sum
		}
		for i := 0; i < nTok; i++ {
			out[i] = F32(exps[i] * inv)
		}
		return Span(out), nil
	})
	// attention weighted values: attn_values(probs, V, d) → span d
	// V: n_tok x d row-major
	nat("attn_values", func(args []Value) (Value, error) {
		if len(args) < 3 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "attn_values(probs, V, d)"}
		}
		p := args[0].Span
		V := args[1].Span
		d := args[2].AsInt()
		nTok := len(p)
		if d <= 0 || len(V) < nTok*d {
			return Void(), &Error{Msg: "attn_values: dimension mismatch"}
		}
		out := make([]Value, d)
		for i := 0; i < d; i++ {
			acc := 0.0
			for t := 0; t < nTok; t++ {
				acc += p[t].AsFloat() * V[t*d+i].AsFloat()
			}
			out[i] = F32(acc)
		}
		return Span(out), nil
	})
	// Full single-head attention: attn(q, K, V, d) → span d
	nat("attn", func(args []Value) (Value, error) {
		if len(args) < 4 || args[0].Kind != VSpan || args[1].Kind != VSpan || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "attn(q, K, V, d)"}
		}
		// call scores then values via direct logic (avoid Env get races)
		q, K, V := args[0].Span, args[1].Span, args[2].Span
		d := args[3].AsInt()
		if d <= 0 {
			return Void(), &Error{Msg: "attn: d must be > 0"}
		}
		nTok := len(K) / d
		if nTok*d != len(K) || len(V) < nTok*d || len(q) < d {
			return Void(), &Error{Msg: "attn: dimension mismatch"}
		}
		scale := 1.0 / math.Sqrt(float64(d))
		logits := make([]float64, nTok)
		m := -1e30
		for t := 0; t < nTok; t++ {
			acc := 0.0
			for i := 0; i < d; i++ {
				acc += q[i].AsFloat() * K[t*d+i].AsFloat()
			}
			logits[t] = acc * scale
			if logits[t] > m {
				m = logits[t]
			}
		}
		exps := make([]float64, nTok)
		sum := 0.0
		for t := 0; t < nTok; t++ {
			e := math.Exp(logits[t] - m)
			exps[t] = e
			sum += e
		}
		inv := 0.0
		if sum != 0 {
			inv = 1.0 / sum
		}
		out := make([]Value, d)
		for i := 0; i < d; i++ {
			acc := 0.0
			for t := 0; t < nTok; t++ {
				acc += (exps[t] * inv) * V[t*d+i].AsFloat()
			}
			out[i] = F32(acc)
		}
		return Span(out), nil
	})
	// argmax over span → i32 index
	nat("argmax", func(args []Value) (Value, error) {
		if len(args) < 1 || args[0].Kind != VSpan || len(args[0].Span) == 0 {
			return Void(), &Error{Msg: "argmax(x)"}
		}
		best := 0
		bv := args[0].Span[0].AsFloat()
		for i := 1; i < len(args[0].Span); i++ {
			if v := args[0].Span[i].AsFloat(); v > bv {
				bv = v
				best = i
			}
		}
		return I32(best), nil
	})
	// temperature sampling: sample_logits(logits [, temp] [, seed])
	// temp<=0 → greedy argmax. temp>0 → softmax(logits/temp) then categorical draw via deterministic LCG.
	// Optional seed reseeds RNG when provided (v0.82 additive).
	nat("sample_logits", func(args []Value) (Value, error) {
		if len(args) < 1 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "sample_logits(logits [, temp] [, seed])"}
		}
		logits := args[0].Span
		temp := 0.0
		if len(args) >= 2 {
			temp = args[1].AsFloat()
		}
		if len(args) >= 3 {
			s := uint64(args[2].AsInt())
			if s == 0 {
				s = 0x9e3779b97f4a7c15
			}
			trinRNG = s
		}
		if len(logits) == 0 {
			return I32(0), nil
		}
		if temp <= 1e-8 {
			// greedy
			best, bv := 0, logits[0].AsFloat()
			for i := 1; i < len(logits); i++ {
				if v := logits[i].AsFloat(); v > bv {
					bv, best = v, i
				}
			}
			return I32(best), nil
		}
		// softmax(logits/temp) then categorical sample
		m := logits[0].AsFloat() / temp
		for i := 1; i < len(logits); i++ {
			v := logits[i].AsFloat() / temp
			if v > m {
				m = v
			}
		}
		exps := make([]float64, len(logits))
		sum := 0.0
		for i := range logits {
			e := math.Exp(logits[i].AsFloat()/temp - m)
			exps[i] = e
			sum += e
		}
		r := trinRNGNext() * sum
		acc := 0.0
		for i, e := range exps {
			acc += e
			if r <= acc {
				return I32(i), nil
			}
		}
		return I32(len(logits) - 1), nil
	})
	// embed lookup: embed_row(table, idx, dim) where table is V*dim row-major
	nat("embed_row", func(args []Value) (Value, error) {
		if len(args) < 3 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "embed_row(table, idx, dim)"}
		}
		tab := args[0].Span
		idx := args[1].AsInt()
		dim := args[2].AsInt()
		if dim <= 0 || idx < 0 || (idx+1)*dim > len(tab) {
			return Void(), &Error{Msg: "embed_row: OOB"}
		}
		out := make([]Value, dim)
		base := idx * dim
		for i := 0; i < dim; i++ {
			out[i] = F32(tab[base+i].AsFloat())
		}
		return Span(out), nil
	})
	// span_scale alias of scale (s, xs)
	nat("span_scale", func(args []Value) (Value, error) {
		if len(args) < 2 || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "span_scale expects (s, span)"}
		}
		s := args[0].AsFloat()
		out := make([]Value, len(args[1].Span))
		for i, x := range args[1].Span {
			out[i] = F32(s * x.AsFloat())
		}
		return Span(out), nil
	})


	// ---- Weight / tensor I/O (v0.81 production) ----
	// getenv(key) → str (empty if unset) — v1.19
	nat("getenv", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "getenv(key)"}
		}
		return Str(os.Getenv(strArg(args[0]))), nil
	})
	// str_contains(s, sub) → i32 (1 if sub in s)
	nat("str_contains", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "str_contains(s, sub)"}
		}
		if strings.Contains(strArg(args[0]), strArg(args[1])) {
			return I32(1), nil
		}
		return I32(0), nil
	})
	// load_f32_csv(path) → span[f32] from comma/whitespace separated floats
	// Skips empty tokens and full-line `#` comments (and trailing `# ...` on a line).
	nat("load_f32_csv", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "load_f32_csv(path)"}
		}
		path := strArg(args[0])
		b, err := os.ReadFile(path)
		if err != nil {
			return Void(), &Error{Msg: "load_f32_csv: " + err.Error()}
		}
		// Strip # comments per line, then tokenize on comma/whitespace.
		var cleaned strings.Builder
		for _, line := range strings.Split(string(b), "\n") {
			if i := strings.IndexByte(line, '#'); i >= 0 {
				line = line[:i]
			}
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			if cleaned.Len() > 0 {
				cleaned.WriteByte('\n')
			}
			cleaned.WriteString(line)
		}
		fields := strings.FieldsFunc(cleaned.String(), func(r rune) bool {
			return r == ',' || r == ' ' || r == '\n' || r == '\t' || r == '\r' || r == ';'
		})
		out := make([]Value, 0, len(fields))
		for _, f := range fields {
			if f == "" {
				continue
			}
			v, err := strconv.ParseFloat(f, 64)
			if err != nil {
				return Void(), &Error{Msg: "load_f32_csv parse: " + f}
			}
			out = append(out, F32(v))
		}
		return Span(out), nil
	})
	// save_f32_csv(path, span)
	nat("save_f32_csv", func(args []Value) (Value, error) {
		if len(args) < 2 || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "save_f32_csv(path, span)"}
		}
		path := args[0].String()
		if args[0].Kind == VStr {
			path = args[0].Str
		}
		parts := make([]string, len(args[1].Span))
		for i, x := range args[1].Span {
			parts[i] = strconv.FormatFloat(x.AsFloat(), 'g', -1, 64)
		}
		if err := os.WriteFile(path, []byte(strings.Join(parts, ",")), 0o644); err != nil {
			return Void(), &Error{Msg: "save_f32_csv: " + err.Error()}
		}
		return I32(len(parts)), nil
	})
	// load_trit_csv(path) → trit span from integers in {-1,0,1}
	nat("load_trit_csv", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "load_trit_csv(path)"}
		}
		path := args[0].String()
		if args[0].Kind == VStr {
			path = args[0].Str
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return Void(), &Error{Msg: "load_trit_csv: " + err.Error()}
		}
		fields := strings.FieldsFunc(string(b), func(r rune) bool {
			return r == ',' || r == ' ' || r == '\n' || r == '\t' || r == '\r' || r == ';'
		})
		out := make([]Value, 0, len(fields))
		for _, f := range fields {
			if f == "" {
				continue
			}
			v, err := strconv.ParseInt(f, 10, 64)
			if err != nil {
				// allow float text
				fv, err2 := strconv.ParseFloat(f, 64)
				if err2 != nil {
					return Void(), &Error{Msg: "load_trit_csv parse: " + f}
				}
				v = int64(fv)
			}
			out = append(out, Trit(clampTrit(int(v))))
		}
		return Span(out), nil
	})
	// quantize_file_absmean(path) → save companion .trit.csv and return [trits, scale]
	nat("quantize_file_absmean", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "quantize_file_absmean(path)"}
		}
		path := args[0].String()
		if args[0].Kind == VStr {
			path = args[0].Str
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return Void(), &Error{Msg: "quantize_file_absmean: " + err.Error()}
		}
		fields := strings.FieldsFunc(string(b), func(r rune) bool {
			return r == ',' || r == ' ' || r == '\n' || r == '\t' || r == '\r' || r == ';'
		})
		xs := make([]Value, 0, len(fields))
		for _, f := range fields {
			if f == "" {
				continue
			}
			v, err := strconv.ParseFloat(f, 64)
			if err != nil {
				return Void(), &Error{Msg: "quantize_file_absmean parse: " + f}
			}
			xs = append(xs, F32(v))
		}
		n := len(xs)
		if n == 0 {
			return Span([]Value{Span(nil), F32(0)}), nil
		}
		sumAbs := 0.0
		for _, x := range xs {
			v := x.AsFloat()
			if v < 0 {
				v = -v
			}
			sumAbs += v
		}
		scale := sumAbs / float64(n)
		out := make([]Value, n)
		if scale == 0 {
			for i := range out {
				out[i] = Trit(0)
			}
		} else {
			for i, x := range xs {
				r := x.AsFloat() / scale
				t := 0
				if r >= 0.5 {
					t = 1
				} else if r <= -0.5 {
					t = -1
				}
				out[i] = Trit(t)
			}
		}
		// write side-car
		parts := make([]string, n)
		for i, t := range out {
			parts[i] = strconv.Itoa(t.AsInt())
		}
		_ = os.WriteFile(path+".trit.csv", []byte(strings.Join(parts, ",")), 0o644)
		_ = os.WriteFile(path+".scale.txt", []byte(strconv.FormatFloat(scale, 'g', -1, 64)), 0o644)
		return Span([]Value{Span(out), F32(scale)}), nil
	})


	// Multi-head attention convenience (v0.81): split q/K/V into H heads of d_h = d/H
	// mha(q, K, V, d, h) where q len d, K/V n_tok*d, returns span d (concat heads then no output proj)
	nat("mha", func(args []Value) (Value, error) {
		if len(args) < 5 || args[0].Kind != VSpan || args[1].Kind != VSpan || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "mha(q, K, V, d, h)"}
		}
		q, K, V := args[0].Span, args[1].Span, args[2].Span
		d, h := args[3].AsInt(), args[4].AsInt()
		if h <= 0 || d%h != 0 {
			return Void(), &Error{Msg: "mha: d must be divisible by h"}
		}
		dh := d / h
		nTok := len(K) / d
		if nTok*d != len(K) || len(V) < nTok*d || len(q) < d {
			return Void(), &Error{Msg: "mha: dimension mismatch"}
		}
		out := make([]Value, d)
		for head := 0; head < h; head++ {
			// gather head slices
			qs := make([]Value, dh)
			for i := 0; i < dh; i++ {
				qs[i] = q[head*dh+i]
			}
			Ks := make([]Value, nTok*dh)
			Vs := make([]Value, nTok*dh)
			for t := 0; t < nTok; t++ {
				for i := 0; i < dh; i++ {
					Ks[t*dh+i] = K[t*d+head*dh+i]
					Vs[t*dh+i] = V[t*d+head*dh+i]
				}
			}
			// call attn logic inline
			scale := 1.0 / math.Sqrt(float64(dh))
			logits := make([]float64, nTok)
			m := -1e30
			for t := 0; t < nTok; t++ {
				acc := 0.0
				for i := 0; i < dh; i++ {
					acc += qs[i].AsFloat() * Ks[t*dh+i].AsFloat()
				}
				logits[t] = acc * scale
				if logits[t] > m {
					m = logits[t]
				}
			}
			exps := make([]float64, nTok)
			sum := 0.0
			for t := 0; t < nTok; t++ {
				e := math.Exp(logits[t] - m)
				exps[t] = e
				sum += e
			}
			inv := 0.0
			if sum != 0 {
				inv = 1.0 / sum
			}
			for i := 0; i < dh; i++ {
				acc := 0.0
				for t := 0; t < nTok; t++ {
					acc += (exps[t] * inv) * Vs[t*dh+i].AsFloat()
				}
				out[head*dh+i] = F32(acc)
			}
		}
		return Span(out), nil
	})

	// ---- v0.82 production KV + deterministic sampling helpers ----
	// span_concat(a,b) → append elements of b after a
	nat("span_concat", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "span_concat(a, b)"}
		}
		out := make([]Value, 0, len(args[0].Span)+len(args[1].Span))
		out = append(out, args[0].Span...)
		out = append(out, args[1].Span...)
		return Span(out), nil
	})
	// kv_new() → empty span cache; kv_append(cache, row) appends one token row (dim elements)
	nat("kv_new", func(args []Value) (Value, error) {
		return Span(nil), nil
	})
	nat("kv_append", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "kv_append(cache, row)"}
		}
		out := make([]Value, 0, len(args[0].Span)+len(args[1].Span))
		out = append(out, args[0].Span...)
		out = append(out, args[1].Span...)
		return Span(out), nil
	})
	// kv_len(cache, d) → number of tokens stored (len/d)
	nat("kv_len", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "kv_len(cache, d)"}
		}
		d := args[1].AsInt()
		if d <= 0 {
			return Void(), &Error{Msg: "kv_len: d must be > 0"}
		}
		return I32(len(args[0].Span) / d), nil
	})
	// kv_row(cache, idx, d) → token row at idx
	nat("kv_row", func(args []Value) (Value, error) {
		if len(args) < 3 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "kv_row(cache, idx, d)"}
		}
		cache := args[0].Span
		idx, d := args[1].AsInt(), args[2].AsInt()
		if d <= 0 || idx < 0 || (idx+1)*d > len(cache) {
			return Void(), &Error{Msg: "kv_row: OOB"}
		}
		out := make([]Value, d)
		base := idx * d
		for i := 0; i < d; i++ {
			out[i] = F32(cache[base+i].AsFloat())
		}
		return Span(out), nil
	})
	// Deterministic LCG RNG (shared with Go backend runtime)
	nat("rng_seed", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "rng_seed(s)"}
		}
		s := uint64(args[0].AsInt())
		if s == 0 {
			s = 0x9e3779b97f4a7c15
		}
		trinRNG = s
		return I32(0), nil
	})
	nat("rng_uniform", func(args []Value) (Value, error) {
		return F32(trinRNGNext()), nil
	})
	// sample_topk(logits, k [, temp] [, seed]) — keep top-k, renormalize, sample (greedy if temp<=0)
	nat("sample_topk", func(args []Value) (Value, error) {
		if len(args) < 2 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "sample_topk(logits, k [, temp] [, seed])"}
		}
		logits := args[0].Span
		k := args[1].AsInt()
		temp := 0.0
		if len(args) >= 3 {
			temp = args[2].AsFloat()
		}
		if len(args) >= 4 {
			s := uint64(args[3].AsInt())
			if s == 0 {
				s = 0x9e3779b97f4a7c15
			}
			trinRNG = s
		}
		n := len(logits)
		if n == 0 {
			return I32(0), nil
		}
		if k <= 0 || k > n {
			k = n
		}
		// indices sorted by logit desc
		idxs := make([]int, n)
		for i := range idxs {
			idxs[i] = i
		}
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				if logits[idxs[j]].AsFloat() > logits[idxs[i]].AsFloat() {
					idxs[i], idxs[j] = idxs[j], idxs[i]
				}
			}
		}
		top := idxs[:k]
		if temp <= 1e-8 {
			return I32(top[0]), nil
		}
		// softmax over top-k / temp then sample
		m := logits[top[0]].AsFloat() / temp
		for _, i := range top[1:] {
			v := logits[i].AsFloat() / temp
			if v > m {
				m = v
			}
		}
		weights := make([]float64, k)
		sum := 0.0
		for t, i := range top {
			e := math.Exp(logits[i].AsFloat()/temp - m)
			weights[t] = e
			sum += e
		}
		r := trinRNGNext() * sum
		acc := 0.0
		for t, w := range weights {
			acc += w
			if r <= acc {
				return I32(top[t]), nil
			}
		}
		return I32(top[k-1]), nil
	})

	// ---- v0.83–0.85 GGUF / safetensors importers (mmap opens) ----
	// load_safetensors_list(path) → span[str] of tensor names
	nat("load_safetensors_list", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "load_safetensors_list(path)"}
		}
		path := strArg(args[0])
		f, err := weights.OpenSafeTensors(path)
		if err != nil {
			return Void(), &Error{Msg: "load_safetensors_list: " + err.Error()}
		}
		defer f.Close()
		names := f.List()
		out := make([]Value, len(names))
		for i, n := range names {
			out[i] = Str(n)
		}
		return Span(out), nil
	})
	// load_safetensors_tensor(path, name) → span[f32] flat
	nat("load_safetensors_tensor", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "load_safetensors_tensor(path, name)"}
		}
		path, name := strArg(args[0]), strArg(args[1])
		f, err := weights.OpenSafeTensors(path)
		if err != nil {
			return Void(), &Error{Msg: "load_safetensors_tensor: " + err.Error()}
		}
		defer f.Close()
		f32s, _, err := f.LoadF32(name)
		if err != nil {
			return Void(), &Error{Msg: "load_safetensors_tensor: " + err.Error()}
		}
		return f32SliceToSpan(f32s), nil
	})
	// load_safetensors_shape(path, name) → span[i32] dims
	nat("load_safetensors_shape", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "load_safetensors_shape(path, name)"}
		}
		path, name := strArg(args[0]), strArg(args[1])
		f, err := weights.OpenSafeTensors(path)
		if err != nil {
			return Void(), &Error{Msg: "load_safetensors_shape: " + err.Error()}
		}
		defer f.Close()
		info, ok := f.Tensors[name]
		if !ok {
			return Void(), &Error{Msg: "load_safetensors_shape: missing " + name}
		}
		out := make([]Value, len(info.Shape))
		for i, d := range info.Shape {
			out[i] = I32(int(d))
		}
		return Span(out), nil
	})
	// load_gguf_list(path) → span[str]
	nat("load_gguf_list", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "load_gguf_list(path)"}
		}
		path := strArg(args[0])
		f, err := weights.OpenGGUF(path)
		if err != nil {
			return Void(), &Error{Msg: "load_gguf_list: " + err.Error()}
		}
		defer f.Close()
		names := f.List()
		out := make([]Value, len(names))
		for i, n := range names {
			out[i] = Str(n)
		}
		return Span(out), nil
	})
	// load_gguf_tensor(path, name) → span[f32]
	nat("load_gguf_tensor", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "load_gguf_tensor(path, name)"}
		}
		path, name := strArg(args[0]), strArg(args[1])
		f, err := weights.OpenGGUF(path)
		if err != nil {
			return Void(), &Error{Msg: "load_gguf_tensor: " + err.Error()}
		}
		defer f.Close()
		f32s, _, err := f.LoadF32(name)
		if err != nil {
			return Void(), &Error{Msg: "load_gguf_tensor: " + err.Error()}
		}
		return f32SliceToSpan(f32s), nil
	})
	// load_gguf_shape(path, name) → span[i32]
	nat("load_gguf_shape", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "load_gguf_shape(path, name)"}
		}
		path, name := strArg(args[0]), strArg(args[1])
		f, err := weights.OpenGGUF(path)
		if err != nil {
			return Void(), &Error{Msg: "load_gguf_shape: " + err.Error()}
		}
		defer f.Close()
		ti, ok := f.ByName[name]
		if !ok {
			return Void(), &Error{Msg: "load_gguf_shape: missing " + name}
		}
		out := make([]Value, len(ti.Dims))
		for i, d := range ti.Dims {
			out[i] = I32(int(d))
		}
		return Span(out), nil
	})
	// load_gguf_meta(path, key) → string (empty if missing)
	nat("load_gguf_meta", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "load_gguf_meta(path, key)"}
		}
		path, key := strArg(args[0]), strArg(args[1])
		f, err := weights.OpenGGUF(path)
		if err != nil {
			return Void(), &Error{Msg: "load_gguf_meta: " + err.Error()}
		}
		defer f.Close()
		return Str(f.Meta[key]), nil
	})
	// load_weight_tensor(path, name) → span[f32] auto-detect GGUF or safetensors
	nat("load_weight_tensor", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "load_weight_tensor(path, name)"}
		}
		path, name := strArg(args[0]), strArg(args[1])
		f32s, _, err := weights.AutoLoad(path, name)
		if err != nil {
			return Void(), &Error{Msg: "load_weight_tensor: " + err.Error()}
		}
		return f32SliceToSpan(f32s), nil
	})
	// ---- v0.86 retained weight sessions ----
	// open_weight_session(path) → i32 session id (mmap retained)
	nat("open_weight_session", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "open_weight_session(path)"}
		}
		s, err := weights.OpenSession(strArg(args[0]))
		if err != nil {
			return Void(), &Error{Msg: "open_weight_session: " + err.Error()}
		}
		return I32(int(s.ID)), nil
	})
	// close_weight_session(id) → void (idempotent)
	nat("close_weight_session", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "close_weight_session(id)"}
		}
		id := int32(args[0].AsInt())
		if err := weights.CloseSession(id); err != nil {
			return Void(), &Error{Msg: "close_weight_session: " + err.Error()}
		}
		return Void(), nil
	})
	// weight_session_list(id) → span[str]
	nat("weight_session_list", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "weight_session_list(id)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_list: " + err.Error()}
		}
		names, err := s.List()
		if err != nil {
			return Void(), &Error{Msg: "weight_session_list: " + err.Error()}
		}
		out := make([]Value, len(names))
		for i, n := range names {
			out[i] = Str(n)
		}
		return Span(out), nil
	})
	// weight_session_shape(id, name) → span[i32]
	nat("weight_session_shape", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "weight_session_shape(id, name)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_shape: " + err.Error()}
		}
		sh, err := s.Shape(strArg(args[1]))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_shape: " + err.Error()}
		}
		out := make([]Value, len(sh))
		for i, d := range sh {
			out[i] = I32(int(d))
		}
		return Span(out), nil
	})
	// weight_session_tensor(id, name) → span[f32]
	nat("weight_session_tensor", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "weight_session_tensor(id, name)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_tensor: " + err.Error()}
		}
		f32s, _, err := s.LoadF32(strArg(args[1]))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_tensor: " + err.Error()}
		}
		return f32SliceToSpan(f32s), nil
	})
	// weight_session_meta(id, key) → string
	nat("weight_session_meta", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "weight_session_meta(id, key)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_meta: " + err.Error()}
		}
		return Str(s.Meta(strArg(args[1]))), nil
	})

	// weight_session_auto_namemap(id [, nLayersHint]) → style string (v1.17)
	nat("weight_session_auto_namemap", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "weight_session_auto_namemap(id [, nLayersHint])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_auto_namemap: " + err.Error()}
		}
		hint := 0
		if len(args) >= 2 {
			hint = args[1].AsInt()
		}
		m, err := s.AutoNameMap(hint)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_auto_namemap: " + err.Error()}
		}
		return Str(m.Style), nil
	})

	// weight_session_resolve_name(id, short) → on-disk name
	nat("weight_session_resolve_name", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "weight_session_resolve_name(id, short)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_resolve_name: " + err.Error()}
		}
		return Str(s.ResolveName(strArg(args[1]))), nil
	})

	// weight_session_resolve_csv(id, csv) → resolved CSV
	nat("weight_session_resolve_csv", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "weight_session_resolve_csv(id, csv)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_resolve_csv: " + err.Error()}
		}
		return Str(s.ResolveNameCSV(strArg(args[1]))), nil
	})

	// weight_session_namemap_style(id) → style string
	nat("weight_session_namemap_style", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "weight_session_namemap_style(id)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_namemap_style: " + err.Error()}
		}
		return Str(s.GetNameMap().Style), nil
	})

	// weight_session_expand_layers(id, nLayers, role) → resolved CSV of L{i}_role
	nat("weight_session_expand_layers", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Void(), &Error{Msg: "weight_session_expand_layers(id, nLayers, role)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_expand_layers: " + err.Error()}
		}
		csv := weights.ExpandLayerCSV(args[1].AsInt(), strArg(args[2]))
		return Str(s.ResolveNameCSV(csv)), nil
	})
	// weight_session_mapped(id) → i32
	nat("weight_session_mapped", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "weight_session_mapped(id)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_mapped: " + err.Error()}
		}
		if s.Mapped {
			return I32(1), nil
		}
		return I32(0), nil
	})

	// gqa_expand_kv(kv, nTok, nKVHeads, nHeads, dh) — expand packed KV for MHA (v0.92)
	nat("gqa_expand_kv", func(args []Value) (Value, error) {
		if len(args) < 5 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "gqa_expand_kv(kv, nTok, nKVHeads, nHeads, dh)"}
		}
		sp := args[0].Span
		kv := make([]float32, len(sp))
		for i, v := range sp {
			kv[i] = float32(v.AsFloat())
		}
		out, err := weights.GQAExpandKV(kv, args[1].AsInt(), args[2].AsInt(), args[3].AsInt(), args[4].AsInt())
		if err != nil {
			return Void(), &Error{Msg: "gqa_expand_kv: " + err.Error()}
		}
		return f32SliceToSpan(out), nil
	})
	// gqa_repeat_kv_row(row, nKVHeads, nHeads, dh)
	nat("gqa_repeat_kv_row", func(args []Value) (Value, error) {
		if len(args) < 4 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "gqa_repeat_kv_row(row, nKVHeads, nHeads, dh)"}
		}
		sp := args[0].Span
		row := make([]float32, len(sp))
		for i, v := range sp {
			row[i] = float32(v.AsFloat())
		}
		out, err := weights.GQARepeatKVRow(row, args[1].AsInt(), args[2].AsInt(), args[3].AsInt())
		if err != nil {
			return Void(), &Error{Msg: "gqa_repeat_kv_row: " + err.Error()}
		}
		return f32SliceToSpan(out), nil
	})
	// weight_session_attn_proj_gqa(id, q,k,v, x, dIn, nHeads, nKVHeads, dh [, tileRows]) → [q,k,v]
	nat("weight_session_attn_proj_gqa", func(args []Value) (Value, error) {
		if len(args) < 9 || args[4].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_attn_proj_gqa(id, q,k,v, x, dIn, nHeads, nKVHeads, dh [, tileRows])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_attn_proj_gqa: " + err.Error()}
		}
		xspan := args[4].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		tile := 0
		if len(args) >= 10 {
			tile = args[9].AsInt()
		}
		q, k, v, err := s.AttnProjGQA(strArg(args[1]), strArg(args[2]), strArg(args[3]), x, args[5].AsInt(), args[6].AsInt(), args[7].AsInt(), args[8].AsInt(), tile)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_attn_proj_gqa: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(q), f32SliceToSpan(k), f32SliceToSpan(v)}), nil
	})
	// weight_session_gqa(id, q,k,v, x, Kpast, Vpast, dIn, nHeads, nKVHeads, dh [, o] [, pos] [, base] [, tile]) → [y,K,V]
	nat("weight_session_gqa", func(args []Value) (Value, error) {
		if len(args) < 11 || args[4].Kind != VSpan || args[5].Kind != VSpan || args[6].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_gqa(id, q,k,v, x, Kpast, Vpast, dIn, nHeads, nKVHeads, dh [, oName] [, pos] [, base] [, tileRows])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_gqa: " + err.Error()}
		}
		toF32 := func(sp []Value) []float32 {
			o := make([]float32, len(sp))
			for i, v := range sp {
				o[i] = float32(v.AsFloat())
			}
			return o
		}
		x := toF32(args[4].Span)
		Kp := toF32(args[5].Span)
		Vp := toF32(args[6].Span)
		dIn := args[7].AsInt()
		nHeads, nKV, dh := args[8].AsInt(), args[9].AsInt(), args[10].AsInt()
		oName := ""
		pos, base := 0.0, 0.0
		tile := 0
		if len(args) >= 12 {
			oName = strArg(args[11])
		}
		if len(args) >= 13 {
			pos = args[12].AsFloat()
		}
		if len(args) >= 14 {
			base = args[13].AsFloat()
		}
		if len(args) >= 15 {
			tile = args[14].AsInt()
		}
		y, kn, vn, err := s.SessionAttnGQA(strArg(args[1]), strArg(args[2]), strArg(args[3]), oName, x, Kp, Vp, dIn, nHeads, nKV, dh, pos, base, tile)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_gqa: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(y), f32SliceToSpan(kn), f32SliceToSpan(vn)}), nil
	})

// weight_session_attn_proj(id, q, k, v, x, dIn, dOut [, tileRows]) → [q,k,v] (v0.91)
	nat("weight_session_attn_proj", func(args []Value) (Value, error) {
		if len(args) < 7 || args[4].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_attn_proj(id, qName, kName, vName, x, dIn, dOut [, tileRows])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_attn_proj: " + err.Error()}
		}
		xspan := args[4].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		tile := 0
		if len(args) >= 8 {
			tile = args[7].AsInt()
		}
		q, k, v, err := s.AttnProj(strArg(args[1]), strArg(args[2]), strArg(args[3]), x, args[5].AsInt(), args[6].AsInt(), tile)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_attn_proj: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(q), f32SliceToSpan(k), f32SliceToSpan(v)}), nil
	})
	// weight_session_attn(id, q, k, v, x, Kpast, Vpast, dIn, dOut [, oName] [, h] [, pos] [, base] [, tileRows])
	// → [y, Knew, Vnew]
	nat("weight_session_attn", func(args []Value) (Value, error) {
		if len(args) < 9 || args[4].Kind != VSpan || args[5].Kind != VSpan || args[6].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_attn(id, qName, kName, vName, x, Kpast, Vpast, dIn, dOut [, oName] [, h] [, pos] [, base] [, tileRows])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_attn: " + err.Error()}
		}
		toF32 := func(sp []Value) []float32 {
			o := make([]float32, len(sp))
			for i, v := range sp {
				o[i] = float32(v.AsFloat())
			}
			return o
		}
		x := toF32(args[4].Span)
		Kp := toF32(args[5].Span)
		Vp := toF32(args[6].Span)
		dIn, dOut := args[7].AsInt(), args[8].AsInt()
		oName := ""
		h := 1
		pos, base := 0.0, 0.0
		tile := 0
		if len(args) >= 10 {
			oName = strArg(args[9])
		}
		if len(args) >= 11 {
			h = args[10].AsInt()
		}
		if len(args) >= 12 {
			pos = args[11].AsFloat()
		}
		if len(args) >= 13 {
			base = args[12].AsFloat()
		}
		if len(args) >= 14 {
			tile = args[13].AsInt()
		}
		y, kn, vn, err := s.SessionAttn(strArg(args[1]), strArg(args[2]), strArg(args[3]), oName, x, Kp, Vp, dIn, dOut, h, pos, base, tile)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_attn: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(y), f32SliceToSpan(kn), f32SliceToSpan(vn)}), nil
	})
	// weight_session_mha — same as attn with explicit heads (h required after oName)
	nat("weight_session_mha", func(args []Value) (Value, error) {
		if len(args) < 10 || args[4].Kind != VSpan || args[5].Kind != VSpan || args[6].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_mha(id, qName, kName, vName, x, Kpast, Vpast, dIn, dOut, h [, oName] [, pos] [, base] [, tileRows])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_mha: " + err.Error()}
		}
		toF32 := func(sp []Value) []float32 {
			o := make([]float32, len(sp))
			for i, v := range sp {
				o[i] = float32(v.AsFloat())
			}
			return o
		}
		x := toF32(args[4].Span)
		Kp := toF32(args[5].Span)
		Vp := toF32(args[6].Span)
		dIn, dOut, h := args[7].AsInt(), args[8].AsInt(), args[9].AsInt()
		oName := ""
		pos, base := 0.0, 0.0
		tile := 0
		if len(args) >= 11 {
			oName = strArg(args[10])
		}
		if len(args) >= 12 {
			pos = args[11].AsFloat()
		}
		if len(args) >= 13 {
			base = args[12].AsFloat()
		}
		if len(args) >= 14 {
			tile = args[13].AsInt()
		}
		y, kn, vn, err := s.SessionAttn(strArg(args[1]), strArg(args[2]), strArg(args[3]), oName, x, Kp, Vp, dIn, dOut, h, pos, base, tile)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_mha: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(y), f32SliceToSpan(kn), f32SliceToSpan(vn)}), nil
	})

	// weight_session_attn_proj_trit(id, q, k, v, x, dIn, dOut [, bitplane]) → [q,k,v] (v1.10)
	nat("weight_session_attn_proj_trit", func(args []Value) (Value, error) {
		if len(args) < 7 || args[4].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_attn_proj_trit(id, qName, kName, vName, x, dIn, dOut [, bitplane])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_attn_proj_trit: " + err.Error()}
		}
		xspan := args[4].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		bitplane := false
		if len(args) >= 8 {
			bitplane = args[7].AsFloat() != 0 || args[7].AsInt() != 0
		}
		q, k, v, err := s.AttnProjTrit(strArg(args[1]), strArg(args[2]), strArg(args[3]), x, args[5].AsInt(), args[6].AsInt(), bitplane)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_attn_proj_trit: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(q), f32SliceToSpan(k), f32SliceToSpan(v)}), nil
	})
	// weight_session_attn_trit(id, q,k,v, x, Kpast, Vpast, dIn, dOut [, o] [, h] [, pos] [, base] [, bitplane])
	nat("weight_session_attn_trit", func(args []Value) (Value, error) {
		if len(args) < 9 || args[4].Kind != VSpan || args[5].Kind != VSpan || args[6].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_attn_trit(id, qName, kName, vName, x, Kpast, Vpast, dIn, dOut [, oName] [, h] [, pos] [, base] [, bitplane])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_attn_trit: " + err.Error()}
		}
		toF32 := func(sp []Value) []float32 {
			o := make([]float32, len(sp))
			for i, v := range sp {
				o[i] = float32(v.AsFloat())
			}
			return o
		}
		x := toF32(args[4].Span)
		Kp := toF32(args[5].Span)
		Vp := toF32(args[6].Span)
		dIn, dOut := args[7].AsInt(), args[8].AsInt()
		oName := ""
		h := 1
		pos, base := 0.0, 0.0
		bitplane := false
		if len(args) >= 10 {
			oName = strArg(args[9])
		}
		if len(args) >= 11 {
			h = args[10].AsInt()
		}
		if len(args) >= 12 {
			pos = args[11].AsFloat()
		}
		if len(args) >= 13 {
			base = args[12].AsFloat()
		}
		if len(args) >= 14 {
			bitplane = args[13].AsFloat() != 0 || args[13].AsInt() != 0
		}
		y, kn, vn, err := s.SessionAttnTrit(strArg(args[1]), strArg(args[2]), strArg(args[3]), oName, x, Kp, Vp, dIn, dOut, h, pos, base, bitplane)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_attn_trit: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(y), f32SliceToSpan(kn), f32SliceToSpan(vn)}), nil
	})
	// weight_session_mha_trit — same as attn_trit with explicit heads (h required after oName)
	nat("weight_session_mha_trit", func(args []Value) (Value, error) {
		if len(args) < 10 || args[4].Kind != VSpan || args[5].Kind != VSpan || args[6].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_mha_trit(id, qName, kName, vName, x, Kpast, Vpast, dIn, dOut, h [, oName] [, pos] [, base] [, bitplane])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_mha_trit: " + err.Error()}
		}
		toF32 := func(sp []Value) []float32 {
			o := make([]float32, len(sp))
			for i, v := range sp {
				o[i] = float32(v.AsFloat())
			}
			return o
		}
		x := toF32(args[4].Span)
		Kp := toF32(args[5].Span)
		Vp := toF32(args[6].Span)
		dIn, dOut, h := args[7].AsInt(), args[8].AsInt(), args[9].AsInt()
		oName := ""
		pos, base := 0.0, 0.0
		bitplane := false
		if len(args) >= 11 {
			oName = strArg(args[10])
		}
		if len(args) >= 12 {
			pos = args[11].AsFloat()
		}
		if len(args) >= 13 {
			base = args[12].AsFloat()
		}
		if len(args) >= 14 {
			bitplane = args[13].AsFloat() != 0 || args[13].AsInt() != 0
		}
		y, kn, vn, err := s.SessionAttnTrit(strArg(args[1]), strArg(args[2]), strArg(args[3]), oName, x, Kp, Vp, dIn, dOut, h, pos, base, bitplane)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_mha_trit: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(y), f32SliceToSpan(kn), f32SliceToSpan(vn)}), nil
	})
	// weight_session_attn_proj_gqa_trit(id, q,k,v, x, dIn, nHeads, nKVHeads, dh [, bitplane]) → [q,k,v]
	nat("weight_session_attn_proj_gqa_trit", func(args []Value) (Value, error) {
		if len(args) < 9 || args[4].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_attn_proj_gqa_trit(id, q,k,v, x, dIn, nHeads, nKVHeads, dh [, bitplane])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_attn_proj_gqa_trit: " + err.Error()}
		}
		xspan := args[4].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		bitplane := false
		if len(args) >= 10 {
			bitplane = args[9].AsFloat() != 0 || args[9].AsInt() != 0
		}
		q, k, v, err := s.AttnProjGQATrit(strArg(args[1]), strArg(args[2]), strArg(args[3]), x, args[5].AsInt(), args[6].AsInt(), args[7].AsInt(), args[8].AsInt(), bitplane)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_attn_proj_gqa_trit: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(q), f32SliceToSpan(k), f32SliceToSpan(v)}), nil
	})
	// weight_session_gqa_trit(id, q,k,v, x, Kpast, Vpast, dIn, nHeads, nKVHeads, dh [, o] [, pos] [, base] [, bitplane])
	nat("weight_session_gqa_trit", func(args []Value) (Value, error) {
		if len(args) < 11 || args[4].Kind != VSpan || args[5].Kind != VSpan || args[6].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_gqa_trit(id, q,k,v, x, Kpast, Vpast, dIn, nHeads, nKVHeads, dh [, oName] [, pos] [, base] [, bitplane])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_gqa_trit: " + err.Error()}
		}
		toF32 := func(sp []Value) []float32 {
			o := make([]float32, len(sp))
			for i, v := range sp {
				o[i] = float32(v.AsFloat())
			}
			return o
		}
		x := toF32(args[4].Span)
		Kp := toF32(args[5].Span)
		Vp := toF32(args[6].Span)
		dIn := args[7].AsInt()
		nHeads, nKV, dh := args[8].AsInt(), args[9].AsInt(), args[10].AsInt()
		oName := ""
		pos, base := 0.0, 0.0
		bitplane := false
		if len(args) >= 12 {
			oName = strArg(args[11])
		}
		if len(args) >= 13 {
			pos = args[12].AsFloat()
		}
		if len(args) >= 14 {
			base = args[13].AsFloat()
		}
		if len(args) >= 15 {
			bitplane = args[14].AsFloat() != 0 || args[14].AsInt() != 0
		}
		y, kn, vn, err := s.SessionAttnGQATrit(strArg(args[1]), strArg(args[2]), strArg(args[3]), oName, x, Kp, Vp, dIn, nHeads, nKV, dh, pos, base, bitplane)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_gqa_trit: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(y), f32SliceToSpan(kn), f32SliceToSpan(vn)}), nil
	})

	// weight_session_layer_stack(id, wCSV, x, dim [, gainCSV] [, modeCSV] [, tileRows] [, eps])
	// Chains residual fused-linear + RMSNorm layers on one session (v0.90).
	nat("weight_session_layer_stack", func(args []Value) (Value, error) {
		if len(args) < 4 || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_layer_stack(id, wCSV, x, dim [, gainCSV] [, modeCSV] [, tileRows] [, eps])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_layer_stack: " + err.Error()}
		}
		xspan := args[2].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		dim := args[3].AsInt()
		gainCSV, modeCSV := "", ""
		tile, eps := 0, 1e-6
		if len(args) >= 5 {
			gainCSV = strArg(args[4])
		}
		if len(args) >= 6 {
			modeCSV = strArg(args[5])
		}
		if len(args) >= 7 {
			tile = args[6].AsInt()
		}
		if len(args) >= 8 {
			eps = args[7].AsFloat()
		}
		y, err := s.LayerStackCSV(strArg(args[1]), gainCSV, modeCSV, x, dim, tile, eps)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_layer_stack: " + err.Error()}
		}
		return f32SliceToSpan(y), nil
	})
	// weight_session_forward_block(id, wName, x, dim [, gainName] [, mode] [, tileRows] [, eps])
	nat("weight_session_forward_block", func(args []Value) (Value, error) {
		if len(args) < 4 || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_forward_block(id, wName, x, dim [, gainName] [, mode] [, tileRows] [, eps])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_forward_block: " + err.Error()}
		}
		xspan := args[2].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		dim := args[3].AsInt()
		gainName, mode := "", "residual"
		tile, eps := 0, 1e-6
		if len(args) >= 5 {
			gainName = strArg(args[4])
		}
		if len(args) >= 6 {
			mode = strArg(args[5])
		}
		if len(args) >= 7 {
			tile = args[6].AsInt()
		}
		if len(args) >= 8 {
			eps = args[7].AsFloat()
		}
		y, err := s.ForwardBlock(weights.LayerSpec{WName: strArg(args[1]), GainName: gainName, Mode: mode}, x, dim, tile, eps)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_forward_block: " + err.Error()}
		}
		return f32SliceToSpan(y), nil
	})

	// weight_session_layer_stack_trit(id, wCSV, x, dim [, gainCSV] [, modeCSV] [, bitplane] [, eps])
	// Trit-cached residual stack (v1.09): prepack once, reuse packs per layer.
	nat("weight_session_layer_stack_trit", func(args []Value) (Value, error) {
		if len(args) < 4 || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_layer_stack_trit(id, wCSV, x, dim [, gainCSV] [, modeCSV] [, bitplane] [, eps])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_layer_stack_trit: " + err.Error()}
		}
		xspan := args[2].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		dim := args[3].AsInt()
		gainCSV, modeCSV := "", ""
		bitplane := false
		eps := 1e-6
		if len(args) >= 5 {
			gainCSV = strArg(args[4])
		}
		if len(args) >= 6 {
			modeCSV = strArg(args[5])
		}
		if len(args) >= 7 {
			bitplane = args[6].AsFloat() != 0 || args[6].AsInt() != 0
		}
		if len(args) >= 8 {
			eps = args[7].AsFloat()
		}
		y, err := s.LayerStackTritCSV(strArg(args[1]), gainCSV, modeCSV, x, dim, bitplane, eps)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_layer_stack_trit: " + err.Error()}
		}
		return f32SliceToSpan(y), nil
	})
	// weight_session_forward_block_trit(id, wName, x, dim [, gainName] [, mode] [, bitplane] [, eps])
	nat("weight_session_forward_block_trit", func(args []Value) (Value, error) {
		if len(args) < 4 || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_forward_block_trit(id, wName, x, dim [, gainName] [, mode] [, bitplane] [, eps])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_forward_block_trit: " + err.Error()}
		}
		xspan := args[2].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		dim := args[3].AsInt()
		gainName, mode := "", "residual"
		bitplane := false
		eps := 1e-6
		if len(args) >= 5 {
			gainName = strArg(args[4])
		}
		if len(args) >= 6 {
			mode = strArg(args[5])
		}
		if len(args) >= 7 {
			bitplane = args[6].AsFloat() != 0 || args[6].AsInt() != 0
		}
		if len(args) >= 8 {
			eps = args[7].AsFloat()
		}
		y, err := s.ForwardBlockTrit(weights.LayerSpec{WName: strArg(args[1]), GainName: gainName, Mode: mode}, x, dim, bitplane, eps)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_forward_block_trit: " + err.Error()}
		}
		return f32SliceToSpan(y), nil
	})
	// weight_session_prepack_trit_csv(id, wCSV, K) → [built, hits]
	nat("weight_session_prepack_trit_csv", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Void(), &Error{Msg: "weight_session_prepack_trit_csv(id, wCSV, K)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_prepack_trit_csv: " + err.Error()}
		}
		built, hits, err := s.PrepackTritCSV(strArg(args[1]), args[2].AsInt())
		if err != nil {
			return Void(), &Error{Msg: "weight_session_prepack_trit_csv: " + err.Error()}
		}
		return Span([]Value{I32(built), I32(hits)}), nil
	})

	// weight_session_forward_attn_block_trit(id, q,k,v,o, up,down, x, Kpast,Vpast, dim, hidden
	//   [, attnGain] [, ffnGain] [, h] [, pos] [, base] [, bitplane] [, eps]) → [y,K,V]
	// One residual transformer layer on trit-cached QKV/O + FFN (v1.12).
	nat("weight_session_forward_attn_block_trit", func(args []Value) (Value, error) {
		if len(args) < 12 || args[7].Kind != VSpan || args[8].Kind != VSpan || args[9].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_forward_attn_block_trit(id, q,k,v,o, up,down, x, Kpast,Vpast, dim, hidden [, attnGain] [, ffnGain] [, h] [, pos] [, base] [, bitplane] [, eps])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_forward_attn_block_trit: " + err.Error()}
		}
		toF32 := func(sp []Value) []float32 {
			o := make([]float32, len(sp))
			for i, v := range sp {
				o[i] = float32(v.AsFloat())
			}
			return o
		}
		spec := weights.AttnLayerSpec{
			Q: strArg(args[1]), K: strArg(args[2]), V: strArg(args[3]), O: strArg(args[4]),
			Up: strArg(args[5]), Down: strArg(args[6]),
			Hidden: args[11].AsInt(),
		}
		x := toF32(args[7].Span)
		Kp := toF32(args[8].Span)
		Vp := toF32(args[9].Span)
		dim := args[10].AsInt()
		h := 1
		pos, base := 0.0, 0.0
		bitplane := false
		eps := 1e-6
		if len(args) >= 13 {
			spec.AttnGain = strArg(args[12])
		}
		if len(args) >= 14 {
			spec.FFNGain = strArg(args[13])
		}
		if len(args) >= 15 {
			h = args[14].AsInt()
		}
		if len(args) >= 16 {
			pos = args[15].AsFloat()
		}
		if len(args) >= 17 {
			base = args[16].AsFloat()
		}
		if len(args) >= 18 {
			bitplane = args[17].AsFloat() != 0 || args[17].AsInt() != 0
		}
		if len(args) >= 19 {
			eps = args[18].AsFloat()
		}
		y, kn, vn, err := s.ForwardAttnBlockTrit(spec, x, Kp, Vp, dim, h, pos, base, bitplane, eps)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_forward_attn_block_trit: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(y), f32SliceToSpan(kn), f32SliceToSpan(vn)}), nil
	})

	// weight_session_attn_stack_trit(id, qCSV,kCSV,vCSV,oCSV, upCSV,downCSV, x, KpastFlat,VpastFlat,
	//   dim, hidden [, attnGainCSV] [, ffnGainCSV] [, h] [, pos] [, base] [, bitplane] [, eps]) → [y,Kflat,Vflat]
	// Multi-layer residual trit-attn stack with layer-major KV flats (v1.12).
	nat("weight_session_attn_stack_trit", func(args []Value) (Value, error) {
		if len(args) < 13 || args[7].Kind != VSpan || args[8].Kind != VSpan || args[9].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_attn_stack_trit(id, qCSV,kCSV,vCSV,oCSV, upCSV,downCSV, x, KpastFlat,VpastFlat, dim, hidden [, attnGainCSV] [, ffnGainCSV] [, h] [, pos] [, base] [, bitplane] [, eps])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_attn_stack_trit: " + err.Error()}
		}
		toF32 := func(sp []Value) []float32 {
			o := make([]float32, len(sp))
			for i, v := range sp {
				o[i] = float32(v.AsFloat())
			}
			return o
		}
		x := toF32(args[7].Span)
		Kp := toF32(args[8].Span)
		Vp := toF32(args[9].Span)
		dim, hidden := args[10].AsInt(), args[11].AsInt()
		attnGainCSV, ffnGainCSV := "", ""
		h := 1
		pos, base := 0.0, 0.0
		bitplane := false
		eps := 1e-6
		// required through hidden @11; optionals: attnGain, ffnGain, h, pos, base, bitplane, eps
		if len(args) >= 13 {
			attnGainCSV = strArg(args[12])
		}
		if len(args) >= 14 {
			ffnGainCSV = strArg(args[13])
		}
		if len(args) >= 15 {
			h = args[14].AsInt()
		}
		if len(args) >= 16 {
			pos = args[15].AsFloat()
		}
		if len(args) >= 17 {
			base = args[16].AsFloat()
		}
		if len(args) >= 18 {
			bitplane = args[17].AsFloat() != 0 || args[17].AsInt() != 0
		}
		if len(args) >= 19 {
			eps = args[18].AsFloat()
		}
		y, kn, vn, err := s.AttnStackTritCSV(
			strArg(args[1]), strArg(args[2]), strArg(args[3]), strArg(args[4]),
			strArg(args[5]), strArg(args[6]),
			attnGainCSV, ffnGainCSV,
			x, Kp, Vp, dim, hidden, h, pos, base, bitplane, eps,
		)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_attn_stack_trit: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(y), f32SliceToSpan(kn), f32SliceToSpan(vn)}), nil
	})

	// weight_session_forward_gqa_block_trit(id, q,k,v,o, up,down, x, Kpast,Vpast, dim, hidden,
	//   nHeads, nKVHeads, dh [, attnGain] [, ffnGain] [, pos] [, base] [, bitplane] [, eps]) → [y,K,V]
	// One residual GQA+FFN layer on trit-cached projections (v1.14).
	nat("weight_session_forward_gqa_block_trit", func(args []Value) (Value, error) {
		if len(args) < 15 || args[7].Kind != VSpan || args[8].Kind != VSpan || args[9].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_forward_gqa_block_trit(id, q,k,v,o, up,down, x, Kpast,Vpast, dim, hidden, nHeads, nKVHeads, dh [, attnGain] [, ffnGain] [, pos] [, base] [, bitplane] [, eps])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_forward_gqa_block_trit: " + err.Error()}
		}
		toF32 := func(sp []Value) []float32 {
			o := make([]float32, len(sp))
			for i, v := range sp {
				o[i] = float32(v.AsFloat())
			}
			return o
		}
		spec := weights.AttnLayerSpec{
			Q: strArg(args[1]), K: strArg(args[2]), V: strArg(args[3]), O: strArg(args[4]),
			Up: strArg(args[5]), Down: strArg(args[6]),
			Hidden: args[11].AsInt(),
		}
		x := toF32(args[7].Span)
		Kp := toF32(args[8].Span)
		Vp := toF32(args[9].Span)
		dim := args[10].AsInt()
		nHeads, nKV, dh := args[12].AsInt(), args[13].AsInt(), args[14].AsInt()
		pos, base := 0.0, 0.0
		bitplane := false
		eps := 1e-6
		if len(args) >= 16 {
			spec.AttnGain = strArg(args[15])
		}
		if len(args) >= 17 {
			spec.FFNGain = strArg(args[16])
		}
		if len(args) >= 18 {
			pos = args[17].AsFloat()
		}
		if len(args) >= 19 {
			base = args[18].AsFloat()
		}
		if len(args) >= 20 {
			bitplane = args[19].AsFloat() != 0 || args[19].AsInt() != 0
		}
		if len(args) >= 21 {
			eps = args[20].AsFloat()
		}
		y, kn, vn, err := s.ForwardGQABlockTrit(spec, x, Kp, Vp, dim, nHeads, nKV, dh, pos, base, bitplane, eps)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_forward_gqa_block_trit: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(y), f32SliceToSpan(kn), f32SliceToSpan(vn)}), nil
	})

	// weight_session_gqa_stack_trit(id, qCSV,kCSV,vCSV,oCSV, upCSV,downCSV, x, KpastFlat,VpastFlat,
	//   dim, hidden, nHeads, nKVHeads, dh [, attnGainCSV] [, ffnGainCSV] [, pos] [, base] [, bitplane] [, eps])
	// → [y,Kflat,Vflat] multi-layer residual GQA stack with compact layer-major KV flats (v1.14).
	nat("weight_session_gqa_stack_trit", func(args []Value) (Value, error) {
		if len(args) < 15 || args[7].Kind != VSpan || args[8].Kind != VSpan || args[9].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_gqa_stack_trit(id, qCSV,kCSV,vCSV,oCSV, upCSV,downCSV, x, KpastFlat,VpastFlat, dim, hidden, nHeads, nKVHeads, dh [, attnGainCSV] [, ffnGainCSV] [, pos] [, base] [, bitplane] [, eps])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_gqa_stack_trit: " + err.Error()}
		}
		toF32 := func(sp []Value) []float32 {
			o := make([]float32, len(sp))
			for i, v := range sp {
				o[i] = float32(v.AsFloat())
			}
			return o
		}
		x := toF32(args[7].Span)
		Kp := toF32(args[8].Span)
		Vp := toF32(args[9].Span)
		dim, hidden := args[10].AsInt(), args[11].AsInt()
		nHeads, nKV, dh := args[12].AsInt(), args[13].AsInt(), args[14].AsInt()
		attnGainCSV, ffnGainCSV := "", ""
		pos, base := 0.0, 0.0
		bitplane := false
		eps := 1e-6
		if len(args) >= 16 {
			attnGainCSV = strArg(args[15])
		}
		if len(args) >= 17 {
			ffnGainCSV = strArg(args[16])
		}
		if len(args) >= 18 {
			pos = args[17].AsFloat()
		}
		if len(args) >= 19 {
			base = args[18].AsFloat()
		}
		if len(args) >= 20 {
			bitplane = args[19].AsFloat() != 0 || args[19].AsInt() != 0
		}
		if len(args) >= 21 {
			eps = args[20].AsFloat()
		}
		y, kn, vn, err := s.GQAStackTritCSV(
			strArg(args[1]), strArg(args[2]), strArg(args[3]), strArg(args[4]),
			strArg(args[5]), strArg(args[6]),
			attnGainCSV, ffnGainCSV,
			x, Kp, Vp, dim, hidden, nHeads, nKV, dh, pos, base, bitplane, eps,
		)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_gqa_stack_trit: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(y), f32SliceToSpan(kn), f32SliceToSpan(vn)}), nil
	})

	// weight_session_final_norm(id, x, dim [, gainName] [, eps]) → span
	// Optional final RMSNorm before lm_head (v1.14). Empty gain → unit gain.
	nat("weight_session_final_norm", func(args []Value) (Value, error) {
		if len(args) < 3 || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_final_norm(id, x, dim [, gainName] [, eps])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_final_norm: " + err.Error()}
		}
		toF32 := func(sp []Value) []float32 {
			o := make([]float32, len(sp))
			for i, v := range sp {
				o[i] = float32(v.AsFloat())
			}
			return o
		}
		x := toF32(args[1].Span)
		dim := args[2].AsInt()
		gain := ""
		eps := 1e-6
		if len(args) >= 4 {
			gain = strArg(args[3])
		}
		if len(args) >= 5 {
			eps = args[4].AsFloat()
		}
		y, err := s.FinalNorm(gain, x, dim, eps)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_final_norm: " + err.Error()}
		}
		return f32SliceToSpan(y), nil
	})

	// weight_session_forward_gqa_swiglu_block_trit(id, q,k,v,o, gate,up,down, x, Kpast,Vpast,
	//   dim, hidden, nHeads, nKVHeads, dh [, attnGain] [, ffnGain] [, pos] [, base] [, bitplane] [, eps])
	// → [y,K,V] residual GQA + SwiGLU FFN (v1.15).
	nat("weight_session_forward_gqa_swiglu_block_trit", func(args []Value) (Value, error) {
		if len(args) < 16 || args[8].Kind != VSpan || args[9].Kind != VSpan || args[10].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_forward_gqa_swiglu_block_trit(id, q,k,v,o, gate,up,down, x, Kpast,Vpast, dim, hidden, nHeads, nKVHeads, dh [, attnGain] [, ffnGain] [, pos] [, base] [, bitplane] [, eps])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_forward_gqa_swiglu_block_trit: " + err.Error()}
		}
		toF32 := func(sp []Value) []float32 {
			o := make([]float32, len(sp))
			for i, v := range sp {
				o[i] = float32(v.AsFloat())
			}
			return o
		}
		spec := weights.AttnLayerSpec{
			Q: strArg(args[1]), K: strArg(args[2]), V: strArg(args[3]), O: strArg(args[4]),
			Gate: strArg(args[5]), Up: strArg(args[6]), Down: strArg(args[7]),
			Hidden: args[12].AsInt(),
		}
		x := toF32(args[8].Span)
		Kp := toF32(args[9].Span)
		Vp := toF32(args[10].Span)
		dim := args[11].AsInt()
		nHeads, nKV, dh := args[13].AsInt(), args[14].AsInt(), args[15].AsInt()
		pos, base := 0.0, 0.0
		bitplane := false
		eps := 1e-6
		if len(args) >= 17 {
			spec.AttnGain = strArg(args[16])
		}
		if len(args) >= 18 {
			spec.FFNGain = strArg(args[17])
		}
		if len(args) >= 19 {
			pos = args[18].AsFloat()
		}
		if len(args) >= 20 {
			base = args[19].AsFloat()
		}
		if len(args) >= 21 {
			bitplane = args[20].AsFloat() != 0 || args[20].AsInt() != 0
		}
		if len(args) >= 22 {
			eps = args[21].AsFloat()
		}
		y, kn, vn, err := s.ForwardGQASwiGLUBlockTrit(spec, x, Kp, Vp, dim, nHeads, nKV, dh, pos, base, bitplane, eps)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_forward_gqa_swiglu_block_trit: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(y), f32SliceToSpan(kn), f32SliceToSpan(vn)}), nil
	})

	// weight_session_gqa_swiglu_stack_trit(id, qCSV,kCSV,vCSV,oCSV, gateCSV,upCSV,downCSV,
	//   x, KpastFlat,VpastFlat, dim, hidden, nHeads, nKVHeads, dh
	//   [, attnGainCSV] [, ffnGainCSV] [, pos] [, base] [, bitplane] [, eps]) → [y,Kflat,Vflat]
	nat("weight_session_gqa_swiglu_stack_trit", func(args []Value) (Value, error) {
		if len(args) < 16 || args[8].Kind != VSpan || args[9].Kind != VSpan || args[10].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_gqa_swiglu_stack_trit(id, qCSV,kCSV,vCSV,oCSV, gateCSV,upCSV,downCSV, x, KpastFlat,VpastFlat, dim, hidden, nHeads, nKVHeads, dh [, attnGainCSV] [, ffnGainCSV] [, pos] [, base] [, bitplane] [, eps])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_gqa_swiglu_stack_trit: " + err.Error()}
		}
		toF32 := func(sp []Value) []float32 {
			o := make([]float32, len(sp))
			for i, v := range sp {
				o[i] = float32(v.AsFloat())
			}
			return o
		}
		x := toF32(args[8].Span)
		Kp := toF32(args[9].Span)
		Vp := toF32(args[10].Span)
		dim, hidden := args[11].AsInt(), args[12].AsInt()
		nHeads, nKV, dh := args[13].AsInt(), args[14].AsInt(), args[15].AsInt()
		attnGainCSV, ffnGainCSV := "", ""
		pos, base := 0.0, 0.0
		bitplane := false
		eps := 1e-6
		if len(args) >= 17 {
			attnGainCSV = strArg(args[16])
		}
		if len(args) >= 18 {
			ffnGainCSV = strArg(args[17])
		}
		if len(args) >= 19 {
			pos = args[18].AsFloat()
		}
		if len(args) >= 20 {
			base = args[19].AsFloat()
		}
		if len(args) >= 21 {
			bitplane = args[20].AsFloat() != 0 || args[20].AsInt() != 0
		}
		if len(args) >= 22 {
			eps = args[21].AsFloat()
		}
		y, kn, vn, err := s.GQASwiGLUStackTritCSV(
			strArg(args[1]), strArg(args[2]), strArg(args[3]), strArg(args[4]),
			strArg(args[5]), strArg(args[6]), strArg(args[7]),
			attnGainCSV, ffnGainCSV,
			x, Kp, Vp, dim, hidden, nHeads, nKV, dh, pos, base, bitplane, eps,
		)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_gqa_swiglu_stack_trit: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(y), f32SliceToSpan(kn), f32SliceToSpan(vn)}), nil
	})

	// weight_session_gqa_swiglu_prefill_trit(id, qCSV,kCSV,vCSV,oCSV, gateCSV,upCSV,downCSV,
	//   xs, KpastFlat,VpastFlat, dim, hidden, nHeads, nKVHeads, dh
	//   [, attnGainCSV] [, ffnGainCSV] [, startPos] [, base] [, bitplane] [, eps])
	// → [Yflat, Kflat, Vflat, yLast] multi-token causal prefill (v1.15).
	nat("weight_session_gqa_swiglu_prefill_trit", func(args []Value) (Value, error) {
		if len(args) < 16 || args[8].Kind != VSpan || args[9].Kind != VSpan || args[10].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_gqa_swiglu_prefill_trit(id, qCSV,kCSV,vCSV,oCSV, gateCSV,upCSV,downCSV, xs, KpastFlat,VpastFlat, dim, hidden, nHeads, nKVHeads, dh [, attnGainCSV] [, ffnGainCSV] [, startPos] [, base] [, bitplane] [, eps])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_gqa_swiglu_prefill_trit: " + err.Error()}
		}
		toF32 := func(sp []Value) []float32 {
			o := make([]float32, len(sp))
			for i, v := range sp {
				o[i] = float32(v.AsFloat())
			}
			return o
		}
		xs := toF32(args[8].Span)
		Kp := toF32(args[9].Span)
		Vp := toF32(args[10].Span)
		dim, hidden := args[11].AsInt(), args[12].AsInt()
		nHeads, nKV, dh := args[13].AsInt(), args[14].AsInt(), args[15].AsInt()
		attnGainCSV, ffnGainCSV := "", ""
		startPos, base := 0.0, 0.0
		bitplane := false
		eps := 1e-6
		if len(args) >= 17 {
			attnGainCSV = strArg(args[16])
		}
		if len(args) >= 18 {
			ffnGainCSV = strArg(args[17])
		}
		if len(args) >= 19 {
			startPos = args[18].AsFloat()
		}
		if len(args) >= 20 {
			base = args[19].AsFloat()
		}
		if len(args) >= 21 {
			bitplane = args[20].AsFloat() != 0 || args[20].AsInt() != 0
		}
		if len(args) >= 22 {
			eps = args[21].AsFloat()
		}
		Y, kn, vn, yLast, err := s.PrefillGQASwiGLUStackTritCSV(
			strArg(args[1]), strArg(args[2]), strArg(args[3]), strArg(args[4]),
			strArg(args[5]), strArg(args[6]), strArg(args[7]),
			attnGainCSV, ffnGainCSV,
			xs, Kp, Vp, dim, hidden, nHeads, nKV, dh, startPos, base, bitplane, eps,
		)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_gqa_swiglu_prefill_trit: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(Y), f32SliceToSpan(kn), f32SliceToSpan(vn), f32SliceToSpan(yLast)}), nil
	})

	// weight_session_embed_row(id, name, idx, dim) → span[f32] row of emb (v1.16)
	nat("weight_session_embed_row", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return Void(), &Error{Msg: "weight_session_embed_row(id, name, idx, dim)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_embed_row: " + err.Error()}
		}
		y, err := s.EmbedRow(strArg(args[1]), args[2].AsInt(), args[3].AsInt())
		if err != nil {
			return Void(), &Error{Msg: "weight_session_embed_row: " + err.Error()}
		}
		return f32SliceToSpan(y), nil
	})

	// weight_session_logits_tied(id, embName, h, dim) → span[f32] (v1.16 f32)
	nat("weight_session_logits_tied", func(args []Value) (Value, error) {
		if len(args) < 4 || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_logits_tied(id, embName, h, dim)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_logits_tied: " + err.Error()}
		}
		h := make([]float32, len(args[2].Span))
		for i, v := range args[2].Span {
			h[i] = float32(v.AsFloat())
		}
		y, err := s.LogitsTied(strArg(args[1]), h, args[3].AsInt())
		if err != nil {
			return Void(), &Error{Msg: "weight_session_logits_tied: " + err.Error()}
		}
		return f32SliceToSpan(y), nil
	})

	// weight_session_logits_tied_trit(id, embName, h, dim [, bitplane]) → span[f32] (v1.16)
	nat("weight_session_logits_tied_trit", func(args []Value) (Value, error) {
		if len(args) < 4 || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_logits_tied_trit(id, embName, h, dim [, bitplane])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_logits_tied_trit: " + err.Error()}
		}
		h := make([]float32, len(args[2].Span))
		for i, v := range args[2].Span {
			h[i] = float32(v.AsFloat())
		}
		bitplane := false
		if len(args) >= 5 {
			bitplane = args[4].AsFloat() != 0 || args[4].AsInt() != 0
		}
		y, err := s.LogitsTiedTrit(strArg(args[1]), h, args[3].AsInt(), bitplane)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_logits_tied_trit: " + err.Error()}
		}
		return f32SliceToSpan(y), nil
	})

	// weight_session_gqa_swiglu_stack_alibi(id, qCSV,kCSV,vCSV,oCSV, gateCSV,upCSV,downCSV,
	//   x, KpastFlat,VpastFlat, dim, hidden, nHeads, nKVHeads, dh
	//   [, attnGainCSV] [, ffnGainCSV] [, pos] [, bitplane] [, eps]) → [y,Kflat,Vflat]
	nat("weight_session_gqa_swiglu_stack_alibi", func(args []Value) (Value, error) {
		if len(args) < 16 || args[8].Kind != VSpan || args[9].Kind != VSpan || args[10].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_gqa_swiglu_stack_alibi(id, qCSV,kCSV,vCSV,oCSV, gateCSV,upCSV,downCSV, x, KpastFlat,VpastFlat, dim, hidden, nHeads, nKVHeads, dh [, attnGainCSV] [, ffnGainCSV] [, pos] [, bitplane] [, eps])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_gqa_swiglu_stack_alibi: " + err.Error()}
		}
		toF32 := func(sp []Value) []float32 {
			o := make([]float32, len(sp))
			for i, v := range sp {
				o[i] = float32(v.AsFloat())
			}
			return o
		}
		x := toF32(args[8].Span)
		Kp := toF32(args[9].Span)
		Vp := toF32(args[10].Span)
		dim, hidden := args[11].AsInt(), args[12].AsInt()
		nHeads, nKV, dh := args[13].AsInt(), args[14].AsInt(), args[15].AsInt()
		attnGainCSV, ffnGainCSV := "", ""
		pos := 0.0
		bitplane := false
		eps := 1e-6
		if len(args) >= 17 {
			attnGainCSV = strArg(args[16])
		}
		if len(args) >= 18 {
			ffnGainCSV = strArg(args[17])
		}
		if len(args) >= 19 {
			pos = args[18].AsFloat()
		}
		if len(args) >= 20 {
			bitplane = args[19].AsFloat() != 0 || args[19].AsInt() != 0
		}
		if len(args) >= 21 {
			eps = args[20].AsFloat()
		}
		y, kn, vn, err := s.GQASwiGLUStackTritAlibiCSV(
			strArg(args[1]), strArg(args[2]), strArg(args[3]), strArg(args[4]),
			strArg(args[5]), strArg(args[6]), strArg(args[7]),
			attnGainCSV, ffnGainCSV,
			x, Kp, Vp, dim, hidden, nHeads, nKV, dh, pos, bitplane, eps,
		)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_gqa_swiglu_stack_alibi: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(y), f32SliceToSpan(kn), f32SliceToSpan(vn)}), nil
	})

	// weight_session_gqa_swiglu_prefill_alibi(... same as stack alibi, xs instead of x) → [Y,K,V,yLast]
	nat("weight_session_gqa_swiglu_prefill_alibi", func(args []Value) (Value, error) {
		if len(args) < 16 || args[8].Kind != VSpan || args[9].Kind != VSpan || args[10].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_gqa_swiglu_prefill_alibi(id, qCSV,kCSV,vCSV,oCSV, gateCSV,upCSV,downCSV, xs, KpastFlat,VpastFlat, dim, hidden, nHeads, nKVHeads, dh [, attnGainCSV] [, ffnGainCSV] [, startPos] [, bitplane] [, eps])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_gqa_swiglu_prefill_alibi: " + err.Error()}
		}
		toF32 := func(sp []Value) []float32 {
			o := make([]float32, len(sp))
			for i, v := range sp {
				o[i] = float32(v.AsFloat())
			}
			return o
		}
		xs := toF32(args[8].Span)
		Kp := toF32(args[9].Span)
		Vp := toF32(args[10].Span)
		dim, hidden := args[11].AsInt(), args[12].AsInt()
		nHeads, nKV, dh := args[13].AsInt(), args[14].AsInt(), args[15].AsInt()
		attnGainCSV, ffnGainCSV := "", ""
		startPos := 0.0
		bitplane := false
		eps := 1e-6
		if len(args) >= 17 {
			attnGainCSV = strArg(args[16])
		}
		if len(args) >= 18 {
			ffnGainCSV = strArg(args[17])
		}
		if len(args) >= 19 {
			startPos = args[18].AsFloat()
		}
		if len(args) >= 20 {
			bitplane = args[19].AsFloat() != 0 || args[19].AsInt() != 0
		}
		if len(args) >= 21 {
			eps = args[20].AsFloat()
		}
		Y, kn, vn, yLast, err := s.PrefillGQASwiGLUStackTritAlibiCSV(
			strArg(args[1]), strArg(args[2]), strArg(args[3]), strArg(args[4]),
			strArg(args[5]), strArg(args[6]), strArg(args[7]),
			attnGainCSV, ffnGainCSV,
			xs, Kp, Vp, dim, hidden, nHeads, nKV, dh, startPos, bitplane, eps,
		)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_gqa_swiglu_prefill_alibi: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(Y), f32SliceToSpan(kn), f32SliceToSpan(vn), f32SliceToSpan(yLast)}), nil
	})

	// weight_session_gqa_swiglu_stack_alibi_window(... + window after eps) → [y,K,V]
	nat("weight_session_gqa_swiglu_stack_alibi_window", func(args []Value) (Value, error) {
		if len(args) < 16 || args[8].Kind != VSpan || args[9].Kind != VSpan || args[10].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_gqa_swiglu_stack_alibi_window(id, qCSV,kCSV,vCSV,oCSV, gateCSV,upCSV,downCSV, x, KpastFlat,VpastFlat, dim, hidden, nHeads, nKVHeads, dh [, attnGainCSV] [, ffnGainCSV] [, pos] [, bitplane] [, eps] [, window])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_gqa_swiglu_stack_alibi_window: " + err.Error()}
		}
		toF32 := func(sp []Value) []float32 {
			o := make([]float32, len(sp))
			for i, v := range sp {
				o[i] = float32(v.AsFloat())
			}
			return o
		}
		x := toF32(args[8].Span)
		Kp := toF32(args[9].Span)
		Vp := toF32(args[10].Span)
		dim, hidden := args[11].AsInt(), args[12].AsInt()
		nHeads, nKV, dh := args[13].AsInt(), args[14].AsInt(), args[15].AsInt()
		attnGainCSV, ffnGainCSV := "", ""
		pos := 0.0
		bitplane := false
		eps := 1e-6
		window := 0
		if len(args) >= 17 {
			attnGainCSV = strArg(args[16])
		}
		if len(args) >= 18 {
			ffnGainCSV = strArg(args[17])
		}
		if len(args) >= 19 {
			pos = args[18].AsFloat()
		}
		if len(args) >= 20 {
			bitplane = args[19].AsFloat() != 0 || args[19].AsInt() != 0
		}
		if len(args) >= 21 {
			eps = args[20].AsFloat()
		}
		if len(args) >= 22 {
			window = args[21].AsInt()
		}
		y, kn, vn, err := s.GQASwiGLUStackTritAlibiWindowCSV(
			strArg(args[1]), strArg(args[2]), strArg(args[3]), strArg(args[4]),
			strArg(args[5]), strArg(args[6]), strArg(args[7]),
			attnGainCSV, ffnGainCSV,
			x, Kp, Vp, dim, hidden, nHeads, nKV, dh, pos, bitplane, eps, window,
		)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_gqa_swiglu_stack_alibi_window: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(y), f32SliceToSpan(kn), f32SliceToSpan(vn)}), nil
	})

	// weight_session_gqa_swiglu_prefill_alibi_window → [Y,K,V,yLast]
	nat("weight_session_gqa_swiglu_prefill_alibi_window", func(args []Value) (Value, error) {
		if len(args) < 16 || args[8].Kind != VSpan || args[9].Kind != VSpan || args[10].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_gqa_swiglu_prefill_alibi_window(id, qCSV,kCSV,vCSV,oCSV, gateCSV,upCSV,downCSV, xs, KpastFlat,VpastFlat, dim, hidden, nHeads, nKVHeads, dh [, attnGainCSV] [, ffnGainCSV] [, startPos] [, bitplane] [, eps] [, window])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_gqa_swiglu_prefill_alibi_window: " + err.Error()}
		}
		toF32 := func(sp []Value) []float32 {
			o := make([]float32, len(sp))
			for i, v := range sp {
				o[i] = float32(v.AsFloat())
			}
			return o
		}
		xs := toF32(args[8].Span)
		Kp := toF32(args[9].Span)
		Vp := toF32(args[10].Span)
		dim, hidden := args[11].AsInt(), args[12].AsInt()
		nHeads, nKV, dh := args[13].AsInt(), args[14].AsInt(), args[15].AsInt()
		attnGainCSV, ffnGainCSV := "", ""
		startPos := 0.0
		bitplane := false
		eps := 1e-6
		window := 0
		if len(args) >= 17 {
			attnGainCSV = strArg(args[16])
		}
		if len(args) >= 18 {
			ffnGainCSV = strArg(args[17])
		}
		if len(args) >= 19 {
			startPos = args[18].AsFloat()
		}
		if len(args) >= 20 {
			bitplane = args[19].AsFloat() != 0 || args[19].AsInt() != 0
		}
		if len(args) >= 21 {
			eps = args[20].AsFloat()
		}
		if len(args) >= 22 {
			window = args[21].AsInt()
		}
		Y, kn, vn, yLast, err := s.PrefillGQASwiGLUStackTritAlibiWindowCSV(
			strArg(args[1]), strArg(args[2]), strArg(args[3]), strArg(args[4]),
			strArg(args[5]), strArg(args[6]), strArg(args[7]),
			attnGainCSV, ffnGainCSV,
			xs, Kp, Vp, dim, hidden, nHeads, nKV, dh, startPos, bitplane, eps, window,
		)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_gqa_swiglu_prefill_alibi_window: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(Y), f32SliceToSpan(kn), f32SliceToSpan(vn), f32SliceToSpan(yLast)}), nil
	})

	// weight_session_linear_fused(id, name, x, M, K [, tileRows]) → span[f32] (v0.89 block-stream)
	nat("weight_session_linear_fused", func(args []Value) (Value, error) {
		if len(args) < 5 || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_linear_fused(id, name, x, M, K [, tileRows])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_linear_fused: " + err.Error()}
		}
		xspan := args[2].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		M, K := args[3].AsInt(), args[4].AsInt()
		tile := 0
		if len(args) >= 6 {
			tile = args[5].AsInt()
		}
		y, err := s.LinearFused(strArg(args[1]), x, M, K, tile)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_linear_fused: " + err.Error()}
		}
		return f32SliceToSpan(y), nil
	})
	// weight_session_linear_fused_tile(id, name, x, K, row0, nRows)
	nat("weight_session_linear_fused_tile", func(args []Value) (Value, error) {
		if len(args) < 6 || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_linear_fused_tile(id, name, x, K, row0, nRows)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_linear_fused_tile: " + err.Error()}
		}
		xspan := args[2].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		K, row0, nRows := args[3].AsInt(), args[4].AsInt(), args[5].AsInt()
		y, err := s.LinearRowsFused(strArg(args[1]), x, K, row0, nRows)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_linear_fused_tile: " + err.Error()}
		}
		return f32SliceToSpan(y), nil
	})
	// load_gguf_linear_fused(path, name, x, M, K [, tileRows])
	nat("load_gguf_linear_fused", func(args []Value) (Value, error) {
		if len(args) < 5 || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "load_gguf_linear_fused(path, name, x, M, K [, tileRows])"}
		}
		xspan := args[2].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		M, K := args[3].AsInt(), args[4].AsInt()
		tile := 0
		if len(args) >= 6 {
			tile = args[5].AsInt()
		}
		y, err := weights.LinearGGUFFused(strArg(args[0]), strArg(args[1]), x, M, K, tile)
		if err != nil {
			return Void(), &Error{Msg: "load_gguf_linear_fused: " + err.Error()}
		}
		return f32SliceToSpan(y), nil
	})

	// weight_session_linear(id, name, x, M, K [, tileRows]) → span[f32] (v0.88 tiled matvec)
	nat("weight_session_linear", func(args []Value) (Value, error) {
		if len(args) < 5 || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_linear(id, name, x, M, K [, tileRows])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_linear: " + err.Error()}
		}
		xspan := args[2].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		M, K := args[3].AsInt(), args[4].AsInt()
		tile := 0
		if len(args) >= 6 {
			tile = args[5].AsInt()
		}
		y, err := s.Linear(strArg(args[1]), x, M, K, tile)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_linear: " + err.Error()}
		}
		return f32SliceToSpan(y), nil
	})
	// weight_session_linear_tile(id, name, x, K, row0, nRows) → span[f32]
	nat("weight_session_linear_tile", func(args []Value) (Value, error) {
		if len(args) < 6 || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_linear_tile(id, name, x, K, row0, nRows)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_linear_tile: " + err.Error()}
		}
		xspan := args[2].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		K, row0, nRows := args[3].AsInt(), args[4].AsInt(), args[5].AsInt()
		y, err := s.LinearRows(strArg(args[1]), x, K, row0, nRows)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_linear_tile: " + err.Error()}
		}
		return f32SliceToSpan(y), nil
	})
	// weight_session_linear_trit(id, name, x, M, K [, tileRows [, bitplane]])
	// → [y:span f32, scale:f32]  session load → absmean trit → matvec (v1.06)
	nat("weight_session_linear_trit", func(args []Value) (Value, error) {
		if len(args) < 5 || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_linear_trit(id, name, x, M, K [, tileRows [, bitplane]])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_linear_trit: " + err.Error()}
		}
		xspan := args[2].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		M, K := args[3].AsInt(), args[4].AsInt()
		tile := 0
		bitplane := false
		if len(args) >= 6 {
			tile = args[5].AsInt()
		}
		if len(args) >= 7 {
			bitplane = args[6].AsFloat() != 0 || args[6].AsInt() != 0
		}
		y, scale, err := s.LinearTrit(strArg(args[1]), x, M, K, tile, bitplane)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_linear_trit: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(y), F32(float64(scale))}), nil
	})
	// weight_session_linear_trit_tile(id, name, x, K, row0, nRows [, bitplane]) → [y, scale]
	nat("weight_session_linear_trit_tile", func(args []Value) (Value, error) {
		if len(args) < 6 || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_linear_trit_tile(id, name, x, K, row0, nRows [, bitplane])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_linear_trit_tile: " + err.Error()}
		}
		xspan := args[2].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		K, row0, nRows := args[3].AsInt(), args[4].AsInt(), args[5].AsInt()
		bitplane := false
		if len(args) >= 7 {
			bitplane = args[6].AsFloat() != 0 || args[6].AsInt() != 0
		}
		y, scale, err := s.LinearTritRows(strArg(args[1]), x, K, row0, nRows, bitplane)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_linear_trit_tile: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(y), F32(float64(scale))}), nil
	})
	// load_gguf_linear_trit(path, name, x, M, K [, tileRows [, bitplane]]) → [y, scale]
	nat("load_gguf_linear_trit", func(args []Value) (Value, error) {
		if len(args) < 5 || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "load_gguf_linear_trit(path, name, x, M, K [, tileRows [, bitplane]])"}
		}
		xspan := args[2].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		M, K := args[3].AsInt(), args[4].AsInt()
		tile := 0
		bitplane := false
		if len(args) >= 6 {
			tile = args[5].AsInt()
		}
		if len(args) >= 7 {
			bitplane = args[6].AsFloat() != 0 || args[6].AsInt() != 0
		}
		y, scale, err := weights.LinearTritGGUF(strArg(args[0]), strArg(args[1]), x, M, K, tile, bitplane)
		if err != nil {
			return Void(), &Error{Msg: "load_gguf_linear_trit: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(y), F32(float64(scale))}), nil
	})
	// weight_session_prepack_trit(id, name, K) → i32 built (1=new, 0=cache hit)  (v1.07)
	nat("weight_session_prepack_trit", func(args []Value) (Value, error) {
		if len(args) < 3 {
			return Void(), &Error{Msg: "weight_session_prepack_trit(id, name, K)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_prepack_trit: " + err.Error()}
		}
		built, err := s.PrepackTrit(strArg(args[1]), args[2].AsInt())
		if err != nil {
			return Void(), &Error{Msg: "weight_session_prepack_trit: " + err.Error()}
		}
		if built {
			return I32(1), nil
		}
		return I32(0), nil
	})
	// weight_session_has_trit_pack(id, name) → i32 1/0
	nat("weight_session_has_trit_pack", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "weight_session_has_trit_pack(id, name)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_has_trit_pack: " + err.Error()}
		}
		if s.HasTritPack(strArg(args[1])) {
			return I32(1), nil
		}
		return I32(0), nil
	})
	// weight_session_trit_pack_info(id, name) → [M, K, scale, hit]
	nat("weight_session_trit_pack_info", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "weight_session_trit_pack_info(id, name)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_trit_pack_info: " + err.Error()}
		}
		M, K, sc, ok := s.TritPackInfo(strArg(args[1]))
		hit := 0
		if ok {
			hit = 1
		}
		return Span([]Value{I32(M), I32(K), F32(float64(sc)), I32(hit)}), nil
	})
	// weight_session_drop_trit_pack(id, name) → void
	nat("weight_session_drop_trit_pack", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "weight_session_drop_trit_pack(id, name)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_drop_trit_pack: " + err.Error()}
		}
		s.DropTritPack(strArg(args[1]))
		return Void(), nil
	})
	// weight_session_linear_trit_cached(id, name, x, M, K [, bitplane]) → [y, scale]
	// Uses / builds whole-tensor prepack; bitplane uses packed planes without re-pack.
	nat("weight_session_linear_trit_cached", func(args []Value) (Value, error) {
		if len(args) < 5 || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "weight_session_linear_trit_cached(id, name, x, M, K [, bitplane])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_linear_trit_cached: " + err.Error()}
		}
		xspan := args[2].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		M, K := args[3].AsInt(), args[4].AsInt()
		bitplane := false
		if len(args) >= 6 {
			bitplane = args[5].AsFloat() != 0 || args[5].AsInt() != 0
		}
		y, scale, err := s.LinearTritCached(strArg(args[1]), x, M, K, bitplane)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_linear_trit_cached: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(y), F32(float64(scale))}), nil
	})
	// load_gguf_linear(path, name, x, M, K [, tileRows]) one-shot
	nat("load_gguf_linear", func(args []Value) (Value, error) {
		if len(args) < 5 || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "load_gguf_linear(path, name, x, M, K [, tileRows])"}
		}
		xspan := args[2].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		M, K := args[3].AsInt(), args[4].AsInt()
		tile := 0
		if len(args) >= 6 {
			tile = args[5].AsInt()
		}
		y, err := weights.LinearGGUF(strArg(args[0]), strArg(args[1]), x, M, K, tile)
		if err != nil {
			return Void(), &Error{Msg: "load_gguf_linear: " + err.Error()}
		}
		return f32SliceToSpan(y), nil
	})

	// weight_session_window(id, name, offset, count) → span[f32] (v0.87 partial decode)
	nat("weight_session_window", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return Void(), &Error{Msg: "weight_session_window(id, name, offset, count)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "weight_session_window: " + err.Error()}
		}
		off := args[2].AsInt()
		cnt := args[3].AsInt()
		f32s, err := s.LoadF32Window(strArg(args[1]), off, cnt)
		if err != nil {
			return Void(), &Error{Msg: "weight_session_window: " + err.Error()}
		}
		return f32SliceToSpan(f32s), nil
	})
	// load_gguf_window(path, name, offset, count) — one-shot open
	nat("load_gguf_window", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return Void(), &Error{Msg: "load_gguf_window(path, name, offset, count)"}
		}
		f, err := weights.OpenGGUF(strArg(args[0]))
		if err != nil {
			return Void(), &Error{Msg: "load_gguf_window: " + err.Error()}
		}
		defer f.Close()
		f32s, err := f.LoadF32Window(strArg(args[1]), args[2].AsInt(), args[3].AsInt())
		if err != nil {
			return Void(), &Error{Msg: "load_gguf_window: " + err.Error()}
		}
		return f32SliceToSpan(f32s), nil
	})
	// load_safetensors_window(path, name, offset, count)
	nat("load_safetensors_window", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return Void(), &Error{Msg: "load_safetensors_window(path, name, offset, count)"}
		}
		f, err := weights.OpenSafeTensors(strArg(args[0]))
		if err != nil {
			return Void(), &Error{Msg: "load_safetensors_window: " + err.Error()}
		}
		defer f.Close()
		f32s, err := f.LoadF32Window(strArg(args[1]), args[2].AsInt(), args[3].AsInt())
		if err != nil {
			return Void(), &Error{Msg: "load_safetensors_window: " + err.Error()}
		}
		return f32SliceToSpan(f32s), nil
	})

	// weight_file_mapped(path) → i32 1 if OS mmap succeeded for open probe
	nat("weight_file_mapped", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "weight_file_mapped(path)"}
		}
		path := strArg(args[0])
		var mapped bool
		switch {
		case strings.HasSuffix(strings.ToLower(path), ".gguf"):
			f, err := weights.OpenGGUF(path)
			if err != nil {
				return Void(), &Error{Msg: "weight_file_mapped: " + err.Error()}
			}
			mapped = f.IsMapped()
			_ = f.Close()
		case strings.HasSuffix(strings.ToLower(path), ".safetensors"):
			f, err := weights.OpenSafeTensors(path)
			if err != nil {
				return Void(), &Error{Msg: "weight_file_mapped: " + err.Error()}
			}
			mapped = f.IsMapped()
			_ = f.Close()
		default:
			if f, err := weights.OpenGGUF(path); err == nil {
				mapped = f.IsMapped()
				_ = f.Close()
			} else if f, err := weights.OpenSafeTensors(path); err == nil {
				mapped = f.IsMapped()
				_ = f.Close()
			} else {
				return Void(), &Error{Msg: "weight_file_mapped: cannot open " + path}
			}
		}
		if mapped {
			return I32(1), nil
		}
		return I32(0), nil
	})

	// --- CUDA path (v1.02) — device probe + portable f32 / trit kernels ---
	// On hosts without NVIDIA driver/devices, cuda_available()=0 and kernels
	// still run via the package's CPU fallback (same numeric contract).

	// ---- v1.04 char/token table tokenizer + text I/O ----
	// open_vocab(path) → i32 vocab session id
	nat("open_vocab", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "open_vocab(path)"}
		}
		v, err := weights.OpenVocab(strArg(args[0]))
		if err != nil {
			return Void(), &Error{Msg: "open_vocab: " + err.Error()}
		}
		return I32(int(v.ID)), nil
	})
	// close_vocab(id) → void (idempotent)
	nat("close_vocab", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "close_vocab(id)"}
		}
		_ = weights.CloseVocab(int32(args[0].AsInt()))
		return Void(), nil
	})
	// vocab_size(id) → i32
	nat("vocab_size", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "vocab_size(id)"}
		}
		v, err := weights.GetVocab(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "vocab_size: " + err.Error()}
		}
		return I32(v.Size()), nil
	})
	// vocab_encode(id, prompt) → span[i32]
	nat("vocab_encode", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "vocab_encode(id, prompt)"}
		}
		v, err := weights.GetVocab(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "vocab_encode: " + err.Error()}
		}
		ids := v.Encode(strArg(args[1]))
		out := make([]Value, len(ids))
		for i, id := range ids {
			out[i] = I32(int(id))
		}
		return Span(out), nil
	})
	// vocab_decode(id, ids) → str  (ids = span[i32])
	nat("vocab_decode", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "vocab_decode(id, ids)"}
		}
		v, err := weights.GetVocab(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "vocab_decode: " + err.Error()}
		}
		var ids []int32
		if args[1].Kind == VSpan {
			ids = make([]int32, len(args[1].Span))
			for i, x := range args[1].Span {
				ids[i] = int32(x.AsInt())
			}
		} else {
			// single id
			ids = []int32{int32(args[1].AsInt())}
		}
		return Str(v.Decode(ids)), nil
	})
	// vocab_token(id, tok_id) → str
	nat("vocab_token", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "vocab_token(id, tok_id)"}
		}
		v, err := weights.GetVocab(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "vocab_token: " + err.Error()}
		}
		return Str(v.Token(args[1].AsInt())), nil
	})
	// vocab_id(id, tok) → i32 (-1 if missing)
	nat("vocab_id", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "vocab_id(id, tok)"}
		}
		v, err := weights.GetVocab(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "vocab_id: " + err.Error()}
		}
		return I32(v.IDOf(strArg(args[1]))), nil
	})
	// load_text(path) → str
	nat("load_text", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "load_text(path)"}
		}
		b, err := os.ReadFile(strArg(args[0]))
		if err != nil {
			return Void(), &Error{Msg: "load_text: " + err.Error()}
		}
		return Str(string(b)), nil
	})
	// load_vocab_text(text) → i32  (inline vocab, no file)
	nat("load_vocab_text", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "load_vocab_text(text)"}
		}
		v, err := weights.LoadVocabText("<inline>", strArg(args[0]))
		if err != nil {
			return Void(), &Error{Msg: "load_vocab_text: " + err.Error()}
		}
		return I32(int(v.ID)), nil
	})

	// ---- v1.05 BPE sidecar tokenizer (local vocab+merges, no network) ----
	nat("open_bpe", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "open_bpe(vocab_path [, merges_path])"}
		}
		vocabPath := strArg(args[0])
		mergesPath := ""
		if len(args) >= 2 {
			mergesPath = strArg(args[1])
		}
		t, err := weights.OpenBPE(vocabPath, mergesPath)
		if err != nil {
			return Void(), &Error{Msg: "open_bpe: " + err.Error()}
		}
		return I32(int(t.ID)), nil
	})
	nat("close_bpe", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "close_bpe(id)"}
		}
		_ = weights.CloseBPE(int32(args[0].AsInt()))
		return Void(), nil
	})
	nat("bpe_vocab_size", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "bpe_vocab_size(id)"}
		}
		t, err := weights.GetBPE(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "bpe_vocab_size: " + err.Error()}
		}
		return I32(t.VocabSize()), nil
	})
	nat("bpe_merge_count", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "bpe_merge_count(id)"}
		}
		t, err := weights.GetBPE(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "bpe_merge_count: " + err.Error()}
		}
		return I32(t.MergeCount()), nil
	})
	nat("bpe_encode", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "bpe_encode(id, prompt)"}
		}
		t, err := weights.GetBPE(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "bpe_encode: " + err.Error()}
		}
		ids := t.Encode(strArg(args[1]))
		out := make([]Value, len(ids))
		for i, id := range ids {
			out[i] = I32(int(id))
		}
		return Span(out), nil
	})
	nat("bpe_decode", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "bpe_decode(id, ids)"}
		}
		t, err := weights.GetBPE(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "bpe_decode: " + err.Error()}
		}
		var ids []int32
		if args[1].Kind == VSpan {
			ids = make([]int32, len(args[1].Span))
			for i, x := range args[1].Span {
				ids[i] = int32(x.AsInt())
			}
		} else {
			ids = []int32{int32(args[1].AsInt())}
		}
		return Str(t.Decode(ids)), nil
	})
	nat("bpe_token", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "bpe_token(id, tok_id)"}
		}
		t, err := weights.GetBPE(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "bpe_token: " + err.Error()}
		}
		return Str(t.Token(args[1].AsInt())), nil
	})

	nat("cuda_available", func(args []Value) (Value, error) {
		if cuda.Available() {
			return I32(1), nil
		}
		return I32(0), nil
	})
	nat("cuda_device_count", func(args []Value) (Value, error) {
		return I32(cuda.DeviceCount()), nil
	})
	nat("cuda_backend", func(args []Value) (Value, error) {
		return Str(cuda.Backend()), nil
	})
	nat("cuda_reason", func(args []Value) (Value, error) {
		return Str(cuda.Reason()), nil
	})
	nat("cuda_device_name", func(args []Value) (Value, error) {
		idx := 0
		if len(args) >= 1 {
			idx = args[0].AsInt()
		}
		return Str(cuda.DeviceName(idx)), nil
	})
	// cuda_linear_f32(W, x, M, K) → f32 span[M]
	nat("cuda_linear_f32", func(args []Value) (Value, error) {
		if len(args) < 4 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "cuda_linear_f32(W, x, M, K)"}
		}
		M, K := args[2].AsInt(), args[3].AsInt()
		Wsp, xsp := args[0].Span, args[1].Span
		W := make([]float32, len(Wsp))
		for i, v := range Wsp {
			W[i] = float32(v.AsFloat())
		}
		x := make([]float32, len(xsp))
		for i, v := range xsp {
			x[i] = float32(v.AsFloat())
		}
		y, err := cuda.LinearF32(W, x, M, K)
		if err != nil {
			return Void(), &Error{Msg: err.Error()}
		}
		return f32SliceToSpan(y), nil
	})
	// cuda_gemm_f32(A, B, M, K, N) → f32 span[M*N]
	nat("cuda_gemm_f32", func(args []Value) (Value, error) {
		if len(args) < 5 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "cuda_gemm_f32(A, B, M, K, N)"}
		}
		M, K, N := args[2].AsInt(), args[3].AsInt(), args[4].AsInt()
		Asp, Bsp := args[0].Span, args[1].Span
		A := make([]float32, len(Asp))
		for i, v := range Asp {
			A[i] = float32(v.AsFloat())
		}
		B := make([]float32, len(Bsp))
		for i, v := range Bsp {
			B[i] = float32(v.AsFloat())
		}
		C, err := cuda.GemmF32(A, B, M, K, N)
		if err != nil {
			return Void(), &Error{Msg: err.Error()}
		}
		return f32SliceToSpan(C), nil
	})
	// cuda_linear_trit(W_trit, x, M, K, scale) → f32 span[M]
	nat("cuda_linear_trit", func(args []Value) (Value, error) {
		if len(args) < 5 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "cuda_linear_trit(W, x, M, K, scale)"}
		}
		M, K := args[2].AsInt(), args[3].AsInt()
		scale := float32(args[4].AsFloat())
		Wspan := args[0].Span
		if len(Wspan) < M*K {
			return Void(), &Error{Msg: "cuda_linear_trit: dimension mismatch"}
		}
		W := make([]int32, M*K)
		for i := 0; i < M*K; i++ {
			W[i] = int32(Wspan[i].AsInt())
		}
		xsp := args[1].Span
		x := make([]float32, len(xsp))
		for i, v := range xsp {
			x[i] = float32(v.AsFloat())
		}
		y, err := cuda.LinearTrit(W, x, M, K, scale)
		if err != nil {
			return Void(), &Error{Msg: err.Error()}
		}
		return f32SliceToSpan(y), nil
	})


	// --- CUDA PTX layer (v1.03) ---
	nat("cuda_ptxas_available", func(args []Value) (Value, error) {
		if cuda.PTXASAvailable() {
			return I32(1), nil
		}
		return I32(0), nil
	})
	nat("cuda_ptx_arch", func(args []Value) (Value, error) {
		return Str(cuda.PTXArch()), nil
	})
	// cuda_ptx_source(kind) kind: linear_f32 | gemm_f32 | linear_trit | dequant_linear_q4_0 | dequant_linear_q8_0
	nat("cuda_ptx_source", func(args []Value) (Value, error) {
		kind := "linear_f32"
		if len(args) >= 1 {
			kind = strArg(args[0])
		}
		var mod cuda.PTXModule
		switch kind {
		case "linear_f32", "cuda_linear_f32", cuda.PTXLinearF32Entry:
			mod = cuda.EmitLinearF32PTX("")
		case "gemm_f32", "cuda_gemm_f32", cuda.PTXGemmF32Entry:
			mod = cuda.EmitGemmF32PTX("")
		case "linear_trit", "cuda_linear_trit", cuda.PTXLinearTritEntry:
			mod = cuda.EmitLinearTritPTX("")
		case "dequant_linear_q4_0", "q4_0", cuda.PTXDequantLinearQ4_0Entry:
			mod = cuda.EmitDequantLinearQ4_0PTX("")
		case "dequant_linear_q8_0", "q8_0", cuda.PTXDequantLinearQ8_0Entry:
			mod = cuda.EmitDequantLinearQ8_0PTX("")
		default:
			return Void(), &Error{Msg: "cuda_ptx_source(kind): want linear_f32|gemm_f32|linear_trit|dequant_linear_q4_0|dequant_linear_q8_0"}
		}
		return Str(mod.Source), nil
	})
	nat("cuda_ptx_bytes", func(args []Value) (Value, error) {
		kind := "linear_f32"
		if len(args) >= 1 {
			kind = strArg(args[0])
		}
		st := cuda.PlanLaunch(kind)
		return I32(st.PTXBytes), nil
	})
	nat("cuda_cubin_bytes", func(args []Value) (Value, error) {
		kind := "linear_f32"
		if len(args) >= 1 {
			kind = strArg(args[0])
		}
		st := cuda.PlanLaunch(kind)
		return I32(st.CubinBytes), nil
	})
	// cuda_plan_launch(kind) → span[would(i32), backend(str), reason(str), entry(str), ptx(i32), cubin(i32), ptxas(i32)]
	nat("cuda_plan_launch", func(args []Value) (Value, error) {
		kind := "linear_f32"
		if len(args) >= 1 {
			kind = strArg(args[0])
		}
		st := cuda.PlanLaunch(kind)
		would := 0
		if st.WouldLaunch {
			would = 1
		}
		ptxas := 0
		if st.PTXAS {
			ptxas = 1
		}
		return Span([]Value{
			I32(would),
			Str(st.Backend),
			Str(st.Reason),
			Str(st.Entry),
			I32(st.PTXBytes),
			I32(st.CubinBytes),
			I32(ptxas),
		}), nil
	})
	// cuda_compile_ptx(kind) → i32 cubin byte count (0 if ptxas missing / fail)
	nat("cuda_compile_ptx", func(args []Value) (Value, error) {
		kind := "linear_f32"
		if len(args) >= 1 {
			kind = strArg(args[0])
		}
		var mod cuda.PTXModule
		switch kind {
		case "linear_f32", "cuda_linear_f32", cuda.PTXLinearF32Entry:
			mod = cuda.EmitLinearF32PTX("")
		case "gemm_f32", "cuda_gemm_f32", cuda.PTXGemmF32Entry:
			mod = cuda.EmitGemmF32PTX("")
		case "linear_trit", "cuda_linear_trit", cuda.PTXLinearTritEntry:
			mod = cuda.EmitLinearTritPTX("")
		case "dequant_linear_q4_0", "q4_0", cuda.PTXDequantLinearQ4_0Entry:
			mod = cuda.EmitDequantLinearQ4_0PTX("")
		case "dequant_linear_q8_0", "q8_0", cuda.PTXDequantLinearQ8_0Entry:
			mod = cuda.EmitDequantLinearQ8_0PTX("")
		default:
			return Void(), &Error{Msg: "cuda_compile_ptx(kind)"}
		}
		cub, err := cuda.CompilePTX(mod)
		if err != nil {
			return I32(0), nil // soft fail — host may lack ptxas
		}
		return I32(len(cub.Bytes)), nil
	})
	// cuda_alloc_f32(n) → i32 handle
	nat("cuda_alloc_f32", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "cuda_alloc_f32(n)"}
		}
		b, err := cuda.AllocF32(args[0].AsInt())
		if err != nil {
			return Void(), &Error{Msg: err.Error()}
		}
		return I32(int(cuda.RegisterBuffer(b))), nil
	})
	nat("cuda_free", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "cuda_free(id)"}
		}
		cuda.FreeBuffer(int32(args[0].AsInt()))
		return Void(), nil
	})
	// cuda_copy_hto_d(id, span) → i32 n
	nat("cuda_copy_hto_d", func(args []Value) (Value, error) {
		if len(args) < 2 || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "cuda_copy_hto_d(id, span)"}
		}
		b, err := cuda.GetBuffer(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: err.Error()}
		}
		sp := args[1].Span
		src := make([]float32, len(sp))
		for i, v := range sp {
			src[i] = float32(v.AsFloat())
		}
		if err := b.CopyHtoD(src); err != nil {
			return Void(), &Error{Msg: err.Error()}
		}
		return I32(len(src)), nil
	})
	// cuda_copy_dto_h(id, n) → f32 span
	nat("cuda_copy_dto_h", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "cuda_copy_dto_h(id [, n])"}
		}
		b, err := cuda.GetBuffer(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: err.Error()}
		}
		n := b.N
		if len(args) >= 2 {
			n = args[1].AsInt()
		}
		if n > b.N {
			n = b.N
		}
		dst := make([]float32, n)
		if err := b.CopyDtoH(dst); err != nil {
			return Void(), &Error{Msg: err.Error()}
		}
		return f32SliceToSpan(dst), nil
	})
	nat("cuda_buffer_len", func(args []Value) (Value, error) {
		if len(args) < 1 {
			return Void(), &Error{Msg: "cuda_buffer_len(id)"}
		}
		b, err := cuda.GetBuffer(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: err.Error()}
		}
		return I32(b.N), nil
	})

	// ---- CUDA GGUF inference backend (v1.20) ----
	// Connects weight sessions to CUDA dequant-fused matvec kernels so that
	// GGUF models with Q4_0 / Q8_0 quantisation can run inference on GPU
	// (or CPU fallback) without a separate dequant pass.

	// cuda_session_linear(id, name, x, M, K) → span[f32]
	// Loads tensor `name` from weight session `id`, dispatches to the fused
	// dequant-linear CUDA kernel for Q4_0/Q8_0, or decodes to f32 + linear for others.
	nat("cuda_session_linear", func(args []Value) (Value, error) {
		if len(args) < 5 || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "cuda_session_linear(id, name, x, M, K)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "cuda_session_linear: " + err.Error()}
		}
		xspan := args[2].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		M, K := args[3].AsInt(), args[4].AsInt()
		y, err := s.CUDALinear(strArg(args[1]), x, M, K)
		if err != nil {
			return Void(), &Error{Msg: "cuda_session_linear: " + err.Error()}
		}
		return f32SliceToSpan(y), nil
	})

	// cuda_session_linear_trit(id, name, x, M, K) → [y:span f32, scale:f32]
	// Decodes W from GGUF, absmean-quantizes to trits, runs matvec via CUDA trit kernel.
	nat("cuda_session_linear_trit", func(args []Value) (Value, error) {
		if len(args) < 5 || args[2].Kind != VSpan {
			return Void(), &Error{Msg: "cuda_session_linear_trit(id, name, x, M, K)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "cuda_session_linear_trit: " + err.Error()}
		}
		xspan := args[2].Span
		x := make([]float32, len(xspan))
		for i, v := range xspan {
			x[i] = float32(v.AsFloat())
		}
		M, K := args[3].AsInt(), args[4].AsInt()
		y, scale, err := s.CUDALinearTrit(strArg(args[1]), x, M, K)
		if err != nil {
			return Void(), &Error{Msg: "cuda_session_linear_trit: " + err.Error()}
		}
		return Span([]Value{f32SliceToSpan(y), F32(float64(scale))}), nil
	})

	// cuda_session_tensor_qtype(id, name) → span[type_code:i32, n_elem:i32]
	// Inspects the GGML quant type and element count for a named tensor.
	nat("cuda_session_tensor_qtype", func(args []Value) (Value, error) {
		if len(args) < 2 {
			return Void(), &Error{Msg: "cuda_session_tensor_qtype(id, name)"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "cuda_session_tensor_qtype: " + err.Error()}
		}
		qt, nElem, err := s.TensorQType(strArg(args[1]))
		if err != nil {
			return Void(), &Error{Msg: "cuda_session_tensor_qtype: " + err.Error()}
		}
		return Span([]Value{I32(int(qt)), I32(nElem)}), nil
	})

	// cuda_dequant_linear_q4_0(W_bytes_span, x, M, K) → span[f32]
	// CPU-fallback dequant-fused Q4_0 linear when called with Trin span data.
	nat("cuda_dequant_linear_q4_0", func(args []Value) (Value, error) {
		if len(args) < 4 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "cuda_dequant_linear_q4_0(W_bytes, x, M, K)"}
		}
		M, K := args[2].AsInt(), args[3].AsInt()
		if M <= 0 || K <= 0 || K%32 != 0 {
			return Void(), &Error{Msg: "cuda_dequant_linear_q4_0: bad M/K (K must be multiple of 32)"}
		}
		if len(args[1].Span) < K {
			return Void(), &Error{Msg: "cuda_dequant_linear_q4_0: x span shorter than K"}
		}
		wb := make([]byte, len(args[0].Span))
		for i, v := range args[0].Span {
			wb[i] = byte(v.AsInt())
		}
		x := make([]float32, len(args[1].Span))
		for i, v := range args[1].Span {
			x[i] = float32(v.AsFloat())
		}
		y, err := cuda.DequantLinearQ4_0(wb, x, M, K)
		if err != nil {
			return Void(), &Error{Msg: err.Error()}
		}
		return f32SliceToSpan(y), nil
	})

	// cuda_dequant_linear_q8_0(W_bytes_span, x, M, K) → span[f32]
	nat("cuda_dequant_linear_q8_0", func(args []Value) (Value, error) {
		if len(args) < 4 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "cuda_dequant_linear_q8_0(W_bytes, x, M, K)"}
		}
		M, K := args[2].AsInt(), args[3].AsInt()
		if M <= 0 || K <= 0 || K%32 != 0 {
			return Void(), &Error{Msg: "cuda_dequant_linear_q8_0: bad M/K (K must be multiple of 32)"}
		}
		if len(args[1].Span) < K {
			return Void(), &Error{Msg: "cuda_dequant_linear_q8_0: x span shorter than K"}
		}
		wb := make([]byte, len(args[0].Span))
		for i, v := range args[0].Span {
			wb[i] = byte(v.AsInt())
		}
		x := make([]float32, len(args[1].Span))
		for i, v := range args[1].Span {
			x[i] = float32(v.AsFloat())
		}
		y, err := cuda.DequantLinearQ8_0(wb, x, M, K)
		if err != nil {
			return Void(), &Error{Msg: err.Error()}
		}
		return f32SliceToSpan(y), nil
	})

	// ---- Starlight CNN / BalancedStarlightDetector (v1.19) ----
	// relu_f32(x) → span[f32] elementwise ReLU
	nat("relu_f32", func(args []Value) (Value, error) {
		if len(args) < 1 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "relu_f32(span)"}
		}
		return f32SliceToSpan(weights.ReLU(spanToF32(args[0]))), nil
	})
	// sigmoid_f32(x) → span[f32] elementwise sigmoid
	nat("sigmoid_f32", func(args []Value) (Value, error) {
		if len(args) < 1 || args[0].Kind != VSpan {
			return Void(), &Error{Msg: "sigmoid_f32(span)"}
		}
		return f32SliceToSpan(weights.Sigmoid(spanToF32(args[0]))), nil
	})
	// linear_bias(W, x, b, M, K) → y = W@x + b; b may be empty span (no bias)
	nat("linear_bias", func(args []Value) (Value, error) {
		if len(args) < 5 || args[0].Kind != VSpan || args[1].Kind != VSpan {
			return Void(), &Error{Msg: "linear_bias(W, x, b, M, K)"}
		}
		W := spanToF32(args[0])
		x := spanToF32(args[1])
		var b []float32
		if args[2].Kind == VSpan && len(args[2].Span) > 0 {
			b = spanToF32(args[2])
		}
		M, K := args[3].AsInt(), args[4].AsInt()
		y, err := weights.LinearBias(W, x, b, M, K)
		if err != nil {
			return Void(), &Error{Msg: "linear_bias: " + err.Error()}
		}
		return f32SliceToSpan(y), nil
	})
	// starlight_forward(id, pixel, meta, alpha, lsb, palette, palette_lsb, format, content, H, W [, meta_weight])
	// → span[f32] ConcatOutputs: stego(1)+method(5)+bit_order(3)+embedding(64)=73
	nat("starlight_forward", func(args []Value) (Value, error) {
		if len(args) < 11 {
			return Void(), &Error{Msg: "starlight_forward(id, pixel, meta, alpha, lsb, palette, palette_lsb, format, content, H, W [, meta_weight])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "starlight_forward: " + err.Error()}
		}
		// args: id, pixel, meta, alpha, lsb, palette, palette_lsb, format, content, H, W [, mw]
		// spans occupy indices 1..8; H/W are 9/10
		for i := 1; i <= 8; i++ {
			if args[i].Kind != VSpan {
				return Void(), &Error{Msg: "starlight_forward: args 1..8 must be span[f32]"}
			}
		}
		mw := float32(0.3)
		if len(args) >= 12 {
			mw = float32(args[11].AsFloat())
		}
		in := weights.StarlightInputs{
			Pixel:           spanToF32(args[1]),
			Meta:            spanToF32(args[2]),
			Alpha:           spanToF32(args[3]),
			LSB:             spanToF32(args[4]),
			Palette:         spanToF32(args[5]),
			PaletteLSB:      spanToF32(args[6]),
			FormatFeatures:  spanToF32(args[7]),
			ContentFeatures: spanToF32(args[8]),
			H:               args[9].AsInt(),
			W:               args[10].AsInt(),
			MetaWeight:      mw,
		}
		out, err := weights.StarlightForwardSession(s, in)
		if err != nil {
			return Void(), &Error{Msg: "starlight_forward: " + err.Error()}
		}
		return f32SliceToSpan(out.ConcatOutputs()), nil
	})
	// starlight_demo_forward(id, H, W, seed [, meta_weight]) — LCG inputs + full CNN
	nat("starlight_demo_forward", func(args []Value) (Value, error) {
		if len(args) < 4 {
			return Void(), &Error{Msg: "starlight_demo_forward(id, H, W, seed [, meta_weight])"}
		}
		s, err := weights.GetSession(int32(args[0].AsInt()))
		if err != nil {
			return Void(), &Error{Msg: "starlight_demo_forward: " + err.Error()}
		}
		H, W := args[1].AsInt(), args[2].AsInt()
		if H <= 0 || W <= 0 {
			return Void(), &Error{Msg: "starlight_demo_forward: H,W must be > 0"}
		}
		seed := uint64(args[3].AsInt())
		mw := float32(0.3)
		if len(args) >= 5 {
			mw = float32(args[4].AsFloat())
		}
		g := &weights.StarlightLCG{S: seed}
		in := weights.StarlightInputs{
			Pixel:           g.Fill(3 * H * W),
			Meta:            g.Fill(2048),
			Alpha:           g.Fill(2 * H * W),
			LSB:             g.Fill(3 * H * W),
			Palette:         g.Fill(768),
			PaletteLSB:      g.Fill(1 * H * W),
			FormatFeatures:  g.Fill(6),
			ContentFeatures: g.Fill(6),
			H:               H,
			W:               W,
			MetaWeight:      mw,
		}
		out, err := weights.StarlightForwardSession(s, in)
		if err != nil {
			return Void(), &Error{Msg: "starlight_demo_forward: " + err.Error()}
		}
		return f32SliceToSpan(out.ConcatOutputs()), nil
	})

	_ = math.Abs
}

// spanToF32 converts a VSpan of numeric values to []float32.
func spanToF32(v Value) []float32 {
	if v.Kind != VSpan {
		return nil
	}
	out := make([]float32, len(v.Span))
	for i, x := range v.Span {
		out[i] = float32(x.AsFloat())
	}
	return out
}

func strArg(v Value) string {
	if v.Kind == VStr {
		return v.Str
	}
	return v.String()
}

func f32SliceToSpan(xs []float32) Value {
	out := make([]Value, len(xs))
	for i, x := range xs {
		out[i] = F32(float64(x))
	}
	return Span(out)
}

// package-level deterministic LCG for sample_logits / sample_topk (v0.82)
var trinRNG uint64 = 0x9e3779b97f4a7c15

func trinRNGNext() float64 {
	// xorshift64*
	x := trinRNG
	if x == 0 {
		x = 0x9e3779b97f4a7c15
	}
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	trinRNG = x
	// map to [0,1)
	return float64(x%1000000) / 1000000.0
}

func softMuxArgs(args []Value) (w, pure, each, dy float64) {
	dy = 1
	if len(args) > 0 {
		w = args[0].AsFloat()
	}
	if len(args) > 1 {
		pure = args[1].AsFloat()
	}
	if len(args) > 2 {
		each = args[2].AsFloat()
	}
	if len(args) > 3 {
		dy = args[3].AsFloat()
	}
	if w < 0 {
		w = 0
	}
	if w > 1 {
		w = 1
	}
	return
}

func spanOp(args []Value, op func(a, b float64) float64) (Value, error) {
	if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
		return Void(), &Error{Msg: "span op expects two spans"}
	}
	n := min(len(args[0].Span), len(args[1].Span))
	out := make([]Value, n)
	for i := 0; i < n; i++ {
		out[i] = F32(op(args[0].Span[i].AsFloat(), args[1].Span[i].AsFloat()))
	}
	return Span(out), nil
}

func clampTrit(n int) int {
	if n < -1 {
		return -1
	}
	if n > 1 {
		return 1
	}
	return n
}

// SimdLaneWidth is the preferred multiword kernel width (4×32-bit words = 128 trits).
const SimdLaneWidth = 128

func tritGemm(args []Value, bitplane, simd bool) (Value, error) {
	if len(args) < 5 || args[0].Kind != VSpan || args[1].Kind != VSpan {
		return Void(), &Error{Msg: "trit_gemm(A,B,M,K,N)"}
	}
	A, B := args[0].Span, args[1].Span
	M, K, N := args[2].AsInt(), args[3].AsInt(), args[4].AsInt()
	out := make([]Value, M*N)
	if bitplane {
		// Dense word packing: pack each K-vector into 32-trit words, product via AND/OR, popcount.
		// simd path uses 128-lane multiword chunks (4×u32) then residual words.
		for i := 0; i < M; i++ {
			row := make([]Value, K)
			for t := 0; t < K; t++ {
				row[t] = A[i*K+t]
			}
			ap, an, _ := packTritWords(row)
			for j := 0; j < N; j++ {
				col := make([]Value, K)
				for t := 0; t < K; t++ {
					col[t] = B[t*N+j]
				}
				bp, bn, _ := packTritWords(col)
				var d int
				if simd {
					d = tritDotPlanesSIMD(ap, an, bp, bn, K)
				} else {
					d = tritDotPlanes(ap, an, bp, bn, K)
				}
				out[i*N+j] = I32(d)
			}
		}
		return Span(out), nil
	}
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			acc := 0
			for t := 0; t < K; t++ {
				av := clampTrit(A[i*K+t].AsInt())
				bv := clampTrit(B[t*N+j].AsInt())
				acc += av * bv
			}
			out[i*N+j] = I32(acc)
		}
	}
	return Span(out), nil
}

// packTritWords packs 32 trits per uint32 word (bit 0 = lowest index).
func packTritWords(xs []Value) (pos, neg []uint32, n int) {
	n = len(xs)
	nw := (n + 31) / 32
	if nw == 0 {
		return []uint32{}, []uint32{}, 0
	}
	pos = make([]uint32, nw)
	neg = make([]uint32, nw)
	for i, x := range xs {
		t := clampTrit(x.AsInt())
		w, b := i/32, uint(i%32)
		if t > 0 {
			pos[w] |= 1 << b
		} else if t < 0 {
			neg[w] |= 1 << b
		}
	}
	return pos, neg, n
}

func unpackTritWords(pos, neg []uint32, n int) []int {
	if n < 0 {
		n = 0
	}
	out := make([]int, n)
	for i := 0; i < n; i++ {
		w, b := i/32, uint(i%32)
		var p, g uint32
		if w < len(pos) {
			p = (pos[w] >> b) & 1
		}
		if w < len(neg) {
			g = (neg[w] >> b) & 1
		}
		if p != 0 && g == 0 {
			out[i] = 1
		} else if g != 0 && p == 0 {
			out[i] = -1
		}
	}
	return out
}

func tritDotPlanes(ap, an, bp, bn []uint32, n int) int {
	if n <= 0 {
		return 0
	}
	nw := (n + 31) / 32
	acc := 0
	for w := 0; w < nw; w++ {
		acc += tritWordProduct(ap, an, bp, bn, w, n, nw)
	}
	return acc
}

// tritDotPlanesSIMD is semantic-identical to tritDotPlanes but walks words in
// 4-wide (128-lane) then 8-wide-friendly residual groups for multiword planes.
func tritDotPlanesSIMD(ap, an, bp, bn []uint32, n int) int {
	if n <= 0 {
		return 0
	}
	nw := (n + 31) / 32
	acc := 0
	w := 0
	// Primary: 4×u32 = 128 trit lanes.
	for ; w+4 <= nw; w += 4 {
		acc += tritWordProduct(ap, an, bp, bn, w, n, nw)
		acc += tritWordProduct(ap, an, bp, bn, w+1, n, nw)
		acc += tritWordProduct(ap, an, bp, bn, w+2, n, nw)
		acc += tritWordProduct(ap, an, bp, bn, w+3, n, nw)
	}
	// Secondary residual pair (toward 256-lane blocks when remaining ≥ 2).
	for ; w+2 <= nw; w += 2 {
		acc += tritWordProduct(ap, an, bp, bn, w, n, nw)
		acc += tritWordProduct(ap, an, bp, bn, w+1, n, nw)
	}
	for ; w < nw; w++ {
		acc += tritWordProduct(ap, an, bp, bn, w, n, nw)
	}
	return acc
}

func tritWordProduct(ap, an, bp, bn []uint32, w, n, nw int) int {
	var mask uint32 = 0xffffffff
	if w == nw-1 && n%32 != 0 {
		mask = (uint32(1) << uint(n%32)) - 1
	}
	var awp, awn, bwp, bwn uint32
	if w < len(ap) {
		awp = ap[w] & mask
	}
	if w < len(an) {
		awn = an[w] & mask
	}
	if w < len(bp) {
		bwp = bp[w] & mask
	}
	if w < len(bn) {
		bwn = bn[w] & mask
	}
	pos := (awp & bwp) | (awn & bwn)
	neg := (awp & bwn) | (awn & bwp)
	return bits.OnesCount32(pos) - bits.OnesCount32(neg)
}

func toU32s(xs []Value) []uint32 {
	out := make([]uint32, len(xs))
	for i, x := range xs {
		out[i] = uint32(x.AsInt())
	}
	return out
}

func u32Span(ws []uint32) Value {
	out := make([]Value, len(ws))
	for i, w := range ws {
		out[i] = I32(int(int32(w)))
	}
	return Span(out)
}

func i32Span(ws []uint32) Value { return u32Span(ws) }

func tritSpan(ts []int) Value {
	out := make([]Value, len(ts))
	for i, t := range ts {
		out[i] = Trit(t)
	}
	return Span(out)
}

func planeBinop(args []Value, op func(a, b uint32) uint32) (Value, error) {
	if len(args) < 2 || args[0].Kind != VSpan || args[1].Kind != VSpan {
		return Void(), &Error{Msg: "plane op expects two word spans"}
	}
	a, b := toU32s(args[0].Span), toU32s(args[1].Span)
	n := min(len(a), len(b))
	out := make([]uint32, n)
	for i := 0; i < n; i++ {
		out[i] = op(a[i], b[i])
	}
	return u32Span(out), nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
