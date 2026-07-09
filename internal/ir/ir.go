// Package ir defines TrinIR — a flat, SSA-like intermediate form for
// kernels and soft-mux graphs. Suitable for text dump (emit-ir) and
// future backends.
package ir

import (
	"fmt"
	"strings"

	"github.com/ericchien/trin/internal/ast"
)

// Op is a TrinIR opcode.
type Op string

const (
	OpConstI32  Op = "const.i32"
	OpConstF32  Op = "const.f32"
	OpConstUnit Op = "const.unit"
	OpConstStr  Op = "const.str"
	OpConstBit  Op = "const.bit"
	OpAdd       Op = "add"
	OpSub       Op = "sub"
	OpMul       Op = "mul"
	OpDiv       Op = "div"
	OpCmp       Op = "cmp" // Args[0] unused; Meta holds op
	OpSoftMux   Op = "soft_mux"
	OpCall      Op = "call"
	OpLoad      Op = "load" // local / name
	OpStore     Op = "store"
	OpSpan      Op = "span"
	OpIndex     Op = "index"
	OpRet       Op = "ret"
	OpUnaryNeg  Op = "neg"
	OpUnaryNot  Op = "not"
)

// Value is an SSA temporary or named ref.
type Value struct {
	ID   int
	Name string // optional
}

// Instr is one IR instruction.
type Instr struct {
	Dst  *Value // nil for side-effect-only
	Op   Op
	Args []Value
	// immediates
	I64  int64
	F64  float64
	Str  string
	Meta string // cmp op, call name, load name, etc.
}

// Func is a TrinIR function.
type Func struct {
	Name   string
	Params []string
	Body   []Instr
	NextID int
}

func (f *Func) fresh(hint string) Value {
	id := f.NextID
	f.NextID++
	name := hint
	if name == "" {
		name = "v"
	}
	return Value{ID: id, Name: fmt.Sprintf("%s%d", name, id)}
}

// Module holds all functions.
type Module struct {
	Funcs []*Func
}

// String dumps human-readable TrinIR.
func (m *Module) String() string {
	var b strings.Builder
	b.WriteString("; TrinIR v0.6\n")
	for _, fn := range m.Funcs {
		fmt.Fprintf(&b, "func @%s(", fn.Name)
		for i, p := range fn.Params {
			if i > 0 {
				b.WriteString(", ")
			}
			fmt.Fprintf(&b, "%%%s", p)
		}
		b.WriteString(") {\n")
		for _, in := range fn.Body {
			b.WriteString("  ")
			b.WriteString(formatInstr(in))
			b.WriteByte('\n')
		}
		b.WriteString("}\n\n")
	}
	return b.String()
}

func formatInstr(in Instr) string {
	var b strings.Builder
	if in.Dst != nil {
		fmt.Fprintf(&b, "%%%s = ", in.Dst.Name)
	}
	b.WriteString(string(in.Op))
	switch in.Op {
	case OpConstI32, OpConstBit:
		fmt.Fprintf(&b, " %d", in.I64)
	case OpConstF32, OpConstUnit:
		fmt.Fprintf(&b, " %g", in.F64)
	case OpConstStr:
		fmt.Fprintf(&b, " %q", in.Str)
	case OpCall, OpLoad, OpStore, OpCmp:
		if in.Meta != "" {
			fmt.Fprintf(&b, " @%s", in.Meta)
		}
		for _, a := range in.Args {
			fmt.Fprintf(&b, " %%%s", a.Name)
		}
	default:
		for _, a := range in.Args {
			fmt.Fprintf(&b, " %%%s", a.Name)
		}
	}
	return b.String()
}

// Emit lowers an AST program to TrinIR (best-effort for examples).
func Emit(prog *ast.Program) (*Module, error) {
	m := &Module{}
	for _, it := range prog.Items {
		fn, ok := it.(*ast.FnDef)
		if !ok {
			continue
		}
		f := &Func{Name: fn.Name}
		for _, p := range fn.Params {
			f.Params = append(f.Params, p.Name)
		}
		g := &builder{fn: f, names: map[string]Value{}}
		for _, p := range fn.Params {
			v := f.fresh(p.Name)
			g.names[p.Name] = v
			// params are implicit — no load instr; name maps to virtual
			// rewrite: treat as load for dump clarity
			f.Body = append(f.Body, Instr{Dst: &v, Op: OpLoad, Meta: p.Name})
			g.names[p.Name] = v
		}
		if fn.Body != nil {
			for _, st := range fn.Body.Stmts {
				if err := g.emitStmt(st); err != nil {
					return nil, err
				}
			}
			if fn.Body.Expr != nil {
				v, err := g.emitExpr(fn.Body.Expr)
				if err != nil {
					return nil, err
				}
				f.Body = append(f.Body, Instr{Op: OpRet, Args: []Value{v}})
			} else {
				f.Body = append(f.Body, Instr{Op: OpRet})
			}
		}
		m.Funcs = append(m.Funcs, f)
	}
	return m, nil
}

type builder struct {
	fn    *Func
	names map[string]Value
}

func (g *builder) emitStmt(st ast.Stmt) error {
	switch s := st.(type) {
	case *ast.LetStmt:
		v, err := g.emitExpr(s.Value)
		if err != nil {
			return err
		}
		g.names[s.Name] = v
		// store for readability
		g.fn.Body = append(g.fn.Body, Instr{Op: OpStore, Meta: s.Name, Args: []Value{v}})
	case *ast.ExprStmt:
		_, err := g.emitExpr(s.Expr)
		return err
	case *ast.ReturnStmt:
		if s.Value == nil {
			g.fn.Body = append(g.fn.Body, Instr{Op: OpRet})
			return nil
		}
		v, err := g.emitExpr(s.Value)
		if err != nil {
			return err
		}
		g.fn.Body = append(g.fn.Body, Instr{Op: OpRet, Args: []Value{v}})
	}
	return nil
}

func (g *builder) emitExpr(e ast.Expr) (Value, error) {
	switch n := e.(type) {
	case *ast.Literal:
		dst := g.fn.fresh("c")
		in := Instr{Dst: &dst}
		switch n.Kind {
		case "int":
			in.Op = OpConstI32
			in.I64 = int64(n.Value.(int))
		case "float":
			in.Op = OpConstF32
			in.F64 = n.Value.(float64)
		case "str":
			in.Op = OpConstStr
			in.Str = n.Value.(string)
		case "bool":
			in.Op = OpConstBit
			if n.Value.(bool) {
				in.I64 = 1
			}
		}
		g.fn.Body = append(g.fn.Body, in)
		return dst, nil
	case *ast.Var:
		if v, ok := g.names[n.Name]; ok {
			return v, nil
		}
		dst := g.fn.fresh("ld")
		g.fn.Body = append(g.fn.Body, Instr{Dst: &dst, Op: OpLoad, Meta: n.Name})
		g.names[n.Name] = dst
		return dst, nil
	case *ast.Unary:
		x, err := g.emitExpr(n.Expr)
		if err != nil {
			return Value{}, err
		}
		dst := g.fn.fresh("u")
		op := OpUnaryNeg
		if n.Op == "!" {
			op = OpUnaryNot
		}
		g.fn.Body = append(g.fn.Body, Instr{Dst: &dst, Op: op, Args: []Value{x}})
		return dst, nil
	case *ast.Binary:
		l, err := g.emitExpr(n.Left)
		if err != nil {
			return Value{}, err
		}
		r, err := g.emitExpr(n.Right)
		if err != nil {
			return Value{}, err
		}
		dst := g.fn.fresh("b")
		in := Instr{Dst: &dst, Args: []Value{l, r}}
		switch n.Op {
		case "+":
			in.Op = OpAdd
		case "-":
			in.Op = OpSub
		case "*":
			in.Op = OpMul
		case "/":
			in.Op = OpDiv
		default:
			in.Op = OpCmp
			in.Meta = n.Op
		}
		g.fn.Body = append(g.fn.Body, in)
		return dst, nil
	case *ast.Ternary:
		gate, err := g.emitExpr(n.Gate)
		if err != nil {
			return Value{}, err
		}
		pure, err := g.emitExpr(n.Pure)
		if err != nil {
			return Value{}, err
		}
		each, err := g.emitExpr(n.Each)
		if err != nil {
			return Value{}, err
		}
		dst := g.fn.fresh("mux")
		g.fn.Body = append(g.fn.Body, Instr{Dst: &dst, Op: OpSoftMux, Args: []Value{gate, pure, each}})
		return dst, nil
	case *ast.SpanLit:
		args := make([]Value, 0, len(n.Elems))
		for _, el := range n.Elems {
			v, err := g.emitExpr(el)
			if err != nil {
				return Value{}, err
			}
			args = append(args, v)
		}
		dst := g.fn.fresh("sp")
		g.fn.Body = append(g.fn.Body, Instr{Dst: &dst, Op: OpSpan, Args: args})
		return dst, nil
	case *ast.Index:
		t, err := g.emitExpr(n.Target)
		if err != nil {
			return Value{}, err
		}
		ix, err := g.emitExpr(n.Index)
		if err != nil {
			return Value{}, err
		}
		dst := g.fn.fresh("ix")
		g.fn.Body = append(g.fn.Body, Instr{Dst: &dst, Op: OpIndex, Args: []Value{t, ix}})
		return dst, nil
	case *ast.Call:
		v, ok := n.Callee.(*ast.Var)
		if !ok {
			return Value{}, fmt.Errorf("ir: only named calls")
		}
		// soft_mux call form → OpSoftMux
		if v.Name == "soft_mux" && len(n.Args) == 3 {
			a0, err := g.emitExpr(n.Args[0])
			if err != nil {
				return Value{}, err
			}
			a1, err := g.emitExpr(n.Args[1])
			if err != nil {
				return Value{}, err
			}
			a2, err := g.emitExpr(n.Args[2])
			if err != nil {
				return Value{}, err
			}
			dst := g.fn.fresh("mux")
			g.fn.Body = append(g.fn.Body, Instr{Dst: &dst, Op: OpSoftMux, Args: []Value{a0, a1, a2}})
			return dst, nil
		}
		args := make([]Value, 0, len(n.Args))
		for _, a := range n.Args {
			// skip lambdas as opaque
			if _, ok := a.(*ast.Lambda); ok {
				// represent as const.str for dump
				dst := g.fn.fresh("lam")
				g.fn.Body = append(g.fn.Body, Instr{Dst: &dst, Op: OpConstStr, Str: "<lambda>"})
				args = append(args, dst)
				continue
			}
			av, err := g.emitExpr(a)
			if err != nil {
				return Value{}, err
			}
			args = append(args, av)
		}
		dst := g.fn.fresh("c")
		g.fn.Body = append(g.fn.Body, Instr{Dst: &dst, Op: OpCall, Meta: v.Name, Args: args})
		return dst, nil
	case *ast.Lambda:
		dst := g.fn.fresh("lam")
		g.fn.Body = append(g.fn.Body, Instr{Dst: &dst, Op: OpConstStr, Str: "<lambda>"})
		return dst, nil
	case *ast.Block:
		return Value{}, fmt.Errorf("ir: nested block not supported")
	}
	return Value{}, fmt.Errorf("ir: unsupported expr %T", e)
}
