// Package ad implements reverse-mode AD for Trin f32 / span graphs.
//
// Soft multiplex: y = w * pure + (1-w) * each, w ∈ [0,1]
// Local VJP (seed dy):
//
//	dw = dy * (pure - each)
//	dpure = dy * w
//	deach = dy * (1 - w)
//
// Tape supports +, -, *, /, unary-, soft_mux, sum, and dot.
// Span producers span_add / scale / map_mul / map(lambda) / zip_with(lambda)
// (n-ary, N>=2) / scan(lambda) / filter(lambda) expand to element arithmetic
// nodes when consumed by sum/dot (no multi-output tape nodes needed).
// reduce/fold unrolls a left fold of a scalar lambda body into the same tape.
// filter keep-set is decided at tape-build from forward values; grads only
// flow through kept element values (hard threshold on pred).
// Grad(out, seed) reverse-sweeps the whole graph.
package ad

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/ericchien/trin/internal/ast"
)

// SoftMux forward.
func SoftMux(w, pure, each float64) float64 {
	if w < 0 {
		w = 0
	}
	if w > 1 {
		w = 1
	}
	return w*pure + (1-w)*each
}

// SoftMuxVJP returns (dw, dpure, deach) for a scalar soft-mux node.
func SoftMuxVJP(w, pure, each, dy float64) (dw, dpure, deach float64) {
	if w < 0 {
		w = 0
	}
	if w > 1 {
		w = 1
	}
	dw = dy * (pure - each)
	dpure = dy * w
	deach = dy * (1 - w)
	return
}

// Tape is a reverse-mode tape for f64 arithmetic + soft_mux + sum/dot.
type Tape struct {
	nodes []node
}

type kind int

const (
	kConst kind = iota
	kAdd
	kSub
	kMul
	kDiv
	kNeg
	kSoftMux
	kSum // args = element parent ids
	kDot // args = a0..a{n-1}, b0..b{n-1}; c = n
)

type node struct {
	k       kind
	v       float64
	a, b, c int // parents; c used by soft_mux as each / by-dot as n
	args    []int
}

// Const pushes a leaf and returns its id.
func (t *Tape) Const(v float64) int {
	t.nodes = append(t.nodes, node{k: kConst, v: v})
	return len(t.nodes) - 1
}

// Add pushes a+b.
func (t *Tape) Add(a, b int) int {
	v := t.nodes[a].v + t.nodes[b].v
	t.nodes = append(t.nodes, node{k: kAdd, v: v, a: a, b: b})
	return len(t.nodes) - 1
}

// Sub pushes a-b.
func (t *Tape) Sub(a, b int) int {
	v := t.nodes[a].v - t.nodes[b].v
	t.nodes = append(t.nodes, node{k: kSub, v: v, a: a, b: b})
	return len(t.nodes) - 1
}

// Mul pushes a*b.
func (t *Tape) Mul(a, b int) int {
	v := t.nodes[a].v * t.nodes[b].v
	t.nodes = append(t.nodes, node{k: kMul, v: v, a: a, b: b})
	return len(t.nodes) - 1
}

// Div pushes a/b.
func (t *Tape) Div(a, b int) int {
	v := t.nodes[a].v / t.nodes[b].v
	t.nodes = append(t.nodes, node{k: kDiv, v: v, a: a, b: b})
	return len(t.nodes) - 1
}

// Neg pushes -a.
func (t *Tape) Neg(a int) int {
	v := -t.nodes[a].v
	t.nodes = append(t.nodes, node{k: kNeg, v: v, a: a})
	return len(t.nodes) - 1
}

// SoftMuxNode pushes soft_mux(w, pure, each).
func (t *Tape) SoftMuxNode(w, pure, each int) int {
	v := SoftMux(t.nodes[w].v, t.nodes[pure].v, t.nodes[each].v)
	t.nodes = append(t.nodes, node{k: kSoftMux, v: v, a: w, b: pure, c: each})
	return len(t.nodes) - 1
}

// SumNode pushes sum of element nodes.
func (t *Tape) SumNode(elems []int) int {
	var s float64
	for _, id := range elems {
		s += t.nodes[id].v
	}
	cp := append([]int(nil), elems...)
	t.nodes = append(t.nodes, node{k: kSum, v: s, args: cp})
	return len(t.nodes) - 1
}

// DotNode pushes sum_i a[i]*b[i].
func (t *Tape) DotNode(as, bs []int) int {
	n := len(as)
	if len(bs) < n {
		n = len(bs)
	}
	var s float64
	args := make([]int, 0, 2*n)
	for i := 0; i < n; i++ {
		s += t.nodes[as[i]].v * t.nodes[bs[i]].v
		args = append(args, as[i])
	}
	for i := 0; i < n; i++ {
		args = append(args, bs[i])
	}
	t.nodes = append(t.nodes, node{k: kDot, v: s, c: n, args: args})
	return len(t.nodes) - 1
}

// Value returns the primal at id.
func (t *Tape) Value(id int) float64 { return t.nodes[id].v }

// Grad seeds ∂L/∂out = seed and returns a gradient map id→grad for all nodes.
func (t *Tape) Grad(out int, seed float64) []float64 {
	n := len(t.nodes)
	g := make([]float64, n)
	g[out] = seed
	for i := n - 1; i >= 0; i-- {
		gi := g[i]
		if gi == 0 {
			continue
		}
		nd := t.nodes[i]
		switch nd.k {
		case kConst:
		case kAdd:
			g[nd.a] += gi
			g[nd.b] += gi
		case kSub:
			g[nd.a] += gi
			g[nd.b] -= gi
		case kMul:
			g[nd.a] += gi * t.nodes[nd.b].v
			g[nd.b] += gi * t.nodes[nd.a].v
		case kDiv:
			bv := t.nodes[nd.b].v
			g[nd.a] += gi / bv
			g[nd.b] += -gi * t.nodes[nd.a].v / (bv * bv)
		case kNeg:
			g[nd.a] -= gi
		case kSoftMux:
			dw, dp, de := SoftMuxVJP(t.nodes[nd.a].v, t.nodes[nd.b].v, t.nodes[nd.c].v, gi)
			g[nd.a] += dw
			g[nd.b] += dp
			g[nd.c] += de
		case kSum:
			for _, pid := range nd.args {
				g[pid] += gi
			}
		case kDot:
			nn := nd.c
			for j := 0; j < nn; j++ {
				ai := nd.args[j]
				bi := nd.args[nn+j]
				g[ai] += gi * t.nodes[bi].v
				g[bi] += gi * t.nodes[ai].v
			}
		}
	}
	return g
}

// Result is the value of an expression plus selected gradients.
type Result struct {
	Value float64
	Grads map[string]float64
}

// DifferentiableExpr evaluates a closed f32 expression AST under env and
// returns the value plus partial derivatives w.r.t. each free variable name.
// Supported ops: literals, vars, unary -, +, -, *, /, soft_mux, sum, dot,
// span lits, span_add, scale, map_mul, map((x)=>body),
// zip_with((p0..pN-1)=>body, s0..sN-1) (N>=2), filter((x)=>pred, span),
// scan((acc,x)=>body, init, span), reduce/fold((acc,x)=>body, init, span)
// with scalar body AD.
func DifferentiableExpr(e ast.Expr, env map[string]float64) (Result, error) {
	tp := &Tape{}
	ids := map[string]int{}
	for name, v := range env {
		ids[name] = tp.Const(v)
	}
	out, err := buildTape(tp, e, ids)
	if err != nil {
		return Result{}, err
	}
	g := tp.Grad(out, 1)
	grads := make(map[string]float64, len(ids))
	for name, id := range ids {
		grads[name] = g[id]
	}
	return Result{Value: tp.Value(out), Grads: grads}, nil
}

// GradSum computes sum(xs) and ∂sum/∂xs[i] (=1 each).
func GradSum(xs []float64) (float64, []float64) {
	tp := &Tape{}
	ids := make([]int, len(xs))
	for i, v := range xs {
		ids[i] = tp.Const(v)
	}
	out := tp.SumNode(ids)
	g := tp.Grad(out, 1)
	gs := make([]float64, len(xs))
	for i, id := range ids {
		gs[i] = g[id]
	}
	return tp.Value(out), gs
}

// GradDot computes dot(a,b) and (∂/∂a, ∂/∂b).
func GradDot(a, b []float64) (float64, []float64, []float64) {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	tp := &Tape{}
	as := make([]int, n)
	bs := make([]int, n)
	for i := 0; i < n; i++ {
		as[i] = tp.Const(a[i])
		bs[i] = tp.Const(b[i])
	}
	out := tp.DotNode(as, bs)
	g := tp.Grad(out, 1)
	ga := make([]float64, n)
	gb := make([]float64, n)
	for i := 0; i < n; i++ {
		ga[i] = g[as[i]]
		gb[i] = g[bs[i]]
	}
	return tp.Value(out), ga, gb
}

// GradSpanAdd computes sum(a_i+b_i) and (∂/∂a, ∂/∂b). Each ∂ is 1.
func GradSpanAdd(a, b []float64) (float64, []float64, []float64) {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	tp := &Tape{}
	as := make([]int, n)
	bs := make([]int, n)
	zs := make([]int, n)
	for i := 0; i < n; i++ {
		as[i] = tp.Const(a[i])
		bs[i] = tp.Const(b[i])
		zs[i] = tp.Add(as[i], bs[i])
	}
	out := tp.SumNode(zs)
	g := tp.Grad(out, 1)
	ga := make([]float64, n)
	gb := make([]float64, n)
	for i := 0; i < n; i++ {
		ga[i] = g[as[i]]
		gb[i] = g[bs[i]]
	}
	return tp.Value(out), ga, gb
}

// GradScale computes sum(s*xs[i]) and (∂/∂s, ∂/∂xs).
func GradScale(s float64, xs []float64) (float64, float64, []float64) {
	tp := &Tape{}
	sid := tp.Const(s)
	ids := make([]int, len(xs))
	zs := make([]int, len(xs))
	for i, v := range xs {
		ids[i] = tp.Const(v)
		zs[i] = tp.Mul(sid, ids[i])
	}
	out := tp.SumNode(zs)
	g := tp.Grad(out, 1)
	gxs := make([]float64, len(xs))
	for i, id := range ids {
		gxs[i] = g[id]
	}
	return tp.Value(out), g[sid], gxs
}

func buildTape(tp *Tape, e ast.Expr, ids map[string]int) (int, error) {
	switch n := e.(type) {
	case *ast.Literal:
		switch n.Kind {
		case "int":
			return tp.Const(float64(n.Value.(int))), nil
		case "float":
			return tp.Const(n.Value.(float64)), nil
		case "bool":
			if n.Value.(bool) {
				return tp.Const(1), nil
			}
			return tp.Const(0), nil
		default:
			return 0, fmt.Errorf("ad: non-numeric literal %s", n.Kind)
		}
	case *ast.Var:
		if id, ok := ids[n.Name]; ok {
			return id, nil
		}
		return 0, fmt.Errorf("ad: free variable %q not in env", n.Name)
	case *ast.Unary:
		x, err := buildTape(tp, n.Expr, ids)
		if err != nil {
			return 0, err
		}
		switch n.Op {
		case "-":
			return tp.Neg(x), nil
		case "+":
			return x, nil
		default:
			return 0, fmt.Errorf("ad: unary %s", n.Op)
		}
	case *ast.Binary:
		l, err := buildTape(tp, n.Left, ids)
		if err != nil {
			return 0, err
		}
		r, err := buildTape(tp, n.Right, ids)
		if err != nil {
			return 0, err
		}
		switch n.Op {
		case "+":
			return tp.Add(l, r), nil
		case "-":
			return tp.Sub(l, r), nil
		case "*":
			return tp.Mul(l, r), nil
		case "/":
			return tp.Div(l, r), nil
		default:
			return 0, fmt.Errorf("ad: binary %s not differentiable", n.Op)
		}
	case *ast.Ternary:
		w, err := buildTape(tp, n.Gate, ids)
		if err != nil {
			return 0, err
		}
		p, err := buildTape(tp, n.Pure, ids)
		if err != nil {
			return 0, err
		}
		ea, err := buildTape(tp, n.Each, ids)
		if err != nil {
			return 0, err
		}
		return tp.SoftMuxNode(w, p, ea), nil
	case *ast.SpanLit:
		// Span literal is not a scalar; only valid inside sum/dot via Call.
		return 0, fmt.Errorf("ad: bare span literal (use sum/dot)")
	case *ast.Call:
		v, ok := n.Callee.(*ast.Var)
		if !ok {
			return 0, fmt.Errorf("ad: only named calls")
		}
		switch v.Name {
		case "soft_mux":
			if len(n.Args) != 3 {
				return 0, fmt.Errorf("ad: soft_mux arity")
			}
			w, err := buildTape(tp, n.Args[0], ids)
			if err != nil {
				return 0, err
			}
			p, err := buildTape(tp, n.Args[1], ids)
			if err != nil {
				return 0, err
			}
			ea, err := buildTape(tp, n.Args[2], ids)
			if err != nil {
				return 0, err
			}
			return tp.SoftMuxNode(w, p, ea), nil
		case "sum":
			elems, err := buildSpanArgs(tp, n.Args, ids)
			if err != nil {
				return 0, err
			}
			return tp.SumNode(elems), nil
		case "dot":
			if len(n.Args) != 2 {
				return 0, fmt.Errorf("ad: dot expects 2 spans")
			}
			as, err := buildSpanArg(tp, n.Args[0], ids)
			if err != nil {
				return 0, err
			}
			bs, err := buildSpanArg(tp, n.Args[1], ids)
			if err != nil {
				return 0, err
			}
			return tp.DotNode(as, bs), nil
		case "scale", "map_mul", "span_add", "map", "zip_with", "scan", "filter":
			return 0, fmt.Errorf("ad: %s returns span; wrap with sum/dot", v.Name)
		case "reduce", "fold":
			// reduce((acc,x)=>body, init, span) / fold(...): left-fold scalar AD
			if len(n.Args) != 3 {
				return 0, fmt.Errorf("ad: %s expects (lambda, init, span)", v.Name)
			}
			lam, ok := n.Args[0].(*ast.Lambda)
			if !ok {
				return 0, fmt.Errorf("ad: %s expects lambda first arg", v.Name)
			}
			if len(lam.Params) != 2 {
				return 0, fmt.Errorf("ad: %s lambda expects exactly 2 params", v.Name)
			}
			pacc, px := lam.Params[0].Name, lam.Params[1].Name
			accID, err := buildTape(tp, n.Args[1], ids)
			if err != nil {
				return 0, err
			}
			xs, err := buildSpanArg(tp, n.Args[2], ids)
			if err != nil {
				return 0, err
			}
			for _, xid := range xs {
				child := make(map[string]int, len(ids)+2)
				for k, v := range ids {
					child[k] = v
				}
				child[pacc] = accID
				child[px] = xid
				bid, err := buildTape(tp, lam.Body, child)
				if err != nil {
					return 0, fmt.Errorf("ad: %s body: %w", v.Name, err)
				}
				accID = bid
			}
			return accID, nil
		default:
			return 0, fmt.Errorf("ad: unsupported call %s", v.Name)
		}
	default:
		return 0, fmt.Errorf("ad: unsupported expr %T", e)
	}
}

// buildSpanArgs flattens sum(a,b,c), sum([a,b,c]), or sum(span_producer(...)).
func buildSpanArgs(tp *Tape, args []ast.Expr, ids map[string]int) ([]int, error) {
	if len(args) == 1 {
		if sl, ok := args[0].(*ast.SpanLit); ok {
			return buildSpanLit(tp, sl, ids)
		}
		// single span producer call
		if _, ok := args[0].(*ast.Call); ok {
			return buildSpanArg(tp, args[0], ids)
		}
	}
	out := make([]int, 0, len(args))
	for _, a := range args {
		id, err := buildTape(tp, a, ids)
		if err != nil {
			return nil, err
		}
		out = append(out, id)
	}
	return out, nil
}

func buildSpanArg(tp *Tape, e ast.Expr, ids map[string]int) ([]int, error) {
	switch n := e.(type) {
	case *ast.SpanLit:
		return buildSpanLit(tp, n, ids)
	case *ast.Call:
		v, ok := n.Callee.(*ast.Var)
		if !ok {
			return nil, fmt.Errorf("ad: only named span calls")
		}
		switch v.Name {
		case "span_add":
			if len(n.Args) != 2 {
				return nil, fmt.Errorf("ad: span_add expects 2 spans")
			}
			as, err := buildSpanArg(tp, n.Args[0], ids)
			if err != nil {
				return nil, err
			}
			bs, err := buildSpanArg(tp, n.Args[1], ids)
			if err != nil {
				return nil, err
			}
			nn := len(as)
			if len(bs) < nn {
				nn = len(bs)
			}
			out := make([]int, nn)
			for i := 0; i < nn; i++ {
				out[i] = tp.Add(as[i], bs[i])
			}
			return out, nil
		case "scale", "map_mul":
			// scale(s, span) / map_mul(s, span): elementwise s * x_i (const-mul map subset)
			if len(n.Args) != 2 {
				return nil, fmt.Errorf("ad: %s expects (s, span)", v.Name)
			}
			sid, err := buildTape(tp, n.Args[0], ids)
			if err != nil {
				return nil, err
			}
			xs, err := buildSpanArg(tp, n.Args[1], ids)
			if err != nil {
				return nil, err
			}
			out := make([]int, len(xs))
			for i, xid := range xs {
				out[i] = tp.Mul(sid, xid)
			}
			return out, nil
		case "map":
			// map((x)=>body, span): apply scalar AD body per element
			if len(n.Args) != 2 {
				return nil, fmt.Errorf("ad: map expects (lambda, span)")
			}
			lam, ok := n.Args[0].(*ast.Lambda)
			if !ok {
				return nil, fmt.Errorf("ad: map expects lambda first arg")
			}
			if len(lam.Params) != 1 {
				return nil, fmt.Errorf("ad: map lambda expects exactly 1 param")
			}
			pname := lam.Params[0].Name
			xs, err := buildSpanArg(tp, n.Args[1], ids)
			if err != nil {
				return nil, err
			}
			out := make([]int, len(xs))
			for i, xid := range xs {
				child := make(map[string]int, len(ids)+1)
				for k, v := range ids {
					child[k] = v
				}
				child[pname] = xid
				bid, err := buildTape(tp, lam.Body, child)
				if err != nil {
					return nil, fmt.Errorf("ad: map body: %w", err)
				}
				out[i] = bid
			}
			return out, nil
		case "zip_with":
			// zip_with((p0,...,pN-1)=>body, s0, ..., sN-1) for N>=2
			if len(n.Args) < 3 {
				return nil, fmt.Errorf("ad: zip_with expects (lambda, span, ...)")
			}
			lam, ok := n.Args[0].(*ast.Lambda)
			if !ok {
				return nil, fmt.Errorf("ad: zip_with expects lambda first arg")
			}
			wantParams := len(n.Args) - 1
			if len(lam.Params) != wantParams {
				return nil, fmt.Errorf("ad: zip_with lambda expects exactly %d params", wantParams)
			}
			spans := make([][]int, wantParams)
			nn := -1
			for si := 0; si < wantParams; si++ {
				sp, err := buildSpanArg(tp, n.Args[si+1], ids)
				if err != nil {
					return nil, err
				}
				spans[si] = sp
				if nn < 0 || len(sp) < nn {
					nn = len(sp)
				}
			}
			out := make([]int, nn)
			for i := 0; i < nn; i++ {
				child := make(map[string]int, len(ids)+wantParams)
				for k, v := range ids {
					child[k] = v
				}
				for si := 0; si < wantParams; si++ {
					child[lam.Params[si].Name] = spans[si][i]
				}
				bid, err := buildTape(tp, lam.Body, child)
				if err != nil {
					return nil, fmt.Errorf("ad: zip_with body: %w", err)
				}
				out[i] = bid
			}
			return out, nil
		case "scan":
			// scan((acc,x)=>body, init, span): emit intermediate acc after each step
			if len(n.Args) != 3 {
				return nil, fmt.Errorf("ad: scan expects (lambda, init, span)")
			}
			lam, ok := n.Args[0].(*ast.Lambda)
			if !ok {
				return nil, fmt.Errorf("ad: scan expects lambda first arg")
			}
			if len(lam.Params) != 2 {
				return nil, fmt.Errorf("ad: scan lambda expects exactly 2 params")
			}
			pacc, px := lam.Params[0].Name, lam.Params[1].Name
			accID, err := buildTape(tp, n.Args[1], ids)
			if err != nil {
				return nil, err
			}
			xs, err := buildSpanArg(tp, n.Args[2], ids)
			if err != nil {
				return nil, err
			}
			out := make([]int, len(xs))
			for i, xid := range xs {
				child := make(map[string]int, len(ids)+2)
				for k, v := range ids {
					child[k] = v
				}
				child[pacc] = accID
				child[px] = xid
				bid, err := buildTape(tp, lam.Body, child)
				if err != nil {
					return nil, fmt.Errorf("ad: scan body: %w", err)
				}
				accID = bid
				out[i] = bid
			}
			return out, nil
		case "filter":
			// filter((x)=>pred, span): keep x when pred != 0 (hard threshold)
			if len(n.Args) != 2 {
				return nil, fmt.Errorf("ad: filter expects (lambda, span)")
			}
			lam, ok := n.Args[0].(*ast.Lambda)
			if !ok {
				return nil, fmt.Errorf("ad: filter expects lambda first arg")
			}
			if len(lam.Params) != 1 {
				return nil, fmt.Errorf("ad: filter lambda expects exactly 1 param")
			}
			pname := lam.Params[0].Name
			xs, err := buildSpanArg(tp, n.Args[1], ids)
			if err != nil {
				return nil, err
			}
			out := make([]int, 0, len(xs))
			for _, xid := range xs {
				child := make(map[string]int, len(ids)+1)
				for k, v := range ids {
					child[k] = v
				}
				child[pname] = xid
				// build pred into tape for forward value; disconnected from out so
				// keep-decision free vars don't get hard-threshold grads
				pid, err := buildTape(tp, lam.Body, child)
				if err != nil {
					return nil, fmt.Errorf("ad: filter pred: %w", err)
				}
				if tp.Value(pid) != 0 {
					out = append(out, xid)
				}
			}
			return out, nil
		default:
			return nil, fmt.Errorf("ad: unsupported span producer %s", v.Name)
		}
	default:
		return nil, fmt.Errorf("ad: expected span literal or span_add/scale/map_mul/map/zip_with/scan/filter")
	}
}

func buildSpanLit(tp *Tape, sl *ast.SpanLit, ids map[string]int) ([]int, error) {
	out := make([]int, 0, len(sl.Elems))
	for _, el := range sl.Elems {
		id, err := buildTape(tp, el, ids)
		if err != nil {
			return nil, err
		}
		out = append(out, id)
	}
	return out, nil
}

// GradExpr is a small DSL helper used by builtins: parse a tiny expression string.
// Syntax: numbers, + - * / unary-, soft_mux(a,b,c), sum(...|[...]),
// dot(span,span), span_add(span,span), scale(s,span), map_mul(s,span),
// map((x)=>body, span), zip_with((p0..)=>body, s0..) (n-ary N>=2),
// filter((x)=>pred, span), scan((acc,x)=>body, init, span),
// reduce/fold((acc,x)=>body, init, span), span lits [a,b],
// parentheses, names [A-Za-z_][A-Za-z0-9_]*
func GradExpr(expr string, env map[string]float64) (Result, error) {
	e, err := parseMini(expr)
	if err != nil {
		return Result{}, err
	}
	return DifferentiableExpr(e, env)
}

// --- minimal expression parser (subset) ---

type mini struct {
	s   string
	i   int
	tok string
	val string
}

func parseMini(s string) (ast.Expr, error) {
	m := &mini{s: s}
	m.next()
	e, err := m.parseExpr()
	if err != nil {
		return nil, err
	}
	if m.tok != "eof" {
		return nil, fmt.Errorf("ad parse: trailing junk near %q", m.val)
	}
	return e, nil
}

func (m *mini) next() {
	for m.i < len(m.s) && (m.s[m.i] == ' ' || m.s[m.i] == '\t') {
		m.i++
	}
	if m.i >= len(m.s) {
		m.tok = "eof"
		m.val = ""
		return
	}
	c := m.s[m.i]
	// =>
	if c == '=' && m.i+1 < len(m.s) && m.s[m.i+1] == '>' {
		m.tok = "=>"
		m.val = "=>"
		m.i += 2
		return
	}
	switch c {
	case '+', '-', '*', '/', '(', ')', ',', '[', ']':
		m.tok = string(c)
		m.val = m.tok
		m.i++
		return
	}
	if c >= '0' && c <= '9' || c == '.' {
		j := m.i
		for j < len(m.s) && ((m.s[j] >= '0' && m.s[j] <= '9') || m.s[j] == '.') {
			j++
		}
		m.tok = "num"
		m.val = m.s[m.i:j]
		m.i = j
		return
	}
	if c == '_' || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') {
		j := m.i + 1
		for j < len(m.s) {
			ch := m.s[j]
			if ch == '_' || (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') {
				j++
				continue
			}
			break
		}
		m.tok = "id"
		m.val = m.s[m.i:j]
		m.i = j
		return
	}
	m.tok = "err"
	m.val = string(c)
}

func (m *mini) parseExpr() (ast.Expr, error) {
	return m.parseAdd()
}

func (m *mini) parseAdd() (ast.Expr, error) {
	left, err := m.parseMul()
	if err != nil {
		return nil, err
	}
	for m.tok == "+" || m.tok == "-" {
		op := m.tok
		m.next()
		right, err := m.parseMul()
		if err != nil {
			return nil, err
		}
		left = &ast.Binary{Op: op, Left: left, Right: right}
	}
	return left, nil
}

func (m *mini) parseMul() (ast.Expr, error) {
	left, err := m.parseUnary()
	if err != nil {
		return nil, err
	}
	for m.tok == "*" || m.tok == "/" {
		op := m.tok
		m.next()
		right, err := m.parseUnary()
		if err != nil {
			return nil, err
		}
		left = &ast.Binary{Op: op, Left: left, Right: right}
	}
	return left, nil
}

func (m *mini) parseUnary() (ast.Expr, error) {
	if m.tok == "-" {
		m.next()
		x, err := m.parseUnary()
		if err != nil {
			return nil, err
		}
		return &ast.Unary{Op: "-", Expr: x}, nil
	}
	if m.tok == "+" {
		m.next()
		return m.parseUnary()
	}
	return m.parsePrimary()
}

func (m *mini) parsePrimary() (ast.Expr, error) {
	switch m.tok {
	case "num":
		s := m.val
		m.next()
		if strings.Contains(s, ".") {
			f, err := strconv.ParseFloat(s, 64)
			if err != nil {
				return nil, err
			}
			return &ast.Literal{Kind: "float", Value: f}, nil
		}
		n, err := strconv.Atoi(s)
		if err != nil {
			f, err2 := strconv.ParseFloat(s, 64)
			if err2 != nil {
				return nil, err
			}
			return &ast.Literal{Kind: "float", Value: f}, nil
		}
		return &ast.Literal{Kind: "int", Value: n}, nil
	case "id":
		name := m.val
		m.next()
		if m.tok == "(" {
			m.next()
			// map((x)=>body, span) / zip_with((x,y)=>body, a, b) special forms
			if name == "map" {
				lam, err := m.parseLambdaArg()
				if err != nil {
					return nil, err
				}
				if m.tok != "," {
					return nil, fmt.Errorf("ad parse: map wants ,")
				}
				m.next()
				span, err := m.parseSpanExpr()
				if err != nil {
					return nil, err
				}
				if m.tok != ")" {
					return nil, fmt.Errorf("ad parse: map wants )")
				}
				m.next()
				return &ast.Call{
					Callee: &ast.Var{Name: "map"},
					Args:   []ast.Expr{lam, span},
				}, nil
			}
			if name == "zip_with" {
				lam, err := m.parseLambdaArg()
				if err != nil {
					return nil, err
				}
				var spans []ast.Expr
				for {
					if m.tok != "," {
						break
					}
					m.next()
					sp, err := m.parseSpanExpr()
					if err != nil {
						return nil, err
					}
					spans = append(spans, sp)
				}
				if len(spans) < 2 {
					return nil, fmt.Errorf("ad parse: zip_with wants >=2 spans")
				}
				if m.tok != ")" {
					return nil, fmt.Errorf("ad parse: zip_with wants )")
				}
				m.next()
				args := make([]ast.Expr, 0, 1+len(spans))
				args = append(args, lam)
				args = append(args, spans...)
				return &ast.Call{Callee: &ast.Var{Name: "zip_with"}, Args: args}, nil
			}
			if name == "filter" {
				lam, err := m.parseLambdaArg()
				if err != nil {
					return nil, err
				}
				if m.tok != "," {
					return nil, fmt.Errorf("ad parse: filter wants ,")
				}
				m.next()
				sp, err := m.parseSpanExpr()
				if err != nil {
					return nil, err
				}
				if m.tok != ")" {
					return nil, fmt.Errorf("ad parse: filter wants )")
				}
				m.next()
				return &ast.Call{Callee: &ast.Var{Name: "filter"}, Args: []ast.Expr{lam, sp}}, nil
			}
			if name == "reduce" || name == "fold" || name == "scan" {
				lam, err := m.parseLambdaArg()
				if err != nil {
					return nil, err
				}
				if m.tok != "," {
					return nil, fmt.Errorf("ad parse: %s wants ,", name)
				}
				m.next()
				init, err := m.parseExpr()
				if err != nil {
					return nil, err
				}
				if m.tok != "," {
					return nil, fmt.Errorf("ad parse: %s wants second ,", name)
				}
				m.next()
				sp, err := m.parseSpanExpr()
				if err != nil {
					return nil, err
				}
				if m.tok != ")" {
					return nil, fmt.Errorf("ad parse: %s wants )", name)
				}
				m.next()
				return &ast.Call{
					Callee: &ast.Var{Name: name},
					Args:   []ast.Expr{lam, init, sp},
				}, nil
			}
			var args []ast.Expr
			if m.tok != ")" {
				for {
					// allow nested span producers as args of sum/dot handled elsewhere
					a, err := m.parseExpr()
					if err != nil {
						return nil, err
					}
					args = append(args, a)
					if m.tok == "," {
						m.next()
						continue
					}
					break
				}
			}
			if m.tok != ")" {
				return nil, fmt.Errorf("ad parse: want )")
			}
			m.next()
			return &ast.Call{Callee: &ast.Var{Name: name}, Args: args}, nil
		}
		return &ast.Var{Name: name}, nil
	case "[":
		m.next()
		var elems []ast.Expr
		if m.tok != "]" {
			for {
				e, err := m.parseExpr()
				if err != nil {
					return nil, err
				}
				elems = append(elems, e)
				if m.tok == "," {
					m.next()
					continue
				}
				break
			}
		}
		if m.tok != "]" {
			return nil, fmt.Errorf("ad parse: want ]")
		}
		m.next()
		return &ast.SpanLit{Elems: elems}, nil
	case "(":
		m.next()
		e, err := m.parseExpr()
		if err != nil {
			return nil, err
		}
		if m.tok != ")" {
			return nil, fmt.Errorf("ad parse: want )")
		}
		m.next()
		return e, nil
	default:
		return nil, fmt.Errorf("ad parse: unexpected %q", m.val)
	}
}

// parseLambdaArg parses (x)=>body, (x,y)=>body, or x=>body.
func (m *mini) parseLambdaArg() (*ast.Lambda, error) {
	if m.tok == "(" {
		m.next()
		var params []ast.Param
		if m.tok == "id" {
			for {
				if m.tok != "id" {
					return nil, fmt.Errorf("ad parse: lambda param")
				}
				params = append(params, ast.Param{Name: m.val})
				m.next()
				if m.tok == "," {
					m.next()
					continue
				}
				break
			}
		}
		if m.tok != ")" {
			return nil, fmt.Errorf("ad parse: lambda wants )")
		}
		m.next()
		if m.tok != "=>" {
			return nil, fmt.Errorf("ad parse: lambda wants =>")
		}
		m.next()
		body, err := m.parseExpr()
		if err != nil {
			return nil, err
		}
		return &ast.Lambda{Params: params, Body: body}, nil
	}
	if m.tok == "id" {
		pname := m.val
		savedI, savedTok, savedVal := m.i, m.tok, m.val
		m.next()
		if m.tok != "=>" {
			// not a lambda — leave caller to fail; restore so parseExpr can see id
			m.i, m.tok, m.val = savedI, savedTok, savedVal
			// but we already were on id and advanced... restore to id
			// saved is after first next from outside. Better only support (x)=> form.
			return nil, fmt.Errorf("ad parse: use (x)=>body for map lambda")
		}
		m.next()
		body, err := m.parseExpr()
		if err != nil {
			return nil, err
		}
		return &ast.Lambda{Params: []ast.Param{{Name: pname}}, Body: body}, nil
	}
	return nil, fmt.Errorf("ad parse: expected lambda")
}

// parseSpanExpr parses [..] or a span producer call: span_add/scale/map_mul/map/zip_with/scan/filter(...).
func (m *mini) parseSpanExpr() (ast.Expr, error) {
	if m.tok == "[" {
		return m.parseSpanLit()
	}
	if m.tok != "id" {
		return nil, fmt.Errorf("ad parse: expected span")
	}
	name := m.val
	m.next()
	if m.tok != "(" {
		return nil, fmt.Errorf("ad parse: expected span call")
	}
	m.next()
	switch name {
	case "map":
		lam, err := m.parseLambdaArg()
		if err != nil {
			return nil, err
		}
		if m.tok != "," {
			return nil, fmt.Errorf("ad parse: map wants ,")
		}
		m.next()
		sp, err := m.parseSpanExpr()
		if err != nil {
			return nil, err
		}
		if m.tok != ")" {
			return nil, fmt.Errorf("ad parse: map wants )")
		}
		m.next()
		return &ast.Call{Callee: &ast.Var{Name: "map"}, Args: []ast.Expr{lam, sp}}, nil
	case "zip_with":
		lam, err := m.parseLambdaArg()
		if err != nil {
			return nil, err
		}
		var spans []ast.Expr
		for {
			if m.tok != "," {
				break
			}
			m.next()
			sp, err := m.parseSpanExpr()
			if err != nil {
				return nil, err
			}
			spans = append(spans, sp)
		}
		if len(spans) < 2 {
			return nil, fmt.Errorf("ad parse: zip_with wants >=2 spans")
		}
		if m.tok != ")" {
			return nil, fmt.Errorf("ad parse: zip_with wants )")
		}
		m.next()
		args := make([]ast.Expr, 0, 1+len(spans))
		args = append(args, lam)
		args = append(args, spans...)
		return &ast.Call{Callee: &ast.Var{Name: "zip_with"}, Args: args}, nil
	case "filter":
		lam, err := m.parseLambdaArg()
		if err != nil {
			return nil, err
		}
		if m.tok != "," {
			return nil, fmt.Errorf("ad parse: filter wants ,")
		}
		m.next()
		sp, err := m.parseSpanExpr()
		if err != nil {
			return nil, err
		}
		if m.tok != ")" {
			return nil, fmt.Errorf("ad parse: filter wants )")
		}
		m.next()
		return &ast.Call{Callee: &ast.Var{Name: "filter"}, Args: []ast.Expr{lam, sp}}, nil
	case "scan":
		lam, err := m.parseLambdaArg()
		if err != nil {
			return nil, err
		}
		if m.tok != "," {
			return nil, fmt.Errorf("ad parse: scan wants ,")
		}
		m.next()
		init, err := m.parseExpr()
		if err != nil {
			return nil, err
		}
		if m.tok != "," {
			return nil, fmt.Errorf("ad parse: scan wants second ,")
		}
		m.next()
		sp, err := m.parseSpanExpr()
		if err != nil {
			return nil, err
		}
		if m.tok != ")" {
			return nil, fmt.Errorf("ad parse: scan wants )")
		}
		m.next()
		return &ast.Call{Callee: &ast.Var{Name: "scan"}, Args: []ast.Expr{lam, init, sp}}, nil
	case "span_add":
		a, err := m.parseSpanExpr()
		if err != nil {
			return nil, err
		}
		if m.tok != "," {
			return nil, fmt.Errorf("ad parse: span_add wants ,")
		}
		m.next()
		b, err := m.parseSpanExpr()
		if err != nil {
			return nil, err
		}
		if m.tok != ")" {
			return nil, fmt.Errorf("ad parse: span_add wants )")
		}
		m.next()
		return &ast.Call{Callee: &ast.Var{Name: "span_add"}, Args: []ast.Expr{a, b}}, nil
	case "scale", "map_mul":
		sarg, err := m.parseExpr()
		if err != nil {
			return nil, err
		}
		if m.tok != "," {
			return nil, fmt.Errorf("ad parse: %s wants ,", name)
		}
		m.next()
		sp, err := m.parseSpanExpr()
		if err != nil {
			return nil, err
		}
		if m.tok != ")" {
			return nil, fmt.Errorf("ad parse: %s wants )", name)
		}
		m.next()
		return &ast.Call{Callee: &ast.Var{Name: name}, Args: []ast.Expr{sarg, sp}}, nil
	default:
		return nil, fmt.Errorf("ad parse: unknown span fn %s", name)
	}
}

// parseSpanLit parses [e0, e1, ...].
func (m *mini) parseSpanLit() (*ast.SpanLit, error) {
	if m.tok != "[" {
		return nil, fmt.Errorf("ad parse: want [")
	}
	m.next()
	var elems []ast.Expr
	if m.tok != "]" {
		for {
			e, err := m.parseExpr()
			if err != nil {
				return nil, err
			}
			elems = append(elems, e)
			if m.tok == "," {
				m.next()
				continue
			}
			break
		}
	}
	if m.tok != "]" {
		return nil, fmt.Errorf("ad parse: want ]")
	}
	m.next()
	return &ast.SpanLit{Elems: elems}, nil
}

