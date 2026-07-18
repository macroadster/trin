package codegen

import (
	"fmt"
	"strings"

	"github.com/macroadster/trin/internal/ast"
)

// hasHostFloatExport is true when the program exports loss/kernel for the float host path.
func hasHostFloatExport(prog *ast.Program) bool {
	if prog == nil {
		return false
	}
	for _, it := range prog.Items {
		fn, ok := it.(*ast.FnDef)
		if !ok {
			continue
		}
		switch fn.Name {
		case "loss", "kernel", "soft_kernel":
			return true
		}
	}
	return false
}

// hasProgramMain reports whether the AST defines fn main.
func hasProgramMain(prog *ast.Program) bool {
	if prog == nil {
		return false
	}
	for _, it := range prog.Items {
		if fn, ok := it.(*ast.FnDef); ok && fn.Name == "main" && fn.Body != nil {
			return true
		}
	}
	return false
}

// emitFullProgramMLIR produces a portable full-program MLIR body for @main (v0.23)
// when the program has main and does not export loss/kernel.
// Uses dialect helpers (soft_mux, sum, println/to_str as func.func + llvm.intrinsics notes).
// Returns "" when full-program MLIR should not be emitted.
func emitFullProgramMLIR(prog *ast.Program) string {
	if prog == nil || !hasProgramMain(prog) || hasHostFloatExport(prog) {
		return ""
	}
	var mainFn *ast.FnDef
	for _, it := range prog.Items {
		if fn, ok := it.(*ast.FnDef); ok && fn.Name == "main" {
			mainFn = fn
			break
		}
	}
	if mainFn == nil || mainFn.Body == nil {
		return ""
	}

	// Use the same expression lowering as LLVM full-program to decide if @main is supportable.
	// If emitFullProgram fails, skip MLIR full-program too.
	if emitFullProgram(prog) == "" {
		return ""
	}

	var b strings.Builder
	b.WriteString("  // --- full-program MLIR (v0.23 + dyn tiles v0.32 + residual peel v0.34 + auto peel v0.36/v0.37) — executable via emit-mlir-llvm ---\n")
	b.WriteString("  // trin.full_program = true\n")
	b.WriteString("  // trin.dyn_tile = runtime  (span sum/dot pick 4/8/16 via @trin_hw_vlen_for_n)\n")
	b.WriteString("  // trin.residual_peel = emit  (non-fixed n peels into 16/8/4 tiles + residual; v0.34)\n")
	b.WriteString("  // Runtime: puts/printf via llvm callables; span kernels via @trin_* helpers\n")
	b.WriteString("  // Dynamic tile: fixed 4/8/16 sum/dot lower through @trin_dyn_sum* / @trin_dyn_dot*\n")
	b.WriteString("  // Residual peel: n∉{4,8,16} → max tiles + scalar residual on LLVM path (v0.34/v0.36)\n\n")

	// Lightweight runtime decls (MLIR form matching LLVM full-program helpers)
	b.WriteString("  func.func private @puts(!llvm.ptr) -> i32\n")
	b.WriteString("  func.func private @printf(!llvm.ptr, ...) -> i32\n")
	b.WriteString("  func.func private @snprintf(!llvm.ptr, i64, !llvm.ptr, ...) -> i32\n")
	b.WriteString("  func.func private @strcpy(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n")
	b.WriteString("  func.func private @strcat(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n\n")

	// Outline pure helper note
	b.WriteString("  func.func @trin_to_str_f32(%x: f32) -> !llvm.ptr {\n")
	b.WriteString("    // lowered to snprintf(\"%g\") on LLVM path\n")
	b.WriteString("    %p = llvm.mlir.zero : !llvm.ptr\n")
	b.WriteString("    return %p : !llvm.ptr\n")
	b.WriteString("  }\n\n")

	b.WriteString("  func.func @trin_println_str(%s: !llvm.ptr) {\n")
	b.WriteString("    %n = func.call @puts(%s) : (!llvm.ptr) -> i32\n")
	b.WriteString("    return\n")
	b.WriteString("  }\n\n")

	// Emit a structured @main that mirrors the program's println/soft_mux/sum/dot graph
	// at the MLIR surface (arith + func.call). Exact SSA reuses lowered compiler notes.
	g := &mlirFull{
		b:         &b,
		n:         0,
		names:     map[string]mlirVal{},
		spanElems: map[string][]mlirVal{},
		ok:        true,
	}
	b.WriteString("  func.func @main() -> i32 {\n")
	for _, st := range mainFn.Body.Stmts {
		if err := g.emitStmt(st); err != nil {
			fmt.Fprintf(&b, "    // lower err: %v\n", err)
			g.ok = false
			break
		}
	}
	if g.ok && mainFn.Body.Expr != nil {
		if _, err := g.emitExpr(mainFn.Body.Expr); err != nil {
			fmt.Fprintf(&b, "    // lower expr err: %v\n", err)
			g.ok = false
		}
	}
	if !g.ok {
		return "" // do not ship a broken main surface
	}
	b.WriteString("    %z = arith.constant 0 : i32\n")
	b.WriteString("    return %z : i32\n")
	b.WriteString("  }\n\n")

	return b.String()
}

type mlirVal struct {
	kind string // f32 | i32 | str | span | void
	ssa  string
}

type mlirFull struct {
	b         *strings.Builder
	n         int
	names     map[string]mlirVal
	spanElems map[string][]mlirVal
	ok        bool
}

func (g *mlirFull) fresh(pref string) string {
	g.n++
	return fmt.Sprintf("%%%s%d", pref, g.n)
}

func (g *mlirFull) emitStmt(st ast.Stmt) error {
	switch s := st.(type) {
	case *ast.LetStmt:
		v, err := g.emitExpr(s.Value)
		if err != nil {
			return err
		}
		// v0.47: let (a, b) = pair — best-effort single-name bind of first slot when span
		// Tracking is limited in MLIR full path; bind both names to value for name presence.
		if names := s.BindingNames(); len(names) >= 2 {
			g.names[names[0]] = v
			g.names[names[1]] = v
			return nil
		}
		g.names[s.Name] = v
		return nil
	case *ast.ExprStmt:
		_, err := g.emitExpr(s.Expr)
		return err
	case *ast.ReturnStmt:
		if s.Value == nil {
			return nil
		}
		_, err := g.emitExpr(s.Value)
		return err
	default:
		return fmt.Errorf("stmt %T", st)
	}
}

func (g *mlirFull) asF32(v mlirVal) mlirVal {
	if v.kind == "f32" {
		return v
	}
	if v.kind == "i32" {
		dst := g.fresh("f")
		fmt.Fprintf(g.b, "    %s = arith.sitofp %s : i32 to f32\n", dst, v.ssa)
		return mlirVal{"f32", dst}
	}
	dst := g.fresh("f")
	fmt.Fprintf(g.b, "    %s = arith.constant 0.000000e+00 : f32\n", dst)
	return mlirVal{"f32", dst}
}

func (g *mlirFull) emitExpr(e ast.Expr) (mlirVal, error) {
	switch n := e.(type) {
	case *ast.Literal:
		switch n.Kind {
		case "int":
			dst := g.fresh("i")
			fmt.Fprintf(g.b, "    %s = arith.constant %d : i32\n", dst, n.Value.(int))
			return mlirVal{"i32", dst}, nil
		case "float":
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "    %s = arith.constant %s : f32\n", dst, llvmF32(n.Value.(float64)))
			return mlirVal{"f32", dst}, nil
		case "bool":
			v := 0
			if n.Value.(bool) {
				v = 1
			}
			dst := g.fresh("i")
			fmt.Fprintf(g.b, "    %s = arith.constant %d : i32\n", dst, v)
			return mlirVal{"i32", dst}, nil
		case "string", "str":
			sv, ok := n.Value.(string)
			if !ok {
				return mlirVal{}, fmt.Errorf("str lit")
			}
			// string constants are notes; real materialization is on LLVM lower
			id := g.fresh("str")
			fmt.Fprintf(g.b, "    // str.lit %q → %s\n", sv, id)
			return mlirVal{"str", id}, nil
		default:
			return mlirVal{}, fmt.Errorf("lit %s", n.Kind)
		}
	case *ast.Var:
		if v, ok := g.names[n.Name]; ok {
			return v, nil
		}
		return mlirVal{}, fmt.Errorf("unknown %s", n.Name)
	case *ast.Unary:
		x, err := g.emitExpr(n.Expr)
		if err != nil {
			return mlirVal{}, err
		}
		if n.Op == "-" {
			if x.kind == "i32" {
				z := g.fresh("z")
				fmt.Fprintf(g.b, "    %s = arith.constant 0 : i32\n", z)
				dst := g.fresh("i")
				fmt.Fprintf(g.b, "    %s = arith.subi %s, %s : i32\n", dst, z, x.ssa)
				return mlirVal{"i32", dst}, nil
			}
			xf := g.asF32(x)
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "    %s = arith.negf %s : f32\n", dst, xf.ssa)
			return mlirVal{"f32", dst}, nil
		}
		return x, nil
	case *ast.Binary:
		l, err := g.emitExpr(n.Left)
		if err != nil {
			return mlirVal{}, err
		}
		r, err := g.emitExpr(n.Right)
		if err != nil {
			return mlirVal{}, err
		}
		if l.kind == "i32" && r.kind == "i32" {
			dst := g.fresh("i")
			op := "arith.addi"
			switch n.Op {
			case "-":
				op = "arith.subi"
			case "*":
				op = "arith.muli"
			case "/":
				op = "arith.divsi"
			case "+":
				op = "arith.addi"
			default:
				return mlirVal{}, fmt.Errorf("op %s", n.Op)
			}
			fmt.Fprintf(g.b, "    %s = %s %s, %s : i32\n", dst, op, l.ssa, r.ssa)
			return mlirVal{"i32", dst}, nil
		}
		lf, rf := g.asF32(l), g.asF32(r)
		dst := g.fresh("f")
		op := "arith.addf"
		switch n.Op {
		case "-":
			op = "arith.subf"
		case "*":
			op = "arith.mulf"
		case "/":
			op = "arith.divf"
		case "+":
			op = "arith.addf"
		default:
			return mlirVal{}, fmt.Errorf("op %s", n.Op)
		}
		fmt.Fprintf(g.b, "    %s = %s %s, %s : f32\n", dst, op, lf.ssa, rf.ssa)
		return mlirVal{"f32", dst}, nil
	case *ast.Ternary:
		w, err := g.emitExpr(n.Gate)
		if err != nil {
			return mlirVal{}, err
		}
		p, err := g.emitExpr(n.Pure)
		if err != nil {
			return mlirVal{}, err
		}
		ea, err := g.emitExpr(n.Each)
		if err != nil {
			return mlirVal{}, err
		}
		wf, pf, ef := g.asF32(w), g.asF32(p), g.asF32(ea)
		dst := g.fresh("f")
		fmt.Fprintf(g.b, "    %s = func.call @trin_soft_mux(%s, %s, %s) : (f32, f32, f32) -> f32\n",
			dst, wf.ssa, pf.ssa, ef.ssa)
		return mlirVal{"f32", dst}, nil
	case *ast.SpanLit:
		elems := make([]mlirVal, len(n.Elems))
		for i, el := range n.Elems {
			v, err := g.emitExpr(el)
			if err != nil {
				return mlirVal{}, err
			}
			elems[i] = g.asF32(v)
		}
		id := g.fresh("span")
		g.spanElems[id] = elems
		// annotate length
		fmt.Fprintf(g.b, "    // span %s len=%d\n", id, len(elems))
		return mlirVal{"span", id}, nil
	case *ast.Call:
		return g.emitCall(n)
	case *ast.Index:
		base, err := g.emitExpr(n.Target)
		if err != nil {
			return mlirVal{}, err
		}
		if base.kind != "span" {
			return mlirVal{}, fmt.Errorf("index non-span")
		}
		if lit, ok := n.Index.(*ast.Literal); ok && lit.Kind == "int" {
			i := lit.Value.(int)
			elems := g.spanElems[base.ssa]
			if i < 0 || i >= len(elems) {
				return mlirVal{}, fmt.Errorf("index OOB")
			}
			return elems[i], nil
		}
		return mlirVal{}, fmt.Errorf("dynamic index")
	default:
		return mlirVal{}, fmt.Errorf("unsupported %T", e)
	}
}

func (g *mlirFull) emitCall(n *ast.Call) (mlirVal, error) {
	v, ok := n.Callee.(*ast.Var)
	if !ok {
		return mlirVal{}, fmt.Errorf("call cal")
	}
	switch v.Name {
	case "soft_mux":
		if len(n.Args) != 3 {
			return mlirVal{}, fmt.Errorf("soft_mux arity")
		}
		w, err := g.emitExpr(n.Args[0])
		if err != nil {
			return mlirVal{}, err
		}
		p, err := g.emitExpr(n.Args[1])
		if err != nil {
			return mlirVal{}, err
		}
		e, err := g.emitExpr(n.Args[2])
		if err != nil {
			return mlirVal{}, err
		}
		wf, pf, ef := g.asF32(w), g.asF32(p), g.asF32(e)
		dst := g.fresh("f")
		fmt.Fprintf(g.b, "    %s = func.call @trin_soft_mux(%s, %s, %s) : (f32, f32, f32) -> f32\n",
			dst, wf.ssa, pf.ssa, ef.ssa)
		return mlirVal{"f32", dst}, nil
	case "to_str":
		if len(n.Args) != 1 {
			return mlirVal{}, fmt.Errorf("to_str arity")
		}
		x, err := g.emitExpr(n.Args[0])
		if err != nil {
			return mlirVal{}, err
		}
		dst := g.fresh("s")
		if x.kind == "i32" {
			fmt.Fprintf(g.b, "    // to_str.i32 %s → %s\n", x.ssa, dst)
		} else if x.kind == "str" {
			return x, nil
		} else {
			xf := g.asF32(x)
			fmt.Fprintf(g.b, "    %s = func.call @trin_to_str_f32(%s) : (f32) -> !llvm.ptr\n", dst, xf.ssa)
		}
		return mlirVal{"str", dst}, nil
	case "str_concat":
		if len(n.Args) != 2 {
			return mlirVal{}, fmt.Errorf("str_concat arity")
		}
		a, err := g.emitExpr(n.Args[0])
		if err != nil {
			return mlirVal{}, err
		}
		b, err := g.emitExpr(n.Args[1])
		if err != nil {
			return mlirVal{}, err
		}
		dst := g.fresh("s")
		fmt.Fprintf(g.b, "    // str_concat %s, %s → %s\n", a.ssa, b.ssa, dst)
		return mlirVal{"str", dst}, nil
	case "println", "print":
		if len(n.Args) != 1 {
			return mlirVal{}, fmt.Errorf("println arity")
		}
		s, err := g.emitExpr(n.Args[0])
		if err != nil {
			return mlirVal{}, err
		}
		if s.kind == "str" {
			fmt.Fprintf(g.b, "    func.call @trin_println_str(%s) : (!llvm.ptr) -> ()\n", s.ssa)
		} else {
			xf := g.asF32(s)
			tmp := g.fresh("s")
			fmt.Fprintf(g.b, "    %s = func.call @trin_to_str_f32(%s) : (f32) -> !llvm.ptr\n", tmp, xf.ssa)
			fmt.Fprintf(g.b, "    func.call @trin_println_str(%s) : (!llvm.ptr) -> ()\n", tmp)
		}
		return mlirVal{"void", ""}, nil
	case "sum":
		if len(n.Args) != 1 {
			return mlirVal{}, fmt.Errorf("sum arity")
		}
		sp, err := g.emitExpr(n.Args[0])
		if err != nil {
			return mlirVal{}, err
		}
		if sp.kind != "span" {
			return mlirVal{}, fmt.Errorf("sum non-span")
		}
		elems := g.spanElems[sp.ssa]
		if len(elems) == 0 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "    %s = arith.constant 0.000000e+00 : f32\n", dst)
			return mlirVal{"f32", dst}, nil
		}
		// v0.32: note dynamic tile path for fixed 4/8/16; LLVM full-program
		// actually dispatches via @trin_dyn_sum*. Surface still uses arith
		// for portable MLIR interpretability, plus a dyn-tile marker call.
		if preferDynTileN(len(elems)) {
			nI := g.fresh("i")
			t := g.fresh("i")
			fmt.Fprintf(g.b, "    // rewrite: full-program dyn_tile sum(n=%d) → @trin_dyn_sum%d\n",
				len(elems), len(elems))
			fmt.Fprintf(g.b, "    %s = arith.constant %d : i32\n", nI, len(elems))
			fmt.Fprintf(g.b, "    %s = func.call @trin_hw_vlen_for_n(%s) : (i32) -> i32\n", t, nI)
			// keep value-producing chain for MLIR correctness probes
		}
		// v0.34 residual peel / v0.36 auto peel sibling notes for non-fixed lengths.
		if preferResidualPeel(len(elems)) {
			tiles, residual := peelDynTileChunks(len(elems))
			nI := g.fresh("i")
			t := g.fresh("i")
			fmt.Fprintf(g.b, "    // rewrite: residual peel sum(n=%d) tiles=%v residual=%d → @trin_dyn_sum* + scalar (v0.34/v0.36)\n",
				len(elems), tiles, residual)
			fmt.Fprintf(g.b, "    %s = arith.constant %d : i32\n", nI, len(elems))
			fmt.Fprintf(g.b, "    %s = func.call @trin_hw_vlen_for_n(%s) : (i32) -> i32\n", t, nI)
		}
		// Horner-style add chain for portable MLIR (matches fixed-size demos)
		acc := elems[0]
		for i := 1; i < len(elems); i++ {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "    %s = arith.addf %s, %s : f32\n", dst, acc.ssa, elems[i].ssa)
			acc = mlirVal{"f32", dst}
		}
		return acc, nil
	case "dot":
		if len(n.Args) != 2 {
			return mlirVal{}, fmt.Errorf("dot arity")
		}
		a, err := g.emitExpr(n.Args[0])
		if err != nil {
			return mlirVal{}, err
		}
		b, err := g.emitExpr(n.Args[1])
		if err != nil {
			return mlirVal{}, err
		}
		if a.kind != "span" || b.kind != "span" {
			return mlirVal{}, fmt.Errorf("dot non-span")
		}
		ae, be := g.spanElems[a.ssa], g.spanElems[b.ssa]
		n0 := len(ae)
		if len(be) < n0 {
			n0 = len(be)
		}
		if n0 == 0 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "    %s = arith.constant 0.000000e+00 : f32\n", dst)
			return mlirVal{"f32", dst}, nil
		}
		if preferDynTileN(n0) {
			nI := g.fresh("i")
			t := g.fresh("i")
			fmt.Fprintf(g.b, "    // rewrite: full-program dyn_tile dot(n=%d) → @trin_dyn_dot%d\n", n0, n0)
			fmt.Fprintf(g.b, "    %s = arith.constant %d : i32\n", nI, n0)
			fmt.Fprintf(g.b, "    %s = func.call @trin_hw_vlen_for_n(%s) : (i32) -> i32\n", t, nI)
		}
		if preferResidualPeel(n0) {
			tiles, residual := peelDynTileChunks(n0)
			nI := g.fresh("i")
			t := g.fresh("i")
			fmt.Fprintf(g.b, "    // rewrite: residual peel dot(n=%d) tiles=%v residual=%d (v0.34/v0.36)\n",
				n0, tiles, residual)
			fmt.Fprintf(g.b, "    %s = arith.constant %d : i32\n", nI, n0)
			fmt.Fprintf(g.b, "    %s = func.call @trin_hw_vlen_for_n(%s) : (i32) -> i32\n", t, nI)
		}
		p0 := g.fresh("f")
		fmt.Fprintf(g.b, "    %s = arith.mulf %s, %s : f32\n", p0, ae[0].ssa, be[0].ssa)
		acc := mlirVal{"f32", p0}
		for i := 1; i < n0; i++ {
			p := g.fresh("f")
			fmt.Fprintf(g.b, "    %s = arith.mulf %s, %s : f32\n", p, ae[i].ssa, be[i].ssa)
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "    %s = arith.addf %s, %s : f32\n", dst, acc.ssa, p)
			acc = mlirVal{"f32", dst}
		}
		return acc, nil
	case "span_add":
		if len(n.Args) != 2 {
			return mlirVal{}, fmt.Errorf("span_add arity")
		}
		a, err := g.emitExpr(n.Args[0])
		if err != nil {
			return mlirVal{}, err
		}
		b, err := g.emitExpr(n.Args[1])
		if err != nil {
			return mlirVal{}, err
		}
		if a.kind != "span" || b.kind != "span" {
			return mlirVal{}, fmt.Errorf("span_add non-span")
		}
		ae, be := g.spanElems[a.ssa], g.spanElems[b.ssa]
		n0 := len(ae)
		if len(be) < n0 {
			n0 = len(be)
		}
		out := make([]mlirVal, n0)
		for i := 0; i < n0; i++ {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "    %s = arith.addf %s, %s : f32\n", dst, ae[i].ssa, be[i].ssa)
			out[i] = mlirVal{"f32", dst}
		}
		id := g.fresh("span")
		g.spanElems[id] = out
		return mlirVal{"span", id}, nil
	case "scale", "span_scale", "map_mul":
		if len(n.Args) != 2 {
			return mlirVal{}, fmt.Errorf("scale arity")
		}
		s, err := g.emitExpr(n.Args[0])
		if err != nil {
			return mlirVal{}, err
		}
		a, err := g.emitExpr(n.Args[1])
		if err != nil {
			return mlirVal{}, err
		}
		// also accept scale(span, s) order from some demos — detect span position
		sf := s
		sp := a
		if s.kind == "span" {
			sf = a
			sp = s
		}
		if sp.kind != "span" {
			return mlirVal{}, fmt.Errorf("scale non-span")
		}
		sc := g.asF32(sf)
		elems := g.spanElems[sp.ssa]
		out := make([]mlirVal, len(elems))
		for i, el := range elems {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "    %s = arith.mulf %s, %s : f32\n", dst, sc.ssa, el.ssa)
			out[i] = mlirVal{"f32", dst}
		}
		id := g.fresh("span")
		g.spanElems[id] = out
		return mlirVal{"span", id}, nil
	case "simd_lane_width":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "    %s = arith.constant 128 : i32\n", dst)
		return mlirVal{"i32", dst}, nil
	case "hw_vlen":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "    %s = func.call @trin_hw_vlen() : () -> i32\n", dst)
		return mlirVal{"i32", dst}, nil
	case "hw_vlen_for_n", "dyn_tile":
		if len(n.Args) != 1 {
			return mlirVal{}, fmt.Errorf("%s arity", v.Name)
		}
		av, err := g.emitExpr(n.Args[0])
		if err != nil {
			return mlirVal{}, err
		}
		arg := av
		if av.kind != "i32" {
			xf := g.asF32(av)
			tmp := g.fresh("i")
			fmt.Fprintf(g.b, "    %s = arith.fptosi %s : f32 to i32\n", tmp, xf.ssa)
			arg = mlirVal{"i32", tmp}
		}
		dst := g.fresh("i")
		if v.Name == "dyn_tile" {
			fmt.Fprintf(g.b, "    %s = func.call @trin_dyn_tile(%s) : (i32) -> i32\n", dst, arg.ssa)
		} else {
			fmt.Fprintf(g.b, "    %s = func.call @trin_hw_vlen_for_n(%s) : (i32) -> i32\n", dst, arg.ssa)
		}
		return mlirVal{"i32", dst}, nil
	case "hw_isa_code":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "    %s = func.call @trin_hw_isa_code() : () -> i32\n", dst)
		return mlirVal{"i32", dst}, nil
	default:
		return mlirVal{}, fmt.Errorf("call %s", v.Name)
	}
}
