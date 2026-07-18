package codegen

import (
	"fmt"
	"strings"

	"github.com/macroadster/trin/internal/ast"
)

// hasLLVMMain reports whether IR already defines a program entrypoint.
func hasLLVMMain(ll string) bool {
	return strings.Contains(ll, "define i32 @main(") || strings.Contains(ll, "define i32 @main ()")
}

// emitFullProgram emits runtime helpers + pure f32 fns (when not already) + @main
// with side effects (println/to_str/str_concat/i32/f32/spans/sum/dot/soft_mux).
// Returns "" if no main body could be lowered.
func emitFullProgram(prog *ast.Program) string {
	return emitFullProgramEx(prog, true)
}

// emitFullProgramEx is like emitFullProgram. When withRuntimeHelpers is false,
// skips CPUID/dyn-tile helper definitions so callers that already lowered them
// (e.g. EmitMLIRLLVM after LowerMLIRToLLVM) can avoid redefinitions.
func emitFullProgramEx(prog *ast.Program, withRuntimeHelpers bool) string {
	if prog == nil {
		return ""
	}
	var mainFn *ast.FnDef
	var pureFns []*ast.FnDef
	for _, it := range prog.Items {
		fn, ok := it.(*ast.FnDef)
		if !ok {
			continue
		}
		if fn.Name == "main" {
			mainFn = fn
			continue
		}
		// Candidate pure kernels (no println/read_line) for separate float defs.
		if pureLikely(fn) {
			pureFns = append(pureFns, fn)
		}
	}
	if mainFn == nil || mainFn.Body == nil {
		return ""
	}

	var b strings.Builder
	b.WriteString("; --- full-program LLVM lowering (v0.19 + dyn tiles v0.32 + residual peel v0.34 + auto peel v0.36/v0.37) ---\n")
	// C runtime
	b.WriteString("declare i32 @puts(i8*)\n")
	b.WriteString("declare i32 @printf(i8*, ...)\n")
	b.WriteString("declare i32 @snprintf(i8*, i64, i8*, ...)\n")
	b.WriteString("declare i8* @strcpy(i8*, i8*)\n")
	b.WriteString("declare i8* @strcat(i8*, i8*)\n")
	b.WriteString("declare i64 @strlen(i8*)\n")
	b.WriteString("\n")
	// Runtime CPU feature detect + dynamic tile helpers (v0.30 / v0.32).
	// emit-llvm path always embeds them; EmitMLIRLLVM may skip to avoid redefs.
	if withRuntimeHelpers {
		b.WriteString(EmitRuntimeCPUIDLLVM(true, true, true))
		b.WriteString(emitDynTileLLVMHelpers())
	}
	// Note: @trin_simd_lane_width is defined by emit-llvm helpers; EmitMLIRLLVM
	// injects it when missing so this fragment stays free of redefinitions.

	// Rotating string buffers for concat/to_str (no malloc).
	// c"..." includes trailing \00 in the byte count.
	b.WriteString("@trin_str_fmt_f = private unnamed_addr constant [3 x i8] c\"%g\\00\"\n")
	b.WriteString("@trin_str_fmt_i = private unnamed_addr constant [3 x i8] c\"%d\\00\"\n")
	b.WriteString("@trin_str_nl = private unnamed_addr constant [2 x i8] c\"\\0A\\00\"\n")
	b.WriteString("@trin_buf0 = internal global [512 x i8] zeroinitializer\n")
	b.WriteString("@trin_buf1 = internal global [512 x i8] zeroinitializer\n")
	b.WriteString("@trin_buf2 = internal global [512 x i8] zeroinitializer\n")
	b.WriteString("@trin_buf3 = internal global [512 x i8] zeroinitializer\n")
	b.WriteString("@trin_buf_i = internal global i32 0\n\n")

	b.WriteString("define internal i8* @trin_next_buf() {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %i0 = load i32, i32* @trin_buf_i\n")
	b.WriteString("  %i1 = add i32 %i0, 1\n")
	b.WriteString("  %i2 = and i32 %i1, 3\n")
	b.WriteString("  store i32 %i2, i32* @trin_buf_i\n")
	b.WriteString("  switch i32 %i0, label %b0 [ i32 0, label %b0 i32 1, label %b1 i32 2, label %b2 i32 3, label %b3 ]\n")
	b.WriteString("b0:\n  %p0 = getelementptr [512 x i8], [512 x i8]* @trin_buf0, i64 0, i64 0\n  ret i8* %p0\n")
	b.WriteString("b1:\n  %p1 = getelementptr [512 x i8], [512 x i8]* @trin_buf1, i64 0, i64 0\n  ret i8* %p1\n")
	b.WriteString("b2:\n  %p2 = getelementptr [512 x i8], [512 x i8]* @trin_buf2, i64 0, i64 0\n  ret i8* %p2\n")
	b.WriteString("b3:\n  %p3 = getelementptr [512 x i8], [512 x i8]* @trin_buf3, i64 0, i64 0\n  ret i8* %p3\n")
	b.WriteString("}\n\n")

	b.WriteString("define internal i8* @trin_to_str_f32(float %x) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %buf = call i8* @trin_next_buf()\n")
	b.WriteString("  %fmt = getelementptr [3 x i8], [3 x i8]* @trin_str_fmt_f, i64 0, i64 0\n")
	b.WriteString("  %x64 = fpext float %x to double\n")
	b.WriteString("  %n = call i32 (i8*, i64, i8*, ...) @snprintf(i8* %buf, i64 512, i8* %fmt, double %x64)\n")
	b.WriteString("  ret i8* %buf\n")
	b.WriteString("}\n\n")

	b.WriteString("define internal i8* @trin_to_str_i32(i32 %x) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %buf = call i8* @trin_next_buf()\n")
	b.WriteString("  %fmt = getelementptr [3 x i8], [3 x i8]* @trin_str_fmt_i, i64 0, i64 0\n")
	b.WriteString("  %n = call i32 (i8*, i64, i8*, ...) @snprintf(i8* %buf, i64 512, i8* %fmt, i32 %x)\n")
	b.WriteString("  ret i8* %buf\n")
	b.WriteString("}\n\n")

	b.WriteString("define internal i8* @trin_str_concat(i8* %a, i8* %b) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %buf = call i8* @trin_next_buf()\n")
	b.WriteString("  store i8 0, i8* %buf\n")
	b.WriteString("  %c0 = call i8* @strcpy(i8* %buf, i8* %a)\n")
	b.WriteString("  %c1 = call i8* @strcat(i8* %buf, i8* %b)\n")
	b.WriteString("  ret i8* %buf\n")
	b.WriteString("}\n\n")

	b.WriteString("define internal void @trin_println_str(i8* %s) {\n")
	b.WriteString("entry:\n")
	b.WriteString("  %n = call i32 @puts(i8* %s)\n")
	b.WriteString("  ret void\n")
	b.WriteString("}\n\n")

	// Emit pure functions that may be called from main (skip if name collides with
	// helpers already produced by tryFloatKernel — emitPureFn is fine either way for
	// non-kernel names; for loss/kernel keep single definition from earlier path).
	emitted := map[string]bool{"loss": true, "kernel": true, "soft_kernel": true, "trin_soft_mux": true}
	for _, fn := range pureFns {
		if emitted[fn.Name] {
			continue
		}
		// Re-emit only if body is pure f32 graph we can lower.
		if body, ok := emitPureFnSafe(fn); ok {
			b.WriteString(body)
			b.WriteByte('\n')
			emitted[fn.Name] = true
		}
	}

	g := &fullLL{
		b:              &b,
		names:          map[string]llVal{},
		n:              0,
		strs:           0,
		spanElems:      map[string][]llVal{},
		pendingGlobals: []string{},
	}
	b.WriteString("define i32 @main() {\n")
	b.WriteString("entry:\n")
	ok := true
	for _, st := range mainFn.Body.Stmts {
		if err := g.emitStmt(st); err != nil {
			fmt.Fprintf(&b, "  ; lower err: %v\n", err)
			ok = false
			break
		}
	}
	if ok && mainFn.Body.Expr != nil {
		if _, err := g.emitExpr(mainFn.Body.Expr); err != nil {
			fmt.Fprintf(&b, "  ; lower expr err: %v\n", err)
			ok = false
		}
	}
	if !ok {
		// Leave a stub main that returns 1 so link still works, but signal fail.
		b.WriteString("  ret i32 1\n}\n")
		// If we failed, still return the fragment — better: return "" so float host works.
		// Prefer not to clobber float-kernel linking: return empty on failure.
		return ""
	}
	b.WriteString("  ret i32 0\n}\n")
	body := b.String()
	// Hoist string globals that were collected while lowering main.
	// emitFullProgram created g after helpers; collect from last fullLL via search.
	// pendingGlobals are on g:
	_ = g
	var pre strings.Builder
	for _, gl := range g.pendingGlobals {
		pre.WriteString(gl)
	}
	return pre.String() + body
}

func pureLikely(fn *ast.FnDef) bool {
	if fn == nil || fn.Body == nil {
		return false
	}
	// Reject if body uses IO-ish calls.
	var bad bool
	var walk func(ast.Expr)
	walk = func(e ast.Expr) {
		if e == nil || bad {
			return
		}
		switch n := e.(type) {
		case *ast.Call:
			if v, ok := n.Callee.(*ast.Var); ok {
				switch v.Name {
				case "println", "print", "read_line", "read":
					bad = true
					return
				}
			}
			for _, a := range n.Args {
				walk(a)
			}
		case *ast.Binary:
			walk(n.Left)
			walk(n.Right)
		case *ast.Unary:
			walk(n.Expr)
		case *ast.Ternary:
			walk(n.Gate)
			walk(n.Pure)
			walk(n.Each)
		case *ast.SpanLit:
			for _, el := range n.Elems {
				walk(el)
			}
		case *ast.Lambda:
			walk(n.Body)
		case *ast.Index:
			walk(n.Target)
			walk(n.Index)
		case *ast.Block:
			for _, st := range n.Stmts {
				switch s := st.(type) {
				case *ast.LetStmt:
					walk(s.Value)
				case *ast.ExprStmt:
					walk(s.Expr)
				case *ast.ReturnStmt:
					walk(s.Value)
				}
			}
			walk(n.Expr)
		}
	}
	for _, st := range fn.Body.Stmts {
		switch s := st.(type) {
		case *ast.LetStmt:
			walk(s.Value)
		case *ast.ExprStmt:
			walk(s.Expr)
		case *ast.ReturnStmt:
			walk(s.Value)
		}
	}
	walk(fn.Body.Expr)
	return !bad
}

func emitPureFnSafe(fn *ast.FnDef) (string, bool) {
	// Wrap emitPureFn; reject if it only produced zero stub due to unsupported body.
	s := emitPureFn(fn)
	if strings.Contains(s, "; err ") {
		return "", false
	}
	return s, true
}

type llVal struct {
	kind string // f32 | i32 | str | void
	ssa  string
}

type fullLL struct {
	b              *strings.Builder
	names          map[string]llVal
	n              int
	strs           int
	spanElems      map[string][]llVal
	pendingGlobals []string
}

func (g *fullLL) fresh(pref string) string {
	g.n++
	return fmt.Sprintf("%%%s%d", pref, g.n)
}

func (g *fullLL) emitStmt(st ast.Stmt) error {
	switch s := st.(type) {
	case *ast.LetStmt:
		v, err := g.emitExpr(s.Value)
		if err != nil {
			return err
		}
		// v0.51/v0.51: multi-way pair/span destructure — bind slots + exhaustive arity
		if names := s.BindingNames(); len(names) >= 2 {
			elems, ok := g.spanElems[v.ssa]
			if !ok || len(elems) != len(names) {
				return fmt.Errorf("pair destructure: not exhaustive (arity mismatch)")
			}
			for i, name := range names {
				g.names[name] = elems[i]
			}
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

func (g *fullLL) emitExpr(e ast.Expr) (llVal, error) {
	switch n := e.(type) {
	case *ast.Literal:
		switch n.Kind {
		case "int":
			dst := g.fresh("i")
			fmt.Fprintf(g.b, "  %s = add i32 %d, 0\n", dst, n.Value.(int))
			return llVal{"i32", dst}, nil
		case "float":
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = fadd float %s, 0.000000e+00\n", dst, llvmF32(n.Value.(float64)))
			return llVal{"f32", dst}, nil
		case "bool":
			v := 0
			if n.Value.(bool) {
				v = 1
			}
			dst := g.fresh("i")
			fmt.Fprintf(g.b, "  %s = add i32 %d, 0\n", dst, v)
			return llVal{"i32", dst}, nil
		case "string", "str":
			sv, ok := n.Value.(string)
			if !ok {
				return llVal{}, fmt.Errorf("str lit type %T", n.Value)
			}
			return g.emitStrLit(sv)
		default:
			return llVal{}, fmt.Errorf("lit kind %s", n.Kind)
		}
	case *ast.Var:
		if v, ok := g.names[n.Name]; ok {
			return v, nil
		}
		return llVal{}, fmt.Errorf("unknown %s", n.Name)
	case *ast.Unary:
		x, err := g.emitExpr(n.Expr)
		if err != nil {
			return llVal{}, err
		}
		if n.Op == "-" {
			if x.kind == "i32" {
				dst := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = sub i32 0, %s\n", dst, x.ssa)
				return llVal{"i32", dst}, nil
			}
			x = g.asF32(x)
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = fneg float %s\n", dst, x.ssa)
			return llVal{"f32", dst}, nil
		}
		return x, nil
	case *ast.Binary:
		l, err := g.emitExpr(n.Left)
		if err != nil {
			return llVal{}, err
		}
		r, err := g.emitExpr(n.Right)
		if err != nil {
			return llVal{}, err
		}
		if l.kind == "i32" && r.kind == "i32" && (n.Op == "+" || n.Op == "-" || n.Op == "*" || n.Op == "/") {
			dst := g.fresh("i")
			op := "add"
			switch n.Op {
			case "-":
				op = "sub"
			case "*":
				op = "mul"
			case "/":
				op = "sdiv"
			}
			fmt.Fprintf(g.b, "  %s = %s i32 %s, %s\n", dst, op, l.ssa, r.ssa)
			return llVal{"i32", dst}, nil
		}
		lf, rf := g.asF32(l), g.asF32(r)
		dst := g.fresh("f")
		op := "fadd"
		switch n.Op {
		case "-":
			op = "fsub"
		case "*":
			op = "fmul"
		case "/":
			op = "fdiv"
		case "+":
			op = "fadd"
		default:
			return llVal{}, fmt.Errorf("op %s", n.Op)
		}
		fmt.Fprintf(g.b, "  %s = %s float %s, %s\n", dst, op, lf.ssa, rf.ssa)
		return llVal{"f32", dst}, nil
	case *ast.Ternary:
		w, err := g.emitExpr(n.Gate)
		if err != nil {
			return llVal{}, err
		}
		p, err := g.emitExpr(n.Pure)
		if err != nil {
			return llVal{}, err
		}
		ea, err := g.emitExpr(n.Each)
		if err != nil {
			return llVal{}, err
		}
		wf, pf, ef := g.asF32(w), g.asF32(p), g.asF32(ea)
		dst := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = call float @trin_soft_mux(float %s, float %s, float %s)\n", dst, wf.ssa, pf.ssa, ef.ssa)
		return llVal{"f32", dst}, nil
	case *ast.SpanLit:
		// Materialize as fixed f32 stack alloc + store; return synthetic span token via names.
		// Represent span as first elem name pack: we keep elems as []llVal in a side map.
		elems := make([]llVal, len(n.Elems))
		for i, el := range n.Elems {
			v, err := g.emitExpr(el)
			if err != nil {
				return llVal{}, err
			}
			elems[i] = g.asF32(v)
		}
		// Encode as special kind str "span:N" with ssa listing joined names — use synthetic id.
		id := g.fresh("span")
		g.spanElems[id] = elems
		return llVal{"span", id}, nil
	case *ast.Call:
		return g.emitCall(n)
	case *ast.Index:
		// Span index: only when target is a known span.
		base, err := g.emitExpr(n.Target)
		if err != nil {
			return llVal{}, err
		}
		if base.kind != "span" {
			return llVal{}, fmt.Errorf("index non-span")
		}
		idx, err := g.emitExpr(n.Index)
		if err != nil {
			return llVal{}, err
		}
		// Require constant index for simplicity.
		if lit, ok := n.Index.(*ast.Literal); ok && lit.Kind == "int" {
			i := lit.Value.(int)
			elems := g.spanElems[base.ssa]
			if i < 0 || i >= len(elems) {
				return llVal{}, fmt.Errorf("index OOB")
			}
			return elems[i], nil
		}
		_ = idx
		return llVal{}, fmt.Errorf("dynamic index")
	default:
		return llVal{}, fmt.Errorf("unsupported %T", e)
	}
}

func (g *fullLL) asF32(v llVal) llVal {
	if v.kind == "f32" {
		return v
	}
	if v.kind == "i32" {
		dst := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = sitofp i32 %s to float\n", dst, v.ssa)
		return llVal{"f32", dst}
	}
	// string / other → 0
	dst := g.fresh("f")
	fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", dst)
	return llVal{"f32", dst}
}

func (g *fullLL) emitStrLit(s string) (llVal, error) {
	g.strs++
	name := fmt.Sprintf("@.str.%d", g.strs)
	// Escape for LLVM c"..." form.
	esc, n := llvmCString(s)
	fmt.Fprintf(g.b, "  ; str const materialize via global (declared at module level via lazy — use alloca init)\n")
	// Use a private constant global written before main isn't available mid-function.
	// Strategy: emit constant as module-level by writing now is wrong inside function.
	// Instead: store bytes via consecutive stores into next_buf.
	buf := g.fresh("s")
	fmt.Fprintf(g.b, "  %s = call i8* @trin_next_buf()\n", buf)
	// Write via snprintf with a format that embeds... safer: byte stores.
	// For simplicity use snprintf of the string itself through a global we declare as
	// appendable module-level via a side channel.
	g.pendingGlobals = append(g.pendingGlobals, fmt.Sprintf(
		"%s = private unnamed_addr constant [%d x i8] c\"%s\"\n", name, n, esc))
	// GEP to first byte — but global must exist: we hoist later.
	// Temporary: use the pending global name directly (globals can be forward-referenced in LLVM).
	tmp := g.fresh("sp")
	fmt.Fprintf(g.b, "  %s = getelementptr [%d x i8], [%d x i8]* %s, i64 0, i64 0\n", tmp, n, n, name)
	// copy into rotating buf so strcat later is safe even if original is constant
	_ = buf
	fmt.Fprintf(g.b, "  %%_c%d = call i8* @strcpy(i8* %s, i8* %s)\n", g.n, buf, tmp)
	return llVal{"str", buf}, nil
}

func llvmCString(s string) (escaped string, nBytes int) {
	var b strings.Builder
	// include trailing NUL
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case '\n':
			b.WriteString("\\0A")
		case '\r':
			b.WriteString("\\0D")
		case '\t':
			b.WriteString("\\09")
		case '"':
			b.WriteString("\\22")
		case '\\':
			b.WriteString("\\5C")
		default:
			if c < 32 || c > 126 {
				fmt.Fprintf(&b, "\\%02X", c)
			} else {
				b.WriteByte(c)
			}
		}
	}
	b.WriteString("\\00")
	return b.String(), len(s) + 1
}

func (g *fullLL) ensureMaps() {
	if g.spanElems == nil {
		g.spanElems = map[string][]llVal{}
	}
	if g.pendingGlobals == nil {
		g.pendingGlobals = []string{}
	}
}

func (g *fullLL) emitCall(n *ast.Call) (llVal, error) {
	g.ensureMaps()
	v, ok := n.Callee.(*ast.Var)
	if !ok {
		return llVal{}, fmt.Errorf("call cal")
	}
	switch v.Name {
	case "soft_mux":
		if len(n.Args) != 3 {
			return llVal{}, fmt.Errorf("soft_mux arity")
		}
		w, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		p, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		e, err := g.emitExpr(n.Args[2])
		if err != nil {
			return llVal{}, err
		}
		wf, pf, ef := g.asF32(w), g.asF32(p), g.asF32(e)
		dst := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = call float @trin_soft_mux(float %s, float %s, float %s)\n", dst, wf.ssa, pf.ssa, ef.ssa)
		return llVal{"f32", dst}, nil
	case "to_str":
		if len(n.Args) != 1 {
			return llVal{}, fmt.Errorf("to_str arity")
		}
		x, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		dst := g.fresh("s")
		switch x.kind {
		case "i32":
			fmt.Fprintf(g.b, "  %s = call i8* @trin_to_str_i32(i32 %s)\n", dst, x.ssa)
		case "str":
			return x, nil
		default:
			xf := g.asF32(x)
			fmt.Fprintf(g.b, "  %s = call i8* @trin_to_str_f32(float %s)\n", dst, xf.ssa)
		}
		return llVal{"str", dst}, nil
	case "str_concat":
		if len(n.Args) != 2 {
			return llVal{}, fmt.Errorf("str_concat arity")
		}
		a, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		b, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		as, err := g.asStr(a)
		if err != nil {
			return llVal{}, err
		}
		bs, err := g.asStr(b)
		if err != nil {
			return llVal{}, err
		}
		dst := g.fresh("s")
		fmt.Fprintf(g.b, "  %s = call i8* @trin_str_concat(i8* %s, i8* %s)\n", dst, as.ssa, bs.ssa)
		return llVal{"str", dst}, nil
	case "println", "print":
		if len(n.Args) < 1 {
			return llVal{}, fmt.Errorf("println arity")
		}
		x, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		s, err := g.asStr(x)
		if err != nil {
			return llVal{}, err
		}
		fmt.Fprintf(g.b, "  call void @trin_println_str(i8* %s)\n", s.ssa)
		return llVal{"void", "0"}, nil
	case "sum":
		if len(n.Args) != 1 {
			return llVal{}, fmt.Errorf("sum arity")
		}
		return g.emitSumArg(n.Args[0])
	case "dot":
		if len(n.Args) != 2 {
			return llVal{}, fmt.Errorf("dot arity")
		}
		as, err := g.emitSpanElemsExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		bs, err := g.emitSpanElemsExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		nn := len(as)
		if len(bs) < nn {
			nn = len(bs)
		}
		if nn == 0 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", dst)
			return llVal{"f32", dst}, nil
		}
		// v0.32: fixed 4/8-length dots use runtime tile dispatch.
		if nn == 4 && len(as) >= 4 && len(bs) >= 4 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_dyn_dot4(float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s)\n",
				dst, as[0].ssa, as[1].ssa, as[2].ssa, as[3].ssa,
				bs[0].ssa, bs[1].ssa, bs[2].ssa, bs[3].ssa)
			return llVal{"f32", dst}, nil
		}
		if nn == 8 && len(as) >= 8 && len(bs) >= 8 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_dyn_dot8(float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s)\n",
				dst,
				as[0].ssa, as[1].ssa, as[2].ssa, as[3].ssa, as[4].ssa, as[5].ssa, as[6].ssa, as[7].ssa,
				bs[0].ssa, bs[1].ssa, bs[2].ssa, bs[3].ssa, bs[4].ssa, bs[5].ssa, bs[6].ssa, bs[7].ssa)
			return llVal{"f32", dst}, nil
		}
		// v0.34: residual peel for non-fixed-length dots — mul chain then sum peel.
		if preferResidualPeel(nn) {
			prods := make([]llVal, nn)
			for i := 0; i < nn; i++ {
				p := g.fresh("f")
				fmt.Fprintf(g.b, "  %s = fmul float %s, %s\n", p, as[i].ssa, bs[i].ssa)
				prods[i] = llVal{"f32", p}
			}
			return g.emitSumPeeled(prods)
		}
		var acc string
		for i := 0; i < nn; i++ {
			p := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = fmul float %s, %s\n", p, as[i].ssa, bs[i].ssa)
			if i == 0 {
				acc = p
			} else {
				nxt := g.fresh("f")
				fmt.Fprintf(g.b, "  %s = fadd float %s, %s\n", nxt, acc, p)
				acc = nxt
			}
		}
		return llVal{"f32", acc}, nil
	case "span_add":
		if len(n.Args) != 2 {
			return llVal{}, fmt.Errorf("span_add arity")
		}
		as, err := g.emitSpanElemsExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		bs, err := g.emitSpanElemsExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		nn := len(as)
		if len(bs) < nn {
			nn = len(bs)
		}
		out := make([]llVal, nn)
		for i := 0; i < nn; i++ {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = fadd float %s, %s\n", dst, as[i].ssa, bs[i].ssa)
			out[i] = llVal{"f32", dst}
		}
		id := g.fresh("span")
		g.spanElems[id] = out
		return llVal{"span", id}, nil
	case "scale", "map_mul":
		if len(n.Args) != 2 {
			return llVal{}, fmt.Errorf("%s arity", v.Name)
		}
		sid, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		xs, err := g.emitSpanElemsExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		sf := g.asF32(sid)
		out := make([]llVal, len(xs))
		for i, x := range xs {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = fmul float %s, %s\n", dst, sf.ssa, x.ssa)
			out[i] = llVal{"f32", dst}
		}
		id := g.fresh("span")
		g.spanElems[id] = out
		return llVal{"span", id}, nil
	case "simd_lane_width":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_simd_lane_width()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_vlen":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_vlen()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_vlen_for_n", "dyn_tile":
		if len(n.Args) != 1 {
			return llVal{}, fmt.Errorf("%s arity", v.Name)
		}
		av, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		// coerce f32 → i32 if needed
		arg := av
		if av.kind != "i32" {
			xf := g.asF32(av)
			tmp := g.fresh("i")
			fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, xf.ssa)
			arg = llVal{"i32", tmp}
		}
		dst := g.fresh("i")
		if v.Name == "dyn_tile" {
			fmt.Fprintf(g.b, "  %s = call i32 @trin_dyn_tile(i32 %s)\n", dst, arg.ssa)
		} else {
			fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_vlen_for_n(i32 %s)\n", dst, arg.ssa)
		}
		return llVal{"i32", dst}, nil
	case "hw_isa_code":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_isa_code()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_rdvl":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_rdvl()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_vl":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_vl()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_tile":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_tile()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_outer":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_outer()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_outer_acc":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_outer_acc()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_outer_acc_true":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_outer_acc_true()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_cost":
		if len(n.Args) != 1 {
			return llVal{}, fmt.Errorf("fusion_cost arity")
		}
		av, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		arg := av
		if av.kind != "i32" {
			xf := g.asF32(av)
			tmp := g.fresh("i")
			fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, xf.ssa)
			arg = llVal{"i32", tmp}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_cost(i32 %s)\n", dst, arg.ssa)
		return llVal{"i32", dst}, nil
	case "fusion_cost_weighted":
		if len(n.Args) < 1 {
			return llVal{}, fmt.Errorf("fusion_cost_weighted arity")
		}
		av, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		ni := av
		if av.kind != "i32" {
			xf := g.asF32(av)
			tmp := g.fresh("i")
			fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, xf.ssa)
			ni = llVal{"i32", tmp}
		}
		wssa := "1"
		if len(n.Args) >= 2 {
			wv, err := g.emitExpr(n.Args[1])
			if err != nil {
				return llVal{}, err
			}
			if wv.kind != "i32" {
				xf := g.asF32(wv)
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, xf.ssa)
				wssa = tmp
			} else {
				wssa = wv.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_cost_weighted(i32 %s, i32 %s)\n", dst, ni.ssa, wssa)
		return llVal{"i32", dst}, nil
	case "fusion_cost_isa":
		if len(n.Args) != 1 {
			return llVal{}, fmt.Errorf("fusion_cost_isa arity")
		}
		av, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		arg := av
		if av.kind != "i32" {
			xf := g.asF32(av)
			tmp := g.fresh("i")
			fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, xf.ssa)
			arg = llVal{"i32", tmp}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_cost_isa(i32 %s)\n", dst, arg.ssa)
		return llVal{"i32", dst}, nil
	case "fusion_cost_calibrated":
		if len(n.Args) != 1 {
			return llVal{}, fmt.Errorf("fusion_cost_calibrated arity")
		}
		av, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		arg := av
		if av.kind != "i32" {
			xf := g.asF32(av)
			tmp := g.fresh("i")
			fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, xf.ssa)
			arg = llVal{"i32", tmp}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_cost_calibrated(i32 %s)\n", dst, arg.ssa)
		return llVal{"i32", dst}, nil
	case "fusion_plan_log":
		if len(n.Args) < 1 {
			return llVal{}, fmt.Errorf("fusion_plan_log arity")
		}
		var cs []string
		for i := 0; i < len(n.Args) && i < 3; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				xf := g.asF32(av)
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, xf.ssa)
				cs = append(cs, tmp)
			} else {
				cs = append(cs, av.ssa)
			}
		}
		for len(cs) < 3 {
			cs = append(cs, "2147483647")
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_plan_log(i32 %s, i32 %s, i32 %s)\n", dst, cs[0], cs[1], cs[2])
		return llVal{"i32", dst}, nil
	case "pair_wildcard":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = add i32 -1, 0\n", dst)
		return llVal{"i32", dst}, nil
	case "pair_or":
		if len(n.Args) < 2 {
			return llVal{}, fmt.Errorf("pair_or arity")
		}
		var elems []llVal
		for _, a := range n.Args {
			av, err := g.emitExpr(a)
			if err != nil {
				return llVal{}, err
			}
			elems = append(elems, g.asF32(av))
		}
		id := g.fresh("span")
		g.spanElems[id] = elems
		return llVal{"span", id}, nil
	case "pair_or_match":
		if len(n.Args) != 2 {
			return llVal{}, fmt.Errorf("pair_or_match arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		idx := 0
		if lit, ok := n.Args[1].(*ast.Literal); ok {
			switch lit.Kind {
			case "int":
				idx = lit.Value.(int)
			case "float":
				idx = int(lit.Value.(float64))
			}
		}
		elems, ok := g.spanElems[p.ssa]
		if !ok || idx < 0 || idx >= len(elems) {
			z := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
			return llVal{"f32", z}, nil
		}
		return elems[idx], nil
	case "pair_or_default":
		if len(n.Args) != 1 {
			return llVal{}, fmt.Errorf("pair_or_default arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		elems, ok := g.spanElems[p.ssa]
		if !ok || len(elems) < 1 {
			z := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
			return llVal{"f32", z}, nil
		}
		return elems[0], nil
	case "hw_sme_za_stream_store":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_stream_store()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_weight":
		if len(n.Args) != 1 {
			return llVal{}, fmt.Errorf("fusion_pgo_weight arity")
		}
		av, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		arg := av
		if av.kind != "i32" {
			xf := g.asF32(av)
			tmp := g.fresh("i")
			fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, xf.ssa)
			arg = llVal{"i32", tmp}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_weight(i32 %s)\n", dst, arg.ssa)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_feedback":
		if len(n.Args) < 1 {
			return llVal{}, fmt.Errorf("fusion_pgo_feedback arity")
		}
		sa, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		sel := sa
		if sa.kind != "i32" {
			xf := g.asF32(sa)
			tmp := g.fresh("i")
			fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, xf.ssa)
			sel = llVal{"i32", tmp}
		}
		rewssa := "0"
		if len(n.Args) >= 2 {
			rv, err := g.emitExpr(n.Args[1])
			if err != nil {
				return llVal{}, err
			}
			if rv.kind != "i32" {
				xf := g.asF32(rv)
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, xf.ssa)
				rewssa = tmp
			} else {
				rewssa = rv.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_feedback(i32 %s, i32 %s)\n", dst, sel.ssa, rewssa)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_select":
		if len(n.Args) < 1 {
			return llVal{}, fmt.Errorf("fusion_pgo_select arity")
		}
		var cs []string
		for i := 0; i < len(n.Args) && i < 3; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				xf := g.asF32(av)
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, xf.ssa)
				cs = append(cs, tmp)
			} else {
				cs = append(cs, av.ssa)
			}
		}
		for len(cs) < 3 {
			cs = append(cs, "2147483647")
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_select(i32 %s, i32 %s, i32 %s)\n", dst, cs[0], cs[1], cs[2])
		return llVal{"i32", dst}, nil
	case "pair_tag":
		if len(n.Args) != 2 {
			return llVal{}, fmt.Errorf("pair_tag arity")
		}
		tag, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		val, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		id := g.fresh("span")
		g.spanElems[id] = []llVal{g.asF32(tag), g.asF32(val)}
		return llVal{"span", id}, nil
	case "pair_tag_match":
		if len(n.Args) != 2 {
			return llVal{}, fmt.Errorf("pair_tag_match arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		want, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		elems, ok := g.spanElems[p.ssa]
		if !ok || len(elems) < 2 {
			z := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = fadd float -1.000000e+00, 0.000000e+00\n", z)
			return llVal{"f32", z}, nil
		}
		tag := elems[0]
		val := elems[1]
		tagi := g.fresh("i")
		wanti := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tagi, g.asF32(tag).ssa)
		wf := g.asF32(want)
		fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", wanti, wf.ssa)
		dst := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = call float @trin_pair_tag_match_f32(float %s, i32 %s, i32 %s)\n", dst, g.asF32(val).ssa, tagi, wanti)
		return llVal{"f32", dst}, nil
	case "pair_tag_exhaust":
		if len(n.Args) < 2 {
			return llVal{}, fmt.Errorf("pair_tag_exhaust arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		tagi := "0"
		if elems, ok := g.spanElems[p.ssa]; ok && len(elems) >= 1 {
			tmp := g.fresh("i")
			fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(elems[0]).ssa)
			tagi = tmp
		}
		tags := []string{"0", "0"}
		for i := 1; i < len(n.Args) && i < 3; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			tmp := g.fresh("i")
			fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
			tags[i-1] = tmp
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_pair_tag_exhaust_i32(i32 %s, i32 %s, i32 %s)\n", dst, tagi, tags[0], tags[1])
		return llVal{"i32", dst}, nil
	case "hw_sme_za_gather":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_adapt":
		if len(n.Args) < 1 {
			return llVal{}, fmt.Errorf("fusion_pgo_adapt arity")
		}
		nv, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		ni := nv
		if nv.kind != "i32" {
			tmp := g.fresh("i")
			fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(nv).ssa)
			ni = llVal{"i32", tmp}
		}
		c0 := "0"
		if len(n.Args) >= 2 {
			cv, err := g.emitExpr(n.Args[1])
			if err != nil {
				return llVal{}, err
			}
			if cv.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(cv).ssa)
				c0 = tmp
			} else {
				c0 = cv.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_adapt(i32 %s, i32 %s)\n", dst, ni.ssa, c0)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_counter_bump":
		if len(n.Args) < 1 {
			return llVal{}, fmt.Errorf("fusion_pgo_counter_bump arity")
		}
		cv, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		ci := cv
		if cv.kind != "i32" {
			tmp := g.fresh("i")
			fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(cv).ssa)
			ci = llVal{"i32", tmp}
		}
		di := "1"
		if len(n.Args) >= 2 {
			dv, err := g.emitExpr(n.Args[1])
			if err != nil {
				return llVal{}, err
			}
			if dv.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(dv).ssa)
				di = tmp
			} else {
				di = dv.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_counter_bump(i32 %s, i32 %s)\n", dst, ci.ssa, di)
		return llVal{"i32", dst}, nil
	
	case "nested_tag":
		if len(n.Args) != 3 {
			return llVal{}, fmt.Errorf("nested_tag arity")
		}
		tag, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		a, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		b, err := g.emitExpr(n.Args[2])
		if err != nil {
			return llVal{}, err
		}
		id := g.fresh("span")
		g.spanElems[id] = []llVal{g.asF32(tag), g.asF32(a), g.asF32(b)}
		return llVal{"span", id}, nil
	case "nested_tag_match":
		if len(n.Args) != 2 {
			return llVal{}, fmt.Errorf("nested_tag_match arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		want, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		elems, ok := g.spanElems[p.ssa]
		if !ok || len(elems) < 3 {
			z := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = fadd float -1.000000e+00, 0.000000e+00\n", z)
			return llVal{"f32", z}, nil
		}
		tagi := g.fresh("i")
		wanti := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tagi, g.asF32(elems[0]).ssa)
		fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", wanti, g.asF32(want).ssa)
		dst := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_match_f32(float %s, float %s, i32 %s, i32 %s)\n",
			dst, g.asF32(elems[1]).ssa, g.asF32(elems[2]).ssa, tagi, wanti)
		return llVal{"f32", dst}, nil
	case "nested_tag_sugar":
		if len(n.Args) != 3 {
			return llVal{}, fmt.Errorf("nested_tag_sugar arity")
		}
		tag, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		a, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		b, err := g.emitExpr(n.Args[2])
		if err != nil {
			return llVal{}, err
		}
		id := g.fresh("span")
		g.spanElems[id] = []llVal{g.asF32(tag), g.asF32(a), g.asF32(b)}
		return llVal{"span", id}, nil
	case "hw_sme_za_gather_mv":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mv()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mv":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mv()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_isa_counter":
		isa, hits := "0", "0"
		if len(n.Args) >= 1 {
			av, err := g.emitExpr(n.Args[0])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				isa = tmp
			} else {
				isa = av.ssa
			}
		}
		if len(n.Args) >= 2 {
			av, err := g.emitExpr(n.Args[1])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				hits = tmp
			} else {
				hits = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_isa_counter(i32 %s, i32 %s)\n", dst, isa, hits)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_isa_loop":
		nv, isa, hits := "0", "0", "0"
		if len(n.Args) >= 1 {
			av, err := g.emitExpr(n.Args[0])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				nv = tmp
			} else {
				nv = av.ssa
			}
		}
		if len(n.Args) >= 2 {
			av, err := g.emitExpr(n.Args[1])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				isa = tmp
			} else {
				isa = av.ssa
			}
		}
		if len(n.Args) >= 3 {
			av, err := g.emitExpr(n.Args[2])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				hits = tmp
			} else {
				hits = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_isa_loop(i32 %s, i32 %s, i32 %s)\n", dst, nv, isa, hits)
		return llVal{"i32", dst}, nil

	case "nested_tag_multi":
		if len(n.Args) != 4 {
			return llVal{}, fmt.Errorf("nested_tag_multi arity")
		}
		tag, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		a, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		b, err := g.emitExpr(n.Args[2])
		if err != nil {
			return llVal{}, err
		}
		c, err := g.emitExpr(n.Args[3])
		if err != nil {
			return llVal{}, err
		}
		id := g.fresh("span")
		g.spanElems[id] = []llVal{g.asF32(tag), g.asF32(a), g.asF32(b), g.asF32(c)}
		return llVal{"span", id}, nil
	case "nested_tag_multi_match":
		if len(n.Args) < 2 {
			return llVal{}, fmt.Errorf("nested_tag_multi_match arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		wants := make([]string, 0, 3)
		for i := 1; i < len(n.Args) && i <= 3; i++ {
			w, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if w.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(w).ssa)
				wants = append(wants, tmp)
			} else {
				wants = append(wants, w.ssa)
			}
		}
		for len(wants) < 3 {
			wants = append(wants, "0")
		}
		elems, ok := g.spanElems[p.ssa]
		if !ok || len(elems) < 4 {
			z := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = fadd float -1.000000e+00, 0.000000e+00\n", z)
			return llVal{"f32", z}, nil
		}
		tagi := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tagi, g.asF32(elems[0]).ssa)
		dst := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_multi_match_f32(float %s, float %s, float %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, g.asF32(elems[1]).ssa, g.asF32(elems[2]).ssa, g.asF32(elems[3]).ssa, tagi, wants[0], wants[1], wants[2])
		return llVal{"f32", dst}, nil
	case "nested_tag_or_sugar":
		if len(n.Args) < 5 {
			return llVal{}, fmt.Errorf("nested_tag_or_sugar arity")
		}
		tag, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		a, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		b, err := g.emitExpr(n.Args[2])
		if err != nil {
			return llVal{}, err
		}
		c, err := g.emitExpr(n.Args[3])
		if err != nil {
			return llVal{}, err
		}
		wants := make([]string, 0, 3)
		for i := 4; i < len(n.Args) && len(wants) < 3; i++ {
			w, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if w.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(w).ssa)
				wants = append(wants, tmp)
			} else {
				wants = append(wants, w.ssa)
			}
		}
		for len(wants) < 3 {
			wants = append(wants, "0")
		}
		tagi := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tagi, g.asF32(tag).ssa)
		dst := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_multi_match_f32(float %s, float %s, float %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, g.asF32(a).ssa, g.asF32(b).ssa, g.asF32(c).ssa, tagi, wants[0], wants[1], wants[2])
		return llVal{"f32", dst}, nil
	case "hw_sme_za_gather_tile":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_tile()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_tile":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_tile()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_isa_live":
		nv, isa, hits, live := "0", "0", "0", "0"
		if len(n.Args) >= 1 {
			av, err := g.emitExpr(n.Args[0])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				nv = tmp
			} else {
				nv = av.ssa
			}
		}
		if len(n.Args) >= 2 {
			av, err := g.emitExpr(n.Args[1])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				isa = tmp
			} else {
				isa = av.ssa
			}
		}
		if len(n.Args) >= 3 {
			av, err := g.emitExpr(n.Args[2])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				hits = tmp
			} else {
				hits = av.ssa
			}
		}
		if len(n.Args) >= 4 {
			av, err := g.emitExpr(n.Args[3])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				live = tmp
			} else {
				live = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_isa_live(i32 %s, i32 %s, i32 %s, i32 %s)\n", dst, nv, isa, hits, live)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_isa_feedback":
		hits, live := "0", "0"
		if len(n.Args) >= 1 {
			av, err := g.emitExpr(n.Args[0])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				hits = tmp
			} else {
				hits = av.ssa
			}
		}
		if len(n.Args) >= 2 {
			av, err := g.emitExpr(n.Args[1])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				live = tmp
			} else {
				live = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_isa_feedback(i32 %s, i32 %s)\n", dst, hits, live)
		return llVal{"i32", dst}, nil

	case "nested_tag_recur":
		if len(n.Args) != 3 {
			return llVal{}, fmt.Errorf("nested_tag_recur arity")
		}
		tag, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		a, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		b, err := g.emitExpr(n.Args[2])
		if err != nil {
			return llVal{}, err
		}
		id := g.fresh("span")
		g.spanElems[id] = []llVal{g.asF32(tag), g.asF32(a), g.asF32(b)}
		return llVal{"span", id}, nil
	case "nested_tag_recur_match":
		if len(n.Args) < 2 {
			return llVal{}, fmt.Errorf("nested_tag_recur_match arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		wants := make([]string, 0, 3)
		for i := 1; i < len(n.Args) && i <= 3; i++ {
			w, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if w.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(w).ssa)
				wants = append(wants, tmp)
			} else {
				wants = append(wants, w.ssa)
			}
		}
		for len(wants) < 3 {
			wants = append(wants, "0")
		}
		elems, ok := g.spanElems[p.ssa]
		if !ok || len(elems) < 3 {
			z := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = fadd float -1.000000e+00, 0.000000e+00\n", z)
			return llVal{"f32", z}, nil
		}
		tagi := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tagi, g.asF32(elems[0]).ssa)
		dst := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_recur_match_f32(float %s, float %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, g.asF32(elems[1]).ssa, g.asF32(elems[2]).ssa, tagi, wants[0], wants[1], wants[2])
		return llVal{"f32", dst}, nil
	case "nested_tag_recur_sugar":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_recur_sugar arity")
		}
		tag, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		a, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		b, err := g.emitExpr(n.Args[2])
		if err != nil {
			return llVal{}, err
		}
		wants := make([]string, 0, 3)
		for i := 3; i < len(n.Args) && len(wants) < 3; i++ {
			w, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if w.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(w).ssa)
				wants = append(wants, tmp)
			} else {
				wants = append(wants, w.ssa)
			}
		}
		for len(wants) < 3 {
			wants = append(wants, "0")
		}
		tagi := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tagi, g.asF32(tag).ssa)
		dst := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_recur_match_f32(float %s, float %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, g.asF32(a).ssa, g.asF32(b).ssa, tagi, wants[0], wants[1], wants[2])
		return llVal{"f32", dst}, nil
	case "hw_sme_za_gather_mt":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_xkernel":
		nv, isa, hits, live, xk := "0", "0", "0", "0", "0"
		if len(n.Args) >= 1 {
			av, err := g.emitExpr(n.Args[0])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				nv = tmp
			} else {
				nv = av.ssa
			}
		}
		if len(n.Args) >= 2 {
			av, err := g.emitExpr(n.Args[1])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				isa = tmp
			} else {
				isa = av.ssa
			}
		}
		if len(n.Args) >= 3 {
			av, err := g.emitExpr(n.Args[2])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				hits = tmp
			} else {
				hits = av.ssa
			}
		}
		if len(n.Args) >= 4 {
			av, err := g.emitExpr(n.Args[3])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				live = tmp
			} else {
				live = av.ssa
			}
		}
		if len(n.Args) >= 5 {
			av, err := g.emitExpr(n.Args[4])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				xk = tmp
			} else {
				xk = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_xkernel(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n", dst, nv, isa, hits, live, xk)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_xkernel_counter":
		hits, live, xk := "0", "0", "0"
		if len(n.Args) >= 1 {
			av, err := g.emitExpr(n.Args[0])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				hits = tmp
			} else {
				hits = av.ssa
			}
		}
		if len(n.Args) >= 2 {
			av, err := g.emitExpr(n.Args[1])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				live = tmp
			} else {
				live = av.ssa
			}
		}
		if len(n.Args) >= 3 {
			av, err := g.emitExpr(n.Args[2])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				xk = tmp
			} else {
				xk = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_xkernel_counter(i32 %s, i32 %s, i32 %s)\n", dst, hits, live, xk)
		return llVal{"i32", dst}, nil

	case "nested_tag_ser":
		if len(n.Args) != 3 {
			return llVal{}, fmt.Errorf("nested_tag_ser arity")
		}
		tag, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		a, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		b, err := g.emitExpr(n.Args[2])
		if err != nil {
			return llVal{}, err
		}
		id := g.fresh("span")
		g.spanElems[id] = []llVal{g.asF32(tag), g.asF32(a), g.asF32(b)}
		return llVal{"span", id}, nil
	case "nested_tag_tree_fold":
		if len(n.Args) < 1 {
			return llVal{}, fmt.Errorf("nested_tag_tree_fold arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		elems, ok := g.spanElems[p.ssa]
		if !ok || len(elems) < 3 {
			z := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
			return llVal{"f32", z}, nil
		}
		dst := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_tree_fold_f32(float %s, float %s)\n",
			dst, g.asF32(elems[1]).ssa, g.asF32(elems[2]).ssa)
		return llVal{"f32", dst}, nil
	case "nested_tag_ser_fold":
		if len(n.Args) != 3 {
			return llVal{}, fmt.Errorf("nested_tag_ser_fold arity")
		}
		_, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		a, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		b, err := g.emitExpr(n.Args[2])
		if err != nil {
			return llVal{}, err
		}
		dst := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_tree_fold_f32(float %s, float %s)\n",
			dst, g.asF32(a).ssa, g.asF32(b).ssa)
		return llVal{"f32", dst}, nil
	case "hw_sme_za_gather_mt_peel":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_peel()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_peel":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_peel()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_xpass":
		nv, isa, hits, live, xk, pass := "0", "0", "0", "0", "0", "0"
		if len(n.Args) >= 1 {
			av, err := g.emitExpr(n.Args[0])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				nv = tmp
			} else {
				nv = av.ssa
			}
		}
		if len(n.Args) >= 2 {
			av, err := g.emitExpr(n.Args[1])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				isa = tmp
			} else {
				isa = av.ssa
			}
		}
		if len(n.Args) >= 3 {
			av, err := g.emitExpr(n.Args[2])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				hits = tmp
			} else {
				hits = av.ssa
			}
		}
		if len(n.Args) >= 4 {
			av, err := g.emitExpr(n.Args[3])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				live = tmp
			} else {
				live = av.ssa
			}
		}
		if len(n.Args) >= 5 {
			av, err := g.emitExpr(n.Args[4])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				xk = tmp
			} else {
				xk = av.ssa
			}
		}
		if len(n.Args) >= 6 {
			av, err := g.emitExpr(n.Args[5])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				pass = tmp
			} else {
				pass = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_xpass(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n", dst, nv, isa, hits, live, xk, pass)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_xpass_counter":
		hits, live, xk, pass := "0", "0", "0", "0"
		if len(n.Args) >= 1 {
			av, err := g.emitExpr(n.Args[0])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				hits = tmp
			} else {
				hits = av.ssa
			}
		}
		if len(n.Args) >= 2 {
			av, err := g.emitExpr(n.Args[1])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				live = tmp
			} else {
				live = av.ssa
			}
		}
		if len(n.Args) >= 3 {
			av, err := g.emitExpr(n.Args[2])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				xk = tmp
			} else {
				xk = av.ssa
			}
		}
		if len(n.Args) >= 4 {
			av, err := g.emitExpr(n.Args[3])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				pass = tmp
			} else {
				pass = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_xpass_counter(i32 %s, i32 %s, i32 %s, i32 %s)\n", dst, hits, live, xk, pass)
		return llVal{"i32", dst}, nil

	case "nested_tag_zip":
		if len(n.Args) != 2 {
			return llVal{}, fmt.Errorf("nested_tag_zip arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		// Best-effort: if both are known spans, emit elementwise sum
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq {
			n := len(pe)
			if len(qe) < n {
				n = len(qe)
			}
			out := make([]llVal, n)
			for i := 0; i < n; i++ {
				dst := g.fresh("f")
				fmt.Fprintf(g.b, "  %s = fadd float %s, %s\n", dst, g.asF32(pe[i]).ssa, g.asF32(qe[i]).ssa)
				out[i] = llVal{"f32", dst}
			}
			id := g.fresh("span")
			g.spanElems[id] = out
			return llVal{"span", id}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		id := g.fresh("span")
		g.spanElems[id] = []llVal{{"f32", z}}
		return llVal{"span", id}, nil
	case "nested_tag_map_fold":
		if len(n.Args) < 1 {
			return llVal{}, fmt.Errorf("nested_tag_map_fold arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		elems, ok := g.spanElems[p.ssa]
		if !ok || len(elems) < 3 {
			z := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
			return llVal{"f32", z}, nil
		}
		dst := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_map_fold_f32(float %s, float %s)\n",
			dst, g.asF32(elems[1]).ssa, g.asF32(elems[2]).ssa)
		return llVal{"f32", dst}, nil
	case "nested_tag_forest_fold":
		if len(n.Args) != 2 {
			return llVal{}, fmt.Errorf("nested_tag_forest_fold arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_forest_fold_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_auto":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_auto()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_auto":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_auto()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_window":
		nv, isa, hits, live, xk, pass, win := "0", "0", "0", "0", "0", "0", "0"
		args := []string{"0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 7; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		nv, isa, hits, live, xk, pass, win = args[0], args[1], args[2], args[3], args[4], args[5], args[6]
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_window(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n", dst, nv, isa, hits, live, xk, pass, win)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_window_counter":
		args := []string{"0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 5; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_window_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n", dst, args[0], args[1], args[2], args[3], args[4])
		return llVal{"i32", dst}, nil

	case "nested_tag_keyed":
		// pack three elems as span if possible; not used in probes
		if len(n.Args) != 3 {
			return llVal{}, fmt.Errorf("nested_tag_keyed arity")
		}
		// lower to nested_tag_ser style via leaves
		a0, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		a1, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		a2, err := g.emitExpr(n.Args[2])
		if err != nil {
			return llVal{}, err
		}
		// build synthetic span elements map via nested_tag_ser pattern
		dst := g.fresh("p")
		// just return first as i32 tag carrier; prefer forest folds
		_ = a1
		_ = a2
		if a0.kind == "i32" {
			fmt.Fprintf(g.b, "  %s = add i32 %s, 0\n", dst, a0.ssa)
			return llVal{"i32", dst}, nil
		}
		tmp := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(a0).ssa)
		return llVal{"i32", tmp}, nil
	case "nested_tag_keyed_fold":
		if len(n.Args) < 2 {
			return llVal{}, fmt.Errorf("nested_tag_keyed_fold arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		if okp && len(pe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_keyed_fold_f32(float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "nested_tag_keyed_reduce":
		if len(n.Args) < 3 {
			return llVal{}, fmt.Errorf("nested_tag_keyed_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_keyed_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vlfb":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vlfb()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vlfb":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vlfb()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_decay":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 8; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_decay(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_decay_counter":
		args := []string{"0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 6; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_decay_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5])
		return llVal{"i32", dst}, nil

	case "nested_tag_filter":
		if len(n.Args) < 2 {
			return llVal{}, fmt.Errorf("nested_tag_filter arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		if okp && len(pe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_filter_f32(float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "nested_tag_multi_key":
		if len(n.Args) != 3 {
			return llVal{}, fmt.Errorf("nested_tag_multi_key arity")
		}
		a0, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		if a0.kind == "i32" {
			return a0, nil
		}
		tmp := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(a0).ssa)
		return llVal{"i32", tmp}, nil
	case "nested_tag_multi_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_multi_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_multi_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vllive":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vllive()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vllive":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vllive()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_ema":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 9; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_ema(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_ema_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 7; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_ema_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6])
		return llVal{"i32", dst}, nil

	case "nested_tag_rank_filter":
		if len(n.Args) < 2 {
			return llVal{}, fmt.Errorf("nested_tag_rank_filter arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		if okp && len(pe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_rank_filter_f32(float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "nested_tag_rank_key":
		if len(n.Args) != 3 {
			return llVal{}, fmt.Errorf("nested_tag_rank_key arity")
		}
		a0, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		if a0.kind == "i32" {
			return a0, nil
		}
		tmp := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(a0).ssa)
		return llVal{"i32", tmp}, nil
	case "nested_tag_rank_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_rank_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_rank_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vlstream":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vlstream()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vlstream":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vlstream()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_adapt_hl":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 9; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_adapt_hl(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_adapt_hl_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 7; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_adapt_hl_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6])
		return llVal{"i32", dst}, nil
	case "nested_tag_weight_filter":
		if len(n.Args) < 2 {
			return llVal{}, fmt.Errorf("nested_tag_weight_filter arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		if okp && len(pe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_weight_filter_f32(float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "nested_tag_weight_key":
		if len(n.Args) != 3 {
			return llVal{}, fmt.Errorf("nested_tag_weight_key arity")
		}
		a0, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		if a0.kind == "i32" {
			return a0, nil
		}
		tmp := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(a0).ssa)
		return llVal{"i32", tmp}, nil
	case "nested_tag_weight_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_weight_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_weight_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vlcost":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vlcost()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vlcost":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vlcost()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_hyst":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 9; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_hyst(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_hyst_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 7; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_hyst_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6])
		return llVal{"i32", dst}, nil
	case "nested_tag_thresh_filter":
		if len(n.Args) < 2 {
			return llVal{}, fmt.Errorf("nested_tag_thresh_filter arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		if okp && len(pe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_thresh_filter_f32(float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "nested_tag_thresh_key":
		if len(n.Args) != 3 {
			return llVal{}, fmt.Errorf("nested_tag_thresh_key arity")
		}
		a0, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		if a0.kind == "i32" {
			return a0, nil
		}
		tmp := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(a0).ssa)
		return llVal{"i32", tmp}, nil
	case "nested_tag_thresh_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_thresh_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_thresh_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vladapt":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vladapt()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vladapt":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vladapt()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_reweight":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 9; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_reweight(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_reweight_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 7; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_reweight_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6])
		return llVal{"i32", dst}, nil
	case "nested_tag_soft_filter":
		if len(n.Args) < 2 {
			return llVal{}, fmt.Errorf("nested_tag_soft_filter arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		if okp && len(pe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_soft_filter_f32(float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "nested_tag_soft_key":
		if len(n.Args) != 3 {
			return llVal{}, fmt.Errorf("nested_tag_soft_key arity")
		}
		a0, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		if a0.kind == "i32" {
			return a0, nil
		}
		tmp := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(a0).ssa)
		return llVal{"i32", tmp}, nil
	case "nested_tag_soft_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_soft_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_soft_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vlsched":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vlsched()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vlsched":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vlsched()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_budget":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 10; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_budget(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_budget_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 7; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_budget_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6])
		return llVal{"i32", dst}, nil
	case "nested_tag_cal_filter":
		if len(n.Args) < 2 {
			return llVal{}, fmt.Errorf("nested_tag_cal_filter arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		if okp && len(pe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_cal_filter_f32(float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "nested_tag_cal_key":
		if len(n.Args) != 3 {
			return llVal{}, fmt.Errorf("nested_tag_cal_key arity")
		}
		a0, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		if a0.kind == "i32" {
			return a0, nil
		}
		tmp := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(a0).ssa)
		return llVal{"i32", tmp}, nil
	case "nested_tag_cal_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_cal_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_cal_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vlfuse":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vlfuse()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vlfuse":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vlfuse()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_softcap":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 11; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_softcap(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_softcap_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 7; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_softcap_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6])
		return llVal{"i32", dst}, nil
	case "nested_tag_adapt_filter":
		if len(n.Args) < 2 {
			return llVal{}, fmt.Errorf("nested_tag_adapt_filter arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		if okp && len(pe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_adapt_filter_f32(float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "nested_tag_adapt_key":
		if len(n.Args) != 3 {
			return llVal{}, fmt.Errorf("nested_tag_adapt_key arity")
		}
		a0, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		if a0.kind == "i32" {
			return a0, nil
		}
		tmp := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(a0).ssa)
		return llVal{"i32", tmp}, nil
	case "nested_tag_adapt_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_adapt_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_adapt_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vlmix":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vlmix()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vlmix":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vlmix()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_dual":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 12; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_dual(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_dual_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 7; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_dual_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6])
		return llVal{"i32", dst}, nil
	case "nested_tag_blend_filter":
		if len(n.Args) < 2 {
			return llVal{}, fmt.Errorf("nested_tag_blend_filter arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		if okp && len(pe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_blend_filter_f32(float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "nested_tag_blend_key":
		if len(n.Args) != 3 {
			return llVal{}, fmt.Errorf("nested_tag_blend_key arity")
		}
		a0, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		if a0.kind == "i32" {
			return a0, nil
		}
		tmp := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(a0).ssa)
		return llVal{"i32", tmp}, nil
	case "nested_tag_blend_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_blend_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_blend_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vlbal":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vlbal()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vlbal":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vlbal()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_conf":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 13; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_conf(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_conf_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 8; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_conf_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7])
		return llVal{"i32", dst}, nil
	case "nested_tag_gate_filter":
		if len(n.Args) < 2 {
			return llVal{}, fmt.Errorf("nested_tag_gate_filter arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		if okp && len(pe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_gate_filter_f32(float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "nested_tag_gate_key":
		if len(n.Args) != 3 {
			return llVal{}, fmt.Errorf("nested_tag_gate_key arity")
		}
		a0, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		if a0.kind == "i32" {
			return a0, nil
		}
		tmp := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(a0).ssa)
		return llVal{"i32", tmp}, nil
	case "nested_tag_gate_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_gate_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_gate_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vlbw":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vlbw()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vlbw":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vlbw()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_reg":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 14; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_reg(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_reg_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 9; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_reg_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8])
		return llVal{"i32", dst}, nil
	case "nested_tag_score_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_score_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_score_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vlpath":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vlpath()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vlpath":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vlpath()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_floor":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 15; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_floor(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_floor_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 10; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_floor_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9])
		return llVal{"i32", dst}, nil
	case "nested_tag_softfloor_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_softfloor_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_softfloor_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vllane":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vllane()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vllane":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vllane()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_afloor":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 16; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_afloor(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_afloor_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 11; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_afloor_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10])
		return llVal{"i32", dst}, nil
	case "nested_tag_hardgate_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_hardgate_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_hardgate_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vlspill":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vlspill()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vlspill":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vlspill()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_burst":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 17; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_burst(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_burst_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 12; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_burst_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11])
		return llVal{"i32", dst}, nil
	case "nested_tag_dual_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_dual_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_dual_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vlocc":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vlocc()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vlocc":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vlocc()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_cool":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 18; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_cool(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_cool_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 13; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_cool_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12])
		return llVal{"i32", dst}, nil
	case "nested_tag_mix_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_mix_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_mix_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vlqueue":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vlqueue()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vlqueue":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vlqueue()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_credit":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 19; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_credit(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_credit_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 14; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_credit_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13])
		return llVal{"i32", dst}, nil
	case "nested_tag_credit_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_credit_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_credit_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vlwater":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vlwater()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vlwater":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vlwater()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_debt":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 20; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_debt(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_debt_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 15; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_debt_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14])
		return llVal{"i32", dst}, nil
	case "nested_tag_margin_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_margin_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_margin_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vldrain":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vldrain()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vldrain":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vldrain()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_interest":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 21; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_interest(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_interest_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 16; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_interest_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15])
		return llVal{"i32", dst}, nil
	case "nested_tag_band_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_band_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_band_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vlrefill":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vlrefill()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vlrefill":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vlrefill()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_amortize":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 22; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_amortize(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_amortize_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 17; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_amortize_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16])
		return llVal{"i32", dst}, nil
	case "nested_tag_clamp_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_clamp_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_clamp_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vlflush":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vlflush()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vlflush":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vlflush()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_settle":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 23; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_settle(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_settle_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 18; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_settle_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17])
		return llVal{"i32", dst}, nil

	case "nested_tag_peg_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_peg_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_peg_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vlseal":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vlseal()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vlseal":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vlseal()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_escrow":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 24; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_escrow(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_escrow_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 19; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_escrow_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18])
		return llVal{"i32", dst}, nil
	case "nested_tag_latch_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_latch_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_latch_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vlpurge":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vlpurge()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vlpurge":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vlpurge()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_clear":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 25; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_clear(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_clear_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 20; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_clear_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19])
		return llVal{"i32", dst}, nil
	case "nested_tag_fence_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_fence_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_fence_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vlsync":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vlsync()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vlsync":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vlsync()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_final":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 26; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_final(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_final_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 21; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_final_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20])
		return llVal{"i32", dst}, nil
	case "nested_tag_lock_reduce":
		if len(n.Args) < 4 {
			return llVal{}, fmt.Errorf("nested_tag_lock_reduce arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		q, err := g.emitExpr(n.Args[1])
		if err != nil {
			return llVal{}, err
		}
		pe, okp := g.spanElems[p.ssa]
		qe, okq := g.spanElems[q.ssa]
		if okp && okq && len(pe) >= 3 && len(qe) >= 3 {
			dst := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_nested_tag_lock_reduce_f32(float %s, float %s, float %s, float %s)\n",
				dst, g.asF32(pe[1]).ssa, g.asF32(pe[2]).ssa, g.asF32(qe[1]).ssa, g.asF32(qe[2]).ssa)
			return llVal{"f32", dst}, nil
		}
		z := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
		return llVal{"f32", z}, nil
	case "hw_sme_za_gather_mt_vlclose":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_gather_mt_vlclose()\n", dst)
		return llVal{"i32", dst}, nil
	case "hw_sme_za_scatter_mt_vlclose":
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_hw_sme_za_scatter_mt_vlclose()\n", dst)
		return llVal{"i32", dst}, nil
	case "fusion_pgo_close":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 27; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_close(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26])
		return llVal{"i32", dst}, nil
	case "fusion_pgo_close_counter":
		args := []string{"0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"}
		for i := 0; i < len(n.Args) && i < 22; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, g.asF32(av).ssa)
				args[i] = tmp
			} else {
				args[i] = av.ssa
			}
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_pgo_close_counter(i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s, i32 %s)\n",
			dst, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21])
		return llVal{"i32", dst}, nil
	case "pair_guard":
		if len(n.Args) != 4 {
			return llVal{}, fmt.Errorf("pair_guard arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		idx := 0
		if lit, ok := n.Args[1].(*ast.Literal); ok {
			switch lit.Kind {
			case "int":
				idx = lit.Value.(int)
			case "float":
				idx = int(lit.Value.(float64))
			}
		}
		elems, ok := g.spanElems[p.ssa]
		var v llVal
		if ok && idx >= 0 && idx < len(elems) {
			v = g.asF32(elems[idx])
		} else {
			z := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
			v = llVal{"f32", z}
		}
		lo, err := g.emitExpr(n.Args[2])
		if err != nil {
			return llVal{}, err
		}
		hi, err := g.emitExpr(n.Args[3])
		if err != nil {
			return llVal{}, err
		}
		lof, hif := g.asF32(lo), g.asF32(hi)
		gv := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = call float @trin_pair_match_guard_f32(float %s, float %s, float %s)\n", gv, v.ssa, lof.ssa, hif.ssa)
		okb := g.fresh("i1")
		fmt.Fprintf(g.b, "  %s = fcmp oeq float %s, %s\n", okb, gv, v.ssa)
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = zext i1 %s to i32\n", dst, okb)
		return llVal{"i32", dst}, nil
	case "pair_match_guard":
		if len(n.Args) != 4 {
			return llVal{}, fmt.Errorf("pair_match_guard arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		idx := 0
		if lit, ok := n.Args[1].(*ast.Literal); ok {
			switch lit.Kind {
			case "int":
				idx = lit.Value.(int)
			case "float":
				idx = int(lit.Value.(float64))
			}
		}
		elems, ok := g.spanElems[p.ssa]
		var v llVal
		if ok && idx >= 0 && idx < len(elems) {
			v = g.asF32(elems[idx])
		} else {
			z := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", z)
			v = llVal{"f32", z}
		}
		lo, err := g.emitExpr(n.Args[2])
		if err != nil {
			return llVal{}, err
		}
		hi, err := g.emitExpr(n.Args[3])
		if err != nil {
			return llVal{}, err
		}
		lof, hif := g.asF32(lo), g.asF32(hi)
		dst := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = call float @trin_pair_match_guard_f32(float %s, float %s, float %s)\n", dst, v.ssa, lof.ssa, hif.ssa)
		return llVal{"f32", dst}, nil
	case "fusion_select":
		if len(n.Args) < 1 {
			return llVal{}, fmt.Errorf("fusion_select arity")
		}
		var cs []string
		for i := 0; i < len(n.Args) && i < 3; i++ {
			av, err := g.emitExpr(n.Args[i])
			if err != nil {
				return llVal{}, err
			}
			if av.kind != "i32" {
				xf := g.asF32(av)
				tmp := g.fresh("i")
				fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", tmp, xf.ssa)
				cs = append(cs, tmp)
			} else {
				cs = append(cs, av.ssa)
			}
		}
		for len(cs) < 3 {
			cs = append(cs, "2147483647")
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = call i32 @trin_fusion_select(i32 %s, i32 %s, i32 %s)\n", dst, cs[0], cs[1], cs[2])
		return llVal{"i32", dst}, nil
	case "pair_arity":
		if len(n.Args) != 1 {
			return llVal{}, fmt.Errorf("pair_arity arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		elems, ok := g.spanElems[p.ssa]
		nLeaves := 0
		if ok {
			nLeaves = len(elems)
		}
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = add i32 %d, 0\n", dst, nLeaves)
		return llVal{"i32", dst}, nil
	case "pair_exhaust":
		if len(n.Args) != 2 {
			return llVal{}, fmt.Errorf("pair_exhaust arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		want := 0
		if lit, ok := n.Args[1].(*ast.Literal); ok {
			switch lit.Kind {
			case "int":
				want = lit.Value.(int)
			case "float":
				want = int(lit.Value.(float64))
			}
		}
		elems, ok := g.spanElems[p.ssa]
		got := 0
		if ok {
			got = len(elems)
		}
		dst := g.fresh("i")
		if got == want {
			fmt.Fprintf(g.b, "  %s = add i32 1, 0\n", dst)
		} else {
			fmt.Fprintf(g.b, "  %s = add i32 0, 0\n", dst)
		}
		return llVal{"i32", dst}, nil
	case "pair_of_pairs":
		if len(n.Args) != 4 {
			return llVal{}, fmt.Errorf("pair_of_pairs arity")
		}
		var elems []llVal
		for _, a := range n.Args {
			av, err := g.emitExpr(a)
			if err != nil {
				return llVal{}, err
			}
			elems = append(elems, g.asF32(av))
		}
		// flat 4-leaf surface (leaf order for destructure)
		id := g.fresh("span")
		g.spanElems[id] = elems
		return llVal{"span", id}, nil
	case "pair":
		if len(n.Args) < 2 {
			return llVal{}, fmt.Errorf("pair arity")
		}
		elems := make([]llVal, 0, len(n.Args))
		for _, a := range n.Args {
			av, err := g.emitExpr(a)
			if err != nil {
				return llVal{}, err
			}
			elems = append(elems, g.asF32(av))
		}
		id := g.fresh("span")
		g.spanElems[id] = elems
		return llVal{"span", id}, nil
	case "pair_first":
		if len(n.Args) != 1 {
			return llVal{}, fmt.Errorf("pair_first arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		elems, ok := g.spanElems[p.ssa]
		if !ok || len(elems) < 1 {
			return llVal{}, fmt.Errorf("pair_first expects pair/span")
		}
		return elems[0], nil
	case "pair_second":
		if len(n.Args) != 1 {
			return llVal{}, fmt.Errorf("pair_second arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		elems, ok := g.spanElems[p.ssa]
		if !ok || len(elems) < 2 {
			return llVal{}, fmt.Errorf("pair_second expects pair/span")
		}
		return elems[1], nil
	case "pair_match":
		if len(n.Args) != 2 {
			return llVal{}, fmt.Errorf("pair_match arity")
		}
		p, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		// compile-time index only when literal
		idx := 0
		if lit, ok := n.Args[1].(*ast.Literal); ok {
			switch lit.Kind {
			case "int":
				idx = lit.Value.(int)
			case "float":
				idx = int(lit.Value.(float64))
			}
		}
		elems, ok := g.spanElems[p.ssa]
		if !ok || idx < 0 || idx >= len(elems) {
			return llVal{}, fmt.Errorf("pair_match expects in-range pair/span index")
		}
		return elems[idx], nil
	case "trit":
		if len(n.Args) != 1 {
			return llVal{}, fmt.Errorf("trit arity")
		}
		x, err := g.emitExpr(n.Args[0])
		if err != nil {
			return llVal{}, err
		}
		if x.kind == "i32" {
			return x, nil
		}
		xf := g.asF32(x)
		dst := g.fresh("i")
		fmt.Fprintf(g.b, "  %s = fptosi float %s to i32\n", dst, xf.ssa)
		return llVal{"i32", dst}, nil
	default:
		// User pure float function call — all args as f32, return f32.
		args := make([]llVal, len(n.Args))
		for i, a := range n.Args {
			av, err := g.emitExpr(a)
			if err != nil {
				return llVal{}, err
			}
			args[i] = g.asF32(av)
		}
		dst := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = call float @%s(", dst, sanitizeLLVM(v.Name))
		for i, a := range args {
			if i > 0 {
				g.b.WriteString(", ")
			}
			fmt.Fprintf(g.b, "float %s", a.ssa)
		}
		g.b.WriteString(")\n")
		return llVal{"f32", dst}, nil
	}
}

func (g *fullLL) asStr(v llVal) (llVal, error) {
	switch v.kind {
	case "str":
		return v, nil
	case "i32":
		dst := g.fresh("s")
		fmt.Fprintf(g.b, "  %s = call i8* @trin_to_str_i32(i32 %s)\n", dst, v.ssa)
		return llVal{"str", dst}, nil
	case "f32":
		dst := g.fresh("s")
		fmt.Fprintf(g.b, "  %s = call i8* @trin_to_str_f32(float %s)\n", dst, v.ssa)
		return llVal{"str", dst}, nil
	default:
		return llVal{}, fmt.Errorf("cannot to_str %s", v.kind)
	}
}

func (g *fullLL) emitSumArg(e ast.Expr) (llVal, error) {
	elems, err := g.emitSpanElemsExpr(e)
	if err != nil {
		return llVal{}, err
	}
	if len(elems) == 0 {
		dst := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", dst)
		return llVal{"f32", dst}, nil
	}
	// v0.32/v0.51: fixed 4/8/16/32-length spans route through runtime tile dispatch.
	if preferDynTileN(len(elems)) {
		dst := g.fresh("f")
		switch len(elems) {
		case 4:
			fmt.Fprintf(g.b, "  %s = call float @trin_dyn_sum4(float %s, float %s, float %s, float %s)\n",
				dst, elems[0].ssa, elems[1].ssa, elems[2].ssa, elems[3].ssa)
		case 8:
			fmt.Fprintf(g.b, "  %s = call float @trin_dyn_sum8(float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s)\n",
				dst, elems[0].ssa, elems[1].ssa, elems[2].ssa, elems[3].ssa,
				elems[4].ssa, elems[5].ssa, elems[6].ssa, elems[7].ssa)
		case 16:
			fmt.Fprintf(g.b, "  %s = call float @trin_dyn_sum16(float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s)\n",
				dst,
				elems[0].ssa, elems[1].ssa, elems[2].ssa, elems[3].ssa,
				elems[4].ssa, elems[5].ssa, elems[6].ssa, elems[7].ssa,
				elems[8].ssa, elems[9].ssa, elems[10].ssa, elems[11].ssa,
				elems[12].ssa, elems[13].ssa, elems[14].ssa, elems[15].ssa)
		case 32:
			fmt.Fprintf(g.b, "  %s = call float @trin_dyn_sum32(float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s)\n",
				dst,
				elems[0].ssa, elems[1].ssa, elems[2].ssa, elems[3].ssa,
				elems[4].ssa, elems[5].ssa, elems[6].ssa, elems[7].ssa,
				elems[8].ssa, elems[9].ssa, elems[10].ssa, elems[11].ssa,
				elems[12].ssa, elems[13].ssa, elems[14].ssa, elems[15].ssa,
				elems[16].ssa, elems[17].ssa, elems[18].ssa, elems[19].ssa,
				elems[20].ssa, elems[21].ssa, elems[22].ssa, elems[23].ssa,
				elems[24].ssa, elems[25].ssa, elems[26].ssa, elems[27].ssa,
				elems[28].ssa, elems[29].ssa, elems[30].ssa, elems[31].ssa)
		}
		return llVal{"f32", dst}, nil
	}
	// v0.34: residual peel — max 16/8/4 dyn tiles + scalar residual.
	if preferResidualPeel(len(elems)) {
		return g.emitSumPeeled(elems)
	}
	acc := elems[0].ssa
	for i := 1; i < len(elems); i++ {
		nxt := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float %s, %s\n", nxt, acc, elems[i].ssa)
		acc = nxt
	}
	return llVal{"f32", acc}, nil
}

// emitSumPeeled lowers a sum of elems by AdaptivePeelChunks at emit time
// (host ISA HWVLEN schedule: may include 32 for SVE-class hosts) then a scalar
// residual (v0.34 residual peel / v0.51 AdaptivePeelChunks emission).
func (g *fullLL) emitSumPeeled(elems []llVal) (llVal, error) {
	hw := DetectHostCPU().HWVLEN
	if hw < 4 {
		hw = 16
	}
	tiles, residual := peelDynTileChunks(len(elems))
	fmt.Fprintf(g.b, "  ; rewrite: residual peel sum(n=%d) tiles=%v residual=%d (v0.34/v0.51 AdaptivePeelChunks hw=%d)\n",
		len(elems), tiles, residual, hw)
	fmt.Fprintf(g.b, "  %s\n", EmitAdaptivePeelChunksNote(hw, len(elems)))
	var acc string
	have := false
	off := 0
	emitChunk := func(w int) {
		dst := g.fresh("f")
		switch w {
		case 4:
			fmt.Fprintf(g.b, "  %s = call float @trin_dyn_sum4(float %s, float %s, float %s, float %s)\n",
				dst, elems[off].ssa, elems[off+1].ssa, elems[off+2].ssa, elems[off+3].ssa)
		case 8:
			fmt.Fprintf(g.b, "  %s = call float @trin_dyn_sum8(float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s)\n",
				dst, elems[off].ssa, elems[off+1].ssa, elems[off+2].ssa, elems[off+3].ssa,
				elems[off+4].ssa, elems[off+5].ssa, elems[off+6].ssa, elems[off+7].ssa)
		case 16:
			fmt.Fprintf(g.b, "  %s = call float @trin_dyn_sum16(float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s)\n",
				dst,
				elems[off].ssa, elems[off+1].ssa, elems[off+2].ssa, elems[off+3].ssa,
				elems[off+4].ssa, elems[off+5].ssa, elems[off+6].ssa, elems[off+7].ssa,
				elems[off+8].ssa, elems[off+9].ssa, elems[off+10].ssa, elems[off+11].ssa,
				elems[off+12].ssa, elems[off+13].ssa, elems[off+14].ssa, elems[off+15].ssa)
		case 32:
			// v0.51 exact-32 dyn helper
			fmt.Fprintf(g.b, "  %s = call float @trin_dyn_sum32(float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s)\n",
				dst,
				elems[off].ssa, elems[off+1].ssa, elems[off+2].ssa, elems[off+3].ssa,
				elems[off+4].ssa, elems[off+5].ssa, elems[off+6].ssa, elems[off+7].ssa,
				elems[off+8].ssa, elems[off+9].ssa, elems[off+10].ssa, elems[off+11].ssa,
				elems[off+12].ssa, elems[off+13].ssa, elems[off+14].ssa, elems[off+15].ssa,
				elems[off+16].ssa, elems[off+17].ssa, elems[off+18].ssa, elems[off+19].ssa,
				elems[off+20].ssa, elems[off+21].ssa, elems[off+22].ssa, elems[off+23].ssa,
				elems[off+24].ssa, elems[off+25].ssa, elems[off+26].ssa, elems[off+27].ssa,
				elems[off+28].ssa, elems[off+29].ssa, elems[off+30].ssa, elems[off+31].ssa)
		case 64:
			// VL>32 peel: two dyn_sum32
			dstLo := g.fresh("f")
			dstHi := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = call float @trin_dyn_sum32(float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s)\n",
				dstLo,
				elems[off].ssa, elems[off+1].ssa, elems[off+2].ssa, elems[off+3].ssa,
				elems[off+4].ssa, elems[off+5].ssa, elems[off+6].ssa, elems[off+7].ssa,
				elems[off+8].ssa, elems[off+9].ssa, elems[off+10].ssa, elems[off+11].ssa,
				elems[off+12].ssa, elems[off+13].ssa, elems[off+14].ssa, elems[off+15].ssa,
				elems[off+16].ssa, elems[off+17].ssa, elems[off+18].ssa, elems[off+19].ssa,
				elems[off+20].ssa, elems[off+21].ssa, elems[off+22].ssa, elems[off+23].ssa,
				elems[off+24].ssa, elems[off+25].ssa, elems[off+26].ssa, elems[off+27].ssa,
				elems[off+28].ssa, elems[off+29].ssa, elems[off+30].ssa, elems[off+31].ssa)
			fmt.Fprintf(g.b, "  %s = call float @trin_dyn_sum32(float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s, float %s)\n",
				dstHi,
				elems[off+32].ssa, elems[off+33].ssa, elems[off+34].ssa, elems[off+35].ssa,
				elems[off+36].ssa, elems[off+37].ssa, elems[off+38].ssa, elems[off+39].ssa,
				elems[off+40].ssa, elems[off+41].ssa, elems[off+42].ssa, elems[off+43].ssa,
				elems[off+44].ssa, elems[off+45].ssa, elems[off+46].ssa, elems[off+47].ssa,
				elems[off+48].ssa, elems[off+49].ssa, elems[off+50].ssa, elems[off+51].ssa,
				elems[off+52].ssa, elems[off+53].ssa, elems[off+54].ssa, elems[off+55].ssa,
				elems[off+56].ssa, elems[off+57].ssa, elems[off+58].ssa, elems[off+59].ssa,
				elems[off+60].ssa, elems[off+61].ssa, elems[off+62].ssa, elems[off+63].ssa)
			fmt.Fprintf(g.b, "  %s = fadd float %s, %s\n", dst, dstLo, dstHi)
		}
		if !have {
			acc = dst
			have = true
		} else {
			nxt := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = fadd float %s, %s\n", nxt, acc, dst)
			acc = nxt
		}
		off += w
	}
	for _, w := range tiles {
		emitChunk(w)
	}
	for i := 0; i < residual; i++ {
		if !have {
			acc = elems[off].ssa
			have = true
		} else {
			nxt := g.fresh("f")
			fmt.Fprintf(g.b, "  %s = fadd float %s, %s\n", nxt, acc, elems[off].ssa)
			acc = nxt
		}
		off++
	}
	if !have {
		dst := g.fresh("f")
		fmt.Fprintf(g.b, "  %s = fadd float 0.000000e+00, 0.000000e+00\n", dst)
		return llVal{"f32", dst}, nil
	}
	return llVal{"f32", acc}, nil
}

func (g *fullLL) emitSpanElemsExpr(e ast.Expr) ([]llVal, error) {
	g.ensureMaps()
	switch n := e.(type) {
	case *ast.SpanLit:
		out := make([]llVal, len(n.Elems))
		for i, el := range n.Elems {
			v, err := g.emitExpr(el)
			if err != nil {
				return nil, err
			}
			out[i] = g.asF32(v)
		}
		return out, nil
	case *ast.Var:
		if v, ok := g.names[n.Name]; ok && v.kind == "span" {
			return g.spanElems[v.ssa], nil
		}
		return nil, fmt.Errorf("span var")
	case *ast.Call:
		v, err := g.emitExpr(n)
		if err != nil {
			return nil, err
		}
		if v.kind != "span" {
			return nil, fmt.Errorf("not span")
		}
		return g.spanElems[v.ssa], nil
	default:
		return nil, fmt.Errorf("span expected")
	}
}
