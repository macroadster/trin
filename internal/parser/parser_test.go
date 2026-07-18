package parser

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/macroadster/trin/internal/ast"
)

func examplesDir(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("no caller")
	}
	return filepath.Join(filepath.Dir(file), "..", "..", "examples")
}

func TestParsePairType(t *testing.T) {
	// Language-level pair[T] surface (v0.46) + destructure/return (v0.47)
	src := `
fn id(p: pair[f32]) -> pair[f32] {
    p
}
fn main() {
    let x: pair[f32] = pair(1.0, 2.0);
    let a = pair_first(x);
    let b = pair_second(x);
    let (c, d) = x;
    println(to_str(a + b + c + d));
}
`
	prog, err := ParseSource(src)
	if err != nil {
		t.Fatalf("parse pair type: %v", err)
	}
	if len(prog.Items) < 2 {
		t.Fatalf("expected pair-typed fns, got %d items", len(prog.Items))
	}
}

func TestParsePairDestructure(t *testing.T) {
	src := `
fn make(a: f32, b: f32) -> pair[f32] {
    pair(a, b)
}
fn main() {
    let (x, y) = make(40.0, 70.0);
    println(to_str(x + y));
}
`
	prog, err := ParseSource(src)
	if err != nil {
		t.Fatalf("parse pair destructure: %v", err)
	}
	// main body should contain a multi-name let
	found := false
	for _, it := range prog.Items {
		fn, ok := it.(*ast.FnDef)
		if !ok || fn.Name != "main" || fn.Body == nil {
			continue
		}
		for _, st := range fn.Body.Stmts {
			if ls, ok := st.(*ast.LetStmt); ok && len(ls.Names) == 2 {
				found = true
			}
		}
	}
	if !found {
		t.Fatal("expected let (a, b) destructure in main")
	}
}

func TestParseNestedMultiwayPair(t *testing.T) {
	// v0.48: multi-way and nested destructure patterns
	src := `
fn main() {
    let (a, b, c, d) = pair(1.0, 2.0, 3.0, 4.0);
    let (x, (y, z)) = pair(10.0, pair(20.0, 30.0));
    let m = pair_match(pair(1.0, 2.0, 3.0), 1);
    println(to_str(a + b + c + d + x + y + z + m));
}
`
	prog, err := ParseSource(src)
	if err != nil {
		t.Fatalf("parse nested multiway: %v", err)
	}
	var multi, nested bool
	for _, it := range prog.Items {
		fn, ok := it.(*ast.FnDef)
		if !ok || fn.Body == nil {
			continue
		}
		for _, st := range fn.Body.Stmts {
			ls, ok := st.(*ast.LetStmt)
			if !ok {
				continue
			}
			if len(ls.Names) == 4 {
				multi = true
			}
			if len(ls.Names) == 3 {
				nested = true
			}
		}
	}
	if !multi {
		t.Fatal("expected multi-way let (a,b,c,d)")
	}
	if !nested {
		t.Fatal("expected nested let (x,(y,z)) → 3 names")
	}
}

func TestParseExamples(t *testing.T) {
	dir := examplesDir(t)
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if filepath.Ext(e.Name()) != ".trin" {
			continue
		}
		t.Run(e.Name(), func(t *testing.T) {
			src, err := os.ReadFile(filepath.Join(dir, e.Name()))
			if err != nil {
				t.Fatal(err)
			}
			prog, err := ParseSource(string(src))
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			if len(prog.Items) == 0 {
				t.Fatal("empty program")
			}
		})
	}
}


func TestParseLoopSugar(t *testing.T) {
	src := `
fn main() {
    let a = when (true) do 1;
    let b = over [1.0, 2.0] as x do x + 1.0;
    let c = for i in range(3) do i * i;
    let d = range(1, 4, 1);
    println(to_str(a));
}
`
	prog, err := ParseSource(src)
	if err != nil {
		t.Fatalf("parse loop sugar: %v", err)
	}
	// ensure when desugars to Ternary and over/for to map Call
	var sawTern, sawMap bool
	var walk func(ast.Expr)
	walk = func(e ast.Expr) {
		if e == nil {
			return
		}
		switch n := e.(type) {
		case *ast.Ternary:
			sawTern = true
			walk(n.Gate); walk(n.Pure); walk(n.Each)
		case *ast.Call:
			if v, ok := n.Callee.(*ast.Var); ok && v.Name == "map" {
				sawMap = true
			}
			for _, a := range n.Args {
				walk(a)
			}
		case *ast.Binary:
			walk(n.Left); walk(n.Right)
		case *ast.Unary:
			walk(n.Expr)
		case *ast.Lambda:
			walk(n.Body)
		case *ast.SpanLit:
			for _, el := range n.Elems {
				walk(el)
			}
		case *ast.Index:
			walk(n.Target); walk(n.Index)
		case *ast.Block:
			for _, st := range n.Stmts {
				if ls, ok := st.(*ast.LetStmt); ok {
					walk(ls.Value)
				}
				if es, ok := st.(*ast.ExprStmt); ok {
					walk(es.Expr)
				}
			}
			if n.Expr != nil {
				walk(n.Expr)
			}
		}
	}
	for _, it := range prog.Items {
		fn, ok := it.(*ast.FnDef)
		if !ok || fn.Body == nil {
			continue
		}
		walk(fn.Body)
	}
	if !sawTern {
		t.Fatal("expected when → Ternary desugar")
	}
	if !sawMap {
		t.Fatal("expected over/for → map Call desugar")
	}
}
