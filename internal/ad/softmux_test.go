package ad

import (
	"math"
	"testing"

	"github.com/ericchien/trin/internal/ast"
)

func almost(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestSoftMuxVJP(t *testing.T) {
	w, pure, each := 0.75, 10.0, 2.0
	y := SoftMux(w, pure, each)
	if !almost(y, 8.0) {
		t.Fatalf("y=%v want 8", y)
	}
	dw, dp, de := SoftMuxVJP(w, pure, each, 1)
	if !almost(dw, pure-each) || !almost(dp, w) || !almost(de, 1-w) {
		t.Fatalf("vjp got %v %v %v", dw, dp, de)
	}
	eps := 1e-6
	num := (SoftMux(w+eps, pure, each) - SoftMux(w-eps, pure, each)) / (2 * eps)
	if !almost(dw, num) {
		t.Fatalf("dw fd %v vs %v", num, dw)
	}
}

func TestTapeComposed(t *testing.T) {
	// z = soft_mux(w, a*x, b) ; w=0.5, a=3, x=4, b=2
	tp := &Tape{}
	w := tp.Const(0.5)
	a := tp.Const(3)
	x := tp.Const(4)
	b := tp.Const(2)
	pure := tp.Mul(a, x)
	z := tp.SoftMuxNode(w, pure, b)
	if !almost(tp.Value(z), 7) {
		t.Fatalf("z=%v", tp.Value(z))
	}
	g := tp.Grad(z, 1)
	if !almost(g[w], 10) || !almost(g[a], 2) || !almost(g[x], 1.5) || !almost(g[b], 0.5) {
		t.Fatalf("grads w=%v a=%v x=%v b=%v", g[w], g[a], g[x], g[b])
	}
}

func TestTapeSubDivNeg(t *testing.T) {
	// y = -(a - b) / c   a=9 b=3 c=2 → -(6)/2 = -3
	tp := &Tape{}
	a := tp.Const(9)
	b := tp.Const(3)
	c := tp.Const(2)
	d := tp.Sub(a, b)
	q := tp.Div(d, c)
	y := tp.Neg(q)
	if !almost(tp.Value(y), -3) {
		t.Fatalf("y=%v", tp.Value(y))
	}
	g := tp.Grad(y, 1)
	if !almost(g[a], -0.5) || !almost(g[b], 0.5) || !almost(g[c], 1.5) {
		t.Fatalf("g a=%v b=%v c=%v", g[a], g[b], g[c])
	}
}

func TestGradExprMini(t *testing.T) {
	env := map[string]float64{"w": 0.4, "a": 2, "x": 3, "b": 1, "c": 5}
	r, err := GradExpr("soft_mux(w, a*x + b, c) * a", env)
	if err != nil {
		t.Fatal(err)
	}
	if !almost(r.Value, 11.6) {
		t.Fatalf("val=%v", r.Value)
	}
	if !almost(r.Grads["a"], 8.2) {
		t.Fatalf("da=%v want 8.2", r.Grads["a"])
	}
	if !almost(r.Grads["w"], 4) {
		t.Fatalf("dw=%v", r.Grads["w"])
	}
}

func TestDifferentiableExprAST(t *testing.T) {
	e := &ast.Ternary{
		Gate: &ast.Var{Name: "w"},
		Pure: &ast.Binary{Op: "*", Left: &ast.Var{Name: "a"}, Right: &ast.Var{Name: "x"}},
		Each: &ast.Var{Name: "b"},
	}
	r, err := DifferentiableExpr(e, map[string]float64{"w": 0.5, "a": 3, "x": 4, "b": 2})
	if err != nil {
		t.Fatal(err)
	}
	if !almost(r.Value, 7) || !almost(r.Grads["a"], 2) {
		t.Fatalf("%+v", r)
	}
}

func TestSumDotTape(t *testing.T) {
	// sum([1,2,3]) = 6, grad all 1
	s, gs := GradSum([]float64{1, 2, 3})
	if !almost(s, 6) || !almost(gs[0], 1) || !almost(gs[2], 1) {
		t.Fatalf("sum %v %v", s, gs)
	}
	// dot([1,2,3],[4,5,6]) = 32; da=[4,5,6] db=[1,2,3]
	d, ga, gb := GradDot([]float64{1, 2, 3}, []float64{4, 5, 6})
	if !almost(d, 32) {
		t.Fatalf("dot=%v", d)
	}
	if !almost(ga[0], 4) || !almost(ga[1], 5) || !almost(gb[2], 3) {
		t.Fatalf("ga=%v gb=%v", ga, gb)
	}
}

func TestGradExprSumDot(t *testing.T) {
	// y = soft_mux(w, sum([a,b,c]), dot([x,y],[p,q]))
	// w=0.5, a=1,b=2,c=3 → sum=6
	// x=1,y=2,p=3,q=4 → dot=11
	// y = 0.5*6 + 0.5*11 = 8.5
	// d/da = w = 0.5
	// d/dx = w*p? wait pure is sum; each is dot
	// d y / d a = w * 1 = 0.5
	// d y / d x = (1-w) * p = 0.5 * 3 = 1.5
	// d y / d w = sum - dot = 6-11 = -5
	env := map[string]float64{
		"w": 0.5, "a": 1, "b": 2, "c": 3,
		"x": 1, "y": 2, "p": 3, "q": 4,
	}
	r, err := GradExpr("soft_mux(w, sum([a,b,c]), dot([x,y],[p,q]))", env)
	if err != nil {
		t.Fatal(err)
	}
	if !almost(r.Value, 8.5) {
		t.Fatalf("val=%v", r.Value)
	}
	if !almost(r.Grads["a"], 0.5) || !almost(r.Grads["c"], 0.5) {
		t.Fatalf("da=%v dc=%v", r.Grads["a"], r.Grads["c"])
	}
	if !almost(r.Grads["x"], 1.5) || !almost(r.Grads["q"], 1.0) {
		t.Fatalf("dx=%v dq=%v", r.Grads["x"], r.Grads["q"])
	}
	if !almost(r.Grads["w"], -5) {
		t.Fatalf("dw=%v", r.Grads["w"])
	}
}

func TestGradExprLossSumScale(t *testing.T) {
	// loss = sum([a*x, b*x, c*x]) = x*(a+b+c)
	// a=1,b=2,c=3,x=4 → 24; d/dx=6; d/da=4
	r, err := GradExpr("sum([a*x, b*x, c*x])", map[string]float64{"a": 1, "b": 2, "c": 3, "x": 4})
	if err != nil {
		t.Fatal(err)
	}
	if !almost(r.Value, 24) || !almost(r.Grads["x"], 6) || !almost(r.Grads["a"], 4) {
		t.Fatalf("%+v", r)
	}
}

func TestGradSpanAddScale(t *testing.T) {
	// sum(span_add([1,2],[3,4])) = 10; da=db=1
	s, ga, gb := GradSpanAdd([]float64{1, 2}, []float64{3, 4})
	if !almost(s, 10) || !almost(ga[0], 1) || !almost(gb[1], 1) {
		t.Fatalf("span_add %v %v %v", s, ga, gb)
	}
	// sum(scale(2, [1,2,3])) = 12; ds=6; dxi=2
	v, ds, gxs := GradScale(2, []float64{1, 2, 3})
	if !almost(v, 12) || !almost(ds, 6) || !almost(gxs[0], 2) || !almost(gxs[2], 2) {
		t.Fatalf("scale %v %v %v", v, ds, gxs)
	}
}

func TestGradExprSpanAddMapMul(t *testing.T) {
	// y = sum(span_add([a,b],[c,d])) = a+b+c+d
	env := map[string]float64{"a": 1, "b": 2, "c": 3, "d": 4}
	r, err := GradExpr("sum(span_add([a,b],[c,d]))", env)
	if err != nil {
		t.Fatal(err)
	}
	if !almost(r.Value, 10) || !almost(r.Grads["a"], 1) || !almost(r.Grads["d"], 1) {
		t.Fatalf("%+v", r)
	}
	// loss = sum(map_mul(s, [x,y,z])) = s*(x+y+z)
	// s=2, x=1,y=2,z=3 → 12; ds=6; dx=2
	env2 := map[string]float64{"s": 2, "x": 1, "y": 2, "z": 3}
	r2, err := GradExpr("sum(map_mul(s, [x,y,z]))", env2)
	if err != nil {
		t.Fatal(err)
	}
	if !almost(r2.Value, 12) || !almost(r2.Grads["s"], 6) || !almost(r2.Grads["x"], 2) {
		t.Fatalf("%+v", r2)
	}
	// composed: soft_mux(w, sum(span_add([a,b],[c,d])), sum(scale(s,[x,y])))
	// w=0.25, a=b=c=d=1 → sum_add=4; s=3,x=2,y=2 → scale sum=12
	// y = 0.25*4 + 0.75*12 = 1+9=10
	// d/dw = 4-12 = -8; d/da = w = 0.25; d/s = (1-w)*(x+y)=0.75*4=3
	env3 := map[string]float64{"w": 0.25, "a": 1, "b": 1, "c": 1, "d": 1, "s": 3, "x": 2, "y": 2}
	r3, err := GradExpr("soft_mux(w, sum(span_add([a,b],[c,d])), sum(scale(s,[x,y])))", env3)
	if err != nil {
		t.Fatal(err)
	}
	if !almost(r3.Value, 10) || !almost(r3.Grads["w"], -8) || !almost(r3.Grads["a"], 0.25) || !almost(r3.Grads["s"], 3) {
		t.Fatalf("%+v", r3)
	}
}

func TestGradExprMapLambda(t *testing.T) {
	// sum(map((x)=>x*a+b, [1,2,3])) = a*(1+2+3)+3*b
	// a=2,b=1 → 2*6+3=15; da=6; db=3; d/dx0 = a = 2 if x were free — x are lits
	env := map[string]float64{"a": 2, "b": 1}
	r, err := GradExpr("sum(map((x)=>x*a+b, [1,2,3]))", env)
	if err != nil {
		t.Fatal(err)
	}
	if !almost(r.Value, 15) || !almost(r.Grads["a"], 6) || !almost(r.Grads["b"], 3) {
		t.Fatalf("%+v", r)
	}

	// free span elements: sum(map((x)=>x*x + s, [u,v])) = u^2+v^2+2s
	// u=3,v=4,s=1 → 9+16+2=27; du=6; dv=8; ds=2
	env2 := map[string]float64{"u": 3, "v": 4, "s": 1}
	r2, err := GradExpr("sum(map((x)=>x*x+s, [u,v]))", env2)
	if err != nil {
		t.Fatal(err)
	}
	if !almost(r2.Value, 27) || !almost(r2.Grads["u"], 6) || !almost(r2.Grads["v"], 8) || !almost(r2.Grads["s"], 2) {
		t.Fatalf("%+v", r2)
	}

	// soft_mux + map: soft_mux(w, sum(map((x)=>x*a,[p,q])), sum(map((x)=>x+b,[r])))
	// w=0.5 a=2 p=3 q=4 → map pure sum = 14; b=1 r=5 → each sum = 6; y = 0.5*14+0.5*6=10
	// dw = 14-6 = 8; da = w*(p+q)=0.5*7=3.5; db = (1-w)*1=0.5; dp=w*a=1
	env3 := map[string]float64{"w": 0.5, "a": 2, "p": 3, "q": 4, "b": 1, "r": 5}
	r3, err := GradExpr("soft_mux(w, sum(map((x)=>x*a, [p,q])), sum(map((x)=>x+b, [r])))", env3)
	if err != nil {
		t.Fatal(err)
	}
	if !almost(r3.Value, 10) || !almost(r3.Grads["w"], 8) || !almost(r3.Grads["a"], 3.5) || !almost(r3.Grads["b"], 0.5) || !almost(r3.Grads["p"], 1) {
		t.Fatalf("%+v", r3)
	}
}

func TestDifferentiableExprMapAST(t *testing.T) {
	// sum(map((x)=>x*2, [a,b])) = 2a+2b
	e := &ast.Call{
		Callee: &ast.Var{Name: "sum"},
		Args: []ast.Expr{
			&ast.Call{
				Callee: &ast.Var{Name: "map"},
				Args: []ast.Expr{
					&ast.Lambda{
						Params: []ast.Param{{Name: "x"}},
						Body:   &ast.Binary{Op: "*", Left: &ast.Var{Name: "x"}, Right: &ast.Literal{Kind: "int", Value: 2}},
					},
					&ast.SpanLit{Elems: []ast.Expr{&ast.Var{Name: "a"}, &ast.Var{Name: "b"}}},
				},
			},
		},
	}
	r, err := DifferentiableExpr(e, map[string]float64{"a": 3, "b": 4})
	if err != nil {
		t.Fatal(err)
	}
	if !almost(r.Value, 14) || !almost(r.Grads["a"], 2) || !almost(r.Grads["b"], 2) {
		t.Fatalf("%+v", r)
	}
}


func TestGradExprNestedMap(t *testing.T) {
	// sum(map((x)=>x*s, map((y)=>y+a, [u,v]))) = s*((u+a)+(v+a))
	env := map[string]float64{"u": 1, "v": 2, "a": 3, "s": 4}
	r, err := GradExpr("sum(map((x)=>x*s, map((y)=>y+a, [u,v])))", env)
	if err != nil {
		t.Fatal(err)
	}
	if !almost(r.Value, 36) || !almost(r.Grads["u"], 4) || !almost(r.Grads["v"], 4) || !almost(r.Grads["a"], 8) || !almost(r.Grads["s"], 9) {
		t.Fatalf("%+v", r)
	}
	// triple nest: sum(map((x)=>x+1, map((y)=>y*2, map((z)=>z-w, [p])))) = ((p-w)*2)+1
	// p=5 w=1 → ((5-1)*2)+1=9; dp=2; dw=-2
	env2 := map[string]float64{"p": 5, "w": 1}
	r2, err := GradExpr("sum(map((x)=>x+1, map((y)=>y*2, map((z)=>z-w, [p]))))", env2)
	if err != nil {
		t.Fatal(err)
	}
	if !almost(r2.Value, 9) || !almost(r2.Grads["p"], 2) || !almost(r2.Grads["w"], -2) {
		t.Fatalf("%+v", r2)
	}
}

func TestGradExprReduce(t *testing.T) {
	// reduce((acc,x)=>acc+x*s, 0, [u,v,w]) = s*(u+v+w)
	env := map[string]float64{"u": 1, "v": 2, "w": 3, "s": 4}
	r, err := GradExpr("reduce((acc,x)=>acc+x*s, 0, [u,v,w])", env)
	if err != nil {
		t.Fatal(err)
	}
	if !almost(r.Value, 24) || !almost(r.Grads["u"], 4) || !almost(r.Grads["v"], 4) || !almost(r.Grads["w"], 4) || !almost(r.Grads["s"], 6) {
		t.Fatalf("%+v", r)
	}
	// fold((acc,x)=>acc*x+b, a, [c,d]) = ((a*c+b)*d+b); a=2 c=3 d=4 b=1 → 29
	env2 := map[string]float64{"a": 2, "b": 1, "c": 3, "d": 4}
	r2, err := GradExpr("fold((acc,x)=>acc*x+b, a, [c,d])", env2)
	if err != nil {
		t.Fatal(err)
	}
	if !almost(r2.Value, 29) || !almost(r2.Grads["a"], 12) || !almost(r2.Grads["b"], 5) || !almost(r2.Grads["c"], 8) || !almost(r2.Grads["d"], 7) {
		t.Fatalf("%+v", r2)
	}
	// reduce over map: reduce((acc,x)=>acc+x, 0, map((z)=>z*s,[p,q])) = s*(p+q)
	env3 := map[string]float64{"p": 2, "q": 5, "s": 3}
	r3, err := GradExpr("reduce((acc,x)=>acc+x, 0, map((z)=>z*s, [p,q]))", env3)
	if err != nil {
		t.Fatal(err)
	}
	if !almost(r3.Value, 21) || !almost(r3.Grads["p"], 3) || !almost(r3.Grads["q"], 3) || !almost(r3.Grads["s"], 7) {
		t.Fatalf("%+v", r3)
	}
}

func TestGradExprZipWith(t *testing.T) {
	// sum(zip_with((x,y)=>x*s+y, [u,v], [p,q])) = s*(u+v)+(p+q)
	env := map[string]float64{"u": 1, "v": 2, "p": 3, "q": 4, "s": 5}
	r, err := GradExpr("sum(zip_with((x,y)=>x*s+y, [u,v], [p,q]))", env)
	if err != nil {
		t.Fatal(err)
	}
	// value = 5*(1+2)+(3+4)=22; du=dv=5; dp=dq=1; ds=u+v=3
	if !almost(r.Value, 22) || !almost(r.Grads["u"], 5) || !almost(r.Grads["v"], 5) || !almost(r.Grads["p"], 1) || !almost(r.Grads["q"], 1) || !almost(r.Grads["s"], 3) {
		t.Fatalf("%+v", r)
	}
	// zip_with over map: sum(zip_with((x,y)=>x+y*w, map((z)=>z+1,[a,b]), [c,d]))
	// = (a+1 + c*w) + (b+1 + d*w); a=1 b=2 c=3 d=4 w=0.5 → 2+1.5 + 3+2 = 8.5
	env2 := map[string]float64{"a": 1, "b": 2, "c": 3, "d": 4, "w": 0.5}
	r2, err := GradExpr("sum(zip_with((x,y)=>x+y*w, map((z)=>z+1, [a,b]), [c,d]))", env2)
	if err != nil {
		t.Fatal(err)
	}
	if !almost(r2.Value, 8.5) || !almost(r2.Grads["a"], 1) || !almost(r2.Grads["b"], 1) || !almost(r2.Grads["c"], 0.5) || !almost(r2.Grads["d"], 0.5) || !almost(r2.Grads["w"], 7) {
		t.Fatalf("%+v", r2)
	}
}

func TestGradExprZipWith3(t *testing.T) {
	// sum(zip_with((x,y,z)=>x*s+y+z, [u,v], [p,q], [r,t])) = s*(u+v)+(p+q)+(r+t)
	env := map[string]float64{"u": 1, "v": 2, "p": 3, "q": 4, "r": 5, "t": 6, "s": 7}
	r, err := GradExpr("sum(zip_with((x,y,z)=>x*s+y+z, [u,v], [p,q], [r,t]))", env)
	if err != nil {
		t.Fatal(err)
	}
	// 7*(1+2)+(3+4)+(5+6)=21+7+11=39; du=dv=7; dp=dq=dr=dt=1; ds=3
	if !almost(r.Value, 39) || !almost(r.Grads["u"], 7) || !almost(r.Grads["v"], 7) ||
		!almost(r.Grads["p"], 1) || !almost(r.Grads["q"], 1) || !almost(r.Grads["r"], 1) ||
		!almost(r.Grads["t"], 1) || !almost(r.Grads["s"], 3) {
		t.Fatalf("%+v", r)
	}
}

func TestGradExprScan(t *testing.T) {
	// sum(scan((acc,x)=>acc+x*s, 0, [a,b,c])) = s*(3a+2b+c)
	env := map[string]float64{"a": 1, "b": 2, "c": 3, "s": 4}
	r, err := GradExpr("sum(scan((acc,x)=>acc+x*s, 0, [a,b,c]))", env)
	if err != nil {
		t.Fatal(err)
	}
	// value=4*(3+4+3)=40; da=12 db=8 dc=4 ds=10
	if !almost(r.Value, 40) || !almost(r.Grads["a"], 12) || !almost(r.Grads["b"], 8) ||
		!almost(r.Grads["c"], 4) || !almost(r.Grads["s"], 10) {
		t.Fatalf("%+v", r)
	}
	// compose with reduce + ternary zip
	env2 := map[string]float64{"p": 2, "q": 3, "r": 4, "w": 0.5}
	r2, err := GradExpr("reduce((acc,x)=>acc+x, 0, zip_with((x,y,z)=>x+y*w+z, [p], [q], [r]))", env2)
	if err != nil {
		t.Fatal(err)
	}
	// 2+3*0.5+4=7.5; dp=1 dq=0.5 dr=1 dw=3
	if !almost(r2.Value, 7.5) || !almost(r2.Grads["p"], 1) || !almost(r2.Grads["q"], 0.5) ||
		!almost(r2.Grads["r"], 1) || !almost(r2.Grads["w"], 3) {
		t.Fatalf("%+v", r2)
	}
}

func TestGradExprFilterNary(t *testing.T) {
	// 4-ary zip
	env := map[string]float64{"a": 1, "b": 2, "c": 10, "d": 20, "e": 0.5, "f": 1.5, "g": 2, "h": 3}
	r, err := GradExpr("sum(zip_with((x,y,z,w)=>x*w+y+z, [a,b], [c,d], [e,f], [g,h]))", env)
	if err != nil {
		t.Fatal(err)
	}
	// 40; da=2 db=3 dc=1 dg=1 dh=2
	if !almost(r.Value, 40) || !almost(r.Grads["a"], 2) || !almost(r.Grads["b"], 3) ||
		!almost(r.Grads["c"], 1) || !almost(r.Grads["g"], 1) || !almost(r.Grads["h"], 2) {
		t.Fatalf("%+v", r)
	}
	// filter hard keep
	env2 := map[string]float64{"p": 1, "q": 2, "r": 3, "k": 2}
	r2, err := GradExpr("sum(filter((x)=>x-k, [p,q,r]))", env2)
	if err != nil {
		t.Fatal(err)
	}
	// keep p,r sum=4; dq=0 (dropped); dk=0 (pred disconnected from out)
	if !almost(r2.Value, 4) || !almost(r2.Grads["p"], 1) || !almost(r2.Grads["q"], 0) ||
		!almost(r2.Grads["r"], 1) || !almost(r2.Grads["k"], 0) {
		t.Fatalf("%+v", r2)
	}
	// reduce over 4-ary zip
	env3 := map[string]float64{"p": 1, "q": 2, "r": 3, "s": 4}
	r3, err := GradExpr("reduce((acc,x)=>acc+x, 0, zip_with((x,y,z,w)=>x+y+z*w, [p], [q], [r], [s]))", env3)
	if err != nil {
		t.Fatal(err)
	}
	if !almost(r3.Value, 15) || !almost(r3.Grads["p"], 1) || !almost(r3.Grads["q"], 1) ||
		!almost(r3.Grads["r"], 4) || !almost(r3.Grads["s"], 3) {
		t.Fatalf("%+v", r3)
	}
}
