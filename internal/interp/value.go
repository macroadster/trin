package interp

import (
	"fmt"
	"strings"
)

type Kind int

const (
	VVoid Kind = iota
	VBit
	VTrit
	VUnit
	VI32
	VF32
	VStr
	VSpan
	VFn
	VNative
)

type Value struct {
	Kind Kind
	Bit  int
	Trit int
	Unit float64
	I32  int
	F32  float64
	Str  string
	Span []Value
	Fn   *Closure
	Nat  NativeFn
}

type NativeFn func(args []Value) (Value, error)

type Closure struct {
	Params []string
	Body   any // ast.Expr — kept as any to avoid cycles in interface storage
	Env    *Env
}

func Void() Value             { return Value{Kind: VVoid} }
func Bit(v int) Value         { return Value{Kind: VBit, Bit: v} }
func Trit(v int) Value        { return Value{Kind: VTrit, Trit: v} }
func Unit(v float64) Value    { return Value{Kind: VUnit, Unit: v} }
func I32(v int) Value         { return Value{Kind: VI32, I32: v} }
func F32(v float64) Value     { return Value{Kind: VF32, F32: v} }
func Str(v string) Value      { return Value{Kind: VStr, Str: v} }
func Span(items []Value) Value {
	return Value{Kind: VSpan, Span: items}
}

func (v Value) String() string {
	switch v.Kind {
	case VVoid:
		return ""
	case VBit:
		return fmt.Sprintf("%d", v.Bit)
	case VTrit:
		return fmt.Sprintf("%d", v.Trit)
	case VUnit:
		// match Python: print bare float of unit.v inside str_concat path via number; to_str for unit uses .v
		s := fmt.Sprintf("%v", v.Unit)
		return s
	case VI32:
		return fmt.Sprintf("%d", v.I32)
	case VF32:
		// Python str(2.0) => "2.0"; clamp tiny float noise for AD demos
		if v.F32 == float64(int64(v.F32)) {
			return fmt.Sprintf("%.1f", v.F32)
		}
		// banker's-safe positive rounding to ~12 decimals
		x := v.F32
		sign := 1.0
		if x < 0 {
			sign = -1
			x = -x
		}
		r := float64(int64(x*1e12+0.5)) / 1e12 * sign
		return fmt.Sprintf("%g", r)
	case VStr:
		return v.Str
	case VSpan:
		parts := make([]string, len(v.Span))
		for i, x := range v.Span {
			parts[i] = x.String()
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case VFn:
		return fmt.Sprintf("<fn/%d>", len(v.Fn.Params))
	case VNative:
		return "<native>"
	default:
		return "?"
	}
}

func (v Value) AsFloat() float64 {
	switch v.Kind {
	case VBit:
		return float64(v.Bit)
	case VTrit:
		return float64(v.Trit)
	case VUnit:
		return v.Unit
	case VI32:
		return float64(v.I32)
	case VF32:
		return v.F32
	default:
		return 0
	}
}

func (v Value) AsInt() int {
	switch v.Kind {
	case VBit:
		return v.Bit
	case VTrit:
		return v.Trit
	case VUnit:
		return int(v.Unit)
	case VI32:
		return v.I32
	case VF32:
		return int(v.F32)
	default:
		return 0
	}
}

type Env struct {
	Parent   *Env
	Bindings map[string]Value
}

func NewEnv(parent *Env) *Env {
	return &Env{Parent: parent, Bindings: map[string]Value{}}
}

func (e *Env) Get(name string) (Value, bool) {
	if v, ok := e.Bindings[name]; ok {
		return v, true
	}
	if e.Parent != nil {
		return e.Parent.Get(name)
	}
	return Void(), false
}

func (e *Env) Set(name string, v Value) {
	e.Bindings[name] = v
}

func GateWeight(v Value) (float64, error) {
	switch v.Kind {
	case VBit:
		return float64(v.Bit), nil
	case VUnit:
		w := v.Unit
		if w < 0 {
			w = 0
		}
		if w > 1 {
			w = 1
		}
		return w, nil
	case VTrit:
		switch v.Trit {
		case -1:
			return 0, nil
		case 0:
			return 0.5, nil
		case 1:
			return 1, nil
		}
	case VI32:
		if v.I32 == 0 || v.I32 == 1 {
			// Python rejects plain int as gate
			return 0, fmt.Errorf("value of type int cannot be used as gate")
		}
	}
	return 0, fmt.Errorf("value cannot be used as gate: %v", v)
}

func Blend(w float64, a, b Value) Value {
	if a.Kind == VUnit || b.Kind == VUnit {
		return Unit(w*a.AsFloat() + (1-w)*b.AsFloat())
	}
	if (a.Kind == VI32 || a.Kind == VF32) && (b.Kind == VI32 || b.Kind == VF32) {
		return F32(w*a.AsFloat() + (1-w)*b.AsFloat())
	}
	if a.Kind == VBit && b.Kind == VBit {
		if w*float64(a.Bit)+(1-w)*float64(b.Bit) >= 0.5 {
			return Bit(1)
		}
		return Bit(0)
	}
	if a.Kind == VStr && b.Kind == VStr {
		if w >= 0.5 {
			return a
		}
		return b
	}
	if a.Kind == VSpan && b.Kind == VSpan && len(a.Span) == len(b.Span) {
		out := make([]Value, len(a.Span))
		for i := range a.Span {
			out[i] = Blend(w, a.Span[i], b.Span[i])
		}
		return Span(out)
	}
	if w >= 0.5 {
		return a
	}
	return b
}

func TruthyBit(v Value) Value {
	switch v.Kind {
	case VBit:
		return v
	case VUnit:
		if v.Unit >= 0.5 {
			return Bit(1)
		}
		return Bit(0)
	case VI32:
		if v.I32 != 0 {
			return Bit(1)
		}
		return Bit(0)
	case VF32:
		if v.F32 != 0 {
			return Bit(1)
		}
		return Bit(0)
	case VStr:
		if len(v.Str) > 0 {
			return Bit(1)
		}
		return Bit(0)
	case VSpan:
		if len(v.Span) > 0 {
			return Bit(1)
		}
		return Bit(0)
	default:
		return Bit(0)
	}
}
