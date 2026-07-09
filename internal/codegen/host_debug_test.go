package codegen

import (
	"strings"
	"testing"
)

func TestNullaryHost(t *testing.T) {
	// Synthetic float-only module (no @main).
	ll := `
define float @trin_vec_loop_probe() {
entry:
  ret float 3.600000e+01
}
`
	if llvmFnArity(ll, "trin_vec_loop_probe") != 0 {
		t.Fatalf("arity")
	}
	host := LLVMHostDriver("trin_vec_loop_probe", ll)
	for _, want := range []string{"trin_vec_loop_probe", "printf", "36"} {
		// host doesn't hardcode 36 — just call
		_ = want
	}
	if !strings.Contains(host, "trin_vec_loop_probe") || !strings.Contains(host, "printf") {
		t.Fatalf("host:\n%s", host)
	}
	if !strings.Contains(host, "trin_vec_loop_probe()") && !strings.Contains(host, "trin_vec_loop_probe(void)") {
		t.Fatalf("nullary call:\n%s", host)
	}
	// With @main present, host must be empty.
	llMain := ll + "\ndefine i32 @main() {\nentry:\n  ret i32 0\n}\n"
	if LLVMHostDriver("trin_vec_loop_probe", llMain) != "" {
		t.Fatal("host should be empty when @main present")
	}
}
