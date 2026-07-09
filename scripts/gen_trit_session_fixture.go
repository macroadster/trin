//go:build ignore

package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/ericchien/trin/internal/weights"
)

// Writes demos/weights/trit_session.gguf — F32 row-major M×K for session linear_trit demos.
func main() {
	root := "."
	if len(os.Args) > 1 {
		root = os.Args[1]
	}
	M, K := 32, 256
	vals := make([]float32, M*K)
	pat := []float32{1.2, 0, -0.9, 0.4, -1.1, 0.05, 0.8, -0.3, 0.6, -0.6, 0.15, -0.2}
	for i := range vals {
		vals[i] = pat[i%len(pat)] * (1 + 0.01*float32(i%17))
	}
	out := filepath.Join(root, "demos/weights/trit_session.gguf")
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		panic(err)
	}
	if err := weights.WriteGGUFF32(out, map[string][]float32{"weight": vals},
		map[string][]uint64{"weight": {uint64(M), uint64(K)}},
		map[string]string{"general.name": "trin-trit-session"}); err != nil {
		panic(err)
	}
	fmt.Printf("wrote %s M=%d K=%d n=%d\n", out, M, K, len(vals))
}
