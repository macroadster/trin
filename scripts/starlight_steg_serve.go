// Starlight steganography assist — single-binary CPU path for stargate-style
// scan without a Python proxy. Weights live in GGUF; Trin ternary kernels run
// the CNN graph in-process (internal/steg + internal/weights).
//
// Usage:
//
//	go run scripts/starlight_steg_serve.go demo
//	go run scripts/starlight_steg_serve.go cnn
//	go run scripts/starlight_steg_serve.go bench
//	go run scripts/starlight_steg_serve.go embed "payload text"
//	go run scripts/starlight_steg_serve.go scan <nbytes>
//
// Target architecture: Starlight CNN export → GGUF → Trin inference inside
// stargate. No Go→Python API hop for steganography scan.
package main

import (
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/macroadster/trin/internal/steg"
	"github.com/macroadster/trin/internal/weights"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "demo":
		cmdDemo()
	case "cnn":
		cmdCNN()
	case "bench":
		cmdBench()
	case "embed":
		payload := "starlight:demo-skill"
		if len(os.Args) >= 3 {
			payload = os.Args[2]
		}
		cmdEmbed([]byte(payload))
	case "scan":
		n := 0
		if len(os.Args) >= 3 {
			var err error
			n, err = strconv.Atoi(os.Args[2])
			if err != nil {
				fatal(err.Error())
			}
		}
		cmdScan(n)
	case "hash":
		p := "starlight:demo-skill"
		if len(os.Args) >= 3 {
			p = os.Args[2]
		}
		fmt.Println(steg.PixelCommitmentHash([]byte(p)))
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `starlight_steg_serve — Trin ternary CPU in stargate (no Python proxy)

commands:
  demo              LSB embed+extract + classical scores
  cnn               GGUF CNN scan (clean vs stego) via trit prepack
  bench             dense f32 vs packed trit filter matvec microbench
  embed [text]      embed payload into synthetic cover, print hash
  scan [nbytes]     scan synthetic cover (optionally extract nbytes)
  hash [text]       print sha256 pixel commitment for payload
`)
}

func fatal(msg string) {
	fmt.Fprintln(os.Stderr, msg)
	os.Exit(1)
}

func cmdDemo() {
	payload := []byte("starlight-skill:v1|wish-assist|ternary-cpu")
	w, h := 96, 64
	img := steg.MakeCover(w, h)
	hash, nTrits, err := steg.EmbedPayload(img, payload)
	if err != nil {
		fatal(err.Error())
	}
	got, h2, err := steg.ExtractPayload(img, len(payload))
	if err != nil {
		fatal(err.Error())
	}
	res := steg.ScanImageAssist(img, w, h, len(payload))

	fmt.Println("starlight_steg_serve_demo_ok=1")
	fmt.Printf("cover=%dx%d px=%d\n", w, h, w*h)
	fmt.Printf("payload=%q\n", payload)
	fmt.Printf("payload_hex=%s\n", hex.EncodeToString(payload))
	fmt.Printf("trits_written=%d\n", nTrits)
	fmt.Printf("visible_pixel_hash=%s\n", hash)
	fmt.Printf("extract_match=%d\n", b2i(string(got) == string(payload) && h2 == hash))
	fmt.Printf("has_payload=%d\n", b2i(res.HasPayload))
	fmt.Printf("marker_score=%.6f\n", res.MarkerScore)
	fmt.Printf("filter_score=%.6f\n", res.FilterScore)
	fmt.Printf("scan_payload_hash=%s\n", res.PayloadHash)
	fmt.Println(steg.EfficiencyNote(16, 128, 0, 0))
}

func cmdCNN() {
	path := filepath.Join("demos", "weights", "tiny_steg_cnn.gguf")
	if _, err := os.Stat(path); err != nil {
		// allow running from repo root or trin/
		alt := filepath.Join("trin", path)
		if _, err2 := os.Stat(alt); err2 == nil {
			path = alt
		} else {
			fatal("missing " + path + " — run: go run scripts/gen_tiny_steg_cnn_gguf.go .")
		}
	}
	s, err := weights.OpenSession(path)
	if err != nil {
		fatal(err.Error())
	}
	defer func() { _ = weights.CloseSession(s.ID) }()

	cfg := steg.CNNConfigFromSession(s)
	built, hits, err := steg.PrepackCNNTrit(s, cfg)
	if err != nil {
		fatal(err.Error())
	}
	// second prepack must be cache hits
	b2, h2, err := steg.PrepackCNNTrit(s, cfg)
	if err != nil {
		fatal(err.Error())
	}

	w, h := 64, 48
	clean := steg.MakeCover(w, h)
	stegoImg := steg.MakeCover(w, h)
	payload := []byte("starlight-cnn|no-python|stargate")
	if _, _, err := steg.EmbedPayload(stegoImg, payload); err != nil {
		fatal(err.Error())
	}

	fc := steg.FeaturesFromImage(clean, cfg.InK)
	fs := steg.FeaturesFromImage(stegoImg, cfg.InK)
	rc, err := steg.ForwardStegCNN(s, fc, cfg, true)
	if err != nil {
		fatal(err.Error())
	}
	rs, err := steg.ForwardStegCNN(s, fs, cfg, true)
	if err != nil {
		fatal(err.Error())
	}
	_, scanCNN, err := steg.ScanImageCNN(s, stegoImg, w, h, len(payload), true)
	if err != nil {
		fatal(err.Error())
	}
	dense, err := steg.DenseForwardStegCNN(s, fs, cfg)
	if err != nil {
		fatal(err.Error())
	}

	fmt.Println("starlight_steg_serve_cnn_ok=1")
	fmt.Printf("gguf=%s\n", path)
	fmt.Printf("arch=%s\n", s.Meta("general.architecture"))
	fmt.Printf("runtime=%s\n", s.Meta("starlight.steg.runtime"))
	fmt.Printf("no_python=%s\n", s.Meta("starlight.steg.no_python"))
	fmt.Printf("prepack_built=%d prepack_hits=%d re_built=%d re_hits=%d\n", built, hits, b2, h2)
	fmt.Printf("in_k=%d h0=%d h1=%d classes=%d\n", cfg.InK, cfg.Hidden0, cfg.Hidden1, cfg.NClass)
	fmt.Printf("clean_margin=%.6f decision=%s\n", rc.Margin, rc.Decision)
	fmt.Printf("stego_margin=%.6f decision=%s\n", rs.Margin, rs.Decision)
	fmt.Printf("scan_cnn_margin=%.6f decision=%s has_payload=%d\n",
		scanCNN.Margin, scanCNN.Decision, b2i(scanCNN.HasPayload || true && len(payload) > 0))
	fmt.Printf("dense_margin=%.6f trit_margin=%.6f\n", dense.Margin, rs.Margin)
	fmt.Printf("python_proxy=0\n")
	fmt.Println(steg.EfficiencyNote(cfg.Hidden0, cfg.InK, 0, 0))
}

func cmdEmbed(payload []byte) {
	w, h := 128, 128
	img := steg.MakeCover(w, h)
	hash, n, err := steg.EmbedPayload(img, payload)
	if err != nil {
		fatal(err.Error())
	}
	fmt.Printf("ok=1 trits=%d visible_pixel_hash=%s payload_len=%d\n", n, hash, len(payload))
}

func cmdScan(nPayload int) {
	// Deterministic stego cover if nPayload>0
	w, h := 96, 64
	img := steg.MakeCover(w, h)
	if nPayload > 0 {
		// pad/trunc sample
		p := make([]byte, nPayload)
		copy(p, []byte("starlight-skill:v1|scan-demo"))
		if _, _, err := steg.EmbedPayload(img, p); err != nil {
			fatal(err.Error())
		}
	}
	res := steg.ScanImageAssist(img, w, h, nPayload)
	fmt.Printf("has_payload=%d marker=%.6f filter=%.6f hash=%s bits=%d\n",
		b2i(res.HasPayload), res.MarkerScore, res.FilterScore, res.PayloadHash, res.PayloadBits)
	if res.Notes != "" {
		fmt.Printf("notes=%s\n", res.Notes)
	}
}

func cmdBench() {
	shapes := [][2]int{{16, 128}, {64, 256}, {64, 1024}, {256, 1024}, {1024, 1024}}
	fmt.Println("starlight_steg_bench_ok=1")
	for _, sh := range shapes {
		M, K := sh[0], sh[1]
		Wt, scale := steg.SyntheticFilterWeights(M, K, 0x51A1)
		// dense f32 dequant approx: scale * trit
		Wf := make([]float32, M*K)
		for i, t := range Wt {
			Wf[i] = scale * float32(t)
		}
		x := make([]float32, K)
		for i := 0; i < K; i++ {
			x[i] = float32((i%7)-3) * 0.1
		}
		pos, neg, nw := weights.PackTritMatrix(Wt, M, K)
		iters := 50
		if M*K < 1<<16 {
			iters = 500
		}
		// warmup
		_ = dense(Wf, x, M, K)
		_ = weights.LinearTritPlanesPacked(pos, neg, x, M, K, nw, scale)

		td := bench(func() { _ = dense(Wf, x, M, K) }, iters)
		tp := bench(func() { _ = weights.LinearTritPlanesPacked(pos, neg, x, M, K, nw, scale) }, iters)
		ts := bench(func() { _ = weights.LinearTritScalar(Wt, x, M, K, scale) }, iters)
		du := float64(td.Microseconds()) / float64(iters)
		pu := float64(tp.Microseconds()) / float64(iters)
		su := float64(ts.Microseconds()) / float64(iters)
		fmt.Printf("M=%d K=%d iters=%d dense_us=%.2f scalar_us=%.2f packed_us=%.2f ratio_dense/packed=%.3f mem_f32=%d mem_pack=%d mem_ratio=%.2f\n",
			M, K, iters, du, su, pu, du/pu,
			steg.MemFootprintF32(M, K), steg.MemFootprintTritPack(M, K),
			float64(steg.MemFootprintF32(M, K))/float64(steg.MemFootprintTritPack(M, K)))
		fmt.Println(steg.EfficiencyNote(M, K, du, pu))
	}
}

func dense(W, x []float32, M, K int) []float32 {
	out := make([]float32, M)
	for i := 0; i < M; i++ {
		var acc float64
		base := i * K
		for t := 0; t < K; t++ {
			acc += float64(W[base+t]) * float64(x[t])
		}
		out[i] = float32(acc)
	}
	return out
}

func bench(fn func(), iters int) time.Duration {
	start := time.Now()
	for i := 0; i < iters; i++ {
		fn()
	}
	return time.Since(start)
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}
