package weights

import (
	"path/filepath"
	"testing"
)

func TestNameMapLlamaResolve(t *testing.T) {
	m := NameMapLlama(2, true)
	if m.Resolve("tok_emb") != "token_embd.weight" {
		t.Fatalf("tok_emb %s", m.Resolve("tok_emb"))
	}
	if m.Resolve("lm_head") != "token_embd.weight" {
		t.Fatal("tied lm_head")
	}
	if m.Resolve("L0_q") != "blk.0.attn_q.weight" {
		t.Fatal(m.Resolve("L0_q"))
	}
	if m.Resolve("L1_gate") != "blk.1.ffn_gate.weight" {
		t.Fatal(m.Resolve("L1_gate"))
	}
	if m.Resolve("final_norm") != "output_norm.weight" {
		t.Fatal(m.Resolve("final_norm"))
	}
	// unknown → identity
	if m.Resolve("zzz") != "zzz" {
		t.Fatal("unknown")
	}
	csv := m.ResolveCSV("L0_q,L1_q")
	if csv != "blk.0.attn_q.weight,blk.1.attn_q.weight" {
		t.Fatalf("csv %s", csv)
	}
	u := NameMapLlama(1, false)
	if u.Resolve("lm_head") != "output.weight" {
		t.Fatal("untied")
	}
}

func TestDetectNameMap(t *testing.T) {
	id := DetectNameMap([]string{"tok_emb", "L0_q"}, 0)
	if id.Style != "identity" {
		t.Fatalf("style %s", id.Style)
	}
	llama := DetectNameMap([]string{
		"token_embd.weight",
		"output_norm.weight",
		"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
		"blk.0.attn_output.weight",
		"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
		"blk.0.attn_norm.weight", "blk.0.ffn_norm.weight",
		"blk.1.attn_q.weight", "blk.1.attn_k.weight", "blk.1.attn_v.weight",
		"blk.1.attn_output.weight",
		"blk.1.ffn_gate.weight", "blk.1.ffn_up.weight", "blk.1.ffn_down.weight",
		"blk.1.attn_norm.weight", "blk.1.ffn_norm.weight",
	}, 0)
	if llama.Style != "llama" {
		t.Fatalf("style %s", llama.Style)
	}
	if llama.Resolve("L1_down") != "blk.1.ffn_down.weight" {
		t.Fatal(llama.Resolve("L1_down"))
	}
	// tied: no output.weight
	if llama.Resolve("lm_head") != "token_embd.weight" {
		t.Fatal("detect tied")
	}
	q, k, v, o, gate, up, down, an, fn := ExpandGQASwiGLUCSVs(llama, 2)
	if q != "blk.0.attn_q.weight,blk.1.attn_q.weight" {
		t.Fatalf("q %s", q)
	}
	if gate != "blk.0.ffn_gate.weight,blk.1.ffn_gate.weight" {
		t.Fatalf("gate %s", gate)
	}
	_ = k
	_ = v
	_ = o
	_ = up
	_ = down
	_ = an
	_ = fn
}

func TestSessionNameMapResolveLoad(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "llama_tiny.gguf")
	// same tiny layout as tied_alibi but llama names
	tokEmb := []float32{
		0.1, 0, 0, 0,
		1, 0.2, 0, 0.1,
		0, 1, 0.1, 0,
		0.2, 0, 1, 0,
		0, 0.1, 0, 1,
		0.5, 0.5, 0, 0,
		2, 2, 2, 2,
		0, 0, 0, 0.5,
	}
	I4 := []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
	k0 := []float32{1, 0.1, 0, 0, 0, 1, 0.1, 0}
	v0 := []float32{0.8, 0, 0.1, 0, 0, 0.8, 0, 0.1}
	gate0 := []float32{
		1, 0, 0, 0.05, 0, 1, 0, 0, 0, 0, 1, 0, 0.05, 0, 0, 1,
		0.5, 0.5, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0.5,
	}
	down0 := []float32{
		1, 0, 0, 0, 0.5, 0, 0, 0.5,
		0, 1, 0, 0, 0, 0.5, 0, 0,
		0, 0, 1, 0, 0, 0, 0.5, 0,
		0, 0, 0, 1, 0.5, 0, 0, 0.5,
	}
	q1 := []float32{0.9, 0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0.9}
	k1 := []float32{1, 0, 0, 0.05, 0, 1, 0.05, 0}
	v1 := []float32{0.75, 0, 0.05, 0, 0, 0.75, 0, 0.05}
	gate1 := []float32{
		0.9, 0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0.9,
		0.25, 0.25, 0, 0, 0, 0.25, 0.25, 0, 0, 0, 0.25, 0.25, 0.25, 0, 0, 0.25,
	}
	down1 := []float32{
		0.5, 0, 0, 0, 0.25, 0, 0, 0.25,
		0, 0.5, 0, 0, 0, 0.25, 0, 0,
		0, 0, 0.5, 0, 0, 0, 0.25, 0,
		0, 0, 0, 0.5, 0.25, 0, 0, 0.25,
	}
	g := []float32{1, 1, 1, 1}
	tensors := map[string][]float32{
		"token_embd.weight":       tokEmb,
		"output_norm.weight":      g,
		"blk.0.attn_q.weight":     I4,
		"blk.0.attn_k.weight":     k0,
		"blk.0.attn_v.weight":     v0,
		"blk.0.attn_output.weight": I4,
		"blk.0.ffn_gate.weight":   gate0,
		"blk.0.ffn_up.weight":     gate0,
		"blk.0.ffn_down.weight":   down0,
		"blk.0.attn_norm.weight":  g,
		"blk.0.ffn_norm.weight":   g,
		"blk.1.attn_q.weight":     q1,
		"blk.1.attn_k.weight":     k1,
		"blk.1.attn_v.weight":     v1,
		"blk.1.attn_output.weight": I4,
		"blk.1.ffn_gate.weight":   gate1,
		"blk.1.ffn_up.weight":     gate1,
		"blk.1.ffn_down.weight":   down1,
		"blk.1.attn_norm.weight":  g,
		"blk.1.ffn_norm.weight":   g,
	}
	shapes := map[string][]uint64{
		"token_embd.weight": {8, 4}, "output_norm.weight": {4},
		"blk.0.attn_q.weight": {4, 4}, "blk.0.attn_k.weight": {2, 4}, "blk.0.attn_v.weight": {2, 4},
		"blk.0.attn_output.weight": {4, 4},
		"blk.0.ffn_gate.weight": {8, 4}, "blk.0.ffn_up.weight": {8, 4}, "blk.0.ffn_down.weight": {4, 8},
		"blk.0.attn_norm.weight": {4}, "blk.0.ffn_norm.weight": {4},
		"blk.1.attn_q.weight": {4, 4}, "blk.1.attn_k.weight": {2, 4}, "blk.1.attn_v.weight": {2, 4},
		"blk.1.attn_output.weight": {4, 4},
		"blk.1.ffn_gate.weight": {8, 4}, "blk.1.ffn_up.weight": {8, 4}, "blk.1.ffn_down.weight": {4, 8},
		"blk.1.attn_norm.weight": {4}, "blk.1.ffn_norm.weight": {4},
	}
	if err := WriteGGUFF32(path, tensors, shapes, map[string]string{
		"general.name": "llama-tiny-tied", "trin.tied_emb": "1",
	}); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)

	// without map, short name fails
	if _, err := s.EmbedRow("tok_emb", 1, 4); err == nil {
		t.Fatal("expected miss without map")
	}

	m, err := s.AutoNameMap(0)
	if err != nil {
		t.Fatal(err)
	}
	if m.Style != "llama" {
		t.Fatalf("style %s", m.Style)
	}
	// short name works via map
	row, err := s.EmbedRow("tok_emb", 6, 4)
	if err != nil {
		t.Fatal(err)
	}
	if !near(row[0], 2) {
		t.Fatalf("row %v", row)
	}
	// prepack via short CSV
	built, hits, err := s.PrepackTritCSV("tok_emb,L0_q,L0_k,L0_v,L0_o,L0_gate,L0_up,L1_q,L1_k,L1_v,L1_o,L1_gate,L1_up", 4)
	if err != nil {
		t.Fatal(err)
	}
	if built != 13 || hits != 0 {
		t.Fatalf("built=%d hits=%d", built, hits)
	}
	// down K=8
	b2, h2, err := s.PrepackTritCSV("L0_down,L1_down", 8)
	if err != nil {
		t.Fatal(err)
	}
	if b2 != 2 || h2 != 0 {
		t.Fatalf("down built=%d", b2)
	}
	// second pass all hits (short names resolve to same on-disk keys)
	b3, h3, err := s.PrepackTritCSV("tok_emb,L0_q,L0_k,L0_v,L0_o,L0_gate,L0_up,L1_q,L1_k,L1_v,L1_o,L1_gate,L1_up", 4)
	if err != nil {
		t.Fatal(err)
	}
	if b3 != 0 || h3 != 13 {
		t.Fatalf("hit fail built=%d hits=%d", b3, h3)
	}
	if !s.HasTritPack("tok_emb") || !s.HasTritPack("L0_gate") {
		t.Fatal("has pack short")
	}
	if !s.HasTritPack("token_embd.weight") {
		t.Fatal("has pack real")
	}

	// ALiBi prefill with short CSVs (resolved internally via Load paths)
	dim, hidden := 4, 8
	nHeads, nKV, dh := 2, 1, 2
	xs := append(append([]float32{}, row...), row...) // dummy two tokens of F emb
	// better: token ids 1,2
	x0, _ := s.EmbedRow("tok_emb", 1, 4)
	x1, _ := s.EmbedRow("tok_emb", 2, 4)
	xs = append(append([]float32{}, x0...), x1...)
	Yf, kP, vP, yLast, err := s.PrefillGQASwiGLUStackTritAlibiCSV(
		"L0_q,L1_q", "L0_k,L1_k", "L0_v,L1_v", "L0_o,L1_o",
		"L0_gate,L1_gate", "L0_up,L1_up", "L0_down,L1_down",
		"L0_an,L1_an", "L0_fn,L1_fn",
		xs, nil, nil, dim, hidden, nHeads, nKV, dh, 0, true, 1e-6,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(Yf) != 8 || len(kP) != 2*2*2 || len(yLast) != 4 {
		t.Fatalf("shapes Y=%d K=%d y=%d", len(Yf), len(kP), len(yLast))
	}
	// expand API
	q, _, _, _, gate, _, _, _, _ := ExpandGQASwiGLUCSVs(s.GetNameMap(), 2)
	if q != "blk.0.attn_q.weight,blk.1.attn_q.weight" {
		t.Fatalf("expand %s", q)
	}
	yn, err := s.FinalNorm("final_norm", yLast, dim, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	lg, err := s.LogitsTied("tok_emb", yn, dim)
	if err != nil {
		t.Fatal(err)
	}
	if len(lg) != 8 {
		t.Fatalf("logits %d", len(lg))
	}
	// also via real name
	lg2, err := s.LogitsTied("token_embd.weight", yn, dim)
	if err != nil {
		t.Fatal(err)
	}
	for i := range lg {
		if !near(lg[i], lg2[i]) {
			t.Fatalf("tied parity %v vs %v", lg, lg2)
		}
	}
	_ = vP
	_ = gate
}
