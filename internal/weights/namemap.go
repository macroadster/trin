package weights

import (
	"fmt"
	"strconv"
	"strings"
)

// NameMap resolves short/canonical Trin names to on-disk tensor names.
// Used for llama.cpp-style GGUF layouts and Trin’s own short names (identity).
//
// Canonical keys (short form used by prior demos):
//
//	tok_emb, lm_head, final_norm
//	L{i}_q, L{i}_k, L{i}_v, L{i}_o, L{i}_gate, L{i}_up, L{i}_down, L{i}_an, L{i}_fn
//
// Llama/gguf aliases (llama.cpp export style):
//
//	token_embd.weight, output.weight, output_norm.weight
//	blk.{i}.attn_q.weight / attn_k / attn_v / attn_output
//	blk.{i}.ffn_gate / ffn_up / ffn_down / attn_norm / ffn_norm
type NameMap struct {
	// Alias: short → real. When empty/nil, identity for unknown.
	Alias map[string]string
	// Style records how the map was built: "identity" | "llama" | "custom"
	Style string
}

// Resolve returns the on-disk name for short, or short itself if unmapped.
func (m *NameMap) Resolve(short string) string {
	if m == nil || m.Alias == nil {
		return short
	}
	if r, ok := m.Alias[short]; ok && r != "" {
		return r
	}
	return short
}

// ResolveCSV maps each comma-separated short name.
func (m *NameMap) ResolveCSV(csv string) string {
	parts := splitCSV(csv)
	if len(parts) == 0 {
		return ""
	}
	out := make([]string, len(parts))
	for i, p := range parts {
		out[i] = m.Resolve(p)
	}
	return strings.Join(out, ",")
}

// Invert returns real→short (first wins on collisions).
func (m *NameMap) Invert() map[string]string {
	if m == nil || m.Alias == nil {
		return nil
	}
	inv := make(map[string]string, len(m.Alias))
	for k, v := range m.Alias {
		if v == "" {
			continue
		}
		if _, ok := inv[v]; !ok {
			inv[v] = k
		}
	}
	return inv
}

// LlamaAliasLayer fills short→llama names for one layer index i.
func LlamaAliasLayer(i int) map[string]string {
	si := strconv.Itoa(i)
	blk := "blk." + si + "."
	return map[string]string{
		"L" + si + "_q":    blk + "attn_q.weight",
		"L" + si + "_k":    blk + "attn_k.weight",
		"L" + si + "_v":    blk + "attn_v.weight",
		"L" + si + "_o":    blk + "attn_output.weight",
		"L" + si + "_gate": blk + "ffn_gate.weight",
		"L" + si + "_up":   blk + "ffn_up.weight",
		"L" + si + "_down": blk + "ffn_down.weight",
		"L" + si + "_an":   blk + "attn_norm.weight",
		"L" + si + "_fn":   blk + "ffn_norm.weight",
	}
}

// LlamaAliases builds a full map for nLayers (0..n-1) plus emb/head/norm.
// When tied is true, lm_head aliases to token_embd.weight (no output.weight).
func LlamaAliases(nLayers int, tied bool) map[string]string {
	m := map[string]string{
		"tok_emb":    "token_embd.weight",
		"final_norm": "output_norm.weight",
	}
	if tied {
		m["lm_head"] = "token_embd.weight"
	} else {
		m["lm_head"] = "output.weight"
	}
	for i := 0; i < nLayers; i++ {
		for k, v := range LlamaAliasLayer(i) {
			m[k] = v
		}
	}
	return m
}

// NameMapLlama returns a NameMap with llama.cpp-style aliases.
func NameMapLlama(nLayers int, tied bool) *NameMap {
	return &NameMap{Alias: LlamaAliases(nLayers, tied), Style: "llama"}
}

// NameMapIdentity returns a passthrough map (Style=identity).
func NameMapIdentity() *NameMap {
	return &NameMap{Alias: map[string]string{}, Style: "identity"}
}

// DetectNameMap inspects session tensor names and returns the best adapter.
// Prefers llama layout when token_embd.weight or blk.0.attn_q.weight exists;
// otherwise identity.
func DetectNameMap(names []string, nLayersHint int) *NameMap {
	set := make(map[string]bool, len(names))
	for _, n := range names {
		set[n] = true
	}
	hasLlama := set["token_embd.weight"] || set["blk.0.attn_q.weight"] || set["blk.0.attn_q"]
	if !hasLlama {
		// also detect bare names without .weight suffix
		for n := range set {
			if strings.HasPrefix(n, "blk.") && strings.Contains(n, "attn_q") {
				hasLlama = true
				break
			}
		}
	}
	if !hasLlama {
		return NameMapIdentity()
	}
	// count layers
	nL := nLayersHint
	if nL <= 0 {
		for i := 0; i < 256; i++ {
			a := fmt.Sprintf("blk.%d.attn_q.weight", i)
			b := fmt.Sprintf("blk.%d.attn_q", i)
			if set[a] || set[b] {
				nL = i + 1
			} else if i > 0 {
				break
			} else {
				break
			}
		}
		if nL <= 0 {
			nL = 1
		}
	}
	tied := !set["output.weight"] && !set["output"]
	m := NameMapLlama(nL, tied)
	// If tensors omit ".weight" suffix, rewrite aliases.
	if !set["token_embd.weight"] && set["token_embd"] {
		m = stripWeightSuffix(m)
	} else if set["token_embd.weight"] {
		// default llama with .weight is fine
	} else if set["token_embd"] {
		m = stripWeightSuffix(m)
	}
	// Prefer present suffix form per key
	m.Alias = preferPresent(m.Alias, set)
	return m
}

func stripWeightSuffix(m *NameMap) *NameMap {
	out := make(map[string]string, len(m.Alias))
	for k, v := range m.Alias {
		out[k] = strings.TrimSuffix(v, ".weight")
	}
	return &NameMap{Alias: out, Style: m.Style}
}

func preferPresent(alias map[string]string, set map[string]bool) map[string]string {
	out := make(map[string]string, len(alias))
	for k, v := range alias {
		if set[v] {
			out[k] = v
			continue
		}
		// try with/without .weight
		if strings.HasSuffix(v, ".weight") {
			alt := strings.TrimSuffix(v, ".weight")
			if set[alt] {
				out[k] = alt
				continue
			}
		} else {
			alt := v + ".weight"
			if set[alt] {
				out[k] = alt
				continue
			}
		}
		out[k] = v // keep even if missing; resolve will fail later at load
	}
	return out
}

// ExpandLayerCSV builds a CSV of short names L0_x,...,L{n-1}_x for role suffix.
// role is "q","k","v","o","gate","up","down","an","fn".
func ExpandLayerCSV(nLayers int, role string) string {
	if nLayers <= 0 {
		return ""
	}
	parts := make([]string, nLayers)
	for i := 0; i < nLayers; i++ {
		parts[i] = fmt.Sprintf("L%d_%s", i, role)
	}
	return strings.Join(parts, ",")
}

// ExpandGQASwiGLUCSVs returns resolved CSVs for nLayers using m (or identity).
func ExpandGQASwiGLUCSVs(m *NameMap, nLayers int) (q, k, v, o, gate, up, down, an, fn string) {
	if m == nil {
		m = NameMapIdentity()
	}
	q = m.ResolveCSV(ExpandLayerCSV(nLayers, "q"))
	k = m.ResolveCSV(ExpandLayerCSV(nLayers, "k"))
	v = m.ResolveCSV(ExpandLayerCSV(nLayers, "v"))
	o = m.ResolveCSV(ExpandLayerCSV(nLayers, "o"))
	gate = m.ResolveCSV(ExpandLayerCSV(nLayers, "gate"))
	up = m.ResolveCSV(ExpandLayerCSV(nLayers, "up"))
	down = m.ResolveCSV(ExpandLayerCSV(nLayers, "down"))
	an = m.ResolveCSV(ExpandLayerCSV(nLayers, "an"))
	fn = m.ResolveCSV(ExpandLayerCSV(nLayers, "fn"))
	return
}

// Session name-map helpers on WeightSession

// SetNameMap attaches a name map (nil clears → identity).
func (s *WeightSession) SetNameMap(m *NameMap) {
	if s == nil {
		return
	}
	s.nameMu.Lock()
	s.nameMap = m
	s.nameMu.Unlock()
}

// GetNameMap returns the active map or identity.
func (s *WeightSession) GetNameMap() *NameMap {
	if s == nil {
		return NameMapIdentity()
	}
	s.nameMu.Lock()
	m := s.nameMap
	s.nameMu.Unlock()
	if m == nil {
		return NameMapIdentity()
	}
	return m
}

// AutoNameMap detects and sets a map from listed tensors.
func (s *WeightSession) AutoNameMap(nLayersHint int) (*NameMap, error) {
	names, err := s.List()
	if err != nil {
		return nil, err
	}
	m := DetectNameMap(names, nLayersHint)
	s.SetNameMap(m)
	return m, nil
}

// ResolveName applies the session map (identity if unset).
func (s *WeightSession) ResolveName(short string) string {
	return s.GetNameMap().Resolve(short)
}

// ResolveNameCSV applies the session map to a CSV.
func (s *WeightSession) ResolveNameCSV(csv string) string {
	return s.GetNameMap().ResolveCSV(csv)
}
