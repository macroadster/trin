package weights

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

// CharVocab is a minimal character / whole-token table tokenizer (v1.04).
// Tokens may be multi-byte strings (e.g. "<pad>", "<eos>"); encode prefers
// longest matching token at each position, then single-byte if present.
type CharVocab struct {
	ID   int32
	Path string
	// Tokens[i] is the string for id i.
	Tokens []string
	// ids maps token string → id (first wins on duplicates).
	ids map[string]int
	// maxLen is longest token string length for greedy encode.
	maxLen int
}

var (
	tokMu   sync.Mutex
	tokMap  = map[int32]*CharVocab{}
	tokNext int32 = 1
)

// LoadVocabFile reads a one-token-per-line vocab file (blank lines skipped).
// Lines may use Go-style escapes; leading/trailing spaces are kept except newline trim.
func LoadVocabFile(path string) (*CharVocab, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return LoadVocabText(path, string(b))
}

// LoadVocabText builds a vocab from newline-separated token strings.
func LoadVocabText(path, text string) (*CharVocab, error) {
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	toks := make([]string, 0, len(lines))
	for _, ln := range lines {
		// keep empty trailing? skip pure empty lines (file end)
		if ln == "" {
			continue
		}
		toks = append(toks, ln)
	}
	if len(toks) == 0 {
		return nil, fmt.Errorf("vocab empty: %s", path)
	}
	v := &CharVocab{Path: path, Tokens: toks, ids: make(map[string]int, len(toks))}
	for i, t := range toks {
		if _, ok := v.ids[t]; !ok {
			v.ids[t] = i
		}
		if len(t) > v.maxLen {
			v.maxLen = len(t)
		}
	}
	tokMu.Lock()
	id := atomic.AddInt32(&tokNext, 1) - 1
	if id <= 0 {
		id = 1
		tokNext = 2
	}
	v.ID = id
	tokMap[id] = v
	tokMu.Unlock()
	return v, nil
}

// OpenVocab is LoadVocabFile with registry.
func OpenVocab(path string) (*CharVocab, error) { return LoadVocabFile(path) }

// GetVocab returns a live vocab session.
func GetVocab(id int32) (*CharVocab, error) {
	tokMu.Lock()
	defer tokMu.Unlock()
	v, ok := tokMap[id]
	if !ok || v == nil {
		return nil, fmt.Errorf("vocab session %d not found (closed?)", id)
	}
	return v, nil
}

// CloseVocab removes the id (idempotent).
func CloseVocab(id int32) error {
	tokMu.Lock()
	delete(tokMap, id)
	tokMu.Unlock()
	return nil
}

// Size is the number of token ids.
func (v *CharVocab) Size() int {
	if v == nil {
		return 0
	}
	return len(v.Tokens)
}

// Token returns the string for id, or "" if OOB.
func (v *CharVocab) Token(id int) string {
	if v == nil || id < 0 || id >= len(v.Tokens) {
		return ""
	}
	return v.Tokens[id]
}

// IDOf returns the id for token string, or -1.
func (v *CharVocab) IDOf(tok string) int {
	if v == nil {
		return -1
	}
	if i, ok := v.ids[tok]; ok {
		return i
	}
	return -1
}

// Encode greedy longest-match: scan prompt left-to-right, prefer longest token
// that matches at the current index. Unknown single runes fall back to id 0
// if "<unk>" exists, else skip one byte and continue (count as unknown).
func (v *CharVocab) Encode(prompt string) []int32 {
	if v == nil || prompt == "" {
		return nil
	}
	unk := -1
	if i, ok := v.ids["<unk>"]; ok {
		unk = i
	} else if i, ok := v.ids["<pad>"]; ok {
		unk = i
	}
	out := make([]int32, 0, len(prompt))
	i := 0
	for i < len(prompt) {
		matched := false
		maxTry := v.maxLen
		if maxTry > len(prompt)-i {
			maxTry = len(prompt) - i
		}
		for L := maxTry; L >= 1; L-- {
			sub := prompt[i : i+L]
			if id, ok := v.ids[sub]; ok {
				out = append(out, int32(id))
				i += L
				matched = true
				break
			}
		}
		if !matched {
			if unk >= 0 {
				out = append(out, int32(unk))
			}
			// advance one byte to avoid infinite loop
			i++
		}
	}
	return out
}

// Decode joins token ids into a string (unknown ids become "").
func (v *CharVocab) Decode(ids []int32) string {
	if v == nil || len(ids) == 0 {
		return ""
	}
	var b strings.Builder
	for _, id := range ids {
		if id >= 0 && int(id) < len(v.Tokens) {
			b.WriteString(v.Tokens[id])
		}
	}
	return b.String()
}

// DecodeOne is a single-id decode.
func (v *CharVocab) DecodeOne(id int) string { return v.Token(id) }

// VocabCount is for tests.
func VocabCount() int {
	tokMu.Lock()
	defer tokMu.Unlock()
	return len(tokMap)
}

// MetaInt parses a GGUF meta string value as int (empty → 0).
func MetaInt(s string) int {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0
	}
	return n
}
