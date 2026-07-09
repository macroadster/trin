package weights

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"unicode/utf8"
)

// BPETokenizer is a minimal GPT-2 style BPE loader (v1.05).
// Sidecar files only — no network fetch.
//
//   vocab path: one token per line (id = line index), same as CharVocab
//   merges path: optional rank order (one "a b" pair per line; # comments + blank skipped)
//
// Encode: byte→char vocab tokens when present, else greedily apply BPE merges by rank.
// Without merges, falls back to longest-match CharVocab semantics on the same table.
type BPETokenizer struct {
	ID     int32
	Vocab  *CharVocab
	// merges maps "left right" → rank (lower rank merges first)
	merges map[string]int
	// maxRank for unknown pairs
	maxRank int
}

var (
	bpeMu   sync.Mutex
	bpeMap  = map[int32]*BPETokenizer{}
	bpeNext int32 = 1
)

// OpenBPE loads vocab + optional merges from local paths.
// mergesPath may be "" — then encode uses CharVocab.Encode only.
func OpenBPE(vocabPath, mergesPath string) (*BPETokenizer, error) {
	v, err := LoadVocabFile(vocabPath)
	if err != nil {
		return nil, err
	}
	// LoadVocabFile registers a CharVocab; keep it under this BPE session.
	t := &BPETokenizer{Vocab: v, merges: map[string]int{}, maxRank: 0}
	if mergesPath != "" {
		if err := t.loadMerges(mergesPath); err != nil {
			_ = CloseVocab(v.ID)
			return nil, err
		}
	}
	bpeMu.Lock()
	id := atomic.AddInt32(&bpeNext, 1) - 1
	if id <= 0 {
		id = 1
		bpeNext = 2
	}
	t.ID = id
	bpeMap[id] = t
	bpeMu.Unlock()
	return t, nil
}

// OpenBPEText builds from in-memory vocab text + merges text (for tests/inline).
func OpenBPEText(vocabText, mergesText string) (*BPETokenizer, error) {
	v, err := LoadVocabText("<inline-bpe-vocab>", vocabText)
	if err != nil {
		return nil, err
	}
	t := &BPETokenizer{Vocab: v, merges: map[string]int{}, maxRank: 0}
	if mergesText != "" {
		if err := t.parseMerges(mergesText); err != nil {
			_ = CloseVocab(v.ID)
			return nil, err
		}
	}
	bpeMu.Lock()
	id := atomic.AddInt32(&bpeNext, 1) - 1
	if id <= 0 {
		id = 1
		bpeNext = 2
	}
	t.ID = id
	bpeMap[id] = t
	bpeMu.Unlock()
	return t, nil
}

func (t *BPETokenizer) loadMerges(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var b strings.Builder
	sc := bufio.NewScanner(f)
	// allow long lines
	buf := make([]byte, 0, 64*1024)
	sc.Buffer(buf, 1024*1024)
	for sc.Scan() {
		b.WriteString(sc.Text())
		b.WriteByte('\n')
	}
	if err := sc.Err(); err != nil {
		return err
	}
	return t.parseMerges(b.String())
}

func (t *BPETokenizer) parseMerges(text string) error {
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	rank := 0
	for _, ln := range lines {
		ln = strings.TrimSpace(ln)
		if ln == "" || strings.HasPrefix(ln, "#") {
			continue
		}
		// optional "#version: ..." header in HF merges
		if strings.HasPrefix(ln, "version") {
			continue
		}
		parts := strings.Fields(ln)
		if len(parts) < 2 {
			return fmt.Errorf("bad merge line %q", ln)
		}
		// only first two tokens form the pair; ignore trailing rank numbers if present
		key := parts[0] + " " + parts[1]
		if _, ok := t.merges[key]; !ok {
			t.merges[key] = rank
			rank++
		}
	}
	t.maxRank = rank
	return nil
}

// GetBPE returns a live BPE session.
func GetBPE(id int32) (*BPETokenizer, error) {
	bpeMu.Lock()
	defer bpeMu.Unlock()
	t, ok := bpeMap[id]
	if !ok || t == nil {
		return nil, fmt.Errorf("bpe session %d not found (closed?)", id)
	}
	return t, nil
}

// CloseBPE closes BPE + underlying vocab (idempotent).
func CloseBPE(id int32) error {
	bpeMu.Lock()
	t, ok := bpeMap[id]
	delete(bpeMap, id)
	bpeMu.Unlock()
	if ok && t != nil && t.Vocab != nil {
		_ = CloseVocab(t.Vocab.ID)
	}
	return nil
}

// MergeCount is number of loaded merge pairs.
func (t *BPETokenizer) MergeCount() int {
	if t == nil {
		return 0
	}
	return len(t.merges)
}

// VocabSize mirrors CharVocab.Size.
func (t *BPETokenizer) VocabSize() int {
	if t == nil || t.Vocab == nil {
		return 0
	}
	return t.Vocab.Size()
}

// Decode joins token ids via underlying vocab.
func (t *BPETokenizer) Decode(ids []int32) string {
	if t == nil || t.Vocab == nil {
		return ""
	}
	return t.Vocab.Decode(ids)
}

// Token returns the string for id.
func (t *BPETokenizer) Token(id int) string {
	if t == nil || t.Vocab == nil {
		return ""
	}
	return t.Vocab.Token(id)
}

// Encode applies BPE when merges exist; otherwise CharVocab longest-match.
func (t *BPETokenizer) Encode(prompt string) []int32 {
	if t == nil || t.Vocab == nil {
		return nil
	}
	if len(t.merges) == 0 {
		return t.Vocab.Encode(prompt)
	}
	if prompt == "" {
		return nil
	}
	// Split into UTF-8 runes as initial symbols; each rune must exist in vocab
	// or we fall back to byte-level / unk.
	symbols := splitInitial(prompt, t.Vocab)
	// Greedy BPE: repeatedly merge the lowest-rank adjacent pair until none remain.
	for {
		bestRank := t.maxRank + 1
		bestI := -1
		for i := 0; i+1 < len(symbols); i++ {
			key := symbols[i] + " " + symbols[i+1]
			if r, ok := t.merges[key]; ok && r < bestRank {
				bestRank = r
				bestI = i
			}
		}
		if bestI < 0 {
			break
		}
		merged := symbols[bestI] + symbols[bestI+1]
		// Replace pair with merged symbol
		symbols = append(symbols[:bestI], append([]string{merged}, symbols[bestI+2:]...)...)
	}
	// Map symbols → ids (unk if missing)
	unk := t.Vocab.IDOf("<unk>")
	if unk < 0 {
		unk = t.Vocab.IDOf("<pad>")
	}
	out := make([]int32, 0, len(symbols))
	for _, s := range symbols {
		id := t.Vocab.IDOf(s)
		if id < 0 {
			if unk >= 0 {
				out = append(out, int32(unk))
			}
			continue
		}
		out = append(out, int32(id))
	}
	return out
}

// splitInitial breaks prompt into vocab-known atoms: prefer multi-byte specials
// only as single runes here; specials longer than one rune that appear at starts
// of runs are not auto-matched (use char path for pure table). For BPE, start
// with UTF-8 runes, then single bytes for invalid sequences.
func splitInitial(prompt string, v *CharVocab) []string {
	out := make([]string, 0, len(prompt))
	for i := 0; i < len(prompt); {
		r, size := utf8.DecodeRuneInString(prompt[i:])
		if r == utf8.RuneError && size == 1 {
			// raw byte token
			s := prompt[i : i+1]
			out = append(out, s)
			i++
			continue
		}
		s := prompt[i : i+size]
		// if rune string not in vocab, fall back to bytes of that rune
		if v.IDOf(s) < 0 {
			for j := 0; j < size; j++ {
				out = append(out, prompt[i+j:i+j+1])
			}
		} else {
			out = append(out, s)
		}
		i += size
	}
	return out
}

// BPECount is for tests.
func BPECount() int {
	bpeMu.Lock()
	defer bpeMu.Unlock()
	return len(bpeMap)
}

// FormatMerge is a small helper for demos.
func FormatMerge(a, b string) string { return a + " " + b }

// ParseRank is for tolerant merge files that end with an integer rank.
func ParseRank(s string) (int, bool) {
	n, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return 0, false
	}
	return n, true
}
