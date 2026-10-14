package weights

import (
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"
)

// qwenTokenizer is the GPT-2 byte-level BPE stored in a Qwen2 GGUF.
type qwenTokenizer struct {
	tokens   []string
	ids      map[string]int
	types    []int
	ranks    map[[2]string]int
	byteTo   []string
	runeTo   map[rune]byte
	specials []qwenSpecial
	re       *regexp.Regexp
	bos      int
	eos      int
	addBOS   bool
}

type qwenSpecial struct {
	text string
	id   int
}

func loadQwenTokenizer(s *WeightSession) (*qwenTokenizer, error) {
	toks, err := metaStrings(s, "tokenizer.ggml.tokens")
	if err != nil {
		return nil, err
	}
	if len(toks) == 0 {
		return nil, fmt.Errorf("qwen2: empty tokenizer")
	}
	merges, err := metaStrings(s, "tokenizer.ggml.merges")
	if err != nil {
		return nil, err
	}
	types := metaInts(s, "tokenizer.ggml.token_type")
	t := &qwenTokenizer{
		tokens: toks,
		ids:    make(map[string]int, len(toks)),
		types:  types,
		ranks:  make(map[[2]string]int, len(merges)),
		byteTo: nil,
		bos:    metaInt(s, "tokenizer.ggml.bos_token_id", -1),
		eos:    metaInt(s, "tokenizer.ggml.eos_token_id", -1),
		addBOS: metaBool(s, "tokenizer.ggml.add_bos_token"),
	}
	for i, tok := range toks {
		if _, ok := t.ids[tok]; !ok {
			t.ids[tok] = i
		}
	}
	for i, m := range merges {
		left, right, ok := splitMerge(m)
		if !ok {
			continue
		}
		key := [2]string{left, right}
		if _, exists := t.ranks[key]; !exists {
			t.ranks[key] = i
		}
	}
	for i, tok := range toks {
		typ := 1
		if i < len(types) {
			typ = types[i]
		}
		if typ == 3 || typ == 4 {
			t.specials = append(t.specials, qwenSpecial{text: tok, id: i})
		}
	}
	// Longest special first so "<|im_start|>" wins over a shorter prefix.
	for i := 0; i < len(t.specials); i++ {
		for j := i + 1; j < len(t.specials); j++ {
			if len(t.specials[j].text) > len(t.specials[i].text) {
				t.specials[i], t.specials[j] = t.specials[j], t.specials[i]
			}
		}
	}
	t.byteTo, t.runeTo = gpt2ByteEncoder()
	t.re = regexp.MustCompile(`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+`)
	return t, nil
}

func (t *qwenTokenizer) Encode(text string) []int {
	if t == nil {
		return nil
	}
	var ids []int
	if t.addBOS && t.bos >= 0 {
		ids = append(ids, t.bos)
	}
	for i := 0; i < len(text); {
		if id, n := t.matchSpecial(text[i:]); n > 0 {
			ids = append(ids, id)
			i += n
			continue
		}
		j := i + 1
		for j < len(text) {
			if _, n := t.matchSpecial(text[j:]); n > 0 {
				break
			}
			j++
		}
		ids = append(ids, t.encodePiece(text[i:j])...)
		i = j
	}
	return ids
}

func (t *qwenTokenizer) Decode(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(t.tokens) {
			continue
		}
		if t.isControl(id) {
			continue
		}
		b.WriteString(t.decodeToken(t.tokens[id]))
	}
	return b.String()
}

func (t *qwenTokenizer) decodeToken(s string) string {
	if len(t.runeTo) == 0 {
		return s
	}
	buf := make([]byte, 0, len(s))
	for _, r := range s {
		b, ok := t.runeTo[r]
		if !ok {
			buf = append(buf, string(r)...)
			continue
		}
		buf = append(buf, b)
	}
	return string(buf)
}

func (t *qwenTokenizer) isControl(id int) bool {
	if id < 0 || id >= len(t.types) {
		return false
	}
	return t.types[id] == 3
}

func (t *qwenTokenizer) matchSpecial(s string) (int, int) {
	for _, sp := range t.specials {
		if sp.text != "" && strings.HasPrefix(s, sp.text) {
			return sp.id, len(sp.text)
		}
	}
	return 0, 0
}

func (t *qwenTokenizer) encodePiece(s string) []int {
	var ids []int
	rest := s
	for len(rest) > 0 {
		loc := t.re.FindStringIndex(rest)
		if loc == nil || loc[0] != 0 {
			ids = append(ids, t.bpeBytes(rest[:1])...)
			rest = rest[1:]
			continue
		}
		ids = append(ids, t.bpeBytes(rest[:loc[1]])...)
		rest = rest[loc[1]:]
	}
	return ids
}

func (t *qwenTokenizer) bpeBytes(s string) []int {
	if s == "" {
		return nil
	}
	if id, ok := t.ids[s]; ok && utf8.RuneCountInString(s) == 1 {
		return []int{id}
	}
	syms := make([]string, 0, len(s))
	for _, b := range []byte(s) {
		syms = append(syms, t.byteTo[b])
	}
	for len(syms) >= 2 {
		bestRank := int(^uint(0) >> 1)
		best := -1
		for i := 0; i < len(syms)-1; i++ {
			r, ok := t.ranks[[2]string{syms[i], syms[i+1]}]
			if ok && r < bestRank {
				bestRank = r
				best = i
			}
		}
		if best < 0 {
			break
		}
		merged := syms[best] + syms[best+1]
		next := make([]string, 0, len(syms)-1)
		next = append(next, syms[:best]...)
		next = append(next, merged)
		next = append(next, syms[best+2:]...)
		syms = next
	}
	ids := make([]int, 0, len(syms))
	for _, sym := range syms {
		if id, ok := t.ids[sym]; ok {
			ids = append(ids, id)
			continue
		}
		// Unknown piece: fall back to raw bytes of this symbol.
		for _, b := range []byte(sym) {
			if id, ok := t.ids[t.byteTo[b]]; ok {
				ids = append(ids, id)
			}
		}
	}
	return ids
}

func splitMerge(s string) (string, string, bool) {
	// llama.cpp writes "left right". A left token that is itself a space is
	// stored with that leading space, so split on the first space only when
	// the left side is non-empty. A line that starts with a space means the
	// left token is a single space and the separator is the second space.
	if strings.HasPrefix(s, " ") {
		rest := s[1:]
		i := strings.IndexByte(rest, ' ')
		if i < 0 {
			return "", "", false
		}
		return " ", rest[i+1:], true
	}
	i := strings.IndexByte(s, ' ')
	if i <= 0 || i >= len(s)-1 {
		return "", "", false
	}
	return s[:i], s[i+1:], true
}

func gpt2ByteEncoder() ([]string, map[rune]byte) {
	enc := make([]string, 256)
	dec := make(map[rune]byte, 256)
	present := make([]bool, 256)
	var printable []int
	for b := int('!'); b <= int('~'); b++ {
		printable = append(printable, b)
	}
	for b := 0xA1; b <= 0xAC; b++ {
		printable = append(printable, b)
	}
	for b := 0xAE; b <= 0xFF; b++ {
		printable = append(printable, b)
	}
	for _, b := range printable {
		present[b] = true
		enc[b] = string(rune(b))
	}
	n := 0
	for b := 0; b < 256; b++ {
		if present[b] {
			continue
		}
		enc[b] = string(rune(256 + n))
		n++
	}
	for b := 0; b < 256; b++ {
		r, size := utf8.DecodeRuneInString(enc[b])
		if r == utf8.RuneError || size != len(enc[b]) {
			continue
		}
		dec[r] = byte(b)
	}
	return enc, dec
}

func metaStrings(s *WeightSession, key string) ([]string, error) {
	v := s.MetaValue(key)
	arr, ok := v.([]any)
	if !ok {
		return nil, fmt.Errorf("qwen2: metadata %s is not an array", key)
	}
	out := make([]string, len(arr))
	for i, item := range arr {
		str, ok := item.(string)
		if !ok {
			return nil, fmt.Errorf("qwen2: metadata %s[%d] is %T", key, i, item)
		}
		out[i] = str
	}
	return out, nil
}

func metaInts(s *WeightSession, key string) []int {
	v := s.MetaValue(key)
	arr, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]int, len(arr))
	for i, item := range arr {
		out[i] = anyInt(item)
	}
	return out
}

func metaInt(s *WeightSession, key string, fallback int) int {
	v := s.MetaValue(key)
	if v == nil {
		return fallback
	}
	if n := anyInt(v); n != 0 || fmt.Sprint(v) == "0" {
		return n
	}
	return fallback
}

func metaBool(s *WeightSession, key string) bool {
	v, ok := s.MetaValue(key).(bool)
	return ok && v
}

func anyInt(v any) int {
	switch n := v.(type) {
	case uint64:
		return int(n)
	case uint32:
		return int(n)
	case int64:
		return int(n)
	case int32:
		return int(n)
	case int:
		return n
	default:
		return 0
	}
}
