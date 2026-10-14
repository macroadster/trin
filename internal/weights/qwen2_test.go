package weights

import (
	"context"
	"os"
	"reflect"
	"testing"
	"time"
)

func opusQwenPath() string {
	if p := os.Getenv("OPUS_QWEN_GGUF"); p != "" {
		return p
	}
	return "/Users/eric/sandbox/starlight/starlight/datasets/opus_submission_2026/model/qwen2.5-0.5b-instruct-q8_0.gguf"
}

func loadOpusQwen(t *testing.T) *Qwen2 {
	t.Helper()
	path := opusQwenPath()
	if _, err := os.Stat(path); err != nil {
		t.Skip(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = CloseSession(s.ID) })
	m, err := LoadQwen2(s)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestQwen2TokenizerMatchesLlama(t *testing.T) {
	m := loadOpusQwen(t)
	cases := []struct {
		text string
		ids  []int
	}{
		{"Hi", []int{13048}},
		{"Hello", []int{9707}},
		{"No hidden text.", []int{2753, 8177, 1467, 13}},
		{"<|im_start|>user\nHi<|im_end|>", []int{151644, 872, 198, 13048, 151645}},
	}
	for _, tc := range cases {
		got := m.tok.Encode(tc.text)
		if !reflect.DeepEqual(got, tc.ids) {
			t.Errorf("encode %q\n got %v\nwant %v", tc.text, got, tc.ids)
		}
	}
}

func TestQwen2HiMatchesLlamaCLI(t *testing.T) {
	m := loadOpusQwen(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	prompt := "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
	t.Logf("prompt ids %v", m.tok.Encode(prompt))
	start := time.Now()
	got, err := m.Generate(ctx, prompt, 1)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("generate %q in %s", got, time.Since(start))
	if got != "Hello" {
		t.Fatalf("next text %q want Hello", got)
	}
}
