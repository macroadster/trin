package weights

import "fmt"

// tensorNElem returns element count and shape for a session tensor.
func (s *WeightSession) tensorNElem(name string) (nElem int, shape []int64, err error) {
	sh, err := s.Shape(name)
	if err != nil {
		return 0, nil, err
	}
	n := 1
	for _, d := range sh {
		if d <= 0 {
			return 0, nil, fmt.Errorf("tensor %q zero/neg dim", name)
		}
		n *= int(d)
	}
	return n, sh, nil
}

// LinearRows computes y = W[row0:row0+nRows, :] @ x where W is MxK row-major
// stored under name. Only the row window is decoded (block-local for Q*).
func (s *WeightSession) LinearRows(name string, x []float32, K, row0, nRows int) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("nil session")
	}
	if K <= 0 || nRows <= 0 {
		return nil, fmt.Errorf("linear_rows: bad K/nRows")
	}
	if len(x) < K {
		return nil, fmt.Errorf("linear_rows: x short (need %d have %d)", K, len(x))
	}
	if row0 < 0 {
		return nil, fmt.Errorf("linear_rows: negative row0")
	}
	nElem, _, err := s.tensorNElem(name)
	if err != nil {
		return nil, err
	}
	if nElem%K != 0 {
		return nil, fmt.Errorf("linear_rows: nElem=%d not divisible by K=%d", nElem, K)
	}
	M := nElem / K
	if row0+nRows > M {
		return nil, fmt.Errorf("linear_rows OOB: row0=%d nRows=%d M=%d", row0, nRows, M)
	}
	off := row0 * K
	count := nRows * K
	W, err := s.LoadF32Window(name, off, count)
	if err != nil {
		return nil, err
	}
	out := make([]float32, nRows)
	for i := 0; i < nRows; i++ {
		acc := float32(0)
		base := i * K
		for t := 0; t < K; t++ {
			acc += W[base+t] * x[t]
		}
		out[i] = acc
	}
	return out, nil
}

// Linear computes y = W @ x for W MxK row-major, decoding row tiles of size tileRows.
// tileRows <= 0 defaults to 8 (or M if smaller).
func (s *WeightSession) Linear(name string, x []float32, M, K, tileRows int) ([]float32, error) {
	if M <= 0 || K <= 0 {
		return nil, fmt.Errorf("linear: bad M/K")
	}
	if len(x) < K {
		return nil, fmt.Errorf("linear: x short")
	}
	nElem, _, err := s.tensorNElem(name)
	if err != nil {
		return nil, err
	}
	if nElem < M*K {
		return nil, fmt.Errorf("linear: tensor too small nElem=%d need %d", nElem, M*K)
	}
	if tileRows <= 0 {
		tileRows = 8
	}
	if tileRows > M {
		tileRows = M
	}
	out := make([]float32, M)
	for r := 0; r < M; r += tileRows {
		n := tileRows
		if r+n > M {
			n = M - r
		}
		y, err := s.LinearRows(name, x, K, r, n)
		if err != nil {
			return nil, err
		}
		copy(out[r:r+n], y)
	}
	return out, nil
}

// LinearGGUF is a one-shot open + Linear for a GGUF path.
func LinearGGUF(path, name string, x []float32, M, K, tileRows int) ([]float32, error) {
	f, err := OpenGGUF(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	// temporary session-like path: use file methods via synthetic session
	s := &WeightSession{Kind: "gguf", Path: path, gguf: f, Mapped: f.Mapped}
	return s.Linear(name, x, M, K, tileRows)
}
