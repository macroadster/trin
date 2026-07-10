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

// LinearTritRows loads a row window, absmean-quantizes to trits, and matvecs.
// bitplane selects packed set-bit accumulate (v1.05 path) vs scalar 4-unroll.
// Returned scale is the absmean scale applied to every output row in the window.
func (s *WeightSession) LinearTritRows(name string, x []float32, K, row0, nRows int, bitplane bool) (y []float32, scale float32, err error) {
	if s == nil {
		return nil, 0, fmt.Errorf("nil session")
	}
	if K <= 0 || nRows <= 0 {
		return nil, 0, fmt.Errorf("linear_trit_rows: bad K/nRows")
	}
	if len(x) < K {
		return nil, 0, fmt.Errorf("linear_trit_rows: x short (need %d have %d)", K, len(x))
	}
	if row0 < 0 {
		return nil, 0, fmt.Errorf("linear_trit_rows: negative row0")
	}
	nElem, _, err := s.tensorNElem(name)
	if err != nil {
		return nil, 0, err
	}
	if nElem%K != 0 {
		return nil, 0, fmt.Errorf("linear_trit_rows: nElem=%d not divisible by K=%d", nElem, K)
	}
	M := nElem / K
	if row0+nRows > M {
		return nil, 0, fmt.Errorf("linear_trit_rows OOB: row0=%d nRows=%d M=%d", row0, nRows, M)
	}
	wF, err := s.LoadF32Window(name, row0*K, nRows*K)
	if err != nil {
		return nil, 0, err
	}
	W, scale := QuantizeAbsMeanTrits(wF)
	if bitplane {
		y = LinearTritPlanes(W, x[:K], nRows, K, scale)
	} else {
		y = LinearTritScalar(W, x[:K], nRows, K, scale)
	}
	return y, scale, nil
}

// LinearTrit is the session-backed quantize + trit matvec path (v1.06).
// GGUF/safetensors row tiles → absmean trit → scalar or packed bitplane matvec.
// tileRows <= 0 defaults to 8 (or M if smaller).
func (s *WeightSession) LinearTrit(name string, x []float32, M, K, tileRows int, bitplane bool) (y []float32, scale float32, err error) {
	if M <= 0 || K <= 0 {
		return nil, 0, fmt.Errorf("linear_trit: bad M/K")
	}
	if len(x) < K {
		return nil, 0, fmt.Errorf("linear_trit: x short")
	}
	nElem, _, err := s.tensorNElem(name)
	if err != nil {
		return nil, 0, err
	}
	if nElem < M*K {
		return nil, 0, fmt.Errorf("linear_trit: tensor too small nElem=%d need %d", nElem, M*K)
	}
	if tileRows <= 0 {
		tileRows = 8
	}
	if tileRows > M {
		tileRows = M
	}
	out := make([]float32, M)
	var usedScale float32
	for r := 0; r < M; r += tileRows {
		n := tileRows
		if r+n > M {
			n = M - r
		}
		part, sc, err := s.LinearTritRows(name, x, K, r, n, bitplane)
		if err != nil {
			return nil, 0, err
		}
		// First tile sets the report scale; later tiles re-quantize independently
		// (same as one-shot window path when M is small enough for one tile).
		if r == 0 {
			usedScale = sc
		}
		copy(out[r:r+n], part)
	}
	return out, usedScale, nil
}

// LinearTritGGUF is a one-shot open + LinearTrit for a GGUF path.
func LinearTritGGUF(path, name string, x []float32, M, K, tileRows int, bitplane bool) ([]float32, float32, error) {
	f, err := OpenGGUF(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	s := &WeightSession{Kind: "gguf", Path: path, gguf: f, Mapped: f.Mapped}
	return s.LinearTrit(name, x, M, K, tileRows, bitplane)
}

// PrepackTrit quantizes tensor name as absmean trits once and caches trit + packed planes.
// Uses the full flat body (global scale). Re-packing the same name is a no-op (returns existing).
// Returns whether a new pack was built (false = cache hit / already present).
// Name is resolved through the session NameMap; packs are keyed by on-disk name.
func (s *WeightSession) PrepackTrit(name string, K int) (built bool, err error) {
	if s == nil {
		return false, fmt.Errorf("nil session")
	}
	if K <= 0 {
		return false, fmt.Errorf("prepack_trit: bad K")
	}
	name = s.resolveOnDisk(name)
	s.tritMu.Lock()
	if s.tritPack != nil {
		if p := s.tritPack[name]; p != nil && p.K == K {
			s.tritMu.Unlock()
			return false, nil
		}
	}
	s.tritMu.Unlock()

	nElem, _, err := s.tensorNElem(name)
	if err != nil {
		return false, err
	}
	if nElem%K != 0 {
		return false, fmt.Errorf("prepack_trit: nElem=%d not divisible by K=%d", nElem, K)
	}
	M := nElem / K
	wF, err := s.LoadF32Window(name, 0, M*K)
	if err != nil {
		return false, err
	}
	W, scale := QuantizeAbsMeanTrits(wF)
	pos, neg, nw := PackTritMatrix(W, M, K)
	pack := &TritPack{
		Name:  name,
		M:     M,
		K:     K,
		Scale: scale,
		W:     W,
		Pos:   pos,
		Neg:   neg,
		NW:    nw,
	}
	s.tritMu.Lock()
	if s.tritPack == nil {
		s.tritPack = map[string]*TritPack{}
	}
	// Another caller may have won the race — prefer first successful pack.
	if existing := s.tritPack[name]; existing != nil && existing.K == K {
		s.tritMu.Unlock()
		return false, nil
	}
	s.tritPack[name] = pack
	s.tritMu.Unlock()
	return true, nil
}

// HasTritPack reports whether name is already prepacked (for demos / counters).
func (s *WeightSession) HasTritPack(name string) bool {
	if s == nil {
		return false
	}
	name = s.resolveOnDisk(name)
	s.tritMu.Lock()
	defer s.tritMu.Unlock()
	if s.tritPack == nil {
		return false
	}
	return s.tritPack[name] != nil
}

// TritPackInfo returns (M, K, scale, 1/0 hit) when a pack exists.
func (s *WeightSession) TritPackInfo(name string) (M, K int, scale float32, ok bool) {
	if s == nil {
		return 0, 0, 0, false
	}
	name = s.resolveOnDisk(name)
	s.tritMu.Lock()
	defer s.tritMu.Unlock()
	if s.tritPack == nil {
		return 0, 0, 0, false
	}
	p := s.tritPack[name]
	if p == nil {
		return 0, 0, 0, false
	}
	return p.M, p.K, p.Scale, true
}

// DropTritPack removes one cached pack (no-op if missing).
func (s *WeightSession) DropTritPack(name string) {
	if s == nil {
		return
	}
	name = s.resolveOnDisk(name)
	s.tritMu.Lock()
	if s.tritPack != nil {
		delete(s.tritPack, name)
	}
	s.tritMu.Unlock()
}

// LinearTritCached uses a whole-tensor prepack when available (or builds one).
// Always uses the full-tensor absmean scale (parity with LinearTrit(..., tileRows=M)).
// bitplane selects scalar int8 path vs packed pos/neg planes (no re-pack on hit).
func (s *WeightSession) LinearTritCached(name string, x []float32, M, K int, bitplane bool) (y []float32, scale float32, err error) {
	if s == nil {
		return nil, 0, fmt.Errorf("nil session")
	}
	if M <= 0 || K <= 0 {
		return nil, 0, fmt.Errorf("linear_trit_cached: bad M/K")
	}
	if len(x) < K {
		return nil, 0, fmt.Errorf("linear_trit_cached: x short")
	}
	name = s.resolveOnDisk(name)
	if _, err := s.PrepackTrit(name, K); err != nil {
		return nil, 0, err
	}
	s.tritMu.Lock()
	p := s.tritPack[name]
	s.tritMu.Unlock()
	if p == nil {
		return nil, 0, fmt.Errorf("linear_trit_cached: pack missing after prepack")
	}
	if M > p.M {
		return nil, 0, fmt.Errorf("linear_trit_cached: M=%d > pack M=%d", M, p.M)
	}
	if K != p.K {
		return nil, 0, fmt.Errorf("linear_trit_cached: K mismatch")
	}
	scale = p.Scale
	if bitplane {
		nw := p.NW
		y = LinearTritPlanesPacked(p.Pos[:M*nw], p.Neg[:M*nw], x[:K], M, K, nw, scale)
	} else {
		y = LinearTritScalar(p.W[:M*K], x[:K], M, K, scale)
	}
	return y, scale, nil
}
