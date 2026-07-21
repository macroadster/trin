// Package steg provides ultra-low-bit CPU helpers for Starlight steganography
// workflows on top of Trin's trit / bitplane inference kernels.
//
// Design intent:
//   - Starlight wishes carry a visible pixel commitment (hex hash) and hide
//     payload skill/metadata in image channels.
//   - Classical LSB extract is already cheap; the ternary win is for
//     repeated filter/scan inference (packed trit matvec) and denser storage
//     of quantised detect kernels / residual maps — not for replacing one-shot
//     byte walks on tiny images.
//   - Everything here is pure Go, mmap-free, and callable from a single binary
//     serve wrapper without Python/torch.
package steg

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"math/bits"
	"strings"

	"github.com/macroadster/trin/internal/weights"
)

// RGB is a packed sRGB pixel (0–255 per channel).
type RGB struct {
	R, G, B uint8
}

// ScanResult is a cheap CPU scan report for tooling / Starlight assist.
type ScanResult struct {
	Width, Height int
	PayloadBits   int
	PayloadHex    string
	PayloadHash   string // sha256 hex of recovered payload bytes
	MarkerScore   float64
	FilterScore   float64
	HasPayload    bool
	Notes         string
}

// PixelCommitmentHash is Starlight-style visible pixel id:
// sha256(payload) as lowercase hex (64 chars). Matches the "visible_pixel_hash"
// shape used by wishes/contracts (content commitment), not an on-image digest.
func PixelCommitmentHash(payload []byte) string {
	sum := sha256.Sum256(payload)
	return hex.EncodeToString(sum[:])
}

// EncodeTritLSB embeds trits ∈ {-1,0,+1} into the low 2 bits of the blue
// channel (capacity = len(img) trits). Encoding:
//
//	00 → 0, 01 → +1, 10 → -1, 11 reserved (maps to 0 on decode).
//
// Cover RGB is mutated in place. Returns bits written.
func EncodeTritLSB(img []RGB, trits []int8) int {
	n := len(trits)
	if n > len(img) {
		n = len(img)
	}
	for i := 0; i < n; i++ {
		code := uint8(0)
		switch weights.ClampTritInt(int(trits[i])) {
		case 1:
			code = 1
		case -1:
			code = 2
		}
		img[i].B = (img[i].B &^ 0x3) | code
	}
	return n
}

// DecodeTritLSB recovers trits previously written by EncodeTritLSB.
func DecodeTritLSB(img []RGB, n int) []int8 {
	if n < 0 {
		n = 0
	}
	if n > len(img) {
		n = len(img)
	}
	out := make([]int8, n)
	for i := 0; i < n; i++ {
		switch img[i].B & 0x3 {
		case 1:
			out[i] = 1
		case 2:
			out[i] = -1
		default:
			out[i] = 0
		}
	}
	return out
}

// TritsFromPayload maps payload bytes → length*4 trits using only codes
// {00→0, 01→+1, 10→−1}. Raw ASCII bit patterns use 11 often, which the
// LSB codec treats as reserved/0 — so each payload bit is expanded as:
//
//	0 → trit 0,  1 → trit +1
//
// (two trits would suffice for bits, but four-per-byte keeps a uniform
// pixel:trit layout with EncodeTritLSB: one trit per pixel).
// Internally we store two bits of payload per two pixels via bit pairs
// that never emit code 11: bit0→(code 0/1), bit1→(code 0/1).
func TritsFromPayload(payload []byte) []int8 {
	out := make([]int8, len(payload)*8) // 1 trit per payload bit
	for i, b := range payload {
		for k := 0; k < 8; k++ {
			if (b>>uint(7-k))&1 == 1 {
				out[i*8+k] = 1
			} else {
				out[i*8+k] = 0
			}
		}
	}
	return out
}

// PayloadFromTrits inverse of TritsFromPayload (bit-aligned; trit!=0 → bit 1).
func PayloadFromTrits(trits []int8) []byte {
	n := len(trits) / 8
	out := make([]byte, n)
	for i := 0; i < n; i++ {
		var b byte
		for k := 0; k < 8; k++ {
			if weights.ClampTritInt(int(trits[i*8+k])) != 0 {
				b |= 1 << uint(7-k)
			}
		}
		out[i] = b
	}
	return out
}

// EmbedPayload encodes payload into img blue LSB trits. Requires
// len(img) >= 8*len(payload) (one trit/pixel per payload bit).
// Returns commitment hash and trit count.
func EmbedPayload(img []RGB, payload []byte) (hash string, nTrits int, err error) {
	need := 8 * len(payload)
	if len(img) < need {
		return "", 0, fmt.Errorf("steg: image capacity %d trits < need %d", len(img), need)
	}
	tr := TritsFromPayload(payload)
	n := EncodeTritLSB(img, tr)
	return PixelCommitmentHash(payload), n, nil
}

// ExtractPayload recovers nBytes from img blue LSB trits.
func ExtractPayload(img []RGB, nBytes int) ([]byte, string, error) {
	if nBytes < 0 {
		return nil, "", fmt.Errorf("steg: negative length")
	}
	need := 8 * nBytes
	if len(img) < need {
		return nil, "", fmt.Errorf("steg: image capacity %d < need %d", len(img), need)
	}
	tr := DecodeTritLSB(img, need)
	payload := PayloadFromTrits(tr)
	return payload, PixelCommitmentHash(payload), nil
}

// ChannelResidualTrits builds a residual trit map from one channel against its
// local mean estimate (here: previous pixel in scan order). Useful as a sparse
// feature input to bitplane kernels for steganalysis-style scoring.
func ChannelResidualTrits(img []RGB, which byte /*0=R,1=G,2=B*/) []int8 {
	out := make([]int8, len(img))
	var prev float64
	for i, p := range img {
		var c float64
		switch which {
		case 1:
			c = float64(p.G)
		case 2:
			c = float64(p.B)
		default:
			c = float64(p.R)
		}
		d := c - prev
		if d > 2 {
			out[i] = 1
		} else if d < -2 {
			out[i] = -1
		}
		prev = c
	}
	return out
}

// PackResidualPlanes packs a trit residual map into pos/neg uint32 words.
func PackResidualPlanes(trits []int8) (pos, neg []uint32, n int) {
	return packInt8(trits)
}

func packInt8(row []int8) (pos, neg []uint32, n int) {
	n = len(row)
	nw := (n + 31) / 32
	if nw == 0 {
		return []uint32{}, []uint32{}, n
	}
	pos = make([]uint32, nw)
	neg = make([]uint32, nw)
	for i, t := range row {
		w, b := i/32, uint(i%32)
		if t > 0 {
			pos[w] |= 1 << b
		} else if t < 0 {
			neg[w] |= 1 << b
		}
	}
	return pos, neg, n
}

// MarkerScorePopcount scores how much a residual plane differs from a
// "clean" reference (xor then popcount density). Higher → more structured
// low-bit perturbation (payload-like). Reference may be nil (scores density
// of non-zeros in plane).
func MarkerScorePopcount(pos, neg []uint32, refPos, refNeg []uint32) float64 {
	if len(pos) == 0 && len(neg) == 0 {
		return 0
	}
	var ones, bitsN int
	nw := len(pos)
	if len(neg) > nw {
		nw = len(neg)
	}
	for w := 0; w < nw; w++ {
		var p, g uint32
		if w < len(pos) {
			p = pos[w]
		}
		if w < len(neg) {
			g = neg[w]
		}
		if refPos != nil || refNeg != nil {
			var rp, rg uint32
			if w < len(refPos) {
				rp = refPos[w]
			}
			if w < len(refNeg) {
				rg = refNeg[w]
			}
			p ^= rp
			g ^= rg
		}
		ones += bits.OnesCount32(p) + bits.OnesCount32(g)
		bitsN += 64 // pos+neg word pair
	}
	if bitsN == 0 {
		return 0
	}
	return float64(ones) / float64(bitsN)
}

// SyntheticFilterWeights builds a deterministic M×K trit filter bank for
// patch scan inference demos (no external weight file).
func SyntheticFilterWeights(M, K int, seed uint32) (trits []int8, scale float32) {
	if M < 1 {
		M = 1
	}
	if K < 1 {
		K = 1
	}
	trits = make([]int8, M*K)
	f := make([]float32, M*K)
	s := seed
	if s == 0 {
		s = 0xC0FFEE
	}
	for i := 0; i < M*K; i++ {
		// xorshift32
		s ^= s << 13
		s ^= s >> 17
		s ^= s << 5
		v := float32(int32(s%7) - 3) // -3..3
		f[i] = v
	}
	return weights.QuantizeAbsMeanTrits(f)
}

// FilterScore runs a prepacked trit matvec of a residual feature vector x
// (length K, typically a flat patch or downsampled channel) against M filters.
// Returns mean abs activation — higher for images that match the filter bank.
func FilterScore(pos, neg []uint32, x []float32, M, K, nw int, scale float32) float64 {
	if M <= 0 || K <= 0 || len(x) < K {
		return 0
	}
	y := weights.LinearTritPlanesPacked(pos, neg, x, M, K, nw, scale)
	var acc float64
	for _, v := range y {
		if v < 0 {
			acc -= float64(v)
		} else {
			acc += float64(v)
		}
	}
	return acc / float64(M)
}

// FeatureFromBlue builds an f32 feature row of length K from blue-channel
// samples (wrapped). Used as the activation path into trit filters.
func FeatureFromBlue(img []RGB, K int) []float32 {
	if K < 1 {
		return nil
	}
	x := make([]float32, K)
	if len(img) == 0 {
		return x
	}
	for i := 0; i < K; i++ {
		x[i] = (float32(img[i%len(img)].B) - 127.5) / 127.5
	}
	return x
}

// MakeCover generates a smooth W×H cover image (deterministic).
func MakeCover(w, h int) []RGB {
	if w < 1 {
		w = 1
	}
	if h < 1 {
		h = 1
	}
	img := make([]RGB, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			img[i] = RGB{
				R: uint8(40 + (x*3+y*5)%180),
				G: uint8(30 + (x*7+y*2)%190),
				B: uint8(50 + (x*2+y*11)%170),
			}
		}
	}
	return img
}

// ScanImageAssist performs a pure-CPU scan: optional payload extract when
// nPayloadBytes > 0, residual marker density, and ternary filter score.
func ScanImageAssist(img []RGB, width, height, nPayloadBytes int) ScanResult {
	res := ScanResult{Width: width, Height: height}
	if width*height != len(img) && width > 0 && height > 0 {
		res.Notes = "width*height != len(img); using len(img) only"
	}

	// Residual density on blue (payload lives in blue LSBs).
	rt := ChannelResidualTrits(img, 2)
	pos, neg, _ := PackResidualPlanes(rt)
	res.MarkerScore = MarkerScorePopcount(pos, neg, nil, nil)

	// Ternary filter bank on blue features.
	M, K := 16, 128
	if len(img) < 16 {
		K = 32
		M = 4
	}
	Wt, scale := SyntheticFilterWeights(M, K, 0x51A1)
	wpos, wneg, nw := weights.PackTritMatrix(Wt, M, K)
	x := FeatureFromBlue(img, K)
	res.FilterScore = FilterScore(wpos, wneg, x, M, K, nw, scale)

	if nPayloadBytes > 0 {
		payload, hash, err := ExtractPayload(img, nPayloadBytes)
		if err != nil {
			res.Notes = strings.TrimSpace(res.Notes + " extract: " + err.Error())
		} else {
			res.PayloadBits = nPayloadBytes * 8
			res.PayloadHex = hex.EncodeToString(payload)
			res.PayloadHash = hash
			// crude presence: any non-zero trit-coded byte
			for _, b := range payload {
				if b != 0 {
					res.HasPayload = true
					break
				}
			}
		}
	} else {
		// Heuristic: structured residual + filter energy
		res.HasPayload = res.MarkerScore > 0.08 && res.FilterScore > 0.05
	}
	return res
}

// MemFootprintF32 estimates dense f32 weight bytes for M×K plus x.
func MemFootprintF32(M, K int) int {
	return M*K*4 + K*4
}

// MemFootprintTritPack estimates packed pos/neg planes + x.
func MemFootprintTritPack(M, K int) int {
	nw := (K + 31) / 32
	return M*nw*8 + K*4
}

// EfficiencyNote returns a short fit summary for docs / serve stdout.
func EfficiencyNote(M, K int, denseUs, packedUs float64) string {
	mf := MemFootprintF32(M, K)
	mt := MemFootprintTritPack(M, K)
	memR := float64(mf) / math.Max(1, float64(mt))
	var lat string
	if packedUs <= 0 {
		lat = "latency n/a"
	} else if denseUs/packedUs >= 1.05 {
		lat = fmt.Sprintf("packed faster (dense/packed=%.2f)", denseUs/packedUs)
	} else if denseUs/packedUs <= 0.95 {
		lat = fmt.Sprintf("dense faster on this host (dense/packed=%.2f) — use packed for memory", denseUs/packedUs)
	} else {
		lat = "latency ≈ parity"
	}
	return fmt.Sprintf("M=%d K=%d mem_ratio_f32/pack=%.1fx %s; win-domain=CPU kernel reuse + weight pressure, not one-shot LSB extract",
		M, K, memR, lat)
}
