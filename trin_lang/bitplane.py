"""
Bitplane-packed trits for Trin v0.3.

Each trit t ∈ {-1,0,+1} splits into two bits:
  pos = 1 iff t == +1
  neg = 1 iff t == -1
  (never both 1; both 0 means zero)

Trit product a*b ∈ {-1,0,+1}:
  product_pos = (Ap ∧ Bp) ∨ (An ∧ Bn)
  product_neg = (Ap ∧ Bn) ∨ (An ∧ Bp)
  value       = product_pos - product_neg

Word packing: 64 trits → two uint64 words (pos, neg) for SIMD-friendly loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


def clamp_trit(v: int) -> int:
    if v < -1:
        return -1
    if v > 1:
        return 1
    return int(v)


def trit_to_bits(t: int) -> Tuple[int, int]:
    t = clamp_trit(t)
    if t > 0:
        return 1, 0
    if t < 0:
        return 0, 1
    return 0, 0


def bits_to_trit(pos: int, neg: int) -> int:
    if pos and neg:
        # illegal; prefer zero for robustness
        return 0
    if pos:
        return 1
    if neg:
        return -1
    return 0


def pack_trits(trits: Sequence[int]) -> Tuple[List[int], List[int]]:
    """Elementwise bitplanes (lists of 0/1)."""
    pos: List[int] = []
    neg: List[int] = []
    for t in trits:
        p, n = trit_to_bits(int(t))
        pos.append(p)
        neg.append(n)
    return pos, neg


def unpack_trits(pos: Sequence[int], neg: Sequence[int]) -> List[int]:
    n = min(len(pos), len(neg))
    return [bits_to_trit(int(pos[i]), int(neg[i])) for i in range(n)]


def pack_words(trits: Sequence[int]) -> Tuple[List[int], List[int], int]:
    """
    Pack trits into uint64 bitplanes.
    Returns (pos_words, neg_words, n_trits).
    """
    n = len(trits)
    nwords = (n + 63) // 64
    pos_w = [0] * nwords
    neg_w = [0] * nwords
    for i, t in enumerate(trits):
        p, nbit = trit_to_bits(int(t))
        w = i // 64
        b = i % 64
        if p:
            pos_w[w] |= 1 << b
        if nbit:
            neg_w[w] |= 1 << b
    return pos_w, neg_w, n


def unpack_words(pos_w: Sequence[int], neg_w: Sequence[int], n: int) -> List[int]:
    out: List[int] = []
    for i in range(n):
        w = i // 64
        b = i % 64
        mask = 1 << b
        p = 1 if (pos_w[w] & mask) else 0
        nb = 1 if (neg_w[w] & mask) else 0
        out.append(bits_to_trit(p, nb))
    return out


def product_bits(ap: int, an: int, bp: int, bn: int) -> Tuple[int, int]:
    """Single-trit bitplane product → (pos, neg)."""
    pp = (ap & bp) | (an & bn)
    pn = (ap & bn) | (an & bp)
    return pp, pn


def product_value(ap: int, an: int, bp: int, bn: int) -> int:
    pp, pn = product_bits(ap, an, bp, bn)
    return pp - pn


def popcount(x: int) -> int:
    return int(x).bit_count()


@dataclass
class Bitplanes:
    """Word-packed bitplanes for a flat trit vector of length n."""

    pos_words: List[int]
    neg_words: List[int]
    n: int

    @classmethod
    def from_trits(cls, trits: Sequence[int]) -> "Bitplanes":
        pw, nw, n = pack_words(trits)
        return cls(pw, nw, n)

    def to_trits(self) -> List[int]:
        return unpack_words(self.pos_words, self.neg_words, self.n)


def trit_dot_bitplane(a: Sequence[int], b: Sequence[int]) -> int:
    """Σ a_i * b_i using bitplane product + popcount (word path when padded)."""
    n = min(len(a), len(b))
    # element path for correctness + short vectors
    acc = 0
    for i in range(n):
        ap, an = trit_to_bits(int(a[i]))
        bp, bn = trit_to_bits(int(b[i]))
        acc += product_value(ap, an, bp, bn)
    return acc


def trit_dot_words(ap: Sequence[int], an: Sequence[int], bp: Sequence[int], bn: Sequence[int], n: int) -> int:
    """
    Dot product from word-packed planes.
    product_pos bits count as +1, product_neg as -1 (per bit position up to n).
    """
    if n <= 0:
        return 0
    nwords = (n + 63) // 64
    acc = 0
    for w in range(nwords):
        if w == nwords - 1 and (n % 64) != 0:
            mask = (1 << (n % 64)) - 1
        else:
            mask = (1 << 64) - 1
        awp = (ap[w] if w < len(ap) else 0) & mask
        awn = (an[w] if w < len(an) else 0) & mask
        bwp = (bp[w] if w < len(bp) else 0) & mask
        bwn = (bn[w] if w < len(bn) else 0) & mask
        pos = (awp & bwp) | (awn & bwn)
        neg = (awp & bwn) | (awn & bwp)
        acc += popcount(pos) - popcount(neg)
    return acc


def trit_gemm_scalar(A: Sequence[int], B: Sequence[int], M: int, K: int, N: int) -> List[int]:
    """Reference row-major GEMM on trit vectors."""
    if len(A) < M * K or len(B) < K * N:
        raise ValueError("shape mismatch")
    out: List[int] = []
    for i in range(M):
        for j in range(N):
            acc = 0
            for t in range(K):
                acc += clamp_trit(int(A[i * K + t])) * clamp_trit(int(B[t * N + j]))
            out.append(acc)
    return out


def trit_gemm_bitplane(A: Sequence[int], B: Sequence[int], M: int, K: int, N: int) -> List[int]:
    """
    GEMM via bitplane encoding.
    For each output (i,j): pack row A[i,:] and col B[:,j] into words and popcount-dot.
    """
    if len(A) < M * K or len(B) < K * N:
        raise ValueError("shape mismatch")
    out: List[int] = []
    for i in range(M):
        row = [clamp_trit(int(A[i * K + t])) for t in range(K)]
        row_bp = Bitplanes.from_trits(row)
        for j in range(N):
            col = [clamp_trit(int(B[t * N + j])) for t in range(K)]
            col_bp = Bitplanes.from_trits(col)
            acc = trit_dot_words(
                row_bp.pos_words,
                row_bp.neg_words,
                col_bp.pos_words,
                col_bp.neg_words,
                K,
            )
            out.append(acc)
    return out


def assert_gemm_equiv(
    A: Sequence[int], B: Sequence[int], M: int, K: int, N: int
) -> List[int]:
    s = trit_gemm_scalar(A, B, M, K, N)
    b = trit_gemm_bitplane(A, B, M, K, N)
    if s != b:
        raise AssertionError(f"bitplane != scalar: {b} vs {s}")
    return s
