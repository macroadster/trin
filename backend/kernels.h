/* trin kernels v0.20 — span ops + trit GEMM + bitplane SIMD + vectorizable loops */
#ifndef TRIN_KERNELS_H
#define TRIN_KERNELS_H

#include <stddef.h>
#include <stdint.h>

/* -------- branchless unit mux -------- */
static inline float trin_soft_mux_f32(float g, float pure, float each) {
    return g * pure + (1.0f - g) * each;
}

/* -------- span kernels (auto-vectorizable) -------- */

static inline void trin_span_add_f32(const float *a, const float *b, float *out, size_t n) {
    size_t i;
    for (i = 0; i < n; i++) out[i] = a[i] + b[i];
}

static inline void trin_span_mul_f32(const float *a, const float *b, float *out, size_t n) {
    size_t i;
    for (i = 0; i < n; i++) out[i] = a[i] * b[i];
}

static inline void trin_span_scale_f32(float s, const float *a, float *out, size_t n) {
    size_t i;
    for (i = 0; i < n; i++) out[i] = s * a[i];
}

static inline float trin_dot_f32(const float *a, const float *b, size_t n) {
    float acc = 0.0f;
    size_t i;
    for (i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
}

static inline void trin_span_soft_mux_f32(
    float g, const float *pure, const float *each, float *out, size_t n
) {
    const float og = 1.0f - g;
    size_t i;
    for (i = 0; i < n; i++) out[i] = g * pure[i] + og * each[i];
}

/* -------- scalar trit GEMM -------- */

static inline int8_t trin_clamp_trit(int v) {
    if (v < -1) return -1;
    if (v > 1) return 1;
    return (int8_t)v;
}

static inline void trin_trit_gemm(
    const int8_t *A, const int8_t *B, int32_t *C, int M, int K, int N
) {
    int i, j, t;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            int32_t acc = 0;
            for (t = 0; t < K; t++) {
                acc += (int32_t)A[i * K + t] * (int32_t)B[t * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

static inline int32_t trin_trit_dot_i8(const int8_t *a, const int8_t *b, int n) {
    int32_t acc = 0;
    int i;
    for (i = 0; i < n; i++) acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

/* -------- bitplane trit: pos/neg planes --------
 *
 * trit t:
 *   pos=1,neg=0 → +1
 *   pos=0,neg=1 → -1
 *   pos=0,neg=0 →  0
 *
 * product (boolean):
 *   p_pos = (Ap & Bp) | (An & Bn)
 *   p_neg = (Ap & Bn) | (An & Bp)
 *   value = popcount(p_pos) - popcount(p_neg)   over K lanes
 */

static inline void trin_trit_to_bits(int8_t t, uint8_t *pos, uint8_t *neg) {
    if (t > 0) { *pos = 1; *neg = 0; }
    else if (t < 0) { *pos = 0; *neg = 1; }
    else { *pos = 0; *neg = 0; }
}

static inline int8_t trin_bits_to_trit(uint8_t pos, uint8_t neg) {
    if (pos && !neg) return 1;
    if (neg && !pos) return -1;
    return 0;
}

/* Elementwise pack: pos[i]/neg[i] as 0/1 bytes */
static inline void trin_pack_bitplanes(
    const int8_t *trits, uint8_t *pos, uint8_t *neg, size_t n
) {
    size_t i;
    for (i = 0; i < n; i++) {
        trin_trit_to_bits(trits[i], &pos[i], &neg[i]);
    }
}

static inline void trin_unpack_bitplanes(
    const uint8_t *pos, const uint8_t *neg, int8_t *trits, size_t n
) {
    size_t i;
    for (i = 0; i < n; i++) {
        trits[i] = trin_bits_to_trit(pos[i], neg[i]);
    }
}

/* Pack up to 64 trits into one uint64 pair (low bit = index 0) */
static inline void trin_pack_u64(
    const int8_t *trits, size_t n, uint64_t *pos_w, uint64_t *neg_w
) {
    uint64_t p = 0, g = 0;
    size_t i;
    for (i = 0; i < n && i < 64; i++) {
        uint8_t bp, bn;
        trin_trit_to_bits(trits[i], &bp, &bn);
        if (bp) p |= (uint64_t)1 << i;
        if (bn) g |= (uint64_t)1 << i;
    }
    *pos_w = p;
    *neg_w = g;
}

/* Portable popcount */
static inline int trin_popcount64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll((unsigned long long)x);
#else
    int c = 0;
    while (x) { x &= x - 1; c++; }
    return c;
#endif
}

/* One u64 plane word product (mask applied by caller). */
static inline int32_t trin_trit_word_product_u64(
    uint64_t awp, uint64_t awn, uint64_t bwp, uint64_t bwn
) {
    uint64_t pos = (awp & bwp) | (awn & bwn);
    uint64_t neg = (awp & bwn) | (awn & bwp);
    return (int32_t)trin_popcount64(pos) - (int32_t)trin_popcount64(neg);
}

/* Word-plane trit dot: sum_i a_i * b_i for n trits packed in 64-bit words */
static inline int32_t trin_trit_dot_bitplane_u64(
    const uint64_t *ap, const uint64_t *an,
    const uint64_t *bp, const uint64_t *bn,
    int n
) {
    int nwords = (n + 63) / 64;
    int32_t acc = 0;
    int w;
    for (w = 0; w < nwords; w++) {
        uint64_t mask;
        if (w == nwords - 1 && (n % 64) != 0) {
            mask = ((uint64_t)1 << (n % 64)) - 1u;
        } else {
            mask = ~((uint64_t)0);
        }
        uint64_t awp = ap[w] & mask;
        uint64_t awn = an[w] & mask;
        uint64_t bwp = bp[w] & mask;
        uint64_t bwn = bn[w] & mask;
        acc += trin_trit_word_product_u64(awp, awn, bwp, bwn);
    }
    return acc;
}

/*
 * Multiword SIMD-friendly path: walk 2×u64 (=128 trit lanes) then residual word.
 * Semantic-identical to trin_trit_dot_bitplane_u64.
 */
static inline int32_t trin_trit_dot_bitplane_u64_simd(
    const uint64_t *ap, const uint64_t *an,
    const uint64_t *bp, const uint64_t *bn,
    int n
) {
    int nwords = (n + 63) / 64;
    int32_t acc = 0;
    int w = 0;
    for (; w + 2 <= nwords; w += 2) {
        uint64_t m0 = ~((uint64_t)0);
        uint64_t m1 = ~((uint64_t)0);
        if (w + 1 == nwords - 1 && (n % 64) != 0) {
            m1 = ((uint64_t)1 << (n % 64)) - 1u;
        }
        acc += trin_trit_word_product_u64(ap[w] & m0, an[w] & m0, bp[w] & m0, bn[w] & m0);
        acc += trin_trit_word_product_u64(ap[w + 1] & m1, an[w + 1] & m1, bp[w + 1] & m1, bn[w + 1] & m1);
    }
    for (; w < nwords; w++) {
        uint64_t mask = ~((uint64_t)0);
        if (w == nwords - 1 && (n % 64) != 0) {
            mask = ((uint64_t)1 << (n % 64)) - 1u;
        }
        acc += trin_trit_word_product_u64(ap[w] & mask, an[w] & mask, bp[w] & mask, bn[w] & mask);
    }
    return acc;
}

/* Preferred multiword kernel width (128 trit lanes). */
#ifndef TRIN_SIMD_LANE_WIDTH
#define TRIN_SIMD_LANE_WIDTH 128
#endif

/*
 * Bitplane GEMM: same math as trin_trit_gemm, using pos/neg planes per row/col.
 * A, B still int8 trits (caller may pre-pack); this builds u64 planes for each K-vector.
 */
static inline void trin_trit_gemm_bitplane(
    const int8_t *A, const int8_t *B, int32_t *C, int M, int K, int N
) {
    int i, j, t;
    int nwords = (K + 63) / 64;
    /* stack-friendly limit; for huge K use heap in real drivers */
    uint64_t row_p[16], row_n[16], col_p[16], col_n[16];
    if (nwords > 16) {
        /* fallback scalar if K > 1024 in this header demo */
        trin_trit_gemm(A, B, C, M, K, N);
        return;
    }
    for (i = 0; i < M; i++) {
        /* pack row i */
        for (t = 0; t < nwords; t++) {
            size_t base = (size_t)t * 64;
            size_t chunk = (base + 64 <= (size_t)K) ? 64 : (size_t)K - base;
            trin_pack_u64(A + i * K + (int)base, chunk, &row_p[t], &row_n[t]);
            if (chunk < 64) {
                /* clear high bits already handled by pack of chunk length */
            }
        }
        for (j = 0; j < N; j++) {
            int8_t colbuf[1024];
            int kk;
            for (kk = 0; kk < K && kk < 1024; kk++) {
                colbuf[kk] = B[kk * N + j];
            }
            for (t = 0; t < nwords; t++) {
                size_t base = (size_t)t * 64;
                size_t chunk = (base + 64 <= (size_t)K) ? 64 : (size_t)K - base;
                trin_pack_u64(colbuf + (int)base, chunk, &col_p[t], &col_n[t]);
            }
            C[i * N + j] = trin_trit_dot_bitplane_u64(row_p, row_n, col_p, col_n, K);
        }
    }
}

#endif /* TRIN_KERNELS_H */
