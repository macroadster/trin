/* Standalone demo of Trin span + trit + bitplane kernels. */
#include <stdio.h>
#include "kernels.h"

int main(void) {
    float a[4] = {1.f, 2.f, 3.f, 4.f};
    float b[4] = {1.f, 1.f, 1.f, 1.f};
    float out[4];
    trin_span_add_f32(a, b, out, 4);
    printf("span_add: %.0f %.0f %.0f %.0f\n", out[0], out[1], out[2], out[3]);
    printf("dot: %.0f\n", trin_dot_f32(a, b, 4));

    int8_t A[4] = {1, 0, 0, 1};
    int8_t B[4] = {1, -1, 0, 1};
    int32_t Cs[4], Cb[4];
    trin_trit_gemm(A, B, Cs, 2, 2, 2);
    trin_trit_gemm_bitplane(A, B, Cb, 2, 2, 2);
    printf("trit_gemm C: %d %d %d %d\n", Cs[0], Cs[1], Cs[2], Cs[3]);
    printf("bitplane  C: %d %d %d %d\n", Cb[0], Cb[1], Cb[2], Cb[3]);
    /* expected: [1,-1, 0,1] both paths */
    return 0;
}
