/*
 * Copyright (c) 2025, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>
#include <math.h>

#include "aom_dsp_rtcd.h"
#include "cdef.h"
#include "definitions.h"
#include "mem_neon.h"

#if !FIX_CDEF_MSE
static inline uint64_t dist_8xn_8bit_neon_dotprod(const uint8_t *src, const uint8_t *dst, const int32_t dstride,
                                                  const int32_t coeff_shift, uint8_t height,
                                                  uint8_t subsampling_factor) {
    uint32x4_t ss = vdupq_n_u32(0);
    uint32x4_t dd = vdupq_n_u32(0);
    uint32x4_t s2 = vdupq_n_u32(0);
    uint32x4_t sd = vdupq_n_u32(0);
    uint32x4_t d2 = vdupq_n_u32(0);

    do {
        const uint8x16_t s = vcombine_u8(vld1_u8(src), vld1_u8(src + subsampling_factor * 8));
        const uint8x16_t d = vcombine_u8(vld1_u8(dst), vld1_u8(dst + subsampling_factor * dstride));

        ss = vdotq_u32(ss, s, vdupq_n_u8(1));
        dd = vdotq_u32(dd, d, vdupq_n_u8(1));

        s2 = vdotq_u32(s2, s, s);
        d2 = vdotq_u32(d2, d, d);
        sd = vdotq_u32(sd, s, d);

        src += 8 * 2 * subsampling_factor; // width * 2 lines per iter. * subsampling
        dst += dstride * 2 * subsampling_factor;
        height -= 2 * subsampling_factor;
    } while (height != 0);

    uint32_t sum_s = vaddvq_u32(ss);
    uint32_t sum_d = vaddvq_u32(dd);

    uint64_t sum_s2 = vaddlvq_u32(s2);
    uint64_t sum_d2 = vaddlvq_u32(d2);
    uint64_t sum_sd = vaddlvq_u32(sd);

    /* Compute the variance -- the calculation cannot go negative. */
    uint64_t svar = sum_s2 - ((sum_s * sum_s + 32) >> 6);
    uint64_t dvar = sum_d2 - ((sum_d * sum_d + 32) >> 6);
    return (uint64_t)floor(.5 +
                           (sum_d2 + sum_s2 - 2 * sum_sd) * .5 * (svar + dvar + (400 << 2 * coeff_shift)) /
                               (sqrt((20000 << 4 * coeff_shift) + svar * (double)dvar)));
}
#endif

static inline void mse_4xn_8bit_neon_dotprod(const uint8_t *src, const uint8_t *dst, const int32_t dstride,
                                             uint32x4_t *sse, uint8_t height, uint8_t subsampling_factor) {
    do {
        const uint8x16_t s = load_u8_4x4(src, 4 * subsampling_factor);
        const uint8x16_t d = load_u8_4x4(dst, dstride * subsampling_factor);

        const uint8x16_t abs = vabdq_u8(d, s);
        *sse                 = vdotq_u32(*sse, abs, abs);

        src += 4 * 4 * subsampling_factor; // with * 2 rows per iter * subsampling
        dst += 4 * dstride * subsampling_factor;
        height -= 4 * subsampling_factor;
    } while (height != 0);
}

static inline void mse_8xn_8bit_neon_dotprod(const uint8_t *src, const uint8_t *dst, const int32_t dstride,
                                             uint32x4_t *sse, uint8_t height, uint8_t subsampling_factor) {
    do {
        const uint8x16_t s = vcombine_u8(vld1_u8(src), vld1_u8(src + subsampling_factor * 8));
        const uint8x16_t d = vcombine_u8(vld1_u8(dst), vld1_u8(dst + subsampling_factor * dstride));

        const uint8x16_t abs = vabdq_u8(s, d);
        *sse                 = vdotq_u32(*sse, abs, abs);

        src += 8 * 2 * subsampling_factor;
        dst += 2 * subsampling_factor * dstride;
        height -= 2 * subsampling_factor;
    } while (height != 0);
}

uint64_t svt_aom_compute_cdef_dist_8bit_neon_dotprod(const uint8_t *dst8, int32_t dstride, const uint8_t *src8,
                                                     const CdefList *dlist, int32_t cdef_count, BlockSize bsize,
#if FIX_CDEF_MSE
                                                     int32_t coeff_shift, uint8_t subsampling_factor) {
#else
                                                     int32_t coeff_shift, int32_t pli, uint8_t subsampling_factor) {
#endif
    uint64_t sum;
    int32_t  bi, bx, by;

#if !FIX_CDEF_MSE
    if (bsize == BLOCK_8X8 && pli == 0) {
        sum = 0;
        for (bi = 0; bi < cdef_count; bi++) {
            by = dlist[bi].by;
            bx = dlist[bi].bx;
            sum += dist_8xn_8bit_neon_dotprod(
                src8, dst8 + 8 * by * dstride + 8 * bx, dstride, coeff_shift, 8, subsampling_factor);
            src8 += 8 * 8;
        }
    } else {
#endif
        uint32x4_t mse = vdupq_n_u32(0);
        if (bsize == BLOCK_8X8) {
            for (bi = 0; bi < cdef_count; bi++) {
                by = dlist[bi].by;
                bx = dlist[bi].bx;
                mse_8xn_8bit_neon_dotprod(src8, dst8 + 8 * by * dstride + 8 * bx, dstride, &mse, 8, subsampling_factor);
                src8 += 8 * 8;
            }
        } else if (bsize == BLOCK_4X8) {
            for (bi = 0; bi < cdef_count; bi++) {
                by = dlist[bi].by;
                bx = dlist[bi].bx;
                mse_4xn_8bit_neon_dotprod(src8, dst8 + 8 * by * dstride + 4 * bx, dstride, &mse, 8, subsampling_factor);
                src8 += 4 * 8;
            }
        } else if (bsize == BLOCK_8X4) {
            for (bi = 0; bi < cdef_count; bi++) {
                by = dlist[bi].by;
                bx = dlist[bi].bx;
                mse_8xn_8bit_neon_dotprod(src8, dst8 + 4 * by * dstride + 8 * bx, dstride, &mse, 4, subsampling_factor);
                src8 += 8 * 4;
            }
        } else {
            assert(bsize == BLOCK_4X4);
            for (bi = 0; bi < cdef_count; bi++) {
                by = dlist[bi].by;
                bx = dlist[bi].bx;
                mse_4xn_8bit_neon_dotprod(src8, dst8 + 4 * by * dstride + 4 * bx, dstride, &mse, 4, subsampling_factor);
                src8 += 4 * 4;
            }
        }
        sum = vaddlvq_u32(mse);
#if !FIX_CDEF_MSE
    }
#endif
    return sum >> 2 * coeff_shift;
}
