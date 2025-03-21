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
#include "neon_sve_bridge.h"

#if !FIX_CDEF_MSE
static inline uint64_t dist_8xn_16bit_sve(const uint16_t *src, const uint16_t *dst, const int32_t dstride,
                                          const int32_t coeff_shift, uint8_t height, uint8_t subsampling_factor) {
    uint64x2_t ss = vdupq_n_u64(0);
    uint64x2_t dd = vdupq_n_u64(0);
    uint64x2_t s2 = vdupq_n_u64(0);
    uint64x2_t sd = vdupq_n_u64(0);
    uint64x2_t d2 = vdupq_n_u64(0);

    do {
        const uint16x8_t s0 = vld1q_u16(src);
        const uint16x8_t s1 = vld1q_u16(src + subsampling_factor * 8);

        const uint16x8_t d0 = vld1q_u16(dst);
        const uint16x8_t d1 = vld1q_u16(dst + subsampling_factor * dstride);

        ss = svt_udotq_u16(ss, s0, vdupq_n_u16(1));
        ss = svt_udotq_u16(ss, s1, vdupq_n_u16(1));
        dd = svt_udotq_u16(dd, d0, vdupq_n_u16(1));
        dd = svt_udotq_u16(dd, d1, vdupq_n_u16(1));

        s2 = svt_udotq_u16(s2, s0, s0);
        s2 = svt_udotq_u16(s2, s1, s1);
        sd = svt_udotq_u16(sd, s0, d0);
        sd = svt_udotq_u16(sd, s1, d1);
        d2 = svt_udotq_u16(d2, d0, d0);
        d2 = svt_udotq_u16(d2, d1, d1);

        src += 8 * 2 * subsampling_factor;
        dst += 2 * subsampling_factor * dstride;
        height -= 2 * subsampling_factor;
    } while (height != 0);

    uint64_t sum_s = vaddvq_u64(ss);
    uint64_t sum_d = vaddvq_u64(dd);

    uint64_t sum_s2 = vaddvq_u64(s2);
    uint64_t sum_d2 = vaddvq_u64(d2);
    uint64_t sum_sd = vaddvq_u64(sd);

    /* Compute the variance -- the calculation cannot go negative. */
    uint64_t svar = sum_s2 - ((sum_s * sum_s + 32) >> 6);
    uint64_t dvar = sum_d2 - ((sum_d * sum_d + 32) >> 6);
    return (uint64_t)floor(.5 +
                           (sum_d2 + sum_s2 - 2 * sum_sd) * .5 * (svar + dvar + (400 << 2 * coeff_shift)) /
                               (sqrt((20000 << 4 * coeff_shift) + svar * (double)dvar)));
}
#endif

static inline void mse_8xn_16bit_sve(const uint16_t *src, const uint16_t *dst, const int32_t dstride, uint64x2_t *sse,
                                     uint8_t height, uint8_t subsampling_factor) {
    do {
        const uint16x8_t s0 = vld1q_u16(src);
        const uint16x8_t s1 = vld1q_u16(src + subsampling_factor * 8);
        const uint16x8_t d0 = vld1q_u16(dst);
        const uint16x8_t d1 = vld1q_u16(dst + subsampling_factor * dstride);

        const uint16x8_t abs0 = vabdq_u16(s0, d0);
        const uint16x8_t abs1 = vabdq_u16(s1, d1);

        *sse = svt_udotq_u16(*sse, abs0, abs0);
        *sse = svt_udotq_u16(*sse, abs1, abs1);

        src += 8 * 2 * subsampling_factor;
        dst += 2 * subsampling_factor * dstride;
        height -= 2 * subsampling_factor;
    } while (height != 0);
}

static inline void mse_4xn_16bit_sve(const uint16_t *src, const uint16_t *dst, const int32_t dstride, uint64x2_t *sse,
                                     uint8_t height, uint8_t subsampling_factor) {
    do {
        const uint16x8_t s0 = load_u16_4x2(src, 4 * subsampling_factor);
        const uint16x8_t s1 = load_u16_4x2(src + 2 * 4 * subsampling_factor, 4 * subsampling_factor);
        const uint16x8_t d0 = load_u16_4x2(dst, dstride * subsampling_factor);
        const uint16x8_t d1 = load_u16_4x2(dst + 2 * dstride * subsampling_factor, dstride * subsampling_factor);

        const uint16x8_t abs0 = vabdq_u16(s0, d0);
        const uint16x8_t abs1 = vabdq_u16(s1, d1);
        *sse                  = svt_udotq_u16(*sse, abs0, abs0);
        *sse                  = svt_udotq_u16(*sse, abs1, abs1);

        src += 4 * 4 * subsampling_factor;
        dst += 4 * subsampling_factor * dstride;
        height -= 4 * subsampling_factor;
    } while (height != 0);
}

uint64_t svt_aom_compute_cdef_dist_16bit_sve(const uint16_t *dst, int32_t dstride, const uint16_t *src,
                                             const CdefList *dlist, int32_t cdef_count, BlockSize bsize,
#if FIX_CDEF_MSE
                                             int32_t coeff_shift, uint8_t subsampling_factor) {
#else
                                             int32_t coeff_shift, int32_t pli, uint8_t subsampling_factor) {
#endif
    uint64_t sum;
    int32_t  bi, bx, by;

#if !FIX_CDEF_MSE
    if ((bsize == BLOCK_8X8) && (pli == 0)) {
        sum = 0;
        for (bi = 0; bi < cdef_count; bi++) {
            by = dlist[bi].by;
            bx = dlist[bi].bx;
            sum += dist_8xn_16bit_sve(
                src, dst + 8 * by * dstride + 8 * bx, dstride, coeff_shift, 8, subsampling_factor);
            src += 8 * 8;
        }
    } else {
#endif
        uint64x2_t mse64 = vdupq_n_u64(0);

        if (bsize == BLOCK_8X8) {
            for (bi = 0; bi < cdef_count; bi++) {
                by = dlist[bi].by;
                bx = dlist[bi].bx;
                mse_8xn_16bit_sve(src, dst + (8 * by + 0) * dstride + 8 * bx, dstride, &mse64, 8, subsampling_factor);
                src += 8 * 8;
            }
        } else if (bsize == BLOCK_4X8) {
            for (bi = 0; bi < cdef_count; bi++) {
                by = dlist[bi].by;
                bx = dlist[bi].bx;
                mse_4xn_16bit_sve(src, dst + (8 * by + 0) * dstride + 4 * bx, dstride, &mse64, 8, subsampling_factor);
                src += 4 * 8;
            }
        } else if (bsize == BLOCK_8X4) {
            for (bi = 0; bi < cdef_count; bi++) {
                by = dlist[bi].by;
                bx = dlist[bi].bx;
                mse_8xn_16bit_sve(src, dst + 4 * by * dstride + 8 * bx, dstride, &mse64, 4, subsampling_factor);
                src += 8 * 4;
            }
        } else {
            assert(bsize == BLOCK_4X4);
            for (bi = 0; bi < cdef_count; bi++) {
                by = dlist[bi].by;
                bx = dlist[bi].bx;
                mse_4xn_16bit_sve(src, dst + 4 * by * dstride + 4 * bx, dstride, &mse64, 4, subsampling_factor);
                src += 4 * 4;
            }
        }

        sum = vaddvq_u64(mse64);
#if !FIX_CDEF_MSE
    }
#endif

    return sum >> 2 * coeff_shift;
}
