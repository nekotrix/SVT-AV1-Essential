/*
 * Copyright (c) 2024, Alliance for Open Media. All rights reserved
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
static inline uint64_t dist_8xn_8bit_neon(const uint8_t *src, const uint8_t *dst, const int32_t dstride,
                                          const int32_t coeff_shift, uint8_t height, uint8_t subsampling_factor) {
    uint16x8_t ss = vdupq_n_u16(0);
    uint16x8_t dd = vdupq_n_u16(0);
    uint32x4_t s2 = vdupq_n_u32(0);
    uint32x4_t sd = vdupq_n_u32(0);
    uint32x4_t d2 = vdupq_n_u32(0);

    do {
        const uint16x8_t s0 = vmovl_u8(vld1_u8(src));
        const uint16x8_t s1 = vmovl_u8(vld1_u8(src + subsampling_factor * 8));
        const uint16x8_t d0 = vmovl_u8(vld1_u8(dst));
        const uint16x8_t d1 = vmovl_u8(vld1_u8(dst + subsampling_factor * dstride));

        ss = vaddq_u16(ss, s0);
        ss = vaddq_u16(ss, s1);
        dd = vaddq_u16(dd, d0);
        dd = vaddq_u16(dd, d1);

        s2 = vmlal_u16(s2, vget_low_u16(s0), vget_low_u16(s0));
        s2 = vmlal_u16(s2, vget_high_u16(s0), vget_high_u16(s0));
        s2 = vmlal_u16(s2, vget_low_u16(s1), vget_low_u16(s1));
        s2 = vmlal_u16(s2, vget_high_u16(s1), vget_high_u16(s1));

        sd = vmlal_u16(sd, vget_low_u16(s0), vget_low_u16(d0));
        sd = vmlal_u16(sd, vget_high_u16(s0), vget_high_u16(d0));
        sd = vmlal_u16(sd, vget_low_u16(s1), vget_low_u16(d1));
        sd = vmlal_u16(sd, vget_high_u16(s1), vget_high_u16(d1));

        d2 = vmlal_u16(d2, vget_low_u16(d0), vget_low_u16(d0));
        d2 = vmlal_u16(d2, vget_high_u16(d0), vget_high_u16(d0));
        d2 = vmlal_u16(d2, vget_low_u16(d1), vget_low_u16(d1));
        d2 = vmlal_u16(d2, vget_high_u16(d1), vget_high_u16(d1));

        src += 8 * 2 * subsampling_factor; // width * 2 lines per iter. * subsampling
        dst += dstride * 2 * subsampling_factor;
        height -= 2 * subsampling_factor;
    } while (height != 0);

    uint32_t sum_s = vaddlvq_u16(ss);
    uint32_t sum_d = vaddlvq_u16(dd);

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

static inline void mse_4xn_8bit_neon(const uint8_t *src, const uint8_t *dst, const int32_t dstride, uint32x4_t *sse,
                                     uint8_t height, uint8_t subsampling_factor) {
    do {
        const uint8x8_t s = load_u8_4x2(src, 4 * subsampling_factor);
        const uint8x8_t d = load_u8_4x2(dst, dstride * subsampling_factor);

        const uint16x8_t abs = vabdl_u8(d, s);

        *sse = vmlal_u16(*sse, vget_low_u16(abs), vget_low_u16(abs));
        *sse = vmlal_u16(*sse, vget_high_u16(abs), vget_high_u16(abs));

        src += 2 * 4 * subsampling_factor; // with * 2 rows per iter * subsampling
        dst += 2 * dstride * subsampling_factor;
        height -= 2 * subsampling_factor;
    } while (height != 0);
}

static inline void mse_8xn_8bit_neon(const uint8_t *src, const uint8_t *dst, const int32_t dstride, uint32x4_t *sse,
                                     uint8_t height, uint8_t subsampling_factor) {
    uint32x4_t mse0 = vdupq_n_u32(0);
    uint32x4_t mse1 = vdupq_n_u32(0);

    do {
        const uint8x8_t s0 = vld1_u8(src);
        const uint8x8_t s1 = vld1_u8(src + subsampling_factor * 8);
        const uint8x8_t d0 = vld1_u8(dst);
        const uint8x8_t d1 = vld1_u8(dst + subsampling_factor * dstride);

        const uint16x8_t abs0 = vabdl_u8(d0, s0);
        const uint16x8_t abs1 = vabdl_u8(d1, s1);

        mse0 = vmlal_u16(mse0, vget_low_u16(abs0), vget_low_u16(abs0));
        mse0 = vmlal_u16(mse0, vget_high_u16(abs0), vget_high_u16(abs0));
        mse1 = vmlal_u16(mse1, vget_low_u16(abs1), vget_low_u16(abs1));
        mse1 = vmlal_u16(mse1, vget_high_u16(abs1), vget_high_u16(abs1));

        src += 8 * 2 * subsampling_factor;
        dst += 2 * subsampling_factor * dstride;
        height -= 2 * subsampling_factor;
    } while (height != 0);
    *sse = vaddq_u32(*sse, mse0);
    *sse = vaddq_u32(*sse, mse1);
}

uint64_t svt_aom_compute_cdef_dist_8bit_neon(const uint8_t *dst8, int32_t dstride, const uint8_t *src8,
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
            sum += dist_8xn_8bit_neon(
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
                mse_8xn_8bit_neon(src8, dst8 + 8 * by * dstride + 8 * bx, dstride, &mse, 8, subsampling_factor);
                src8 += 8 * 8;
            }
        } else if (bsize == BLOCK_4X8) {
            for (bi = 0; bi < cdef_count; bi++) {
                by = dlist[bi].by;
                bx = dlist[bi].bx;
                mse_4xn_8bit_neon(src8, dst8 + 8 * by * dstride + 4 * bx, dstride, &mse, 8, subsampling_factor);
                src8 += 4 * 8;
            }
        } else if (bsize == BLOCK_8X4) {
            for (bi = 0; bi < cdef_count; bi++) {
                by = dlist[bi].by;
                bx = dlist[bi].bx;
                mse_8xn_8bit_neon(src8, dst8 + 4 * by * dstride + 8 * bx, dstride, &mse, 4, subsampling_factor);
                src8 += 8 * 4;
            }
        } else {
            assert(bsize == BLOCK_4X4);
            for (bi = 0; bi < cdef_count; bi++) {
                by = dlist[bi].by;
                bx = dlist[bi].bx;
                mse_4xn_8bit_neon(src8, dst8 + 4 * by * dstride + 4 * bx, dstride, &mse, 4, subsampling_factor);
                src8 += 4 * 4;
            }
        }

        sum = vaddlvq_u32(mse);
#if !FIX_CDEF_MSE
    }
#endif
    return sum >> 2 * coeff_shift;
}

#if !FIX_CDEF_MSE
static inline uint64_t dist_8xn_16bit_neon(const uint16_t *src, const uint16_t *dst, const int32_t dstride,
                                           const int32_t coeff_shift, uint8_t height, uint8_t subsampling_factor) {
    uint16x8_t ss = vdupq_n_u16(0);
    uint16x8_t dd = vdupq_n_u16(0);
    uint32x4_t s2 = vdupq_n_u32(0);
    uint32x4_t sd = vdupq_n_u32(0);
    uint32x4_t d2 = vdupq_n_u32(0);

    do {
        const uint16x8_t s0 = vld1q_u16(src);
        const uint16x8_t s1 = vld1q_u16(src + subsampling_factor * 8);

        const uint16x8_t d0 = vld1q_u16(dst);
        const uint16x8_t d1 = vld1q_u16(dst + subsampling_factor * dstride);

        ss = vaddq_u16(ss, s0);
        ss = vaddq_u16(ss, s1);
        dd = vaddq_u16(dd, d0);
        dd = vaddq_u16(dd, d1);

        s2 = vmlal_u16(s2, vget_low_u16(s0), vget_low_u16(s0));
        s2 = vmlal_u16(s2, vget_high_u16(s0), vget_high_u16(s0));
        s2 = vmlal_u16(s2, vget_low_u16(s1), vget_low_u16(s1));
        s2 = vmlal_u16(s2, vget_high_u16(s1), vget_high_u16(s1));

        sd = vmlal_u16(sd, vget_low_u16(s0), vget_low_u16(d0));
        sd = vmlal_u16(sd, vget_high_u16(s0), vget_high_u16(d0));
        sd = vmlal_u16(sd, vget_low_u16(s1), vget_low_u16(d1));
        sd = vmlal_u16(sd, vget_high_u16(s1), vget_high_u16(d1));

        d2 = vmlal_u16(d2, vget_low_u16(d0), vget_low_u16(d0));
        d2 = vmlal_u16(d2, vget_high_u16(d0), vget_high_u16(d0));
        d2 = vmlal_u16(d2, vget_low_u16(d1), vget_low_u16(d1));
        d2 = vmlal_u16(d2, vget_high_u16(d1), vget_high_u16(d1));

        src += 8 * 2 * subsampling_factor;
        dst += 2 * subsampling_factor * dstride;
        height -= 2 * subsampling_factor;
    } while (height != 0);

    uint32x4_t ss_s32 = vpaddlq_u16(ss);
    uint32x4_t dd_s32 = vpaddlq_u16(dd);
    uint64_t   sum_s  = vaddlvq_u32(ss_s32);
    uint64_t   sum_d  = vaddlvq_u32(dd_s32);

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

static inline uint32x4_t mse_8xn_16bit_neon(const uint16_t *src, const uint16_t *dst, const int32_t dstride,
                                            uint8_t height, uint8_t subsampling_factor) {
    uint32x4_t sse0 = vdupq_n_u32(0);
    uint32x4_t sse1 = vdupq_n_u32(0);

    do {
        const uint16x8_t s0 = vld1q_u16(src);
        const uint16x8_t s1 = vld1q_u16(src + subsampling_factor * 8);
        const uint16x8_t d0 = vld1q_u16(dst);
        const uint16x8_t d1 = vld1q_u16(dst + subsampling_factor * dstride);

        const uint16x8_t abs0 = vabdq_u16(d0, s0);
        const uint16x8_t abs1 = vabdq_u16(d1, s1);

        sse0 = vmlal_u16(sse0, vget_low_u16(abs0), vget_low_u16(abs0));
        sse0 = vmlal_u16(sse0, vget_high_u16(abs0), vget_high_u16(abs0));
        sse1 = vmlal_u16(sse1, vget_low_u16(abs1), vget_low_u16(abs1));
        sse1 = vmlal_u16(sse1, vget_high_u16(abs1), vget_high_u16(abs1));

        src += 8 * 2 * subsampling_factor;
        dst += 2 * subsampling_factor * dstride;
        height -= 2 * subsampling_factor;
    } while (height != 0);

    return vaddq_u32(sse0, sse1);
}

static inline uint32x4_t mse_4xn_16bit_neon(const uint16_t *src, const uint16_t *dst, const int32_t dstride,
                                            uint8_t height, uint8_t subsampling_factor) {
    uint32x4_t sse = vdupq_n_u32(0);

    do {
        const uint16x8_t s0 = load_u16_4x2(src, 4 * subsampling_factor);
        const uint16x8_t d0 = load_u16_4x2(dst, dstride * subsampling_factor);

        const uint16x8_t diff_0 = vabdq_u16(d0, s0);

        sse = vmlal_u16(sse, vget_low_u16(diff_0), vget_low_u16(diff_0));
        sse = vmlal_u16(sse, vget_high_u16(diff_0), vget_high_u16(diff_0));

        src += 2 * 4 * subsampling_factor; // with * 4 rows per iter * subsampling
        dst += 2 * subsampling_factor * dstride;
        height -= 2 * subsampling_factor;
    } while (height != 0);

    return sse;
}

uint64_t svt_aom_compute_cdef_dist_16bit_neon(const uint16_t *dst, int32_t dstride, const uint16_t *src,
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
            sum += dist_8xn_16bit_neon(
                src, dst + 8 * by * dstride + 8 * bx, dstride, coeff_shift, 8, subsampling_factor);
            src += 8 * 8;
        }
    } else {
#endif
        uint64x2_t mse64 = vdupq_n_u64(0);

        if (bsize == BLOCK_8X8) {
            for (bi = 0; bi < cdef_count; bi++) {
                by               = dlist[bi].by;
                bx               = dlist[bi].bx;
                uint32x4_t mse32 = mse_8xn_16bit_neon(
                    src, dst + 8 * by * dstride + 8 * bx, dstride, 8, subsampling_factor);
                mse64 = vpadalq_u32(mse64, mse32);
                src += 8 * 8;
            }
        } else if (bsize == BLOCK_4X8) {
            for (bi = 0; bi < cdef_count; bi++) {
                by               = dlist[bi].by;
                bx               = dlist[bi].bx;
                uint32x4_t mse32 = mse_4xn_16bit_neon(
                    src, dst + 8 * by * dstride + 4 * bx, dstride, 8, subsampling_factor);
                mse64 = vpadalq_u32(mse64, mse32);
                src += 4 * 8;
            }
        } else if (bsize == BLOCK_8X4) {
            for (bi = 0; bi < cdef_count; bi++) {
                by               = dlist[bi].by;
                bx               = dlist[bi].bx;
                uint32x4_t mse32 = mse_8xn_16bit_neon(
                    src, dst + 4 * by * dstride + 8 * bx, dstride, 4, subsampling_factor);
                mse64 = vpadalq_u32(mse64, mse32);
                src += 8 * 4;
            }
        } else {
            assert(bsize == BLOCK_4X4);
            for (bi = 0; bi < cdef_count; bi++) {
                by               = dlist[bi].by;
                bx               = dlist[bi].bx;
                uint32x4_t mse32 = mse_4xn_16bit_neon(
                    src, dst + 4 * by * dstride + 4 * bx, dstride, 4, subsampling_factor);
                mse64 = vpadalq_u32(mse64, mse32);
                src += 4 * 4;
            }
        }

        sum = vaddvq_u64(mse64);
#if !FIX_CDEF_MSE
    }
#endif

    return sum >> 2 * coeff_shift;
}

uint64_t svt_search_one_dual_neon(int *lev0, int *lev1, int nb_strengths, uint64_t **mse[2], int sb_count, int start_gi,
                                  int end_gi) {
    if (end_gi >= 48) {
        return svt_search_one_dual_c(lev0, lev1, nb_strengths, mse, sb_count, start_gi, end_gi);
    }

    uint64_t      tot_mse[TOTAL_STRENGTHS][TOTAL_STRENGTHS];
    int32_t       i, j;
    uint64_t      best_tot_mse    = (uint64_t)1 << 63;
    int32_t       best_id0        = 0;
    int32_t       best_id1        = 0;
    const int32_t total_strengths = end_gi;
    memset(tot_mse, 0, sizeof(tot_mse));
    /* Loop over the filter blocks in the frame */
    for (i = 0; i < sb_count; i++) {
        int32_t  gi;
        uint64_t best_mse = (uint64_t)1 << 63;
        /* Loop over the already selected nb_strengths (Luma_strength,
           Chroma_strength) pairs, and find the pair that has the smallest mse
           (best_mse) for the current filter block.*/
        /* Find best mse among already selected options. */
        for (gi = 0; gi < nb_strengths; gi++) {
            uint64_t curr = mse[0][i][lev0[gi]];
            curr += mse[1][i][lev1[gi]];
            if (curr < best_mse)
                best_mse = curr;
        }
        /* Loop over the set of available (Luma_strength, Chroma_strength)
           pairs, identify any that provide an mse better than best_mse from the
           step above for the current filter block, and update any corresponding
           total mse (tot_mse[j][k]). */
        /* Find best mse when adding each possible new option. */
        uint64x2_t best0 = vdupq_n_u64(best_mse);
        for (j = start_gi; j < total_strengths; j++) {
            int32_t    k;
            uint64x2_t curr0 = vld1q_dup_u64(&mse[0][i][j]);
            for (k = start_gi; k < total_strengths; k += 4) {
                uint64x2_t curr_v0 = vaddq_u64(curr0, vld1q_u64(&mse[1][i][k]));
                uint64x2_t curr_v1 = vaddq_u64(curr0, vld1q_u64(&mse[1][i][k + 2]));

                uint64x2_t comp0   = vcltq_u64(curr_v0, best0);
                uint64x2_t comp1   = vcltq_u64(curr_v1, best0);
                uint64x2_t best_v0 = vbslq_u64(comp0, curr_v0, best0);
                uint64x2_t best_v1 = vbslq_u64(comp1, curr_v1, best0);

                uint64x2_t tot_mse_v0 = vld1q_u64(&tot_mse[j][k]);
                uint64x2_t tot_mse_v1 = vld1q_u64(&tot_mse[j][k + 2]);
                vst1q_u64(&tot_mse[j][k], vaddq_u64(tot_mse_v0, best_v0));
                vst1q_u64(&tot_mse[j][k + 2], vaddq_u64(tot_mse_v1, best_v1));
            }
        }
    }
    /* Loop over the additionally searched (Luma_strength, Chroma_strength) pairs
       from the step above, and identify any such pair that provided the best mse for
       the whole frame. The identified pair would be added to the set of already selected pairs. */
    for (j = start_gi; j < total_strengths; j++) { // Loop over the additionally searched luma strengths
        int32_t k;
        for (k = start_gi; k < total_strengths; k++) { // Loop over the additionally searched chroma strengths
            if (tot_mse[j][k] < best_tot_mse) {
                best_tot_mse = tot_mse[j][k];
                best_id0     = j; // index for the best luma strength
                best_id1     = k; // index for the best chroma strength
            }
        }
    }
    lev0[nb_strengths] = best_id0; // Add the identified luma strength to the list of selected luma strengths
    lev1[nb_strengths] = best_id1; // Add the identified chroma strength to the list of selected chroma strengths
    return best_tot_mse;
}
