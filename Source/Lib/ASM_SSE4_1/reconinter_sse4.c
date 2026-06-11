/*
 * Copyright (c) 2018, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <emmintrin.h> // SSE2
#include <smmintrin.h> /* SSE4.1 */

#include "definitions.h"

void svt_av1_build_compound_diffwtd_mask_d16_sse4_1(uint8_t *mask, DIFFWTD_MASK_TYPE mask_type,
                                                    const CONV_BUF_TYPE *src0, int src0_stride,
                                                    const CONV_BUF_TYPE *src1, int src1_stride, int h, int w,
                                                    ConvolveParams *conv_params, int bd) {
    const int     which_inverse = (mask_type == DIFFWTD_38) ? 0 : 1;
    const int     mask_base     = 38;
    int           round         = 2 * FILTER_BITS - conv_params->round_0 - conv_params->round_1 + (bd - 8);
    const __m128i round_const   = _mm_set1_epi16((1 << round) >> 1);
    const __m128i mask_base_16  = _mm_set1_epi16(mask_base);
    const __m128i clip_diff     = _mm_set1_epi16(AOM_BLEND_A64_MAX_ALPHA);
    const __m128i add_const     = _mm_set1_epi16((which_inverse ? AOM_BLEND_A64_MAX_ALPHA : 0));
    const __m128i add_sign      = _mm_set1_epi16((which_inverse ? -1 : 1));

    int i, j;
    // When rounding constant is added, there is a possibility of overflow.
    // However that much precision is not required. Code should very well work for
    // other values of DIFF_FACTOR_LOG2 and AOM_BLEND_A64_MAX_ALPHA as well. But
    // there is a possibility of corner case bugs.
    assert(DIFF_FACTOR_LOG2 == 4);
    assert(AOM_BLEND_A64_MAX_ALPHA == 64);
    for (i = 0; i < h; ++i) {
        for (j = 0; j < w; j += 8) {
            const __m128i data_src0 = _mm_loadu_si128((__m128i *)&src0[(i * src0_stride) + j]);
            const __m128i data_src1 = _mm_loadu_si128((__m128i *)&src1[(i * src1_stride) + j]);

            const __m128i diffa       = _mm_subs_epu16(data_src0, data_src1);
            const __m128i diffb       = _mm_subs_epu16(data_src1, data_src0);
            const __m128i diff        = _mm_max_epu16(diffa, diffb);
            const __m128i diff_round  = _mm_srli_epi16(_mm_adds_epu16(diff, round_const), round);
            const __m128i diff_factor = _mm_srli_epi16(diff_round, DIFF_FACTOR_LOG2);
            const __m128i diff_mask   = _mm_adds_epi16(diff_factor, mask_base_16);
            __m128i       diff_clamp  = _mm_min_epi16(diff_mask, clip_diff);
            // clamp to 0 can be skipped since we are using add and saturate
            // instruction

            const __m128i diff_sign     = _mm_sign_epi16(diff_clamp, add_sign);
            const __m128i diff_const_16 = _mm_add_epi16(diff_sign, add_const);

            // 8 bit conversion and saturation to uint8
            const __m128i res_8 = _mm_packus_epi16(diff_const_16, diff_const_16);

            // Store values into the destination buffer
            __m128i *const dst = (__m128i *)&mask[i * w + j];

            if ((w - j) > 4) {
                _mm_storel_epi64(dst, res_8);
            } else { // w==4
                *(uint32_t *)dst = _mm_cvtsi128_si32(res_8);
            }
        }
    }
}
