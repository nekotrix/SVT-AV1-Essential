/*
 * Copyright (c) 2022, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>

#include "common_dsp_rtcd.h"
#include "definitions.h"
#include "mem_neon.h"

static INLINE int16x4_t clip3_s16(const int16x4_t val, const int16x4_t low, const int16x4_t high) {
    return vmin_s16(vmax_s16(val, low), high);
}

static INLINE uint16x8_t convert_to_unsigned_pixel_u16(int16x8_t val, int bitdepth) {
    const int16x8_t  low  = vdupq_n_s16(0);
    const uint16x8_t high = vdupq_n_u16((1 << bitdepth) - 1);

    return vminq_u16(vreinterpretq_u16_s16(vmaxq_s16(val, low)), high);
}

// (abs(p1 - p0) > thresh) || (abs(q1 - q0) > thresh)
static INLINE uint16x4_t hev(const uint16x8_t abd_p0p1_q0q1, const uint16_t thresh) {
    const uint16x8_t a = vcgtq_u16(abd_p0p1_q0q1, vdupq_n_u16(thresh));
    return vorr_u16(vget_low_u16(a), vget_high_u16(a));
}

// abs(p0 - q0) * 2 + abs(p1 - q1) / 2 <= outer_thresh
static INLINE uint16x4_t outer_threshold(const uint16x4_t p1, const uint16x4_t p0, const uint16x4_t q0,
                                         const uint16x4_t q1, const uint16_t outer_thresh) {
    const uint16x4_t abd_p0q0    = vabd_u16(p0, q0);
    const uint16x4_t abd_p1q1    = vabd_u16(p1, q1);
    const uint16x4_t p0q0_double = vshl_n_u16(abd_p0q0, 1);
    const uint16x4_t p1q1_half   = vshr_n_u16(abd_p1q1, 1);
    const uint16x4_t sum         = vadd_u16(p0q0_double, p1q1_half);
    return vcle_u16(sum, vdup_n_u16(outer_thresh));
}

// abs(p1 - p0) <= inner_thresh && abs(q1 - q0) <= inner_thresh &&
//   outer_threshold()
static INLINE uint16x4_t needs_filter4(const uint16x8_t abd_p0p1_q0q1, const uint16_t inner_thresh,
                                       const uint16x4_t outer_mask) {
    const uint16x8_t a          = vcleq_u16(abd_p0p1_q0q1, vdupq_n_u16(inner_thresh));
    const uint16x4_t inner_mask = vand_u16(vget_low_u16(a), vget_high_u16(a));
    return vand_u16(inner_mask, outer_mask);
}

static INLINE void filter4_masks(const uint16x8_t p0q0, const uint16x8_t p1q1, const uint16_t hev_thresh,
                                 const uint16x4_t outer_mask, const uint16_t inner_thresh, uint16x4_t *const hev_mask,
                                 uint16x4_t *const needs_filter4_mask) {
    const uint16x8_t p0p1_q0q1 = vabdq_u16(p0q0, p1q1);
    // This includes cases where needs_filter4() is not true and so filter2
    // will not be applied.
    const uint16x4_t hev_tmp_mask = hev(p0p1_q0q1, hev_thresh);

    *needs_filter4_mask = needs_filter4(p0p1_q0q1, inner_thresh, outer_mask);

    // filter2 will only be applied if both needs_filter4() and hev() are true.
    *hev_mask = vand_u16(hev_tmp_mask, *needs_filter4_mask);
}

static INLINE void filter4(const uint16x8_t p0q0, const uint16x8_t p0q1, const uint16x8_t p1q1,
                           const uint16x4_t hev_mask, int bitdepth, uint16x8_t *const p1q1_result,
                           uint16x8_t *const p0q0_result) {
    const uint16x8_t q0p1 = vextq_u16(p0q0, p1q1, 4);
    // a = 3 * (q0 - p0) + Clip3(p1 - q1, min_signed_val, max_signed_val);
    // q0mp0 means "q0 minus p0".
    const int16x8_t q0mp0_p1mq1 = vreinterpretq_s16_u16(vsubq_u16(q0p1, p0q1));
    const int16x4_t q0mp0_3     = vmul_n_s16(vget_low_s16(q0mp0_p1mq1), 3);

    // If this is for filter2 then include |p1mq1|. Otherwise zero it.
    const int16x4_t min_signed_pixel = vdup_n_s16(-(1 << (bitdepth - 1)));
    const int16x4_t max_signed_pixel = vdup_n_s16((1 << (bitdepth - 1)) - 1);
    const int16x4_t p1mq1            = vget_high_s16(q0mp0_p1mq1);
    const int16x4_t p1mq1_saturated  = clip3_s16(p1mq1, min_signed_pixel, max_signed_pixel);
    const int16x4_t hev_option       = vand_s16(vreinterpret_s16_u16(hev_mask), p1mq1_saturated);

    const int16x4_t a = vadd_s16(q0mp0_3, hev_option);

    // We can not shift with rounding because the clamp comes *before* the
    // shifting.
    // a1 = Clip3(a + 4, min_signed_val, max_signed_val) >> 3;
    // a2 = Clip3(a + 3, min_signed_val, max_signed_val) >> 3;
    const int16x4_t plus_four  = clip3_s16(vadd_s16(a, vdup_n_s16(4)), min_signed_pixel, max_signed_pixel);
    const int16x4_t plus_three = clip3_s16(vadd_s16(a, vdup_n_s16(3)), min_signed_pixel, max_signed_pixel);
    const int16x4_t a1         = vshr_n_s16(plus_four, 3);
    const int16x4_t a2         = vshr_n_s16(plus_three, 3);

    // a3 = (a1 + 1) >> 1;
    const int16x4_t a3 = vrshr_n_s16(a1, 1);

    const int16x8_t a3_ma3  = vcombine_s16(a3, vneg_s16(a3));
    const int16x8_t p1q1_a3 = vaddq_s16(vreinterpretq_s16_u16(p1q1), a3_ma3);

    // Need to shift the second term or we end up with a2_ma2.
    const int16x8_t a2_ma1 = vcombine_s16(a2, vneg_s16(a1));
    const int16x8_t p0q0_a = vaddq_s16(vreinterpretq_s16_u16(p0q0), a2_ma1);
    *p1q1_result           = convert_to_unsigned_pixel_u16(p1q1_a3, bitdepth);
    *p0q0_result           = convert_to_unsigned_pixel_u16(p0q0_a, bitdepth);
}

void svt_aom_highbd_lpf_horizontal_4_neon(uint16_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit,
                                          const uint8_t *thresh, int bd) {
    uint16x4_t src[4];
    load_u16_4x4(s - 2 * pitch, pitch, &src[0], &src[1], &src[2], &src[3]);

    // Adjust thresholds to bitdepth.
    const int        outer_thresh = *blimit << (bd - 8);
    const int        inner_thresh = *limit << (bd - 8);
    const int        hev_thresh   = *thresh << (bd - 8);
    const uint16x4_t outer_mask   = outer_threshold(src[0], src[1], src[2], src[3], outer_thresh);
    uint16x4_t       hev_mask;
    uint16x4_t       needs_filter4_mask;
    const uint16x8_t p0q0 = vcombine_u16(src[1], src[2]);
    const uint16x8_t p1q1 = vcombine_u16(src[0], src[3]);
    filter4_masks(p0q0, p1q1, hev_thresh, outer_mask, inner_thresh, &hev_mask, &needs_filter4_mask);

    if (vget_lane_u64(vreinterpret_u64_u16(needs_filter4_mask), 0) == 0) {
        // None of the values will be filtered.
        return;
    }

    // Copy the masks to the high bits for packed comparisons later.
    const uint16x8_t hev_mask_8           = vcombine_u16(hev_mask, hev_mask);
    const uint16x8_t needs_filter4_mask_8 = vcombine_u16(needs_filter4_mask, needs_filter4_mask);

    uint16x8_t       f_p1q1;
    uint16x8_t       f_p0q0;
    const uint16x8_t p0q1 = vcombine_u16(src[1], src[3]);
    filter4(p0q0, p0q1, p1q1, hev_mask, bd, &f_p1q1, &f_p0q0);

    // Already integrated the hev mask when calculating the filtered values.
    const uint16x8_t p0q0_output = vbslq_u16(needs_filter4_mask_8, f_p0q0, p0q0);

    // p1/q1 are unmodified if only hev() is true. This works because it was and'd
    // with |needs_filter4_mask| previously.
    const uint16x8_t p1q1_mask   = veorq_u16(hev_mask_8, needs_filter4_mask_8);
    const uint16x8_t p1q1_output = vbslq_u16(p1q1_mask, f_p1q1, p1q1);

    store_u16_4x4(s - 2 * pitch,
                  pitch,
                  vget_low_u16(p1q1_output),
                  vget_low_u16(p0q0_output),
                  vget_high_u16(p0q0_output),
                  vget_high_u16(p1q1_output));
}
