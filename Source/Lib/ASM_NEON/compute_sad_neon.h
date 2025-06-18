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

#ifndef COMPUTE_SAD_NEON_H
#define COMPUTE_SAD_NEON_H

#include <arm_neon.h>

#include "aom_dsp_rtcd.h"
#include "sum_neon.h"

#if __GNUC__
#define svt_ctzll(id, x) id = (unsigned long)__builtin_ctzll(x)
#elif defined(_MSC_VER)
#include <intrin.h>

#define svt_ctzll(id, x) _BitScanForward64(&id, x)
#endif

/* Find the position of the first occurrence of 'value' in the vector 'x'.
 * Returns the position (index) of the first occurrence of 'value' in the vector 'x'. */
static inline uint16_t findposq_u32(uint32x4_t x, uint32_t value) {
    uint32x4_t val_mask = vdupq_n_u32(value);

    /* Pack the information in the lower 64 bits of the register by considering only alternate
     * 16-bit lanes. */
    uint16x4_t is_one = vmovn_u32(vceqq_u32(x, val_mask));

    /* Get the lower 64 bits from the 128-bit register. */
    uint64_t idx = vget_lane_u64(vreinterpret_u64_u16(is_one), 0);

    /* Calculate the position as an index, dividing by 16 to account for 16-bit lanes. */
    uint64_t res;
    svt_ctzll(res, idx);
    return res >> 4;
}

static inline void update_best_sad_u32(uint32x4_t sad4, uint64_t *best_sad, int16_t *x_search_center,
                                       int16_t *y_search_center, int16_t x_search_index, int16_t y_search_index) {
    /* Find the minimum SAD value out of the 4 search spaces. */
    uint64_t temp_sad = vminvq_u32(sad4);

    if (temp_sad < *best_sad) {
        *best_sad        = temp_sad;
        *x_search_center = (int16_t)(x_search_index + findposq_u32(sad4, temp_sad));
        *y_search_center = y_search_index;
    }
}

/* Find the position of the first occurrence of 'value' in the vector 'x'.
 * Returns the position (index) of the first occurrence of 'value' in the vector 'x'. */
static inline uint16_t findposq_u16(uint16x8_t x, uint16_t value) {
    uint16x8_t val_mask = vdupq_n_u16(value);

    /* Pack the information in the lower 64 bits of the register by considering only alternate
     * 8-bit lanes. */
    uint8x8_t is_one = vmovn_u16(vceqq_u16(x, val_mask));

    /* Get the lower 64 bits from the 128-bit register. */
    uint64_t idx = vget_lane_u64(vreinterpret_u64_u8(is_one), 0);

    /* Calculate the position as an index, dividing by 8 to account for 8-bit lanes. */
    uint64_t res;
    svt_ctzll(res, idx);
    return res >> 3;
}

static inline void update_best_sad_u16(uint16x8_t sad8, uint64_t *best_sad, int16_t *x_search_center,
                                       int16_t *y_search_center, int16_t x_search_index, int16_t y_search_index) {
    /* Find the minimum SAD value out of the 8 search spaces. */
    uint64_t temp_sad = vminvq_u16(sad8);

    if (temp_sad < *best_sad) {
        *best_sad        = temp_sad;
        *x_search_center = (int16_t)(x_search_index + findposq_u16(sad8, temp_sad));
        *y_search_center = y_search_index;
    }
}

static inline void update_best_sad(uint64_t temp_sad, uint64_t *best_sad, int16_t *x_search_center,
                                   int16_t *y_search_center, int16_t x_search_index, int16_t y_search_index) {
    if (temp_sad < *best_sad) {
        *best_sad        = temp_sad;
        *x_search_center = x_search_index;
        *y_search_center = y_search_index;
    }
}

/* Return a uint16x8 vector with 'n' lanes filled with 0 and the others filled with 65535
 * The valid range for 'n' is 0 to 7 */
static inline uint16x8_t prepare_maskq_u16(uint16_t n) {
    uint64_t mask    = UINT64_MAX;
    mask             = mask << (8 * n);
    uint8x16_t mask8 = vcombine_u8(vcreate_u8(mask), vdup_n_u8(0));
    return vreinterpretq_u16_u8(vzip1q_u8(mask8, mask8));
}

/* Return a uint32x4 vector with 'n' lanes filled with 0 and the others filled with 4294967295
 * The valid range for 'n' is 0 to 4 */
static inline uint32x4_t prepare_maskq_u32(uint16_t n) {
    uint64_t mask    = UINT64_MAX;
    mask             = n < 4 ? (mask << (16 * n)) : 0;
    uint16x8_t mask8 = vcombine_u16(vcreate_u16(mask), vdup_n_u16(0));
    return vreinterpretq_u32_u16(vzip1q_u16(mask8, mask8));
}

static inline uint32_t sad8xh_neon(const uint8_t *src_ptr, uint32_t src_stride, const uint8_t *ref_ptr,
                                   uint32_t ref_stride, uint32_t h) {
    uint16x8_t sum = vdupq_n_u16(0);
    do {
        uint8x8_t s = vld1_u8(src_ptr);
        uint8x8_t r = vld1_u8(ref_ptr);

        sum = vabal_u8(sum, s, r);

        src_ptr += src_stride;
        ref_ptr += ref_stride;
    } while (--h != 0);

    return vaddlvq_u16(sum);
}

static inline uint32x4_t sad8xhx4d_neon(const uint8_t *src, uint32_t src_stride, const uint8_t *ref,
                                        uint32_t ref_stride, uint32_t h) {
    uint16x8_t sum[4];
    sum[0] = vdupq_n_u16(0);
    sum[1] = vdupq_n_u16(0);
    sum[2] = vdupq_n_u16(0);
    sum[3] = vdupq_n_u16(0);

    do {
        uint8x8_t s = vld1_u8(src);
        sum[0]      = vabal_u8(sum[0], s, vld1_u8(ref + 0));
        sum[1]      = vabal_u8(sum[1], s, vld1_u8(ref + 1));
        sum[2]      = vabal_u8(sum[2], s, vld1_u8(ref + 2));
        sum[3]      = vabal_u8(sum[3], s, vld1_u8(ref + 3));

        src += src_stride;
        ref += ref_stride;
    } while (--h != 0);

    return horizontal_add_4d_u16x8(sum);
}

static inline void svt_sad_loop_kernel8xh_neon(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride,
                                               uint32_t block_height, uint64_t *best_sad, int16_t *x_search_center,
                                               int16_t *y_search_center, uint32_t src_stride_raw,
                                               int16_t search_area_width, int16_t search_area_height) {
    for (int y_search_index = 0; y_search_index < search_area_height; y_search_index++) {
        int16_t x_search_index;
        for (x_search_index = 0; x_search_index <= search_area_width - 4; x_search_index += 4) {
            /* Get the SAD of 4 search spaces aligned along the width and store it in 'sad4'. */
            uint32x4_t sad4 = sad8xhx4d_neon(src, src_stride, ref + x_search_index, ref_stride, block_height);
            update_best_sad_u32(sad4, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
        }

        for (; x_search_index < search_area_width; x_search_index++) {
            /* Get the SAD of 1 search spaces aligned along the width and store it in 'temp_sad'. */
            uint64_t temp_sad = sad8xh_neon(src, src_stride, ref + x_search_index, ref_stride, block_height);
            update_best_sad(temp_sad, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
        }
        ref += src_stride_raw;
    }
}

#endif // COMPUTE_SAD_NEON_H
