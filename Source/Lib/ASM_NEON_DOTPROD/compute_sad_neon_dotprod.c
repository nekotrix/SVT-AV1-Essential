/*
 * Copyright (c) 2023, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>

#include "aom_dsp_rtcd.h"
#include "common_dsp_rtcd.h"
#include "compute_sad_c.h"
#include "mcomp.h"
#include "mem_neon.h"
#include "sum_neon.h"
#include "utility.h"

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
    uint64_t temp_sad;

    /* Find the minimum SAD value out of the 4 search spaces. */
    temp_sad = vminvq_u32(sad4);

    if (temp_sad < *best_sad) {
        *best_sad        = temp_sad;
        *x_search_center = (int16_t)(x_search_index + findposq_u32(sad4, temp_sad));
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

static inline unsigned int sadwxh_neon_dotprod(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,
                                               int ref_stride, int w, int h) {
    // Only two accumulators are required for optimal instruction throughput of
    // the ABD, UDOT sequence on CPUs with either 2 or 4 Neon pipes.
    uint32x4_t sum[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};

    do {
        int j = 0;
        do {
            uint8x16_t s0, s1, r0, r1, diff0, diff1;

            s0     = vld1q_u8(src_ptr + j);
            r0     = vld1q_u8(ref_ptr + j);
            diff0  = vabdq_u8(s0, r0);
            sum[0] = vdotq_u32(sum[0], diff0, vdupq_n_u8(1));

            s1     = vld1q_u8(src_ptr + j + 16);
            r1     = vld1q_u8(ref_ptr + j + 16);
            diff1  = vabdq_u8(s1, r1);
            sum[1] = vdotq_u32(sum[1], diff1, vdupq_n_u8(1));

            j += 32;
        } while (j < w);

        src_ptr += src_stride;
        ref_ptr += ref_stride;
    } while (--h != 0);

    return vaddvq_u32(vaddq_u32(sum[0], sum[1]));
}

static inline unsigned int sad32xh_neon_dotprod(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,
                                                int ref_stride, int h) {
    return sadwxh_neon_dotprod(src_ptr, src_stride, ref_ptr, ref_stride, 32, h);
}

static inline unsigned int sad64xh_neon_dotprod(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,
                                                int ref_stride, int h) {
    return sadwxh_neon_dotprod(src_ptr, src_stride, ref_ptr, ref_stride, 64, h);
}

static inline void sad16_neon_dotprod(uint8x16_t src, uint8x16_t ref, uint32x4_t *const sad_sum) {
    uint8x16_t abs_diff = vabdq_u8(src, ref);
    *sad_sum            = vdotq_u32(*sad_sum, abs_diff, vdupq_n_u8(1));
}

static inline uint32x4_t sadwxhx4d_large_neon_dotprod(const uint8_t *src, int src_stride, const uint8_t *ref,
                                                      int ref_stride, int w, int h) {
    uint32x4_t sum_lo[4] = {vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)};
    uint32x4_t sum_hi[4] = {vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)};
    uint32x4_t sum[4];

    do {
        int            j       = w;
        const uint8_t *ref_ptr = ref;
        const uint8_t *src_ptr = src;
        do {
            const uint8x16_t s0 = vld1q_u8(src_ptr);
            sad16_neon_dotprod(s0, vld1q_u8(ref_ptr + 0), &sum_lo[0]);
            sad16_neon_dotprod(s0, vld1q_u8(ref_ptr + 1), &sum_lo[1]);
            sad16_neon_dotprod(s0, vld1q_u8(ref_ptr + 2), &sum_lo[2]);
            sad16_neon_dotprod(s0, vld1q_u8(ref_ptr + 3), &sum_lo[3]);

            const uint8x16_t s1 = vld1q_u8(src_ptr + 16);
            sad16_neon_dotprod(s1, vld1q_u8(ref_ptr + 0 + 16), &sum_hi[0]);
            sad16_neon_dotprod(s1, vld1q_u8(ref_ptr + 1 + 16), &sum_hi[1]);
            sad16_neon_dotprod(s1, vld1q_u8(ref_ptr + 2 + 16), &sum_hi[2]);
            sad16_neon_dotprod(s1, vld1q_u8(ref_ptr + 3 + 16), &sum_hi[3]);

            j -= 32;
            ref_ptr += 32;
            src_ptr += 32;
        } while (j != 0);

        src += src_stride;
        ref += ref_stride;
    } while (--h != 0);

    sum[0] = vaddq_u32(sum_lo[0], sum_hi[0]);
    sum[1] = vaddq_u32(sum_lo[1], sum_hi[1]);
    sum[2] = vaddq_u32(sum_lo[2], sum_hi[2]);
    sum[3] = vaddq_u32(sum_lo[3], sum_hi[3]);

    return horizontal_add_4d_u32x4(sum);
}

static inline uint32x4_t sad64xhx4d_neon_dotprod(const uint8_t *src, uint32_t src_stride, const uint8_t *ref,
                                                 uint32_t ref_stride, uint32_t h) {
    return sadwxhx4d_large_neon_dotprod(src, src_stride, ref, ref_stride, 64, h);
}

static inline uint32x4_t sad32xhx4d_neon_dotprod(const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride,
                                                 int h) {
    return sadwxhx4d_large_neon_dotprod(src, src_stride, ref, ref_stride, 32, h);
}

static inline uint32x4_t sad16xhx4d_neon_dotprod(const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride,
                                                 int h) {
    uint32x4_t sum[4] = {vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)};

    do {
        const uint8x16_t s = vld1q_u8(src);
        sad16_neon_dotprod(s, vld1q_u8(ref + 0), &sum[0]);
        sad16_neon_dotprod(s, vld1q_u8(ref + 1), &sum[1]);
        sad16_neon_dotprod(s, vld1q_u8(ref + 2), &sum[2]);
        sad16_neon_dotprod(s, vld1q_u8(ref + 3), &sum[3]);

        src += src_stride;
        ref += ref_stride;
    } while (--h != 0);

    return horizontal_add_4d_u32x4(sum);
}

static inline unsigned int sad16xh_neon_dotprod(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,
                                                int ref_stride, int h) {
    uint32x4_t sum[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};

    do {
        uint8x16_t s0, r0, diff0;

        s0     = vld1q_u8(src_ptr);
        r0     = vld1q_u8(ref_ptr);
        diff0  = vabdq_u8(s0, r0);
        sum[0] = vdotq_u32(sum[0], diff0, vdupq_n_u8(1));

        src_ptr += src_stride;
        ref_ptr += ref_stride;

        uint8x16_t s1, r1, diff1;

        s1     = vld1q_u8(src_ptr);
        r1     = vld1q_u8(ref_ptr);
        diff1  = vabdq_u8(s1, r1);
        sum[1] = vdotq_u32(sum[1], diff1, vdupq_n_u8(1));

        src_ptr += src_stride;
        ref_ptr += ref_stride;

        h -= 2;
    } while (h > 1);

    if (h) {
        uint8x16_t s0, r0, diff0;

        s0     = vld1q_u8(src_ptr);
        r0     = vld1q_u8(ref_ptr);
        diff0  = vabdq_u8(s0, r0);
        sum[0] = vdotq_u32(sum[0], diff0, vdupq_n_u8(1));
    }

    return vaddvq_u32(vaddq_u32(sum[0], sum[1]));
}

static inline void svt_sad_loop_kernel16xh_neon_dotprod(uint8_t *src, uint32_t src_stride, uint8_t *ref,
                                                        uint32_t ref_stride, uint32_t block_height, uint64_t *best_sad,
                                                        int16_t *x_search_center, int16_t *y_search_center,
                                                        uint32_t src_stride_raw, uint8_t skip_search_line,
                                                        int16_t search_area_width, int16_t search_area_height) {
    int y_search_start = 0;
    int y_search_step  = 1;

    if (block_height <= 16 && skip_search_line) {
        ref += src_stride_raw;
        src_stride_raw *= 2;
        y_search_start = 1;
        y_search_step  = 2;
    }

    for (int y_search_index = y_search_start; y_search_index < search_area_height; y_search_index += y_search_step) {
        for (int x_search_index = 0; x_search_index < search_area_width; x_search_index += 8) {
            /* Get the SAD of 8 search spaces aligned along the width and store it in 'sad4'. */
            uint32x4_t sad4_0 = sad16xhx4d_neon_dotprod(
                src, src_stride, ref + x_search_index, ref_stride, block_height);
            uint32x4_t sad4_1 = sad16xhx4d_neon_dotprod(
                src, src_stride, ref + x_search_index + 4, ref_stride, block_height);
            update_best_sad_u32(sad4_0, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
            update_best_sad_u32(sad4_1, best_sad, x_search_center, y_search_center, x_search_index + 4, y_search_index);
        }

        ref += src_stride_raw;
    }
}

static inline void svt_sad_loop_kernel16xh_small_neon_dotprod(uint8_t *src, uint32_t src_stride, uint8_t *ref,
                                                              uint32_t ref_stride, uint32_t block_height,
                                                              uint64_t *best_sad, int16_t *x_search_center,
                                                              int16_t *y_search_center, uint32_t src_stride_raw,
                                                              uint8_t skip_search_line, int16_t search_area_width,
                                                              int16_t search_area_height) {
    int y_search_start = 0;
    int y_search_step  = 1;

    if (block_height <= 16 && skip_search_line) {
        ref += src_stride_raw;
        src_stride_raw *= 2;
        y_search_start = 1;
        y_search_step  = 2;
    }

    for (int y_search_index = y_search_start; y_search_index < search_area_height; y_search_index += y_search_step) {
        int x_search_index;
        for (x_search_index = 0; x_search_index <= search_area_width - 4; x_search_index += 4) {
            /* Get the SAD of 4 search spaces aligned along the width and store it in 'sad4'. */
            uint32x4_t sad4_0 = sad16xhx4d_neon_dotprod(
                src, src_stride, ref + x_search_index, ref_stride, block_height);
            update_best_sad_u32(sad4_0, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
        }
        for (; x_search_index < search_area_width; x_search_index++) {
            /* Get the SAD of 1 search space aligned along the width and store it in 'temp_sad'. */
            uint64_t temp_sad = sad16xh_neon_dotprod(src, src_stride, ref + x_search_index, ref_stride, block_height);
            update_best_sad(temp_sad, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
        }

        ref += src_stride_raw;
    }
}

static inline void svt_sad_loop_kernel32xh_neon_dotprod(uint8_t *src, uint32_t src_stride, uint8_t *ref,
                                                        uint32_t ref_stride, uint32_t block_height, uint64_t *best_sad,
                                                        int16_t *x_search_center, int16_t *y_search_center,
                                                        uint32_t src_stride_raw, int16_t search_area_width,
                                                        int16_t search_area_height) {
    for (int y_search_index = 0; y_search_index < search_area_height; y_search_index++) {
        for (int x_search_index = 0; x_search_index < search_area_width; x_search_index += 8) {
            /* Get the SAD of 4 search spaces aligned along the width and store it in 'sad4'. */
            uint32x4_t sad4_0 = sad32xhx4d_neon_dotprod(
                src, src_stride, ref + x_search_index, ref_stride, block_height);
            uint32x4_t sad4_1 = sad32xhx4d_neon_dotprod(
                src, src_stride, ref + x_search_index + 4, ref_stride, block_height);
            update_best_sad_u32(sad4_0, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
            update_best_sad_u32(sad4_1, best_sad, x_search_center, y_search_center, x_search_index + 4, y_search_index);
        }

        ref += src_stride_raw;
    }
}

static inline void svt_sad_loop_kernel32xh_small_neon_dotprod(uint8_t *src, uint32_t src_stride, uint8_t *ref,
                                                              uint32_t ref_stride, uint32_t block_height,
                                                              uint64_t *best_sad, int16_t *x_search_center,
                                                              int16_t *y_search_center, uint32_t src_stride_raw,
                                                              int16_t search_area_width, int16_t search_area_height) {
    for (int y_search_index = 0; y_search_index < search_area_height; y_search_index++) {
        int x_search_index;
        for (x_search_index = 0; x_search_index <= search_area_width - 4; x_search_index += 4) {
            /* Get the SAD of 4 search spaces aligned along the width and store it in 'sad4'. */
            uint32x4_t sad4_0 = sad32xhx4d_neon_dotprod(
                src, src_stride, ref + x_search_index, ref_stride, block_height);
            update_best_sad_u32(sad4_0, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
        }
        for (; x_search_index < search_area_width; x_search_index++) {
            /* Get the SAD of 1 search space aligned along the width and store it in 'temp_sad'. */
            uint64_t temp_sad = sad32xh_neon_dotprod(src, src_stride, ref + x_search_index, ref_stride, block_height);
            update_best_sad(temp_sad, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
        }

        ref += src_stride_raw;
    }
}

static inline void svt_sad_loop_kernel64xh_neon_dotprod(uint8_t *src, uint32_t src_stride, uint8_t *ref,
                                                        uint32_t ref_stride, uint32_t block_height, uint64_t *best_sad,
                                                        int16_t *x_search_center, int16_t *y_search_center,
                                                        uint32_t src_stride_raw, int16_t search_area_width,
                                                        int16_t search_area_height) {
    for (int y_search_index = 0; y_search_index < search_area_height; y_search_index++) {
        for (int x_search_index = 0; x_search_index < search_area_width; x_search_index += 8) {
            /* Get the SAD of 4 search spaces aligned along the width and store it in 'sad4'. */
            uint32x4_t sad4_0 = sad64xhx4d_neon_dotprod(
                src, src_stride, ref + x_search_index, ref_stride, block_height);
            uint32x4_t sad4_1 = sad64xhx4d_neon_dotprod(
                src, src_stride, ref + x_search_index + 4, ref_stride, block_height);
            update_best_sad_u32(sad4_0, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
            update_best_sad_u32(sad4_1, best_sad, x_search_center, y_search_center, x_search_index + 4, y_search_index);
        }

        ref += src_stride_raw;
    }
}

static inline void svt_sad_loop_kernel64xh_small_neon_dotprod(uint8_t *src, uint32_t src_stride, uint8_t *ref,
                                                              uint32_t ref_stride, uint32_t block_height,
                                                              uint64_t *best_sad, int16_t *x_search_center,
                                                              int16_t *y_search_center, uint32_t src_stride_raw,
                                                              int16_t search_area_width, int16_t search_area_height) {
    for (int y_search_index = 0; y_search_index < search_area_height; y_search_index++) {
        int x_search_index;
        for (x_search_index = 0; x_search_index <= search_area_width - 4; x_search_index += 4) {
            /* Get the SAD of 4 search spaces aligned along the width and store it in 'sad4'. */
            uint32x4_t sad4_0 = sad64xhx4d_neon_dotprod(
                src, src_stride, ref + x_search_index, ref_stride, block_height);
            update_best_sad_u32(sad4_0, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
        }
        for (; x_search_index < search_area_width; x_search_index++) {
            /* Get the SAD of 1 search space aligned along the width and store it in 'temp_sad'. */
            uint64_t temp_sad = sad64xh_neon_dotprod(src, src_stride, ref + x_search_index, ref_stride, block_height);
            update_best_sad(temp_sad, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
        }

        ref += src_stride_raw;
    }
}

void svt_sad_loop_kernel_neon_dotprod(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride,
                                      uint32_t block_height, uint32_t block_width, uint64_t *best_sad,
                                      int16_t *x_search_center, int16_t *y_search_center, uint32_t src_stride_raw,
                                      uint8_t skip_search_line, int16_t search_area_width, int16_t search_area_height) {
    *best_sad = UINT64_MAX;
    // Most of the time search_area_width is a multiple of 8, so specialize for this case so that we run only sad4d.
    if (search_area_width % 8 == 0) {
        switch (block_width) {
        case 16: {
            svt_sad_loop_kernel16xh_neon_dotprod(src,
                                                 src_stride,
                                                 ref,
                                                 ref_stride,
                                                 block_height,
                                                 best_sad,
                                                 x_search_center,
                                                 y_search_center,
                                                 src_stride_raw,
                                                 skip_search_line,
                                                 search_area_width,
                                                 search_area_height);
            break;
        }
        case 32: {
            svt_sad_loop_kernel32xh_neon_dotprod(src,
                                                 src_stride,
                                                 ref,
                                                 ref_stride,
                                                 block_height,
                                                 best_sad,
                                                 x_search_center,
                                                 y_search_center,
                                                 src_stride_raw,
                                                 search_area_width,
                                                 search_area_height);
            break;
        }
        case 64: {
            svt_sad_loop_kernel64xh_neon_dotprod(src,
                                                 src_stride,
                                                 ref,
                                                 ref_stride,
                                                 block_height,
                                                 best_sad,
                                                 x_search_center,
                                                 y_search_center,
                                                 src_stride_raw,
                                                 search_area_width,
                                                 search_area_height);
            break;
        }
        default: {
            svt_sad_loop_kernel_neon(src,
                                     src_stride,
                                     ref,
                                     ref_stride,
                                     block_height,
                                     block_width,
                                     best_sad,
                                     x_search_center,
                                     y_search_center,
                                     src_stride_raw,
                                     skip_search_line,
                                     search_area_width,
                                     search_area_height);
            break;
        }
        }

    } else {
        switch (block_width) {
        case 16: {
            svt_sad_loop_kernel16xh_small_neon_dotprod(src,
                                                       src_stride,
                                                       ref,
                                                       ref_stride,
                                                       block_height,
                                                       best_sad,
                                                       x_search_center,
                                                       y_search_center,
                                                       src_stride_raw,
                                                       skip_search_line,
                                                       search_area_width,
                                                       search_area_height);
            break;
        }
        case 32: {
            svt_sad_loop_kernel32xh_small_neon_dotprod(src,
                                                       src_stride,
                                                       ref,
                                                       ref_stride,
                                                       block_height,
                                                       best_sad,
                                                       x_search_center,
                                                       y_search_center,
                                                       src_stride_raw,
                                                       search_area_width,
                                                       search_area_height);
            break;
        }
        case 64: {
            svt_sad_loop_kernel64xh_small_neon_dotprod(src,
                                                       src_stride,
                                                       ref,
                                                       ref_stride,
                                                       block_height,
                                                       best_sad,
                                                       x_search_center,
                                                       y_search_center,
                                                       src_stride_raw,
                                                       search_area_width,
                                                       search_area_height);
            break;
        }
        default: {
            svt_sad_loop_kernel_neon(src,
                                     src_stride,
                                     ref,
                                     ref_stride,
                                     block_height,
                                     block_width,
                                     best_sad,
                                     x_search_center,
                                     y_search_center,
                                     src_stride_raw,
                                     skip_search_line,
                                     search_area_width,
                                     search_area_height);
            break;
        }
        }
    }
}
