/*
 * Copyright (c) 2023, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>
#include <assert.h>

#include "common_dsp_rtcd.h"
#include "definitions.h"
#include "intra_prediction.h"

void svt_av1_filter_intra_edge_high_neon(uint16_t *p, int sz, int strength) {
    if (!strength)
        return;
    assert(sz >= 0 && sz <= 129);

    DECLARE_ALIGNED(16, static const uint16_t, idx[8]) = {0, 1, 2, 3, 4, 5, 6, 7};
    const uint16x8_t index                             = vld1q_u16(idx);

    uint16_t edge[160]; // Max value of sz + enough padding for vector accesses.
    memcpy(edge + 1, p, sz * sizeof(*p));

    // Populate extra space appropriately.
    edge[0]      = edge[1];
    edge[sz + 1] = edge[sz];
    edge[sz + 2] = edge[sz];

    // Don't overwrite first pixel.
    uint16_t *dst = p + 1;
    sz--;

    if (strength == 1) { // Filter: {4, 8, 4}.
        const uint16_t *src = edge + 1;

        while (sz >= 8) {
            uint16x8_t s0 = vld1q_u16(src);
            uint16x8_t s1 = vld1q_u16(src + 1);
            uint16x8_t s2 = vld1q_u16(src + 2);

            // Make use of the identity:
            // (4*a + 8*b + 4*c) >> 4 == (a + (b << 1) + c) >> 2
            uint16x8_t t0  = vaddq_u16(s0, s2);
            uint16x8_t t1  = vaddq_u16(s1, s1);
            uint16x8_t sum = vaddq_u16(t0, t1);
            uint16x8_t res = vrshrq_n_u16(sum, 2);

            vst1q_u16(dst, res);

            src += 8;
            dst += 8;
            sz -= 8;
        }

        if (sz > 0) { // Handle sz < 8 to avoid modifying out-of-bounds values.
            uint16x8_t s0 = vld1q_u16(src);
            uint16x8_t s1 = vld1q_u16(src + 1);
            uint16x8_t s2 = vld1q_u16(src + 2);

            // Make use of the identity:
            // (4*a + 8*b + 4*c) >> 4 == (a + (b << 1) + c) >> 2
            uint16x8_t t0  = vaddq_u16(s0, s2);
            uint16x8_t t1  = vaddq_u16(s1, s1);
            uint16x8_t sum = vaddq_u16(t0, t1);
            uint16x8_t res = vrshrq_n_u16(sum, 2);

            // Mask off out-of-bounds indices.
            uint16x8_t current_dst = vld1q_u16(dst);
            uint16x8_t mask        = vcgtq_u16(vdupq_n_u16(sz), index);
            res                    = vbslq_u16(mask, res, current_dst);

            vst1q_u16(dst, res);
        }
    } else if (strength == 2) { // Filter: {5, 6, 5}.
        const uint16_t *src = edge + 1;

        const uint16x8x3_t filter = {{vdupq_n_u16(5), vdupq_n_u16(6), vdupq_n_u16(5)}};
        while (sz >= 8) {
            uint16x8_t s0 = vld1q_u16(src);
            uint16x8_t s1 = vld1q_u16(src + 1);
            uint16x8_t s2 = vld1q_u16(src + 2);

            uint16x8_t accum = vmulq_u16(s0, filter.val[0]);
            accum            = vmlaq_u16(accum, s1, filter.val[1]);
            accum            = vmlaq_u16(accum, s2, filter.val[2]);
            uint16x8_t res   = vrshrq_n_u16(accum, 4);

            vst1q_u16(dst, res);

            src += 8;
            dst += 8;
            sz -= 8;
        }

        if (sz > 0) { // Handle sz < 8 to avoid modifying out-of-bounds values.
            uint16x8_t s0 = vld1q_u16(src);
            uint16x8_t s1 = vld1q_u16(src + 1);
            uint16x8_t s2 = vld1q_u16(src + 2);

            uint16x8_t accum = vmulq_u16(s0, filter.val[0]);
            accum            = vmlaq_u16(accum, s1, filter.val[1]);
            accum            = vmlaq_u16(accum, s2, filter.val[2]);
            uint16x8_t res   = vrshrq_n_u16(accum, 4);

            // Mask off out-of-bounds indices.
            uint16x8_t current_dst = vld1q_u16(dst);
            uint16x8_t mask        = vcgtq_u16(vdupq_n_u16(sz), index);
            res                    = vbslq_u16(mask, res, current_dst);

            vst1q_u16(dst, res);
        }
    } else { // Filter {2, 4, 4, 4, 2}.
        const uint16_t *src = edge;

        while (sz >= 8) {
            uint16x8_t s0 = vld1q_u16(src);
            uint16x8_t s1 = vld1q_u16(src + 1);
            uint16x8_t s2 = vld1q_u16(src + 2);
            uint16x8_t s3 = vld1q_u16(src + 3);
            uint16x8_t s4 = vld1q_u16(src + 4);

            // Make use of the identity:
            // (2*a + 4*b + 4*c + 4*d + 2*e) >> 4 == (a + ((b + c + d) << 1) + e) >> 3
            uint16x8_t t0  = vaddq_u16(s0, s4);
            uint16x8_t t1  = vaddq_u16(s1, s2);
            t1             = vaddq_u16(t1, s3);
            t1             = vaddq_u16(t1, t1);
            uint16x8_t sum = vaddq_u16(t0, t1);
            uint16x8_t res = vrshrq_n_u16(sum, 3);

            vst1q_u16(dst, res);

            src += 8;
            dst += 8;
            sz -= 8;
        }

        if (sz > 0) { // Handle sz < 8 to avoid modifying out-of-bounds values.
            uint16x8_t s0 = vld1q_u16(src);
            uint16x8_t s1 = vld1q_u16(src + 1);
            uint16x8_t s2 = vld1q_u16(src + 2);
            uint16x8_t s3 = vld1q_u16(src + 3);
            uint16x8_t s4 = vld1q_u16(src + 4);

            // Make use of the identity:
            // (2*a + 4*b + 4*c + 4*d + 2*e) >> 4 == (a + ((b + c + d) << 1) + e) >> 3
            uint16x8_t t0  = vaddq_u16(s0, s4);
            uint16x8_t t1  = vaddq_u16(s1, s2);
            t1             = vaddq_u16(t1, s3);
            t1             = vaddq_u16(t1, t1);
            uint16x8_t sum = vaddq_u16(t0, t1);
            uint16x8_t res = vrshrq_n_u16(sum, 3);

            // Mask off out-of-bounds indices.
            uint16x8_t current_dst = vld1q_u16(dst);
            uint16x8_t mask        = vcgtq_u16(vdupq_n_u16(sz), index);
            res                    = vbslq_u16(mask, res, current_dst);

            vst1q_u16(dst, res);
        }
    }
}

#define SMOOTH_WEIGHT_LOG2_SCALE 8

// clang-format off
const uint16_t sm_weight_arrays_u16[2 * MAX_BLOCK_DIM] = {
    // bs = 4
    255, 149,  85,  64,
    // bs = 8
    255, 197, 146, 105,  73,  50,  37,  32,
    // bs = 16
    255, 225, 196, 170, 145, 123, 102,  84,  68,  54,  43,  33,  26,  20,  17,  16,
    // bs = 32
    255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101,  92,  83,  74,
     66,  59,  52,  45,  39,  34,  29,  25,  21,  17,  14,  12,  10,   9,   8,   8,
    // bs = 64
    255, 248, 240, 233, 225, 218, 210, 203, 196, 189, 182, 176, 169, 163, 156, 150,
    144, 138, 133, 127, 121, 116, 111, 106, 101,  96,  91,  86,  82,  77,  73,  69,
     65,  61,  57,  54,  50,  47,  44,  41,  38,  35,  32,  29,  27,  25,  22,  20,
     18,  16,  15,  13,  12,  10,   9,   8,   7,   6,   6,   5,   5,   4,   4,   4,
};
// clang-format on

static INLINE void highbd_smooth_v_4xh_neon(uint16_t *dst, ptrdiff_t stride, const uint16_t *const top_row,
                                            const uint16_t *const left_column, const int height) {
    const uint16_t        bottom_left = left_column[height - 1];
    const uint16_t *const weights_y   = sm_weight_arrays_u16 + height - 4;

    const uint16x4_t top_v         = vld1_u16(top_row);
    const uint16x4_t bottom_left_v = vdup_n_u16(bottom_left);

    for (int y = 0; y < height; ++y) {
        const uint32x4_t weighted_bl  = vmull_n_u16(bottom_left_v, 256 - weights_y[y]);
        const uint32x4_t weighted_top = vmlal_n_u16(weighted_bl, top_v, weights_y[y]);
        vst1_u16(dst, vrshrn_n_u32(weighted_top, SMOOTH_WEIGHT_LOG2_SCALE));

        dst += stride;
    }
}

static INLINE void highbd_smooth_v_8xh_neon(uint16_t *dst, const ptrdiff_t stride, const uint16_t *const top_row,
                                            const uint16_t *const left_column, const int height) {
    const uint16_t        bottom_left = left_column[height - 1];
    const uint16_t *const weights_y   = sm_weight_arrays_u16 + height - 4;

    const uint16x4_t top_low       = vld1_u16(top_row);
    const uint16x4_t top_high      = vld1_u16(top_row + 4);
    const uint16x4_t bottom_left_v = vdup_n_u16(bottom_left);

    for (int y = 0; y < height; ++y) {
        const uint32x4_t weighted_bl = vmull_n_u16(bottom_left_v, 256 - weights_y[y]);

        const uint32x4_t weighted_top_low = vmlal_n_u16(weighted_bl, top_low, weights_y[y]);
        vst1_u16(dst, vrshrn_n_u32(weighted_top_low, SMOOTH_WEIGHT_LOG2_SCALE));

        const uint32x4_t weighted_top_high = vmlal_n_u16(weighted_bl, top_high, weights_y[y]);
        vst1_u16(dst + 4, vrshrn_n_u32(weighted_top_high, SMOOTH_WEIGHT_LOG2_SCALE));
        dst += stride;
    }
}

#define HIGHBD_SMOOTH_V_NXM(W, H)                                                                 \
    void svt_aom_highbd_smooth_v_predictor_##W##x##H##_neon(                                      \
        uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd) { \
        (void)bd;                                                                                 \
        highbd_smooth_v_##W##xh_neon(dst, y_stride, above, left, H);                              \
    }

HIGHBD_SMOOTH_V_NXM(4, 4)
HIGHBD_SMOOTH_V_NXM(4, 8)
HIGHBD_SMOOTH_V_NXM(4, 16)
HIGHBD_SMOOTH_V_NXM(8, 4)
HIGHBD_SMOOTH_V_NXM(8, 8)
HIGHBD_SMOOTH_V_NXM(8, 16)
HIGHBD_SMOOTH_V_NXM(8, 32)

#undef HIGHBD_SMOOTH_V_NXM

// For width 16 and above.
#define HIGHBD_SMOOTH_V_PREDICTOR(W)                                                                                   \
    static INLINE void highbd_smooth_v_##W##xh_neon(uint16_t             *dst,                                         \
                                                    const ptrdiff_t       stride,                                      \
                                                    const uint16_t *const top_row,                                     \
                                                    const uint16_t *const left_column,                                 \
                                                    const int             height) {                                                \
        const uint16_t        bottom_left = left_column[height - 1];                                                   \
        const uint16_t *const weights_y   = sm_weight_arrays_u16 + height - 4;                                         \
                                                                                                                       \
        uint16x8_t top_vals[(W) >> 3];                                                                                 \
        for (int i = 0; i < (W) >> 3; ++i) {                                                                           \
            const int x = i << 3;                                                                                      \
            top_vals[i] = vld1q_u16(top_row + x);                                                                      \
        }                                                                                                              \
                                                                                                                       \
        const uint16x4_t bottom_left_v = vdup_n_u16(bottom_left);                                                      \
        for (int y = 0; y < height; ++y) {                                                                             \
            const uint32x4_t weighted_bl = vmull_n_u16(bottom_left_v, 256 - weights_y[y]);                             \
                                                                                                                       \
            uint16_t *dst_x = dst;                                                                                     \
            for (int i = 0; i < (W) >> 3; ++i) {                                                                       \
                const uint32x4_t weighted_top_low = vmlal_n_u16(weighted_bl, vget_low_u16(top_vals[i]), weights_y[y]); \
                vst1_u16(dst_x, vrshrn_n_u32(weighted_top_low, SMOOTH_WEIGHT_LOG2_SCALE));                             \
                                                                                                                       \
                const uint32x4_t weighted_top_high = vmlal_n_u16(                                                      \
                    weighted_bl, vget_high_u16(top_vals[i]), weights_y[y]);                                            \
                vst1_u16(dst_x + 4, vrshrn_n_u32(weighted_top_high, SMOOTH_WEIGHT_LOG2_SCALE));                        \
                dst_x += 8;                                                                                            \
            }                                                                                                          \
            dst += stride;                                                                                             \
        }                                                                                                              \
    }

HIGHBD_SMOOTH_V_PREDICTOR(16)
HIGHBD_SMOOTH_V_PREDICTOR(32)
HIGHBD_SMOOTH_V_PREDICTOR(64)

#undef HIGHBD_SMOOTH_V_PREDICTOR

#define HIGHBD_SMOOTH_V_NXM_WIDE(W, H)                                                            \
    void svt_aom_highbd_smooth_v_predictor_##W##x##H##_neon(                                      \
        uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd) { \
        (void)bd;                                                                                 \
        highbd_smooth_v_##W##xh_neon(dst, y_stride, above, left, H);                              \
    }

HIGHBD_SMOOTH_V_NXM_WIDE(16, 4)
HIGHBD_SMOOTH_V_NXM_WIDE(16, 8)
HIGHBD_SMOOTH_V_NXM_WIDE(16, 16)
HIGHBD_SMOOTH_V_NXM_WIDE(16, 32)
HIGHBD_SMOOTH_V_NXM_WIDE(16, 64)
HIGHBD_SMOOTH_V_NXM_WIDE(32, 8)
HIGHBD_SMOOTH_V_NXM_WIDE(32, 16)
HIGHBD_SMOOTH_V_NXM_WIDE(32, 32)
HIGHBD_SMOOTH_V_NXM_WIDE(32, 64)
HIGHBD_SMOOTH_V_NXM_WIDE(64, 16)
HIGHBD_SMOOTH_V_NXM_WIDE(64, 32)
HIGHBD_SMOOTH_V_NXM_WIDE(64, 64)

#undef HIGHBD_SMOOTH_V_NXM_WIDE

// 256 - v = vneg_s8(v)
static inline uint16x4_t negate_s8(const uint16x4_t v) { return vreinterpret_u16_s8(vneg_s8(vreinterpret_s8_u16(v))); }

static INLINE void highbd_smooth_h_4xh_neon(uint16_t *dst, ptrdiff_t stride, const uint16_t *const top_row,
                                            const uint16_t *left_column, int height) {
    const uint16_t top_right = top_row[3];

    const uint16x4_t weights_x        = vld1_u16(sm_weight_arrays_u16);
    const uint16x4_t scaled_weights_x = negate_s8(weights_x);

    const uint32x4_t weighted_tr = vmull_n_u16(scaled_weights_x, top_right);
    do {
        uint16x4_t       left_col       = vld1_u16(left_column);
        const uint32x4_t weighted_left0 = vmlal_lane_u16(weighted_tr, weights_x, left_col, 0);
        const uint32x4_t weighted_left1 = vmlal_lane_u16(weighted_tr, weights_x, left_col, 1);
        const uint32x4_t weighted_left2 = vmlal_lane_u16(weighted_tr, weights_x, left_col, 2);
        const uint32x4_t weighted_left3 = vmlal_lane_u16(weighted_tr, weights_x, left_col, 3);

        vst1_u16(dst + 0 * stride, vrshrn_n_u32(weighted_left0, SMOOTH_WEIGHT_LOG2_SCALE));
        vst1_u16(dst + 1 * stride, vrshrn_n_u32(weighted_left1, SMOOTH_WEIGHT_LOG2_SCALE));
        vst1_u16(dst + 2 * stride, vrshrn_n_u32(weighted_left2, SMOOTH_WEIGHT_LOG2_SCALE));
        vst1_u16(dst + 3 * stride, vrshrn_n_u32(weighted_left3, SMOOTH_WEIGHT_LOG2_SCALE));

        dst += 4 * stride;
        left_column += 4;
        height -= 4;
    } while (height != 0);
}

static INLINE void highbd_smooth_h_8xh_neon(uint16_t *dst, ptrdiff_t stride, const uint16_t *const top_row,
                                            const uint16_t *const left_column, const int height) {
    const uint16_t top_right = top_row[7];

    const uint16x8_t weights_x = vld1q_u16(sm_weight_arrays_u16 + 4);

    const uint32x4_t weighted_tr_low  = vmull_n_u16(negate_s8(vget_low_u16(weights_x)), top_right);
    const uint32x4_t weighted_tr_high = vmull_n_u16(negate_s8(vget_high_u16(weights_x)), top_right);

    for (int y = 0; y < height; ++y) {
        const uint16_t   left_y            = left_column[y];
        const uint32x4_t weighted_left_low = vmlal_n_u16(weighted_tr_low, vget_low_u16(weights_x), left_y);
        vst1_u16(dst, vrshrn_n_u32(weighted_left_low, SMOOTH_WEIGHT_LOG2_SCALE));

        const uint32x4_t weighted_left_high = vmlal_n_u16(weighted_tr_high, vget_high_u16(weights_x), left_y);
        vst1_u16(dst + 4, vrshrn_n_u32(weighted_left_high, SMOOTH_WEIGHT_LOG2_SCALE));
        dst += stride;
    }
}

#define HIGHBD_SMOOTH_H_NXM(W, H)                                                                 \
    void svt_aom_highbd_smooth_h_predictor_##W##x##H##_neon(                                      \
        uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd) { \
        (void)bd;                                                                                 \
        highbd_smooth_h_##W##xh_neon(dst, y_stride, above, left, H);                              \
    }

HIGHBD_SMOOTH_H_NXM(4, 4)
HIGHBD_SMOOTH_H_NXM(4, 8)
HIGHBD_SMOOTH_H_NXM(4, 16)
HIGHBD_SMOOTH_H_NXM(8, 4)
HIGHBD_SMOOTH_H_NXM(8, 8)
HIGHBD_SMOOTH_H_NXM(8, 16)
HIGHBD_SMOOTH_H_NXM(8, 32)

#undef HIGHBD_SMOOTH_H_NXM

// For width 16 and above.
#define HIGHBD_SMOOTH_H_PREDICTOR(W)                                                                               \
    static INLINE void highbd_smooth_h_##W##xh_neon(uint16_t             *dst,                                     \
                                                    ptrdiff_t             stride,                                  \
                                                    const uint16_t *const top_row,                                 \
                                                    const uint16_t *const left_column,                             \
                                                    const int             height) {                                            \
        const uint16_t top_right = top_row[(W)-1];                                                                 \
                                                                                                                   \
        uint16x4_t weights_x_low[(W) >> 3];                                                                        \
        uint16x4_t weights_x_high[(W) >> 3];                                                                       \
        uint32x4_t weighted_tr_low[(W) >> 3];                                                                      \
        uint32x4_t weighted_tr_high[(W) >> 3];                                                                     \
        for (int i = 0; i < (W) >> 3; ++i) {                                                                       \
            const int x         = i << 3;                                                                          \
            weights_x_low[i]    = vld1_u16(sm_weight_arrays_u16 + (W)-4 + x);                                      \
            weighted_tr_low[i]  = vmull_n_u16(negate_s8(weights_x_low[i]), top_right);                             \
            weights_x_high[i]   = vld1_u16(sm_weight_arrays_u16 + (W) + x);                                        \
            weighted_tr_high[i] = vmull_n_u16(negate_s8(weights_x_high[i]), top_right);                            \
        }                                                                                                          \
                                                                                                                   \
        for (int y = 0; y < height; ++y) {                                                                         \
            uint16_t      *dst_x  = dst;                                                                           \
            const uint16_t left_y = left_column[y];                                                                \
            for (int i = 0; i < (W) >> 3; ++i) {                                                                   \
                const uint32x4_t weighted_left_low = vmlal_n_u16(weighted_tr_low[i], weights_x_low[i], left_y);    \
                vst1_u16(dst_x, vrshrn_n_u32(weighted_left_low, SMOOTH_WEIGHT_LOG2_SCALE));                        \
                                                                                                                   \
                const uint32x4_t weighted_left_high = vmlal_n_u16(weighted_tr_high[i], weights_x_high[i], left_y); \
                vst1_u16(dst_x + 4, vrshrn_n_u32(weighted_left_high, SMOOTH_WEIGHT_LOG2_SCALE));                   \
                dst_x += 8;                                                                                        \
            }                                                                                                      \
            dst += stride;                                                                                         \
        }                                                                                                          \
    }

HIGHBD_SMOOTH_H_PREDICTOR(16)
HIGHBD_SMOOTH_H_PREDICTOR(32)
HIGHBD_SMOOTH_H_PREDICTOR(64)

#undef HIGHBD_SMOOTH_H_PREDICTOR

#define HIGHBD_SMOOTH_H_NXM_WIDE(W, H)                                                            \
    void svt_aom_highbd_smooth_h_predictor_##W##x##H##_neon(                                      \
        uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd) { \
        (void)bd;                                                                                 \
        highbd_smooth_h_##W##xh_neon(dst, y_stride, above, left, H);                              \
    }

HIGHBD_SMOOTH_H_NXM_WIDE(16, 4)
HIGHBD_SMOOTH_H_NXM_WIDE(16, 8)
HIGHBD_SMOOTH_H_NXM_WIDE(16, 16)
HIGHBD_SMOOTH_H_NXM_WIDE(16, 32)
HIGHBD_SMOOTH_H_NXM_WIDE(16, 64)
HIGHBD_SMOOTH_H_NXM_WIDE(32, 8)
HIGHBD_SMOOTH_H_NXM_WIDE(32, 16)
HIGHBD_SMOOTH_H_NXM_WIDE(32, 32)
HIGHBD_SMOOTH_H_NXM_WIDE(32, 64)
HIGHBD_SMOOTH_H_NXM_WIDE(64, 16)
HIGHBD_SMOOTH_H_NXM_WIDE(64, 32)
HIGHBD_SMOOTH_H_NXM_WIDE(64, 64)

#undef HIGHBD_SMOOTH_H_NXM_WIDE

static INLINE void highbd_smooth_4xh_neon(uint16_t *dst, ptrdiff_t stride, const uint16_t *const top_row,
                                          const uint16_t *const left_column, const int height) {
    const uint16_t        top_right   = top_row[3];
    const uint16_t        bottom_left = left_column[height - 1];
    const uint16_t *const weights_y   = sm_weight_arrays_u16 + height - 4;

    const uint16x4_t top_v            = vld1_u16(top_row);
    const uint16x4_t bottom_left_v    = vdup_n_u16(bottom_left);
    const uint16x4_t weights_x_v      = vld1_u16(sm_weight_arrays_u16);
    const uint16x4_t scaled_weights_x = negate_s8(weights_x_v);
    const uint32x4_t weighted_tr      = vmull_n_u16(scaled_weights_x, top_right);

    for (int y = 0; y < height; ++y) {
        // Each variable in the running summation is named for the last item to be
        // accumulated.
        const uint32x4_t weighted_top  = vmlal_n_u16(weighted_tr, top_v, weights_y[y]);
        const uint32x4_t weighted_left = vmlal_n_u16(weighted_top, weights_x_v, left_column[y]);
        const uint32x4_t weighted_bl   = vmlal_n_u16(weighted_left, bottom_left_v, 256 - weights_y[y]);

        const uint16x4_t pred = vrshrn_n_u32(weighted_bl, SMOOTH_WEIGHT_LOG2_SCALE + 1);
        vst1_u16(dst, pred);
        dst += stride;
    }
}

// Common code between 8xH and [16|32|64]xH.
static INLINE void highbd_calculate_pred8(uint16_t *dst, const uint32x4_t weighted_corners_low,
                                          const uint32x4_t weighted_corners_high, const uint16x8_t top_vals,
                                          const uint16x8_t weights_x, const uint16_t left_y, const uint16_t weight_y) {
    // Each variable in the running summation is named for the last item to be
    // accumulated.
    const uint32x4_t weighted_top_low   = vmlal_n_u16(weighted_corners_low, vget_low_u16(top_vals), weight_y);
    const uint32x4_t weighted_edges_low = vmlal_n_u16(weighted_top_low, vget_low_u16(weights_x), left_y);

    const uint16x4_t pred_low = vshrn_n_u32(weighted_edges_low, SMOOTH_WEIGHT_LOG2_SCALE + 1);
    vst1_u16(dst, pred_low);

    const uint32x4_t weighted_top_high   = vmlal_n_u16(weighted_corners_high, vget_high_u16(top_vals), weight_y);
    const uint32x4_t weighted_edges_high = vmlal_n_u16(weighted_top_high, vget_high_u16(weights_x), left_y);

    const uint16x4_t pred_high = vshrn_n_u32(weighted_edges_high, SMOOTH_WEIGHT_LOG2_SCALE + 1);
    vst1_u16(dst + 4, pred_high);
}

static INLINE void highbd_smooth_8xh_neon(uint16_t *dst, ptrdiff_t stride, const uint16_t *const top_row,
                                          const uint16_t *const left_column, const int height) {
    const uint16_t        top_right   = top_row[7];
    const uint16_t        bottom_left = left_column[height - 1];
    const uint16_t *const weights_y   = sm_weight_arrays_u16 + height - 4;

    const uint16x8_t top_vals         = vld1q_u16(top_row);
    const uint16x4_t bottom_left_v    = vdup_n_u16(bottom_left);
    const uint16x8_t weights_x        = vld1q_u16(sm_weight_arrays_u16 + 4);
    const uint32x4_t offset           = vdupq_n_u32(1 << SMOOTH_WEIGHT_LOG2_SCALE);
    const uint32x4_t weighted_tr_low  = vmull_n_u16(negate_s8(vget_low_u16(weights_x)), top_right);
    const uint32x4_t weighted_tr_high = vmull_n_u16(negate_s8(vget_high_u16(weights_x)), top_right);

    for (int y = 0; y < height; ++y) {
        const uint32x4_t weighted_bl           = vmlal_n_u16(offset, bottom_left_v, 256 - weights_y[y]);
        const uint32x4_t weighted_corners_low  = vaddq_u32(weighted_bl, weighted_tr_low);
        const uint32x4_t weighted_corners_high = vaddq_u32(weighted_bl, weighted_tr_high);
        highbd_calculate_pred8(
            dst, weighted_corners_low, weighted_corners_high, top_vals, weights_x, left_column[y], weights_y[y]);
        dst += stride;
    }
}

#define HIGHBD_SMOOTH_NXM(W, H)                                                                   \
    void svt_aom_highbd_smooth_predictor_##W##x##H##_neon(                                        \
        uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd) { \
        (void)bd;                                                                                 \
        highbd_smooth_##W##xh_neon(dst, y_stride, above, left, H);                                \
    }

HIGHBD_SMOOTH_NXM(4, 4)
HIGHBD_SMOOTH_NXM(4, 8)
HIGHBD_SMOOTH_NXM(8, 4)
HIGHBD_SMOOTH_NXM(8, 8)
HIGHBD_SMOOTH_NXM(4, 16)
HIGHBD_SMOOTH_NXM(8, 16)
HIGHBD_SMOOTH_NXM(8, 32)

#undef HIGHBD_SMOOTH_NXM

// For width 16 and above.
#define HIGHBD_SMOOTH_PREDICTOR(W)                                                                    \
    static INLINE void highbd_smooth_##W##xh_neon(uint16_t             *dst,                          \
                                                  ptrdiff_t             stride,                       \
                                                  const uint16_t *const top_row,                      \
                                                  const uint16_t *const left_column,                  \
                                                  const int             height) {                                 \
        const uint16_t        top_right   = top_row[(W)-1];                                           \
        const uint16_t        bottom_left = left_column[height - 1];                                  \
        const uint16_t *const weights_y   = sm_weight_arrays_u16 + height - 4;                        \
                                                                                                      \
        /* Precompute weighted values that don't vary with |y|. */                                    \
        uint32x4_t       weighted_tr_low[(W) >> 3];                                                   \
        uint32x4_t       weighted_tr_high[(W) >> 3];                                                  \
        const uint32x4_t offset = vdupq_n_u32(1 << SMOOTH_WEIGHT_LOG2_SCALE);                         \
        for (int i = 0; i < (W) >> 3; ++i) {                                                          \
            const int        x              = i << 3;                                                 \
            const uint16x4_t weights_x_low  = vld1_u16(sm_weight_arrays_u16 + (W)-4 + x);             \
            weighted_tr_low[i]              = vmull_n_u16(negate_s8(weights_x_low), top_right);       \
            const uint16x4_t weights_x_high = vld1_u16(sm_weight_arrays_u16 + (W) + x);               \
            weighted_tr_high[i]             = vmull_n_u16(negate_s8(weights_x_high), top_right);      \
        }                                                                                             \
                                                                                                      \
        const uint16x4_t bottom_left_v = vdup_n_u16(bottom_left);                                     \
        for (int y = 0; y < height; ++y) {                                                            \
            const uint32x4_t weighted_bl = vmlal_n_u16(offset, bottom_left_v, 256 - weights_y[y]);    \
            uint16_t        *dst_x       = dst;                                                       \
            for (int i = 0; i < (W) >> 3; ++i) {                                                      \
                const int        x                     = i << 3;                                      \
                const uint16x8_t top_vals              = vld1q_u16(top_row + x);                      \
                const uint32x4_t weighted_corners_low  = vaddq_u32(weighted_bl, weighted_tr_low[i]);  \
                const uint32x4_t weighted_corners_high = vaddq_u32(weighted_bl, weighted_tr_high[i]); \
                /* Accumulate weighted edge values and store. */                                      \
                const uint16x8_t weights_x = vld1q_u16(sm_weight_arrays_u16 + (W)-4 + x);             \
                highbd_calculate_pred8(dst_x,                                                         \
                                       weighted_corners_low,                                          \
                                       weighted_corners_high,                                         \
                                       top_vals,                                                      \
                                       weights_x,                                                     \
                                       left_column[y],                                                \
                                       weights_y[y]);                                                 \
                dst_x += 8;                                                                           \
            }                                                                                         \
            dst += stride;                                                                            \
        }                                                                                             \
    }

HIGHBD_SMOOTH_PREDICTOR(16)
HIGHBD_SMOOTH_PREDICTOR(32)
HIGHBD_SMOOTH_PREDICTOR(64)

#undef HIGHBD_SMOOTH_PREDICTOR

#define HIGHBD_SMOOTH_NXM_WIDE(W, H)                                                              \
    void svt_aom_highbd_smooth_predictor_##W##x##H##_neon(                                        \
        uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd) { \
        (void)bd;                                                                                 \
        highbd_smooth_##W##xh_neon(dst, y_stride, above, left, H);                                \
    }

HIGHBD_SMOOTH_NXM_WIDE(16, 4)
HIGHBD_SMOOTH_NXM_WIDE(16, 8)
HIGHBD_SMOOTH_NXM_WIDE(16, 16)
HIGHBD_SMOOTH_NXM_WIDE(16, 32)
HIGHBD_SMOOTH_NXM_WIDE(16, 64)
HIGHBD_SMOOTH_NXM_WIDE(32, 8)
HIGHBD_SMOOTH_NXM_WIDE(32, 16)
HIGHBD_SMOOTH_NXM_WIDE(32, 32)
HIGHBD_SMOOTH_NXM_WIDE(32, 64)
HIGHBD_SMOOTH_NXM_WIDE(64, 16)
HIGHBD_SMOOTH_NXM_WIDE(64, 32)
HIGHBD_SMOOTH_NXM_WIDE(64, 64)

#undef HIGHBD_SMOOTH_NXM_WIDE

// -----------------------------------------------------------------------------
// V_PRED

static INLINE uint16x8x2_t load_uint16x8x2(uint16_t const *ptr) {
    uint16x8x2_t x;
    // Clang/gcc uses ldp here.
    x.val[0] = vld1q_u16(ptr);
    x.val[1] = vld1q_u16(ptr + 8);
    return x;
}

static INLINE void store_uint16x8x2(uint16_t *ptr, uint16x8x2_t x) {
    vst1q_u16(ptr, x.val[0]);
    vst1q_u16(ptr + 8, x.val[1]);
}

static INLINE void vertical4xh_neon(uint16_t *dst, ptrdiff_t stride, const uint16_t *const above, int height) {
    const uint16x4_t row = vld1_u16(above);
    int              y   = height;
    do {
        vst1_u16(dst, row);
        vst1_u16(dst + stride, row);
        dst += stride << 1;
        y -= 2;
    } while (y != 0);
}

static INLINE void vertical8xh_neon(uint16_t *dst, ptrdiff_t stride, const uint16_t *const above, int height) {
    const uint16x8_t row = vld1q_u16(above);
    int              y   = height;
    do {
        vst1q_u16(dst, row);
        vst1q_u16(dst + stride, row);
        dst += stride << 1;
        y -= 2;
    } while (y != 0);
}

static INLINE void vertical16xh_neon(uint16_t *dst, ptrdiff_t stride, const uint16_t *const above, int height) {
    const uint16x8x2_t row = load_uint16x8x2(above);
    int                y   = height;
    do {
        store_uint16x8x2(dst, row);
        store_uint16x8x2(dst + stride, row);
        dst += stride << 1;
        y -= 2;
    } while (y != 0);
}

static INLINE uint16x8x4_t load_uint16x8x4(uint16_t const *ptr) {
    uint16x8x4_t x;
    // Clang/gcc uses ldp here.
    x.val[0] = vld1q_u16(ptr);
    x.val[1] = vld1q_u16(ptr + 8);
    x.val[2] = vld1q_u16(ptr + 16);
    x.val[3] = vld1q_u16(ptr + 24);
    return x;
}

static INLINE void store_uint16x8x4(uint16_t *ptr, uint16x8x4_t x) {
    vst1q_u16(ptr, x.val[0]);
    vst1q_u16(ptr + 8, x.val[1]);
    vst1q_u16(ptr + 16, x.val[2]);
    vst1q_u16(ptr + 24, x.val[3]);
}

static INLINE void vertical32xh_neon(uint16_t *dst, ptrdiff_t stride, const uint16_t *const above, int height) {
    const uint16x8x4_t row = load_uint16x8x4(above);
    int                y   = height;
    do {
        store_uint16x8x4(dst, row);
        store_uint16x8x4(dst + stride, row);
        dst += stride << 1;
        y -= 2;
    } while (y != 0);
}

static INLINE void vertical64xh_neon(uint16_t *dst, ptrdiff_t stride, const uint16_t *const above, int height) {
    uint16_t          *dst32 = dst + 32;
    const uint16x8x4_t row   = load_uint16x8x4(above);
    const uint16x8x4_t row32 = load_uint16x8x4(above + 32);
    int                y     = height;
    do {
        store_uint16x8x4(dst, row);
        store_uint16x8x4(dst32, row32);
        store_uint16x8x4(dst + stride, row);
        store_uint16x8x4(dst32 + stride, row32);
        dst += stride << 1;
        dst32 += stride << 1;
        y -= 2;
    } while (y != 0);
}

#define HIGHBD_V_NXM(W, H)                                                                      \
    void svt_aom_highbd_v_predictor_##W##x##H##_neon(                                           \
        uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd) { \
        (void)left;                                                                             \
        (void)bd;                                                                               \
        vertical##W##xh_neon(dst, stride, above, H);                                            \
    }

HIGHBD_V_NXM(4, 4)
HIGHBD_V_NXM(4, 8)
HIGHBD_V_NXM(4, 16)

HIGHBD_V_NXM(8, 4)
HIGHBD_V_NXM(8, 8)
HIGHBD_V_NXM(8, 16)
HIGHBD_V_NXM(8, 32)

HIGHBD_V_NXM(16, 4)
HIGHBD_V_NXM(16, 8)
HIGHBD_V_NXM(16, 16)
HIGHBD_V_NXM(16, 32)
HIGHBD_V_NXM(16, 64)

HIGHBD_V_NXM(32, 8)
HIGHBD_V_NXM(32, 16)
HIGHBD_V_NXM(32, 32)
HIGHBD_V_NXM(32, 64)

HIGHBD_V_NXM(64, 16)
HIGHBD_V_NXM(64, 32)
HIGHBD_V_NXM(64, 64)
