/*
 * Copyright (c) 2024, Intel Corporation
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
#include "aom_dsp_rtcd.h"
#include "mem_neon.h"
#include "inv_transforms.h"
#include "transforms.h"
#include "transpose_neon.h"
#include "definitions.h"

// Constants are stored in pairs, where symmetrical constants in the
// cospi array are stored adjacent in memory, i.e.:
//   f(i,j) = (int)round(cos(PI*j/128) * (1<<(cos_bit_min+i)))
// and then in memory we store 4-tuples of constants together as:
//   f2(i,j) = [ f(i,j), f(i,64-j) ]
const int32_t av1_cospi_arr_s32_data[4][66] = {
    {
        1024, 0,   1024, 25,  1023, 50,  1021, 75,  1019, 100, 1016, 125, 1013, 150, 1009, 175, 1004,
        200,  999, 224,  993, 249,  987, 273,  980, 297,  972, 321,  964, 345,  955, 369,  946, 392,
        936,  415, 926,  438, 915,  460, 903,  483, 891,  505, 878,  526, 865,  548, 851,  569, 837,
        590,  822, 610,  807, 630,  792, 650,  775, 669,  759, 688,  742, 706,  724, 724,
    },
    {
        2048, 0,    2047, 50,   2046, 100,  2042, 151,  2038, 201,  2033, 251,  2026, 301,  2018, 350,  2009,
        400,  1998, 449,  1987, 498,  1974, 546,  1960, 595,  1945, 642,  1928, 690,  1911, 737,  1892, 784,
        1872, 830,  1851, 876,  1829, 921,  1806, 965,  1782, 1009, 1757, 1053, 1730, 1096, 1703, 1138, 1674,
        1179, 1645, 1220, 1615, 1260, 1583, 1299, 1551, 1338, 1517, 1375, 1483, 1412, 1448, 1448,
    },
    {
        4096, 0,    4095, 101,  4091, 201,  4085, 301,  4076, 401,  4065, 501,  4052, 601,  4036, 700,  4017,
        799,  3996, 897,  3973, 995,  3948, 1092, 3920, 1189, 3889, 1285, 3857, 1380, 3822, 1474, 3784, 1567,
        3745, 1660, 3703, 1751, 3659, 1842, 3612, 1931, 3564, 2019, 3513, 2106, 3461, 2191, 3406, 2276, 3349,
        2359, 3290, 2440, 3229, 2520, 3166, 2598, 3102, 2675, 3035, 2751, 2967, 2824, 2896, 2896,
    },
    {
        8192, 0,    8190, 201,  8182, 402,  8170, 603,  8153, 803,  8130, 1003, 8103, 1202, 8071, 1401, 8035,
        1598, 7993, 1795, 7946, 1990, 7895, 2185, 7839, 2378, 7779, 2570, 7713, 2760, 7643, 2948, 7568, 3135,
        7489, 3320, 7405, 3503, 7317, 3683, 7225, 3862, 7128, 4038, 7027, 4212, 6921, 4383, 6811, 4551, 6698,
        4717, 6580, 4880, 6458, 5040, 6333, 5197, 6203, 5351, 6070, 5501, 5933, 5649, 5793, 5793,
    }};

static inline const int32_t *cospi_arr_s32(int n) { return av1_cospi_arr_s32_data[n - cos_bit_min]; }

static inline void ud_adjust_input_and_stride(int ud_flip, int16_t **input, uint32_t *stride, int out_size) {
    if (ud_flip) {
        *input  = *input + (out_size - 1) * *stride;
        *stride = -*stride;
    }
}

#define LOAD_BUFFER_4XH(h, shift)                                                                                    \
    static AOM_FORCE_INLINE void load_buffer_4x##h##_(const int16_t *input, int32x4_t *in, int stride, int fliplr) { \
        if (fliplr) {                                                                                                \
            for (int i = 0; i < (h); ++i) {                                                                          \
                int16x4_t a = vld1_s16(input + i * stride);                                                          \
                a           = vrev64_s16(a);                                                                         \
                in[i]       = vshll_n_s16(a, shift);                                                                 \
            }                                                                                                        \
        } else {                                                                                                     \
            for (int i = 0; i < (h); ++i) {                                                                          \
                int16x4_t a = vld1_s16(input + i * stride);                                                          \
                in[i]       = vshll_n_s16(a, shift);                                                                 \
            }                                                                                                        \
        }                                                                                                            \
    }

LOAD_BUFFER_4XH(4, 2)
LOAD_BUFFER_4XH(8, 2)
LOAD_BUFFER_4XH(16, 2)
LOAD_BUFFER_4XH(32, 2)
LOAD_BUFFER_4XH(64, 0)

#define LOAD_BUFFER_WXH(w, h, shift)                                                    \
    static AOM_FORCE_INLINE void load_buffer_##w##x##h##_(                              \
        const int16_t *input, int32x4_t *in, int stride, int fliplr) {                  \
        assert(w >= 8);                                                                 \
        if (fliplr) {                                                                   \
            for (int i = 0; i < (h); ++i) {                                             \
                for (int j = 0; j < (w) / 8; ++j) {                                     \
                    int16x8_t a                = vld1q_s16(input + i * stride + j * 8); \
                    a                          = vrev64q_s16(a);                        \
                    int j2                     = (w) / 8 - j - 1;                       \
                    in[i + (h) * (2 * j2 + 0)] = vshll_n_s16(vget_high_s16(a), shift);  \
                    in[i + (h) * (2 * j2 + 1)] = vshll_n_s16(vget_low_s16(a), shift);   \
                }                                                                       \
            }                                                                           \
        } else {                                                                        \
            for (int i = 0; i < (h); ++i) {                                             \
                for (int j = 0; j < (w) / 8; ++j) {                                     \
                    int16x8_t a               = vld1q_s16(input + i * stride + j * 8);  \
                    in[i + (h) * (2 * j + 0)] = vshll_n_s16(vget_low_s16(a), shift);    \
                    in[i + (h) * (2 * j + 1)] = vshll_n_s16(vget_high_s16(a), shift);   \
                }                                                                       \
            }                                                                           \
        }                                                                               \
    }

LOAD_BUFFER_WXH(8, 8, 2)
LOAD_BUFFER_WXH(16, 16, 2)
LOAD_BUFFER_WXH(16, 64, 0)
LOAD_BUFFER_WXH(32, 64, 0)
LOAD_BUFFER_WXH(64, 16, 2)
LOAD_BUFFER_WXH(64, 32, 2)

#define STORE_BUFFER_WXH(w, h)                                                                           \
    static AOM_FORCE_INLINE void store_buffer_##w##x##h(const int32x4_t *in, int32_t *out, int stride) { \
        for (int i = 0; i < (w); ++i) {                                                                  \
            for (int j = 0; j < (h) / 4; ++j) { vst1q_s32(&out[i * stride + j * 4], in[i + j * (w)]); }  \
        }                                                                                                \
    }

STORE_BUFFER_WXH(4, 4)
STORE_BUFFER_WXH(8, 8)
STORE_BUFFER_WXH(16, 16)

static AOM_FORCE_INLINE void highbd_fdct4_x4_neon(const int32x4_t *in, int32x4_t *out, int bit) {
    const int32_t *const cospi      = cospi_arr_s32(bit);
    const int32x4_t      cospi32    = vdupq_n_s32(cospi[2 * 32]);
    const int32x2_t      cospi16_48 = vld1_s32(&cospi[2 * 16]);

    const int32x4_t a0 = vaddq_s32(in[0], in[3]);
    const int32x4_t a1 = vsubq_s32(in[0], in[3]);
    const int32x4_t a2 = vaddq_s32(in[1], in[2]);
    const int32x4_t a3 = vsubq_s32(in[1], in[2]);

    const int32x4_t b0 = vmulq_s32(a0, cospi32);
    const int32x4_t b1 = vmulq_lane_s32(a1, cospi16_48, 1);
    const int32x4_t b2 = vmulq_s32(a2, cospi32);
    const int32x4_t b3 = vmulq_lane_s32(a3, cospi16_48, 1);

    const int32x4_t c0 = vaddq_s32(b0, b2);
    const int32x4_t c1 = vsubq_s32(b0, b2);
    const int32x4_t c2 = vmlaq_lane_s32(b3, a1, cospi16_48, 0);
    const int32x4_t c3 = vmlsq_lane_s32(b1, a3, cospi16_48, 0);

    const int32x4_t v_bit = vdupq_n_s32(-bit);
    const int32x4_t d0    = vrshlq_s32(c0, v_bit);
    const int32x4_t d1    = vrshlq_s32(c1, v_bit);
    const int32x4_t d2    = vrshlq_s32(c2, v_bit);
    const int32x4_t d3    = vrshlq_s32(c3, v_bit);

    out[0] = d0;
    out[1] = d2;
    out[2] = d1;
    out[3] = d3;
}

static AOM_FORCE_INLINE void highbd_fadst4_x4_neon(const int32x4_t *in, int32x4_t *out, int bit) {
    const int32x4_t sinpi = vld1q_s32(sinpi_arr(bit) + 1);

    const int32x4_t a0 = vaddq_s32(in[0], in[1]);
    const int32x4_t a1 = vmulq_lane_s32(in[0], vget_low_s32(sinpi), 0);
    const int32x4_t a2 = vmulq_lane_s32(in[0], vget_high_s32(sinpi), 1);
    const int32x4_t a3 = vmulq_lane_s32(in[2], vget_high_s32(sinpi), 0);

    const int32x4_t b0 = vmlaq_lane_s32(a1, in[1], vget_low_s32(sinpi), 1);
    const int32x4_t b1 = vmlsq_lane_s32(a2, in[1], vget_low_s32(sinpi), 0);
    const int32x4_t b2 = vsubq_s32(a0, in[3]);

    const int32x4_t c0 = vmlaq_lane_s32(b0, in[3], vget_high_s32(sinpi), 1);
    const int32x4_t c1 = vmlaq_lane_s32(b1, in[3], vget_low_s32(sinpi), 1);
    const int32x4_t c2 = vmulq_lane_s32(b2, vget_high_s32(sinpi), 0);

    const int32x4_t d0 = vaddq_s32(c0, a3);
    const int32x4_t d1 = vsubq_s32(c1, a3);
    const int32x4_t d2 = vsubq_s32(c1, c0);

    const int32x4_t e0 = vaddq_s32(d2, a3);

    const int32x4_t v_bit = vdupq_n_s32(-bit);
    out[0]                = vrshlq_s32(d0, v_bit);
    out[1]                = vrshlq_s32(c2, v_bit);
    out[2]                = vrshlq_s32(d1, v_bit);
    out[3]                = vrshlq_s32(e0, v_bit);
}

static AOM_FORCE_INLINE void highbd_fidentity4_x4_neon(const int32x4_t *in, int32x4_t *out, int bit) {
    (void)bit;
    int32x4_t fact = vdupq_n_s32(new_sqrt2);

    for (int i = 0; i < 4; i++) {
        const int32x4_t a_low = vmulq_s32(in[i], fact);
        out[i]                = vrshrq_n_s32(a_low, new_sqrt2_bits);
    }
}

void svt_av1_fwd_txfm2d_4x4_neon(int16_t *input, int32_t *output, uint32_t input_stride, TxType tx_type, uint8_t bd) {
    (void)bd;

    int ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);
    ud_adjust_input_and_stride(ud_flip, &input, &input_stride, 4);

    // Workspace for column/row-wise transforms.
    int32x4_t buf[4];

    switch (tx_type) {
    case DCT_DCT:
        load_buffer_4x4_(input, buf, input_stride, 0);
        highbd_fdct4_x4_neon(buf, buf, fwd_cos_bit_col[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        highbd_fdct4_x4_neon(buf, buf, fwd_cos_bit_row[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        store_buffer_4x4(buf, output, /*stride=*/4);
        break;
    case ADST_DCT:
        load_buffer_4x4_(input, buf, input_stride, 0);
        highbd_fadst4_x4_neon(buf, buf, fwd_cos_bit_col[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        highbd_fdct4_x4_neon(buf, buf, fwd_cos_bit_row[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        store_buffer_4x4(buf, output, /*stride=*/4);
        break;
    case DCT_ADST:
        load_buffer_4x4_(input, buf, input_stride, 0);
        highbd_fdct4_x4_neon(buf, buf, fwd_cos_bit_col[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        highbd_fadst4_x4_neon(buf, buf, fwd_cos_bit_row[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        store_buffer_4x4(buf, output, /*stride=*/4);
        break;
    case ADST_ADST:
        load_buffer_4x4_(input, buf, input_stride, 0);
        highbd_fadst4_x4_neon(buf, buf, fwd_cos_bit_col[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        highbd_fadst4_x4_neon(buf, buf, fwd_cos_bit_row[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        store_buffer_4x4(buf, output, /*stride=*/4);
        break;
    case FLIPADST_DCT:
        load_buffer_4x4_(input, buf, input_stride, 0);
        highbd_fadst4_x4_neon(buf, buf, fwd_cos_bit_col[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        highbd_fdct4_x4_neon(buf, buf, fwd_cos_bit_row[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        store_buffer_4x4(buf, output, /*stride=*/4);
        break;
    case DCT_FLIPADST:
        load_buffer_4x4_(input, buf, input_stride, 1);
        highbd_fdct4_x4_neon(buf, buf, fwd_cos_bit_col[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        highbd_fadst4_x4_neon(buf, buf, fwd_cos_bit_row[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        store_buffer_4x4(buf, output, /*stride=*/4);
        break;
    case FLIPADST_FLIPADST:
        load_buffer_4x4_(input, buf, input_stride, 1);
        highbd_fadst4_x4_neon(buf, buf, fwd_cos_bit_col[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        highbd_fadst4_x4_neon(buf, buf, fwd_cos_bit_row[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        store_buffer_4x4(buf, output, /*stride=*/4);
        break;
    case ADST_FLIPADST:
        load_buffer_4x4_(input, buf, input_stride, 1);
        highbd_fadst4_x4_neon(buf, buf, fwd_cos_bit_col[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        highbd_fadst4_x4_neon(buf, buf, fwd_cos_bit_row[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        store_buffer_4x4(buf, output, /*stride=*/4);
        break;
    case FLIPADST_ADST:
        load_buffer_4x4_(input, buf, input_stride, 0);
        highbd_fadst4_x4_neon(buf, buf, fwd_cos_bit_col[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        highbd_fadst4_x4_neon(buf, buf, fwd_cos_bit_row[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        store_buffer_4x4(buf, output, /*stride=*/4);
        break;
    case IDTX:
        load_buffer_4x4_(input, buf, input_stride, 0);
        highbd_fidentity4_x4_neon(buf, buf, fwd_cos_bit_col[0][0]);
        highbd_fidentity4_x4_neon(buf, buf, fwd_cos_bit_row[0][0]);
        store_buffer_4x4(buf, output, /*stride=*/4);
        break;
    case V_DCT:
        load_buffer_4x4_(input, buf, input_stride, 0);
        highbd_fdct4_x4_neon(buf, buf, fwd_cos_bit_col[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        highbd_fidentity4_x4_neon(buf, buf, fwd_cos_bit_row[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        store_buffer_4x4(buf, output, /*stride=*/4);
        break;
    case H_DCT:
        load_buffer_4x4_(input, buf, input_stride, 0);
        highbd_fidentity4_x4_neon(buf, buf, fwd_cos_bit_row[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        highbd_fdct4_x4_neon(buf, buf, fwd_cos_bit_col[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        store_buffer_4x4(buf, output, /*stride=*/4);
        break;
    case V_ADST:
        load_buffer_4x4_(input, buf, input_stride, 0);
        highbd_fadst4_x4_neon(buf, buf, fwd_cos_bit_col[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        highbd_fidentity4_x4_neon(buf, buf, fwd_cos_bit_row[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        store_buffer_4x4(buf, output, /*stride=*/4);
        break;
    case H_ADST:
        load_buffer_4x4_(input, buf, input_stride, 0);
        highbd_fidentity4_x4_neon(buf, buf, fwd_cos_bit_row[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        highbd_fadst4_x4_neon(buf, buf, fwd_cos_bit_col[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        store_buffer_4x4(buf, output, /*stride=*/4);
        break;
    case V_FLIPADST:
        load_buffer_4x4_(input, buf, input_stride, 0);
        highbd_fadst4_x4_neon(buf, buf, fwd_cos_bit_row[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        highbd_fidentity4_x4_neon(buf, buf, fwd_cos_bit_row[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        store_buffer_4x4(buf, output, /*stride=*/4);
        break;
    case H_FLIPADST:
        load_buffer_4x4_(input, buf, input_stride, 1);
        highbd_fidentity4_x4_neon(buf, buf, fwd_cos_bit_row[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        highbd_fadst4_x4_neon(buf, buf, fwd_cos_bit_row[0][0]);
        transpose_arrays_s32_4x4(buf, buf);
        store_buffer_4x4(buf, output, /*stride=*/4);
        break;
    default: assert(0);
    }
}

#define SHIFT_LOOP_HELPER(name, type, intrinsic, arg)              \
    static inline void name(const type *in, type *out, int size) { \
        int i = 0;                                                 \
        do { out[i] = intrinsic(in[i], arg); } while (++i < size); \
    }

SHIFT_LOOP_HELPER(shift_right_2_round_s32_x4, int32x4_t, vrshrq_n_s32, 2)
SHIFT_LOOP_HELPER(shift_right_4_round_s32_x4, int32x4_t, vrshrq_n_s32, 4)

// Addition instructions have slightly better performance compared to shift
// instructions on some micro-architectures, so use these for shifts by one.

SHIFT_LOOP_HELPER(shift_right_1_round_s32_x4, int32x4_t, vrhaddq_s32, vdupq_n_s32(0))

// A note on butterfly helper naming:
//
// butterfly_[weight_indices]_neon
// e.g. butterfly_0312_neon
//                ^ Weights are applied as indices 0, 3, 2, 1
//                  (see more detail below)
//
// Weight indices are treated as an index into the 4-tuple of the weight
// itself, plus related and negated constants: w=(w0, 1-w0, -w0, w0-1).
// This is then represented in the helper naming by referring to the lane index
// in the loaded tuple that each multiply is performed with:
//
//         in0   in1
//      /------------
// out0 |  w[0]  w[1]   ==>  out0 = in0 * w[0] + in1 * w[1]
// out1 |  w[2]  w[3]   ==>  out1 = in0 * w[2] + in1 * w[3]
//
// So for indices 0321 from the earlier example, we end up with:
//
//          in0       in1
//      /------------------
// out0 | (lane 0) (lane 3)   ==>  out0 = in0 *  w0 + in1 * (w0-1)
// out1 | (lane 2) (lane 1)   ==>  out1 = in0 * -w0 + in1 * (1-w0)

#define butterfly_half_neon(wvec, lane0, lane1, in0, in1, out, v_bit)                \
    do {                                                                             \
        int32x2x2_t wvecs = {{wvec, vneg_s32(wvec)}};                                \
        int32x4_t   x     = vmulq_lane_s32(in0, wvecs.val[lane0 / 2], lane0 % 2);    \
        x                 = vmlaq_lane_s32(x, in1, wvecs.val[lane1 / 2], lane1 % 2); \
        *out              = vrshlq_s32(x, v_bit);                                    \
    } while (false)

static AOM_FORCE_INLINE void butterfly_0112_neon(const int32_t *cospi, const int widx0, const int32x4_t n0,
                                                 const int32x4_t n1, int32x4_t *out0, int32x4_t *out1,
                                                 const int32x4_t v_bit) {
    int32x2_t w01 = vld1_s32(cospi + 2 * widx0);
    butterfly_half_neon(w01, 0, 1, n0, n1, out0, v_bit);
    butterfly_half_neon(w01, 1, 2, n0, n1, out1, v_bit);
}

static AOM_FORCE_INLINE void butterfly_2312_neon(const int32_t *cospi, const int widx0, const int32x4_t n0,
                                                 const int32x4_t n1, int32x4_t *out0, int32x4_t *out1,
                                                 const int32x4_t v_bit) {
    int32x2_t w01 = vld1_s32(cospi + 2 * widx0);
    butterfly_half_neon(w01, 2, 3, n0, n1, out0, v_bit);
    butterfly_half_neon(w01, 1, 2, n0, n1, out1, v_bit);
}

static AOM_FORCE_INLINE void butterfly_0332_neon(const int32_t *cospi, const int widx0, const int32x4_t n0,
                                                 const int32x4_t n1, int32x4_t *out0, int32x4_t *out1,
                                                 const int32x4_t v_bit) {
    int32x2_t w01 = vld1_s32(cospi + 2 * widx0);
    butterfly_half_neon(w01, 0, 3, n0, n1, out0, v_bit);
    butterfly_half_neon(w01, 3, 2, n0, n1, out1, v_bit);
}

static AOM_FORCE_INLINE void butterfly_0130_neon(const int32_t *cospi, const int widx0, const int32x4_t n0,
                                                 const int32x4_t n1, int32x4_t *out0, int32x4_t *out1,
                                                 const int32x4_t v_bit) {
    int32x2_t w01 = vld1_s32(cospi + 2 * widx0);
    butterfly_half_neon(w01, 0, 1, n0, n1, out0, v_bit);
    butterfly_half_neon(w01, 3, 0, n0, n1, out1, v_bit);
}

static AOM_FORCE_INLINE void butterfly_cospi32_0002_neon(const int32_t *cospi, const int32x4_t n0, const int32x4_t n1,
                                                         int32x4_t *out0, int32x4_t *out1, const int32x4_t v_bit) {
    int32x2_t w01 = vld1_s32(cospi + 2 * 32);
    butterfly_half_neon(w01, 0, 0, n0, n1, out0, v_bit);
    butterfly_half_neon(w01, 0, 2, n0, n1, out1, v_bit);
}

static AOM_FORCE_INLINE void butterfly_cospi32_0222_neon(const int32_t *cospi, const int32x4_t n0, const int32x4_t n1,
                                                         int32x4_t *out0, int32x4_t *out1, const int32x4_t v_bit) {
    int32x2_t w01 = vld1_s32(cospi + 2 * 32);
    butterfly_half_neon(w01, 0, 2, n0, n1, out0, v_bit);
    butterfly_half_neon(w01, 2, 2, n0, n1, out1, v_bit);
}

// Butterfly pre-processing:
// e.g. n=4:
//   out[0] = in[0] + in[3]
//   out[1] = in[1] + in[2]
//   out[2] = in[1] - in[2]
//   out[3] = in[0] - in[3]

static AOM_FORCE_INLINE void butterfly_dct_pre(const int32x4_t *input, int32x4_t *output, int n) {
    for (int i = 0; i < n / 2; ++i) { output[i] = vaddq_s32(input[i], input[n - i - 1]); }
    for (int i = 0; i < n / 2; ++i) { output[n / 2 + i] = vsubq_s32(input[n / 2 - i - 1], input[n / 2 + i]); }
}

// Butterfly post-processing:
// e.g. n=8:
//   out[0] = in0[0] + in1[3];
//   out[1] = in0[1] + in1[2];
//   out[2] = in0[1] - in1[2];
//   out[3] = in0[0] - in1[3];
//   out[4] = in0[7] - in1[4];
//   out[5] = in0[6] - in1[5];
//   out[6] = in0[6] + in1[5];
//   out[7] = in0[7] + in1[4];

static AOM_FORCE_INLINE void butterfly_dct_post(const int32x4_t *in0, const int32x4_t *in1, int32x4_t *output, int n) {
    for (int i = 0; i < n / 4; ++i) { output[i] = vaddq_s32(in0[i], in1[n / 2 - i - 1]); }
    for (int i = 0; i < n / 4; ++i) { output[n / 4 + i] = vsubq_s32(in0[n / 4 - i - 1], in1[n / 4 + i]); }
    for (int i = 0; i < n / 4; ++i) { output[n / 2 + i] = vsubq_s32(in0[n - i - 1], in1[n / 2 + i]); }
    for (int i = 0; i < n / 4; ++i) {
        output[(3 * n) / 4 + i] = vaddq_s32(in0[(3 * n) / 4 + i], in1[(3 * n) / 4 - i - 1]);
    }
}

static AOM_FORCE_INLINE void highbd_fdct8_x4_neon(const int32x4_t *in, int32x4_t *out, int bit) {
    const int32_t *const cospi = cospi_arr_s32(bit);
    const int32x4_t      v_bit = vdupq_n_s32(-bit);

    // stage 1
    int32x4_t a[8];
    butterfly_dct_pre(in, a, 8);

    // stage 2
    int32x4_t b[8];
    butterfly_dct_pre(a, b, 4);
    butterfly_0130_neon(cospi, 32, a[5], a[6], &b[6], &b[5], v_bit);

    // stage 3
    int32x4_t c[8];
    butterfly_0130_neon(cospi, 32, b[1], b[0], &c[0], &c[1], v_bit);
    butterfly_0112_neon(cospi, 16, b[3], b[2], &c[2], &c[3], v_bit);
    butterfly_dct_post(a + 4, b + 4, c + 4, 4);

    // stage 4-5
    butterfly_0112_neon(cospi, 8, c[7], c[4], &out[1], &out[7], v_bit);
    butterfly_0130_neon(cospi, 24, c[5], c[6], &out[5], &out[3], v_bit);

    out[0] = c[0];
    out[2] = c[2];
    out[4] = c[1];
    out[6] = c[3];
}

static AOM_FORCE_INLINE void highbd_fadst8_x4_neon(const int32x4_t *in, int32x4_t *out, int bit) {
    const int32_t *const cospi = cospi_arr_s32(bit);
    const int32x4_t      v_bit = vdupq_n_s32(-bit);

    int32x4_t u0, u1, u2, u3, u4, u5, u6, u7;
    int32x4_t v0, v1, v2, v3, v4, v5, v6, v7;

    // stage 0-1
    u0 = in[0];
    u1 = in[7];
    u2 = in[3];
    u3 = in[4];
    u4 = in[1];
    u5 = in[6];
    u6 = in[2];
    u7 = in[5];

    // stage 2
    v0 = u0;
    v1 = u1;
    butterfly_cospi32_0222_neon(cospi, u3, u2, &v2, &v3, v_bit);
    v4 = u4;
    v5 = u5;
    butterfly_cospi32_0002_neon(cospi, u6, u7, &v7, &v6, v_bit);

    // stage 3
    u0 = vaddq_s32(v0, v2);
    u1 = vsubq_s32(v3, v1);
    u2 = vsubq_s32(v0, v2);
    u3 = vaddq_s32(v1, v3);
    u4 = vsubq_s32(v6, v4);
    u5 = vaddq_s32(v5, v7);
    u6 = vaddq_s32(v4, v6);
    u7 = vsubq_s32(v5, v7);

    // stage 4
    v0 = u0;
    v1 = u1;
    v2 = u2;
    v3 = u3;

    butterfly_0112_neon(cospi, 16, u4, u5, &v4, &v5, v_bit);
    butterfly_0112_neon(cospi, 16, u7, u6, &v6, &v7, v_bit);

    // stage 5
    u0 = vaddq_s32(v0, v4);
    u1 = vaddq_s32(v1, v5);
    u2 = vaddq_s32(v2, v6);
    u3 = vsubq_s32(v7, v3);
    u4 = vsubq_s32(v0, v4);
    u5 = vsubq_s32(v1, v5);
    u6 = vsubq_s32(v2, v6);
    u7 = vaddq_s32(v3, v7);

    // stage 6
    butterfly_0112_neon(cospi, 4, u0, u1, &v0, &v1, v_bit);
    butterfly_0112_neon(cospi, 20, u2, u3, &v2, &v3, v_bit);
    butterfly_0130_neon(cospi, 28, u5, u4, &v4, &v5, v_bit);
    butterfly_0112_neon(cospi, 12, u6, u7, &v7, &v6, v_bit);

    // stage 7
    out[0] = v1;
    out[1] = v6;
    out[2] = v3;
    out[3] = v4;
    out[4] = v5;
    out[5] = v2;
    out[6] = v7;
    out[7] = v0;
}

static AOM_FORCE_INLINE void highbd_fidentity8_x4_neon(const int32x4_t *in, int32x4_t *out, int bit) {
    (void)bit;
    out[0] = vshlq_n_s32(in[0], 1);
    out[1] = vshlq_n_s32(in[1], 1);
    out[2] = vshlq_n_s32(in[2], 1);
    out[3] = vshlq_n_s32(in[3], 1);
    out[4] = vshlq_n_s32(in[4], 1);
    out[5] = vshlq_n_s32(in[5], 1);
    out[6] = vshlq_n_s32(in[6], 1);
    out[7] = vshlq_n_s32(in[7], 1);
}

static AOM_FORCE_INLINE void highbd_fdct8_xn_neon(const int32x4_t *in, int32x4_t *out, int bit, int howmany) {
    const int stride = 8;
    int       i      = 0;
    do { highbd_fdct8_x4_neon(in + i * stride, out + i * stride, bit); } while (++i < howmany);
}

static AOM_FORCE_INLINE void highbd_fadst8_xn_neon(const int32x4_t *in, int32x4_t *out, int bit, int howmany) {
    const int stride = 8;
    int       i      = 0;
    do { highbd_fadst8_x4_neon(in + i * stride, out + i * stride, bit); } while (++i < howmany);
}

static AOM_FORCE_INLINE void highbd_fidentity8_xn_neon(const int32x4_t *in, int32x4_t *out, int bit, int howmany) {
    (void)bit;
    const int stride = 8;
    int       i      = 0;
    do { highbd_fidentity8_x4_neon(in + i * stride, out + i * stride, bit); } while (++i < howmany);
}

void svt_av1_fwd_txfm2d_8x8_neon(int16_t *input, int32_t *output, uint32_t stride, TxType tx_type, uint8_t bd) {
    (void)bd;

    int ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);
    ud_adjust_input_and_stride(ud_flip, &input, &stride, 8);

    // Workspaces for column/row-wise transforms.
    int32x4_t buf0[16], buf1[16];

    switch (tx_type) {
    case DCT_DCT:
        load_buffer_8x8_(input, buf0, stride, 0);
        highbd_fdct8_xn_neon(buf0, buf0, fwd_cos_bit_col[1][1], 2);
        shift_right_1_round_s32_x4(buf0, buf0, 16);
        transpose_arrays_s32_8x8(buf0, buf1);
        highbd_fdct8_xn_neon(buf1, buf1, fwd_cos_bit_row[1][1], 2);
        transpose_arrays_s32_8x8(buf1, buf0);
        store_buffer_8x8(buf0, output, /*stride=*/8);
        break;
    case ADST_DCT:
        load_buffer_8x8_(input, buf0, stride, 0);
        highbd_fadst8_xn_neon(buf0, buf0, fwd_cos_bit_col[1][1], 2);
        shift_right_1_round_s32_x4(buf0, buf0, 16);
        transpose_arrays_s32_8x8(buf0, buf1);
        highbd_fdct8_xn_neon(buf1, buf1, fwd_cos_bit_row[1][1], 2);
        transpose_arrays_s32_8x8(buf1, buf0);
        store_buffer_8x8(buf0, output, /*stride=*/8);
        break;
    case DCT_ADST:
        load_buffer_8x8_(input, buf0, stride, 0);
        highbd_fdct8_xn_neon(buf0, buf0, fwd_cos_bit_col[1][1], 2);
        shift_right_1_round_s32_x4(buf0, buf0, 16);
        transpose_arrays_s32_8x8(buf0, buf1);
        highbd_fadst8_xn_neon(buf1, buf1, fwd_cos_bit_row[1][1], 2);
        transpose_arrays_s32_8x8(buf1, buf0);
        store_buffer_8x8(buf0, output, /*stride=*/8);
        break;
    case ADST_ADST:
        load_buffer_8x8_(input, buf0, stride, 0);
        highbd_fadst8_xn_neon(buf0, buf0, fwd_cos_bit_col[1][1], 2);
        shift_right_1_round_s32_x4(buf0, buf0, 16);
        transpose_arrays_s32_8x8(buf0, buf1);
        highbd_fadst8_xn_neon(buf1, buf1, fwd_cos_bit_row[1][1], 2);
        transpose_arrays_s32_8x8(buf1, buf0);
        store_buffer_8x8(buf0, output, /*stride=*/8);
        break;
    case FLIPADST_DCT:
        load_buffer_8x8_(input, buf0, stride, 0);
        highbd_fadst8_xn_neon(buf0, buf0, fwd_cos_bit_col[1][1], 2);
        shift_right_1_round_s32_x4(buf0, buf0, 16);
        transpose_arrays_s32_8x8(buf0, buf1);
        highbd_fdct8_xn_neon(buf1, buf1, fwd_cos_bit_row[1][1], 2);
        transpose_arrays_s32_8x8(buf1, buf0);
        store_buffer_8x8(buf0, output, /*stride=*/8);
        break;
    case DCT_FLIPADST:
        load_buffer_8x8_(input, buf0, stride, 1);
        highbd_fdct8_xn_neon(buf0, buf0, fwd_cos_bit_col[1][1], 2);
        shift_right_1_round_s32_x4(buf0, buf0, 16);
        transpose_arrays_s32_8x8(buf0, buf1);
        highbd_fadst8_xn_neon(buf1, buf1, fwd_cos_bit_row[1][1], 2);
        transpose_arrays_s32_8x8(buf1, buf0);
        store_buffer_8x8(buf0, output, /*stride=*/8);
        break;
    case FLIPADST_FLIPADST:
        load_buffer_8x8_(input, buf0, stride, 1);
        highbd_fadst8_xn_neon(buf0, buf0, fwd_cos_bit_col[1][1], 2);
        shift_right_1_round_s32_x4(buf0, buf0, 16);
        transpose_arrays_s32_8x8(buf0, buf1);
        highbd_fadst8_xn_neon(buf1, buf1, fwd_cos_bit_row[1][1], 2);
        transpose_arrays_s32_8x8(buf1, buf0);
        store_buffer_8x8(buf0, output, /*stride=*/8);
        break;
    case ADST_FLIPADST:
        load_buffer_8x8_(input, buf0, stride, 1);
        highbd_fadst8_xn_neon(buf0, buf0, fwd_cos_bit_col[1][1], 2);
        shift_right_1_round_s32_x4(buf0, buf0, 16);
        transpose_arrays_s32_8x8(buf0, buf1);
        highbd_fadst8_xn_neon(buf1, buf1, fwd_cos_bit_row[1][1], 2);
        transpose_arrays_s32_8x8(buf1, buf0);
        store_buffer_8x8(buf0, output, /*stride=*/8);
        break;
    case FLIPADST_ADST:
        load_buffer_8x8_(input, buf0, stride, 0);
        highbd_fadst8_xn_neon(buf0, buf0, fwd_cos_bit_col[1][1], 2);
        shift_right_1_round_s32_x4(buf0, buf0, 16);
        transpose_arrays_s32_8x8(buf0, buf1);
        highbd_fadst8_xn_neon(buf1, buf1, fwd_cos_bit_row[1][1], 2);
        transpose_arrays_s32_8x8(buf1, buf0);
        store_buffer_8x8(buf0, output, /*stride=*/8);
        break;
    case IDTX:
        load_buffer_8x8_(input, buf0, stride, 0);
        highbd_fidentity8_xn_neon(buf0, buf0, fwd_cos_bit_col[1][1], 2);
        shift_right_1_round_s32_x4(buf0, buf0, 16);
        highbd_fidentity8_xn_neon(buf0, buf0, fwd_cos_bit_col[1][1], 2);
        store_buffer_8x8(buf0, output, /*stride=*/8);
        break;
    case V_DCT:
        load_buffer_8x8_(input, buf0, stride, 0);
        highbd_fdct8_xn_neon(buf0, buf0, fwd_cos_bit_col[1][1], 2);
        shift_right_1_round_s32_x4(buf0, buf0, 16);
        transpose_arrays_s32_8x8(buf0, buf1);
        highbd_fidentity8_xn_neon(buf1, buf1, fwd_cos_bit_col[1][1], 2);
        transpose_arrays_s32_8x8(buf1, buf0);
        store_buffer_8x8(buf0, output, /*stride=*/8);
        break;
    case H_DCT:
        load_buffer_8x8_(input, buf0, stride, 0);
        highbd_fidentity8_xn_neon(buf0, buf0, fwd_cos_bit_col[1][1], 2);
        shift_right_1_round_s32_x4(buf0, buf0, 16);
        transpose_arrays_s32_8x8(buf0, buf1);
        highbd_fdct8_xn_neon(buf1, buf1, fwd_cos_bit_col[1][1], 2);
        transpose_arrays_s32_8x8(buf1, buf0);
        store_buffer_8x8(buf0, output, /*stride=*/8);
        break;
    case V_ADST:
        load_buffer_8x8_(input, buf0, stride, 0);
        highbd_fadst8_xn_neon(buf0, buf0, fwd_cos_bit_col[1][1], 2);
        shift_right_1_round_s32_x4(buf0, buf0, 16);
        transpose_arrays_s32_8x8(buf0, buf1);
        highbd_fidentity8_xn_neon(buf1, buf1, fwd_cos_bit_col[1][1], 2);
        transpose_arrays_s32_8x8(buf1, buf0);
        store_buffer_8x8(buf0, output, /*stride=*/8);
        break;
    case H_ADST:
        load_buffer_8x8_(input, buf0, stride, 0);
        highbd_fidentity8_xn_neon(buf0, buf0, fwd_cos_bit_col[1][1], 2);
        shift_right_1_round_s32_x4(buf0, buf0, 16);
        transpose_arrays_s32_8x8(buf0, buf1);
        highbd_fadst8_xn_neon(buf1, buf1, fwd_cos_bit_col[1][1], 2);
        transpose_arrays_s32_8x8(buf1, buf0);
        store_buffer_8x8(buf0, output, /*stride=*/8);
        break;
    case V_FLIPADST:
        load_buffer_8x8_(input, buf0, stride, 0);
        highbd_fadst8_xn_neon(buf0, buf0, fwd_cos_bit_col[1][1], 2);
        shift_right_1_round_s32_x4(buf0, buf0, 16);
        transpose_arrays_s32_8x8(buf0, buf1);
        highbd_fidentity8_xn_neon(buf1, buf1, fwd_cos_bit_col[1][1], 2);
        transpose_arrays_s32_8x8(buf1, buf0);
        store_buffer_8x8(buf0, output, /*stride=*/8);
        break;
    case H_FLIPADST:
        load_buffer_8x8_(input, buf0, stride, 1);
        highbd_fidentity8_xn_neon(buf0, buf0, fwd_cos_bit_col[1][1], 2);
        shift_right_1_round_s32_x4(buf0, buf0, 16);
        transpose_arrays_s32_8x8(buf0, buf1);
        highbd_fadst8_xn_neon(buf1, buf1, fwd_cos_bit_col[1][1], 2);
        transpose_arrays_s32_8x8(buf1, buf0);
        store_buffer_8x8(buf0, output, /*stride=*/8);
        break;
    default: assert(0);
    }
}

static void highbd_fdct16_x4_neon(const int32x4_t *in, int32x4_t *out, int bit) {
    const int32_t *const cospi = cospi_arr_s32(bit);
    const int32x4_t      v_bit = vdupq_n_s32(-bit);

    int32x4_t u[16], v[16];

    // stage 1
    butterfly_dct_pre(in, u, 16);

    // stage 2
    butterfly_dct_pre(u, v, 8);
    v[8] = u[8];
    v[9] = u[9];
    butterfly_cospi32_0002_neon(cospi, u[13], u[10], &v[13], &v[10], v_bit);
    butterfly_cospi32_0002_neon(cospi, u[12], u[11], &v[12], &v[11], v_bit);
    v[14] = u[14];
    v[15] = u[15];

    // stage 3
    butterfly_dct_pre(v, u, 4);
    u[4] = v[4];
    butterfly_cospi32_0002_neon(cospi, v[6], v[5], &u[6], &u[5], v_bit);
    u[7] = v[7];
    butterfly_dct_post(v + 8, v + 8, u + 8, 8);

    // stage 4
    butterfly_cospi32_0002_neon(cospi, u[0], u[1], &v[0], &v[1], v_bit);
    butterfly_0112_neon(cospi, 16, u[3], u[2], &v[2], &v[3], v_bit);
    butterfly_dct_post(u + 4, u + 4, v + 4, 4);
    v[8] = u[8];
    butterfly_0112_neon(cospi, 16, u[14], u[9], &v[14], &v[9], v_bit);
    butterfly_2312_neon(cospi, 16, u[13], u[10], &v[10], &v[13], v_bit);
    v[11] = u[11];
    v[12] = u[12];
    v[15] = u[15];

    // stage 5
    u[0] = v[0];
    u[1] = v[1];
    u[2] = v[2];
    u[3] = v[3];
    butterfly_0112_neon(cospi, 8, v[7], v[4], &u[4], &u[7], v_bit);
    butterfly_0130_neon(cospi, 24, v[5], v[6], &u[5], &u[6], v_bit);
    butterfly_dct_post(v + 8, v + 8, u + 8, 4);
    butterfly_dct_post(v + 12, v + 12, u + 12, 4);

    // stage 6
    v[0] = u[0];
    v[1] = u[1];
    v[2] = u[2];
    v[3] = u[3];
    v[4] = u[4];
    v[5] = u[5];
    v[6] = u[6];
    v[7] = u[7];
    butterfly_0112_neon(cospi, 4, u[15], u[8], &v[8], &v[15], v_bit);
    butterfly_0130_neon(cospi, 28, u[9], u[14], &v[9], &v[14], v_bit);
    butterfly_0112_neon(cospi, 20, u[13], u[10], &v[10], &v[13], v_bit);
    butterfly_0130_neon(cospi, 12, u[11], u[12], &v[11], &v[12], v_bit);

    out[0]  = v[0];
    out[1]  = v[8];
    out[2]  = v[4];
    out[3]  = v[12];
    out[4]  = v[2];
    out[5]  = v[10];
    out[6]  = v[6];
    out[7]  = v[14];
    out[8]  = v[1];
    out[9]  = v[9];
    out[10] = v[5];
    out[11] = v[13];
    out[12] = v[3];
    out[13] = v[11];
    out[14] = v[7];
    out[15] = v[15];
}

static void highbd_fadst16_x4_neon(const int32x4_t *in, int32x4_t *out, int bit) {
    const int32_t *const cospi = cospi_arr_s32(bit);
    const int32x4_t      v_bit = vdupq_n_s32(-bit);

    int32x4_t u[16], v[16];

    // stage 0-1
    u[0]  = in[0];
    u[1]  = in[15];
    u[2]  = in[7];
    u[3]  = in[8];
    u[4]  = in[3];
    u[5]  = in[12];
    u[6]  = in[4];
    u[7]  = in[11];
    u[8]  = in[1];
    u[9]  = in[14];
    u[10] = in[6];
    u[11] = in[9];
    u[12] = in[2];
    u[13] = in[13];
    u[14] = in[5];
    u[15] = in[10];

    // stage 2
    v[0] = u[0];
    v[1] = u[1];
    butterfly_cospi32_0222_neon(cospi, u[3], u[2], &v[2], &v[3], v_bit);
    v[4] = u[4];
    v[5] = u[5];
    butterfly_cospi32_0002_neon(cospi, u[6], u[7], &v[7], &v[6], v_bit);
    v[8] = u[8];
    v[9] = u[9];
    butterfly_cospi32_0002_neon(cospi, u[10], u[11], &v[11], &v[10], v_bit);
    v[12] = u[12];
    v[13] = u[13];
    butterfly_cospi32_0222_neon(cospi, u[15], u[14], &v[14], &v[15], v_bit);

    // stage 3
    u[0]  = vaddq_s32(v[0], v[2]);
    u[1]  = vsubq_s32(v[3], v[1]);
    u[2]  = vsubq_s32(v[0], v[2]);
    u[3]  = vaddq_s32(v[1], v[3]);
    u[4]  = vsubq_s32(v[6], v[4]);
    u[5]  = vaddq_s32(v[5], v[7]);
    u[6]  = vaddq_s32(v[4], v[6]);
    u[7]  = vsubq_s32(v[5], v[7]);
    u[8]  = vsubq_s32(v[10], v[8]);
    u[9]  = vaddq_s32(v[9], v[11]);
    u[10] = vaddq_s32(v[8], v[10]);
    u[11] = vsubq_s32(v[9], v[11]);
    u[12] = vaddq_s32(v[12], v[14]);
    u[13] = vsubq_s32(v[15], v[13]);
    u[14] = vsubq_s32(v[12], v[14]);
    u[15] = vaddq_s32(v[13], v[15]);

    // stage 4
    v[0] = u[0];
    v[1] = u[1];
    v[2] = u[2];
    v[3] = u[3];
    butterfly_0112_neon(cospi, 16, u[4], u[5], &v[4], &v[5], v_bit);
    butterfly_0112_neon(cospi, 16, u[7], u[6], &v[6], &v[7], v_bit);

    v[8]  = u[8];
    v[9]  = u[9];
    v[10] = u[10];
    v[11] = u[11];

    butterfly_0112_neon(cospi, 16, u[12], u[13], &v[12], &v[13], v_bit);
    butterfly_0332_neon(cospi, 16, u[14], u[15], &v[15], &v[14], v_bit);

    // stage 5
    u[0]  = vaddq_s32(v[0], v[4]);
    u[1]  = vaddq_s32(v[1], v[5]);
    u[2]  = vaddq_s32(v[2], v[6]);
    u[3]  = vsubq_s32(v[7], v[3]);
    u[4]  = vsubq_s32(v[0], v[4]);
    u[5]  = vsubq_s32(v[1], v[5]);
    u[6]  = vsubq_s32(v[2], v[6]);
    u[7]  = vaddq_s32(v[3], v[7]);
    u[8]  = vaddq_s32(v[8], v[12]);
    u[9]  = vaddq_s32(v[9], v[13]);
    u[10] = vsubq_s32(v[14], v[10]);
    u[11] = vaddq_s32(v[11], v[15]);
    u[12] = vsubq_s32(v[8], v[12]);
    u[13] = vsubq_s32(v[9], v[13]);
    u[14] = vaddq_s32(v[10], v[14]);
    u[15] = vsubq_s32(v[11], v[15]);

    // stage 6
    v[0] = u[0];
    v[1] = u[1];
    v[2] = u[2];
    v[3] = u[3];
    v[4] = u[4];
    v[5] = u[5];
    v[6] = u[6];
    v[7] = u[7];

    butterfly_0112_neon(cospi, 8, u[8], u[9], &v[8], &v[9], v_bit);
    butterfly_0130_neon(cospi, 8, u[12], u[13], &v[13], &v[12], v_bit);
    butterfly_0130_neon(cospi, 24, u[11], u[10], &v[10], &v[11], v_bit);
    butterfly_0130_neon(cospi, 24, u[14], u[15], &v[14], &v[15], v_bit);

    // stage 7
    u[0]  = vaddq_s32(v[0], v[8]);
    u[1]  = vaddq_s32(v[1], v[9]);
    u[2]  = vaddq_s32(v[2], v[10]);
    u[3]  = vaddq_s32(v[3], v[11]);
    u[4]  = vaddq_s32(v[4], v[12]);
    u[5]  = vaddq_s32(v[5], v[13]);
    u[6]  = vaddq_s32(v[6], v[14]);
    u[7]  = vsubq_s32(v[15], v[7]);
    u[8]  = vsubq_s32(v[0], v[8]);
    u[9]  = vsubq_s32(v[1], v[9]);
    u[10] = vsubq_s32(v[2], v[10]);
    u[11] = vsubq_s32(v[3], v[11]);
    u[12] = vsubq_s32(v[4], v[12]);
    u[13] = vsubq_s32(v[5], v[13]);
    u[14] = vsubq_s32(v[6], v[14]);
    u[15] = vaddq_s32(v[7], v[15]);

    // stage 8
    butterfly_0112_neon(cospi, 2, u[0], u[1], &v[0], &v[1], v_bit);
    butterfly_0112_neon(cospi, 10, u[2], u[3], &v[2], &v[3], v_bit);
    butterfly_0112_neon(cospi, 18, u[4], u[5], &v[4], &v[5], v_bit);
    butterfly_0112_neon(cospi, 26, u[6], u[7], &v[6], &v[7], v_bit);
    butterfly_0130_neon(cospi, 30, u[9], u[8], &v[8], &v[9], v_bit);
    butterfly_0130_neon(cospi, 22, u[11], u[10], &v[10], &v[11], v_bit);
    butterfly_0130_neon(cospi, 14, u[13], u[12], &v[12], &v[13], v_bit);
    butterfly_0112_neon(cospi, 6, u[14], u[15], &v[15], &v[14], v_bit);

    // stage 9
    out[0]  = v[1];
    out[1]  = v[14];
    out[2]  = v[3];
    out[3]  = v[12];
    out[4]  = v[5];
    out[5]  = v[10];
    out[6]  = v[7];
    out[7]  = v[8];
    out[8]  = v[9];
    out[9]  = v[6];
    out[10] = v[11];
    out[11] = v[4];
    out[12] = v[13];
    out[13] = v[2];
    out[14] = v[15];
    out[15] = v[0];
}

static void highbd_fidentity16_x4_neon(const int32x4_t *in, int32x4_t *out, int bit) {
    (void)bit;
    const int32x4_t fact = vdupq_n_s32(2 * new_sqrt2);

    for (int i = 0; i < 16; i++) {
        int32x4_t a = vmulq_s32(in[i], fact);
        out[i]      = vrshrq_n_s32(a, new_sqrt2_bits);
    }
}

static void highbd_fdct16_xn_neon(const int32x4_t *in, int32x4_t *out, int bit, const int howmany) {
    const int stride = 16;
    int       i      = 0;
    do { highbd_fdct16_x4_neon(in + i * stride, out + i * stride, bit); } while (++i < howmany);
}

static void highbd_fadst16_xn_neon(const int32x4_t *in, int32x4_t *out, int bit, int howmany) {
    const int stride = 16;
    int       i      = 0;
    do { highbd_fadst16_x4_neon(in + i * stride, out + i * stride, bit); } while (++i < howmany);
}

static void highbd_fidentity16_xn_neon(const int32x4_t *in, int32x4_t *out, int bit, int howmany) {
    const int stride = 16;
    int       i      = 0;
    do { highbd_fidentity16_x4_neon(in + i * stride, out + i * stride, bit); } while (++i < howmany);
}

void svt_av1_fwd_txfm2d_16x16_neon(int16_t *input, int32_t *output, uint32_t stride, TxType tx_type, uint8_t bd) {
    (void)bd;
    int ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);
    ud_adjust_input_and_stride(ud_flip, &input, &stride, 16);

    // Workspaces for column/row-wise transforms.
    int32x4_t buf0[64], buf1[64];

    switch (tx_type) {
    case DCT_DCT:
        load_buffer_16x16_(input, buf0, stride, 0);
        highbd_fdct16_xn_neon(buf0, buf0, fwd_cos_bit_col[2][2], 4);
        shift_right_2_round_s32_x4(buf0, buf0, 64);
        transpose_arrays_s32_16x16(buf0, buf1);
        highbd_fdct16_xn_neon(buf1, buf1, fwd_cos_bit_row[2][2], 4);
        transpose_arrays_s32_16x16(buf1, buf0);
        store_buffer_16x16(buf0, output, /*stride=*/16);
        break;
    case ADST_DCT:
        load_buffer_16x16_(input, buf0, stride, 0);
        highbd_fadst16_xn_neon(buf0, buf0, fwd_cos_bit_col[2][2], 4);
        shift_right_2_round_s32_x4(buf0, buf0, 64);
        transpose_arrays_s32_16x16(buf0, buf1);
        highbd_fdct16_xn_neon(buf1, buf1, fwd_cos_bit_row[2][2], 4);
        transpose_arrays_s32_16x16(buf1, buf0);
        store_buffer_16x16(buf0, output, /*stride=*/16);
        break;
    case DCT_ADST:
        load_buffer_16x16_(input, buf0, stride, 0);
        highbd_fdct16_xn_neon(buf0, buf0, fwd_cos_bit_col[2][2], 4);
        shift_right_2_round_s32_x4(buf0, buf0, 64);
        transpose_arrays_s32_16x16(buf0, buf1);
        highbd_fadst16_xn_neon(buf1, buf1, fwd_cos_bit_row[2][2], 4);
        transpose_arrays_s32_16x16(buf1, buf0);
        store_buffer_16x16(buf0, output, /*stride=*/16);
        break;
    case ADST_ADST:
        load_buffer_16x16_(input, buf0, stride, 0);
        highbd_fadst16_xn_neon(buf0, buf0, fwd_cos_bit_col[2][2], 4);
        shift_right_2_round_s32_x4(buf0, buf0, 64);
        transpose_arrays_s32_16x16(buf0, buf1);
        highbd_fadst16_xn_neon(buf1, buf1, fwd_cos_bit_row[2][2], 4);
        transpose_arrays_s32_16x16(buf1, buf0);
        store_buffer_16x16(buf0, output, /*stride=*/16);
        break;
    case FLIPADST_DCT:
        load_buffer_16x16_(input, buf0, stride, 0);
        highbd_fadst16_xn_neon(buf0, buf0, fwd_cos_bit_col[2][2], 4);
        shift_right_2_round_s32_x4(buf0, buf0, 64);
        transpose_arrays_s32_16x16(buf0, buf1);
        highbd_fdct16_xn_neon(buf1, buf1, fwd_cos_bit_row[2][2], 4);
        transpose_arrays_s32_16x16(buf1, buf0);
        store_buffer_16x16(buf0, output, /*stride=*/16);
        break;
    case DCT_FLIPADST:
        load_buffer_16x16_(input, buf0, stride, 1);
        highbd_fdct16_xn_neon(buf0, buf0, fwd_cos_bit_col[2][2], 4);
        shift_right_2_round_s32_x4(buf0, buf0, 64);
        transpose_arrays_s32_16x16(buf0, buf1);
        highbd_fadst16_xn_neon(buf1, buf1, fwd_cos_bit_row[2][2], 4);
        transpose_arrays_s32_16x16(buf1, buf0);
        store_buffer_16x16(buf0, output, /*stride=*/16);
        break;
    case FLIPADST_FLIPADST:
        load_buffer_16x16_(input, buf0, stride, 1);
        highbd_fadst16_xn_neon(buf0, buf0, fwd_cos_bit_col[2][2], 4);
        shift_right_2_round_s32_x4(buf0, buf0, 64);
        transpose_arrays_s32_16x16(buf0, buf1);
        highbd_fadst16_xn_neon(buf1, buf1, fwd_cos_bit_row[2][2], 4);
        transpose_arrays_s32_16x16(buf1, buf0);
        store_buffer_16x16(buf0, output, /*stride=*/16);
        break;
    case ADST_FLIPADST:
        load_buffer_16x16_(input, buf0, stride, 1);
        highbd_fadst16_xn_neon(buf0, buf0, fwd_cos_bit_col[2][2], 4);
        shift_right_2_round_s32_x4(buf0, buf0, 64);
        transpose_arrays_s32_16x16(buf0, buf1);
        highbd_fadst16_xn_neon(buf1, buf1, fwd_cos_bit_row[2][2], 4);
        transpose_arrays_s32_16x16(buf1, buf0);
        store_buffer_16x16(buf0, output, /*stride=*/16);
        break;
    case FLIPADST_ADST:
        load_buffer_16x16_(input, buf0, stride, 0);
        highbd_fadst16_xn_neon(buf0, buf0, fwd_cos_bit_col[2][2], 4);
        shift_right_2_round_s32_x4(buf0, buf0, 64);
        transpose_arrays_s32_16x16(buf0, buf1);
        highbd_fadst16_xn_neon(buf1, buf1, fwd_cos_bit_row[2][2], 4);
        transpose_arrays_s32_16x16(buf1, buf0);
        store_buffer_16x16(buf0, output, /*stride=*/16);
        break;
    case IDTX:
        load_buffer_16x16_(input, buf0, stride, 0);
        highbd_fidentity16_xn_neon(buf0, buf1, fwd_cos_bit_col[2][2], 4);
        shift_right_2_round_s32_x4(buf1, buf1, 64);
        highbd_fidentity16_xn_neon(buf1, buf0, fwd_cos_bit_row[2][2], 4);
        store_buffer_16x16(buf0, output, /*stride=*/16);
        break;
    case V_DCT:
        load_buffer_16x16_(input, buf0, stride, 0);
        highbd_fdct16_xn_neon(buf0, buf0, fwd_cos_bit_col[2][2], 4);
        shift_right_2_round_s32_x4(buf0, buf0, 64);
        transpose_arrays_s32_16x16(buf0, buf1);
        highbd_fidentity16_xn_neon(buf1, buf1, fwd_cos_bit_row[2][2], 4);
        transpose_arrays_s32_16x16(buf1, buf0);
        store_buffer_16x16(buf0, output, /*stride=*/16);
        break;
    case H_DCT:
        load_buffer_16x16_(input, buf0, stride, 0);
        highbd_fidentity16_xn_neon(buf0, buf0, fwd_cos_bit_col[2][2], 4);
        shift_right_2_round_s32_x4(buf0, buf0, 64);
        transpose_arrays_s32_16x16(buf0, buf1);
        highbd_fdct16_xn_neon(buf1, buf1, fwd_cos_bit_row[2][2], 4);
        transpose_arrays_s32_16x16(buf1, buf0);
        store_buffer_16x16(buf0, output, /*stride=*/16);
        break;
    case V_ADST:
        load_buffer_16x16_(input, buf0, stride, 0);
        highbd_fadst16_xn_neon(buf0, buf0, fwd_cos_bit_col[2][2], 4);
        shift_right_2_round_s32_x4(buf0, buf0, 64);
        transpose_arrays_s32_16x16(buf0, buf1);
        highbd_fidentity16_xn_neon(buf1, buf1, fwd_cos_bit_row[2][2], 4);
        transpose_arrays_s32_16x16(buf1, buf0);
        store_buffer_16x16(buf0, output, /*stride=*/16);
        break;
    case H_ADST:
        load_buffer_16x16_(input, buf0, stride, 0);
        highbd_fidentity16_xn_neon(buf0, buf0, fwd_cos_bit_col[2][2], 4);
        shift_right_2_round_s32_x4(buf0, buf0, 64);
        transpose_arrays_s32_16x16(buf0, buf1);
        highbd_fadst16_xn_neon(buf1, buf1, fwd_cos_bit_row[2][2], 4);
        transpose_arrays_s32_16x16(buf1, buf0);
        store_buffer_16x16(buf0, output, /*stride=*/16);
        break;
    case V_FLIPADST:
        load_buffer_16x16_(input, buf0, stride, 0);
        highbd_fadst16_xn_neon(buf0, buf0, fwd_cos_bit_col[2][2], 4);
        shift_right_2_round_s32_x4(buf0, buf0, 64);
        transpose_arrays_s32_16x16(buf0, buf1);
        highbd_fidentity16_xn_neon(buf1, buf1, fwd_cos_bit_row[2][2], 4);
        transpose_arrays_s32_16x16(buf1, buf0);
        store_buffer_16x16(buf0, output, /*stride=*/16);
        break;
    case H_FLIPADST:
        load_buffer_16x16_(input, buf0, stride, 1);
        highbd_fidentity16_xn_neon(buf0, buf0, fwd_cos_bit_col[2][2], 4);
        shift_right_2_round_s32_x4(buf0, buf0, 64);
        transpose_arrays_s32_16x16(buf0, buf1);
        highbd_fadst16_xn_neon(buf1, buf1, fwd_cos_bit_row[2][2], 4);
        transpose_arrays_s32_16x16(buf1, buf0);
        store_buffer_16x16(buf0, output, /*stride=*/16);
        break;
    default: assert(0);
    }
}

static AOM_FORCE_INLINE void round_rect_array_s32_neon(const int32x4_t *input, int32x4_t *output, const int size) {
    const int32x4_t sqrt2 = vdupq_n_s32(new_sqrt2);
    int             i     = 0;
    do {
        const int32x4_t r1 = vmulq_s32(input[i], sqrt2);
        output[i]          = vrshrq_n_s32(r1, new_sqrt2_bits);
    } while (++i < size);
}

typedef void (*fwd_transform_1d_col_neon)(const int16_t *in, int32x4_t *out, int stride, int bit, int lr_flip);
typedef void (*fwd_transform_1d_col_many_neon)(const int16_t *in, int32x4_t *out, int stride, int bit, int lr_flip,
                                               int howmany, int hm_stride);

typedef void (*fwd_transform_1d_row_neon)(const int32x4_t *in, int32x4_t *out, int bit);
typedef void (*fwd_transform_1d_row_many_neon)(const int32x4_t *in, int32x4_t *out, int bit, int howmany,
                                               int hm_stride);

// Construct component kernels that include the load_buffer and store_buffer
// stages to avoid the need to spill loaded data to the stack between these and
// the txfm kernel calls.
// The TRANSFORM_*_ONE cases are only ever called in situations where the
// howmany parameter would be one, so no need for the loop at all in these
// cases.

#define TRANSFORM_COL_ONE(name, n)                                                       \
    static void highbd_##name##_col_neon(                                                \
        const int16_t *input, int32x4_t *output, int stride, int cos_bit, int lr_flip) { \
        int32x4_t buf0[n];                                                               \
        load_buffer_4x##n##_(input, buf0, stride, lr_flip);                              \
        highbd_##name##_x4_neon(buf0, output, cos_bit);                                  \
    }

#define TRANSFORM_COL_MANY(name, n)                                                                                  \
    static void highbd_##name##_col_many_neon(                                                                       \
        const int16_t *input, int32x4_t *output, int stride, int cos_bit, int lr_flip, int howmany, int hm_stride) { \
        int i = 0;                                                                                                   \
        do {                                                                                                         \
            int32x4_t buf0[n];                                                                                       \
            load_buffer_4x##n##_(input + 4 * i, buf0, stride, lr_flip);                                              \
            highbd_##name##_x4_neon(buf0, output + i * hm_stride, cos_bit);                                          \
        } while (++i < howmany);                                                                                     \
    }

#define TRANSFORM_ROW_ONE(name, n)                                                                 \
    static void highbd_##name##_row_neon(const int32x4_t *input, int32x4_t *output, int cos_bit) { \
        highbd_##name##_x4_neon(input, output, cos_bit);                                           \
    }

#define TRANSFORM_ROW_RECT_ONE(name, n)                                                                 \
    static void highbd_##name##_row_rect_neon(const int32x4_t *input, int32x4_t *output, int cos_bit) { \
        highbd_##name##_x4_neon(input, output, cos_bit);                                                \
    }

#define TRANSFORM_ROW_MANY(name, n)                                                                                    \
    static void highbd_##name##_row_many_neon(                                                                         \
        const int32x4_t *input, int32x4_t *output, int cos_bit, int howmany, int hm_stride) {                          \
        int i = 0;                                                                                                     \
        do { highbd_##name##_x4_neon(input + hm_stride * i, output + hm_stride * i, cos_bit); } while (++i < howmany); \
    }

#define TRANSFORM_ROW_RECT_MANY(name, n)                                                      \
    static void highbd_##name##_row_rect_many_neon(                                           \
        const int32x4_t *input, int32x4_t *output, int cos_bit, int howmany, int hm_stride) { \
        int i = 0;                                                                            \
        do {                                                                                  \
            highbd_##name##_x4_neon(input + hm_stride * i, output + hm_stride * i, cos_bit);  \
            round_rect_array_s32_neon(output + hm_stride * i, output + hm_stride * i, (n));   \
        } while (++i < howmany);                                                              \
    }

TRANSFORM_COL_ONE(fdct8, 8)
TRANSFORM_COL_ONE(fadst8, 8)
TRANSFORM_COL_ONE(fidentity8, 8)

TRANSFORM_COL_MANY(fdct4, 4)
TRANSFORM_COL_MANY(fadst4, 4)
TRANSFORM_COL_MANY(fidentity4, 4)
TRANSFORM_COL_MANY(fdct8, 8)
TRANSFORM_COL_MANY(fadst8, 8)
TRANSFORM_COL_MANY(fidentity8, 8)
TRANSFORM_COL_MANY(fdct16, 16)
TRANSFORM_COL_MANY(fadst16, 16)
TRANSFORM_COL_MANY(fidentity16, 16)

TRANSFORM_ROW_ONE(fdct16, 16)
TRANSFORM_ROW_ONE(fadst16, 16)
TRANSFORM_ROW_ONE(fidentity16, 16)

TRANSFORM_ROW_MANY(fdct4, 4)
TRANSFORM_ROW_MANY(fadst4, 4)
TRANSFORM_ROW_MANY(fidentity4, 4)
TRANSFORM_ROW_MANY(fdct8, 8)
TRANSFORM_ROW_MANY(fadst8, 8)
TRANSFORM_ROW_MANY(fidentity8, 8)

TRANSFORM_ROW_RECT_ONE(fdct8, 8)
TRANSFORM_ROW_RECT_ONE(fadst8, 8)
TRANSFORM_ROW_RECT_ONE(fidentity8, 8)

TRANSFORM_ROW_RECT_MANY(fdct4, 4)
TRANSFORM_ROW_RECT_MANY(fadst4, 4)
TRANSFORM_ROW_RECT_MANY(fidentity4, 4)
TRANSFORM_ROW_RECT_MANY(fdct8, 8)
TRANSFORM_ROW_RECT_MANY(fadst8, 8)
TRANSFORM_ROW_RECT_MANY(fidentity8, 8)
TRANSFORM_ROW_RECT_MANY(fdct16, 16)
TRANSFORM_ROW_RECT_MANY(fadst16, 16)
TRANSFORM_ROW_RECT_MANY(fidentity16, 16)

static const fwd_transform_1d_col_neon col_highbd_txfm8_x4_arr[TX_TYPES] = {
    highbd_fdct8_col_neon, // DCT_DCT
    highbd_fadst8_col_neon, // ADST_DCT
    highbd_fdct8_col_neon, // DCT_ADST
    highbd_fadst8_col_neon, // ADST_ADST
    highbd_fadst8_col_neon, // FLIPADST_DCT
    highbd_fdct8_col_neon, // DCT_FLIPADST
    highbd_fadst8_col_neon, // FLIPADST_FLIPADST
    highbd_fadst8_col_neon, // ADST_FLIPADST
    highbd_fadst8_col_neon, // FLIPADST_ADST
    highbd_fidentity8_col_neon, // IDTX
    highbd_fdct8_col_neon, // V_DCT
    highbd_fidentity8_col_neon, // H_DCT
    highbd_fadst8_col_neon, // V_ADST
    highbd_fidentity8_col_neon, // H_ADST
    highbd_fadst8_col_neon, // V_FLIPADST
    highbd_fidentity8_col_neon // H_FLIPADST
};

static const fwd_transform_1d_col_many_neon col_highbd_txfm16_xn_arr[TX_TYPES] = {
    highbd_fdct16_col_many_neon, // DCT_DCT
    highbd_fadst16_col_many_neon, // ADST_DCT
    highbd_fdct16_col_many_neon, // DCT_ADST
    highbd_fadst16_col_many_neon, // ADST_ADST
    highbd_fadst16_col_many_neon, // FLIPADST_DCT
    highbd_fdct16_col_many_neon, // DCT_FLIPADST
    highbd_fadst16_col_many_neon, // FLIPADST_FLIPADST
    highbd_fadst16_col_many_neon, // ADST_FLIPADST
    highbd_fadst16_col_many_neon, // FLIPADST_ADST
    highbd_fidentity16_col_many_neon, // IDTX
    highbd_fdct16_col_many_neon, // V_DCT
    highbd_fidentity16_col_many_neon, // H_DCT
    highbd_fadst16_col_many_neon, // V_ADST
    highbd_fidentity16_col_many_neon, // H_ADST
    highbd_fadst16_col_many_neon, // V_FLIPADST
    highbd_fidentity16_col_many_neon // H_FLIPADST
};

static const fwd_transform_1d_row_many_neon row_rect_highbd_txfm4_xn_arr[TX_TYPES] = {
    highbd_fdct4_row_rect_many_neon, // DCT_DCT
    highbd_fdct4_row_rect_many_neon, // ADST_DCT
    highbd_fadst4_row_rect_many_neon, // DCT_ADST
    highbd_fadst4_row_rect_many_neon, // ADST_ADST
    highbd_fdct4_row_rect_many_neon, // FLIPADST_DCT
    highbd_fadst4_row_rect_many_neon, // DCT_FLIPADST
    highbd_fadst4_row_rect_many_neon, // FLIPADST_FLIPADST
    highbd_fadst4_row_rect_many_neon, // ADST_FLIPADST
    highbd_fadst4_row_rect_many_neon, // FLIPADST_ADST
    highbd_fidentity4_row_rect_many_neon, // IDTX
    highbd_fidentity4_row_rect_many_neon, // V_DCT
    highbd_fdct4_row_rect_many_neon, // H_DCT
    highbd_fidentity4_row_rect_many_neon, // V_ADST
    highbd_fadst4_row_rect_many_neon, // H_ADST
    highbd_fidentity4_row_rect_many_neon, // V_FLIPADST
    highbd_fadst4_row_rect_many_neon // H_FLIPADST
};

void svt_av1_fwd_txfm2d_4x8_neon(int16_t *input, int32_t *output, uint32_t stride, TxType tx_type, uint8_t bd) {
    (void)bd;
    int                                  bitcol   = fwd_cos_bit_col[0][1];
    int                                  bitrow   = fwd_cos_bit_row[0][1];
    const fwd_transform_1d_col_neon      col_txfm = col_highbd_txfm8_x4_arr[tx_type];
    const fwd_transform_1d_row_many_neon row_txfm = row_rect_highbd_txfm4_xn_arr[tx_type];

    int ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);
    ud_adjust_input_and_stride(ud_flip, &input, &stride, 8);

    // Column-wise transform.
    int32x4_t buf0[8];
    col_txfm(input, buf0, stride, bitcol, lr_flip);
    shift_right_1_round_s32_x4(buf0, buf0, 8);

    int32x4_t buf1[8];
    transpose_arrays_s32_4x8(buf0, buf1);

    // Row-wise transform.
    row_txfm(buf1, buf0, bitrow, /*howmany=*/2, /*hm_stride=*/4);
    transpose_arrays_s32_4x8(buf0, (int32x4_t *)output);
}

static const fwd_transform_1d_col_many_neon col_highbd_txfm4_xn_arr[TX_TYPES] = {
    highbd_fdct4_col_many_neon, // DCT_DCT
    highbd_fadst4_col_many_neon, // ADST_DCT
    highbd_fdct4_col_many_neon, // DCT_ADST
    highbd_fadst4_col_many_neon, // ADST_ADST
    highbd_fadst4_col_many_neon, // FLIPADST_DCT
    highbd_fdct4_col_many_neon, // DCT_FLIPADST
    highbd_fadst4_col_many_neon, // FLIPADST_FLIPADST
    highbd_fadst4_col_many_neon, // ADST_FLIPADST
    highbd_fadst4_col_many_neon, // FLIPADST_ADST
    highbd_fidentity4_col_many_neon, // IDTX
    highbd_fdct4_col_many_neon, // V_DCT
    highbd_fidentity4_col_many_neon, // H_DCT
    highbd_fadst4_col_many_neon, // V_ADST
    highbd_fidentity4_col_many_neon, // H_ADST
    highbd_fadst4_col_many_neon, // V_FLIPADST
    highbd_fidentity4_col_many_neon // H_FLIPADST
};

static const fwd_transform_1d_row_neon row_highbd_txfm8_x4_arr[TX_TYPES] = {
    highbd_fdct8_row_rect_neon, // DCT_DCT
    highbd_fdct8_row_rect_neon, // ADST_DCT
    highbd_fadst8_row_rect_neon, // DCT_ADST
    highbd_fadst8_row_rect_neon, // ADST_ADST
    highbd_fdct8_row_rect_neon, // FLIPADST_DCT
    highbd_fadst8_row_rect_neon, // DCT_FLIPADST
    highbd_fadst8_row_rect_neon, // FLIPADST_FLIPADST
    highbd_fadst8_row_rect_neon, // ADST_FLIPADST
    highbd_fadst8_row_rect_neon, // FLIPADST_ADST
    highbd_fidentity8_row_rect_neon, // IDTX
    highbd_fidentity8_row_rect_neon, // V_DCT
    highbd_fdct8_row_rect_neon, // H_DCT
    highbd_fidentity8_row_rect_neon, // V_ADST
    highbd_fadst8_row_rect_neon, // H_ADST
    highbd_fidentity8_row_rect_neon, // V_FLIPADST
    highbd_fadst8_row_rect_neon // H_FLIPADST
};

static const fwd_transform_1d_row_many_neon row_rect_highbd_txfm8_xn_arr[TX_TYPES] = {
    highbd_fdct8_row_rect_many_neon, // DCT_DCT
    highbd_fdct8_row_rect_many_neon, // ADST_DCT
    highbd_fadst8_row_rect_many_neon, // DCT_ADST
    highbd_fadst8_row_rect_many_neon, // ADST_ADST
    highbd_fdct8_row_rect_many_neon, // FLIPADST_DCT
    highbd_fadst8_row_rect_many_neon, // DCT_FLIPADST
    highbd_fadst8_row_rect_many_neon, // FLIPADST_FLIPADST
    highbd_fadst8_row_rect_many_neon, // ADST_FLIPADST
    highbd_fadst8_row_rect_many_neon, // FLIPADST_ADST
    highbd_fidentity8_row_rect_many_neon, // IDTX
    highbd_fidentity8_row_rect_many_neon, // V_DCT
    highbd_fdct8_row_rect_many_neon, // H_DCT
    highbd_fidentity8_row_rect_many_neon, // V_ADST
    highbd_fadst8_row_rect_many_neon, // H_ADST
    highbd_fidentity8_row_rect_many_neon, // V_FLIPADST
    highbd_fadst8_row_rect_many_neon // H_FLIPADST
};

static const fwd_transform_1d_row_many_neon row_highbd_txfm4_xn_arr[TX_TYPES] = {
    highbd_fdct4_row_many_neon, // DCT_DCT
    highbd_fdct4_row_many_neon, // ADST_DCT
    highbd_fadst4_row_many_neon, // DCT_ADST
    highbd_fadst4_row_many_neon, // ADST_ADST
    highbd_fdct4_row_many_neon, // FLIPADST_DCT
    highbd_fadst4_row_many_neon, // DCT_FLIPADST
    highbd_fadst4_row_many_neon, // FLIPADST_FLIPADST
    highbd_fadst4_row_many_neon, // ADST_FLIPADST
    highbd_fadst4_row_many_neon, // FLIPADST_ADST
    highbd_fidentity4_row_many_neon, // IDTX
    highbd_fidentity4_row_many_neon, // V_DCT
    highbd_fdct4_row_many_neon, // H_DCT
    highbd_fidentity4_row_many_neon, // V_ADST
    highbd_fadst4_row_many_neon, // H_ADST
    highbd_fidentity4_row_many_neon, // V_FLIPADST
    highbd_fadst4_row_many_neon // H_FLIPADST
};

static const fwd_transform_1d_row_many_neon row_highbd_txfm8_xn_arr[TX_TYPES] = {
    highbd_fdct8_row_many_neon, // DCT_DCT
    highbd_fdct8_row_many_neon, // ADST_DCT
    highbd_fadst8_row_many_neon, // DCT_ADST
    highbd_fadst8_row_many_neon, // ADST_ADST
    highbd_fdct8_row_many_neon, // FLIPADST_DCT
    highbd_fadst8_row_many_neon, // DCT_FLIPADST
    highbd_fadst8_row_many_neon, // FLIPADST_FLIPADST
    highbd_fadst8_row_many_neon, // ADST_FLIPADST
    highbd_fadst8_row_many_neon, // FLIPADST_ADST
    highbd_fidentity8_row_many_neon, // IDTX
    highbd_fidentity8_row_many_neon, // V_DCT
    highbd_fdct8_row_many_neon, // H_DCT
    highbd_fidentity8_row_many_neon, // V_ADST
    highbd_fadst8_row_many_neon, // H_ADST
    highbd_fidentity8_row_many_neon, // V_FLIPADST
    highbd_fadst8_row_many_neon // H_FLIPADST
};

void svt_av1_fwd_txfm2d_4x16_neon(int16_t *input, int32_t *output, uint32_t stride, TxType tx_type, uint8_t bd) {
    (void)bd;
    int                                  bitcol   = fwd_cos_bit_col[0][2];
    int                                  bitrow   = fwd_cos_bit_row[0][2];
    const fwd_transform_1d_col_many_neon col_txfm = col_highbd_txfm16_xn_arr[tx_type];
    const fwd_transform_1d_row_many_neon row_txfm = row_highbd_txfm4_xn_arr[tx_type];

    int ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);
    ud_adjust_input_and_stride(ud_flip, &input, &stride, 16);

    // Column-wise transform.
    int32x4_t buf0[16];
    if (lr_flip) {
        col_txfm(input,
                 buf0,
                 stride,
                 bitcol,
                 /*lr_flip=*/1,
                 /*howmany=*/1,
                 /*hm_stride=*/0);
    } else {
        col_txfm(input,
                 buf0,
                 stride,
                 bitcol,
                 /*lr_flip=*/0,
                 /*howmany=*/1,
                 /*hm_stride=*/0);
    }
    shift_right_1_round_s32_x4(buf0, buf0, 16);

    int32x4_t buf1[16];
    transpose_arrays_s32_4x16(buf0, buf1);

    // Row-wise transform.
    row_txfm(buf1, buf0, bitrow, /*howmany=*/4, /*hm_stride=*/4);
    transpose_arrays_s32_4x16(buf0, (int32x4_t *)output);
}

static INLINE void transpose_elems_s32_8x4(const int32x4_t *in, int32x4_t *out) {
    transpose_elems_s32_4x4(in[0], in[1], in[2], in[3], &out[0], &out[2], &out[4], &out[6]);
    transpose_elems_s32_4x4(in[4], in[5], in[6], in[7], &out[1], &out[3], &out[5], &out[7]);
}

static INLINE void transpose_8xh(const int32x4_t *in, int32x4_t *out, int n) {
    for (int i = 0; i < n; i += 8) { transpose_elems_s32_8x4(in + i, out + i); }
}

void svt_av1_fwd_txfm2d_8x4_neon(int16_t *input, int32_t *coeff, uint32_t stride, TxType tx_type, uint8_t bd) {
    (void)bd;
    const int                            bitcol   = fwd_cos_bit_col[1][0];
    const int                            bitrow   = fwd_cos_bit_row[1][0];
    const fwd_transform_1d_col_many_neon col_txfm = col_highbd_txfm4_xn_arr[tx_type];
    const fwd_transform_1d_row_neon      row_txfm = row_highbd_txfm8_x4_arr[tx_type];

    int ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);
    ud_adjust_input_and_stride(ud_flip, &input, &stride, 4);

    // Column-wise transform.
    int32x4_t buf0[8];
    if (lr_flip) {
        col_txfm(input,
                 buf0 + 4,
                 stride,
                 bitcol,
                 /*lr_flip=*/1,
                 /*howmany=*/2,
                 /*hm_stride=*/-4);
    } else {
        col_txfm(input,
                 buf0,
                 stride,
                 bitcol,
                 /*lr_flip=*/0,
                 /*howmany=*/2,
                 /*hm_stride=*/4);
    }

    shift_right_1_round_s32_x4(buf0, buf0, 8);

    int32x4_t buf1[8];
    transpose_arrays_s32_8x4(buf0, buf1);

    // Row-wise transform.
    row_txfm(buf1, buf0, bitrow);
    round_rect_array_s32_neon(buf0, buf1, 8);
    transpose_8xh(buf1, (int32x4_t *)coeff, 8);
}

void svt_av1_fwd_txfm2d_8x16_neon(int16_t *input, int32_t *coeff, uint32_t stride, TxType tx_type, uint8_t bd) {
    (void)bd;
    const fwd_transform_1d_col_many_neon col_txfm = col_highbd_txfm16_xn_arr[tx_type];
    const fwd_transform_1d_row_many_neon row_txfm = row_rect_highbd_txfm8_xn_arr[tx_type];
    int                                  bit      = fwd_cos_bit_col[1][2];

    int ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);
    ud_adjust_input_and_stride(ud_flip, &input, &stride, 16);

    // Column-wise transform.
    int32x4_t buf0[32];
    if (lr_flip) {
        col_txfm(input,
                 buf0 + 16,
                 stride,
                 bit,
                 /*lr_flip=*/1,
                 /*howmany=*/2,
                 /*hm_stride=*/-16);
    } else {
        col_txfm(input,
                 buf0,
                 stride,
                 bit,
                 /*lr_flip=*/0,
                 /*howmany=*/2,
                 /*hm_stride=*/16);
    }
    shift_right_2_round_s32_x4(buf0, buf0, 32);

    int32x4_t buf1[32];
    transpose_arrays_s32_8x16(buf0, buf1);

    // Row-wise transform.
    row_txfm(buf1, buf0, bit, /*howmany=*/4, /*hm_stride=*/8);
    transpose_8xh(buf0, (int32x4_t *)coeff, 32);
}

static void highbd_fdct32_x4_neon(const int32x4_t *input, int32x4_t *output, int cos_bit) {
    const int32_t *const cospi     = cospi_arr_s32(cos_bit);
    const int32x4_t      v_cos_bit = vdupq_n_s32(-cos_bit);

    // Workspaces for intermediate transform steps.
    int32x4_t buf0[32];
    int32x4_t buf1[32];

    // stage 1
    butterfly_dct_pre(input, buf1, 32);

    // stage 2
    butterfly_dct_pre(buf1, buf0, 16);
    buf0[16] = buf1[16];
    buf0[17] = buf1[17];
    buf0[18] = buf1[18];
    buf0[19] = buf1[19];
    butterfly_0112_neon(cospi, 32, buf1[27], buf1[20], &buf0[27], &buf0[20], v_cos_bit);
    butterfly_0112_neon(cospi, 32, buf1[26], buf1[21], &buf0[26], &buf0[21], v_cos_bit);
    butterfly_0112_neon(cospi, 32, buf1[25], buf1[22], &buf0[25], &buf0[22], v_cos_bit);
    butterfly_0112_neon(cospi, 32, buf1[24], buf1[23], &buf0[24], &buf0[23], v_cos_bit);
    buf0[28] = buf1[28];
    buf0[29] = buf1[29];
    buf0[30] = buf1[30];
    buf0[31] = buf1[31];

    // stage 3
    butterfly_dct_pre(buf0, buf1, 8);
    buf1[8] = buf0[8];
    buf1[9] = buf0[9];
    butterfly_0112_neon(cospi, 32, buf0[13], buf0[10], &buf1[13], &buf1[10], v_cos_bit);
    butterfly_0112_neon(cospi, 32, buf0[12], buf0[11], &buf1[12], &buf1[11], v_cos_bit);
    buf1[14] = buf0[14];
    buf1[15] = buf0[15];
    butterfly_dct_post(buf0 + 16, buf0 + 16, buf1 + 16, 16);

    // stage 4
    butterfly_dct_pre(buf1, buf0, 4);
    buf0[4] = buf1[4];
    butterfly_0112_neon(cospi, 32, buf1[6], buf1[5], &buf0[6], &buf0[5], v_cos_bit);
    buf0[7] = buf1[7];
    butterfly_dct_post(buf1 + 8, buf1 + 8, buf0 + 8, 8);
    buf0[16] = buf1[16];
    buf0[17] = buf1[17];
    butterfly_0112_neon(cospi, 16, buf1[29], buf1[18], &buf0[29], &buf0[18], v_cos_bit);
    butterfly_0112_neon(cospi, 16, buf1[28], buf1[19], &buf0[28], &buf0[19], v_cos_bit);
    butterfly_2312_neon(cospi, 16, buf1[27], buf1[20], &buf0[20], &buf0[27], v_cos_bit);
    butterfly_2312_neon(cospi, 16, buf1[26], buf1[21], &buf0[21], &buf0[26], v_cos_bit);
    buf0[22] = buf1[22];
    buf0[23] = buf1[23];
    buf0[24] = buf1[24];
    buf0[25] = buf1[25];
    buf0[30] = buf1[30];
    buf0[31] = buf1[31];

    // stage 5
    butterfly_0112_neon(cospi, 32, buf0[0], buf0[1], &buf1[0], &buf1[1], v_cos_bit);
    butterfly_0112_neon(cospi, 16, buf0[3], buf0[2], &buf1[2], &buf1[3], v_cos_bit);
    butterfly_dct_post(buf0 + 4, buf0 + 4, buf1 + 4, 4);
    buf1[8] = buf0[8];
    butterfly_0112_neon(cospi, 16, buf0[14], buf0[9], &buf1[14], &buf1[9], v_cos_bit);
    butterfly_2312_neon(cospi, 16, buf0[13], buf0[10], &buf1[10], &buf1[13], v_cos_bit);
    buf1[11] = buf0[11];
    buf1[12] = buf0[12];
    buf1[15] = buf0[15];
    butterfly_dct_post(buf0 + 16, buf0 + 16, buf1 + 16, 8);
    butterfly_dct_post(buf0 + 24, buf0 + 24, buf1 + 24, 8);

    // stage 6
    buf0[0] = buf1[0];
    buf0[1] = buf1[1];
    buf0[2] = buf1[2];
    buf0[3] = buf1[3];

    butterfly_0112_neon(cospi, 8, buf1[7], buf1[4], &buf0[4], &buf0[7], v_cos_bit);
    butterfly_0112_neon(cospi, 8, buf1[30], buf1[17], &buf0[30], &buf0[17], v_cos_bit);
    butterfly_2312_neon(cospi, 8, buf1[29], buf1[18], &buf0[18], &buf0[29], v_cos_bit);
    butterfly_dct_post(buf1 + 8, buf1 + 8, buf0 + 8, 4);
    butterfly_dct_post(buf1 + 12, buf1 + 12, buf0 + 12, 4);
    buf0[16] = buf1[16];
    buf0[19] = buf1[19];
    buf0[20] = buf1[20];

    butterfly_0130_neon(cospi, 24, buf1[5], buf1[6], &buf0[5], &buf0[6], v_cos_bit);
    butterfly_0130_neon(cospi, 24, buf1[21], buf1[26], &buf0[26], &buf0[21], v_cos_bit);
    butterfly_0332_neon(cospi, 24, buf1[25], buf1[22], &buf0[25], &buf0[22], v_cos_bit);

    buf0[23] = buf1[23];
    buf0[24] = buf1[24];
    buf0[27] = buf1[27];
    buf0[28] = buf1[28];
    buf0[31] = buf1[31];

    // stage 7
    buf1[0] = buf0[0];
    buf1[1] = buf0[1];
    buf1[2] = buf0[2];
    buf1[3] = buf0[3];
    buf1[4] = buf0[4];
    buf1[5] = buf0[5];
    buf1[6] = buf0[6];
    buf1[7] = buf0[7];
    butterfly_0112_neon(cospi, 4, buf0[15], buf0[8], &buf1[8], &buf1[15], v_cos_bit);
    butterfly_0130_neon(cospi, 28, buf0[9], buf0[14], &buf1[9], &buf1[14], v_cos_bit);
    butterfly_0112_neon(cospi, 20, buf0[13], buf0[10], &buf1[10], &buf1[13], v_cos_bit);
    butterfly_0130_neon(cospi, 12, buf0[11], buf0[12], &buf1[11], &buf1[12], v_cos_bit);
    butterfly_dct_post(buf0 + 16, buf0 + 16, buf1 + 16, 4);
    butterfly_dct_post(buf0 + 20, buf0 + 20, buf1 + 20, 4);
    butterfly_dct_post(buf0 + 24, buf0 + 24, buf1 + 24, 4);
    butterfly_dct_post(buf0 + 28, buf0 + 28, buf1 + 28, 4);

    // stage 8
    buf0[0]  = buf1[0];
    buf0[1]  = buf1[1];
    buf0[2]  = buf1[2];
    buf0[3]  = buf1[3];
    buf0[4]  = buf1[4];
    buf0[5]  = buf1[5];
    buf0[6]  = buf1[6];
    buf0[7]  = buf1[7];
    buf0[8]  = buf1[8];
    buf0[9]  = buf1[9];
    buf0[10] = buf1[10];
    buf0[11] = buf1[11];
    buf0[12] = buf1[12];
    buf0[13] = buf1[13];
    buf0[14] = buf1[14];
    buf0[15] = buf1[15];
    butterfly_0112_neon(cospi, 2, buf1[31], buf1[16], &buf0[16], &buf0[31], v_cos_bit);
    butterfly_0130_neon(cospi, 30, buf1[17], buf1[30], &buf0[17], &buf0[30], v_cos_bit);
    butterfly_0112_neon(cospi, 18, buf1[29], buf1[18], &buf0[18], &buf0[29], v_cos_bit);
    butterfly_0130_neon(cospi, 14, buf1[19], buf1[28], &buf0[19], &buf0[28], v_cos_bit);
    butterfly_0112_neon(cospi, 10, buf1[27], buf1[20], &buf0[20], &buf0[27], v_cos_bit);
    butterfly_0130_neon(cospi, 22, buf1[21], buf1[26], &buf0[21], &buf0[26], v_cos_bit);
    butterfly_0112_neon(cospi, 26, buf1[25], buf1[22], &buf0[22], &buf0[25], v_cos_bit);
    butterfly_0130_neon(cospi, 6, buf1[23], buf1[24], &buf0[23], &buf0[24], v_cos_bit);

    // stage 9
    output[0]  = buf0[0];
    output[1]  = buf0[16];
    output[2]  = buf0[8];
    output[3]  = buf0[24];
    output[4]  = buf0[4];
    output[5]  = buf0[20];
    output[6]  = buf0[12];
    output[7]  = buf0[28];
    output[8]  = buf0[2];
    output[9]  = buf0[18];
    output[10] = buf0[10];
    output[11] = buf0[26];
    output[12] = buf0[6];
    output[13] = buf0[22];
    output[14] = buf0[14];
    output[15] = buf0[30];
    output[16] = buf0[1];
    output[17] = buf0[17];
    output[18] = buf0[9];
    output[19] = buf0[25];
    output[20] = buf0[5];
    output[21] = buf0[21];
    output[22] = buf0[13];
    output[23] = buf0[29];
    output[24] = buf0[3];
    output[25] = buf0[19];
    output[26] = buf0[11];
    output[27] = buf0[27];
    output[28] = buf0[7];
    output[29] = buf0[23];
    output[30] = buf0[15];
    output[31] = buf0[31];
}

static void highbd_fidentity32_x4_neon(const int32x4_t *input, int32x4_t *output, int cos_bit) {
    (void)cos_bit;
    for (int i = 0; i < 32; i++) { output[i] = vshlq_n_s32(input[i], 2); }
}

TRANSFORM_COL_MANY(fdct32, 32)
TRANSFORM_COL_MANY(fidentity32, 32)

static const fwd_transform_1d_col_many_neon col_highbd_txfm32_x4_arr[TX_TYPES] = {
    highbd_fdct32_col_many_neon, // DCT_DCT
    NULL, // ADST_DCT
    NULL, // DCT_ADST
    NULL, // ADST_ADST
    NULL, // FLIPADST_DCT
    NULL, // DCT_FLIPADST
    NULL, // FLIPADST_FLIPADST
    NULL, // ADST_FLIPADST
    NULL, // FLIPADST_ADST
    highbd_fidentity32_col_many_neon, // IDTX
    highbd_fdct32_col_many_neon, // V_DCT
    highbd_fidentity32_col_many_neon, // H_DCT
    NULL, // V_ADST
    NULL, // H_ADST
    NULL, // V_FLIPADST
    NULL // H_FLIPADST
};

void svt_av1_fwd_txfm2d_8x32_neon(int16_t *input, int32_t *coeff, uint32_t stride, TxType tx_type, uint8_t bd) {
    (void)bd;
    const fwd_transform_1d_col_many_neon col_txfm = col_highbd_txfm32_x4_arr[tx_type];
    const fwd_transform_1d_row_many_neon row_txfm = row_highbd_txfm8_xn_arr[tx_type];
    int                                  bitcol   = fwd_cos_bit_col[1][3];
    int                                  bitrow   = fwd_cos_bit_row[1][3];

    // Column-wise transform.
    int32x4_t buf0[64];
    col_txfm(input,
             buf0,
             stride,
             bitcol,
             /*lr_flip=*/0,
             /*howmany=*/2,
             /*hm_stride=*/32);
    shift_right_2_round_s32_x4(buf0, buf0, 64);

    int32x4_t buf1[64];
    transpose_arrays_s32_8x32(buf0, buf1);

    // Row-wise transform.
    row_txfm(buf1, buf0, bitrow, /*howmany=*/8, /*hm_stride=*/8);
    transpose_8xh(buf0, (int32x4_t *)coeff, 64);
}

static const fwd_transform_1d_row_neon row_highbd_txfm16_xn_arr[TX_TYPES] = {
    highbd_fdct16_row_neon, // DCT_DCT
    highbd_fdct16_row_neon, // ADST_DCT
    highbd_fadst16_row_neon, // DCT_ADST
    highbd_fadst16_row_neon, // ADST_ADST
    highbd_fdct16_row_neon, // FLIPADST_DCT
    highbd_fadst16_row_neon, // DCT_FLIPADST
    highbd_fadst16_row_neon, // FLIPADST_FLIPADST
    highbd_fadst16_row_neon, // ADST_FLIPADST
    highbd_fadst16_row_neon, // FLIPADST_ADST
    highbd_fidentity16_row_neon, // IDTX
    highbd_fidentity16_row_neon, // V_DCT
    highbd_fdct16_row_neon, // H_DCT
    highbd_fidentity16_row_neon, // V_ADST
    highbd_fadst16_row_neon, // H_ADST
    highbd_fidentity16_row_neon, // V_FLIPADST
    highbd_fadst16_row_neon // H_FLIPADST
};

static INLINE void transpose_elems_s32_16x4(const int32x4_t *in, int32x4_t *out) {
    transpose_elems_s32_4x4(in[0], in[1], in[2], in[3], &out[0], &out[4], &out[8], &out[12]);
    transpose_elems_s32_4x4(in[4], in[5], in[6], in[7], &out[1], &out[5], &out[9], &out[13]);
    transpose_elems_s32_4x4(in[8], in[9], in[10], in[11], &out[2], &out[6], &out[10], &out[14]);
    transpose_elems_s32_4x4(in[12], in[13], in[14], in[15], &out[3], &out[7], &out[11], &out[15]);
}

static INLINE void transpose_16xh(const int32x4_t *in, int32x4_t *out, int n) {
    for (int i = 0; i < n; i += 16) { transpose_elems_s32_16x4(in + i, out + i); }
}

void svt_av1_fwd_txfm2d_16x4_neon(int16_t *input, int32_t *coeff, uint32_t stride, TxType tx_type, uint8_t bd) {
    (void)bd;
    int                                  bitcol   = fwd_cos_bit_col[2][0];
    int                                  bitrow   = fwd_cos_bit_row[2][0];
    const fwd_transform_1d_col_many_neon col_txfm = col_highbd_txfm4_xn_arr[tx_type];
    const fwd_transform_1d_row_neon      row_txfm = row_highbd_txfm16_xn_arr[tx_type];

    int ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);
    ud_adjust_input_and_stride(ud_flip, &input, &stride, 4);

    // Column-wise transform.
    int32x4_t buf0[16];
    if (lr_flip) {
        col_txfm(input,
                 buf0 + 3 * 4,
                 stride,
                 bitcol,
                 /*lr_flip=*/1,
                 /*howmany=*/4,
                 /*hm_stride=*/-4);
    } else {
        col_txfm(input,
                 buf0,
                 stride,
                 bitcol,
                 /*lr_flip=*/0,
                 /*howmany=*/4,
                 /*hm_stride=*/4);
    }

    shift_right_1_round_s32_x4(buf0, buf0, 16);
    transpose_arrays_s32_4x16(buf0, buf0);

    int32x4_t buf1[16];
    // Row-wise transform.
    row_txfm(buf0, buf1, bitrow);

    transpose_16xh(buf1, (int32x4_t *)coeff, 16);
}

static const fwd_transform_1d_col_many_neon col_highbd_txfm8_xn_arr[TX_TYPES] = {
    highbd_fdct8_col_many_neon, // DCT_DCT
    highbd_fadst8_col_many_neon, // ADST_DCT
    highbd_fdct8_col_many_neon, // DCT_ADST
    highbd_fadst8_col_many_neon, // ADST_ADST
    highbd_fadst8_col_many_neon, // FLIPADST_DCT
    highbd_fdct8_col_many_neon, // DCT_FLIPADST
    highbd_fadst8_col_many_neon, // FLIPADST_FLIPADST
    highbd_fadst8_col_many_neon, // ADST_FLIPADST
    highbd_fadst8_col_many_neon, // FLIPADST_ADST
    highbd_fidentity8_col_many_neon, // IDTX
    highbd_fdct8_col_many_neon, // V_DCT
    highbd_fidentity8_col_many_neon, // H_DCT
    highbd_fadst8_col_many_neon, // V_ADST
    highbd_fidentity8_col_many_neon, // H_ADST
    highbd_fadst8_col_many_neon, // V_FLIPADST
    highbd_fidentity8_col_many_neon // H_FLIPADST
};

static const fwd_transform_1d_row_many_neon row_rect_highbd_txfm16_xn_arr[TX_TYPES] = {
    highbd_fdct16_row_rect_many_neon, // DCT_DCT
    highbd_fdct16_row_rect_many_neon, // ADST_DCT
    highbd_fadst16_row_rect_many_neon, // DCT_ADST
    highbd_fadst16_row_rect_many_neon, // ADST_ADST
    highbd_fdct16_row_rect_many_neon, // FLIPADST_DCT
    highbd_fadst16_row_rect_many_neon, // DCT_FLIPADST
    highbd_fadst16_row_rect_many_neon, // FLIPADST_FLIPADST
    highbd_fadst16_row_rect_many_neon, // ADST_FLIPADST
    highbd_fadst16_row_rect_many_neon, // FLIPADST_ADST
    highbd_fidentity16_row_rect_many_neon, // IDTX
    highbd_fidentity16_row_rect_many_neon, // V_DCT
    highbd_fdct16_row_rect_many_neon, // H_DCT
    highbd_fidentity16_row_rect_many_neon, // V_ADST
    highbd_fadst16_row_rect_many_neon, // H_ADST
    highbd_fidentity16_row_rect_many_neon, // V_FLIPADST
    highbd_fadst16_row_rect_many_neon // H_FLIPADST
};

void svt_av1_fwd_txfm2d_16x8_neon(int16_t *input, int32_t *coeff, uint32_t stride, TxType tx_type, uint8_t bd) {
    (void)bd;
    const fwd_transform_1d_col_many_neon col_txfm = col_highbd_txfm8_xn_arr[tx_type];
    const fwd_transform_1d_row_many_neon row_txfm = row_rect_highbd_txfm16_xn_arr[tx_type];
    int                                  bit      = fwd_cos_bit_col[2][1];

    int ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);
    ud_adjust_input_and_stride(ud_flip, &input, &stride, 8);

    // Column-wise transform.
    int32x4_t buf0[32];
    if (lr_flip) {
        col_txfm(input,
                 buf0 + 3 * 8,
                 stride,
                 bit,
                 /*lr_flip=*/1,
                 /*howmany=*/4,
                 /*hm_stride=*/-8);
    } else {
        col_txfm(input,
                 buf0,
                 stride,
                 bit,
                 /*lr_flip=*/0,
                 /*howmany=*/4,
                 /*hm_stride=*/8);
    }
    shift_right_2_round_s32_x4(buf0, buf0, 32);

    int32x4_t buf1[32];
    transpose_arrays_s32_16x8(buf0, buf1);

    // Row-wise transform.
    row_txfm(buf1, buf0, bit, /*howmany=*/2, /*hm_stride=*/16);

    transpose_16xh(buf0, (int32x4_t *)coeff, 32);
}

void svt_av1_fwd_txfm2d_16x32_neon(int16_t *input, int32_t *coeff, uint32_t stride, TxType tx_type, uint8_t bd) {
    (void)bd;
    const fwd_transform_1d_col_many_neon col_txfm = col_highbd_txfm32_x4_arr[tx_type];
    const fwd_transform_1d_row_many_neon row_txfm = row_rect_highbd_txfm16_xn_arr[tx_type];
    int                                  bitcol   = fwd_cos_bit_col[2][3];
    int                                  bitrow   = fwd_cos_bit_row[2][3];

    // Column-wise transform.
    int32x4_t buf0[128];
    col_txfm(input,
             buf0,
             stride,
             bitcol,
             /*lr_flip=*/0,
             /*howmany=*/4,
             /*hm_stride=*/32);
    shift_right_4_round_s32_x4(buf0, buf0, 128);

    int32x4_t buf1[128];
    transpose_arrays_s32_16x32(buf0, buf1);

    // Row-wise transform.
    row_txfm(buf1, buf0, bitrow, /*howmany=*/8, /*hm_stride=*/16);
    transpose_16xh(buf0, (int32x4_t *)coeff, 128);
}

static void highbd_fdct64_x4_neon(const int32x4_t *input, int32x4_t *output, int8_t cos_bit) {
    const int32_t *const cospi     = cospi_arr_s32(cos_bit);
    const int32x4_t      v_cos_bit = vdupq_n_s32(-cos_bit);

    // stage 1
    int32x4_t x1[64];
    butterfly_dct_pre(input, x1, 64);

    // stage 2
    int32x4_t x2[64];
    butterfly_dct_pre(x1, x2, 32);
    x2[32] = x1[32];
    x2[33] = x1[33];
    x2[34] = x1[34];
    x2[35] = x1[35];
    x2[36] = x1[36];
    x2[37] = x1[37];
    x2[38] = x1[38];
    x2[39] = x1[39];
    butterfly_0112_neon(cospi, 32, x1[55], x1[40], &x2[55], &x2[40], v_cos_bit);
    butterfly_0112_neon(cospi, 32, x1[54], x1[41], &x2[54], &x2[41], v_cos_bit);
    butterfly_0112_neon(cospi, 32, x1[53], x1[42], &x2[53], &x2[42], v_cos_bit);
    butterfly_0112_neon(cospi, 32, x1[52], x1[43], &x2[52], &x2[43], v_cos_bit);
    butterfly_0112_neon(cospi, 32, x1[51], x1[44], &x2[51], &x2[44], v_cos_bit);
    butterfly_0112_neon(cospi, 32, x1[50], x1[45], &x2[50], &x2[45], v_cos_bit);
    butterfly_0112_neon(cospi, 32, x1[49], x1[46], &x2[49], &x2[46], v_cos_bit);
    butterfly_0112_neon(cospi, 32, x1[48], x1[47], &x2[48], &x2[47], v_cos_bit);
    x2[56] = x1[56];
    x2[57] = x1[57];
    x2[58] = x1[58];
    x2[59] = x1[59];
    x2[60] = x1[60];
    x2[61] = x1[61];
    x2[62] = x1[62];
    x2[63] = x1[63];

    // stage 3
    int32x4_t x3[64];
    butterfly_dct_pre(x2, x3, 16);
    x3[16] = x2[16];
    x3[17] = x2[17];
    x3[18] = x2[18];
    x3[19] = x2[19];
    butterfly_0112_neon(cospi, 32, x2[27], x2[20], &x3[27], &x3[20], v_cos_bit);
    butterfly_0112_neon(cospi, 32, x2[26], x2[21], &x3[26], &x3[21], v_cos_bit);
    butterfly_0112_neon(cospi, 32, x2[25], x2[22], &x3[25], &x3[22], v_cos_bit);
    butterfly_0112_neon(cospi, 32, x2[24], x2[23], &x3[24], &x3[23], v_cos_bit);
    x3[28] = x2[28];
    x3[29] = x2[29];
    x3[30] = x2[30];
    x3[31] = x2[31];
    butterfly_dct_post(x2 + 32, x2 + 32, x3 + 32, 32);

    // stage 4
    int32x4_t x4[64];
    butterfly_dct_pre(x3, x4, 8);
    x4[8] = x3[8];
    x4[9] = x3[9];
    butterfly_0112_neon(cospi, 32, x3[13], x3[10], &x4[13], &x4[10], v_cos_bit);
    butterfly_0112_neon(cospi, 32, x3[12], x3[11], &x4[12], &x4[11], v_cos_bit);
    x4[14] = x3[14];
    x4[15] = x3[15];
    butterfly_dct_post(x3 + 16, x3 + 16, x4 + 16, 16);
    x4[32] = x3[32];
    x4[33] = x3[33];
    x4[34] = x3[34];
    x4[35] = x3[35];
    butterfly_0112_neon(cospi, 16, x3[59], x3[36], &x4[59], &x4[36], v_cos_bit);
    butterfly_0112_neon(cospi, 16, x3[58], x3[37], &x4[58], &x4[37], v_cos_bit);
    butterfly_0112_neon(cospi, 16, x3[57], x3[38], &x4[57], &x4[38], v_cos_bit);
    butterfly_0112_neon(cospi, 16, x3[56], x3[39], &x4[56], &x4[39], v_cos_bit);
    butterfly_2312_neon(cospi, 16, x3[55], x3[40], &x4[40], &x4[55], v_cos_bit);
    butterfly_2312_neon(cospi, 16, x3[54], x3[41], &x4[41], &x4[54], v_cos_bit);
    butterfly_2312_neon(cospi, 16, x3[53], x3[42], &x4[42], &x4[53], v_cos_bit);
    butterfly_2312_neon(cospi, 16, x3[52], x3[43], &x4[43], &x4[52], v_cos_bit);
    x4[44] = x3[44];
    x4[45] = x3[45];
    x4[46] = x3[46];
    x4[47] = x3[47];
    x4[48] = x3[48];
    x4[49] = x3[49];
    x4[50] = x3[50];
    x4[51] = x3[51];
    x4[60] = x3[60];
    x4[61] = x3[61];
    x4[62] = x3[62];
    x4[63] = x3[63];

    // stage 5
    int32x4_t x5[64];
    butterfly_dct_pre(x4, x5, 4);
    x5[4] = x4[4];
    butterfly_0112_neon(cospi, 32, x4[6], x4[5], &x5[6], &x5[5], v_cos_bit);
    x5[7] = x4[7];
    butterfly_dct_post(x4 + 8, x4 + 8, x5 + 8, 8);
    x5[16] = x4[16];
    x5[17] = x4[17];
    butterfly_0112_neon(cospi, 16, x4[29], x4[18], &x5[29], &x5[18], v_cos_bit);
    butterfly_0112_neon(cospi, 16, x4[28], x4[19], &x5[28], &x5[19], v_cos_bit);
    butterfly_2312_neon(cospi, 16, x4[27], x4[20], &x5[20], &x5[27], v_cos_bit);
    butterfly_2312_neon(cospi, 16, x4[26], x4[21], &x5[21], &x5[26], v_cos_bit);
    x5[22] = x4[22];
    x5[23] = x4[23];
    x5[24] = x4[24];
    x5[25] = x4[25];
    x5[30] = x4[30];
    x5[31] = x4[31];
    butterfly_dct_post(x4 + 32, x4 + 32, x5 + 32, 16);
    butterfly_dct_post(x4 + 48, x4 + 48, x5 + 48, 16);

    // stage 6
    int32x4_t x6[64];
    butterfly_0112_neon(cospi, 32, x5[0], x5[1], &x6[0], &x6[1], v_cos_bit);
    butterfly_0112_neon(cospi, 16, x5[3], x5[2], &x6[2], &x6[3], v_cos_bit);
    butterfly_dct_post(x5 + 4, x5 + 4, x6 + 4, 4);
    x6[8] = x5[8];
    butterfly_0112_neon(cospi, 16, x5[14], x5[9], &x6[14], &x6[9], v_cos_bit);
    butterfly_2312_neon(cospi, 16, x5[13], x5[10], &x6[10], &x6[13], v_cos_bit);
    x6[11] = x5[11];
    x6[12] = x5[12];
    x6[15] = x5[15];
    butterfly_dct_post(x5 + 16, x5 + 16, x6 + 16, 8);
    butterfly_dct_post(x5 + 24, x5 + 24, x6 + 24, 8);
    x6[32] = x5[32];
    x6[33] = x5[33];
    butterfly_0112_neon(cospi, 8, x5[61], x5[34], &x6[61], &x6[34], v_cos_bit);
    butterfly_0112_neon(cospi, 8, x5[60], x5[35], &x6[60], &x6[35], v_cos_bit);
    butterfly_2312_neon(cospi, 8, x5[59], x5[36], &x6[36], &x6[59], v_cos_bit);
    butterfly_2312_neon(cospi, 8, x5[58], x5[37], &x6[37], &x6[58], v_cos_bit);
    x6[38] = x5[38];
    x6[39] = x5[39];
    x6[40] = x5[40];
    x6[41] = x5[41];
    butterfly_0130_neon(cospi, 24, x5[42], x5[53], &x6[53], &x6[42], v_cos_bit);
    butterfly_0130_neon(cospi, 24, x5[43], x5[52], &x6[52], &x6[43], v_cos_bit);
    butterfly_0332_neon(cospi, 24, x5[51], x5[44], &x6[51], &x6[44], v_cos_bit);
    butterfly_0332_neon(cospi, 24, x5[50], x5[45], &x6[50], &x6[45], v_cos_bit);
    x6[46] = x5[46];
    x6[47] = x5[47];
    x6[48] = x5[48];
    x6[49] = x5[49];
    x6[54] = x5[54];
    x6[55] = x5[55];
    x6[56] = x5[56];
    x6[57] = x5[57];
    x6[62] = x5[62];
    x6[63] = x5[63];

    // stage 7
    int32x4_t x7[64];
    x7[0] = x6[0];
    x7[1] = x6[1];
    x7[2] = x6[2];
    x7[3] = x6[3];
    butterfly_0112_neon(cospi, 8, x6[7], x6[4], &x7[4], &x7[7], v_cos_bit);
    butterfly_0130_neon(cospi, 24, x6[5], x6[6], &x7[5], &x7[6], v_cos_bit);
    butterfly_dct_post(x6 + 8, x6 + 8, x7 + 8, 4);
    butterfly_dct_post(x6 + 12, x6 + 12, x7 + 12, 4);
    x7[16] = x6[16];
    butterfly_0112_neon(cospi, 8, x6[30], x6[17], &x7[30], &x7[17], v_cos_bit);
    butterfly_2312_neon(cospi, 8, x6[29], x6[18], &x7[18], &x7[29], v_cos_bit);
    x7[19] = x6[19];
    x7[20] = x6[20];
    butterfly_0130_neon(cospi, 24, x6[21], x6[26], &x7[26], &x7[21], v_cos_bit);
    butterfly_0332_neon(cospi, 24, x6[25], x6[22], &x7[25], &x7[22], v_cos_bit);
    x7[23] = x6[23];
    x7[24] = x6[24];
    x7[27] = x6[27];
    x7[28] = x6[28];
    x7[31] = x6[31];
    butterfly_dct_post(x6 + 32, x6 + 32, x7 + 32, 8);
    butterfly_dct_post(x6 + 40, x6 + 40, x7 + 40, 8);
    butterfly_dct_post(x6 + 48, x6 + 48, x7 + 48, 8);
    butterfly_dct_post(x6 + 56, x6 + 56, x7 + 56, 8);

    // stage 8
    int32x4_t x8[64];
    x8[0] = x7[0];
    x8[1] = x7[1];
    x8[2] = x7[2];
    x8[3] = x7[3];
    x8[4] = x7[4];
    x8[5] = x7[5];
    x8[6] = x7[6];
    x8[7] = x7[7];

    butterfly_0112_neon(cospi, 4, x7[15], x7[8], &x8[8], &x8[15], v_cos_bit);
    butterfly_0130_neon(cospi, 28, x7[9], x7[14], &x8[9], &x8[14], v_cos_bit);
    butterfly_0112_neon(cospi, 20, x7[13], x7[10], &x8[10], &x8[13], v_cos_bit);
    butterfly_0130_neon(cospi, 12, x7[11], x7[12], &x8[11], &x8[12], v_cos_bit);
    butterfly_dct_post(x7 + 16, x7 + 16, x8 + 16, 4);
    butterfly_dct_post(x7 + 20, x7 + 20, x8 + 20, 4);
    butterfly_dct_post(x7 + 24, x7 + 24, x8 + 24, 4);
    butterfly_dct_post(x7 + 28, x7 + 28, x8 + 28, 4);
    x8[32] = x7[32];
    butterfly_0112_neon(cospi, 4, x7[62], x7[33], &x8[62], &x8[33], v_cos_bit);
    butterfly_2312_neon(cospi, 4, x7[61], x7[34], &x8[34], &x8[61], v_cos_bit);
    x8[35] = x7[35];
    x8[36] = x7[36];
    butterfly_0130_neon(cospi, 28, x7[37], x7[58], &x8[58], &x8[37], v_cos_bit);
    butterfly_0332_neon(cospi, 28, x7[57], x7[38], &x8[57], &x8[38], v_cos_bit);
    x8[39] = x7[39];
    x8[40] = x7[40];
    butterfly_0112_neon(cospi, 20, x7[54], x7[41], &x8[54], &x8[41], v_cos_bit);
    butterfly_2312_neon(cospi, 20, x7[53], x7[42], &x8[42], &x8[53], v_cos_bit);
    x8[43] = x7[43];
    x8[44] = x7[44];
    butterfly_0130_neon(cospi, 12, x7[45], x7[50], &x8[50], &x8[45], v_cos_bit);
    butterfly_0332_neon(cospi, 12, x7[49], x7[46], &x8[49], &x8[46], v_cos_bit);
    x8[47] = x7[47];
    x8[48] = x7[48];
    x8[51] = x7[51];
    x8[52] = x7[52];
    x8[55] = x7[55];
    x8[56] = x7[56];
    x8[59] = x7[59];
    x8[60] = x7[60];
    x8[63] = x7[63];

    // stage 9
    int32x4_t x9[64];
    x9[0]  = x8[0];
    x9[1]  = x8[1];
    x9[2]  = x8[2];
    x9[3]  = x8[3];
    x9[4]  = x8[4];
    x9[5]  = x8[5];
    x9[6]  = x8[6];
    x9[7]  = x8[7];
    x9[8]  = x8[8];
    x9[9]  = x8[9];
    x9[10] = x8[10];
    x9[11] = x8[11];
    x9[12] = x8[12];
    x9[13] = x8[13];
    x9[14] = x8[14];
    x9[15] = x8[15];
    butterfly_0112_neon(cospi, 2, x8[31], x8[16], &x9[16], &x9[31], v_cos_bit);
    butterfly_0130_neon(cospi, 30, x8[17], x8[30], &x9[17], &x9[30], v_cos_bit);
    butterfly_0112_neon(cospi, 18, x8[29], x8[18], &x9[18], &x9[29], v_cos_bit);
    butterfly_0130_neon(cospi, 14, x8[19], x8[28], &x9[19], &x9[28], v_cos_bit);
    butterfly_0112_neon(cospi, 10, x8[27], x8[20], &x9[20], &x9[27], v_cos_bit);
    butterfly_0130_neon(cospi, 22, x8[21], x8[26], &x9[21], &x9[26], v_cos_bit);
    butterfly_0112_neon(cospi, 26, x8[25], x8[22], &x9[22], &x9[25], v_cos_bit);
    butterfly_0130_neon(cospi, 6, x8[23], x8[24], &x9[23], &x9[24], v_cos_bit);
    butterfly_dct_post(x8 + 32, x8 + 32, x9 + 32, 4);
    butterfly_dct_post(x8 + 36, x8 + 36, x9 + 36, 4);
    butterfly_dct_post(x8 + 40, x8 + 40, x9 + 40, 4);
    butterfly_dct_post(x8 + 44, x8 + 44, x9 + 44, 4);
    butterfly_dct_post(x8 + 48, x8 + 48, x9 + 48, 4);
    butterfly_dct_post(x8 + 52, x8 + 52, x9 + 52, 4);
    butterfly_dct_post(x8 + 56, x8 + 56, x9 + 56, 4);
    butterfly_dct_post(x8 + 60, x8 + 60, x9 + 60, 4);

    // stage 10
    int32x4_t x10[64];
    x10[0]  = x9[0];
    x10[1]  = x9[1];
    x10[2]  = x9[2];
    x10[3]  = x9[3];
    x10[4]  = x9[4];
    x10[5]  = x9[5];
    x10[6]  = x9[6];
    x10[7]  = x9[7];
    x10[8]  = x9[8];
    x10[9]  = x9[9];
    x10[10] = x9[10];
    x10[11] = x9[11];
    x10[12] = x9[12];
    x10[13] = x9[13];
    x10[14] = x9[14];
    x10[15] = x9[15];
    x10[16] = x9[16];
    x10[17] = x9[17];
    x10[18] = x9[18];
    x10[19] = x9[19];
    x10[20] = x9[20];
    x10[21] = x9[21];
    x10[22] = x9[22];
    x10[23] = x9[23];
    x10[24] = x9[24];
    x10[25] = x9[25];
    x10[26] = x9[26];
    x10[27] = x9[27];
    x10[28] = x9[28];
    x10[29] = x9[29];
    x10[30] = x9[30];
    x10[31] = x9[31];
    butterfly_0112_neon(cospi, 1, x9[63], x9[32], &x10[32], &x10[63], v_cos_bit);
    butterfly_0130_neon(cospi, 31, x9[33], x9[62], &x10[33], &x10[62], v_cos_bit);
    butterfly_0112_neon(cospi, 17, x9[61], x9[34], &x10[34], &x10[61], v_cos_bit);
    butterfly_0130_neon(cospi, 15, x9[35], x9[60], &x10[35], &x10[60], v_cos_bit);
    butterfly_0112_neon(cospi, 9, x9[59], x9[36], &x10[36], &x10[59], v_cos_bit);
    butterfly_0130_neon(cospi, 23, x9[37], x9[58], &x10[37], &x10[58], v_cos_bit);
    butterfly_0112_neon(cospi, 25, x9[57], x9[38], &x10[38], &x10[57], v_cos_bit);
    butterfly_0130_neon(cospi, 7, x9[39], x9[56], &x10[39], &x10[56], v_cos_bit);
    butterfly_0112_neon(cospi, 5, x9[55], x9[40], &x10[40], &x10[55], v_cos_bit);
    butterfly_0130_neon(cospi, 27, x9[41], x9[54], &x10[41], &x10[54], v_cos_bit);
    butterfly_0112_neon(cospi, 21, x9[53], x9[42], &x10[42], &x10[53], v_cos_bit);
    butterfly_0130_neon(cospi, 11, x9[43], x9[52], &x10[43], &x10[52], v_cos_bit);
    butterfly_0112_neon(cospi, 13, x9[51], x9[44], &x10[44], &x10[51], v_cos_bit);
    butterfly_0130_neon(cospi, 19, x9[45], x9[50], &x10[45], &x10[50], v_cos_bit);
    butterfly_0112_neon(cospi, 29, x9[49], x9[46], &x10[46], &x10[49], v_cos_bit);
    butterfly_0130_neon(cospi, 3, x9[47], x9[48], &x10[47], &x10[48], v_cos_bit);

    // stage 11
    output[0]  = x10[0];
    output[1]  = x10[32];
    output[2]  = x10[16];
    output[3]  = x10[48];
    output[4]  = x10[8];
    output[5]  = x10[40];
    output[6]  = x10[24];
    output[7]  = x10[56];
    output[8]  = x10[4];
    output[9]  = x10[36];
    output[10] = x10[20];
    output[11] = x10[52];
    output[12] = x10[12];
    output[13] = x10[44];
    output[14] = x10[28];
    output[15] = x10[60];
    output[16] = x10[2];
    output[17] = x10[34];
    output[18] = x10[18];
    output[19] = x10[50];
    output[20] = x10[10];
    output[21] = x10[42];
    output[22] = x10[26];
    output[23] = x10[58];
    output[24] = x10[6];
    output[25] = x10[38];
    output[26] = x10[22];
    output[27] = x10[54];
    output[28] = x10[14];
    output[29] = x10[46];
    output[30] = x10[30];
    output[31] = x10[62];
    output[32] = x10[1];
    output[33] = x10[33];
    output[34] = x10[17];
    output[35] = x10[49];
    output[36] = x10[9];
    output[37] = x10[41];
    output[38] = x10[25];
    output[39] = x10[57];
    output[40] = x10[5];
    output[41] = x10[37];
    output[42] = x10[21];
    output[43] = x10[53];
    output[44] = x10[13];
    output[45] = x10[45];
    output[46] = x10[29];
    output[47] = x10[61];
    output[48] = x10[3];
    output[49] = x10[35];
    output[50] = x10[19];
    output[51] = x10[51];
    output[52] = x10[11];
    output[53] = x10[43];
    output[54] = x10[27];
    output[55] = x10[59];
    output[56] = x10[7];
    output[57] = x10[39];
    output[58] = x10[23];
    output[59] = x10[55];
    output[60] = x10[15];
    output[61] = x10[47];
    output[62] = x10[31];
    output[63] = x10[63];
}

void svt_av1_fwd_txfm2d_16x64_neon(int16_t *input, int32_t *coeff, uint32_t stride, TxType tx_type, uint8_t bd) {
    (void)bd;
    const int bitcol = fwd_cos_bit_col[2][4];
    const int bitrow = fwd_cos_bit_row[2][4];

    int ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);
    ud_adjust_input_and_stride(ud_flip, &input, &stride, 64);

    // Column-wise transform.
    int32x4_t buf0[256];
    load_buffer_16x64_(input, buf0, stride, lr_flip);
    for (int i = 0; i < 4; i++) { highbd_fdct64_x4_neon(buf0 + i * 64, buf0 + i * 64, bitcol); }
    shift_right_2_round_s32_x4(buf0, buf0, 256);

    int32x4_t buf1[256];
    transpose_arrays_s32_16x64(buf0, buf1);

    // Row-wise transform.
    highbd_fdct16_xn_neon(buf1, buf1, bitrow, 16);
    transpose_16xh(buf1, (int32x4_t *)coeff, 256);
}

TRANSFORM_ROW_MANY(fdct32, 32)
TRANSFORM_ROW_MANY(fidentity32, 32)

static const fwd_transform_1d_row_many_neon row_highbd_txfm32_x4_arr[TX_TYPES] = {
    highbd_fdct32_row_many_neon, // DCT_DCT
    NULL, // ADST_DCT
    NULL, // DCT_ADST
    NULL, // ADST_ADST
    NULL, // FLIPADST_DCT
    NULL, // DCT_FLIPADST
    NULL, // FLIPADST_FLIPADST
    NULL, // ADST_FLIPADST
    NULL, // FLIPADST_ADST
    highbd_fidentity32_row_many_neon, // IDTX
    highbd_fidentity32_row_many_neon, // V_DCT
    highbd_fdct32_row_many_neon, // H_DCT
    NULL, // V_ADST
    NULL, // H_ADST
    NULL, // V_FLIPADST
    NULL // H_FLIPADST
};

static INLINE void transpose_elems_s32_32x4(const int32x4_t *in, int32x4_t *out) {
    transpose_elems_s32_4x4(in[0], in[1], in[2], in[3], &out[0], &out[8], &out[16], &out[24]);
    transpose_elems_s32_4x4(in[4], in[5], in[6], in[7], &out[1], &out[9], &out[17], &out[25]);
    transpose_elems_s32_4x4(in[8], in[9], in[10], in[11], &out[2], &out[10], &out[18], &out[26]);
    transpose_elems_s32_4x4(in[12], in[13], in[14], in[15], &out[3], &out[11], &out[19], &out[27]);
    transpose_elems_s32_4x4(in[16], in[17], in[18], in[19], &out[4], &out[12], &out[20], &out[28]);
    transpose_elems_s32_4x4(in[20], in[21], in[22], in[23], &out[5], &out[13], &out[21], &out[29]);
    transpose_elems_s32_4x4(in[24], in[25], in[26], in[27], &out[6], &out[14], &out[22], &out[30]);
    transpose_elems_s32_4x4(in[28], in[29], in[30], in[31], &out[7], &out[15], &out[23], &out[31]);
}

static INLINE void transpose_32xh(const int32x4_t *in, int32x4_t *out, int n) {
    for (int i = 0; i < n; i += 32) { transpose_elems_s32_32x4(in + i, out + i); }
}

void svt_av1_fwd_txfm2d_32x8_neon(int16_t *input, int32_t *output, uint32_t stride, TxType tx_type, uint8_t bd) {
    (void)bd;
    const fwd_transform_1d_col_many_neon col_txfm = col_highbd_txfm8_xn_arr[tx_type];
    const fwd_transform_1d_row_many_neon row_txfm = row_highbd_txfm32_x4_arr[tx_type];
    int                                  bitcol   = fwd_cos_bit_col[3][1];
    int                                  bitrow   = fwd_cos_bit_row[3][1];

    // Column-wise transform.
    int32x4_t buf0[64];
    col_txfm(input,
             buf0,
             stride,
             bitcol,
             /*lr_flip=*/0,
             /*howmany=*/8,
             /*hm_stride=*/8);
    shift_right_2_round_s32_x4(buf0, buf0, 64);

    int32x4_t buf1[64];
    transpose_arrays_s32_32x8(buf0, buf1);

    // Row-wise transform.
    row_txfm(buf1, buf0, bitrow, /*howmany=*/2, /*hm_stride=*/32);
    transpose_32xh(buf0, (int32x4_t *)output, 64);
}

TRANSFORM_ROW_RECT_MANY(fdct32, 32)
TRANSFORM_ROW_RECT_MANY(fidentity32, 32)

static const fwd_transform_1d_row_many_neon row_rect_highbd_txfm32_x4_arr[TX_TYPES] = {
    highbd_fdct32_row_rect_many_neon, // DCT_DCT
    NULL, // ADST_DCT
    NULL, // DCT_ADST
    NULL, // ADST_ADST
    NULL, // FLIPADST_DCT
    NULL, // DCT_FLIPADST
    NULL, // FLIPADST_FLIPADST
    NULL, // ADST_FLIPADST
    NULL, // FLIPADST_ADST
    highbd_fidentity32_row_rect_many_neon, // IDTX
    NULL, // V_DCT
    NULL, // H_DCT
    NULL, // V_ADST
    NULL, // H_ADST
    NULL, // V_FLIPADST
    NULL // H_FLIPADST
};

void svt_av1_fwd_txfm2d_32x16_neon(int16_t *input, int32_t *output, uint32_t stride, TxType tx_type, uint8_t bd) {
    (void)bd;
    const fwd_transform_1d_col_many_neon col_txfm = col_highbd_txfm16_xn_arr[tx_type];
    const fwd_transform_1d_row_many_neon row_txfm = row_rect_highbd_txfm32_x4_arr[tx_type];
    int                                  bitcol   = fwd_cos_bit_col[3][2];
    int                                  bitrow   = fwd_cos_bit_row[3][2];

    // Column-wise transform.
    int32x4_t buf0[128];
    col_txfm(input,
             buf0,
             stride,
             bitcol,
             /*lr_flip=*/0,
             /*howmany=*/8,
             /*hm_stride=*/16);
    shift_right_4_round_s32_x4(buf0, buf0, 128);

    int32x4_t buf1[128];
    transpose_arrays_s32_32x16(buf0, buf1);

    // Row-wise transform.
    row_txfm(buf1, buf0, bitrow, /*howmany=*/4, /*hm_stride=*/32);
    transpose_32xh(buf0, (int32x4_t *)output, 128);
}

void svt_av1_fwd_txfm2d_32x32_neon(int16_t *input, int32_t *output, uint32_t stride, TxType tx_type, uint8_t bd) {
    (void)bd;
    const fwd_transform_1d_col_many_neon col_txfm = col_highbd_txfm32_x4_arr[tx_type];
    const fwd_transform_1d_row_many_neon row_txfm = row_highbd_txfm32_x4_arr[tx_type];

    // Column-wise transform.
    int32x4_t buf0[256];
    col_txfm(input,
             buf0,
             stride,
             /*cos_bit=*/12,
             /*lr_flip=*/0,
             /*howmany=*/8,
             /*hm_stride=*/32);
    shift_right_4_round_s32_x4(buf0, buf0, 256);

    int32x4_t buf1[256];
    transpose_arrays_s32_32x32(buf0, buf1);

    // Row-wise transform.
    row_txfm(buf1, buf0, /*cos_bit=*/12, /*howmany=*/8, /*hm_stride=*/32);
    transpose_32xh(buf0, (int32x4_t *)output, 256);
}

static AOM_FORCE_INLINE void round_shift2_rect_array_s32_neon(const int32x4_t *input, int32x4_t *output,
                                                              const int size) {
    const int32x4_t sqrt2 = vdupq_n_s32(new_sqrt2);
    int             i     = 0;
    do {
        const int32x4_t r0 = vrshrq_n_s32(input[i], 2);
        const int32x4_t r1 = vmulq_s32(r0, sqrt2);
        output[i]          = vrshrq_n_s32(r1, new_sqrt2_bits);
    } while (++i < size);
}

void svt_av1_fwd_txfm2d_32x64_neon(int16_t *input, int32_t *output, uint32_t stride, TxType tx_type, uint8_t bd) {
    (void)bd;
    (void)tx_type;
    int bitcol = fwd_cos_bit_col[3][4];
    int bitrow = fwd_cos_bit_row[3][4];

    // Column-wise transform.
    int32x4_t buf0[512];
    load_buffer_32x64_(input, buf0, stride, 0);
    for (int i = 0; i < 8; i++) { highbd_fdct64_x4_neon(buf0 + i * 64, buf0 + i * 64, bitcol); }
    shift_right_2_round_s32_x4(buf0, buf0, 512);

    int32x4_t buf1[512];
    transpose_arrays_s32_32x64(buf0, buf1);

    // Row-wise transform.
    for (int i = 0; i < 16; i++) { highbd_fdct32_x4_neon(buf1 + i * 32, buf1 + i * 32, bitrow); }
    round_shift2_rect_array_s32_neon(buf1, buf1, 512);
    transpose_32xh(buf1, (int32x4_t *)output, 512);
}

static INLINE void transpose_elems_s32_64x4(const int32x4_t *in, int32x4_t *out) {
    transpose_elems_s32_4x4(in[0], in[1], in[2], in[3], &out[0], &out[16], &out[32], &out[48]);
    transpose_elems_s32_4x4(in[4], in[5], in[6], in[7], &out[1], &out[17], &out[33], &out[49]);
    transpose_elems_s32_4x4(in[8], in[9], in[10], in[11], &out[2], &out[18], &out[34], &out[50]);
    transpose_elems_s32_4x4(in[12], in[13], in[14], in[15], &out[3], &out[19], &out[35], &out[51]);
    transpose_elems_s32_4x4(in[16], in[17], in[18], in[19], &out[4], &out[20], &out[36], &out[52]);
    transpose_elems_s32_4x4(in[20], in[21], in[22], in[23], &out[5], &out[21], &out[37], &out[53]);
    transpose_elems_s32_4x4(in[24], in[25], in[26], in[27], &out[6], &out[22], &out[38], &out[54]);
    transpose_elems_s32_4x4(in[28], in[29], in[30], in[31], &out[7], &out[23], &out[39], &out[55]);
    transpose_elems_s32_4x4(in[32], in[33], in[34], in[35], &out[8], &out[24], &out[40], &out[56]);
    transpose_elems_s32_4x4(in[36], in[37], in[38], in[39], &out[9], &out[25], &out[41], &out[57]);
    transpose_elems_s32_4x4(in[40], in[41], in[42], in[43], &out[10], &out[26], &out[42], &out[58]);
    transpose_elems_s32_4x4(in[44], in[45], in[46], in[47], &out[11], &out[27], &out[43], &out[59]);
    transpose_elems_s32_4x4(in[48], in[49], in[50], in[51], &out[12], &out[28], &out[44], &out[60]);
    transpose_elems_s32_4x4(in[52], in[53], in[54], in[55], &out[13], &out[29], &out[45], &out[61]);
    transpose_elems_s32_4x4(in[56], in[57], in[58], in[59], &out[14], &out[30], &out[46], &out[62]);
    transpose_elems_s32_4x4(in[60], in[61], in[62], in[63], &out[15], &out[31], &out[47], &out[63]);
}

static INLINE void transpose_64xh(const int32x4_t *in, int32x4_t *out, int n) {
    for (int i = 0; i < n; i += 64) { transpose_elems_s32_64x4(in + i, out + i); }
}

void svt_av1_fwd_txfm2d_64x16_neon(int16_t *input, int32_t *output, uint32_t stride, TxType tx_type, uint8_t bd) {
    (void)bd;
    const int bitcol = fwd_cos_bit_col[4][2];
    const int bitrow = fwd_cos_bit_row[4][2];

    int ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);
    ud_adjust_input_and_stride(ud_flip, &input, &stride, 16);

    // Column-wise transform.
    int32x4_t buf0[256];
    load_buffer_64x16_(input, buf0, stride, lr_flip);
    highbd_fdct16_xn_neon(buf0, buf0, bitcol, 16);
    shift_right_4_round_s32_x4(buf0, buf0, 256);

    int32x4_t buf1[256];
    transpose_arrays_s32_64x16(buf0, buf1);

    // Row-wise transform.
    for (int i = 0; i < 4; i++) { highbd_fdct64_x4_neon(buf1 + i * 64, buf1 + i * 64, bitrow); }
    transpose_64xh(buf1, (int32x4_t *)output, 256);
}

void svt_av1_fwd_txfm2d_64x32_neon(int16_t *input, int32_t *output, uint32_t stride, TxType tx_type, uint8_t bd) {
    (void)bd;
    (void)tx_type;
    int bitcol = fwd_cos_bit_col[4][3];
    int bitrow = fwd_cos_bit_row[4][3];

    // Column-wise transform.
    int32x4_t buf0[512];
    load_buffer_64x32_(input, buf0, stride, 0);
    for (int i = 0; i < 16; i++) { highbd_fdct32_x4_neon(buf0 + i * 32, buf0 + i * 32, bitcol); }
    shift_right_4_round_s32_x4(buf0, buf0, 512);

    int32x4_t buf1[512];
    transpose_arrays_s32_64x32(buf0, buf1);

    // Row-wise transform.
    for (int i = 0; i < 8; i++) { highbd_fdct64_x4_neon(buf1 + i * 64, buf1 + i * 64, bitrow); }
    round_shift2_rect_array_s32_neon(buf1, buf1, 512);
    transpose_64xh(buf1, (int32x4_t *)output, 512);
}

TRANSFORM_COL_MANY(fdct64, 64)
TRANSFORM_ROW_MANY(fdct64, 64)

static INLINE void load_buffer_64x64_neon(const int16_t *input, int32_t stride, int32x4_t *output) {
    int32_t i;

    for (i = 0; i < 64; ++i) {
        output[0]  = vmovl_s16(vld1_s16(input + 0 * 4));
        output[1]  = vmovl_s16(vld1_s16(input + 1 * 4));
        output[2]  = vmovl_s16(vld1_s16(input + 2 * 4));
        output[3]  = vmovl_s16(vld1_s16(input + 3 * 4));
        output[4]  = vmovl_s16(vld1_s16(input + 4 * 4));
        output[5]  = vmovl_s16(vld1_s16(input + 5 * 4));
        output[6]  = vmovl_s16(vld1_s16(input + 6 * 4));
        output[7]  = vmovl_s16(vld1_s16(input + 7 * 4));
        output[8]  = vmovl_s16(vld1_s16(input + 8 * 4));
        output[9]  = vmovl_s16(vld1_s16(input + 9 * 4));
        output[10] = vmovl_s16(vld1_s16(input + 10 * 4));
        output[11] = vmovl_s16(vld1_s16(input + 11 * 4));
        output[12] = vmovl_s16(vld1_s16(input + 12 * 4));
        output[13] = vmovl_s16(vld1_s16(input + 13 * 4));
        output[14] = vmovl_s16(vld1_s16(input + 14 * 4));
        output[15] = vmovl_s16(vld1_s16(input + 15 * 4));

        input += stride;
        output += 16;
    }
}

static INLINE void fidtx64x64_neon(int32x4_t *input, int32x4_t *output, const int8_t cos_bit,
                                   const int8_t *stage_range) {
    (void)cos_bit;
    (void)stage_range;
    const int32_t   bits    = 12; // new_sqrt2_bits = 12
    const int32_t   sqrt    = 4 * 5793; // 4 * new_sqrt2
    const int32_t   col_num = 16;
    const int32x4_t newsqrt = vdupq_n_s32(sqrt);

    const int32_t num_iters = 64 * col_num;
    for (int32_t i = 0; i < num_iters; i++) {
        int32x4_t temp = vmulq_s32(input[i], newsqrt);
        output[i]      = vrshlq_s32(temp, vdupq_n_s32(-bits));
    }
}

void svt_av1_fwd_txfm2d_64x64_neon(int16_t *input, int32_t *output, uint32_t stride, TxType tx_type, uint8_t bd) {
    (void)bd;

    switch (tx_type) {
    case DCT_DCT: {
        // Column-wise transform.
        int32x4_t buf0[1024];
        highbd_fdct64_col_many_neon(input, buf0, stride, 13, /*lr_flip=*/0, /*howmany=*/16, /*hm_stride=*/64);
        shift_right_2_round_s32_x4(buf0, buf0, 1024);

        int32x4_t buf1[1024];
        transpose_arrays_s32_64x64(buf0, buf1);

        // Row-wise transform.
        highbd_fdct64_row_many_neon(buf1, buf0, 10, /*howmany=*/16, /*hm_stride=*/64);
        shift_right_2_round_s32_x4(buf0, buf0, 1024);
        transpose_64xh(buf0, (int32x4_t *)output, 1024);
        break;
    }
    case IDTX: {
        int32x4_t buf0[1024];
        load_buffer_64x64_neon(input, stride, buf0);

        // Column-wise transform.
        int32x4_t buf1[1024];
        fidtx64x64_neon(buf0, buf1, 13, NULL);
        shift_right_2_round_s32_x4(buf1, buf1, 1024);

        // Row-wise transform.
        fidtx64x64_neon(buf1, buf0, 10, NULL);
        shift_right_2_round_s32_x4(buf0, (int32x4_t *)output, 1024);

        break;
    }
    default: assert(0);
    }
}

void svt_aom_transform_config(TxType tx_type, TxSize tx_size, Txfm2dFlipCfg *cfg);

static const int8_t *fwd_txfm_shift_ls[TX_SIZES_ALL] = {
    fwd_shift_4x4,  fwd_shift_8x8,  fwd_shift_16x16, fwd_shift_32x32, fwd_shift_64x64, fwd_shift_4x8,   fwd_shift_8x4,
    fwd_shift_8x16, fwd_shift_16x8, fwd_shift_16x32, fwd_shift_32x16, fwd_shift_32x64, fwd_shift_64x32, fwd_shift_4x16,
    fwd_shift_16x4, fwd_shift_8x32, fwd_shift_32x8,  fwd_shift_16x64, fwd_shift_64x16,
};

#define btf_32_neon_type0(w0, w1, in0, in1, out0, out1, v_cos_bit) \
    do {                                                           \
        out0 = vmulq_n_s32(in0, w0);                               \
        out0 = vmlaq_n_s32(out0, in1, w1);                         \
        out0 = vrshlq_s32(out0, v_cos_bit);                        \
        out1 = vmulq_n_s32(in0, w1);                               \
        out1 = vmlsq_n_s32(out1, in1, w0);                         \
        out1 = vrshlq_s32(out1, v_cos_bit);                        \
    } while (0)

#define btf_32_neon_type1(w0, w1, in0, in1, out0, out1, bit) \
    do { btf_32_neon_type0(w1, w0, in1, in0, out0, out1, bit); } while (0)

static INLINE void load_buffer_4x4(const int16_t *input, int32x4_t *in, uint32_t stride, int flipud, int fliplr,
                                   const int32x4_t *v_shift) {
    int16x4_t v0, v1, v2, v3;

    if (!flipud) {
        v0 = vld1_s16(input + 0 * stride);
        v1 = vld1_s16(input + 1 * stride);
        v2 = vld1_s16(input + 2 * stride);
        v3 = vld1_s16(input + 3 * stride);
    } else {
        v0 = vld1_s16(input + 3 * stride);
        v1 = vld1_s16(input + 2 * stride);
        v2 = vld1_s16(input + 1 * stride);
        v3 = vld1_s16(input + 0 * stride);
    }

    if (fliplr) {
        v0 = vrev64_s16(v0);
        v1 = vrev64_s16(v1);
        v2 = vrev64_s16(v2);
        v3 = vrev64_s16(v3);
    }
    in[0] = vshlq_s32(vmovl_s16(v0), *v_shift);
    in[1] = vshlq_s32(vmovl_s16(v1), *v_shift);
    in[2] = vshlq_s32(vmovl_s16(v2), *v_shift);
    in[3] = vshlq_s32(vmovl_s16(v3), *v_shift);
}

static INLINE void load_buffer_8x8(const int16_t *input, int32x4_t *in, uint32_t stride, int flipud, int fliplr,
                                   const int shift) {
    if (!flipud) {
        in[0] = vreinterpretq_s32_s16(vld1q_s16((input + 0 * stride)));
        in[1] = vreinterpretq_s32_s16(vld1q_s16((input + 1 * stride)));
        in[2] = vreinterpretq_s32_s16(vld1q_s16((input + 2 * stride)));
        in[3] = vreinterpretq_s32_s16(vld1q_s16((input + 3 * stride)));
        in[4] = vreinterpretq_s32_s16(vld1q_s16((input + 4 * stride)));
        in[5] = vreinterpretq_s32_s16(vld1q_s16((input + 5 * stride)));
        in[6] = vreinterpretq_s32_s16(vld1q_s16((input + 6 * stride)));
        in[7] = vreinterpretq_s32_s16(vld1q_s16((input + 7 * stride)));
    } else {
        in[0] = vreinterpretq_s32_s16(vld1q_s16((input + 7 * stride)));
        in[1] = vreinterpretq_s32_s16(vld1q_s16((input + 6 * stride)));
        in[2] = vreinterpretq_s32_s16(vld1q_s16((input + 5 * stride)));
        in[3] = vreinterpretq_s32_s16(vld1q_s16((input + 4 * stride)));
        in[4] = vreinterpretq_s32_s16(vld1q_s16((input + 3 * stride)));
        in[5] = vreinterpretq_s32_s16(vld1q_s16((input + 2 * stride)));
        in[6] = vreinterpretq_s32_s16(vld1q_s16((input + 1 * stride)));
        in[7] = vreinterpretq_s32_s16(vld1q_s16((input + 0 * stride)));
    }

    if (fliplr) {
        in[0] = vreinterpretq_s32_s16(vrev64q_s16(vreinterpretq_s16_s32(in[0])));
        in[0] = vextq_s32(in[0], in[0], 2);
        in[1] = vreinterpretq_s32_s16(vrev64q_s16(vreinterpretq_s16_s32(in[1])));
        in[1] = vextq_s32(in[1], in[1], 2);
        in[2] = vreinterpretq_s32_s16(vrev64q_s16(vreinterpretq_s16_s32(in[2])));
        in[2] = vextq_s32(in[2], in[2], 2);
        in[3] = vreinterpretq_s32_s16(vrev64q_s16(vreinterpretq_s16_s32(in[3])));
        in[3] = vextq_s32(in[3], in[3], 2);
        in[4] = vreinterpretq_s32_s16(vrev64q_s16(vreinterpretq_s16_s32(in[4])));
        in[4] = vextq_s32(in[4], in[4], 2);
        in[5] = vreinterpretq_s32_s16(vrev64q_s16(vreinterpretq_s16_s32(in[5])));
        in[5] = vextq_s32(in[5], in[5], 2);
        in[6] = vreinterpretq_s32_s16(vrev64q_s16(vreinterpretq_s16_s32(in[6])));
        in[6] = vextq_s32(in[6], in[6], 2);
        in[7] = vreinterpretq_s32_s16(vrev64q_s16(vreinterpretq_s16_s32(in[7])));
        in[7] = vextq_s32(in[7], in[7], 2);
    }

    int16x4_t u = vget_high_s16(vreinterpretq_s16_s32(in[4]));
    in[8]       = vmovl_s16(vget_low_s16(vreinterpretq_s16_s32(in[4])));
    in[9]       = vmovl_s16(u);

    u      = vget_high_s16(vreinterpretq_s16_s32(in[5]));
    in[10] = vmovl_s16(vget_low_s16(vreinterpretq_s16_s32(in[5])));
    in[11] = vmovl_s16(u);

    u      = vget_high_s16(vreinterpretq_s16_s32(in[6]));
    in[12] = vmovl_s16(vget_low_s16(vreinterpretq_s16_s32(in[6])));
    in[13] = vmovl_s16(u);

    u      = vget_high_s16(vreinterpretq_s16_s32(in[7]));
    in[14] = vmovl_s16(vget_low_s16(vreinterpretq_s16_s32(in[7])));
    in[15] = vmovl_s16(u);

    u     = vget_high_s16(vreinterpretq_s16_s32(in[3]));
    in[6] = vmovl_s16(vget_low_s16(vreinterpretq_s16_s32(in[3])));
    in[7] = vmovl_s16(u);

    u     = vget_high_s16(vreinterpretq_s16_s32(in[2]));
    in[4] = vmovl_s16(vget_low_s16(vreinterpretq_s16_s32(in[2])));
    in[5] = vmovl_s16(u);

    u     = vget_high_s16(vreinterpretq_s16_s32(in[1]));
    in[2] = vmovl_s16(vget_low_s16(vreinterpretq_s16_s32(in[1])));
    in[3] = vmovl_s16(u);

    u     = vget_high_s16(vreinterpretq_s16_s32(in[0]));
    in[0] = vmovl_s16(vget_low_s16(vreinterpretq_s16_s32(in[0])));
    in[1] = vmovl_s16(u);

    const int32x4_t v_shift = vdupq_n_s32(shift);

    in[0] = vshlq_s32(in[0], v_shift);
    in[1] = vshlq_s32(in[1], v_shift);
    in[2] = vshlq_s32(in[2], v_shift);
    in[3] = vshlq_s32(in[3], v_shift);
    in[4] = vshlq_s32(in[4], v_shift);
    in[5] = vshlq_s32(in[5], v_shift);
    in[6] = vshlq_s32(in[6], v_shift);
    in[7] = vshlq_s32(in[7], v_shift);

    in[8]  = vshlq_s32(in[8], v_shift);
    in[9]  = vshlq_s32(in[9], v_shift);
    in[10] = vshlq_s32(in[10], v_shift);
    in[11] = vshlq_s32(in[11], v_shift);
    in[12] = vshlq_s32(in[12], v_shift);
    in[13] = vshlq_s32(in[13], v_shift);
    in[14] = vshlq_s32(in[14], v_shift);
    in[15] = vshlq_s32(in[15], v_shift);
}

static INLINE void col_txfm_8x8_rounding(int32x4_t *in, const int32x4_t *v_shift) {
    in[0]  = vrshlq_s32(in[0], *v_shift);
    in[1]  = vrshlq_s32(in[1], *v_shift);
    in[2]  = vrshlq_s32(in[2], *v_shift);
    in[3]  = vrshlq_s32(in[3], *v_shift);
    in[4]  = vrshlq_s32(in[4], *v_shift);
    in[5]  = vrshlq_s32(in[5], *v_shift);
    in[6]  = vrshlq_s32(in[6], *v_shift);
    in[7]  = vrshlq_s32(in[7], *v_shift);
    in[8]  = vrshlq_s32(in[8], *v_shift);
    in[9]  = vrshlq_s32(in[9], *v_shift);
    in[10] = vrshlq_s32(in[10], *v_shift);
    in[11] = vrshlq_s32(in[11], *v_shift);
    in[12] = vrshlq_s32(in[12], *v_shift);
    in[13] = vrshlq_s32(in[13], *v_shift);
    in[14] = vrshlq_s32(in[14], *v_shift);
    in[15] = vrshlq_s32(in[15], *v_shift);
}

// Hybrid Transform 16x16
static INLINE void convert_8x8_to_16x16(const int32x4_t *in, int32x4_t *out) {
    int row_index = 0;
    int dst_index = 0;
    int src_index = 0;

    // row 0, 1, .., 7
    do {
        out[dst_index]     = in[src_index];
        out[dst_index + 1] = in[src_index + 1];
        out[dst_index + 2] = in[src_index + 16];
        out[dst_index + 3] = in[src_index + 17];
        dst_index += 4;
        src_index += 2;
        row_index += 1;
    } while (row_index < 8);

    // row 8, 9, ..., 15
    src_index += 16;
    do {
        out[dst_index]     = in[src_index];
        out[dst_index + 1] = in[src_index + 1];
        out[dst_index + 2] = in[src_index + 16];
        out[dst_index + 3] = in[src_index + 17];
        dst_index += 4;
        src_index += 2;
        row_index += 1;
    } while (row_index < 16);
}

static INLINE void load_buffer_16x16(const int16_t *input, int32x4_t *out, uint32_t stride, int flipud, int fliplr,
                                     int shift) {
    int32x4_t in[64];
    // Load 4 8x8 blocks
    const int16_t *topL = input;
    const int16_t *topR = input + 8;
    const int16_t *botL = input + 8 * stride;
    const int16_t *botR = input + 8 * stride + 8;

    const int16_t *tmp;

    if (flipud) {
        // Swap left columns
        tmp  = topL;
        topL = botL;
        botL = tmp;
        // Swap right columns
        tmp  = topR;
        topR = botR;
        botR = tmp;
    }

    if (fliplr) {
        // Swap top rows
        tmp  = topL;
        topL = topR;
        topR = tmp;
        // Swap bottom rows
        tmp  = botL;
        botL = botR;
        botR = tmp;
    }

    // load first 8 columns
    load_buffer_8x8(topL, &in[0], stride, flipud, fliplr, shift);
    load_buffer_8x8(botL, &in[32], stride, flipud, fliplr, shift);

    // load second 8 columns
    load_buffer_8x8(topR, &in[16], stride, flipud, fliplr, shift);
    load_buffer_8x8(botR, &in[48], stride, flipud, fliplr, shift);

    convert_8x8_to_16x16(in, out);
}

static INLINE void load_buffer_32x8n(const int16_t *input, int32x4_t *out, uint32_t stride, int flipud, int fliplr,
                                     int shift, const int height) {
    const int16_t *in     = input;
    int32x4_t     *output = out;
    int            col;
    int32x4_t      v_shift;
    for (col = 0; col < height; col++) {
        in      = input + col * stride;
        output  = out + col * 8;
        v_shift = vdupq_n_s32(shift);
        load_buffer_4x4(in, output, 4, flipud, fliplr, &v_shift);
        load_buffer_4x4((in + 16), (output + 4), 4, flipud, fliplr, &v_shift);
    }
}

typedef void (*fwd_transform_1d_neon)(int32x4_t *in, int32x4_t *out, int bit, const int num_cols);

typedef void (*TxfmFuncNEON)(int32x4_t *input, int32x4_t *output, const int8_t cos_bit, const int8_t *stage_range);

static INLINE void av1_round_shift_array_32_neon(int32x4_t *restrict input, int32x4_t *restrict output, const int size,
                                                 const int bit) {
    const int32x4_t v_bit = vdupq_n_s32(-bit);
    int             i;
    for (i = 0; i < size; i++) { output[i] = vrshlq_s32(input[i], v_bit); }
}

#define REVERSE_FLIP_LR_8_NEON(temp, x, y) \
    do {                                   \
        int16x4_t t = temp[x];             \
        temp[x]     = vrev64_s16(temp[y]); \
        temp[y]     = vrev64_s16(t);       \
    } while (0)

static INLINE void load_buffer_4x8_in_8x8(const int16_t *input, int32x4_t *in, int32_t stride, int32_t flipud,
                                          int32_t fliplr, int32_t shift, int32_t step) {
    int16x4_t temp[8];

    if (!flipud) {
        temp[0] = vld1_s16(input + 0 * stride);
        temp[1] = vld1_s16(input + 0 * stride + 4);
        temp[2] = vld1_s16(input + 1 * stride);
        temp[3] = vld1_s16(input + 1 * stride + 4);
        temp[4] = vld1_s16(input + 2 * stride);
        temp[5] = vld1_s16(input + 2 * stride + 4);
        temp[6] = vld1_s16(input + 3 * stride);
        temp[7] = vld1_s16(input + 3 * stride + 4);

    } else {
        temp[0] = vld1_s16(input + 7 * stride);
        temp[1] = vld1_s16(input + 7 * stride + 4);
        temp[2] = vld1_s16(input + 6 * stride);
        temp[3] = vld1_s16(input + 6 * stride + 4);
        temp[4] = vld1_s16(input + 5 * stride);
        temp[5] = vld1_s16(input + 5 * stride + 4);
        temp[6] = vld1_s16(input + 4 * stride);
        temp[7] = vld1_s16(input + 4 * stride + 4);
    }

    if (fliplr) {
        REVERSE_FLIP_LR_8_NEON(temp, 0, 1);
        REVERSE_FLIP_LR_8_NEON(temp, 2, 3);
        REVERSE_FLIP_LR_8_NEON(temp, 4, 5);
        REVERSE_FLIP_LR_8_NEON(temp, 6, 7);
    }

    int32x4_t vshift = vdupq_n_s32(shift);

    in[0 * step]     = vshlq_s32(vmovl_s16(temp[0]), vshift);
    in[0 * step + 1] = vshlq_s32(vmovl_s16(temp[1]), vshift);
    in[2 * step]     = vshlq_s32(vmovl_s16(temp[2]), vshift);
    in[2 * step + 1] = vshlq_s32(vmovl_s16(temp[3]), vshift);
    in[4 * step]     = vshlq_s32(vmovl_s16(temp[4]), vshift);
    in[4 * step + 1] = vshlq_s32(vmovl_s16(temp[5]), vshift);
    in[6 * step]     = vshlq_s32(vmovl_s16(temp[6]), vshift);
    in[6 * step + 1] = vshlq_s32(vmovl_s16(temp[7]), vshift);
}

static AOM_FORCE_INLINE void load_buffer_4x16_in_16x16(const int16_t *input, int32x4_t *out, int32_t stride,
                                                       int32_t flipud, int32_t fliplr, int32_t shift) {
    // Load 2 4x8 blocks
    if (flipud) {
        const int16_t *top_l = input + 8 * stride;
        const int16_t *top_r = input + 8 * stride + 8;

        if (fliplr) {
            load_buffer_4x8_in_8x8(top_r, &out[0], stride, flipud, fliplr, shift, 2); // load first 4 columns
            load_buffer_4x8_in_8x8(top_l, &out[2], stride, flipud, fliplr, shift, 2); // load second 4 columns
        } else {
            load_buffer_4x8_in_8x8(top_l, &out[0], stride, flipud, fliplr, shift, 2); // load first 4 columns
            load_buffer_4x8_in_8x8(top_r, &out[2], stride, flipud, fliplr, shift, 2); // load second 4 columns
        }
    } else {
        const int16_t *top_l = input;
        const int16_t *top_r = input + 8;

        if (fliplr) {
            load_buffer_4x8_in_8x8(top_r, &out[0], stride, flipud, fliplr, shift, 2); // load first 4 columns
            load_buffer_4x8_in_8x8(top_l, &out[2], stride, flipud, fliplr, shift, 2); // load second 4 columns
        } else {
            load_buffer_4x8_in_8x8(top_l, &out[0], stride, flipud, fliplr, shift, 2); // load first 4 columns
            load_buffer_4x8_in_8x8(top_r, &out[2], stride, flipud, fliplr, shift, 2); // load second 4 columns
        }
    }
}

static void fidtx16x16_N4_neon(const int32x4_t *restrict in, int32x4_t *restrict out, int8_t bit, int32_t col_num,
                               int32_t step) {
    (void)bit;
    const int32_t bits = 12; // new_sqrt2_bits = 12
    const int32_t sqrt = 2 * 5793; // 2 * new_sqrt2

    const int32x4_t newsqrt   = vdupq_n_s32(sqrt);
    const int32_t   num_iters = 16 * col_num;
    const int32x4_t vbits     = vdupq_n_s32(-bits); // sign because we use vshlq_s32 to shift right

    for (int32_t i = 0; i < num_iters / 4; i += step) {
        int32x4_t temp0 = vmulq_s32(in[2 * i], newsqrt);
        int32x4_t temp1 = vmulq_s32(in[2 * i + 1], newsqrt);
        out[2 * i]      = vrshlq_s32(temp0, vbits);
        out[2 * i + 1]  = vrshlq_s32(temp1, vbits);
    }
}

static INLINE void write_buffer_16x16_N4(const int32x4_t *res, int32_t *output) {
    int32_t fact = -1, index = 0;

    const int32x4_t zero = vdupq_n_s32(0);

    for (int32_t i = 0; i < 2; i++) {
        vst1q_s32((output + (++fact) * 16), res[2 * (index)]);
        vst1q_s32((output + fact * 16 + 4), zero);
        vst1q_s32((output + fact * 16 + 8), zero);
        vst1q_s32((output + fact * 16 + 12), zero);

        index += 2;

        vst1q_s32((output + (++fact) * 16), res[2 * (index)]);
        vst1q_s32((output + fact * 16 + 4), zero);
        vst1q_s32((output + fact * 16 + 8), zero);
        vst1q_s32((output + fact * 16 + 12), zero);

        index += 2;
    }

    EB_MEMSET((output + (++fact) * 16), 0, 768);
}

static void fdct16x16_N4_neon(const int32x4_t *restrict in, int32x4_t *restrict out, int8_t bit, const int32_t col_num,
                              int32_t size) {
    const int32_t  *cospi    = cospi_arr(bit);
    const int32x4_t cospi32  = vdupq_n_s32(cospi[32]);
    const int32x4_t cospim32 = vdupq_n_s32(-cospi[32]);
    const int32x4_t cospi48  = vdupq_n_s32(cospi[48]);
    const int32x4_t cospim48 = vdupq_n_s32(-cospi[48]);
    const int32x4_t cospim16 = vdupq_n_s32(-cospi[16]);
    const int32x4_t cospi56  = vdupq_n_s32(cospi[56]);
    const int32x4_t cospi8   = vdupq_n_s32(cospi[8]);
    const int32x4_t cospi60  = vdupq_n_s32(cospi[60]);
    const int32x4_t cospi4   = vdupq_n_s32(cospi[4]);
    const int32x4_t cospi12  = vdupq_n_s32(cospi[12]);
    const int32x4_t cospi52  = vdupq_n_s32(cospi[52]);
    const int32x4_t rnding   = vdupq_n_s32(1 << (bit - 1));
    const int32x4_t vbit     = vdupq_n_s32(-bit); // sign because we use vshlq_s32 to shift right

    int32x4_t u[16], v[16], x;
    int32_t   col;

    for (col = 0; col < size; ++col) {
        // stage 0
        // stage 1

        u[0]  = vaddq_s32(in[0 * col_num + col], in[15 * col_num + col]);
        u[15] = vsubq_s32(in[0 * col_num + col], in[15 * col_num + col]);
        u[1]  = vaddq_s32(in[1 * col_num + col], in[14 * col_num + col]);
        u[14] = vsubq_s32(in[1 * col_num + col], in[14 * col_num + col]);
        u[2]  = vaddq_s32(in[2 * col_num + col], in[13 * col_num + col]);
        u[13] = vsubq_s32(in[2 * col_num + col], in[13 * col_num + col]);
        u[3]  = vaddq_s32(in[3 * col_num + col], in[12 * col_num + col]);
        u[12] = vsubq_s32(in[3 * col_num + col], in[12 * col_num + col]);
        u[4]  = vaddq_s32(in[4 * col_num + col], in[11 * col_num + col]);
        u[11] = vsubq_s32(in[4 * col_num + col], in[11 * col_num + col]);
        u[5]  = vaddq_s32(in[5 * col_num + col], in[10 * col_num + col]);
        u[10] = vsubq_s32(in[5 * col_num + col], in[10 * col_num + col]);
        u[6]  = vaddq_s32(in[6 * col_num + col], in[9 * col_num + col]);
        u[9]  = vsubq_s32(in[6 * col_num + col], in[9 * col_num + col]);
        u[7]  = vaddq_s32(in[7 * col_num + col], in[8 * col_num + col]);
        u[8]  = vsubq_s32(in[7 * col_num + col], in[8 * col_num + col]);

        // stage 2

        v[0] = vaddq_s32(u[0], u[7]);
        v[7] = vsubq_s32(u[0], u[7]);
        v[1] = vaddq_s32(u[1], u[6]);
        v[6] = vsubq_s32(u[1], u[6]);
        v[2] = vaddq_s32(u[2], u[5]);
        v[5] = vsubq_s32(u[2], u[5]);
        v[3] = vaddq_s32(u[3], u[4]);
        v[4] = vsubq_s32(u[3], u[4]);
        v[8] = u[8];
        v[9] = u[9];

        v[10] = vmulq_s32(u[10], cospim32);
        x     = vmulq_s32(u[13], cospi32);
        v[10] = vaddq_s32(v[10], x);
        v[10] = vaddq_s32(v[10], rnding);
        v[10] = vshlq_s32(v[10], vbit);

        v[13] = vmulq_s32(u[10], cospi32);
        x     = vmulq_s32(u[13], cospim32);
        v[13] = vsubq_s32(v[13], x);
        v[13] = vaddq_s32(v[13], rnding);
        v[13] = vshlq_s32(v[13], vbit);

        v[11] = vmulq_s32(u[11], cospim32);
        x     = vmulq_s32(u[12], cospi32);
        v[11] = vaddq_s32(v[11], x);
        v[11] = vaddq_s32(v[11], rnding);
        v[11] = vshlq_s32(v[11], vbit);

        v[12] = vmulq_s32(u[11], cospi32);
        x     = vmulq_s32(u[12], cospim32);
        v[12] = vsubq_s32(v[12], x);
        v[12] = vaddq_s32(v[12], rnding);
        v[12] = vshlq_s32(v[12], vbit);
        v[14] = u[14];
        v[15] = u[15];

        // stage 3

        u[0] = vaddq_s32(v[0], v[3]);
        u[1] = vaddq_s32(v[1], v[2]);
        u[4] = v[4];

        u[5] = vmulq_s32(v[5], cospim32);
        x    = vmulq_s32(v[6], cospi32);
        u[5] = vaddq_s32(u[5], x);
        u[5] = vaddq_s32(u[5], rnding);
        u[5] = vshlq_s32(u[5], vbit);

        u[6] = vmulq_s32(v[5], cospi32);
        x    = vmulq_s32(v[6], cospim32);
        u[6] = vsubq_s32(u[6], x);
        u[6] = vaddq_s32(u[6], rnding);
        u[6] = vshlq_s32(u[6], vbit);

        u[7]  = v[7];
        u[8]  = vaddq_s32(v[8], v[11]);
        u[11] = vsubq_s32(v[8], v[11]);
        u[9]  = vaddq_s32(v[9], v[10]);
        u[10] = vsubq_s32(v[9], v[10]);
        u[12] = vsubq_s32(v[15], v[12]);
        u[15] = vaddq_s32(v[15], v[12]);
        u[13] = vsubq_s32(v[14], v[13]);
        u[14] = vaddq_s32(v[14], v[13]);

        // stage 4

        u[0] = vmulq_s32(u[0], cospi32);
        u[1] = vmulq_s32(u[1], cospi32);
        v[0] = vaddq_s32(u[0], u[1]);
        v[0] = vaddq_s32(v[0], rnding);
        v[0] = vshlq_s32(v[0], vbit);

        v[4] = vaddq_s32(u[4], u[5]);
        v[7] = vaddq_s32(u[7], u[6]);
        v[8] = u[8];

        v[9] = vmulq_s32(u[9], cospim16);
        x    = vmulq_s32(u[14], cospi48);
        v[9] = vaddq_s32(v[9], x);
        v[9] = vaddq_s32(v[9], rnding);
        v[9] = vshlq_s32(v[9], vbit);

        v[14] = vmulq_s32(u[9], cospi48);
        x     = vmulq_s32(u[14], cospim16);
        v[14] = vsubq_s32(v[14], x);
        v[14] = vaddq_s32(v[14], rnding);
        v[14] = vshlq_s32(v[14], vbit);

        v[10] = vmulq_s32(u[10], cospim48);
        x     = vmulq_s32(u[13], cospim16);
        v[10] = vaddq_s32(v[10], x);
        v[10] = vaddq_s32(v[10], rnding);
        v[10] = vshlq_s32(v[10], vbit);

        v[13] = vmulq_s32(u[10], cospim16);
        x     = vmulq_s32(u[13], cospim48);
        v[13] = vsubq_s32(v[13], x);
        v[13] = vaddq_s32(v[13], rnding);
        v[13] = vshlq_s32(v[13], vbit);

        v[11] = u[11];
        v[12] = u[12];
        v[15] = u[15];

        // stage 5

        u[0] = v[0];

        u[4] = vmulq_s32(v[4], cospi56);
        x    = vmulq_s32(v[7], cospi8);
        u[4] = vaddq_s32(u[4], x);
        u[4] = vaddq_s32(u[4], rnding);
        u[4] = vshlq_s32(u[4], vbit);

        u[8]  = vaddq_s32(v[8], v[9]);
        u[11] = vaddq_s32(v[11], v[10]);
        u[12] = vaddq_s32(v[12], v[13]);
        u[15] = vaddq_s32(v[15], v[14]);

        // stage 6

        v[0] = u[0];
        v[4] = u[4];

        v[8] = vmulq_s32(u[8], cospi60);
        x    = vmulq_s32(u[15], cospi4);
        v[8] = vaddq_s32(v[8], x);
        v[8] = vaddq_s32(v[8], rnding);
        v[8] = vshlq_s32(v[8], vbit);

        v[12] = vmulq_s32(u[11], cospi52);
        x     = vmulq_s32(u[12], cospi12);
        v[12] = vsubq_s32(x, v[12]);
        v[12] = vaddq_s32(v[12], rnding);
        v[12] = vshlq_s32(v[12], vbit);

        out[0 * col_num + col] = v[0];
        out[1 * col_num + col] = v[8];
        out[2 * col_num + col] = v[4];
        out[3 * col_num + col] = v[12];
    }
}

static INLINE int32x4_t half_btf_small(const int32x4_t *w0, const int32x4_t *n0, const int32x4_t *w1,
                                       const int32x4_t *n1, int32_t bit) {
    const int32x4_t vbit = vdupq_n_s32(-bit); // sign because we use vshlq_s32 to shift right
    int32x4_t       x    = vmulq_s32(*w0, *n0);
    int32x4_t       y    = vmulq_s32(*w1, *n1);
    x                    = vaddq_s32(x, y);
    x                    = vrshlq_s32(x, vbit);
    return x;
}

static void fadst16x16_N4_neon(const int32x4_t *restrict in, int32x4_t *restrict out, int8_t bit, const int32_t col_num,
                               int32_t size) {
    const int32_t  *cospi    = cospi_arr(bit);
    const int32x4_t cospi32  = vdupq_n_s32(cospi[32]);
    const int32x4_t cospi48  = vdupq_n_s32(cospi[48]);
    const int32x4_t cospi16  = vdupq_n_s32(cospi[16]);
    const int32x4_t cospim16 = vdupq_n_s32(-cospi[16]);
    const int32x4_t cospim48 = vdupq_n_s32(-cospi[48]);
    const int32x4_t cospi8   = vdupq_n_s32(cospi[8]);
    const int32x4_t cospi56  = vdupq_n_s32(cospi[56]);
    const int32x4_t cospim56 = vdupq_n_s32(-cospi[56]);
    const int32x4_t cospim8  = vdupq_n_s32(-cospi[8]);
    const int32x4_t cospi24  = vdupq_n_s32(cospi[24]);
    const int32x4_t cospim24 = vdupq_n_s32(-cospi[24]);
    const int32x4_t cospim40 = vdupq_n_s32(-cospi[40]);
    const int32x4_t cospi40  = vdupq_n_s32(cospi[40]);
    const int32x4_t cospi62  = vdupq_n_s32(cospi[62]);
    const int32x4_t cospim2  = vdupq_n_s32(-cospi[2]);
    const int32x4_t cospi54  = vdupq_n_s32(cospi[54]);
    const int32x4_t cospim10 = vdupq_n_s32(-cospi[10]);
    const int32x4_t cospi50  = vdupq_n_s32(cospi[50]);
    const int32x4_t cospi14  = vdupq_n_s32(cospi[14]);
    const int32x4_t cospi58  = vdupq_n_s32(cospi[58]);
    const int32x4_t cospi6   = vdupq_n_s32(cospi[6]);
    const int32x4_t rnding   = vdupq_n_s32(1 << (bit - 1));
    const int32x4_t vbit     = vdupq_n_s32(-bit); // sign because we use vshlq_s32 to shift right

    int32x4_t u[16], v[16], x, y;
    int32_t   col;

    for (col = 0; col < size; ++col) {
        // stage 0
        // stage 1

        u[0]  = in[0 * col_num + col];
        u[1]  = vnegq_s32(in[15 * col_num + col]);
        u[2]  = vnegq_s32(in[7 * col_num + col]);
        u[3]  = in[8 * col_num + col];
        u[4]  = vnegq_s32(in[3 * col_num + col]);
        u[5]  = in[12 * col_num + col];
        u[6]  = in[4 * col_num + col];
        u[7]  = vnegq_s32(in[11 * col_num + col]);
        u[8]  = vnegq_s32(in[1 * col_num + col]);
        u[9]  = in[14 * col_num + col];
        u[10] = in[6 * col_num + col];
        u[11] = vnegq_s32(in[9 * col_num + col]);
        u[12] = in[2 * col_num + col];
        u[13] = vnegq_s32(in[13 * col_num + col]);
        u[14] = vnegq_s32(in[5 * col_num + col]);
        u[15] = in[10 * col_num + col];

        // stage 2

        v[0] = u[0];
        v[1] = u[1];

        x    = vmulq_s32(u[2], cospi32);
        y    = vmulq_s32(u[3], cospi32);
        v[2] = vaddq_s32(x, y);
        v[2] = vaddq_s32(v[2], rnding);
        v[2] = vshlq_s32(v[2], vbit);

        v[3] = vsubq_s32(x, y);
        v[3] = vaddq_s32(v[3], rnding);
        v[3] = vshlq_s32(v[3], vbit);

        v[4] = u[4];
        v[5] = u[5];

        x    = vmulq_s32(u[6], cospi32);
        y    = vmulq_s32(u[7], cospi32);
        v[6] = vaddq_s32(x, y);
        v[6] = vaddq_s32(v[6], rnding);
        v[6] = vshlq_s32(v[6], vbit);

        v[7] = vsubq_s32(x, y);
        v[7] = vaddq_s32(v[7], rnding);
        v[7] = vshlq_s32(v[7], vbit);

        v[8] = u[8];
        v[9] = u[9];

        x     = vmulq_s32(u[10], cospi32);
        y     = vmulq_s32(u[11], cospi32);
        v[10] = vaddq_s32(x, y);
        v[10] = vaddq_s32(v[10], rnding);
        v[10] = vshlq_s32(v[10], vbit);

        v[11] = vsubq_s32(x, y);
        v[11] = vaddq_s32(v[11], rnding);
        v[11] = vshlq_s32(v[11], vbit);

        v[12] = u[12];
        v[13] = u[13];

        x     = vmulq_s32(u[14], cospi32);
        y     = vmulq_s32(u[15], cospi32);
        v[14] = vaddq_s32(x, y);
        v[14] = vaddq_s32(v[14], rnding);
        v[14] = vshlq_s32(v[14], vbit);

        v[15] = vsubq_s32(x, y);
        v[15] = vaddq_s32(v[15], rnding);
        v[15] = vshlq_s32(v[15], vbit);

        // stage 3

        u[0]  = vaddq_s32(v[0], v[2]);
        u[1]  = vaddq_s32(v[1], v[3]);
        u[2]  = vsubq_s32(v[0], v[2]);
        u[3]  = vsubq_s32(v[1], v[3]);
        u[4]  = vaddq_s32(v[4], v[6]);
        u[5]  = vaddq_s32(v[5], v[7]);
        u[6]  = vsubq_s32(v[4], v[6]);
        u[7]  = vsubq_s32(v[5], v[7]);
        u[8]  = vaddq_s32(v[8], v[10]);
        u[9]  = vaddq_s32(v[9], v[11]);
        u[10] = vsubq_s32(v[8], v[10]);
        u[11] = vsubq_s32(v[9], v[11]);
        u[12] = vaddq_s32(v[12], v[14]);
        u[13] = vaddq_s32(v[13], v[15]);
        u[14] = vsubq_s32(v[12], v[14]);
        u[15] = vsubq_s32(v[13], v[15]);

        // stage 4

        v[0]  = u[0];
        v[1]  = u[1];
        v[2]  = u[2];
        v[3]  = u[3];
        v[4]  = half_btf_small(&cospi16, &u[4], &cospi48, &u[5], bit);
        v[5]  = half_btf_small(&cospi48, &u[4], &cospim16, &u[5], bit);
        v[6]  = half_btf_small(&cospim48, &u[6], &cospi16, &u[7], bit);
        v[7]  = half_btf_small(&cospi16, &u[6], &cospi48, &u[7], bit);
        v[8]  = u[8];
        v[9]  = u[9];
        v[10] = u[10];
        v[11] = u[11];
        v[12] = half_btf_small(&cospi16, &u[12], &cospi48, &u[13], bit);
        v[13] = half_btf_small(&cospi48, &u[12], &cospim16, &u[13], bit);
        v[14] = half_btf_small(&cospim48, &u[14], &cospi16, &u[15], bit);
        v[15] = half_btf_small(&cospi16, &u[14], &cospi48, &u[15], bit);

        // stage 5

        u[0]  = vaddq_s32(v[0], v[4]);
        u[1]  = vaddq_s32(v[1], v[5]);
        u[2]  = vaddq_s32(v[2], v[6]);
        u[3]  = vaddq_s32(v[3], v[7]);
        u[4]  = vsubq_s32(v[0], v[4]);
        u[5]  = vsubq_s32(v[1], v[5]);
        u[6]  = vsubq_s32(v[2], v[6]);
        u[7]  = vsubq_s32(v[3], v[7]);
        u[8]  = vaddq_s32(v[8], v[12]);
        u[9]  = vaddq_s32(v[9], v[13]);
        u[10] = vaddq_s32(v[10], v[14]);
        u[11] = vaddq_s32(v[11], v[15]);
        u[12] = vsubq_s32(v[8], v[12]);
        u[13] = vsubq_s32(v[9], v[13]);
        u[14] = vsubq_s32(v[10], v[14]);
        u[15] = vsubq_s32(v[11], v[15]);

        // stage 6

        v[0]  = u[0];
        v[1]  = u[1];
        v[2]  = u[2];
        v[3]  = u[3];
        v[4]  = u[4];
        v[5]  = u[5];
        v[6]  = u[6];
        v[7]  = u[7];
        v[8]  = half_btf_small(&cospi8, &u[8], &cospi56, &u[9], bit);
        v[9]  = half_btf_small(&cospi56, &u[8], &cospim8, &u[9], bit);
        v[10] = half_btf_small(&cospi40, &u[10], &cospi24, &u[11], bit);
        v[11] = half_btf_small(&cospi24, &u[10], &cospim40, &u[11], bit);
        v[12] = half_btf_small(&cospim56, &u[12], &cospi8, &u[13], bit);
        v[13] = half_btf_small(&cospi8, &u[12], &cospi56, &u[13], bit);
        v[14] = half_btf_small(&cospim24, &u[14], &cospi40, &u[15], bit);
        v[15] = half_btf_small(&cospi40, &u[14], &cospi24, &u[15], bit);

        // stage 7

        u[0]  = vaddq_s32(v[0], v[8]);
        u[1]  = vaddq_s32(v[1], v[9]);
        u[2]  = vaddq_s32(v[2], v[10]);
        u[3]  = vaddq_s32(v[3], v[11]);
        u[12] = vsubq_s32(v[4], v[12]);
        u[13] = vsubq_s32(v[5], v[13]);
        u[14] = vsubq_s32(v[6], v[14]);
        u[15] = vsubq_s32(v[7], v[15]);

        // stage 8
        v[1]  = half_btf_small(&cospi62, &u[0], &cospim2, &u[1], bit);
        v[3]  = half_btf_small(&cospi54, &u[2], &cospim10, &u[3], bit);
        v[12] = half_btf_small(&cospi50, &u[12], &cospi14, &u[13], bit);
        v[14] = half_btf_small(&cospi58, &u[14], &cospi6, &u[15], bit);

        // stage 9
        out[0 * col_num + col] = v[1];
        out[1 * col_num + col] = v[14];
        out[2 * col_num + col] = v[3];
        out[3 * col_num + col] = v[12];
    }
}

static INLINE void transpose_4x4_in_16x16_neon(const int32x4_t *restrict in, int32x4_t *restrict out) {
    int32x4_t u0, u1, u2, u3;

    u0 = vzip1q_s32(in[0], in[4]);
    u1 = vzip2q_s32(in[0], in[4]);
    u2 = vzip1q_s32(in[8], in[12]);
    u3 = vzip2q_s32(in[8], in[12]);

    out[0]  = vreinterpretq_s32_s64(vzip1q_s64(vreinterpretq_s64_s32(u0), vreinterpretq_s64_s32(u2)));
    out[4]  = vreinterpretq_s32_s64(vzip2q_s64(vreinterpretq_s64_s32(u0), vreinterpretq_s64_s32(u2)));
    out[8]  = vreinterpretq_s32_s64(vzip1q_s64(vreinterpretq_s64_s32(u1), vreinterpretq_s64_s32(u3)));
    out[12] = vreinterpretq_s32_s64(vzip2q_s64(vreinterpretq_s64_s32(u1), vreinterpretq_s64_s32(u3)));
}

static AOM_FORCE_INLINE void col_txfm_16x16_N4_rounding(int32x4_t *in, int32_t shift) {
    const int32x4_t vshift = vdupq_n_s32(shift);
    col_txfm_8x8_rounding(&in[0], &vshift);
}

#define TRANSPOSE_2X4X4_NEON(in, x0, x2, x4, x6, out, y0, y1, y2, y3, y4, y5, y6, y7)                      \
    do {                                                                                                   \
        int32x4_t u0, u1, u2, u3, u4, u5, u6, u7;                                                          \
        u0 = vzip1q_s32(in[x0], in[x2]);                                                                   \
        u1 = vzip2q_s32(in[x0], in[x2]);                                                                   \
        u2 = vzip1q_s32(in[x0 + 1], in[x2 + 1]);                                                           \
        u3 = vzip2q_s32(in[x0 + 1], in[x2 + 1]);                                                           \
        u4 = vzip1q_s32(in[x4], in[x6]);                                                                   \
        u5 = vzip2q_s32(in[x4], in[x6]);                                                                   \
        u6 = vzip1q_s32(in[x4 + 1], in[x6 + 1]);                                                           \
        u7 = vzip2q_s32(in[x4 + 1], in[x6 + 1]);                                                           \
                                                                                                           \
        out[y0] = vreinterpretq_s32_s64(vzip1q_s64(vreinterpretq_s64_s32(u0), vreinterpretq_s64_s32(u4))); \
        out[y1] = vreinterpretq_s32_s64(vzip1q_s64(vreinterpretq_s64_s32(u2), vreinterpretq_s64_s32(u6))); \
        out[y2] = vreinterpretq_s32_s64(vzip2q_s64(vreinterpretq_s64_s32(u0), vreinterpretq_s64_s32(u4))); \
        out[y3] = vreinterpretq_s32_s64(vzip2q_s64(vreinterpretq_s64_s32(u2), vreinterpretq_s64_s32(u6))); \
        out[y4] = vreinterpretq_s32_s64(vzip1q_s64(vreinterpretq_s64_s32(u1), vreinterpretq_s64_s32(u5))); \
        out[y5] = vreinterpretq_s32_s64(vzip1q_s64(vreinterpretq_s64_s32(u3), vreinterpretq_s64_s32(u7))); \
        out[y6] = vreinterpretq_s32_s64(vzip2q_s64(vreinterpretq_s64_s32(u1), vreinterpretq_s64_s32(u5))); \
        out[y7] = vreinterpretq_s32_s64(vzip2q_s64(vreinterpretq_s64_s32(u3), vreinterpretq_s64_s32(u7))); \
    } while (0)

/*
 *    Transpose top left block of size 8x8 in 16x16 block
 */
static INLINE void transpose_8x8_in_16x16_neon(const int32x4_t *restrict in, int32x4_t *restrict out) {
    TRANSPOSE_2X4X4_NEON(in, 0, 4, 8, 12, out, 0, 16, 4, 20, 8, 24, 12, 28);
    TRANSPOSE_2X4X4_NEON(in, 16, 20, 24, 28, out, 1, 17, 5, 21, 9, 25, 13, 29);
}

void svt_av1_fwd_txfm2d_16x16_N4_neon(int16_t *input, int32_t *coeff, uint32_t stride, TxType tx_type, uint8_t bd) {
    int32x4_t     in[64], out[64];
    const int8_t *shift   = fwd_txfm_shift_ls[TX_16X16];
    const int32_t txw_idx = get_txw_idx(TX_16X16);
    const int32_t txh_idx = get_txh_idx(TX_16X16);
    const int32_t col_num = 2;

    switch (tx_type) {
    case IDTX:
        load_buffer_4x16_in_16x16(input, in, stride, 0, 0, shift[0]);
        fidtx16x16_N4_neon(in, out, fwd_cos_bit_col[txw_idx][txh_idx], col_num, 2);
        col_txfm_16x16_N4_rounding(out, shift[1]);
        fidtx16x16_N4_neon(out, in, fwd_cos_bit_row[txw_idx][txh_idx], col_num, 2);
        write_buffer_16x16_N4(in, coeff);
        break;
    case DCT_DCT:
        load_buffer_16x16(input, in, stride, 0, 0, shift[0]);
        fdct16x16_N4_neon(in, out, fwd_cos_bit_col[txw_idx][txh_idx], 2 * col_num, 2 * col_num);
        col_txfm_16x16_N4_rounding(out, shift[1]);
        transpose_8x8_in_16x16_neon(out, in); // top-left -> top-left
        transpose_8x8_in_16x16_neon(out + 2, in + 32); // top-right -> bottom-left
        fdct16x16_N4_neon(in, out, fwd_cos_bit_row[txw_idx][txh_idx], 2 * col_num, 2);
        transpose_4x4_in_16x16_neon(out, in);
        write_buffer_16x16_N4(in, coeff);
        break;
    case ADST_DCT:
        load_buffer_16x16(input, in, stride, 0, 0, shift[0]);
        fadst16x16_N4_neon(in, out, fwd_cos_bit_col[txw_idx][txh_idx], 2 * col_num, 2 * col_num);
        col_txfm_16x16_N4_rounding(out, shift[1]);
        transpose_8x8_in_16x16_neon(out, in); // top-left -> top-left
        transpose_8x8_in_16x16_neon(out + 2, in + 32); // top-right -> bottom-left
        fdct16x16_N4_neon(in, out, fwd_cos_bit_row[txw_idx][txh_idx], 2 * col_num, 2);
        transpose_4x4_in_16x16_neon(out, in);
        write_buffer_16x16_N4(in, coeff);
        break;
    case DCT_ADST:
        load_buffer_16x16(input, in, stride, 0, 0, shift[0]);
        fdct16x16_N4_neon(in, out, fwd_cos_bit_col[txw_idx][txh_idx], 2 * col_num, 2 * col_num);
        col_txfm_16x16_N4_rounding(out, shift[1]);
        transpose_8x8_in_16x16_neon(out, in); // top-left -> top-left
        transpose_8x8_in_16x16_neon(out + 2, in + 32); // top-right -> bottom-left
        fadst16x16_N4_neon(in, out, fwd_cos_bit_row[txw_idx][txh_idx], 2 * col_num, 2);
        transpose_4x4_in_16x16_neon(out, in);
        write_buffer_16x16_N4(in, coeff);
        break;
    case ADST_ADST:
        load_buffer_16x16(input, in, stride, 0, 0, shift[0]);
        fadst16x16_N4_neon(in, out, fwd_cos_bit_col[txw_idx][txh_idx], 2 * col_num, 2 * col_num);
        col_txfm_16x16_N4_rounding(out, shift[1]);
        transpose_8x8_in_16x16_neon(out, in); // top-left -> top-left
        transpose_8x8_in_16x16_neon(out + 2, in + 32); // top-right -> bottom-left
        fadst16x16_N4_neon(in, out, fwd_cos_bit_row[txw_idx][txh_idx], 2 * col_num, 2);
        transpose_4x4_in_16x16_neon(out, in);
        write_buffer_16x16_N4(in, coeff);
        break;
    case DCT_FLIPADST:
        load_buffer_16x16(input, in, stride, 0, 1, shift[0]);
        fdct16x16_N4_neon(in, out, fwd_cos_bit_col[txw_idx][txh_idx], 2 * col_num, 2 * col_num);
        col_txfm_16x16_N4_rounding(out, shift[1]);
        transpose_8x8_in_16x16_neon(out, in); // top-left -> top-left
        transpose_8x8_in_16x16_neon(out + 2, in + 32); // top-right -> bottom-left
        fadst16x16_N4_neon(in, out, fwd_cos_bit_row[txw_idx][txh_idx], 2 * col_num, 2);
        transpose_4x4_in_16x16_neon(out, in);
        write_buffer_16x16_N4(in, coeff);
        break;
    case FLIPADST_DCT:
        load_buffer_16x16(input, in, stride, 1, 0, shift[0]);
        fadst16x16_N4_neon(in, out, fwd_cos_bit_col[txw_idx][txh_idx], 2 * col_num, 2 * col_num);
        col_txfm_16x16_N4_rounding(out, shift[1]);
        transpose_8x8_in_16x16_neon(out, in); // top-left -> top-left
        transpose_8x8_in_16x16_neon(out + 2, in + 32); // top-right -> bottom-left
        fdct16x16_N4_neon(in, out, fwd_cos_bit_row[txw_idx][txh_idx], 2 * col_num, 2);
        transpose_4x4_in_16x16_neon(out, in);
        write_buffer_16x16_N4(in, coeff);
        break;
    case FLIPADST_FLIPADST:
        load_buffer_16x16(input, in, stride, 1, 1, shift[0]);
        fadst16x16_N4_neon(in, out, fwd_cos_bit_col[txw_idx][txh_idx], 2 * col_num, 2 * col_num);
        col_txfm_16x16_N4_rounding(out, shift[1]);
        transpose_8x8_in_16x16_neon(out, in); // top-left -> top-left
        transpose_8x8_in_16x16_neon(out + 2, in + 32); // top-right -> bottom-left
        fadst16x16_N4_neon(in, out, fwd_cos_bit_row[txw_idx][txh_idx], 2 * col_num, 2);
        transpose_4x4_in_16x16_neon(out, in);
        write_buffer_16x16_N4(in, coeff);
        break;
    case ADST_FLIPADST:
        load_buffer_16x16(input, in, stride, 0, 1, shift[0]);
        fadst16x16_N4_neon(in, out, fwd_cos_bit_col[txw_idx][txh_idx], 2 * col_num, 2 * col_num);
        col_txfm_16x16_N4_rounding(out, shift[1]);
        transpose_8x8_in_16x16_neon(out, in); // top-left -> top-left
        transpose_8x8_in_16x16_neon(out + 2, in + 32); // top-right -> bottom-left
        fadst16x16_N4_neon(in, out, fwd_cos_bit_row[txw_idx][txh_idx], 2 * col_num, 2);
        transpose_4x4_in_16x16_neon(out, in);
        write_buffer_16x16_N4(in, coeff);
        break;
    case FLIPADST_ADST:
        load_buffer_16x16(input, in, stride, 1, 0, shift[0]);
        fadst16x16_N4_neon(in, out, fwd_cos_bit_col[txw_idx][txh_idx], 2 * col_num, 2 * col_num);
        col_txfm_16x16_N4_rounding(out, shift[1]);
        transpose_8x8_in_16x16_neon(out, in); // top-left -> top-left
        transpose_8x8_in_16x16_neon(out + 2, in + 32); // top-right -> bottom-left
        fadst16x16_N4_neon(in, out, fwd_cos_bit_row[txw_idx][txh_idx], 2 * col_num, 2);
        transpose_4x4_in_16x16_neon(out, in);
        write_buffer_16x16_N4(in, coeff);
        break;
    case V_DCT:
        load_buffer_16x16(input, in, stride, 0, 0, shift[0]);
        fdct16x16_N4_neon(in, out, fwd_cos_bit_col[txw_idx][txh_idx], 2 * col_num, 2);
        col_txfm_16x16_N4_rounding(out, shift[1]);
        fidtx16x16_N4_neon(out, in, fwd_cos_bit_row[txw_idx][txh_idx], col_num, 2);
        write_buffer_16x16_N4(in, coeff);
        break;
    case H_DCT:
        load_buffer_4x16_in_16x16(input, out, stride, 0, 0, shift[0]);
        fidtx16x16_N4_neon(out, in, fwd_cos_bit_col[txw_idx][txh_idx], col_num, 1);
        col_txfm_16x16_N4_rounding(in, shift[1]);
        transpose_8x8_in_16x16_neon(in, out); // top-left -> top-left
        transpose_8x8_in_16x16_neon(in + 2, out + 32); // top-right -> bottom-left
        fdct16x16_N4_neon(out, in, fwd_cos_bit_row[txw_idx][txh_idx], 2 * col_num, 2);
        transpose_4x4_in_16x16_neon(in, out);
        write_buffer_16x16_N4(out, coeff);
        break;
    case V_ADST:
        load_buffer_16x16(input, in, stride, 0, 0, shift[0]);
        fadst16x16_N4_neon(in, out, fwd_cos_bit_col[txw_idx][txh_idx], 2 * col_num, 2);
        col_txfm_16x16_N4_rounding(out, shift[1]);
        fidtx16x16_N4_neon(out, in, fwd_cos_bit_row[txw_idx][txh_idx], col_num, 2);
        write_buffer_16x16_N4(in, coeff);
        break;
    case H_ADST:
        load_buffer_4x16_in_16x16(input, out, stride, 0, 0, shift[0]);
        fidtx16x16_N4_neon(out, in, fwd_cos_bit_col[txw_idx][txh_idx], col_num, 1);
        col_txfm_16x16_N4_rounding(in, shift[1]);
        transpose_8x8_in_16x16_neon(in, out); // top-left -> top-left
        transpose_8x8_in_16x16_neon(in + 2, out + 32); // top-right -> bottom-left
        fadst16x16_N4_neon(out, in, fwd_cos_bit_row[txw_idx][txh_idx], 2 * col_num, 2);
        transpose_4x4_in_16x16_neon(in, out);
        write_buffer_16x16_N4(out, coeff);
        break;
    case V_FLIPADST:
        load_buffer_16x16(input, in, stride, 1, 0, shift[0]);
        fadst16x16_N4_neon(in, out, fwd_cos_bit_col[txw_idx][txh_idx], 2 * col_num, 2 * col_num);
        col_txfm_16x16_N4_rounding(out, shift[1]);
        fidtx16x16_N4_neon(out, in, fwd_cos_bit_row[txw_idx][txh_idx], col_num, 2);
        write_buffer_16x16_N4(in, coeff);
        break;
    case H_FLIPADST:
        load_buffer_4x16_in_16x16(input, out, stride, 0, 1, shift[0]);
        fidtx16x16_N4_neon(out, in, fwd_cos_bit_col[txw_idx][txh_idx], col_num, 1);
        col_txfm_16x16_N4_rounding(in, shift[1]);
        transpose_8x8_in_16x16_neon(in, out); // top-left -> top-left
        transpose_8x8_in_16x16_neon(in + 2, out + 32); // top-right -> bottom-left
        fadst16x16_N4_neon(out, in, fwd_cos_bit_row[txw_idx][txh_idx], 2 * col_num, 2);
        transpose_4x4_in_16x16_neon(in, out);
        write_buffer_16x16_N4(out, coeff);
        break;
    default: assert(0);
    }
    (void)bd;
}

static INLINE void load_buffer_16x16_in_64x64_neon(const int16_t *input, int32_t stride, int32x4_t *output) {
    for (int32_t i = 0; i < 16; ++i) {
        output[0] = vmovl_s16(vld1_s16(input + 0 * 4));
        output[1] = vmovl_s16(vld1_s16(input + 1 * 4));
        output[2] = vmovl_s16(vld1_s16(input + 2 * 4));
        output[3] = vmovl_s16(vld1_s16(input + 3 * 4));

        input += stride;
        output += 16;
    }
}

static void fidtx64x64_N4_neon(const int32x4_t *restrict input, int32x4_t *restrict output) {
    const int32_t   bits    = 12; // new_sqrt2_bits = 12
    const int32_t   sqrt    = 4 * 5793; // 4 * new_sqrt2
    const int32x4_t newsqrt = vdupq_n_s32(sqrt);
    const int32x4_t vbits   = vdupq_n_s32(-bits); // sign because we use vshlq_s32 to shift right

    for (int32_t i = 0; i < 128; i += 8) {
        int32x4_t temp0 = vmulq_s32(input[2 * i + 0], newsqrt);
        int32x4_t temp1 = vmulq_s32(input[2 * i + 1], newsqrt);
        int32x4_t temp2 = vmulq_s32(input[2 * i + 2], newsqrt);
        int32x4_t temp3 = vmulq_s32(input[2 * i + 3], newsqrt);

        output[2 * i + 0] = vrshlq_s32(temp0, vbits);
        output[2 * i + 1] = vrshlq_s32(temp1, vbits);
        output[2 * i + 2] = vrshlq_s32(temp2, vbits);
        output[2 * i + 3] = vrshlq_s32(temp3, vbits);
    }
}

static INLINE void av1_round_shift_array_64_N4_neon(int32x4_t *restrict input, int32x4_t *restrict output,
                                                    const int32_t size, const int32_t bit) {
    const int32x4_t right_bit_bits = vdupq_n_s32(-bit); // sign because we use vshlq_s32 to shift right

    for (int i = 0; i < size; i += 8) {
        output[2 * i + 0] = vrshlq_s32(input[2 * i + 0], right_bit_bits);
        output[2 * i + 1] = vrshlq_s32(input[2 * i + 1], right_bit_bits);
        output[2 * i + 2] = vrshlq_s32(input[2 * i + 2], right_bit_bits);
        output[2 * i + 3] = vrshlq_s32(input[2 * i + 3], right_bit_bits);
    }
}

static INLINE void clear_buffer_wxh_N4(int32x4_t *buff, int32_t num_col, int32_t num_row) {
    const int32x4_t zero = vdupq_n_s32(0);

    assert(num_col > 0);
    assert(num_row > 1);

    if (num_col == 1) {
        for (int i = 0; i < num_row / 4; i++) {
            buff[i * 2] = vreinterpretq_s32_s64(
                vzip1q_s64(vreinterpretq_s64_s32(buff[i * 2]), vreinterpretq_s64_s32(zero)));
            buff[i * 2 + 1] = zero;
        }
    } else if (num_col == 2) {
        for (int i = 0; i < num_row / 4; i++) {
            buff[i * 4 + 1] = zero;
            buff[i * 4 + 2] = zero;
            buff[i * 4 + 3] = zero;
        }
    } else {
        for (int i = 0; i < num_row / 4; i++) {
            for (int j = num_col / 4; j < num_col; j++) {
                buff[2 * (i * num_col + j)]     = zero;
                buff[2 * (i * num_col + j) + 1] = zero;
            }
        }
    }
    //clear bottom
    for (int i = num_row / 4; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            buff[2 * (i * num_col + j)]     = zero;
            buff[2 * (i * num_col + j) + 1] = zero;
        }
    }
}

// out0 =  in0 * w0 + in1 * w1
// out1 = -in1 * w0 + in0 * w1
#define btf_32_type0_neon_new(ww0, ww1, in0, in1, out0, out1, bit) \
    do {                                                           \
        const int32x4_t vbit   = vdupq_n_s32(-bit);                \
        const int32x4_t in0_w0 = vmulq_s32(in0, ww0);              \
        const int32x4_t in1_w1 = vmulq_s32(in1, ww1);              \
        out0                   = vaddq_s32(in0_w0, in1_w1);        \
        out0                   = vrshlq_s32(out0, vbit);           \
        const int32x4_t in0_w1 = vmulq_s32(in0, ww1);              \
        const int32x4_t in1_w0 = vmulq_s32(in1, ww0);              \
        out1                   = vsubq_s32(in0_w1, in1_w0);        \
        out1                   = vrshlq_s32(out1, vbit);           \
    } while (0)

static void av1_fdct64_new_N4_neon(const int32x4_t *restrict input, int32x4_t *restrict output, int8_t cos_bit,
                                   const int32_t col_num, const int32_t stride) {
    const int32_t *cospi   = cospi_arr(cos_bit);
    const int32_t  columns = col_num >> 2;

    const int32x4_t cospi_m32 = vdupq_n_s32(-cospi[32]);
    const int32x4_t cospi_p32 = vdupq_n_s32(cospi[32]);
    const int32x4_t cospi_m16 = vdupq_n_s32(-cospi[16]);
    const int32x4_t cospi_p48 = vdupq_n_s32(cospi[48]);
    const int32x4_t cospi_m48 = vdupq_n_s32(-cospi[48]);
    const int32x4_t cospi_m08 = vdupq_n_s32(-cospi[8]);
    const int32x4_t cospi_p56 = vdupq_n_s32(cospi[56]);
    const int32x4_t cospi_m56 = vdupq_n_s32(-cospi[56]);
    const int32x4_t cospi_m40 = vdupq_n_s32(-cospi[40]);
    const int32x4_t cospi_p24 = vdupq_n_s32(cospi[24]);
    const int32x4_t cospi_m24 = vdupq_n_s32(-cospi[24]);
    const int32x4_t cospi_p08 = vdupq_n_s32(cospi[8]);
    const int32x4_t cospi_p60 = vdupq_n_s32(cospi[60]);
    const int32x4_t cospi_p04 = vdupq_n_s32(cospi[4]);
    const int32x4_t cospi_p28 = vdupq_n_s32(cospi[28]);
    const int32x4_t cospi_p44 = vdupq_n_s32(cospi[44]);
    const int32x4_t cospi_p12 = vdupq_n_s32(cospi[12]);
    const int32x4_t cospi_m04 = vdupq_n_s32(-cospi[4]);
    const int32x4_t cospi_m60 = vdupq_n_s32(-cospi[60]);
    const int32x4_t cospi_m36 = vdupq_n_s32(-cospi[36]);
    const int32x4_t cospi_m28 = vdupq_n_s32(-cospi[28]);
    const int32x4_t cospi_m20 = vdupq_n_s32(-cospi[20]);
    const int32x4_t cospi_m44 = vdupq_n_s32(-cospi[44]);
    const int32x4_t cospi_m52 = vdupq_n_s32(-cospi[52]);
    const int32x4_t cospi_m12 = vdupq_n_s32(-cospi[12]);
    const int32x4_t cospi_p62 = vdupq_n_s32(cospi[62]);
    const int32x4_t cospi_p02 = vdupq_n_s32(cospi[2]);
    const int32x4_t cospi_p14 = vdupq_n_s32(cospi[14]);
    const int32x4_t cospi_m50 = vdupq_n_s32(-cospi[50]);
    const int32x4_t cospi_p54 = vdupq_n_s32(cospi[54]);
    const int32x4_t cospi_p10 = vdupq_n_s32(cospi[10]);
    const int32x4_t cospi_p06 = vdupq_n_s32(cospi[6]);
    const int32x4_t cospi_m58 = vdupq_n_s32(-cospi[58]);
    const int32x4_t cospi_p63 = vdupq_n_s32(cospi[63]);
    const int32x4_t cospi_p01 = vdupq_n_s32(cospi[1]);
    const int32x4_t cospi_p15 = vdupq_n_s32(cospi[15]);
    const int32x4_t cospi_m49 = vdupq_n_s32(-cospi[49]);
    const int32x4_t cospi_p55 = vdupq_n_s32(cospi[55]);
    const int32x4_t cospi_p09 = vdupq_n_s32(cospi[9]);
    const int32x4_t cospi_p07 = vdupq_n_s32(cospi[7]);
    const int32x4_t cospi_m57 = vdupq_n_s32(-cospi[57]);
    const int32x4_t cospi_p59 = vdupq_n_s32(cospi[59]);
    const int32x4_t cospi_p05 = vdupq_n_s32(cospi[5]);
    const int32x4_t cospi_p11 = vdupq_n_s32(cospi[11]);
    const int32x4_t cospi_m53 = vdupq_n_s32(-cospi[53]);
    const int32x4_t cospi_p51 = vdupq_n_s32(cospi[51]);
    const int32x4_t cospi_p13 = vdupq_n_s32(cospi[13]);
    const int32x4_t cospi_p03 = vdupq_n_s32(cospi[3]);
    const int32x4_t cospi_m61 = vdupq_n_s32(-cospi[61]);

    for (int32_t col = 0; col < columns; col++) {
        const int32x4_t *in  = &input[col];
        int32x4_t       *out = &output[col];

        // stage 1
        int32x4_t x1[64];
        x1[0]  = vaddq_s32(in[2 * 0 * stride], in[2 * 63 * stride]);
        x1[63] = vsubq_s32(in[2 * 0 * stride], in[2 * 63 * stride]);
        x1[1]  = vaddq_s32(in[2 * 1 * stride], in[2 * 62 * stride]);
        x1[62] = vsubq_s32(in[2 * 1 * stride], in[2 * 62 * stride]);
        x1[2]  = vaddq_s32(in[2 * 2 * stride], in[2 * 61 * stride]);
        x1[61] = vsubq_s32(in[2 * 2 * stride], in[2 * 61 * stride]);
        x1[3]  = vaddq_s32(in[2 * 3 * stride], in[2 * 60 * stride]);
        x1[60] = vsubq_s32(in[2 * 3 * stride], in[2 * 60 * stride]);
        x1[4]  = vaddq_s32(in[2 * 4 * stride], in[2 * 59 * stride]);
        x1[59] = vsubq_s32(in[2 * 4 * stride], in[2 * 59 * stride]);
        x1[5]  = vaddq_s32(in[2 * 5 * stride], in[2 * 58 * stride]);
        x1[58] = vsubq_s32(in[2 * 5 * stride], in[2 * 58 * stride]);
        x1[6]  = vaddq_s32(in[2 * 6 * stride], in[2 * 57 * stride]);
        x1[57] = vsubq_s32(in[2 * 6 * stride], in[2 * 57 * stride]);
        x1[7]  = vaddq_s32(in[2 * 7 * stride], in[2 * 56 * stride]);
        x1[56] = vsubq_s32(in[2 * 7 * stride], in[2 * 56 * stride]);
        x1[8]  = vaddq_s32(in[2 * 8 * stride], in[2 * 55 * stride]);
        x1[55] = vsubq_s32(in[2 * 8 * stride], in[2 * 55 * stride]);
        x1[9]  = vaddq_s32(in[2 * 9 * stride], in[2 * 54 * stride]);
        x1[54] = vsubq_s32(in[2 * 9 * stride], in[2 * 54 * stride]);
        x1[10] = vaddq_s32(in[2 * 10 * stride], in[2 * 53 * stride]);
        x1[53] = vsubq_s32(in[2 * 10 * stride], in[2 * 53 * stride]);
        x1[11] = vaddq_s32(in[2 * 11 * stride], in[2 * 52 * stride]);
        x1[52] = vsubq_s32(in[2 * 11 * stride], in[2 * 52 * stride]);
        x1[12] = vaddq_s32(in[2 * 12 * stride], in[2 * 51 * stride]);
        x1[51] = vsubq_s32(in[2 * 12 * stride], in[2 * 51 * stride]);
        x1[13] = vaddq_s32(in[2 * 13 * stride], in[2 * 50 * stride]);
        x1[50] = vsubq_s32(in[2 * 13 * stride], in[2 * 50 * stride]);
        x1[14] = vaddq_s32(in[2 * 14 * stride], in[2 * 49 * stride]);
        x1[49] = vsubq_s32(in[2 * 14 * stride], in[2 * 49 * stride]);
        x1[15] = vaddq_s32(in[2 * 15 * stride], in[2 * 48 * stride]);
        x1[48] = vsubq_s32(in[2 * 15 * stride], in[2 * 48 * stride]);
        x1[16] = vaddq_s32(in[2 * 16 * stride], in[2 * 47 * stride]);
        x1[47] = vsubq_s32(in[2 * 16 * stride], in[2 * 47 * stride]);
        x1[17] = vaddq_s32(in[2 * 17 * stride], in[2 * 46 * stride]);
        x1[46] = vsubq_s32(in[2 * 17 * stride], in[2 * 46 * stride]);
        x1[18] = vaddq_s32(in[2 * 18 * stride], in[2 * 45 * stride]);
        x1[45] = vsubq_s32(in[2 * 18 * stride], in[2 * 45 * stride]);
        x1[19] = vaddq_s32(in[2 * 19 * stride], in[2 * 44 * stride]);
        x1[44] = vsubq_s32(in[2 * 19 * stride], in[2 * 44 * stride]);
        x1[20] = vaddq_s32(in[2 * 20 * stride], in[2 * 43 * stride]);
        x1[43] = vsubq_s32(in[2 * 20 * stride], in[2 * 43 * stride]);
        x1[21] = vaddq_s32(in[2 * 21 * stride], in[2 * 42 * stride]);
        x1[42] = vsubq_s32(in[2 * 21 * stride], in[2 * 42 * stride]);
        x1[22] = vaddq_s32(in[2 * 22 * stride], in[2 * 41 * stride]);
        x1[41] = vsubq_s32(in[2 * 22 * stride], in[2 * 41 * stride]);
        x1[23] = vaddq_s32(in[2 * 23 * stride], in[2 * 40 * stride]);
        x1[40] = vsubq_s32(in[2 * 23 * stride], in[2 * 40 * stride]);
        x1[24] = vaddq_s32(in[2 * 24 * stride], in[2 * 39 * stride]);
        x1[39] = vsubq_s32(in[2 * 24 * stride], in[2 * 39 * stride]);
        x1[25] = vaddq_s32(in[2 * 25 * stride], in[2 * 38 * stride]);
        x1[38] = vsubq_s32(in[2 * 25 * stride], in[2 * 38 * stride]);
        x1[26] = vaddq_s32(in[2 * 26 * stride], in[2 * 37 * stride]);
        x1[37] = vsubq_s32(in[2 * 26 * stride], in[2 * 37 * stride]);
        x1[27] = vaddq_s32(in[2 * 27 * stride], in[2 * 36 * stride]);
        x1[36] = vsubq_s32(in[2 * 27 * stride], in[2 * 36 * stride]);
        x1[28] = vaddq_s32(in[2 * 28 * stride], in[2 * 35 * stride]);
        x1[35] = vsubq_s32(in[2 * 28 * stride], in[2 * 35 * stride]);
        x1[29] = vaddq_s32(in[2 * 29 * stride], in[2 * 34 * stride]);
        x1[34] = vsubq_s32(in[2 * 29 * stride], in[2 * 34 * stride]);
        x1[30] = vaddq_s32(in[2 * 30 * stride], in[2 * 33 * stride]);
        x1[33] = vsubq_s32(in[2 * 30 * stride], in[2 * 33 * stride]);
        x1[31] = vaddq_s32(in[2 * 31 * stride], in[2 * 32 * stride]);
        x1[32] = vsubq_s32(in[2 * 31 * stride], in[2 * 32 * stride]);

        // stage 2
        int32x4_t x2[64];
        x2[0]  = vaddq_s32(x1[0], x1[31]);
        x2[31] = vsubq_s32(x1[0], x1[31]);
        x2[1]  = vaddq_s32(x1[1], x1[30]);
        x2[30] = vsubq_s32(x1[1], x1[30]);
        x2[2]  = vaddq_s32(x1[2], x1[29]);
        x2[29] = vsubq_s32(x1[2], x1[29]);
        x2[3]  = vaddq_s32(x1[3], x1[28]);
        x2[28] = vsubq_s32(x1[3], x1[28]);
        x2[4]  = vaddq_s32(x1[4], x1[27]);
        x2[27] = vsubq_s32(x1[4], x1[27]);
        x2[5]  = vaddq_s32(x1[5], x1[26]);
        x2[26] = vsubq_s32(x1[5], x1[26]);
        x2[6]  = vaddq_s32(x1[6], x1[25]);
        x2[25] = vsubq_s32(x1[6], x1[25]);
        x2[7]  = vaddq_s32(x1[7], x1[24]);
        x2[24] = vsubq_s32(x1[7], x1[24]);
        x2[8]  = vaddq_s32(x1[8], x1[23]);
        x2[23] = vsubq_s32(x1[8], x1[23]);
        x2[9]  = vaddq_s32(x1[9], x1[22]);
        x2[22] = vsubq_s32(x1[9], x1[22]);
        x2[10] = vaddq_s32(x1[10], x1[21]);
        x2[21] = vsubq_s32(x1[10], x1[21]);
        x2[11] = vaddq_s32(x1[11], x1[20]);
        x2[20] = vsubq_s32(x1[11], x1[20]);
        x2[12] = vaddq_s32(x1[12], x1[19]);
        x2[19] = vsubq_s32(x1[12], x1[19]);
        x2[13] = vaddq_s32(x1[13], x1[18]);
        x2[18] = vsubq_s32(x1[13], x1[18]);
        x2[14] = vaddq_s32(x1[14], x1[17]);
        x2[17] = vsubq_s32(x1[14], x1[17]);
        x2[15] = vaddq_s32(x1[15], x1[16]);
        x2[16] = vsubq_s32(x1[15], x1[16]);
        x2[32] = x1[32];
        x2[33] = x1[33];
        x2[34] = x1[34];
        x2[35] = x1[35];
        x2[36] = x1[36];
        x2[37] = x1[37];
        x2[38] = x1[38];
        x2[39] = x1[39];
        btf_32_type0_neon_new(cospi_m32, cospi_p32, x1[40], x1[55], x2[40], x2[55], cos_bit);
        btf_32_type0_neon_new(cospi_m32, cospi_p32, x1[41], x1[54], x2[41], x2[54], cos_bit);
        btf_32_type0_neon_new(cospi_m32, cospi_p32, x1[42], x1[53], x2[42], x2[53], cos_bit);
        btf_32_type0_neon_new(cospi_m32, cospi_p32, x1[43], x1[52], x2[43], x2[52], cos_bit);
        btf_32_type0_neon_new(cospi_m32, cospi_p32, x1[44], x1[51], x2[44], x2[51], cos_bit);
        btf_32_type0_neon_new(cospi_m32, cospi_p32, x1[45], x1[50], x2[45], x2[50], cos_bit);
        btf_32_type0_neon_new(cospi_m32, cospi_p32, x1[46], x1[49], x2[46], x2[49], cos_bit);
        btf_32_type0_neon_new(cospi_m32, cospi_p32, x1[47], x1[48], x2[47], x2[48], cos_bit);
        x2[56] = x1[56];
        x2[57] = x1[57];
        x2[58] = x1[58];
        x2[59] = x1[59];
        x2[60] = x1[60];
        x2[61] = x1[61];
        x2[62] = x1[62];
        x2[63] = x1[63];

        // stage 3
        int32x4_t x3[64];
        x3[0]  = vaddq_s32(x2[0], x2[15]);
        x3[15] = vsubq_s32(x2[0], x2[15]);
        x3[1]  = vaddq_s32(x2[1], x2[14]);
        x3[14] = vsubq_s32(x2[1], x2[14]);
        x3[2]  = vaddq_s32(x2[2], x2[13]);
        x3[13] = vsubq_s32(x2[2], x2[13]);
        x3[3]  = vaddq_s32(x2[3], x2[12]);
        x3[12] = vsubq_s32(x2[3], x2[12]);
        x3[4]  = vaddq_s32(x2[4], x2[11]);
        x3[11] = vsubq_s32(x2[4], x2[11]);
        x3[5]  = vaddq_s32(x2[5], x2[10]);
        x3[10] = vsubq_s32(x2[5], x2[10]);
        x3[6]  = vaddq_s32(x2[6], x2[9]);
        x3[9]  = vsubq_s32(x2[6], x2[9]);
        x3[7]  = vaddq_s32(x2[7], x2[8]);
        x3[8]  = vsubq_s32(x2[7], x2[8]);
        x3[16] = x2[16];
        x3[17] = x2[17];
        x3[18] = x2[18];
        x3[19] = x2[19];
        btf_32_type0_neon_new(cospi_m32, cospi_p32, x2[20], x2[27], x3[20], x3[27], cos_bit);
        btf_32_type0_neon_new(cospi_m32, cospi_p32, x2[21], x2[26], x3[21], x3[26], cos_bit);
        btf_32_type0_neon_new(cospi_m32, cospi_p32, x2[22], x2[25], x3[22], x3[25], cos_bit);
        btf_32_type0_neon_new(cospi_m32, cospi_p32, x2[23], x2[24], x3[23], x3[24], cos_bit);
        x3[28] = x2[28];
        x3[29] = x2[29];
        x3[30] = x2[30];
        x3[31] = x2[31];
        x3[32] = vaddq_s32(x2[32], x2[47]);
        x3[47] = vsubq_s32(x2[32], x2[47]);
        x3[33] = vaddq_s32(x2[33], x2[46]);
        x3[46] = vsubq_s32(x2[33], x2[46]);
        x3[34] = vaddq_s32(x2[34], x2[45]);
        x3[45] = vsubq_s32(x2[34], x2[45]);
        x3[35] = vaddq_s32(x2[35], x2[44]);
        x3[44] = vsubq_s32(x2[35], x2[44]);
        x3[36] = vaddq_s32(x2[36], x2[43]);
        x3[43] = vsubq_s32(x2[36], x2[43]);
        x3[37] = vaddq_s32(x2[37], x2[42]);
        x3[42] = vsubq_s32(x2[37], x2[42]);
        x3[38] = vaddq_s32(x2[38], x2[41]);
        x3[41] = vsubq_s32(x2[38], x2[41]);
        x3[39] = vaddq_s32(x2[39], x2[40]);
        x3[40] = vsubq_s32(x2[39], x2[40]);
        x3[48] = vsubq_s32(x2[63], x2[48]);
        x3[63] = vaddq_s32(x2[63], x2[48]);
        x3[49] = vsubq_s32(x2[62], x2[49]);
        x3[62] = vaddq_s32(x2[62], x2[49]);
        x3[50] = vsubq_s32(x2[61], x2[50]);
        x3[61] = vaddq_s32(x2[61], x2[50]);
        x3[51] = vsubq_s32(x2[60], x2[51]);
        x3[60] = vaddq_s32(x2[60], x2[51]);
        x3[52] = vsubq_s32(x2[59], x2[52]);
        x3[59] = vaddq_s32(x2[59], x2[52]);
        x3[53] = vsubq_s32(x2[58], x2[53]);
        x3[58] = vaddq_s32(x2[58], x2[53]);
        x3[54] = vsubq_s32(x2[57], x2[54]);
        x3[57] = vaddq_s32(x2[57], x2[54]);
        x3[55] = vsubq_s32(x2[56], x2[55]);
        x3[56] = vaddq_s32(x2[56], x2[55]);

        // stage 4
        int32x4_t x4[64];
        x4[0] = vaddq_s32(x3[0], x3[7]);
        x4[7] = vsubq_s32(x3[0], x3[7]);
        x4[1] = vaddq_s32(x3[1], x3[6]);
        x4[6] = vsubq_s32(x3[1], x3[6]);
        x4[2] = vaddq_s32(x3[2], x3[5]);
        x4[5] = vsubq_s32(x3[2], x3[5]);
        x4[3] = vaddq_s32(x3[3], x3[4]);
        x4[4] = vsubq_s32(x3[3], x3[4]);
        x4[8] = x3[8];
        x4[9] = x3[9];
        btf_32_type0_neon_new(cospi_m32, cospi_p32, x3[10], x3[13], x4[10], x4[13], cos_bit);
        btf_32_type0_neon_new(cospi_m32, cospi_p32, x3[11], x3[12], x4[11], x4[12], cos_bit);
        x4[14] = x3[14];
        x4[15] = x3[15];
        x4[16] = vaddq_s32(x3[16], x3[23]);
        x4[23] = vsubq_s32(x3[16], x3[23]);
        x4[17] = vaddq_s32(x3[17], x3[22]);
        x4[22] = vsubq_s32(x3[17], x3[22]);
        x4[18] = vaddq_s32(x3[18], x3[21]);
        x4[21] = vsubq_s32(x3[18], x3[21]);
        x4[19] = vaddq_s32(x3[19], x3[20]);
        x4[20] = vsubq_s32(x3[19], x3[20]);
        x4[24] = vsubq_s32(x3[31], x3[24]);
        x4[31] = vaddq_s32(x3[31], x3[24]);
        x4[25] = vsubq_s32(x3[30], x3[25]);
        x4[30] = vaddq_s32(x3[30], x3[25]);
        x4[26] = vsubq_s32(x3[29], x3[26]);
        x4[29] = vaddq_s32(x3[29], x3[26]);
        x4[27] = vsubq_s32(x3[28], x3[27]);
        x4[28] = vaddq_s32(x3[28], x3[27]);
        x4[32] = x3[32];
        x4[33] = x3[33];
        x4[34] = x3[34];
        x4[35] = x3[35];
        btf_32_type0_neon_new(cospi_m16, cospi_p48, x3[36], x3[59], x4[36], x4[59], cos_bit);
        btf_32_type0_neon_new(cospi_m16, cospi_p48, x3[37], x3[58], x4[37], x4[58], cos_bit);
        btf_32_type0_neon_new(cospi_m16, cospi_p48, x3[38], x3[57], x4[38], x4[57], cos_bit);
        btf_32_type0_neon_new(cospi_m16, cospi_p48, x3[39], x3[56], x4[39], x4[56], cos_bit);
        btf_32_type0_neon_new(cospi_m48, cospi_m16, x3[40], x3[55], x4[40], x4[55], cos_bit);
        btf_32_type0_neon_new(cospi_m48, cospi_m16, x3[41], x3[54], x4[41], x4[54], cos_bit);
        btf_32_type0_neon_new(cospi_m48, cospi_m16, x3[42], x3[53], x4[42], x4[53], cos_bit);
        btf_32_type0_neon_new(cospi_m48, cospi_m16, x3[43], x3[52], x4[43], x4[52], cos_bit);
        x4[44] = x3[44];
        x4[45] = x3[45];
        x4[46] = x3[46];
        x4[47] = x3[47];
        x4[48] = x3[48];
        x4[49] = x3[49];
        x4[50] = x3[50];
        x4[51] = x3[51];
        x4[60] = x3[60];
        x4[61] = x3[61];
        x4[62] = x3[62];
        x4[63] = x3[63];

        // stage 5
        int32x4_t x5[64];
        x5[0] = vaddq_s32(x4[0], x4[3]);
        x5[1] = vaddq_s32(x4[1], x4[2]);
        x5[4] = x4[4];
        btf_32_type0_neon_new(cospi_m32, cospi_p32, x4[5], x4[6], x5[5], x5[6], cos_bit);
        x5[7]  = x4[7];
        x5[8]  = vaddq_s32(x4[8], x4[11]);
        x5[11] = vsubq_s32(x4[8], x4[11]);
        x5[9]  = vaddq_s32(x4[9], x4[10]);
        x5[10] = vsubq_s32(x4[9], x4[10]);
        x5[12] = vsubq_s32(x4[15], x4[12]);
        x5[15] = vaddq_s32(x4[15], x4[12]);
        x5[13] = vsubq_s32(x4[14], x4[13]);
        x5[14] = vaddq_s32(x4[14], x4[13]);
        x5[16] = x4[16];
        x5[17] = x4[17];
        btf_32_type0_neon_new(cospi_m16, cospi_p48, x4[18], x4[29], x5[18], x5[29], cos_bit);
        btf_32_type0_neon_new(cospi_m16, cospi_p48, x4[19], x4[28], x5[19], x5[28], cos_bit);
        btf_32_type0_neon_new(cospi_m48, cospi_m16, x4[20], x4[27], x5[20], x5[27], cos_bit);
        btf_32_type0_neon_new(cospi_m48, cospi_m16, x4[21], x4[26], x5[21], x5[26], cos_bit);
        x5[22] = x4[22];
        x5[23] = x4[23];
        x5[24] = x4[24];
        x5[25] = x4[25];
        x5[30] = x4[30];
        x5[31] = x4[31];
        x5[32] = vaddq_s32(x4[32], x4[39]);
        x5[39] = vsubq_s32(x4[32], x4[39]);
        x5[33] = vaddq_s32(x4[33], x4[38]);
        x5[38] = vsubq_s32(x4[33], x4[38]);
        x5[34] = vaddq_s32(x4[34], x4[37]);
        x5[37] = vsubq_s32(x4[34], x4[37]);
        x5[35] = vaddq_s32(x4[35], x4[36]);
        x5[36] = vsubq_s32(x4[35], x4[36]);
        x5[40] = vsubq_s32(x4[47], x4[40]);
        x5[47] = vaddq_s32(x4[47], x4[40]);
        x5[41] = vsubq_s32(x4[46], x4[41]);
        x5[46] = vaddq_s32(x4[46], x4[41]);
        x5[42] = vsubq_s32(x4[45], x4[42]);
        x5[45] = vaddq_s32(x4[45], x4[42]);
        x5[43] = vsubq_s32(x4[44], x4[43]);
        x5[44] = vaddq_s32(x4[44], x4[43]);
        x5[48] = vaddq_s32(x4[48], x4[55]);
        x5[55] = vsubq_s32(x4[48], x4[55]);
        x5[49] = vaddq_s32(x4[49], x4[54]);
        x5[54] = vsubq_s32(x4[49], x4[54]);
        x5[50] = vaddq_s32(x4[50], x4[53]);
        x5[53] = vsubq_s32(x4[50], x4[53]);
        x5[51] = vaddq_s32(x4[51], x4[52]);
        x5[52] = vsubq_s32(x4[51], x4[52]);
        x5[56] = vsubq_s32(x4[63], x4[56]);
        x5[63] = vaddq_s32(x4[63], x4[56]);
        x5[57] = vsubq_s32(x4[62], x4[57]);
        x5[62] = vaddq_s32(x4[62], x4[57]);
        x5[58] = vsubq_s32(x4[61], x4[58]);
        x5[61] = vaddq_s32(x4[61], x4[58]);
        x5[59] = vsubq_s32(x4[60], x4[59]);
        x5[60] = vaddq_s32(x4[60], x4[59]);

        // stage 6
        int32x4_t x6[64];
        out[2 * 0 * stride] = half_btf_small(&cospi_p32, &x5[0], &cospi_p32, &x5[1], cos_bit);
        x6[4]               = vaddq_s32(x5[4], x5[5]);
        x6[7]               = vaddq_s32(x5[7], x5[6]);
        x6[8]               = x5[8];
        btf_32_type0_neon_new(cospi_m16, cospi_p48, x5[9], x5[14], x6[9], x6[14], cos_bit);
        btf_32_type0_neon_new(cospi_m48, cospi_m16, x5[10], x5[13], x6[10], x6[13], cos_bit);
        x6[11] = x5[11];
        x6[12] = x5[12];
        x6[15] = x5[15];
        x6[16] = vaddq_s32(x5[16], x5[19]);
        x6[19] = vsubq_s32(x5[16], x5[19]);
        x6[17] = vaddq_s32(x5[17], x5[18]);
        x6[18] = vsubq_s32(x5[17], x5[18]);
        x6[20] = vsubq_s32(x5[23], x5[20]);
        x6[23] = vaddq_s32(x5[23], x5[20]);
        x6[21] = vsubq_s32(x5[22], x5[21]);
        x6[22] = vaddq_s32(x5[22], x5[21]);
        x6[24] = vaddq_s32(x5[24], x5[27]);
        x6[27] = vsubq_s32(x5[24], x5[27]);
        x6[25] = vaddq_s32(x5[25], x5[26]);
        x6[26] = vsubq_s32(x5[25], x5[26]);
        x6[28] = vsubq_s32(x5[31], x5[28]);
        x6[31] = vaddq_s32(x5[31], x5[28]);
        x6[29] = vsubq_s32(x5[30], x5[29]);
        x6[30] = vaddq_s32(x5[30], x5[29]);
        x6[32] = x5[32];
        x6[33] = x5[33];
        btf_32_type0_neon_new(cospi_m08, cospi_p56, x5[34], x5[61], x6[34], x6[61], cos_bit);
        btf_32_type0_neon_new(cospi_m08, cospi_p56, x5[35], x5[60], x6[35], x6[60], cos_bit);
        btf_32_type0_neon_new(cospi_m56, cospi_m08, x5[36], x5[59], x6[36], x6[59], cos_bit);
        btf_32_type0_neon_new(cospi_m56, cospi_m08, x5[37], x5[58], x6[37], x6[58], cos_bit);
        x6[38] = x5[38];
        x6[39] = x5[39];
        x6[40] = x5[40];
        x6[41] = x5[41];
        btf_32_type0_neon_new(cospi_m40, cospi_p24, x5[42], x5[53], x6[42], x6[53], cos_bit);
        btf_32_type0_neon_new(cospi_m40, cospi_p24, x5[43], x5[52], x6[43], x6[52], cos_bit);
        btf_32_type0_neon_new(cospi_m24, cospi_m40, x5[44], x5[51], x6[44], x6[51], cos_bit);
        btf_32_type0_neon_new(cospi_m24, cospi_m40, x5[45], x5[50], x6[45], x6[50], cos_bit);
        x6[46] = x5[46];
        x6[47] = x5[47];
        x6[48] = x5[48];
        x6[49] = x5[49];
        x6[54] = x5[54];
        x6[55] = x5[55];
        x6[56] = x5[56];
        x6[57] = x5[57];
        x6[62] = x5[62];
        x6[63] = x5[63];

        // stage 7
        int32x4_t x7[64];
        out[2 * 8 * stride] = half_btf_small(&cospi_p56, &x6[4], &cospi_p08, &x6[7], cos_bit);
        x7[8]               = vaddq_s32(x6[8], x6[9]);
        x7[11]              = vaddq_s32(x6[11], x6[10]);
        x7[12]              = vaddq_s32(x6[12], x6[13]);
        x7[15]              = vaddq_s32(x6[15], x6[14]);
        x7[16]              = x6[16];
        btf_32_type0_neon_new(cospi_m08, cospi_p56, x6[17], x6[30], x7[17], x7[30], cos_bit);
        btf_32_type0_neon_new(cospi_m56, cospi_m08, x6[18], x6[29], x7[18], x7[29], cos_bit);
        x7[19] = x6[19];
        x7[20] = x6[20];
        btf_32_type0_neon_new(cospi_m40, cospi_p24, x6[21], x6[26], x7[21], x7[26], cos_bit);
        btf_32_type0_neon_new(cospi_m24, cospi_m40, x6[22], x6[25], x7[22], x7[25], cos_bit);
        x7[23] = x6[23];
        x7[24] = x6[24];
        x7[27] = x6[27];
        x7[28] = x6[28];
        x7[31] = x6[31];
        x7[32] = vaddq_s32(x6[32], x6[35]);
        x7[35] = vsubq_s32(x6[32], x6[35]);
        x7[33] = vaddq_s32(x6[33], x6[34]);
        x7[34] = vsubq_s32(x6[33], x6[34]);
        x7[36] = vsubq_s32(x6[39], x6[36]);
        x7[39] = vaddq_s32(x6[39], x6[36]);
        x7[37] = vsubq_s32(x6[38], x6[37]);
        x7[38] = vaddq_s32(x6[38], x6[37]);
        x7[40] = vaddq_s32(x6[40], x6[43]);
        x7[43] = vsubq_s32(x6[40], x6[43]);
        x7[41] = vaddq_s32(x6[41], x6[42]);
        x7[42] = vsubq_s32(x6[41], x6[42]);
        x7[44] = vsubq_s32(x6[47], x6[44]);
        x7[47] = vaddq_s32(x6[47], x6[44]);
        x7[45] = vsubq_s32(x6[46], x6[45]);
        x7[46] = vaddq_s32(x6[46], x6[45]);
        x7[48] = vaddq_s32(x6[48], x6[51]);
        x7[51] = vsubq_s32(x6[48], x6[51]);
        x7[49] = vaddq_s32(x6[49], x6[50]);
        x7[50] = vsubq_s32(x6[49], x6[50]);
        x7[52] = vsubq_s32(x6[55], x6[52]);
        x7[55] = vaddq_s32(x6[55], x6[52]);
        x7[53] = vsubq_s32(x6[54], x6[53]);
        x7[54] = vaddq_s32(x6[54], x6[53]);
        x7[56] = vaddq_s32(x6[56], x6[59]);
        x7[59] = vsubq_s32(x6[56], x6[59]);
        x7[57] = vaddq_s32(x6[57], x6[58]);
        x7[58] = vsubq_s32(x6[57], x6[58]);
        x7[60] = vsubq_s32(x6[63], x6[60]);
        x7[63] = vaddq_s32(x6[63], x6[60]);
        x7[61] = vsubq_s32(x6[62], x6[61]);
        x7[62] = vaddq_s32(x6[62], x6[61]);

        // stage 8
        int32x4_t x8[40];
        out[2 * 4 * stride]  = half_btf_small(&cospi_p60, &x7[8], &cospi_p04, &x7[15], cos_bit);
        out[2 * 12 * stride] = half_btf_small(&cospi_p12, &x7[12], &cospi_m52, &x7[11], cos_bit);
        x8[0]                = vaddq_s32(x7[16], x7[17]);
        x8[1]                = vaddq_s32(x7[19], x7[18]);
        x8[2]                = vaddq_s32(x7[20], x7[21]);
        x8[3]                = vaddq_s32(x7[23], x7[22]);
        x8[4]                = vaddq_s32(x7[24], x7[25]);
        x8[5]                = vaddq_s32(x7[27], x7[26]);
        x8[6]                = vaddq_s32(x7[28], x7[29]);
        x8[7]                = vaddq_s32(x7[31], x7[30]);
        x8[8]                = x7[32];
        btf_32_type0_neon_new(cospi_m04, cospi_p60, x7[33], x7[62], x8[9], x8[32], cos_bit);
        btf_32_type0_neon_new(cospi_m60, cospi_m04, x7[34], x7[61], x8[10], x8[33], cos_bit);
        x8[11] = x7[35];
        x8[12] = x7[36];
        btf_32_type0_neon_new(cospi_m36, cospi_p28, x7[37], x7[58], x8[13], x8[34], cos_bit);
        btf_32_type0_neon_new(cospi_m28, cospi_m36, x7[38], x7[57], x8[14], x8[35], cos_bit);
        x8[15] = x7[39];
        x8[16] = x7[40];
        btf_32_type0_neon_new(cospi_m20, cospi_p44, x7[41], x7[54], x8[17], x8[36], cos_bit);
        btf_32_type0_neon_new(cospi_m44, cospi_m20, x7[42], x7[53], x8[18], x8[37], cos_bit);
        x8[19] = x7[43];
        x8[20] = x7[44];
        btf_32_type0_neon_new(cospi_m52, cospi_p12, x7[45], x7[50], x8[21], x8[38], cos_bit);
        btf_32_type0_neon_new(cospi_m12, cospi_m52, x7[46], x7[49], x8[22], x8[39], cos_bit);
        x8[23] = x7[47];
        x8[24] = x7[48];
        x8[25] = x7[51];
        x8[26] = x7[52];
        x8[27] = x7[55];
        x8[28] = x7[56];
        x8[29] = x7[59];
        x8[30] = x7[60];
        x8[31] = x7[63];

        // stage 9
        int32x4_t x9[16];
        out[2 * 2 * stride]  = half_btf_small(&cospi_p62, &x8[0], &cospi_p02, &x8[7], cos_bit);
        out[2 * 14 * stride] = half_btf_small(&cospi_p14, &x8[6], &cospi_m50, &x8[1], cos_bit);
        out[2 * 10 * stride] = half_btf_small(&cospi_p54, &x8[2], &cospi_p10, &x8[5], cos_bit);
        out[2 * 6 * stride]  = half_btf_small(&cospi_p06, &x8[4], &cospi_m58, &x8[3], cos_bit);
        x9[0]                = vaddq_s32(x8[8], x8[9]);
        x9[1]                = vaddq_s32(x8[11], x8[10]);
        x9[2]                = vaddq_s32(x8[12], x8[13]);
        x9[3]                = vaddq_s32(x8[15], x8[14]);
        x9[4]                = vaddq_s32(x8[16], x8[17]);
        x9[5]                = vaddq_s32(x8[19], x8[18]);
        x9[6]                = vaddq_s32(x8[20], x8[21]);
        x9[7]                = vaddq_s32(x8[23], x8[22]);
        x9[8]                = vaddq_s32(x8[24], x8[39]);
        x9[9]                = vaddq_s32(x8[25], x8[38]);
        x9[10]               = vaddq_s32(x8[26], x8[37]);
        x9[11]               = vaddq_s32(x8[27], x8[36]);
        x9[12]               = vaddq_s32(x8[28], x8[35]);
        x9[13]               = vaddq_s32(x8[29], x8[34]);
        x9[14]               = vaddq_s32(x8[30], x8[33]);
        x9[15]               = vaddq_s32(x8[31], x8[32]);

        // stage 10
        out[2 * 1 * stride]  = half_btf_small(&cospi_p63, &x9[0], &cospi_p01, &x9[15], cos_bit);
        out[2 * 15 * stride] = half_btf_small(&cospi_p15, &x9[14], &cospi_m49, &x9[1], cos_bit);
        out[2 * 9 * stride]  = half_btf_small(&cospi_p55, &x9[2], &cospi_p09, &x9[13], cos_bit);
        out[2 * 7 * stride]  = half_btf_small(&cospi_p07, &x9[12], &cospi_m57, &x9[3], cos_bit);
        out[2 * 5 * stride]  = half_btf_small(&cospi_p59, &x9[4], &cospi_p05, &x9[11], cos_bit);
        out[2 * 11 * stride] = half_btf_small(&cospi_p11, &x9[10], &cospi_m53, &x9[5], cos_bit);
        out[2 * 13 * stride] = half_btf_small(&cospi_p51, &x9[6], &cospi_p13, &x9[9], cos_bit);
        out[2 * 3 * stride]  = half_btf_small(&cospi_p03, &x9[8], &cospi_m61, &x9[7], cos_bit);
    }
}

static INLINE void transpose_16x16_in_64x64_neon(const int32x4_t *restrict in, int32x4_t *restrict out) {
    TRANSPOSE_2X4X4_NEON(in, 0, 16, 32, 48, out, 0, 64, 16, 80, 32, 96, 48, 112);
    TRANSPOSE_2X4X4_NEON(in, 64, 80, 96, 112, out, 1, 65, 17, 81, 33, 97, 49, 113);
    TRANSPOSE_2X4X4_NEON(in, 2, 18, 34, 50, out, 128, 192, 144, 208, 160, 224, 176, 240);
    TRANSPOSE_2X4X4_NEON(in, 66, 82, 98, 114, out, 129, 193, 145, 209, 161, 225, 177, 241);

    TRANSPOSE_2X4X4_NEON(in, 128, 144, 160, 176, out, 2, 66, 18, 82, 34, 98, 50, 114);
    TRANSPOSE_2X4X4_NEON(in, 192, 208, 224, 240, out, 3, 67, 19, 83, 35, 99, 51, 115);
    TRANSPOSE_2X4X4_NEON(in, 130, 146, 162, 178, out, 130, 194, 146, 210, 162, 226, 178, 242);
    TRANSPOSE_2X4X4_NEON(in, 194, 210, 226, 242, out, 131, 195, 147, 211, 163, 227, 179, 243);
}

void svt_av1_fwd_txfm2d_64x64_N4_neon(int16_t *input, int32_t *output, uint32_t stride, TxType tx_type, uint8_t bd) {
    (void)bd;
    int32x4_t     buf1[1024];
    int32x4_t    *buf2    = (int32x4_t *)output;
    const int32_t txw_idx = tx_size_wide_log2[TX_64X64] - tx_size_wide_log2[0];
    const int32_t txh_idx = tx_size_high_log2[TX_64X64] - tx_size_high_log2[0];
    const int8_t *shift   = fwd_txfm_shift_ls[TX_64X64];

    switch (tx_type) {
    case IDTX:
        load_buffer_16x16_in_64x64_neon(input, stride, buf2);
        fidtx64x64_N4_neon(buf2, buf1);
        av1_round_shift_array_64_N4_neon(buf1, buf2, 512 / 4, -shift[1]);
        fidtx64x64_N4_neon(buf2, buf1);
        av1_round_shift_array_64_N4_neon(buf1, (int32x4_t *)output, 512 / 4, -shift[2]);
        clear_buffer_wxh_N4((int32x4_t *)output, 8, 64);
        break;
    case DCT_DCT:
        load_buffer_64x64_neon(input, stride, buf2);
        av1_fdct64_new_N4_neon(buf2, buf1, fwd_cos_bit_col[txw_idx][txh_idx], 64, 8);
        av1_round_shift_array_32_neon(buf1, buf2, 1024 / 4, -shift[1]);
        transpose_16x16_in_64x64_neon(buf2, buf1);
        transpose_16x16_in_64x64_neon(buf2 + 4, buf1 + 256);
        transpose_16x16_in_64x64_neon(buf2 + 8, buf1 + 512);
        transpose_16x16_in_64x64_neon(buf2 + 12, buf1 + 768);
        av1_fdct64_new_N4_neon(buf1, buf2, fwd_cos_bit_row[txw_idx][txh_idx], 16, 8);
        av1_round_shift_array_64_N4_neon(buf2, buf1, 512 / 4, -shift[2]);
        transpose_16x16_in_64x64_neon(buf1, (int32x4_t *)output); //top-left
        clear_buffer_wxh_N4((int32x4_t *)output, 8, 64);
        break;
    default: assert(0);
    }
}

static INLINE void load_buffer_8x8_in_32x32_neon(const int16_t *restrict input, int32x4_t *restrict output,
                                                 int32_t stride) {
    for (int32_t i = 0; i < 8; ++i) {
        output[0] = vmovl_s16(vld1_s16(input + 0));
        output[1] = vmovl_s16(vld1_s16(input + 4));

        input += stride;
        output += 8;
    }
}

static INLINE void av1_round_shift_array_32_N4_neon(int32x4_t *input, int32x4_t *output, const int32_t size,
                                                    const int32_t bit) {
    const int32x4_t vbits = vdupq_n_s32(-bit);
    for (int32_t i = 0; i < size; i += 4) {
        output[2 * i]     = vrshlq_s32(input[2 * i + 0], vbits);
        output[2 * i + 1] = vrshlq_s32(input[2 * i + 1], vbits);
    }
}

static void fidtx_wxh_N4_neon(const int32x4_t *restrict input, int32x4_t *restrict output, int32_t size, int32_t step) {
    for (int32_t i = 0; i < size; i += step) {
        output[2 * i]     = vshlq_n_s32(input[2 * i], 2);
        output[2 * i + 1] = vshlq_n_s32(input[2 * i + 1], 2);
    }
}

static INLINE void write_buffer_32x32_N4(const int32x4_t *res, int32_t *output) {
    const int32x4_t zero = vdupq_n_s32(0);
    uint32_t        i;

    for (i = 0; i < 8; i++) {
        vst1q_s32(output + i * 32 + 0, res[i * 8]);
        vst1q_s32(output + i * 32 + 4, res[i * 8 + 1]);
        vst1q_s32(output + i * 32 + 8, zero);
        vst1q_s32(output + i * 32 + 12, zero);
        vst1q_s32(output + i * 32 + 16, zero);
        vst1q_s32(output + i * 32 + 20, zero);
        vst1q_s32(output + i * 32 + 24, zero);
        vst1q_s32(output + i * 32 + 28, zero);
    }

    for (; i < 32; i++) {
        vst1q_s32(output + i * 32 + 0, zero);
        vst1q_s32(output + i * 32 + 4, zero);
        vst1q_s32(output + i * 32 + 8, zero);
        vst1q_s32(output + i * 32 + 12, zero);
        vst1q_s32(output + i * 32 + 16, zero);
        vst1q_s32(output + i * 32 + 20, zero);
        vst1q_s32(output + i * 32 + 24, zero);
        vst1q_s32(output + i * 32 + 28, zero);
    }
}

static INLINE void load_buffer_32x32_neon(const int16_t *restrict input, int32x4_t *restrict output, int32_t stride) {
    for (int32_t i = 0; i < 32; ++i) {
        output[0] = vmovl_s16(vld1_s16(input + 0 * 4));
        output[1] = vmovl_s16(vld1_s16(input + 1 * 4));
        output[2] = vmovl_s16(vld1_s16(input + 2 * 4));
        output[3] = vmovl_s16(vld1_s16(input + 3 * 4));
        output[4] = vmovl_s16(vld1_s16(input + 4 * 4));
        output[5] = vmovl_s16(vld1_s16(input + 5 * 4));
        output[6] = vmovl_s16(vld1_s16(input + 6 * 4));
        output[7] = vmovl_s16(vld1_s16(input + 7 * 4));

        input += stride;
        output += 8;
    }
}

static void av1_fdct32_new_N4_neon(const int32x4_t *input, int32x4_t *output, int8_t cos_bit, const int32_t col_num,
                                   const int32_t stride) {
    const int32_t *cospi   = cospi_arr(cos_bit);
    const int32_t  columns = col_num >> 2;

    const int32x4_t cospi_m32 = vdupq_n_s32(-cospi[32]);
    const int32x4_t cospi_p32 = vdupq_n_s32(cospi[32]);
    const int32x4_t cospi_m16 = vdupq_n_s32(-cospi[16]);
    const int32x4_t cospi_p48 = vdupq_n_s32(cospi[48]);
    const int32x4_t cospi_m48 = vdupq_n_s32(-cospi[48]);
    const int32x4_t cospi_m08 = vdupq_n_s32(-cospi[8]);
    const int32x4_t cospi_p56 = vdupq_n_s32(cospi[56]);
    const int32x4_t cospi_m56 = vdupq_n_s32(-cospi[56]);
    const int32x4_t cospi_m40 = vdupq_n_s32(-cospi[40]);
    const int32x4_t cospi_p24 = vdupq_n_s32(cospi[24]);
    const int32x4_t cospi_m24 = vdupq_n_s32(-cospi[24]);
    const int32x4_t cospi_p08 = vdupq_n_s32(cospi[8]);
    const int32x4_t cospi_p04 = vdupq_n_s32(cospi[4]);
    const int32x4_t cospi_p60 = vdupq_n_s32(cospi[60]);
    const int32x4_t cospi_m52 = vdupq_n_s32(-cospi[52]);
    const int32x4_t cospi_p12 = vdupq_n_s32(cospi[12]);
    const int32x4_t cospi_p02 = vdupq_n_s32(cospi[2]);
    const int32x4_t cospi_p06 = vdupq_n_s32(cospi[6]);
    const int32x4_t cospi_p62 = vdupq_n_s32(cospi[62]);
    const int32x4_t cospi_m50 = vdupq_n_s32(-cospi[50]);
    const int32x4_t cospi_p14 = vdupq_n_s32(cospi[14]);
    const int32x4_t cospi_p10 = vdupq_n_s32(cospi[10]);
    const int32x4_t cospi_p54 = vdupq_n_s32(cospi[54]);
    const int32x4_t cospi_m58 = vdupq_n_s32(-cospi[58]);

    int32x4_t buf0[32];
    int32x4_t buf1[32];

    for (int32_t col = 0; col < columns; col++) {
        const int32x4_t *in  = &input[col];
        int32x4_t       *out = &output[col];

        // stage 0
        // stage 1
        buf1[0]  = vaddq_s32(in[2 * 0 * stride], in[2 * 31 * stride]);
        buf1[31] = vsubq_s32(in[2 * 0 * stride], in[2 * 31 * stride]);
        buf1[1]  = vaddq_s32(in[2 * 1 * stride], in[2 * 30 * stride]);
        buf1[30] = vsubq_s32(in[2 * 1 * stride], in[2 * 30 * stride]);
        buf1[2]  = vaddq_s32(in[2 * 2 * stride], in[2 * 29 * stride]);
        buf1[29] = vsubq_s32(in[2 * 2 * stride], in[2 * 29 * stride]);
        buf1[3]  = vaddq_s32(in[2 * 3 * stride], in[2 * 28 * stride]);
        buf1[28] = vsubq_s32(in[2 * 3 * stride], in[2 * 28 * stride]);
        buf1[4]  = vaddq_s32(in[2 * 4 * stride], in[2 * 27 * stride]);
        buf1[27] = vsubq_s32(in[2 * 4 * stride], in[2 * 27 * stride]);
        buf1[5]  = vaddq_s32(in[2 * 5 * stride], in[2 * 26 * stride]);
        buf1[26] = vsubq_s32(in[2 * 5 * stride], in[2 * 26 * stride]);
        buf1[6]  = vaddq_s32(in[2 * 6 * stride], in[2 * 25 * stride]);
        buf1[25] = vsubq_s32(in[2 * 6 * stride], in[2 * 25 * stride]);
        buf1[7]  = vaddq_s32(in[2 * 7 * stride], in[2 * 24 * stride]);
        buf1[24] = vsubq_s32(in[2 * 7 * stride], in[2 * 24 * stride]);
        buf1[8]  = vaddq_s32(in[2 * 8 * stride], in[2 * 23 * stride]);
        buf1[23] = vsubq_s32(in[2 * 8 * stride], in[2 * 23 * stride]);
        buf1[9]  = vaddq_s32(in[2 * 9 * stride], in[2 * 22 * stride]);
        buf1[22] = vsubq_s32(in[2 * 9 * stride], in[2 * 22 * stride]);
        buf1[10] = vaddq_s32(in[2 * 10 * stride], in[2 * 21 * stride]);
        buf1[21] = vsubq_s32(in[2 * 10 * stride], in[2 * 21 * stride]);
        buf1[11] = vaddq_s32(in[2 * 11 * stride], in[2 * 20 * stride]);
        buf1[20] = vsubq_s32(in[2 * 11 * stride], in[2 * 20 * stride]);
        buf1[12] = vaddq_s32(in[2 * 12 * stride], in[2 * 19 * stride]);
        buf1[19] = vsubq_s32(in[2 * 12 * stride], in[2 * 19 * stride]);
        buf1[13] = vaddq_s32(in[2 * 13 * stride], in[2 * 18 * stride]);
        buf1[18] = vsubq_s32(in[2 * 13 * stride], in[2 * 18 * stride]);
        buf1[14] = vaddq_s32(in[2 * 14 * stride], in[2 * 17 * stride]);
        buf1[17] = vsubq_s32(in[2 * 14 * stride], in[2 * 17 * stride]);
        buf1[15] = vaddq_s32(in[2 * 15 * stride], in[2 * 16 * stride]);
        buf1[16] = vsubq_s32(in[2 * 15 * stride], in[2 * 16 * stride]);

        // stage 2
        buf0[0]  = vaddq_s32(buf1[0], buf1[15]);
        buf0[15] = vsubq_s32(buf1[0], buf1[15]);
        buf0[1]  = vaddq_s32(buf1[1], buf1[14]);
        buf0[14] = vsubq_s32(buf1[1], buf1[14]);
        buf0[2]  = vaddq_s32(buf1[2], buf1[13]);
        buf0[13] = vsubq_s32(buf1[2], buf1[13]);
        buf0[3]  = vaddq_s32(buf1[3], buf1[12]);
        buf0[12] = vsubq_s32(buf1[3], buf1[12]);
        buf0[4]  = vaddq_s32(buf1[4], buf1[11]);
        buf0[11] = vsubq_s32(buf1[4], buf1[11]);
        buf0[5]  = vaddq_s32(buf1[5], buf1[10]);
        buf0[10] = vsubq_s32(buf1[5], buf1[10]);
        buf0[6]  = vaddq_s32(buf1[6], buf1[9]);
        buf0[9]  = vsubq_s32(buf1[6], buf1[9]);
        buf0[7]  = vaddq_s32(buf1[7], buf1[8]);
        buf0[8]  = vsubq_s32(buf1[7], buf1[8]);
        buf0[16] = buf1[16];
        buf0[17] = buf1[17];
        buf0[18] = buf1[18];
        buf0[19] = buf1[19];
        btf_32_type0_neon_new(cospi_m32, cospi_p32, buf1[20], buf1[27], buf0[20], buf0[27], cos_bit);
        btf_32_type0_neon_new(cospi_m32, cospi_p32, buf1[21], buf1[26], buf0[21], buf0[26], cos_bit);
        btf_32_type0_neon_new(cospi_m32, cospi_p32, buf1[22], buf1[25], buf0[22], buf0[25], cos_bit);
        btf_32_type0_neon_new(cospi_m32, cospi_p32, buf1[23], buf1[24], buf0[23], buf0[24], cos_bit);
        buf0[28] = buf1[28];
        buf0[29] = buf1[29];
        buf0[30] = buf1[30];
        buf0[31] = buf1[31];

        // stage 3
        buf1[0] = vaddq_s32(buf0[0], buf0[7]);
        buf1[7] = vsubq_s32(buf0[0], buf0[7]);
        buf1[1] = vaddq_s32(buf0[1], buf0[6]);
        buf1[6] = vsubq_s32(buf0[1], buf0[6]);
        buf1[2] = vaddq_s32(buf0[2], buf0[5]);
        buf1[5] = vsubq_s32(buf0[2], buf0[5]);
        buf1[3] = vaddq_s32(buf0[3], buf0[4]);
        buf1[4] = vsubq_s32(buf0[3], buf0[4]);
        buf1[8] = buf0[8];
        buf1[9] = buf0[9];
        btf_32_type0_neon_new(cospi_m32, cospi_p32, buf0[10], buf0[13], buf1[10], buf1[13], cos_bit);
        btf_32_type0_neon_new(cospi_m32, cospi_p32, buf0[11], buf0[12], buf1[11], buf1[12], cos_bit);
        buf1[14] = buf0[14];
        buf1[15] = buf0[15];
        buf1[16] = vaddq_s32(buf0[16], buf0[23]);
        buf1[23] = vsubq_s32(buf0[16], buf0[23]);
        buf1[17] = vaddq_s32(buf0[17], buf0[22]);
        buf1[22] = vsubq_s32(buf0[17], buf0[22]);
        buf1[18] = vaddq_s32(buf0[18], buf0[21]);
        buf1[21] = vsubq_s32(buf0[18], buf0[21]);
        buf1[19] = vaddq_s32(buf0[19], buf0[20]);
        buf1[20] = vsubq_s32(buf0[19], buf0[20]);
        buf1[24] = vsubq_s32(buf0[31], buf0[24]);
        buf1[31] = vaddq_s32(buf0[31], buf0[24]);
        buf1[25] = vsubq_s32(buf0[30], buf0[25]);
        buf1[30] = vaddq_s32(buf0[30], buf0[25]);
        buf1[26] = vsubq_s32(buf0[29], buf0[26]);
        buf1[29] = vaddq_s32(buf0[29], buf0[26]);
        buf1[27] = vsubq_s32(buf0[28], buf0[27]);
        buf1[28] = vaddq_s32(buf0[28], buf0[27]);

        // stage 4
        buf0[0] = vaddq_s32(buf1[0], buf1[3]);
        buf0[1] = vaddq_s32(buf1[1], buf1[2]);
        buf0[4] = buf1[4];
        btf_32_type0_neon_new(cospi_m32, cospi_p32, buf1[5], buf1[6], buf0[5], buf0[6], cos_bit);
        buf0[7]  = buf1[7];
        buf0[8]  = vaddq_s32(buf1[8], buf1[11]);
        buf0[11] = vsubq_s32(buf1[8], buf1[11]);
        buf0[9]  = vaddq_s32(buf1[9], buf1[10]);
        buf0[10] = vsubq_s32(buf1[9], buf1[10]);
        buf0[12] = vsubq_s32(buf1[15], buf1[12]);
        buf0[15] = vaddq_s32(buf1[15], buf1[12]);
        buf0[13] = vsubq_s32(buf1[14], buf1[13]);
        buf0[14] = vaddq_s32(buf1[14], buf1[13]);
        buf0[16] = buf1[16];
        buf0[17] = buf1[17];
        btf_32_type0_neon_new(cospi_m16, cospi_p48, buf1[18], buf1[29], buf0[18], buf0[29], cos_bit);
        btf_32_type0_neon_new(cospi_m16, cospi_p48, buf1[19], buf1[28], buf0[19], buf0[28], cos_bit);
        btf_32_type0_neon_new(cospi_m48, cospi_m16, buf1[20], buf1[27], buf0[20], buf0[27], cos_bit);
        btf_32_type0_neon_new(cospi_m48, cospi_m16, buf1[21], buf1[26], buf0[21], buf0[26], cos_bit);
        buf0[22] = buf1[22];
        buf0[23] = buf1[23];
        buf0[24] = buf1[24];
        buf0[25] = buf1[25];
        buf0[30] = buf1[30];
        buf0[31] = buf1[31];

        // stage 5
        buf1[0] = half_btf_small(&cospi_p32, &buf0[0], &cospi_p32, &buf0[1], cos_bit);
        buf1[4] = vaddq_s32(buf0[4], buf0[5]);
        buf1[7] = vaddq_s32(buf0[7], buf0[6]);
        buf1[8] = buf0[8];
        btf_32_type0_neon_new(cospi_m16, cospi_p48, buf0[9], buf0[14], buf1[9], buf1[14], cos_bit);
        btf_32_type0_neon_new(cospi_m48, cospi_m16, buf0[10], buf0[13], buf1[10], buf1[13], cos_bit);
        buf1[11] = buf0[11];
        buf1[12] = buf0[12];
        buf1[15] = buf0[15];
        buf1[16] = vaddq_s32(buf0[16], buf0[19]);
        buf1[19] = vsubq_s32(buf0[16], buf0[19]);
        buf1[17] = vaddq_s32(buf0[17], buf0[18]);
        buf1[18] = vsubq_s32(buf0[17], buf0[18]);
        buf1[20] = vsubq_s32(buf0[23], buf0[20]);
        buf1[23] = vaddq_s32(buf0[23], buf0[20]);
        buf1[21] = vsubq_s32(buf0[22], buf0[21]);
        buf1[22] = vaddq_s32(buf0[22], buf0[21]);
        buf1[24] = vaddq_s32(buf0[24], buf0[27]);
        buf1[27] = vsubq_s32(buf0[24], buf0[27]);
        buf1[25] = vaddq_s32(buf0[25], buf0[26]);
        buf1[26] = vsubq_s32(buf0[25], buf0[26]);
        buf1[28] = vsubq_s32(buf0[31], buf0[28]);
        buf1[31] = vaddq_s32(buf0[31], buf0[28]);
        buf1[29] = vsubq_s32(buf0[30], buf0[29]);
        buf1[30] = vaddq_s32(buf0[30], buf0[29]);

        // stage 6
        buf0[0]  = buf1[0];
        buf0[4]  = half_btf_small(&cospi_p56, &buf1[4], &cospi_p08, &buf1[7], cos_bit);
        buf0[8]  = vaddq_s32(buf1[8], buf1[9]);
        buf0[11] = vaddq_s32(buf1[11], buf1[10]);
        buf0[12] = vaddq_s32(buf1[12], buf1[13]);
        buf0[15] = vaddq_s32(buf1[15], buf1[14]);
        buf0[16] = buf1[16];
        btf_32_type0_neon_new(cospi_m08, cospi_p56, buf1[17], buf1[30], buf0[17], buf0[30], cos_bit);
        btf_32_type0_neon_new(cospi_m56, cospi_m08, buf1[18], buf1[29], buf0[18], buf0[29], cos_bit);
        buf0[19] = buf1[19];
        buf0[20] = buf1[20];
        btf_32_type0_neon_new(cospi_m40, cospi_p24, buf1[21], buf1[26], buf0[21], buf0[26], cos_bit);
        btf_32_type0_neon_new(cospi_m24, cospi_m40, buf1[22], buf1[25], buf0[22], buf0[25], cos_bit);
        buf0[23] = buf1[23];
        buf0[24] = buf1[24];
        buf0[27] = buf1[27];
        buf0[28] = buf1[28];
        buf0[31] = buf1[31];

        // stage 7
        buf1[0]  = buf0[0];
        buf1[4]  = buf0[4];
        buf1[8]  = half_btf_small(&cospi_p60, &buf0[8], &cospi_p04, &buf0[15], cos_bit);
        buf1[12] = half_btf_small(&cospi_p12, &buf0[12], &cospi_m52, &buf0[11], cos_bit);

        buf1[16] = vaddq_s32(buf0[16], buf0[17]);
        buf1[19] = vaddq_s32(buf0[19], buf0[18]);
        buf1[20] = vaddq_s32(buf0[20], buf0[21]);
        buf1[23] = vaddq_s32(buf0[23], buf0[22]);
        buf1[24] = vaddq_s32(buf0[24], buf0[25]);
        buf1[27] = vaddq_s32(buf0[27], buf0[26]);
        buf1[28] = vaddq_s32(buf0[28], buf0[29]);
        buf1[31] = vaddq_s32(buf0[31], buf0[30]);

        // stage 8
        buf0[0]  = buf1[0];
        buf0[4]  = buf1[4];
        buf0[8]  = buf1[8];
        buf0[12] = buf1[12];
        buf0[16] = half_btf_small(&cospi_p62, &buf1[16], &cospi_p02, &buf1[31], cos_bit);
        buf0[28] = half_btf_small(&cospi_p14, &buf1[28], &cospi_m50, &buf1[19], cos_bit);
        buf0[20] = half_btf_small(&cospi_p54, &buf1[20], &cospi_p10, &buf1[27], cos_bit);
        buf0[24] = half_btf_small(&cospi_p06, &buf1[24], &cospi_m58, &buf1[23], cos_bit);

        // stage 9
        out[2 * 0 * stride] = buf0[0];
        out[2 * 1 * stride] = buf0[16];
        out[2 * 2 * stride] = buf0[8];
        out[2 * 3 * stride] = buf0[24];
        out[2 * 4 * stride] = buf0[4];
        out[2 * 5 * stride] = buf0[20];
        out[2 * 6 * stride] = buf0[12];
        out[2 * 7 * stride] = buf0[28];
    }
}

static AOM_FORCE_INLINE void fdct32x32_N4_col_neon(const int32x4_t *input, int32x4_t *output, const int8_t cos_bit) {
    const int32_t txfm_size   = 32;
    const int32_t num_per_256 = 8;
    int32_t       col_num     = txfm_size / num_per_256;
    av1_fdct32_new_N4_neon(input, output, cos_bit, txfm_size, col_num);
}

static INLINE void transpose_8x8_in_32x32_neon(const int32x4_t *restrict in, int32x4_t *restrict out) {
    TRANSPOSE_2X4X4_NEON(in, 0, 8, 16, 24, out, 0, 32, 8, 40, 16, 48, 24, 56);
    TRANSPOSE_2X4X4_NEON(in, 32, 40, 48, 56, out, 1, 33, 9, 41, 17, 49, 25, 57);
}

static AOM_FORCE_INLINE void fdct32x32_N4_row_neon(const int32x4_t *restrict input, int32x4_t *restrict output,
                                                   const int8_t cos_bit) {
    const int32_t txfm_size   = 32;
    const int32_t num_per_256 = 8;
    int32_t       col_num     = txfm_size / num_per_256;
    av1_fdct32_new_N4_neon(input, output, cos_bit, txfm_size / 4, col_num);
}

static INLINE void load_buffer_8x32_in_32x32_neon(const int16_t *restrict input, int32x4_t *restrict output,
                                                  int32_t stride) {
    for (int32_t i = 0; i < 32; ++i) {
        output[0] = vmovl_s16(vld1_s16(input + 0));
        output[1] = vmovl_s16(vld1_s16(input + 4));

        input += stride;
        output += 8;
    }
}

static INLINE void load_buffer_32x8_in_32x32_neon(const int16_t *restrict input, int32x4_t *restrict output,
                                                  int32_t stride) {
    for (int32_t i = 0; i < 8; ++i) {
        output[0] = vmovl_s16(vld1_s16(input + 0 * 4));
        output[1] = vmovl_s16(vld1_s16(input + 1 * 4));
        output[2] = vmovl_s16(vld1_s16(input + 2 * 4));
        output[3] = vmovl_s16(vld1_s16(input + 3 * 4));
        output[4] = vmovl_s16(vld1_s16(input + 4 * 4));
        output[5] = vmovl_s16(vld1_s16(input + 5 * 4));
        output[6] = vmovl_s16(vld1_s16(input + 6 * 4));
        output[7] = vmovl_s16(vld1_s16(input + 7 * 4));

        input += stride;
        output += 8;
    }
}

void svt_av1_fwd_txfm2d_32x32_N4_neon(int16_t *input, int32_t *output, uint32_t stride, TxType tx_type, uint8_t bd) {
    const int8_t *shift           = fwd_txfm_shift_ls[TX_32X32];
    const int32_t txw_idx         = tx_size_wide_log2[TX_32X32] - tx_size_wide_log2[0];
    const int32_t txh_idx         = tx_size_high_log2[TX_32X32] - tx_size_high_log2[0];
    const int8_t  cos_bit_col     = fwd_cos_bit_col[txw_idx][txh_idx];
    const int8_t  cos_bit_row     = fwd_cos_bit_row[txw_idx][txh_idx];
    const int32_t txfm2d_size_128 = 32 * 32 / 4;
    int32x4_t     buf[256];
    int32x4_t    *out = (int32x4_t *)output;
    (void)bd;
    switch (tx_type) {
    case IDTX:
        load_buffer_8x8_in_32x32_neon(input, buf, stride);
        av1_round_shift_array_32_N4_neon(buf, out, 32, -shift[0]);
        fidtx_wxh_N4_neon(out, buf, 32, 4);
        av1_round_shift_array_32_N4_neon(buf, out, 32, -shift[1]);
        fidtx_wxh_N4_neon(out, buf, 32, 4);
        av1_round_shift_array_32_N4_neon(buf, buf, 32, -shift[2]);
        write_buffer_32x32_N4(buf, output);
        break;
    case DCT_DCT:
        load_buffer_32x32_neon(input, buf, stride);
        av1_round_shift_array_32_neon(buf, out, txfm2d_size_128, -shift[0]);
        fdct32x32_N4_col_neon(out, buf, cos_bit_col);
        av1_round_shift_array_32_neon(buf, out, txfm2d_size_128 / 4, -shift[1]);
        transpose_8x8_in_32x32_neon(out, buf);
        transpose_8x8_in_32x32_neon(out + 2, buf + 64);
        transpose_8x8_in_32x32_neon(out + 4, buf + 128);
        transpose_8x8_in_32x32_neon(out + 6, buf + 192);
        fdct32x32_N4_row_neon(buf, out, cos_bit_row);
        av1_round_shift_array_32_N4_neon(out, out, 32, -shift[2]);
        transpose_8x8_in_32x32_neon(out, buf);
        write_buffer_32x32_N4(buf, output);
        break;
    case V_DCT:
        load_buffer_8x32_in_32x32_neon(input, buf, stride);
        av1_round_shift_array_32_neon(buf, out, txfm2d_size_128, -shift[0]);
        fdct32x32_N4_row_neon(out, buf, cos_bit_col);
        av1_round_shift_array_32_N4_neon(buf, out, 32, -shift[1]);
        fidtx_wxh_N4_neon(out, buf, 32, 4);
        av1_round_shift_array_32_N4_neon(buf, buf, 32, -shift[2]);
        write_buffer_32x32_N4(buf, output);
        break;
    case H_DCT:
        load_buffer_32x8_in_32x32_neon(input, buf, stride);
        av1_round_shift_array_32_neon(buf, out, txfm2d_size_128 / 4, -shift[0]);
        fidtx_wxh_N4_neon(out, buf, 32, 1);
        av1_round_shift_array_32_neon(buf, out, txfm2d_size_128 / 4, -shift[1]);
        transpose_8x8_in_32x32_neon(out, buf);
        transpose_8x8_in_32x32_neon(out + 2, buf + 64);
        transpose_8x8_in_32x32_neon(out + 4, buf + 128);
        transpose_8x8_in_32x32_neon(out + 6, buf + 192);
        fdct32x32_N4_row_neon(buf, out, cos_bit_row);
        av1_round_shift_array_32_N4_neon(out, out, 32, -shift[2]);
        transpose_8x8_in_32x32_neon(out, buf);
        write_buffer_32x32_N4(buf, output);
        break;
    default: assert(0);
    }
}

static void fdct8x8_N4_neon(const int32x4_t *in, int32x4_t *out, int8_t bit, const int32_t col_num) {
    const int32_t  *cospi    = cospi_arr(bit);
    const int32x4_t cospi32  = vdupq_n_s32(cospi[32]);
    const int32x4_t cospim32 = vdupq_n_s32(-cospi[32]);
    const int32x4_t cospi56  = vdupq_n_s32(cospi[56]);
    const int32x4_t cospi8   = vdupq_n_s32(cospi[8]);
    int32x4_t       u[16], v[16];

    // stage 0
    // stage 1
    u[0]  = vaddq_s32(in[2 * 0 * col_num], in[2 * 7 * col_num]);
    u[1]  = vaddq_s32(in[2 * 0 * col_num + 1], in[2 * 7 * col_num + 1]);
    v[14] = vsubq_s32(in[2 * 0 * col_num], in[2 * 7 * col_num]);
    v[15] = vsubq_s32(in[2 * 0 * col_num + 1], in[2 * 7 * col_num + 1]);
    u[2]  = vaddq_s32(in[2 * 1 * col_num], in[2 * 6 * col_num]);
    u[3]  = vaddq_s32(in[2 * 1 * col_num + 1], in[2 * 6 * col_num + 1]);
    u[12] = vsubq_s32(in[2 * 1 * col_num], in[2 * 6 * col_num]);
    u[13] = vsubq_s32(in[2 * 1 * col_num + 1], in[2 * 6 * col_num + 1]);
    u[4]  = vaddq_s32(in[2 * 2 * col_num], in[2 * 5 * col_num]);
    u[5]  = vaddq_s32(in[2 * 2 * col_num + 1], in[2 * 5 * col_num + 1]);
    u[10] = vsubq_s32(in[2 * 2 * col_num], in[2 * 5 * col_num]);
    u[11] = vsubq_s32(in[2 * 2 * col_num + 1], in[2 * 5 * col_num + 1]);
    u[6]  = vaddq_s32(in[2 * 3 * col_num], in[2 * 4 * col_num]);
    u[7]  = vaddq_s32(in[2 * 3 * col_num + 1], in[2 * 4 * col_num + 1]);
    v[8]  = vsubq_s32(in[2 * 3 * col_num], in[2 * 4 * col_num]);
    v[9]  = vsubq_s32(in[2 * 3 * col_num + 1], in[2 * 4 * col_num + 1]);

    // stage 2
    v[0] = vaddq_s32(u[0], u[6]);
    v[1] = vaddq_s32(u[1], u[7]);
    v[2] = vaddq_s32(u[2], u[4]);
    v[3] = vaddq_s32(u[3], u[5]);

    v[10] = vmulq_s32(u[10], cospim32);
    v[11] = vmulq_s32(u[11], cospim32);
    v[12] = vmulq_s32(u[12], cospi32);
    v[13] = vmulq_s32(u[13], cospi32);
    v[10] = vaddq_s32(v[10], v[12]);
    v[11] = vaddq_s32(v[11], v[13]);
    v[10] = vrshlq_s32(v[10], vdupq_n_s32(-bit));
    v[11] = vrshlq_s32(v[11], vdupq_n_s32(-bit));

    u[0]  = vmulq_s32(u[10], cospi32);
    u[1]  = vmulq_s32(u[11], cospi32);
    v[12] = vmulq_s32(u[12], cospim32);
    v[13] = vmulq_s32(u[13], cospim32);
    v[12] = vsubq_s32(u[0], v[12]);
    v[13] = vsubq_s32(u[1], v[13]);
    v[12] = vrshlq_s32(v[12], vdupq_n_s32(-bit));
    v[13] = vrshlq_s32(v[13], vdupq_n_s32(-bit));

    // stage 3
    // type 0
    v[0] = vmulq_s32(v[0], cospi32);
    v[1] = vmulq_s32(v[1], cospi32);
    v[2] = vmulq_s32(v[2], cospi32);
    v[3] = vmulq_s32(v[3], cospi32);
    u[0] = vaddq_s32(v[0], v[2]);
    u[1] = vaddq_s32(v[1], v[3]);
    u[0] = vrshlq_s32(u[0], vdupq_n_s32(-bit));
    u[1] = vrshlq_s32(u[1], vdupq_n_s32(-bit));

    u[8]  = vaddq_s32(v[8], v[10]);
    u[9]  = vaddq_s32(v[9], v[11]);
    u[14] = vaddq_s32(v[14], v[12]);
    u[15] = vaddq_s32(v[15], v[13]);

    // stage 4
    // stage 5
    v[0]                     = vmulq_s32(u[8], cospi56);
    v[1]                     = vmulq_s32(u[9], cospi56);
    v[2]                     = vmulq_s32(u[14], cospi8);
    v[3]                     = vmulq_s32(u[15], cospi8);
    v[0]                     = vaddq_s32(v[0], v[2]);
    v[1]                     = vaddq_s32(v[1], v[3]);
    out[2 * 1 * col_num]     = vrshlq_s32(v[0], vdupq_n_s32(-bit));
    out[2 * 1 * col_num + 1] = vrshlq_s32(v[1], vdupq_n_s32(-bit));

    out[2 * 0 * col_num]     = u[0];
    out[2 * 0 * col_num + 1] = u[1];
}

static INLINE void transpose_8nx8n_N4_quad_neon(const int32x4_t *input, int32x4_t *output, const int32_t width,
                                                const int32_t height) {
    const int32_t numcol = height >> 3;
    const int32_t numrow = width >> 3;

    int32_t calc_numcol = numcol >> 2;
    int32_t calc_numrow = numrow >> 2;
    if (!calc_numcol) {
        calc_numcol = 1;
    }
    if (!calc_numrow) {
        calc_numrow = 1;
    }

    for (int32_t j = 0; j < calc_numrow; j++) {
        for (int32_t i = 0; i < calc_numcol; i++) {
            TRANSPOSE_2X4X4_NEON(input,
                                 2 * (i * width + j + (numrow * 0)),
                                 2 * (i * width + j + (numrow * 1)),
                                 2 * (i * width + j + (numrow * 2)),
                                 2 * (i * width + j + (numrow * 3)),
                                 output,
                                 2 * (j * height + i + (numcol * 0)),
                                 2 * (j * height + i + (numcol * 4)),
                                 2 * (j * height + i + (numcol * 1)),
                                 2 * (j * height + i + (numcol * 5)),
                                 2 * (j * height + i + (numcol * 2)),
                                 2 * (j * height + i + (numcol * 6)),
                                 2 * (j * height + i + (numcol * 3)),
                                 2 * (j * height + i + (numcol * 7)));
            TRANSPOSE_2X4X4_NEON(input,
                                 2 * (i * width + j + (numrow * 4)),
                                 2 * (i * width + j + (numrow * 5)),
                                 2 * (i * width + j + (numrow * 6)),
                                 2 * (i * width + j + (numrow * 7)),
                                 output,
                                 2 * (j * height + i + (numcol * 0)) + 1,
                                 2 * (j * height + i + (numcol * 4)) + 1,
                                 2 * (j * height + i + (numcol * 1)) + 1,
                                 2 * (j * height + i + (numcol * 5)) + 1,
                                 2 * (j * height + i + (numcol * 2)) + 1,
                                 2 * (j * height + i + (numcol * 6)) + 1,
                                 2 * (j * height + i + (numcol * 3)) + 1,
                                 2 * (j * height + i + (numcol * 7)) + 1);
        }
    }
}

static INLINE void load_buffer_16_neon(const int16_t *input, int32x4_t *in, int32_t stride, int32_t shift) {
    int16x4_t temp[4];

    temp[0] = vld1_s16(input + 0 * stride);
    temp[1] = vld1_s16(input + 0 * stride + 4);
    temp[2] = vld1_s16(input + 1 * stride);
    temp[3] = vld1_s16(input + 1 * stride + 4);

    const int32x4_t vshift = vdupq_n_s32(shift);

    in[0] = vshlq_s32(vmovl_s16(temp[0]), vshift);
    in[1] = vshlq_s32(vmovl_s16(temp[1]), vshift);
    in[2] = vshlq_s32(vmovl_s16(temp[2]), vshift);
    in[3] = vshlq_s32(vmovl_s16(temp[3]), vshift);
}

static AOM_FORCE_INLINE void col_txfm_32x8_N4_rounding(int32x4_t *in, int32_t shift) {
    in[0] = vrshlq_s32(in[0], vdupq_n_s32(-shift));
    in[1] = vrshlq_s32(in[1], vdupq_n_s32(-shift));
    in[8] = vrshlq_s32(in[8], vdupq_n_s32(-shift));
    in[9] = vrshlq_s32(in[9], vdupq_n_s32(-shift));
}

static void fidtx32x8_N2_neon(const int32x4_t *input, int32x4_t *output, int8_t cos_bit, const int32_t col_num,
                              int32_t row_num) {
    (void)cos_bit;
    for (int32_t i = 0; i < row_num; i++) {
        output[2 * i * col_num]     = vshlq_n_s32(input[2 * i * col_num], 1);
        output[2 * i * col_num + 1] = vshlq_n_s32(input[2 * i * col_num + 1], 1);
    }
}

static AOM_FORCE_INLINE void load_buffer_16x8n(const int16_t *input, int32x4_t *out, int32_t stride, int32_t shift,
                                               const int32_t height) {
    for (int32_t col = 0; col < height; col++) {
        const int16_t *in     = input + col * stride;
        int32x4_t     *output = out + 2 * col * 4;
        load_buffer_16_neon(in, output, 8, shift);
    }
}

static INLINE void transpose_8nx8n_N4_half_neon(const int32x4_t *input, int32x4_t *output, const int32_t width,
                                                const int32_t height) {
    const int32_t numcol      = height >> 3;
    const int32_t numrow      = width >> 3;
    int32_t       calc_numcol = numcol >> 2;

    if (!calc_numcol) {
        calc_numcol = 1;
    }

    for (int32_t j = 0; j < numrow; j++) {
        for (int32_t i = 0; i < calc_numcol; i++) {
            TRANSPOSE_2X4X4_NEON(input,
                                 2 * (i * width + j + (numrow * 0)),
                                 2 * (i * width + j + (numrow * 1)),
                                 2 * (i * width + j + (numrow * 2)),
                                 2 * (i * width + j + (numrow * 3)),
                                 output,
                                 2 * (j * height + i + (numcol * 0)),
                                 2 * (j * height + i + (numcol * 4)),
                                 2 * (j * height + i + (numcol * 1)),
                                 2 * (j * height + i + (numcol * 5)),
                                 2 * (j * height + i + (numcol * 2)),
                                 2 * (j * height + i + (numcol * 6)),
                                 2 * (j * height + i + (numcol * 3)),
                                 2 * (j * height + i + (numcol * 7)));
            TRANSPOSE_2X4X4_NEON(input,
                                 2 * (i * width + j + (numrow * 4)),
                                 2 * (i * width + j + (numrow * 5)),
                                 2 * (i * width + j + (numrow * 6)),
                                 2 * (i * width + j + (numrow * 7)),
                                 output,
                                 2 * (j * height + i + (numcol * 0)) + 1,
                                 2 * (j * height + i + (numcol * 4)) + 1,
                                 2 * (j * height + i + (numcol * 1)) + 1,
                                 2 * (j * height + i + (numcol * 5)) + 1,
                                 2 * (j * height + i + (numcol * 2)) + 1,
                                 2 * (j * height + i + (numcol * 6)) + 1,
                                 2 * (j * height + i + (numcol * 3)) + 1,
                                 2 * (j * height + i + (numcol * 7)) + 1);
        }
    }
}

void svt_av1_fwd_txfm2d_32x8_N4_neon(int16_t *input, int32_t *output, uint32_t stride, TxType tx_type, uint8_t bd) {
    int32x4_t     in[64];
    int32x4_t    *outcoef = (int32x4_t *)output;
    const int8_t *shift   = fwd_txfm_shift_ls[TX_32X8];
    const int32_t txw_idx = get_txw_idx(TX_32X8);
    const int32_t txh_idx = get_txh_idx(TX_32X8);
    int8_t        bitcol  = fwd_cos_bit_col[txw_idx][txh_idx];
    int8_t        bitrow  = fwd_cos_bit_row[txw_idx][txh_idx];

    const int32_t txfm_size_col = tx_size_wide[TX_32X8];
    const int32_t txfm_size_row = tx_size_high[TX_32X8];
    const int32_t num_row       = txfm_size_row >> 3;
    const int32_t num_col       = txfm_size_col >> 3;

    switch (tx_type) {
    case IDTX:
        load_buffer_16x8n(input, in, stride, shift[0], txfm_size_row / 4);
        fidtx32x8_N2_neon(in, in, bitcol, num_col, 2);
        col_txfm_32x8_N4_rounding(&in[0], -shift[1]);
        fidtx_wxh_N4_neon(in, outcoef, 8, 4);
        clear_buffer_wxh_N4(outcoef, num_col, txfm_size_row);
        break;
    case DCT_DCT:
        load_buffer_32x8n(input, in, stride, 0, 0, shift[0], txfm_size_row);
        for (int32_t i = 0; i < num_col; i++) { fdct8x8_N4_neon((in + 2 * i), (in + 2 * i), bitcol, num_col); }
        col_txfm_16x16_N4_rounding(&in[0], shift[1]);
        transpose_8nx8n_N4_half_neon(in, outcoef, txfm_size_col, txfm_size_row);
        av1_fdct32_new_N4_neon(outcoef, in, bitrow, 8, num_row);
        transpose_8nx8n_N4_quad_neon(in, outcoef, txfm_size_row, txfm_size_col);
        clear_buffer_wxh_N4(outcoef, num_col, txfm_size_row);
        break;
    default: assert(0);
    }
    (void)bd;
}
