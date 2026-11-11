/*
 * Compile-time RTCD selection for the tool.
 * Default kernels are built from the upstream source files. The --fast
 * covariance specialization is selected explicitly below.
 */

#include <float.h>

#define RTCD_C
#define AOM_RTCD_C
#include "aom_dsp_rtcd.h"
#include "common_dsp_rtcd.h"
#include "pack_unpack_c.h"

/* svt_compressed_packmsb_c() handles complete groups of four samples only.
 * Keep support for even frame widths whose chroma has a partial group.
 */
static void svt_fg_compressed_packmsb(uint8_t* in8_bit_buffer, uint32_t in8_stride, uint8_t* inn_bit_buffer,
                                    uint32_t inn_stride, uint16_t* out16_bit_buffer, uint32_t out_stride,
                                    uint32_t width, uint32_t height) {
    svt_compressed_packmsb_c(
        in8_bit_buffer, in8_stride, inn_bit_buffer, inn_stride, out16_bit_buffer, out_stride, width, height);
    for (uint32_t row = 0; row < height; ++row) {
        for (uint32_t col = width & ~3u; col < width; ++col) {
            const uint8_t n_bit_pixel = (inn_bit_buffer[col / 4 + row * inn_stride] >> (6 - 2 * (col & 3))) & 3;
            out16_bit_buffer[col + row * out_stride] = (in8_bit_buffer[col + row * in8_stride] << 2) | n_bit_pixel;
        }
    }
}

#if defined(SVT_FG_USE_AVX2)
void svt_fg_add_block_observations_internal_avx2(uint32_t n, double val, double recp_sqr_norm,
                                                double* buffer, double* buffer_norm, double* b, double* A);
#endif

void svt_fg_tool_init_svt_shims(int fast) {
#if FLT_EVAL_METHOD != 0
    fast = 0;
#endif
    (void)fast;
    svt_memcpy = svt_memcpy_c;
    svt_memset = svt_memset_c;

#if defined(SVT_FG_USE_AVX2)
    svt_aom_fft2x2_float   = svt_aom_fft2x2_float_c;
    svt_aom_fft4x4_float   = svt_aom_fft4x4_float_sse2;
    svt_aom_fft8x8_float   = svt_aom_fft8x8_float_avx2;
    svt_aom_fft16x16_float = svt_aom_fft16x16_float_avx2;
    svt_aom_fft32x32_float = svt_aom_fft32x32_float_avx2;

    svt_aom_ifft2x2_float   = svt_aom_ifft2x2_float_c;
    svt_aom_ifft4x4_float   = svt_aom_ifft4x4_float_sse2;
    svt_aom_ifft8x8_float   = svt_aom_ifft8x8_float_avx2;
    svt_aom_ifft16x16_float = svt_aom_ifft16x16_float_avx2;
    svt_aom_ifft32x32_float = svt_aom_ifft32x32_float_avx2;

    // Tool --fast specialization; normal mode calls the unmodified upstream kernel.
    svt_av1_add_block_observations_internal = fast
        ? svt_fg_add_block_observations_internal_avx2 : svt_av1_add_block_observations_internal_avx2;
    svt_av1_pointwise_multiply              = svt_av1_pointwise_multiply_avx2;
    svt_av1_apply_window_function_to_plane  = svt_av1_apply_window_function_to_plane_avx2;
    svt_aom_noise_tx_filter                 = svt_aom_noise_tx_filter_avx2;
    svt_aom_flat_block_finder_extract_block = svt_aom_flat_block_finder_extract_block_avx2;
#else
    svt_aom_fft2x2_float   = svt_aom_fft2x2_float_c;
    svt_aom_fft4x4_float   = svt_aom_fft4x4_float_c;
    svt_aom_fft8x8_float   = svt_aom_fft8x8_float_c;
    svt_aom_fft16x16_float = svt_aom_fft16x16_float_c;
    svt_aom_fft32x32_float = svt_aom_fft32x32_float_c;

    svt_aom_ifft2x2_float   = svt_aom_ifft2x2_float_c;
    svt_aom_ifft4x4_float   = svt_aom_ifft4x4_float_c;
    svt_aom_ifft8x8_float   = svt_aom_ifft8x8_float_c;
    svt_aom_ifft16x16_float = svt_aom_ifft16x16_float_c;
    svt_aom_ifft32x32_float = svt_aom_ifft32x32_float_c;

    svt_av1_add_block_observations_internal = svt_av1_add_block_observations_internal_c;
    svt_av1_pointwise_multiply              = svt_av1_pointwise_multiply_c;
    svt_av1_apply_window_function_to_plane  = svt_av1_apply_window_function_to_plane_c;
    svt_aom_noise_tx_filter                 = svt_aom_noise_tx_filter_c;
    svt_aom_flat_block_finder_extract_block = svt_aom_flat_block_finder_extract_block_c;
#endif

    svt_compressed_packmsb    = svt_fg_compressed_packmsb;
    svt_unpack_and_2bcompress = svt_unpack_and_2bcompress_c;
}
