/*
 * Copyright (c) 2017, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */

/*
 * Compile the upstream implementation in this translation unit so the external
 * denoised-input adapter can use its private modeling helpers. Do not compile
 * noise_model.c separately for the tool.
 */
#include "noise_model.c"
#include "svt_fg_noise_model.h"
#include <float.h>

/*
 * Externally-denoised variant of svt_aom_denoise_and_model_run() above. The input
 * preparation replaces the Wiener filter; flat-block selection, noise modeling,
 * parameter conversion and cleanup follow the upstream function in the same order.
 */
int32_t svt_aom_denoise_and_model_run_with_external_denoised(struct AomDenoiseAndModel* ctx, EbPictureBufferDesc* sd,
                                                             EbPictureBufferDesc* denoised_sd,
                                                             AomFilmGrain*        film_grain,
                                                             int32_t              use_highbd) {
    const int32_t block_size = ctx->block_size;
    uint8_t*      raw_data[3];
    uint8_t*      denoised_data[3];
    int32_t       chroma_sub_log2[2] = {1, 1}; //todo: send chroma subsampling
    int32_t       strides[3]         = {sd->y_stride, sd->u_stride, sd->v_stride};
    int32_t       ok                 = 1;

    if (sd->width != denoised_sd->width || sd->height != denoised_sd->height ||
        sd->y_stride != denoised_sd->y_stride || sd->u_stride != denoised_sd->u_stride ||
        sd->v_stride != denoised_sd->v_stride) {
        SVT_ERROR("Source and externally denoised buffers must have matching dimensions and strides\n");
        return 0;
    }

    if (!denoise_and_model_realloc_if_necessary(ctx, sd, use_highbd)) {
        SVT_ERROR("Unable to realloc buffers\n");
        return 0;
    }

    if (!use_highbd) { // 8 bits input
        raw_data[0]      = sd->y_buffer;
        raw_data[1]      = sd->u_buffer;
        raw_data[2]      = sd->v_buffer;
        denoised_data[0] = denoised_sd->y_buffer;
        denoised_data[1] = denoised_sd->u_buffer;
        denoised_data[2] = denoised_sd->v_buffer;
    } else { // 10 bits input
        uint16_t* denoised_packed[3] = {
            (uint16_t*)ctx->denoised[0],
            (uint16_t*)ctx->denoised[1],
            (uint16_t*)ctx->denoised[2],
        };
        svt_aom_pack_2d_pic(sd, ctx->packed);
        svt_aom_pack_2d_pic(denoised_sd, denoised_packed);

        raw_data[0]      = (uint8_t*)(ctx->packed[0]);
        raw_data[1]      = (uint8_t*)(ctx->packed[1]);
        raw_data[2]      = (uint8_t*)(ctx->packed[2]);
        denoised_data[0] = (uint8_t*)denoised_packed[0];
        denoised_data[1] = (uint8_t*)denoised_packed[1];
        denoised_data[2] = (uint8_t*)denoised_packed[2];
    }

    const uint8_t* const data[3]     = {raw_data[0], raw_data[1], raw_data[2]};
    const uint8_t* const denoised[3] = {denoised_data[0], denoised_data[1], denoised_data[2]};

    svt_aom_flat_block_finder_run(
        &ctx->flat_block_finder, data[0], sd->width, sd->height, strides[0], ctx->flat_blocks);

    const AomNoiseStatus status = noise_model_update(&ctx->noise_model,
                                                     data,
                                                     denoised,
                                                     sd->width,
                                                     sd->height,
                                                     strides,
                                                     chroma_sub_log2,
                                                     ctx->flat_blocks,
                                                     block_size);

    int32_t have_noise_estimate = 0;
    if (status == AOM_NOISE_STATUS_OK || status == AOM_NOISE_STATUS_DIFFERENT_NOISE_TYPE) {
        svt_aom_noise_model_save_latest(&ctx->noise_model);
        have_noise_estimate = 1;
    }

    film_grain->apply_grain = 0;
    if (have_noise_estimate) {
        if (!svt_aom_noise_model_get_grain_parameters(&ctx->noise_model, film_grain)) {
            SVT_ERROR("Unable to get grain parameters.\n");
            ok = 0;
            goto cleanup;
        }
        film_grain->apply_grain = 1;
    }

cleanup:
    svt_aom_flat_block_finder_free(&ctx->flat_block_finder);
    svt_aom_noise_model_free(&ctx->noise_model);
    free(ctx->flat_blocks);
    ctx->flat_blocks = NULL;

    return ok;
}

/*
 * Tool --fast additions. The upstream implementation above is unchanged.
 *
 * These variants retain noise_model.c's loops, arithmetic and error-diffusion
 * order. Quantization divides an integer in [0, (1 << bit_depth) - 1] by the
 * same normalization for every pixel. Compute each of those float divisions
 * once, then use its bit-identical value in the original recurrence. This is
 * not reciprocal multiplication or reduced-precision/8-bit modeling.
 */
#define DITHER_AND_QUANTIZE_FAST(INT_TYPE, suffix)                                                                      \
    static void dither_and_quantize_fast_##suffix(float*    result,                                                     \
                                             int32_t   result_stride,                                                   \
                                             INT_TYPE* denoised,                                                        \
                                             int32_t   w,                                                               \
                                             int32_t   h,                                                               \
                                             int32_t   stride,                                                          \
                                             int32_t   chroma_sub_w,                                                    \
                                             int32_t   chroma_sub_h,                                                    \
                                             int32_t   block_size,                                                      \
                                             float     block_normalization,                                             \
                                             const float* quantized) {                                                  \
        for (int32_t y = 0; y < (h >> chroma_sub_h); ++y) {                                                             \
            for (int32_t x = 0; x < (w >> chroma_sub_w); ++x) {                                                         \
                const int32_t result_idx = (y + (block_size >> chroma_sub_h)) * result_stride + x +                     \
                    (block_size >> chroma_sub_w);                                                                       \
                INT_TYPE    new_val      = (INT_TYPE)AOMMIN(AOMMAX(result[result_idx] * block_normalization + 0.5f, 0), \
                                                    block_normalization);                                       \
                const float err          = -(quantized[new_val] - result[result_idx]);                                  \
                denoised[y * stride + x] = new_val;                                                                     \
                if (x + 1 < (w >> chroma_sub_w)) {                                                                      \
                    result[result_idx + 1] += err * 7.0f / 16.0f;                                                       \
                }                                                                                                       \
                if (y + 1 < (h >> chroma_sub_h)) {                                                                      \
                    if (x > 0) {                                                                                        \
                        result[result_idx + result_stride - 1] += err * 3.0f / 16.0f;                                   \
                    }                                                                                                   \
                    result[result_idx + result_stride] += err * 5.0f / 16.0f;                                           \
                    if (x + 1 < (w >> chroma_sub_w)) {                                                                  \
                        result[result_idx + result_stride + 1] += err * 1.0f / 16.0f;                                   \
                    }                                                                                                   \
                }                                                                                                       \
            }                                                                                                           \
        }                                                                                                               \
    }

DITHER_AND_QUANTIZE_FAST(uint8_t, lowbd);
DITHER_AND_QUANTIZE_FAST(uint16_t, highbd);

int32_t svt_fg_wiener_denoise_2d_fast(const uint8_t* const data[3], uint8_t* denoised[3], int32_t w, int32_t h,
                                  int32_t stride[3], int32_t chroma_sub[2], float noise_psd[3], int32_t block_size,
                                  int32_t bit_depth, int32_t use_highbd) {
    const float *window_full = NULL, *window_chroma = NULL;
    float*       plane = NULL;
    DECLARE_ALIGNED(32, float, *block);
    block                          = NULL;
    double *               block_d = NULL, *plane_d = NULL;
    struct aom_noise_tx_t* tx_full       = NULL;
    struct aom_noise_tx_t* tx_chroma     = NULL;
    const int32_t          num_blocks_w  = (w + block_size - 1) / block_size;
    const int32_t          num_blocks_h  = (h + block_size - 1) / block_size;
    const int32_t          result_stride = (num_blocks_w + 2) * block_size;
    const int32_t          result_height = (num_blocks_h + 2) * block_size;
    float*                 result        = NULL;
    int32_t                init_success  = 1;
    AomFlatBlockFinder     block_finder_full;
    AomFlatBlockFinder     block_finder_chroma;
    const float            k_block_normalization = (float)((1 << bit_depth) - 1);
    // Tool --fast: exact float divisions reused by the original quantizer.
    assert(bit_depth == 8 || bit_depth == 10);
    float quantized[1024];
    for (int32_t i = 0; i <= (int32_t)k_block_normalization; ++i) {
        quantized[i] = (float)i / k_block_normalization;
    }
    if (chroma_sub[0] != chroma_sub[1]) {
        SVT_ERROR(
            "svt_aom_wiener_denoise_2d doesn't handle different chroma "
            "subsampling");
        return 0;
    }
    init_success &= svt_aom_flat_block_finder_init(&block_finder_full, block_size, bit_depth, use_highbd);
    result      = (float*)malloc((num_blocks_h + 2) * block_size * result_stride * sizeof(*result));
    plane       = (float*)malloc(block_size * block_size * sizeof(*plane));
    block       = (float*)svt_aom_memalign(32, 2 * block_size * block_size * sizeof(*block));
    block_d     = (double*)malloc(block_size * block_size * sizeof(*block_d));
    plane_d     = (double*)malloc(block_size * block_size * sizeof(*plane_d));
    window_full = get_half_cos_window(block_size);
    tx_full     = svt_aom_noise_tx_malloc(block_size);

    if (chroma_sub[0] != 0) {
        init_success &= svt_aom_flat_block_finder_init(
            &block_finder_chroma, block_size >> chroma_sub[0], bit_depth, use_highbd);
        window_chroma = get_half_cos_window(block_size >> chroma_sub[0]);
        tx_chroma     = svt_aom_noise_tx_malloc(block_size >> chroma_sub[0]);
    } else {
        window_chroma = window_full;
        tx_chroma     = tx_full;
    }

    init_success &= (int32_t)((tx_full != NULL) && (tx_chroma != NULL) && (plane != NULL) && (plane_d != NULL) &&
                              (block != NULL) && (block_d != NULL) && (window_full != NULL) &&
                              (window_chroma != NULL) && (result != NULL));
    for (int32_t c = init_success ? 0 : 3; c < 3; ++c) {
        const float*           window_function = c == 0 ? window_full : window_chroma;
        AomFlatBlockFinder*    block_finder    = &block_finder_full;
        const int32_t          chroma_sub_h    = c > 0 ? chroma_sub[1] : 0;
        const int32_t          chroma_sub_w    = c > 0 ? chroma_sub[0] : 0;
        struct aom_noise_tx_t* tx              = (c > 0 && chroma_sub[0] > 0) ? tx_chroma : tx_full;
        if (!data[c] || !denoised[c]) {
            continue;
        }
        if (c > 0 && chroma_sub[0] != 0) {
            block_finder = &block_finder_chroma;
        }
        memset(result, 0, sizeof(*result) * result_stride * result_height);
        // Do overlapped block processing (half overlapped). The block rows can
        // easily be done in parallel
        for (int32_t offsy = 0; offsy < (block_size >> chroma_sub_h); offsy += (block_size >> chroma_sub_h) / 2) {
            for (int32_t offsx = 0; offsx < (block_size >> chroma_sub_w); offsx += (block_size >> chroma_sub_w) / 2) {
                // Pad the boundary when processing each block-set.
                for (int32_t by = -1; by < num_blocks_h; ++by) {
                    for (int32_t bx = -1; bx < num_blocks_w; ++bx) {
                        const int32_t pixels_per_block = (block_size >> chroma_sub_w) * (block_size >> chroma_sub_h);
                        svt_aom_flat_block_finder_extract_block(block_finder,
                                                                data[c],
                                                                w >> chroma_sub_w,
                                                                h >> chroma_sub_h,
                                                                stride[c],
                                                                bx * (block_size >> chroma_sub_w) + offsx,
                                                                by * (block_size >> chroma_sub_h) + offsy,
                                                                plane_d,
                                                                block_d);
                        svt_av1_pointwise_multiply(window_function, plane, block, plane_d, block_d, pixels_per_block);
                        svt_aom_noise_tx_forward(tx, block);
                        svt_aom_noise_tx_filter(tx->block_size, tx->tx_block, noise_psd[c]);
                        svt_aom_noise_tx_inverse(tx, block);

                        // Apply window function to the plane approximation (we will apply
                        // it to the sum of plane + block when composing the results).

                        const int y_size  = (block_size >> chroma_sub_h);
                        const int x_size  = (block_size >> chroma_sub_w);
                        float* result_ptr = result + ((by + 1) * y_size + offsy) * result_stride + (bx + 1) * x_size +
                            offsx;
                        svt_av1_apply_window_function_to_plane(
                            y_size, x_size, result_ptr, result_stride, block, plane, window_function);
                    }
                }
            }
        }
        if (use_highbd) {
            dither_and_quantize_fast_highbd(result,
                                       result_stride,
                                       (uint16_t*)denoised[c],
                                       w,
                                       h,
                                       stride[c],
                                       chroma_sub_w,
                                       chroma_sub_h,
                                       block_size,
                                       k_block_normalization, quantized);
        } else {
            dither_and_quantize_fast_lowbd(result,
                                      result_stride,
                                      denoised[c],
                                      w,
                                      h,
                                      stride[c],
                                      chroma_sub_w,
                                      chroma_sub_h,
                                      block_size,
                                      k_block_normalization, quantized);
        }
    }
    free(result);
    free(plane);
    svt_aom_free(block);
    free(plane_d);
    free(block_d);

    svt_aom_noise_tx_free(tx_full);

    svt_aom_flat_block_finder_free(&block_finder_full);
    if (chroma_sub[0] != 0) {
        svt_aom_flat_block_finder_free(&block_finder_chroma);
        svt_aom_noise_tx_free(tx_chroma);
    }
    return init_success;
}

int32_t svt_aom_denoise_and_model_run_fast(struct AomDenoiseAndModel* ctx, EbPictureBufferDesc* sd, AomFilmGrain* film_grain,
                                      int32_t use_highbd) {
#if FLT_EVAL_METHOD != 0
    // The lookup assumes float evaluation without excess precision.
    return svt_aom_denoise_and_model_run(ctx, sd, film_grain, use_highbd);
#endif
    const int32_t block_size = ctx->block_size;
    uint8_t*      raw_data[3];
    int32_t       chroma_sub_log2[2] = {1, 1}; //future todo: send chroma subsampling
    int32_t       strides[3]         = {sd->y_stride, sd->u_stride, sd->v_stride};

    if (!denoise_and_model_realloc_if_necessary(ctx, sd, use_highbd)) {
        SVT_ERROR("Unable to realloc buffers\n");
        return 0;
    }

    if (!use_highbd) { // 8 bits input
        raw_data[0] = sd->y_buffer;
        raw_data[1] = sd->u_buffer;
        raw_data[2] = sd->v_buffer;
    } else { // 10 bits input
        svt_aom_pack_2d_pic(sd, ctx->packed);

        raw_data[0] = (uint8_t*)(ctx->packed[0]);
        raw_data[1] = (uint8_t*)(ctx->packed[1]);
        raw_data[2] = (uint8_t*)(ctx->packed[2]);
    }

    const uint8_t* const data[3] = {raw_data[0], raw_data[1], raw_data[2]};

    svt_aom_flat_block_finder_run(
        &ctx->flat_block_finder, data[0], sd->width, sd->height, strides[0], ctx->flat_blocks);

    if (!svt_fg_wiener_denoise_2d_fast(data,
                                   ctx->denoised,
                                   sd->width,
                                   sd->height,
                                   strides,
                                   chroma_sub_log2,
                                   ctx->noise_psd,
                                   block_size,
                                   ctx->bit_depth,
                                   use_highbd)) {
        SVT_ERROR("Unable to denoise image\n");
        return 0;
    }

    const AomNoiseStatus status = noise_model_update(&ctx->noise_model,
                                                     data,
                                                     (const uint8_t* const*)ctx->denoised,
                                                     sd->width,
                                                     sd->height,
                                                     strides,
                                                     chroma_sub_log2,
                                                     ctx->flat_blocks,
                                                     block_size);

    int32_t have_noise_estimate = 0;
    if (status == AOM_NOISE_STATUS_OK || status == AOM_NOISE_STATUS_DIFFERENT_NOISE_TYPE) {
        svt_aom_noise_model_save_latest(&ctx->noise_model);
        have_noise_estimate = 1;
    }

    film_grain->apply_grain = 0;
    if (have_noise_estimate) {
        if (!svt_aom_noise_model_get_grain_parameters(&ctx->noise_model, film_grain)) {
            SVT_ERROR("Unable to get grain parameters.\n");
            return 0;
        }
        film_grain->apply_grain = 1;

        if (ctx->denoise_apply) {
            if (!use_highbd) {
                if (svt_memcpy != NULL) {
                    svt_memcpy(raw_data[0], ctx->denoised[0], (strides[0] * sd->height) << use_highbd);
                    svt_memcpy(
                        raw_data[1], ctx->denoised[1], (strides[1] * (sd->height >> chroma_sub_log2[0])) << use_highbd);
                    svt_memcpy(
                        raw_data[2], ctx->denoised[2], (strides[2] * (sd->height >> chroma_sub_log2[0])) << use_highbd);
                } else {
                    svt_memcpy_c(raw_data[0], ctx->denoised[0], (strides[0] * sd->height) << use_highbd);
                    svt_memcpy_c(
                        raw_data[1], ctx->denoised[1], (strides[1] * (sd->height >> chroma_sub_log2[0])) << use_highbd);
                    svt_memcpy_c(
                        raw_data[2], ctx->denoised[2], (strides[2] * (sd->height >> chroma_sub_log2[0])) << use_highbd);
                }
            } else {
                unpack_2d_pic(ctx->denoised, sd);
            }
        }
    }
    svt_aom_flat_block_finder_free(&ctx->flat_block_finder);
    svt_aom_noise_model_free(&ctx->noise_model);
    free(ctx->flat_blocks);
    ctx->flat_blocks = NULL;

    return 1;
}
