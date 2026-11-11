/*
* Copyright(c) 2022 Intel Corporation
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#include "EbConfigMacros.h"

#if CONFIG_ENABLE_FILM_GRAIN

#include <immintrin.h>
#include "definitions.h"
#include "common_dsp_rtcd.h"
#include "aom_dsp_rtcd.h"
#include "noise_util.h"
#include "noise_model.h"

/*
 * Tool --fast addition.
 * The wrapper specializes only n, allowing the compiler to remove loop-control
 * work for the 24/25-coefficient systems used by the grain model. Other sizes
 * use the original kernel. No symmetry shortcut or reduction reassociation.
 */
static INLINE void add_block_observations_internal_avx2(uint32_t n, const double val, const double recp_sqr_norm,
                                                  double* buffer, double* buffer_norm, double* b, double* A) {
    uint32_t      i;
    const __m256d recp_spr_pd = _mm256_set1_pd(recp_sqr_norm);
    const __m256d val_pd      = _mm256_set1_pd(val);
    __m256d       buffer_pd, tmp_pd;

    for (i = 0; i + 8 - 1 < n; i += 8) {
        buffer_pd = _mm256_loadu_pd(buffer + i);
        buffer_pd = _mm256_mul_pd(buffer_pd, recp_spr_pd);
        _mm256_storeu_pd(buffer_norm + i, buffer_pd);
        buffer_pd = _mm256_mul_pd(buffer_pd, val_pd);
        tmp_pd    = _mm256_loadu_pd(b + i);
        tmp_pd    = _mm256_add_pd(tmp_pd, buffer_pd);
        _mm256_storeu_pd(b + i, tmp_pd);

        buffer_pd = _mm256_loadu_pd(buffer + i + 4);
        buffer_pd = _mm256_mul_pd(buffer_pd, recp_spr_pd);
        _mm256_storeu_pd(buffer_norm + i + 4, buffer_pd);
        buffer_pd = _mm256_mul_pd(buffer_pd, val_pd);
        tmp_pd    = _mm256_loadu_pd(b + i + 4);
        tmp_pd    = _mm256_add_pd(tmp_pd, buffer_pd);
        _mm256_storeu_pd(b + i + 4, tmp_pd);
    }
    for (; i < n; ++i) {
        buffer_norm[i] = buffer[i] * recp_sqr_norm;
        b[i] += buffer_norm[i] * val;
    }

    for (i = 0; i < n; ++i) {
        uint32_t      j              = 0;
        const double  buffer_norm_i  = buffer_norm[i];
        const __m256d buffer_norm_pd = _mm256_set1_pd(buffer_norm_i);

        for (j = 0; j + 8 - 1 < n; j += 8) {
            buffer_pd = _mm256_loadu_pd(buffer + j);
            tmp_pd    = _mm256_loadu_pd(A + i * n + j);
            buffer_pd = _mm256_mul_pd(buffer_pd, buffer_norm_pd);
            tmp_pd    = _mm256_add_pd(tmp_pd, buffer_pd);
            _mm256_storeu_pd(A + i * n + j, tmp_pd);

            buffer_pd = _mm256_loadu_pd(buffer + j + 4);
            tmp_pd    = _mm256_loadu_pd(A + i * n + j + 4);
            buffer_pd = _mm256_mul_pd(buffer_pd, buffer_norm_pd);
            tmp_pd    = _mm256_add_pd(tmp_pd, buffer_pd);
            _mm256_storeu_pd(A + i * n + j + 4, tmp_pd);
        }
        for (; j < n; ++j) {
            A[i * n + j] += (buffer_norm_i * buffer[j]);
        }
    }
}


void svt_fg_add_block_observations_internal_avx2(uint32_t n, const double val, const double recp_sqr_norm,
                                                double* buffer, double* buffer_norm, double* b, double* A) {
    if (n == 24) {
        add_block_observations_internal_avx2(24, val, recp_sqr_norm, buffer, buffer_norm, b, A);
    } else if (n == 25) {
        add_block_observations_internal_avx2(25, val, recp_sqr_norm, buffer, buffer_norm, b, A);
    } else {
        svt_av1_add_block_observations_internal_avx2(n, val, recp_sqr_norm, buffer, buffer_norm, b, A);
    }
}
#endif
