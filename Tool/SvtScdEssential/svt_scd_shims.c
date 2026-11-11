/*
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0 in LICENSE and PATENTS.
 */

#define RTCD_C
#define AOM_RTCD_C
#include "aom_dsp_rtcd.h"
#include "common_dsp_rtcd.h"
#include "me_sad_calculation.h"
#include "motion_estimation.h"

#if defined(SVT_SCD_USE_AVX2)
#define AOM_ARCH_X86_64 ARCH_X86_64
#include "third_party/aom/aom_ports/x86.h"
#endif

void svt_scd_setup_rtcd(void) {
    svt_memcpy                   = svt_memcpy_c;
    svt_memset                   = svt_memset_c;
    svt_initialize_buffer_32bits = svt_initialize_buffer_32bits_c;
    downsample_2d                = svt_aom_downsample_2d_c;
#if defined(SVT_SCD_USE_AVX2)
    const int flags = x86_simd_caps();
    if (flags & HAS_AVX2) {
        downsample_2d = svt_aom_downsample_2d_avx2;
    } else if (flags & HAS_SSE4_1) {
        downsample_2d = svt_aom_downsample_2d_sse4_1;
    }
#endif
}
