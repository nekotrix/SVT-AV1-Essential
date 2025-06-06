/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */
#ifndef EbEncCdef_h
#define EbEncCdef_h

#include "definitions.h"
#include "cdef.h"
#if CLN_FUNCS_HEADER
#include "pcs.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif
typedef void (*CdefFilterBlockFunc)(uint8_t* dst8, uint16_t* dst16, int32_t dstride, const uint16_t* in,
                                    int32_t pri_strength, int32_t sec_strength, int32_t dir, int32_t pri_damping,
                                    int32_t sec_damping, int32_t bsize, int32_t coeff_shift,
                                    uint8_t subsampling_factor);

#if CLN_FUNCS_HEADER
int32_t svt_sb_compute_cdef_list(PictureControlSet* pcs, const Av1Common* const cm, int32_t mi_row, int32_t mi_col,
                                 CdefList* dlist, BlockSize bs);
void    finish_cdef_search(PictureControlSet* pcs);
void    svt_av1_cdef_frame(struct SequenceControlSet* scs, PictureControlSet* pcs);
#else
void copy_cdef_16bit_to_16bit(uint16_t* dst, int32_t dstride, uint16_t* src, CdefList* dlist, int32_t cdef_count,
                              int32_t bsize);
#endif
#ifdef __cplusplus
}
#endif
#endif // AV1_COMMON_CDEF_H_
