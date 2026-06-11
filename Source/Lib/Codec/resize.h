/*
 * Copyright(c) 2019 Intel Corporation
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */

#ifndef EbResize_h
#define EbResize_h

#include "definitions.h"
#include "pic_buffer_desc.h"
#include "inter_prediction.h"
#include "sequence_control_set.h"
#include "reference_object.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RS_SUBPEL_BITS 6
#define RS_SUBPEL_MASK ((1 << RS_SUBPEL_BITS) - 1)
#define RS_SCALE_SUBPEL_BITS 14
#define RS_SCALE_SUBPEL_MASK ((1 << RS_SCALE_SUBPEL_BITS) - 1)
#define RS_SCALE_EXTRA_BITS (RS_SCALE_SUBPEL_BITS - RS_SUBPEL_BITS)
#define RS_SCALE_EXTRA_OFF (1 << (RS_SCALE_EXTRA_BITS - 1))
#define UPSCALE_NORMATIVE_TAPS 8

extern const int16_t svt_av1_resize_filter_normative[(1 << RS_SUBPEL_BITS)][UPSCALE_NORMATIVE_TAPS];

void svt_av1_upscale_normative_rows(const Av1Common *cm, const uint8_t *src, int src_stride, uint8_t *dst,
                                    int dst_stride, int rows, int sub_x, int bd, bool is_16bit_pipeline);

extern const int16_t      svt_aom_av1_down2_symeven_half_filter[4];
extern const int16_t      av1_down2_symodd_half_filter[4];
extern const InterpKernel svt_aom_av1_filteredinterp_filters500[(1 << RS_SUBPEL_BITS)];
extern const InterpKernel svt_aom_av1_filteredinterp_filters625[(1 << RS_SUBPEL_BITS)];
extern const InterpKernel svt_aom_av1_filteredinterp_filters750[(1 << RS_SUBPEL_BITS)];
extern const InterpKernel svt_aom_av1_filteredinterp_filters875[(1 << RS_SUBPEL_BITS)];

#define filteredinterp_filters1000 svt_av1_resize_filter_normative

typedef struct {
    uint16_t encoding_width;
    uint16_t encoding_height;
    uint8_t  resize_denom;
} resize_params_type;

void scale_source_references(SequenceControlSet *scs, PictureParentControlSet *pcs, EbPictureBufferDesc *input_pic);

void svt_aom_scale_rec_references(PictureControlSet *pcs, EbPictureBufferDesc *input_pic);

void svt_aom_use_scaled_rec_refs_if_needed(PictureControlSet *pcs, EbPictureBufferDesc *input_pic,
                                           EbReferenceObject *ref_obj, EbPictureBufferDesc **ref_pic, uint8_t hbd_md);

void svt_aom_use_scaled_source_refs_if_needed(PictureParentControlSet *pcs, EbPictureBufferDesc *input_pic,
                                              EbPaReferenceObject *ref_obj, EbPictureBufferDesc **ref_pic_ptr,
                                              EbPictureBufferDesc **quarter_ref_pic_ptr,
                                              EbPictureBufferDesc **sixteenth_ref_pic_ptr);

void scale_pcs_params(SequenceControlSet *scs, PictureParentControlSet *pcs, resize_params_type spr_params,
                      uint16_t source_width, uint16_t source_height);

// resize picture for scaling-ref
void svt_aom_init_resize_picture(SequenceControlSet *scs, PictureParentControlSet *pcs);

void svt_aom_reset_resized_picture(SequenceControlSet *scs, PictureParentControlSet *pcs,
                                   EbPictureBufferDesc *input_pic);

uint8_t svt_aom_get_denom_idx(uint8_t scale_denom);

EbErrorType svt_aom_downscaled_source_buffer_desc_ctor(EbPictureBufferDesc **picture_ptr,
                                                       EbPictureBufferDesc  *picture_ptr_for_reference,
                                                       resize_params_type  spr_params);

EbErrorType svt_aom_resize_frame(const EbPictureBufferDesc *src, EbPictureBufferDesc *dst, int bd, const int num_planes,
                                 const uint32_t ss_x, const uint32_t ss_y, uint8_t is_packed,
                                 uint32_t buffer_enable_mask, uint8_t is_2bcompress);

#ifdef __cplusplus
}
#endif
#endif // EbResize_h
