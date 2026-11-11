/*
 * Copyright(c) 2019 Intel Corporation
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0 in LICENSE and PATENTS.
 */

// Compile the encoder implementations here to access their private SCD helpers.
#include "pic_analysis_process.c"
#include "pd_process.c"
#include "svt_scd_wrappers.h"

void svt_scd_gather_statistics(SequenceControlSet* scs, PictureParentControlSet* pcs, EbPictureBufferDesc* pic,
                               uint64_t* sum) {
    sub_sample_luma_generate_pixel_intensity_histogram_bins(scs, pcs, pic, sum);
}

void svt_scd_detect_scene(SequenceControlSet* scs, PictureParentControlSet* pcs, PictureDecisionContext* ctx) {
    perform_scene_change_detection(scs, pcs, ctx);
}

void svt_scd_copy_histograms(PictureParentControlSet* pcs, PictureDecisionContext* ctx) {
    copy_histograms(pcs, ctx);
}
