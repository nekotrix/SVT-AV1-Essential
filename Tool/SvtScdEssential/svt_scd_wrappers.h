/*
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0 in LICENSE and PATENTS.
 */
#ifndef SVT_SCD_WRAPPERS_H
#define SVT_SCD_WRAPPERS_H

#include "pcs.h"
#include "pd_process.h"
#include "sequence_control_set.h"

void        svt_scd_setup_rtcd(void);
void        svt_scd_gather_statistics(SequenceControlSet* scs, PictureParentControlSet* pcs, EbPictureBufferDesc* pic,
                                      uint64_t* sum);
void        svt_scd_detect_scene(SequenceControlSet* scs, PictureParentControlSet* pcs, PictureDecisionContext* ctx);
void        svt_scd_copy_histograms(PictureParentControlSet* pcs, PictureDecisionContext* ctx);
int32_t     svt_scd_default_intra_period(SequenceControlSet* scs);
int32_t     svt_scd_default_min_intra_period(SequenceControlSet* scs);
EbErrorType svt_scd_str_to_keyint(const char* value, int32_t* out, bool* multiply);

#endif
