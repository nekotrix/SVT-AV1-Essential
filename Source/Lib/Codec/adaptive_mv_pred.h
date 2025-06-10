/*
* Copyright(c) 2019 Intel Corporation
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#ifndef EbAdaptiveMotionVectorPrediction_h
#define EbAdaptiveMotionVectorPrediction_h

#include "utility.h"
#include "pcs.h"
#include "coding_unit.h"
#include "neighbor_arrays.h"
#include "enc_warped_motion.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ModeDecisionContext;
extern EbErrorType clip_mv(uint32_t blk_org_x, uint32_t blk_org_y, int16_t *mv_x, int16_t *mv_y, uint32_t picture_width,
                           uint32_t picture_height, uint32_t tb_size);
void               svt_aom_init_xd(PictureControlSet *pcs, struct ModeDecisionContext *ctx);
void svt_aom_generate_av1_mvp_table(struct ModeDecisionContext *ctx, BlkStruct *blk_ptr, const BlockGeom *blk_geom,
                                    uint16_t blk_org_x, uint16_t blk_org_y, MvReferenceFrame *ref_frames,
                                    uint32_t tot_refs, PictureControlSet *pcs);

#if CLN_UNIFY_MV_TYPE
void svt_aom_get_av1_mv_pred_drl(struct ModeDecisionContext *ctx, BlkStruct *blk_ptr, MvReferenceFrame ref_frame,
                                 uint8_t is_compound, PredictionMode mode, uint8_t drl_index, Mv nearestmv[2],
                                 Mv nearmv[2], Mv ref_mv[2]);
#else
void svt_aom_get_av1_mv_pred_drl(struct ModeDecisionContext *ctx, BlkStruct *blk_ptr, MvReferenceFrame ref_frame,
                                 uint8_t is_compound, PredictionMode mode, uint8_t drl_index, IntMv nearestmv[2],
                                 IntMv nearmv[2], IntMv ref_mv[2]);
#endif
MbModeInfo *get_mbmi(PictureControlSet *pcs, uint32_t blk_org_x, uint32_t blk_org_y);
void        svt_aom_update_mi_map(BlkStruct *blk_ptr, uint32_t blk_org_x, uint32_t blk_org_y, const BlockGeom *blk_geom,
                                  PictureControlSet *pcs, struct ModeDecisionContext *ctx);
#if CLN_WM_CTRLS
#if !CLN_WM_SAMPLES
uint8_t wm_find_samples(BlkStruct *blk_ptr, MvReferenceFrame rf0, PictureControlSet *pcs, int32_t *pts,
                        int32_t *pts_inref);
void    svt_aom_wm_count_samples(BlkStruct *blk_ptr, const BlockSize sb_size, uint8_t ref_frame_type,
                                 PictureControlSet *pcs, uint8_t *num_samples);
#endif
#if CLN_MV_UNIT
bool svt_aom_warped_motion_parameters(struct ModeDecisionContext *ctx, const Mv mv, const BlockGeom *blk_geom,
                                      const MvReferenceFrame ref_frame, WarpedMotionParams *wm_params,
                                      uint8_t *num_samples, uint16_t lower_band_th, uint16_t upper_band_th,
                                      bool shut_approx);
#else
#if CLN_WM_SAMPLES
bool svt_aom_warped_motion_parameters(struct ModeDecisionContext *ctx, MvUnit *mv_unit, const BlockGeom *blk_geom,
                                      uint8_t ref_frame_type, WarpedMotionParams *wm_params, uint8_t *num_samples,
                                      uint16_t lower_band_th, uint16_t upper_band_th, bool shut_approx);
#else
bool svt_aom_warped_motion_parameters(PictureControlSet *pcs, BlkStruct *blk_ptr, MvUnit *mv_unit,
                                      const BlockGeom *blk_geom, uint8_t ref_frame_type, WarpedMotionParams *wm_params,
                                      uint8_t *num_samples, uint16_t lower_band_th, uint16_t upper_band_th,
                                      bool shut_approx);
#endif
#endif
#else
uint16_t wm_find_samples(BlkStruct *blk_ptr, const BlockGeom *blk_geom, uint16_t blk_org_x, uint16_t blk_org_y,
                         MvReferenceFrame rf0, PictureControlSet *pcs, int32_t *pts, int32_t *pts_inref,
                         int *adjacent_samples, int *top_left_present, int *top_right_present);
void     svt_aom_wm_count_samples(BlkStruct *blk_ptr, const BlockSize sb_size, const BlockGeom *blk_geom,
                                  uint16_t blk_org_x, uint16_t blk_org_y, uint8_t ref_frame_type, PictureControlSet *pcs,
                                  uint16_t *num_samples);
bool     svt_aom_warped_motion_parameters(PictureControlSet *pcs, BlkStruct *blk_ptr, MvUnit *mv_unit,
                                          const BlockGeom *blk_geom, uint16_t blk_org_x, uint16_t blk_org_y,
                                          uint8_t ref_frame_type, WarpedMotionParams *wm_params, uint16_t *num_samples,
                                          uint8_t min_neighbour_perc, uint8_t corner_perc_bias, uint16_t lower_band_th,
                                          uint16_t upper_band_th, bool shut_approx);
#endif
#if CLN_WM_SAMPLES
void svt_aom_init_wm_samples(PictureControlSet *pcs, struct ModeDecisionContext *ctx);
#endif
static INLINE bool has_overlappable_candidates(const BlkStruct *blk_ptr) {
    return (blk_ptr->overlappable_neighbors != 0);
}

void svt_av1_count_overlappable_neighbors(const PictureControlSet *pcs, BlkStruct *blk_ptr, const BlockSize bsize,
                                          int32_t mi_row, int32_t mi_col);

#if CLN_UNIFY_MV_TYPE
void svt_av1_find_best_ref_mvs_from_stack(int allow_hp, CandidateMv ref_mv_stack[][MAX_REF_MV_STACK_SIZE],
                                          MacroBlockD *xd, MvReferenceFrame ref_frame, Mv *nearest_mv, Mv *near_mv,
                                          int is_integer);
int svt_aom_is_dv_valid(const Mv dv, const MacroBlockD *xd, int mi_row, int mi_col, BlockSize bsize, int mib_size_log2);

Mv svt_aom_gm_get_motion_vector_enc(const WarpedMotionParams *gm, int32_t allow_hp, BlockSize bsize, int32_t mi_col,
                                    int32_t mi_row, int32_t is_integer);
#else
void svt_av1_find_best_ref_mvs_from_stack(int allow_hp, CandidateMv ref_mv_stack[][MAX_REF_MV_STACK_SIZE],
                                          MacroBlockD *xd, MvReferenceFrame ref_frame, IntMv *nearest_mv,
                                          IntMv *near_mv, int is_integer);
int svt_aom_is_dv_valid(const MV dv, const MacroBlockD *xd, int mi_row, int mi_col, BlockSize bsize, int mib_size_log2);

IntMv svt_aom_gm_get_motion_vector_enc(const WarpedMotionParams *gm, int32_t allow_hp, BlockSize bsize, int32_t mi_col,
                                       int32_t mi_row, int32_t is_integer);
#endif
#ifdef __cplusplus
}
#endif
#endif // EbAdaptiveMotionVectorPrediction_h
