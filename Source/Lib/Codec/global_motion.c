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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "global_motion.h"
#include "utility.h"
#include "corner_detect.h"
#include "corner_match.h"
#include "ransac.h"

#include "enc_warped_motion.h"

#if !CLN_RANSAC
#define MIN_INLIER_PROB 0.1
#endif

#if !CLN_WMMAT
#define MIN_TRANS_THRESH (1 * GM_TRANS_DECODE_FACTOR)
#endif

// Border over which to compute the global motion
#define ERRORADV_BORDER 0

#if OPT_GM_RFN_EARLY_EXIT
static const double erroradv_tr[]      = {0.65, 0.50, 0.45};
static const double erroradv_prod_tr[] = {20000, 15000, 14000};
#else
static const double erroradv_tr[]      = {0.65, 0.60, 0.65};
static const double erroradv_prod_tr[] = {20000, 18000, 16000};
#endif

int svt_av1_is_enough_erroradvantage(double best_erroradvantage, int params_cost, int erroradv_type) {
    assert(erroradv_type < GM_ERRORADV_TR_TYPES);
    return best_erroradvantage < erroradv_tr[erroradv_type] &&
        best_erroradvantage * params_cost < erroradv_prod_tr[erroradv_type];
}

static void convert_to_params(const double* params, int32_t* model) {
    int i;
#if !CLN_WMMAT
    int alpha_present = 0;
#endif
    model[0]          = (int32_t)floor(params[0] * (1 << GM_TRANS_PREC_BITS) + 0.5);
    model[1]          = (int32_t)floor(params[1] * (1 << GM_TRANS_PREC_BITS) + 0.5);
    model[0]          = (int32_t)clamp(model[0], GM_TRANS_MIN, GM_TRANS_MAX) * GM_TRANS_DECODE_FACTOR;
    model[1]          = (int32_t)clamp(model[1], GM_TRANS_MIN, GM_TRANS_MAX) * GM_TRANS_DECODE_FACTOR;

    for (i = 2; i < 6; ++i) {
        const int diag_value = ((i == 2 || i == 5) ? (1 << GM_ALPHA_PREC_BITS) : 0);
        model[i]             = (int32_t)floor(params[i] * (1 << GM_ALPHA_PREC_BITS) + 0.5);
        model[i]             = (int32_t)clamp(model[i] - diag_value, GM_ALPHA_MIN, GM_ALPHA_MAX);
#if !CLN_WMMAT
        alpha_present |= (model[i] != 0);
#endif
        model[i] = (model[i] + diag_value) * GM_ALPHA_DECODE_FACTOR;
    }
#if !CLN_WMMAT
    for (; i < 8; ++i) {
        model[i] = (int32_t)floor(params[i] * (1 << GM_ROW3HOMO_PREC_BITS) + 0.5);
        model[i] = (int32_t)clamp(model[i], GM_ROW3HOMO_MIN, GM_ROW3HOMO_MAX) * GM_ROW3HOMO_DECODE_FACTOR;
        alpha_present |= (model[i] != 0);
    }

    if (!alpha_present) {
        if (abs(model[0]) < MIN_TRANS_THRESH && abs(model[1]) < MIN_TRANS_THRESH) {
            model[0] = 0;
            model[1] = 0;
        }
    }
#endif
}

static INLINE TransformationType get_wmtype(const EbWarpedMotionParams* gm) {
    if (gm->wmmat[5] == (1 << WARPEDMODEL_PREC_BITS) && !gm->wmmat[4] && gm->wmmat[2] == (1 << WARPEDMODEL_PREC_BITS) &&
        !gm->wmmat[3]) {
        return ((!gm->wmmat[1] && !gm->wmmat[0]) ? IDENTITY : TRANSLATION);
    }
    if (gm->wmmat[2] == gm->wmmat[5] && gm->wmmat[3] == -gm->wmmat[4])
        return ROTZOOM;
    else
        return AFFINE;
}

void svt_av1_convert_model_to_params(const double *params, EbWarpedMotionParams *model) {
    convert_to_params(params, model->wmmat);
    model->wmtype  = get_wmtype(model);
    model->invalid = 0;
}

#if CLN_GM
// Adds some offset to a global motion parameter and handles
// all of the necessary precision shifts, clamping, and
// zero-centering.
static int32_t add_param_offset(int param_index, int32_t param_value, int32_t offset) {
    const int scale_vals[2] = {GM_TRANS_PREC_DIFF, GM_ALPHA_PREC_DIFF};
    const int clamp_vals[2] = {GM_TRANS_MAX, GM_ALPHA_MAX};
    // type of param: 0 - translation, 1 - affine
    const int param_type      = (param_index < 2 ? 0 : 1);
    const int is_one_centered = (param_index == 2 || param_index == 5);

    // Make parameter zero-centered and offset the shift that was done to make
    // it compatible with the warped model
    param_value = (param_value - (is_one_centered << WARPEDMODEL_PREC_BITS)) >> scale_vals[param_type];
    // Add desired offset to the rescaled/zero-centered parameter
    param_value += offset;
    // Clamp the parameter so it does not overflow the number of bits allotted
    // to it in the bitstream
    param_value = (int32_t)clamp(param_value, -clamp_vals[param_type], clamp_vals[param_type]);
    // Rescale the parameter to WARPEDMODEL_PRECISION_BITS so it is compatible
    // with the warped motion library
    param_value *= (1 << scale_vals[param_type]);

    // Undo the zero-centering step if necessary
    return param_value + (is_one_centered << WARPEDMODEL_PREC_BITS);
}
#else
// Adds some offset to a global motion parameter and handles
// all of the necessary precision shifts, clamping, and
// zero-centering.
static int32_t add_param_offset(int param_index, int32_t param_value, int32_t offset) {
    const int scale_vals[3] = {GM_TRANS_PREC_DIFF, GM_ALPHA_PREC_DIFF, GM_ROW3HOMO_PREC_DIFF};
    const int clamp_vals[3] = {GM_TRANS_MAX, GM_ALPHA_MAX, GM_ROW3HOMO_MAX};
    // type of param: 0 - translation, 1 - affine, 2 - homography
    const int param_type      = (param_index < 2 ? 0 : (param_index < 6 ? 1 : 2));
    const int is_one_centered = (param_index == 2 || param_index == 5);

    // Make parameter zero-centered and offset the shift that was done to make
    // it compatible with the warped model
    param_value = (param_value - (is_one_centered << WARPEDMODEL_PREC_BITS)) >> scale_vals[param_type];
    // Add desired offset to the rescaled/zero-centered parameter
    param_value += offset;
    // Clamp the parameter so it does not overflow the number of bits allotted
    // to it in the Bitstream
    param_value = (int32_t)clamp(param_value, -clamp_vals[param_type], clamp_vals[param_type]);
    // Rescale the parameter to WARPEDMODEL_PRECISION_BITS so it is compatible
    // with the warped motion library
    param_value *= (1 << scale_vals[param_type]);

    // Undo the zero-centering step if necessary
    return param_value + (is_one_centered << WARPEDMODEL_PREC_BITS);
}
#endif

static void force_wmtype(EbWarpedMotionParams* wm, TransformationType wmtype) {
    switch (wmtype) {
    case IDENTITY:
        wm->wmmat[0] = 0;
        wm->wmmat[1] = 0;
        AOM_FALLTHROUGH_INTENDED;
    case TRANSLATION:
        wm->wmmat[2] = 1 << WARPEDMODEL_PREC_BITS;
        wm->wmmat[3] = 0;
        AOM_FALLTHROUGH_INTENDED;
    case ROTZOOM:
        wm->wmmat[4] = -wm->wmmat[3];
        wm->wmmat[5] = wm->wmmat[2];
        AOM_FALLTHROUGH_INTENDED;
#if CLN_WMMAT
    case AFFINE: break;
#else
    case AFFINE: wm->wmmat[6] = wm->wmmat[7] = 0; break;
#endif
    default: assert(0);
    }
    wm->wmtype = wmtype;
}

#if OPT_GM_RFN_EARLY_EXIT
int64_t svt_av1_refine_integerized_param(GmControls* gm_ctrls, EbWarpedMotionParams* wm, TransformationType wmtype,
                                         uint8_t* ref, int r_width, int r_height, int r_stride, uint8_t* dst,
                                         int d_width, int d_height, int d_stride, int n_refinements, uint8_t chess_refn,
                                         int64_t best_frame_error, uint32_t pic_sad, int params_cost) {
#else
int64_t svt_av1_refine_integerized_param(EbWarpedMotionParams *wm, TransformationType wmtype, uint8_t *ref, int r_width,
                                         int r_height, int r_stride, uint8_t *dst, int d_width, int d_height,
                                         int d_stride, int n_refinements, uint8_t chess_refn,
                                         int64_t best_frame_error) {
#endif
    static const int max_trans_model_params[TRANS_TYPES] = {0, 2, 4, 6};
    const int        border                              = ERRORADV_BORDER;
    int              i                                   = 0, p;
    int              n_params                            = max_trans_model_params[wmtype];
    int32_t         *param_mat                           = wm->wmmat;
    int64_t          step_error, best_error;
    int32_t          step;
    int32_t         *param;
    int32_t          curr_param;
    int32_t          best_param;

    force_wmtype(wm, wmtype);
    best_error = svt_av1_warp_error(wm,
                                    ref,
                                    r_width,
                                    r_height,
                                    r_stride,
                                    dst + border * d_stride + border,
                                    border,
                                    border,
                                    d_width - 2 * border,
                                    d_height - 2 * border,
                                    d_stride,
                                    0,
                                    0,
                                    chess_refn,
                                    best_frame_error);
    best_error = AOMMIN(best_error, best_frame_error);
#if OPT_GM_RFN_EARLY_EXIT
    if (gm_ctrls->rfn_early_exit &&
        !svt_av1_is_enough_erroradvantage((double)best_error / pic_sad, params_cost, GM_ERRORADV_TR_1)) {
        return best_error;
    }
#endif
    step = 1 << (5 - 1); //initial step=16
    for (i = 0; i < n_refinements; i++, step >>= 1) {
        for (p = 0; p < n_params; ++p) {
#if !OPT_GM_PARAM_REFIN
            int step_dir = 0;
#endif
            // Skip searches for parameters that are forced to be 0
            param      = param_mat + p;
            curr_param = *param;
            best_param = curr_param;
            // look to the left
            *param     = add_param_offset(p, curr_param, -step);
            step_error = svt_av1_warp_error(wm,
                                            ref,
                                            r_width,
                                            r_height,
                                            r_stride,
                                            dst + border * d_stride + border,
                                            border,
                                            border,
                                            d_width - 2 * border,
                                            d_height - 2 * border,
                                            d_stride,
                                            0,
                                            0,
                                            chess_refn,
                                            best_error);
            if (step_error < best_error) {
                best_error = step_error;
                best_param = *param;
#if !OPT_GM_PARAM_REFIN
                step_dir = -1;
#endif
            }

            // look to the right
            *param     = add_param_offset(p, curr_param, step);
            step_error = svt_av1_warp_error(wm,
                                            ref,
                                            r_width,
                                            r_height,
                                            r_stride,
                                            dst + border * d_stride + border,
                                            border,
                                            border,
                                            d_width - 2 * border,
                                            d_height - 2 * border,
                                            d_stride,
                                            0,
                                            0,
                                            chess_refn,
                                            best_error);
            if (step_error < best_error) {
                best_error = step_error;
                best_param = *param;
#if !OPT_GM_PARAM_REFIN
                step_dir = 1;
#endif
            }
            *param = best_param;

#if !OPT_GM_PARAM_REFIN
            // look to the direction chosen above repeatedly until error increases
            // for the biggest step size
            while (step_dir) {
                *param     = add_param_offset(p, best_param, step * step_dir);
                step_error = svt_av1_warp_error(wm,
                                                ref,
                                                r_width,
                                                r_height,
                                                r_stride,
                                                dst + border * d_stride + border,
                                                border,
                                                border,
                                                d_width - 2 * border,
                                                d_height - 2 * border,
                                                d_stride,
                                                0,
                                                0,
                                                chess_refn,
                                                best_error);
                if (step_error < best_error) {
                    best_error = step_error;
                    best_param = *param;
                } else {
                    *param   = best_param;
                    step_dir = 0;
                }
            }
#endif
        }
    }
    force_wmtype(wm, wmtype);
    wm->wmtype = get_wmtype(wm);
    return best_error;
}

#if !CLN_RANSAC // done inside ransac()
static void get_inliers_from_indices(MotionModel* params, int* correspondences) {
    int* inliers_tmp = (int*)svt_aom_malloc(2 * MAX_CORNERS * sizeof(*inliers_tmp));
    memset(inliers_tmp, 0, 2 * MAX_CORNERS * sizeof(*inliers_tmp));

    for (int i = 0; i < params->num_inliers; i++) {
        int index              = params->inliers[i];
        inliers_tmp[2 * i]     = correspondences[4 * index];
        inliers_tmp[2 * i + 1] = correspondences[4 * index + 1];
    }
    svt_memcpy(params->inliers, inliers_tmp, sizeof(*inliers_tmp) * 2 * MAX_CORNERS);
    svt_aom_free(inliers_tmp);
}
#endif

#if CLN_GM
#if OPT_GM_CORESP_FROM_MV
// Generate the corresponding points for the current ref frame. The corners of the current frame are input.
// The function will compute the corners of the ref frame and then generate the correspondence points.
static void correspondence_from_corners(GmControls* gm_ctrls, uint8_t* frm_buffer, int frm_width, int frm_height,
                                        int frm_stride, int* frm_corners, int num_frm_corners, uint8_t* ref,
                                        int ref_stride, Correspondence* correspondences, int* num_correspondences) {
    int ref_corners[2 * MAX_CORNERS];

    int num_ref_corners = svt_av1_fast_corner_detect(
        (unsigned char*)ref, frm_width, frm_height, ref_stride, ref_corners, MAX_CORNERS);

    num_ref_corners = num_ref_corners * gm_ctrls->corners / 4;
    num_frm_corners = num_frm_corners * gm_ctrls->corners / 4;

    // find correspondences between the two images
    *num_correspondences = svt_av1_determine_correspondence(frm_buffer,
                                                            (int*)frm_corners,
                                                            num_frm_corners,
                                                            ref,
                                                            (int*)ref_corners,
                                                            num_ref_corners,
                                                            frm_width,
                                                            frm_height,
                                                            frm_stride,
                                                            ref_stride,
                                                            correspondences,
                                                            gm_ctrls->match_sz);
}

static void correspondence_from_mvs(PictureParentControlSet* pcs, Correspondence* correspondences,
                                    int* num_correspondences, uint8_t list_idx, uint8_t ref_idx) {
    int count_correspondences = 0;
    // 0: 64x64, 1: 32x32, 2: 16x16, 3: 8x8
    const CorrespondenceMethod mv_search_lvl = pcs->gm_ctrls.correspondence_method;
    assert(mv_search_lvl < CORNERS);
    const int      block_size        = 64 >> mv_search_lvl;
    const int      blocks_per_line   = 1 << mv_search_lvl;
    const int      num_blocks_per_sb = blocks_per_line * blocks_per_line;
    const int      starting_n_idx    = mv_search_lvl == MV_64x64 ? 0
                : mv_search_lvl == MV_32x32                      ? 1
                : mv_search_lvl == MV_16x16                      ? 5
                                                                 : 21 /*MV_8x8*/;
    const uint16_t pic_b64_width     = (uint16_t)((pcs->aligned_width + pcs->scs->b64_size - 1) / pcs->scs->b64_size);
    const uint16_t pic_b64_height    = (uint16_t)((pcs->aligned_height + pcs->scs->b64_size - 1) / pcs->scs->b64_size);
    assert(pcs->b64_total_count == pic_b64_width * pic_b64_height);

    for (uint16_t b64_y = 0; b64_y < pic_b64_height; b64_y++) {
        for (uint16_t b64_x = 0; b64_x < pic_b64_width; b64_x++) {
            uint16_t b64_idx = b64_y * pic_b64_width + b64_x;
            for (int i = 0; i < num_blocks_per_sb; i++) {
                // If the starting x/y position is outside the frame, don't include it
                if ((b64_x * pcs->scs->b64_size) + (i % blocks_per_line) * block_size >= pcs->aligned_width ||
                    (b64_y * pcs->scs->b64_size) + (i / blocks_per_line) * block_size >= pcs->aligned_height)
                    continue;
                uint8_t n_idx = starting_n_idx + i;

                if (!pcs->enable_me_8x8) {
                    if (n_idx >= MAX_SB64_PU_COUNT_NO_8X8)
                        n_idx = me_idx_85_8x8_to_16x16_conversion[n_idx - MAX_SB64_PU_COUNT_NO_8X8];
                    if (!pcs->enable_me_16x16)
                        if (n_idx >= MAX_SB64_PU_COUNT_WO_16X16)
                            n_idx = me_idx_16x16_to_parent_32x32_conversion[n_idx - MAX_SB64_PU_COUNT_WO_16X16];
                }

                uint8_t      total_me_cnt  = pcs->pa_me_data->me_results[b64_idx]->total_me_candidate_index[n_idx];
                MeCandidate* me_cand_array = &(
                    pcs->pa_me_data->me_results[b64_idx]->me_candidate_array[n_idx * pcs->pa_me_data->max_cand]);

                // Find MV for the block for the appropriate reference frame
                Mv   mv;
                bool found_mv = false;
                for (uint32_t me_cand_i = 0; me_cand_i < total_me_cnt; ++me_cand_i) {
                    const MeCandidate* me_cand = &me_cand_array[me_cand_i];
                    assert(me_cand->direction <= 2);

                    // don't consider bipred candidates
                    if (me_cand->direction == 2)
                        continue;

                    if (me_cand->direction == 0) {
                        if (list_idx == me_cand->ref0_list && ref_idx == me_cand->ref_idx_l0) {
                            mv.x = pcs->pa_me_data->me_results[b64_idx]
                                       ->me_mv_array[n_idx * pcs->pa_me_data->max_refs +
                                                     (list_idx ? pcs->pa_me_data->max_l0 : 0) + ref_idx]
                                       .x_mv;
                            mv.y = pcs->pa_me_data->me_results[b64_idx]
                                       ->me_mv_array[n_idx * pcs->pa_me_data->max_refs +
                                                     (list_idx ? pcs->pa_me_data->max_l0 : 0) + ref_idx]
                                       .y_mv;
                            found_mv = true;
                            break;
                        }
                    }
                    if (me_cand->direction == 1) {
                        if (list_idx == me_cand->ref1_list && ref_idx == me_cand->ref_idx_l1) {
                            mv.x = pcs->pa_me_data->me_results[b64_idx]
                                       ->me_mv_array[n_idx * pcs->pa_me_data->max_refs +
                                                     (list_idx ? pcs->pa_me_data->max_l0 : 0) + ref_idx]
                                       .x_mv;
                            mv.y = pcs->pa_me_data->me_results[b64_idx]
                                       ->me_mv_array[n_idx * pcs->pa_me_data->max_refs +
                                                     (list_idx ? pcs->pa_me_data->max_l0 : 0) + ref_idx]
                                       .y_mv;
                            found_mv = true;
                            break;
                        }
                    }
                }

                if (found_mv) {
                    // clang-format off
                    const int shift = pcs->gm_downsample_level == GM_DOWN ? 1
                                    : pcs->gm_downsample_level == GM_DOWN16 ? 2
                                    : 0;
                    const uint8_t b64_size = pcs->scs->b64_size;
                    correspondences[count_correspondences].x =
                        ((b64_x * b64_size) + (i % blocks_per_line) * block_size) >> shift; // x
                    correspondences[count_correspondences].y =
                        ((b64_y * b64_size) + (i / blocks_per_line) * block_size) >> shift; // y
                    correspondences[count_correspondences].rx =
                        ((b64_x * b64_size) + (i % blocks_per_line) * block_size + mv.x) >> shift; // rx
                    correspondences[count_correspondences].ry =
                        ((b64_y * b64_size) + (i / blocks_per_line) * block_size + mv.y) >> shift; // ry
                    count_correspondences++;
                    // clang-format on
                }
            }
        }
    }
    *num_correspondences = count_correspondences;
}

// Generate the corresponding points for the current ref frame. The corners of the current frame are input.
// The function will compute the corners of the ref frame and then generate the correspondence points.
void gm_compute_correspondence(PictureParentControlSet* pcs, uint8_t* frm_buffer, int frm_width, int frm_height,
                               int frm_stride, int* frm_corners, int num_frm_corners, uint8_t* ref, int ref_stride,
                               Correspondence* correspondences, int* num_correspondences, uint8_t list_idx,
                               uint8_t ref_idx) {
    if (pcs->gm_ctrls.correspondence_method == CORNERS) {
        correspondence_from_corners(&pcs->gm_ctrls,
                                    frm_buffer,
                                    frm_width,
                                    frm_height,
                                    frm_stride,
                                    frm_corners,
                                    num_frm_corners,
                                    ref,
                                    ref_stride,
                                    correspondences,
                                    num_correspondences);
    } else {
        assert(pcs->gm_ctrls.correspondence_method <= MV_8x8 && pcs->gm_ctrls.correspondence_method >= MV_64x64);
        correspondence_from_mvs(pcs, correspondences, num_correspondences, list_idx, ref_idx);
    }
}
#else
// Generate the corresponding points for the current ref frame. The corners of the current frame are input.
// The function will compute the corners of the ref frame and then generate the correspondence points.
void gm_compute_correspondence(GmControls *gm_ctrls, uint8_t *frm_buffer, int frm_width, int frm_height, int frm_stride,
                               int *frm_corners, int num_frm_corners, uint8_t *ref, int ref_stride,
                               Correspondence *correspondences, int *num_correspondences) {
    int ref_corners[2 * MAX_CORNERS];

    int num_ref_corners = svt_av1_fast_corner_detect(
        (unsigned char *)ref, frm_width, frm_height, ref_stride, ref_corners, MAX_CORNERS);

    num_ref_corners = num_ref_corners * gm_ctrls->corners / 4;
    num_frm_corners = num_frm_corners * gm_ctrls->corners / 4;

    // find correspondences between the two images
    *num_correspondences = svt_av1_determine_correspondence(frm_buffer,
                                                            (int *)frm_corners,
                                                            num_frm_corners,
                                                            ref,
                                                            (int *)ref_corners,
                                                            num_ref_corners,
                                                            frm_width,
                                                            frm_height,
                                                            frm_stride,
                                                            ref_stride,
                                                            correspondences,
                                                            gm_ctrls->match_sz);
}
#endif

// Take the input correspondences and determine the params for the gm type via ransac
#if CLN_RANSAC
void determine_gm_params(TransformationType type, MotionModel* params_by_motion, int num_motions,
                         Correspondence* correspondences, int num_correspondences) {
    bool mem_alloc_failed;
    svt_aom_ransac(correspondences, num_correspondences, type, params_by_motion, num_motions, &mem_alloc_failed);
#else
void determine_gm_params(TransformationType type, int *num_inliers_by_motion, MotionModel *params_by_motion,
                         int num_motions, Correspondence *correspondences, int num_correspondences) {
    RansacFunc ransac = svt_av1_get_ransac_type(type);
    ransac((int *)correspondences, num_correspondences, num_inliers_by_motion, params_by_motion, num_motions);

    // Set num_inliers = 0 for motions with too few inliers so they are ignored.
    for (int i = 0; i < num_motions; ++i) {
        if (num_inliers_by_motion[i] < MIN_INLIER_PROB * num_correspondences || num_correspondences == 0) {
            num_inliers_by_motion[i] = 0;
        } else {
            get_inliers_from_indices(&params_by_motion[i], (int *)correspondences);
        }
    }
#endif
}
#else
static int compute_global_motion_feature_based(TransformationType type, uint8_t corners, unsigned char *frm_buffer,
                                               int frm_width, int frm_height, int frm_stride, int *frm_corners,
                                               int num_frm_corners, uint8_t *ref, int ref_stride, int bit_depth,
                                               int *num_inliers_by_motion, MotionModel *params_by_motion,
                                               int num_motions, uint8_t match_sz) {
    (void)bit_depth;
    assert(bit_depth == EB_EIGHT_BIT);
    int            i;
    int            num_ref_corners;
    int            num_correspondences;
    int           *correspondences;
    int            ref_corners[2 * MAX_CORNERS];
    unsigned char *ref_buffer = ref;
    RansacFunc     ransac     = svt_av1_get_ransac_type(type);

    num_ref_corners = svt_av1_fast_corner_detect(
        ref_buffer, frm_width, frm_height, ref_stride, ref_corners, MAX_CORNERS);

    num_ref_corners = num_ref_corners * corners / 4;

    // find correspondences between the two images
    correspondences     = (int *)malloc(num_frm_corners * 4 * sizeof(*correspondences));
    num_correspondences = svt_av1_determine_correspondence(frm_buffer,
                                                           (int *)frm_corners,
                                                           num_frm_corners,
                                                           ref_buffer,
                                                           (int *)ref_corners,
                                                           num_ref_corners,
                                                           frm_width,
                                                           frm_height,
                                                           frm_stride,
                                                           ref_stride,
                                                           correspondences,
                                                           match_sz);
    ransac(correspondences, num_correspondences, num_inliers_by_motion, params_by_motion, num_motions);

    // Set num_inliers = 0 for motions with too few inliers so they are ignored.
    for (i = 0; i < num_motions; ++i) {
        if (num_inliers_by_motion[i] < MIN_INLIER_PROB * num_correspondences || num_correspondences == 0) {
            num_inliers_by_motion[i] = 0;
        } else {
            get_inliers_from_indices(&params_by_motion[i], correspondences);
        }
    }

    free(correspondences);

    // Return true if any one of the motions has inliers.
    for (i = 0; i < num_motions; ++i) {
        if (num_inliers_by_motion[i] > 0)
            return 1;
    }
    return 0;
}

int svt_av1_compute_global_motion(TransformationType type, uint8_t corners, unsigned char *frm_buffer, int frm_width,
                                  int frm_height, int frm_stride, int *frm_corners, int num_frm_corners, uint8_t *ref,
                                  int ref_stride, int bit_depth, GlobalMotionEstimationType gm_estimation_type,
                                  int *num_inliers_by_motion, MotionModel *params_by_motion, int num_motions,
                                  uint8_t match_sz) {
    switch (gm_estimation_type) {
    case GLOBAL_MOTION_FEATURE_BASED:
        return compute_global_motion_feature_based(type,
                                                   corners,
                                                   frm_buffer,
                                                   frm_width,
                                                   frm_height,
                                                   frm_stride,
                                                   frm_corners,
                                                   num_frm_corners,
                                                   ref,
                                                   ref_stride,
                                                   bit_depth,
                                                   num_inliers_by_motion,
                                                   params_by_motion,
                                                   num_motions,
                                                   match_sz);
    default: assert(0 && "Unknown global motion estimation type");
    }
    return 0;
}
#endif
