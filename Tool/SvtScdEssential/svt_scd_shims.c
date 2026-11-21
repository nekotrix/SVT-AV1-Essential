/*
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0 in LICENSE and PATENTS.
 */

#define RTCD_C
#define AOM_RTCD_C
#include "aom_dsp_rtcd.h"
#include "common_dsp_rtcd.h"
#include "common_utils.h"
#include "enc_mode_config.h"
#include "global_me.h"
#include "inter_prediction.h"
#include "me_sad_calculation.h"
#include "motion_estimation.h"
#include "resize.h"
#include "temporal_filtering.h"

#if defined(SVT_SCD_USE_AVX2)
#define AOM_ARCH_X86_64 ARCH_X86_64
#include "third_party/aom/aom_ports/x86.h"
#endif

// Unreachable linker stubs for unused encoder routines pulled in by
// svt_scd_wrappers.c and inter_prediction.h before dead-code elimination.
DECLARE_ALIGNED(256, const InterpKernel, sub_pel_filters_8[SUBPEL_SHIFTS])       = {0};
DECLARE_ALIGNED(256, const InterpKernel, sub_pel_filters_4[SUBPEL_SHIFTS])       = {0};
DECLARE_ALIGNED(256, const InterpKernel, sub_pel_filters_8sharp[SUBPEL_SHIFTS])  = {0};
DECLARE_ALIGNED(256, const InterpKernel, sub_pel_filters_8smooth[SUBPEL_SHIFTS]) = {0};
DECLARE_ALIGNED(256, const InterpKernel, bilinear_filters[SUBPEL_SHIFTS])        = {0};
DECLARE_ALIGNED(256, const InterpKernel, sub_pel_filters_4smooth[SUBPEL_SHIFTS]) = {0};

const uint8_t    eb_num_pels_log2_lookup[BLOCK_SIZES_ALL] = {0};
AomVarianceFnPtr svt_aom_mefn_ptr[BLOCK_SIZES_ALL]        = {0};

void first_pass_frame_end_one_pass(PictureParentControlSet* pcs) {
    (void)pcs;
}

void me_reset_carryover(MotionEstimationData* me_data) {
    (void)me_data;
}

EbErrorType me_update_param(MotionEstimationData* me_data, struct SequenceControlSet* scs) {
    (void)me_data;
    (void)scs;
    return EB_ErrorNone;
}

void svt_aom_assert_err(uint32_t condition, char* err_msg) {
    (void)condition;
    (void)err_msg;
}

EbErrorType svt_aom_denoise_and_model_ctor(AomDenoiseAndModel* object_ptr, EbPtr object_init_data_ptr) {
    (void)object_ptr;
    (void)object_init_data_ptr;
    return EB_ErrorNone;
}

int32_t svt_aom_denoise_and_model_run(struct AomDenoiseAndModel* ctx, EbPictureBufferDesc* sd, AomFilmGrain* film_grain,
                                      int32_t use_highbd) {
    (void)ctx;
    (void)sd;
    (void)film_grain;
    (void)use_highbd;
    return 0;
}

uint8_t svt_aom_derive_gm_level(PictureParentControlSet* pcs, bool super_res_off) {
    (void)pcs;
    (void)super_res_off;
    return 0;
}

void svt_aom_fga_apply_multiplier(AomFilmGrain* grain, float multiplier) {
    (void)grain;
    (void)multiplier;
}

SvtFgaContext* svt_aom_fga_create(int width, int height, int bit_depth) {
    (void)width;
    (void)height;
    (void)bit_depth;
    return NULL;
}

void svt_aom_fga_free(SvtFgaContext* ctx) {
    (void)ctx;
}

int svt_aom_fga_run(SvtFgaContext* ctx, EbPictureBufferDesc* source, const EbPictureBufferDesc* next, bool scene_cut,
                    bool denoise, AomFilmGrain* grain) {
    (void)ctx;
    (void)source;
    (void)next;
    (void)scene_cut;
    (void)denoise;
    (void)grain;
    return 0;
}

void svt_aom_get_max_allocated_me_refs(uint8_t ref_count_used_list0, uint8_t ref_count_used_list1,
                                       uint8_t* max_ref_to_alloc, uint8_t* max_cand_to_alloc) {
    (void)ref_count_used_list0;
    (void)ref_count_used_list1;
    (void)max_ref_to_alloc;
    (void)max_cand_to_alloc;
}

const MiniGopStats* svt_aom_get_mini_gop_stats(const uint32_t mini_gop_index) {
    (void)mini_gop_index;
    return NULL;
}

PredictionStructure* svt_aom_get_prediction_structure(PredictionStructureGroup* prediction_structure_group_ptr,
                                                      PredStructure pred_structure, uint32_t levels_of_hierarchy) {
    (void)prediction_structure_group_ptr;
    (void)pred_structure;
    (void)levels_of_hierarchy;
    return NULL;
}

void svt_aom_get_qp_based_th_scaling_factors(bool enable_qp_based_th_scaling, uint32_t* ret_q_weight,
                                             uint32_t* ret_q_weight_denom, uint32_t qp) {
    (void)enable_qp_based_th_scaling;
    (void)ret_q_weight;
    (void)ret_q_weight_denom;
    (void)qp;
}

void svt_aom_gm_pre_processor(PictureParentControlSet* pcs, PictureParentControlSet** pcs_list) {
    (void)pcs;
    (void)pcs_list;
}

void svt_aom_init_resize_picture(SequenceControlSet* scs, PictureParentControlSet* pcs) {
    (void)scs;
    (void)pcs;
}

int32_t svt_aom_noise_log1p_fp16(int32_t noise_level_fp16) {
    (void)noise_level_fp16;
    return 0;
}

void svt_aom_set_gm_controls(PictureParentControlSet* pcs, uint8_t gm_level) {
    (void)pcs;
    (void)gm_level;
}

void svt_aom_sig_deriv_me_tf(PictureParentControlSet* pcs, MeContext* me_ctx) {
    (void)pcs;
    (void)me_ctx;
}

void svt_aom_sig_deriv_multi_processes_allintra(SequenceControlSet* scs, PictureParentControlSet* pcs) {
    (void)scs;
    (void)pcs;
}

void svt_aom_sig_deriv_multi_processes_default(SequenceControlSet* scs, PictureParentControlSet* pcs) {
    (void)scs;
    (void)pcs;
}

void svt_aom_sig_deriv_multi_processes_rtc(SequenceControlSet* scs, PictureParentControlSet* pcs) {
    (void)scs;
    (void)pcs;
}

uint8_t svt_aom_tf_max_ref_per_struct(uint32_t hierarchical_levels, uint8_t type, bool direction) {
    (void)hierarchical_levels;
    (void)type;
    (void)direction;
    return 0;
}

EbErrorType svt_av1_init_temporal_filtering(PictureParentControlSet** pcs_list, PictureParentControlSet* centre_pcs,
                                            MotionEstimationContext_t* me_context_ptr, int32_t segment_index) {
    (void)pcs_list;
    (void)centre_pcs;
    (void)me_context_ptr;
    (void)segment_index;
    return EB_ErrorNone;
}

EbErrorType svt_get_empty_object(EbFifo* empty_fifo_ptr, EbObjectWrapper** wrapper_dbl_ptr) {
    (void)empty_fifo_ptr;
    (void)wrapper_dbl_ptr;
    return EB_ErrorNone;
}

EbErrorType svt_get_full_object(EbFifo* full_fifo_ptr, EbObjectWrapper** wrapper_dbl_ptr) {
    (void)full_fifo_ptr;
    (void)wrapper_dbl_ptr;
    return EB_ErrorNone;
}

MvReferenceFrame svt_get_ref_frame_type(uint8_t list, uint8_t ref_idx) {
    (void)list;
    (void)ref_idx;
    return NONE_FRAME;
}

EbErrorType svt_object_inc_live_count(EbObjectWrapper* wrapper_ptr, uint32_t increment_number) {
    (void)wrapper_ptr;
    (void)increment_number;
    return EB_ErrorNone;
}

EbErrorType svt_post_full_object(EbObjectWrapper* object_ptr) {
    (void)object_ptr;
    return EB_ErrorNone;
}

EbErrorType svt_release_object(EbObjectWrapper* object_ptr) {
    (void)object_ptr;
    return EB_ErrorNone;
}

EbFifo* svt_system_resource_get_consumer_fifo(const EbSystemResource* resource_ptr, uint32_t index) {
    (void)resource_ptr;
    (void)index;
    return NULL;
}

EbFifo* svt_system_resource_get_producer_fifo(const EbSystemResource* resource_ptr, uint32_t index) {
    (void)resource_ptr;
    (void)index;
    return NULL;
}

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
