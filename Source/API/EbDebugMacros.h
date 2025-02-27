/*
* Copyright(c) 2020 Intel Corporation
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

/*
* This file contains only debug macros that are used during the development
* and are supposed to be cleaned up every tag cycle
* all macros must have the following format:
* - adding a new feature should be prefixed by FTR_
* - tuning a feature should be prefixed by TUNE_
* - enabling a feature should be prefixed by EN_
* - disabling a feature should be prefixed by DIS_
* - bug fixes should be prefixed by FIX_
* - code refactors should be prefixed by RFCTR_
* - code cleanups should be prefixed by CLN_
* - optimizations should be prefixed by OPT_
* - all macros must have a coherent comment explaining what the MACRO is doing
* - #if 0 / #if 1 are not to be used
*/

#ifndef EbDebugMacros_h
#define EbDebugMacros_h

// clang-format off

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// svt-15 macros
#define OPT_MAXP0P1                 1 // Modulate TH for intra modes
#define CLN_REMOVE_TXS_ME_MOD       1 // Remove Me-based modulation of TXS level
#define OPT_TXS_D2_IFF_D1_BEST      0 // Test TXS depth 2 iff depth 1 is best
#define OPT_MDS_FEATS_PRUNING       1 // Optimize the feautres used in MDS1/2 and enable MDS2 pruning when MDS2 is skipped (using MDS1 cost)
#define CLN_MDS_SIGS                1 // Cleanup MD staging signals
#if CLN_MDS_SIGS
#define CLN_VALID_PRED              1 // Rename valid_pred to valid_luma_pred
#define CLN_RENAME_MDS_SKIP         1 // Rename ctx->mds_* signals. Make each signal "do" instead of "skip".
#define CLN_PRED_SIGS               1 // Cleanup signals related to performing prediction in each MDS
#endif
#define OPT_DEPTHS_CTRL             1 // Remove depths_ctrls, and rework depth_refinement_ctrls
#define OPT_INTRA                   1 // Optimize INTRA levels
#define OPT_MDS_BYPASS_NICS         1 // Don't update NIC counts when a stage is bypassed. Update pruning THs to adapt.
#define OPT_FILTER_INTRA            0 // Optimize Filter-INTRA levels
#define OPT_DR_SETTINGS             1 // Optimize the depth-removal settings
#define CLN_HIGH_FREQUENCY          1 // delete high frequency modulation
#define OPT_TXS                     1 // TXS Tuning
#define OPT_REMOVE_NIC_QP_BANDS     1 // Remove QP banding for NICs
#define FIX_COMPOUND_TEST           1 // Fix unit test to enable wedge intrinsic
#define OPT_DELTA_QP                1 // Break-down r0_based_qps_qpm into three separate components and added a fourth signal to specify whether
                                      // to use SB QP during quantization:r0_gen, r0_qps, r0_delta_qp_md,r0_delta_qp_quant
#define TUNE_M5                     1 // Tune M5
#define TUNE_M7                     1 // Tune M7
#define TUNE_M4                     1 // Tune M4
#define TUNE_M3                     1 // Tune M3
#define OPT_USE_EXP_HME_ME          1 // Use an exponential (instead of a linear) QP-based function for HME/ME search area scaling
#define TUNE_M2                     1 // Tune M2
#define OPT_SC_ME                   1 // Opt SC for ME; new variant of the SC classifier that operates at the 8×8 block level instead of the 16×16 block level, aiming to enhance SC detection, particularly for sub-1080p SC clips (sc_class4)
#define OPT_ME                      1 // ME Tuning
#define FIX_GM_CANDS                1 // Allow translation bipred for GM; enable GM for 4xN blocks for rotzoom
#define OPT_NSQ_SEARCH              1 // NSQ SEARCH Tuning
#define OPT_TXT                     1 // TXT Tuning; fixing the onion ring and going more aggressive in txt_group_intra_gt_eq_16x16 in txt level 10
#define OPT_ME_M0_TO_M3             1 // ME Tuning for M0 to M3
#define OPT_ME_M7_TO_M9             1 // ME Tuning for M7 to M9. Note: Use a per-resolution Test Set (e.g., high-frame Test Set) for evaluation, as the reference previously included a resolution check.
#define OPT_USE_EXP_DEPTHS          1 // Use an exponential (instead of a linear) QP-based function for depth-th(s) scaling
#define OPT_USE_EXP_TXT             1 // Use an exponential (instead of a linear) QP-based function for txt-th(s) scaling
#define OPT_USE_EXP_PME             1 // Use an exponential (instead of a linear) QP-based function for pme search area scaling
#define OPT_USE_EXP_TF              1 // Use an exponential (instead of a linear) QP-based function for tf ref-pruning  scaling
#define TUNE_M8                     1 // Tune M8
#define OPT_COMPOUND                1 // Optimize compound search
#define TUNE_M1                     1 // Tune M1
#define OPT_QP_TH                   1 // Change the barrier between linear/exp QP scaling to QP 45/46
#define OPT_USE_EXP_NSQ             1 // Use an exponential QP-based function for NSQ th-pruning
#define OPT_II_PRECOMPUTE           1 // Use precomputed inter-intra pred in MDS0/3
#define OPT_II_MASK_GEN             1 // Generate inter-intra mask at the beginning of the encoder and reuse
#define FIX_INTRA_UPDATES           1 // Ensure intra buffers are updated when inter-intra is enabled
#define FIX_OBMC_QP_BANDING         1 // Fix OBMC QP banding (lossless for all presets)
#define OPT_II_M1                   1 // Enable inter-intra for all frames in M1
#define OPT_TX_SHORT                1 // Create a new set of levels for tx_shortcut_level replacing the old set of levels and tuning them to M3-M10. MR-M2 still use tx_shortcut_level OFF.
#define CLN_USE_NEIGHBOUR_SIG       1 // Cleanup the use_neighbour_info signal. It has been set to 0 everywhere for tx_shortcut_level in OPT_TX_SHORT.
#define TUNE_M0                     1 // Tune M0

//FOR DEBUGGING - Do not remove
#define FIX_AVX512_ICL_RTCD         1 // Correct avx512icl support detection
#define OPT_LD_LATENCY2         1 // Latency optimization for low delay - to keep the Macro for backwards testing until 3.0
#define LOG_ENC_DONE            0 // log encoder job one
#define DEBUG_TPL               0 // Prints to debug TPL
#define DETAILED_FRAME_OUTPUT   0 // Prints detailed frame output from the library for debugging
#define TUNE_CHROMA_SSIM        0 // Allows for Chroma and SSIM BDR-based Tuning
#define TUNE_CQP_CHROMA_SSIM    0 // Tune CQP qp scaling towards improved chroma and SSIM BDR

#define MIN_PIC_PARALLELIZATION 0 // Use the minimum amount of picture parallelization
#define SRM_REPORT              0 // Report SRM status
#define LAD_MG_PRINT            0 // Report LAD
#define RC_NO_R2R               0 // This is a debugging flag for RC and makes encoder to run with no R2R in RC mode
                                  // Note that the speed might impacted significantly
#if !RC_NO_R2R
#define FTR_KF_ON_FLY_SAMPLE      0 // Sample code to signal KF
#define FTR_RES_ON_FLY_SAMPLE     0 // Sample functions to change the resolution on the fly
#define FTR_RATE_ON_FLY_SAMPLE     0 // Sample functions to change bit rate
#endif
// Super-resolution debugging code
#define DEBUG_SCALING           0
#define DEBUG_TF                0
#define DEBUG_UPSCALING         0
#define DEBUG_SUPERRES_RECODE   0
#define DEBUG_SUPERRES_ENERGY   0
#define DEBUG_RC_CAP_LOG        0 // Prints for RC cap

// Switch frame debugging code
#define DEBUG_SFRAME            0

// Variance boost debugging code
#define DEBUG_VAR_BOOST         0
#define DEBUG_VAR_BOOST_QP      0
#define DEBUG_VAR_BOOST_STATS   0

// Quantization matrices
#define DEBUG_QM_LEVEL          0
#define DEBUG_STARTUP_MG_SIZE   0
#define DEBUG_SEGMENT_QP        0
#define DEBUG_ROI               0
#ifdef __cplusplus
}
#endif // __cplusplus

// clang-format on

#endif // EbDebugMacros_h
