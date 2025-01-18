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

#define CLN_LP_LVLS             1 // Change --lp input to represent the levels of parallelization that are actually implemented in the code
#define FTR_LOSSLESS_SUPPORT    1 // Losless coding support
#define FTR_STILL_PICTURE       1 // Still picture support
#define FTR_STARTUP_QP          1 // Add the ability to add an offset to the input-qp for the startup GOP prior to the picture-qp derivation

#define TUNE_MFMV_FD2               1 // Disable MFMV in fd2
#define TUNE_SB64_FD2               1 // Change QP check for SB64 in fd2
#define TUNE_M5                     1 // M5 tuning
#define TUNE_M4                     1 // M4 tuning
#define TUNE_M0                     1 // M0 tuning
#define TUNE_M3                     1 // M3 tuning
#define TUNE_MR                     1 // MR tuning
#define TUNE_M2                     1 // M2 tuning
#define TUNE_M1                     1 // M1 tuning
#define TUNE_M7                     1 // M7 tuning
#define TUNE_M8                     1 // M8 tuning
#define OPT_MDS0_EXIT               1 // Opt mds0 exit: reset the best-mds0-cost for each class to ensure that at least one candidate per class is retained before the mds1 pruning phase. Then, use the block complexity = MIN(me-dist, pme_dist) to switch between the two methods
#define TUNE_M9                     1 // M9 tuning
#define TUNE_M10                    1 // M10 tuning
#define OPT_GM 1
#if OPT_GM
#define FIX_GM_TRANS                1 // Fix assumptions that disallow TRANSLATION GM model
#define CLN_WMMAT                   1 // wm matrix should have 6 entries, not 8, since two entries of the 8 are always 0
#define CLN_GM                      1 // Cleanup the GM code path
#define CLN_RANSAC                  1 // Cleanup the ransac function (port new code from libaom)
#define OPT_GM_PARAM_REFIN          1 // Remove extra refinement from GM
#define OPT_GM_RFN_EARLY_EXIT       1 // Skip GM refinement when unlikely to succeed
#define OPT_GM_CORESP_FROM_MV       1 // Generate the GM correspondence points from ME MVs
#define OPT_GM_LVLS                 1 // Optimize the GM levels used in each preset
#define CLN_UNUSED_GM_SIGS          1 // Remove unused GM signals
#define OPT_GM_LVL_M5               1 // Use GM for 480p/720p only for M5
#endif
#define OPT_DLF_FD2                 1 // Opt DLF for fd2
#define OPT_CDEF_FD2                1 // Opt CDEF for fd2
#define OPT_M6_NEW                  1 // Opt M6 new
#define OPT_M5_NEW                  1 // Opt M5 new
#define OPT_TXS                     1 // Opt txs
#define OPT_ME                      1 // Opt ME
#define OPT_TF                      1 // OPT tf
#define TUNE_M6_BDR_2               1 // OPT M6 in term of BDR
#define TUNE_M5_BDR                 1 // OPT M6 in term of BDR
#define OPT_LAMBDA                  1 // OPTimized lambda modulation: (1) Expanded the QP bands for lambda - weighting; from 2 to 4 bands, (2) Reduced the intra-percentage threshold
#define TUNE_M4_2                   1 // M4 tuning
#define TUNE_M9_2                   1 // M9 tuning
#define OPT_FD_10BIT                1 // Opt fd for 10bit
#define TUNE_LAMBDA_WEIGHT          1 // Tune lambda weight
#define FIX_SUPERRES                1 // Fix setting b64_total_count based on SB count in superres path
#define OPT_FD0_SETTINGS            1 // Unify some fd2 settings into fd0 to improve trade-offs
#define OPT_LOW_DELAY               1 // Opt Low-delay
#define FIX_DLF_ONION_RING          1 // Adopt fd0 dlf level in M10/M11 fd1 to preserve onion ring
#define CLN_SHIFT_M8                1 // Shift M8 to M7
#define CLN_SHIFT_M9                1 // Shift M9 to M8
#define CLN_SHIFT_M10               1 // Shift M10 to M9
#define CLN_SHIFT_M11               1 // Shift M11 to M10
#define FIX_DEFAULT_PRESET          1 // Change default preset to M8 to align with old default
#define OPT_LOW_DELAY_2             1 // Opt Low-delay
#define CLN_LCG_RAND16              1 // Remove duplicate definitions of lcg_rand16()
#define CLN_LPD0_FUNC               1 // Cleanup set_pic_lpd0_lvl to address style check issue
#define TUNE_M3_2                   1 // Tune M3
#define FIX_FAST_PRESET             1 // Tuning for high presets
#define TUNE_FD0_FEATS              1 // Unify CDEF and DLF levels of fd2/fd0 in M10 and M9
#define OPT_CDEF_FD1                1 // Opt CDEF for fd1
#define FIX_PUBLIC_HEADER           1
#if FIX_PUBLIC_HEADER
// EbSvtAv1Enc.h
#define FIX_BOOL                           1 // Replace Bool with the standard bool
#define FIX_HIGH_DYNAMIC_RANGE_INPUT       1 // Remove high_dynamic_range_input as hdr is not supported
#define FIX_COLOR_DESCRIPTION_PRESENT_FLAG 1 // Remove color_description_present_flag as adding color description information is not supported
#define FIX_RATE_CONTROL_MODE              1 // Use SvtAv1RcMode instead of uint32_t
#define FIX_VBV_BUFSIZE                    1 // Remove vbv_bufsize as not used
#define FIX_VBR_BIAS_PCT                   1 // Remove vbr_bias_pct as not used
#define FIX_PRED_STRUCT                    1 // Remove the unused pred_struct signals
// EbSvtAv1.h
#define FIX_EB_SVT_IO_FORMAT               1 // Remove the unused EbSvtIOFormat fields
#define FIX_P_APP_DATA                     1 // Remove p_application_private as not used
#define FTR_SIGNAL_LAYER                   1 // Output the temporal layer index of the encoded picture
#define FTR_SIGNAL_AVERAGE_QP              1 // Output the mean QP across all blocks for the encoded picture
#endif
#define FIX_SVT_AV1_CHECK_VERSION          1 // Remove the SVT_AV1_CHECK_VERSION instances
#define FIX_NO_ENCDEC                      1 // Remove REMOVE_LP1_LPN_DIFF as not used
#define FIX_REMOVE_LP1_LPN_DIFF            1 // remove NO_ENCDEC as not used
//FOR DEBUGGING - Do not remove
#define OPT_LD_LATENCY2         1 // Latency optimization for low delay - to keep the Macro for backwards testing until 3.0
#define LOG_ENC_DONE            0 // log encoder job one
#if !FIX_NO_ENCDEC
#define NO_ENCDEC               0 // bypass encDec to test cmpliance of MD. complained achieved when skip_flag is OFF. Port sample code from VCI-SW_AV1_Candidate1 branch
#endif
#define DEBUG_TPL               0 // Prints to debug TPL
#define DETAILED_FRAME_OUTPUT   0 // Prints detailed frame output from the library for debugging
#define TUNE_CHROMA_SSIM        0 // Allows for Chroma and SSIM BDR-based Tuning
#define TUNE_CQP_CHROMA_SSIM    0 // Tune CQP qp scaling towards improved chroma and SSIM BDR

#define MIN_PIC_PARALLELIZATION 0 // Use the minimum amount of picture parallelization
#define SRM_REPORT              0 // Report SRM status
#define LAD_MG_PRINT            0 // Report LAD
#define RC_NO_R2R               0 // This is a debugging flag for RC and makes encoder to run with no R2R in RC mode
                                  // Note that the speed might impacted significantly
#if FIX_REMOVE_LP1_LPN_DIFF
#if !RC_NO_R2R
#define FTR_KF_ON_FLY_SAMPLE      0 // Sample code to signal KF
#define FTR_RES_ON_FLY_SAMPLE     0 // Sample functions to change the resolution on the fly
#define FTR_RATE_ON_FLY_SAMPLE     0 // Sample functions to change bit rate
#endif
#else
#if RC_NO_R2R
#define REMOVE_LP1_LPN_DIFF     1 // Disallow single-thread/multi-thread differences
#else
#define REMOVE_LP1_LPN_DIFF     0 // Disallow single-thread/multi-thread differences
#define FTR_KF_ON_FLY_SAMPLE      0 // Sample code to signal KF
#define FTR_RES_ON_FLY_SAMPLE     0 // Sample functions to change the resolution on the fly
#define FTR_RATE_ON_FLY_SAMPLE    0 // Sample functions to change bit rate
#endif
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
