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

#define FTR_FRAME_RATE_ON_THE_FLY   1 // Add ability to change frame rate on the fly (without inserting keyframe)
#define OPT_RATE_ON_THE_FLY_NO_KF   1 // Add ability to change bitrate on the fly without inserting keyframe
#define FTR_PER_FRAME_QUALITY       1 // Add ability to compute quality for specific frame
#define FTR_SFRAME_FLEX             1 // Add S-Frame Flexible ARF Mode
#if FTR_SFRAME_FLEX
#define FTR_SFRAME_POSI             1 // Add parameter to allow user insert S-Frames by picture number
#if FTR_SFRAME_POSI
#define FTR_SFRAME_QP               1 // Add parameter to allow user set QP of S-Frame
#define FTR_SFRAME_DEC_POSI         1 // New S-Frame mode to insert S-Frames at specific position in decode order
#endif // FTR_SFRAME_POSI
#endif // FTR_SFRAME_FLEX
#define FIX_TUNE_SSIM               1 // Fix SSIM mode
#define RFCTR_PARSE_LIST            1 // Refactor parameter parsing list and support the range of int8 and uint8
#define FIX_QUEUE_DEADLOCK          1 // Use min-heap instead of queue to manage out-of-order decode orders
#define FIX_INTRA_BLUR_QP62         1 // Intra lambda-weight tuning for INTRA frames at high QPs (>=62) to reduce blurriness
#define FIX_FPS_CALC                1 // Fix frame-rate derivation to handle < 1 fps

#define CLN_MDC_FUNCS               1 // Clean-up unused MDC functions and variables
#define CLN_RECON_FUNC              1 // Clean-up MD recon function
#define CLN_REMOVE_OIS_FLAG         1 // Clean-up OIS functions and variables
#define FIX_EOB_COEF_CTX            1 // Fix the number of contexts used for signaling the EOB
#define FIX_IND_UV_SEARCH_TX        1 // Set MD stage to 0 for independent chroma search to prevent using uninit'd data to skip TX
#define FIX_CHROMA_SKIP             1 // Properly update chroma settings when chroma is skipped
#define CLN_MOVE_CHROMA_CHECK       1 // Move chroma complexity check inside MD stage check for PD1
#define CLN_MDS0_DIST_PD1           1 // Fix fast-cost derivation
#define FTR_STILL_IMAGE_UP_TO_M12   1 // Still-image up to M12
#define FTR_DEPTH_REMOVAL_INTRA     1 // Add support for variance-based depth removal for INTRA frames to speed up still-image processing
#define TUNE_STILL_IMAGE_0          1 // Perform a first round of tuning for still-image
#define OPT_MD_SIGNALS              1 // Clean up INTRA-level derivation, fix PD0/PD1 interactions, and adaptively set the intra-edge-filter flag
#define FIX_TUNE_SSIM_LAMBDA        1 // Fix SSIM mode lambda: fix per-SB lambda for balanced inter-depth decisions, and use default factors when deriving the lambda scaling factors
#define TUNE_STILL_IMAGE_1          1 // Perform a second round of tuning for still-image
#define OPT_DEFAULT_LAMBDA_MULT     1 // Use default lambda at RDOQ
#define FTR_USE_HADAMARD_MDS0       1 // Use Hadamard at mds0
#define TUNE_STILL_IMAGE_2          1 // Perform a third round of tuning for still-image
#define OPT_SSIM_METRIC             1 // Remove SSIM distortion calculations for INTRA frames
#define OPT_RECON_OPERATIONS        1 // Remove unnecessary reconstruction operations
#define OPT_FD2_FD1_STILL_IMAGE     1 // Add fast-decode for still-image

#define OPT_DEFAULT_6L              1 // Set 6L as default structure for most modes
#define EN_M11_RA                   1 // Enable M11 for RA
#define EN_FLAT_ALL_PRESETS         1 // Enable flat pred structure for all RTC presets
#define TUNE_RTC_RA_PRESETS         1 // Preset tuning for RTC mode and RA
#define CLN_ME_SCALING              1 // Remove min ME search area scaling based on non-zero framerate (which is always true)
#define OPT_RTC_FACTORS             1 // Use same RTC CBR factors as libaom
#define FIX_FRAMES_SINCE_KEY        1 // align to libaom count
#define OPT_ME_DIST_IN_RC           1 // Use 64x64 ME distortion for adjust_q_cbr_flat; update at every frame
#define FIX_RATE_SPIKES             1 // Fix bit allocation behaviour that causes every 4th frame to be huge in flat pred structure
#define OPT_LPD1_RTC                1 // Simplify and improve LPD1 for RTC
#define OPT_REMOVE_ENH_BASE         1 // Remove LD enhanced base frame
#define OPT_SUBPEL_TH               1 // Optimize subpel skipping equation
#define OPT_SKIP_CANDS_LPD1         1 // Optimize how candidates are skipped in lpd1
#define OPT_RATE_EST_FAST           1 // bypass some rate estimation steps for speed
#define OPT_LPD1_TX_SKIP            1 // Optimize LPD1 TX skipping based on full cost estimate
#define OPT_LPD0_RTC                1 // Unify LPD0-classifier between rtc and non-rtc
#define OPT_RTC_VLPD0_DEPTH         1 // Enable depth early exit for VLPD0 in RTC M12. Aggressive tx shortcut level for RTC M12.
#define CLN_UNUSED_SIGS             1 // Remove unused signals
#define CLN_MDS0_DIST_LPD1          1 // Don't shift MDS0 variance in LPD1; use full lambda for variance
#define CLN_MDS0_DIST_LPD0          1 // Don't shift MDS0 variance in LPD0
#define FIX_10BIT_BYPASS_ED         1 // Use proper lambda during MD when bypassing encdec for 10bit
#define OPT_RPS_MRP_4_REFS          1 // Reduce the number of reference pictures stored in LD
#define OPT_ENABLE_MRP_FLAT         1 // Enable multiple reference frames to be used for flat prediction structure
#define OPT_DR_RTC                  1 // Unify depth-removal between rtc and non-rtc
#if OPT_DR_RTC
#define OPT_B8                      1 // b8 for up to M10
#define OPT_DR_T_INFO               1 // Use collocated min blk size @ to modulate dr deviation-threshold
#define OPT_DR_COST_TH              1 // Enhance the granularity of the dr cost-threshold multipliers
#endif
#define FIX_DISALLOW_8X8            1 // Fix when 8x8 are needed at pic boundaries
#define TUNE_RTC_RA_PRESETS_2       1 // Preset tuning for RTC mode and RA
#define CLN_DLF_DEF                 1 // Clean dlf-level def
#define OPT_CYCLIC_REFRESH          1
#if OPT_CYCLIC_REFRESH
#define OPT_CR_CTRL                 1 // Use avg_frame_low_motion and avg_frame_qindex, reflecting stationary block, and average Q-index, from past frames, to decide whether to keep or disable cyclic-refresh for the current frame
#define OPT_BOOST_MODULATION        1 // Replace the step-based rate_boost_fac logic by quadratic scaling, so the boost increases gradually with deviation rather than jumping at fixed thresholds
#define FIX_LAMBDA_FLAT             1 // Use the the actual layer-index and hierarchical-level
#endif
#define OPT_RDOQ_RTC                1 // Use default lambda at RDOQ for RTC
#define TUNE_RTC_FLAT               1 // Preset tuning flat for RTC
#define TUNE_RTC_3L                 1 // Preset tuning 3L for RTC

#define CLN_REMOVE_SS_PIN           1 // Remove options to pin execution to certain cores/sockets (--ss/--pin)
#define CLN_REMOVE_TPL_SIG          1 // Remove enable_tpl_la signal (it is not used)
#define CLN_AQ_MODE                 1 // Rename enable_adaptive_quantization to aq_mode to reflect that it has more than 2 levels
#define CLN_REMOVE_CHANNELS         1 // Remove multiple channels from app since each channel invokes separate library call anyway

//FOR DEBUGGING - Do not remove
#define LOG_ENC_DONE            0 // log encoder job one
#define DEBUG_TPL               0 // Prints to debug TPL
#define DETAILED_FRAME_OUTPUT   0 // Prints detailed frame output from the library for debugging
#define DEBUG_BUFFERS           0 // Print process count and segments info
#define TUNE_CHROMA_SSIM        0 // Allows for Chroma and SSIM BDR-based Tuning
#define TUNE_CQP_CHROMA_SSIM    0 // Tune CQP qp scaling towards improved chroma and SSIM BDR

#define MIN_PIC_PARALLELIZATION 0 // Use the minimum amount of picture parallelization
#define SRM_REPORT              0 // Report SRM status
#define LAD_MG_PRINT            0 // Report LAD
#define RC_NO_R2R               0 // This is a debugging flag for RC and makes encoder to run with no R2R in RC mode
                                  // Note that the speed might impacted significantly
#if !RC_NO_R2R
#define FTR_KF_ON_FLY_SAMPLE         0 // Sample code to signal KF
#define FTR_RES_ON_FLY_SAMPLE        0 // Sample functions to change the resolution on the fly
#define FTR_RATE_ON_FLY_SAMPLE       0 // Sample functions to change bit rate
#if FTR_FRAME_RATE_ON_THE_FLY
#define FTR_FRAME_RATE_ON_FLY_SAMPLE 0 // Sample functions to change frame rate
#endif
#if FTR_PER_FRAME_QUALITY
#define FTR_PER_FRAME_QUALITY_SAMPLE 0 // Sample functions to compute PSNR per frame
#endif
#endif
// Super-resolution debugging code
#define DEBUG_SCALING           0
#define DEBUG_TF                0
#define DEBUG_SUPERRES_RECODE   0
#define DEBUG_SUPERRES_ENERGY   0
#define DEBUG_RC_CAP_LOG        0 // Prints for RC cap

// Switch frame debugging code
#define DEBUG_SFRAME            0

// Variance Boost debugging code
#define DEBUG_VAR_BOOST         0
#define DEBUG_VAR_BOOST_QP      0
#define DEBUG_VAR_BOOST_STATS   0

// Anti-alias aware screen content mode debugging code
#define DEBUG_AA_SCM            0

// QP scaling debugging code
#define DEBUG_QP_SCALING        0

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
