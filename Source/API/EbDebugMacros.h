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
#define OPT_SC_ME                   1 // Opt SC for ME; new variant of the SC classifier that operates at the 8x8 block level instead of the 16x16 block level, aiming to enhance SC detection, particularly for sub-1080p SC clips (sc_class4)
#define OPT_ME                      1 // ME Tuning
#define FIX_GM_CANDS                1 // Allow translation bipred for GM; enable GM for 4xN blocks for rotzoom
#define OPT_NSQ_SEARCH_M5_LVL       1 // Change M5 level of the NSQ SEARCH from 15 to 16.
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
#define OPT_OBMC                    1 // Opt OBMC; Lossless optimizations and level redefinition
#define OPT_NSQ_SEARCH_LEVELS       1 // Optimize NSQ search levels.
#define CLN_REMOVE_DEC_STRUCT       1 // Remove old struct used by the decoder
#define CLN_REMOVE_MODE_INFO        1 // Remove ModeInfo struct that has only one element; point directly to that element instead
#define OPT_NSQ_SEARCH_M7_LVL       1 // Change M7 level of the NSQ SEARCH from 18 to 17.
#define CLN_CALCULATE_VARIANCE      1 // Remove the calculation and use of variance throughout the encoder.
#define CLN_MOVE_FIELDS_MBMI        1 // Move fields from EcBlkStruct to BlockModeInfo
#define CLN_WM_CTRLS                1 // Remove unused WM features and make num_proj_ref uint8_t type
#define CLN_WM_SAMPLES              1 // Change how we compute num_proj_ref
#define CLN_MDS0                    1 // Simplified MDS0: chroma-blind, and variance as metric for distortion derivation
#define TUNE_MR                     1 // Tune MR
#define TUNE_M7_2                   1 // Tune M7 (slower M7)
#define CLN_M10_DEPTH_REFINEMENT    1 // Remove the r0-modulation for depth refinement in M10.
#define CLN_UNIFY_MV_TYPE           1 // Use Mv struct instead of IntMv/MV
#define CLN_MOVE_MV_FIELDS          1 // Aggregate MV-related definitions and functions in mv.h - TODO: When removing macros, remove a file
#define CLN_CAND_REF_FRAME          1 // Store ref frames as an array of two refs in ModeDecisionCandidate
#define CLN_MV_IDX                  1 // Store unipred MVs in the 0th index
#define CLN_GET_REF_PIC             1 // Change svt_aom_get_ref_pic_buffer to take the ref frame instead of list idx/ref idx
#define CLN_CAND_INJ                1 // Cleanup cand injection funcs
#define CLN_MOVE_FUNCS              1 // Move block info funcs to block_structures.h to avoid duplicate definitions
#define OPT_NSQ_GEOM_MR_M0_LVL      1 // change the MR/M0 level of the NSQ SEARCH to 1 (for regular/low coeff lvl).
#define FIX_IFS_10BIT               1 // Enable OBMC and inter-intra during IFS search for 10bit
#define FTR_RTC_MODE                1 // Create a new rtc API and use it to enable rtc settings throughout the encoder.
#define CLN_MBMI_IN_CAND            1 // Use BlockModeInfo struct in MD candidates
#define OPT_MRP                     1 // Create a new set of levels for MRP, and tune MRP for M3-M5.
#define CLN_MBMI_IN_BLKSTRUCT       1 // Use BlockModeInfo struct in BlkStruct
#define FIX_NEW_NR_NRST_LIST        1 // Fix list idx in new/near/nearest cand injection
#define OPT_NSQ_SEARCH_LEVELS_2     1 // Develop a new level for NSQ Search (level 19), and set M10 & M9 to that level.
#define CLN_INTER_PRED_FUNC         1 // Cleanup inter prediction functions
#define CLN_INTER_PRED_FUNC_LPD1    1 // Cleanup LPD1 inter prediction functions
#define CLN_INTER_PRED_FUNC_LPD0    1 // Cleanup LPD0 inter prediction functions
#define CLN_MERGE_WM_INTER_PRED     1 // Combine WM pred func with regular inter pred function
#define OPT_ENABLE_COMP_GM          1 // Enable DIFF and WEDGE compound for GM
#define FIX_GM_MOTION               1 // GM cands should be SIMPLE_TRANSLATION and WM info should be obtained from the gm params
#define CLN_COMPOUND_CHECKS         1 // Cleanup compound shortcuts; merge dist_based_ref_prune signals for inter-inter compound types
#define OPT_NSQ_SEARCH_LEVELS_3     1 // Optimize NSQ search levels (level 15 - level 19).
#define OPT_NSQ_GEOM                1 // Remove level 4 and changing M8-M10 level.
#define OPT_NO_GM_IDENTITY          1 // Add GM opt to skip injecting GM IDENTITY cands
#define CLN_UNUSED_SIGS             1 // Remove unused signals
#define OPT_ENABLE_GM_M5            1 // Enable GM in M5
#define CLN_MV_UNIT                 1 // Remove MvUnit struct
#define CLN_REMOVE_MVCAND           1 // Remove MvCandidate struct type b/c redundant to Mv
#define CLN_INJ_NON_SIMPLE_MODES    1 // Move checks for allowed non-simple modes inside inj_non_simple_modes()
#define FIX_OPT_MRP_M4_M5           1 // Fix an onion ring issue in the new MRP changes in M4/M5.
#define CLN_IF_PARAMS               1 // Cleanup unnecessary calls to filter params
#define CLN_MV_ARRAYS               1 // Change the type of arrays that store MVs from int16_t to Mv.
#define CLN_MV_BEST_PRED_FUNC       1 // Simplify the number and type of arguments passed to the function svt_aom_choose_best_av1_mv_pred()
#define CLN_MV_MD_SUBPEL_FUNC       1 // Simplify the number and type of arguments passed to the function md_subpel_search()
#define CLN_MV_ME_MV_XY             1 // Create a new Mv to hold me_mv_x and me_mv_y and use it.
#define FIX_IFS_MDS0                1 // Add interp filter rate to fast cost when IFS is performed at MDS0
#define FIX_IFS_MDS1                1 // Perform IFS at a later MD stage if IFS-MD-stage is bypassed
#define FIX_PME_REF_MV              1 // Fix ref mv in PME search
#define CLN_IFS                     1 // Skip pred in IFS search for fullpel cands; keep pred as valid
#define TUNE_MR_2                   1 // MR Tuning; slowing down MR with good BDR gain.
#define TUNE_REV_TUNE_M5_DIFFS      1 // Reverse two changes in TUNE_M5 for BDR gain: NIC level and Skip INTRA
#define OPT_ALLINTRA                1 // Optimize allintra configuration
#define CLN_REMOVE_LDP              1 // Remove unused SVT_AV1_PRED_LOW_DELAY_P
#define OPT_CDEF_FD2_M5             1 // Change the cdef recon level that M5 uses in FD2 to level 1.
#define FIX_CDEF_MSE                1 // Fix CDEF metric - use the same for luma and chroma
#define OPT_SHUT_TF_LD              1 // Disable TF for LD
#define OPT_NIC_TUNE_LPD1           1 // Adopting a new NIC level in M5 and adopting the pic lpd1 M7 level in M5.
#define OPT_NEW_LD_RPS              1 // New rps for LD
#define TUNE_M0_2                   1 // M0 Tuning; slowing down M0 with good trade-offs and spacing
#define OPT_MDS0_M4                 1 // MDS0 Tuning. M4 takes the M5 level.
#define OPT_CDEF_LEVEL1             1 // Use a more agressive level of zero_fs_cost_bias for decoder cycles gain.
#define OPT_REF_INFO                1 // Opt features that rely on ref info to not use info from higher layers and use more refs
#define TUNE_M1_2                   1 // M1 Tuning; Better spacing
#define FIX_NIC_QP_SCALING          1 // Fix QP scaling for NIC pruning thresholds
#define OPT_ALLINTRA_STILLIMAGE     1 // Tune all-intra coding
#define FIX_STILLIMAGE_HRES         1 // Remove >4K preset restriction for still-image coding
#define OPT_NIC_2_PME_TUNE_DEP_REF  1 // Optimize NIC level 13; Optimize MD_PME level 3; Adopt the Depth Refinement M4 level in M5.
#define CLN_MISC                    1 // Miscellaneous cleanups
#define OPT_NIC_3                   1 // Optimize NIC Level 13 more.
#define TUNE_M2_2                   1 // M2 Tuning; Good speed up.
#define TUNE_M3_2                   1 // M3 Tuning; Good BD-rate gain with small speed up.
#define TUNE_M4_2                   1 // M4 Tuning ; Good trade-offs.
#define OPT_ALLINTRA_STILLIMAGE_2   1 // Tune all-intra coding
#define FIX_LAMBDA                  1 // Fix the input to def_arf_rd_multiplier() to use q instead of qinde
#define FTR_USE_KEY_IF_ALLINTRA     1 // Use key-frame for all-intra coding
#define OPT_CDEF_LEVEL2             1 // Fix the onion ring in cdef_recon_level 2, affects M8 in all FDs and M7 in FD2.
#define OPT_NIC_4                   1 // Tuning NIC M0/M1 levels.
#define TUNE_M5_2                   1 // M5 Tuning; adopt same changes for RA/LD.
#define OPT_NIC_5                   1 // Tuning NIC M7-M10 levels.
#define FIX_NIC_LVL_ONION_RING      1 // Fix the onion ring in NIC levels 14-20.
#define OPT_RTC                     1 // Opt rtc cbr
#define CLN_NIC_LEVELS              1 // Clean NIC levels; Change the NIC level used for PD0 to level 10 (previously named 19).
#define TUNE_M4_LD                  1 // M4 Tuning; same chnages for RA & LD.
#define TUNE_REENABLE_TF_LD_RTC     1 // Re-enable TF for RTC mode
#define TUNE_M10_10BIT              1 // Tuning M10 for 10bit; 8bit is also impacted.
#define TUNE_M9_10BIT               1 // Tuning M9 for 10bit; 8bit is also impacted.
#define OPT_ALLINTRA_STILLIMAGE_3   1 // Tune all-intra coding
#define TUNE_M7_3                   1 // M7 Tuning; Impact RA and LD (common changes).
#define OPT_LD_MEM                  1 // Reduce memory usage (lossless)
#define OPT_LD_MEM_2                1 // Reduce memory usage (lossless)
#define TUNE_NIC_M5_LVL             1 // use more aggressive NIC level for M5 (extra speed-up).
#define TUNE_M8_2                   1 // M8 Tuning; Better trade-offs.
#define FIX_LD_CBR_CRASH            1 // Bypass cyclic_sb_qp_derivation() if sb128
#define OPT_LD_MEM_3                1 // Reduce memory usage (lossless)
#define CLN_4K_1080P_CHECKS         1 // Remove useless 4K/1080p checks
#define CLN_CDEF_LEVEL3             1 // Remove redundant cdef_recon_level 3 (which is equal to level 2).
#define TUNE_M5_3                   1 // M5 Tuning; Extra speed-up.
#define TUNE_M9                     1 // M9 Tuning; Impact RA and LD (common changes).
#define TUNE_M7_4                   1 // M7 Tuning; Impact RA/LD and 8bit/10bit (common changes).
#define CLN_CDEF_SEARCH_LEVEL7      1 // Unify the use of cdef_search_level 7 for M8+ presets.
#define CLN_PIC_LPD1_LEVEL          1 // Unify the pic_lpd1_level settings for M8 and M9.
#define FIX_R2R                     1 // Bypass unpacking of obmc_buff_0 and obmc_buff_1 when generated for 8-bit, and use 10bit for ifs when hybrid
#define CLN_REMOVE_P_SLICE          1 // Remove P_SLICE - fixes bug in OPT_NEW_LD_RPS
#define CLN_VARIANCE_DR             1 // Remove the use of variance for depth removal.
#define TUNE_NEW_M6                 1 // Create a new M6
#define CLN_OBMC_BUILD_PRED         1 // Add a per-plane loop for left-pred, and remove num_planes from the input(s)
#define FIX_R2R_2                   1 // Re-compute obmc_buff_0 and obmc_buff_1 if 10bit compensation, but pre-computation done using 8bit
#define TUNE_M9_SC                  1 // Tuning M9 for SC.
#define TUNE_M8_SC                  1 // Tuning M8 for SC.
#define TUNE_M7_5                   1 // Tuning M7 to fix the M8-M7-M6 spacing.
#define TUNE_M0_3                   1 // Tune M0
#define TUNE_M7_6                   1 // M7 Tuning.
#define TUNE_M1_3                   1 // Tune M1 (still-image, all-intra)
#define TUNE_M3_3                   1 // Tune M3
#define TUNE_M7_SC                  1 // Tuning M7 for SC.
#define TUNE_M6_SC                  1 // Tuning M6 for SC.
#define FIX_R2R_3                   1 // Compute ii data if 10bit compensation, but pre-computation done using 8bit
#define FIX_STARTUP_MG              1 // Do not block stratup-mg if stratup-mg = default (hierarchical_levels)
#define TUNE_M3_SC                  1 // Tuning M3 for SC.
#define TUNE_M5_4                   1 // M5 Tuning.
#define TUNE_LD_RTC                 1 // Tune LD RTC
#define TUNE_VBR                    1 // Adopt the MRP CRF levels in VBR for M5+, and the TPL group + TPL params CRF levels in VBR for M9; Fix the M10 MRP level for LD non-RTC.
#define CLN_REMOVE_PSQ_FEAT         1 // remove psq_cplx_lvl under nsq search controls because it is unused
#define CLN_REMOVE_FORCE_1_CAND     1 // Remove force_1_cand_th under nic pruning controls because it is unused
#define FIX_SSSE_MDS2               1 // Don't skip MDS2 for inter cands if spatial SSE is first used in MDS2
#define TUNE_M2_3                   1 // Tune M2 (still-image, all-intra)
#define TUNE_M1_SC                  1 // Tuning M1 for SC.
#define TUNE_M6_2                   1 // Tune the new M6.
#define TUNE_M8_3                   1 // Tune M8 for better slope.
#define TUNE_M3_4                   1 // Tune M3 for better slope.
#define TUNE_M3_5                   1 // Tune M3 (still-image, all-intra)
#define CLN_CAND_INJ_2              1 // Cleanup cand injection funcs
#define CLN_FULL_LOOP_INPUTS        1 // Cleanup inputs to the full loop
#define CLN_SUBRES_DET              1 // Move subres detector to its own function
#define CLN_TXS_CHECKS              1 // Move checks that change start/end TX depth into get_start_end_tx_depth
#define FIX_TPL_RESULTS_USE         1 // Fix how TPL results are accessed to ensure valid data is used
#define CLN_FUNCS_HEADER            1 // Move stray function declarations to header files
#define TUNE_RTC_USE_LD             1 // If rtc is specified in the CLI, force LD to be used, rather than disabling rtc
#define CLN_ME_DIST_MOD             1 // Limit the use of me_dist_mod to NSQ.
#define CLN_TXS_MIN_SQ_SIZE         1 // Remove the TXS sub-signal min_sq_size as it is always set to 0.
#define CLN_BYPASS_TX_ZCOEFF        1 // Remove the tx_shortcut_level sub-signal bypass_tx_when_zcoeff as it is useless.
#define CLN_INTER_COMP_LVLS         1 // Remove inter_comp_mode level 5, along with sub-signals use_rate and distortion_exit_th.
#define CLN_GMV_UNUSED_SIGS         1 // Remove useless GMV level signals (that are always set to 0).
#define TUNE_M6_SC_2                1 // Tuning M6 (2) For SC.
#define OPT_SC_ME_2                 1 // Re-optimize SC for ME (all presets); activate ME booster to MR-M8 for SC class 1.
// rtc opts
#define OPT_FIFO_MEM                1 // Reduce memory used by fifos
#define CLN_SEG_COUNTS              1 // Remove unnecessary segment counts
#define OPT_PIC_MGR_Q               1 // Reduce the size of the pic manager queue
#define OPT_REF_Q                   1 // Reduce the size of ref queues
#define OPT_PD_REORDER_Q            1 // Reduce the size of PD reorder queue
#define FIX_REST_ONE_SEG_LP1        1 // Use one restoration segment for lp1
#define CLN_REMOVE_IRC_Q            1 // Remove initial_rate_control_reorder_queue because it's not used.  TODO: Remove a file when removing macros
#define CLN_REMOVE_10BIT_FORMAT     1 // Remove unused scs->ten_bit_format
#define CLN_REMOVE_DATA_LL          1 // Remove linked list in pcs supposedly for meta data, but that is always NULL
#define OPT_PACK_Q                  1 // Reduce the size of packetization reorder queue
#define OPT_OUTPUT_STREAM_Q         1 // Reduce the size of the output stream fifo
#define FTR_RTC_M11_M12             1 // Allow Preset 11 and 12 if rtc
#define OPT_RTC_B8                  1 // Add a high-level control for 8x8 block
#if OPT_RTC_B8
#define FTR_RTC_GEOM                1
#define FTR_RTC_MI_GRID             1
#endif
#define OPT_RTC_M10                 1 // Speed-up M10
#if OPT_RTC_M10
#define OPT_RTC_RDOQ                1
#define OPT_RTC_TXT                 1
#define OPT_RTC_PME                 1
#define OPT_RTC_MRP                 1
#define OPT_RTC_SUBPEL              1
#define OPT_RTC_INTRA               1
#endif
#define TUNE_RTC_M10                1 // Tuning M10 for more speed.
#define CLN_EC                      1 // Update ec code to match libaom
#define OPT_ENCDEC_MEM              1 // Remove enc dec buffers when encdec is bypassed; make buffers sb-size dependent
#define TUNE_RTC_M11                1 // Tuning M11 for Extra speed.
#define OPT_CR_FLOW_CHANGE          1 // Update control flow for cyclic refresh logic (lossless)
#if OPT_CR_FLOW_CHANGE
#define OPT_CR_ESTIMATE             1 // Take the cycle-refresh modulation into account when deriving the projected frame size to improve the accuracy of the correction factor
#define OPT_CR_LIMIT                1 // Update the equation used in the derivation of the adjustment limit
#define OPT_CR_ADJUST               1 // Adaptively derive the percent refresh and the rate-ratio-delta-qp based on the overshoot or undershoot of the target in the current frame
#define OPT_RATE_BOOST_FAC          1 // Use the distortion per segment to modulate rate_boost_fac
#define OPT_LAMBDA                  1 // Opt lambda modulation
#define OPT_UPDATE_GET_BIT          1 // Update the bpmb enumerato derivation
#define OPT_CR_CAP                  1 // Remove delta-qp capping
#endif
#define OPT_PIC_BUFFS               1 // Reduce number of pic buffs when few refs are used
#define TUNE_RTC_M11_2              1 // add more speed-up changes to M11.
#define OPT_SHUT_COEFF_LVL          1 // Shut coeff-level if rtc
#define CLN_AVG_ME_DISTORTION       1 // Clean avg me-dist
#define OPT_M12                     1 // OPT M12
#if OPT_M12
#define SHUT_TF                     1
#define OPT_DR                      1
#endif
#define FTR_ADD_FLAT_IPP            1 // Add flat IPP rps structure that uses only the previous pic as ref
#define OPT_CDEF_LVL8               1 // Opt CDEF lvl 8 for M11 rtc
#define OPT_LD_CQP_MEM              1 // Reduce number of pics buffs for LD CQP when few refs are used
#define OPT_MEM_FLAT_IPP            1 // Reduce pic buffs for flat-ipp structure
#define TUNE_RTC_M11_3              1 // Adopt changes that improve the M11 slope.
#define TUNE_RTC_M10_2              1 // Adopt changes that yiled better trade-offs for M10.
#define OPT_I_CHECK                 1 // Give the intra frame an ON (more conservative) level instead of OFF in M11.
#define OPT_BASE_TO_I_CHECK         1 // Change base checks into i-checks in M12, in preperation for the new flat IPP structure.
#define TUNE_RTC_M11_4              1 // Adopt changes that yield good BD-rate gain with acceptable speed loss to M11.
#define OPT_CDEF_UV_FROM_Y          1 // Allow CDEF UV filters to be taken from Y
#define TUNE_RTC_M10_3              1 // Tuning M10 for BD-rate gain and speedup.
#define TUNE_RTC_M11_5              1 // Obtain better trade-offs for M11.
#define OPT_CBR_FLAT                1 // Opt CBR for flat
#define TUNE_RTC_M11_6              1 // Adopt aggressive NIC level in M11.
#define TUNE_RTC_M12                1 // Adopt a change that yields a good BD-rate gain with not much slow down.
#define TUNE_RTC_M12_2              1 // Adopt more aggressive lpd0 (NSC) in M12 and M11, md_subpel, and me_sa (NSC) for M12 only.
#define TUNE_RTC_M9                 1 // Adopt changes with better trade-offs to M9.
#define TUNE_RTC_M8                 1 // Adopt changes with better trade-offs to M8.
#define FTR_RTC_FLAT                1 // Use --hierarchical-levels 0 to enable a flat prediction structure
#define TUNE_RTC_M10_4              1 // New M10 RTC NSC adoption.
#define TUNE_RTC_PSNR_M7            1 // PSNR focused M7 tuning.
#define TUNE_RTC_PSNR_M8            1 // PSNR focused M8 tuning.
#define TUNE_RTC_PSNR_M9            1 // PSNR focused M9 tuning.
#define TUNE_RTC_PSNR_M10           1 // PSNR focused M10 tuning.
#define TUNE_RTC_PSNR_M11           1 // PSNR focused M11 tuning.
#define TUNE_RTC_PSNR_M12           1 // PSNR focused M12 tuning.
#define CLN_REMOVE_LPD0_SHIFT       1 // Remove ld_enhanced_base_frame pic_lpd0 modulator for M7-M11.
#define OPT_LAMBDA_RTC              1 // SB-based lambda modulation using exclusively ME delta-QP, and lambda boost for kf
#define OPT_RTC_PSNR_M12            1 // Speeding up M12
#define FTR_SFRAME_RA               1 // Enable S-Frame feature in Random Access Mode
#if FTR_SFRAME_RA
#define FIX_NEAREST_ARF_RA          1 // Fix issue of S-Frame nearest ARF mode finding wrong ARF position in RA mode, since frames in decode order
#define FIX_SFRAME_PRUNE_REF0       1 // Prune RefList0 when forward and backward both refer to S-Frame in RA mode, [LAST/LAST2/LAST3/GOLD] ref frame MVs to S-Frame are in reversed direction
#define EN_SFRAME_E2E_TEST          0 // Enable S-Frame test cases in E2E test, default is disabled
#endif
#define FIX_SFRAME_ORDER_HINT       1 // Fix issue of dpd_order_hint is not relative with key position

//FOR DEBUGGING - Do not remove
#define OPT_LD_LATENCY2         1 // Latency optimization for low delay - to keep the Macro for backwards testing until 3.0
#define LOG_ENC_DONE            0 // log encoder job one
#define DEBUG_TPL               0 // Prints to debug TPL
#define DETAILED_FRAME_OUTPUT   0 // Prints detailed frame output from the library for debugging
#if CLN_SEG_COUNTS
#define DEBUG_BUFFERS           0 // Print process count and segments info
#endif
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
#if !CLN_FUNCS_HEADER
#define DEBUG_UPSCALING         0
#endif
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
