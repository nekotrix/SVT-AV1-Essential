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

#ifndef EbBlockStructures_h
#define EbBlockStructures_h

#include "definitions.h"
#include "segmentation_params.h"
#include "av1_structs.h"
#if CLN_MOVE_MV_FIELDS
#include "mv.h"
#else
#if CLN_UNIFY_MV_TYPE
#include "motion_vector_unit.h"
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_TILE_WIDTH (4096) // Max Tile width in pixels
#define MAX_TILE_AREA (4096 * 2304) // Maximum tile area in pixels

#if !CLN_UNIFY_MV_TYPE
typedef struct MV {
    int16_t row;
    int16_t col;
} MV;

typedef union IntMv {
    uint32_t as_int;
    MV       as_mv;
} IntMv; /* facilitates faster equality tests and copies */
typedef struct mv32 {
    int32_t row;
    int32_t col;
} MV32;
#endif
#if !CLN_MOVE_MV_FIELDS
#define GET_MV_RAWPEL(x) (((x) + 3 + ((x) >= 0)) >> 3)
#define GET_MV_SUBPEL(x) ((x) * 8)

// The motion vector in units of full pixel
typedef struct fullpel_mv {
    int16_t row;
    int16_t col;
} FULLPEL_MV;
#if CLN_UNIFY_MV_TYPE
static INLINE int is_zero_mv(const Mv *mv) { return *((const uint32_t *)mv) == 0; }

static AOM_INLINE Mv get_fullmv_from_mv(const Mv *subpel_mv) {
    const Mv full_mv = {{(int16_t)GET_MV_RAWPEL(subpel_mv->x), (int16_t)GET_MV_RAWPEL(subpel_mv->y)}};
    return full_mv;
}

static AOM_INLINE Mv get_mv_from_fullmv(const Mv *full_mv) {
    const Mv subpel_mv = {{(int16_t)GET_MV_SUBPEL(full_mv->x), (int16_t)GET_MV_SUBPEL(full_mv->y)}};
    return subpel_mv;
}
#else
static const MV         kZeroMv     = {0, 0};
static const FULLPEL_MV kZeroFullMv = {0, 0};
static INLINE int       is_zero_mv(const MV *mv) { return *((const uint32_t *)mv) == 0; }

static AOM_INLINE FULLPEL_MV get_fullmv_from_mv(const MV *subpel_mv) {
    const FULLPEL_MV full_mv = {(int16_t)GET_MV_RAWPEL(subpel_mv->row), (int16_t)GET_MV_RAWPEL(subpel_mv->col)};
    return full_mv;
}

static AOM_INLINE MV get_mv_from_fullmv(const FULLPEL_MV *full_mv) {
    const MV subpel_mv = {(int16_t)GET_MV_SUBPEL(full_mv->row), (int16_t)GET_MV_SUBPEL(full_mv->col)};
    return subpel_mv;
}
#endif
#endif
typedef struct OisMbResults {
    int64_t intra_cost;
    int32_t intra_mode;
} OisMbResults;
#if !CLN_MOVE_MV_FIELDS
#if CLN_UNIFY_MV_TYPE
typedef struct CandidateMv {
    Mv      this_mv;
    Mv      comp_mv;
    int32_t weight;
} CandidateMv;
#else
typedef struct CandidateMv {
    IntMv   this_mv;
    IntMv   comp_mv;
    int32_t weight;
} CandidateMv;
#endif
#endif

typedef struct TileInfo {
    int32_t mi_row_start, mi_row_end;
    int32_t mi_col_start, mi_col_end;
    int32_t tg_horz_boundary;
    int32_t tile_row;
    int32_t tile_col;
    int32_t tile_rs_index; //tile index in raster order
} TileInfo;

#if !CLN_REMOVE_DEC_STRUCT
#define INTER_TX_SIZE_BUF_LEN 16
#define TXK_TYPE_BUF_LEN 64
#endif

typedef struct FilterIntraModeInfo {
    /*!< Specifies the type of intra filtering, and can represent any of the following:
         * FILTER_DC_PRED, FILTER_V_PRED, FILTER_H_PRED, FILTER_D157_PRED, FILTER_PAETH_PRED */
    FilterIntraMode filter_intra_mode;

    /*!< This bit specifies whether or not intra filtering can be used. */
    uint8_t use_filter_intra;
} FilterIntraModeInfo_t;

typedef struct InterIntraModeParams {
    /*!< Specifies the type of intra prediction to be used */
    InterIntraMode interintra_mode;

    /*!< equal to 1 specifies that wedge blending should be used.
            * wedge_interintra equal to 0 specifies that intra blending should be used. */
    uint8_t wedge_interintra;

    /*!< Used to derive the direction and offset of the wedge mask used during blending. */
    uint8_t interintra_wedge_index;

    /*!< Specifies the sign of the wedge blend. */
    // int interintra_wedge_sign; Always 0
} InterIntraModeParams;
#if CLN_MOVE_FIELDS_MBMI
typedef struct BlockModeInfo {
    /*! \brief The prediction mode used */
    PredictionMode mode;
    /*! \brief The UV mode when intra is used */
    UvPredictionMode uv_mode; // Only for INTRA blocks

    /*****************************************************************************
   * \name Inter Mode Info
   ****************************************************************************/
    /**@{*/
#if CLN_MV_IDX
    /*! \brief The motion vectors used by the current inter mode. Unipred MV stored
   in idx 0.*/
    Mv mv[2];
#else
    /*! \brief The motion vectors used by the current inter mode */
#if CLN_UNIFY_MV_TYPE
    Mv mv[2];
#else
    IntMv mv[2];
#endif
#endif
    /*! \brief The reference frames for the MV */
    MvReferenceFrame ref_frame[2];
    /*! \brief Filter used in subpel interpolation. */
    uint32_t interp_filters;
    /*! \brief Struct that stores the data used in interinter compound mode. */
    InterInterCompoundData interinter_comp;
    /*! \brief The motion mode used by the inter prediction. */
    MotionMode motion_mode;
    /*! \brief Number of samples used by warp causal */
#if CLN_WM_CTRLS
    uint8_t num_proj_ref;
#else
    uint16_t num_proj_ref;
#endif
    /*! \brief The type of intra mode used by inter-intra */
    InterIntraMode interintra_mode;
    /*! \brief The type of wedge used in interintra mode. */
#if CLN_MBMI_IN_CAND
    int8_t interintra_wedge_index;
#else
    uint8_t interintra_wedge_index;
#endif

    /*****************************************************************************
     * \name Intra Mode Info
     ****************************************************************************/
    /**@{*/
    /*! \brief Directional mode delta: the angle is base angle + (angle_delta *
      * step). */
    int8_t angle_delta[PLANE_TYPES];
    /*! \brief The type of filter intra mode used (if applicable). */
    uint8_t filter_intra_mode;
    /*! \brief Chroma from Luma: Joint sign of alpha Cb and alpha Cr */
    uint8_t cfl_alpha_signs;
    /*! \brief Chroma from Luma: Index of the alpha Cb and alpha Cr combination */
    uint8_t cfl_alpha_idx;

    uint8_t tx_depth;
    uint8_t is_interintra_used : 1;
    uint8_t use_wedge_interintra : 1;
    /*! \brief Indicates if masked compound is used(1) or not (0). */
    uint8_t comp_group_idx : 1;
    /*!< 0 indicates that a distance based weighted scheme should be used for blending.
     *   1 indicates that the averaging scheme should be used for blending.*/
    uint8_t compound_idx : 1;
    // possible values: 0,1; skip coeff only. as defined in section 6.10.11 of the av1 text
    uint8_t skip : 1;

    /*!< 1 indicates that this block will use some default settings and skip mode info.
     * 0 indicates that the mode info is not skipped. */
    // possible values: 0,1; skip mode_info + coeff. as defined in section 6.10.10 of the av1 text
    uint8_t skip_mode : 1;
    /*! \brief Whether intrabc is used. */
    uint8_t use_intrabc : 1;
} BlockModeInfo;
#else
#if CLN_REMOVE_DEC_STRUCT
typedef struct BlockModeInfo {
    // Only for INTER blocks
    IntMv mv[2];

    /*!< Specifies the type of filter used in inter prediction. Values 0..3 are allowed
        * with the same interpretation as for interpolation_filter. One filter type is specified
        * for the vertical filter direction and one for the horizontal filter direction.*/
    uint32_t interp_filters;

    MvReferenceFrame ref_frame[2]; // Only for INTER blocks
    BlockSize        bsize;
    PredictionMode   mode;
    PartitionType    partition;
    UvPredictionMode uv_mode; // Only for INTRA blocks

    uint8_t tx_depth;
    uint8_t comp_group_idx : 1; // possible values: 0,1
    /*!< 0 indicates that a distance based weighted scheme should be used for blending.
     *   1 indicates that the averaging scheme should be used for blending.*/
    uint8_t compound_idx : 1; // possible values: 0,1
    // possible values: 0,1; skip coeff only. as defined in section 6.10.11 of the av1 text
    uint8_t skip : 1;

    /*!< 1 indicates that this block will use some default settings and skip mode info.
     * 0 indicates that the mode info is not skipped. */
    // possible values: 0,1; skip mode_info + coeff. as defined in section 6.10.10 of the av1 text
    uint8_t skip_mode : 1;
    uint8_t use_intrabc : 1; // possible values: 0,1

    uint8_t segment_id;
} BlockModeInfo;
#else
typedef struct BlockModeInfoEnc {
    // Only for INTER blocks
    IntMv mv[2];

    /*!< Specifies the type of filter used in inter prediction. Values 0..3 are allowed
        * with the same interpretation as for interpolation_filter. One filter type is specified
        * for the vertical filter direction and one for the horizontal filter direction.*/
    uint32_t interp_filters;

    MvReferenceFrame ref_frame[2]; // Only for INTER blocks
    BlockSize        bsize;
    PredictionMode   mode;
    PartitionType    partition;
    UvPredictionMode uv_mode; // Only for INTRA blocks

    uint8_t tx_depth;
    uint8_t comp_group_idx : 1; // possible values: 0,1
    /*!< 0 indicates that a distance based weighted scheme should be used for blending.
     *   1 indicates that the averaging scheme should be used for blending.*/
    uint8_t compound_idx : 1; // possible values: 0,1
    // possible values: 0,1; skip coeff only. as defined in section 6.10.11 of the av1 text
    uint8_t skip : 1;

    /*!< 1 indicates that this block will use some default settings and skip mode info.
     * 0 indicates that the mode info is not skipped. */
    // possible values: 0,1; skip mode_info + coeff. as defined in section 6.10.10 of the av1 text
    uint8_t skip_mode : 1;
    uint8_t use_intrabc : 1; // possible values: 0,1

    uint8_t segment_id;

#if MODE_INFO_DBG
    int32_t mi_row;
    int32_t mi_col;
#endif
} BlockModeInfoEnc;

typedef struct BlockModeInfo {
    // Common for both INTER and INTRA blocks
    BlockSize      bsize;
    PredictionMode mode;
    int8_t         skip;

    PartitionType partition;

    /*!< 1 indicates that this block will use some default settings and skip mode info.
            * 0 indicates that the mode info is not skipped. */
    int8_t skip_mode;

    /*!< Specifies which segment is associated with the current intra block being decoded. */
    int8_t segment_id;

    /*!< Equal to 1 specifies that the segment_id is taken from the segmentation map. */
    int8_t seg_id_predicted;

    /*!< For Lossy mode   : Specifies number of TUs in a block for each plane
             For Lossless mode: Specifies number of TUs for a block of size other than
                                128x128, 128x64, 64x128 and 64x64 - computed based on blocksize */
    uint8_t num_tus[MAX_MB_PLANE - 1];

    /*!< Offset of first transform info from strat of SB pointer for each plane */
    uint16_t first_txb_offset[MAX_MB_PLANE - 1];

    // Only for INTRA blocks
    UvPredictionMode uv_mode;

    uint8_t use_intrabc;

    // Only for INTER blocks

    MvReferenceFrame ref_frame[2];
    IntMv            mv[2];

    uint16_t ref_mv_idx;

    // interinter members

    InterIntraModeParams interintra_mode_params;

    /*!< Specifies the type of motion compensation to perform. */
    MotionMode motion_mode;

    InterIntraMode is_inter_intra;

    /*!< 0 indicates that a distance based weighted scheme should be used for blending.
         *   1 indicates that the averaging scheme should be used for blending.*/
    uint8_t compound_idx;

    InterInterCompoundData inter_inter_compound;
    FilterIntraModeInfo_t  filter_intra_mode_info;

    /*!< Specifies how the motion vector used by inter prediction is obtained when using compound prediction. */
    uint8_t compound_mode;

    /*!< Specifies the type of filter used in inter prediction. Values 0..3 are allowed
        * with the same interpretation as for interpolation_filter. One filter type is specified
        * for the vertical filter direction and one for the horizontal filter direction.*/
    uint32_t interp_filters;

    /*!< Index of the alpha Cb and alpha Cr combination */
    uint8_t cfl_alpha_idx;

    /*!< Contains the sign of the alpha values for U and V packed together into a single syntax element. */
    uint8_t cfl_alpha_signs;

    /*!< The actual prediction angle is the base angle + (angle_delta * step). */
    int8_t angle_delta[PLANE_TYPES];

    // Number of base colors for Y (0) and UV (1)
    uint8_t palette_size[MAX_MB_PLANE - 1];

    /*mi_row & mi_col wrt a super block*/
    int8_t mi_row_in_sb;
    int8_t mi_col_in_sb;

#if MODE_INFO_DBG
    int32_t mi_row;
    int32_t mi_col;
#endif
} BlockModeInfo;
#endif
#endif
typedef struct MbModeInfo {
#if !CLN_MOVE_FIELDS_MBMI
#if CONFIG_RD_DEBUG
    RD_STATS rd_stats;
    int32_t  mi_row;
    int32_t  mi_col;
#endif
#endif
#if CLN_REMOVE_DEC_STRUCT
    BlockModeInfo block_mi;
#else
    BlockModeInfoEnc block_mi;
#endif
#if CLN_MOVE_FIELDS_MBMI
    BlockSize     bsize;
    PartitionType partition;
    uint8_t       segment_id;
#endif
    PaletteLumaModeInfo palette_mode_info;
    int8_t              cdef_strength;
} MbModeInfo;

#if CLN_MOVE_FUNCS
static AOM_INLINE int has_second_ref(const BlockModeInfo *block_mi) { return block_mi->ref_frame[1] > INTRA_FRAME; }

static AOM_INLINE int has_uni_comp_refs(const BlockModeInfo *block_mi) {
    return has_second_ref(block_mi) &&
        (!((block_mi->ref_frame[0] >= BWDREF_FRAME) ^ (block_mi->ref_frame[1] >= BWDREF_FRAME)));
}

static AOM_INLINE int is_intrabc_block(const BlockModeInfo *block_mi) { return block_mi->use_intrabc; }

static AOM_INLINE int is_inter_block(const BlockModeInfo *block_mi) {
    return is_intrabc_block(block_mi) || block_mi->ref_frame[0] > INTRA_FRAME;
}
#endif
void svt_av1_tile_set_col(TileInfo *tile, const TilesInfo *tiles_info, int32_t mi_cols, int col);
void svt_av1_tile_set_row(TileInfo *tile, TilesInfo *tiles_info, int32_t mi_rows, int row);

static INLINE int32_t tile_log2(int32_t blk_size, int32_t target) {
    int32_t k;
    for (k = 0; (blk_size << k) < target; k++) {}
    return k;
}

#ifdef __cplusplus
}
#endif
#endif // EbBlockStructures_h
