/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AOM_AV1_COMMON_MV_H_
#define AOM_AV1_COMMON_MV_H_

#include "definitions.h"
#if !CLN_MOVE_MV_FIELDS
#include "block_structures.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if CLN_MOVE_MV_FIELDS
#define INVALID_MV 0x80008000
#endif
#if CLN_UNIFY_MV_TYPE
#define CHECK_MV_EQUAL(mv1, mv2) (((mv1).y == (mv2).y) && ((mv1).x == (mv2).x))
#else
#define MARK_MV_INVALID(mv) \
    do { ((int_mv*)(mv))->as_int = INVALID_MV; } while (0);
#define CHECK_MV_EQUAL(x, y) (((x).row == (y).row) && ((x).col == (y).col))

typedef union int_mv {
    uint32_t   as_int;
    MV         as_mv;
    FULLPEL_MV as_fullmv;
} int_mv; /* facilitates faster equality tests and copies */
#endif

// The mv limit for fullpel mvs
typedef struct {
    int col_min;
    int col_max;
    int row_min;
    int row_max;
} FullMvLimits;

// The mv limit for subpel mvs
typedef struct {
    int col_min;
    int col_max;
    int row_min;
    int row_max;
} SubpelMvLimits;

#if CLN_MOVE_MV_FIELDS
#pragma pack(push, 1)
typedef union Mv {
    struct {
        int16_t x;
        int16_t y;
    };
    uint32_t as_int; /* facilitates faster equality tests and copies */
} Mv;
#pragma pack(pop)

#if !CLN_MV_UNIT
typedef struct MvUnit {
    Mv      mv[MAX_NUM_OF_REF_PIC_LIST];
    uint8_t pred_direction;
} MvUnit;
#endif

typedef struct CandidateMv {
    Mv      this_mv;
    Mv      comp_mv;
    int32_t weight;
} CandidateMv;

#define GET_MV_RAWPEL(x) (((x) + 3 + ((x) >= 0)) >> 3)
#define GET_MV_SUBPEL(x) ((x) * 8)

static AOM_INLINE Mv get_fullmv_from_mv(const Mv* subpel_mv) {
    const Mv full_mv = {{(int16_t)GET_MV_RAWPEL(subpel_mv->x), (int16_t)GET_MV_RAWPEL(subpel_mv->y)}};
    return full_mv;
}

static AOM_INLINE Mv get_mv_from_fullmv(const Mv* full_mv) {
    const Mv subpel_mv = {{(int16_t)GET_MV_SUBPEL(full_mv->x), (int16_t)GET_MV_SUBPEL(full_mv->y)}};
    return subpel_mv;
}
static INLINE void clamp_mv(Mv* mv, int32_t min_col, int32_t max_col, int32_t min_row, int32_t max_row) {
    mv->x = (int16_t)clamp(mv->x, min_col, max_col);
    mv->y = (int16_t)clamp(mv->y, min_row, max_row);
}
#endif
#ifdef __cplusplus
} // extern "C"
#endif

#endif // AOM_AV1_COMMON_MV_H_
