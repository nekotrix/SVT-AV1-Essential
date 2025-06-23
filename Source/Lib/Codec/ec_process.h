/*
* Copyright(c) 2019 Intel Corporation
* Copyright (c) 2019, Alliance for Open Media. All rights reserved
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#ifndef EbEntropyCodingProcess_h
#define EbEntropyCodingProcess_h

#include "definitions.h"

#include "sys_resource_manager.h"
#include "pic_buffer_desc.h"
#include "enc_inter_prediction.h"
#include "entropy_coding.h"
#include "coding_unit.h"
#include "object.h"

/**************************************
 * Enc Dec Context
 **************************************/
typedef struct EntropyCodingContext {
    EbDctor dctor;
    EbFifo *enc_dec_input_fifo_ptr;
    EbFifo *entropy_coding_output_fifo_ptr; // to packetization
#if !OPT_FIFO_MEM
    EbFifo *rate_control_output_fifo_ptr; // feedback to rate control
#endif
    uint32_t sb_total_count;
    // Coding Unit Workspace---------------------------
    EbPictureBufferDesc *coeff_buffer_sb; //Used to hold quantized coeff for one TB in EncPass.

    //  Context Variables---------------------------------
    BlkStruct *blk_ptr;
    //const CodedBlockStats           *cu_stats;
    uint32_t blk_index;
    uint8_t  cu_depth;
    uint32_t cu_size;
    uint32_t cu_size_log2;
    uint32_t blk_org_x;
    uint32_t blk_org_y;
    uint32_t sb_origin_x;
    uint32_t sb_origin_y;
#if !OPT_LD_MEM_3
    uint32_t pu_itr;
#endif
    uint32_t pu_origin_x;
    uint32_t pu_origin_y;
    uint32_t pu_width;
    uint32_t pu_height;
#if !CLN_UNUSED_SIGS
    MvUnit mv_unit;
#endif

    uint32_t txb_itr;
    uint32_t txb_origin_x;
    uint32_t txb_origin_y;
    uint32_t txb_size;

    // MCP Context
    bool        is_16bit; //enable 10 bit encode in CL
    int32_t     coded_area_sb;
    int32_t     coded_area_sb_uv;
    TOKENEXTRA *tok;
    MbModeInfo *mbmi;
#if OPT_LD_MEM_2
    /*!
     * cdef_transmitted[i] is true if CDEF strength for ith CDEF unit in the
     * current superblock has already been read from (decoder) / written to
     * (encoder) the bitstream; and false otherwise.
     * More detail:
     * 1. CDEF strength is transmitted only once per CDEF unit, in the 1st
     * non-skip coding block. So, we need this array to keep track of whether CDEF
     * strengths for the given CDEF units have been transmitted yet or not.
     * 2. Superblock size can be either 128x128 or 64x64, but CDEF unit size is
     * fixed to be 64x64. So, there may be 4 CDEF units within a superblock (if
     * superblock size is 128x128). Hence the array size is 4.
     * 3. In the current implementation, CDEF strength for this CDEF unit is
     * stored in the MB_MODE_INFO of the 1st block in this CDEF unit (inside
     * pcs->mi_grid_base).
     */
    bool cdef_transmitted[4];

    /**
   * \name Default values for the two restoration filters for each plane.
   * Default values for the two restoration filters for each plane.
   * These values are used as reference values when writing the bitstream. That
   * is, we transmit the delta between the actual values in
   * pcs->rst_info[plane].unit_info[runit_idx] and these reference values.
   */
    WienerInfo  wiener_info[MAX_MB_PLANE];
    SgrprojInfo sgrproj_info[MAX_MB_PLANE];
#endif
} EntropyCodingContext;

/**************************************
 * Extern Function Declarations
 **************************************/
#if OPT_FIFO_MEM
EbErrorType svt_aom_entropy_coding_context_ctor(EbThreadContext *thread_ctx, const EbEncHandle *enc_handle_ptr,
                                                int index);
#else
extern EbErrorType svt_aom_entropy_coding_context_ctor(EbThreadContext *thread_ctx, const EbEncHandle *enc_handle_ptr,
                                                       int index, int rate_control_index);
#endif

extern void *svt_aom_entropy_coding_kernel(void *input_ptr);

#endif // EbEntropyCodingProcess_h
