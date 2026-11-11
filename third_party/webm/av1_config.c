/*
 * Copyright (c) 2018, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "third_party/webm/av1_config.h"

// ---------------------------------------------------------------------------
// Minimal bitreader
// ---------------------------------------------------------------------------

typedef struct {
    const uint8_t *buf;
    const uint8_t *end;
    uint32_t       bit_offset;
    int            error;
} BitReader;

static void br_init(BitReader *br, const uint8_t *buf, size_t sz) {
    br->buf        = buf;
    br->end        = buf + sz;
    br->bit_offset = 0;
    br->error      = 0;
}

static uint32_t br_read_bit(BitReader *br) {
    if (br->error) return 0;
    size_t   byte_idx = br->bit_offset >> 3;
    uint32_t bit_idx  = 7 - (br->bit_offset & 7);
    if (br->buf + byte_idx >= br->end) {
        br->error = 1;
        return 0;
    }
    br->bit_offset++;
    return (br->buf[byte_idx] >> bit_idx) & 1;
}

static uint32_t br_read_bits(BitReader *br, int n) {
    uint32_t val = 0;
    for (int i = 0; i < n; i++)
        val = (val << 1) | br_read_bit(br);
    return val;
}

// Read UVLC (used for num_ticks_per_picture_minus_1 in timing_info)
static uint32_t br_read_uvlc(BitReader *br) {
    int leading_zeros = 0;
    while (leading_zeros < 32 && !br_read_bit(br))
        leading_zeros++;
    if (leading_zeros == 32) return UINT32_MAX;
    return (1u << leading_zeros) - 1 + br_read_bits(br, leading_zeros);
}

// ---------------------------------------------------------------------------
// OBU header parsing
// ---------------------------------------------------------------------------

// Returns the total byte length of the OBU header (1 or 2 bytes),
// the obu_type, and advances past the LEB128 size field.
// Sets *payload_sz to the size of the OBU payload, or SIZE_MAX if no size field.
static int parse_obu_header(const uint8_t *buf, size_t buf_sz,
                             int *obu_type, size_t *header_sz,
                             size_t *payload_sz) {
    if (buf_sz == 0) return -1;

    uint8_t header_byte   = buf[0];
    if (header_byte & 0x80) return -1; // forbidden bit

    *obu_type              = (header_byte >> 3) & 0xF;
    int extension_flag     = (header_byte >> 2) & 0x1;
    int has_size_field     = (header_byte >> 1) & 0x1;

    *header_sz = 1 + (extension_flag ? 1 : 0);
    if (*header_sz > buf_sz) return -1;

    if (has_size_field) {
        size_t leb_val  = 0;
        size_t leb_bytes = 0;
        for (int i = 0; i < 8; i++) {
            size_t idx = *header_sz + i;
            if (idx >= buf_sz) return -1;
            uint8_t b  = buf[idx];
            leb_val   |= (size_t)(b & 0x7F) << (i * 7);
            leb_bytes++;
            if (!(b & 0x80)) break;
        }
        *header_sz  += leb_bytes;
        *payload_sz  = leb_val;
    } else {
        *payload_sz = buf_sz - *header_sz;
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Sequence header sub-parsers
// ---------------------------------------------------------------------------

static int skip_timing_info(BitReader *br) {
    br_read_bits(br, 32); // num_units_in_display_tick
    br_read_bits(br, 32); // time_scale
    if (br_read_bit(br))  // equal_picture_interval
        br_read_uvlc(br); // num_ticks_per_picture_minus_1
    return br->error ? -1 : 0;
}

static int skip_decoder_model_info(BitReader *br, int *buf_delay_len) {
    *buf_delay_len = (int)br_read_bits(br, 5) + 1;
    br_read_bits(br, 32); // num_units_in_decoding_tick
    br_read_bits(br, 5);  // buffer_removal_time_length_minus_1
    br_read_bits(br, 5);  // frame_presentation_time_length_minus_1
    return br->error ? -1 : 0;
}

static int skip_operating_parameters_info(BitReader *br, int buf_delay_len) {
    br_read_bits(br, buf_delay_len); // decoder_buffer_delay
    br_read_bits(br, buf_delay_len); // encoder_buffer_delay
    br_read_bit(br);                 // low_delay_mode_flag
    return br->error ? -1 : 0;
}

static int parse_color_config(BitReader *br, Av1Config *config) {
    config->high_bitdepth = br_read_bit(br);

    int bit_depth = 8;
    if (config->seq_profile == 2 && config->high_bitdepth) {
        config->twelve_bit = br_read_bit(br);
        bit_depth = config->twelve_bit ? 12 : 10;
    } else {
        bit_depth = config->high_bitdepth ? 10 : 8;
    }

    if (config->seq_profile != 1)
        config->monochrome = br_read_bit(br);

    // CICP constants (from AV1 spec, no aom headers needed)
    const int CP_BT_709   = 1;
    const int TC_SRGB     = 13;
    const int MC_IDENTITY = 0;

    int color_primaries          = 2; // CP_UNSPECIFIED
    int transfer_characteristics = 2; // TC_UNSPECIFIED
    int matrix_coefficients      = 2; // MC_UNSPECIFIED

    if (br_read_bit(br)) { // color_description_present_flag
        color_primaries          = (int)br_read_bits(br, 8);
        transfer_characteristics = (int)br_read_bits(br, 8);
        matrix_coefficients      = (int)br_read_bits(br, 8);
    }

    if (config->monochrome) {
        br_read_bit(br); // color_range
        config->chroma_subsampling_x = 1;
        config->chroma_subsampling_y = 1;
    } else if (color_primaries          == CP_BT_709 &&
               transfer_characteristics == TC_SRGB   &&
               matrix_coefficients      == MC_IDENTITY) {
        config->chroma_subsampling_x = 0;
        config->chroma_subsampling_y = 0;
    } else {
        br_read_bit(br); // color_range
        if (config->seq_profile == 0) {
            config->chroma_subsampling_x = 1;
            config->chroma_subsampling_y = 1;
        } else if (config->seq_profile == 1) {
            config->chroma_subsampling_x = 0;
            config->chroma_subsampling_y = 0;
        } else {
            if (bit_depth == 12) {
                config->chroma_subsampling_x = br_read_bit(br);
                if (config->chroma_subsampling_x)
                    config->chroma_subsampling_y = br_read_bit(br);
            } else {
                config->chroma_subsampling_x = 1;
                config->chroma_subsampling_y = 0;
            }
        }
        if (config->chroma_subsampling_x && config->chroma_subsampling_y)
            config->chroma_sample_position = (uint8_t)br_read_bits(br, 2);
    }

    br_read_bit(br); // separate_uv_delta_q
    return br->error ? -1 : 0;
}

static int parse_sequence_header(const uint8_t *buf, size_t sz, Av1Config *config) {
    BitReader br_inst;
    BitReader *br = &br_inst;
    br_init(br, buf, sz);

    config->seq_profile = (uint8_t)br_read_bits(br, 3);
    int still_picture                = (int)br_read_bit(br);
    int reduced_still_picture_header = (int)br_read_bit(br);

    if (reduced_still_picture_header) {
        config->seq_level_idx_0 = (uint8_t)br_read_bits(br, 5);
        config->seq_tier_0      = 0;
    } else {
        int timing_info_present = (int)br_read_bit(br);
        int has_decoder_model   = 0;
        int buf_delay_len       = 0;

        if (timing_info_present) {
            if (skip_timing_info(br) != 0) return -1;
            has_decoder_model = (int)br_read_bit(br);
            if (has_decoder_model)
                if (skip_decoder_model_info(br, &buf_delay_len) != 0) return -1;
        }

        config->initial_presentation_delay_present = (uint8_t)br_read_bit(br);

        uint32_t op_count = br_read_bits(br, 5) + 1; // operating_points_cnt_minus_1
        for (uint32_t i = 0; i < op_count; i++) {
            br_read_bits(br, 12); // operating_point_idc
            uint8_t seq_level_idx = (uint8_t)br_read_bits(br, 5);
            uint8_t seq_tier      = 0;
            if (seq_level_idx > 7)
                seq_tier = (uint8_t)br_read_bit(br);

            if (has_decoder_model) {
                if (br_read_bit(br)) // decoder_model_present_for_this_op
                    if (skip_operating_parameters_info(br, buf_delay_len) != 0) return -1;
            }

            if (config->initial_presentation_delay_present) {
                if (br_read_bit(br)) // initial_presentation_delay_present_for_this_op
                    br_read_bits(br, 4);
            }

            if (i == 0) {
                config->seq_level_idx_0 = seq_level_idx;
                config->seq_tier_0      = seq_tier;
            }
        }
    }

    // Skip frame size fields we don't need
    int frame_width_bits  = (int)br_read_bits(br, 4) + 1;
    int frame_height_bits = (int)br_read_bits(br, 4) + 1;
    br_read_bits(br, frame_width_bits);  // max_frame_width_minus_1
    br_read_bits(br, frame_height_bits); // max_frame_height_minus_1

    if (!reduced_still_picture_header) {
        int frame_id_numbers_present = (int)br_read_bit(br);
        if (frame_id_numbers_present) {
            br_read_bits(br, 4); // delta_frame_id_length_minus_2
            br_read_bits(br, 3); // additional_frame_id_length_minus_1
        }
    }

    // Skip feature flags we don't need
    br_read_bit(br); // use_128x128_superblock
    br_read_bit(br); // enable_filter_intra
    br_read_bit(br); // enable_intra_edge_filter

    if (!reduced_still_picture_header) {
        br_read_bit(br); // enable_interintra_compound
        br_read_bit(br); // enable_masked_compound
        br_read_bit(br); // enable_warped_motion
        br_read_bit(br); // enable_dual_filter

        if (br_read_bit(br)) { // enable_order_hint
            br_read_bit(br);   // enable_dist_wtd_comp
            br_read_bit(br);   // enable_ref_frame_mvs
        }

        const int SELECT_SCREEN_CONTENT_TOOLS = 2;
        int seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
        if (!br_read_bit(br)) // seq_choose_screen_content_tools
            seq_force_screen_content_tools = (int)br_read_bit(br);

        if (seq_force_screen_content_tools > 0)
            if (!br_read_bit(br)) // seq_choose_integer_mv
                br_read_bit(br);  // seq_force_integer_mv

        if (br_read_bit(br)) // enable_order_hint
            br_read_bits(br, 3); // order_hint_bits_minus_1
    }

    br_read_bit(br); // enable_superres
    br_read_bit(br); // enable_cdef
    br_read_bit(br); // enable_restoration

    if (parse_color_config(br, config) != 0) {
        fprintf(stderr, "av1c: color_config() parse failed.\n");
        return -1;
    }

    (void)still_picture;
    return br->error ? -1 : 0;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

int get_av1config_from_obu(const uint8_t *buffer, size_t length, int is_annexb,
                           Av1Config *config) {
    if (!buffer || length == 0 || !config) return -1;
    (void)is_annexb; // SVT-AV1 always produces standard OBUs, never Annex B

    int    obu_type   = 0;
    size_t header_sz  = 0;
    size_t payload_sz = 0;
    if (parse_obu_header(buffer, length, &obu_type, &header_sz, &payload_sz) != 0)
        return -1;
    if (obu_type != 1) // OBU_SEQUENCE_HEADER
        return -1;
    if (header_sz + payload_sz > length)
        return -1;

    memset(config, 0, sizeof(*config));
    config->marker  = 1;
    config->version = 1;
    return parse_sequence_header(buffer + header_sz, payload_sz, config);
}

int write_av1config(const Av1Config *config, size_t capacity,
                    size_t *bytes_written, uint8_t *buffer) {
    if (!config || !buffer || capacity < 4 || !bytes_written) return -1;

    buffer[0] = (config->marker << 7) | config->version;
    buffer[1] = (config->seq_profile << 5) | config->seq_level_idx_0;
    buffer[2] = (config->seq_tier_0 << 7)         | (config->high_bitdepth << 6)       |
                (config->twelve_bit << 5)          | (config->monochrome << 4)          |
                (config->chroma_subsampling_x << 3)| (config->chroma_subsampling_y << 2)|
                config->chroma_sample_position;
    buffer[3] = config->initial_presentation_delay_present << 4;
    if (config->initial_presentation_delay_present)
        buffer[3] |= config->initial_presentation_delay_minus_one;

    *bytes_written = 4;
    return 0;
}