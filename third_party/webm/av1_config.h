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
#ifndef AV1_CONFIG_H_
#define AV1_CONFIG_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _Av1Config {
    uint8_t marker;
    uint8_t version;
    uint8_t seq_profile;
    uint8_t seq_level_idx_0;
    uint8_t seq_tier_0;
    uint8_t high_bitdepth;
    uint8_t twelve_bit;
    uint8_t monochrome;
    uint8_t chroma_subsampling_x;
    uint8_t chroma_subsampling_y;
    uint8_t chroma_sample_position;
    uint8_t initial_presentation_delay_present;
    uint8_t initial_presentation_delay_minus_one;
} Av1Config;

int get_av1config_from_obu(const uint8_t *buffer, size_t length, int is_annexb,
                           Av1Config *config);
int write_av1config(const Av1Config *config, size_t capacity,
                    size_t *bytes_written, uint8_t *buffer);

#ifdef __cplusplus
}
#endif

#endif  // AV1_CONFIG_H_