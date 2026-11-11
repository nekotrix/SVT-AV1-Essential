/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */
#ifndef WEBMENC_H_
#define WEBMENC_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct WebmOutputContext {
    FILE   *stream;
    int64_t last_pts_ns;
    void   *writer;
    void   *segment;
    uint64_t video_track_uid;
    char    *encoder_settings;
};

typedef struct {
    uint32_t       width;
    uint32_t       height;
    int            timebase_num;
    int            timebase_den;
} SvtWebmEncConfig;

typedef struct {
    const void *buf;
    size_t      sz;
    int64_t     pts_ticks;
    int         is_keyframe;
} SvtWebmFrameData;

char *extract_encoder_settings(const char **argv, int argc);

int write_webm_file_header(struct WebmOutputContext *webm_ctx,
                           const SvtWebmEncConfig   *cfg,
                           const uint8_t            *seq_header_obu,
                           size_t                    seq_header_obu_sz,
                           const char               *version,
                           const char               *encoder_settings);

int write_webm_block(struct WebmOutputContext *webm_ctx,
                     const SvtWebmEncConfig   *cfg,
                     const SvtWebmFrameData   *frame);

int write_webm_file_footer(struct WebmOutputContext *webm_ctx);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // WEBMENC_H_