/*
* Copyright(c) 2022 Intel Corporation
*
* This source code is subject to the terms of the BSD 3-Clause Clear License and
* the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "app_config.h"
#include "app_output_ivf.h"

#define AV1_FOURCC 0x31305641 // used for ivf header
#define IVF_STREAM_HEADER_SIZE 32
#define IVF_FRAME_HEADER_SIZE 12

static __inline void mem_put_le32(void* vmem, int32_t val) {
    uint8_t* mem = (uint8_t*)vmem;

    mem[0] = (uint8_t)((val >> 0) & 0xff);
    mem[1] = (uint8_t)((val >> 8) & 0xff);
    mem[2] = (uint8_t)((val >> 16) & 0xff);
    mem[3] = (uint8_t)((val >> 24) & 0xff);
}

static __inline void mem_put_le16(void* vmem, int32_t val) {
    uint8_t* mem = (uint8_t*)vmem;

    mem[0] = (uint8_t)((val >> 0) & 0xff);
    mem[1] = (uint8_t)((val >> 8) & 0xff);
}

void write_ivf_stream_header(EbConfig* app_cfg, int32_t length) {
    char header[IVF_STREAM_HEADER_SIZE] = {'D', 'K', 'I', 'F'};
    mem_put_le16(header + 4, 0); // version
    mem_put_le16(header + 6, 32); // header size
    mem_put_le32(header + 8, AV1_FOURCC); // fourcc
    mem_put_le16(header + 12, app_cfg->input_padded_width); // width
    mem_put_le16(header + 14, app_cfg->input_padded_height); // height
    mem_put_le32(header + 16, app_cfg->config.frame_rate_numerator); // rate
    mem_put_le32(header + 20, app_cfg->config.frame_rate_denominator); // scale
    mem_put_le32(header + 24, length); // length
    mem_put_le32(header + 28, 0); // unused
    fwrite(header, 1, IVF_STREAM_HEADER_SIZE, app_cfg->bitstream_file);
}

void write_ivf_frame_header(EbConfig* app_cfg, uint32_t byte_count) {
    char header[IVF_FRAME_HEADER_SIZE];

    mem_put_le32(&header[0], (int32_t)byte_count);
    mem_put_le32(&header[4], (int32_t)(app_cfg->ivf_count & 0xFFFFFFFF));
    mem_put_le32(&header[8], (int32_t)(app_cfg->ivf_count >> 32));

    app_cfg->ivf_count++;
    fwrite(header, 1, IVF_FRAME_HEADER_SIZE, app_cfg->bitstream_file);
}

#ifdef CONFIG_WEBM_IO

// Scans a buffer of OBUs and returns a pointer to the sequence header OBU
// (obu_type == 1) and its total byte length, or NULL if not found.
static const uint8_t *find_sequence_header_obu(const uint8_t *buf, size_t sz,
                                                size_t *obu_sz) {
    size_t i = 0;
    while (i < sz) {
        uint8_t header_byte  = buf[i];
        if (header_byte & 0x80) return NULL; // forbidden bit set

        int    obu_type        = (header_byte >> 3) & 0xF;
        int    extension_flag  = (header_byte >> 2) & 0x1;
        int    has_size_field  = (header_byte >> 1) & 0x1;

        size_t header_sz = 1 + (extension_flag ? 1 : 0);
        if (i + header_sz > sz) return NULL;

        size_t payload_sz = 0;
        size_t leb_bytes  = 0;
        if (has_size_field) {
            for (int j = 0; j < 8; j++) {
                if (i + header_sz + j >= sz) return NULL;
                uint8_t b  = buf[i + header_sz + j];
                payload_sz |= (size_t)(b & 0x7F) << (j * 7);
                leb_bytes++;
                if (!(b & 0x80)) break;
            }
        } else {
            payload_sz = sz - i - header_sz;
        }

        size_t total = header_sz + leb_bytes + payload_sz;

        if (obu_type == 1) { // OBU_SEQUENCE_HEADER
            *obu_sz = total;
            return &buf[i];
        }

        i += total;
    }
    return NULL;
}

#ifdef LIBDOVI_FOUND
static int get_dovi_config(const EbConfig* app_cfg, uint8_t config[24]) {
    const DoviRpuDataHeader* header = dovi_rpu_get_header(app_cfg->dovi_rpus->list[0]);
    if (!header) {
        fprintf(stderr, "webmenc> Unable to read Dolby Vision RPU header.\n");
        return -1;
    }
    const uint8_t compatibility = header->bl_video_full_range_flag      ? 0
        : app_cfg->config.transfer_characteristics == EB_CICP_TC_HLG    ? 4
        : app_cfg->config.transfer_characteristics == EB_CICP_TC_BT_709 ? 2
                                                                        : 1;
    dovi_rpu_free_header(header);

    const DoviVdrDmData* dm         = dovi_rpu_get_vdr_dm_data(app_cfg->dovi_rpus->list[0]);
    const bool           compressed = dm && dm->compressed;
    dovi_rpu_free_vdr_dm_data(dm);

    // Dolby Vision levels: maximum width and decoded pixels per second.
    const struct {
        uint32_t width;
        uint64_t pps;
    } levels[] = {
        {1280, 1280ULL * 720 * 24},
        {1280, 1280ULL * 720 * 30},
        {1920, 1920ULL * 1080 * 24},
        {2560, 1920ULL * 1080 * 30},
        {3840, 1920ULL * 1080 * 60},
        {3840, 3840ULL * 2160 * 24},
        {3840, 3840ULL * 2160 * 30},
        {3840, 3840ULL * 2160 * 48},
        {3840, 3840ULL * 2160 * 60},
        {3840, 3840ULL * 2160 * 120},
        {7680, 3840ULL * 2160 * 120},
        {7680, 7680ULL * 4320 * 60},
        {7680, 7680ULL * 4320 * 120},
    };

    const uint64_t pixels  = (uint64_t)app_cfg->input_padded_width * app_cfg->input_padded_height;
    const uint64_t samples = pixels * app_cfg->config.frame_rate_numerator;
    unsigned       level;
    for (level = 0; level < sizeof(levels) / sizeof(levels[0]); level++) {
        if (app_cfg->input_padded_width <= levels[level].width &&
            samples <= levels[level].pps * app_cfg->config.frame_rate_denominator) {
            break;
        }
    }
    if (level == sizeof(levels) / sizeof(levels[0])) {
        fprintf(stderr, "webmenc> Video dimensions and frame rate exceed Dolby Vision level limits.\n");
        return -1;
    }
    level++;
    config[0] = 1; // dv_version_major
    config[2] = (10 << 1) | (level >> 5); // dv_profile, dv_level
    config[3] = (level << 3) | 5; // dv_level, RPU and base layer present
    config[4] = (compatibility << 4) | (compressed << 2);
    return 0;
}
#endif

int write_webm_stream_header(EbConfig *app_cfg,
                             const uint8_t *buf, size_t buf_sz) {
    app_cfg->webm_ctx.stream    = app_cfg->bitstream_file;
    app_cfg->webm_ctx.last_pts_ns = -1;

    size_t         seq_obu_sz = 0;
    const uint8_t *seq_obu    = find_sequence_header_obu(buf, buf_sz, &seq_obu_sz);
    if (!seq_obu) {
        fprintf(stderr, "webmenc> Sequence header OBU not found in first packet.\n");
        return -1;
    }

    SvtWebmEncConfig cfg = {
        .width             = app_cfg->input_padded_width,
        .height            = app_cfg->input_padded_height,
        .timebase_num      = app_cfg->config.frame_rate_denominator,
        .timebase_den      = app_cfg->config.frame_rate_numerator,
    };

#ifdef LIBDOVI_FOUND
    uint8_t dovi_config[24] = {0};
    if (app_cfg->dovi_rpus && app_cfg->dovi_rpus->len) {
        if (get_dovi_config(app_cfg, dovi_config)) {
            return -1;
        }
        cfg.dovi_config = dovi_config;
    }
#endif

    const char *svt_version = svt_av1_get_version();
    char version[128];
    snprintf(version, sizeof(version), "SVT-AV1-Essential %s", svt_version);

    char *enc_settings = NULL;
    if (app_cfg->argc > 0 && app_cfg->argv)
        enc_settings = extract_encoder_settings(app_cfg->argv, app_cfg->argc);

    int result = write_webm_file_header(&app_cfg->webm_ctx, &cfg,
                                        seq_obu, seq_obu_sz,
                                        version, enc_settings);
    free(enc_settings);
    return result;
}

int write_webm_frame_data(EbConfig *app_cfg,
                          const uint8_t *buf, size_t sz,
                          int64_t pts, int is_keyframe) {
    SvtWebmEncConfig cfg = {
        .width        = app_cfg->input_padded_width,
        .height       = app_cfg->input_padded_height,
        .timebase_num = app_cfg->config.frame_rate_denominator,
        .timebase_den = app_cfg->config.frame_rate_numerator,
    };

    SvtWebmFrameData frame = {
        .buf        = buf,
        .sz         = sz,
        .pts_ticks  = pts,
        .is_keyframe = is_keyframe,
    };

    return write_webm_block(&app_cfg->webm_ctx, &cfg, &frame);
}

int write_webm_stream_footer(EbConfig *app_cfg) {
    return write_webm_file_footer(&app_cfg->webm_ctx);
}

#endif // CONFIG_WEBM_IO
