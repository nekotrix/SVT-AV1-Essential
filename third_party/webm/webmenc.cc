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

#include "third_party/webm/webmenc.h"

#include <stdio.h>
#include <string.h>

#include <memory>
#include <new>
#include <string>

#include "third_party/webm/av1_config.h"
#include "third_party/webm/libwebm/mkvmuxer/mkvmuxer.h"
#include "third_party/webm/libwebm/mkvmuxer/mkvmuxerutil.h"
#include "third_party/webm/libwebm/mkvmuxer/mkvwriter.h"

namespace {
const int      kVideoTrackNumber = 1;

int skip_input_output_arg(const char *arg) {
    if (strcmp(arg, "-i") == 0 || strcmp(arg, "--input") == 0)
        return 2;
    if (strcmp(arg, "-b") == 0 || strcmp(arg, "--output") == 0)
        return 2;
    
    return 0;
}
}  // namespace

char *extract_encoder_settings(const char **argv, int argc) {
    size_t total_size = 1; // just null terminator to start
    int i = 1;
    while (i < argc) {
        int num_skip = skip_input_output_arg(argv[i]);
        i += num_skip;
        if (num_skip == 0) {
            total_size += strlen(argv[i]) + 1;
            ++i;
        }
    }
    char *result = static_cast<char *>(malloc(total_size));
    if (result == nullptr) return nullptr;
    char *cur = result;
    i = 1;
    while (i < argc) {
        int num_skip = skip_input_output_arg(argv[i]);
        i += num_skip;
        if (num_skip == 0) {
            cur += snprintf(cur, total_size, "%s ", argv[i]);
            ++i;
        }
    }
    // trim trailing space
    if (cur > result) *(cur - 1) = '\0';
    else *cur = '\0';
    return result;
}

int write_webm_file_header(struct WebmOutputContext *webm_ctx,
                           const SvtWebmEncConfig   *cfg,
                           const uint8_t            *seq_header_obu,
                           size_t                    seq_header_obu_sz,
                           const char               *version,
                           const char               *encoder_settings) {
    std::unique_ptr<mkvmuxer::MkvWriter> writer(
        new (std::nothrow) mkvmuxer::MkvWriter(webm_ctx->stream));
    std::unique_ptr<mkvmuxer::Segment> segment(
        new (std::nothrow) mkvmuxer::Segment());
    if (writer == nullptr || segment == nullptr) {
        fprintf(stderr, "webmenc> mkvmuxer objects alloc failed, out of memory?\n");
        return -1;
    }

    if (!segment->Init(writer.get())) {
        fprintf(stderr, "webmenc> mkvmuxer Init failed.\n");
        return -1;
    }

    segment->set_mode(mkvmuxer::Segment::kFile);
    segment->OutputCues(true);

    mkvmuxer::SegmentInfo *const info = segment->GetSegmentInfo();
    if (!info) {
        fprintf(stderr, "webmenc> Cannot retrieve Segment Info.\n");
        return -1;
    }

    info->set_timecode_scale(1000000);  // 1ms in ns
    std::string writing_app = "SVT-AV1-Essential";
    info->set_writing_app(writing_app.c_str());

    const uint64_t video_track_id =
        segment->AddVideoTrack(static_cast<int>(cfg->width),
                               static_cast<int>(cfg->height),
                               kVideoTrackNumber);
    mkvmuxer::VideoTrack *const video_track =
        static_cast<mkvmuxer::VideoTrack *>(
            segment->GetTrackByNumber(video_track_id));
    if (!video_track) {
        fprintf(stderr, "webmenc> Video track creation failed.\n");
        return -1;
    }

    video_track->set_codec_id("V_AV1");
    video_track->set_language("und");
    video_track->set_default_duration(1000000000ULL * cfg->timebase_num / cfg->timebase_den);

    // Build AV1CodecConfigurationRecord from the sequence header OBU
    // provided by the caller (extracted from the first SVT-AV1 output packet)
    if (!seq_header_obu || seq_header_obu_sz == 0) {
        fprintf(stderr, "webmenc> No sequence header OBU provided.\n");
        return -1;
    }

    Av1Config av1_config;
    if (get_av1config_from_obu(seq_header_obu, seq_header_obu_sz,
                               false, &av1_config) != 0) {
        fprintf(stderr, "webmenc> Failed to parse AV1 sequence header OBU.\n");
        return -1;
    }

    uint8_t av1_config_buffer[4] = { 0 };
    size_t  bytes_written = 0;
    if (write_av1config(&av1_config, sizeof(av1_config_buffer),
                        &bytes_written, av1_config_buffer) != 0) {
        fprintf(stderr, "webmenc> Failed to serialize AV1Config.\n");
        return -1;
    }

    if (!video_track->SetCodecPrivate(av1_config_buffer,
                                      sizeof(av1_config_buffer))) {
        fprintf(stderr, "webmenc> Unable to set AV1 config.\n");
        return -1;
    }

    if (version != nullptr || encoder_settings != nullptr) {
      mkvmuxer::Tag *tag = segment->AddTag();
      if (tag == nullptr) {
          fprintf(stderr, "webmenc> Unable to allocate tag.\n");
          return -1;
      }
      tag->set_track_uid(video_track->uid());
      if (version != nullptr) {
        if (!tag->add_simple_tag("ENCODER", version)) {
            fprintf(stderr, "webmenc> Unable to write ENCODER tag.\n");
            return -1;
        }
      }
      if (encoder_settings != nullptr) {
        if (!tag->add_simple_tag("ENCODER_SETTINGS", encoder_settings)) {
            fprintf(stderr, "webmenc> Unable to write ENCODER_SETTINGS tag.\n");
            return -1;
        }
      }
    }

    webm_ctx->writer  = writer.release();
    webm_ctx->segment = segment.release();
    return 0;
}

int write_webm_block(struct WebmOutputContext *webm_ctx,
                     const SvtWebmEncConfig   *cfg,
                     const SvtWebmFrameData   *frame) {
    if (!webm_ctx->segment) {
        fprintf(stderr, "webmenc> segment is NULL.\n");
        return -1;
    }
    mkvmuxer::Segment *const segment =
        reinterpret_cast<mkvmuxer::Segment *>(webm_ctx->segment);

    int64_t pts_ns = frame->pts_ticks * 1000000000ll *
                     cfg->timebase_num / cfg->timebase_den;
    if (pts_ns <= webm_ctx->last_pts_ns)
        pts_ns = webm_ctx->last_pts_ns + 1000000;
    webm_ctx->last_pts_ns = pts_ns;

    if (!segment->AddFrame(static_cast<const uint8_t *>(frame->buf),
                           frame->sz, kVideoTrackNumber,
                           pts_ns, frame->is_keyframe)) {
        fprintf(stderr, "webmenc> AddFrame failed.\n");
        return -1;
    }
    return 0;
}

int write_webm_file_footer(struct WebmOutputContext *webm_ctx) {
    if (!webm_ctx->writer || !webm_ctx->segment) {
        fprintf(stderr, "webmenc> segment or writer NULL.\n");
        return -1;
    }
    mkvmuxer::MkvWriter *const writer =
        reinterpret_cast<mkvmuxer::MkvWriter *>(webm_ctx->writer);
    mkvmuxer::Segment *const segment =
        reinterpret_cast<mkvmuxer::Segment *>(webm_ctx->segment);
    const bool ok = segment->Finalize();
    delete segment;
    delete writer;
    webm_ctx->writer  = NULL;
    webm_ctx->segment = NULL;

    if (!ok) {
        fprintf(stderr, "webmenc> Segment::Finalize failed.\n");
        return -1;
    }
    return 0;
}