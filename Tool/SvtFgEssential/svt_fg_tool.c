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

/*
* Copyright(c) 2019 Intel Corporation
*
* This source code is subject to the terms of the BSD 3-Clause Clear License and
* the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

/*
 * Standalone SVT-AV1 film-grain estimation and synthesis app.
 *
 * CLI parsing, frame I/O, table export and worker scheduling. Estimation uses
 * svt_aom_denoise_and_model_run() from Source/Lib/Codec/noise_model.c;
 * synthesis uses svt_av1_add_film_grain_run() from grainSynthesis.c.
 */

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#define FG_ISATTY(stream) _isatty(_fileno(stream))
#define strcasecmp _stricmp
#else
#include <pthread.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>
#define FG_ISATTY(stream) isatty(fileno(stream))
#endif

#if HAVE_FFMS2
#include "ffms.h"
#endif

#include "EbSvtAv1Formats.h"
#include "grainSynthesis.h"
#include "pack_unpack_c.h"
#include "pic_operators.h"
#include "sequence_control_set.h"
#include "svt_fg_noise_model.h"
#include "film_grain_analysis.h"
#include "film_grain_seeds.h"
#include "pic_buffer_desc.h"

#define FG_TOOL_MAX_LINE 4096
#define FG_PROGRESS_INTERVAL_MS 500
#define FG_STDIN_PATH "-"
#define FG_STDIN_DISPLAY "<stdin>"

typedef enum VideoContainer {
    VIDEO_RAW   = 0,
    VIDEO_Y4M   = 1,
    VIDEO_FFMS2 = 2,
} VideoContainer;

typedef enum InputFormat {
    INPUT_AUTO = 0,
    INPUT_RAW,
    INPUT_FFMS2,
} InputFormat;

typedef struct VideoInfo {
    int            width;
    int            height;
    int            bit_depth;
    int            fps_num;
    int            fps_den;
    VideoContainer container;
    char           chroma[32];
} VideoInfo;

typedef struct VideoReader {
    FILE*       file;
    VideoInfo   info;
    char        path[1024];
    const char* input_file_path;
    int         is_pipe;
#if HAVE_FFMS2
    FFMS_VideoSource* ffms_video_source;
    FFMS_Index*      ffms_index;
    int              ffms_frame_num;
    int              ffms_num_frames;
    int              ffms_pixel_format;
#endif
    uint8_t   prefix[16];
    size_t    prefix_size;
    size_t    prefix_pos;
    uint64_t  bytes_read;
} VideoReader;

typedef struct VideoWriter {
    FILE*     file;
    VideoInfo info;
    char      path[1024];
} VideoWriter;

typedef struct FilmGrainTableEntry {
    int64_t      start_time;
    int64_t      end_time;
    AomFilmGrain params;
} FilmGrainTableEntry;

typedef struct TimestampCache {
    uint64_t* timestamps;
    int       count;
    int       capacity;
    int       round_down;
    double    accumulator;
    double    frame_duration;
} TimestampCache;

typedef struct Frame8 {
    int      bit_depth;
    int      width;
    int      height;
    int      model_width;
    int      model_height;
    int      y_stride;
    int      uv_stride;
    uint8_t* y;
    uint8_t* u;
    uint8_t* v;
    uint8_t* y2;
    uint8_t* u2;
    uint8_t* v2;
} Frame8;

typedef struct PackedFrame10 {
    int       width;
    int       height;
    int       y_stride;
    int       uv_stride;
    uint16_t* y;
    uint16_t* u;
    uint16_t* v;
} PackedFrame10;

typedef enum SeedMode {
    SEED_MODE_SVT_DEFAULT = 0,
    SEED_MODE_SINGLE      = 1,
    SEED_MODE_LIST        = 2,
} SeedMode;

typedef enum TableEventMode {
    TABLE_EVENTS_SVT_RA    = 0,
    TABLE_EVENTS_PER_FRAME = 1,
} TableEventMode;

typedef struct Options {
    const char*    source_path;
    const char*    target_path;
    const char*    output_path;
    const char*    denoised_output_path;
    const char*    table_path;
    InputFormat    source_format;
    InputFormat    target_format;
    int            fast;
    int            width;
    int            height;
    int            frames;
    int            strength;
    int            adaptive;
    int            bit_depth;
    int            fps_num;
    int            fps_den;
    int            preset;
    int            hierarchical_levels;
    int            intra_refresh_type;
    SeedMode       seed_mode;
    uint16_t       default_seed;
    uint16_t*      seeds;
    uint32_t       seed_count;
    int*           keyframes;
    int            keyframe_count;
    int            jobs;
    int            svt_ra_gop_size;
    TableEventMode table_event_mode;
} Options;

typedef struct ProgressState {
    int    enabled;
    int    total_frames;
    int    completed_frames;
    int    last_line_len;
    double start_time;
    double last_fps;
    double eta_deadline;
#if defined(_WIN32)
    CRITICAL_SECTION lock;
#else
    pthread_mutex_t lock;
#endif
    int lock_initialized;
} ProgressState;

typedef struct ProgressMonitor {
    ProgressState* progress;
    int            stop;
#if defined(_WIN32)
    HANDLE thread;
    HANDLE stop_event;
#else
    pthread_t       thread;
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
#endif
    int started;
} ProgressMonitor;

typedef struct WorkSlot {
    Frame8             source_frame;
    Frame8             target_frame;
    AomDenoiseAndModel denoise_model;
    AomFilmGrain       grain;
    int                fast;
    int                frame_index;
    int                ok;
    char               error[256];
} WorkSlot;

typedef struct EstimateThread {
    WorkSlot*      slot;
    ProgressState* progress;
    int            use_external_denoised;
    int            started;
#if defined(_WIN32)
    HANDLE thread;
#else
    pthread_t thread;
#endif
} EstimateThread;

void svt_fg_tool_init_svt_shims(int fast);

static void usage(FILE* fp, const char* argv0, int full) {
    fprintf(fp,
            "SVT-FG-Essential\n\n"
            "Usage: %s [--source input] [--table output.tbl] [options]\n",
            argv0);
    if (!full) {
        fprintf(fp,
            "\n"
            "Basic options:\n"
            "  --source FILE         Noisy input video (required).\n"
            "  --target FILE         Previously denoised input for grain estimation.\n"
            "  --table FILE          Save AV1 film-grain parameters.\n"
            "  --strength -1|1..50   Denoising strength (default 6); -1 estimates grain automatically.\n"
            "  --jobs N              Frames to analyze in parallel (default 2).\n"
            "                        Used with manual strength or --target.\n"
            "  --keyint N            Periodic keyframe interval in frames; 0 disables periodic keyframes.\n"
            "  --keyframes LIST|FILE|@FILE\n"
            "                        Keyframe numbers or a list file.\n"
            "  --seed S              Set a fixed grain-pattern seed (1..65535).\n"
            "                        -1 uses a sequence optimized to reduce repeating grain patterns.\n"
            "  --seeds S1,S2,...|-1  Cycle through seeds; -1 uses an optimized sequence.\n"
            "  --fast                Enable extra optimizations without changing results.\n"
            "  -h, --help            Show the full help\n");
        return;
    }
    fprintf(fp,
            "\n"
            "Input and output options:\n"
            "  --source FILE         Noisy source video (required); --input and -i are aliases.\n"
            "                        Native inputs: 8-bit or 10-bit 4:2:0 Y4M or planar YUV.\n"
            "                        Other video formats require FFMS2 support.\n"
            "                        Use '-', 'stdin' or a named pipe for streaming input.\n"
            "  --target FILE         Previously denoised video; --denoised-input is an alias.\n"
            "                        Estimate grain from the difference between source and target.\n"
            "                        Otherwise, denoising is performed internally.\n"
            "                        Only one input may read from stdin or a pipe.\n"
            "  --table FILE          Save AV1 film-grain parameters for use with an encoder.\n"
            "                        Choose at least one of --table, --output or --denoised-output.\n"
            "  --output FILE         Save video with estimated grain added; -o is an alias.\n"
            "                        Grain is added to --target, or to the internally denoised source.\n"
            "                        A .y4m extension writes Y4M; other extensions write raw YUV.\n"
            "  --denoised-output FILE\n"
            "                        Save denoised video, using --target if supplied.\n"
            "                        Output format follows the same rule as --output.\n"
            "  --frames N            Maximum number of frames to process; -n is an alias.\n"
            "                        Default: process all frames.\n"
            "  --width W             Raw input width in pixels; -w is an alias.\n"
            "  --height H            Raw input height in pixels; -hgt is an alias.\n"
            "                        Raw input requires both dimensions, each an even number.\n"
            "  --bit-depth 8|10      Raw input bit depth (default 8).\n"
            "  --fps-num N           Frame-rate numerator.\n"
            "  --fps-denom D         Frame-rate denominator; specify both FPS options together.\n"
            "                        Example: --fps-num 24000 --fps-denom 1001.\n"
            "                        Default: input frame rate, or 24/1 for raw input.\n"
            "  --source-format auto|raw|ffms2\n"
            "                        Source reader (default auto).\n"
            "                        auto reads .y4m/.yuv/.raw and pipes directly; other extensions\n"
            "                        use FFMS2. raw forces planar YUV; ffms2 forces video decoding.\n"
            "  --target-format auto|raw|ffms2\n"
            "                        Target reader (default auto); same choices as above.\n"
            "                        At most one input may use FFMS2.\n"
#if HAVE_FFMS2
            "                        FFMS2 support: enabled in this build.\n"
#else
            "                        FFMS2 support: disabled in this build.\n"
#endif
            "\n"
            "Grain options:\n"
            "  --strength -1|1..50   Denoising strength (default 6); --noise-level is an alias.\n"
            "                        Higher positive values remove more noise before grain estimation.\n"
            "                        -1 estimates grain automatically, without a strength setting.\n"
            "                        Automatic table-only runs skip denoising; video output enables it.\n"
            "                        With --target, grain is estimated from the supplied pair instead.\n"
            "  --adaptive 0|1        Adapt denoising block size to video resolution (default 0).\n"
            "                        Applies to manual strength or analysis with --target.\n"
            "  --jobs N              Frames to analyze in parallel (1..256, default 2).\n"
            "                        Applies to manual strength or analysis with --target.\n"
            "                        Automatic analysis processes one frame at a time.\n"
            "  --seed S              Number controlling the generated grain pattern (1..65535).\n"
            "                        A fixed value keeps the same random seed on every frame.\n"
            "                        -1 uses a tested sequence chosen to reduce visible repeating\n"
            "                        grain patterns. Default: change the seed for each frame.\n"
            "  --seeds S1,S2,...|-1  Cycle through a list of seeds, one per frame, then repeat.\n"
            "                        List values: 1..65535. -1 uses a tested sequence chosen to\n"
            "                        reduce visible repeating grain patterns.\n"
            "                        Use either --seed or --seeds, not both.\n"
            "  --fast                Enable extra optimizations without changing results.\n"
            "\n"
            "Keyframe and table options:\n"
            "  --keyint N            Periodic keyframe interval in frames; 0 disables periodic keyframes.\n"
            "                        Default: derive the interval from frame rate and encoder settings.\n"
            "  --keyframes LIST|FILE|@FILE\n"
            "                        Additional keyframe positions, starting from frame 0.\n"
            "                        Accepts comma-separated numbers or a list file.\n"
            "                        Example: --keyframes 0,48,108. Use @FILE to always read a file.\n"
            "                        Automatic grain analysis restarts at these frames.\n"
            "  --preset N            Encoder preset for automatic keyframe spacing (0..13, default 8).\n"
            "  --hierarchical-levels N\n"
            "                        Encoder frame hierarchy (2..5) for keyframe spacing.\n"
            "                        Default: follow preset and video resolution.\n"
            "  --irefresh-type N     Encoder keyframe type: 1=forward, 2=closed (default 2).\n"
            "                        Use the same value as your encoder for table timing.\n"
            "  --table-events svt-ra|per-frame\n"
            "                        When to write grain-table updates (default svt-ra).\n"
            "                        svt-ra follows the encoder's update schedule; per-frame writes\n"
            "                        an update for every frame. Automatic analysis always uses per-frame.\n"
            "                        --table-event-mode is an alias.\n"
            "  -h, --help            Show the full help\n");
}

static int parse_int_arg(const char* text, int min_value, int max_value, int* out) {
    char* end = NULL;
    long  v;

    errno = 0;
    v     = strtol(text, &end, 10);
    if (errno || end == text || *end != '\0' || v < min_value || v > max_value) {
        return 0;
    }
    *out = (int)v;
    return 1;
}

static int compare_ints(const void* a, const void* b) {
    const int av = *(const int*)a;
    const int bv = *(const int*)b;
    return (av > bv) - (av < bv);
}

static char* trim_token(char* text) {
    char* end;
    while (*text == ' ' || *text == '\t' || *text == '\r' || *text == '\n') {
        ++text;
    }
    end = text + strlen(text);
    while (end > text && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n')) {
        *--end = '\0';
    }
    return text;
}

static int parse_frame_number_arg(char* text, int* out) {
    text = trim_token(text);
    if (*text == '#') {
        ++text;
    }
    return parse_int_arg(text, 0, 2147483647, out);
}

static int parse_frame_list(const char* text, int** frames, int* frame_count) {
    char* copy;
    char* token;
    int*  parsed   = NULL;
    int   count    = 0;
    int   capacity = 0;

    if (!text || !*text) {
        return 0;
    }

    copy = (char*)malloc(strlen(text) + 1);
    if (!copy) {
        return 0;
    }
    strcpy(copy, text);

    token = strtok(copy, ",");
    while (token) {
        int  frame_number;
        int* grown;
        if (!parse_frame_number_arg(token, &frame_number)) {
            free(parsed);
            free(copy);
            return 0;
        }
        if (count == capacity) {
            if (capacity > (INT_MAX - 1) / 2 || (size_t)(capacity ? capacity * 2 : 64) > SIZE_MAX / sizeof(*parsed)) {
                free(parsed);
                free(copy);
                return 0;
            }
            capacity = capacity ? capacity * 2 : 64;
            grown = (int*)realloc(parsed, sizeof(*parsed) * (size_t)capacity);
            if (!grown) {
                free(parsed);
                free(copy);
                return 0;
            }
            parsed = grown;
        }
        parsed[count++] = frame_number;
        token           = strtok(NULL, ",");
    }

    free(copy);
    if (count == 0) {
        free(parsed);
        return 0;
    }

    qsort(parsed, (size_t)count, sizeof(*parsed), compare_ints);
    int unique_count = 0;
    for (int i = 0; i < count; ++i) {
        if (unique_count == 0 || parsed[i] != parsed[unique_count - 1]) {
            parsed[unique_count++] = parsed[i];
        }
    }
    *frames      = parsed;
    *frame_count = unique_count;
    return 1;
}

static int parse_keyframes(const char* value, int** frames, int* frame_count) {
    if (*value != '@' && parse_frame_list(value, frames, frame_count)) {
        return 1;
    }

    const char* path = *value == '@' ? value + 1 : value;
    FILE*       file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "Unable to open keyframe list %s\n", path);
        return 0;
    }
    size_t capacity = 4096;
    size_t size     = 0;
    char*  text     = (char*)malloc(capacity);
    int    ok       = 0;
    if (!text) {
        goto cleanup;
    }
    for (;;) {
        size_t got = fread(text + size, 1, capacity - size - 1, file);
        if (memchr(text + size, '\0', got)) {
            fprintf(stderr, "NUL byte in keyframe list %s\n", path);
            goto cleanup;
        }
        size += got;
        if (ferror(file)) {
            goto cleanup;
        }
        if (feof(file)) {
            break;
        }
        if (capacity > SIZE_MAX / 2) {
            goto cleanup;
        }
        char* grown = (char*)realloc(text, capacity * 2);
        if (!grown) {
            goto cleanup;
        }
        text = grown;
        capacity *= 2;
    }
    text[size] = '\0';
    ok = parse_frame_list(text, frames, frame_count);
cleanup:
    free(text);
    if (fclose(file)) {
        ok = 0;
    }
    return ok;
}

static int options_ensure_keyframe_zero(Options* opt) {
    if (opt->keyframe_count > 0 && opt->keyframes[0] == 0) {
        return 1;
    }
    int* grown = (int*)realloc(opt->keyframes, sizeof(*opt->keyframes) * (size_t)(opt->keyframe_count + 1));
    if (!grown) {
        fprintf(stderr, "Unable to allocate keyframe list\n");
        return 0;
    }
    opt->keyframes = grown;
    memmove(opt->keyframes + 1, opt->keyframes, sizeof(*opt->keyframes) * (size_t)opt->keyframe_count);
    opt->keyframes[0] = 0;
    opt->keyframe_count++;
    return 1;
}

static int parse_bool_arg(const char* text, int* out) {
    if (!strcmp(text, "1") || !strcmp(text, "true") || !strcmp(text, "TRUE")) {
        *out = 1;
        return 1;
    }
    if (!strcmp(text, "0") || !strcmp(text, "false") || !strcmp(text, "FALSE")) {
        *out = 0;
        return 1;
    }
    return 0;
}

static int next_arg(int argc, char** argv, int* i, const char** out) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "Missing value for %s\n", argv[*i]);
        return 0;
    }
    *out = argv[++(*i)];
    return 1;
}

static int path_is_stdin(const char* path) {
    return !strcmp(path, "stdin") || !strcmp(path, FG_STDIN_PATH);
}

static int input_path_is_pipe(const char* path) {
    if (path_is_stdin(path)) {
        return 1;
    }
#if !defined(_WIN32)
    struct stat statbuf;
    return !stat(path, &statbuf) && S_ISFIFO(statbuf.st_mode);
#else
    return 0;
#endif
}

static int path_uses_ffms2(const char* path, InputFormat format) {
    if (format != INPUT_AUTO) {
        return format == INPUT_FFMS2;
    }
    if (input_path_is_pipe(path)) {
        return 0;
    }
    const char* ext = strrchr(path, '.');
    if (!ext || strchr(ext, '/') || strchr(ext, '\\')) {
        return 0;
    }
    return strcasecmp(ext, ".yuv") && strcasecmp(ext, ".raw") && strcasecmp(ext, ".y4m");
}

static int parse_input_format(const char* value, InputFormat* format) {
    if (!strcmp(value, "auto")) {
        *format = INPUT_AUTO;
    } else if (!strcmp(value, "raw")) {
        *format = INPUT_RAW;
    } else if (!strcmp(value, "ffms2")) {
        *format = INPUT_FFMS2;
    } else {
        fprintf(stderr, "Input format must be auto, raw or ffms2\n");
        return 0;
    }
    return 1;
}

static int parse_options(int argc, char** argv, Options* opt) {
    memset(opt, 0, sizeof(*opt));
    opt->frames              = -1;
    opt->strength            = 6;
    opt->adaptive            = 0;
    opt->bit_depth           = 8;
    opt->seed_mode           = SEED_MODE_SVT_DEFAULT;
    opt->default_seed        = 7391;
    opt->jobs                = 2;
    opt->svt_ra_gop_size      = -1;
    opt->preset              = 8;
    opt->hierarchical_levels = -1;
    opt->intra_refresh_type  = 2;
    opt->table_event_mode    = TABLE_EVENTS_SVT_RA;

    for (int i = 1; i < argc; ++i) {
        const char* value = NULL;
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(stdout, argv[0], 1);
            return 0;
        } else if (!strcmp(argv[i], "--source") || !strcmp(argv[i], "--input") || !strcmp(argv[i], "-i")) {
            if (!next_arg(argc, argv, &i, &opt->source_path)) {
                return 0;
            }
        } else if (!strcmp(argv[i], "--target") || !strcmp(argv[i], "--denoised-input")) {
            if (!next_arg(argc, argv, &i, &opt->target_path)) {
                return 0;
            }
        } else if (!strcmp(argv[i], "--fast")) {
            opt->fast = 1;
        } else if (!strcmp(argv[i], "--source-format")) {
            if (!next_arg(argc, argv, &i, &value) || !parse_input_format(value, &opt->source_format)) {
                return 0;
            }
        } else if (!strcmp(argv[i], "--target-format")) {
            if (!next_arg(argc, argv, &i, &value) || !parse_input_format(value, &opt->target_format)) {
                return 0;
            }
        } else if (!strcmp(argv[i], "--output") || !strcmp(argv[i], "-o")) {
            if (!next_arg(argc, argv, &i, &opt->output_path)) {
                return 0;
            }
        } else if (!strcmp(argv[i], "--denoised-output")) {
            if (!next_arg(argc, argv, &i, &opt->denoised_output_path)) {
                return 0;
            }
        } else if (!strcmp(argv[i], "--table")) {
            if (!next_arg(argc, argv, &i, &opt->table_path)) {
                return 0;
            }
        } else if (!strcmp(argv[i], "--frames") || !strcmp(argv[i], "-n")) {
            if (!next_arg(argc, argv, &i, &value) || !parse_int_arg(value, 1, 2147483647, &opt->frames)) {
                fprintf(stderr, "Invalid --frames value\n");
                return 0;
            }
        } else if (!strcmp(argv[i], "--width") || !strcmp(argv[i], "-w")) {
            if (!next_arg(argc, argv, &i, &value) || !parse_int_arg(value, 2, 65535, &opt->width)) {
                fprintf(stderr, "Invalid --width value\n");
                return 0;
            }
        } else if (!strcmp(argv[i], "--height") || !strcmp(argv[i], "-hgt")) {
            if (!next_arg(argc, argv, &i, &value) || !parse_int_arg(value, 2, 65535, &opt->height)) {
                fprintf(stderr, "Invalid --height value\n");
                return 0;
            }
        } else if (!strcmp(argv[i], "--bit-depth")) {
            if (!next_arg(argc, argv, &i, &value) || !parse_int_arg(value, 8, 10, &opt->bit_depth) ||
                (opt->bit_depth != 8 && opt->bit_depth != 10)) {
                fprintf(stderr, "Invalid --bit-depth value; expected 8 or 10\n");
                return 0;
            }
        } else if (!strcmp(argv[i], "--strength") || !strcmp(argv[i], "--noise-level")) {
            if (!next_arg(argc, argv, &i, &value) || !parse_int_arg(value, -1, 50, &opt->strength) ||
                opt->strength == 0) {
                fprintf(stderr, "Invalid --strength value; expected -1 or 1..50\n");
                return 0;
            }
        } else if (!strcmp(argv[i], "--adaptive")) {
            if (!next_arg(argc, argv, &i, &value) || !parse_bool_arg(value, &opt->adaptive)) {
                fprintf(stderr, "Invalid --adaptive value; expected 0 or 1\n");
                return 0;
            }
        } else if (!strcmp(argv[i], "--jobs")) {
            if (!next_arg(argc, argv, &i, &value) || !parse_int_arg(value, 1, 256, &opt->jobs)) {
                fprintf(stderr, "Invalid --jobs value; expected 1..256\n");
                return 0;
            }
        } else if (!strcmp(argv[i], "--fps-num")) {
            if (!next_arg(argc, argv, &i, &value) || !parse_int_arg(value, 1, INT_MAX, &opt->fps_num)) {
                fprintf(stderr, "Invalid --fps-num value\n");
                return 0;
            }
        } else if (!strcmp(argv[i], "--fps-denom")) {
            if (!next_arg(argc, argv, &i, &value) || !parse_int_arg(value, 1, INT_MAX, &opt->fps_den)) {
                fprintf(stderr, "Invalid --fps-denom value\n");
                return 0;
            }
        } else if (!strcmp(argv[i], "--preset")) {
            if (!next_arg(argc, argv, &i, &value) || !parse_int_arg(value, 0, 13, &opt->preset)) {
                fprintf(stderr, "Invalid --preset value; expected 0..13\n");
                return 0;
            }
        } else if (!strcmp(argv[i], "--hierarchical-levels")) {
            if (!next_arg(argc, argv, &i, &value) || !parse_int_arg(value, 2, 5, &opt->hierarchical_levels)) {
                fprintf(stderr, "Invalid --hierarchical-levels value; expected 2..5\n");
                return 0;
            }
        } else if (!strcmp(argv[i], "--irefresh-type")) {
            if (!next_arg(argc, argv, &i, &value) || !parse_int_arg(value, 1, 2, &opt->intra_refresh_type)) {
                fprintf(stderr, "Invalid --irefresh-type value; expected 1 or 2\n");
                return 0;
            }
        } else if (!strcmp(argv[i], "--keyint")) {
            if (!next_arg(argc, argv, &i, &value) || !parse_int_arg(value, 0, 2147483647, &opt->svt_ra_gop_size)) {
                fprintf(stderr, "Invalid --keyint value; expected 0 or a positive frame count\n");
                return 0;
            }
        } else if (!strcmp(argv[i], "--keyframes")) {
            if (opt->keyframe_count > 0) {
                fprintf(stderr, "Only one --keyframes list may be specified\n");
                return 0;
            }
            if (!next_arg(argc, argv, &i, &value) ||
                !parse_keyframes(value, &opt->keyframes, &opt->keyframe_count)) {
                fprintf(stderr, "Invalid --keyframes value; expected a comma-separated frame list or file\n");
                return 0;
            }
        } else if (!strcmp(argv[i], "--table-events") || !strcmp(argv[i], "--table-event-mode")) {
            if (!next_arg(argc, argv, &i, &value)) {
                return 0;
            }
            if (!strcmp(value, "svt-ra") || !strcmp(value, "svt") || !strcmp(value, "ra")) {
                opt->table_event_mode = TABLE_EVENTS_SVT_RA;
            } else if (!strcmp(value, "per-frame") || !strcmp(value, "frame") || !strcmp(value, "all")) {
                opt->table_event_mode = TABLE_EVENTS_PER_FRAME;
            } else {
                fprintf(stderr, "Invalid --table-events value; expected svt-ra or per-frame\n");
                return 0;
            }
        } else if (!strcmp(argv[i], "--seed")) {
            if (opt->seed_mode != SEED_MODE_SVT_DEFAULT) {
                fprintf(stderr, "Only one of --seed or --seeds may be specified\n");
                return 0;
            }
            if (!next_arg(argc, argv, &i, &value) ||
                svt_aom_parse_film_grain_seeds(value, &opt->seeds, &opt->seed_count) != 1 ||
                (opt->seed_count != 1 && strcmp(value, "-1"))) {
                free(opt->seeds);
                opt->seeds      = NULL;
                opt->seed_count = 0;
                fprintf(stderr, "Invalid --seed value; expected 1..65535 or -1\n");
                return 0;
            }
            opt->seed_mode = opt->seed_count == 1 ? SEED_MODE_SINGLE : SEED_MODE_LIST;
        } else if (!strcmp(argv[i], "--seeds")) {
            if (opt->seed_mode != SEED_MODE_SVT_DEFAULT) {
                fprintf(stderr, "Only one of --seed or --seeds may be specified\n");
                return 0;
            }
            if (!next_arg(argc, argv, &i, &value) ||
                svt_aom_parse_film_grain_seeds(value, &opt->seeds, &opt->seed_count) != 1) {
                fprintf(stderr, "Invalid --seeds value; expected a comma-separated list of 1..65535 values or -1\n");
                return 0;
            }
            opt->seed_mode = SEED_MODE_LIST;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 0;
        }
    }

    if (!opt->source_path) {
        fprintf(stderr, "--source is required\n");
        return 0;
    }
    if (!opt->output_path && !opt->denoised_output_path && !opt->table_path) {
        fprintf(stderr, "Specify at least one of --output, --denoised-output, or --table\n");
        return 0;
    }
    if ((opt->fps_num && !opt->fps_den) || (!opt->fps_num && opt->fps_den)) {
        fprintf(stderr, "--fps-num and --fps-denom must be specified together\n");
        return 0;
    }
    if ((opt->width && !opt->height) || (!opt->width && opt->height)) {
        fprintf(stderr, "--width and --height must be specified together\n");
        return 0;
    }
    if (opt->target_path && input_path_is_pipe(opt->source_path) && input_path_is_pipe(opt->target_path)) {
        fprintf(stderr, "Only one of --source and --target may read from a pipe or stdin\n");
        return 0;
    }
    if (opt->target_path && path_uses_ffms2(opt->source_path, opt->source_format) &&
        path_uses_ffms2(opt->target_path, opt->target_format)) {
        fprintf(stderr, "Only one of --source and --target may use FFMS2\n");
        return 0;
    }
    if (opt->keyframe_count > 0) {
        if (!options_ensure_keyframe_zero(opt)) {
            return 0;
        }
    }

    return 1;
}

/* YUV/Y4M input and output */
static int path_ends_with_y4m(const char* path) {
    const size_t len = strlen(path);
    if (len < 4) {
        return 0;
    }
    const char* ext = path + len - 4;
    return (ext[0] == '.' && (ext[1] == 'y' || ext[1] == 'Y') && (ext[2] == '4') &&
            (ext[3] == 'm' || ext[3] == 'M'));
}

static int y4m_chroma_bit_depth(const char* chroma, int* bit_depth) {
    if (!strcmp(chroma, "420") || !strcmp(chroma, "420jpeg") || !strcmp(chroma, "420mpeg2") ||
        !strcmp(chroma, "420paldv")) {
        *bit_depth = 8;
        return 1;
    }
    if (!strcmp(chroma, "420p10")) {
        *bit_depth = 10;
        return 1;
    }
    return 0;
}

static const char* skip_ascii_whitespace(const char* text) {
    while (*text == ' ' || *text == '\t' || *text == '\r' || *text == '\n') {
        ++text;
    }
    return text;
}

static void print_line_prefix(FILE* file, const char* line) {
    fputs("prefix bytes:", file);
    for (int i = 0; line[i] && line[i] != '\n' && i < 16; ++i) {
        fprintf(file, " %02X", (unsigned char)line[i]);
    }
    fputs(" ascii:\"", file);
    for (int i = 0; line[i] && line[i] != '\n' && i < 32; ++i) {
        const unsigned char ch = (unsigned char)line[i];
        fputc(ch >= 32 && ch <= 126 ? ch : '.', file);
    }
    fputc('"', file);
}

static int looks_like_windows_text_pipe_corruption(const char* line) {
    int question_count = 0;
    while (*line == '?') {
        ++question_count;
        ++line;
    }
    return question_count > 0 && strstr(line, "FRAME") != NULL;
}

static int read_line_continue(FILE* file, char* line, size_t capacity, size_t n) {
    int ch;

    while ((ch = fgetc(file)) != EOF) {
        if (n + 1 >= capacity) {
            fprintf(stderr, "Input line exceeds %zu bytes\n", capacity);
            return -1;
        }
        line[n++] = (char)ch;
        if (ch == '\n') {
            break;
        }
    }

    if (ferror(file)) {
        fprintf(stderr, "Unable to read input line\n");
        return -1;
    }
    if (n == 0 && ch == EOF) {
        return 0;
    }
    line[n] = '\0';
    return 1;
}

static int read_line(FILE* file, char* line, size_t capacity) {
    return read_line_continue(file, line, capacity, 0);
}

static int parse_y4m_header(char* header, VideoInfo* info) {
    char* token = strtok(header, " \t\r\n");
    if (!token || strcmp(token, "YUV4MPEG2")) {
        fprintf(stderr, "Invalid Y4M header\n");
        return 0;
    }

    info->fps_num   = 24;
    info->fps_den   = 1;
    info->bit_depth = 8;
    strcpy(info->chroma, "420jpeg");

    while ((token = strtok(NULL, " \t\r\n")) != NULL) {
        if (token[0] == 'W') {
            if (!parse_int_arg(token + 1, 2, 65535, &info->width)) {
                fprintf(stderr, "Invalid Y4M width token: %s\n", token);
                return 0;
            }
        } else if (token[0] == 'H') {
            if (!parse_int_arg(token + 1, 2, 65535, &info->height)) {
                fprintf(stderr, "Invalid Y4M height token: %s\n", token);
                return 0;
            }
        } else if (token[0] == 'F') {
            char* colon = strchr(token + 1, ':');
            if (!colon) {
                fprintf(stderr, "Invalid Y4M frame rate token: %s\n", token);
                return 0;
            }
            *colon = '\0';
            if (!parse_int_arg(token + 1, 1, INT_MAX, &info->fps_num) ||
                !parse_int_arg(colon + 1, 1, INT_MAX, &info->fps_den)) {
                fprintf(stderr, "Invalid Y4M frame rate\n");
                return 0;
            }
        } else if (token[0] == 'C') {
            strncpy(info->chroma, token + 1, sizeof(info->chroma) - 1);
            info->chroma[sizeof(info->chroma) - 1] = '\0';
        }
    }

    if (info->width <= 0 || info->height <= 0) {
        fprintf(stderr, "Y4M header did not contain valid W/H tokens\n");
        return 0;
    }
    if ((info->width & 1) || (info->height & 1)) {
        fprintf(stderr, "Only even-sized 4:2:0 frames are supported\n");
        return 0;
    }
    if (!y4m_chroma_bit_depth(info->chroma, &info->bit_depth)) {
        fprintf(stderr, "Only 8-bit and 10-bit 4:2:0 Y4M is supported; got C%s\n", info->chroma);
        return 0;
    }

    return 1;
}

static int reader_set_raw_info(VideoReader* reader, int raw_width, int raw_height, int raw_bit_depth) {
    if (raw_width <= 0 || raw_height <= 0) {
        fprintf(stderr, "Raw input %s requires --width and --height\n", reader->path);
        return 0;
    }
    if ((raw_width & 1) || (raw_height & 1)) {
        fprintf(stderr, "Only even-sized 4:2:0 frames are supported\n");
        return 0;
    }
    reader->info.container = VIDEO_RAW;
    reader->info.width     = raw_width;
    reader->info.height    = raw_height;
    reader->info.bit_depth = raw_bit_depth;
    reader->info.fps_num   = 24;
    reader->info.fps_den   = 1;
    strcpy(reader->info.chroma, raw_bit_depth == 10 ? "420p10" : "420");
    return 1;
}

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

/* Adapted from SVT-AV1-Essential app_context.c, revision 8b8da08a.
 * Encoder configuration/metadata handling is replaced by VideoReader fields.
 */
#if HAVE_FFMS2
static int init_ffms2(VideoReader *reader) {
    FFMS_Init(0, 0);

    FFMS_ErrorInfo err_info;
    char errmsg[1024];
    err_info.Buffer = errmsg;
    err_info.BufferSize = sizeof(errmsg);
    err_info.ErrorType = FFMS_ERROR_SUCCESS;
    err_info.SubType = FFMS_ERROR_SUCCESS;

    FFMS_Indexer *indexer = FFMS_CreateIndexer(reader->input_file_path, &err_info);
    if (!indexer) {
        fprintf(stderr, "FFMS2 Error: Failed to create indexer for %s: %s\n",
                reader->input_file_path, errmsg);
        return 0;
    }

    // Index the file (in memory only)
    FFMS_Index *index = FFMS_DoIndexing2(indexer, FFMS_IEH_ABORT, &err_info);
    if (!index) {
        fprintf(stderr, "FFMS2 Error: Failed to index %s: %s\n",
                reader->input_file_path, errmsg);
        return 0;
    }

    // Get first video track
    int track_num = FFMS_GetFirstTrackOfType(index, FFMS_TYPE_VIDEO, &err_info);
    if (track_num < 0) {
        fprintf(stderr, "FFMS2 Error: No video tracks found in %s\n", reader->input_file_path);
        FFMS_DestroyIndex(index);
        return 0;
    }

    FFMS_VideoSource *video_source = FFMS_CreateVideoSource(
        reader->input_file_path, track_num, index, -1, FFMS_SEEK_NORMAL, &err_info);
    if (!video_source) {
        fprintf(stderr, "FFMS2 Error: Failed to create video source: %s\n", errmsg);
        FFMS_DestroyIndex(index);
        return 0;
    }

    const FFMS_VideoProperties *props = FFMS_GetVideoProperties(video_source);

    // Check first frame
    const FFMS_Frame *test_frame = FFMS_GetFrame(video_source, 0, &err_info);
    if (!test_frame) {
        fprintf(stderr, "FFMS2 Error: Failed to get test frame: %s\n", errmsg);
        FFMS_DestroyVideoSource(video_source);
        FFMS_DestroyIndex(index);
        return 0;
    }

    const int width = test_frame->EncodedWidth;
    const int height = test_frame->EncodedHeight;
    if (width < 2 || width > 65535 || height < 2 || height > 65535 || (width & 1) || (height & 1) ||
        props->NumFrames <= 0) {
        fprintf(stderr, "FFMS2 Error: Invalid dimensions or frame count\n");
        goto fail;
    }

    // Force output to 10-bit, as upstream does.
    const char *pixel_format = "yuv420p10le";
    int pixfmts[2];
    pixfmts[0] = FFMS_GetPixFmt(pixel_format);
    pixfmts[1] = -1;

    if (pixfmts[0] < 0) {
        fprintf(stderr, "FFMS2 Error: Unsupported pixel format %s\n", pixel_format);
        goto fail;
    }
    if (FFMS_SetOutputFormatV2(video_source, pixfmts,
                                width, height,
                                FFMS_RESIZER_BILINEAR, &err_info)) {
        fprintf(stderr, "FFMS2 Error: Failed to set output format to %s: %s\n", pixel_format, errmsg);
        goto fail;
    }

    reader->ffms_video_source = video_source;
    reader->ffms_index = index;
    reader->ffms_num_frames = props->NumFrames;
    reader->ffms_pixel_format = pixfmts[0];
    reader->info.container = VIDEO_FFMS2;
    reader->info.width = width;
    reader->info.height = height;
    reader->info.fps_num = props->FPSNumerator;
    reader->info.fps_den = props->FPSDenominator;
    reader->info.bit_depth = 10;
    strcpy(reader->info.chroma, "420p10");

    fprintf(stderr, "FFMS2: Initialized video source - %dx%d, %d frames, %.2f fps\n",
            width, height, props->NumFrames,
            props->FPSDenominator > 0 ? (double)props->FPSNumerator / props->FPSDenominator : 0.0);
    return 1;

fail:
    FFMS_DestroyVideoSource(video_source);
    FFMS_DestroyIndex(index);
    return 0;
}
#endif

static int reader_open(VideoReader* reader, const char* path, int raw_width, int raw_height, int raw_bit_depth,
                       InputFormat format) {
    char   magic[9];
    char   header[FG_TOOL_MAX_LINE];
    size_t got;
    int    is_stdin = path_is_stdin(path);

    memset(reader, 0, sizeof(*reader));
    reader->input_file_path = path;
    strncpy(reader->path, is_stdin ? FG_STDIN_DISPLAY : path, sizeof(reader->path) - 1);

    if (path_uses_ffms2(path, format)) {
        if (input_path_is_pipe(path)) {
            fprintf(stderr, "FFMS2 requires a seekable file: %s\n", path);
            return 0;
        }
#if HAVE_FFMS2
        return init_ffms2(reader);
#else
        fprintf(stderr, "This build does not have FFMS2 support; rebuild with USE_FFMS2 enabled\n");
        return 0;
#endif
    }

    if (is_stdin) {
        reader->file    = stdin;
        reader->is_pipe = 1;
#if defined(_WIN32)
        _setmode(_fileno(stdin), _O_BINARY);
#endif
    } else {
        reader->file = fopen(path, "rb");
        if (!reader->file) {
            fprintf(stderr, "Unable to open input %s\n", path);
            return 0;
        }
    }

    if (reader->file != stdin) {
#ifdef _WIN32
        HANDLE handle = (HANDLE)_get_osfhandle(_fileno(reader->file));
        if (handle == INVALID_HANDLE_VALUE) {
            fprintf(stderr, "Unable to get input handle for %s\n", path);
            return 0;
        }
        reader->is_pipe = GetFileType(handle) == FILE_TYPE_PIPE;
#else
        int         fd = fileno(reader->file);
        struct stat statbuf;
        if (fstat(fd, &statbuf)) {
            fprintf(stderr, "Unable to stat input %s\n", path);
            return 0;
        }
        reader->is_pipe = S_ISFIFO(statbuf.st_mode);
#endif
    }

    got = fread(magic, 1, sizeof(magic), reader->file);

    if (format != INPUT_RAW && got == sizeof(magic) && !memcmp(magic, "YUV4MPEG2", sizeof(magic))) {
        int ok;
        reader->info.container = VIDEO_Y4M;
        memcpy(header, magic, sizeof(magic));
        if (read_line_continue(reader->file, header, sizeof(header), sizeof(magic)) != 1) {
            fprintf(stderr, "Unable to read Y4M header from %s\n", reader->path);
            return 0;
        }
        ok = parse_y4m_header(header, &reader->info);
        if (!ok) {
            return 0;
        }
    } else {
        if (got > 0) {
            memcpy(reader->prefix, magic, got);
            reader->prefix_size = got;
            reader->prefix_pos  = 0;
        }
        if (!reader_set_raw_info(reader, raw_width, raw_height, raw_bit_depth)) {
            return 0;
        }
    }

    return 1;
}

static uint64_t y4m_frame_payload_size(const VideoInfo* info) {
    const uint64_t luma_samples   = (uint64_t)info->width * (uint64_t)info->height;
    const uint64_t chroma_samples = luma_samples / 2;
    const uint64_t sample_bytes   = info->bit_depth > 8 ? 2 : 1;
    return (luma_samples + chroma_samples) * sample_bytes;
}

static int seek_forward_bytes(FILE* file, uint64_t bytes) {
    int ch;
    if (bytes == 0) {
        return 1;
    }
    --bytes;
    while (bytes > 0) {
        const long step = bytes > (uint64_t)INT_MAX ? INT_MAX : (long)bytes;
        if (fseek(file, step, SEEK_CUR)) {
            return 0;
        }
        bytes -= (uint64_t)step;
    }
    ch = fgetc(file);
    return ch != EOF;
}

static int reader_read_y4m_frame_header(VideoReader* reader, const char* context);

static int reader_infer_y4m_frame_count(VideoReader* reader, int* frame_count) {
    fpos_t         original_pos;
    const uint64_t payload_size = y4m_frame_payload_size(&reader->info);
    int            count        = 0;
    int            ok           = 0;
    *frame_count = -1;
    if (reader->info.container != VIDEO_Y4M || reader->is_pipe) {
        return 0;
    }
    if (fgetpos(reader->file, &original_pos)) {
        fprintf(stderr, "Unable to save Y4M read position for %s\n", reader->path);
        return 0;
    }

    for (;;) {
        const int line_status = reader_read_y4m_frame_header(reader, "while counting");
        if (line_status == 0) {
            ok = 1;
            break;
        }
        if (line_status < 0) {
            break;
        }
        if (!seek_forward_bytes(reader->file, payload_size)) {
            fprintf(stderr, "Truncated Y4M frame payload while counting %s\n", reader->path);
            break;
        }
        if (count == INT_MAX) {
            fprintf(stderr, "Y4M frame count exceeds supported range in %s\n", reader->path);
            break;
        }
        ++count;
    }

    if (fsetpos(reader->file, &original_pos)) {
        fprintf(stderr, "Unable to restore Y4M read position for %s\n", reader->path);
        return 0;
    }
    if (!ok) {
        return 0;
    }

    *frame_count = count;
    return 1;
}

static int reader_read_y4m_frame_header(VideoReader* reader, const char* context) {
    char frame_header[FG_TOOL_MAX_LINE];

    for (;;) {
        const int   line_status = read_line(reader->file, frame_header, sizeof(frame_header));
        const char* line;
        if (line_status == 0) {
            return 0;
        }
        if (line_status < 0) {
            return -1;
        }
        line = skip_ascii_whitespace(frame_header);
        if (!*line) {
            continue;
        }
        if (!strncmp(line, "FRAME", 5) &&
            (line[5] == ' ' || line[5] == '\t' || line[5] == '\r' || line[5] == '\n')) {
            return 1;
        }
        if (context && *context) {
            fprintf(stderr, "Invalid or missing Y4M frame header %s in %s; ", context, reader->path);
        } else {
            fprintf(stderr, "Invalid or missing Y4M frame header in %s; ", reader->path);
        }
        print_line_prefix(stderr, frame_header);
        if (looks_like_windows_text_pipe_corruption(frame_header)) {
            fprintf(stderr,
                    "\nThe stdin stream looks text-transcoded before reaching this tool. On Windows, run the ffmpeg "
                    "pipe through cmd.exe or a binary-safe shell instead of Windows PowerShell's native pipeline.");
        }
        fputc('\n', stderr);
        return -1;
    }
}

static void reader_close(VideoReader* reader) {
#if HAVE_FFMS2
    if (reader->ffms_video_source) {
        FFMS_DestroyVideoSource(reader->ffms_video_source);
        reader->ffms_video_source = NULL;
    }
    if (reader->ffms_index) {
        FFMS_DestroyIndex(reader->ffms_index);
        reader->ffms_index = NULL;
    }
#endif
    if (reader->file) {
        if (reader->file != stdin) {
            fclose(reader->file);
        }
        reader->file = NULL;
    }
}

static int writer_open(VideoWriter* writer, const char* path, const VideoInfo* ref_info) {
    memset(writer, 0, sizeof(*writer));
    strncpy(writer->path, path, sizeof(writer->path) - 1);
    writer->info           = *ref_info;
    writer->info.container = path_ends_with_y4m(path) ? VIDEO_Y4M : VIDEO_RAW;

    writer->file = fopen(path, "wb");
    if (!writer->file) {
        fprintf(stderr, "Unable to open output %s\n", path);
        return 0;
    }

    if (writer->info.container == VIDEO_Y4M) {
        fprintf(writer->file,
                "YUV4MPEG2 W%d H%d F%d:%d Ip A0:0 C%s\n",
                writer->info.width,
                writer->info.height,
                writer->info.fps_num,
                writer->info.fps_den,
                writer->info.bit_depth == 10 ? "420p10" : "420jpeg");
    }

    return 1;
}

static int writer_close(VideoWriter* writer) {
    if (writer->file) {
        const int status = fclose(writer->file);
        writer->file = NULL;
        if (status) {
            fprintf(stderr, "Unable to close output %s\n", writer->path);
            return 0;
        }
    }
    return 1;
}

/* Frame buffers and SVT picture descriptors */
static int align_power_of_two(int value, int alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

static int frame_alloc(Frame8* frame, int width, int height, int bit_depth) {
    const int    model_width  = align_power_of_two(width, MIN_BLOCK_SIZE);
    const int    model_height = align_power_of_two(height, MIN_BLOCK_SIZE);
    const int    y_stride     = model_width;
    const int    uv_stride    = model_width / 2;
    const size_t y_size       = (size_t)y_stride * (size_t)model_height;
    const size_t uv_size      = (size_t)uv_stride * (size_t)(model_height / 2);
    const size_t y_low_size   = bit_depth > 8 ? y_size / 4 : 0;
    const size_t uv_low_size  = bit_depth > 8 ? uv_size / 4 : 0;

    memset(frame, 0, sizeof(*frame));
    if (y_stride > UINT16_MAX || model_height > UINT16_MAX ||
        y_size > (size_t)INT_MAX / (bit_depth > 8 ? 2 : 1) ||
        (uint64_t)(align_power_of_two(width, 32) + 64) * (align_power_of_two(height, 32) + 64) > INT_MAX) {
        fprintf(stderr, "Frame dimensions exceed SVT buffer limits\n");
        return 0;
    }
    frame->bit_depth    = bit_depth;
    frame->width        = width;
    frame->height       = height;
    frame->model_width  = model_width;
    frame->model_height = model_height;
    frame->y_stride     = y_stride;
    frame->uv_stride    = uv_stride;
    frame->y            = (uint8_t*)calloc(1, y_size);
    frame->u            = (uint8_t*)calloc(1, uv_size);
    frame->v            = (uint8_t*)calloc(1, uv_size);
    if (bit_depth > 8) {
        frame->y2 = (uint8_t*)calloc(1, y_low_size);
        frame->u2 = (uint8_t*)calloc(1, uv_low_size);
        frame->v2 = (uint8_t*)calloc(1, uv_low_size);
    }
    if (!frame->y || !frame->u || !frame->v || (bit_depth > 8 && (!frame->y2 || !frame->u2 || !frame->v2))) {
        fprintf(stderr, "Unable to allocate frame buffers\n");
        return 0;
    }
    return 1;
}

static void frame_free(Frame8* frame) {
    free(frame->y);
    free(frame->u);
    free(frame->v);
    free(frame->y2);
    free(frame->u2);
    free(frame->v2);
    memset(frame, 0, sizeof(*frame));
}

static int reader_read_exact(VideoReader* reader, uint8_t* dst, size_t bytes) {
    size_t copied = 0;
    if (reader->prefix_pos < reader->prefix_size) {
        size_t available = reader->prefix_size - reader->prefix_pos;
        if (available > bytes) {
            available = bytes;
        }
        memcpy(dst, reader->prefix + reader->prefix_pos, available);
        reader->prefix_pos += available;
        copied += available;
    }
    if (copied < bytes) {
        copied += fread(dst + copied, 1, bytes - copied, reader->file);
    }
    reader->bytes_read += copied;
    return copied == bytes;
}

static uint8_t get_low2_sample(const uint8_t* low_plane, int low_stride, int x, int y) {
    return (uint8_t)((low_plane[y * low_stride + (x >> 2)] >> (6 - 2 * (x & 3))) & 3);
}

static void set_low2_sample(uint8_t* low_plane, int low_stride, int x, int y, uint8_t value) {
    const int shift = 6 - 2 * (x & 3);
    uint8_t*  byte  = &low_plane[y * low_stride + (x >> 2)];
    *byte           = (uint8_t)((*byte & ~(3 << shift)) | ((value & 3) << shift));
}

static int read_plane_8(VideoReader* reader, uint8_t* plane, int width, int height, int stride) {
    for (int y = 0; y < height; ++y) {
        if (!reader_read_exact(reader, plane + y * stride, (size_t)width)) {
            return 0;
        }
    }
    return 1;
}

static int write_plane_8(FILE* file, const uint8_t* plane, int width, int height, int stride) {
    for (int y = 0; y < height; ++y) {
        if (fwrite(plane + y * stride, 1, (size_t)width, file) != (size_t)width) {
            return 0;
        }
    }
    return 1;
}

static void split_plane_10_row(const uint8_t* row, uint8_t* plane, uint8_t* low_plane, int width, int stride, int y) {
    const int low_stride = stride / 4;
    memset(low_plane + y * low_stride, 0, (size_t)low_stride);
    for (int x = 0; x < width; ++x) {
        uint16_t sample = (uint16_t)(row[2 * x] | (row[2 * x + 1] << 8));
        if (sample > 1023) {
            sample = 1023;
        }
        plane[y * stride + x] = (uint8_t)(sample >> 2);
        set_low2_sample(low_plane, low_stride, x, y, (uint8_t)(sample & 3));
    }
}

static int read_plane_10(VideoReader* reader, uint8_t* plane, uint8_t* low_plane, int width, int height, int stride) {
    const size_t row_bytes  = (size_t)width * 2;
    uint8_t*     row        = (uint8_t*)malloc(row_bytes);
    int          ok         = 0;
    if (!row) {
        fprintf(stderr, "Unable to allocate 10-bit input row\n");
        return 0;
    }
    for (int y = 0; y < height; ++y) {
        if (!reader_read_exact(reader, row, row_bytes)) {
            goto cleanup;
        }
        split_plane_10_row(row, plane, low_plane, width, stride, y);
    }
    ok = 1;
cleanup:
    free(row);
    return ok;
}

static int write_plane_10(FILE* file, const uint8_t* plane, const uint8_t* low_plane, int width, int height,
                          int stride) {
    const int    low_stride = stride / 4;
    const size_t row_bytes  = (size_t)width * 2;
    uint8_t*     row        = (uint8_t*)malloc(row_bytes);
    int          ok         = 0;
    if (!row) {
        fprintf(stderr, "Unable to allocate 10-bit output row\n");
        return 0;
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint16_t sample = (uint16_t)((plane[y * stride + x] << 2) | get_low2_sample(low_plane, low_stride, x, y));
            row[2 * x]      = (uint8_t)(sample & 0xFF);
            row[2 * x + 1]  = (uint8_t)(sample >> 8);
        }
        if (fwrite(row, 1, row_bytes, file) != row_bytes) {
            goto cleanup;
        }
    }
    ok = 1;
cleanup:
    free(row);
    return ok;
}

#if HAVE_FFMS2
static int ffms2_read_input_frames(VideoReader *reader, Frame8 *input_ptr) {
    FFMS_VideoSource *video_source = reader->ffms_video_source;
    int frame_num = reader->ffms_frame_num;

    // Check if we've reached the end
    if (frame_num >= reader->ffms_num_frames) {
        return 0;
    }

    // Get frame from FFMS2
    FFMS_ErrorInfo err_info;
    char errmsg[1024];
    err_info.Buffer = errmsg;
    err_info.BufferSize = sizeof(errmsg);
    err_info.ErrorType = FFMS_ERROR_SUCCESS;
    err_info.SubType = FFMS_ERROR_SUCCESS;

    const FFMS_Frame *frame = FFMS_GetFrame(video_source, frame_num, &err_info);
    if (!frame) {
        fprintf(stderr, "FFMS2 Error getting frame %d: %s\n", frame_num, errmsg);
        return -1;
    }
    if (frame->ScaledWidth != input_ptr->width || frame->ScaledHeight != input_ptr->height ||
        frame->ConvertedPixelFormat != reader->ffms_pixel_format) {
        fprintf(stderr, "FFMS2 Error: Frame dimensions or pixel format changed at frame %d\n", frame_num);
        return -1;
    }

    const uint32_t width = input_ptr->width;
    const uint32_t height = input_ptr->height;
    const uint32_t chroma_width = width >> 1;
    const uint32_t chroma_height = height >> 1;
    for (int plane = 0; plane < 3; ++plane) {
        const int row_bytes = (plane ? chroma_width : width) * 2;
        if (!frame->Data[plane] || frame->Linesize[plane] < row_bytes) {
            fprintf(stderr, "FFMS2 Error: Invalid plane stride at frame %d\n", frame_num);
            return -1;
        }
    }

    // Copy frame data (10-bit = 2 bytes per sample)
    const uint8_t *src_y = frame->Data[0];
    const uint8_t *src_u = frame->Data[1];
    const uint8_t *src_v = frame->Data[2];

    for (uint32_t y = 0; y < height; y++) {
        split_plane_10_row(src_y, input_ptr->y, input_ptr->y2, width, input_ptr->y_stride, y);
        src_y += frame->Linesize[0];
    }

    for (uint32_t y = 0; y < chroma_height; y++) {
        split_plane_10_row(src_u, input_ptr->u, input_ptr->u2, chroma_width, input_ptr->uv_stride, y);
        split_plane_10_row(src_v, input_ptr->v, input_ptr->v2, chroma_width, input_ptr->uv_stride, y);
        src_u += frame->Linesize[1];
        src_v += frame->Linesize[2];
    }

    ++reader->ffms_frame_num;
    return 1;
}
#endif

static int frame_read(VideoReader* reader, Frame8* frame) {
#if HAVE_FFMS2
    if (reader->info.container == VIDEO_FFMS2) {
        return ffms2_read_input_frames(reader, frame);
    }
#endif
    const int uv_width  = frame->width / 2;
    const int uv_height = frame->height / 2;

    if (reader->info.container == VIDEO_Y4M) {
        int line_status = reader_read_y4m_frame_header(reader, "");
        if (line_status == 0) {
            return 0;
        }
        if (line_status < 0) {
            return -1;
        }
    }

    const uint64_t payload_start = reader->bytes_read;
    if (reader->info.bit_depth > 8) {
        if (!read_plane_10(reader, frame->y, frame->y2, frame->width, frame->height, frame->y_stride)) {
            goto luma_read_failed;
        }
        if (!read_plane_10(reader, frame->u, frame->u2, uv_width, uv_height, frame->uv_stride) ||
            !read_plane_10(reader, frame->v, frame->v2, uv_width, uv_height, frame->uv_stride)) {
            fprintf(stderr, "Truncated frame in %s\n", reader->path);
            return -1;
        }
    } else {
        if (!read_plane_8(reader, frame->y, frame->width, frame->height, frame->y_stride)) {
            goto luma_read_failed;
        }
        if (!read_plane_8(reader, frame->u, uv_width, uv_height, frame->uv_stride) ||
            !read_plane_8(reader, frame->v, uv_width, uv_height, frame->uv_stride)) {
            fprintf(stderr, "Truncated frame in %s\n", reader->path);
            return -1;
        }
        // Match the encoder's (sample << 2) + 1 upshift.
        if (frame->bit_depth > 8) {
            memset(frame->y2, 0x55, (size_t)(frame->y_stride / 4) * frame->model_height);
            memset(frame->u2, 0x55, (size_t)(frame->uv_stride / 4) * (frame->model_height / 2));
            memset(frame->v2, 0x55, (size_t)(frame->uv_stride / 4) * (frame->model_height / 2));
        }
    }
    return 1;

luma_read_failed:
    if (reader->info.container == VIDEO_RAW && reader->bytes_read == payload_start && !ferror(reader->file)) {
        return 0;
    }
    fprintf(stderr, "Truncated or unreadable frame in %s\n", reader->path);
    return -1;
}

static int frame_write(VideoWriter* writer, const Frame8* frame) {
    const int uv_width  = frame->width / 2;
    const int uv_height = frame->height / 2;

    if (!writer->file) {
        return 1;
    }
    if (writer->info.container == VIDEO_Y4M) {
        if (fputs("FRAME\n", writer->file) < 0) {
            return 0;
        }
    }
    if (writer->info.bit_depth > 8) {
        return write_plane_10(writer->file, frame->y, frame->y2, frame->width, frame->height, frame->y_stride) &&
            write_plane_10(writer->file, frame->u, frame->u2, uv_width, uv_height, frame->uv_stride) &&
            write_plane_10(writer->file, frame->v, frame->v2, uv_width, uv_height, frame->uv_stride);
    }
    return write_plane_8(writer->file, frame->y, frame->width, frame->height, frame->y_stride) &&
        write_plane_8(writer->file, frame->u, uv_width, uv_height, frame->uv_stride) &&
        write_plane_8(writer->file, frame->v, uv_width, uv_height, frame->uv_stride);
}

static void frame_copy(Frame8* dst, const Frame8* src) {
    const size_t y_size      = (size_t)src->y_stride * (size_t)src->height;
    const size_t uv_size     = (size_t)src->uv_stride * (size_t)(src->height / 2);
    const size_t y_low_size  = src->bit_depth > 8 ? (size_t)(src->y_stride / 4) * (size_t)src->height : 0;
    const size_t uv_low_size = src->bit_depth > 8 ? (size_t)(src->uv_stride / 4) * (size_t)(src->height / 2) : 0;
    memcpy(dst->y, src->y, y_size);
    memcpy(dst->u, src->u, uv_size);
    memcpy(dst->v, src->v, uv_size);
    if (src->bit_depth > 8) {
        memcpy(dst->y2, src->y2, y_low_size);
        memcpy(dst->u2, src->u2, uv_low_size);
        memcpy(dst->v2, src->v2, uv_low_size);
    }
}

static EbPictureBufferDesc make_picture_desc(Frame8* frame) {
    EbPictureBufferDesc pic;
    memset(&pic, 0, sizeof(pic));
    pic.y_buffer         = frame->y;
    pic.u_buffer         = frame->u;
    pic.v_buffer         = frame->v;
    pic.y_buffer_bit_inc = frame->y2;
    pic.u_buffer_bit_inc = frame->u2;
    pic.v_buffer_bit_inc = frame->v2;
    pic.y_stride         = (uint16_t)frame->y_stride;
    pic.u_stride         = (uint16_t)frame->uv_stride;
    pic.v_stride         = (uint16_t)frame->uv_stride;
    pic.y_stride_bit_inc = (uint16_t)frame->y_stride;
    pic.u_stride_bit_inc = (uint16_t)frame->uv_stride;
    pic.v_stride_bit_inc = (uint16_t)frame->uv_stride;
    pic.max_width        = (uint16_t)frame->model_width;
    pic.max_height       = (uint16_t)frame->model_height;
    pic.width            = (uint16_t)frame->model_width;
    pic.height           = (uint16_t)frame->model_height;
    pic.bit_depth        = frame->bit_depth > 8 ? EB_TEN_BIT : EB_EIGHT_BIT;
    pic.color_format     = EB_YUV420;
    return pic;
}

static int packed_frame10_alloc(PackedFrame10* frame, int width, int height) {
    memset(frame, 0, sizeof(*frame));
    frame->width     = width;
    frame->height    = height;
    frame->y_stride  = width;
    frame->uv_stride = width / 2;
    frame->y         = (uint16_t*)calloc((size_t)frame->y_stride * (size_t)height, sizeof(*frame->y));
    frame->u         = (uint16_t*)calloc((size_t)frame->uv_stride * (size_t)(height / 2), sizeof(*frame->u));
    frame->v         = (uint16_t*)calloc((size_t)frame->uv_stride * (size_t)(height / 2), sizeof(*frame->v));
    if (!frame->y || !frame->u || !frame->v) {
        fprintf(stderr, "Unable to allocate packed 10-bit frame buffers\n");
        return 0;
    }
    return 1;
}

static void packed_frame10_free(PackedFrame10* frame) {
    free(frame->y);
    free(frame->u);
    free(frame->v);
    memset(frame, 0, sizeof(*frame));
}

static void frame_to_packed10(const Frame8* src, PackedFrame10* dst) {
    svt_aom_compressed_pack_sb(src->y,
                                src->y_stride,
                                src->y2,
                                src->y_stride / 4,
                                dst->y,
                                dst->y_stride,
                                src->width,
                                src->height);
    svt_aom_compressed_pack_sb(src->u,
                                src->uv_stride,
                                src->u2,
                                src->uv_stride / 4,
                                dst->u,
                                dst->uv_stride,
                                src->width / 2,
                                src->height / 2);
    svt_aom_compressed_pack_sb(src->v,
                                src->uv_stride,
                                src->v2,
                                src->uv_stride / 4,
                                dst->v,
                                dst->uv_stride,
                                src->width / 2,
                                src->height / 2);
}

static void frame_from_packed10(PackedFrame10* src, Frame8* dst) {
    svt_unpack_and_2bcompress_c(src->y,
                                 src->y_stride,
                                 dst->y,
                                 dst->y_stride,
                                 dst->y2,
                                 dst->y_stride / 4,
                                 src->width,
                                 src->height);
    svt_unpack_and_2bcompress_c(src->u,
                                 src->uv_stride,
                                 dst->u,
                                 dst->uv_stride,
                                 dst->u2,
                                 dst->uv_stride / 4,
                                 src->width / 2,
                                 src->height / 2);
    svt_unpack_and_2bcompress_c(src->v,
                                 src->uv_stride,
                                 dst->v,
                                 dst->uv_stride,
                                 dst->v2,
                                 dst->uv_stride / 4,
                                 src->width / 2,
                                 src->height / 2);
}

/* From enc_handle.c:compute_default_intra_period(). Only the state access is
 * replaced by arguments; keep the upstream arithmetic and refresh adjustment.
 */
static int32_t compute_default_intra_period(double frame_rate, int32_t hierarchical_levels, int32_t intra_refresh_type) {
    double  fps           = frame_rate;
    int32_t mini_gop_size = (1 << hierarchical_levels);

    // use a 5-sec gop by default.
    int32_t intra_period = (int)((fps + mini_gop_size) / mini_gop_size) * mini_gop_size * 5;
    if (intra_refresh_type == 1) {
        intra_period -= 1;
    }

    return intra_period;
}

static int set_table_config(Options* opt, VideoInfo* info) {
    if (opt->fps_num) {
        info->fps_num = opt->fps_num;
        info->fps_den = opt->fps_den;
    }
    const double frame_rate = (double)info->fps_num / info->fps_den;
    if (frame_rate > 240) {
        fprintf(stderr, "The maximum allowed frame rate is 240 fps\n");
        return 0;
    }
    if (opt->svt_ra_gop_size < 0) {
        ResolutionRange input_resolution;
        svt_aom_derive_input_resolution(&input_resolution, (uint32_t)info->width * info->height);

        // enc_handle.c:copy_api_from_app(), random-access CRF configuration.
        int32_t hierarchical_levels = opt->hierarchical_levels;
        if (hierarchical_levels < 0) {
            hierarchical_levels = (input_resolution >= INPUT_SIZE_1080p_RANGE && opt->preset >= 8) ||
                    !(opt->preset <= 8) || input_resolution >= INPUT_SIZE_8K_RANGE
                ? 4
                : 5;
        }
        // Upstream computes the intra period before forcing the hierarchy for
        // forward keyframes in set_param_based_on_input().
        opt->svt_ra_gop_size = compute_default_intra_period(frame_rate, hierarchical_levels, opt->intra_refresh_type) + 1;
    }
    return 1;
}

/* Film-grain seeds and table export */
static uint16_t advance_seed(uint16_t seed) {
    seed = (uint16_t)(seed + 3381);
    if (!seed) {
        seed = (uint16_t)(seed + 7391);
    }
    return seed;
}

static uint16_t seed_for_frame(const Options* opt, int frame_index, uint16_t* default_seed) {
    if (opt->seed_mode == SEED_MODE_SINGLE) {
        return opt->seeds[0];
    }
    if (opt->seed_mode == SEED_MODE_LIST) {
        return opt->seeds[frame_index % opt->seed_count];
    }

    {
        const uint16_t seed = *default_seed;
        *default_seed       = advance_seed(*default_seed);
        return seed;
    }
}

static int timestamp_cache_init(TimestampCache* cache, const VideoInfo* info) {
    const int    fps_num    = info->fps_num > 0 ? info->fps_num : 24;
    const int    fps_den    = info->fps_den > 0 ? info->fps_den : 1;
    const double frame_rate = (double)fps_num / fps_den;
    memset(cache, 0, sizeof(*cache));
    cache->capacity   = 256;
    cache->timestamps = (uint64_t*)calloc((size_t)cache->capacity, sizeof(*cache->timestamps));
    if (!cache->timestamps) {
        fprintf(stderr, "Unable to allocate timestamp cache\n");
        return 0;
    }
    cache->count          = 1;
    cache->accumulator    = 0.0;
    cache->frame_duration = (double)10000000 / frame_rate;
    cache->timestamps[0]  = 0;
    return 1;
}

static void timestamp_cache_free(TimestampCache* cache) {
    free(cache->timestamps);
    memset(cache, 0, sizeof(*cache));
}

static int timestamp_cache_reserve(TimestampCache* cache, int frame_index) {
    if (frame_index < cache->capacity) {
        return 1;
    }
    int new_capacity = cache->capacity;
    while (new_capacity <= frame_index) {
        if (new_capacity > INT_MAX / 2) {
            fprintf(stderr, "Timestamp cache capacity overflow\n");
            return 0;
        }
        new_capacity *= 2;
    }
    uint64_t* grown = (uint64_t*)realloc(cache->timestamps, (size_t)new_capacity * sizeof(*cache->timestamps));
    if (!grown) {
        fprintf(stderr, "Unable to grow timestamp cache\n");
        return 0;
    }
    cache->timestamps = grown;
    cache->capacity   = new_capacity;
    return 1;
}

static uint64_t timestamp_for_frame(TimestampCache* cache, int frame_index) {
    if (frame_index <= 0) {
        return 0;
    }
    if (!timestamp_cache_reserve(cache, frame_index)) {
        return 0;
    }
    // Automatic per-frame entries must start no later than the exact frame time
    // used by the encoder's table lookup. Retain legacy rounding otherwise.
    while (cache->count <= frame_index) {
        cache->accumulator += cache->frame_duration;
        cache->timestamps[cache->count] = (uint64_t)(cache->round_down ? floor(cache->accumulator)
                                                                    : ceil(cache->accumulator));
        cache->count++;
    }
    return cache->timestamps[frame_index];
}

static int svt_ra_gop_start_for_frame(const Options* opt, int frame_index) {
    int gop_start = 0;
    for (int i = 0; i < opt->keyframe_count && opt->keyframes[i] <= frame_index; ++i) {
        gop_start = opt->keyframes[i];
    }
    // SVT restarts the intra-period counter at a forced keyframe.
    if (opt->svt_ra_gop_size > 0) {
        gop_start += ((frame_index - gop_start) / opt->svt_ra_gop_size) * opt->svt_ra_gop_size;
    }
    return gop_start;
}

static int should_write_table_event(const Options* opt, int frame_index) {
    int gop_start;
    int offset;
    if (opt->table_event_mode == TABLE_EVENTS_PER_FRAME) {
        return 1;
    }
    gop_start = svt_ra_gop_start_for_frame(opt, frame_index);
    offset    = frame_index - gop_start;
    return offset == 0 ? frame_index == 0 || opt->intra_refresh_type != 1 : (offset & 1);
}

/* aom_dsp/grain_table.c, AOM revision 0d4bddb983. The upstream writer is
 * unchanged apart from the entry/parameter types. Preserve its text format.
 */
static void grain_table_entry_write(FILE *file,
                                    FilmGrainTableEntry *entry) {
  const AomFilmGrain *pars = &entry->params;
  fprintf(file, "E %" PRId64 " %" PRId64 " %d %d %d\n", entry->start_time,
          entry->end_time, pars->apply_grain, pars->random_seed,
          pars->update_parameters);
  if (pars->update_parameters) {
    fprintf(file, "\tp %d %d %d %d %d %d %d %d %d %d %d %d\n",
            pars->ar_coeff_lag, pars->ar_coeff_shift, pars->grain_scale_shift,
            pars->scaling_shift, pars->chroma_scaling_from_luma,
            pars->overlap_flag, pars->cb_mult, pars->cb_luma_mult,
            pars->cb_offset, pars->cr_mult, pars->cr_luma_mult,
            pars->cr_offset);
    fprintf(file, "\tsY %d ", pars->num_y_points);
    for (int i = 0; i < pars->num_y_points; ++i) {
      fprintf(file, " %d %d", pars->scaling_points_y[i][0],
              pars->scaling_points_y[i][1]);
    }
    fprintf(file, "\n\tsCb %d", pars->num_cb_points);
    for (int i = 0; i < pars->num_cb_points; ++i) {
      fprintf(file, " %d %d", pars->scaling_points_cb[i][0],
              pars->scaling_points_cb[i][1]);
    }
    fprintf(file, "\n\tsCr %d", pars->num_cr_points);
    for (int i = 0; i < pars->num_cr_points; ++i) {
      fprintf(file, " %d %d", pars->scaling_points_cr[i][0],
              pars->scaling_points_cr[i][1]);
    }
    fprintf(file, "\n\tcY");
    const int n = 2 * pars->ar_coeff_lag * (pars->ar_coeff_lag + 1);
    for (int i = 0; i < n; ++i) {
      fprintf(file, " %d", pars->ar_coeffs_y[i]);
    }
    fprintf(file, "\n\tcCb");
    for (int i = 0; i <= n; ++i) {
      fprintf(file, " %d", pars->ar_coeffs_cb[i]);
    }
    fprintf(file, "\n\tcCr");
    for (int i = 0; i <= n; ++i) {
      fprintf(file, " %d", pars->ar_coeffs_cr[i]);
    }
    fprintf(file, "\n");
  }
}

static int open_fgs_table(const char* path, FILE** out_file) {
    FILE* file = fopen(path, "w");
    if (!file) {
        fprintf(stderr, "Unable to open table output %s\n", path);
        return 0;
    }
    fprintf(file, "filmgrn1\n");
    *out_file = file;
    return 1;
}

static int write_fgs_table_entry(FILE* file, TimestampCache* timestamps, int start_frame, int end_frame,
                                  const AomFilmGrain* grain) {
    if (!timestamp_cache_reserve(timestamps, end_frame)) {
        return 0;
    }
    FilmGrainTableEntry entry;
    entry.start_time = (int64_t)timestamp_for_frame(timestamps, start_frame);
    entry.end_time   = (int64_t)timestamp_for_frame(timestamps, end_frame);
    entry.params     = *grain;
    grain_table_entry_write(file, &entry);
    return !ferror(file);
}

static int same_dimensions(const VideoInfo* a, const VideoInfo* b) {
    return a->width == b->width && a->height == b->height;
}

/* Progress display */
static double fg_now_seconds(void) {
#if defined(_WIN32)
    return (double)GetTickCount64() * 0.001;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static int fg_terminal_width(void) {
    const char* columns = getenv("COLUMNS");
    int         width   = 80;
    if (columns && *columns) {
        int parsed = 0;
        if (parse_int_arg(columns, 20, 512, &parsed)) {
            width = parsed;
        }
    }
    return width < 20 ? 20 : width;
}

static int fg_progress_enabled(void) {
    const char* env = getenv("SVT_FG_PROGRESS");
    if (env && *env) {
        if (!strcmp(env, "0") || !strcmp(env, "false") || !strcmp(env, "FALSE") || !strcmp(env, "off") ||
            !strcmp(env, "OFF")) {
            return 0;
        }
        return 1;
    }
    return FG_ISATTY(stderr);
}

static void format_duration(double seconds, char* out, size_t out_capacity) {
    int total;
    int hours;
    int minutes;
    int secs;

    if (seconds < 0.0) {
        seconds = 0.0;
    }
    if (seconds < 60.0) {
        snprintf(out, out_capacity, "%.1fs", seconds);
        return;
    }

    total   = (int)(seconds + 0.5);
    hours   = total / 3600;
    minutes = (total / 60) % 60;
    secs    = total % 60;
    if (hours > 0) {
        snprintf(out, out_capacity, "%d:%02d:%02d", hours, minutes, secs);
    } else {
        snprintf(out, out_capacity, "%02d:%02d", minutes, secs);
    }
}

static void format_duration_clock(double seconds, char* out, size_t out_capacity) {
    int total;
    int hours;
    int minutes;
    int secs;

    if (seconds < 0.0) {
        seconds = 0.0;
    }
    total   = (int)(seconds + 0.5);
    hours   = total / 3600;
    minutes = (total / 60) % 60;
    secs    = total % 60;
    if (hours > 0) {
        snprintf(out, out_capacity, "%d:%02d:%02d", hours, minutes, secs);
    } else {
        snprintf(out, out_capacity, "%02d:%02d", minutes, secs);
    }
}

static void progress_init(ProgressState* progress, int total_frames) {
    memset(progress, 0, sizeof(*progress));
    progress->enabled          = fg_progress_enabled();
    progress->total_frames     = total_frames;
    progress->completed_frames = 0;
    progress->start_time       = fg_now_seconds();
    progress->last_fps         = 0.0;
    progress->eta_deadline     = -1.0;
#if defined(_WIN32)
    InitializeCriticalSection(&progress->lock);
    progress->lock_initialized = 1;
#else
    if (pthread_mutex_init(&progress->lock, NULL) == 0) {
        progress->lock_initialized = 1;
    } else {
        progress->enabled = 0;
    }
#endif
}

static void progress_destroy(ProgressState* progress) {
    if (!progress->lock_initialized) {
        return;
    }
#if defined(_WIN32)
    DeleteCriticalSection(&progress->lock);
#else
    pthread_mutex_destroy(&progress->lock);
#endif
    progress->lock_initialized = 0;
}

static void progress_lock(ProgressState* progress) {
    if (!progress->lock_initialized) {
        return;
    }
#if defined(_WIN32)
    EnterCriticalSection(&progress->lock);
#else
    pthread_mutex_lock(&progress->lock);
#endif
}

static void progress_unlock(ProgressState* progress) {
    if (!progress->lock_initialized) {
        return;
    }
#if defined(_WIN32)
    LeaveCriticalSection(&progress->lock);
#else
    pthread_mutex_unlock(&progress->lock);
#endif
}

static void progress_snapshot(ProgressState* progress, int* completed, int* total, double* start_time, double* fps,
                              double* eta_deadline, int* last_line_len) {
    progress_lock(progress);
    *completed     = progress->completed_frames;
    *total         = progress->total_frames;
    *start_time    = progress->start_time;
    *fps           = progress->last_fps;
    *eta_deadline  = progress->eta_deadline;
    *last_line_len = progress->last_line_len;
    progress_unlock(progress);
}

static void progress_set_last_line_len(ProgressState* progress, int last_line_len) {
    progress_lock(progress);
    progress->last_line_len = last_line_len;
    progress_unlock(progress);
}

static void progress_note_frame(ProgressState* progress) {
    if (!progress->enabled) {
        return;
    }
    progress_lock(progress);
    const double now     = fg_now_seconds();
    const double elapsed = now - progress->start_time;
    progress->completed_frames++;
    if (elapsed > 0.0) {
        progress->last_fps = (double)progress->completed_frames / elapsed;
    }
    if (progress->total_frames > 0 && progress->completed_frames > 0 && progress->last_fps > 0.0) {
        const int remaining = progress->total_frames - progress->completed_frames;
        progress->eta_deadline = now + ((remaining > 0 ? remaining : 0) / progress->last_fps);
    }
    progress_unlock(progress);
}

static void progress_render(ProgressState* progress) {
    char   line[256];
    char   count_text[48];
    char   elapsed_text[32];
    char   eta_text[32];
    char   fps_text[32];
    char   bar[48];
    int    completed;
    int    total;
    int    previous_len;
    int    width;
    int    line_len;
    double start_time;
    double fps;
    double eta_deadline;
    double now;
    double elapsed;

    if (!progress->enabled) {
        return;
    }

    progress_snapshot(progress, &completed, &total, &start_time, &fps, &eta_deadline, &previous_len);

    now     = fg_now_seconds();
    elapsed = now - start_time;
    format_duration_clock(elapsed, elapsed_text, sizeof(elapsed_text));
    if (eta_deadline >= 0.0 && total >= 0 && completed < total) {
        format_duration_clock(eta_deadline - now, eta_text, sizeof(eta_text));
    } else if (total >= 0 && completed >= total) {
        snprintf(eta_text, sizeof(eta_text), "00:00");
    } else {
        snprintf(eta_text, sizeof(eta_text), "--:--");
    }
    if (fps > 0.0) {
        snprintf(fps_text, sizeof(fps_text), "%.2f fps", fps);
    } else {
        snprintf(fps_text, sizeof(fps_text), "-- fps");
    }
    if (total >= 0) {
        snprintf(count_text, sizeof(count_text), "%d/%d", completed, total);
    } else {
        snprintf(count_text, sizeof(count_text), "%d/?", completed);
    }

    width = fg_terminal_width();
    if (width >= 70 && total > 0) {
        const int bar_width = width >= 110 ? 30 : (width >= 84 ? 20 : 12);
        int       filled    = (int)((int64_t)completed * bar_width / total);
        if (filled < 0) {
            filled = 0;
        } else if (filled > bar_width) {
            filled = bar_width;
        }
        bar[0] = '[';
        for (int i = 0; i < bar_width; ++i) {
            bar[i + 1] = i < filled ? '#' : '-';
        }
        bar[bar_width + 1] = ']';
        bar[bar_width + 2] = '\0';
        snprintf(line,
                 sizeof(line),
                 "%s %s  %s  elapsed %s  eta %s",
                 bar,
                 count_text,
                 fps_text,
                 elapsed_text,
                 eta_text);
    } else {
        snprintf(line,
                 sizeof(line),
                 "%s  %s  elapsed %s  eta %s",
                 count_text,
                 fps_text,
                 elapsed_text,
                 eta_text);
    }

    line_len = (int)strlen(line);
    if (line_len >= width) {
        const int clipped = width - 1;
        if (clipped > 3) {
            line[clipped - 3] = '.';
            line[clipped - 2] = '.';
            line[clipped - 1] = '.';
            line[clipped]     = '\0';
            line_len          = clipped;
        } else {
            line[clipped] = '\0';
            line_len      = clipped;
        }
    }

    fprintf(stderr, "\r%s", line);
    for (int i = line_len; i < previous_len; ++i) {
        fputc(' ', stderr);
    }
    fflush(stderr);
    progress_set_last_line_len(progress, line_len);
}

static void progress_finish(ProgressState* progress) {
    if (progress->enabled && progress->last_line_len > 0) {
        progress_render(progress);
        fputc('\n', stderr);
        fflush(stderr);
        progress->last_line_len = 0;
    }
}

#if defined(_WIN32)
static DWORD WINAPI progress_monitor_thread(void* arg) {
    ProgressMonitor* monitor = (ProgressMonitor*)arg;
    while (WaitForSingleObject(monitor->stop_event, FG_PROGRESS_INTERVAL_MS) == WAIT_TIMEOUT) {
        progress_render(monitor->progress);
    }
    progress_render(monitor->progress);
    return 0;
}
#else
static void* progress_monitor_thread(void* arg) {
    ProgressMonitor* monitor = (ProgressMonitor*)arg;
    pthread_mutex_lock(&monitor->mutex);
    while (!monitor->stop) {
        struct timespec timeout;
        progress_render(monitor->progress);
        timespec_get(&timeout, TIME_UTC);
        timeout.tv_nsec += FG_PROGRESS_INTERVAL_MS * 1000000L;
        if (timeout.tv_nsec >= 1000000000L) {
            timeout.tv_sec++;
            timeout.tv_nsec -= 1000000000L;
        }
        pthread_cond_timedwait(&monitor->cond, &monitor->mutex, &timeout);
    }
    pthread_mutex_unlock(&monitor->mutex);
    progress_render(monitor->progress);
    return NULL;
}
#endif

static void progress_monitor_start(ProgressMonitor* monitor, ProgressState* progress) {
    memset(monitor, 0, sizeof(*monitor));
    monitor->progress = progress;
    if (!progress->enabled) {
        return;
    }
#if defined(_WIN32)
    monitor->stop_event = CreateEvent(NULL, TRUE, FALSE, NULL);
    if (!monitor->stop_event) {
        progress_render(progress);
        return;
    }
    monitor->thread  = CreateThread(NULL, 0, progress_monitor_thread, monitor, 0, NULL);
    monitor->started = monitor->thread != NULL;
    if (!monitor->started) {
        CloseHandle(monitor->stop_event);
        monitor->stop_event = NULL;
    }
#else
    if (pthread_mutex_init(&monitor->mutex, NULL) != 0) {
        progress_render(progress);
        return;
    }
    if (pthread_cond_init(&monitor->cond, NULL) != 0) {
        pthread_mutex_destroy(&monitor->mutex);
        progress_render(progress);
        return;
    }
    monitor->started = pthread_create(&monitor->thread, NULL, progress_monitor_thread, monitor) == 0;
    if (!monitor->started) {
        pthread_cond_destroy(&monitor->cond);
        pthread_mutex_destroy(&monitor->mutex);
    }
#endif
    if (!monitor->started) {
        progress_render(progress);
    }
}

static void progress_monitor_stop(ProgressMonitor* monitor) {
    if (!monitor->started) {
        return;
    }
#if defined(_WIN32)
    monitor->stop = 1;
    SetEvent(monitor->stop_event);
    WaitForSingleObject(monitor->thread, INFINITE);
    CloseHandle(monitor->thread);
    CloseHandle(monitor->stop_event);
    monitor->stop_event = NULL;
#else
    pthread_mutex_lock(&monitor->mutex);
    monitor->stop = 1;
    pthread_cond_signal(&monitor->cond);
    pthread_mutex_unlock(&monitor->mutex);
    pthread_join(monitor->thread, NULL);
    pthread_cond_destroy(&monitor->cond);
    pthread_mutex_destroy(&monitor->mutex);
#endif
    monitor->started = 0;
}

/* Estimation workers and serialized synthesis */
static int init_denoise_context(AomDenoiseAndModel* denoise_model, const Options* opt, const Frame8* frame) {
    DenoiseAndModelInitData init;
    memset(&init, 0, sizeof(init));
    init.noise_level          = (uint16_t)AOMMAX(0, opt->strength);
    init.encoder_bit_depth    = frame->bit_depth > 8 ? EB_TEN_BIT : EB_EIGHT_BIT;
    init.encoder_color_format = EB_YUV420;
    init.width                = (uint16_t)frame->model_width;
    init.height               = (uint16_t)frame->model_height;
    init.y_stride             = (uint16_t)frame->y_stride;
    init.u_stride             = (uint16_t)frame->uv_stride;
    init.v_stride             = (uint16_t)frame->uv_stride;
    init.denoise_apply        = 1;
    init.adaptive_film_grain  = opt->adaptive ? 1 : 0;
    if (svt_aom_denoise_and_model_ctor(denoise_model, (EbPtr)&init) != EB_ErrorNone) {
        fprintf(stderr, "Unable to initialize SVT denoise/model context\n");
        return 0;
    }
    return 1;
}

static int work_slot_init(WorkSlot* slot, const Options* opt, const VideoInfo* info, int have_target) {
    memset(slot, 0, sizeof(*slot));
    if (!frame_alloc(&slot->source_frame, info->width, info->height, info->bit_depth)) {
        return 0;
    }
    if (have_target && !frame_alloc(&slot->target_frame, info->width, info->height, info->bit_depth)) {
        return 0;
    }
    slot->fast = opt->fast;
    return init_denoise_context(&slot->denoise_model, opt, &slot->source_frame);
}

static void work_slot_free(WorkSlot* slot) {
    if (slot->denoise_model.dctor) {
        slot->denoise_model.dctor(&slot->denoise_model);
    }
    frame_free(&slot->source_frame);
    frame_free(&slot->target_frame);
    memset(slot, 0, sizeof(*slot));
}

/* pic_analysis_process.c pads to MIN_BLOCK_SIZE before film-grain estimation.
 * Reuse the same padding kernels; crop only when writing the visible frame.
 */
static void pad_model_frame(Frame8* frame, uint16_t* packed[3]) {
    if (frame->width == frame->model_width && frame->height == frame->model_height) {
        return;
    }
    uint8_t* planes[3]     = {frame->y, frame->u, frame->v};
    uint8_t* low_planes[3] = {frame->y2, frame->u2, frame->v2};
    if (frame->bit_depth > 8) {
        EbPictureBufferDesc pic = make_picture_desc(frame);
        svt_aom_pack_2d_pic(&pic, packed);
    }
    for (int c = 0; c < 3; ++c) {
        const int sub        = c > 0;
        const int width      = frame->width >> sub;
        const int height     = frame->height >> sub;
        const int stride     = c ? frame->uv_stride : frame->y_stride;
        const int pad_right  = (frame->model_width - frame->width) >> sub;
        const int pad_bottom = (frame->model_height - frame->height) >> sub;
        if (frame->bit_depth > 8) {
            svt_aom_pad_input_picture_16bit(packed[c], stride, width, height, pad_right, pad_bottom);
            svt_unpack_and_2bcompress_c(packed[c],
                                         stride,
                                         planes[c],
                                         stride,
                                         low_planes[c],
                                         stride / 4,
                                         width + pad_right,
                                         height + pad_bottom);
        } else {
            pad_input_picture(planes[c], stride, width, height, pad_right, pad_bottom);
        }
    }
}

static void estimate_slot(WorkSlot* slot, int use_external_denoised) {
    EbPictureBufferDesc source_pic;
    EbPictureBufferDesc target_pic;
    slot->ok       = 0;
    slot->error[0] = '\0';

    pad_model_frame(&slot->source_frame, slot->denoise_model.packed);
    source_pic = make_picture_desc(&slot->source_frame);
    if (use_external_denoised) {
        pad_model_frame(&slot->target_frame, slot->denoise_model.packed);
        target_pic = make_picture_desc(&slot->target_frame);
        if (!svt_aom_denoise_and_model_run_with_external_denoised(
                &slot->denoise_model, &source_pic, &target_pic, &slot->grain, slot->source_frame.bit_depth > 8)) {
            snprintf(slot->error, sizeof(slot->error), "SVT grain estimation failed at frame %d", slot->frame_index);
            return;
        }
    } else {
        // Tool --fast dispatch; both paths retain the input bit depth and upstream math.
        int32_t (*run)(struct AomDenoiseAndModel*, EbPictureBufferDesc*, AomFilmGrain*, int32_t) = slot->fast
            ? svt_aom_denoise_and_model_run_fast : svt_aom_denoise_and_model_run;
        if (!run(&slot->denoise_model, &source_pic, &slot->grain, slot->source_frame.bit_depth > 8)) {
            snprintf(slot->error, sizeof(slot->error), "SVT grain estimation failed at frame %d", slot->frame_index);
            return;
        }
    }
    slot->grain.bit_depth = slot->source_frame.bit_depth;
    slot->ok              = 1;
}

static void estimate_thread_execute(EstimateThread* thread) {
    estimate_slot(thread->slot, thread->use_external_denoised);
    if (thread->slot->ok) {
        progress_note_frame(thread->progress);
    }
}

#if defined(_WIN32)
static DWORD WINAPI estimate_thread_main(void* arg) {
    estimate_thread_execute((EstimateThread*)arg);
    return 0;
}
#else
static void* estimate_thread_main(void* arg) {
    estimate_thread_execute((EstimateThread*)arg);
    return NULL;
}
#endif

static int estimate_thread_start(EstimateThread* thread, WorkSlot* slot, ProgressState* progress,
                                 int use_external_denoised) {
    memset(thread, 0, sizeof(*thread));
    thread->slot                  = slot;
    thread->progress              = progress;
    thread->use_external_denoised = use_external_denoised;
#if defined(_WIN32)
    thread->thread  = CreateThread(NULL, 0, estimate_thread_main, thread, 0, NULL);
    thread->started = thread->thread != NULL;
#else
    thread->started = pthread_create(&thread->thread, NULL, estimate_thread_main, thread) == 0;
#endif
    return thread->started;
}

static void estimate_thread_join(EstimateThread* thread) {
    if (!thread->started) {
        return;
    }
#if defined(_WIN32)
    WaitForSingleObject(thread->thread, INFINITE);
    CloseHandle(thread->thread);
    thread->thread = NULL;
#else
    pthread_join(thread->thread, NULL);
#endif
    thread->started = 0;
}

static void synthesize_frame(Frame8* synth_frame, PackedFrame10* synth_packed, const AomFilmGrain* grain) {
    if (!grain->apply_grain) {
        return;
    }
    if (synth_frame->bit_depth > 8) {
        frame_to_packed10(synth_frame, synth_packed);
        svt_av1_add_film_grain_run(grain,
                                   (uint8_t*)synth_packed->y,
                                   (uint8_t*)synth_packed->u,
                                   (uint8_t*)synth_packed->v,
                                   synth_packed->height,
                                   synth_packed->width,
                                   synth_packed->y_stride,
                                   synth_packed->uv_stride,
                                   1,
                                   1,
                                   1);
        frame_from_packed10(synth_packed, synth_frame);
    } else {
        svt_av1_add_film_grain_run(grain,
                                   synth_frame->y,
                                   synth_frame->u,
                                   synth_frame->v,
                                   synth_frame->height,
                                   synth_frame->width,
                                   synth_frame->y_stride,
                                   synth_frame->uv_stride,
                                   0,
                                   1,
                                   1);
    }
}

static void estimate_batch(WorkSlot* slots, int batch_count, ProgressState* progress, int use_external_denoised) {
    ProgressMonitor monitor;
    EstimateThread* threads = NULL;

    progress_monitor_start(&monitor, progress);

    if (batch_count > 1) {
        threads = (EstimateThread*)calloc((size_t)batch_count, sizeof(*threads));
    }

    if (threads) {
        for (int i = 0; i < batch_count; ++i) {
            if (!estimate_thread_start(&threads[i], &slots[i], progress, use_external_denoised)) {
                estimate_slot(&slots[i], use_external_denoised);
                if (slots[i].ok) {
                    progress_note_frame(progress);
                }
            }
        }
        for (int i = 0; i < batch_count; ++i) {
            estimate_thread_join(&threads[i]);
        }
        free(threads);
    } else {
        for (int i = 0; i < batch_count; ++i) {
            estimate_slot(&slots[i], use_external_denoised);
            if (slots[i].ok) {
                progress_note_frame(progress);
            }
        }
    }

    progress_monitor_stop(&monitor);
}

int main(int argc, char** argv) {
    Options        opt;
    VideoReader    source;
    VideoReader    target;
    VideoWriter    output;
    VideoWriter    denoised_output;
    TimestampCache timestamps;
    FILE*          table_file;
    Frame8         synth_frame;
    PackedFrame10  synth_packed;
    WorkSlot*      slots;
    SvtFgaContext* automatic     = NULL;
    Frame8         lookahead     = {0};
    Frame8         future        = {0};
    int            next_status   = 0;
    int            future_status = 0;
    int            slot_count    = 0;
    ProgressState  progress;
    int            progress_started     = 0;
    int            have_target          = 0;
    int            have_output          = 0;
    int            have_denoised_output = 0;
    int            processed            = 0;
    int            estimates            = 0;
    uint16_t       default_seed;
    AomFilmGrain   table_grain;
    int            table_start_frame = -1;
    int            exit_code = 1;

    ProgressMonitor auto_monitor = {0};

    memset(&source, 0, sizeof(source));
    memset(&target, 0, sizeof(target));
    memset(&output, 0, sizeof(output));
    memset(&denoised_output, 0, sizeof(denoised_output));
    memset(&timestamps, 0, sizeof(timestamps));
    table_file = NULL;
    memset(&synth_frame, 0, sizeof(synth_frame));
    memset(&synth_packed, 0, sizeof(synth_packed));
    memset(&progress, 0, sizeof(progress));
    slots = NULL;

    if (argc == 1) {
        usage(stdout, argv[0], 0);
        return 0;
    }

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(stdout, argv[0], 1);
            return 0;
        }
    }

    if (!parse_options(argc, argv, &opt)) {
        free(opt.seeds);
        free(opt.keyframes);
        return 1;
    }

    svt_fg_tool_init_svt_shims(opt.fast);

    if (!reader_open(&source, opt.source_path, opt.width, opt.height, opt.bit_depth, opt.source_format)) {
        goto cleanup;
    }
    if (!set_table_config(&opt, &source.info)) {
        goto cleanup;
    }
    if (opt.frames < 0 && source.info.container == VIDEO_Y4M && !source.is_pipe) {
        int inferred_frames = -1;
        if (!reader_infer_y4m_frame_count(&source, &inferred_frames)) {
            goto cleanup;
        }
        opt.frames = inferred_frames;
    }
#if HAVE_FFMS2
    if (opt.frames < 0 && source.info.container == VIDEO_FFMS2) {
        opt.frames = source.ffms_num_frames;
    }
#endif
    if (opt.target_path) {
        have_target = 1;
        if (!reader_open(&target, opt.target_path, source.info.width, source.info.height,
                         source.info.bit_depth, opt.target_format)) {
            goto cleanup;
        }
        if (target.info.container == VIDEO_RAW) {
            target.info.fps_num = source.info.fps_num;
            target.info.fps_den = source.info.fps_den;
        }
        if (source.is_pipe && target.is_pipe) {
            fprintf(stderr, "Only one of --source and --target may read from a pipe or stdin\n");
            goto cleanup;
        }
        if (!same_dimensions(&source.info, &target.info)) {
            fprintf(stderr, "Source and target dimensions differ\n");
            goto cleanup;
        }
        if (source.info.bit_depth != target.info.bit_depth) {
            fprintf(stderr, "Source and target bit depths differ\n");
            goto cleanup;
        }
    }
    if (!timestamp_cache_init(&timestamps, &source.info)) {
        goto cleanup;
    }

    const int auto_mode  = opt.strength == -1 && !have_target;
    VideoInfo model_info = source.info;
    timestamps.round_down = auto_mode;
    if (auto_mode) {
        model_info.bit_depth = 10;
    }
    slot_count          = auto_mode ? 1 : opt.jobs;
    slots               = (WorkSlot*)calloc((size_t)slot_count, sizeof(*slots));
    if (!slots) {
        fprintf(stderr, "Unable to allocate work slots\n");
        goto cleanup;
    }
    for (int i = 0; i < slot_count; ++i) {
        if (!work_slot_init(&slots[i], &opt, &model_info, have_target)) {
            goto cleanup;
        }
    }

    if (auto_mode) {
        automatic = svt_aom_fga_create(
            slots[0].source_frame.model_width, slots[0].source_frame.model_height, model_info.bit_depth);
        if (!automatic || !frame_alloc(&lookahead, source.info.width, source.info.height, model_info.bit_depth) ||
            !frame_alloc(&future, source.info.width, source.info.height, model_info.bit_depth)) {
            goto cleanup;
        }
        next_status = frame_read(&source, &lookahead);
        if (next_status < 0) {
            goto cleanup;
        }
        future_status = next_status && opt.frames != 1 ? frame_read(&source, &future) : 0;
        if (future_status < 0) {
            goto cleanup;
        }
    }

    if (opt.output_path) {
        const VideoInfo* ref = have_target ? &target.info : &source.info;
        have_output = 1;
        if (!frame_alloc(&synth_frame, source.info.width, source.info.height, model_info.bit_depth) ||
            (model_info.bit_depth > 8 && !packed_frame10_alloc(&synth_packed, source.info.width, source.info.height)) ||
            !writer_open(&output, opt.output_path, ref)) {
            goto cleanup;
        }
    }
    if (opt.denoised_output_path) {
        have_denoised_output = 1;
        if (!writer_open(&denoised_output, opt.denoised_output_path, &source.info)) {
            goto cleanup;
        }
    }
    if (opt.table_path && !open_fgs_table(opt.table_path, &table_file)) {
        goto cleanup;
    }

    progress_init(&progress, opt.frames);
    progress_started = 1;
    progress_render(&progress);
    if (auto_mode) {
        progress_monitor_start(&auto_monitor, &progress);
    }

    default_seed = opt.default_seed;
    while (opt.frames < 0 || processed < opt.frames) {
        int batch_count = 0;

        if (auto_mode) {
            if (!next_status) {
                break;
            }
            WorkSlot*    slot      = &slots[0];
            const Frame8 old_frame = slot->source_frame;
            slot->source_frame     = lookahead;
            lookahead              = future;
            future                 = old_frame;
            next_status            = future_status;
            future_status = next_status && (opt.frames < 0 || processed + 2 < opt.frames)
                ? frame_read(&source, &future)
                : 0;
            if (future_status < 0) {
                goto cleanup;
            }
            memset(&slot->grain, 0, sizeof(slot->grain));
            slot->grain.random_seed = seed_for_frame(&opt, processed, &default_seed);
            slot->frame_index       = processed++;
            pad_model_frame(&slot->source_frame, slot->denoise_model.packed);
            // The encoder's PD window excludes the terminal next picture.
            const int have_next = next_status && future_status && opt.svt_ra_gop_size != 1;
            if (have_next) {
                pad_model_frame(&lookahead, slot->denoise_model.packed);
            }
            EbPictureBufferDesc pic  = make_picture_desc(&slot->source_frame);
            EbPictureBufferDesc next = make_picture_desc(&lookahead);
            const int scene_cut = opt.keyframe_count &&
                bsearch(&slot->frame_index, opt.keyframes, opt.keyframe_count, sizeof(*opt.keyframes), compare_ints);
            slot->ok = svt_aom_fga_run(automatic,
                                      &pic,
                                      have_next ? &next : NULL,
                                      scene_cut,
                                      opt.output_path || opt.denoised_output_path,
                                      &slot->grain);
            if (!slot->ok) {
                snprintf(slot->error,
                         sizeof(slot->error),
                         "Automatic grain estimation failed at frame %d",
                         slot->frame_index);
            } else {
                progress_note_frame(&progress);
            }
            batch_count = 1;
        }

        for (; !auto_mode && batch_count < slot_count && (opt.frames < 0 || processed < opt.frames); ++batch_count) {
            WorkSlot* slot = &slots[batch_count];
            int       read_status;

            read_status = frame_read(&source, &slot->source_frame);
            if (read_status == 0) {
                break;
            }
            if (read_status < 0) {
                goto cleanup;
            }
            if (have_target) {
                read_status = frame_read(&target, &slot->target_frame);
                if (read_status == 0) {
                    fprintf(stderr, "Target ended before source at frame %d\n", processed);
                    goto cleanup;
                }
                if (read_status < 0) {
                    goto cleanup;
                }
            }

            memset(&slot->grain, 0, sizeof(slot->grain));
            slot->grain.random_seed = seed_for_frame(&opt, processed, &default_seed);
            slot->frame_index       = processed;
            slot->ok                = 0;
            slot->error[0]          = '\0';
            ++processed;
        }

        if (batch_count == 0) {
            break;
        }

        if (!auto_mode) {
            estimate_batch(slots, batch_count, &progress, have_target);
        }

        for (int i = 0; i < batch_count; ++i) {
            WorkSlot* slot = &slots[i];
            if (!slot->ok) {
                fprintf(stderr, "%s\n", slot->error[0] ? slot->error : "SVT grain estimation failed");
                goto cleanup;
            }

            if (slot->grain.apply_grain) {
                ++estimates;
            }
            if (table_file && (auto_mode || should_write_table_event(&opt, slot->frame_index))) {
                if (table_start_frame >= 0 &&
                    !write_fgs_table_entry(table_file, &timestamps, table_start_frame, slot->frame_index, &table_grain)) {
                    fprintf(stderr, "Unable to write grain table\n");
                    goto cleanup;
                }
                table_grain       = slot->grain;
                table_start_frame = slot->frame_index;
            }

            if (have_denoised_output &&
                !frame_write(&denoised_output, have_target ? &slot->target_frame : &slot->source_frame)) {
                fprintf(stderr, "Unable to write denoised frame %d\n", slot->frame_index);
                goto cleanup;
            }

            if (have_output) {
                frame_copy(&synth_frame, have_target ? &slot->target_frame : &slot->source_frame);
                synthesize_frame(&synth_frame, &synth_packed, &slot->grain);
                if (!frame_write(&output, &synth_frame)) {
                    fprintf(stderr, "Unable to write output frame %d\n", slot->frame_index);
                    goto cleanup;
                }
            }
        }
    }

    // The final interval must stop at EOF even when the input length was unknown.
    if (table_file && table_start_frame >= 0 &&
        !write_fgs_table_entry(table_file,
                               &timestamps,
                               table_start_frame,
                               processed,
                               &table_grain)) {
        fprintf(stderr, "Unable to write grain table\n");
        goto cleanup;
    }
    progress_monitor_stop(&auto_monitor);
    progress_finish(&progress);
    {
        char duration_text[32];
        format_duration(fg_now_seconds() - progress.start_time, duration_text, sizeof(duration_text));
        fprintf(stderr,
                "Processed %d frame(s) in %s; %d frame(s) produced grain parameters.\n",
                processed,
                duration_text,
                estimates);
    }
    exit_code = 0;

cleanup:
    progress_monitor_stop(&auto_monitor);
    svt_aom_fga_free(automatic);
    frame_free(&lookahead);
    frame_free(&future);
    if (progress_started && exit_code != 0) {
        progress_finish(&progress);
    }
    if (progress_started) {
        progress_destroy(&progress);
    }
    if (slots) {
        for (int i = 0; i < slot_count; ++i) {
            work_slot_free(&slots[i]);
        }
        free(slots);
    }
    if (table_file && fclose(table_file)) {
        fprintf(stderr, "Unable to close grain table %s\n", opt.table_path);
        exit_code = 1;
    }
    frame_free(&synth_frame);
    packed_frame10_free(&synth_packed);
    reader_close(&source);
    reader_close(&target);
    if (!writer_close(&output)) {
        exit_code = 1;
    }
    if (!writer_close(&denoised_output)) {
        exit_code = 1;
    }
    timestamp_cache_free(&timestamps);
    free(opt.seeds);
    free(opt.keyframes);
    if (exit_code == 0 && opt.table_path) {
        fprintf(stderr, "Wrote fgs table: %s\n", opt.table_path);
    }
    return exit_code;
}
