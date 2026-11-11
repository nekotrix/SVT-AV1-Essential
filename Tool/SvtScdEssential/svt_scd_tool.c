/*
Copyright(c) 2019 Intel Corporation (extracted scalar/pre-analysis code).
BSD 3-Clause Clear License
The Clear BSD License

Copyright (c) 2021, Alliance for Open Media

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted (subject to the limitations in the disclaimer below)
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in
   the documentation and/or other materials provided with the distribution.

3. Neither the name of the Alliance for Open Media nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

BSD 2-Clause License (SIMD kernels)
Copyright (c) 2017, Alliance for Open Media. All rights reserved.
Copyright(c) 2019 Intel Corporation.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

*/

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#if HAVE_FFMS2
#include <ffms.h>
#endif
#ifdef _WIN32
#define strcasecmp _stricmp
#else
#include <strings.h>
#endif
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#define SCD_ISATTY(stream) _isatty(_fileno(stream))
#else
#include <pthread.h>
#include <unistd.h>
#define SCD_ISATTY(stream) isatty(fileno(stream))
#endif
#include "svt_scd_wrappers.h"
#include "pic_analysis_process.h"

#define SIMD_GUARD_BYTES 32u
#define MAX_INTRA_PERIOD (INT32_MAX - 1)
#define IO_BUFFER_SIZE 65536
#define SCD_PROGRESS_INTERVAL_MS 500

typedef struct VideoInfo {
    unsigned width;
    unsigned height;
    unsigned fps_num;
    unsigned fps_den;
    unsigned bit_depth;
    unsigned bytes_per_sample;
    size_t chroma_bytes_per_frame;
    char chroma_tag[64];
} VideoInfo;

typedef struct VideoReader {
    FILE *file;
#if HAVE_FFMS2
    FFMS_Index *index;
    FFMS_VideoSource *video_source;
    int frame_number;
    int frame_count;
    int pixel_format;
#endif
} VideoReader;

typedef struct KeyframeList {
    uint64_t *frame;
    size_t count;
    size_t capacity;
} KeyframeList;

typedef struct Options {
    const char *input_path;
    const char *output_path;
    int64_t keyint;
    int64_t min_keyint;
    int multiply_keyint;
    int allintra;
    int preset;
    int hierarchical_levels;
    int intra_refresh_type;
    int enable_tf;
} Options;

typedef struct ProgressState {
    int     enabled;
    int64_t total_frames;
    int64_t completed_frames;
    int     last_line_len;
    double  start_time;
    double  last_fps;
    double  eta_deadline;
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

static void usage(FILE *fp, const char *argv0, int full) {
    fprintf(fp,
        "SVT-SCD-Essential\n\n"
        "Usage: %s [input] [output.cfg] [options]\n"
        "\n"
        "Basic options:\n"
        "  --keyint N            Periodic keyframes: -2=automatic, -1/0=off, >0=frame interval\n"
        "                        Auto follows frame rate and the selected mini-GOP; Ns uses seconds.\n"
        "  --min-keyint N        Minimum keyframe spacing: -1=automatic, 0=off, >0=frames\n"
        "                        Auto follows frame rate and the selected mini-GOP.\n"
        "  -h, --help            Show the full help\n",
        argv0);
    if (!full) {
        return;
    }
    fprintf(fp,
        "\n"
        "Advanced options:\n"
        "  --preset N            Encoder preset (0..13, default 8)\n"
        "  --hierarchical-levels N\n"
        "                        Hierarchy (2..5, default follows preset and resolution)\n"
        "  --irefresh-type N     Keyframe type (1=forward, 2=closed, default 2)\n"
        "  --enable-tf N         Encoder temporal-filtering setting (0..3, default 1)\n"
        "\n"
        "Stdin, pipes and .y4m files use Y4M; other file extensions select FFMS2.\n"
#if HAVE_FFMS2
        "FFMS2 support: enabled.\n"
#else
        "FFMS2 support: disabled in this build.\n"
#endif
        "\n"
        "Writes an SVT-AV1 config line, like: ForceKeyFrames : 0f,123f,456f\n");
}

/* Keep parser safety checks and the intentional minimum-zero exception
 * outside svt_scd_str_to_keyint(), so the upstream function stays byte-for-byte intact.
 */
static int parse_keyint(const char *s, int64_t *out, int *multiply, int minimum) {
    char *end;
    long value;
    int32_t parsed;
    bool multi = *multiply != 0;

    errno = 0;
    value = strtol(s, &end, 0);
    if (errno || end == s || value > INT32_MAX || value < -2)
        return 0;
    if (*end != '\0' && !(*end == 's' && end[1] == '\0'))
        return 0;
    if (svt_scd_str_to_keyint(s, &parsed, &multi) != EB_ErrorNone)
        return 0;
    if (minimum && value == 0 && *end == '\0')
        parsed = 0;
    *out = parsed;
    *multiply = multi;
    return 1;
}

static int parse_uint(const char *s, unsigned *out) {
    char *end = NULL;
    unsigned long v;
    errno = 0;
    v = strtoul(s, &end, 10);
    if (errno || end == s || *end != '\0' || v > UINT_MAX)
        return 0;
    *out = (unsigned)v;
    return 1;
}

/* Progress display */
static double scd_now_seconds(void) {
#if defined(_WIN32)
    return (double)GetTickCount64() * 0.001;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static int scd_terminal_width(void) {
    const char* columns = getenv("COLUMNS");
    int         width   = 80;
    if (columns && *columns) {
        unsigned parsed = 0;
        if (parse_uint(columns, &parsed) && parsed >= 20 && parsed <= 512) {
            width = parsed;
        }
    }
    return width < 20 ? 20 : width;
}

static int scd_progress_enabled(void) {
    const char* env = getenv("SVT_SCD_PROGRESS");
    if (env && *env) {
        if (!strcmp(env, "0") || !strcmp(env, "false") || !strcmp(env, "FALSE") || !strcmp(env, "off") ||
            !strcmp(env, "OFF")) {
            return 0;
        }
        return 1;
    }
    return SCD_ISATTY(stderr);
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

static void progress_init(ProgressState* progress, int64_t total_frames) {
    memset(progress, 0, sizeof(*progress));
    progress->enabled          = scd_progress_enabled();
    progress->total_frames     = total_frames;
    progress->completed_frames = 0;
    progress->start_time       = scd_now_seconds();
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

static void progress_snapshot(ProgressState* progress, int64_t* completed, int64_t* total, double* start_time, double* fps,
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
    const double now     = scd_now_seconds();
    const double elapsed = now - progress->start_time;
    progress->completed_frames++;
    if (elapsed > 0.0) {
        progress->last_fps = (double)progress->completed_frames / elapsed;
    }
    if (progress->total_frames > 0 && progress->completed_frames > 0 && progress->last_fps > 0.0) {
        const int64_t remaining = progress->total_frames - progress->completed_frames;
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
    int64_t completed;
    int64_t total;
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

    now     = scd_now_seconds();
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
        snprintf(count_text, sizeof(count_text), "%" PRId64 "/%" PRId64, completed, total);
    } else {
        snprintf(count_text, sizeof(count_text), "%" PRId64 "/?", completed);
    }

    width = scd_terminal_width();
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
    while (WaitForSingleObject(monitor->stop_event, SCD_PROGRESS_INTERVAL_MS) == WAIT_TIMEOUT) {
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
        timeout.tv_nsec += SCD_PROGRESS_INTERVAL_MS * 1000000L;
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

static int parse_options(int argc, char **argv, Options *opt) {
    int positional = 0;
    const char *keyint_value = NULL, *min_keyint_value = NULL;
    int i;
    opt->input_path = "-";
    opt->output_path = "-";
    opt->keyint = -2;
    opt->min_keyint = -1;
    opt->multiply_keyint = 0;
    opt->allintra = 0;
    opt->preset = 8;
    opt->hierarchical_levels = -1;
    opt->intra_refresh_type = 2;
    opt->enable_tf = 1;

    for (i = 1; i < argc; ++i) {
        const char *a = argv[i];
        const char *value = NULL;
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout, argv[0], 1);
            exit(0);
        }
        if (!strcmp(a, "--preset") || !strcmp(a, "--hierarchical-levels") ||
            !strcmp(a, "--irefresh-type") || !strcmp(a, "--enable-tf")) {
            unsigned parsed;
            const unsigned minimum = !strcmp(a, "--hierarchical-levels") ? 2 : !strcmp(a, "--irefresh-type") ? 1 : 0;
            const unsigned maximum = !strcmp(a, "--preset") ? 13 : !strcmp(a, "--hierarchical-levels") ? 5 :
                !strcmp(a, "--irefresh-type") ? 2 : 3;
            if (++i >= argc || !parse_uint(argv[i], &parsed) || parsed < minimum || parsed > maximum) {
                fprintf(stderr, "Invalid %s value\n", a);
                return 0;
            }
            if (!strcmp(a, "--preset")) opt->preset = (int)parsed;
            else if (!strcmp(a, "--hierarchical-levels")) opt->hierarchical_levels = (int)parsed;
            else if (!strcmp(a, "--irefresh-type")) opt->intra_refresh_type = (int)parsed;
            else opt->enable_tf = (int)parsed;
            continue;
        }
        if (!strcmp(a, "--keyint") || !strcmp(a, "--min-keyint")) {
            if (++i >= argc) {
                fprintf(stderr, "Missing value after %s\n", a);
                return 0;
            }
            value = argv[i];
            if (!strcmp(a, "--keyint"))
                keyint_value = value;
            else
                min_keyint_value = value;
            continue;
        }
        if (a[0] == '-') {
            if (!strcmp(a, "-")) {
                /* positional stdin/stdout marker */
            } else {
                fprintf(stderr, "Unknown option: %s\n", a);
                return 0;
            }
        }
        if (positional == 0)
            opt->input_path = a;
        else if (positional == 1)
            opt->output_path = a;
        else {
            fprintf(stderr, "Too many positional arguments: %s\n", a);
            return 0;
        }
        ++positional;
    }

    /* The encoder CLI visits its option table in keyint, min-keyint order,
     * independently of argv order. This matters for its shared seconds flag.
     */
    if (keyint_value && !parse_keyint(keyint_value, &opt->keyint,
                                      &opt->multiply_keyint, 0)) {
        fprintf(stderr, "Invalid --keyint value: %s\n", keyint_value);
        return 0;
    }
    if (min_keyint_value && !parse_keyint(min_keyint_value, &opt->min_keyint,
                                          &opt->multiply_keyint, 1)) {
        fprintf(stderr, "Invalid --min-keyint value: %s\n", min_keyint_value);
        return 0;
    }
    return 1;
}

static int starts_with(const char *s, const char *prefix) {
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

static int read_y4m_header(FILE *in, VideoInfo *info) {
    char line[4096];
    char *tok;
    unsigned w = 0, h = 0;
    unsigned fps_num = 0, fps_den = 1;
    unsigned depth = 8;
    unsigned sub_x = 2, sub_y = 2;
    int have_chroma = 0;

    if (!fgets(line, sizeof(line), in)) {
        fprintf(stderr, "Unable to read Y4M header\n");
        return 0;
    }
    if (!starts_with(line, "YUV4MPEG2")) {
        fprintf(stderr, "Input is not a YUV4MPEG2 stream\n");
        return 0;
    }

    for (tok = strtok(line + 9, " \t\r\n"); tok; tok = strtok(NULL, " \t\r\n")) {
        if (tok[0] == 'W') {
            if (!parse_uint(tok + 1, &w)) {
                fprintf(stderr, "Invalid Y4M width token: %s\n", tok);
                return 0;
            }
        } else if (tok[0] == 'H') {
            if (!parse_uint(tok + 1, &h)) {
                fprintf(stderr, "Invalid Y4M height token: %s\n", tok);
                return 0;
            }
        } else if (tok[0] == 'F') {
            char *colon = strchr(tok + 1, ':');
            if (colon) {
                *colon = '\0';
                if (!parse_uint(tok + 1, &fps_num) || !parse_uint(colon + 1, &fps_den) || fps_den == 0) {
                    fprintf(stderr, "Invalid Y4M frame-rate token\n");
                    return 0;
                }
            }
        } else if (tok[0] == 'C') {
            have_chroma = 1;
            if (strlen(tok) >= sizeof(info->chroma_tag)) {
                fprintf(stderr, "Y4M chroma tag is too long\n");
                return 0;
            }
            strcpy(info->chroma_tag, tok);
            if (strstr(tok, "p10")) depth = 10;
            else if (strstr(tok, "p12") || strstr(tok, "p16")) depth = 16;
            else depth = 8;

            /* This standalone port targets the usual encoder input formats only. */
            if (!starts_with(tok, "C420")) {
                fprintf(stderr, "Only YUV420 Y4M input is supported (got %s)\n", tok);
                return 0;
            }
            sub_x = 2;
            sub_y = 2;
        }
    }

    if (!w || !h) {
        fprintf(stderr, "Y4M header is missing a valid W or H token\n");
        return 0;
    }
    /* Match SVT-AV1-Essential's supported visible input limits. */
    if (w > 16384 || h > 8704 || (uint64_t)w * h > SIZE_MAX / 4) {
        fprintf(stderr, "Y4M frame dimensions exceed the supported limits\n");
        return 0;
    }
    /* Y4M's omitted C tag means 8-bit 4:2:0. */
    if (!have_chroma) {
        strcpy(info->chroma_tag, "C420");
        sub_x = 2;
        sub_y = 2;
        depth = 8;
    }
    if (depth != 8 && depth != 10) {
        fprintf(stderr, "Only 8-bit and 10-bit YUV420 Y4M input is supported\n");
        return 0;
    }

    info->width = w;
    info->height = h;
    info->fps_num = fps_num;
    info->fps_den = fps_den;
    info->bit_depth = depth;
    info->bytes_per_sample = depth > 8 ? 2u : 1u;
    if (starts_with(info->chroma_tag, "Cmono")) {
        info->chroma_bytes_per_frame = 0;
    } else {
        size_t cw = ((size_t)w + sub_x - 1) / sub_x;
        size_t ch = ((size_t)h + sub_y - 1) / sub_y;
        info->chroma_bytes_per_frame = 2u * cw * ch * info->bytes_per_sample;
    }
    return 1;
}

static int read_exact(FILE *fp, void *buffer, size_t bytes) {
    unsigned char *p = (unsigned char *)buffer;
    while (bytes) {
        size_t n = fread(p, 1, bytes, fp);
        if (n == 0)
            return 0;
        p += n;
        bytes -= n;
    }
    return 1;
}

static int skip_exact(FILE *fp, size_t bytes) {
    unsigned char scratch[IO_BUFFER_SIZE];
    while (bytes) {
        size_t want = bytes < sizeof(scratch) ? bytes : sizeof(scratch);
        size_t n = fread(scratch, 1, want, fp);
        if (n == 0)
            return 0;
        bytes -= n;
    }
    return 1;
}

static void pad_input_frame(unsigned char *padded_y, unsigned width, unsigned height,
                            unsigned padded_width, unsigned padded_height) {
    for (unsigned y = 0; y < height; ++y) {
        unsigned char *row = padded_y + (size_t)y * padded_width;
        memset(row + width, row[width - 1], padded_width - width);
    }
    for (unsigned y = height; y < padded_height; ++y)
        memcpy(padded_y + (size_t)y * padded_width,
               padded_y + (size_t)(height - 1) * padded_width, padded_width);
}

static int read_y4m_frame(FILE *in, const VideoInfo *info, unsigned char *raw_y,
                          unsigned char *padded_y, unsigned padded_width,
                          unsigned padded_height) {
    char line[4096];
    size_t samples = (size_t)info->width * info->height;
    size_t y_bytes = samples * info->bytes_per_sample;
    size_t i;
    unsigned y;

    if (!fgets(line, sizeof(line), in)) {
        if (feof(in)) return 0;
        fprintf(stderr, "Failed reading Y4M frame marker\n");
        return -1;
    }
    if (!starts_with(line, "FRAME")) {
        fprintf(stderr, "Expected Y4M FRAME marker; got: %.64s\n", line);
        return -1;
    }
    if (!read_exact(in, raw_y, y_bytes)) {
        fprintf(stderr, "Truncated Y4M luma plane\n");
        return -1;
    }
    if (!skip_exact(in, info->chroma_bytes_per_frame)) {
        fprintf(stderr, "Truncated Y4M chroma planes\n");
        return -1;
    }

    if (info->bytes_per_sample == 1) {
        for (y = 0; y < info->height; ++y) {
            unsigned char *dst = padded_y + (size_t)y * padded_width;
            const unsigned char *src = raw_y + (size_t)y * info->width;
            memcpy(dst, src, info->width);
        }
    } else {
        const unsigned shift = info->bit_depth - 8;
        for (i = 0; i < samples; ++i) {
            /* High-bit-depth Y4M samples are little-endian words. */
            unsigned v = (unsigned)raw_y[2 * i] | ((unsigned)raw_y[2 * i + 1] << 8);
            raw_y[i] = (unsigned char)((v >> shift) > 255u ? 255u : (v >> shift));
        }
        for (y = 0; y < info->height; ++y) {
            unsigned char *dst = padded_y + (size_t)y * padded_width;
            const unsigned char *src = raw_y + (size_t)y * info->width;
            memcpy(dst, src, info->width);
        }
    }
    pad_input_frame(padded_y, info->width, info->height, padded_width, padded_height);
    return 1;
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

static int64_t infer_y4m_frame_count(FILE *file, const VideoInfo *info) {
    fpos_t original_pos;
    char line[4096];
    const uint64_t payload_size = (uint64_t)info->width * info->height * info->bytes_per_sample +
        info->chroma_bytes_per_frame;
    int64_t count = 0;

    if (fgetpos(file, &original_pos)) return -1;
    for (;;) {
        if (!fgets(line, sizeof(line), file)) {
            if (ferror(file)) count = -1;
            break;
        }
        if (!starts_with(line, "FRAME") || !seek_forward_bytes(file, payload_size) || count == INT64_MAX) {
            count = -1;
            break;
        }
        ++count;
    }
    if (fsetpos(file, &original_pos)) return -1;
    return count;
}

static int input_is_pipe(const char *path) {
    if (!strcmp(path, "-")) return 1;
#ifndef _WIN32
    struct stat st;
    return !stat(path, &st) && S_ISFIFO(st.st_mode);
#else
    return !strncmp(path, "\\\\.\\pipe\\", 9);
#endif
}

#if HAVE_FFMS2
static int init_ffms2(VideoReader *reader, const char *path, VideoInfo *info) {
    FFMS_ErrorInfo error;
    char message[1024];
    FFMS_Init(0, 0);
    error.Buffer = message;
    error.BufferSize = sizeof(message);
    error.ErrorType = FFMS_ERROR_SUCCESS;
    error.SubType = FFMS_ERROR_SUCCESS;

    FFMS_Indexer *indexer = FFMS_CreateIndexer(path, &error);
    if (!indexer) {
        fprintf(stderr, "FFMS2: Cannot open %s: %s\n", path, message);
        return 0;
    }
    reader->index = FFMS_DoIndexing2(indexer, FFMS_IEH_ABORT, &error);
    if (!reader->index) {
        fprintf(stderr, "FFMS2: Cannot index %s: %s\n", path, message);
        return 0;
    }
    int track = FFMS_GetFirstTrackOfType(reader->index, FFMS_TYPE_VIDEO, &error);
    if (track < 0) {
        fprintf(stderr, "FFMS2: No video track: %s\n", message);
        return 0;
    }
    reader->video_source = FFMS_CreateVideoSource(path, track, reader->index, -1, FFMS_SEEK_NORMAL, &error);
    if (!reader->video_source) {
        fprintf(stderr, "FFMS2: Cannot create video source: %s\n", message);
        return 0;
    }
    const FFMS_VideoProperties *props = FFMS_GetVideoProperties(reader->video_source);
    const FFMS_Frame *frame = FFMS_GetFrame(reader->video_source, 0, &error);
    if (!frame) {
        fprintf(stderr, "FFMS2: Cannot read first frame: %s\n", message);
        return 0;
    }
    const int width = frame->EncodedWidth, height = frame->EncodedHeight;
    if (width < 1 || width > 16384 || height < 1 || height > 8704 || props->NumFrames <= 0 ||
        props->FPSNumerator <= 0 || props->FPSDenominator <= 0 ||
        (uint64_t)props->FPSNumerator > UINT_MAX || (uint64_t)props->FPSDenominator > UINT_MAX) {
        fprintf(stderr, "FFMS2: Invalid dimensions, frame count or frame rate\n");
        return 0;
    }
    // Keep ten-bit samples intact until the same truncation as the Y4M reader.
    int formats[2] = {FFMS_GetPixFmt("yuv420p10le"), -1};
    if (formats[0] < 0) {
        fprintf(stderr, "FFMS2: Unsupported output pixel format yuv420p10le\n");
        return 0;
    }
    if (FFMS_SetOutputFormatV2(reader->video_source, formats, width, height,
                               FFMS_RESIZER_BILINEAR, &error)) {
        fprintf(stderr, "FFMS2: Cannot set output format: %s\n", message);
        return 0;
    }
    reader->frame_count = props->NumFrames;
    reader->pixel_format = formats[0];
    info->width = width;
    info->height = height;
    info->fps_num = props->FPSNumerator;
    info->fps_den = props->FPSDenominator;
    info->bit_depth = 10;
    info->bytes_per_sample = 2;
    return 1;
}

static int read_ffms2_frame(VideoReader *reader, const VideoInfo *info, unsigned char *padded_y,
                             unsigned padded_width, unsigned padded_height) {
    if (reader->frame_number >= reader->frame_count) return 0;
    FFMS_ErrorInfo error;
    char message[1024];
    error.Buffer = message;
    error.BufferSize = sizeof(message);
    error.ErrorType = FFMS_ERROR_SUCCESS;
    error.SubType = FFMS_ERROR_SUCCESS;
    const FFMS_Frame *frame = FFMS_GetFrame(reader->video_source, reader->frame_number, &error);
    if (!frame) {
        fprintf(stderr, "FFMS2: Cannot read frame %d: %s\n", reader->frame_number, message);
        return -1;
    }
    if (frame->ScaledWidth != (int)info->width || frame->ScaledHeight != (int)info->height ||
        frame->ConvertedPixelFormat != reader->pixel_format || !frame->Data[0] ||
        frame->Linesize[0] < (int)info->width * 2) {
        fprintf(stderr, "FFMS2: Invalid frame dimensions, pixel format or stride\n");
        return -1;
    }
    for (unsigned y = 0; y < info->height; ++y) {
        const uint8_t *src = frame->Data[0] + (size_t)y * frame->Linesize[0];
        uint8_t *dst = padded_y + (size_t)y * padded_width;
        for (unsigned x = 0; x < info->width; ++x) {
            unsigned sample = (unsigned)src[2 * x] | ((unsigned)src[2 * x + 1] << 8);
            dst[x] = (uint8_t)((sample >> 2) > 255 ? 255 : sample >> 2);
        }
    }
    pad_input_frame(padded_y, info->width, info->height, padded_width, padded_height);
    ++reader->frame_number;
    return 1;
}
#endif

static int reader_open(VideoReader *reader, const Options *opt, VideoInfo *info) {
    int pipe = input_is_pipe(opt->input_path);
    const char *extension = strrchr(opt->input_path, '.');
    if (extension && (strchr(extension, '/') || strchr(extension, '\\'))) extension = NULL;
    if (!pipe && extension && strcasecmp(extension, ".y4m")) {
#if HAVE_FFMS2
        return init_ffms2(reader, opt->input_path, info);
#else
        fprintf(stderr, "This build does not have FFMS2 support; rebuild with USE_FFMS2 enabled\n");
        return 0;
#endif
    }
    reader->file = !strcmp(opt->input_path, "-") ? stdin : fopen(opt->input_path, "rb");
    if (!reader->file) {
        fprintf(stderr, "Cannot open input '%s': %s\n", opt->input_path, strerror(errno));
        return 0;
    }
    return read_y4m_header(reader->file, info);
}

static int reader_read_frame(VideoReader *reader, const VideoInfo *info, unsigned char *raw_y,
                              unsigned char *padded_y, unsigned padded_width, unsigned padded_height) {
#if HAVE_FFMS2
    if (reader->video_source)
        return read_ffms2_frame(reader, info, padded_y, padded_width, padded_height);
#endif
    return read_y4m_frame(reader->file, info, raw_y, padded_y, padded_width, padded_height);
}

static void reader_close(VideoReader *reader) {
#if HAVE_FFMS2
    if (reader->video_source) FFMS_DestroyVideoSource(reader->video_source);
    if (reader->index) FFMS_DestroyIndex(reader->index);
#endif
    if (reader->file && reader->file != stdin) fclose(reader->file);
}

static int append_keyframe(KeyframeList *list, uint64_t frame) {
    uint64_t *tmp;
    size_t new_capacity;
    if (list->count && list->frame[list->count - 1] == frame)
        return 1;
    if (list->count && list->frame[list->count - 1] > frame) {
        fprintf(stderr, "Internal error: keyframes are not in display order\n");
        return 0;
    }
    if (list->count == list->capacity) {
        new_capacity = list->capacity ? list->capacity * 2 : 32;
        if (new_capacity < list->capacity || new_capacity > SIZE_MAX / sizeof(*list->frame)) {
            fprintf(stderr, "Too many keyframes\n");
            return 0;
        }
        tmp = (uint64_t *)realloc(list->frame, new_capacity * sizeof(*list->frame));
        if (!tmp) {
            fprintf(stderr, "Out of memory while growing keyframe list\n");
            return 0;
        }
        list->frame = tmp;
        list->capacity = new_capacity;
    }
    list->frame[list->count++] = frame;
    return 1;
}

static int write_config(FILE *out, const KeyframeList *list) {
    size_t i;
    if (fprintf(out, "ForceKeyFrames : ") < 0)
        return 0;
    for (i = 0; i < list->count; ++i) {
        if (i && fprintf(out, ",") < 0)
            return 0;
        if (fprintf(out, "%" PRIu64 "f", list->frame[i]) < 0)
            return 0;
    }
    return fprintf(out, "\n") >= 0;
}

/* Safe bounds from the preceding standalone version are checked before the
 * upstream helpers. The extreme-rate fallback avoids signed overflow in the
 * original multiply-by-five expression while preserving the previous result.
 */
static int adapter_resolve_configuration(Options *opt, const VideoInfo *info, SequenceControlSet *scs) {
    EbSvtAv1EncConfiguration *config = &scs->static_config;
    double groups;

    memset(scs, 0, sizeof(*scs));
    config->intra_period_length = (int32_t)opt->keyint;
    config->min_intra_period_length = (int32_t)opt->min_keyint;
    config->intra_refresh_type = opt->intra_refresh_type;
    config->hierarchical_levels = opt->hierarchical_levels;
    if (opt->hierarchical_levels < 0) {
        const uint64_t samples = (uint64_t)info->width * info->height;
        // Random-access CRF defaults from enc_handle.c:copy_api_from_app().
        config->hierarchical_levels = (samples >= INPUT_SIZE_720p_TH && opt->preset >= 8) ||
            opt->preset > 8 || samples >= INPUT_SIZE_4K_TH ? 4 : 5;
    }
    config->enc_mode = opt->preset;
    config->scene_change_detection = 1;
    config->multiply_keyint = opt->multiply_keyint != 0;
    config->frame_rate_numerator = info->fps_num && info->fps_den ? info->fps_num : 60000;
    config->frame_rate_denominator = info->fps_num && info->fps_den ? info->fps_den : 1000;
    scs->frame_rate = (double)config->frame_rate_numerator / config->frame_rate_denominator;
    scs->allintra = (scs->static_config.intra_period_length == 0);
    /* Preserve this fixed-profile port's 4x4 regions, including tiny inputs. */
    scs->picture_analysis_number_of_regions_per_width = MAX_NUMBER_OF_REGIONS_IN_WIDTH;
    scs->picture_analysis_number_of_regions_per_height = MAX_NUMBER_OF_REGIONS_IN_HEIGHT;

    const int mini_gop_frames = 1 << config->hierarchical_levels;
    groups = (scs->frame_rate + mini_gop_frames) / mini_gop_frames;
    if (config->intra_period_length == -2 && groups > INT32_MAX / mini_gop_frames) {
        fprintf(stderr, "Frame rate is too large for automatic keyint\n");
        return 0;
    }
    if (config->intra_period_length == -2) {
        config->intra_period_length = groups > INT32_MAX / (mini_gop_frames * 5)
            ? 288 : svt_scd_default_intra_period(scs);
        scs->allintra = (scs->static_config.intra_period_length == 0);
    } else if (config->multiply_keyint) {
        double scaled = scs->frame_rate * config->intra_period_length;
        if (scaled < INT32_MIN || scaled > INT32_MAX) {
            fprintf(stderr, "Seconds-based keyint exceeds the supported range\n");
            return 0;
        }
        config->intra_period_length = (int32_t)scaled;
    }
    if (config->intra_period_length == -1 || scs->allintra) {
        config->min_intra_period_length = 0;
    } else if (config->min_intra_period_length == -1) {
        if (groups > INT32_MAX / mini_gop_frames) {
            fprintf(stderr, "Frame rate is too large for automatic min-keyint\n");
            return 0;
        }
        config->min_intra_period_length = svt_scd_default_min_intra_period(scs);
    }
    if (config->intra_period_length < -2 || config->intra_period_length > MAX_INTRA_PERIOD ||
        config->min_intra_period_length < -1 || config->min_intra_period_length > MAX_INTRA_PERIOD) {
        fprintf(stderr, "Resolved keyint is outside the upstream 32-bit range\n");
        return 0;
    }
    if ((config->intra_period_length >= 0 && config->min_intra_period_length > config->intra_period_length) ||
        (config->intra_period_length < 0 && config->min_intra_period_length > 0)) {
        fprintf(stderr, "--min-keyint must not exceed --keyint\n");
        return 0;
    }
    // Forward keyframes force hierarchy 4 after default keyints have been derived.
    if (config->intra_refresh_type == SVT_AV1_FWDKF_REFRESH) config->hierarchical_levels = 4;
    if (scs->allintra) config->hierarchical_levels = 2;
    config->enable_tf = scs->allintra ? 0 : opt->enable_tf;
    // BASE TF uses one future picture plus modulation, bounded by the hierarchy.
    scs->scd_delay = config->enable_tf
        ? MAX(2, MIN(1 + TF_MAX_EXTENSION, MIN(1 << config->hierarchical_levels, TF_MAX_BASE_REF_PICS)))
        : 2;
    opt->allintra = scs->allintra;
    return 1;
}

typedef struct StandaloneHistogram {
    uint32_t bins[MAX_NUMBER_OF_REGIONS_IN_WIDTH][MAX_NUMBER_OF_REGIONS_IN_HEIGHT][HISTOGRAM_NUMBER_OF_BINS];
    uint32_t *rows[MAX_NUMBER_OF_REGIONS_IN_WIDTH][MAX_NUMBER_OF_REGIONS_IN_HEIGHT];
    uint32_t **regions[MAX_NUMBER_OF_REGIONS_IN_WIDTH];
} StandaloneHistogram;

typedef struct StandalonePicture {
    PictureParentControlSet pcs;
    EbPictureBufferDesc enhanced_pic;
    StandaloneHistogram histogram;
} StandalonePicture;

typedef struct StandaloneContext {
    PictureDecisionContext pd_ctx;
    EncodeContext enc_ctx;
    StandaloneHistogram previous_histogram;
    uint32_t ahd_running_avg[MAX_NUMBER_OF_REGIONS_IN_WIDTH][MAX_NUMBER_OF_REGIONS_IN_HEIGHT];
    uint32_t *ahd_rows[MAX_NUMBER_OF_REGIONS_IN_WIDTH];
} StandaloneContext;

static uint32_t ***adapter_init_histogram(StandaloneHistogram *histogram) {
    for (unsigned x = 0; x < MAX_NUMBER_OF_REGIONS_IN_WIDTH; ++x) {
        histogram->regions[x] = histogram->rows[x];
        for (unsigned y = 0; y < MAX_NUMBER_OF_REGIONS_IN_HEIGHT; ++y)
            histogram->rows[x][y] = histogram->bins[x][y];
    }
    return histogram->regions;
}

static void adapter_init_context(StandaloneContext *context) {
    memset(context, 0, sizeof(*context));
    context->pd_ctx.reset_running_avg = true;
    context->pd_ctx.transition_detected = -1;
    context->pd_ctx.prev_picture_histogram = adapter_init_histogram(&context->previous_histogram);
    for (unsigned x = 0; x < MAX_NUMBER_OF_REGIONS_IN_WIDTH; ++x)
        context->ahd_rows[x] = context->ahd_running_avg[x];
    context->pd_ctx.ahd_running_avg = context->ahd_rows;
}

static EbPictureBufferDesc adapter_picture_desc(uint8_t *buffer_y, unsigned width, unsigned height) {
    EbPictureBufferDesc picture = {0};
    picture.y_buffer = buffer_y;
    picture.y_stride = (uint16_t)width;
    picture.width = (uint16_t)width;
    picture.height = (uint16_t)height;
    return picture;
}

/* Upstream's histogram routine assumes every region has nonzero area. Keep
 * the previous port's defined zero-area behavior for sub-16-pixel inputs in
 * this adapter, rather than editing the upstream routine or rejecting inputs.
 */
static void adapter_gather_tiny_picture_statistics(PictureParentControlSet *pcs, EbPictureBufferDesc *input_pic) {
    unsigned base_w = input_pic->width / MAX_NUMBER_OF_REGIONS_IN_WIDTH;
    unsigned base_h = input_pic->height / MAX_NUMBER_OF_REGIONS_IN_HEIGHT;
    for (unsigned rx = 0; rx < MAX_NUMBER_OF_REGIONS_IN_WIDTH; ++rx) {
        for (unsigned ry = 0; ry < MAX_NUMBER_OF_REGIONS_IN_HEIGHT; ++ry) {
            unsigned rw = base_w + (rx == MAX_NUMBER_OF_REGIONS_IN_WIDTH - 1
                ? input_pic->width - MAX_NUMBER_OF_REGIONS_IN_WIDTH * base_w : 0);
            unsigned rh = base_h + (ry == MAX_NUMBER_OF_REGIONS_IN_HEIGHT - 1
                ? input_pic->height - MAX_NUMBER_OF_REGIONS_IN_HEIGHT * base_h : 0);
            uint64_t sum = 0, area = (uint64_t)rw * rh;
            for (unsigned bin = 0; bin < HISTOGRAM_NUMBER_OF_BINS; ++bin)
                pcs->picture_histogram[rx][ry][bin] = 16;
            for (unsigned y = ry * base_h; y < ry * base_h + rh; ++y) {
                for (unsigned x = rx * base_w; x < rx * base_w + rw; ++x) {
                    uint8_t value = input_pic->y_buffer[(size_t)y * input_pic->y_stride + x];
                    pcs->picture_histogram[rx][ry][value] += 16;
                    sum += value;
                }
            }
            pcs->average_intensity_per_region[rx][ry] = area ? (uint8_t)((sum + (area >> 1)) / area) : 0;
        }
    }
}

/* Bridge contiguous Y4M working planes to the original buffer descriptors.
 * The border is zero: the reader already extends visible edges to n.8;
 * the separate SIMD guard bytes cover vector tail reads.
 */
static void adapter_analyze_picture(SequenceControlSet *scs, StandalonePicture *picture,
                                    uint64_t picture_number, uint8_t *padded_y,
                                    unsigned padded_width, unsigned padded_height,
                                    uint8_t *quarter, uint8_t *lowres) {
    PictureParentControlSet *pcs = &picture->pcs;
    EbPictureBufferDesc input_padded_pic = adapter_picture_desc(padded_y, padded_width, padded_height);
    EbPictureBufferDesc quarter_picture = adapter_picture_desc(quarter, padded_width / 2, padded_height / 2);
    EbPictureBufferDesc sixteenth_picture = adapter_picture_desc(lowres, padded_width / 4, padded_height / 4);
    uint64_t sum_avg_intensity_ttl_regions_luma;

    memset(pcs, 0, sizeof(*pcs));
    picture->enhanced_pic = input_padded_pic;
    pcs->enhanced_pic = &picture->enhanced_pic;
    pcs->picture_number = picture_number;
    pcs->picture_histogram = adapter_init_histogram(&picture->histogram);
    pcs->enable_hme_flag = pcs->enable_hme_level0_flag = pcs->enable_hme_level1_flag = true;
    pcs->tf_enable_hme_flag = pcs->tf_enable_hme_level0_flag = pcs->tf_enable_hme_level1_flag = true;

    svt_aom_downsample_filtering_input_picture(pcs, &input_padded_pic, &quarter_picture, &sixteenth_picture);
    if (sixteenth_picture.width < MAX_NUMBER_OF_REGIONS_IN_WIDTH ||
        sixteenth_picture.height < MAX_NUMBER_OF_REGIONS_IN_HEIGHT)
        adapter_gather_tiny_picture_statistics(pcs, &sixteenth_picture);
    else
        svt_scd_gather_statistics(
            scs, pcs, &sixteenth_picture, &sum_avg_intensity_ttl_regions_luma);
}

/* This is an adapter around statements from svt_aom_picture_decision_kernel,
 * not a replacement encoder kernel. Excerpts below retain their indentation.
 * Forced-input keyframes, S-frames, GOP construction and encoding are outside
 * this tool. Frame zero is initialized separately, as in the preceding port.
 */
static int adapter_process_picture(SequenceControlSet* scs, EncodeContext* enc_ctx,
    PictureDecisionContext* ctx, PictureParentControlSet* pcs, bool window_avail,
    KeyframeList* keyframes) {
    const bool allintra = scs->allintra;

    if (!allintra && window_avail)
        svt_scd_detect_scene(scs, pcs, ctx);

            // If the Intra period length is 0, then introduce an intra for every picture
            if (allintra) {
                pcs->idr_flag = true;
                pcs->cra_flag = false;
            }
            // If an #IntraPeriodLength has passed since the last Intra, then introduce a CRA or IDR based on Intra Refresh type
            else if (scs->static_config.intra_period_length != -1) {

                pcs->cra_flag =
                    (scs->static_config.intra_refresh_type != SVT_AV1_FWDKF_REFRESH) ?
                    pcs->cra_flag :
                    enc_ctx->intra_period_position == (uint32_t)scs->static_config.intra_period_length ?
                    true :
                    pcs->cra_flag;

                pcs->idr_flag =
                    (scs->static_config.intra_refresh_type != SVT_AV1_KF_REFRESH) ?
                    pcs->idr_flag :
                    enc_ctx->intra_period_position == (uint32_t)scs->static_config.intra_period_length ?

                    true :
                    pcs->idr_flag;
            }
            // Enforce minimum keyframe distance ctx->is_scene_change_detected  && enc_ctx->intra_period_position > 1
            if (scs->static_config.min_intra_period_length > 0 && pcs->picture_number != 0) {
                if ((pcs->cra_flag || pcs->idr_flag) && (enc_ctx->intra_period_position < (uint32_t)scs->static_config.min_intra_period_length)) {
                    // Too soon to place another keyframe
                    pcs->cra_flag = false;
                    pcs->idr_flag = false;
                    // SVT_LOG("\nToo soon to place another idr keyframe at %d, ipp = %d\n", pcs->picture_number, enc_ctx->intra_period_position);
                }
            }
    if ((pcs->idr_flag || pcs->cra_flag) && !append_keyframe(keyframes, pcs->picture_number))
        return 0;
    svt_scd_copy_histograms(pcs, ctx);

            // Increment the Intra Period Position
            enc_ctx->intra_period_position =
                (pcs->idr_flag == true ||
                pcs->cra_flag == true) ?
                0 : enc_ctx->intra_period_position + 1;

    return 1;
}

int main(int argc, char **argv) {
    Options opt;
    FILE *out = NULL;
    VideoReader reader = {0};
    VideoInfo info;
    KeyframeList keyframes = {0};
    SequenceControlSet *scs = NULL;
    StandaloneContext *context = NULL;
    unsigned padded_width, padded_height;
    size_t raw_size, padded_size, lowres_size;
    unsigned char *raw_y = NULL, *padded_y = NULL, *quarter = NULL, *lowres = NULL;
    StandalonePicture *pictures = NULL;
    int status = 1;
    ProgressState progress = {0};
    ProgressMonitor monitor = {0};
    int progress_started = 0;

    if (argc == 1) {
        usage(stdout, argv[0], 0);
        return 0;
    }

    memset(&info, 0, sizeof(info));
    if (!parse_options(argc, argv, &opt)) {
        usage(stderr, argv[0], 0);
        return 2;
    }

    scs = calloc(1, sizeof(*scs));
    context = calloc(1, sizeof(*context));
    if (!scs || !context) {
        fprintf(stderr, "Out of memory allocating SCD contexts\n");
        goto cleanup;
    }
    adapter_init_context(context);
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
#endif

    if (!reader_open(&reader, &opt, &info))
        goto cleanup;

    if (!adapter_resolve_configuration(&opt, &info, scs))
        goto cleanup;

    svt_scd_setup_rtcd();

    padded_width = (info.width + 7u) & ~7u;
    padded_height = (info.height + 7u) & ~7u;
    raw_size = (size_t)info.width * info.height * info.bytes_per_sample;
    padded_size = (size_t)padded_width * padded_height;
    lowres_size = (size_t)(padded_width / 4) * (padded_height / 4);

    if (reader.file) raw_y = (unsigned char *)malloc(raw_size ? raw_size : 1);
    padded_y = (unsigned char *)calloc(padded_size + SIMD_GUARD_BYTES, 1);
    if ((reader.file && !raw_y) || !padded_y) {
        fprintf(stderr, "Out of memory allocating input frame buffers\n");
        goto cleanup;
    }
    {
        quarter = (unsigned char *)calloc(padded_size / 4 + SIMD_GUARD_BYTES, 1);
        lowres = (unsigned char *)malloc(lowres_size ? lowres_size : 1);
        pictures = (StandalonePicture *)calloc(scs->scd_delay + 2u, sizeof(*pictures));
        if (!quarter || !lowres || !pictures) {
            fprintf(stderr, "Out of memory allocating SCD buffers\n");
            goto cleanup;
        }
    }

    progress_init(&progress, -1);
    progress_started = 1;
    if (progress.enabled) {
#if HAVE_FFMS2
        if (reader.video_source) progress.total_frames = reader.frame_count;
        else
#endif
            progress.total_frames = infer_y4m_frame_count(reader.file, &info);
    }
    progress_render(&progress);
    progress_monitor_start(&monitor, &progress);

    /* Keep N-1 through N+scd_delay. One extra successful read proves that
     * N+scd_delay is not the EOS frame. Only N+1's averages enter the detector,
     * but the full lookahead must pass upstream's availability check.
     */
    {
        const uint64_t ring_size = scs->scd_delay + 2u;
        uint64_t frames_read = 0, current_index = 1;
        int r;
        while ((r = reader_read_frame(&reader, &info, raw_y, padded_y,
                                   padded_width, padded_height)) == 1) {
            if (frames_read == 0 && !append_keyframe(&keyframes, 0))
                goto cleanup;
            if (frames_read >= scs->scd_delay + 2u) {
                PictureParentControlSet *pcs = &pictures[current_index % ring_size].pcs;
                pcs->pd_window[0] = &pictures[(current_index - 1) % ring_size].pcs;
                pcs->pd_window[1] = pcs;
                for (unsigned window_index = 0; window_index < scs->scd_delay; ++window_index)
                    pcs->pd_window[2 + window_index] = &pictures[(current_index + 1 + window_index) % ring_size].pcs;
                if (!adapter_process_picture(scs, &context->enc_ctx, &context->pd_ctx,
                                              pcs, true, &keyframes))
                    goto cleanup;
                ++current_index;
            }
            adapter_analyze_picture(scs, &pictures[frames_read % ring_size], frames_read,
                                    padded_y, padded_width, padded_height, quarter, lowres);
            if (frames_read == 0)
                svt_scd_copy_histograms(&pictures[0].pcs, &context->pd_ctx);
            ++frames_read;
            progress_note_frame(&progress);
        }
        if (r < 0) goto cleanup;
        if (frames_read == 0) {
            fprintf(stderr, "Input stream contains no frames\n");
            goto cleanup;
        }
        /* EOS invalidates all remaining detection windows. */
        for (; current_index < frames_read; ++current_index)
            if (!adapter_process_picture(scs, &context->enc_ctx, &context->pd_ctx,
                                          &pictures[current_index % ring_size].pcs, false, &keyframes))
                goto cleanup;
    }

    progress_monitor_stop(&monitor);
    progress_finish(&progress);

    if (strcmp(opt.output_path, "-") == 0)
        out = stdout;
    else {
        out = fopen(opt.output_path, "wb");
        if (!out) {
            fprintf(stderr, "Cannot open output '%s': %s\n", opt.output_path, strerror(errno));
            goto cleanup;
        }
    }
    if (!write_config(out, &keyframes) || fflush(out) != 0) {
        fprintf(stderr, "Failed writing keyframe config\n");
        goto cleanup;
    }
    status = 0;

cleanup:
    progress_monitor_stop(&monitor);
    if (progress_started) {
        progress_finish(&progress);
        progress_destroy(&progress);
    }
    if (out && out != stdout && fclose(out) != 0 && status == 0) {
        fprintf(stderr, "Failed closing output\n");
        status = 1;
    }
    reader_close(&reader);
    free(raw_y);
    free(padded_y);
    free(quarter);
    free(lowres);
    free(pictures);
    free(keyframes.frame);
    free(context);
    free(scs);
    return status;
}
