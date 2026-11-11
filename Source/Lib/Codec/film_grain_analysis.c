/*
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0 in LICENSE and PATENTS.
 */

#include "EbConfigMacros.h"

#if CONFIG_ENABLE_FILM_GRAIN
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "film_grain_analysis.h"
#include "noise_util.h"

#define FGA_BLOCK_SIZE 32
#define FGA_MAX_BLOCKS 256
#define FGA_INTERVAL 16

typedef struct SvtFgaFrame {
    const uint16_t* plane[3];
    int             stride[3];
    int             width, height, bit_depth;
} SvtFgaFrame;

typedef struct SvtFgaStats {
    int    flat_blocks, used_blocks, neighbours, scene_cut;
    double noise_std;
} SvtFgaStats;

typedef struct FgaMotion {
    int dx, dy;
    int sad;
} FgaMotion;

struct SvtFgaContext {
    int                width, height, bit_depth;
    int                blocks_w, blocks_h, motion_w, motion_h, coarse_w, coarse_h;
    uint16_t*          coarse[3];
    int16_t*           residual[3];
    uint8_t*           flat;
    double*            variance;
    double*            scratch;
    FgaMotion*         motion[2];
    AomFlatBlockFinder finder;
    AomNoiseModel      model;
    AomFilmGrain       last_grain;
    double             last_variance;
    int                failures;
    uint16_t*          input[3];
    uint16_t*          future[3];
    uint16_t*          previous[3];
    uint16_t*          denoised[3];
    double             noise_std[3];
    int                have_previous;
    int                until_analysis;
    int                denoise;
    int                denoised_ready;
};

SvtFgaContext* svt_aom_fga_create(int width, int height, int bit_depth) {
    if (width < 32 || height < 32 || (width & 1) || (height & 1) || (bit_depth != 8 && bit_depth != 10)) {
        return NULL;
    }
    SvtFgaContext* ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        return NULL;
    }
    ctx->width       = width;
    ctx->height      = height;
    ctx->bit_depth   = bit_depth;
    ctx->blocks_w    = (width + 31) / 32;
    ctx->blocks_h    = (height + 31) / 32;
    ctx->motion_w    = (width + 15) / 16;
    ctx->motion_h    = (height + 15) / 16;
    ctx->coarse_w    = (width + 3) / 4;
    ctx->coarse_h    = (height + 3) / 4;
    const int blocks = ctx->blocks_w * ctx->blocks_h;
    ctx->flat        = calloc(blocks, 1);
    ctx->variance    = calloc(blocks, sizeof(*ctx->variance));
    ctx->scratch     = calloc(blocks, sizeof(*ctx->scratch));
    for (int i = 0; i < 3; ++i) {
        ctx->coarse[i]       = calloc((size_t)ctx->coarse_w * ctx->coarse_h, sizeof(uint16_t));
        const size_t samples = (size_t)(width >> (i > 0)) * (height >> (i > 0));
        ctx->residual[i]     = calloc(samples, sizeof(int16_t));
        ctx->input[i]        = calloc(samples, sizeof(uint16_t));
        ctx->future[i]       = calloc(samples, sizeof(uint16_t));
        ctx->previous[i]     = calloc(samples, sizeof(uint16_t));
        if (!ctx->coarse[i] || !ctx->residual[i] || !ctx->input[i] || !ctx->future[i] || !ctx->previous[i]) {
            goto fail;
        }
    }
    for (int i = 0; i < 2; ++i) {
        ctx->motion[i] = calloc((size_t)ctx->motion_w * ctx->motion_h, sizeof(FgaMotion));
        if (!ctx->motion[i]) {
            goto fail;
        }
    }
    const AomNoiseModelParams params = {AOM_NOISE_SHAPE_SQUARE, 3, bit_depth, 1};
    if (!ctx->flat || !ctx->variance || !ctx->scratch ||
        !svt_aom_flat_block_finder_init(&ctx->finder, FGA_BLOCK_SIZE, bit_depth, 1) ||
        !svt_aom_noise_model_init(&ctx->model, params)) {
        goto fail;
    }
    return ctx;
fail:
    svt_aom_fga_free(ctx);
    return NULL;
}

void svt_aom_fga_free(SvtFgaContext* ctx) {
    if (!ctx) {
        return;
    }
    for (int i = 0; i < 3; ++i) {
        free(ctx->coarse[i]);
        free(ctx->residual[i]);
        free(ctx->input[i]);
        free(ctx->future[i]);
        free(ctx->previous[i]);
        free(ctx->denoised[i]);
    }
    free(ctx->motion[0]);
    free(ctx->motion[1]);
    free(ctx->flat);
    free(ctx->variance);
    free(ctx->scratch);
    svt_aom_flat_block_finder_free(&ctx->finder);
    svt_aom_noise_model_free(&ctx->model);
    free(ctx);
}

static void downsample(const SvtFgaFrame* frame, uint16_t* dst, int stride) {
    for (int y = 0; y < frame->height; y += 4) {
        for (int x = 0; x < frame->width; x += 4) {
            int sum = 0, count = 0;
            for (int dy = y; dy < AOMMIN(y + 4, frame->height); ++dy) {
                for (int dx = x; dx < AOMMIN(x + 4, frame->width); ++dx) {
                    sum += frame->plane[0][dy * frame->stride[0] + dx];
                    ++count;
                }
            }
            dst[(y / 4) * stride + x / 4] = (sum + count / 2) / count;
        }
    }
}

static int coarse_sad(const SvtFgaContext* ctx, const uint16_t* ref, int x, int y, int dx, int dy, int w, int h) {
    int sad = 0;
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            sad += abs(ctx->coarse[0][(y + row) * ctx->coarse_w + x + col] -
                       ref[(y + row + dy) * ctx->coarse_w + x + col + dx]);
        }
    }
    return sad;
}

static int full_sad(const SvtFgaFrame* source, const SvtFgaFrame* ref, int x, int y, int dx, int dy, int w, int h) {
    int sad = 0;
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            sad += abs(source->plane[0][(y + row) * source->stride[0] + x + col] -
                       ref->plane[0][(y + row + dy) * ref->stride[0] + x + col + dx]);
        }
    }
    return sad;
}

static double find_motion(SvtFgaContext* ctx, const SvtFgaFrame* source, const SvtFgaFrame* reference, int ref_index) {
    const uint16_t* ref     = ctx->coarse[ref_index + 1];
    double          sad_sum = 0;
    int             count   = 0;
    const int       range   = ctx->width > 2048 ? 5 : 3;
    for (int by = 0; by < ctx->motion_h; ++by) {
        for (int bx = 0; bx < ctx->motion_w; ++bx) {
            if (!ctx->flat[(by / 2) * ctx->blocks_w + bx / 2]) {
                continue;
            }
            const int x = bx * 4, y = by * 4;
            const int w = AOMMIN(4, ctx->coarse_w - x), h = AOMMIN(4, ctx->coarse_h - y);
            FgaMotion best  = {0, 0, coarse_sad(ctx, ref, x, y, 0, 0, w, h)};
            int       score = 17 * best.sad;
            for (int dy = -range; dy <= range; ++dy) {
                if (y + dy < 0 || y + dy + h > ctx->coarse_h) {
                    continue;
                }
                for (int dx = -range; dx <= range; ++dx) {
                    if ((!dx && !dy) || x + dx < 0 || x + dx + w > ctx->coarse_w) {
                        continue;
                    }
                    int sad = coarse_sad(ctx, ref, x, y, dx, dy, w, h);
                    if (sad * 20 < score) {
                        score = sad * 20;
                        best  = (FgaMotion){dx * 4, dy * 4, sad};
                    }
                }
            }
            {
                const int px = bx * 16, py = by * 16;
                const int pw = AOMMIN(16, ctx->width - px), ph = AOMMIN(16, ctx->height - py);
                const int zero_sad        = full_sad(source, reference, px, py, 0, 0, pw, ph);
                best.dx                   = AOMMAX(-px, AOMMIN(ctx->width - px - pw, best.dx));
                best.dy                   = AOMMAX(-py, AOMMIN(ctx->height - py - ph, best.dy));
                best.sad                  = full_sad(source, reference, px, py, best.dx, best.dy, pw, ph);
                const FgaMotion starts[2] = {best, {0, 0, zero_sad}};
                for (int start = 0; start < 2; ++start) {
                    FgaMotion candidate = starts[start];
                    for (int step = 2; step; step /= 2) {
                        const FgaMotion centre = candidate;
                        for (int dy = -step; dy <= step; dy += step) {
                            const int ry = py + centre.dy + dy;
                            if (ry < 0 || ry + ph > ctx->height) {
                                continue;
                            }
                            for (int dx = -step; dx <= step; dx += step) {
                                const int rx = px + centre.dx + dx;
                                if (rx < 0 || rx + pw > ctx->width) {
                                    continue;
                                }
                                const int value = full_sad(
                                    source, reference, px, py, centre.dx + dx, centre.dy + dy, pw, ph);
                                if (value < candidate.sad ||
                                    (value == candidate.sad &&
                                     abs(centre.dx + dx) + abs(centre.dy + dy) <
                                         abs(candidate.dx) + abs(candidate.dy))) {
                                    candidate = (FgaMotion){centre.dx + dx, centre.dy + dy, value};
                                }
                            }
                        }
                    }
                    if (candidate.sad < best.sad ||
                        (candidate.sad == best.sad &&
                         abs(candidate.dx) + abs(candidate.dy) < abs(best.dx) + abs(best.dy))) {
                        best = candidate;
                    }
                }
                if (20 * best.sad >= 17 * zero_sad) {
                    best = (FgaMotion){0, 0, zero_sad};
                }
            }
            ctx->motion[ref_index][by * ctx->motion_w + bx] = best;
            sad_sum += best.sad;
            count += AOMMIN(16, ctx->width - bx * 16) * AOMMIN(16, ctx->height - by * 16);
        }
    }
    return count ? sad_sum / count / (1 << (ctx->bit_depth - 8)) : 0;
}

static double histogram_distance(const SvtFgaContext* ctx, int ref_index) {
    int       hist[2][32] = {{0}};
    const int count       = ctx->coarse_w * ctx->coarse_h;
    for (int i = 0; i < count; ++i) {
        ++hist[0][ctx->coarse[0][i] >> (ctx->bit_depth - 5)];
        ++hist[1][ctx->coarse[ref_index + 1][i] >> (ctx->bit_depth - 5)];
    }
    double distance = 0;
    for (int i = 0; i < 32; ++i) {
        distance += abs(hist[0][i] - hist[1][i]);
    }
    return distance / count;
}

static int compare_double(const void* a, const void* b) {
    const double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static int reset_model(SvtFgaContext* ctx) {
    const AomNoiseModelParams params = {AOM_NOISE_SHAPE_SQUARE, 3, ctx->bit_depth, 1};
    svt_aom_noise_model_free(&ctx->model);
    memset(&ctx->last_grain, 0, sizeof(ctx->last_grain));
    ctx->last_variance = 0;
    ctx->failures      = 0;
    return svt_aom_noise_model_init(&ctx->model, params);
}

static int valid_frame(const SvtFgaContext* ctx, const SvtFgaFrame* frame, int packed) {
    if (frame->width != ctx->width || frame->height != ctx->height || frame->bit_depth != ctx->bit_depth) {
        return 0;
    }
    for (int c = 0; c < 3; ++c) {
        const int width = ctx->width >> (c > 0);
        if (!frame->plane[c] || frame->stride[c] < width || (packed && frame->stride[c] != width)) {
            return 0;
        }
    }
    return 1;
}

static int spatial_residual(SvtFgaContext* ctx, const SvtFgaFrame* source) {
    double             plane[32 * 32], block[32 * 32];
    AomFlatBlockFinder chroma = {0};
    if (!svt_aom_flat_block_finder_init(&chroma, 16, ctx->bit_depth, 1)) {
        return 0;
    }
    for (int c = 0; c < 3; ++c) {
        const int                 sub = c > 0, size = 32 >> sub;
        const int                 width = ctx->width >> sub, height = ctx->height >> sub;
        const AomFlatBlockFinder* finder = c ? &chroma : &ctx->finder;
        for (int by = 0; by < ctx->blocks_h; ++by) {
            for (int bx = 0; bx < ctx->blocks_w; ++bx) {
                const int i = by * ctx->blocks_w + bx;
                if (!ctx->flat[i]) {
                    continue;
                }
                svt_aom_flat_block_finder_extract_block_c(finder,
                                                          (const uint8_t*)source->plane[c],
                                                          width,
                                                          height,
                                                          source->stride[c],
                                                          bx * size,
                                                          by * size,
                                                          plane,
                                                          block);
                if (!c) {
                    double variance = 0, third = 0, fourth = 0;
                    for (int j = 0; j < size * size; ++j) {
                        const double v = block[j];
                        variance += v * v;
                        third += v * v * v;
                        fourth += v * v * v * v;
                    }
                    variance /= size * size;
                    third /= size * size;
                    fourth /= size * size;
                    // Reject edges and regular patterns in the spatial fallback.
                    if (variance <= 1e-12 || fourth < 2.4 * variance * variance || fourth > 4.5 * variance * variance ||
                        fabs(third) > 0.5 * pow(variance, 1.5)) {
                        ctx->flat[i] = 0;
                        continue;
                    }
                }
                for (int y = 0; y < AOMMIN(size, height - by * size); ++y) {
                    for (int x = 0; x < AOMMIN(size, width - bx * size); ++x) {
                        ctx->residual[c][(by * size + y) * width + bx * size + x] = (int16_t)lrint(
                            block[y * size + x] * ((1 << ctx->bit_depth) - 1) * 4);
                    }
                }
            }
        }
    }
    svt_aom_flat_block_finder_free(&chroma);
    return 1;
}

static double channel_variance(const SvtFgaContext* ctx, int c, double scale) {
    const int sub = c > 0, width = ctx->width >> sub, height = ctx->height >> sub, size = 32 >> sub;
    double    total  = 0;
    int       blocks = 0;
    for (int by = 0; by < ctx->blocks_h; ++by) {
        for (int bx = 0; bx < ctx->blocks_w; ++bx) {
            if (!ctx->flat[by * ctx->blocks_w + bx]) {
                continue;
            }
            const int w = AOMMIN(size, width - bx * size), h = AOMMIN(size, height - by * size);
            double    sum = 0, squares = 0;
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const double value = ctx->residual[c][(by * size + y) * width + bx * size + x] * scale;
                    sum += value;
                    squares += value * value;
                }
            }
            total += AOMMAX(0, squares / (w * h) - (sum / (w * h)) * (sum / (w * h)));
            ++blocks;
        }
    }
    return total / AOMMAX(1, blocks);
}

static int denoise_frame(SvtFgaContext* ctx, const SvtFgaFrame* source) {
    const uint8_t* data[3];
    uint8_t*       denoised[3];
    float          psd[3];
    int32_t        sub[2]     = {1, 1};
    int32_t        strides[3] = {ctx->width, ctx->width / 2, ctx->width / 2};
    for (int c = 0; c < 3; ++c) {
        if (!ctx->denoised[c]) {
            ctx->denoised[c] = calloc((size_t)(ctx->width >> (c > 0)) * (ctx->height >> (c > 0)), sizeof(uint16_t));
            if (!ctx->denoised[c]) {
                return 0;
            }
        }
        data[c]     = (const uint8_t*)source->plane[c];
        denoised[c] = (uint8_t*)ctx->denoised[c];
        psd[c]      = svt_aom_noise_psd_get_default_value(32 >> (c > 0), (float)ctx->noise_std[c]);
    }
    return svt_aom_wiener_denoise_2d(data, denoised, ctx->width, ctx->height, strides, sub, psd, 32, ctx->bit_depth, 1);
}

static int analyze(SvtFgaContext* ctx, const SvtFgaFrame* source, const SvtFgaFrame* previous, const SvtFgaFrame* next,
                   AomFilmGrain* grain, SvtFgaStats* stats) {
    const SvtFgaFrame* refs[2] = {previous, next};
    memset(stats, 0, sizeof(*stats));
    downsample(source, ctx->coarse[0], ctx->coarse_w);
    for (int i = 0; i < 2; ++i) {
        if (!refs[i]) {
            continue;
        }
        if (!valid_frame(ctx, refs[i], 0)) {
            return 0;
        }
        downsample(refs[i], ctx->coarse[i + 1], ctx->coarse_w);
        if (histogram_distance(ctx, i) > 0.5) {
            if (i == 0) {
                stats->scene_cut = 1;
            }
            refs[i] = NULL;
        }
    }
    if (stats->scene_cut && !reset_model(ctx)) {
        return 0;
    }
    const int blocks = ctx->blocks_w * ctx->blocks_h;
    svt_aom_flat_block_finder_run(
        &ctx->finder, (const uint8_t*)source->plane[0], ctx->width, ctx->height, source->stride[0], ctx->flat);
    for (int i = 0; i < blocks; ++i) {
        stats->flat_blocks += ctx->flat[i] != 0;
    }
    const int limit = AOMMIN(FGA_MAX_BLOCKS, stats->flat_blocks);
    int       seen  = 0;
    for (int i = 0; i < blocks; ++i) {
        if (!ctx->flat[i]) {
            continue;
        }
        ctx->flat[i] = (seen + 1) * limit / AOMMAX(1, stats->flat_blocks) >
            seen * limit / AOMMAX(1, stats->flat_blocks);
        ++seen;
    }
    int repeated = 0;
    for (int i = 0; i < 2; ++i) {
        if (!refs[i]) {
            continue;
        }
        int equal = 1;
        for (int y = 0; y < ctx->height && equal; ++y) {
            equal = !memcmp(source->plane[0] + y * source->stride[0],
                            refs[i]->plane[0] + y * refs[i]->stride[0],
                            ctx->width * sizeof(uint16_t));
        }
        repeated |= equal;
    }
    double sad[2] = {0};
    for (int i = 0; i < 2; ++i) {
        if (refs[i]) {
            sad[i] = find_motion(ctx, source, refs[i], i);
        }
    }
    if (refs[0] && refs[1]) {
        if (sad[0] > 2 * sad[1] + 1) {
            refs[0] = NULL;
        } else if (sad[1] > 2 * sad[0] + 1) {
            refs[1] = NULL;
        } else {
            refs[sad[0] <= sad[1] ? 1 : 0] = NULL;
        }
    }
    const int neighbours = (refs[0] != NULL) + (refs[1] != NULL);
    stats->neighbours    = neighbours;
    if (limit <= 1) {
        goto no_estimate;
    }
    double scale = 1.0 / sqrt(2.0);
    if (!neighbours || repeated) {
        if (!spatial_residual(ctx, source)) {
            return 0;
        }
        scale = 0.25;
    } else {
        for (int c = 0; c < 3; ++c) {
            const int sub = c > 0, width = ctx->width >> sub, height = ctx->height >> sub;
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    if (!ctx->flat[((y << sub) / 32) * ctx->blocks_w + (x << sub) / 32]) {
                        continue;
                    }
                    int residual = neighbours * source->plane[c][y * source->stride[c] + x];
                    for (int i = 0; i < 2; ++i) {
                        if (!refs[i]) {
                            continue;
                        }
                        const FgaMotion mv = ctx->motion[i][((y << sub) / 16) * ctx->motion_w + (x << sub) / 16];
                        const int       rx = AOMMAX(0, AOMMIN(width - 1, x + (mv.dx >> sub)));
                        const int       ry = AOMMAX(0, AOMMIN(height - 1, y + (mv.dy >> sub)));
                        residual -= refs[i]->plane[c][ry * refs[i]->stride[c] + rx];
                    }
                    ctx->residual[c][y * width + x] = (int16_t)residual;
                }
            }
        }
    }
    double total_variance = 0;
    for (int i = 0; i < blocks; ++i) {
        if (!ctx->flat[i]) {
            continue;
        }
        int    x = (i % ctx->blocks_w) * 32, y = (i / ctx->blocks_w) * 32;
        int    w = AOMMIN(32, ctx->width - x), h = AOMMIN(32, ctx->height - y);
        double sum = 0, squares = 0;
        for (int dy = 0; dy < h; ++dy) {
            for (int dx = 0; dx < w; ++dx) {
                double value = ctx->residual[0][(y + dy) * ctx->width + x + dx] * scale;
                sum += value;
                squares += value * value;
            }
        }
        ctx->variance[i] = AOMMAX(0, squares / (w * h) - (sum / (w * h)) * (sum / (w * h)));
    }
    for (int bin = 0; bin < 16; ++bin) {
        int count = 0;
        for (int i = 0; i < blocks; ++i) {
            if (!ctx->flat[i]) {
                continue;
            }
            int x = (i % ctx->blocks_w) * 8, y = (i / ctx->blocks_w) * 8;
            if ((ctx->coarse[0][y * ctx->coarse_w + x] >> (ctx->bit_depth - 4)) == bin) {
                ctx->scratch[count++] = ctx->variance[i];
            }
        }
        if (count < 8) {
            continue;
        }
        qsort(ctx->scratch, count, sizeof(double), compare_double);
        const double threshold = 4 * ctx->scratch[count / 2];
        for (int i = 0; i < blocks; ++i) {
            int x = (i % ctx->blocks_w) * 8, y = (i / ctx->blocks_w) * 8;
            if ((ctx->coarse[0][y * ctx->coarse_w + x] >> (ctx->bit_depth - 4)) == bin &&
                ctx->variance[i] > threshold) {
                ctx->flat[i] = 0;
            }
        }
    }
    for (int i = 0; i < blocks; ++i) {
        if (!ctx->flat[i]) {
            continue;
        }
        ++stats->used_blocks;
        total_variance += ctx->variance[i];
    }
    total_variance /= AOMMAX(1, stats->used_blocks);
    stats->noise_std = sqrt(total_variance) / (1 << (ctx->bit_depth - 8));
    if (total_variance < 1e-8) {
        if (!reset_model(ctx)) {
            return 0;
        }
        goto no_estimate;
    }
    if (stats->used_blocks >= 64 && ctx->last_variance > 0 &&
        (total_variance > 4 * ctx->last_variance || ctx->last_variance > 4 * total_variance)) {
        if (!reset_model(ctx)) {
            return 0;
        }
    }
    ctx->last_variance = total_variance;
    for (int c = 0; c < 3; ++c) {
        ctx->noise_std[c] = sqrt(channel_variance(ctx, c, scale)) / (1 << (ctx->bit_depth - 8));
    }
    if (ctx->denoise) {
        if (!denoise_frame(ctx, source)) {
            return 0;
        }
        ctx->denoised_ready = 1;
        for (int c = 0; c < 3; ++c) {
            const int width = ctx->width >> (c > 0), height = ctx->height >> (c > 0);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    ctx->residual[c][y * width + x] = source->plane[c][y * source->stride[c] + x] -
                        ctx->denoised[c][y * width + x];
                }
            }
        }
        scale = 1.0;
    }
    if (stats->used_blocks > 1) {
        const uint8_t* data[3] = {
            (const uint8_t*)source->plane[0], (const uint8_t*)source->plane[1], (const uint8_t*)source->plane[2]};
        const int16_t* residual[3] = {ctx->residual[0], ctx->residual[1], ctx->residual[2]};
        int32_t        strides[3]  = {ctx->width, ctx->width / 2, ctx->width / 2};
        if (svt_aom_noise_model_update_residual(
                &ctx->model, data, residual, ctx->width, ctx->height, strides, ctx->flat, FGA_BLOCK_SIZE, scale, 0.5) ==
                AOM_NOISE_STATUS_OK &&
            svt_aom_noise_model_get_grain_parameters(&ctx->model, grain)) {
            grain->bit_depth = ctx->bit_depth;
            ctx->last_grain  = *grain;
            ctx->failures    = 0;
            return 1;
        }
    }
no_estimate:
    if (++ctx->failures >= 3) {
        memset(&ctx->last_grain, 0, sizeof(ctx->last_grain));
    }
    const uint16_t seed = grain->random_seed;
    *grain              = ctx->last_grain;
    grain->random_seed  = seed;
    grain->bit_depth    = ctx->bit_depth;
    return 1;
}

static int valid_picture(const SvtFgaContext* ctx, const EbPictureBufferDesc* pic) {
    if (pic->width != ctx->width || pic->height != ctx->height || (int)pic->bit_depth != ctx->bit_depth ||
        pic->color_format != EB_YUV420) {
        return 0;
    }
    for (int c = 0; c < 3; ++c) {
        const int width = ctx->width >> (c > 0);
        if (!pic->buffer[c] || pic->stride[c] < width) {
            return 0;
        }
        if (ctx->bit_depth > 8 && (!pic->buffer_bit_inc[c] || (pic->stride_bit_inc[c] >> 2) < (width + 3) / 4)) {
            return 0;
        }
    }
    return 1;
}

static void read_picture(const SvtFgaContext* ctx, const EbPictureBufferDesc* pic, uint16_t* planes[3]) {
    for (int c = 0; c < 3; ++c) {
        const int width = ctx->width >> (c > 0), height = ctx->height >> (c > 0);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int value = pic->buffer[c][y * pic->stride[c] + x];
                if (ctx->bit_depth > 8) {
                    value = (value << 2) |
                        ((pic->buffer_bit_inc[c][y * (pic->stride_bit_inc[c] >> 2) + x / 4] >> (6 - 2 * (x & 3))) & 3);
                }
                planes[c][y * width + x] = value;
            }
        }
    }
}

static void write_picture(const SvtFgaContext* ctx, EbPictureBufferDesc* pic) {
    for (int c = 0; c < 3; ++c) {
        const int width = ctx->width >> (c > 0), height = ctx->height >> (c > 0);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const uint16_t value                   = ctx->denoised[c][y * width + x];
                pic->buffer[c][y * pic->stride[c] + x] = (uint8_t)(value >> (ctx->bit_depth > 8 ? 2 : 0));
                if (ctx->bit_depth > 8) {
                    const int shift = 6 - 2 * (x & 3);
                    uint8_t*  low   = &pic->buffer_bit_inc[c][y * (pic->stride_bit_inc[c] >> 2) + x / 4];
                    *low            = (uint8_t)((*low & ~(3 << shift)) | ((value & 3) << shift));
                }
            }
        }
    }
}

int svt_aom_fga_run(SvtFgaContext* ctx, EbPictureBufferDesc* source, const EbPictureBufferDesc* next, bool scene_cut,
                    bool denoise, AomFilmGrain* grain) {
    if (!ctx || !source || !grain || !valid_picture(ctx, source) || (next && !valid_picture(ctx, next))) {
        return 0;
    }
    const uint16_t seed     = grain->random_seed;
    SvtFgaFrame    input    = {{ctx->input[0], ctx->input[1], ctx->input[2]},
                               {ctx->width, ctx->width / 2, ctx->width / 2},
                               ctx->width,
                               ctx->height,
                               ctx->bit_depth};
    SvtFgaFrame    previous = {{ctx->previous[0], ctx->previous[1], ctx->previous[2]},
                               {ctx->width, ctx->width / 2, ctx->width / 2},
                               ctx->width,
                               ctx->height,
                               ctx->bit_depth};
    SvtFgaFrame    future   = {{ctx->future[0], ctx->future[1], ctx->future[2]},
                               {ctx->width, ctx->width / 2, ctx->width / 2},
                               ctx->width,
                               ctx->height,
                               ctx->bit_depth};
    read_picture(ctx, source, ctx->input);
    downsample(&input, ctx->coarse[0], ctx->coarse_w);
    if (ctx->have_previous) {
        downsample(&previous, ctx->coarse[1], ctx->coarse_w);
        scene_cut |= histogram_distance(ctx, 0) > 0.5;
    }
    if (scene_cut) {
        if (!reset_model(ctx)) {
            return 0;
        }
        ctx->have_previous  = 0;
        ctx->until_analysis = 0;
    }
    ctx->denoise        = denoise;
    ctx->denoised_ready = 0;
    if (!ctx->until_analysis) {
        SvtFgaStats stats;
        if (next) {
            read_picture(ctx, next, ctx->future);
        }
        if (!analyze(ctx, &input, ctx->have_previous ? &previous : NULL, next ? &future : NULL, grain, &stats)) {
            return 0;
        }
        ctx->until_analysis = FGA_INTERVAL;
    } else {
        *grain = ctx->last_grain;
    }
    --ctx->until_analysis;
    for (int c = 0; c < 3; ++c) {
        memcpy(ctx->previous[c],
               ctx->input[c],
               (size_t)(ctx->width >> (c > 0)) * (ctx->height >> (c > 0)) * sizeof(uint16_t));
    }
    ctx->have_previous = 1;
    grain->random_seed = seed;
    grain->bit_depth   = ctx->bit_depth;
    if (denoise && grain->apply_grain) {
        if (!ctx->denoised_ready && !denoise_frame(ctx, &input)) {
            return 0;
        }
        write_picture(ctx, source);
    }
    return 1;
}

#endif
