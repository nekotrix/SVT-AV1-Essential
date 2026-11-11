/*
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0 in LICENSE and PATENTS.
 */
#ifndef SVT_FILM_GRAIN_ANALYSIS_H
#define SVT_FILM_GRAIN_ANALYSIS_H

#include "noise_model.h"

#if CONFIG_ENABLE_FILM_GRAIN
#ifdef __cplusplus
extern "C" {
#endif

typedef struct SvtFgaContext SvtFgaContext;

SvtFgaContext* svt_aom_fga_create(int width, int height, int bit_depth);
void           svt_aom_fga_free(SvtFgaContext* ctx);
// Call in display order. The optional next picture must be unfiltered.
int svt_aom_fga_run(SvtFgaContext* ctx, EbPictureBufferDesc* source, const EbPictureBufferDesc* next, bool scene_cut,
                    bool denoise, AomFilmGrain* grain);

#ifdef __cplusplus
}
#endif
#endif
#endif
