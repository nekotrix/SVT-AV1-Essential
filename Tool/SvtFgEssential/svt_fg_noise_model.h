/*
 * Copyright (c) 2017, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */

#ifndef SVT_FG_NOISE_MODEL_H_
#define SVT_FG_NOISE_MODEL_H_

#include "noise_model.h"

int32_t svt_aom_denoise_and_model_run_with_external_denoised(struct AomDenoiseAndModel* ctx, EbPictureBufferDesc* sd,
                                                             EbPictureBufferDesc* denoised_sd,
                                                             AomFilmGrain*        film_grain,
                                                             int32_t              use_highbd);

/* Tool-only variant; upstream's entry point remains the default. */
int32_t svt_aom_denoise_and_model_run_fast(struct AomDenoiseAndModel* ctx, EbPictureBufferDesc* sd,
                                        AomFilmGrain* film_grain, int32_t use_highbd);

#endif // SVT_FG_NOISE_MODEL_H_
