/*
* Copyright(c) 2019 Intel Corporation
* Copyright (c) 2016, Alliance for Open Media. All rights reserved
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#include "sequence_control_set.h"
#include "svt_log.h"

EbErrorType svt_aom_derive_input_resolution(ResolutionRange* input_resolution, uint32_t inputSize) {
    EbErrorType return_error = EB_ErrorNone;
    if (inputSize < INPUT_SIZE_240p_TH) {
        *input_resolution = INPUT_SIZE_240p_RANGE;
    } else if (inputSize < INPUT_SIZE_360p_TH) {
        *input_resolution = INPUT_SIZE_360p_RANGE;
    } else if (inputSize < INPUT_SIZE_480p_TH) {
        *input_resolution = INPUT_SIZE_480p_RANGE;
    } else if (inputSize < INPUT_SIZE_720p_TH) {
        *input_resolution = INPUT_SIZE_720p_RANGE;
    } else if (inputSize < INPUT_SIZE_1080p_TH) {
        *input_resolution = INPUT_SIZE_1080p_RANGE;
    } else if (inputSize < INPUT_SIZE_4K_TH) {
        *input_resolution = INPUT_SIZE_4K_RANGE;
    } else {
        *input_resolution = INPUT_SIZE_8K_RANGE;
    }

    return return_error;
}

void svt_aom_assert_err(uint32_t condition, char* err_msg) {
    assert(condition);
    if (!condition) {
        SVT_ERROR("\n %s \n", err_msg);
    }
}
