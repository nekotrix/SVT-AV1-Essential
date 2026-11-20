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

#include <stdlib.h>
#include "svt_scd_wrappers.h"
#include "svt_log.h"

// These configuration helpers are private to enc_handle.c and enc_settings.c.
int32_t svt_scd_default_intra_period(
    SequenceControlSet       *scs){
    int32_t intra_period               = 0;
    EbSvtAv1EncConfiguration   *config = &scs->static_config;
    double  fps                        = scs->frame_rate;
    int32_t mini_gop_size              = (1 << (config->hierarchical_levels));

    intra_period                       = ((int)((fps + mini_gop_size) / mini_gop_size)*(mini_gop_size));
    intra_period                       = intra_period * 5; // use a 5-sec gop by default.
    if (intra_period > 300)
        intra_period = 288;
    if (config->intra_refresh_type == 1)
        intra_period -= 1;

    return intra_period;
}

int32_t svt_scd_default_min_intra_period(
    SequenceControlSet       *scs){
    int32_t min_intra_period           = 0;
    EbSvtAv1EncConfiguration   *config = &scs->static_config;
    double  fps                        = scs->frame_rate;
    int32_t mini_gop_size              = (1 << (config->hierarchical_levels));

    min_intra_period                   = ((int)((fps + mini_gop_size) / mini_gop_size)*(mini_gop_size));
    if (config->intra_refresh_type == 1)
        min_intra_period -= 1;

    return min_intra_period;
}

EbErrorType svt_scd_str_to_keyint(const char *nptr, int32_t *out, bool *multi) {
    char      *suff;
    const long keyint = strtol(nptr, &suff, 0);

    if (keyint > INT32_MAX || keyint < -2)
        return EB_ErrorBadParameter;

    switch (*suff) {
    case 's':
        // signal we need to multiply keyint * frame_rate
        *multi = true;
        *out   = keyint;
        break;
    case '\0':
        *multi = false;
        *out   = keyint < 0 ? keyint : keyint - 1;
        break;
    default:
        // else leave as untouched, we have an invalid keyint
        SVT_ERROR("Invalid keyint value: %s\n", nptr);
        return EB_ErrorBadParameter;
    }

    return EB_ErrorNone;
}
