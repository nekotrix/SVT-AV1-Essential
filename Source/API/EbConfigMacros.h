/*
* Copyright(c) 2025 Meta Platforms, Inc. and affiliates.
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

/*
* This file contains configuration macros that control which parts of code are used
* Macros could be fed via command line, so all macros here must check if they are
* already defined!
* All macros must have the following format:
* - all macros must be prefixed with CONFIG_
*/

#ifndef EbConfigMacros_h
#define EbConfigMacros_h

// clang-format off

#ifndef CONFIG_ENABLE_QUANT_MATRIX
#define CONFIG_ENABLE_QUANT_MATRIX          1
#endif

#ifndef CONFIG_ENABLE_OBMC
#define CONFIG_ENABLE_OBMC                  1
#endif

#ifndef CONFIG_ENABLE_FILM_GRAIN
#define CONFIG_ENABLE_FILM_GRAIN            1
#endif

// clang-format on

#endif // EbConfigMacros_h
