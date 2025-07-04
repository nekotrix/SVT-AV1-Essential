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
