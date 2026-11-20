/*
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0 in LICENSE and PATENTS.
 */
#ifndef SVT_AV1_FILM_GRAIN_SEEDS_H
#define SVT_AV1_FILM_GRAIN_SEEDS_H

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Ranked seed pool, arranged to reduce repetition between consecutive frames. */
static const uint16_t svt_aom_film_grain_seeds[] = {
    60958, 57496, 7258,  62261, 28894, 26402, 56151, 13141, 14661, 39234, 15157, 17083, 3665,  41433, 30821, 58794,
    61525, 17569, 41650, 62778, 44084, 30893, 47701, 15342, 19817, 60452, 40388, 21622, 650,   28098, 44306, 47333,
    36106, 46702, 24977, 17113, 39462, 29071, 19506, 21413, 52961, 15285, 55210, 43404, 40655, 63101, 10011, 46233,
    64313, 25267, 61931, 29375, 52771, 3662,  30479, 2445,  61245, 41348, 20942, 39918, 55502, 43773, 38680, 60279,
    52805, 32961, 36391, 35161, 28515, 9118,  15527, 39977, 15774, 39747, 5997,  45565, 47897, 5153,  62775, 27846,
    56379, 22984, 1033,  55912, 59267, 62564, 41701, 29522, 7995,  14364, 21477, 59650, 787,   25858, 511,   41909,
    37076, 47623, 58031, 46739, 15660, 37195, 14935, 63647, 36210, 35169, 30048, 21264, 3820,  10329, 45387, 48525,
    21313, 57602, 48096, 7128,  25840, 22011, 10528, 17177, 50630, 10630, 37586, 2501,  3129,  43838, 30569, 39517,
    57794, 5913,  59624, 50485, 51361, 10811, 9185,  59870, 45486, 48219, 42864, 14301, 60523, 6500,  33421, 24303,
    51759, 49028, 10073, 17547, 52078, 8391,  49430, 51614, 1575,  52384, 2162,  1554,  39815, 55278, 17029, 55905,
    57030, 56041, 24419, 11492, 44805, 51406, 37591, 15777, 10163, 1382,  36523, 18916, 26511, 41020, 54835, 31930,
    14612, 8383,  63290, 49933, 15239, 50926, 21260, 27010, 26778, 8939,  5480,  35519, 9908,  16073, 22996, 12100,
    15482, 2601,  10378, 5300,  61340, 18531, 33685, 21014, 21369, 5473,  9019,  6551,  56241, 64506, 4122,  14936,
    55779, 47965, 29785, 10478, 28910, 3642,  64209, 36603, 44472, 30571, 41229, 29592, 3997,  26712, 52552, 16078,
    45461, 36398, 325,   16689, 26337, 14080, 24539, 31662, 58647, 22077, 37494, 23654, 17889, 21104, 55267, 36064,
    10181, 35045, 22641, 41970, 3150,  21003, 29273, 40300, 57693, 61496, 16522, 46099, 40324, 39187, 45978, 15374,
    30773, 17096, 40294, 52854, 62853, 18523, 31602, 38997, 49494, 9851,  9391,  15409, 28440, 36078, 29233, 31980,
    8588,  32121, 31890, 51736, 50422, 47066, 17709, 22509, 46023, 48469, 59088, 49394, 53022, 65001, 9295,  7584,
    9807,  486,   52893, 18480, 10428, 3067,  35960, 18475, 15113, 53827, 48215, 10142, 34479, 1394,  47809, 27708,
    14796, 29821, 18479, 37483, 19812, 44167, 37990, 14890, 1222,  7704,  57088, 64063, 1348,  61124, 29041, 63480,
    40668,
};

/* Returns 1 on success, 0 for invalid input, or -1 on allocation failure. */
static inline int svt_aom_parse_film_grain_seeds(const char* text, uint16_t** seeds, uint32_t* seed_count) {
    uint16_t*   parsed;
    size_t      count = 0;
    const char* p     = text;

    if (!text || !*text) {
        return 0;
    }
    if (!strcmp(text, "-1")) {
        count  = sizeof(svt_aom_film_grain_seeds) / sizeof(svt_aom_film_grain_seeds[0]);
        parsed = (uint16_t*)malloc(count * sizeof(*parsed));
        if (!parsed) {
            return -1;
        }
        memcpy(parsed, svt_aom_film_grain_seeds, count * sizeof(*parsed));
    } else {
        do {
            char* end;
            errno           = 0;
            const long seed = strtol(p, &end, 10);
            if (errno || end == p || seed < 1 || seed > UINT16_MAX || (*end && *end != ',')) {
                return 0;
            }
            if (++count > UINT32_MAX || count > SIZE_MAX / sizeof(*parsed)) {
                return 0;
            }
            p = *end ? end + 1 : end;
            if (*end && !*p) {
                return 0;
            }
        } while (*p);
        parsed = (uint16_t*)malloc(count * sizeof(*parsed));
        if (!parsed) {
            return -1;
        }
        p = text;
        for (size_t i = 0; i < count; ++i) {
            char* end;
            parsed[i] = (uint16_t)strtol(p, &end, 10);
            p         = *end ? end + 1 : end;
        }
    }
    *seeds      = parsed;
    *seed_count = (uint32_t)count;
    return 1;
}

#endif // SVT_AV1_FILM_GRAIN_SEEDS_H
