/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */
#include <memory.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

#include "ransac.h"
#include "mathutils.h"
#include "random.h"
#include "common_dsp_rtcd.h"
#include "utility.h"

#define MAX_MINPTS 4
#define MAX_DEGENERATE_ITER 10
#define MINPTS_MULTIPLIER 5

#if CLN_RANSAC
#define INLIER_THRESHOLD_SQUARED 1.5625 /*(1.25 * 1.25)*/

// Number of initial models to generate
#define NUM_TRIALS 20

// Number of times to refine the best model found
#define NUM_REFINES 5
#else
#define INLIER_THRESHOLD_POW2 1.5625 /*(1.25 * 1.25)*/
#endif
#define MIN_TRIALS 20

////////////////////////////////////////////////////////////////////////////////
// ransac
#if !CLN_RANSAC
typedef int (*IsDegenerateFunc)(double *p);
typedef int (*FindTransformationFunc)(int points, double *points1, double *points2, double *params);
typedef void (*ProjectPointsDoubleFunc)(double *mat, double *points, double *proj, int n, int stride_points,
                                        int stride_proj);

static void project_points_double_translation(double *mat, double *points, double *proj, int n, int stride_points,
                                              int stride_proj) {
    int i;
    for (i = 0; i < n; ++i) {
        const double x = *(points++), y = *(points++);
        *(proj++) = x + mat[0];
        *(proj++) = y + mat[1];
        points += stride_points - 2;
        proj += stride_proj - 2;
    }
}

static void project_points_double_rotzoom(double *mat, double *points, double *proj, int n, int stride_points,
                                          int stride_proj) {
    int i;
    for (i = 0; i < n; ++i) {
        const double x = *(points++), y = *(points++);
        *(proj++) = mat[2] * x + mat[3] * y + mat[0];
        *(proj++) = -mat[3] * x + mat[2] * y + mat[1];
        points += stride_points - 2;
        proj += stride_proj - 2;
    }
}

static void project_points_double_affine(double *mat, double *points, double *proj, int n, int stride_points,
                                         int stride_proj) {
    int i;
    for (i = 0; i < n; ++i) {
        const double x = *(points++), y = *(points++);
        *(proj++) = mat[2] * x + mat[3] * y + mat[0];
        *(proj++) = mat[4] * x + mat[5] * y + mat[1];
        points += stride_points - 2;
        proj += stride_proj - 2;
    }
}

static void normalize_homography(double *pts, int n, double *T) {
    double *p       = pts;
    double  mean[2] = {0, 0};
    double  msqe    = 0;
    double  scale;
    int     i;

    assert(n > 0);
    for (i = 0; i < n; ++i, p += 2) {
        mean[0] += p[0];
        mean[1] += p[1];
    }
    mean[0] /= n;
    mean[1] /= n;
    for (p = pts, i = 0; i < n; ++i, p += 2) {
        p[0] -= mean[0];
        p[1] -= mean[1];
        msqe += sqrt(p[0] * p[0] + p[1] * p[1]);
    }
    msqe /= n;
    scale = (msqe == 0 ? 1.0 : CONST_SQRT2 / msqe);
    T[0]  = scale;
    T[1]  = 0;
    T[2]  = -scale * mean[0];
    T[3]  = 0;
    T[4]  = scale;
    T[5]  = -scale * mean[1];
    T[6]  = 0;
    T[7]  = 0;
    T[8]  = 1;
    for (p = pts, i = 0; i < n; ++i, p += 2) {
        p[0] *= scale;
        p[1] *= scale;
    }
}

static void invnormalize_mat(double *T, double *iT) {
    double is = 1.0 / T[0];
    double m0 = -T[2] * is;
    double m1 = -T[5] * is;
    iT[0]     = is;
    iT[1]     = 0;
    iT[2]     = m0;
    iT[3]     = 0;
    iT[4]     = is;
    iT[5]     = m1;
    iT[6]     = 0;
    iT[7]     = 0;
    iT[8]     = 1;
}

static void denormalize_homography(double *params, double *t1, double *t2) {
    double i_t2[9];
    double params2[9];
    invnormalize_mat(t2, i_t2);
    multiply_mat(params, t1, params2, 3, 3, 3);
    multiply_mat(i_t2, params2, params, 3, 3, 3);
}

static void denormalize_affine_reorder(double *params, double *t1, double *t2) {
#if CLN_WMMAT
    double params_denorm[9];
#else
    double params_denorm[MAX_PARAMDIM];
#endif
    params_denorm[0] = params[0];
    params_denorm[1] = params[1];
    params_denorm[2] = params[4];
    params_denorm[3] = params[2];
    params_denorm[4] = params[3];
    params_denorm[5] = params[5];
    params_denorm[6] = params_denorm[7] = 0;
    params_denorm[8]                    = 1;
    denormalize_homography(params_denorm, t1, t2);
    params[0] = params_denorm[2];
    params[1] = params_denorm[5];
    params[2] = params_denorm[0];
    params[3] = params_denorm[1];
    params[4] = params_denorm[3];
    params[5] = params_denorm[4];
#if !CLN_WMMAT
    params[6] = params[7] = 0;
#endif
}

static void denormalize_rotzoom_reorder(double *params, double *t1, double *t2) {
#if CLN_WMMAT
    double params_denorm[9];
#else
    double params_denorm[MAX_PARAMDIM];
#endif
    params_denorm[0] = params[0];
    params_denorm[1] = params[1];
    params_denorm[2] = params[2];
    params_denorm[3] = -params[1];
    params_denorm[4] = params[0];
    params_denorm[5] = params[3];
    params_denorm[6] = params_denorm[7] = 0;
    params_denorm[8]                    = 1;
    denormalize_homography(params_denorm, t1, t2);
    params[0] = params_denorm[2];
    params[1] = params_denorm[5];
    params[2] = params_denorm[0];
    params[3] = params_denorm[1];
    params[4] = -params[3];
    params[5] = params[2];
#if !CLN_WMMAT
    params[6] = params[7] = 0;
#endif
}

static void denormalize_translation_reorder(double *params, double *t1, double *t2) {
#if CLN_WMMAT
    double params_denorm[9];
#else
    double params_denorm[MAX_PARAMDIM];
#endif
    params_denorm[0] = 1;
    params_denorm[1] = 0;
    params_denorm[2] = params[0];
    params_denorm[3] = 0;
    params_denorm[4] = 1;
    params_denorm[5] = params[1];
    params_denorm[6] = params_denorm[7] = 0;
    params_denorm[8]                    = 1;
    denormalize_homography(params_denorm, t1, t2);
    params[0] = params_denorm[2];
    params[1] = params_denorm[5];
    params[2] = params[5] = 1;
    params[3] = params[4] = 0;
#if !CLN_WMMAT
    params[6] = params[7] = 0;
#endif
}

static int find_translation(int np, double *pts1, double *pts2, double *mat) {
    double sumx, sumy;

    double t1[9], t2[9];
    normalize_homography(pts1, np, t1);
    normalize_homography(pts2, np, t2);

    sumx = 0;
    sumy = 0;
    for (int i = 0; i < np; ++i) {
        double dx = *(pts2++);
        double dy = *(pts2++);
        double sx = *(pts1++);
        double sy = *(pts1++);

        sumx += dx - sx;
        sumy += dy - sy;
    }
    mat[0] = sumx / np;
    mat[1] = sumy / np;
    denormalize_translation_reorder(mat, t1, t2);
    return 0;
}

static int find_rotzoom(int np, double *pts1, double *pts2, double *mat) {
    const int np2  = np * 2;
    double   *a    = (double *)malloc(sizeof(*a) * (np2 * 5 + 20));
    double   *b    = a + np2 * 4;
    double   *temp = b + np2;

    double t1[9], t2[9];
    normalize_homography(pts1, np, t1);
    normalize_homography(pts2, np, t2);
    assert(a != NULL);
    for (int i = 0; i < np; ++i) {
        double dx = *(pts2++);
        double dy = *(pts2++);
        double sx = *(pts1++);
        double sy = *(pts1++);

        a[i * 2 * 4 + 0]       = sx;
        a[i * 2 * 4 + 1]       = sy;
        a[i * 2 * 4 + 2]       = 1;
        a[i * 2 * 4 + 3]       = 0;
        a[(i * 2 + 1) * 4 + 0] = sy;
        a[(i * 2 + 1) * 4 + 1] = -sx;
        a[(i * 2 + 1) * 4 + 2] = 0;
        a[(i * 2 + 1) * 4 + 3] = 1;

        b[2 * i]     = dx;
        b[2 * i + 1] = dy;
    }
    if (!least_squares(4, a, np2, 4, b, temp, mat)) {
        free(a);
        return 1;
    }
    denormalize_rotzoom_reorder(mat, t1, t2);
    free(a);
    return 0;
}

static int find_affine(int np, double *pts1, double *pts2, double *mat) {
    assert(np > 0);
    const int np2 = np * 2;
    double   *a   = (double *)malloc(sizeof(*a) * (np2 * 7 + 42));
    if (a == NULL)
        return 1;
    double *b    = a + np2 * 6;
    double *temp = b + np2;

    double t1[9], t2[9];
    normalize_homography(pts1, np, t1);
    normalize_homography(pts2, np, t2);

    for (int i = 0; i < np; ++i) {
        double dx = *(pts2++);
        double dy = *(pts2++);
        double sx = *(pts1++);
        double sy = *(pts1++);

        a[i * 2 * 6 + 0]       = sx;
        a[i * 2 * 6 + 1]       = sy;
        a[i * 2 * 6 + 2]       = 0;
        a[i * 2 * 6 + 3]       = 0;
        a[i * 2 * 6 + 4]       = 1;
        a[i * 2 * 6 + 5]       = 0;
        a[(i * 2 + 1) * 6 + 0] = 0;
        a[(i * 2 + 1) * 6 + 1] = 0;
        a[(i * 2 + 1) * 6 + 2] = sx;
        a[(i * 2 + 1) * 6 + 3] = sy;
        a[(i * 2 + 1) * 6 + 4] = 0;
        a[(i * 2 + 1) * 6 + 5] = 1;

        b[2 * i]     = dx;
        b[2 * i + 1] = dy;
    }
    if (!least_squares(6, a, np2, 6, b, temp, mat)) {
        free(a);
        return 1;
    }
    denormalize_affine_reorder(mat, t1, t2);
    free(a);
    return 0;
}

static int get_rand_indices(int npoints, int minpts, int *indices, unsigned int *seed) {
    int i, j;
    int ptr = lcg_rand16(seed) % npoints;
    if (minpts > npoints)
        return 0;
    indices[0] = ptr;
    ptr        = (ptr == npoints - 1 ? 0 : ptr + 1);
    i          = 1;
    while (i < minpts) {
        int index = lcg_rand16(seed) % npoints;
        while (index) {
            ptr = (ptr == npoints - 1 ? 0 : ptr + 1);
            for (j = 0; j < i; ++j) {
                if (indices[j] == ptr)
                    break;
            }
            if (j == i)
                index--;
        }
        indices[i++] = ptr;
    }
    return 1;
}
#endif

#if CLN_RANSAC
// Return -1 if 'a' is a better motion, 1 if 'b' is better, 0 otherwise.
static int compare_motions(const void *arg_a, const void *arg_b) {
    const RANSAC_MOTION *motion_a = (RANSAC_MOTION *)arg_a;
    const RANSAC_MOTION *motion_b = (RANSAC_MOTION *)arg_b;

    if (motion_a->num_inliers > motion_b->num_inliers)
        return -1;
    if (motion_a->num_inliers < motion_b->num_inliers)
        return 1;
    if (motion_a->sse < motion_b->sse)
        return -1;
    if (motion_a->sse > motion_b->sse)
        return 1;
    return 0;
}
#else
typedef struct {
    int    num_inliers;
    double variance;
    int   *inlier_indices;
} RANSAC_MOTION;

// Return -1 if 'a' is a better motion, 1 if 'b' is better, 0 otherwise.
static int compare_motions(const void *arg_a, const void *arg_b) {
    const RANSAC_MOTION *motion_a = (RANSAC_MOTION *)arg_a;
    const RANSAC_MOTION *motion_b = (RANSAC_MOTION *)arg_b;

    if (motion_a->num_inliers > motion_b->num_inliers)
        return -1;
    if (motion_a->num_inliers < motion_b->num_inliers)
        return 1;
    if (motion_a->variance < motion_b->variance)
        return -1;
    if (motion_a->variance > motion_b->variance)
        return 1;
    return 0;
}
#endif

static int is_better_motion(const RANSAC_MOTION *motion_a, const RANSAC_MOTION *motion_b) {
    return compare_motions(motion_a, motion_b) < 0;
}

#if !CLN_RANSAC
static void copy_points_at_indices(double *dest, const double *src, const int *indices, int num_points) {
    for (int i = 0; i < num_points; ++i) {
        const int index = indices[i];
        dest[i * 2]     = src[index * 2];
        dest[i * 2 + 1] = src[index * 2 + 1];
    }
}
#endif

#if CLN_RANSAC
static void score_translation(const double *mat, const Correspondence *points, int num_points, RANSAC_MOTION *model) {
    model->num_inliers = 0;
    model->sse         = 0.0;

    for (int i = 0; i < num_points; ++i) {
        const double x1 = points[i].x;
        const double y1 = points[i].y;
        const double x2 = points[i].rx;
        const double y2 = points[i].ry;

        const double proj_x = x1 + mat[0];
        const double proj_y = y1 + mat[1];

        const double dx  = proj_x - x2;
        const double dy  = proj_y - y2;
        const double sse = dx * dx + dy * dy;

        if (sse < INLIER_THRESHOLD_SQUARED) {
            model->inlier_indices[model->num_inliers++] = i;
            model->sse += sse;
        }
    }
}

static void score_affine(const double *mat, const Correspondence *points, int num_points, RANSAC_MOTION *model) {
    model->num_inliers = 0;
    model->sse         = 0.0;

    for (int i = 0; i < num_points; ++i) {
        const double x1 = points[i].x;
        const double y1 = points[i].y;
        const double x2 = points[i].rx;
        const double y2 = points[i].ry;

        const double proj_x = mat[2] * x1 + mat[3] * y1 + mat[0];
        const double proj_y = mat[4] * x1 + mat[5] * y1 + mat[1];

        const double dx  = proj_x - x2;
        const double dy  = proj_y - y2;
        const double sse = dx * dx + dy * dy;

        if (sse < INLIER_THRESHOLD_SQUARED) {
            model->inlier_indices[model->num_inliers++] = i;
            model->sse += sse;
        }
    }
}

static bool find_translation(const Correspondence *points, const int *indices, int num_indices, double *params) {
    double sumx = 0;
    double sumy = 0;

    for (int i = 0; i < num_indices; ++i) {
        int          index = indices[i];
        const double sx    = points[index].x;
        const double sy    = points[index].y;
        const double dx    = points[index].rx;
        const double dy    = points[index].ry;

        sumx += dx - sx;
        sumy += dy - sy;
    }

    params[0] = sumx / num_indices;
    params[1] = sumy / num_indices;
    params[2] = 1;
    params[3] = 0;
    params[4] = 0;
    params[5] = 1;
    return true;
}

static bool find_rotzoom(const Correspondence *points, const int *indices, int num_indices, double *params) {
    const int n = 4; // Size of least-squares problem
    double    mat[4 * 4]; // Accumulator for A'A
    double    y[4]; // Accumulator for A'b
    double    a[4]; // Single row of A

    least_squares_init(mat, y, n);
    for (int i = 0; i < num_indices; ++i) {
        int          index = indices[i];
        const double sx    = points[index].x;
        const double sy    = points[index].y;
        const double dx    = points[index].rx;
        const double dy    = points[index].ry;

        a[0]     = 1;
        a[1]     = 0;
        a[2]     = sx;
        a[3]     = sy;
        double b = dx; // Single element of b
        least_squares_accumulate(mat, y, a, b, n);

        a[0] = 0;
        a[1] = 1;
        a[2] = sy;
        a[3] = -sx;
        b    = dy;
        least_squares_accumulate(mat, y, a, b, n);
    }

    // Fill in params[0] .. params[3] with output model
    if (!least_squares_solve(mat, y, params, n)) {
        return false;
    }

    // Fill in remaining parameters
    params[4] = -params[3];
    params[5] = params[2];

    return true;
}

static bool find_affine(const Correspondence *points, const int *indices, int num_indices, double *params) {
    // Note: The least squares problem for affine models is 6-dimensional,
    // but it splits into two independent 3-dimensional subproblems.
    // Solving these two subproblems separately and recombining at the end
    // results in less total computation than solving the 6-dimensional
    // problem directly.
    //
    // The two subproblems correspond to all the parameters which contribute
    // to the x output of the model, and all the parameters which contribute
    // to the y output, respectively.

    const int n = 3; // Size of each least-squares problem
    double    mat[2][3 * 3]; // Accumulator for A'A
    double    y[2][3]; // Accumulator for A'b
    double    x[2][3]; // Output vector
    double    a[2][3]; // Single row of A
    double    b[2]; // Single element of b

    least_squares_init(mat[0], y[0], n);
    least_squares_init(mat[1], y[1], n);
    for (int i = 0; i < num_indices; ++i) {
        int          index = indices[i];
        const double sx    = points[index].x;
        const double sy    = points[index].y;
        const double dx    = points[index].rx;
        const double dy    = points[index].ry;

        a[0][0] = 1;
        a[0][1] = sx;
        a[0][2] = sy;
        b[0]    = dx;
        least_squares_accumulate(mat[0], y[0], a[0], b[0], n);

        a[1][0] = 1;
        a[1][1] = sx;
        a[1][2] = sy;
        b[1]    = dy;
        least_squares_accumulate(mat[1], y[1], a[1], b[1], n);
    }

    if (!least_squares_solve(mat[0], y[0], x[0], n)) {
        return false;
    }
    if (!least_squares_solve(mat[1], y[1], x[1], n)) {
        return false;
    }

    // Rearrange least squares result to form output model
    params[0] = x[0][0];
    params[1] = x[1][0];
    params[2] = x[0][1];
    params[3] = x[0][2];
    params[4] = x[1][1];
    params[5] = x[1][2];

    return true;
}

// Returns true on success, false on error
static bool ransac_internal(const Correspondence *matched_points, int npoints, MotionModel *motion_models,
                            int num_desired_motions, const RansacModelInfo *model_info, bool *mem_alloc_failed) {
    assert(npoints >= 0);
    int  i       = 0;
    int  minpts  = model_info->minpts;
    bool ret_val = true;

    unsigned int seed = (unsigned int)npoints;

    int indices[MAX_MINPTS] = {0};

    // Store information for the num_desired_motions best transformations found
    // and the worst motion among them, as well as the motion currently under
    // consideration.
    RANSAC_MOTION *motions, *worst_kept_motion = NULL;
    RANSAC_MOTION  current_motion;

    // Store the parameters and the indices of the inlier points for the motion
    // currently under consideration.
    double params_this_motion[MAX_PARAMDIM];

    // Initialize output models, as a fallback in case we can't find a model
    for (i = 0; i < num_desired_motions; i++) {
        memcpy(motion_models[i].params, kIdentityParams, MAX_PARAMDIM * sizeof(*(motion_models[i].params)));
        motion_models[i].num_inliers = 0;
    }

    if (npoints < minpts * MINPTS_MULTIPLIER || npoints == 0) {
        return false;
    }

    int min_inliers = AOMMAX((int)(MIN_INLIER_PROB * npoints), minpts);

    motions = (RANSAC_MOTION *)calloc(num_desired_motions, sizeof(RANSAC_MOTION));

    // Allocate one large buffer which will be carved up to store the inlier
    // indices for the current motion plus the num_desired_motions many
    // output models
    // This allows us to keep the allocation/deallocation logic simple, without
    // having to (for example) check that `motions` is non-null before allocating
    // the inlier arrays
    int *inlier_buffer = (int *)malloc(sizeof(*inlier_buffer) * npoints * (num_desired_motions + 1));

    if (!(motions && inlier_buffer)) {
        ret_val           = false;
        *mem_alloc_failed = true;
        goto finish_ransac;
    }

    // Once all our allocations are known-good, we can fill in our structures
    worst_kept_motion = motions;

    for (i = 0; i < num_desired_motions; ++i) { motions[i].inlier_indices = inlier_buffer + i * npoints; }
    memset(&current_motion, 0, sizeof(current_motion));
    current_motion.inlier_indices = inlier_buffer + num_desired_motions * npoints;

    for (int trial_count = 0; trial_count < NUM_TRIALS; trial_count++) {
        lcg_pick(npoints, minpts, indices, &seed);

        if (!model_info->find_transformation(matched_points, indices, minpts, params_this_motion)) {
            continue;
        }

        model_info->score_model(params_this_motion, matched_points, npoints, &current_motion);

        if (current_motion.num_inliers < min_inliers) {
            // Reject models with too few inliers
            continue;
        }

        if (is_better_motion(&current_motion, worst_kept_motion)) {
            // This motion is better than the worst currently kept motion. Remember
            // the inlier points and sse. The parameters for each kept motion
            // will be recomputed later using only the inliers.
            worst_kept_motion->num_inliers = current_motion.num_inliers;
            worst_kept_motion->sse         = current_motion.sse;

            // Rather than copying the (potentially many) inlier indices from
            // current_motion.inlier_indices to worst_kept_motion->inlier_indices,
            // we can swap the underlying pointers.
            //
            // This is okay because the next time current_motion.inlier_indices
            // is used will be in the next trial, where we ignore its previous
            // contents anyway. And both arrays will be deallocated together at the
            // end of this function, so there are no lifetime issues.
            int *tmp                          = worst_kept_motion->inlier_indices;
            worst_kept_motion->inlier_indices = current_motion.inlier_indices;
            current_motion.inlier_indices     = tmp;

            // Determine the new worst kept motion and its num_inliers and sse.
            for (i = 0; i < num_desired_motions; ++i) {
                if (is_better_motion(worst_kept_motion, &motions[i])) {
                    worst_kept_motion = &motions[i];
                }
            }
        }
    }

    // Sort the motions, best first.
    qsort(motions, num_desired_motions, sizeof(RANSAC_MOTION), compare_motions);

    // Refine each of the best N models using iterative estimation.
    //
    // The idea here is loosely based on the iterative method from
    // "Locally Optimized RANSAC" by O. Chum, J. Matas and Josef Kittler:
    // https://cmp.felk.cvut.cz/ftp/articles/matas/chum-dagm03.pdf
    //
    // However, we implement a simpler version than their proposal, and simply
    // refit the model repeatedly until the number of inliers stops increasing,
    // with a cap on the number of iterations to defend against edge cases which
    // only improve very slowly.
    for (i = 0; i < num_desired_motions; ++i) {
        if (motions[i].num_inliers <= 0) {
            // Output model has already been initialized to the identity model,
            // so just skip setup
            continue;
        }

        bool bad_model = false;
        for (int refine_count = 0; refine_count < NUM_REFINES; refine_count++) {
            int num_inliers = motions[i].num_inliers;
            assert(num_inliers >= min_inliers);

            if (!model_info->find_transformation(
                    matched_points, motions[i].inlier_indices, num_inliers, params_this_motion)) {
                // In the unlikely event that this model fitting fails, we don't have a
                // good fallback. So leave this model set to the identity model
                bad_model = true;
                break;
            }

            // Score the newly generated model
            model_info->score_model(params_this_motion, matched_points, npoints, &current_motion);

            // At this point, there are three possibilities:
            // 1) If we found more inliers, keep refining.
            // 2) If we found the same number of inliers but a lower SSE, we want to
            //    keep the new model, but further refinement is unlikely to gain much.
            //    So commit to this new model
            // 3) It is possible, but very unlikely, that the new model will have
            //    fewer inliers. If it does happen, we probably just lost a few
            //    borderline inliers. So treat the same as case (2).
            if (current_motion.num_inliers > motions[i].num_inliers) {
                motions[i].num_inliers        = current_motion.num_inliers;
                motions[i].sse                = current_motion.sse;
                int *tmp                      = motions[i].inlier_indices;
                motions[i].inlier_indices     = current_motion.inlier_indices;
                current_motion.inlier_indices = tmp;
            } else {
                // Refined model is no better, so stop
                // This shouldn't be significantly worse than the previous model,
                // so it's fine to use the parameters in params_this_motion.
                // This saves us from having to cache the previous iteration's params.
                break;
            }
        }

        if (bad_model)
            continue;

        // Fill in output struct
        memcpy(motion_models[i].params, params_this_motion, MAX_PARAMDIM * sizeof(*motion_models[i].params));
        for (int j = 0; j < motions[i].num_inliers; j++) {
            int                   index         = motions[i].inlier_indices[j];
            const Correspondence *corr          = &matched_points[index];
            motion_models[i].inliers[2 * j + 0] = (int)rint(corr->x);
            motion_models[i].inliers[2 * j + 1] = (int)rint(corr->y);
        }
        motion_models[i].num_inliers = motions[i].num_inliers;
    }

finish_ransac:
    free(inlier_buffer);
    free(motions);

    return ret_val;
}

static const RansacModelInfo ransac_model_info[TRANS_TYPES] = {
    // IDENTITY
    {NULL, NULL, 0},
    // TRANSLATION
    {find_translation, score_translation, 1},
    // ROTZOOM
    {find_rotzoom, score_affine, 2},
    // AFFINE
    {find_affine, score_affine, 3},
};

// Returns true on success, false on error
bool svt_aom_ransac(const Correspondence *matched_points, int npoints, TransformationType type,
                    MotionModel *motion_models, int num_desired_motions, bool *mem_alloc_failed) {
    assert(type > IDENTITY && type < TRANS_TYPES);

    return ransac_internal(
        matched_points, npoints, motion_models, num_desired_motions, &ransac_model_info[type], mem_alloc_failed);
}
#else
static const double k_infinite_variance = 1e12;

static void clear_motion(RANSAC_MOTION *motion, int num_points) {
    motion->num_inliers = 0;
    motion->variance    = k_infinite_variance;
    memset(motion->inlier_indices, 0, sizeof(*motion->inlier_indices) * num_points);
}

static int ransac(const int *matched_points, int npoints, int *num_inliers_by_motion, MotionModel *params_by_motion,
                  int num_desired_motions, int minpts, IsDegenerateFunc is_degenerate,
                  FindTransformationFunc find_transformation, ProjectPointsDoubleFunc projectpoints) {
    int trial_count = 0;
    int ret_val     = 0;

    unsigned int seed = (unsigned int)npoints;

    int indices[MAX_MINPTS] = {0};

    double *points1, *points2;
    double *corners1, *corners2;
    double *image1_coord;

    // Store information for the num_desired_motions best transformations found
    // and the worst motion among them, as well as the motion currently under
    // consideration.
    RANSAC_MOTION *motions, *worst_kept_motion = NULL;
    RANSAC_MOTION  current_motion;

    // Store the parameters and the indices of the inlier points for the motion
    // currently under consideration.
    double params_this_motion[MAX_PARAMDIM];

    double *cnp1, *cnp2;

    for (int i = 0; i < num_desired_motions; ++i) num_inliers_by_motion[i] = 0;

    if (npoints < minpts * MINPTS_MULTIPLIER || npoints == 0)
        return 1;

    points1      = (double *)malloc(sizeof(*points1) * npoints * 2);
    points2      = (double *)malloc(sizeof(*points2) * npoints * 2);
    corners1     = (double *)malloc(sizeof(*corners1) * npoints * 2);
    corners2     = (double *)malloc(sizeof(*corners2) * npoints * 2);
    image1_coord = (double *)malloc(sizeof(*image1_coord) * npoints * 2);

    motions = (RANSAC_MOTION *)malloc(sizeof(RANSAC_MOTION) * num_desired_motions);
    assert(motions != NULL);
    for (int i = 0; i < num_desired_motions; ++i) {
        motions[i].inlier_indices = (int *)malloc(sizeof(*motions->inlier_indices) * npoints);
        clear_motion(motions + i, npoints);
    }
    current_motion.inlier_indices = (int *)malloc(sizeof(*current_motion.inlier_indices) * npoints);
    clear_motion(&current_motion, npoints);

    worst_kept_motion = motions;

    if (!(points1 && points2 && corners1 && corners2 && image1_coord && motions && current_motion.inlier_indices)) {
        ret_val = 1;
        goto finish_ransac;
    }

    cnp1 = corners1;
    cnp2 = corners2;
    for (int i = 0; i < npoints; ++i) {
        *(cnp1++) = *(matched_points++);
        *(cnp1++) = *(matched_points++);
        *(cnp2++) = *(matched_points++);
        *(cnp2++) = *(matched_points++);
    }

    while (MIN_TRIALS > trial_count) {
        double sum_distance         = 0.0;
        double sum_distance_squared = 0.0;

        clear_motion(&current_motion, npoints);

        int degenerate          = 1;
        int num_degenerate_iter = 0;

        while (degenerate) {
            num_degenerate_iter++;
            if (!get_rand_indices(npoints, minpts, indices, &seed)) {
                ret_val = 1;
                goto finish_ransac;
            }

            copy_points_at_indices(points1, corners1, indices, minpts);
            copy_points_at_indices(points2, corners2, indices, minpts);

            degenerate = is_degenerate(points1);
            if (num_degenerate_iter > MAX_DEGENERATE_ITER) {
                ret_val = 1;
                goto finish_ransac;
            }
        }

        if (find_transformation(minpts, points1, points2, params_this_motion)) {
            trial_count++;
            continue;
        }

        projectpoints(params_this_motion, corners1, image1_coord, npoints, 2, 2);

        for (int i = 0; i < npoints; ++i) {
            double dx            = image1_coord[i * 2] - corners2[i * 2];
            double dy            = image1_coord[i * 2 + 1] - corners2[i * 2 + 1];
            double distance_pow2 = dx * dx + dy * dy;

            if (distance_pow2 < INLIER_THRESHOLD_POW2) {
                current_motion.inlier_indices[current_motion.num_inliers++] = i;
                sum_distance += sqrt(distance_pow2);
                sum_distance_squared += distance_pow2;
            }
        }

        if (current_motion.num_inliers >= worst_kept_motion->num_inliers && current_motion.num_inliers > 1) {
            double mean_distance;
            mean_distance           = sum_distance / ((double)current_motion.num_inliers);
            current_motion.variance = sum_distance_squared / ((double)current_motion.num_inliers - 1.0) -
                mean_distance * mean_distance * ((double)current_motion.num_inliers) /
                    ((double)current_motion.num_inliers - 1.0);
            if (is_better_motion(&current_motion, worst_kept_motion)) {
                // This motion is better than the worst currently kept motion. Remember
                // the inlier points and variance. The parameters for each kept motion
                // will be recomputed later using only the inliers.
                worst_kept_motion->num_inliers = current_motion.num_inliers;
                worst_kept_motion->variance    = current_motion.variance;
                if (svt_memcpy != NULL)
                    svt_memcpy(worst_kept_motion->inlier_indices,
                               current_motion.inlier_indices,
                               sizeof(*current_motion.inlier_indices) * npoints);
                else
                    svt_memcpy_c(worst_kept_motion->inlier_indices,
                                 current_motion.inlier_indices,
                                 sizeof(*current_motion.inlier_indices) * npoints);
                assert(npoints > 0);
                // Determine the new worst kept motion and its num_inliers and variance.
                for (int i = 0; i < num_desired_motions; ++i) {
                    if (is_better_motion(worst_kept_motion, &motions[i])) {
                        worst_kept_motion = &motions[i];
                    }
                }
            }
        }
        trial_count++;
    }

    // Sort the motions, best first.
    qsort(motions, num_desired_motions, sizeof(RANSAC_MOTION), compare_motions);

    // Recompute the motions using only the inliers.
    for (int i = 0; i < num_desired_motions; ++i) {
        if (motions[i].num_inliers >= minpts) {
            copy_points_at_indices(points1, corners1, motions[i].inlier_indices, motions[i].num_inliers);
            copy_points_at_indices(points2, corners2, motions[i].inlier_indices, motions[i].num_inliers);

            find_transformation(motions[i].num_inliers, points1, points2, params_by_motion[i].params);

            params_by_motion[i].num_inliers = motions[i].num_inliers;
            if (svt_memcpy != NULL)
                svt_memcpy(params_by_motion[i].inliers,
                           motions[i].inlier_indices,
                           sizeof(*motions[i].inlier_indices) * npoints);
            else
                svt_memcpy_c(params_by_motion[i].inliers,
                             motions[i].inlier_indices,
                             sizeof(*motions[i].inlier_indices) * npoints);
        }
        num_inliers_by_motion[i] = motions[i].num_inliers;
    }

finish_ransac:
    free(points1);
    free(points2);
    free(corners1);
    free(corners2);
    free(image1_coord);
    free(current_motion.inlier_indices);
    if (motions) {
        for (int i = 0; i < num_desired_motions; ++i) free(motions[i].inlier_indices);
        free(motions);
    }

    return ret_val;
}

static int is_collinear3(double *p1, double *p2, double *p3) {
    static const double collinear_eps = 1e-3;
    const double        v             = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]);
    return fabs(v) < collinear_eps;
}

static int is_degenerate_translation(double *p) {
    return (p[0] - p[2]) * (p[0] - p[2]) + (p[1] - p[3]) * (p[1] - p[3]) <= 2;
}

static int is_degenerate_affine(double *p) { return is_collinear3(p, p + 2, p + 4); }

static int ransac_translation(int *matched_points, int npoints, int *num_inliers_by_motion,
                              MotionModel *params_by_motion, int num_desired_motions) {
    return ransac(matched_points,
                  npoints,
                  num_inliers_by_motion,
                  params_by_motion,
                  num_desired_motions,
                  3,
                  is_degenerate_translation,
                  find_translation,
                  project_points_double_translation);
}

static int ransac_rotzoom(int *matched_points, int npoints, int *num_inliers_by_motion, MotionModel *params_by_motion,
                          int num_desired_motions) {
    return ransac(matched_points,
                  npoints,
                  num_inliers_by_motion,
                  params_by_motion,
                  num_desired_motions,
                  3,
                  is_degenerate_affine,
                  find_rotzoom,
                  project_points_double_rotzoom);
}

static int ransac_affine(int *matched_points, int npoints, int *num_inliers_by_motion, MotionModel *params_by_motion,
                         int num_desired_motions) {
    return ransac(matched_points,
                  npoints,
                  num_inliers_by_motion,
                  params_by_motion,
                  num_desired_motions,
                  3,
                  is_degenerate_affine,
                  find_affine,
                  project_points_double_affine);
}

RansacFunc svt_av1_get_ransac_type(TransformationType type) {
    switch (type) {
    case AFFINE: return ransac_affine;
    case ROTZOOM: return ransac_rotzoom;
    case TRANSLATION: return ransac_translation;
    default: assert(0); return NULL;
    }
}
#endif
