/*
 * --------------------------------------------------------------------------
 * GSKS (General Stride Kernel Summation)
 * --------------------------------------------------------------------------
 * Copyright (C) 2015, The University of Texas at Austin
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 *
 * ks.h
 *
 * Chenhan D. Yu - Department of Computer Science, 
 *                 The University of Texas at Austin
 *
 *
 * Purpose: 
 *
 *
 * Todo:
 *
 *
 * Modification:
 *
 *
 * */

#ifndef __KS_H__
#define __KS_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define KS_RHS 1
#define KS_NUM_THREAD 20

typedef enum { 
  KS_GAUSSIAN, 
  KS_POLYNOMIAL, 
  KS_LAPLACE, 
  KS_GAUSSIAN_VAR_BANDWIDTH,
  KS_TANH,
  KS_QUARTIC,
  KS_MULTIQUADRATIC,
  KS_EPANECHNIKOV
} ks_type;

struct kernel_s {
  ks_type type;
  double powe;
  double scal;
  double cons;
  // The following variables are designed for the variable gaussian kernel.
  double *hi;
  double *hj;
};

typedef struct kernel_s ks_t;

void dgsks(
    ks_t   *kernel,
    int    m,
    int    n,
    int    k,
    double *u,
    int    *umap,
    double *XA,
    double *XA2,
    int    *amap,
    double *XB,
    double *XB2,
    int    *bmap,
    double *w,
    int    *wmap
    );

void dgsks_ref(
    ks_t   *kernel,
    int    m,
    int    n,
    int    k,
    double *u,
    int    *umap,
    double *XA,
    double *XA2,
    int    *amap,
    double *XB,
    double *XB2,
    int    *bmap,
    double *w,
    int    *wmap
    );

double *ks_malloc_aligned(
    int    m,
    int    n,
    int    size
    );

#endif // defined __KS_H__
