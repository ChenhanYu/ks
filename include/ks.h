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
#include <immintrin.h> // AVX

#define DKS_SIMD_ALIGN_SIZE 32
#define DKS_MC 104
#define DKS_NC 4096
#define DKS_KC 256
#define DKS_MR 8
#define DKS_NR 4
#define DKS_PACK_MC 96
#define DKS_PACK_NC 4096
#define DKS_PACK_MR 8
#define DKS_PACK_NR 4

#define KS_RHS 3
#define KS_NUM_THREAD 20

typedef union {
  __m256d v;
  double d[ 4 ];
  __m256i i;
  unsigned long long u[ 4 ];
} v4df_t;

typedef union {
  __m128i v;
  int d[ 4 ];
} v4li_t;

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

struct aux_s {
  double *a_next;
  double *b_next;
  double *c_buff;
  int    pc;
};

typedef struct aux_s aux_t;

struct kernel_s {
  ks_type type;
  double powe;
  double scal;
  double cons;
  // The following variables are designed for the variable gaussian kernel.
  double *h;
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
