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

#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <immintrin.h> // AVX

#define DKS_SIMD_ALIGN_SIZE 32
#define MIC_DKS_SIMD_ALIGN_SIZE 64

#define DKS_MC 104
#define DKS_NC 4096
#define DKS_KC 256
#define DKS_MR 8
#define DKS_NR 4

#define DKS_PACK_MC 96
#define DKS_PACK_NC 4096
#define DKS_PACK_MR 8
#define DKS_PACK_NR 4



#define KS_NUM_THREAD 20
//#define KS_NUM_THREAD 1

//#define KS_OMP_PARALLEL 1
#define KS_NUM_THD_MC 60







#define MIC_KS_NUM_THREAD 60
#define MIC_KS_JC_NT 6
#define MIC_KS_IR_NT 40


#define MIC_DKS_MC 14400
#define MIC_DKS_NC 120

#define MIC_DKS_KC 240

#define MIC_DKS_MR 8
#define MIC_DKS_NR 30

#define MIC_DKS_PACK_MC 14400
#define MIC_DKS_PACK_NC 128

#define MIC_DKS_PACK_MR 8
#define MIC_DKS_PACK_NR 32




//#define min( i, j ) ( (i)<(j) ? (i): (j) )

typedef union {
  __m256d v;
  double d[ 4 ];
  __m256i i;
  unsigned long long u[ 4 ];
} v4df_t;

//typedef union {
//  __m512d v;
//  double d[ 8 ];
//} v8df_t;

typedef union {
  __m128i v;
  int d[ 4 ];
} v4li_t;

//typedef union {
//  __m512i v;
//  int d[ 16 ];
//} v16i_t;




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
  //double *packh;
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

void dgsks_mic_var2(
    ks_t   *kernel,
    int    m,
    int    n,
    int    k,
    double *u,
    double *XA,
    double *XA2,
    int    *amap,
    double *XB,
    double *XB2,
    int    *bmap,
    double *w,
    int    *wmap
    );

void dgsks_ref_mic(
    ks_t   *kernel,
    int    m,
    int    n,
    int    k,
    double *u,
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
