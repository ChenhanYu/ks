#ifndef __KS_MIC_H__
#define __KS_MIC_H__


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h> // AVX

#define MIC_DKS_SIMD_ALIGN_SIZE 64
#define MIC_DKS_MC 14400
#define MIC_DKS_NC 120
#define MIC_DKS_KC 240
#define MIC_DKS_MR 8
#define MIC_DKS_NR 30
#define MIC_DKS_PACK_MC 14400
#define MIC_DKS_PACK_NC 128
#define MIC_DKS_PACK_MR 8
#define MIC_DKS_PACK_NR 32
#define MIC_KS_NUM_THREAD 60
#define MIC_KS_JC_NT 6
#define MIC_KS_IR_NT 40

typedef union {
  __m512d v;
  double d[ 8 ];
} v8df_t;

typedef union {
  __m512i v;
  int d[ 16 ];
} v16i_t;

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

#endif // defined __KS_MIC_H__
