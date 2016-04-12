#include <immintrin.h> // AVX
#include <math.h>
#include <ks.h>
#include <gsks_internal.h>
#include <avx_type.h>


void ks_tanh_int_d8x4(
    int    k,
    int    rhs,
    double *u,
    double *aa, // NOP
    double *a,
    double *bb, // NOP
    double *b,
    double *w,
    double *c,
    ks_t   *ker,
    aux_t  *aux
    )
{
  int    i, rhs_left;
  double scal = ker->scal;
  double cons = ker->cons;


  v4df_t    c03_0,    c03_1,    c03_2,    c03_3;
  v4df_t    c47_0,    c47_1,    c47_2,    c47_3;
  v4df_t tmpc03_0, tmpc03_1, tmpc03_2, tmpc03_3;
  v4df_t tmpc47_0, tmpc47_1, tmpc47_2, tmpc47_3;
  v4df_t u03, u47;
  v4df_t a03, a47, A03, A47; // prefetched A 
  v4df_t b0, b1, b2, b3, B0; // prefetched B
  v4df_t c_tmp, aa_tmp, bb_tmp, w_tmp;


  // Rank-k update segment
  #include "ks_rank_k_int_d8x4.h"


  // Accumulate
  if ( aux->pc ) {
    tmpc03_0.v = _mm256_load_pd( (double*)( c      ) );
    c03_0.v    = _mm256_add_pd( tmpc03_0.v, c03_0.v );
    tmpc47_0.v = _mm256_load_pd( (double*)( c + 4  ) );
    c47_0.v    = _mm256_add_pd( tmpc47_0.v, c47_0.v );
    tmpc03_1.v = _mm256_load_pd( (double*)( c + 8  ) );
    c03_1.v    = _mm256_add_pd( tmpc03_1.v, c03_1.v );
    tmpc47_1.v = _mm256_load_pd( (double*)( c + 12 ) );
    c47_1.v    = _mm256_add_pd( tmpc47_1.v, c47_1.v );
    tmpc03_2.v = _mm256_load_pd( (double*)( c + 16 ) );
    c03_2.v    = _mm256_add_pd( tmpc03_2.v, c03_2.v );
    tmpc47_2.v = _mm256_load_pd( (double*)( c + 20 ) );
    c47_2.v    = _mm256_add_pd( tmpc47_2.v, c47_2.v );
    tmpc03_3.v = _mm256_load_pd( (double*)( c + 24 ) );
    c03_3.v    = _mm256_add_pd( tmpc03_3.v, c03_3.v );
    tmpc47_3.v = _mm256_load_pd( (double*)( c + 28 ) );
    c47_3.v    = _mm256_add_pd( tmpc47_3.v, c47_3.v );
  }


  // Scale before the kernel evaluation
  c_tmp.v  = _mm256_broadcast_sd( &scal );
  c03_0.v  = _mm256_mul_pd( c_tmp.v, c03_0.v );
  c03_1.v  = _mm256_mul_pd( c_tmp.v, c03_1.v );
  c03_2.v  = _mm256_mul_pd( c_tmp.v, c03_2.v );
  c03_3.v  = _mm256_mul_pd( c_tmp.v, c03_3.v );
  c47_0.v  = _mm256_mul_pd( c_tmp.v, c47_0.v );
  c47_1.v  = _mm256_mul_pd( c_tmp.v, c47_1.v );
  c47_2.v  = _mm256_mul_pd( c_tmp.v, c47_2.v );
  c47_3.v  = _mm256_mul_pd( c_tmp.v, c47_3.v );


  // Shift before the kernel evaluation
  c_tmp.v  = _mm256_broadcast_sd( &cons );
  c03_0.v  = _mm256_add_pd( c_tmp.v, c03_0.v );
  c03_1.v  = _mm256_add_pd( c_tmp.v, c03_1.v );
  c03_2.v  = _mm256_add_pd( c_tmp.v, c03_2.v );
  c03_3.v  = _mm256_add_pd( c_tmp.v, c03_3.v );
  c47_0.v  = _mm256_add_pd( c_tmp.v, c47_0.v );
  c47_1.v  = _mm256_add_pd( c_tmp.v, c47_1.v );
  c47_2.v  = _mm256_add_pd( c_tmp.v, c47_2.v );
  c47_3.v  = _mm256_add_pd( c_tmp.v, c47_3.v );


  // Preload u03, u47
  u03.v    = _mm256_load_pd( (double*)u );
  u47.v    = _mm256_load_pd( (double*)( u + 4 ) );


  // Prefetch u and w
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u + 8 ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( w ) );


  // c = tanh( c );
#ifdef GSKS_USE_VML
  c03_0.v  = _mm256_tanh_pd( c03_0.v );
  c03_1.v  = _mm256_tanh_pd( c03_1.v );
  c03_2.v  = _mm256_tanh_pd( c03_2.v );
  c03_3.v  = _mm256_tanh_pd( c03_3.v );
  c47_0.v  = _mm256_tanh_pd( c47_0.v );
  c47_1.v  = _mm256_tanh_pd( c47_1.v );
  c47_2.v  = _mm256_tanh_pd( c47_2.v );
  c47_3.v  = _mm256_tanh_pd( c47_3.v );
#else
  c03_0.d[ 0 ] = tanh( c03_0.d[ 0 ] );
  c03_0.d[ 1 ] = tanh( c03_0.d[ 1 ] );
  c03_0.d[ 2 ] = tanh( c03_0.d[ 2 ] );
  c03_0.d[ 3 ] = tanh( c03_0.d[ 3 ] );
  c03_1.d[ 0 ] = tanh( c03_1.d[ 0 ] );
  c03_1.d[ 1 ] = tanh( c03_1.d[ 1 ] );
  c03_1.d[ 2 ] = tanh( c03_1.d[ 2 ] );
  c03_1.d[ 3 ] = tanh( c03_1.d[ 3 ] );
  c03_2.d[ 0 ] = tanh( c03_2.d[ 0 ] );
  c03_2.d[ 1 ] = tanh( c03_2.d[ 1 ] );
  c03_2.d[ 2 ] = tanh( c03_2.d[ 2 ] );
  c03_2.d[ 3 ] = tanh( c03_2.d[ 3 ] );
  c03_3.d[ 0 ] = tanh( c03_3.d[ 0 ] );
  c03_3.d[ 1 ] = tanh( c03_3.d[ 1 ] );
  c03_3.d[ 2 ] = tanh( c03_3.d[ 2 ] );
  c03_3.d[ 3 ] = tanh( c03_3.d[ 3 ] );
  c47_0.d[ 0 ] = tanh( c47_0.d[ 0 ] );
  c47_0.d[ 1 ] = tanh( c47_0.d[ 1 ] );
  c47_0.d[ 2 ] = tanh( c47_0.d[ 2 ] );
  c47_0.d[ 3 ] = tanh( c47_0.d[ 3 ] );
  c47_1.d[ 0 ] = tanh( c47_1.d[ 0 ] );
  c47_1.d[ 1 ] = tanh( c47_1.d[ 1 ] );
  c47_1.d[ 2 ] = tanh( c47_1.d[ 2 ] );
  c47_1.d[ 3 ] = tanh( c47_1.d[ 3 ] );
  c47_2.d[ 0 ] = tanh( c47_2.d[ 0 ] );
  c47_2.d[ 1 ] = tanh( c47_2.d[ 1 ] );
  c47_2.d[ 2 ] = tanh( c47_2.d[ 2 ] );
  c47_2.d[ 3 ] = tanh( c47_2.d[ 3 ] );
  c47_3.d[ 0 ] = tanh( c47_3.d[ 0 ] );
  c47_3.d[ 1 ] = tanh( c47_3.d[ 1 ] );
  c47_3.d[ 2 ] = tanh( c47_3.d[ 2 ] );
  c47_3.d[ 3 ] = tanh( c47_3.d[ 3 ] );
#endif
  
  
  // Multiple rhs kernel summation.
  #include "ks_kernel_summation_int_d8x4.h"

}
