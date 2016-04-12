#include <immintrin.h> // AVX
#include <ks.h>
#include <gsks_internal.h>
#include <avx_type.h>


void ks_quartic_int_d8x4(
    int    k,
    int    rhs,
    double *u,
    double *aa,
    double *a,
    double *bb,
    double *b,
    double *w,
    double *c,
    ks_t   *ker, // NOP
    aux_t  *aux
    )
{
  int    i, rhs_left;
  double neg2  = -2.0;
  double dzero =  0.0;
  double done  =  1.0;
  double mdone = -1.0;
  double alpha = ( 15.0 / 16.0 );


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


  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( aa ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( bb ) );


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


  // Scale -2
  aa_tmp.v = _mm256_broadcast_sd( &neg2 );
  c03_0.v  = _mm256_mul_pd( aa_tmp.v, c03_0.v );
  c03_1.v  = _mm256_mul_pd( aa_tmp.v, c03_1.v );
  c03_2.v  = _mm256_mul_pd( aa_tmp.v, c03_2.v );
  c03_3.v  = _mm256_mul_pd( aa_tmp.v, c03_3.v );
  c47_0.v  = _mm256_mul_pd( aa_tmp.v, c47_0.v );
  c47_1.v  = _mm256_mul_pd( aa_tmp.v, c47_1.v );
  c47_2.v  = _mm256_mul_pd( aa_tmp.v, c47_2.v );
  c47_3.v  = _mm256_mul_pd( aa_tmp.v, c47_3.v );


  aa_tmp.v = _mm256_load_pd( (double*)aa );
  c03_0.v  = _mm256_add_pd( aa_tmp.v, c03_0.v );
  c03_1.v  = _mm256_add_pd( aa_tmp.v, c03_1.v );
  c03_2.v  = _mm256_add_pd( aa_tmp.v, c03_2.v );
  c03_3.v  = _mm256_add_pd( aa_tmp.v, c03_3.v );


  aa_tmp.v = _mm256_load_pd( (double*)( aa + 4 ) );
  c47_0.v  = _mm256_add_pd( aa_tmp.v, c47_0.v );
  c47_1.v  = _mm256_add_pd( aa_tmp.v, c47_1.v );
  c47_2.v  = _mm256_add_pd( aa_tmp.v, c47_2.v );
  c47_3.v  = _mm256_add_pd( aa_tmp.v, c47_3.v );
  

  // Prefetch u
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u ) );


  bb_tmp.v = _mm256_broadcast_sd( (double*)bb );
  c03_0.v  = _mm256_add_pd( bb_tmp.v, c03_0.v );
  c47_0.v  = _mm256_add_pd( bb_tmp.v, c47_0.v );

  bb_tmp.v = _mm256_broadcast_sd( (double*)( bb + 1 ) );
  c03_1.v  = _mm256_add_pd( bb_tmp.v, c03_1.v );
  c47_1.v  = _mm256_add_pd( bb_tmp.v, c47_1.v );

  bb_tmp.v = _mm256_broadcast_sd( (double*)( bb + 2 ) );
  c03_2.v  = _mm256_add_pd( bb_tmp.v, c03_2.v );
  c47_2.v  = _mm256_add_pd( bb_tmp.v, c47_2.v );

  bb_tmp.v = _mm256_broadcast_sd( (double*)( bb + 3 ) );
  c03_3.v  = _mm256_add_pd( bb_tmp.v, c03_3.v );
  c47_3.v  = _mm256_add_pd( bb_tmp.v, c47_3.v );


  // Check if there is any illegle value 
  c_tmp.v  = _mm256_broadcast_sd( &dzero );
  c03_0.v  = _mm256_max_pd( c_tmp.v, c03_0.v );
  c03_1.v  = _mm256_max_pd( c_tmp.v, c03_1.v );
  c03_2.v  = _mm256_max_pd( c_tmp.v, c03_2.v );
  c03_3.v  = _mm256_max_pd( c_tmp.v, c03_3.v );
  c47_0.v  = _mm256_max_pd( c_tmp.v, c47_0.v );
  c47_1.v  = _mm256_max_pd( c_tmp.v, c47_1.v );
  c47_2.v  = _mm256_max_pd( c_tmp.v, c47_2.v );
  c47_3.v  = _mm256_max_pd( c_tmp.v, c47_3.v );


  // If c > 1, then c = 1.
  c_tmp.v  = _mm256_broadcast_sd( &done );
  c03_0.v  = _mm256_min_pd( c_tmp.v, c03_0.v );
  c03_1.v  = _mm256_min_pd( c_tmp.v, c03_1.v );
  c03_2.v  = _mm256_min_pd( c_tmp.v, c03_2.v );
  c03_3.v  = _mm256_min_pd( c_tmp.v, c03_3.v );
  c47_0.v  = _mm256_min_pd( c_tmp.v, c47_0.v );
  c47_1.v  = _mm256_min_pd( c_tmp.v, c47_1.v );
  c47_2.v  = _mm256_min_pd( c_tmp.v, c47_2.v );
  c47_3.v  = _mm256_min_pd( c_tmp.v, c47_3.v );


  // Preload u03, u47
  u03.v    = _mm256_load_pd( (double*)u );
  u47.v    = _mm256_load_pd( (double*)( u + 4 ) );


  // Prefetch u and w
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u + 8 ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( w ) );


  // c = ( 15 / 16 ) * ( 1 - c )^2;
  c_tmp.v  = _mm256_broadcast_sd( &mdone );
  c03_0.v  = _mm256_add_pd( c_tmp.v, c03_0.v );
  c03_1.v  = _mm256_add_pd( c_tmp.v, c03_1.v );
  c03_2.v  = _mm256_add_pd( c_tmp.v, c03_2.v );
  c03_3.v  = _mm256_add_pd( c_tmp.v, c03_3.v );
  c47_0.v  = _mm256_add_pd( c_tmp.v, c47_0.v );
  c47_1.v  = _mm256_add_pd( c_tmp.v, c47_1.v );
  c47_2.v  = _mm256_add_pd( c_tmp.v, c47_2.v );
  c47_3.v  = _mm256_add_pd( c_tmp.v, c47_3.v );

  c_tmp.v  = _mm256_broadcast_sd( &alpha );
  c03_0.v  = _mm256_mul_pd( c03_0.v, c03_0.v );
  c03_1.v  = _mm256_mul_pd( c03_1.v, c03_1.v );
  c03_2.v  = _mm256_mul_pd( c03_2.v, c03_2.v );
  c03_3.v  = _mm256_mul_pd( c03_3.v, c03_3.v );
  c47_0.v  = _mm256_mul_pd( c47_0.v, c47_0.v );
  c47_1.v  = _mm256_mul_pd( c47_1.v, c47_1.v );
  c47_2.v  = _mm256_mul_pd( c47_2.v, c47_2.v );
  c47_3.v  = _mm256_mul_pd( c47_3.v, c47_3.v );

  c03_0.v  = _mm256_mul_pd( c_tmp.v, c03_0.v );
  c03_1.v  = _mm256_mul_pd( c_tmp.v, c03_1.v );
  c03_2.v  = _mm256_mul_pd( c_tmp.v, c03_2.v );
  c03_3.v  = _mm256_mul_pd( c_tmp.v, c03_3.v );
  c47_0.v  = _mm256_mul_pd( c_tmp.v, c47_0.v );
  c47_1.v  = _mm256_mul_pd( c_tmp.v, c47_1.v );
  c47_2.v  = _mm256_mul_pd( c_tmp.v, c47_2.v );
  c47_3.v  = _mm256_mul_pd( c_tmp.v, c47_3.v );

  // Multiple rhs kernel summation.
  #include "ks_kernel_summation_int_d8x4.h"

}
