#include <immintrin.h> // AVX
#include <ks.h>


void ks_tanh_int_d8x4(
    int    k,
    double scal,
    double cons,
    double *u,
    double *a,
    double *b,
    double *w,
    aux_t  *aux
    )
{
  int    i;

  v4df_t c03_0, c03_1, c03_2, c03_3;
  v4df_t c47_0, c47_1, c47_2, c47_3;
  v4df_t tmpc03_0, tmpc03_1, tmpc03_2, tmpc03_3;
  v4df_t tmpc47_0, tmpc47_1, tmpc47_2, tmpc47_3;
  v4df_t c_tmp;
  v4df_t u03, u47;
  v4df_t a03, a47;
  v4df_t A03, A47; // prefetched A 

  v4df_t b0, b1, b2, b3;
  v4df_t B0;       // prefetched B

  v4df_t w_tmp;


  // Rank-k update segment
  #include "ks_rank_k_int_d8x4.h"


  // Prefetch u
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u ) );


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


  // Prefetch w
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( w ) );


  // c = tanh( c );
  c03_0.v  = _mm256_tanh_pd( c03_0.v );
  c03_1.v  = _mm256_tanh_pd( c03_1.v );
  c03_2.v  = _mm256_tanh_pd( c03_2.v );
  c03_3.v  = _mm256_tanh_pd( c03_3.v );
  c47_0.v  = _mm256_tanh_pd( c47_0.v );
  c47_1.v  = _mm256_tanh_pd( c47_1.v );
  c47_2.v  = _mm256_tanh_pd( c47_2.v );
  c47_3.v  = _mm256_tanh_pd( c47_3.v );
  

  // u = C * w
  w_tmp.v  = _mm256_broadcast_sd( (double*)w );
  c03_0.v  = _mm256_mul_pd( w_tmp.v, c03_0.v );
  c47_0.v  = _mm256_mul_pd( w_tmp.v, c47_0.v );
  u03.v    = _mm256_add_pd( u03.v, c03_0.v );
  u47.v    = _mm256_add_pd( u47.v, c47_0.v );
 

  w_tmp.v  = _mm256_broadcast_sd( (double*)( w + 1 ) );
  c03_1.v  = _mm256_mul_pd( w_tmp.v, c03_1.v );
  c47_1.v  = _mm256_mul_pd( w_tmp.v, c47_1.v );
  u03.v    = _mm256_add_pd( u03.v, c03_1.v );
  u47.v    = _mm256_add_pd( u47.v, c47_1.v );


  w_tmp.v  = _mm256_broadcast_sd( (double*)( w + 2 ) );
  c03_2.v  = _mm256_mul_pd( w_tmp.v, c03_2.v );
  c47_2.v  = _mm256_mul_pd( w_tmp.v, c47_2.v );
  u03.v    = _mm256_add_pd( u03.v, c03_2.v );
  u47.v    = _mm256_add_pd( u47.v, c47_2.v );


  w_tmp.v  = _mm256_broadcast_sd( (double*)( w + 3 ) );
  c03_3.v  = _mm256_mul_pd( w_tmp.v, c03_3.v );
  c47_3.v  = _mm256_mul_pd( w_tmp.v, c47_3.v );
  u03.v    = _mm256_add_pd( u03.v, c03_3.v );
  u47.v    = _mm256_add_pd( u47.v, c47_3.v );


  _mm256_store_pd( (double*)u, u03.v );
  _mm256_store_pd( (double*)( u + 4 ), u47.v );
}
