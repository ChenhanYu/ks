#include <math.h>
#include <immintrin.h> // AVX
#include <ks.h>
#include <avx_type.h>

void gaussian_ref_d8x6(
    int    k,
    int    rhs,
    double *h,
    double *u,
    double *aa,
    double *a,
    double *bb,
    double *b,
    double *w,
    double *c,
    ks_t   *ker,
    aux_t  *aux
    )
{
  int    i, j, p;
  double K[ 8 * 6 ] = {{ 0.0 }};

  #include <rank_k_ref_d8x6.h>

  // Gaussian kernel
  for ( j = 0; j < 6; j ++ ) {
	for ( i = 0; i < 8; i ++ ) { 
	  K[ j * 8 + i ] = aa[ i ] - 2.0 * K[ j * 8 + i ] + bb[ j ];
      K[ j * 8 + i ] = exp( ker->scal * K[ j * 8 + i ] );
	  u[ i ] += K[ j * 8 + i ] * w[ j ];
	}
  }
}


void gaussian_int_d8x6(
    int    k,
    int    rhs,
    double *h,
    double *u,
    double *aa,
    double *a,
    double *bb,
    double *b,
    double *w,
    double *c,
    ks_t   *ker,
    aux_t  *aux
    )
{
  int    i;
  double alpha = ker->scal;
  // 16 registers.
  v4df_t c03_0, c03_1, c03_2, c03_3, c03_4, c03_5;
  v4df_t c47_0, c47_1, c47_2, c47_3, c47_4, c47_5;
  v4df_t a03, a47, b0, b1;

  #include <rank_k_int_d8x6.h>

  /*
  // Prefetch aa and bb
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( aa ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( bb ) );

  // Scale -2
  a03.v   = _mm256_broadcast_sd( &neg2 );
  c03_0.v = _mm256_mul_pd( a03.v, c03_0.v );
  c03_1.v = _mm256_mul_pd( a03.v, c03_1.v );
  c03_2.v = _mm256_mul_pd( a03.v, c03_2.v );
  c03_3.v = _mm256_mul_pd( a03.v, c03_3.v );
  c03_4.v = _mm256_mul_pd( a03.v, c03_4.v );
  c03_5.v = _mm256_mul_pd( a03.v, c03_5.v );

  c47_0.v = _mm256_mul_pd( a03.v, c47_0.v );
  c47_1.v = _mm256_mul_pd( a03.v, c47_1.v );
  c47_2.v = _mm256_mul_pd( a03.v, c47_2.v );
  c47_3.v = _mm256_mul_pd( a03.v, c47_3.v );
  c47_4.v = _mm256_mul_pd( a03.v, c47_4.v );
  c47_5.v = _mm256_mul_pd( a03.v, c47_5.v );

  a03.v   = _mm256_load_pd( (double*)aa );
  c03_0.v = _mm256_add_pd( a03.v, c03_0.v );
  c03_1.v = _mm256_add_pd( a03.v, c03_1.v );
  c03_2.v = _mm256_add_pd( a03.v, c03_2.v );
  c03_3.v = _mm256_add_pd( a03.v, c03_3.v );
  c03_4.v = _mm256_add_pd( a03.v, c03_4.v );
  c03_5.v = _mm256_add_pd( a03.v, c03_5.v );

  a47.v   = _mm256_load_pd( (double*)( aa + 4 ) );
  c47_0.v = _mm256_add_pd( a47.v, c47_0.v );
  c47_1.v = _mm256_add_pd( a47.v, c47_1.v );
  c47_2.v = _mm256_add_pd( a47.v, c47_2.v );
  c47_3.v = _mm256_add_pd( a47.v, c47_3.v );
  c47_4.v = _mm256_add_pd( a47.v, c47_4.v );
  c47_5.v = _mm256_add_pd( a47.v, c47_5.v );
  
  b0.v    = _mm256_broadcast_sd( (double*)( bb     ) );
  c03_0.v = _mm256_add_pd( b0.v, c03_0.v );
  c47_0.v = _mm256_add_pd( b0.v, c47_0.v );

  b1.v    = _mm256_broadcast_sd( (double*)( bb + 1 ) );
  c03_1.v = _mm256_add_pd( b1.v, c03_1.v );
  c47_1.v = _mm256_add_pd( b1.v, c47_1.v );

  b0.v    = _mm256_broadcast_sd( (double*)( bb + 2 ) );
  c03_2.v = _mm256_add_pd( b0.v, c03_2.v );
  c47_2.v = _mm256_add_pd( b0.v, c47_2.v );

  b1.v    = _mm256_broadcast_sd( (double*)( bb + 3 ) );
  c03_3.v = _mm256_add_pd( b1.v, c03_3.v );
  c47_3.v = _mm256_add_pd( b1.v, c47_3.v );

  b0.v    = _mm256_broadcast_sd( (double*)( bb + 4 ) );
  c03_4.v = _mm256_add_pd( b0.v, c03_4.v );
  c47_4.v = _mm256_add_pd( b0.v, c47_4.v );

  b1.v    = _mm256_broadcast_sd( (double*)( bb + 5 ) );
  c03_5.v = _mm256_add_pd( b1.v, c03_5.v );
  c47_5.v = _mm256_add_pd( b1.v, c47_5.v );

  // Check if there is any illegle value 
  a03.v   = _mm256_broadcast_sd( &dzero );
  c03_0.v = _mm256_max_pd( a03.v, c03_0.v );
  c03_1.v = _mm256_max_pd( a03.v, c03_1.v );
  c03_2.v = _mm256_max_pd( a03.v, c03_2.v );
  c03_3.v = _mm256_max_pd( a03.v, c03_3.v );
  c03_4.v = _mm256_max_pd( a03.v, c03_4.v );
  c03_5.v = _mm256_max_pd( a03.v, c03_5.v );

  c47_0.v = _mm256_max_pd( a03.v, c47_0.v );
  c47_1.v = _mm256_max_pd( a03.v, c47_1.v );
  c47_2.v = _mm256_max_pd( a03.v, c47_2.v );
  c47_3.v = _mm256_max_pd( a03.v, c47_3.v );
  c47_4.v = _mm256_max_pd( a03.v, c47_4.v );
  c47_5.v = _mm256_max_pd( a03.v, c47_5.v );
  */

  #include <sq2nrm_int_d8x6.h>

  // Scale before the kernel evaluation
  a03.v   = _mm256_broadcast_sd( &alpha );
  c03_0.v = _mm256_mul_pd( a03.v, c03_0.v );
  c03_1.v = _mm256_mul_pd( a03.v, c03_1.v );
  c03_2.v = _mm256_mul_pd( a03.v, c03_2.v );
  c03_3.v = _mm256_mul_pd( a03.v, c03_3.v );
  c03_4.v = _mm256_mul_pd( a03.v, c03_4.v );
  c03_5.v = _mm256_mul_pd( a03.v, c03_5.v );

  c47_0.v = _mm256_mul_pd( a03.v, c47_0.v );
  c47_1.v = _mm256_mul_pd( a03.v, c47_1.v );
  c47_2.v = _mm256_mul_pd( a03.v, c47_2.v );
  c47_3.v = _mm256_mul_pd( a03.v, c47_3.v );
  c47_4.v = _mm256_mul_pd( a03.v, c47_4.v );
  c47_5.v = _mm256_mul_pd( a03.v, c47_5.v );

  // Prefetch u, w
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( w ) );

  // c = exp( c )
  c03_0.v = _mm256_exp_pd( c03_0.v );
  c03_1.v = _mm256_exp_pd( c03_1.v );
  c03_2.v = _mm256_exp_pd( c03_2.v );
  c03_3.v = _mm256_exp_pd( c03_3.v );
  c03_4.v = _mm256_exp_pd( c03_4.v );
  c03_5.v = _mm256_exp_pd( c03_5.v );

  c47_0.v = _mm256_exp_pd( c47_0.v );
  c47_1.v = _mm256_exp_pd( c47_1.v );
  c47_2.v = _mm256_exp_pd( c47_2.v );
  c47_3.v = _mm256_exp_pd( c47_3.v );
  c47_4.v = _mm256_exp_pd( c47_4.v );
  c47_5.v = _mm256_exp_pd( c47_5.v );

  // Preload u03, u47
  a03.v    = _mm256_load_pd( (double*)  u       );
  a47.v    = _mm256_load_pd( (double*)( u + 4 ) );

  // Multiple rhs weighted sum.
  #include<weighted_sum_int_d8x6.h>
}
