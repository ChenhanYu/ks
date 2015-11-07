#include <immintrin.h> // AVX
#include <ks.h>

// IEEE-754 double
// 0      7 8     15 16    23 24    31 32    39 40    47 48    55 56    63
// SEEEEEEE EEEEMMMM MMMMMMMM MMMMMMMM MMMMMMMM MMMMMMMM MMMMMMMM MMMMMMMM
//
// Exponent Max ( Exponent Mask ) 0x 7F F0 00 00 00 00 00 00
// 01111111 11110000 00000000 00000000 00000000 00000000 00000000 00000000
//
// Mantissa Max ( Mantissa Mask ) 0x 00 0F FF FF FF FF FF FF
// 00000000 00001111 11111111 11111111 11111111 11111111 11111111 11111111
//
// Extract exponent from x
// maske = _mm256_set1_pd( 0x7FF0000000000000 );
// i = _mm256_castpd_si256( x )
// e = _mm256_and_pf
//
//

void ks_polynomial_int_d8x4(
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
  int    i, rhs_left;
  double dzero = 0.0;
  double neg2  = -2.0;
  double powe  = ker->powe;
  double scal  = ker->scal;
  double cons  = ker->cons;


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


  // C = scal * C
  aa_tmp.v = _mm256_broadcast_sd( &scal );
  c03_0.v  = _mm256_mul_pd( aa_tmp.v, c03_0.v );
  c03_1.v  = _mm256_mul_pd( aa_tmp.v, c03_1.v );
  c03_2.v  = _mm256_mul_pd( aa_tmp.v, c03_2.v );
  c03_3.v  = _mm256_mul_pd( aa_tmp.v, c03_3.v );
  c47_0.v  = _mm256_mul_pd( aa_tmp.v, c47_0.v );
  c47_1.v  = _mm256_mul_pd( aa_tmp.v, c47_1.v );
  c47_2.v  = _mm256_mul_pd( aa_tmp.v, c47_2.v );
  c47_3.v  = _mm256_mul_pd( aa_tmp.v, c47_3.v );


  // C += cons
  aa_tmp.v = _mm256_broadcast_sd( &cons );
  c03_0.v  = _mm256_add_pd( aa_tmp.v, c03_0.v );
  c03_1.v  = _mm256_add_pd( aa_tmp.v, c03_1.v );
  c03_2.v  = _mm256_add_pd( aa_tmp.v, c03_2.v );
  c03_3.v  = _mm256_add_pd( aa_tmp.v, c03_3.v );
  c47_0.v  = _mm256_add_pd( aa_tmp.v, c47_0.v );
  c47_1.v  = _mm256_add_pd( aa_tmp.v, c47_1.v );
  c47_2.v  = _mm256_add_pd( aa_tmp.v, c47_2.v );
  c47_3.v  = _mm256_add_pd( aa_tmp.v, c47_3.v );


  // Preload u03, u47
  u03.v    = _mm256_load_pd( u     );
  u47.v    = _mm256_load_pd( u + 4 );


  // Prefetch u and w
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u + 8 ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( w     ) );


  // c = pow( c, powe );
  if ( powe == 2.0 ) {
    c03_0.v   = _mm256_mul_pd( c03_0.v, c03_0.v ); 
    c03_1.v   = _mm256_mul_pd( c03_1.v, c03_1.v ); 
    c03_2.v   = _mm256_mul_pd( c03_2.v, c03_2.v ); 
    c03_3.v   = _mm256_mul_pd( c03_3.v, c03_3.v ); 
    c47_0.v   = _mm256_mul_pd( c47_0.v, c47_0.v ); 
    c47_1.v   = _mm256_mul_pd( c47_1.v, c47_1.v ); 
    c47_2.v   = _mm256_mul_pd( c47_2.v, c47_2.v ); 
    c47_3.v   = _mm256_mul_pd( c47_3.v, c47_3.v );
  }
  else if ( powe == 4.0 ) {
    c03_0.v   = _mm256_mul_pd( c03_0.v, c03_0.v ); 
    c03_1.v   = _mm256_mul_pd( c03_1.v, c03_1.v ); 
    c03_2.v   = _mm256_mul_pd( c03_2.v, c03_2.v ); 
    c03_3.v   = _mm256_mul_pd( c03_3.v, c03_3.v ); 
    c47_0.v   = _mm256_mul_pd( c47_0.v, c47_0.v ); 
    c47_1.v   = _mm256_mul_pd( c47_1.v, c47_1.v ); 
    c47_2.v   = _mm256_mul_pd( c47_2.v, c47_2.v ); 
    c47_3.v   = _mm256_mul_pd( c47_3.v, c47_3.v );
    c03_0.v   = _mm256_mul_pd( c03_0.v, c03_0.v ); 
    c03_1.v   = _mm256_mul_pd( c03_1.v, c03_1.v ); 
    c03_2.v   = _mm256_mul_pd( c03_2.v, c03_2.v ); 
    c03_3.v   = _mm256_mul_pd( c03_3.v, c03_3.v ); 
    c47_0.v   = _mm256_mul_pd( c47_0.v, c47_0.v ); 
    c47_1.v   = _mm256_mul_pd( c47_1.v, c47_1.v ); 
    c47_2.v   = _mm256_mul_pd( c47_2.v, c47_2.v ); 
    c47_3.v   = _mm256_mul_pd( c47_3.v, c47_3.v );
  }
  else {
#ifdef USE_VML    
    c_tmp.v   = _mm256_broadcast_sd( &powe );
    c03_0.v   = _mm256_pow_pd( c03_0.v, c_tmp.v ); 
    c03_1.v   = _mm256_pow_pd( c03_1.v, c_tmp.v ); 
    c03_2.v   = _mm256_pow_pd( c03_2.v, c_tmp.v ); 
    c03_3.v   = _mm256_pow_pd( c03_3.v, c_tmp.v ); 
    c47_0.v   = _mm256_pow_pd( c47_0.v, c_tmp.v ); 
    c47_1.v   = _mm256_pow_pd( c47_1.v, c_tmp.v ); 
    c47_2.v   = _mm256_pow_pd( c47_2.v, c_tmp.v ); 
    c47_3.v   = _mm256_pow_pd( c47_3.v, c_tmp.v );
#else
    c03_0.d[ 0 ] = pow( c03_0.d[ 0 ], powe );
    c03_0.d[ 1 ] = pow( c03_0.d[ 1 ], powe );
    c03_0.d[ 2 ] = pow( c03_0.d[ 2 ], powe );
    c03_0.d[ 3 ] = pow( c03_0.d[ 3 ], powe );
    c03_1.d[ 0 ] = pow( c03_1.d[ 0 ], powe );
    c03_1.d[ 1 ] = pow( c03_1.d[ 1 ], powe );
    c03_1.d[ 2 ] = pow( c03_1.d[ 2 ], powe );
    c03_1.d[ 3 ] = pow( c03_1.d[ 3 ], powe );
    c03_2.d[ 0 ] = pow( c03_2.d[ 0 ], powe );
    c03_2.d[ 1 ] = pow( c03_2.d[ 1 ], powe );
    c03_2.d[ 2 ] = pow( c03_2.d[ 2 ], powe );
    c03_2.d[ 3 ] = pow( c03_2.d[ 3 ], powe );
    c03_3.d[ 0 ] = pow( c03_3.d[ 0 ], powe );
    c03_3.d[ 1 ] = pow( c03_3.d[ 1 ], powe );
    c03_3.d[ 2 ] = pow( c03_3.d[ 2 ], powe );
    c03_3.d[ 3 ] = pow( c03_3.d[ 3 ], powe );
    c47_0.d[ 0 ] = pow( c47_0.d[ 0 ], powe );
    c47_0.d[ 1 ] = pow( c47_0.d[ 1 ], powe );
    c47_0.d[ 2 ] = pow( c47_0.d[ 2 ], powe );
    c47_0.d[ 3 ] = pow( c47_0.d[ 3 ], powe );
    c47_1.d[ 0 ] = pow( c47_1.d[ 0 ], powe );
    c47_1.d[ 1 ] = pow( c47_1.d[ 1 ], powe );
    c47_1.d[ 2 ] = pow( c47_1.d[ 2 ], powe );
    c47_1.d[ 3 ] = pow( c47_1.d[ 3 ], powe );
    c47_2.d[ 0 ] = pow( c47_2.d[ 0 ], powe );
    c47_2.d[ 1 ] = pow( c47_2.d[ 1 ], powe );
    c47_2.d[ 2 ] = pow( c47_2.d[ 2 ], powe );
    c47_2.d[ 3 ] = pow( c47_2.d[ 3 ], powe );
    c47_3.d[ 0 ] = pow( c47_3.d[ 0 ], powe );
    c47_3.d[ 1 ] = pow( c47_3.d[ 1 ], powe );
    c47_3.d[ 2 ] = pow( c47_3.d[ 2 ], powe );
    c47_3.d[ 3 ] = pow( c47_3.d[ 3 ], powe );
#endif
  }
 

  // Multiple rhs kernel summation.
  #include "ks_kernel_summation_int_d8x4.h"
  
}
