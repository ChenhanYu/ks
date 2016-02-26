#include <immintrin.h> // AVX
#include <ks.h>
#include <avx_type.h>

void ks_gaussian_int_d8x4(
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
  double neg2 = -2.0;
  double dzero = 0.0;
  double alpha = ker->scal;

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


  //if ( c03_0.d[ 0 ] != c03_0.d[ 0 ] ) printf( "rank-k c03_0[ 0 ] nan\n" );
  //if ( c03_0.d[ 1 ] != c03_0.d[ 1 ] ) printf( "rank-k c03_0[ 1 ] nan\n" );
  //if ( c03_0.d[ 2 ] != c03_0.d[ 2 ] ) printf( "rank-k c03_0[ 2 ] nan\n" );
  //if ( c03_0.d[ 3 ] != c03_0.d[ 3 ] ) printf( "rank-k c03_0[ 3 ] nan\n" );

  //if ( c03_1.d[ 0 ] != c03_1.d[ 0 ] ) printf( "rank-k c03_1[ 0 ] nan\n" );
  //if ( c03_1.d[ 1 ] != c03_1.d[ 1 ] ) printf( "rank-k c03_1[ 1 ] nan\n" );
  //if ( c03_1.d[ 2 ] != c03_1.d[ 2 ] ) printf( "rank-k c03_1[ 2 ] nan\n" );
  //if ( c03_1.d[ 3 ] != c03_1.d[ 3 ] ) printf( "rank-k c03_1[ 3 ] nan\n" );

  //if ( c03_2.d[ 0 ] != c03_2.d[ 0 ] ) printf( "rank-k c03_2[ 0 ] nan\n" );
  //if ( c03_2.d[ 1 ] != c03_2.d[ 1 ] ) printf( "rank-k c03_2[ 1 ] nan\n" );
  //if ( c03_2.d[ 2 ] != c03_2.d[ 2 ] ) printf( "rank-k c03_2[ 2 ] nan\n" );
  //if ( c03_2.d[ 3 ] != c03_2.d[ 3 ] ) printf( "rank-k c03_2[ 3 ] nan\n" );

  //if ( c03_3.d[ 0 ] != c03_3.d[ 0 ] ) printf( "rank-k c03_3[ 0 ] nan\n" );
  //if ( c03_3.d[ 1 ] != c03_3.d[ 1 ] ) printf( "rank-k c03_3[ 1 ] nan\n" );
  //if ( c03_3.d[ 2 ] != c03_3.d[ 2 ] ) printf( "rank-k c03_3[ 2 ] nan\n" );
  //if ( c03_3.d[ 3 ] != c03_3.d[ 3 ] ) printf( "rank-k c03_3[ 3 ] nan\n" );

  //if ( c47_0.d[ 0 ] != c47_0.d[ 0 ] ) printf( "rank-k c47_0[ 0 ] nan\n" );
  //if ( c47_0.d[ 1 ] != c47_0.d[ 1 ] ) printf( "rank-k c47_0[ 1 ] nan\n" );
  //if ( c47_0.d[ 2 ] != c47_0.d[ 2 ] ) printf( "rank-k c47_0[ 2 ] nan\n" );
  //if ( c47_0.d[ 3 ] != c47_0.d[ 3 ] ) printf( "rank-k c47_0[ 3 ] nan\n" );

  //if ( c47_1.d[ 0 ] != c47_1.d[ 0 ] ) printf( "rank-k c47_1[ 0 ] nan\n" );
  //if ( c47_1.d[ 1 ] != c47_1.d[ 1 ] ) printf( "rank-k c47_1[ 1 ] nan\n" );
  //if ( c47_1.d[ 2 ] != c47_1.d[ 2 ] ) printf( "rank-k c47_1[ 2 ] nan\n" );
  //if ( c47_1.d[ 3 ] != c47_1.d[ 3 ] ) printf( "rank-k c47_1[ 3 ] nan\n" );

  //if ( c47_2.d[ 0 ] != c47_2.d[ 0 ] ) printf( "rank-k c47_2[ 0 ] nan\n" );
  //if ( c47_2.d[ 1 ] != c47_2.d[ 1 ] ) printf( "rank-k c47_2[ 1 ] nan\n" );
  //if ( c47_2.d[ 2 ] != c47_2.d[ 2 ] ) printf( "rank-k c47_2[ 2 ] nan\n" );
  //if ( c47_2.d[ 3 ] != c47_2.d[ 3 ] ) printf( "rank-k c47_2[ 3 ] nan\n" );

  //if ( c47_3.d[ 0 ] != c47_3.d[ 0 ] ) printf( "rank-k c47_3[ 0 ] nan\n" );
  //if ( c47_3.d[ 1 ] != c47_3.d[ 1 ] ) printf( "rank-k c47_3[ 1 ] nan\n" );
  //if ( c47_3.d[ 2 ] != c47_3.d[ 2 ] ) printf( "rank-k c47_3[ 2 ] nan\n" );
  //if ( c47_3.d[ 3 ] != c47_3.d[ 3 ] ) printf( "rank-k c47_3[ 3 ] nan\n" );

  //if ( aa[ 0 ] != aa[ 0 ] ) printf( "aa[ 0 ] nan\n" );
  //if ( aa[ 1 ] != aa[ 1 ] ) printf( "aa[ 1 ] nan\n" );
  //if ( aa[ 2 ] != aa[ 2 ] ) printf( "aa[ 2 ] nan\n" );
  //if ( aa[ 3 ] != aa[ 3 ] ) printf( "aa[ 3 ] nan\n" );
  //if ( aa[ 4 ] != aa[ 4 ] ) printf( "aa[ 4 ] nan\n" );
  //if ( aa[ 5 ] != aa[ 5 ] ) printf( "aa[ 5 ] nan\n" );
  //if ( aa[ 6 ] != aa[ 6 ] ) printf( "aa[ 6 ] nan\n" );
  //if ( aa[ 7 ] != aa[ 7 ] ) printf( "aa[ 7 ] nan\n" );

  //if ( bb[ 0 ] != bb[ 0 ] ) printf( "bb[ 0 ] nan\n" );
  //if ( bb[ 1 ] != bb[ 1 ] ) printf( "bb[ 1 ] nan\n" );
  //if ( bb[ 2 ] != bb[ 2 ] ) printf( "bb[ 2 ] nan\n" );
  //if ( bb[ 3 ] != bb[ 3 ] ) printf( "bb[ 3 ] nan\n" );



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


  //if ( c03_1.d[ 0 ] != c03_1.d[ 0 ] ) printf( "max c03_1[ 0 ] nan\n" );
  //if ( c03_1.d[ 1 ] != c03_1.d[ 1 ] ) printf( "max c03_1[ 1 ] nan\n" );
  //if ( c03_1.d[ 2 ] != c03_1.d[ 2 ] ) printf( "max c03_1[ 2 ] nan\n" );
  //if ( c03_1.d[ 3 ] != c03_1.d[ 3 ] ) printf( "max c03_1[ 3 ] nan\n" );

  //if ( c03_2.d[ 0 ] != c03_2.d[ 0 ] ) printf( "max c03_2[ 0 ] nan\n" );
  //if ( c03_2.d[ 1 ] != c03_2.d[ 1 ] ) printf( "max c03_2[ 1 ] nan\n" );
  //if ( c03_2.d[ 2 ] != c03_2.d[ 2 ] ) printf( "max c03_2[ 2 ] nan\n" );
  //if ( c03_2.d[ 3 ] != c03_2.d[ 3 ] ) printf( "max c03_2[ 3 ] nan\n" );

  //if ( c03_3.d[ 0 ] != c03_3.d[ 0 ] ) printf( "max c03_3[ 0 ] nan\n" );
  //if ( c03_3.d[ 1 ] != c03_3.d[ 1 ] ) printf( "max c03_3[ 1 ] nan\n" );
  //if ( c03_3.d[ 2 ] != c03_3.d[ 2 ] ) printf( "max c03_3[ 2 ] nan\n" );
  //if ( c03_3.d[ 3 ] != c03_3.d[ 3 ] ) printf( "max c03_3[ 3 ] nan\n" );

  //if ( c47_0.d[ 0 ] != c47_0.d[ 0 ] ) printf( "max c47_0[ 0 ] nan\n" );
  //if ( c47_0.d[ 1 ] != c47_0.d[ 1 ] ) printf( "max c47_0[ 1 ] nan\n" );
  //if ( c47_0.d[ 2 ] != c47_0.d[ 2 ] ) printf( "max c47_0[ 2 ] nan\n" );
  //if ( c47_0.d[ 3 ] != c47_0.d[ 3 ] ) printf( "max c47_0[ 3 ] nan\n" );

  //if ( c47_1.d[ 0 ] != c47_1.d[ 0 ] ) printf( "max c47_1[ 0 ] nan\n" );
  //if ( c47_1.d[ 1 ] != c47_1.d[ 1 ] ) printf( "max c47_1[ 1 ] nan\n" );
  //if ( c47_1.d[ 2 ] != c47_1.d[ 2 ] ) printf( "max c47_1[ 2 ] nan\n" );
  //if ( c47_1.d[ 3 ] != c47_1.d[ 3 ] ) printf( "max c47_1[ 3 ] nan\n" );

  //if ( c47_2.d[ 0 ] != c47_2.d[ 0 ] ) printf( "max c47_2[ 0 ] nan\n" );
  //if ( c47_2.d[ 1 ] != c47_2.d[ 1 ] ) printf( "max c47_2[ 1 ] nan\n" );
  //if ( c47_2.d[ 2 ] != c47_2.d[ 2 ] ) printf( "max c47_2[ 2 ] nan\n" );
  //if ( c47_2.d[ 3 ] != c47_2.d[ 3 ] ) printf( "max c47_2[ 3 ] nan\n" );

  //if ( c47_3.d[ 0 ] != c47_3.d[ 0 ] ) printf( "max c47_3[ 0 ] nan\n" );
  //if ( c47_3.d[ 1 ] != c47_3.d[ 1 ] ) printf( "max c47_3[ 1 ] nan\n" );
  //if ( c47_3.d[ 2 ] != c47_3.d[ 2 ] ) printf( "max c47_3[ 2 ] nan\n" );
  //if ( c47_3.d[ 3 ] != c47_3.d[ 3 ] ) printf( "max c47_3[ 3 ] nan\n" );




  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( aa ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( bb ) );


  // Accumulate







  // Scale before the kernel evaluation
  aa_tmp.v = _mm256_broadcast_sd( &alpha );
  c03_0.v  = _mm256_mul_pd( aa_tmp.v, c03_0.v );
  c03_1.v  = _mm256_mul_pd( aa_tmp.v, c03_1.v );
  c03_2.v  = _mm256_mul_pd( aa_tmp.v, c03_2.v );
  c03_3.v  = _mm256_mul_pd( aa_tmp.v, c03_3.v );
  c47_0.v  = _mm256_mul_pd( aa_tmp.v, c47_0.v );
  c47_1.v  = _mm256_mul_pd( aa_tmp.v, c47_1.v );
  c47_2.v  = _mm256_mul_pd( aa_tmp.v, c47_2.v );
  c47_3.v  = _mm256_mul_pd( aa_tmp.v, c47_3.v );


  // Preload u03, u47
  u03.v    = _mm256_load_pd( (double*)u );
  u47.v    = _mm256_load_pd( (double*)( u + 4 ) );


  // Prefetch u and w
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u + 8 ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( w ) );


  // c = exp( c );
  #include "ks_exp_int_d8x4.h"


  // Multiple rhs kernel summation.
  #include "ks_kernel_summation_int_d8x4.h"

  //if ( c03_0.d[ 0 ] != c03_0.d[ 0 ] ) printf( "c03_0[ 0 ] nan\n" );
  //if ( c03_0.d[ 1 ] != c03_0.d[ 1 ] ) printf( "c03_0[ 1 ] nan\n" );
  //if ( c03_0.d[ 2 ] != c03_0.d[ 2 ] ) printf( "c03_0[ 2 ] nan\n" );
  //if ( c03_0.d[ 3 ] != c03_0.d[ 3 ] ) printf( "c03_0[ 3 ] nan\n" );

  //if ( c03_1.d[ 0 ] != c03_1.d[ 0 ] ) printf( "c03_1[ 0 ] nan\n" );
  //if ( c03_1.d[ 1 ] != c03_1.d[ 1 ] ) printf( "c03_1[ 1 ] nan\n" );
  //if ( c03_1.d[ 2 ] != c03_1.d[ 2 ] ) printf( "c03_1[ 2 ] nan\n" );
  //if ( c03_1.d[ 3 ] != c03_1.d[ 3 ] ) printf( "c03_1[ 3 ] nan\n" );

  //if ( c03_2.d[ 0 ] != c03_2.d[ 0 ] ) printf( "c03_2[ 0 ] nan\n" );
  //if ( c03_2.d[ 1 ] != c03_2.d[ 1 ] ) printf( "c03_2[ 1 ] nan\n" );
  //if ( c03_2.d[ 2 ] != c03_2.d[ 2 ] ) printf( "c03_2[ 2 ] nan\n" );
  //if ( c03_2.d[ 3 ] != c03_2.d[ 3 ] ) printf( "c03_2[ 3 ] nan\n" );

  //if ( c03_3.d[ 0 ] != c03_3.d[ 0 ] ) printf( "c03_3[ 0 ] nan\n" );
  //if ( c03_3.d[ 1 ] != c03_3.d[ 1 ] ) printf( "c03_3[ 1 ] nan\n" );
  //if ( c03_3.d[ 2 ] != c03_3.d[ 2 ] ) printf( "c03_3[ 2 ] nan\n" );
  //if ( c03_3.d[ 3 ] != c03_3.d[ 3 ] ) printf( "c03_3[ 3 ] nan\n" );

  //if ( c47_0.d[ 0 ] != c47_0.d[ 0 ] ) printf( "c47_0[ 0 ] nan\n" );
  //if ( c47_0.d[ 1 ] != c47_0.d[ 1 ] ) printf( "c47_0[ 1 ] nan\n" );
  //if ( c47_0.d[ 2 ] != c47_0.d[ 2 ] ) printf( "c47_0[ 2 ] nan\n" );
  //if ( c47_0.d[ 3 ] != c47_0.d[ 3 ] ) printf( "c47_0[ 3 ] nan\n" );

  //if ( c47_1.d[ 0 ] != c47_1.d[ 0 ] ) printf( "c47_1[ 0 ] nan\n" );
  //if ( c47_1.d[ 1 ] != c47_1.d[ 1 ] ) printf( "c47_1[ 1 ] nan\n" );
  //if ( c47_1.d[ 2 ] != c47_1.d[ 2 ] ) printf( "c47_1[ 2 ] nan\n" );
  //if ( c47_1.d[ 3 ] != c47_1.d[ 3 ] ) printf( "c47_1[ 3 ] nan\n" );

  //if ( c47_2.d[ 0 ] != c47_2.d[ 0 ] ) printf( "c47_2[ 0 ] nan\n" );
  //if ( c47_2.d[ 1 ] != c47_2.d[ 1 ] ) printf( "c47_2[ 1 ] nan\n" );
  //if ( c47_2.d[ 2 ] != c47_2.d[ 2 ] ) printf( "c47_2[ 2 ] nan\n" );
  //if ( c47_2.d[ 3 ] != c47_2.d[ 3 ] ) printf( "c47_2[ 3 ] nan\n" );

  //if ( c47_3.d[ 0 ] != c47_3.d[ 0 ] ) printf( "c47_3[ 0 ] nan\n" );
  //if ( c47_3.d[ 1 ] != c47_3.d[ 1 ] ) printf( "c47_3[ 1 ] nan\n" );
  //if ( c47_3.d[ 2 ] != c47_3.d[ 2 ] ) printf( "c47_3[ 2 ] nan\n" );
  //if ( c47_3.d[ 3 ] != c47_3.d[ 3 ] ) printf( "c47_3[ 3 ] nan\n" );

  //if ( u[ 0 ] != u[ 0 ] ) printf( "u[ 0 ] nan\n" );
  //if ( u[ 1 ] != u[ 1 ] ) printf( "u[ 1 ] nan\n" );
  //if ( u[ 2 ] != u[ 2 ] ) printf( "u[ 2 ] nan\n" );
  //if ( u[ 3 ] != u[ 3 ] ) printf( "u[ 3 ] nan\n" );
  //if ( u[ 4 ] != u[ 4 ] ) printf( "u[ 4 ] nan\n" );
  //if ( u[ 5 ] != u[ 5 ] ) printf( "u[ 5 ] nan\n" );
  //if ( u[ 6 ] != u[ 6 ] ) printf( "u[ 6 ] nan\n" );
  //if ( u[ 7 ] != u[ 7 ] ) printf( "u[ 7 ] nan\n" );

  //if ( w[ 0 ] != w[ 0 ] ) printf( "w[ 0 ] nan\n" );
  //if ( w[ 1 ] != w[ 1 ] ) printf( "w[ 1 ] nan\n" );
  //if ( w[ 2 ] != w[ 2 ] ) printf( "w[ 2 ] nan\n" );
  //if ( w[ 3 ] != w[ 3 ] ) printf( "w[ 3 ] nan\n" );
}
