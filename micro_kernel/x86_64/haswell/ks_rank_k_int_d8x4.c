#include <immintrin.h> // AVX
#include <ks.h>


void ks_rank_k_int_d8x4(
    int    k,
    double *a,
    double *b,
    double *c,
    int    ldc,
    aux_t  *aux
    )
{
  int    i;
  double neg2 = -2.0;
  v4df_t c03_0, c03_1, c03_2, c03_3;
  v4df_t c47_0, c47_1, c47_2, c47_3;
  v4df_t tmpc03_0, tmpc03_1, tmpc03_2, tmpc03_3;
  v4df_t tmpc47_0, tmpc47_1, tmpc47_2, tmpc47_3;
  v4df_t c_tmp;
  v4df_t a03, a47;
  v4df_t A03, A47; // prefetched A 
  v4df_t b0, b1, b2, b3;
  v4df_t B0; // prefetched B
  v4df_t aa_tmp, bb_tmp;


  #include "ks_rank_k_int_d8x4.h"

 
  if ( aux->pc != 0 ) {
    tmpc03_0.v = _mm256_load_pd( (double*)( c      ) );
    tmpc47_0.v = _mm256_load_pd( (double*)( c + 4  ) );

    tmpc03_1.v = _mm256_load_pd( (double*)( c + 8  ) );
    tmpc47_1.v = _mm256_load_pd( (double*)( c + 12 ) );

    tmpc03_2.v = _mm256_load_pd( (double*)( c + 16 ) );
    tmpc47_2.v = _mm256_load_pd( (double*)( c + 20 ) );

    tmpc03_3.v = _mm256_load_pd( (double*)( c + 24 ) );
    tmpc47_3.v = _mm256_load_pd( (double*)( c + 28 ) );
    

    c03_0.v    = _mm256_add_pd( tmpc03_0.v, c03_0.v );
    c47_0.v    = _mm256_add_pd( tmpc47_0.v, c47_0.v );

    c03_1.v    = _mm256_add_pd( tmpc03_1.v, c03_1.v );
    c47_1.v    = _mm256_add_pd( tmpc47_1.v, c47_1.v );

    c03_2.v    = _mm256_add_pd( tmpc03_2.v, c03_2.v );
    c47_2.v    = _mm256_add_pd( tmpc47_2.v, c47_2.v );

    c03_3.v    = _mm256_add_pd( tmpc03_3.v, c03_3.v );
    c47_3.v    = _mm256_add_pd( tmpc47_3.v, c47_3.v );
  }


  // packed
  _mm256_store_pd( (double*)( c      ), c03_0.v );
  _mm256_store_pd( (double*)( c + 4  ), c47_0.v );

  _mm256_store_pd( (double*)( c + 8  ), c03_1.v );
  _mm256_store_pd( (double*)( c + 12 ), c47_1.v );

  _mm256_store_pd( (double*)( c + 16 ), c03_2.v );
  _mm256_store_pd( (double*)( c + 20 ), c47_2.v );

  _mm256_store_pd( (double*)( c + 24 ), c03_3.v );
  _mm256_store_pd( (double*)( c + 28 ), c47_3.v );


  //printf( "ldc = %d\n", ldc );
  //printf( "%lf, %lf, %lf, %lf\n", c[0], c[ ldc + 0], c[ ldc * 2 + 0], c[ ldc * 3 + 0] );
  //printf( "%lf, %lf, %lf, %lf\n", c[1], c[ ldc + 1], c[ ldc * 2 + 1], c[ ldc * 3 + 1] );
  //printf( "%lf, %lf, %lf, %lf\n", c[2], c[ ldc + 2], c[ ldc * 2 + 2], c[ ldc * 3 + 2] );
  //printf( "%lf, %lf, %lf, %lf\n", c[3], c[ ldc + 3], c[ ldc * 2 + 3], c[ ldc * 3 + 3] );
  //printf( "%lf, %lf, %lf, %lf\n", c[4], c[ ldc + 4], c[ ldc * 2 + 4], c[ ldc * 3 + 4] );
  //printf( "%lf, %lf, %lf, %lf\n", c[5], c[ ldc + 5], c[ ldc * 2 + 5], c[ ldc * 3 + 5] );
  //printf( "%lf, %lf, %lf, %lf\n", c[6], c[ ldc + 6], c[ ldc * 2 + 6], c[ ldc * 3 + 6] );
  //printf( "%lf, %lf, %lf, %lf\n", c[7], c[ ldc + 7], c[ ldc * 2 + 7], c[ ldc * 3 + 7] );
}
