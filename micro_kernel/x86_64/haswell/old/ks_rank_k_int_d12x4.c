#include <immintrin.h> // AVX
#include <ks.h>


void ks_rank_k_int_d12x4(
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
  v4df_t c81_0, c81_1, c81_2, c81_3;
  v4df_t c_tmp;
  v4df_t a03, a47, a81;
  v4df_t b0, b1, b2, b3;

  __asm__ volatile( "prefetcht0 768(%0)    \n\t" : :"r"( c ) );
  __asm__ volatile( "prefetcht0 1536(%0)    \n\t" : :"r"( c ) );
  __asm__ volatile( "prefetcht0 2304(%0)    \n\t" : :"r"( c ) );

  //#include "ks_rank_k_int_d12x4.h"
  #include "ks_rank_k_opt_d12x4.h"

 
  if ( aux->pc != 0 ) {
    c_tmp.v = _mm256_load_pd( (double*)( c      ) );
    c03_0.v = _mm256_add_pd( c_tmp.v, c03_0.v );
    _mm256_store_pd( (double*)( c      ), c03_0.v );

    c_tmp.v = _mm256_load_pd( (double*)( c + 4  ) );
    c47_0.v = _mm256_add_pd( c_tmp.v, c47_0.v );
    _mm256_store_pd( (double*)( c + 4  ), c47_0.v );

    c_tmp.v = _mm256_load_pd( (double*)( c + 8  ) );
    c81_0.v = _mm256_add_pd( c_tmp.v, c81_0.v );
    _mm256_store_pd( (double*)( c + 8  ), c81_0.v );

    c_tmp.v = _mm256_load_pd( (double*)( c + 12 ) );
    c03_1.v = _mm256_add_pd( c_tmp.v, c03_1.v );
    _mm256_store_pd( (double*)( c + 12 ), c03_1.v );

    c_tmp.v = _mm256_load_pd( (double*)( c + 16 ) );
    c47_1.v = _mm256_add_pd( c_tmp.v, c47_1.v );
    _mm256_store_pd( (double*)( c + 16 ), c47_1.v );

    c_tmp.v = _mm256_load_pd( (double*)( c + 20 ) );
    c81_1.v = _mm256_add_pd( c_tmp.v, c81_1.v );
    _mm256_store_pd( (double*)( c + 20 ), c81_1.v );

    c_tmp.v = _mm256_load_pd( (double*)( c + 24 ) );
    c03_2.v = _mm256_add_pd( c_tmp.v, c03_2.v );
    _mm256_store_pd( (double*)( c + 24 ), c03_2.v );

    c_tmp.v = _mm256_load_pd( (double*)( c + 28 ) );
    c47_2.v = _mm256_add_pd( c_tmp.v, c47_2.v );
    _mm256_store_pd( (double*)( c + 28 ), c47_2.v );

    c_tmp.v = _mm256_load_pd( (double*)( c + 32 ) );
    c81_2.v = _mm256_add_pd( c_tmp.v, c81_2.v );
    _mm256_store_pd( (double*)( c + 32 ), c81_2.v );

    c_tmp.v = _mm256_load_pd( (double*)( c + 36 ) );
    c03_3.v = _mm256_add_pd( c_tmp.v, c03_3.v );
    _mm256_store_pd( (double*)( c + 36 ), c03_3.v );

    c_tmp.v = _mm256_load_pd( (double*)( c + 40 ) );
    c47_3.v = _mm256_add_pd( c_tmp.v, c47_3.v );
    _mm256_store_pd( (double*)( c + 40 ), c47_3.v );

    c_tmp.v = _mm256_load_pd( (double*)( c + 44 ) );
    c81_3.v = _mm256_add_pd( c_tmp.v, c81_3.v );
    _mm256_store_pd( (double*)( c + 44 ), c81_3.v );
  }
  else {
    // packed
    _mm256_store_pd( (double*)( c      ), c03_0.v );
    _mm256_store_pd( (double*)( c + 4  ), c47_0.v );
    _mm256_store_pd( (double*)( c + 8  ), c81_0.v );

    _mm256_store_pd( (double*)( c + 12 ), c03_1.v );
    _mm256_store_pd( (double*)( c + 16 ), c47_1.v );
    _mm256_store_pd( (double*)( c + 20 ), c81_1.v );

    _mm256_store_pd( (double*)( c + 24 ), c03_2.v );
    _mm256_store_pd( (double*)( c + 28 ), c47_2.v );
    _mm256_store_pd( (double*)( c + 32 ), c81_2.v );

    _mm256_store_pd( (double*)( c + 36 ), c03_3.v );
    _mm256_store_pd( (double*)( c + 40 ), c47_3.v );
    _mm256_store_pd( (double*)( c + 44 ), c81_3.v );
  }

}
