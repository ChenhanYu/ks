#include <immintrin.h> // AVX
#include <ks.h>


void ks_rank_k_int_d8x4_unroll_4(
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
  double *b_next;

  v4df_t c03_0, c03_1, c03_2, c03_3;
  v4df_t c47_0, c47_1, c47_2, c47_3;
  v4df_t tmpc03_0, tmpc03_1, tmpc03_2, tmpc03_3;
  v4df_t tmpc47_0, tmpc47_1, tmpc47_2, tmpc47_3;
  v4df_t c_tmp0, c_tmp1;
  v4df_t a03, a47;
  v4df_t b0;
  v4df_t b0x5, b0x3_0, b0x3_1;


  // In this case, k must be kc.
  int k_iter = k / 4;
  int k_left = k % 4;

  b_next = aux->b_next;
  b_next -= 4 * DKS_NR; 


  // Load a03
  a03.v = _mm256_load_pd( a );

  // Load ( b0, b1, b2, b3 )
  b0.v  = _mm256_load_pd( b );

  // Permute b0 to get b0x5
  b0x5.v = _mm256_permute_pd( b0.v, 0x5 );


  // Prefetch c03_0, c47_0
  __asm__ volatile( "prefetcht0  3 * 8(%0)    \n\t" : :"r"( c ) );

  // Prefetch c03_1, c47_1
  __asm__ volatile( "prefetcht0 11 * 8(%0)    \n\t" : :"r"( c ) );

  // Prefetch c03_2, c47_2
  __asm__ volatile( "prefetcht0 19 * 8(%0)    \n\t" : :"r"( c ) );

  // Prefetch c03_2, c47_2
  __asm__ volatile( "prefetcht0 27 * 8(%0)    \n\t" : :"r"( c ) );


  // Set the rank-kc update buffer to zero
  c03_0.v = _mm256_setzero_pd();
  c03_1.v = _mm256_setzero_pd();
  c03_2.v = _mm256_setzero_pd();
  c03_3.v = _mm256_setzero_pd();
  c47_0.v = _mm256_setzero_pd();
  c47_1.v = _mm256_setzero_pd();
  c47_2.v = _mm256_setzero_pd();
  c47_3.v = _mm256_setzero_pd();


  // Main loop
  for ( i = 0; i < k_iter; ++ i ) {
    b_next += 4 * DKS_NR;


    // Interation #0


    // Load a47
    a47.v    = _mm256_load_pd( a + 4 );
    c_tmp0.v = _mm256_mul_pd( a03.v, b0.v );
    b0x3_0.v = _mm256_permute2f128_pd( b0.v, b0.v, 0x3 );
    c_tmp1.v = _mm256_mul_pd( a03.v, b0x5.v );   
    b0x3_1.v = _mm256_permute2f128_pd( b0x5.v, b0x5.v, 0x3 );
    c03_0.v  = _mm256_add_pd( c03_0.v, c_tmp0.v );
    c03_1.v  = _mm256_add_pd( c03_1.v, c_tmp1.v );

    // Prefetch a03, a47 for 8 iteration later
    __asm__ volatile( "prefetcht0 16 * 32(%0)    \n\t" : :"r"( a ) );
    c_tmp0.v = _mm256_mul_pd( a47.v, b0.v );
    b0.v     = _mm256_load_pd( b + 4 );                   // Preload b0
    c_tmp1.v = _mm256_mul_pd( a47.v, b0x5.v );   
    b0x5.v   = _mm256_permute_pd( b0.v, 0x5 );            // Permute b0 to get b0x5
    c47_0.v  = _mm256_add_pd( c47_0.v, c_tmp0.v );
    c47_1.v  = _mm256_add_pd( c47_1.v, c_tmp1.v );

    c_tmp0.v = _mm256_mul_pd( a03.v, b0x3_0.v );
    c_tmp1.v = _mm256_mul_pd( a03.v, b0x3_1.v );
    a03.v    = _mm256_load_pd( a + 8 );                   // Preload a03
    c03_2.v  = _mm256_add_pd( c03_2.v, c_tmp0.v );
    c03_3.v  = _mm256_add_pd( c03_3.v, c_tmp1.v );
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( b_next ) );

    c_tmp0.v = _mm256_mul_pd( a47.v, b0x3_0.v );
    c_tmp1.v = _mm256_mul_pd( a47.v, b0x3_1.v );
    c47_2.v  = _mm256_add_pd( c47_2.v, c_tmp0.v );
    c47_3.v  = _mm256_add_pd( c47_3.v, c_tmp1.v );


    // Iteration #1
   

    a47.v    = _mm256_load_pd( a + 12 );
    c_tmp0.v = _mm256_mul_pd( a03.v, b0.v );              // The same as iter 0
    b0x3_0.v = _mm256_permute2f128_pd( b0.v, b0.v, 0x3 );
    c_tmp1.v = _mm256_mul_pd( a03.v, b0x5.v );   
    b0x3_1.v = _mm256_permute2f128_pd( b0x5.v, b0x5.v, 0x3 );
    c03_0.v  = _mm256_add_pd( c03_0.v, c_tmp0.v );
    c03_1.v  = _mm256_add_pd( c03_1.v, c_tmp1.v );

    // Prefetch a03, a47 for 8 iteration later
    __asm__ volatile( "prefetcht0 18 * 32(%0)    \n\t" : :"r"( a ) );
    c_tmp0.v = _mm256_mul_pd( a47.v, b0.v );
    b0.v     = _mm256_load_pd( b + 8 );                   // Preload b0 fot iter 2
    c_tmp1.v = _mm256_mul_pd( a47.v, b0x5.v );   
    b0x5.v   = _mm256_permute_pd( b0.v, 0x5 ); 
    c47_0.v  = _mm256_add_pd( c47_0.v, c_tmp0.v );
    c47_1.v  = _mm256_add_pd( c47_1.v, c_tmp1.v );

    c_tmp0.v = _mm256_mul_pd( a03.v, b0x3_0.v );
    c_tmp1.v = _mm256_mul_pd( a03.v, b0x3_1.v );
    a03.v    = _mm256_load_pd( a + 16 );                  // Preload a03
    c03_2.v  = _mm256_add_pd( c03_2.v, c_tmp0.v );
    c03_3.v  = _mm256_add_pd( c03_3.v, c_tmp1.v );

    c_tmp0.v = _mm256_mul_pd( a47.v, b0x3_0.v );
    c_tmp1.v = _mm256_mul_pd( a47.v, b0x3_1.v );
    c47_2.v  = _mm256_add_pd( c47_2.v, c_tmp0.v );
    c47_3.v  = _mm256_add_pd( c47_3.v, c_tmp1.v );


    // Iteration #2 ( increase b pointer here )


    a47.v    = _mm256_load_pd( a + 20 );
    c_tmp0.v = _mm256_mul_pd( a03.v, b0.v );              // The same as iter 0
    b0x3_0.v = _mm256_permute2f128_pd( b0.v, b0.v, 0x3 );
    c_tmp1.v = _mm256_mul_pd( a03.v, b0x5.v );   
    b0x3_1.v = _mm256_permute2f128_pd( b0x5.v, b0x5.v, 0x3 );
    c03_0.v  = _mm256_add_pd( c03_0.v, c_tmp0.v );
    c03_1.v  = _mm256_add_pd( c03_1.v, c_tmp1.v );

    // Prefetch a03, a47 for 8 iteration later
    __asm__ volatile( "prefetcht0 20 * 32(%0)    \n\t" : :"r"( a ) );
    c_tmp0.v = _mm256_mul_pd( a47.v, b0.v );
    b0.v     = _mm256_load_pd( b + 12 );                  // Preload b0 fot iter 3
    b += 4 * DKS_NR;                                      // Increase b by 4 * nr
    c_tmp1.v = _mm256_mul_pd( a47.v, b0x5.v );   
    b0x5.v   = _mm256_permute_pd( b0.v, 0x5 ); 
    c47_0.v  = _mm256_add_pd( c47_0.v, c_tmp0.v );
    c47_1.v  = _mm256_add_pd( c47_1.v, c_tmp1.v );

    c_tmp0.v = _mm256_mul_pd( a03.v, b0x3_0.v );
    c_tmp1.v = _mm256_mul_pd( a03.v, b0x3_1.v );
    a03.v    = _mm256_load_pd( a + 24 );                  // Preload a03
    c03_2.v  = _mm256_add_pd( c03_2.v, c_tmp0.v );
    c03_3.v  = _mm256_add_pd( c03_3.v, c_tmp1.v );
    __asm__ volatile( "prefetcht0 2 * 32(%0)    \n\t" : :"r"( b_next ) );

    c_tmp0.v = _mm256_mul_pd( a47.v, b0x3_0.v );
    c_tmp1.v = _mm256_mul_pd( a47.v, b0x3_1.v );
    c47_2.v  = _mm256_add_pd( c47_2.v, c_tmp0.v );
    c47_3.v  = _mm256_add_pd( c47_3.v, c_tmp1.v );


    // Iteration #3 ( increase a pointer here )


    a47.v    = _mm256_load_pd( a + 28 );
    a += 32;                                              // Increase a by 4 * mr
    c_tmp0.v = _mm256_mul_pd( a03.v, b0.v );              // The same as iter 0
    b0x3_0.v = _mm256_permute2f128_pd( b0.v, b0.v, 0x3 );
    c_tmp1.v = _mm256_mul_pd( a03.v, b0x5.v );   
    b0x3_1.v = _mm256_permute2f128_pd( b0x5.v, b0x5.v, 0x3 );
    c03_0.v  = _mm256_add_pd( c03_0.v, c_tmp0.v );
    c03_1.v  = _mm256_add_pd( c03_1.v, c_tmp1.v );

    // Prefetch a03, a47 for 8 iteration later
    __asm__ volatile( "prefetcht0 14 * 32(%0)    \n\t" : :"r"( a ) );
    c_tmp0.v = _mm256_mul_pd( a47.v, b0.v );
    b0.v     = _mm256_load_pd( b );                       // Preload b0 fot iter 0
    c_tmp1.v = _mm256_mul_pd( a47.v, b0x5.v );   
    b0x5.v   = _mm256_permute_pd( b0.v, 0x5 ); 
    c47_0.v  = _mm256_add_pd( c47_0.v, c_tmp0.v );
    c47_1.v  = _mm256_add_pd( c47_1.v, c_tmp1.v );

    c_tmp0.v = _mm256_mul_pd( a03.v, b0x3_0.v );
    c_tmp1.v = _mm256_mul_pd( a03.v, b0x3_1.v );
    a03.v    = _mm256_load_pd( a );                       // Preload a03
    c03_2.v  = _mm256_add_pd( c03_2.v, c_tmp0.v );
    c03_3.v  = _mm256_add_pd( c03_3.v, c_tmp1.v );

    c_tmp0.v = _mm256_mul_pd( a47.v, b0x3_0.v );
    c_tmp1.v = _mm256_mul_pd( a47.v, b0x3_1.v );
    c47_2.v  = _mm256_add_pd( c47_2.v, c_tmp0.v );
    c47_3.v  = _mm256_add_pd( c47_3.v, c_tmp1.v );
  }

 
  if ( aux->pc != 0 ) {
    // packed
    tmpc03_0.v = _mm256_load_pd( c      );
    c03_0.v    = _mm256_add_pd( tmpc03_0.v, c03_0.v );

    tmpc47_0.v = _mm256_load_pd( c + 4  );
    c47_0.v    = _mm256_add_pd( tmpc47_0.v, c47_0.v );

    tmpc03_1.v = _mm256_load_pd( c + 8  );
    c03_1.v    = _mm256_add_pd( tmpc03_1.v, c03_1.v );

    tmpc47_1.v = _mm256_load_pd( c + 12 );
    c47_1.v    = _mm256_add_pd( tmpc47_1.v, c47_1.v );

    tmpc03_2.v = _mm256_load_pd( c + 16 );
    c03_2.v    = _mm256_add_pd( tmpc03_2.v, c03_2.v );

    tmpc47_2.v = _mm256_load_pd( c + 20 );
    c47_2.v    = _mm256_add_pd( tmpc47_2.v, c47_2.v );

    tmpc03_3.v = _mm256_load_pd( c + 24 );
    c03_3.v    = _mm256_add_pd( tmpc03_3.v, c03_3.v );

    tmpc47_3.v = _mm256_load_pd( c + 28 );
    c47_3.v    = _mm256_add_pd( tmpc47_3.v, c47_3.v );
  }


  // packed
  _mm256_store_pd( c     , c03_0.v );
  _mm256_store_pd( c + 4 , c47_0.v );

  _mm256_store_pd( c + 8 , c03_1.v );
  _mm256_store_pd( c + 12, c47_1.v );

  _mm256_store_pd( c + 16, c03_2.v );
  _mm256_store_pd( c + 20, c47_2.v );

  _mm256_store_pd( c + 24, c03_3.v );
  _mm256_store_pd( c + 28, c47_3.v );

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
