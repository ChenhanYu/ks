  int k_iter = k / 2;
  int k_left = k % 2;

  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( a ) );
  __asm__ volatile( "prefetcht2 0(%0)    \n\t" : :"r"( aux->b_next ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( c ) );

  // Load a03
  a03.v = _mm256_load_pd(      (double*)a         ); // ymm0
  // Load a47
  a47.v = _mm256_load_pd(      (double*)( a + 4 ) ); // ymm1
  // Load a81
  a81.v = _mm256_load_pd(      (double*)( a + 8 ) ); // ymm2


  c03_0.v = _mm256_setzero_pd();
  c03_1.v = _mm256_setzero_pd();
  c03_2.v = _mm256_setzero_pd();
  c03_3.v = _mm256_setzero_pd();

  c47_0.v = _mm256_setzero_pd();
  c47_1.v = _mm256_setzero_pd();
  c47_2.v = _mm256_setzero_pd();
  c47_3.v = _mm256_setzero_pd();

  c81_0.v = _mm256_setzero_pd();
  c81_1.v = _mm256_setzero_pd();
  c81_2.v = _mm256_setzero_pd();
  c81_3.v = _mm256_setzero_pd();

  for ( i = 0; i < k_iter; ++ i ) {
    __asm__ volatile( "prefetcht0 512(%0)    \n\t" : :"r"(a) );

    // Iteration #0
    b0.v    = _mm256_broadcast_sd( (double*)b );          // ymm3
    c03_0.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_0.v );
    c47_0.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_0.v );
    c81_0.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_0.v );

    b0.v    = _mm256_broadcast_sd( (double*)( b + 1 ) );  // ymm3
    c03_1.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_1.v );
    c47_1.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_1.v );
    c81_1.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_1.v );

    b0.v    = _mm256_broadcast_sd( (double*)( b + 2 ) );  // ymm3
    c03_2.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_2.v );
    c47_2.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_2.v );
    c81_2.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_2.v );

    b0.v    = _mm256_broadcast_sd( (double*)( b + 3 ) );  // ymm3
    c03_3.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_3.v );
    a03.v   = _mm256_load_pd(      (double*)( a + 12 ) ); // ymm0
    c47_3.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_3.v );
    a47.v   = _mm256_load_pd(      (double*)( a + 16 ) ); // ymm1
    c81_3.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_3.v );
    a81.v   = _mm256_load_pd(      (double*)( a + 20 ) ); // ymm2

    __asm__ volatile( "prefetcht0 1280(%0)    \n\t" : :"r"(a) );

    // Iteration #1
    b0.v    = _mm256_broadcast_sd( (double*)( b + 4 ) );  // ymm3
    c03_0.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_0.v );
    c47_0.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_0.v );
    c81_0.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_0.v );

    b0.v    = _mm256_broadcast_sd( (double*)( b + 5 ) );  // ymm3
    c03_1.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_1.v );
    c47_1.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_1.v );
    c81_1.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_1.v );

    b0.v    = _mm256_broadcast_sd( (double*)( b + 6 ) );  // ymm3
    c03_2.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_2.v );
    c47_2.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_2.v );
    c81_2.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_2.v );

    b0.v    = _mm256_broadcast_sd( (double*)( b + 7 ) );  // ymm3
    c03_3.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_3.v );
    a03.v   = _mm256_load_pd(      (double*)( a + 24 ) ); // ymm0
    c47_3.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_3.v );
    a47.v   = _mm256_load_pd(      (double*)( a + 28 ) ); // ymm1
    c81_3.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_3.v );
    a81.v   = _mm256_load_pd(      (double*)( a + 32 ) ); // ymm2

    //__asm__ volatile( "prefetcht0 704(%0)    \n\t" : :"r"(a) );

    //// Iteration #2
    //b0.v    = _mm256_broadcast_sd( (double*)( b + 8 ) );  // ymm3
    //c03_0.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_0.v );
    //c47_0.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_0.v );
    //c81_0.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_0.v );

    //b0.v    = _mm256_broadcast_sd( (double*)( b + 9 ) );  // ymm3
    //c03_1.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_1.v );
    //c47_1.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_1.v );
    //c81_1.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_1.v );

    //b0.v    = _mm256_broadcast_sd( (double*)( b + 10 ) ); // ymm3
    //c03_2.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_2.v );
    //c47_2.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_2.v );
    //c81_2.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_2.v );

    //b0.v    = _mm256_broadcast_sd( (double*)( b + 11 ) ); // ymm3
    //c03_3.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_3.v );
    //a03.v   = _mm256_load_pd(      (double*)( a + 36 ) ); // ymm0
    //c47_3.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_3.v );
    //a47.v   = _mm256_load_pd(      (double*)( a + 40 ) ); // ymm1
    //c81_3.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_3.v );
    //a81.v   = _mm256_load_pd(      (double*)( a + 44 ) ); // ymm2

    //// Iteration #3
    //b0.v    = _mm256_broadcast_sd( (double*)( b + 12 ) ); // ymm3
    //c03_0.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_0.v );
    //c47_0.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_0.v );
    //c81_0.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_0.v );

    //b0.v    = _mm256_broadcast_sd( (double*)( b + 13 ) ); // ymm3
    //c03_1.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_1.v );
    //c47_1.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_1.v );
    //c81_1.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_1.v );

    //b0.v    = _mm256_broadcast_sd( (double*)( b + 14 ) ); // ymm3
    //c03_2.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_2.v );
    //c47_2.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_2.v );
    //c81_2.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_2.v );

    //b0.v    = _mm256_broadcast_sd( (double*)( b + 15 ) ); // ymm3
    //c03_3.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_3.v );
    //a03.v   = _mm256_load_pd(      (double*)( a + 48 ) ); // ymm0
    //c47_3.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_3.v );
    //a47.v   = _mm256_load_pd(      (double*)( a + 52 ) ); // ymm1
    //c81_3.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_3.v );
    //a81.v   = _mm256_load_pd(      (double*)( a + 56 ) ); // ymm2

    //a += 48;
    //b += 16;

    a += 24;
    b += 8;
  }


  for ( i = 0; i < k_left; ++ i ) {
    __asm__ volatile( "prefetcht0 512(%0)    \n\t" : :"r"(a) );

    b0.v    = _mm256_broadcast_sd( (double*)b );          // ymm3
    c03_0.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_0.v );
    c47_0.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_0.v );
    c81_0.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_0.v );

    b0.v    = _mm256_broadcast_sd( (double*)( b + 1 ) );  // ymm3
    c03_1.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_1.v );
    c47_1.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_1.v );
    c81_1.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_1.v );

    b0.v    = _mm256_broadcast_sd( (double*)( b + 2 ) );  // ymm3
    c03_2.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_2.v );
    c47_2.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_2.v );
    c81_2.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_2.v );

    b0.v    = _mm256_broadcast_sd( (double*)( b + 3 ) );  // ymm3
    c03_3.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_3.v );
    a03.v   = _mm256_load_pd(      (double*)( a + 12 ) ); // ymm0
    c47_3.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_3.v );
    a47.v   = _mm256_load_pd(      (double*)( a + 16 ) ); // ymm1
    c81_3.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_3.v );
    a81.v   = _mm256_load_pd(      (double*)( a + 20 ) ); // ymm2

    a += 12;
    b += 4;
  }
