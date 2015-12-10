  int k_iter = k;

  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( a ) );
  __asm__ volatile( "prefetcht2 0(%0)    \n\t" : :"r"( aux->b_next ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( c ) );


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


  for ( i = 0; i < k_iter; i ++ ) {
    // Load a03
    a03.v = _mm256_load_pd(      (double*)a         );
    // Load a47
    a47.v = _mm256_load_pd(      (double*)( a + 4 ) );
    // Load a81
    a81.v = _mm256_load_pd(      (double*)( a + 8 ) );

    // Broadcast b0
    b0.v = _mm256_broadcast_sd( (double*)b );
    c03_0.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_0.v );
    c47_0.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_0.v );
    c81_0.v = _mm256_fmadd_pd( a81.v, b0.v,  c81_0.v );

    // Broadcast b1
    b1.v = _mm256_broadcast_sd( (double*)( b + 1 ) );
    c03_1.v = _mm256_fmadd_pd( a03.v, b1.v,  c03_1.v );
    c47_1.v = _mm256_fmadd_pd( a47.v, b1.v,  c47_1.v );
    c81_1.v = _mm256_fmadd_pd( a81.v, b1.v,  c81_1.v );

    // Broadcast b2
    b2.v = _mm256_broadcast_sd( (double*)( b + 2 ) );
    c03_2.v = _mm256_fmadd_pd( a03.v, b2.v,  c03_2.v );
    c47_2.v = _mm256_fmadd_pd( a47.v, b2.v,  c47_2.v );
    c81_2.v = _mm256_fmadd_pd( a81.v, b2.v,  c81_2.v );

    // Broadcast b3
    b3.v = _mm256_broadcast_sd( (double*)( b + 3 ) );
    c03_3.v = _mm256_fmadd_pd( a03.v, b3.v,  c03_3.v );
    c47_3.v = _mm256_fmadd_pd( a47.v, b3.v,  c47_3.v );
    c81_3.v = _mm256_fmadd_pd( a81.v, b3.v,  c81_3.v );

    a += 12;
    b += 4;
  }
