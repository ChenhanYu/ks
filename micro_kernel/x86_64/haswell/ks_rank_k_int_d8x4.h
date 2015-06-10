  int k_iter = k / 2;
  int k_left = k % 2;

  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( a ) );
  __asm__ volatile( "prefetcht2 0(%0)    \n\t" : :"r"( aux->b_next ) );


  c03_0.v = _mm256_setzero_pd();
  c03_1.v = _mm256_setzero_pd();
  c03_2.v = _mm256_setzero_pd();
  c03_3.v = _mm256_setzero_pd();
  c47_0.v = _mm256_setzero_pd();
  c47_1.v = _mm256_setzero_pd();
  c47_2.v = _mm256_setzero_pd();
  c47_3.v = _mm256_setzero_pd();


  // Load a03
  a03.v = _mm256_load_pd(      (double*)a         );
  // Load a47
  a47.v = _mm256_load_pd(      (double*)( a + 4 ) );
  // Load (b0,b1,b2,b3)
  b0.v  = _mm256_load_pd(      (double*)b         );



  for ( i = 0; i < k_iter; ++i ) {
    __asm__ volatile( "prefetcht0 192(%0)    \n\t" : :"r"(a) );

    // Preload A03
    A03.v = _mm256_load_pd(      (double*)( a + 8 ) );

    c03_0.v = _mm256_fmadd_pd( a03.v, b0.v,  c03_0.v );
    c47_0.v = _mm256_fmadd_pd( a47.v, b0.v,  c47_0.v );

    // Preload A47
    A47.v = _mm256_load_pd(      (double*)( a + 12 ) );

    // Shuffle b ( 1, 0, 3, 2 )
    b1.v  = _mm256_shuffle_pd( b0.v, b0.v, 0x5 );

    c03_1.v = _mm256_fmadd_pd( a03.v, b1.v, c03_1.v );
    c47_1.v = _mm256_fmadd_pd( a47.v, b1.v, c47_1.v );

    // Permute b ( 3, 2, 1, 0 )
    b2.v  = _mm256_permute2f128_pd( b1.v, b1.v, 0x1 );

    // Preload B0
    B0.v  = _mm256_load_pd(      (double*)( b + 4 ) );

    c03_2.v = _mm256_fmadd_pd( a03.v, b2.v, c03_2.v );
    c47_2.v = _mm256_fmadd_pd( a47.v, b2.v, c47_2.v );

    // Shuffle b ( 3, 2, 1, 0 )
    b3.v  = _mm256_shuffle_pd( b2.v, b2.v, 0x5 );

    c03_3.v = _mm256_fmadd_pd( a03.v, b3.v, c03_3.v );
    c47_3.v = _mm256_fmadd_pd( a47.v, b3.v, c47_3.v );


    // Iteration #1
    __asm__ volatile( "prefetcht0 512(%0)    \n\t" : :"r"(a) );

    // Preload a03 ( next iteration )
    a03.v = _mm256_load_pd(      (double*)( a + 16 ) );

    c03_0.v = _mm256_fmadd_pd( A03.v, B0.v, c03_0.v );

    b1.v  = _mm256_shuffle_pd( B0.v, B0.v, 0x5 );

    c47_0.v = _mm256_fmadd_pd( A47.v, B0.v, c47_0.v );
    c03_1.v = _mm256_fmadd_pd( A03.v, b1.v, c03_1.v );

    // Preload a47 ( next iteration )
    a47.v = _mm256_load_pd(      (double*)( a + 20 ) );

    // Permute b ( 3, 2, 1, 0 )
    b2.v  = _mm256_permute2f128_pd( b1.v, b1.v, 0x1 );

    c47_1.v = _mm256_fmadd_pd( A47.v, b1.v, c47_1.v );
    c03_2.v = _mm256_fmadd_pd( A03.v, b2.v, c03_2.v );

    // Shuffle b ( 3, 2, 1, 0 )
    b3.v  = _mm256_shuffle_pd( b2.v, b2.v, 0x5 );

    c47_2.v = _mm256_fmadd_pd( A47.v, b2.v, c47_2.v );

    // Load b0 ( next iteration )
    b0.v  = _mm256_load_pd(      (double*)( b + 8 ) );

    c03_3.v = _mm256_fmadd_pd( A03.v, b3.v, c03_3.v );
    c47_3.v = _mm256_fmadd_pd( A47.v, b3.v, c47_3.v );

    a += 16;
    b += 8;
  }


  for ( i = 0; i < k_left; ++i ) {
    a03.v = _mm256_load_pd(      (double*)a         );
    //printf( "a03 = %lf, %lf, %lf, %lf\n", a03.d[0], a03.d[1], a03.d[2], a03.d[3] );

    a47.v = _mm256_load_pd(      (double*)( a + 4 ) );
    //printf( "a47 = %lf, %lf, %lf, %lf\n", a47.d[0], a47.d[1], a47.d[2], a47.d[3] );

    b0.v  = _mm256_load_pd(      (double*)b         );
    //printf( "b0  = %lf, %lf, %lf, %lf\n", b0.d[0], b0.d[1], b0.d[2], b0.d[3] );

    c03_0.v = _mm256_fmadd_pd( a03.v, b0.v, c03_0.v );
    c47_0.v = _mm256_fmadd_pd( a03.v, b0.v, c47_0.v );

    // Shuffle b ( 1, 0, 3, 2 )
    b1.v  = _mm256_shuffle_pd( b0.v, b0.v, 0x5 );

    c03_1.v = _mm256_fmadd_pd( a03.v, b1.v, c03_1.v );
    c47_1.v = _mm256_fmadd_pd( a47.v, b1.v, c47_1.v );

    // Permute b ( 3, 2, 1, 0 )
    b2.v  = _mm256_permute2f128_pd( b1.v, b1.v, 0x1 );

    c03_2.v = _mm256_fmadd_pd( a03.v, b2.v, c03_2.v );
    c47_2.v = _mm256_fmadd_pd( a47.v, b2.v, c47_2.v );

    // Shuffle b ( 3, 2, 1, 0 )
    b3.v  = _mm256_shuffle_pd( b2.v, b2.v, 0x5 );

    c03_3.v = _mm256_fmadd_pd( a03.v, b3.v, c03_3.v );
    c47_3.v = _mm256_fmadd_pd( a47.v, b3.v, c47_3.v );

    a += 8;
    b += 4;
  }
 

  tmpc03_0.v = _mm256_blend_pd( c03_0.v, c03_1.v, 0x6 );
  tmpc03_1.v = _mm256_blend_pd( c03_1.v, c03_0.v, 0x6 );
  
  tmpc03_2.v = _mm256_blend_pd( c03_2.v, c03_3.v, 0x6 );
  tmpc03_3.v = _mm256_blend_pd( c03_3.v, c03_2.v, 0x6 );

  tmpc47_0.v = _mm256_blend_pd( c47_0.v, c47_1.v, 0x6 );
  tmpc47_1.v = _mm256_blend_pd( c47_1.v, c47_0.v, 0x6 );

  tmpc47_2.v = _mm256_blend_pd( c47_2.v, c47_3.v, 0x6 );
  tmpc47_3.v = _mm256_blend_pd( c47_3.v, c47_2.v, 0x6 );

  c03_0.v    = _mm256_permute2f128_pd( tmpc03_0.v, tmpc03_2.v, 0x30 );
  c03_3.v    = _mm256_permute2f128_pd( tmpc03_2.v, tmpc03_0.v, 0x30 );

  c03_1.v    = _mm256_permute2f128_pd( tmpc03_1.v, tmpc03_3.v, 0x30 );
  c03_2.v    = _mm256_permute2f128_pd( tmpc03_3.v, tmpc03_1.v, 0x30 );

  c47_0.v    = _mm256_permute2f128_pd( tmpc47_0.v, tmpc47_2.v, 0x30 );
  c47_3.v    = _mm256_permute2f128_pd( tmpc47_2.v, tmpc47_0.v, 0x30 );

  c47_1.v    = _mm256_permute2f128_pd( tmpc47_1.v, tmpc47_3.v, 0x30 );
  c47_2.v    = _mm256_permute2f128_pd( tmpc47_3.v, tmpc47_1.v, 0x30 );
