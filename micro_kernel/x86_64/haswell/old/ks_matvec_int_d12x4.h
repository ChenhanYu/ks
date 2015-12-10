// begin ks_matvec_int_d12x4

  for ( i = 0; i < rhs; i ++ ) {
    w_tmp.v  = _mm256_broadcast_sd( (double*)w );
    a03.v    = _mm256_mul_pd( w_tmp.v, c03_0.v );
    a47.v    = _mm256_mul_pd( w_tmp.v, c47_0.v );
    a81.v    = _mm256_mul_pd( w_tmp.v, c81_0.v );
    u03.v    = _mm256_add_pd( u03.v, a03.v );
    u47.v    = _mm256_add_pd( u47.v, a47.v );
    u81.v    = _mm256_add_pd( u81.v, a81.v );

    w_tmp.v  = _mm256_broadcast_sd( (double*)( w + 1 ) );
    a03.v    = _mm256_mul_pd( w_tmp.v, c03_1.v );
    a47.v    = _mm256_mul_pd( w_tmp.v, c47_1.v );
    a81.v    = _mm256_mul_pd( w_tmp.v, c81_1.v );
    u03.v    = _mm256_add_pd( u03.v, a03.v );
    u47.v    = _mm256_add_pd( u47.v, a47.v );
    u81.v    = _mm256_add_pd( u81.v, a81.v );

    w_tmp.v  = _mm256_broadcast_sd( (double*)( w + 2 ) );
    a03.v    = _mm256_mul_pd( w_tmp.v, c03_2.v );
    a47.v    = _mm256_mul_pd( w_tmp.v, c47_2.v );
    a81.v    = _mm256_mul_pd( w_tmp.v, c81_2.v );
    u03.v    = _mm256_add_pd( u03.v, a03.v );
    u47.v    = _mm256_add_pd( u47.v, a47.v );
    u81.v    = _mm256_add_pd( u81.v, a81.v );

    w_tmp.v  = _mm256_broadcast_sd( (double*)( w + 3 ) );
    a03.v    = _mm256_mul_pd( w_tmp.v, c03_3.v );
    a47.v    = _mm256_mul_pd( w_tmp.v, c47_3.v );
    a81.v    = _mm256_mul_pd( w_tmp.v, c81_3.v );
    u03.v    = _mm256_add_pd( u03.v, a03.v );
    u47.v    = _mm256_add_pd( u47.v, a47.v );
    u81.v    = _mm256_add_pd( u81.v, a81.v );

    _mm256_store_pd( u     , u03.v );
    _mm256_store_pd( u + 4 , u47.v );
    _mm256_store_pd( u + 8 , u81.v );

    u += 12;
    w += 4;
  }

// end ks_matvec_int_d12x4
