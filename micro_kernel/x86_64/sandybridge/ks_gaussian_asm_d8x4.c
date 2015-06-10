#include <immintrin.h> // AVX
#include <ks.h>


void ks_gaussian_asm_d8x4(
    int    k,
    double alpha,
    double *u,
    double *aa,
    double *a,
    double *bb,
    double *b,
    double *w,
    aux_t  *aux
    )
{
  int    i;
  double neg2 = -2.0;
  double dzero = 0.0;
  double *c;
  v4df_t c03_0, c03_1, c03_2, c03_3;
  v4df_t c47_0, c47_1, c47_2, c47_3;
  v4df_t u03, u47;
  v4df_t aa_tmp, bb_tmp;
  v4df_t w_tmp;

  posix_memalign( &c, (size_t)DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * DKS_MR * DKS_NR );

  //printf( "pc = %d\n", aux->pc );
  aux->pc = 0;

  // Call the rank-k update kernel
  ks_rank_k_asm_d8x4(
      k,
      a,
      b,
      c,
      0,
      aux
      );

  // Load c from memory, hopefully it's still in L1.
  c03_0.v = _mm256_load_pd( c      );
  c47_0.v = _mm256_load_pd( c + 4  );

  c03_1.v = _mm256_load_pd( c + 8  );
  c47_1.v = _mm256_load_pd( c + 12 );

  c03_2.v = _mm256_load_pd( c + 16 );
  c47_2.v = _mm256_load_pd( c + 20 );

  c03_3.v = _mm256_load_pd( c + 24 );
  c47_3.v = _mm256_load_pd( c + 28 );















  // Inline vdExp()
  const double log2e  =  1.4426950408889634073599;
  const double maxlog =  7.09782712893383996843e2; // log( 2**1024 )
  const double minlog = -7.08396418532264106224e2; // log( 2**-1024 )
  const double one    =  1.0;
  const double c1     =  6.93145751953125E-1;
  const double c2     =  1.42860682030941723212E-6;

  // Original Remez Order 11 coefficients
  const double w11    =  3.5524625185478232665958141148891055719216674475023e-8;
  const double w10    =  2.5535368519306500343384723775435166753084614063349e-7;
  const double w9     =  2.77750562801295315877005242757916081614772210463065e-6;
  const double w8     =  2.47868893393199945541176652007657202642495832996107e-5;
  const double w7     =  1.98419213985637881240770890090795533564573406893163e-4;
  const double w6     =  1.3888869684178659239014256260881685824525255547326e-3;
  const double w5     =  8.3333337052009872221152811550156335074160546333973e-3;
  const double w4     =  4.1666666621080810610346717440523105184720007971655e-2;
  const double w3     =  0.166666666669960803484477734308515404418108830469798;
  const double w2     =  0.499999999999877094481580370323249951329122224389189;
  const double w1     =  1.0000000000000017952745258419615282194236357388884;
  const double w0     =  0.99999999999999999566016490920259318691496540598896;


  v4df_t a03_0, a03_1, a03_2, a03_3;
  v4df_t a47_0, a47_1, a47_2, a47_3;
  v4df_t p03_0, p03_1, p03_2, p03_3;
  v4df_t p47_0, p47_1, p47_2, p47_3;
  v4df_t y, l2e, tmp, p;
  v4li_t k03_0, k03_1, k03_2, k03_3;
  v4li_t k47_0, k47_1, k47_2, k47_3;
  v4li_t offset;
  v4li_t k1, k2;
  __m128d p1, p2;



  // Prefetch aa and bb
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( aa ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( bb ) );



  //printf( "rank-k\n" );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[0], c03_1.d[0], c03_2.d[0], c03_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[1], c03_1.d[1], c03_2.d[1], c03_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[2], c03_1.d[2], c03_2.d[2], c03_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[3], c03_1.d[3], c03_2.d[3], c03_3.d[3] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[0], c47_1.d[0], c47_2.d[0], c47_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[1], c47_1.d[1], c47_2.d[1], c47_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[2], c47_1.d[2], c47_2.d[2], c47_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[3], c47_1.d[3], c47_2.d[3], c47_3.d[3] );



  
  aa_tmp.v = _mm256_broadcast_sd( &neg2 );

  c03_0.v  = _mm256_mul_pd( aa_tmp.v, c03_0.v );
  c03_1.v  = _mm256_mul_pd( aa_tmp.v, c03_1.v );
  c03_2.v  = _mm256_mul_pd( aa_tmp.v, c03_2.v );
  c03_3.v  = _mm256_mul_pd( aa_tmp.v, c03_3.v );
  c47_0.v  = _mm256_mul_pd( aa_tmp.v, c47_0.v );
  c47_1.v  = _mm256_mul_pd( aa_tmp.v, c47_1.v );
  c47_2.v  = _mm256_mul_pd( aa_tmp.v, c47_2.v );
  c47_3.v  = _mm256_mul_pd( aa_tmp.v, c47_3.v );


  //printf( "scale -2 \n" );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[0], c03_1.d[0], c03_2.d[0], c03_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[1], c03_1.d[1], c03_2.d[1], c03_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[2], c03_1.d[2], c03_2.d[2], c03_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[3], c03_1.d[3], c03_2.d[3], c03_3.d[3] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[0], c47_1.d[0], c47_2.d[0], c47_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[1], c47_1.d[1], c47_2.d[1], c47_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[2], c47_1.d[2], c47_2.d[2], c47_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[3], c47_1.d[3], c47_2.d[3], c47_3.d[3] );


  aa_tmp.v = _mm256_load_pd( (double*)aa );
  c03_0.v  = _mm256_add_pd( aa_tmp.v, c03_0.v );
  c03_1.v  = _mm256_add_pd( aa_tmp.v, c03_1.v );
  c03_2.v  = _mm256_add_pd( aa_tmp.v, c03_2.v );
  c03_3.v  = _mm256_add_pd( aa_tmp.v, c03_3.v );

  //printf( "aa03 = %lf, %lf, %lf, %lf\n", aa_tmp.d[0], aa_tmp.d[1], aa_tmp.d[2], aa_tmp.d[3] );
  //printf( "bb03 = %lf, %lf, %lf, %lf\n", bb[ 0 ], bb[ 1 ], bb[ 2 ], bb[ 3 ] );

  aa_tmp.v = _mm256_load_pd( (double*)( aa + 4 ) );
  c47_0.v  = _mm256_add_pd( aa_tmp.v, c47_0.v );
  c47_1.v  = _mm256_add_pd( aa_tmp.v, c47_1.v );
  c47_2.v  = _mm256_add_pd( aa_tmp.v, c47_2.v );
  c47_3.v  = _mm256_add_pd( aa_tmp.v, c47_3.v );
  

  //printf( "add a^2\n" );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[0], c03_1.d[0], c03_2.d[0], c03_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[1], c03_1.d[1], c03_2.d[1], c03_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[2], c03_1.d[2], c03_2.d[2], c03_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[3], c03_1.d[3], c03_2.d[3], c03_3.d[3] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[0], c47_1.d[0], c47_2.d[0], c47_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[1], c47_1.d[1], c47_2.d[1], c47_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[2], c47_1.d[2], c47_2.d[2], c47_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[3], c47_1.d[3], c47_2.d[3], c47_3.d[3] );




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
  tmp.v    = _mm256_broadcast_sd( &dzero );
  c03_0.v  = _mm256_max_pd( tmp.v, c03_0.v );
  c03_1.v  = _mm256_max_pd( tmp.v, c03_1.v );
  c03_2.v  = _mm256_max_pd( tmp.v, c03_2.v );
  c03_3.v  = _mm256_max_pd( tmp.v, c03_3.v );
  c47_0.v  = _mm256_max_pd( tmp.v, c47_0.v );
  c47_1.v  = _mm256_max_pd( tmp.v, c47_1.v );
  c47_2.v  = _mm256_max_pd( tmp.v, c47_2.v );
  c47_3.v  = _mm256_max_pd( tmp.v, c47_3.v );



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



  //printf( "square distance\n" );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[0], c03_1.d[0], c03_2.d[0], c03_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[1], c03_1.d[1], c03_2.d[1], c03_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[2], c03_1.d[2], c03_2.d[2], c03_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[3], c03_1.d[3], c03_2.d[3], c03_3.d[3] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[0], c47_1.d[0], c47_2.d[0], c47_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[1], c47_1.d[1], c47_2.d[1], c47_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[2], c47_1.d[2], c47_2.d[2], c47_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[3], c47_1.d[3], c47_2.d[3], c47_3.d[3] );

  //for ( i = 0; i < 4; i++ ) {
  //  if ( c03_0.d[ i ] != c03_0.d[ i ] ) {
  //    printf( "error Nan: c03_0[ %d ]\n", i );
  //  }
  //  if ( c03_1.d[ i ] != c03_1.d[ i ] ) {
  //    printf( "error Nan: c03_1[ %d ]\n", i );
  //  }
  //  if ( c03_2.d[ i ] != c03_2.d[ i ] ) {
  //    printf( "error Nan: c03_2[ %d ]\n", i );
  //  }
  //  if ( c03_3.d[ i ] != c03_3.d[ i ] ) {
  //    printf( "error Nan: c03_3[ %d ]\n", i );
  //  }
  //  if ( c47_0.d[ i ] != c47_0.d[ i ] ) {
  //    printf( "error Nan: c47_0[ %d ]\n", i );
  //  }
  //  if ( c47_1.d[ i ] != c47_1.d[ i ] ) {
  //    printf( "error Nan: c47_1[ %d ]\n", i );
  //  }
  //  if ( c47_2.d[ i ] != c47_2.d[ i ] ) {
  //    printf( "error Nan: c47_2[ %d ]\n", i );
  //  }
  //  if ( c47_3.d[ i ] != c47_3.d[ i ] ) {
  //    printf( "error Nan: c47_3[ %d ]\n", i );
  //  }
  //}



  tmp.v     = _mm256_broadcast_sd( &maxlog );
  c03_0.v   = _mm256_min_pd( tmp.v, c03_0.v ); 
  c03_1.v   = _mm256_min_pd( tmp.v, c03_1.v ); 
  c03_2.v   = _mm256_min_pd( tmp.v, c03_2.v ); 
  c03_3.v   = _mm256_min_pd( tmp.v, c03_3.v ); 
  c47_0.v   = _mm256_min_pd( tmp.v, c47_0.v ); 
  c47_1.v   = _mm256_min_pd( tmp.v, c47_1.v ); 
  c47_2.v   = _mm256_min_pd( tmp.v, c47_2.v ); 
  c47_3.v   = _mm256_min_pd( tmp.v, c47_3.v ); 
  tmp.v     = _mm256_broadcast_sd( &minlog );
  c03_0.v   = _mm256_max_pd( tmp.v, c03_0.v ); 
  c03_1.v   = _mm256_max_pd( tmp.v, c03_1.v ); 
  c03_2.v   = _mm256_max_pd( tmp.v, c03_2.v ); 
  c03_3.v   = _mm256_max_pd( tmp.v, c03_3.v ); 
  c47_0.v   = _mm256_max_pd( tmp.v, c47_0.v ); 
  c47_1.v   = _mm256_max_pd( tmp.v, c47_1.v ); 
  c47_2.v   = _mm256_max_pd( tmp.v, c47_2.v ); 
  c47_3.v   = _mm256_max_pd( tmp.v, c47_3.v ); 

  // a = c / log2e
  // c = a * ln2 = k * ln2 + w, ( w in [ -ln2, ln2 ] )
  l2e.v         = _mm256_broadcast_sd( &log2e );
  a03_0.v       = _mm256_mul_pd( l2e.v, c03_0.v );
  a03_1.v       = _mm256_mul_pd( l2e.v, c03_1.v );
  a03_2.v       = _mm256_mul_pd( l2e.v, c03_2.v );
  a03_3.v       = _mm256_mul_pd( l2e.v, c03_3.v );
  a47_0.v       = _mm256_mul_pd( l2e.v, c47_0.v );
  a47_1.v       = _mm256_mul_pd( l2e.v, c47_1.v );
  a47_2.v       = _mm256_mul_pd( l2e.v, c47_2.v );
  a47_3.v       = _mm256_mul_pd( l2e.v, c47_3.v );

  // Check if a < 0 
  tmp.v         = _mm256_setzero_pd();
  p03_0.v       = _mm256_cmp_pd( a03_0.v, tmp.v, 1 );
  p03_1.v       = _mm256_cmp_pd( a03_1.v, tmp.v, 1 );
  p03_2.v       = _mm256_cmp_pd( a03_2.v, tmp.v, 1 );
  p03_3.v       = _mm256_cmp_pd( a03_3.v, tmp.v, 1 );
  p47_0.v       = _mm256_cmp_pd( a47_0.v, tmp.v, 1 );
  p47_1.v       = _mm256_cmp_pd( a47_1.v, tmp.v, 1 );
  p47_2.v       = _mm256_cmp_pd( a47_2.v, tmp.v, 1 );
  p47_3.v       = _mm256_cmp_pd( a47_3.v, tmp.v, 1 );
  tmp.v         = _mm256_broadcast_sd( &one );
  p03_0.v       = _mm256_and_pd( tmp.v, p03_0.v );
  p03_1.v       = _mm256_and_pd( tmp.v, p03_1.v );
  p03_2.v       = _mm256_and_pd( tmp.v, p03_2.v );
  p03_3.v       = _mm256_and_pd( tmp.v, p03_3.v );
  p47_0.v       = _mm256_and_pd( tmp.v, p47_0.v );
  p47_1.v       = _mm256_and_pd( tmp.v, p47_1.v );
  p47_2.v       = _mm256_and_pd( tmp.v, p47_2.v );
  p47_3.v       = _mm256_and_pd( tmp.v, p47_3.v );
  // If a < 0 ( w < 0 ), then a - 1 =  ( k - 1 ) + w / ln2 
  a03_0.v       = _mm256_sub_pd( a03_0.v, p03_0.v );
  a03_1.v       = _mm256_sub_pd( a03_1.v, p03_1.v );
  a03_2.v       = _mm256_sub_pd( a03_2.v, p03_2.v );
  a03_3.v       = _mm256_sub_pd( a03_3.v, p03_3.v );
  a47_0.v       = _mm256_sub_pd( a47_0.v, p47_0.v );
  a47_1.v       = _mm256_sub_pd( a47_1.v, p47_1.v );
  a47_2.v       = _mm256_sub_pd( a47_2.v, p47_2.v );
  a47_3.v       = _mm256_sub_pd( a47_3.v, p47_3.v );
  // Compute floor( a ) by two conversions
  // if a < 0, p = k - 1
  // else    , p = k
  k03_0.v       = _mm256_cvttpd_epi32( a03_0.v );
  k03_1.v       = _mm256_cvttpd_epi32( a03_1.v );
  k03_2.v       = _mm256_cvttpd_epi32( a03_2.v );
  k03_3.v       = _mm256_cvttpd_epi32( a03_3.v );
  k47_0.v       = _mm256_cvttpd_epi32( a47_0.v );
  k47_1.v       = _mm256_cvttpd_epi32( a47_1.v );
  k47_2.v       = _mm256_cvttpd_epi32( a47_2.v );
  k47_3.v       = _mm256_cvttpd_epi32( a47_3.v );
  p03_0.v       = _mm256_cvtepi32_pd( k03_0.v );
  p03_1.v       = _mm256_cvtepi32_pd( k03_1.v );
  p03_2.v       = _mm256_cvtepi32_pd( k03_2.v );
  p03_3.v       = _mm256_cvtepi32_pd( k03_3.v );
  p47_0.v       = _mm256_cvtepi32_pd( k47_0.v );
  p47_1.v       = _mm256_cvtepi32_pd( k47_1.v );
  p47_2.v       = _mm256_cvtepi32_pd( k47_2.v );
  p47_3.v       = _mm256_cvtepi32_pd( k47_3.v );

  // ---------------------
  // x -= p * ln2
  // ---------------------
  // c1 = ln2
  // if a < 0, a = ( k - 1 ) * ln2
  // else    , a = k * ln2
  // if a < 0, x -= ( k - 1 ) * ln2
  // else    , x -= k * ln2
  //
  tmp.v         = _mm256_broadcast_sd( &c1 );
  a03_0.v       = _mm256_mul_pd( tmp.v, p03_0.v );
  a03_1.v       = _mm256_mul_pd( tmp.v, p03_1.v );
  a03_2.v       = _mm256_mul_pd( tmp.v, p03_2.v );
  a03_3.v       = _mm256_mul_pd( tmp.v, p03_3.v );
  a47_0.v       = _mm256_mul_pd( tmp.v, p47_0.v );
  a47_1.v       = _mm256_mul_pd( tmp.v, p47_1.v );
  a47_2.v       = _mm256_mul_pd( tmp.v, p47_2.v );
  a47_3.v       = _mm256_mul_pd( tmp.v, p47_3.v );
  c03_0.v       = _mm256_sub_pd( c03_0.v, a03_0.v );
  c03_1.v       = _mm256_sub_pd( c03_1.v, a03_1.v );
  c03_2.v       = _mm256_sub_pd( c03_2.v, a03_2.v );
  c03_3.v       = _mm256_sub_pd( c03_3.v, a03_3.v );
  c47_0.v       = _mm256_sub_pd( c47_0.v, a47_0.v );
  c47_1.v       = _mm256_sub_pd( c47_1.v, a47_1.v );
  c47_2.v       = _mm256_sub_pd( c47_2.v, a47_2.v );
  c47_3.v       = _mm256_sub_pd( c47_3.v, a47_3.v );
  tmp.v         = _mm256_broadcast_sd( &c2 );
  a03_0.v       = _mm256_mul_pd( tmp.v, p03_0.v );
  a03_1.v       = _mm256_mul_pd( tmp.v, p03_1.v );
  a03_2.v       = _mm256_mul_pd( tmp.v, p03_2.v );
  a03_3.v       = _mm256_mul_pd( tmp.v, p03_3.v );
  a47_0.v       = _mm256_mul_pd( tmp.v, p47_0.v );
  a47_1.v       = _mm256_mul_pd( tmp.v, p47_1.v );
  a47_2.v       = _mm256_mul_pd( tmp.v, p47_2.v );
  a47_3.v       = _mm256_mul_pd( tmp.v, p47_3.v );
  c03_0.v       = _mm256_sub_pd( c03_0.v, a03_0.v );
  c03_1.v       = _mm256_sub_pd( c03_1.v, a03_1.v );
  c03_2.v       = _mm256_sub_pd( c03_2.v, a03_2.v );
  c03_3.v       = _mm256_sub_pd( c03_3.v, a03_3.v );
  c47_0.v       = _mm256_sub_pd( c47_0.v, a47_0.v );
  c47_1.v       = _mm256_sub_pd( c47_1.v, a47_1.v );
  c47_2.v       = _mm256_sub_pd( c47_2.v, a47_2.v );
  c47_3.v       = _mm256_sub_pd( c47_3.v, a47_3.v );


  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[0], c03_1.d[0], c03_2.d[0], c03_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[1], c03_1.d[1], c03_2.d[1], c03_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[2], c03_1.d[2], c03_2.d[2], c03_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[3], c03_1.d[3], c03_2.d[3], c03_3.d[3] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[0], c47_1.d[0], c47_2.d[0], c47_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[1], c47_1.d[1], c47_2.d[1], c47_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[2], c47_1.d[2], c47_2.d[2], c47_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[3], c47_1.d[3], c47_2.d[3], c47_3.d[3] );


  // Prefetch u
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u ) );



  // Compute e^x using polynomial approximation
  // a = w10 + w11 * x
  tmp.v         = _mm256_broadcast_sd( &w11 );
  //tmp.v         = _mm256_broadcast_sd( &w9 );
  a03_0.v       = _mm256_mul_pd( c03_0.v, tmp.v );
  a03_1.v       = _mm256_mul_pd( c03_1.v, tmp.v );
  a03_2.v       = _mm256_mul_pd( c03_2.v, tmp.v );
  a03_3.v       = _mm256_mul_pd( c03_3.v, tmp.v );
  a47_0.v       = _mm256_mul_pd( c47_0.v, tmp.v );
  a47_1.v       = _mm256_mul_pd( c47_1.v, tmp.v );
  a47_2.v       = _mm256_mul_pd( c47_2.v, tmp.v );
  a47_3.v       = _mm256_mul_pd( c47_3.v, tmp.v );
  tmp.v         = _mm256_broadcast_sd( &w10 );
  //tmp.v         = _mm256_broadcast_sd( &w8 );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );
  a03_1.v       = _mm256_add_pd( a03_1.v, tmp.v );
  a03_2.v       = _mm256_add_pd( a03_2.v, tmp.v );
  a03_3.v       = _mm256_add_pd( a03_3.v, tmp.v );
  a47_0.v       = _mm256_add_pd( a47_0.v, tmp.v );
  a47_1.v       = _mm256_add_pd( a47_1.v, tmp.v );
  a47_2.v       = _mm256_add_pd( a47_2.v, tmp.v );
  a47_3.v       = _mm256_add_pd( a47_3.v, tmp.v );


  // a = w8 + ( w9 + ( w10 + w11 * x ) * x ) * x
  tmp.v         = _mm256_broadcast_sd( &w9 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_1.v       = _mm256_mul_pd( a03_1.v, c03_1.v );
  a03_2.v       = _mm256_mul_pd( a03_2.v, c03_2.v );
  a03_3.v       = _mm256_mul_pd( a03_3.v, c03_3.v );
  a47_0.v       = _mm256_mul_pd( a47_0.v, c47_0.v );
  a47_1.v       = _mm256_mul_pd( a47_1.v, c47_1.v );
  a47_2.v       = _mm256_mul_pd( a47_2.v, c47_2.v );
  a47_3.v       = _mm256_mul_pd( a47_3.v, c47_3.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );
  a03_1.v       = _mm256_add_pd( a03_1.v, tmp.v );
  a03_2.v       = _mm256_add_pd( a03_2.v, tmp.v );
  a03_3.v       = _mm256_add_pd( a03_3.v, tmp.v );
  a47_0.v       = _mm256_add_pd( a47_0.v, tmp.v );
  a47_1.v       = _mm256_add_pd( a47_1.v, tmp.v );
  a47_2.v       = _mm256_add_pd( a47_2.v, tmp.v );
  a47_3.v       = _mm256_add_pd( a47_3.v, tmp.v );
  tmp.v         = _mm256_broadcast_sd( &w8 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_1.v       = _mm256_mul_pd( a03_1.v, c03_1.v );
  a03_2.v       = _mm256_mul_pd( a03_2.v, c03_2.v );
  a03_3.v       = _mm256_mul_pd( a03_3.v, c03_3.v );
  a47_0.v       = _mm256_mul_pd( a47_0.v, c47_0.v );
  a47_1.v       = _mm256_mul_pd( a47_1.v, c47_1.v );
  a47_2.v       = _mm256_mul_pd( a47_2.v, c47_2.v );
  a47_3.v       = _mm256_mul_pd( a47_3.v, c47_3.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );
  a03_1.v       = _mm256_add_pd( a03_1.v, tmp.v );
  a03_2.v       = _mm256_add_pd( a03_2.v, tmp.v );
  a03_3.v       = _mm256_add_pd( a03_3.v, tmp.v );
  a47_0.v       = _mm256_add_pd( a47_0.v, tmp.v );
  a47_1.v       = _mm256_add_pd( a47_1.v, tmp.v );
  a47_2.v       = _mm256_add_pd( a47_2.v, tmp.v );
  a47_3.v       = _mm256_add_pd( a47_3.v, tmp.v );


  tmp.v         = _mm256_broadcast_sd( &w7 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_1.v       = _mm256_mul_pd( a03_1.v, c03_1.v );
  a03_2.v       = _mm256_mul_pd( a03_2.v, c03_2.v );
  a03_3.v       = _mm256_mul_pd( a03_3.v, c03_3.v );
  a47_0.v       = _mm256_mul_pd( a47_0.v, c47_0.v );
  a47_1.v       = _mm256_mul_pd( a47_1.v, c47_1.v );
  a47_2.v       = _mm256_mul_pd( a47_2.v, c47_2.v );
  a47_3.v       = _mm256_mul_pd( a47_3.v, c47_3.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );
  a03_1.v       = _mm256_add_pd( a03_1.v, tmp.v );
  a03_2.v       = _mm256_add_pd( a03_2.v, tmp.v );
  a03_3.v       = _mm256_add_pd( a03_3.v, tmp.v );
  a47_0.v       = _mm256_add_pd( a47_0.v, tmp.v );
  a47_1.v       = _mm256_add_pd( a47_1.v, tmp.v );
  a47_2.v       = _mm256_add_pd( a47_2.v, tmp.v );
  a47_3.v       = _mm256_add_pd( a47_3.v, tmp.v );
  tmp.v         = _mm256_broadcast_sd( &w6 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_1.v       = _mm256_mul_pd( a03_1.v, c03_1.v );
  a03_2.v       = _mm256_mul_pd( a03_2.v, c03_2.v );
  a03_3.v       = _mm256_mul_pd( a03_3.v, c03_3.v );
  a47_0.v       = _mm256_mul_pd( a47_0.v, c47_0.v );
  a47_1.v       = _mm256_mul_pd( a47_1.v, c47_1.v );
  a47_2.v       = _mm256_mul_pd( a47_2.v, c47_2.v );
  a47_3.v       = _mm256_mul_pd( a47_3.v, c47_3.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );
  a03_1.v       = _mm256_add_pd( a03_1.v, tmp.v );
  a03_2.v       = _mm256_add_pd( a03_2.v, tmp.v );
  a03_3.v       = _mm256_add_pd( a03_3.v, tmp.v );
  a47_0.v       = _mm256_add_pd( a47_0.v, tmp.v );
  a47_1.v       = _mm256_add_pd( a47_1.v, tmp.v );
  a47_2.v       = _mm256_add_pd( a47_2.v, tmp.v );
  a47_3.v       = _mm256_add_pd( a47_3.v, tmp.v );


  tmp.v         = _mm256_broadcast_sd( &w5 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_1.v       = _mm256_mul_pd( a03_1.v, c03_1.v );
  a03_2.v       = _mm256_mul_pd( a03_2.v, c03_2.v );
  a03_3.v       = _mm256_mul_pd( a03_3.v, c03_3.v );
  a47_0.v       = _mm256_mul_pd( a47_0.v, c47_0.v );
  a47_1.v       = _mm256_mul_pd( a47_1.v, c47_1.v );
  a47_2.v       = _mm256_mul_pd( a47_2.v, c47_2.v );
  a47_3.v       = _mm256_mul_pd( a47_3.v, c47_3.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );
  a03_1.v       = _mm256_add_pd( a03_1.v, tmp.v );
  a03_2.v       = _mm256_add_pd( a03_2.v, tmp.v );
  a03_3.v       = _mm256_add_pd( a03_3.v, tmp.v );
  a47_0.v       = _mm256_add_pd( a47_0.v, tmp.v );
  a47_1.v       = _mm256_add_pd( a47_1.v, tmp.v );
  a47_2.v       = _mm256_add_pd( a47_2.v, tmp.v );
  a47_3.v       = _mm256_add_pd( a47_3.v, tmp.v );
  tmp.v         = _mm256_broadcast_sd( &w4 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_1.v       = _mm256_mul_pd( a03_1.v, c03_1.v );
  a03_2.v       = _mm256_mul_pd( a03_2.v, c03_2.v );
  a03_3.v       = _mm256_mul_pd( a03_3.v, c03_3.v );
  a47_0.v       = _mm256_mul_pd( a47_0.v, c47_0.v );
  a47_1.v       = _mm256_mul_pd( a47_1.v, c47_1.v );
  a47_2.v       = _mm256_mul_pd( a47_2.v, c47_2.v );
  a47_3.v       = _mm256_mul_pd( a47_3.v, c47_3.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );
  a03_1.v       = _mm256_add_pd( a03_1.v, tmp.v );
  a03_2.v       = _mm256_add_pd( a03_2.v, tmp.v );
  a03_3.v       = _mm256_add_pd( a03_3.v, tmp.v );
  a47_0.v       = _mm256_add_pd( a47_0.v, tmp.v );
  a47_1.v       = _mm256_add_pd( a47_1.v, tmp.v );
  a47_2.v       = _mm256_add_pd( a47_2.v, tmp.v );
  a47_3.v       = _mm256_add_pd( a47_3.v, tmp.v );


  // Prefetch w
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( w ) );
  // Preload u03
  u03.v    = _mm256_load_pd( (double*)u );


  tmp.v         = _mm256_broadcast_sd( &w3 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_1.v       = _mm256_mul_pd( a03_1.v, c03_1.v );
  a03_2.v       = _mm256_mul_pd( a03_2.v, c03_2.v );
  a03_3.v       = _mm256_mul_pd( a03_3.v, c03_3.v );
  a47_0.v       = _mm256_mul_pd( a47_0.v, c47_0.v );
  a47_1.v       = _mm256_mul_pd( a47_1.v, c47_1.v );
  a47_2.v       = _mm256_mul_pd( a47_2.v, c47_2.v );
  a47_3.v       = _mm256_mul_pd( a47_3.v, c47_3.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );
  a03_1.v       = _mm256_add_pd( a03_1.v, tmp.v );
  a03_2.v       = _mm256_add_pd( a03_2.v, tmp.v );
  a03_3.v       = _mm256_add_pd( a03_3.v, tmp.v );
  a47_0.v       = _mm256_add_pd( a47_0.v, tmp.v );
  a47_1.v       = _mm256_add_pd( a47_1.v, tmp.v );
  a47_2.v       = _mm256_add_pd( a47_2.v, tmp.v );
  a47_3.v       = _mm256_add_pd( a47_3.v, tmp.v );
  tmp.v         = _mm256_broadcast_sd( &w2 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_1.v       = _mm256_mul_pd( a03_1.v, c03_1.v );
  a03_2.v       = _mm256_mul_pd( a03_2.v, c03_2.v );
  a03_3.v       = _mm256_mul_pd( a03_3.v, c03_3.v );
  a47_0.v       = _mm256_mul_pd( a47_0.v, c47_0.v );
  a47_1.v       = _mm256_mul_pd( a47_1.v, c47_1.v );
  a47_2.v       = _mm256_mul_pd( a47_2.v, c47_2.v );
  a47_3.v       = _mm256_mul_pd( a47_3.v, c47_3.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );
  a03_1.v       = _mm256_add_pd( a03_1.v, tmp.v );
  a03_2.v       = _mm256_add_pd( a03_2.v, tmp.v );
  a03_3.v       = _mm256_add_pd( a03_3.v, tmp.v );
  a47_0.v       = _mm256_add_pd( a47_0.v, tmp.v );
  a47_1.v       = _mm256_add_pd( a47_1.v, tmp.v );
  a47_2.v       = _mm256_add_pd( a47_2.v, tmp.v );
  a47_3.v       = _mm256_add_pd( a47_3.v, tmp.v );


  tmp.v         = _mm256_broadcast_sd( &w1 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_1.v       = _mm256_mul_pd( a03_1.v, c03_1.v );
  a03_2.v       = _mm256_mul_pd( a03_2.v, c03_2.v );
  a03_3.v       = _mm256_mul_pd( a03_3.v, c03_3.v );
  a47_0.v       = _mm256_mul_pd( a47_0.v, c47_0.v );
  a47_1.v       = _mm256_mul_pd( a47_1.v, c47_1.v );
  a47_2.v       = _mm256_mul_pd( a47_2.v, c47_2.v );
  a47_3.v       = _mm256_mul_pd( a47_3.v, c47_3.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );
  a03_1.v       = _mm256_add_pd( a03_1.v, tmp.v );
  a03_2.v       = _mm256_add_pd( a03_2.v, tmp.v );
  a03_3.v       = _mm256_add_pd( a03_3.v, tmp.v );
  a47_0.v       = _mm256_add_pd( a47_0.v, tmp.v );
  a47_1.v       = _mm256_add_pd( a47_1.v, tmp.v );
  a47_2.v       = _mm256_add_pd( a47_2.v, tmp.v );
  a47_3.v       = _mm256_add_pd( a47_3.v, tmp.v );
  tmp.v         = _mm256_broadcast_sd( &w0 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_1.v       = _mm256_mul_pd( a03_1.v, c03_1.v );
  a03_2.v       = _mm256_mul_pd( a03_2.v, c03_2.v );
  a03_3.v       = _mm256_mul_pd( a03_3.v, c03_3.v );
  a47_0.v       = _mm256_mul_pd( a47_0.v, c47_0.v );
  a47_1.v       = _mm256_mul_pd( a47_1.v, c47_1.v );
  a47_2.v       = _mm256_mul_pd( a47_2.v, c47_2.v );
  a47_3.v       = _mm256_mul_pd( a47_3.v, c47_3.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );
  a03_1.v       = _mm256_add_pd( a03_1.v, tmp.v );
  a03_2.v       = _mm256_add_pd( a03_2.v, tmp.v );
  a03_3.v       = _mm256_add_pd( a03_3.v, tmp.v );
  a47_0.v       = _mm256_add_pd( a47_0.v, tmp.v );
  a47_1.v       = _mm256_add_pd( a47_1.v, tmp.v );
  a47_2.v       = _mm256_add_pd( a47_2.v, tmp.v );
  a47_3.v       = _mm256_add_pd( a47_3.v, tmp.v );


  // Preload u47
  u47.v    = _mm256_load_pd( (double*)( u + 4 ) );


  offset.v      = _mm_setr_epi32( 1023, 1023, 0, 0 );
  k1.v          = _mm_set_epi32( 0, 0, k03_0.d[ 1 ], k03_0.d[ 0 ]);
  k2.v          = _mm_set_epi32( 0, 0, k03_0.d[ 3 ], k03_0.d[ 2 ]);
  k1.v          = _mm_add_epi32( k1.v, offset.v );
  k2.v          = _mm_add_epi32( k2.v, offset.v );
  k1.v          = _mm_slli_epi32( k1.v, 20 );
  k2.v          = _mm_slli_epi32( k2.v, 20 );
  k1.v          = _mm_shuffle_epi32( k1.v, _MM_SHUFFLE( 1, 3, 0, 2 ) );
  k2.v          = _mm_shuffle_epi32( k2.v, _MM_SHUFFLE( 1, 3, 0, 2 ) );
  p1            = _mm_castsi128_pd( k1.v );
  p2            = _mm_castsi128_pd( k2.v );
  p03_0.v       = _mm256_set_m128d( p2, p1 );
  k1.v          = _mm_set_epi32( 0, 0, k03_1.d[ 1 ], k03_1.d[ 0 ]);
  k2.v          = _mm_set_epi32( 0, 0, k03_1.d[ 3 ], k03_1.d[ 2 ]);
  k1.v          = _mm_add_epi32( k1.v, offset.v );
  k2.v          = _mm_add_epi32( k2.v, offset.v );
  k1.v          = _mm_slli_epi32( k1.v, 20 );
  k2.v          = _mm_slli_epi32( k2.v, 20 );
  k1.v          = _mm_shuffle_epi32( k1.v, _MM_SHUFFLE( 1, 3, 0, 2 ) );
  k2.v          = _mm_shuffle_epi32( k2.v, _MM_SHUFFLE( 1, 3, 0, 2 ) );
  p1            = _mm_castsi128_pd( k1.v );
  p2            = _mm_castsi128_pd( k2.v );
  p03_1.v       = _mm256_set_m128d( p2, p1 );
  k1.v          = _mm_set_epi32( 0, 0, k03_2.d[ 1 ], k03_2.d[ 0 ]);
  k2.v          = _mm_set_epi32( 0, 0, k03_2.d[ 3 ], k03_2.d[ 2 ]);
  k1.v          = _mm_add_epi32( k1.v, offset.v );
  k2.v          = _mm_add_epi32( k2.v, offset.v );
  k1.v          = _mm_slli_epi32( k1.v, 20 );
  k2.v          = _mm_slli_epi32( k2.v, 20 );
  k1.v          = _mm_shuffle_epi32( k1.v, _MM_SHUFFLE( 1, 3, 0, 2 ) );
  k2.v          = _mm_shuffle_epi32( k2.v, _MM_SHUFFLE( 1, 3, 0, 2 ) );
  p1            = _mm_castsi128_pd( k1.v );
  p2            = _mm_castsi128_pd( k2.v );
  p03_2.v       = _mm256_set_m128d( p2, p1 );
  k1.v          = _mm_set_epi32( 0, 0, k03_3.d[ 1 ], k03_3.d[ 0 ]);
  k2.v          = _mm_set_epi32( 0, 0, k03_3.d[ 3 ], k03_3.d[ 2 ]);
  k1.v          = _mm_add_epi32( k1.v, offset.v );
  k2.v          = _mm_add_epi32( k2.v, offset.v );
  k1.v          = _mm_slli_epi32( k1.v, 20 );
  k2.v          = _mm_slli_epi32( k2.v, 20 );
  k1.v          = _mm_shuffle_epi32( k1.v, _MM_SHUFFLE( 1, 3, 0, 2 ) );
  k2.v          = _mm_shuffle_epi32( k2.v, _MM_SHUFFLE( 1, 3, 0, 2 ) );
  p1            = _mm_castsi128_pd( k1.v );
  p2            = _mm_castsi128_pd( k2.v );
  p03_3.v       = _mm256_set_m128d( p2, p1 );
  k1.v          = _mm_set_epi32( 0, 0, k47_0.d[ 1 ], k47_0.d[ 0 ]);
  k2.v          = _mm_set_epi32( 0, 0, k47_0.d[ 3 ], k47_0.d[ 2 ]);
  k1.v          = _mm_add_epi32( k1.v, offset.v );
  k2.v          = _mm_add_epi32( k2.v, offset.v );
  k1.v          = _mm_slli_epi32( k1.v, 20 );
  k2.v          = _mm_slli_epi32( k2.v, 20 );
  k1.v          = _mm_shuffle_epi32( k1.v, _MM_SHUFFLE( 1, 3, 0, 2 ) );
  k2.v          = _mm_shuffle_epi32( k2.v, _MM_SHUFFLE( 1, 3, 0, 2 ) );
  p1            = _mm_castsi128_pd( k1.v );
  p2            = _mm_castsi128_pd( k2.v );
  p47_0.v       = _mm256_set_m128d( p2, p1 );
  k1.v          = _mm_set_epi32( 0, 0, k47_1.d[ 1 ], k47_1.d[ 0 ]);
  k2.v          = _mm_set_epi32( 0, 0, k47_1.d[ 3 ], k47_1.d[ 2 ]);
  k1.v          = _mm_add_epi32( k1.v, offset.v );
  k2.v          = _mm_add_epi32( k2.v, offset.v );
  k1.v          = _mm_slli_epi32( k1.v, 20 );
  k2.v          = _mm_slli_epi32( k2.v, 20 );
  k1.v          = _mm_shuffle_epi32( k1.v, _MM_SHUFFLE( 1, 3, 0, 2 ) );
  k2.v          = _mm_shuffle_epi32( k2.v, _MM_SHUFFLE( 1, 3, 0, 2 ) );
  p1            = _mm_castsi128_pd( k1.v );
  p2            = _mm_castsi128_pd( k2.v );
  p47_1.v       = _mm256_set_m128d( p2, p1 );
  k1.v          = _mm_set_epi32( 0, 0, k47_2.d[ 1 ], k47_2.d[ 0 ]);
  k2.v          = _mm_set_epi32( 0, 0, k47_2.d[ 3 ], k47_2.d[ 2 ]);
  k1.v          = _mm_add_epi32( k1.v, offset.v );
  k2.v          = _mm_add_epi32( k2.v, offset.v );
  k1.v          = _mm_slli_epi32( k1.v, 20 );
  k2.v          = _mm_slli_epi32( k2.v, 20 );
  k1.v          = _mm_shuffle_epi32( k1.v, _MM_SHUFFLE( 1, 3, 0, 2 ) );
  k2.v          = _mm_shuffle_epi32( k2.v, _MM_SHUFFLE( 1, 3, 0, 2 ) );
  p1            = _mm_castsi128_pd( k1.v );
  p2            = _mm_castsi128_pd( k2.v );
  p47_2.v       = _mm256_set_m128d( p2, p1 );
  k1.v          = _mm_set_epi32( 0, 0, k47_3.d[ 1 ], k47_3.d[ 0 ]);
  k2.v          = _mm_set_epi32( 0, 0, k47_3.d[ 3 ], k47_3.d[ 2 ]);
  k1.v          = _mm_add_epi32( k1.v, offset.v );
  k2.v          = _mm_add_epi32( k2.v, offset.v );
  k1.v          = _mm_slli_epi32( k1.v, 20 );
  k2.v          = _mm_slli_epi32( k2.v, 20 );
  k1.v          = _mm_shuffle_epi32( k1.v, _MM_SHUFFLE( 1, 3, 0, 2 ) );
  k2.v          = _mm_shuffle_epi32( k2.v, _MM_SHUFFLE( 1, 3, 0, 2 ) );
  p1            = _mm_castsi128_pd( k1.v );
  p2            = _mm_castsi128_pd( k2.v );
  p47_3.v       = _mm256_set_m128d( p2, p1 );
  
 
  //u03.v    = _mm256_load_pd( (double*)u );
  //u47.v    = _mm256_load_pd( (double*)( u + 4 ) );


  c03_0.v       = _mm256_mul_pd( a03_0.v, p03_0.v );
  c03_1.v       = _mm256_mul_pd( a03_1.v, p03_1.v );
  c03_2.v       = _mm256_mul_pd( a03_2.v, p03_2.v );
  c03_3.v       = _mm256_mul_pd( a03_3.v, p03_3.v );
  c47_0.v       = _mm256_mul_pd( a47_0.v, p47_0.v );
  c47_1.v       = _mm256_mul_pd( a47_1.v, p47_1.v );
  c47_2.v       = _mm256_mul_pd( a47_2.v, p47_2.v );
  c47_3.v       = _mm256_mul_pd( a47_3.v, p47_3.v );



  //for ( i = 0; i < 4; i++ ) {
  //  if ( c03_0.d[ i ] != c03_0.d[ i ] ) {
  //    printf( "error exp Nan: c03_0[ %d ]\n", i );
  //  }
  //  if ( c03_1.d[ i ] != c03_1.d[ i ] ) {
  //    printf( "error exp Nan: c03_1[ %d ]\n", i );
  //  }
  //  if ( c03_2.d[ i ] != c03_2.d[ i ] ) {
  //    printf( "error exp Nan: c03_2[ %d ]\n", i );
  //  }
  //  if ( c03_3.d[ i ] != c03_3.d[ i ] ) {
  //    printf( "error exp Nan: c03_3[ %d ]\n", i );
  //  }
  //  if ( c47_0.d[ i ] != c47_0.d[ i ] ) {
  //    printf( "error exp Nan: c47_0[ %d ]\n", i );
  //  }
  //  if ( c47_1.d[ i ] != c47_1.d[ i ] ) {
  //    printf( "error exp Nan: c47_1[ %d ]\n", i );
  //  }
  //  if ( c47_2.d[ i ] != c47_2.d[ i ] ) {
  //    printf( "error exp Nan: c47_2[ %d ]\n", i );
  //  }
  //  if ( c47_3.d[ i ] != c47_3.d[ i ] ) {
  //    printf( "error exp Nan: c47_3[ %d ]\n", i );
  //  }
  //}




  //printf( "exp\n" );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[0], c03_1.d[0], c03_2.d[0], c03_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[1], c03_1.d[1], c03_2.d[1], c03_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[2], c03_1.d[2], c03_2.d[2], c03_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[3], c03_1.d[3], c03_2.d[3], c03_3.d[3] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[0], c47_1.d[0], c47_2.d[0], c47_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[1], c47_1.d[1], c47_2.d[1], c47_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[2], c47_1.d[2], c47_2.d[2], c47_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[3], c47_1.d[3], c47_2.d[3], c47_3.d[3] );

  //printf( "w\n" );
  //printf( "%lf, %lf, %lf, %lf\n", w[0], w[3], w[3], w[3] );


  //u03.v    = _mm256_load_pd( (double*)u );
  //u47.v    = _mm256_load_pd( (double*)( u + 4 ) );

  w_tmp.v  = _mm256_broadcast_sd( (double*)w );
  c03_0.v  = _mm256_mul_pd( w_tmp.v, c03_0.v );
  c47_0.v  = _mm256_mul_pd( w_tmp.v, c47_0.v );
  u03.v    = _mm256_add_pd( u03.v, c03_0.v );
  u47.v    = _mm256_add_pd( u47.v, c47_0.v );
 

  //for ( i = 0; i < 4; i++ ) {
  //  if ( w_tmp.d[ i ] != w_tmp.d[ i ] ) {
  //    printf( "error w_tmp Nan: w_tmp[ %d ]\n", i );
  //  }
  //}


  w_tmp.v  = _mm256_broadcast_sd( (double*)( w + 1 ) );
  c03_1.v  = _mm256_mul_pd( w_tmp.v, c03_1.v );
  c47_1.v  = _mm256_mul_pd( w_tmp.v, c47_1.v );
  u03.v    = _mm256_add_pd( u03.v, c03_1.v );
  u47.v    = _mm256_add_pd( u47.v, c47_1.v );


  //for ( i = 0; i < 4; i++ ) {
  //  if ( w_tmp.d[ i ] != w_tmp.d[ i ] ) {
  //    printf( "error w_tmp Nan: w_tmp[ %d ]\n", i );
  //  }
  //}

  w_tmp.v  = _mm256_broadcast_sd( (double*)( w + 2 ) );
  c03_2.v  = _mm256_mul_pd( w_tmp.v, c03_2.v );
  c47_2.v  = _mm256_mul_pd( w_tmp.v, c47_2.v );
  u03.v    = _mm256_add_pd( u03.v, c03_2.v );
  u47.v    = _mm256_add_pd( u47.v, c47_2.v );


  //for ( i = 0; i < 4; i++ ) {
  //  if ( w_tmp.d[ i ] != w_tmp.d[ i ] ) {
  //    printf( "error w_tmp Nan: w_tmp[ %d ]\n", i );
  //  }
  //}

  w_tmp.v  = _mm256_broadcast_sd( (double*)( w + 3 ) );
  c03_3.v  = _mm256_mul_pd( w_tmp.v, c03_3.v );
  c47_3.v  = _mm256_mul_pd( w_tmp.v, c47_3.v );
  u03.v    = _mm256_add_pd( u03.v, c03_3.v );
  u47.v    = _mm256_add_pd( u47.v, c47_3.v );


  //for ( i = 0; i < 4; i++ ) {
  //  if ( w_tmp.d[ i ] != w_tmp.d[ i ] ) {
  //    printf( "error w_tmp Nan: w_tmp[ %d ]\n", i );
  //  }
  //}



  _mm256_store_pd( (double*)u, u03.v );
  _mm256_store_pd( (double*)( u + 4 ), u47.v );


  //for ( i = 0; i < 4; i++ ) {
  //  if ( c03_0.d[ i ] != c03_0.d[ i ] ) {
  //    printf( "error gemv Nan: c03_0[ %d ]\n", i );
  //    exit( 1 );
  //  }
  //  if ( c03_1.d[ i ] != c03_1.d[ i ] ) {
  //    printf( "error gemv Nan: c03_1[ %d ]\n", i );
  //    exit( 1 );
  //  }
  //  if ( c03_2.d[ i ] != c03_2.d[ i ] ) {
  //    printf( "error gemv Nan: c03_2[ %d ]\n", i );
  //    exit( 1 );
  //  }
  //  if ( c03_3.d[ i ] != c03_3.d[ i ] ) {
  //    printf( "error gemv Nan: c03_3[ %d ]\n", i );
  //    exit( 1 );
  //  }
  //  if ( c47_0.d[ i ] != c47_0.d[ i ] ) {
  //    printf( "error gemv Nan: c47_0[ %d ]\n", i );
  //    exit( 1 );
  //  }
  //  if ( c47_1.d[ i ] != c47_1.d[ i ] ) {
  //    printf( "error gemv Nan: c47_1[ %d ]\n", i );
  //    exit( 1 );
  //  }
  //  if ( c47_2.d[ i ] != c47_2.d[ i ] ) {
  //    printf( "error gemv Nan: c47_2[ %d ]\n", i );
  //    exit( 1 );
  //  }
  //  if ( c47_3.d[ i ] != c47_3.d[ i ] ) {
  //    printf( "error gemv Nan: c47_3[ %d ]\n", i );
  //    exit( 1 );
  //  }
  //}


  //for ( i = 0; i < 4; i ++ ) {
  //  if ( w[ i ] != w[ i ] ) {
  //    printf( "GSKS error w Nan: w03[ %d ]\n", i );
  //  }
  //}


  //for ( i = 0; i < 4; i++ ) {
  //  if ( u03.d[ i ] != u03.d[ i ] ) {
  //    printf( "GSKS error u Nan: u03[ %d ]\n", i );
  //  }
  //  if ( u47.d[ i ] != u47.d[ i ] ) {
  //    printf( "GSKS error u Nan: u47[ %d ]\n", i );
  //  }
  //}



  //printf( "%lf\n", u03.d[0] );
  //printf( "%lf\n", u03.d[1] );
  //printf( "%lf\n", u03.d[2] );
  //printf( "%lf\n", u03.d[3] );
  //printf( "%lf\n", u47.d[0] );
  //printf( "%lf\n", u47.d[1] );
  //printf( "%lf\n", u47.d[2] );
  //printf( "%lf\n", u47.d[3] );
  
  free( c );
}
