#include <immintrin.h> // AVX
#include <ks.h>


void exp_int_d4(
    double *x
    )
{
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

  // Remez Order 11 polynomail approximation
  //const double w0     =  9.9999999999999999694541216787022234814339814028865e-1;
  //const double w1     =  1.0000000000000013347525109964212249781265243645457;
  //const double w2     =  4.9999999999990426011279542064313207349934058355357e-1;
  //const double w3     =  1.6666666666933781279020916199156875162816850273886e-1;
  //const double w4     =  4.1666666628388978913396218847247771982698350546174e-2;
  //const double w5     =  8.3333336552944126722390410619859929515740995889372e-3;
  //const double w6     =  1.3888871805082296012945081624687544823497126781709e-3;
  //const double w7     =  1.9841863599469418342286677256362193951266072398489e-4;
  //const double w8     =  2.4787899938611697691690479138150629377630767114546e-5;
  //const double w9     =  2.7764095757136528235740765949934667970688427190168e-6;
  //const double w10    =  2.5602485412126369546033948405199058329040797134573e-7;
  //const double w11    =  3.5347283721656121939634391175390704621351283546671e-8;

  // Remez Order 9 polynomail approximation
//  const double w0     =  9.9999999999998657717890998293462356769270934668652e-1;
//  const double w1     =  1.0000000000041078023971691258305486059867172736079;
//  const double w2     =  4.9999999979496223000111361187419539211772440139043e-1;
//  const double w3     =  1.6666667059968250851708016603646727895353772273675e-1;
//  const double w4     =  4.1666628655740875994884332519499013211594753124142e-2;
//  const double w5     =  8.3335428149736685441705398632467122758546893330069e-3;
//  const double w6     =  1.3881912931358424526285652289974115047170651985345e-3;
//  const double w7     =  1.9983735415194021112767942931416179152416729204150e-4;
//  const double w8     =  2.3068467290270483679711135625155862511780587976925e-5;
//  const double w9     =  3.8865682386514872192656192137071689334005518164704e-6;


  v4df_t c03_0;
  v4df_t a03_0;
  v4df_t p03_0;

  v4df_t y, l2e, tmp, p;
  v4li_t k03_0;
  v4li_t offset;
  v4li_t k1, k2;
  __m128d p1, p2;

  
  c03_0.v   = _mm256_load_pd( x );

  tmp.v     = _mm256_broadcast_sd( &maxlog );
  c03_0.v   = _mm256_min_pd( tmp.v, c03_0.v ); 
  tmp.v     = _mm256_broadcast_sd( &minlog );
  c03_0.v   = _mm256_max_pd( tmp.v, c03_0.v ); 

  // a = c / log2e
  // c = a * ln2 = k * ln2 + w, ( w in [ -ln2, ln2 ] )
  l2e.v         = _mm256_broadcast_sd( &log2e );
  a03_0.v       = _mm256_mul_pd( l2e.v, c03_0.v );

  // Check if a < 0 
  tmp.v         = _mm256_setzero_pd();
  p03_0.v       = _mm256_cmp_pd( a03_0.v, tmp.v, 1 );
  tmp.v         = _mm256_broadcast_sd( &one );
  p03_0.v       = _mm256_and_pd( tmp.v, p03_0.v );
  // If a < 0 ( w < 0 ), then a - 1 =  ( k - 1 ) + w / ln2 
  a03_0.v       = _mm256_sub_pd( a03_0.v, p03_0.v );
  // Compute floor( a ) by two conversions
  // if a < 0, p = k - 1
  // else    , p = k
  k03_0.v       = _mm256_cvttpd_epi32( a03_0.v );
  p03_0.v       = _mm256_cvtepi32_pd( k03_0.v );

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
  c03_0.v       = _mm256_sub_pd( c03_0.v, a03_0.v );
  tmp.v         = _mm256_broadcast_sd( &c2 );
  a03_0.v       = _mm256_mul_pd( tmp.v, p03_0.v );
  c03_0.v       = _mm256_sub_pd( c03_0.v, a03_0.v );



  // Compute e^x using polynomial approximation
  // a = w10 + w11 * x
  tmp.v         = _mm256_broadcast_sd( &w11 );
  //tmp.v         = _mm256_broadcast_sd( &w9 );
  a03_0.v       = _mm256_mul_pd( c03_0.v, tmp.v );
  tmp.v         = _mm256_broadcast_sd( &w10 );
  //tmp.v         = _mm256_broadcast_sd( &w8 );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );


  // a = w8 + ( w9 + ( w10 + w11 * x ) * x ) * x
  tmp.v         = _mm256_broadcast_sd( &w9 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );
  tmp.v         = _mm256_broadcast_sd( &w8 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );


  tmp.v         = _mm256_broadcast_sd( &w7 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );
  tmp.v         = _mm256_broadcast_sd( &w6 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );


  tmp.v         = _mm256_broadcast_sd( &w5 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );
  tmp.v         = _mm256_broadcast_sd( &w4 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );


  tmp.v         = _mm256_broadcast_sd( &w3 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );
  tmp.v         = _mm256_broadcast_sd( &w2 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );


  tmp.v         = _mm256_broadcast_sd( &w1 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );
  tmp.v         = _mm256_broadcast_sd( &w0 );
  a03_0.v       = _mm256_mul_pd( a03_0.v, c03_0.v );
  a03_0.v       = _mm256_add_pd( a03_0.v, tmp.v );


  offset.v      = _mm_setr_epi32( 1023, 1023, 0, 0 );

  //printf( "offset\n" );
  //printf( "%d, %d, %d, %d\n", offset.d[ 0 ], offset.d[ 1 ], offset.d[ 2 ], offset.d[ 3 ] );

  k1.v          = _mm_set_epi32( 0, 0, k03_0.d[ 1 ], k03_0.d[ 0 ]);
  k2.v          = _mm_set_epi32( 0, 0, k03_0.d[ 3 ], k03_0.d[ 2 ]);

  //printf( "k1\n" );
  //printf( "%d, %d, %d, %d\n", k1.d[ 0 ], k1.d[ 1 ], k1.d[ 2 ], k1.d[ 3 ] );

  k1.v          = _mm_add_epi32( k1.v, offset.v );
  k2.v          = _mm_add_epi32( k2.v, offset.v );

  //printf( "k1 after\n" );
  //printf( "%d, %d, %d, %d\n", k1.d[ 0 ], k1.d[ 1 ], k1.d[ 2 ], k1.d[ 3 ] );


  k1.v          = _mm_slli_epi32( k1.v, 20 );

  //printf( "k1 shift\n" );
  //printf( "%d, %d, %d, %d\n", k1.d[ 0 ], k1.d[ 1 ], k1.d[ 2 ], k1.d[ 3 ] );

  k2.v          = _mm_slli_epi32( k2.v, 20 );
  k1.v          = _mm_shuffle_epi32( k1.v, _MM_SHUFFLE( 1, 3, 0, 2 ) );

  //printf( "k1 shuffle\n" );
  //printf( "%d, %d, %d, %d\n", k1.d[ 0 ], k1.d[ 1 ], k1.d[ 2 ], k1.d[ 3 ] );

  k2.v          = _mm_shuffle_epi32( k2.v, _MM_SHUFFLE( 1, 3, 0, 2 ) );
  p1            = _mm_castsi128_pd( k1.v );
  p2            = _mm_castsi128_pd( k2.v );
  p03_0.v       = _mm256_set_m128d( p2, p1 );

  //printf( "%E, %E, %E, %E\n", p03_0.d[ 0 ], p03_0.d[ 1 ], p03_0.d[ 2 ], p03_0.d[ 3 ] );

  c03_0.v       = _mm256_mul_pd( a03_0.v, p03_0.v );

  _mm256_store_pd( (double*)x, c03_0.v );
}
