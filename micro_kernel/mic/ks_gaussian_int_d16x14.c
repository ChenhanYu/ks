#include <immintrin.h> // AVX
#include <ks.h>


void ks_gaussian_int_d16x14(
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
  int    i, k_iter, k_left;
  int    izero = 0;
  double neg2 = -2.0;
  double dzero = 0.0;
  double dmone = -1.0;
  int    perm[ 16 ] = { 8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7 };
  int    mask[ 16 ] = { 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 
	                    0, 0, 0, 0, 0, 0, 0, 0 };

  v8df_t c007_0, c007_1,  c007_2,  c007_3;
  v8df_t c007_4, c007_5,  c007_6,  c007_7;

  v8df_t c815_0,  c815_1,  c815_2,  c815_3;
  v8df_t c815_4,  c815_5,  c815_6,  c815_7;

  v8df_t a007, a815, b_tmp;
  v8df_t A007, A815;

  v8df_t l2e;

  v8df_t a007_0, a007_1,  a007_2,  a007_3;
  v8df_t a007_4, a007_5,  a007_6,  a007_7;

  v8df_t a815_0,  a815_1,  a815_2,  a815_3;
  v8df_t a815_4, a815_5,  a815_6,  a815_7; 
 
  v8df_t p007_0,  p007_1,  p007_2,  p007_3;
  v8df_t p007_4, p007_5,  p007_6,  p007_7;

  v8df_t p815_0, p815_1,  p815_2,  p815_3;
  v8df_t p815_4, p815_5,  p815_6,  p815_7;

  v16i_t k007_0,  k007_1,  k007_2,  k007_3;
  v16i_t k007_4, k007_5,  k007_6,  k007_7;

  v16i_t k815_0, k815_1,  k815_2,  k815_3;
  v16i_t k815_4, k815_5,  k815_6,  k815_7;
  
  
  v16i_t iperm;
 

  // Inline vdExp()
  const double log2e  =  1.4426950408889634073599;
  const double maxlog =  7.09782712893383996843e2; // log( 2**1024 )
  const double minlog = -7.08396418532264106224e2; // log( 2**-1024 )
  const double one    =  1.0;
  const double c1     = -6.93145751953125E-1;
  const double c2     = -1.42860682030941723212E-6;

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
  
  
 
  
  k_iter = k / 2;
  k_left = k % 2;


  // Prefetch a, a+8, b
  __asm__ volatile( "vprefetch0 0(%0)      \n\t" : :"r"(a) );
  __asm__ volatile( "vprefetch0 64(%0)     \n\t" : :"r"(a) );
  __asm__ volatile( "vprefetch0 0(%0)      \n\t" : :"r"(b) );




  // TODO: need to clean the c buffer.
  c007_0.v   = _mm512_extload_pd( &dzero, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  c007_1.v   = _mm512_mask_mov_pd( c007_0.v, 0x0, c007_0.v );
  c007_2.v   = _mm512_mask_mov_pd( c007_0.v, 0x0, c007_0.v );
  c007_3.v   = _mm512_mask_mov_pd( c007_0.v, 0x0, c007_0.v );
  c007_4.v   = _mm512_mask_mov_pd( c007_0.v, 0x0, c007_0.v );
  c007_5.v   = _mm512_mask_mov_pd( c007_0.v, 0x0, c007_0.v );
  c007_6.v   = _mm512_mask_mov_pd( c007_0.v, 0x0, c007_0.v );
  c007_7.v   = _mm512_mask_mov_pd( c007_0.v, 0x0, c007_0.v );
  
  c815_0.v   = _mm512_mask_mov_pd( c007_0.v, 0x0, c007_0.v );
  c815_1.v   = _mm512_mask_mov_pd( c007_0.v, 0x0, c007_0.v );
  c815_2.v   = _mm512_mask_mov_pd( c007_0.v, 0x0, c007_0.v );
  c815_3.v   = _mm512_mask_mov_pd( c007_0.v, 0x0, c007_0.v );
  c815_4.v   = _mm512_mask_mov_pd( c007_0.v, 0x0, c007_0.v );
  c815_5.v   = _mm512_mask_mov_pd( c007_0.v, 0x0, c007_0.v );
  c815_6.v   = _mm512_mask_mov_pd( c007_0.v, 0x0, c007_0.v );
  c815_7.v   = _mm512_mask_mov_pd( c007_0.v, 0x0, c007_0.v );




//  // Main loop
//  for ( i = 0; i < k_iter; i ++ ) { 
//
//
//	a007.v    = _mm512_load_pd( a );
//	a815.v    = _mm512_load_pd( a + 8 );
//
//	//printf( "a007\n" );
//	//printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", 
//	//	a007.d[ 0 ], a007.d[ 1 ], a007.d[ 2 ],  a007.d[ 3 ], 
//	//	a007.d[ 4 ], a007.d[ 5 ], a007.d[ 6 ],  a007.d[ 7 ] );
//	//printf( "a815\n" );
//	//printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", 
//	//	a815.d[ 0 ], a815.d[ 1 ], a815.d[ 2 ],  a815.d[ 3 ], 
//	//	a815.d[ 4 ], a815.d[ 5 ], a815.d[ 6 ],  a815.d[ 7 ] );
//
//	b_tmp.v   = _mm512_extload_pd( b, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
//
//    //printf( "b[ 0 ] = %lf\n", b[ 0 ] );
//
//	//printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", 
//	//	b_tmp.d[ 0 ], b_tmp.d[ 1 ], b_tmp.d[ 2 ],  b_tmp.d[ 3 ], 
//	//	b_tmp.d[ 4 ], b_tmp.d[ 5 ], b_tmp.d[ 6 ],  b_tmp.d[ 7 ] );
//
//
//
//	c007_0.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_0.v ); 
//	c815_0.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_0.v ); 
//	
//	b_tmp.v   = _mm512_extload_pd( b + 1, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
//	c007_1.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_1.v ); 
//	c815_1.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_1.v ); 
//
//    __asm__ volatile( "vprefetch0 128(%0)      \n\t" : :"r"(a) );
//
//	b_tmp.v   = _mm512_extload_pd( b + 2, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
//	c007_2.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_2.v ); 
//	c815_2.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_2.v ); 
//
//	b_tmp.v   = _mm512_extload_pd( b + 3, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
//	c007_3.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_3.v ); 
//	c815_3.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_3.v ); 
//
//    __asm__ volatile( "vprefetch0 192(%0)      \n\t" : :"r"(a) );
//
//	b_tmp.v   = _mm512_extload_pd( b + 4, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
//	c007_4.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_4.v ); 
//	c815_4.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_4.v ); 
//
//	b_tmp.v   = _mm512_extload_pd( b + 5, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
//	c007_5.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_5.v ); 
//	c815_5.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_5.v ); 
//
//    __asm__ volatile( "vprefetch0 64(%0)      \n\t" : :"r"(b) );
//
//	b_tmp.v   = _mm512_extload_pd( b + 6, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
//	c007_6.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_6.v ); 
//	c815_6.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_6.v ); 
//
//	b_tmp.v   = _mm512_extload_pd( b + 7, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
//	c007_7.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_7.v ); 
//	c815_7.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_7.v ); 
//
//	//b_tmp.v   = _mm512_extload_pd( b + 8, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
//	//c007_8.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_8.v ); 
//	//c815_8.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_8.v ); 
//
//	//b_tmp.v   = _mm512_extload_pd( b + 9, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
//	//c007_9.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_9.v ); 
//	//c815_9.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_9.v ); 
//
//	//b_tmp.v   = _mm512_extload_pd( b + 10, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
//	//c007_10.v = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_10.v ); 
//	//c815_10.v = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_10.v ); 
//
//	//b_tmp.v   = _mm512_extload_pd( b + 11, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
//	//c007_11.v = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_11.v ); 
//	//c815_11.v = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_11.v ); 
//
//	//b_tmp.v   = _mm512_extload_pd( b + 12, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
//	//c007_12.v = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_12.v ); 
//	//c815_12.v = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_12.v ); 
//
//	//b_tmp.v   = _mm512_extload_pd( b + 13, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
//	//c007_13.v = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_13.v ); 
//	//c815_13.v = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_13.v );
//
//	a += 16;
//	b += 8;
//  }



  a007.v    = _mm512_load_pd( a );
  a815.v    = _mm512_load_pd( a + 8 );


  // Main loop
  for ( i = 0; i < k_iter; i ++ ) {
    // Iteration #0
    __asm__ volatile( "vprefetch0 256(%0)      \n\t" : :"r"(a) );
    __asm__ volatile( "vprefetch0 320(%0)      \n\t" : :"r"(a) );

	// Preload A007
    A007.v    = _mm512_load_pd( a + 16 );

	b_tmp.v   = _mm512_extload_pd( b, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_0.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_0.v ); 
	c815_0.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_0.v ); 

	b_tmp.v   = _mm512_extload_pd( b + 1, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_1.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_1.v ); 
	c815_1.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_1.v );

    __asm__ volatile( "vprefetch0 64(%0)      \n\t" : :"r"(b) );

	b_tmp.v   = _mm512_extload_pd( b + 2, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_2.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_2.v ); 
	c815_2.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_2.v ); 

	b_tmp.v   = _mm512_extload_pd( b + 3, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_3.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_3.v ); 
	c815_3.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_3.v ); 

	// Preload A815
    A815.v    = _mm512_load_pd( a + 24 );

	b_tmp.v   = _mm512_extload_pd( b + 4, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_4.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_4.v ); 
	c815_4.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_4.v ); 

	b_tmp.v   = _mm512_extload_pd( b + 5, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_5.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_5.v ); 
	c815_5.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_5.v ); 

	b_tmp.v   = _mm512_extload_pd( b + 6, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_6.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_6.v ); 
	c815_6.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_6.v ); 

	b_tmp.v   = _mm512_extload_pd( b + 7, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_7.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_7.v ); 
	c815_7.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_7.v ); 

    // Iteration #1
    __asm__ volatile( "vprefetch0 384(%0)      \n\t" : :"r"(a) );
    __asm__ volatile( "vprefetch0 448(%0)      \n\t" : :"r"(a) );

	// Preload a007
    a007.v    = _mm512_load_pd( a + 32 );

	b_tmp.v   = _mm512_extload_pd( b + 8, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_0.v  = _mm512_fmadd_pd( A007.v, b_tmp.v, c007_0.v ); 
	c815_0.v  = _mm512_fmadd_pd( A815.v, b_tmp.v, c815_0.v ); 

	b_tmp.v   = _mm512_extload_pd( b + 9, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_1.v  = _mm512_fmadd_pd( A007.v, b_tmp.v, c007_1.v ); 
	c815_1.v  = _mm512_fmadd_pd( A815.v, b_tmp.v, c815_1.v );

    __asm__ volatile( "vprefetch0 128(%0)      \n\t" : :"r"(b) );

	b_tmp.v   = _mm512_extload_pd( b + 10, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_2.v  = _mm512_fmadd_pd( A007.v, b_tmp.v, c007_2.v ); 
	c815_2.v  = _mm512_fmadd_pd( A815.v, b_tmp.v, c815_2.v ); 

	b_tmp.v   = _mm512_extload_pd( b + 11, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_3.v  = _mm512_fmadd_pd( A007.v, b_tmp.v, c007_3.v ); 
	c815_3.v  = _mm512_fmadd_pd( A815.v, b_tmp.v, c815_3.v ); 

	// Preload a815
    a815.v    = _mm512_load_pd( a + 40 );

	b_tmp.v   = _mm512_extload_pd( b + 12, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_4.v  = _mm512_fmadd_pd( A007.v, b_tmp.v, c007_4.v ); 
	c815_4.v  = _mm512_fmadd_pd( A815.v, b_tmp.v, c815_4.v ); 

	b_tmp.v   = _mm512_extload_pd( b + 13, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_5.v  = _mm512_fmadd_pd( A007.v, b_tmp.v, c007_5.v ); 
	c815_5.v  = _mm512_fmadd_pd( A815.v, b_tmp.v, c815_5.v ); 


	b_tmp.v   = _mm512_extload_pd( b + 14, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_6.v  = _mm512_fmadd_pd( A007.v, b_tmp.v, c007_6.v ); 
	c815_6.v  = _mm512_fmadd_pd( A815.v, b_tmp.v, c815_6.v ); 

	b_tmp.v   = _mm512_extload_pd( b + 15, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_7.v  = _mm512_fmadd_pd( A007.v, b_tmp.v, c007_7.v ); 
	c815_7.v  = _mm512_fmadd_pd( A815.v, b_tmp.v, c815_7.v ); 

	a += 32;
	b += 16;
  }


//  printf( "rank-k:c007_0\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", 
//	  c007_0.d[ 0 ], c007_0.d[ 1 ], c007_0.d[ 2 ],  c007_0.d[ 3 ], 
//	  c007_0.d[ 4 ], c007_0.d[ 5 ], c007_0.d[ 6 ],  c007_0.d[ 7 ] );
//
//  printf( "rank-k:c815_0\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
//    c815_0.d[ 0 ], c815_0.d[ 1 ], c815_0.d[ 2 ], c815_0.d[ 3 ], 
//	c815_0.d[ 4 ], c815_0.d[ 5 ], c815_0.d[ 6 ], c815_0.d[ 7 ] );





  // TODO: we can possibly combine add + mul to fma


  //printf( "finish rank-k\n" );

  // Scale with -2
  __asm__ volatile( "vprefetch0 0(%0)      \n\t" : :"r"(bb) );
  __asm__ volatile( "vprefetch0 64(%0)      \n\t" : :"r"(aa) );
  a007.v    = _mm512_load_pd( aa );
  b_tmp.v   = _mm512_extload_pd( &neg2, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );

  c007_0.v  = _mm512_fmadd_pd( c007_0.v, b_tmp.v, a007.v ); 
  c007_1.v  = _mm512_fmadd_pd( c007_1.v, b_tmp.v, a007.v ); 
  c007_2.v  = _mm512_fmadd_pd( c007_2.v, b_tmp.v, a007.v ); 
  c007_3.v  = _mm512_fmadd_pd( c007_3.v, b_tmp.v, a007.v ); 
  c007_4.v  = _mm512_fmadd_pd( c007_4.v, b_tmp.v, a007.v ); 
  c007_5.v  = _mm512_fmadd_pd( c007_5.v, b_tmp.v, a007.v ); 
  c007_6.v  = _mm512_fmadd_pd( c007_6.v, b_tmp.v, a007.v ); 
  c007_7.v  = _mm512_fmadd_pd( c007_7.v, b_tmp.v, a007.v ); 

  a815.v    = _mm512_load_pd( aa + 8 );

  c815_0.v  = _mm512_fmadd_pd( c815_0.v, b_tmp.v, a815.v ); 
  c815_1.v  = _mm512_fmadd_pd( c815_1.v, b_tmp.v, a815.v ); 
  c815_2.v  = _mm512_fmadd_pd( c815_2.v, b_tmp.v, a815.v ); 
  c815_3.v  = _mm512_fmadd_pd( c815_3.v, b_tmp.v, a815.v ); 
  c815_4.v  = _mm512_fmadd_pd( c815_4.v, b_tmp.v, a815.v ); 
  c815_5.v  = _mm512_fmadd_pd( c815_5.v, b_tmp.v, a815.v ); 
  c815_6.v  = _mm512_fmadd_pd( c815_6.v, b_tmp.v, a815.v ); 
  c815_7.v  = _mm512_fmadd_pd( c815_7.v, b_tmp.v, a815.v ); 


  b_tmp.v   = _mm512_extload_pd( bb, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  c007_0.v  = _mm512_add_pd( c007_0.v, b_tmp.v );
  c815_0.v  = _mm512_add_pd( c815_0.v, b_tmp.v );

  b_tmp.v   = _mm512_extload_pd( bb + 1, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  c007_1.v  = _mm512_add_pd( c007_1.v, b_tmp.v );
  c815_1.v  = _mm512_add_pd( c815_1.v, b_tmp.v );

  b_tmp.v   = _mm512_extload_pd( bb + 2, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  c007_2.v  = _mm512_add_pd( c007_2.v, b_tmp.v );
  c815_2.v  = _mm512_add_pd( c815_2.v, b_tmp.v );

  b_tmp.v   = _mm512_extload_pd( bb + 3, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  c007_3.v  = _mm512_add_pd( c007_3.v, b_tmp.v );
  c815_3.v  = _mm512_add_pd( c815_3.v, b_tmp.v );

  b_tmp.v   = _mm512_extload_pd( bb + 4, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  c007_4.v  = _mm512_add_pd( c007_4.v, b_tmp.v );
  c815_4.v  = _mm512_add_pd( c815_4.v, b_tmp.v );

  b_tmp.v   = _mm512_extload_pd( bb + 5, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  c007_5.v  = _mm512_add_pd( c007_5.v, b_tmp.v );
  c815_5.v  = _mm512_add_pd( c815_5.v, b_tmp.v );

  b_tmp.v   = _mm512_extload_pd( bb + 6, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  c007_6.v  = _mm512_add_pd( c007_6.v, b_tmp.v );
  c815_6.v  = _mm512_add_pd( c815_6.v, b_tmp.v );

  b_tmp.v   = _mm512_extload_pd( bb + 7, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  c007_7.v  = _mm512_add_pd( c007_7.v, b_tmp.v );
  c815_7.v  = _mm512_add_pd( c815_7.v, b_tmp.v );

//  printf( "\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", 
//	  c007_0.d[ 0 ], c007_0.d[ 1 ], c007_0.d[ 2 ],  c007_0.d[ 3 ], 
//	  c007_0.d[ 4 ], c007_0.d[ 5 ], c007_0.d[ 6 ],  c007_0.d[ 7 ] );


//  printf( "square distance: c007_0\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", 
//	  c007_0.d[ 0 ], c007_0.d[ 1 ], c007_0.d[ 2 ],  c007_0.d[ 3 ], 
//	  c007_0.d[ 4 ], c007_0.d[ 5 ], c007_0.d[ 6 ],  c007_0.d[ 7 ] );
//
//  printf( "square distance: c815_0\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
//    c815_0.d[ 0 ], c815_0.d[ 1 ], c815_0.d[ 2 ], c815_0.d[ 3 ], 
//	c815_0.d[ 4 ], c815_0.d[ 5 ], c815_0.d[ 6 ], c815_0.d[ 7 ] );



  // Check if any square value is negative due the edge case.
  b_tmp.v   = _mm512_extload_pd( &dzero, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  c007_0.v  = _mm512_gmax_pd( c007_0.v, b_tmp.v );
  c007_1.v  = _mm512_gmax_pd( c007_1.v, b_tmp.v );
  c007_2.v  = _mm512_gmax_pd( c007_2.v, b_tmp.v );
  c007_3.v  = _mm512_gmax_pd( c007_3.v, b_tmp.v );
  c007_4.v  = _mm512_gmax_pd( c007_4.v, b_tmp.v );
  c007_5.v  = _mm512_gmax_pd( c007_5.v, b_tmp.v );
  c007_6.v  = _mm512_gmax_pd( c007_6.v, b_tmp.v );
  c007_7.v  = _mm512_gmax_pd( c007_7.v, b_tmp.v );

  c815_0.v  = _mm512_gmax_pd( c815_0.v, b_tmp.v );
  c815_1.v  = _mm512_gmax_pd( c815_1.v, b_tmp.v );
  c815_2.v  = _mm512_gmax_pd( c815_2.v, b_tmp.v );
  c815_3.v  = _mm512_gmax_pd( c815_3.v, b_tmp.v );
  c815_4.v  = _mm512_gmax_pd( c815_4.v, b_tmp.v );
  c815_5.v  = _mm512_gmax_pd( c815_5.v, b_tmp.v );
  c815_6.v  = _mm512_gmax_pd( c815_6.v, b_tmp.v );
  c815_7.v  = _mm512_gmax_pd( c815_7.v, b_tmp.v );



//  printf( "max( c007_0, 0.0 )\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", 
//	  c007_0.d[ 0 ], c007_0.d[ 1 ], c007_0.d[ 2 ],  c007_0.d[ 3 ], 
//	  c007_0.d[ 4 ], c007_0.d[ 5 ], c007_0.d[ 6 ],  c007_0.d[ 7 ] );
//
//  printf( "max( c815_0, 0.0 )\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
//    c815_0.d[ 0 ], c815_0.d[ 1 ], c815_0.d[ 2 ], c815_0.d[ 3 ], 
//	c815_0.d[ 4 ], c815_0.d[ 5 ], c815_0.d[ 6 ], c815_0.d[ 7 ] );



  // Scale with alpha
  b_tmp.v   = _mm512_extload_pd( &alpha, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );

  c007_0.v  = _mm512_mul_pd( c007_0.v, b_tmp.v );
  c007_1.v  = _mm512_mul_pd( c007_1.v, b_tmp.v );
  c007_2.v  = _mm512_mul_pd( c007_2.v, b_tmp.v );
  c007_3.v  = _mm512_mul_pd( c007_3.v, b_tmp.v );
  c007_4.v  = _mm512_mul_pd( c007_4.v, b_tmp.v );
  c007_5.v  = _mm512_mul_pd( c007_5.v, b_tmp.v );
  c007_6.v  = _mm512_mul_pd( c007_6.v, b_tmp.v );
  c007_7.v  = _mm512_mul_pd( c007_7.v, b_tmp.v );

  c815_0.v  = _mm512_mul_pd( c815_0.v, b_tmp.v );
  c815_1.v  = _mm512_mul_pd( c815_1.v, b_tmp.v );
  c815_2.v  = _mm512_mul_pd( c815_2.v, b_tmp.v );
  c815_3.v  = _mm512_mul_pd( c815_3.v, b_tmp.v );
  c815_4.v  = _mm512_mul_pd( c815_4.v, b_tmp.v );
  c815_5.v  = _mm512_mul_pd( c815_5.v, b_tmp.v );
  c815_6.v  = _mm512_mul_pd( c815_6.v, b_tmp.v );
  c815_7.v  = _mm512_mul_pd( c815_7.v, b_tmp.v );

//  printf( "scale c815_0\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
//    c815_0.d[ 0 ], c815_0.d[ 1 ], c815_0.d[ 2 ], c815_0.d[ 3 ], 
//	c815_0.d[ 4 ], c815_0.d[ 5 ], c815_0.d[ 6 ], c815_0.d[ 7 ] );






  // exp()
  b_tmp.v   = _mm512_extload_pd( &maxlog, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  c007_0.v  = _mm512_gmin_pd( c007_0.v, b_tmp.v );
  c007_1.v  = _mm512_gmin_pd( c007_1.v, b_tmp.v );
  c007_2.v  = _mm512_gmin_pd( c007_2.v, b_tmp.v );
  c007_3.v  = _mm512_gmin_pd( c007_3.v, b_tmp.v );
  c007_4.v  = _mm512_gmin_pd( c007_4.v, b_tmp.v );
  c007_5.v  = _mm512_gmin_pd( c007_5.v, b_tmp.v );
  c007_6.v  = _mm512_gmin_pd( c007_6.v, b_tmp.v );
  c007_7.v  = _mm512_gmin_pd( c007_7.v, b_tmp.v );

  c815_0.v  = _mm512_gmin_pd( c815_0.v, b_tmp.v );
  c815_1.v  = _mm512_gmin_pd( c815_1.v, b_tmp.v );
  c815_2.v  = _mm512_gmin_pd( c815_2.v, b_tmp.v );
  c815_3.v  = _mm512_gmin_pd( c815_3.v, b_tmp.v );
  c815_4.v  = _mm512_gmin_pd( c815_4.v, b_tmp.v );
  c815_5.v  = _mm512_gmin_pd( c815_5.v, b_tmp.v );
  c815_6.v  = _mm512_gmin_pd( c815_6.v, b_tmp.v );
  c815_7.v  = _mm512_gmin_pd( c815_7.v, b_tmp.v );

  b_tmp.v   = _mm512_extload_pd( &minlog, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  c007_0.v  = _mm512_gmax_pd( c007_0.v, b_tmp.v );
  c007_1.v  = _mm512_gmax_pd( c007_1.v, b_tmp.v );
  c007_2.v  = _mm512_gmax_pd( c007_2.v, b_tmp.v );
  c007_3.v  = _mm512_gmax_pd( c007_3.v, b_tmp.v );
  c007_4.v  = _mm512_gmax_pd( c007_4.v, b_tmp.v );
  c007_5.v  = _mm512_gmax_pd( c007_5.v, b_tmp.v );
  c007_6.v  = _mm512_gmax_pd( c007_6.v, b_tmp.v );
  c007_7.v  = _mm512_gmax_pd( c007_7.v, b_tmp.v );

  c815_0.v  = _mm512_gmax_pd( c815_0.v, b_tmp.v );
  c815_1.v  = _mm512_gmax_pd( c815_1.v, b_tmp.v );
  c815_2.v  = _mm512_gmax_pd( c815_2.v, b_tmp.v );
  c815_3.v  = _mm512_gmax_pd( c815_3.v, b_tmp.v );
  c815_4.v  = _mm512_gmax_pd( c815_4.v, b_tmp.v );
  c815_5.v  = _mm512_gmax_pd( c815_5.v, b_tmp.v );
  c815_6.v  = _mm512_gmax_pd( c815_6.v, b_tmp.v );
  c815_7.v  = _mm512_gmax_pd( c815_7.v, b_tmp.v );



  // a = c / log2
  l2e.v     = _mm512_extload_pd( &log2e, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  b_tmp.v   = _mm512_extload_pd( &dmone, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );

  a007_0.v  = _mm512_fmadd_pd( c007_0.v, l2e.v, b_tmp.v ); 
  a007_1.v  = _mm512_fmadd_pd( c007_1.v, l2e.v, b_tmp.v ); 
  a007_2.v  = _mm512_fmadd_pd( c007_2.v, l2e.v, b_tmp.v ); 
  a007_3.v  = _mm512_fmadd_pd( c007_3.v, l2e.v, b_tmp.v ); 
  a007_4.v  = _mm512_fmadd_pd( c007_4.v, l2e.v, b_tmp.v ); 
  a007_5.v  = _mm512_fmadd_pd( c007_5.v, l2e.v, b_tmp.v ); 
  a007_6.v  = _mm512_fmadd_pd( c007_6.v, l2e.v, b_tmp.v ); 
  a007_7.v  = _mm512_fmadd_pd( c007_7.v, l2e.v, b_tmp.v ); 

  a815_0.v  = _mm512_fmadd_pd( c815_0.v, l2e.v, b_tmp.v ); 
  a815_1.v  = _mm512_fmadd_pd( c815_1.v, l2e.v, b_tmp.v ); 
  a815_2.v  = _mm512_fmadd_pd( c815_2.v, l2e.v, b_tmp.v ); 
  a815_3.v  = _mm512_fmadd_pd( c815_3.v, l2e.v, b_tmp.v ); 
  a815_4.v  = _mm512_fmadd_pd( c815_4.v, l2e.v, b_tmp.v ); 
  a815_5.v  = _mm512_fmadd_pd( c815_5.v, l2e.v, b_tmp.v ); 
  a815_6.v  = _mm512_fmadd_pd( c815_6.v, l2e.v, b_tmp.v ); 
  a815_7.v  = _mm512_fmadd_pd( c815_7.v, l2e.v, b_tmp.v ); 


//  printf( "a - 1\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", 
//	  a007_0.d[ 0 ], a007_0.d[ 1 ], a007_0.d[ 2 ],  a007_0.d[ 3 ], 
//	  a007_0.d[ 4 ], a007_0.d[ 5 ], a007_0.d[ 6 ],  a007_0.d[ 7 ] );



  // Convert a to integer
  k007_0.v  = _mm512_cvtfxpnt_roundpd_epi32lo( a007_0.v, _MM_FROUND_TO_POS_INF );
  k007_1.v  = _mm512_cvtfxpnt_roundpd_epi32lo( a007_1.v, _MM_FROUND_TO_POS_INF );
  k007_2.v  = _mm512_cvtfxpnt_roundpd_epi32lo( a007_2.v, _MM_FROUND_TO_POS_INF );
  k007_3.v  = _mm512_cvtfxpnt_roundpd_epi32lo( a007_3.v, _MM_FROUND_TO_POS_INF );
  k007_4.v  = _mm512_cvtfxpnt_roundpd_epi32lo( a007_4.v, _MM_FROUND_TO_POS_INF );
  k007_5.v  = _mm512_cvtfxpnt_roundpd_epi32lo( a007_5.v, _MM_FROUND_TO_POS_INF );
  k007_6.v  = _mm512_cvtfxpnt_roundpd_epi32lo( a007_6.v, _MM_FROUND_TO_POS_INF );
  k007_7.v  = _mm512_cvtfxpnt_roundpd_epi32lo( a007_7.v, _MM_FROUND_TO_POS_INF );

  k815_0.v  = _mm512_cvtfxpnt_roundpd_epi32lo( a815_0.v, _MM_FROUND_TO_POS_INF );
  k815_1.v  = _mm512_cvtfxpnt_roundpd_epi32lo( a815_1.v, _MM_FROUND_TO_POS_INF );
  k815_2.v  = _mm512_cvtfxpnt_roundpd_epi32lo( a815_2.v, _MM_FROUND_TO_POS_INF );
  k815_3.v  = _mm512_cvtfxpnt_roundpd_epi32lo( a815_3.v, _MM_FROUND_TO_POS_INF );
  k815_4.v  = _mm512_cvtfxpnt_roundpd_epi32lo( a815_4.v, _MM_FROUND_TO_POS_INF );
  k815_5.v  = _mm512_cvtfxpnt_roundpd_epi32lo( a815_5.v, _MM_FROUND_TO_POS_INF );
  k815_6.v  = _mm512_cvtfxpnt_roundpd_epi32lo( a815_6.v, _MM_FROUND_TO_POS_INF );
  k815_7.v  = _mm512_cvtfxpnt_roundpd_epi32lo( a815_7.v, _MM_FROUND_TO_POS_INF );




  // Covert integer back to double
  p007_0.v  = _mm512_cvtepi32lo_pd( k007_0.v );
  p007_1.v  = _mm512_cvtepi32lo_pd( k007_1.v );
  p007_2.v  = _mm512_cvtepi32lo_pd( k007_2.v );
  p007_3.v  = _mm512_cvtepi32lo_pd( k007_3.v );
  p007_4.v  = _mm512_cvtepi32lo_pd( k007_4.v );
  p007_5.v  = _mm512_cvtepi32lo_pd( k007_5.v );
  p007_6.v  = _mm512_cvtepi32lo_pd( k007_6.v );
  p007_7.v  = _mm512_cvtepi32lo_pd( k007_7.v );

  p815_0.v  = _mm512_cvtepi32lo_pd( k815_0.v );
  p815_1.v  = _mm512_cvtepi32lo_pd( k815_1.v );
  p815_2.v  = _mm512_cvtepi32lo_pd( k815_2.v );
  p815_3.v  = _mm512_cvtepi32lo_pd( k815_3.v );
  p815_4.v  = _mm512_cvtepi32lo_pd( k815_4.v );
  p815_5.v  = _mm512_cvtepi32lo_pd( k815_5.v );
  p815_6.v  = _mm512_cvtepi32lo_pd( k815_6.v );
  p815_7.v  = _mm512_cvtepi32lo_pd( k815_7.v );

//  printf( "p007_0\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", 
//	  p007_0.d[ 0 ], p007_0.d[ 1 ], p007_0.d[ 2 ],  p007_0.d[ 3 ], 
//	  p007_0.d[ 4 ], p007_0.d[ 5 ], p007_0.d[ 6 ],  p007_0.d[ 7 ] );
//
//  printf( "cp815_0\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
//    p815_0.d[ 0 ], p815_0.d[ 1 ], p815_0.d[ 2 ], p815_0.d[ 3 ], 
//	p815_0.d[ 4 ], p815_0.d[ 5 ], p815_0.d[ 6 ], p815_0.d[ 7 ] );


  b_tmp.v   = _mm512_extload_pd( &c1, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );

  c007_0.v  = _mm512_fmadd_pd( p007_0.v, b_tmp.v, c007_0.v ); 
  c007_1.v  = _mm512_fmadd_pd( p007_1.v, b_tmp.v, c007_1.v ); 
  c007_2.v  = _mm512_fmadd_pd( p007_2.v, b_tmp.v, c007_2.v ); 
  c007_3.v  = _mm512_fmadd_pd( p007_3.v, b_tmp.v, c007_3.v ); 
  c007_4.v  = _mm512_fmadd_pd( p007_4.v, b_tmp.v, c007_4.v ); 
  c007_5.v  = _mm512_fmadd_pd( p007_5.v, b_tmp.v, c007_5.v ); 
  c007_6.v  = _mm512_fmadd_pd( p007_6.v, b_tmp.v, c007_6.v ); 
  c007_7.v  = _mm512_fmadd_pd( p007_7.v, b_tmp.v, c007_7.v ); 

  c815_0.v  = _mm512_fmadd_pd( p815_0.v, b_tmp.v, c815_0.v ); 
  c815_1.v  = _mm512_fmadd_pd( p815_1.v, b_tmp.v, c815_1.v ); 
  c815_2.v  = _mm512_fmadd_pd( p815_2.v, b_tmp.v, c815_2.v ); 
  c815_3.v  = _mm512_fmadd_pd( p815_3.v, b_tmp.v, c815_3.v ); 
  c815_4.v  = _mm512_fmadd_pd( p815_4.v, b_tmp.v, c815_4.v ); 
  c815_5.v  = _mm512_fmadd_pd( p815_5.v, b_tmp.v, c815_5.v ); 
  c815_6.v  = _mm512_fmadd_pd( p815_6.v, b_tmp.v, c815_6.v ); 
  c815_7.v  = _mm512_fmadd_pd( p815_7.v, b_tmp.v, c815_7.v ); 


  b_tmp.v   = _mm512_extload_pd( &c2, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  c007_0.v  = _mm512_fmadd_pd( p007_0.v, b_tmp.v, c007_0.v ); 
  c007_1.v  = _mm512_fmadd_pd( p007_1.v, b_tmp.v, c007_1.v ); 
  c007_2.v  = _mm512_fmadd_pd( p007_2.v, b_tmp.v, c007_2.v ); 
  c007_3.v  = _mm512_fmadd_pd( p007_3.v, b_tmp.v, c007_3.v ); 
  c007_4.v  = _mm512_fmadd_pd( p007_4.v, b_tmp.v, c007_4.v ); 
  c007_5.v  = _mm512_fmadd_pd( p007_5.v, b_tmp.v, c007_5.v ); 
  c007_6.v  = _mm512_fmadd_pd( p007_6.v, b_tmp.v, c007_6.v ); 
  c007_7.v  = _mm512_fmadd_pd( p007_7.v, b_tmp.v, c007_7.v ); 

  c815_0.v  = _mm512_fmadd_pd( p815_0.v, b_tmp.v, c815_0.v ); 
  c815_1.v  = _mm512_fmadd_pd( p815_1.v, b_tmp.v, c815_1.v ); 
  c815_2.v  = _mm512_fmadd_pd( p815_2.v, b_tmp.v, c815_2.v ); 
  c815_3.v  = _mm512_fmadd_pd( p815_3.v, b_tmp.v, c815_3.v ); 
  c815_4.v  = _mm512_fmadd_pd( p815_4.v, b_tmp.v, c815_4.v ); 
  c815_5.v  = _mm512_fmadd_pd( p815_5.v, b_tmp.v, c815_5.v ); 
  c815_6.v  = _mm512_fmadd_pd( p815_6.v, b_tmp.v, c815_6.v ); 
  c815_7.v  = _mm512_fmadd_pd( p815_7.v, b_tmp.v, c815_7.v ); 


  __asm__ volatile( "vprefetch0 0(%0)      \n\t" : :"r"(mask) );


  // Nested polynomail evaluation
  l2e.v     = _mm512_extload_pd( &w11, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  b_tmp.v   = _mm512_extload_pd( &w10, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );

  a007_0.v  = _mm512_fmadd_pd( c007_0.v, l2e.v, b_tmp.v ); 
  a007_1.v  = _mm512_fmadd_pd( c007_1.v, l2e.v, b_tmp.v ); 
  a007_2.v  = _mm512_fmadd_pd( c007_2.v, l2e.v, b_tmp.v ); 
  a007_3.v  = _mm512_fmadd_pd( c007_3.v, l2e.v, b_tmp.v ); 
  a007_4.v  = _mm512_fmadd_pd( c007_4.v, l2e.v, b_tmp.v ); 
  a007_5.v  = _mm512_fmadd_pd( c007_5.v, l2e.v, b_tmp.v ); 
  a007_6.v  = _mm512_fmadd_pd( c007_6.v, l2e.v, b_tmp.v ); 
  a007_7.v  = _mm512_fmadd_pd( c007_7.v, l2e.v, b_tmp.v ); 

  a815_0.v  = _mm512_fmadd_pd( c815_0.v, l2e.v, b_tmp.v ); 
  a815_1.v  = _mm512_fmadd_pd( c815_1.v, l2e.v, b_tmp.v ); 
  a815_2.v  = _mm512_fmadd_pd( c815_2.v, l2e.v, b_tmp.v ); 
  a815_3.v  = _mm512_fmadd_pd( c815_3.v, l2e.v, b_tmp.v ); 
  a815_4.v  = _mm512_fmadd_pd( c815_4.v, l2e.v, b_tmp.v ); 
  a815_5.v  = _mm512_fmadd_pd( c815_5.v, l2e.v, b_tmp.v ); 
  a815_6.v  = _mm512_fmadd_pd( c815_6.v, l2e.v, b_tmp.v ); 
  a815_7.v  = _mm512_fmadd_pd( c815_7.v, l2e.v, b_tmp.v ); 

  
  

  b_tmp.v   = _mm512_extload_pd( &w9, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  a007_0.v  = _mm512_fmadd_pd( a007_0.v, c007_0.v, b_tmp.v ); 
  a007_1.v  = _mm512_fmadd_pd( a007_1.v, c007_1.v, b_tmp.v ); 
  a007_2.v  = _mm512_fmadd_pd( a007_2.v, c007_2.v, b_tmp.v ); 
  a007_3.v  = _mm512_fmadd_pd( a007_3.v, c007_3.v, b_tmp.v ); 
  a007_4.v  = _mm512_fmadd_pd( a007_4.v, c007_4.v, b_tmp.v ); 
  a007_5.v  = _mm512_fmadd_pd( a007_5.v, c007_5.v, b_tmp.v ); 
  a007_6.v  = _mm512_fmadd_pd( a007_6.v, c007_6.v, b_tmp.v ); 
  a007_7.v  = _mm512_fmadd_pd( a007_7.v, c007_7.v, b_tmp.v ); 

  a815_0.v  = _mm512_fmadd_pd( a815_0.v, c815_0.v, b_tmp.v ); 
  a815_1.v  = _mm512_fmadd_pd( a815_1.v, c815_1.v, b_tmp.v ); 
  a815_2.v  = _mm512_fmadd_pd( a815_2.v, c815_2.v, b_tmp.v ); 
  a815_3.v  = _mm512_fmadd_pd( a815_3.v, c815_3.v, b_tmp.v ); 
  a815_4.v  = _mm512_fmadd_pd( a815_4.v, c815_4.v, b_tmp.v ); 
  a815_5.v  = _mm512_fmadd_pd( a815_5.v, c815_5.v, b_tmp.v ); 
  a815_6.v  = _mm512_fmadd_pd( a815_6.v, c815_6.v, b_tmp.v ); 
  a815_7.v  = _mm512_fmadd_pd( a815_7.v, c815_7.v, b_tmp.v ); 

  b_tmp.v   = _mm512_extload_pd( &w8, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  a007_0.v  = _mm512_fmadd_pd( a007_0.v, c007_0.v, b_tmp.v ); 
  a007_1.v  = _mm512_fmadd_pd( a007_1.v, c007_1.v, b_tmp.v ); 
  a007_2.v  = _mm512_fmadd_pd( a007_2.v, c007_2.v, b_tmp.v ); 
  a007_3.v  = _mm512_fmadd_pd( a007_3.v, c007_3.v, b_tmp.v ); 
  a007_4.v  = _mm512_fmadd_pd( a007_4.v, c007_4.v, b_tmp.v ); 
  a007_5.v  = _mm512_fmadd_pd( a007_5.v, c007_5.v, b_tmp.v ); 
  a007_6.v  = _mm512_fmadd_pd( a007_6.v, c007_6.v, b_tmp.v ); 
  a007_7.v  = _mm512_fmadd_pd( a007_7.v, c007_7.v, b_tmp.v ); 

  a815_0.v  = _mm512_fmadd_pd( a815_0.v, c815_0.v, b_tmp.v ); 
  a815_1.v  = _mm512_fmadd_pd( a815_1.v, c815_1.v, b_tmp.v ); 
  a815_2.v  = _mm512_fmadd_pd( a815_2.v, c815_2.v, b_tmp.v ); 
  a815_3.v  = _mm512_fmadd_pd( a815_3.v, c815_3.v, b_tmp.v ); 
  a815_4.v  = _mm512_fmadd_pd( a815_4.v, c815_4.v, b_tmp.v ); 
  a815_5.v  = _mm512_fmadd_pd( a815_5.v, c815_5.v, b_tmp.v ); 
  a815_6.v  = _mm512_fmadd_pd( a815_6.v, c815_6.v, b_tmp.v ); 
  a815_7.v  = _mm512_fmadd_pd( a815_7.v, c815_7.v, b_tmp.v ); 

  b_tmp.v   = _mm512_extload_pd( &w7, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  a007_0.v  = _mm512_fmadd_pd( a007_0.v, c007_0.v, b_tmp.v ); 
  a007_1.v  = _mm512_fmadd_pd( a007_1.v, c007_1.v, b_tmp.v ); 
  a007_2.v  = _mm512_fmadd_pd( a007_2.v, c007_2.v, b_tmp.v ); 
  a007_3.v  = _mm512_fmadd_pd( a007_3.v, c007_3.v, b_tmp.v ); 
  a007_4.v  = _mm512_fmadd_pd( a007_4.v, c007_4.v, b_tmp.v ); 
  a007_5.v  = _mm512_fmadd_pd( a007_5.v, c007_5.v, b_tmp.v ); 
  a007_6.v  = _mm512_fmadd_pd( a007_6.v, c007_6.v, b_tmp.v ); 
  a007_7.v  = _mm512_fmadd_pd( a007_7.v, c007_7.v, b_tmp.v ); 

  a815_0.v  = _mm512_fmadd_pd( a815_0.v, c815_0.v, b_tmp.v ); 
  a815_1.v  = _mm512_fmadd_pd( a815_1.v, c815_1.v, b_tmp.v ); 
  a815_2.v  = _mm512_fmadd_pd( a815_2.v, c815_2.v, b_tmp.v ); 
  a815_3.v  = _mm512_fmadd_pd( a815_3.v, c815_3.v, b_tmp.v ); 
  a815_4.v  = _mm512_fmadd_pd( a815_4.v, c815_4.v, b_tmp.v ); 
  a815_5.v  = _mm512_fmadd_pd( a815_5.v, c815_5.v, b_tmp.v ); 
  a815_6.v  = _mm512_fmadd_pd( a815_6.v, c815_6.v, b_tmp.v ); 
  a815_7.v  = _mm512_fmadd_pd( a815_7.v, c815_7.v, b_tmp.v ); 

  b_tmp.v   = _mm512_extload_pd( &w6, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  a007_0.v  = _mm512_fmadd_pd( a007_0.v, c007_0.v, b_tmp.v ); 
  a007_1.v  = _mm512_fmadd_pd( a007_1.v, c007_1.v, b_tmp.v ); 
  a007_2.v  = _mm512_fmadd_pd( a007_2.v, c007_2.v, b_tmp.v ); 
  a007_3.v  = _mm512_fmadd_pd( a007_3.v, c007_3.v, b_tmp.v ); 
  a007_4.v  = _mm512_fmadd_pd( a007_4.v, c007_4.v, b_tmp.v ); 
  a007_5.v  = _mm512_fmadd_pd( a007_5.v, c007_5.v, b_tmp.v ); 
  a007_6.v  = _mm512_fmadd_pd( a007_6.v, c007_6.v, b_tmp.v ); 
  a007_7.v  = _mm512_fmadd_pd( a007_7.v, c007_7.v, b_tmp.v ); 

  a815_0.v  = _mm512_fmadd_pd( a815_0.v, c815_0.v, b_tmp.v ); 
  a815_1.v  = _mm512_fmadd_pd( a815_1.v, c815_1.v, b_tmp.v ); 
  a815_2.v  = _mm512_fmadd_pd( a815_2.v, c815_2.v, b_tmp.v ); 
  a815_3.v  = _mm512_fmadd_pd( a815_3.v, c815_3.v, b_tmp.v ); 
  a815_4.v  = _mm512_fmadd_pd( a815_4.v, c815_4.v, b_tmp.v ); 
  a815_5.v  = _mm512_fmadd_pd( a815_5.v, c815_5.v, b_tmp.v ); 
  a815_6.v  = _mm512_fmadd_pd( a815_6.v, c815_6.v, b_tmp.v ); 
  a815_7.v  = _mm512_fmadd_pd( a815_7.v, c815_7.v, b_tmp.v ); 

  b_tmp.v   = _mm512_extload_pd( &w5, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  a007_0.v  = _mm512_fmadd_pd( a007_0.v, c007_0.v, b_tmp.v ); 
  a007_1.v  = _mm512_fmadd_pd( a007_1.v, c007_1.v, b_tmp.v ); 
  a007_2.v  = _mm512_fmadd_pd( a007_2.v, c007_2.v, b_tmp.v ); 
  a007_3.v  = _mm512_fmadd_pd( a007_3.v, c007_3.v, b_tmp.v ); 
  a007_4.v  = _mm512_fmadd_pd( a007_4.v, c007_4.v, b_tmp.v ); 
  a007_5.v  = _mm512_fmadd_pd( a007_5.v, c007_5.v, b_tmp.v ); 
  a007_6.v  = _mm512_fmadd_pd( a007_6.v, c007_6.v, b_tmp.v ); 
  a007_7.v  = _mm512_fmadd_pd( a007_7.v, c007_7.v, b_tmp.v ); 

  a815_0.v  = _mm512_fmadd_pd( a815_0.v, c815_0.v, b_tmp.v ); 
  a815_1.v  = _mm512_fmadd_pd( a815_1.v, c815_1.v, b_tmp.v ); 
  a815_2.v  = _mm512_fmadd_pd( a815_2.v, c815_2.v, b_tmp.v ); 
  a815_3.v  = _mm512_fmadd_pd( a815_3.v, c815_3.v, b_tmp.v ); 
  a815_4.v  = _mm512_fmadd_pd( a815_4.v, c815_4.v, b_tmp.v ); 
  a815_5.v  = _mm512_fmadd_pd( a815_5.v, c815_5.v, b_tmp.v ); 
  a815_6.v  = _mm512_fmadd_pd( a815_6.v, c815_6.v, b_tmp.v ); 
  a815_7.v  = _mm512_fmadd_pd( a815_7.v, c815_7.v, b_tmp.v ); 

  b_tmp.v   = _mm512_extload_pd( &w4, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  a007_0.v  = _mm512_fmadd_pd( a007_0.v, c007_0.v, b_tmp.v ); 
  a007_1.v  = _mm512_fmadd_pd( a007_1.v, c007_1.v, b_tmp.v ); 
  a007_2.v  = _mm512_fmadd_pd( a007_2.v, c007_2.v, b_tmp.v ); 
  a007_3.v  = _mm512_fmadd_pd( a007_3.v, c007_3.v, b_tmp.v ); 
  a007_4.v  = _mm512_fmadd_pd( a007_4.v, c007_4.v, b_tmp.v ); 
  a007_5.v  = _mm512_fmadd_pd( a007_5.v, c007_5.v, b_tmp.v ); 
  a007_6.v  = _mm512_fmadd_pd( a007_6.v, c007_6.v, b_tmp.v ); 
  a007_7.v  = _mm512_fmadd_pd( a007_7.v, c007_7.v, b_tmp.v ); 

  a815_0.v  = _mm512_fmadd_pd( a815_0.v, c815_0.v, b_tmp.v ); 
  a815_1.v  = _mm512_fmadd_pd( a815_1.v, c815_1.v, b_tmp.v ); 
  a815_2.v  = _mm512_fmadd_pd( a815_2.v, c815_2.v, b_tmp.v ); 
  a815_3.v  = _mm512_fmadd_pd( a815_3.v, c815_3.v, b_tmp.v ); 
  a815_4.v  = _mm512_fmadd_pd( a815_4.v, c815_4.v, b_tmp.v ); 
  a815_5.v  = _mm512_fmadd_pd( a815_5.v, c815_5.v, b_tmp.v ); 
  a815_6.v  = _mm512_fmadd_pd( a815_6.v, c815_6.v, b_tmp.v ); 
  a815_7.v  = _mm512_fmadd_pd( a815_7.v, c815_7.v, b_tmp.v ); 

  b_tmp.v   = _mm512_extload_pd( &w3, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  a007_0.v  = _mm512_fmadd_pd( a007_0.v, c007_0.v, b_tmp.v ); 
  a007_1.v  = _mm512_fmadd_pd( a007_1.v, c007_1.v, b_tmp.v ); 
  a007_2.v  = _mm512_fmadd_pd( a007_2.v, c007_2.v, b_tmp.v ); 
  a007_3.v  = _mm512_fmadd_pd( a007_3.v, c007_3.v, b_tmp.v ); 
  a007_4.v  = _mm512_fmadd_pd( a007_4.v, c007_4.v, b_tmp.v ); 
  a007_5.v  = _mm512_fmadd_pd( a007_5.v, c007_5.v, b_tmp.v ); 
  a007_6.v  = _mm512_fmadd_pd( a007_6.v, c007_6.v, b_tmp.v ); 
  a007_7.v  = _mm512_fmadd_pd( a007_7.v, c007_7.v, b_tmp.v ); 

  a815_0.v  = _mm512_fmadd_pd( a815_0.v, c815_0.v, b_tmp.v ); 
  a815_1.v  = _mm512_fmadd_pd( a815_1.v, c815_1.v, b_tmp.v ); 
  a815_2.v  = _mm512_fmadd_pd( a815_2.v, c815_2.v, b_tmp.v ); 
  a815_3.v  = _mm512_fmadd_pd( a815_3.v, c815_3.v, b_tmp.v ); 
  a815_4.v  = _mm512_fmadd_pd( a815_4.v, c815_4.v, b_tmp.v ); 
  a815_5.v  = _mm512_fmadd_pd( a815_5.v, c815_5.v, b_tmp.v ); 
  a815_6.v  = _mm512_fmadd_pd( a815_6.v, c815_6.v, b_tmp.v ); 
  a815_7.v  = _mm512_fmadd_pd( a815_7.v, c815_7.v, b_tmp.v ); 

  b_tmp.v   = _mm512_extload_pd( &w2, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  a007_0.v  = _mm512_fmadd_pd( a007_0.v, c007_0.v, b_tmp.v ); 
  a007_1.v  = _mm512_fmadd_pd( a007_1.v, c007_1.v, b_tmp.v ); 
  a007_2.v  = _mm512_fmadd_pd( a007_2.v, c007_2.v, b_tmp.v ); 
  a007_3.v  = _mm512_fmadd_pd( a007_3.v, c007_3.v, b_tmp.v ); 
  a007_4.v  = _mm512_fmadd_pd( a007_4.v, c007_4.v, b_tmp.v ); 
  a007_5.v  = _mm512_fmadd_pd( a007_5.v, c007_5.v, b_tmp.v ); 
  a007_6.v  = _mm512_fmadd_pd( a007_6.v, c007_6.v, b_tmp.v ); 
  a007_7.v  = _mm512_fmadd_pd( a007_7.v, c007_7.v, b_tmp.v ); 

  a815_0.v  = _mm512_fmadd_pd( a815_0.v, c815_0.v, b_tmp.v ); 
  a815_1.v  = _mm512_fmadd_pd( a815_1.v, c815_1.v, b_tmp.v ); 
  a815_2.v  = _mm512_fmadd_pd( a815_2.v, c815_2.v, b_tmp.v ); 
  a815_3.v  = _mm512_fmadd_pd( a815_3.v, c815_3.v, b_tmp.v ); 
  a815_4.v  = _mm512_fmadd_pd( a815_4.v, c815_4.v, b_tmp.v ); 
  a815_5.v  = _mm512_fmadd_pd( a815_5.v, c815_5.v, b_tmp.v ); 
  a815_6.v  = _mm512_fmadd_pd( a815_6.v, c815_6.v, b_tmp.v ); 
  a815_7.v  = _mm512_fmadd_pd( a815_7.v, c815_7.v, b_tmp.v ); 

  b_tmp.v   = _mm512_extload_pd( &w1, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  a007_0.v  = _mm512_fmadd_pd( a007_0.v, c007_0.v, b_tmp.v ); 
  a007_1.v  = _mm512_fmadd_pd( a007_1.v, c007_1.v, b_tmp.v ); 
  a007_2.v  = _mm512_fmadd_pd( a007_2.v, c007_2.v, b_tmp.v ); 
  a007_3.v  = _mm512_fmadd_pd( a007_3.v, c007_3.v, b_tmp.v ); 
  a007_4.v  = _mm512_fmadd_pd( a007_4.v, c007_4.v, b_tmp.v ); 
  a007_5.v  = _mm512_fmadd_pd( a007_5.v, c007_5.v, b_tmp.v ); 
  a007_6.v  = _mm512_fmadd_pd( a007_6.v, c007_6.v, b_tmp.v ); 
  a007_7.v  = _mm512_fmadd_pd( a007_7.v, c007_7.v, b_tmp.v ); 

  a815_0.v  = _mm512_fmadd_pd( a815_0.v, c815_0.v, b_tmp.v ); 
  a815_1.v  = _mm512_fmadd_pd( a815_1.v, c815_1.v, b_tmp.v ); 
  a815_2.v  = _mm512_fmadd_pd( a815_2.v, c815_2.v, b_tmp.v ); 
  a815_3.v  = _mm512_fmadd_pd( a815_3.v, c815_3.v, b_tmp.v ); 
  a815_4.v  = _mm512_fmadd_pd( a815_4.v, c815_4.v, b_tmp.v ); 
  a815_5.v  = _mm512_fmadd_pd( a815_5.v, c815_5.v, b_tmp.v ); 
  a815_6.v  = _mm512_fmadd_pd( a815_6.v, c815_6.v, b_tmp.v ); 
  a815_7.v  = _mm512_fmadd_pd( a815_7.v, c815_7.v, b_tmp.v ); 

  b_tmp.v   = _mm512_extload_pd( &w0, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  a007_0.v  = _mm512_fmadd_pd( a007_0.v, c007_0.v, b_tmp.v ); 
  a007_1.v  = _mm512_fmadd_pd( a007_1.v, c007_1.v, b_tmp.v ); 
  a007_2.v  = _mm512_fmadd_pd( a007_2.v, c007_2.v, b_tmp.v ); 
  a007_3.v  = _mm512_fmadd_pd( a007_3.v, c007_3.v, b_tmp.v ); 
  a007_4.v  = _mm512_fmadd_pd( a007_4.v, c007_4.v, b_tmp.v ); 
  a007_5.v  = _mm512_fmadd_pd( a007_5.v, c007_5.v, b_tmp.v ); 
  a007_6.v  = _mm512_fmadd_pd( a007_6.v, c007_6.v, b_tmp.v ); 
  a007_7.v  = _mm512_fmadd_pd( a007_7.v, c007_7.v, b_tmp.v ); 

  a815_0.v  = _mm512_fmadd_pd( a815_0.v, c815_0.v, b_tmp.v ); 
  a815_1.v  = _mm512_fmadd_pd( a815_1.v, c815_1.v, b_tmp.v ); 
  a815_2.v  = _mm512_fmadd_pd( a815_2.v, c815_2.v, b_tmp.v ); 
  a815_3.v  = _mm512_fmadd_pd( a815_3.v, c815_3.v, b_tmp.v ); 
  a815_4.v  = _mm512_fmadd_pd( a815_4.v, c815_4.v, b_tmp.v ); 
  a815_5.v  = _mm512_fmadd_pd( a815_5.v, c815_5.v, b_tmp.v ); 
  a815_6.v  = _mm512_fmadd_pd( a815_6.v, c815_6.v, b_tmp.v ); 
  a815_7.v  = _mm512_fmadd_pd( a815_7.v, c815_7.v, b_tmp.v ); 


//  printf( "a poly:\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", 
//	  a007_0.d[ 0 ], a007_0.d[ 1 ], a007_0.d[ 2 ],  a007_0.d[ 3 ], 
//	  a007_0.d[ 4 ], a007_0.d[ 5 ], a007_0.d[ 6 ],  a007_0.d[ 7 ] );


  __asm__ volatile( "vprefetch0 0(%0)      \n\t" : :"r"(u) );
  __asm__ volatile( "vprefetch0 64(%0)      \n\t" : :"r"(u) );


  // Shift p to the exponant
  iperm.v   = _mm512_load_epi32( mask );


//  printf( "k-lo\n" );
//  printf( "%d, %d, %d, %d, %d, %d, %d, %d\n",
//    k007_0.d[ 0 ], k007_0.d[ 1 ], k007_0.d[ 2 ], k007_0.d[ 3 ], 
//	k007_0.d[ 4 ], k007_0.d[ 5 ], k007_0.d[ 6 ], k007_0.d[ 7 ] );
//
//  printf( "k-hi\n" );
//  printf( "%d, %d, %d, %d, %d, %d, %d, %d\n",
//    k007_0.d[ 8 ], k007_0.d[ 9 ], k007_0.d[ 10 ], k007_0.d[ 11 ], 
//	k007_0.d[ 12 ], k007_0.d[ 13 ], k007_0.d[ 14 ], k007_0.d[ 15 ] ); 
//
//  printf( "mask\n" );
//  printf( "%d, %d, %d, %d, %d, %d, %d, %d\n",
//	  iperm.d[ 0 ], iperm.d[ 2 ], iperm.d[ 4 ], iperm.d[ 6 ],
//	  iperm.d[ 8 ], iperm.d[ 10 ], iperm.d[ 12 ], iperm.d[ 14 ] );
//
//  k007_0.v  = _mm512_permutevar_epi32( iperm.v, k007_0.v );
//
//  printf( "k-lo\n" );
//  printf( "%d, %d, %d, %d, %d, %d, %d, %d\n",
//    k007_0.d[ 0 ], k007_0.d[ 1 ], k007_0.d[ 2 ], k007_0.d[ 3 ], 
//	k007_0.d[ 4 ], k007_0.d[ 5 ], k007_0.d[ 6 ], k007_0.d[ 7 ] );
//
//  printf( "k-hi\n" );
//  printf( "%d, %d, %d, %d, %d, %d, %d, %d\n",
//    k007_0.d[ 8 ], k007_0.d[ 9 ], k007_0.d[ 10 ], k007_0.d[ 11 ], 
//	k007_0.d[ 12 ], k007_0.d[ 13 ], k007_0.d[ 14 ], k007_0.d[ 15 ] ); 


  __asm__ volatile( "vprefetch0 0(%0)      \n\t" : :"r"(w) );

  k007_0.v  = _mm512_add_epi32( iperm.v, k007_0.v );
  k007_1.v  = _mm512_add_epi32( iperm.v, k007_1.v );
  k007_2.v  = _mm512_add_epi32( iperm.v, k007_2.v );
  k007_3.v  = _mm512_add_epi32( iperm.v, k007_3.v );
  k007_4.v  = _mm512_add_epi32( iperm.v, k007_4.v );
  k007_5.v  = _mm512_add_epi32( iperm.v, k007_5.v );
  k007_6.v  = _mm512_add_epi32( iperm.v, k007_6.v );
  k007_7.v  = _mm512_add_epi32( iperm.v, k007_7.v );

  k815_0.v  = _mm512_add_epi32( iperm.v, k815_0.v );
  k815_1.v  = _mm512_add_epi32( iperm.v, k815_1.v );
  k815_2.v  = _mm512_add_epi32( iperm.v, k815_2.v );
  k815_3.v  = _mm512_add_epi32( iperm.v, k815_3.v );
  k815_4.v  = _mm512_add_epi32( iperm.v, k815_4.v );
  k815_5.v  = _mm512_add_epi32( iperm.v, k815_5.v );
  k815_6.v  = _mm512_add_epi32( iperm.v, k815_6.v );
  k815_7.v  = _mm512_add_epi32( iperm.v, k815_7.v );

  iperm.v   = _mm512_load_epi32( perm );
  k007_0.v  = _mm512_permutevar_epi32( iperm.v, k007_0.v );
  k007_1.v  = _mm512_permutevar_epi32( iperm.v, k007_1.v );
  k007_2.v  = _mm512_permutevar_epi32( iperm.v, k007_2.v );
  k007_3.v  = _mm512_permutevar_epi32( iperm.v, k007_3.v );
  k007_4.v  = _mm512_permutevar_epi32( iperm.v, k007_4.v );
  k007_5.v  = _mm512_permutevar_epi32( iperm.v, k007_5.v );
  k007_6.v  = _mm512_permutevar_epi32( iperm.v, k007_6.v );
  k007_7.v  = _mm512_permutevar_epi32( iperm.v, k007_7.v );

  k815_0.v  = _mm512_permutevar_epi32( iperm.v, k815_0.v );
  k815_1.v  = _mm512_permutevar_epi32( iperm.v, k815_1.v );
  k815_2.v  = _mm512_permutevar_epi32( iperm.v, k815_2.v );
  k815_3.v  = _mm512_permutevar_epi32( iperm.v, k815_3.v );
  k815_4.v  = _mm512_permutevar_epi32( iperm.v, k815_4.v );
  k815_5.v  = _mm512_permutevar_epi32( iperm.v, k815_5.v );
  k815_6.v  = _mm512_permutevar_epi32( iperm.v, k815_6.v );
  k815_7.v  = _mm512_permutevar_epi32( iperm.v, k815_7.v );

  k007_0.v  = _mm512_slli_epi32( k007_0.v, 20 );
  k007_1.v  = _mm512_slli_epi32( k007_1.v, 20 );
  k007_2.v  = _mm512_slli_epi32( k007_2.v, 20 );
  k007_3.v  = _mm512_slli_epi32( k007_3.v, 20 );
  k007_4.v  = _mm512_slli_epi32( k007_4.v, 20 );
  k007_5.v  = _mm512_slli_epi32( k007_5.v, 20 );
  k007_6.v  = _mm512_slli_epi32( k007_6.v, 20 );
  k007_7.v  = _mm512_slli_epi32( k007_7.v, 20 );

  k815_0.v  = _mm512_slli_epi32( k815_0.v, 20 );
  k815_1.v  = _mm512_slli_epi32( k815_1.v, 20 );
  k815_2.v  = _mm512_slli_epi32( k815_2.v, 20 );
  k815_3.v  = _mm512_slli_epi32( k815_3.v, 20 );
  k815_4.v  = _mm512_slli_epi32( k815_4.v, 20 );
  k815_5.v  = _mm512_slli_epi32( k815_5.v, 20 );
  k815_6.v  = _mm512_slli_epi32( k815_6.v, 20 );
  k815_7.v  = _mm512_slli_epi32( k815_7.v, 20 );


//  printf( "k-lo\n" );
//  printf( "%d, %d, %d, %d, %d, %d, %d, %d\n",
//    k007_0.d[ 0 ], k007_0.d[ 1 ], k007_0.d[ 2 ], k007_0.d[ 3 ], 
//	k007_0.d[ 4 ], k007_0.d[ 5 ], k007_0.d[ 6 ], k007_0.d[ 7 ] );
//
//  printf( "k-hi\n" );
//  printf( "%d, %d, %d, %d, %d, %d, %d, %d\n",
//    k007_0.d[ 8 ], k007_0.d[ 9 ], k007_0.d[ 10 ], k007_0.d[ 11 ], 
//	k007_0.d[ 12 ], k007_0.d[ 13 ], k007_0.d[ 14 ], k007_0.d[ 15 ] ); 



  p007_0.v  = _mm512_castsi512_pd( k007_0.v );
  p007_1.v  = _mm512_castsi512_pd( k007_1.v );
  p007_2.v  = _mm512_castsi512_pd( k007_2.v );
  p007_3.v  = _mm512_castsi512_pd( k007_3.v );
  p007_4.v  = _mm512_castsi512_pd( k007_4.v );
  p007_5.v  = _mm512_castsi512_pd( k007_5.v );
  p007_6.v  = _mm512_castsi512_pd( k007_6.v );
  p007_7.v  = _mm512_castsi512_pd( k007_7.v );

  p815_0.v  = _mm512_castsi512_pd( k815_0.v );
  p815_1.v  = _mm512_castsi512_pd( k815_1.v );
  p815_2.v  = _mm512_castsi512_pd( k815_2.v );
  p815_3.v  = _mm512_castsi512_pd( k815_3.v );
  p815_4.v  = _mm512_castsi512_pd( k815_4.v );
  p815_5.v  = _mm512_castsi512_pd( k815_5.v );
  p815_6.v  = _mm512_castsi512_pd( k815_6.v );
  p815_7.v  = _mm512_castsi512_pd( k815_7.v );

  // Preload u007
  a007.v    = _mm512_load_pd( u );

  c007_0.v  = _mm512_mul_pd( p007_0.v, a007_0.v );
  c007_1.v  = _mm512_mul_pd( p007_1.v, a007_1.v );
  c007_2.v  = _mm512_mul_pd( p007_2.v, a007_2.v );
  c007_3.v  = _mm512_mul_pd( p007_3.v, a007_3.v );
  c007_4.v  = _mm512_mul_pd( p007_4.v, a007_4.v );
  c007_5.v  = _mm512_mul_pd( p007_5.v, a007_5.v );
  c007_6.v  = _mm512_mul_pd( p007_6.v, a007_6.v );
  c007_7.v  = _mm512_mul_pd( p007_7.v, a007_7.v );

  // Preload u815
  a815.v    = _mm512_load_pd( u + 8 );

  c815_0.v  = _mm512_mul_pd( p815_0.v, a815_0.v );
  c815_1.v  = _mm512_mul_pd( p815_1.v, a815_1.v );
  c815_2.v  = _mm512_mul_pd( p815_2.v, a815_2.v );
  c815_3.v  = _mm512_mul_pd( p815_3.v, a815_3.v );
  c815_4.v  = _mm512_mul_pd( p815_4.v, a815_4.v );
  c815_5.v  = _mm512_mul_pd( p815_5.v, a815_5.v );
  c815_6.v  = _mm512_mul_pd( p815_6.v, a815_6.v );
  c815_7.v  = _mm512_mul_pd( p815_7.v, a815_7.v );


//  printf( "u007\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
//    u[ 0 ], u[ 1 ], u[ 2 ], u[ 3 ], 
//	u[ 4 ], u[ 5 ], u[ 6 ], u[ 7 ] );
//
//  printf( "w007\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
//    w[ 0 ], w[ 1 ], w[ 2 ], w[ 3 ], 
//	w[ 4 ], w[ 5 ], w[ 6 ], w[ 7 ] );
//
//
//  printf( "c007_0\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
//    c007_0.d[ 0 ], c007_0.d[ 1 ], c007_0.d[ 2 ], c007_0.d[ 3 ], 
//	c007_0.d[ 4 ], c007_0.d[ 5 ], c007_0.d[ 6 ], c007_0.d[ 7 ] );
//
//  printf( "c007_1\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
//    c007_1.d[ 0 ], c007_1.d[ 1 ], c007_1.d[ 2 ], c007_1.d[ 3 ], 
//	c007_1.d[ 4 ], c007_1.d[ 5 ], c007_1.d[ 6 ], c007_1.d[ 7 ] );
//
//  printf( "c007_2\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
//    c007_2.d[ 0 ], c007_2.d[ 1 ], c007_2.d[ 2 ], c007_2.d[ 3 ], 
//	c007_2.d[ 4 ], c007_2.d[ 5 ], c007_2.d[ 6 ], c007_2.d[ 7 ] );
//
//  printf( "c007_3\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
//    c007_3.d[ 0 ], c007_3.d[ 1 ], c007_3.d[ 2 ], c007_3.d[ 3 ], 
//	c007_3.d[ 4 ], c007_3.d[ 5 ], c007_3.d[ 6 ], c007_3.d[ 7 ] );
//
//  printf( "c007_4\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
//    c007_4.d[ 0 ], c007_4.d[ 1 ], c007_4.d[ 2 ], c007_4.d[ 3 ], 
//	c007_4.d[ 4 ], c007_4.d[ 5 ], c007_4.d[ 6 ], c007_4.d[ 7 ] );
//
//  printf( "c007_5\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
//    c007_5.d[ 0 ], c007_5.d[ 1 ], c007_5.d[ 2 ], c007_5.d[ 3 ], 
//	c007_5.d[ 4 ], c007_5.d[ 5 ], c007_5.d[ 6 ], c007_5.d[ 7 ] );
//
//  printf( "c007_6\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
//    c007_6.d[ 0 ], c007_6.d[ 1 ], c007_6.d[ 2 ], c007_6.d[ 3 ], 
//	c007_6.d[ 4 ], c007_6.d[ 5 ], c007_6.d[ 6 ], c007_6.d[ 7 ] );
//
//  printf( "c007_7\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
//    c007_7.d[ 0 ], c007_7.d[ 1 ], c007_7.d[ 2 ], c007_7.d[ 3 ], 
//	c007_7.d[ 4 ], c007_7.d[ 5 ], c007_7.d[ 6 ], c007_7.d[ 7 ] );
//
//
//  printf( "c815_0\n" );
//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
//    c815_0.d[ 0 ], c815_0.d[ 1 ], c815_0.d[ 2 ], c815_0.d[ 3 ], 
//	c815_0.d[ 4 ], c815_0.d[ 5 ], c815_0.d[ 6 ], c815_0.d[ 7 ] );
//

  // gemv() with fma
  b_tmp.v   = _mm512_extload_pd( w, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  a007.v    = _mm512_fmadd_pd( b_tmp.v, c007_0.v, a007.v ); 
  a815.v    = _mm512_fmadd_pd( b_tmp.v, c815_0.v, a815.v ); 

  b_tmp.v   = _mm512_extload_pd( w + 1, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  a007.v    = _mm512_fmadd_pd( b_tmp.v, c007_1.v, a007.v ); 
  a815.v    = _mm512_fmadd_pd( b_tmp.v, c815_1.v, a815.v ); 

  b_tmp.v   = _mm512_extload_pd( w + 2, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  a007.v    = _mm512_fmadd_pd( b_tmp.v, c007_2.v, a007.v ); 
  a815.v    = _mm512_fmadd_pd( b_tmp.v, c815_2.v, a815.v ); 

  b_tmp.v   = _mm512_extload_pd( w + 3, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  a007.v    = _mm512_fmadd_pd( b_tmp.v, c007_3.v, a007.v ); 
  a815.v    = _mm512_fmadd_pd( b_tmp.v, c815_3.v, a815.v ); 

  b_tmp.v   = _mm512_extload_pd( w + 4, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  a007.v    = _mm512_fmadd_pd( b_tmp.v, c007_4.v, a007.v ); 
  a815.v    = _mm512_fmadd_pd( b_tmp.v, c815_4.v, a815.v ); 

  b_tmp.v   = _mm512_extload_pd( w + 5, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  a007.v    = _mm512_fmadd_pd( b_tmp.v, c007_5.v, a007.v ); 
  a815.v    = _mm512_fmadd_pd( b_tmp.v, c815_5.v, a815.v ); 

  b_tmp.v   = _mm512_extload_pd( w + 6, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  a007.v    = _mm512_fmadd_pd( b_tmp.v, c007_6.v, a007.v ); 
  a815.v    = _mm512_fmadd_pd( b_tmp.v, c815_6.v, a815.v ); 

  b_tmp.v   = _mm512_extload_pd( w + 7, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
  a007.v    = _mm512_fmadd_pd( b_tmp.v, c007_7.v, a007.v ); 
  a815.v    = _mm512_fmadd_pd( b_tmp.v, c815_7.v, a815.v ); 






//  if ( aux->pc != 0 ) {
//
//    // packed
//    tmpc03_0.v = _mm256_load_pd( (double*)( c      ) );
//    tmpc47_0.v = _mm256_load_pd( (double*)( c + 4  ) );
//
//    tmpc03_1.v = _mm256_load_pd( (double*)( c + 8  ) );
//    tmpc47_1.v = _mm256_load_pd( (double*)( c + 12 ) );
//
//    tmpc03_2.v = _mm256_load_pd( (double*)( c + 16 ) );
//    tmpc47_2.v = _mm256_load_pd( (double*)( c + 20 ) );
//
//    tmpc03_3.v = _mm256_load_pd( (double*)( c + 24 ) );
//    tmpc47_3.v = _mm256_load_pd( (double*)( c + 28 ) );
//    
//
//    c03_0.v    = _mm256_add_pd( tmpc03_0.v, c03_0.v );
//    c47_0.v    = _mm256_add_pd( tmpc47_0.v, c47_0.v );
//
//    c03_1.v    = _mm256_add_pd( tmpc03_1.v, c03_1.v );
//    c47_1.v    = _mm256_add_pd( tmpc47_1.v, c47_1.v );
//
//    c03_2.v    = _mm256_add_pd( tmpc03_2.v, c03_2.v );
//    c47_2.v    = _mm256_add_pd( tmpc47_2.v, c47_2.v );
//
//    c03_3.v    = _mm256_add_pd( tmpc03_3.v, c03_3.v );
//    c47_3.v    = _mm256_add_pd( tmpc47_3.v, c47_3.v );
//  }
//
//
  // packed
  _mm512_store_pd( u     , a007.v );
  _mm512_store_pd( u + 8 , a815.v );





}
