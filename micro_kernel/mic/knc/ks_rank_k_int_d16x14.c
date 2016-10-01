#include <immintrin.h> // AVX
#include <ks.h>


void ks_rank_k_int_d16x14(
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


  v8df_t c007_0,  c007_1,  c007_2,  c007_3,  c007_4;
  v8df_t c007_5,  c007_6,  c007_7,  c007_8,  c007_9;
  v8df_t c007_10, c007_11, c007_12, c007_13;

  v8df_t c815_0,  c815_1,  c815_2,  c815_3,  c815_4;
  v8df_t c815_5,  c815_6,  c815_7,  c815_8,  c815_9;
  v8df_t c815_10, c815_11, c815_12, c815_13;


  v8df_t a007, a815, b_tmp;

  int k_iter = k;


  // TODO: need to clean the c buffer.




  for ( i = 0; i < k_iter; ++ i ) { 
	a007.v    = _mm512_load_pd( a );
	a815.v    = _mm512_load_pd( a + 8 );

	//printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", 
	//	a007.d[ 0 ], a007.d[ 1 ], a007.d[ 2 ],  a007.d[ 3 ], 
	//	a007.d[ 4 ], a007.d[ 5 ], a007.d[ 6 ],  a007.d[ 7 ] );

	b_tmp.v   = _mm512_extload_pd( b, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );

    //printf( "b[ 0 ] = %lf\n", b[ 0 ] );

	//printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", 
	//	b_tmp.d[ 0 ], b_tmp.d[ 1 ], b_tmp.d[ 2 ],  b_tmp.d[ 3 ], 
	//	b_tmp.d[ 4 ], b_tmp.d[ 5 ], b_tmp.d[ 6 ],  b_tmp.d[ 7 ] );



	c007_0.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_0.v ); 
	c815_0.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_0.v ); 
	
	b_tmp.v   = _mm512_extload_pd( b + 1, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_1.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_1.v ); 
	c815_1.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_1.v ); 

	b_tmp.v   = _mm512_extload_pd( b + 2, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_2.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_2.v ); 
	c815_2.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_2.v ); 

	b_tmp.v   = _mm512_extload_pd( b + 3, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_3.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_3.v ); 
	c815_3.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_3.v ); 

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

	b_tmp.v   = _mm512_extload_pd( b + 8, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_8.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_8.v ); 
	c815_8.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_8.v ); 

	b_tmp.v   = _mm512_extload_pd( b + 9, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_9.v  = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_9.v ); 
	c815_9.v  = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_9.v ); 

	b_tmp.v   = _mm512_extload_pd( b + 10, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_10.v = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_10.v ); 
	c815_10.v = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_10.v ); 

	b_tmp.v   = _mm512_extload_pd( b + 11, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_11.v = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_11.v ); 
	c815_11.v = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_11.v ); 

	b_tmp.v   = _mm512_extload_pd( b + 12, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_12.v = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_12.v ); 
	c815_12.v = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_12.v ); 

	b_tmp.v   = _mm512_extload_pd( b + 13, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0 );
	c007_13.v = _mm512_fmadd_pd( a007.v, b_tmp.v, c007_13.v ); 
	c815_13.v = _mm512_fmadd_pd( a815.v, b_tmp.v, c815_13.v );

	a += 16;
	b += 16;
  }
 

  // simulate kernel summation
  c007_0.v  = _mm512_add_pd( c007_0.v, c007_1.v ); 
  c815_0.v  = _mm512_add_pd( c815_0.v, c815_1.v ); 

  c007_0.v  = _mm512_add_pd( c007_0.v, c007_2.v ); 
  c815_0.v  = _mm512_add_pd( c815_0.v, c815_2.v ); 

  c007_0.v  = _mm512_add_pd( c007_0.v, c007_3.v ); 
  c815_0.v  = _mm512_add_pd( c815_0.v, c815_3.v ); 

  c007_0.v  = _mm512_add_pd( c007_0.v, c007_4.v ); 
  c815_0.v  = _mm512_add_pd( c815_0.v, c815_4.v ); 

  c007_0.v  = _mm512_add_pd( c007_0.v, c007_5.v ); 
  c815_0.v  = _mm512_add_pd( c815_0.v, c815_5.v ); 

  c007_0.v  = _mm512_add_pd( c007_0.v, c007_6.v ); 
  c815_0.v  = _mm512_add_pd( c815_0.v, c815_6.v ); 

  c007_0.v  = _mm512_add_pd( c007_0.v, c007_7.v ); 
  c815_0.v  = _mm512_add_pd( c815_0.v, c815_7.v ); 

  c007_0.v  = _mm512_add_pd( c007_0.v, c007_8.v ); 
  c815_0.v  = _mm512_add_pd( c815_0.v, c815_8.v ); 

  c007_0.v  = _mm512_add_pd( c007_0.v, c007_9.v ); 
  c815_0.v  = _mm512_add_pd( c815_0.v, c815_9.v ); 

  c007_0.v  = _mm512_add_pd( c007_0.v, c007_10.v ); 
  c815_0.v  = _mm512_add_pd( c815_0.v, c815_10.v ); 

  c007_0.v  = _mm512_add_pd( c007_0.v, c007_11.v ); 
  c815_0.v  = _mm512_add_pd( c815_0.v, c815_11.v ); 

  c007_0.v  = _mm512_add_pd( c007_0.v, c007_12.v ); 
  c815_0.v  = _mm512_add_pd( c815_0.v, c815_12.v ); 

  c007_0.v  = _mm512_add_pd( c007_0.v, c007_13.v ); 
  c815_0.v  = _mm512_add_pd( c815_0.v, c815_13.v ); 










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
  _mm512_store_pd( c     , c007_0.v );
  _mm512_store_pd( c + 8 , c815_0.v );

//  _mm512_store_pd( c + 16, c007_1.v );
//  _mm512_store_pd( c + 24, c815_1.v );
//
//  _mm512_store_pd( c + 32, c007_2.v );
//  _mm512_store_pd( c + 40, c815_2.v );
//
//  _mm512_store_pd( c + 48, c007_3.v );
//  _mm512_store_pd( c + 56, c815_3.v );
//
//  _mm512_store_pd( c + 64, c007_4.v );
//  _mm512_store_pd( c + 72, c815_4.v );
//
//  _mm512_store_pd( c + 80, c007_5.v );
//  _mm512_store_pd( c + 88, c815_5.v );
//
//  _mm512_store_pd( c + 96, c007_6.v );
//  _mm512_store_pd( c + 104, c815_6.v );
//
//  _mm512_store_pd( c + 112, c007_7.v );
//  _mm512_store_pd( c + 120, c815_7.v );
//
//  _mm512_store_pd( c + 128, c007_8.v );
//  _mm512_store_pd( c + 136, c815_8.v );
//
//  _mm512_store_pd( c + 144, c007_9.v );
//  _mm512_store_pd( c + 152, c815_9.v );
//
//  _mm512_store_pd( c + 160, c007_10.v );
//  _mm512_store_pd( c + 168, c815_10.v );
//
//  _mm512_store_pd( c + 176, c007_11.v );
//  _mm512_store_pd( c + 184, c815_11.v );
//
//  _mm512_store_pd( c + 192, c007_12.v );
//  _mm512_store_pd( c + 200, c815_12.v );
//
//  _mm512_store_pd( c + 208, c007_13.v );
//  _mm512_store_pd( c + 216, c815_13.v );



  //printf( "ldc = %d\n", ldc );
  

//  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", 
//	  c007_0.d[ 0 ], c007_0.d[ 1 ], c007_0.d[ 2 ],  c007_0.d[ 3 ], 
//	  c007_0.d[ 4 ], c007_0.d[ 5 ], c007_0.d[ 6 ],  c007_0.d[ 7 ] );
//

  //printf( "%lf, %lf, %lf, %lf\n", c[1], c[ ldc + 1], c[ ldc * 2 + 1], c[ ldc * 3 + 1] );
  //printf( "%lf, %lf, %lf, %lf\n", c[2], c[ ldc + 2], c[ ldc * 2 + 2], c[ ldc * 3 + 2] );
  //printf( "%lf, %lf, %lf, %lf\n", c[3], c[ ldc + 3], c[ ldc * 2 + 3], c[ ldc * 3 + 3] );
  //printf( "%lf, %lf, %lf, %lf\n", c[4], c[ ldc + 4], c[ ldc * 2 + 4], c[ ldc * 3 + 4] );
  //printf( "%lf, %lf, %lf, %lf\n", c[5], c[ ldc + 5], c[ ldc * 2 + 5], c[ ldc * 3 + 5] );
  //printf( "%lf, %lf, %lf, %lf\n", c[6], c[ ldc + 6], c[ ldc * 2 + 6], c[ ldc * 3 + 6] );
  //printf( "%lf, %lf, %lf, %lf\n", c[7], c[ ldc + 7], c[ ldc * 2 + 7], c[ ldc * 3 + 7] );
}
