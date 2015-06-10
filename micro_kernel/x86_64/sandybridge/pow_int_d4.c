#include <immintrin.h> // AVX
#include <ks.h>


// IEEE-754 double
// 0      7 8     15 16    23 24    31 32    39 40    47 48    55 56    63
// SEEEEEEE EEEEMMMM MMMMMMMM MMMMMMMM MMMMMMMM MMMMMMMM MMMMMMMM MMMMMMMM
//
// Exponent Max ( Exponent Mask ) 0x 7F F0 00 00 00 00 00 00
// 01111111 11110000 00000000 00000000 00000000 00000000 00000000 00000000
//
// Mantissa Max ( Mantissa Mask ) 0x 00 0F FF FF FF FF FF FF
// 00000000 00001111 11111111 11111111 11111111 11111111 11111111 11111111
//
// Extract exponent from x
// maske = _mm256_set1_pd( 0x7FF0000000000000 );
// i = _mm256_castpd_si256( x )
// e = _mm256_and_pf
//
//
void pow_int_d4(
    double *x,
    double *y
    )
{
  v4df_t c03_0;
  v4df_t a03_0;
  v4df_t p03_0;
  v4df_t maske, maskm, mask2;

  x[ 0 ] = 2.0;
  x[ 1 ] = 5.0;
  x[ 2 ] = 0.5;
  x[ 3 ] = 1.36;

  printf( "original:\n");
  printf( "%lf, %lf, %lf, %lf\n", x[ 0 ], x[ 1 ], x[ 2 ], x[ 3 ] );

  c03_0.v   = _mm256_load_pd( x );
  //p03_0.v   = _mm256_load_pd( y );

  printf( "%lx, %lx, %lx, %lx\n", c03_0.u[ 0 ], c03_0.u[ 1 ], c03_0.u[ 2 ], c03_0.u[ 3 ] );

  // Setup exponent and mantissa masks
  maske.i   = _mm256_set1_epi64x( 0x7FF0000000000000 );
  maskm.i   = _mm256_set1_epi64x( 0x000FFFFFFFFFFFFF );
  mask2.i   = _mm256_set1_epi64x( 0x3FFFFFFFFFFFFFFF );

  c03_0.v   = _mm256_or_pd( c03_0.v, _mm256_castsi256_pd( maske.i ) );
  c03_0.v   = _mm256_and_pd( c03_0.v, _mm256_castsi256_pd( mask2.i ) );




  p03_0.v   = _mm256_and_pd( c03_0.v, _mm256_castsi256_pd( maske.i ) );



  _mm256_store_pd( (double*)x, c03_0.v );
  _mm256_store_pd( (double*)y, p03_0.v );

  printf( "maske: %lx\n", 0x7FF0000000000000 );
  printf( "maske: %lx\n", maske.u[ 0 ] );
  printf( "maskm: %lx\n", maskm.u[ 0 ] );


  printf( "extract mantissa:\n");
  printf( "%lf, %lf, %lf, %lf\n", x[ 0 ], x[ 1 ], x[ 2 ], x[ 3 ] );

  printf( "%lx, %lx, %lx, %lx\n", c03_0.u[ 0 ], c03_0.u[ 1 ], c03_0.u[ 2 ], c03_0.u[ 3 ] );

  printf( "extract exponent:\n");
  printf( "%lf, %lf, %lf, %lf\n", y[ 0 ], y[ 1 ], y[ 2 ], y[ 3 ] );

  printf( "%lx, %lx, %lx, %lx\n", p03_0.u[ 0 ], p03_0.u[ 1 ], p03_0.u[ 2 ], p03_0.u[ 3 ] );
}
