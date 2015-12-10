#include <math.h>
#include <immintrin.h> // AVX
#include <ks.h>
#include <avx_type.h>

void gaussian_int_d8x6(
    int    k,
    int    rhs,
    double *h,
    double *u,
    double *aa,
    double *a,
    double *bb,
    double *b,
    double *w,
    double *c,
    ks_t   *ker,
    aux_t  *aux
    )
{
  int    i, j, p;
  double K[ 8 * 6 ] = {{ 0.0 }};

  #include <rank_k_int_d8x6.h>

  // Gaussian kernel
  for ( j = 0; j < 6; j ++ ) {
	for ( i = 0; i < 8; i ++ ) { 
	  K[ j * 8 + i ] = aa[ i ] - 2.0 * K[ j * 8 + i ] + bb[ j ];
      K[ j * 8 + i ] = exp( ker->scal * K[ j * 8 + i ] );
	  u[ i ] += K[ j * 8 + i ] * w[ j ];
	}
  }
}
