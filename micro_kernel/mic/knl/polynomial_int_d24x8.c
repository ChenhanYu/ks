#include <math.h>
#include <immintrin.h> // AVX
#include <ks.h>
#include <gsks_internal.h>
#include <avx_type.h>

void polynomial_int_s16x6(
    int    k,
    int    rhs,
    //float  *h,
    float  *u,
    float  *aa,
    float  *a,
    float  *bb,
    float  *b,
    float  *w,
    float  *c,
    ks_t   *ker,
    aux_t  *aux
    )
{
  printf( "polynomial_int_s16x6 not yet implemented.\n" );
}

void polynomial_int_d24x8(
    int    k,
    int    rhs,
    //double *h,
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
  int    i;
  double powe  = ker->powe;
  double scal  = ker->scal;
  double cons  = ker->cons;

  // 24 avx512 registers
  v8df_t c07_0, c07_1, c07_2, c07_3, c07_4, c07_5, c07_6, c07_7;
  v8df_t c15_0, c15_1, c15_2, c15_3, c15_4, c15_5, c15_6, c15_7;
  v8df_t c23_0, c23_1, c23_2, c23_3, c23_4, c23_5, c23_6, c23_7;

  // 8 avx512 registers
  v8df_t a07, a15, a23;
  v8df_t A07, A15, A23;
  v8df_t b0, b1;

  #include <rank_k_int_d24x8.h>




  // Prefetch u, w
  //__asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u ) );
  //__asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( w ) );

  // c = c * scal
  a07.v   = _mm512_set1_pd( scal );
  c07_0.v = _mm512_mul_pd( a07.v, c07_0.v );
  c07_1.v = _mm512_mul_pd( a07.v, c07_1.v );
  c07_2.v = _mm512_mul_pd( a07.v, c07_2.v );
  c07_3.v = _mm512_mul_pd( a07.v, c07_3.v );
  c07_4.v = _mm512_mul_pd( a07.v, c07_4.v );
  c07_5.v = _mm512_mul_pd( a07.v, c07_5.v );
  c07_6.v = _mm512_mul_pd( a07.v, c07_6.v );
  c07_7.v = _mm512_mul_pd( a07.v, c07_7.v );

  c15_0.v = _mm512_mul_pd( a07.v, c15_0.v );
  c15_1.v = _mm512_mul_pd( a07.v, c15_1.v );
  c15_2.v = _mm512_mul_pd( a07.v, c15_2.v );
  c15_3.v = _mm512_mul_pd( a07.v, c15_3.v );
  c15_4.v = _mm512_mul_pd( a07.v, c15_4.v );
  c15_5.v = _mm512_mul_pd( a07.v, c15_5.v );
  c15_6.v = _mm512_mul_pd( a07.v, c15_6.v );
  c15_7.v = _mm512_mul_pd( a07.v, c15_7.v );

  c23_0.v = _mm512_mul_pd( a07.v, c23_0.v );
  c23_1.v = _mm512_mul_pd( a07.v, c23_1.v );
  c23_2.v = _mm512_mul_pd( a07.v, c23_2.v );
  c23_3.v = _mm512_mul_pd( a07.v, c23_3.v );
  c23_4.v = _mm512_mul_pd( a07.v, c23_4.v );
  c23_5.v = _mm512_mul_pd( a07.v, c23_5.v );
  c23_6.v = _mm512_mul_pd( a07.v, c23_6.v );
  c23_7.v = _mm512_mul_pd( a07.v, c23_7.v );


  // c = c + cons
  a07.v   = _mm512_set1_pd( cons );
  c07_0.v = _mm512_add_pd( a07.v, c07_0.v );
  c07_1.v = _mm512_add_pd( a07.v, c07_1.v );
  c07_2.v = _mm512_add_pd( a07.v, c07_2.v );
  c07_3.v = _mm512_add_pd( a07.v, c07_3.v );
  c07_4.v = _mm512_add_pd( a07.v, c07_4.v );
  c07_5.v = _mm512_add_pd( a07.v, c07_5.v );
  c07_6.v = _mm512_add_pd( a07.v, c07_6.v );
  c07_7.v = _mm512_add_pd( a07.v, c07_7.v );

  c15_0.v = _mm512_add_pd( a07.v, c15_0.v );
  c15_1.v = _mm512_add_pd( a07.v, c15_1.v );
  c15_2.v = _mm512_add_pd( a07.v, c15_2.v );
  c15_3.v = _mm512_add_pd( a07.v, c15_3.v );
  c15_4.v = _mm512_add_pd( a07.v, c15_4.v );
  c15_5.v = _mm512_add_pd( a07.v, c15_5.v );
  c15_6.v = _mm512_add_pd( a07.v, c15_6.v );
  c15_7.v = _mm512_add_pd( a07.v, c15_7.v );

  c23_0.v = _mm512_add_pd( a07.v, c23_0.v );
  c23_1.v = _mm512_add_pd( a07.v, c23_1.v );
  c23_2.v = _mm512_add_pd( a07.v, c23_2.v );
  c23_3.v = _mm512_add_pd( a07.v, c23_3.v );
  c23_4.v = _mm512_add_pd( a07.v, c23_4.v );
  c23_5.v = _mm512_add_pd( a07.v, c23_5.v );
  c23_6.v = _mm512_add_pd( a07.v, c23_6.v );
  c23_7.v = _mm512_add_pd( a07.v, c23_7.v );

  
  
  // c = pow( c, powe );
  if ( powe == 2.0 ) {
	c07_0.v = _mm512_mul_pd( c07_0.v, c07_0.v );
	c07_1.v = _mm512_mul_pd( c07_1.v, c07_1.v );
	c07_2.v = _mm512_mul_pd( c07_2.v, c07_2.v );
	c07_3.v = _mm512_mul_pd( c07_3.v, c07_3.v );
	c07_4.v = _mm512_mul_pd( c07_4.v, c07_4.v );
	c07_5.v = _mm512_mul_pd( c07_5.v, c07_5.v );
	c07_6.v = _mm512_mul_pd( c07_6.v, c07_6.v );
	c07_7.v = _mm512_mul_pd( c07_7.v, c07_7.v );

	c15_0.v = _mm512_mul_pd( c15_0.v, c15_0.v );
	c15_1.v = _mm512_mul_pd( c15_1.v, c15_1.v );
	c15_2.v = _mm512_mul_pd( c15_2.v, c15_2.v );
	c15_3.v = _mm512_mul_pd( c15_3.v, c15_3.v );
	c15_4.v = _mm512_mul_pd( c15_4.v, c15_4.v );
	c15_5.v = _mm512_mul_pd( c15_5.v, c15_5.v );
	c15_6.v = _mm512_mul_pd( c15_6.v, c15_6.v );
	c15_7.v = _mm512_mul_pd( c15_7.v, c15_7.v );

	c23_0.v = _mm512_mul_pd( c23_0.v, c23_0.v );
	c23_1.v = _mm512_mul_pd( c23_1.v, c23_1.v );
	c23_2.v = _mm512_mul_pd( c23_2.v, c23_2.v );
	c23_3.v = _mm512_mul_pd( c23_3.v, c23_3.v );
	c23_4.v = _mm512_mul_pd( c23_4.v, c23_4.v );
	c23_5.v = _mm512_mul_pd( c23_5.v, c23_5.v );
	c23_6.v = _mm512_mul_pd( c23_6.v, c23_6.v );
	c23_7.v = _mm512_mul_pd( c23_7.v, c23_7.v );
  }
  else if ( powe == 4.0 ) {
	c07_0.v = _mm512_mul_pd( c07_0.v, c07_0.v );
	c07_1.v = _mm512_mul_pd( c07_1.v, c07_1.v );
	c07_2.v = _mm512_mul_pd( c07_2.v, c07_2.v );
	c07_3.v = _mm512_mul_pd( c07_3.v, c07_3.v );
	c07_4.v = _mm512_mul_pd( c07_4.v, c07_4.v );
	c07_5.v = _mm512_mul_pd( c07_5.v, c07_5.v );
	c07_6.v = _mm512_mul_pd( c07_6.v, c07_6.v );
	c07_7.v = _mm512_mul_pd( c07_7.v, c07_7.v );

	c15_0.v = _mm512_mul_pd( c15_0.v, c15_0.v );
	c15_1.v = _mm512_mul_pd( c15_1.v, c15_1.v );
	c15_2.v = _mm512_mul_pd( c15_2.v, c15_2.v );
	c15_3.v = _mm512_mul_pd( c15_3.v, c15_3.v );
	c15_4.v = _mm512_mul_pd( c15_4.v, c15_4.v );
	c15_5.v = _mm512_mul_pd( c15_5.v, c15_5.v );
	c15_6.v = _mm512_mul_pd( c15_6.v, c15_6.v );
	c15_7.v = _mm512_mul_pd( c15_7.v, c15_7.v );

	c23_0.v = _mm512_mul_pd( c23_0.v, c23_0.v );
	c23_1.v = _mm512_mul_pd( c23_1.v, c23_1.v );
	c23_2.v = _mm512_mul_pd( c23_2.v, c23_2.v );
	c23_3.v = _mm512_mul_pd( c23_3.v, c23_3.v );
	c23_4.v = _mm512_mul_pd( c23_4.v, c23_4.v );
	c23_5.v = _mm512_mul_pd( c23_5.v, c23_5.v );
	c23_6.v = _mm512_mul_pd( c23_6.v, c23_6.v );
	c23_7.v = _mm512_mul_pd( c23_7.v, c23_7.v );

	c07_0.v = _mm512_mul_pd( c07_0.v, c07_0.v );
	c07_1.v = _mm512_mul_pd( c07_1.v, c07_1.v );
	c07_2.v = _mm512_mul_pd( c07_2.v, c07_2.v );
	c07_3.v = _mm512_mul_pd( c07_3.v, c07_3.v );
	c07_4.v = _mm512_mul_pd( c07_4.v, c07_4.v );
	c07_5.v = _mm512_mul_pd( c07_5.v, c07_5.v );
	c07_6.v = _mm512_mul_pd( c07_6.v, c07_6.v );
	c07_7.v = _mm512_mul_pd( c07_7.v, c07_7.v );

	c15_0.v = _mm512_mul_pd( c15_0.v, c15_0.v );
	c15_1.v = _mm512_mul_pd( c15_1.v, c15_1.v );
	c15_2.v = _mm512_mul_pd( c15_2.v, c15_2.v );
	c15_3.v = _mm512_mul_pd( c15_3.v, c15_3.v );
	c15_4.v = _mm512_mul_pd( c15_4.v, c15_4.v );
	c15_5.v = _mm512_mul_pd( c15_5.v, c15_5.v );
	c15_6.v = _mm512_mul_pd( c15_6.v, c15_6.v );
	c15_7.v = _mm512_mul_pd( c15_7.v, c15_7.v );

	c23_0.v = _mm512_mul_pd( c23_0.v, c23_0.v );
	c23_1.v = _mm512_mul_pd( c23_1.v, c23_1.v );
	c23_2.v = _mm512_mul_pd( c23_2.v, c23_2.v );
	c23_3.v = _mm512_mul_pd( c23_3.v, c23_3.v );
	c23_4.v = _mm512_mul_pd( c23_4.v, c23_4.v );
	c23_5.v = _mm512_mul_pd( c23_5.v, c23_5.v );
	c23_6.v = _mm512_mul_pd( c23_6.v, c23_6.v );
	c23_7.v = _mm512_mul_pd( c23_7.v, c23_7.v );
  }
  else {
	a07.v   = _mm512_set1_pd( powe );
	c07_0.v = _mm512_pow_pd( a07.v, c07_0.v );
	c07_1.v = _mm512_pow_pd( a07.v, c07_1.v );
	c07_2.v = _mm512_pow_pd( a07.v, c07_2.v );
	c07_3.v = _mm512_pow_pd( a07.v, c07_3.v );
	c07_4.v = _mm512_pow_pd( a07.v, c07_4.v );
	c07_5.v = _mm512_pow_pd( a07.v, c07_5.v );
	c07_6.v = _mm512_pow_pd( a07.v, c07_6.v );
	c07_7.v = _mm512_pow_pd( a07.v, c07_7.v );

	c15_0.v = _mm512_pow_pd( a07.v, c15_0.v );
	c15_1.v = _mm512_pow_pd( a07.v, c15_1.v );
	c15_2.v = _mm512_pow_pd( a07.v, c15_2.v );
	c15_3.v = _mm512_pow_pd( a07.v, c15_3.v );
	c15_4.v = _mm512_pow_pd( a07.v, c15_4.v );
	c15_5.v = _mm512_pow_pd( a07.v, c15_5.v );
	c15_6.v = _mm512_pow_pd( a07.v, c15_6.v );
	c15_7.v = _mm512_pow_pd( a07.v, c15_7.v );

	c23_0.v = _mm512_pow_pd( a07.v, c23_0.v );
	c23_1.v = _mm512_pow_pd( a07.v, c23_1.v );
	c23_2.v = _mm512_pow_pd( a07.v, c23_2.v );
	c23_3.v = _mm512_pow_pd( a07.v, c23_3.v );
	c23_4.v = _mm512_pow_pd( a07.v, c23_4.v );
	c23_5.v = _mm512_pow_pd( a07.v, c23_5.v );
	c23_6.v = _mm512_pow_pd( a07.v, c23_6.v );
	c23_7.v = _mm512_pow_pd( a07.v, c23_7.v );
  }

  // Preload u03, u47
  a07.v    = _mm512_load_pd( u      );
  a15.v    = _mm512_load_pd( u +  8 );
  a23.v    = _mm512_load_pd( u + 16 );

  // Multiple rhs weighted sum.
  #include<weighted_sum_int_d24x8.h>
}
