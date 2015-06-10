#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <omp.h>
#include <ks.h>




void ks_rank_k_asm_d30x8(
    int    k,
    double *a,
    double *b,
    double *c,
    int    ldc,
    aux_t  *aux
	);

void ks_rank_k_asm_d8x30(
    int    k,
    double *a,
    double *b,
    double *c,
    int    ldc,
    aux_t  *aux
	);


void ks_rank_k_int_d16x14(
    int    k,
    double *a,
    double *b,
    double *c,
    int    ldc,
    aux_t  *aux
	);

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
    );


//void ks_gaussian_asm_d8x30(
//    unsigned long long  k,
//    double alpha,
//    double *u,
//    double *aa,
//    double *a,
//    double *bb,
//    double *b,
//    double *w,
//    aux_t  *aux
//    );



void test_gaussian_micro()
{
  int    i, p, iter;
  int    k = 1;
  int    ldc = 16;
  double *a, *b, *c, *aa, *bb, *u, *w;
  double scal = -0.5;
  double ks_beg, ks_time;

  printf( "Here-1\n" );

  posix_memalign( (void**)&a, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * 8 * k );
  posix_memalign( (void**)&b, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * 32 * k );
  posix_memalign( (void**)&c, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * 8 * 32 );
  posix_memalign( (void**)&aa, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * 8 );
  posix_memalign( (void**)&bb, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * 32 );
  posix_memalign( (void**)&u, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * 8 );
  posix_memalign( (void**)&w, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * 32 );

  printf( "Here0\n" );


  for ( i = 0; i < 8; i++ ) {
	aa[ i ] = 0.0;
	for ( p = 0; p < k; p ++ ) {
      a[ p * 8 + i ] = 1.0;
	  aa[ i ] += 1.0 * 1.0;
	}
  }

  for ( i = 0; i < 32; i++ ) {
	bb[ i ] = 0.0;
	for ( p = 0; p < k; p ++ ) {
      b[ p * 32 + i ] = 0.0;
	  bb[ i ] += 0.0 * 0.0;
	}
  }

  for ( i = 0; i < 8; i++ ) {
	u[ i ] = 0.0;
  }

  for ( i = 0; i < 32; i++ ) {
	w[ i ] = 0.0;
  }

  for ( i = 0; i < 30; i++ ) {
	w[ i ] = 1.0;
  }


  printf( "Here\n" );

  ks_beg = omp_get_wtime();


    aux_t aux;
	aux.c_buff = c;
	//for ( iter = 0; iter < 100; iter ++ ) {
	//ks_gaussian_int_d16x14(
	ks_gaussian_asm_d8x30(
		(unsigned long long)k,
		scal,
		u,
		aa,
		a,
		bb,
		b,
		w,
		&aux
		);
	//}


  ks_time = omp_get_wtime() - ks_beg;
  ks_time /= 100;

  printf( "mic ks: %lf sec, %lf\n", ks_time, 
	  (double)( 0.002 * 0.024 * 0.008 * ( k + 18 ) ) / ks_time );
//  printf( "mic ks: %lf sec, %lf\n", ks_time, 
//	  (double)( 0.002 * 0.016 * 0.008 * ( k ) ) / ks_time );

  printf( "%E, %E, %E, %E, %E, %E, %E, %E\n",
	  u[ 0 ], u[ 1 ], u[ 2 ], u[ 3 ], u[ 4 ], u[ 5 ], u[ 6 ], u[ 7 ] );



  free( a );
  free( b );
  free( c );
  free( aa );
  free( bb );
  free( u );
  free( w );
}



void test_rank_k_micro()
{
  int    i, iter;
  int    k = 96;
  int    ldc = 16;
  double *a, *b, *c;
  //aux_t aux;
  double rank_k_beg, rank_k_time;


  posix_memalign( (void**)&a, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * 8 * k );
  posix_memalign( (void**)&b, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * 32 * k );
  posix_memalign( (void**)&c, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * 8 * 32 );

  for ( i = 0; i < 8 * k; i++ ) {
    a[ i ] = 1.0;
  }

  for ( i = 0; i < 32 * k; i++ ) {
    b[ i ] = 1.0;
  }

  rank_k_beg = omp_get_wtime();

    aux_t aux;
	for ( iter = 0; iter < 100; iter ++ ) {
	  ks_rank_k_asm_d30x8(
		  k,
		  a,
		  b,
		  c,
		  ldc,
		  &aux
		  );
	}

  rank_k_time = omp_get_wtime() - rank_k_beg;
  rank_k_time /= 100;

  printf( "mic rank_k: %lf sec, %lf Gflops\n", rank_k_time, 
	  (double)( 0.002 * 0.030 * 0.008 * ( k + 1 ) ) / rank_k_time );

  free( a );
  free( b );
  free( c );
}

int main( int argc, char *argv[] )
{
  test_rank_k_micro();
  test_gaussian_micro();

  return 0;
}
