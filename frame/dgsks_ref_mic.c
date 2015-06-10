/*
 * --------------------------------------------------------------------------
 * GSKS (General Stride Kernel Summation)
 * --------------------------------------------------------------------------
 * Copyright (C) 2014, The University of Texas at Austin
 *
 * dgsks_ref_mic.c
 *
 * Chenhan D. Yu - Department of Computer Science, 
 *                 The University of Texas at Austin
 *
 *
 * Purpose: 
 * this is the reference routine of the double precision general stride
 * kernel summation kernel on the Intel Xeon Phi architecture.
 *
 *
 * Todo:
 *
 *
 * Modification:
 *
 *
 * */


#include <omp.h>
#include <mkl.h>
#include <ks.h>

// This reference function will call MKL
void dgsks_ref_mic(
    ks_t   *kernel,
    int    m,
    int    n,
    int    k,
    double *u,
    double *XA,
    double *XA2,
    int    *alpha,
    double *XB,
    double *XB2,
    int    *beta,
    double *w,
    int    *omega
    )
{
  int    i, j, p;
  double *As, *Bs, *Cs, *us, *ws, *powe;
  int    norm2_distance, const_shift;
  double rank_k_scale;
  double done  =  1.0;
  int    one   =  1;
  double dmone = -1.0;
  double dzero =  0.0;

  double flops;
  double beg;
  double time_setup   = 0.0;
  double time_collect = 0.0;
  double time_dgemm   = 0.0;
  double time_kernel  = 0.0;
  double time_dgemv   = 0.0;



  beg = omp_get_wtime();

  switch ( kernel->type ) {
    case KS_GAUSSIAN:
      //norm2_distance = 1;
      //const_shift = 0;
      rank_k_scale = -2.0;
      break;
    case KS_POLYNOMIAL:
      //norm2_distance = 0;
      //const_shift = 1;
      rank_k_scale = 1.0;
      powe = (double*)malloc( sizeof(double) * m * n );
      for ( i = 0; i < m * n; i++ ) powe[ i ] = kernel->powe;
      break;
    case KS_LAPLACE:
      //norm2_distance = 1;
      //const_shift = 0;
      rank_k_scale = -2.0;
      if ( k < 3 ) {
        printf( "Error dgsks(): laplace kernel only supports k > 2.\n" );
      }
      kernel->powe = 0.5 * ( 2.0 - (double)k );
      kernel->scal = tgamma( 0.5 * k + 1.0 ) / 
        ( (double)k * (double)( k - 2 ) * pow( M_PI, 0.5 * k ) );
      powe = (double*)malloc( sizeof(double) * m * n );
      for ( i = 0; i < m * n; i++ ) powe[ i ] = kernel->powe;
      printf( "powe = %lf, scal = %lf\n", kernel->powe, kernel->scal );
      break;
    default:
      printf( "Error dgsks_ref(): illegal kernel type\n" );
      exit( 1 );
  }

  posix_memalign( (void**)&As, (size_t)DKS_SIMD_ALIGN_SIZE, 
	  sizeof(double) * m * k );
  posix_memalign( (void**)&Bs, (size_t)DKS_SIMD_ALIGN_SIZE, 
	  sizeof(double) * n * k );
  posix_memalign( (void**)&Cs, (size_t)DKS_SIMD_ALIGN_SIZE, 
	  sizeof(double) * m * m );
  posix_memalign( (void**)&us, (size_t)DKS_SIMD_ALIGN_SIZE, 
	  sizeof(double) * m );
  posix_memalign( (void**)&ws, (size_t)DKS_SIMD_ALIGN_SIZE, 
	  sizeof(double) * n );


  //As = (double*)malloc( sizeof(double) * m * k );
  //Bs = (double*)malloc( sizeof(double) * n * k );
  //Cs = (double*)malloc( sizeof(double) * m * n );
  //us = (double*)malloc( sizeof(double) * m );
  //ws = (double*)malloc( sizeof(double) * n );


  time_setup += ( omp_get_wtime() - beg );


  beg = omp_get_wtime();

  // Collect As from XA, us from u
  #pragma omp parallel for private( i, p )
  for ( i = 0; i < m; i ++ ) {
    for ( p = 0; p < k; p ++ ) {
      As[ i * k + p ] = XA[ alpha[ i ] * k + p ];
    }
    us[ i ] = u[ alpha[ i ] ];
  }

  //printf( "Bs: " );
  // Collect Bs from XB, ws from w
  #pragma omp parallel for private( j, p )
  for ( j = 0; j < n; j ++ ) {
    for ( p = 0; p < k; p ++ ) {
      Bs[ j * k + p ] = XB[ beta[ j ] * k + p ];
      //printf( "%lf, ", Bs[ j * k + p ]);
    }
    ws[ j ] = w[ omega[ j ] ];
  }
  //printf( "\n" );

  //printf( "B2s: " );
  //for ( j = 0; j < n; j ++ ) {
  //  printf( "%lf, ", XB2[ beta[ j ] ]);
  //}
  //printf( "\n" );


  // C = -2.0 * A^t * B
  beg = omp_get_wtime();
  dgemm( "T", "N", &m, &n, &k, &rank_k_scale, As, &k, Bs, &k, &dzero, Cs, &m );
  time_dgemm = omp_get_wtime() - beg;


  //for ( i = 0; i < m; i ++ ) {
  //  printf( "%lf, %lf, %lf, %lf\n", 
  //      Cs[ 0 * m + i ], 
  //      Cs[ 1 * m + i ],
  //      Cs[ 2 * m + i ],
  //      Cs[ 3 * m + i ]
  //      );
  //}


  // C[ i ][ j ] += ...
//  for ( j = 0; j < n; j ++ ) {
//    for ( i = 0; i < m; i ++ ) {
//      if ( norm2_distance ) {
//        Cs[ j * m + i ] += XA2[ alpha[ i ] ];
//        Cs[ j * m + i ] += XB2[ beta[ j ] ];
//      }
//      Cs[ j * m + i ] *= kernel->scal;
//      if ( const_shift ) {
//        Cs[ j * m + i ] += kernel->cons;
//      }
//    }
//  }


  //printf( "\n" );
  //for ( i = 0; i < m; i ++ ) {
  //  printf( "%lf, %lf, %lf, %lf\n", 
  //      Cs[ 0 * m + i ], 
  //      Cs[ 1 * m + i ],
  //      Cs[ 2 * m + i ],
  //      Cs[ 3 * m + i ]
  //      );
  //}


  beg = omp_get_wtime();
  switch ( kernel->type ) {
    case KS_GAUSSIAN:
      #pragma omp parallel for private( j, i )
      for ( j = 0; j < n; j ++ ) {
        for ( i = 0; i < m; i ++ ) {
          Cs[ j * m + i ] += XA2[ alpha[ i ] ];
          Cs[ j * m + i ] += XB2[ beta[ j ] ];
          Cs[ j * m + i ] *= kernel->scal;
		  Cs[ j * m + i ] = exp( Cs[ j * m + i ] );
        }
      }
      //vdExp( m * n, Cs, Cs );
      break;
    case KS_POLYNOMIAL:
      for ( j = 0; j < n; j ++ ) {
        for ( i = 0; i < m; i ++ ) {
          Cs[ j * m + i ] *= kernel->scal;
          Cs[ j * m + i ] += kernel->cons;
        }
      }
      vdPow( m * n, Cs, powe, Cs );
      break;
    case KS_LAPLACE:
      for ( j = 0; j < n; j ++ ) {
        for ( i = 0; i < m; i ++ ) {
          Cs[ j * m + i ] += XA2[ alpha[ i ] ];
          Cs[ j * m + i ] += XB2[ beta[ j ] ];
        }
      }
      vdPow( m * n, Cs, powe, Cs );
      for ( j = 0; j < n; j ++ ) {
        for ( i = 0; i < m; i ++ ) {
          Cs[ j * m + i ] *= kernel->scal;
        }
      }
      break;
    default:
      printf( "Error dgsks_ref(): illegal kernel type\n" );
      exit( 1 );
  }
  time_kernel = omp_get_wtime() - beg;


  //printf( "\n" );
  //for ( i = 0; i < m; i ++ ) {
  //  printf( "%lf, %lf, %lf, %lf\n", 
  //      Cs[ 0 * m + i ], 
  //      Cs[ 1 * m + i ],
  //      Cs[ 2 * m + i ],
  //      Cs[ 3 * m + i ]
  //      );
  //}


  // Kernel Summation
  beg = omp_get_wtime();
  dgemv( "N", &m, &n, &done, Cs, &m, ws, &one, &done, us, &one );
  time_dgemv = omp_get_wtime() - beg;


  beg = omp_get_wtime();
  // Assemble us back to u
  #pragma omp parallel for private( i )
  for ( i = 0; i < m; i ++ ) {
    u[ alpha[ i ] ] = us[ i ];
  }
  time_collect += ( omp_get_wtime() - beg );



  beg = omp_get_wtime();
  // Free the temporary buffers
  free( As );
  free( Bs );
  free( Cs );
  free( us );
  free( ws );
  time_setup += ( omp_get_wtime() - beg );

  flops = m * n * (double)( 2 * k );
  flops /= ( 1024.0 * 1024.0 * 1024.0 );

  printf( "%5.3lf, %5.3lf, %5.3lf, %5.3lf, %5.3lf sec, %5.2lf gflops\n",
	  time_setup, time_collect, time_dgemm, time_kernel, time_dgemv, flops / time_dgemm );
}
