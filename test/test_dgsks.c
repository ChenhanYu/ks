/*
 * test_dgsks.c
 *
 * Chenhan D. Yu
 *
 * Department of Computer Science, University of Texas at Austin
 *
 * Purpose: 
 * this is the main function to exam the correctness between dgsks()
 * and dgsks_ref().
 *
 * Todo:
 *
 * Chenhan
 * Apr 27, 2015: readjust the flops count of the polynomail, 
 * laplace, tanh kernel.
 *
 * Modification:
 * Chenhan
 * Apr 27, 2015: New tanh kernel configuration. 
 *
 *
 * */


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <ks.h>

void compute_error(
    int    m,
    int    rhs,
    double *u_test,
    double *u_gold
    )
{
  int    i, p, max_idx;
  double max_err, abs_err, rel_err;
  double tmp, nrm2;

  max_idx = -1;
  max_err = 0.0;
  nrm2    = 0.0;

  for ( i = 0; i < m; i ++ ) {
    for ( p = 0; p < rhs; p ++ ) {
      tmp = fabs( u_test[ i * rhs + p ] - u_gold[ i * rhs + p ] );
      if ( tmp > max_err ) {
        max_err = tmp;
        max_idx = i;
      }
      rel_err += tmp * tmp;
      nrm2    += u_gold[ i * rhs + p ] * u_gold[ i * rhs + p ];
    }
  }

  abs_err = sqrt( rel_err );
  rel_err /= nrm2;
  rel_err = sqrt( rel_err );

  printf( "rel error = %E, abs error = %E, max error = %E, idx = %d\n", 
      rel_err, abs_err, max_err, max_idx );
}


/* 
 * --------------------------------------------------------------------------
 * @brief  This is the test routine to exam the correctness of GSKS. XA and
 *         XB are d leading coordinate tables, and u, w have to be rhs
 *         leading. In this case, dgsks() doesn't need to know the size of
 *         nxa and nxb as long as those index map--amap, bmap, umap and wmap
 *         --are within the legal range.
 *
 * @param  m       Number of target points
 * @param  n       Number of source points
 * @param  k       Data point dimension
 * --------------------------------------------------------------------------
 */
void test_dgsks(
  int m,
  int n,
  int k
    ) 
{
  int    i, j, p, nx, iter, n_iter, rhs;
  int    *amap, *bmap, *wmap, *umap;
  double *XA, *XB, *XA2, *XB2, *u, *w, *h, *umkl;
  double tmp, error, flops;
  double ref_beg, ref_time, dgsks_beg, dgsks_time;
  ks_t   kernel;

  nx     = 4096 * 5;
  rhs    = KS_RHS;
  n_iter = 1;


  // ------------------------------------------------------------------------
  // Memory allocation for all common buffers
  // ------------------------------------------------------------------------
  amap = (int*)malloc( sizeof(int) * m );
  umap = (int*)malloc( sizeof(int) * m );
  bmap = (int*)malloc( sizeof(int) * n );
  wmap = (int*)malloc( sizeof(int) * n );
  XA   = (double*)malloc( sizeof(double) * k * nx );   // k   leading
  XA2  = (double*)malloc( sizeof(double) * nx );
  u    = (double*)malloc( sizeof(double) * nx * rhs ); // rhs leading
  w    = (double*)malloc( sizeof(double) * nx * rhs ); // rhs leading
  umkl = (double*)malloc( sizeof(double) * nx * rhs ); // rhs leading
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Initialization
  // ------------------------------------------------------------------------
  for ( i = 0; i < nx; i ++ ) {
    for ( p = 0; p < rhs; p ++ ) {
      u[ i * rhs + p ]    = 0.0;
      umkl[ i * rhs + p ] = 0.0;
      w[ i * rhs + p ]    = (double)( rand() % 1000 ) / 1000.0;
    }
  }

  for ( i = 0; i < m; i ++ ) {
    //amap[ i ] = i * 2;
    amap[ i ] = i;
    //umap[ i ] = i * 2;
    umap[ i ] = i;
  }

  for ( j = 0; j < n; j ++ ) {
    //bmap[ j ] = j * 2 + 1;
    bmap[ j ] = j;
    //wmap[ j ] = j * 2 + 1;
    wmap[ j ] = j;
  }

  // random[ 0, 0.1 ]
  for ( i = 0; i < nx; i ++ ) {
    for ( p = 0; p < k; p ++ ) {
      XA[ i * k + p ] = (double)( rand() % 100 ) / 1000.0;	
      //printf( "i = %d\n", i );
      //XA[ i * k + p ] = 1.0;	
    }
  }
  // ------------------------------------------------------------------------

  //printf( "XA03 = %lf, %lf, %lf, %lf\n", XA[96], XA[97], XA[98], XA[99] );
  //printf( "XA47 = %lf, %lf, %lf, %lf\n", XA[4], XA[5], XA[6], XA[7] );
  //printf( "XA811 = %lf, %lf, %lf, %lf\n", XA[8], XA[9], XA[10], XA[11] );
  //printf( "XA1215 = %lf, %lf, %lf, %lf\n", XA[12], XA[13], XA[14], XA[15] );


  // random[ 0, 1 ]
//  for ( j = 0; j < n ; j ++ ) {
//    for ( p = 0; p < k; p ++ ) {
//      XB[ j * k + p ] = (double)( rand() % 100 ) / 1000.0;
//      //XB[ j * k + p ] = 1.0;
//    }
//  }
  //printf( "XB03 = %lf, %lf, %lf, %lf\n", XB[0], XB[1], XB[2], XB[3] );

  // ------------------------------------------------------------------------
  // Compute XA2
  // ------------------------------------------------------------------------
  for ( i = 0; i < nx; i ++ ) {
    tmp = 0.0;
    for ( p = 0; p < k; p ++ ) {
      tmp += XA[ i * k + p ] * XA[ i * k + p ];
    }
    XA2[ i ] = tmp;
  }
  // ------------------------------------------------------------------------

  //printf( "XA2811 = %lf, %lf, %lf, %lf\n", XA2[96], XA2[97], XA2[98], XA2[99] );


  // Compute XB2
//  for ( j = 0; j < n; j ++ ) {
//    tmp = 0.0;
//    for ( p = 0; p < k; p ++ ) {
//      tmp += XB[ j * k + p ] * XB[ j * k + p ];
//    }
//    XB2[ j ] = tmp;
//  }

  //printf( "XB203 = %lf, %lf, %lf, %lf\n", XB2[0], XB2[1], XB2[2], XB2[3] );



  // ------------------------------------------------------------------------
  // Use the same coordinate table
  // ------------------------------------------------------------------------
  XB  = XA;
  XB2 = XA2;
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Test Gaussian Kernel
  // ------------------------------------------------------------------------
  //kernel.type = KS_GAUSSIAN;
  //kernel.scal = -0.5;
  //kernel.scal = -1.0 * 0.16 * 0.16;
  //kernel.scal = -5000.0;
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Test Variable Bandwidth Gaussian Kernel
  // ------------------------------------------------------------------------
  kernel.type = KS_GAUSSIAN_VAR_BANDWIDTH;
  kernel.h = malloc( sizeof(double) * nx );
  for ( i = 0; i < nx; i ++ ) {
    //kernel.h[ i ] = -0.5;
    //kernel.h[ i ] = ( -0.5 * i ) / 1000.0 ;
    kernel.h[ i ] = ( 1.0 + 0.5 / ( 1 + exp( -1.0 * XA2[ i ] ) ) );
    kernel.h[ i ] = -1.0 / ( 2.0 * kernel.h[ i ] * kernel.h[ i ] );
  }
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Test Polynomial Kernel
  // ------------------------------------------------------------------------
  //kernel.type = KS_POLYNOMIAL;
  //kernel.powe = 4.0;
  //kernel.scal = 0.1;
  //kernel.cons = 0.1;
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Test Laplace Kernel
  // ------------------------------------------------------------------------
  //kernel.type = KS_LAPLACE;
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Test Tanh Kernel
  // ------------------------------------------------------------------------
  //kernel.type = KS_TANH;
  //kernel.scal = 0.1;
  //kernel.cons = 0.1;
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Test Quartic Kernel
  // ------------------------------------------------------------------------
  //kernel.type = KS_QUARTIC;
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Test Multiquadratic Kernel
  // ------------------------------------------------------------------------
  //kernel.type = KS_MULTIQUADRATIC;
  //kernel.cons = 1.0 * 1.0;
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Test Epanechnikov Kernel
  // ------------------------------------------------------------------------
  //kernel.type = KS_EPANECHNIKOV;
  // ------------------------------------------------------------------------


    dgsks(
        &kernel,
        m,
        n,
        k,
        u,
        umap,     // New feature, a separate ulist
        XA,
        XA2,
        amap,
        XB,
        XB2,
        bmap,
        w,
        wmap
        );


  dgsks_beg = omp_get_wtime();
  // ------------------------------------------------------------------------
  // Call my implementation
  // ------------------------------------------------------------------------
  for ( iter = 0; iter < n_iter; iter ++ ) {
    dgsks(
        &kernel,
        m,
        n,
        k,
        u,
        umap,     // New feature, a separate ulist
        XA,
        XA2,
        amap,
        XB,
        XB2,
        bmap,
        w,
        wmap
        );
  }
  // ------------------------------------------------------------------------


  dgsks_time = omp_get_wtime() - dgsks_beg;





    dgsks_ref(
        &kernel,
        m,
        n,
        k,
        umkl,
        umap,    // New feature, a separate ulist
        XA,
        XA2,
        amap,
        XB,
        XB2,
        bmap,
        w,
        wmap
        );

  ref_beg = omp_get_wtime();
  // ------------------------------------------------------------------------
  // Call the reference function 
  // ------------------------------------------------------------------------
  for ( iter = 0; iter < n_iter; iter ++ ) {
    dgsks_ref(
        &kernel,
        m,
        n,
        k,
        umkl,
        umap,    // New feature, a separate ulist
        XA,
        XA2,
        amap,
        XB,
        XB2,
        bmap,
        w,
        wmap
        );
  }
  // ------------------------------------------------------------------------
  ref_time = omp_get_wtime() - ref_beg;
  ref_time   /= n_iter;
  dgsks_time /= n_iter;


  //error = 0.0;
  //for ( i = 0; i < m; i ++ ) {
  //  if ( fabs( umkl[ i ] - u[ i ] ) > 0.0000000001 ) {
  //    printf( " error\n ");
  //    printf( "umkl[ %d ] = %lf, u[ i ] = %lf\n", i, umkl[ i ], u[ i ] );
  //    break;
  //  }
  //  tmp = umkl[ i ] - u[ i ];
  //  error += tmp * tmp;
  //}

  compute_error( m, rhs, u, umkl );

  //printf( "%lf, %lf\n", umkl[ 0 ], u[ 0 ] );

  switch ( kernel.type ) {
    case KS_GAUSSIAN:
      flops = ( (double)( m * n ) / ( 1024 * 1024 * 1024 ) ) * ( 2 * k + 35 + 2 );
      break;
    case KS_GAUSSIAN_VAR_BANDWIDTH:
      flops = ( (double)( m * n ) / ( 1024 * 1024 * 1024 ) ) * ( 2 * k + 35 );
      free( kernel.h );
      break;
    case KS_POLYNOMIAL:
      // Need to be readjusted
      //flops = ( (double)( m * n ) / ( 1024 * 1024 * 1024 ) ) * ( 2 * k + 50 );
      flops = ( (double)( m * n ) / ( 1024 * 1024 * 1024 ) ) * ( 2 * k + 6 );
      break;
    case KS_LAPLACE:
      flops = ( (double)( m * n ) / ( 1024 * 1024 * 1024 ) ) * ( 2 * k + 60 );
      break;
    case KS_TANH:
      flops = ( (double)( m * n ) / ( 1024 * 1024 * 1024 ) ) * ( 2 * k + 89 );
      break;
    case KS_QUARTIC:
      flops = ( (double)( m * n ) / ( 1024 * 1024 * 1024 ) ) * ( 2 * k + 8 );
      break;
    case KS_MULTIQUADRATIC:
      flops = ( (double)( m * n ) / ( 1024 * 1024 * 1024 ) ) * ( 2 * k + 6 );
      break;
    case KS_EPANECHNIKOV:
      flops = ( (double)( m * n ) / ( 1024 * 1024 * 1024 ) ) * ( 2 * k + 7 );
      break;
    default:
      exit( 1 );
  }

  //printf( "dgsks: %6.4lf secs / %4.1lf Gflops, ref: %6.4lf secs / %4.1lf Gflops, Absolute Error: %E\n", 
  //    dgsks_time, flops / dgsks_time, ref_time, flops / ref_time, sqrt( error ) );

  //printf( "%d, %d, %d, %5.3lf, %5.3lf;\n", 
  //    m, n, k, dgsks_time, ref_time );
  printf( "%d, %d, %d, %5.2lf, %5.2lf;\n", 
      m, n, k, flops / dgsks_time, flops / ref_time );

}

int main( int argc, char *argv[] )
{
  int    m, n, k; 

  sscanf( argv[ 1 ], "%d", &m );
  sscanf( argv[ 2 ], "%d", &n );
  sscanf( argv[ 3 ], "%d", &k );

  //printf( "Test Dgsks: m = %d, n = %d, k = %d\n", m, n, k );

  //printf( "KS_GAUSSIAN              : %d\n", KS_GAUSSIAN );
  //printf( "KS_POLYNOMIAL            : %d\n", KS_POLYNOMIAL );
  //printf( "KS_LAPLACE               : %d\n", KS_LAPLACE );
  //printf( "KS_GAUSSIAN_VAR_BANDWIDTH: %d\n", KS_GAUSSIAN_VAR_BANDWIDTH );
  //printf( "KS_TANH                  : %d\n", KS_TANH );
  //printf( "KS_QUARTIC               : %d\n", KS_QUARTIC );
  //printf( "KS_MULTIQUADRATIC        : %d\n", KS_MULTIQUADRATIC );

  test_dgsks( m, n, k );


  return 0;
}
