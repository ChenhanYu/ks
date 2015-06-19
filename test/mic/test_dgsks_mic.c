#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <ks.h>



void ks_gaussian_int_d16x8(
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



void compute_error(
    int    n,
    double *u_test,
    double *u_gold
    )
{
  int    i;
  int    max_idx;
  double max_err;
  double abs_err;
  double rel_err;
  double tmp;
  double nrm2;

  max_idx = -1;
  max_err = 0.0;
  nrm2    = 0.0;

  for ( i = 0; i < n; i ++ ) {
    tmp = fabs( u_test[ i ] - u_gold[ i ] );
    if ( tmp > max_err ) {
      max_err = tmp;
      max_idx = i;
    }
    rel_err += tmp * tmp;
    nrm2    += u_gold[ i ] * u_gold[ i ];
  }

  abs_err = sqrt( rel_err );
  rel_err /= nrm2;
  rel_err = sqrt( rel_err );

  printf( "rel error = %E, abs error = %E, max error = %E, idx = %d\n", 
      rel_err, abs_err, max_err, max_idx );
}





void test_dgsks(
  int m,
  int n,
  int k
    ) 
{
  int    i, j, p, nx, iter, n_iter;
  int    *amap, *bmap, *wmap;
  double *XA, *XB, *XA2, *XB2, *u, *w, *umkl;
  double tmp, h, error, flops;
  double ref_beg, ref_time, dgsks_beg, dgsks_time;
  ks_t   kernel;

  nx = 4096 * 10;
  n_iter = 1;
  h = -0.5;

  amap = (int*)malloc( sizeof(int) * m );
  bmap = (int*)malloc( sizeof(int) * n );
  wmap = (int*)malloc( sizeof(int) * n );
  XA   = (double*)malloc( sizeof(double) * k * nx );
  XA2  = (double*)malloc( sizeof(double) * nx );
  u    = (double*)malloc( sizeof(double) * nx );
  w    = (double*)malloc( sizeof(double) * nx );
  umkl = (double*)malloc( sizeof(double) * nx );

  //printf( "After Malloc\n" );

  for ( i = 0; i < nx; i ++ ) {
    u[ i ] = 0.0;
    umkl[ i ] = 0.0;
    w[ i ] = 1.0;
  }


  for ( i = 0; i < m; i ++ ) {
    //amap[ i ] = i * 2;
    amap[ i ] = i;
  }

  for ( j = 0; j < n; j ++ ) {
    //bmap[ j ] = j * 2 + 1;
    //wmap[ j ] = j * 2 + 1;
    bmap[ j ] = j;
    wmap[ j ] = j;
  }

  //printf( "check point\n" );

  // random[ 0, 0.1 ]
  for ( i = 0; i < nx; i ++ ) {
    for ( p = 0; p < k; p ++ ) {
      XA[ i * k + p ] = (double)( rand() % 100 ) / 1000.0;	
      //printf( "i = %d\n", i );
      //XA[ i * k + p ] = 1.0;	
    }
  }

  //printf( "XA03 = %lf, %lf, %lf, %lf\n", XA[0], XA[1], XA[2], XA[3] );
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

  // Compute XA2
  for ( i = 0; i < nx; i ++ ) {
    tmp = 0.0;
    for ( p = 0; p < k; p ++ ) {
      tmp += XA[ i * k + p ] * XA[ i * k + p ];
    }
    XA2[ i ] = tmp;
  }

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



  // Use the same coordinate table
  XB = XA;
  XB2 = XA2;



  // Test Gaussian Kernel
  kernel.type = KS_GAUSSIAN;
  kernel.scal = -0.5;
  //kernel.scal = -1.0 * 0.16 * 0.16;
  //kernel.scal = -5000.0;



  // Test Polynomial Kernel
  //kernel.type = KS_POLYNOMIAL;
  //kernel.powe = 3.0;
  //kernel.scal = 0.1;
  //kernel.cons = 0.1;


  // Test Laplace Kernel
  //kernel.type = KS_LAPLACE;



  dgsks_beg = omp_get_wtime();

  // Call my implementation
  for ( iter = 0; iter < n_iter; iter ++ ) {
    //dgsks_mic_var2(
    dgsks_mic_parallel(
        &kernel,
        m,
        n,
        k,
        u,
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


  dgsks_time = omp_get_wtime() - dgsks_beg;

  //printf( "Reference\n" );


  ref_beg = omp_get_wtime();


  // Call the reference function 
  for ( iter = 0; iter < n_iter; iter ++ ) {
    dgsks_ref_mic(
        &kernel,
        m,
        n,
        k,
        umkl,
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

  ref_time = omp_get_wtime() - ref_beg;


  ref_time /= n_iter;
  dgsks_time /= n_iter;


  error = 0.0;

  for ( i = 0; i < m; i ++ ) {
    if ( fabs( umkl[ i ] - u[ i ] ) > 0.00000001 ) {
      printf( " error\n ");
      printf( "umkl[ %d ] = %lf, u[ i ] = %lf\n", i, umkl[ i ], u[ i ] );
      break;
    }
    tmp = umkl[ i ] - u[ i ];
    error += tmp * tmp;
  }

  //compute_error( m, u, umkl );

  //printf( "%lf, %lf\n", umkl[ 0 ], u[ 0 ] );

  switch ( kernel.type ) {
    case KS_GAUSSIAN:
      flops = ( (double)( m * n ) / ( 1024 * 1024 * 1024 ) ) * ( 2 * k + 35 );
      break;
    case KS_POLYNOMIAL:
      // Need to be readjusted
      flops = ( (double)( m * n ) / ( 1024 * 1024 * 1024 ) ) * ( 2 * k + 50 );
      break;
    case KS_LAPLACE:
      flops = ( (double)( m * n ) / ( 1024 * 1024 * 1024 ) ) * ( 2 * k + 50 );
      break;
    default:
      exit( 1 );
  }

  //printf( "dgsks: %6.4lf secs / %4.1lf Gflops, ref: %6.4lf secs / %4.1lf Gflops, Absolute Error: %E\n", 
  //    dgsks_time, flops / dgsks_time, ref_time, flops / ref_time, sqrt( error ) );

  printf( "%d, %d, %d, %5.2lf, %5.2lf;\n", 
      m, n, k, flops / dgsks_time, flops / ref_time );


  free( XA );
  free( XA2 );
  free( amap );
  free( bmap );
  free( wmap );
  free( w );
  free( u );
  free( umkl );
}

int main( int argc, char *argv[] )
{
  int    m, n, k; 

  sscanf( argv[ 1 ], "%d", &m );
  sscanf( argv[ 2 ], "%d", &n );
  sscanf( argv[ 3 ], "%d", &k );

  //printf( "Test Dgsks: m = %d, n = %d, k = %d\n", m, n, k );

  test_dgsks( m, n, k );


  return 0;
}
