#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include <iostream>

extern "C" {
#include <ks.h>
}
#include <omp_dgsks_list_mic.hpp>

void omp_dgsks_list_mic(
    ks_t   *kernel,
    int    k,
    std::vector<double> &u,
    double *XA,
    double *XA2,
    std::vector< std::vector<int> > &alist,
    double *XB,
    double *XB2,
    std::vector< std::vector<int> > &blist,
    double *w,
    std::vector< std::vector<int> > &wlist
    );



void random_permutation( int* x, int nx )
{
  for ( int j = 0; j < nx; j++ ) {
    x[ j ] = j;
  }

  for ( int j = 0; j < nx; j++ ) {
    int target = rand() % nx;
    int tmp;
    if ( target != j ) {
      tmp         = x[ j ];
      x[ j ]      = x[ target ];
      x[ target ] = tmp;
    }
  }
}

// C++ implementation using std::vector
// Create two separate coordinate tables, n_list random lists
// amap[ n_list ], bmap[ n_list ], wmap[ n_list ]
//
//
void test_dgsks_list_mic(
    int    rangem,
    int    rangen,
    int    k,
    int    ntask
    ) 
{
  int    m, n, i, j, p, nx, iter;
  int    *amap, *bmap, *wmap;
  int    *randperm;
  double *XA, *XB, *XA2, *XB2, *u, *w, *umkl;
  double tmp, h, error, flops;
  double ref_beg, ref_time, dgsks_beg, dgsks_time;
  ks_t   kernel;


  // ========================
  std::vector<double> uvec;

  // ========================
  std::vector< std::vector<int> > alist;
  std::vector< std::vector<int> > blist;
  std::vector< std::vector<int> > wlist;

  int    n_list;

  n_list = ntask;

  nx     = 4096 * 5;
  //nx     = rangem * 5;
  //k      = 256;

  flops  = 0.0;

  // Initialize alist, blist and wlist
  alist.resize( n_list );
  blist.resize( n_list );
  wlist.resize( n_list );

  // Random Permutation Array
  randperm = (int*)malloc( sizeof(int) * nx );

  // Random list generation
  for ( i = 0; i < n_list; i++ ) {
    // First decide amap.size() and bmap.size()
    
    
    int na = rand() % rangem + 512; 
    int nb = rand() % rangen + 512; 
    //int na = 569;
    //int nb = 8;
    
    flops += (double)( na * nb );
    
    alist[ i ].resize( na );
    blist[ i ].resize( nb );
    wlist[ i ].resize( nb );
    // Random permutation
    random_permutation( randperm, nx );
    // Random amap
    for ( j = 0; j < na; j++ ) {
      // TODO: nx should be nxa if XA and XB are with different sizes.
      // TODO: replace the random generation as random permutation
      //
      alist[ i ][ j ] = randperm[ j ];

      //if ( alist[ i ][ j ] == 909 ) {
      //  std::cout << j << "\n";
      //}
    }
    // Random permutation
    random_permutation( randperm, nx );
    // Random bmap and wmap
    for ( j = 0; j < nb; j++ ) {
      // TODO: nx should be nxa if XA and XB are with different sizes.
      blist[ i ][ j ] = randperm[ j ];
      wlist[ i ][ j ] = randperm[ j ];
    }
  }

  std::cout << "Check point after rand list\n";

  XA   = (double*)malloc( sizeof(double) * k * nx );
  XA2  = (double*)malloc( sizeof(double) * nx );
  XB   = XA;
  XB2  = XA2;
  u    = (double*)malloc( sizeof(double) * nx );
  w    = (double*)malloc( sizeof(double) * nx );
  umkl = (double*)malloc( sizeof(double) * nx );
  uvec.resize( nx, 0.0 );


  std::cout << "Check point after malloc\n";

  // Initialize u, umkl and w
  for ( i = 0; i < nx; i ++ ) {
    u[ i ] = 0.0;
    umkl[ i ] = 0.0;
    w[ i ] = 1.0;
  }


  std::cout << "Check point after u, umkl, w\n";

  // random[ 0, 1 ]
  for ( i = 0; i < nx; i ++ ) {
    for ( p = 0; p < k; p ++ ) {
      XA[ i * k + p ] = (double)( rand() % 100 ) / 1000.0;	
      //XA[ i * k + p ] = 1.0;	
    }
  }

  std::cout << "Check point after XA\n";

  // Compute XA2
  for ( i = 0; i < nx; i ++ ) {
    tmp = 0.0;
    for ( p = 0; p < k; p ++ ) {
      tmp += XA[ i * k + p ] * XA[ i * k + p ];
    }
    XA2[ i ] = tmp;
  }


  std::cout << "Check point brfore kernel\n";

  // Test Gaussian Kernel
  kernel.type = KS_GAUSSIAN;
  kernel.scal = -0.5;
  //kernel.scal = -5000.0;

  // Test Polynomial Kernel
  //kernel.type = KS_POLYNOMIAL;
  //kernel.powe = 3.0;
  //kernel.scal = 0.1;
  //kernel.cons = 0.1;

  // Test Laplace Kernel
  //kernel.type = KS_LAPLACE;

  // Compute u using dgsks
  dgsks_beg = omp_get_wtime();
  //for ( i = 0; i < n_list; i++ ) {
  //  // TODO: replace amap, bmap, wmap (double checked)
  //  dgsks(
  //      &kernel,
  //      alist[ i ].size(),
  //      blist[ i ].size(),
  //      k,
  //      u,
  //      XA,
  //      XA2,
  //      alist[ i ].data(),
  //      XB,
  //      XB2,
  //      blist[ i ].data(),
  //      w,
  //      wlist[ i ].data()
  //      );
  //}

  //omp_dgsks_list(
  //    &kernel,
  //    k,
  //    uvec,
  //    XA,
  //    XA2,
  //    alist,
  //    XB,
  //    XB2,
  //    blist,
  //    w,
  //    wlist
  //    );

  omp_dgsks_list_symmetric_mic(
      &kernel,
      k,
      uvec,
      XA,
      nx,
      alist,
      blist,
      w,
      wlist
      );



  dgsks_time = omp_get_wtime() - dgsks_beg;

  printf( "Reference\n" );


  // Compute u using mkl reference kernel
  ref_beg = omp_get_wtime();
  for ( i = 0; i < n_list; i++ ) {
    //dgsks_ref_mic(
    //    &kernel,
    //    alist[ i ].size(),
    //    blist[ i ].size(),
    //    k,
    //    umkl,
    //    XA,
    //    XA2,
    //    alist[ i ].data(),
    //    XB,
    //    XB2,
    //    blist[ i ].data(),
    //    w,
    //    wlist[ i ].data()
    //    );
  }
  ref_time = omp_get_wtime() - ref_beg;
  // ========================


  // TODO: implement error checking



  error = 0.0;

  for ( i = 0; i < nx; i ++ ) {
    //if ( fabs( umkl[ i ] - u[ i ] ) > 0.0000000001 ) {
    if ( fabs( umkl[ i ] - uvec[ i ] ) > 0.0000000001 ) {
      printf( " error\n ");
      //printf( "umkl[ %d ] = %E, u[ i ] = %E\n", i, umkl[ i ], u[ i ] );
      //printf( "umkl[ %d ] = %lf, u[ i ] = %lf\n", i, umkl[ i ], uvec[ i ] );
      break;
    }
    //printf( "%E\n", uvec[ i ] );
    //tmp = umkl[ i ] - u[ i ];
    tmp = umkl[ i ] - uvec[ i ];
    error += tmp * tmp;
  }

  //printf( "%lf, %lf\n", umkl[ 0 ], u[ 0 ] );

  switch ( kernel.type ) {
    case KS_GAUSSIAN:
      //flops = ( (double)( m * n ) / ( 1024 * 1024 * 1024 ) ) * ( 2 * k + 35 );
      flops = flops * ( 2 * k + 35 ) / ( 1024 * 1024 * 1024 );
      break;
    case KS_POLYNOMIAL:
      // Need to be readjusted
      //flops = ( (double)( m * n ) / ( 1024 * 1024 * 1024 ) ) * ( 2 * k + 50 );
      flops = flops * ( 2 * k + 50 ) / ( 1024 * 1024 * 1024 );
      break;
    case KS_LAPLACE:
      //flops = ( (double)( m * n ) / ( 1024 * 1024 * 1024 ) ) * ( 2 * k + 50 );
      flops = flops * ( 2 * k + 50 ) / ( 1024 * 1024 * 1024 );
      break;
    default:
      exit( 1 );
  }

  printf( "dgsks: %6.4lf secs / %4.1lf Gflops, ref: %6.4lf secs / %4.1lf Gflops, Absolute Error: %E\n", 
      dgsks_time, flops / dgsks_time, ref_time, flops / ref_time, sqrt( error ) );

  free( XA );
  free( XA2 );

}



int main( int argc, char *argv[] )
{
  int    rangem, rangen, k, ntask; 

  sscanf( argv[ 1 ], "%d", &rangem );
  sscanf( argv[ 2 ], "%d", &rangen );
  sscanf( argv[ 3 ], "%d", &k );
  sscanf( argv[ 4 ], "%d", &ntask );

  test_dgsks_list_mic( rangem, rangen, k, ntask );

  return 0;
}
