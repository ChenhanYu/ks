#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include <ks.h>

void exp_int_d4( double* );


void compute_error(
    int    n,
    double *y,
    double *goal
    )
{
  int    i;
  double error, tmp, nrm2;
  for ( i = 0; i < n; i ++ ) {
    //tmp = fabs( x[ i ] - goal[ i ] );
    //tmp = y[ i ] - goal[ i ];
    tmp = ( y[ i ] - goal[ i ] ) / goal[ i ];
    //printf( "%E\n", tmp );
  }
}

void error_plot(
    int    n,
    double *x,
    double *y,
    double *gmkl,
    double *goal
    )
{
  int    i;
  double err_y, err_mkl;
  
  printf( "exp_err = [\n" );

  for ( i = 0; i < n; i ++ ) {
    err_y   = y[ i ] - goal[ i ];
    err_mkl = gmkl[ i ] - goal[ i ];
    //if ( fabs( err_y ) > 0.0000001 ) {
      printf( "%E, %E, %E;\n", x[ i ], err_y, err_mkl );
    //}
  }

  printf( "];\n" ); 
}

int main()
{
  int    i;
  int    n = 1000;
  double beg, end, inc;
  double x[ n ];
  double y[ n ];
  double goal[ n ];
  double gmkl[ n ];
  
  //beg = 0.0;
  //end = 0.693147180559945;
  
  beg = 1.0;
  end = -4.0;
  
  inc = ( end - beg ) / n;

  for ( i = 0; i < n; i ++ ) {
    x[ i ] = beg + inc * i;
    y[ i ] = x[ i ];
  }

  for ( i = 0; i < n; i ++ ) {
    goal[ i ] = exp( x[ i ] );
  }
 
  vdExp( n, x, gmkl );

  for ( i = 0; i < n; i += 4 ) {
    exp_int_d4( y + i );
  }

  compute_error( n, y, goal );
  compute_error( n, gmkl, goal );
  error_plot( n, x, y, gmkl, goal );



  //pow_int_d4( x, y );


  return 0;
}
