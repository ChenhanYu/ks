/*
 * --------------------------------------------------------------------------
 * GSKS (General Stride Kernel Summation)
 * --------------------------------------------------------------------------
 * Copyright (C) 2015, The University of Texas at Austin
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 *
 * omp_dgsks_list.cpp
 *
 * Chenhan D. Yu - Department of Computer Science, 
 *                 The University of Texas at Austin
 *
 *
 * Purpose: 
 * This is the main file of the task parallel double precision 
 * general stride kernel summation.
 *
 *
 * Todo:
 *
 *
 * Modification:
 *
 *
 * */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include <iostream>
#include <deque>

extern "C" {
#include <ks.h>
}
#include <omp_dgsks_list.hpp> 



void omp_dgsks_list_unsymmetric(
    ks_t   *kernel,
    int    k,
    std::vector<double> &u,
    int    nxa,
    double *XA,
    std::vector< std::vector<int> > &alist,
    int    nxb,
    double *XB,
    std::vector< std::vector<int> > &blist,
    double *w,
    std::vector< std::vector<int> > &wlist
    )
{
  int    i, p;
  double tmp;
  double *XA2, *XB2;

  XA2  = (double*)malloc( sizeof(double) * nxa );
  XB2  = (double*)malloc( sizeof(double) * nxb );

  // Compute XA2
  #pragma omp parallel for 
  for ( i = 0; i < nxa; i ++ ) {
    tmp = 0.0;
    for ( p = 0; p < k; p ++ ) {
      tmp += XA[ i * k + p ] * XA[ i * k + p ];
    }
    XA2[ i ] = tmp;
  }
  
  // Compute XB2
  #pragma omp parallel for 
  for ( i = 0; i < nxb; i ++ ) {
    tmp = 0.0;
    for ( p = 0; p < k; p ++ ) {
      tmp += XB[ i * k + p ] * XB[ i * k + p ];
    }
    XB2[ i ] = tmp;
  }

  // Call omp_dgsks_list()
  omp_dgsks_list(
      kernel,
      k,
      u,
      alist, // Use an unified ulist
      XA,
      XA2,
      alist,
      XB,
      XB2,
      blist,
      w,
      wlist
      );

  free( XA2 );
  free( XB2 );
}

void omp_dgsks_list_symmetric(
    ks_t   *kernel,
    int    k,
    std::vector<double> &u,
    double *XA,
    int    nxa,
    std::vector< std::vector<int> > &alist,
    std::vector< std::vector<int> > &blist,
    double *w,
    std::vector< std::vector<int> > &wlist
    )
{
  int    i, p;
  double tmp;
  double *XA2;

  XA2  = (double*)malloc( sizeof(double) * nxa );

  //printf( "%d, %d\n", nxa, u.size() );


  // Compute XA2
  #pragma omp parallel for 
  for ( i = 0; i < nxa; i ++ ) {
    tmp = 0.0;
    for ( p = 0; p < k; p ++ ) {
      tmp += XA[ i * k + p ] * XA[ i * k + p ];
    }
    XA2[ i ] = tmp;
  }
  
  // Call omp_dgsks_list()
  omp_dgsks_list(
      kernel,
      k,
      u,
      alist, // Use an unified ulist
      XA,
      XA2,
      alist,
      XA,
      XA2,
      blist,
      w,
      wlist
      );

  //for ( i = 0; i < nxa; i ++ ) {
  //  if ( u[ i ] != u[ i ] ) {
  //    printf( "u[ %d ] is NAN\n", i );
  //  }
  //}


  free( XA2 );
}


void omp_dgsks_list_separated_u_unsymmetric(
    ks_t   *kernel,
    int    k,
    std::vector<double> &u,
    std::vector< std::vector<int> > &ulist,
    int    nxa,
    double *XA,
    std::vector< std::vector<int> > &alist,
    int    nxb,
    double *XB,
    std::vector< std::vector<int> > &blist,
    double *w,
    std::vector< std::vector<int> > &wlist
    )
{
  int    i, p;
  double tmp;
  double *XA2, *XB2;

  XA2  = (double*)malloc( sizeof(double) * nxa );
  XB2  = (double*)malloc( sizeof(double) * nxb );

  // Compute XA2
  #pragma omp parallel for 
  for ( i = 0; i < nxa; i ++ ) {
    tmp = 0.0;
    for ( p = 0; p < k; p ++ ) {
      tmp += XA[ i * k + p ] * XA[ i * k + p ];
    }
    XA2[ i ] = tmp;
  }
  
  // Compute XB2
  #pragma omp parallel for 
  for ( i = 0; i < nxb; i ++ ) {
    tmp = 0.0;
    for ( p = 0; p < k; p ++ ) {
      tmp += XB[ i * k + p ] * XB[ i * k + p ];
    }
    XB2[ i ] = tmp;
  }

  // Call omp_dgsks_list()
  omp_dgsks_list(
      kernel,
      k,
      u,
      ulist, // Use a separated ulist
      XA,
      XA2,
      alist,
      XB,
      XB2,
      blist,
      w,
      wlist
      );

  free( XA2 );
  free( XB2 );
}


void omp_dgsks_list_separated_u_symmetric(
    ks_t   *kernel,
    int    k,
    std::vector<double> &u,
    std::vector< std::vector<int> > &ulist,
    double *XA,
    int    nxa,
    std::vector< std::vector<int> > &alist,
    std::vector< std::vector<int> > &blist,
    double *w,
    std::vector< std::vector<int> > &wlist
    )
{
  int    i, p;
  double tmp;
  double *XA2;

  XA2  = (double*)malloc( sizeof(double) * nxa );

  // Compute XA2
  #pragma omp parallel for 
  for ( i = 0; i < nxa; i ++ ) {
    tmp = 0.0;
    for ( p = 0; p < k; p ++ ) {
      tmp += XA[ i * k + p ] * XA[ i * k + p ];
    }
    XA2[ i ] = tmp;
  }
 
  //printf( "Call omp_dgsks_list\n" );

  // Call omp_dgsks_list()
  omp_dgsks_list(
      kernel,
      k,
      u,
      ulist, // Use a separated ulist
      XA,
      XA2,
      alist,
      XA,
      XA2,
      blist,
      w,
      wlist
      );


  free( XA2 );
}


void omp_dgsks_list(
    ks_t   *kernel,
    int    k,
    std::vector<double> &u,
    std::vector< std::vector<int> > &ulist, // New feature, a separate ulist
    double *XA,
    double *XA2,
    std::vector< std::vector<int> > &alist,
    double *XB,
    double *XB2,
    std::vector< std::vector<int> > &blist,
    double *w,
    std::vector< std::vector<int> > &wlist
    )
{
  int    nthd, nu, n_list;
  double *u_local[ KS_NUM_THREAD ];
  std::deque<int> jobs[ KS_NUM_THREAD ];
  double workload[ KS_NUM_THREAD ];

  // Early return
  if ( alist.size() == 0 || blist.size() == 0 ) return;

  // Sanity check
  if ( ( alist.size() != blist.size() ) || ( blist.size() != wlist.size() ) ) {
    printf( "omp_dgsks_list(): alist, blist and wlist must have the same sizes.\n" );
    exit( 1 );
  }


  nu     = u.size();
  n_list = alist.size();

  // Initialize u_local to prevent race conditions 
  for ( int i = 0; i < KS_NUM_THREAD; i++ ) {
    u_local[ i ] = (double*)malloc( sizeof(double) * nu );
    for ( int j = 0; j < nu; j++ ) {
      u_local[ i ][ j ] = 0.0;
    }
  }

  //printf( "Finish Initialize u_local\n" );


  // Initialize workload
  for ( int i = 0; i < KS_NUM_THREAD; i++ ) workload[ i ] = 0.0;
  

  // Enqueue with a load greedy load-balanced strategy.
  // TODO: implement weighted load-balanced stragegy in the future.
  for ( int i = 0; i < n_list; i++ ) {
    int    des     = 0;
    double cost    = ( (double)alist[ i ].size() ) * blist[ i ].size();
    double minload = workload[ des ];
    for ( int j = 0; j < KS_NUM_THREAD; j++ ) {
      if ( workload[ j ] < minload ) {
        des     = j;
        minload = workload[ j ];
      }
    }
    workload[ des ] += cost;
    jobs[ des ].push_back( i );
  }
  

  //printf( "Finish jobs distribution: %d workers\n", KS_NUM_THREAD );


  // Dequeue in parallel with omp parallel for
#pragma omp parallel for num_threads( KS_NUM_THREAD )
  for ( int i = 0; i < KS_NUM_THREAD; i++ ) {
    while ( !jobs[ i ].empty() ) {
      int tar = jobs[ i ].front();

      //printf( "job id = %d\n", tar );

      std::vector<int>& amap = alist[ tar ];
      std::vector<int>& umap = ulist[ tar ]; // New feature, a separate ulist
      std::vector<int>& bmap = blist[ tar ];
      std::vector<int>& wmap = wlist[ tar ];

      //printf( "amap.size() = %d, bmap.size() = %d\n", amap.size(), bmap.size() );

      if ( amap.size() != 0 && bmap.size() != 0 ) {
        dgsks(
            kernel,
            amap.size(),
            bmap.size(),
            k,
            u_local[ i ],
            umap.data(),
            XA,
            XA2,
            amap.data(),
            XB,
            XB2,
            bmap.data(),
            w,
            wmap.data()
            );
      }
      
      //printf( "Finish\n" );

      // Pop the job out to decrease the number of remaining jobs
      jobs[ i ].pop_front();
    }
  }

  // Merge u_local back to u in sequential
  for ( int i = 0; i < KS_NUM_THREAD; i++ ) {
    for ( int j = 0; j < nu; j++ ) {
      u[ j ] += u_local[ i ][ j ];
    }
  }

  // Free u_local and return
  for ( int i = 0; i < KS_NUM_THREAD; i++ ) {
    free( u_local[ i ] );
  }
}

