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
 * dgsks.c
 *
 * Chenhan D. Yu - Department of Computer Science, 
 *                 The University of Texas at Austin
 *
 *
 * Purpose: 
 * this is the main file of the double precision general stride
 * kernel summation kernel.
 *
 *
 * Todo:
 *
 *
 * Modification:
 * Chenhan
 * Mar 16, 2015: Change the omp thread number control to get_env(). Split
 *               the rank-k update and exp kernels to head files.
 *               New tanh kernel.
 *
 * Chenhan
 * Mar 16, 2015: Change dgsks interface to support a separate ulist.
 *               Increase all packing buffer by 1 to deal with the preloading 
 *               segmentation fault in the rank-k update micro-kernel.
 *
 * Chenhan
 * May 05, 2015: Fix the double free bug in kernel->packh. Now I discard 
 *               kernel->packh, but allocate packh in dgsks. In this case,
 *               there is an additional parameter needed to be passed to
 *               the macro-kernel.
 *
 * */

#include <ks.h>
#define min( i, j ) ( (i)<(j) ? (i): (j) )


/* 
 * --------------------------------------------------------------------------
 * @brief  This is the packing routine that packs the target coordinates
 *         into a Z shape contiguous buffer.
 * --------------------------------------------------------------------------
 */
void packA_kcxmc(
    int    m,
    int    k,
    double *XA,
    int    ldXA,
    int    *amap,
    double *packA
    )
{
  int    i, p;
  //double *a_i0_pntr;
  //double *a_i1_pntr;
  //double *a_i2_pntr;
  //double *a_i3_pntr;
  //double *a_i4_pntr;
  //double *a_i5_pntr;
  //double *a_i6_pntr;
  //double *a_i7_pntr;

  //double *packA_check = packA;

  double *a_pntr[ DKS_MR ];

  for ( i = 0; i < m; i ++ ) {
    a_pntr[ i ] = XA + ldXA * amap[ i ];
  }

  for ( i = m; i < DKS_MR; i ++ ) {
    a_pntr[ i ] = XA + ldXA * amap[ 0 ];
  }


  //printf( "packA check#0\n" );
  //if ( m > 7 ) {
  //  a_i0_pntr = XA + ldXA * amap[ 0 ];
  //  a_i1_pntr = XA + ldXA * amap[ 1 ];
  //  a_i2_pntr = XA + ldXA * amap[ 2 ];
  //  a_i3_pntr = XA + ldXA * amap[ 3 ];
  //  a_i4_pntr = XA + ldXA * amap[ 4 ];
  //  a_i5_pntr = XA + ldXA * amap[ 5 ];
  //  a_i6_pntr = XA + ldXA * amap[ 6 ];
  //  a_i7_pntr = XA + ldXA * amap[ 7 ];
  //}
  //else if ( m > 6 ) {
  //  a_i0_pntr = XA + ldXA * amap[ 0 ];
  //  a_i1_pntr = XA + ldXA * amap[ 1 ];
  //  a_i2_pntr = XA + ldXA * amap[ 2 ];
  //  a_i3_pntr = XA + ldXA * amap[ 3 ];
  //  a_i4_pntr = XA + ldXA * amap[ 4 ];
  //  a_i5_pntr = XA + ldXA * amap[ 5 ];
  //  a_i6_pntr = XA + ldXA * amap[ 6 ];
  //  a_i7_pntr = XA + ldXA * amap[ 0 ];
  //}
  //else if ( m > 5 ) {
  //  a_i0_pntr = XA + ldXA * amap[ 0 ];
  //  a_i1_pntr = XA + ldXA * amap[ 1 ];
  //  a_i2_pntr = XA + ldXA * amap[ 2 ];
  //  a_i3_pntr = XA + ldXA * amap[ 3 ];
  //  a_i4_pntr = XA + ldXA * amap[ 4 ];
  //  a_i5_pntr = XA + ldXA * amap[ 5 ];
  //  a_i6_pntr = XA + ldXA * amap[ 0 ];
  //  a_i7_pntr = XA + ldXA * amap[ 0 ];
  //}
  //else if ( m > 4 ) {
  //  a_i0_pntr = XA + ldXA * amap[ 0 ];
  //  a_i1_pntr = XA + ldXA * amap[ 1 ];
  //  a_i2_pntr = XA + ldXA * amap[ 2 ];
  //  a_i3_pntr = XA + ldXA * amap[ 3 ];
  //  a_i4_pntr = XA + ldXA * amap[ 4 ];
  //  a_i5_pntr = XA + ldXA * amap[ 0 ];
  //  a_i6_pntr = XA + ldXA * amap[ 0 ];
  //  a_i7_pntr = XA + ldXA * amap[ 0 ];
  //}
  //else if ( m > 3 ) {
  //  a_i0_pntr = XA + ldXA * amap[ 0 ];
  //  a_i1_pntr = XA + ldXA * amap[ 1 ];
  //  a_i2_pntr = XA + ldXA * amap[ 2 ];
  //  a_i3_pntr = XA + ldXA * amap[ 3 ];
  //  a_i4_pntr = XA + ldXA * amap[ 0 ];
  //  a_i5_pntr = XA + ldXA * amap[ 0 ];
  //  a_i6_pntr = XA + ldXA * amap[ 0 ];
  //  a_i7_pntr = XA + ldXA * amap[ 0 ];
  //}
  //else if ( m > 2 ) {
  //  a_i0_pntr = XA + ldXA * amap[ 0 ];
  //  a_i1_pntr = XA + ldXA * amap[ 1 ];
  //  a_i2_pntr = XA + ldXA * amap[ 2 ];
  //  a_i3_pntr = XA + ldXA * amap[ 0 ];
  //  a_i4_pntr = XA + ldXA * amap[ 0 ];
  //  a_i5_pntr = XA + ldXA * amap[ 0 ];
  //  a_i6_pntr = XA + ldXA * amap[ 0 ];
  //  a_i7_pntr = XA + ldXA * amap[ 0 ];
  //}
  //else if ( m > 1 ) {
  //  a_i0_pntr = XA + ldXA * amap[ 0 ];
  //  a_i1_pntr = XA + ldXA * amap[ 1 ];
  //  a_i2_pntr = XA + ldXA * amap[ 0 ];
  //  a_i3_pntr = XA + ldXA * amap[ 0 ];
  //  a_i4_pntr = XA + ldXA * amap[ 0 ];
  //  a_i5_pntr = XA + ldXA * amap[ 0 ];
  //  a_i6_pntr = XA + ldXA * amap[ 0 ];
  //  a_i7_pntr = XA + ldXA * amap[ 0 ];
  //}
  //else {
  //  a_i0_pntr = XA + ldXA * amap[ 0 ];
  //  a_i1_pntr = XA + ldXA * amap[ 0 ];
  //  a_i2_pntr = XA + ldXA * amap[ 0 ];
  //  a_i3_pntr = XA + ldXA * amap[ 0 ];
  //  a_i4_pntr = XA + ldXA * amap[ 0 ];
  //  a_i5_pntr = XA + ldXA * amap[ 0 ];
  //  a_i6_pntr = XA + ldXA * amap[ 0 ];
  //  a_i7_pntr = XA + ldXA * amap[ 0 ];
  //}


  // loop over rows of XB
  for ( p = 0; p < k; p ++ ) {
    //*packA ++ = *a_i0_pntr++;
    //*packA ++ = *a_i1_pntr++;
    //*packA ++ = *a_i2_pntr++;
    //*packA ++ = *a_i3_pntr++;
    //*packA ++ = *a_i4_pntr++;
    //*packA ++ = *a_i5_pntr++;
    //*packA ++ = *a_i6_pntr++;
    //*packA ++ = *a_i7_pntr++;

    // No loop unrolling
    for ( i = 0; i < DKS_MR; i ++ ) {
      *packA ++ = *a_pntr[ i ] ++;
    }
  }
}


/* 
 * --------------------------------------------------------------------------
 * @brief  This is the packing routine that packs the source coordinates
 *         into a Z shape contiguous buffer.
 * --------------------------------------------------------------------------
 */
void packB_kcxnc(
    int    n,
    int    k,
    double *XB,
    int    ldXB, // ldXB is the original k
    int    *bmap,
    double *packB
    )
{
  int    j, p;
  //double *b_i0_pntr;
  //double *b_i1_pntr;
  //double *b_i2_pntr;
  //double *b_i3_pntr;
  double *packB_check = packB;

  double *b_pntr[ DKS_NR ];

  for ( j = 0; j < n; j ++ ) {
    b_pntr[ j ] = XB + ldXB * bmap[ j ];
  }

  for ( j = n; j < DKS_NR; j ++ ) {
    b_pntr[ j ] = XB + ldXB * bmap[ 0 ];
  }


  //if ( n > 3 ) {
  //  //printf( "packB n = 4\n" );
  //  //printf( "ldXB = %d, bmap[ 0 ] = %d, bmap[ 1 ] = %d, bmap[ 2 ] = %d, bmap[ 3 ] = %d\n", 
  //  //    ldXB, bmap[ 0 ], bmap[ 1 ], bmap[ 2 ], bmap[ 3 ] );
  //  b_i0_pntr = XB + ldXB * bmap[ 0 ];
  //  b_i1_pntr = XB + ldXB * bmap[ 1 ];
  //  b_i2_pntr = XB + ldXB * bmap[ 2 ];
  //  b_i3_pntr = XB + ldXB * bmap[ 3 ];
  //}
  //else if ( n > 2 ) {
  //  //printf( "packB n = 3\n" );
  //  b_i0_pntr = XB + ldXB * bmap[ 0 ];
  //  b_i1_pntr = XB + ldXB * bmap[ 1 ];
  //  b_i2_pntr = XB + ldXB * bmap[ 2 ];
  //  b_i3_pntr = XB + ldXB * bmap[ 0 ];
  //}
  //else if ( n > 1 ) {
  //  //printf( "packB n = 2\n" );
  //  b_i0_pntr = XB + ldXB * bmap[ 0 ];
  //  b_i1_pntr = XB + ldXB * bmap[ 1 ];
  //  b_i2_pntr = XB + ldXB * bmap[ 0 ];
  //  b_i3_pntr = XB + ldXB * bmap[ 0 ];
  //}
  //else {
  //  //printf( "packB n = 1\n" );
  //  b_i0_pntr = XB + ldXB * bmap[ 0 ];
  //  b_i1_pntr = XB + ldXB * bmap[ 0 ];
  //  b_i2_pntr = XB + ldXB * bmap[ 0 ];
  //  b_i3_pntr = XB + ldXB * bmap[ 0 ];
  //}

  //printf( "packB loop k = %d\n", k );

  // loop over rows of XB
  for ( p = 0; p < k; p ++ ) {
    //*packB ++ = *b_i0_pntr++;
    //*packB ++ = *b_i1_pntr++;
    //*packB ++ = *b_i2_pntr++;
    //*packB ++ = *b_i3_pntr++;

    for ( j = 0; j < DKS_NR; j ++ ) {
      *packB ++ = *b_pntr[ j ] ++;
    }
  }

  //printf( "packB end\n" );
}


/* 
 * --------------------------------------------------------------------------
 * @brief  This is macro-kernel of the rank-k update subroutine, containing
 *         the 3 rd and the 2 nd loop. In the very middle of the routine,
 *         a rank-k update micro-kernel will be called to compute the 
 *         1.st loop ( k loop ).
 *
 * @param  m       Number of target points
 * @param  n       Number of source points
 * @param  k       Data point dimension
 * @param  *packA  Packed target coordinates
 * @param  *packB  Packed source coordinates
 * @param  *packC  Packed accumulated rank-k update
 * @param  ldc     Leading dimension of packC
 * @param  pc      This is the 5.th loop counter which indicates whether
 *                 this macro-kernel is first call. The micro-kernel won't
 *                 load the packC if this is the first call.
 * --------------------------------------------------------------------------
 */
void rank_k_macro_kernel(
    int    m,
    int    n,
    int    k,
    double *packA,
    double *packB,
    double *packC,
    int    ldc,
    int    pc
    )
{
  int    i, j, j_next;
  aux_t  aux;

  aux.pc = pc;
  aux.b_next = packB;

  for ( j = 0; j < n; j += DKS_NR ) {
    j_next = j + DKS_NR;
    for ( i = 0; i < m; i += DKS_MR ) {
      if ( i + DKS_MR >= m ) {
        aux.b_next += DKS_NR * k;
      }

      //ks_rank_k_int_d8x4(
      ks_rank_k_asm_d8x4(
      //ks_rank_k_int_d8x4_unroll_4(
          k,
          &packA[ i * k ],
          &packB[ j * k ],
          &packC[ j * ldc + i * DKS_NR ], // packed
          //&packC[ j * ldc + i ],        // nonpacked
          ldc,
          &aux
          );
    }
  }
}


/* 
 * --------------------------------------------------------------------------
 * @brief  This is macro-kernel will be called if k > KC. The macro-kernel
 *         includes the 3.rd and the 2.nd loop, calling a mr x nr 
 *         micro-kernel which will load the accumulated rank-k update in 
 *         packC.
 *
 * @param  *kernel This structure is used to specified the type of the kernel.
 * @param  m       Number of target points
 * @param  n       Number of source points
 * @param  k       Data point dimension
 * @param  *packu  Packed potential vector, packu = u[ umap[] ];
 * @param  *packA  Packed target coordinates
 * @param  *packA2 Packed target square 2-norm
 * @param  *packB  Packed source coordinates
 * @param  *packB2 Packed source square 2-norm
 * @param  *packw  Packed weight vector, packw = w[ wmap[] ];
 * @param  *packC  Packed accumulated rank-k update
 * @param  ldc     Leading dimension of packC
 * @param  pc      This is the 5.th loop counter which indicates whether
 *                 this macro-kernel is first call. The micro-kernel won't
 *                 load the packC if this is the first call.
 * --------------------------------------------------------------------------
 */
void dgsks_macro_kernel_var2(
    ks_t   *kernel,
    int    m,
    int    n,
    int    k,
    double *packu,
    double *packA,
    double *packA2,
    double *packB,
    double *packB2,
    double *packw,
    double *packh,
    double *packC,
    int    ldc,
    int    pc
    )
{
  int    i, j, j_next, tid;
  aux_t  aux;

  aux.pc = pc;
  aux.b_next = packB;

  // We need to access private packh[ tid ]
  tid = omp_get_thread_num();

  switch ( kernel->type ) {
    case KS_GAUSSIAN:
      for ( j = 0; j < n; j += DKS_NR ) {
        j_next = j + DKS_NR;
        for ( i = 0; i < m; i += DKS_MR ) {
          if ( i + DKS_MR >= m ) {
            aux.b_next += DKS_NR * k;
          }

          ks_gaussian_asm_d8x4_var2(
          //ks_gaussian_svml_d8x4_var2(
              k,
              kernel->scal,
              packu + i,
              packA2 + i,
              &packA[ i * k ],
              packB2 + j,
              &packB[ j * k ],
              packw + j,
              &packC[ j * ldc + i * DKS_NR ], // packed
              //&packC[ j * ldc + i ],        // nonpacked
              ldc,
              &aux
              );
        }
      }
      break;
    case KS_GAUSSIAN_VAR_BANDWIDTH:
      for ( j = 0; j < n; j += DKS_NR ) {
        j_next = j + DKS_NR;
        for ( i = 0; i < m; i += DKS_MR ) {
          if ( i + DKS_MR >= m ) {
            aux.b_next += DKS_NR * k;
          }

          ks_variable_bandwidth_gaussian_asm_d8x4_var2(
              k,
              //kernel->packh + j,
              packh + j,
              packu + i,
              packA2 + i,
              &packA[ i * k ],
              packB2 + j,
              &packB[ j * k ],
              packw + j,
              &packC[ j * ldc + i * DKS_NR ],
              ldc,
              &aux
              );
        }
      }
      break;
    case KS_POLYNOMIAL:
      printf( "Error dgsks_macro_kernel_var2(): polynomial kernel hasn't been implemented.\n" );
      break;
    case KS_LAPLACE:
      printf( "Error dgsks_macro_kernel_var2(): laplace kernel hasn't been implemented.\n" );
      break;
    case KS_TANH:
      printf( "Error dgsks_macro_kernel_var2(): tanh kernel hasn't been implemented.\n" );
      break;
    case KS_QUARTIC:
      printf( "Error dgsks_macro_kernel_var2(): quartic kernel hasn't been implemented.\n" );
      break;
    case KS_MULTIQUADRATIC:
      printf( "Error dgsks_macro_kernel_var2(): multiquadratic kernel hasn't been implemented.\n" );
      break;
    case KS_EPANECHNIKOV:
      printf( "Error dgsks_macro_kernel_var2(): epanechnikov kernel hasn't been implemented.\n" );
      break;
    default:
      printf( "Error dgsks_macro_kernel_var2(): illegal kernel type\n" );
      exit( 1 );
  }
}


/* 
 * --------------------------------------------------------------------------
 * @brief  This is macro-kernel will be called if k <= KC. The macro-kernel
 *         includes the 3.rd and the 2.nd loop, calling a mr x nr 
 *         micro-kernel.
 *
 * @param  *kernel This structure is used to specified the type of the kernel.
 * @param  m       Number of target points
 * @param  n       Number of source points
 * @param  k       Data point dimension
 * @param  *packu  Packed potential vector, packu = u[ umap[] ];
 * @param  *packA  Packed target coordinates
 * @param  *packA2 Packed target square 2-norm
 * @param  *packB  Packed source coordinates
 * @param  *packB2 Packed source square 2-norm
 * @param  *packw  Packed weight vector, packw = w[ wmap[] ];
 * --------------------------------------------------------------------------
 */
void dgsks_macro_kernel(
    ks_t   *kernel,
    int    m,
    int    n,
    int    k,
    double *packu,
    double *packA,
    double *packA2,
    double *packB,
    double *packB2,
    double *packw,
    double *packh
    )
{
  int    i, j, j_next, tid;
  aux_t  aux;

  aux.b_next = packB;

  tid = omp_get_thread_num();

  switch ( kernel->type ) {
    case KS_GAUSSIAN:
      for ( j = 0; j < n; j += DKS_NR ) {
        j_next = j + DKS_NR;
        for ( i = 0; i < m; i += DKS_MR ) {
          if ( i + DKS_MR >= m ) {
            aux.b_next += DKS_NR * k;
          }
          //ks_gaussian_int_d8x4(
          //ks_gaussian_asm_d8x4(
          ks_gaussian_svml_d8x4(
              k,
              kernel->scal,
              packu + i,
              packA2 + i,
              &packA[ i * k ],
              packB2 + j,
              &packB[ j * k ],
              packw + j,
              &aux
              );
        }
      }
      break;
    case KS_GAUSSIAN_VAR_BANDWIDTH:
      for ( j = 0; j < n; j += DKS_NR ) {
        j_next = j + DKS_NR;
        for ( i = 0; i < m; i += DKS_MR ) {
          if ( i + DKS_MR >= m ) {
            aux.b_next += DKS_NR * k;
          }
          ks_variable_bandwidth_gaussian_int_d8x4(
              k,
              //kernel->packh + j,
              packh + j,
              packu + i,
              packA2 + i,
              &packA[ i * k ],
              packB2 + j,
              &packB[ j * k ],
              packw + j,
              &aux
              );
        }
      }
      break;
    case KS_POLYNOMIAL:
      //printf( "Error dgsks_macro_kernel(): polynomial kernel hasn't been implemented.\n" );
      for ( j = 0; j < n; j += DKS_NR ) {
        j_next = j + DKS_NR;
        for ( i = 0; i < m; i += DKS_MR ) {
          if ( i + DKS_MR >= m ) {
            aux.b_next += DKS_NR * k;
          }

          ks_polynomial_int_d8x4(
              k,
              kernel->powe,
              kernel->scal,
              kernel->cons,
              packu + i,
              packA2 + i,
              &packA[ i * k ],
              packB2 + j,
              &packB[ j * k ],
              packw + j,
              &aux
              );
        }
      }
      break;
    case KS_LAPLACE:
      //printf( "Error dgsks_macro_kernel(): laplace kernel hasn't been implemented.\n" );
      for ( j = 0; j < n; j += DKS_NR ) {
        j_next = j + DKS_NR;
        for ( i = 0; i < m; i += DKS_MR ) {
          if ( i + DKS_MR >= m ) {
            aux.b_next += DKS_NR * k;
          }
          ks_laplace3d_int_d8x4(
              k,
              kernel->powe,
              kernel->scal,
              packu + i,
              packA2 + i,
              &packA[ i * k ],
              packB2 + j,
              &packB[ j * k ],
              packw + j,
              &aux
              );
        }
      }
      break;
    case KS_TANH:
      //printf( "tanh micro-kernel\n" );
      for ( j = 0; j < n; j += DKS_NR ) {
        j_next = j + DKS_NR;
        for ( i = 0; i < m; i += DKS_MR ) {
          if ( i + DKS_MR >= m ) {
            aux.b_next += DKS_NR * k;
          }
          ks_tanh_int_d8x4(
              k,
              kernel->scal,
              kernel->cons,
              packu + i,
              &packA[ i * k ],
              &packB[ j * k ],
              packw + j,
              &aux
              );
        }
      }
      break;
    case KS_QUARTIC:
      //printf( "quartic micro-kernel\n" );
      for ( j = 0; j < n; j += DKS_NR ) {
        j_next = j + DKS_NR;
        for ( i = 0; i < m; i += DKS_MR ) {
          if ( i + DKS_MR >= m ) {
            aux.b_next += DKS_NR * k;
          }
          ks_quartic_int_d8x4(
              k,
              packu + i,
              packA2 + i,
              &packA[ i * k ],
              packB2 + j,
              &packB[ j * k ],
              packw + j,
              &aux
              );
        }
      }
      break;
    case KS_MULTIQUADRATIC:
      //printf( "multiquadratic micro-kernel\n" );
      for ( j = 0; j < n; j += DKS_NR ) {
        j_next = j + DKS_NR;
        for ( i = 0; i < m; i += DKS_MR ) {
          if ( i + DKS_MR >= m ) {
            aux.b_next += DKS_NR * k;
          }
          ks_multiquadratic_int_d8x4(
              k,
              kernel->cons,
              packu + i,
              packA2 + i,
              &packA[ i * k ],
              packB2 + j,
              &packB[ j * k ],
              packw + j,
              &aux
              );
        }
      }
      break;
    case KS_EPANECHNIKOV:
      //printf( "epanechnikov micro-kernel\n" );
      for ( j = 0; j < n; j += DKS_NR ) {
        j_next = j + DKS_NR;
        for ( i = 0; i < m; i += DKS_MR ) {
          if ( i + DKS_MR >= m ) {
            aux.b_next += DKS_NR * k;
          }
          ks_epanechnikov_int_d8x4(
              k,
              packu + i,
              packA2 + i,
              &packA[ i * k ],
              packB2 + j,
              &packB[ j * k ],
              packw + j,
              &aux
              );
        }
      }
      break;
    default:
      printf( "Error dgsks_macro_kernel(): illegal kernel type\n" );
      exit( 1 );
  }
}


/* 
 * --------------------------------------------------------------------------
 * @brief  This is the main routine of the double precision general stride
 *         kernel summation.
 *
 * @param  *kernel This structure is used to specified the type of the kernel.
 * @param  m       Number of target points
 * @param  n       Number of source points
 * @param  k       Data point dimension
 * @param  *u      Potential vector
 * @param  *umap   Potential vector index map
 * @param  *XA     Target coordinate table [ k * nxa ]
 * @param  *XA2    Target square 2-norm table
 * @param  *alpha  Target points index map
 * @param  *XB     Source coordinate table [ k * nxb ]
 * @param  *XB2    Source square 2-norm table
 * @param  *beta   Source points index map
 * @param  *w      Weight vector
 * @param  *omega  Weight vector index map
 * --------------------------------------------------------------------------
 */
void dgsks(
    ks_t   *kernel,
    int    m,
    int    n,
    int    k,
    double *u,
    int    *umap,         // New feature, a separate ulist
    double *XA,
    double *XA2,
    int    *amap,
    double *XB,
    double *XB2,
    int    *bmap,
    double *w,
    int    *wmap
    )
{
  int    i, j, p;
  int    ic, ib, jc, jb, pc, pb;
  int    ir, jr;
  int    pack_norm, pack_bandwidth, ks_ic_nt;
  int    ldc, padn;
  double *packA, *packB, *packC, *packh, *packw, *packu, *packA2, *packB2;
  char   *str;


  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) {
    printf( "dgsks(): early return\n" );
    return;
  }


  // Sequential is the default situation.
  ks_ic_nt = 1;

  
  // Check the environment variable.
  str = getenv( "KS_IC_NT" );
  if ( str != NULL ) {
    ks_ic_nt = (int)strtol( str, NULL, 10 );
  }


  packA  = ks_malloc_aligned( DKS_KC, ( DKS_MC + 1 ) * ks_ic_nt, sizeof(double) ); 
  packA2 = ks_malloc_aligned(      1, ( DKS_MC + 1 ) * ks_ic_nt, sizeof(double) ); 
  packu  = ks_malloc_aligned(      1, ( DKS_MC + 1 ) * ks_ic_nt, sizeof(double) ); 
  packB  = ks_malloc_aligned( DKS_KC, ( DKS_NC + 1 )           , sizeof(double) ); 
  packB2 = ks_malloc_aligned(      1, ( DKS_NC + 1 )           , sizeof(double) ); 
  packw  = ks_malloc_aligned(      1, ( DKS_NC + 1 )           , sizeof(double) ); 



  switch ( kernel->type ) {
    case KS_GAUSSIAN:
      //printf( "dgsks(): Gaussian kernel\n" );
      pack_bandwidth = 0;
      pack_norm      = 1;
      break;
    case KS_GAUSSIAN_VAR_BANDWIDTH:
      if ( !kernel->h ) {
        printf( "Error dgsks(): bandwidth vector has been initialized yet.\n" );
      }
      pack_bandwidth = 1;
      pack_norm      = 1;
      //kernel->packh = ks_malloc_aligned( 1, ( DKS_NC + 1 ), sizeof(double) ); 
      packh          = ks_malloc_aligned( 1, ( DKS_NC + 1 ), sizeof(double) ); 
      break;
    case KS_POLYNOMIAL:
      pack_bandwidth = 0;
      pack_norm      = 0;
      break;
    case KS_LAPLACE:
      pack_bandwidth = 0;
      pack_norm      = 1;
      if ( k < 3 ) {
        printf( "Error dgsks(): laplace kernel only supports k > 2.\n" );
      }
      kernel->powe = 0.5 * ( 2.0 - (double)k );
      kernel->scal = tgamma( 0.5 * k + 1.0 ) / 
        ( (double)k * (double)( k - 2 ) * pow( M_PI, 0.5 * k ) );
      break;
    case KS_TANH:
      pack_bandwidth = 0;
      pack_norm      = 0;
      break;
    case KS_QUARTIC:
      pack_bandwidth = 0;
      pack_norm      = 1;
      break;
    case KS_MULTIQUADRATIC:
      pack_bandwidth = 0;
      pack_norm      = 1;
      break;
    case KS_EPANECHNIKOV:
      pack_bandwidth = 0;
      pack_norm      = 1;
      break;
    default:
      printf( "Error dgsks(): illegal kernel type\n" );
      exit( 1 );
  }


  if ( k > DKS_KC ) {
    ldc  = ( ( m - 1 ) / DKS_MR + 1 ) * DKS_MR;
    padn = DKS_NC;
    if ( n < DKS_NC ) {
      padn = ( ( n - 1 ) / DKS_NR + 1 ) * DKS_NR;
    }

    // nonpacked
    packC = ks_malloc_aligned( ldc, padn, sizeof(double) ); 


    for ( jc = 0; jc < n; jc += DKS_NC ) {           // 6-th loop
      jb = min( n - jc, DKS_NC );
      for ( pc = 0; pc < k; pc += DKS_KC ) {         // 5-th loop
        pb = min( k - pc, DKS_KC );

        //printf( "Here, pb = %d\n", pb ); 

        // packB, packw, packbb
        #pragma omp parallel for num_threads( ks_ic_nt ) private( jr )
        for ( j = 0; j < jb; j += DKS_NR ) {
          
          if ( pc + DKS_KC >= k ) {
            // Initialize w
            for ( jr = 0; jr < DKS_NR; jr ++ ) {
              packw[ j + jr ] = 0.0;
            }
            // packw, packB2, packh (alternatively)
            for ( jr = 0; jr < min( jb - j, DKS_NR ); jr ++ ) {
              packw[ j + jr ] = w[ wmap[ jc + j + jr ] ];
              if ( pack_norm ) {
                packB2[ j + jr ] = XB2[ bmap[ jc + j + jr ] ];
              }
              if ( pack_bandwidth ) {
                //kernel->packh[ j + jr ] = kernel->h[ bmap[ jc + j + jr ] ];
                packh[ j + jr ] = kernel->h[ bmap[ jc + j + jr ] ];
              }
            }
          }

          packB_kcxnc(
              min( jb - j, DKS_NR ),
              pb,
              &XB[ pc ],
              k, // should be ldXB instead
              &bmap[ jc + j ],
              //&packB[ j * k ]
              &packB[ j * pb ]
              );
        }

        //printf( "After PackB\n" );
        //printf( "PackB: %lf, %lf, %lf, %lf\n", packB[ 0 ], packB[ 1 ], packB[ 2 ], packB[ 3 ] );
        //printf( "PackB: %lf, %lf, %lf, %lf\n", packB[ 4 ], packB[ 5 ], packB[ 6 ], packB[ 7 ] );
        
        #pragma omp parallel for num_threads( ks_ic_nt ) private( ic, ib, i, ir )
        for ( ic = 0; ic < m; ic += DKS_MC ) {       // 4-th loop

          // Get the thread id ( 0 ~ 9 )
          int     tid = omp_get_thread_num();
//          int     tid = 0;

          ib = min( m - ic, DKS_MC );
          for ( i = 0; i < ib; i += DKS_MR ) {
            if ( pc + DKS_KC >= k ) {
              for ( ir = 0; ir < min( ib - i, DKS_MR ); ir ++ ) {
                // -----------------------------------------------------------------
                // Unified ulist ( u and A share amap ) 
                // ----------------------------------------------------------------- 
                //packu[ tid * DKS_MC + i + ir ] = u[ amap[ ic + i + ir ] ];
                //packu[ i + ir ] = u[ amap[ ic + i + ir ] ];
                // -----------------------------------------------------------------
                // Separate ulist ( u has a separate umap )
                // -----------------------------------------------------------------
                packu[ tid * DKS_MC + i + ir ] = u[ umap[ ic + i + ir ] ];
                // -----------------------------------------------------------------
                if ( pack_norm ) {
                  packA2[ tid * DKS_MC + i + ir ] = XA2[ amap[ ic + i + ir ] ];
                  //packA2[ i + ir ] = XA2[ amap[ ic + i + ir ] ];
                }
              }
            }
            packA_kcxmc(
                min( ib - i, DKS_MR ),
                pb,
                &XA[ pc ],
                k,
                &amap[ ic + i ],
                &packA[ tid * DKS_MC * pb + i * pb ]
                //&packA[ tid * DKS_MC * DKS_KC + i * pb ]
                //&packA[ i * pb ]
                );
          }
          //printf( "After PackA\n" );
          //printf( "PackA: %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", 
          //    packA[ 0 ], packA[ 1 ], packA[ 2 ], packA[ 3 ],
          //    packA[ 4 ], packA[ 5 ], packA[ 6 ], packA[ 7 ]
          //    );
          //printf( "PackA: %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", 
          //    packA[ 8 ], packA[ 9 ], packA[ 10 ], packA[ 11 ],
          //    packA[ 12 ], packA[ 13 ], packA[ 14 ], packA[ 15 ]
          //    );

          // Check if this is the last kc interation
          if ( pc + DKS_KC < k ) {
            // call the macro kernel
            rank_k_macro_kernel(
                ib,
                jb,
                pb,
                packA + tid * DKS_MC * pb,
                //packA + tid * DKS_MC * DKS_KC,
                //packA,
                packB,
                &packC[ ic * padn ], // packed
                //&packC[ ic ],        // nonpacked
                ( ( ib - 1 ) / DKS_MR + 1 ) * DKS_MR, // packed
                //ldc,                                // nonpacked
                pc
                );
          }
          else {

            /* call the macro kernel */
            dgsks_macro_kernel_var2(                      // 1~3 loops
                kernel,
                ib,
                jb,
                pb,
                packu + tid * DKS_MC,
                //packu,
                packA + tid * DKS_MC * pb,
                //packA + tid * DKS_MC * DKS_KC,
                //packA,
                packA2 + tid * DKS_MC,
                //packA2,
                packB,
                packB2,
                packw,
                packh,
                &packC[ ic * padn ],                // packed
                //&packC[ ic ],                       // nonpacked
                ( ( ib - 1 ) / DKS_MR + 1 ) * DKS_MR, // packed
                //ldc,                                // nonpacked
                pc
                );


            //printf( "Packu: %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", 
            //    packu[ 0 ], packu[ 1 ], packu[ 2 ], packu[ 3 ],
            //    packu[ 4 ], packu[ 5 ], packu[ 6 ], packu[ 7 ]
            //    );

            /* Unpack u */
            for ( i = 0; i < ib; i += DKS_MR ) {
              for ( ir = 0; ir < min( ib - i, DKS_MR ); ir ++ ) {
                // -----------------------------------------------------------------
                // Unified ulist ( u and A share amap ) 
                // ----------------------------------------------------------------- 
                //u[ amap[ ic + i + ir ] ] = packu[ tid * DKS_MC + i + ir ];
                //u[ amap[ ic + i + ir ] ] = packu[ i + ir ];
                // -----------------------------------------------------------------
                // Separate ulist ( u has a separate umap )
                // -----------------------------------------------------------------
                u[ umap[ ic + i + ir ] ] = packu[ tid * DKS_MC + i + ir ];
                // -----------------------------------------------------------------
              }
            }
          }
        }
      }
    }

    free( packC );
  }
  else {

    //printf( "Before dgsks main loop\n" );

    for ( jc = 0; jc < n; jc += DKS_NC ) {           // 6-th loop
      jb = min( n - jc, DKS_NC );
      for ( pc = 0; pc < k; pc += DKS_KC ) {         // 5-th loop
        pb = min( k - pc, DKS_KC );

        // packB, packw, packbb
        #pragma omp parallel for num_threads( ks_ic_nt ) private( jr )
        for ( j = 0; j < jb; j += DKS_NR ) {
          // Initialize w
          for ( jr = 0; jr < DKS_NR; jr ++ ) {
            packw[ j + jr ] = 0.0;
          }
          //printf( "dgsks(): packw & packB2, j = %d, jb = %d\n", j, jb );
          // packw and packB2
          for ( jr = 0; jr < min( jb - j, DKS_NR ); jr ++ ) {
            //printf( "dgsks(): packw & packB2, j = %d, jr = %d\n", j, jr );
            packw[ j + jr ] = w[ wmap[ jc + j + jr ] ];
            if ( pack_norm ) {
              packB2[ j + jr ] = XB2[ bmap[ jc + j + jr ] ];
            }
            if ( pack_bandwidth ) {
              //kernel->packh[ j + jr ] = kernel->h[ bmap[ jc + j + jr ] ];
              packh[ j + jr ] = kernel->h[ bmap[ jc + j + jr ] ];
            }
          }
          //printf( "dgsks(): packB, jc = %d, j = %d, jb = %d, k = %d\n", jc, j, jb, k );
          //printf( "bmap pointer: jc + j = %d\n", jc + j );
          //printf( "packB pointer: j * k = %d\n", j * k );

          // packB
          packB_kcxnc(
              min( jb - j, DKS_NR ),
              pb,
              XB,
              k, // should be ldXB instead
              &bmap[ jc + j ],
              &packB[ j * k ]
              );
        }

        //printf( "dgsks(): 4-th loop, jc = %d, pc = %d\n", jc, pc );

        #pragma omp parallel for num_threads( ks_ic_nt ) private( ic, ib, i, ir )
        for ( ic = 0; ic < m; ic += DKS_MC ) {       // 4-th loop

          // Get the thread id ( 0 ~ 9 )
          int     tid = omp_get_thread_num();
          //int     tid = 0;

          ib = min( m - ic, DKS_MC );
          for ( i = 0; i < ib; i += DKS_MR ) {
            for ( ir = 0; ir < min( ib - i, DKS_MR ); ir ++ ) {

              // -----------------------------------------------------------------
              // Unified ulist ( u and A share amap ) 
              // ----------------------------------------------------------------- 
              //packu[ tid * DKS_MC + i + ir ] = u[ amap[ ic + i + ir ] ];
              //packu[ i + ir ] = u[ amap[ ic + i + ir ] ];
              // -----------------------------------------------------------------
              // Separate ulist ( u has a separate umap )
              // -----------------------------------------------------------------
              packu[ tid * DKS_MC + i + ir ] = u[ umap[ ic + i + ir ] ];
              // ----------------------------------------------------------------- 
              if ( pack_norm ) {
                packA2[ tid * DKS_MC + i + ir ] = XA2[ amap[ ic + i + ir ] ];
                //packA2[ i + ir ] = XA2[ amap[ ic + i + ir ] ];
              }
            }
            //printf( "i = %d, ib = %d, min = %d\n", i, ib, min( ib - i, DKS_MR ) );
            packA_kcxmc(
                min( ib - i, DKS_MR ),
                pb,
                XA,
                k,
                &amap[ ic + i ],
                &packA[ tid * DKS_MC * pb + i * pb ]
                //&packA[ tid * DKS_MC * DKS_KC + i * k ]
                //&packA[ i * k ]
                );
          }
          //printf( "PackA: %lf, %lf, %lf, %lf\n", packA[ 8 ], packA[ 9 ], packA[ 10 ], packA[ 11 ] );

          dgsks_macro_kernel(                      // 1~3 loops
              kernel,
              ib,
              jb,
              pb,
              //packu,
              packu + tid * DKS_MC,
              //packA,
              packA + tid * DKS_MC * pb,
              //packA + tid * DKS_MC * DKS_KC,
              //packA2,
              packA2 + tid * DKS_MC,
              packB,
              packB2,
              packw,
              packh
              );

          for ( i = 0; i < ib; i += DKS_MR ) {
            for ( ir = 0; ir < min( ib - i, DKS_MR ); ir ++ ) {
              // -----------------------------------------------------------------
              // Unified ulist ( u and A share amap ) 
              // ----------------------------------------------------------------- 
              //u[ amap[ ic + i + ir ] ] = packu[ tid * DKS_MC + i + ir ]; // This is possible a concurrent write.
              //u[ amap[ ic + i + ir ] ] = packu[ i + ir ];
              // -----------------------------------------------------------------
              // Separate ulist ( u has a separate umap )
              // -----------------------------------------------------------------
              u[ umap[ ic + i + ir ] ] = packu[ tid * DKS_MC + i + ir ];
              // -----------------------------------------------------------------
            }
          }
        }
      }
    }

  }


  // -----------------------------------------------------------------
  // Free all packing buffers.
  // -----------------------------------------------------------------
  free( packA );
  free( packB );
  free( packu );
  free( packw );
  free( packA2 );
  free( packB2 );
  // -----------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Free kernel dependent buffers
  // ------------------------------------------------------------------------
  switch ( kernel->type ) {
    case KS_GAUSSIAN:
      break;
    case KS_GAUSSIAN_VAR_BANDWIDTH:
      //free( kernel->packh );
      free( packh );
      break;
    case KS_POLYNOMIAL:
      break;
    case KS_LAPLACE:
      break;
    case KS_TANH:
      break;
    case KS_QUARTIC:
      break;
    case KS_MULTIQUADRATIC:
      break;
    case KS_EPANECHNIKOV:
      break;
    default:
      printf( "Error dgsks(): illegal kernel type\n" );
      exit( 1 );
  }
  // ------------------------------------------------------------------------
}
