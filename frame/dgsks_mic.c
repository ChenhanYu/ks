/*
 * --------------------------------------------------------------------------
 * GSKS (General Stride Kernel Summation)
 * --------------------------------------------------------------------------
 * Copyright (C) 2014, The University of Texas at Austin
 *
 * dgsks_mic.c
 *
 * Chenhan D. Yu - Department of Computer Science, 
 *                 The University of Texas at Austin
 *
 *
 * Purpose: 
 * this is the main file of the double precision general stride
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
#include <ks.h>
#define min( i, j ) ( (i)<(j) ? (i): (j) )



struct thread_comm_s {
  void   *obj;
  double *packu;
  double *packA;
  double *packA2;
  double *packw;
  double *packB;
  double *packB2;
  int    nthd;
  int    signal;
  int    arrived;
  omp_lock_t lock;
};

typedef struct thread_comm_s ks_comm_t;


void packA_kcxmc(
    int    m,
    int    k,
    double *XA,
    int    ldXA,
    int    *amap,
    double *packA
    )
{

  int    i;
  double *a_i0_pntr;
  double *a_i1_pntr;
  double *a_i2_pntr;
  double *a_i3_pntr;
  double *a_i4_pntr;
  double *a_i5_pntr;
  double *a_i6_pntr;
  double *a_i7_pntr;

  double *packA_check = packA;

  //printf( "packA check#0\n" );
  if ( m > 7 ) {
    a_i0_pntr = XA + ldXA * amap[ 0 ];
    a_i1_pntr = XA + ldXA * amap[ 1 ];
    a_i2_pntr = XA + ldXA * amap[ 2 ];
    a_i3_pntr = XA + ldXA * amap[ 3 ];
    a_i4_pntr = XA + ldXA * amap[ 4 ];
    a_i5_pntr = XA + ldXA * amap[ 5 ];
    a_i6_pntr = XA + ldXA * amap[ 6 ];
    a_i7_pntr = XA + ldXA * amap[ 7 ];
  }
  else if ( m > 6 ) {
    a_i0_pntr = XA + ldXA * amap[ 0 ];
    a_i1_pntr = XA + ldXA * amap[ 1 ];
    a_i2_pntr = XA + ldXA * amap[ 2 ];
    a_i3_pntr = XA + ldXA * amap[ 3 ];
    a_i4_pntr = XA + ldXA * amap[ 4 ];
    a_i5_pntr = XA + ldXA * amap[ 5 ];
    a_i6_pntr = XA + ldXA * amap[ 6 ];
    a_i7_pntr = XA + ldXA * amap[ 0 ];
  }
  else if ( m > 5 ) {
    a_i0_pntr = XA + ldXA * amap[ 0 ];
    a_i1_pntr = XA + ldXA * amap[ 1 ];
    a_i2_pntr = XA + ldXA * amap[ 2 ];
    a_i3_pntr = XA + ldXA * amap[ 3 ];
    a_i4_pntr = XA + ldXA * amap[ 4 ];
    a_i5_pntr = XA + ldXA * amap[ 5 ];
    a_i6_pntr = XA + ldXA * amap[ 0 ];
    a_i7_pntr = XA + ldXA * amap[ 0 ];
  }
  else if ( m > 4 ) {
    a_i0_pntr = XA + ldXA * amap[ 0 ];
    a_i1_pntr = XA + ldXA * amap[ 1 ];
    a_i2_pntr = XA + ldXA * amap[ 2 ];
    a_i3_pntr = XA + ldXA * amap[ 3 ];
    a_i4_pntr = XA + ldXA * amap[ 4 ];
    a_i5_pntr = XA + ldXA * amap[ 0 ];
    a_i6_pntr = XA + ldXA * amap[ 0 ];
    a_i7_pntr = XA + ldXA * amap[ 0 ];
  }
  else if ( m > 3 ) {
    a_i0_pntr = XA + ldXA * amap[ 0 ];
    a_i1_pntr = XA + ldXA * amap[ 1 ];
    a_i2_pntr = XA + ldXA * amap[ 2 ];
    a_i3_pntr = XA + ldXA * amap[ 3 ];
    a_i4_pntr = XA + ldXA * amap[ 0 ];
    a_i5_pntr = XA + ldXA * amap[ 0 ];
    a_i6_pntr = XA + ldXA * amap[ 0 ];
    a_i7_pntr = XA + ldXA * amap[ 0 ];
  }
  else if ( m > 2 ) {
    a_i0_pntr = XA + ldXA * amap[ 0 ];
    a_i1_pntr = XA + ldXA * amap[ 1 ];
    a_i2_pntr = XA + ldXA * amap[ 2 ];
    a_i3_pntr = XA + ldXA * amap[ 0 ];
    a_i4_pntr = XA + ldXA * amap[ 0 ];
    a_i5_pntr = XA + ldXA * amap[ 0 ];
    a_i6_pntr = XA + ldXA * amap[ 0 ];
    a_i7_pntr = XA + ldXA * amap[ 0 ];
  }
  else if ( m > 1 ) {
    a_i0_pntr = XA + ldXA * amap[ 0 ];
    a_i1_pntr = XA + ldXA * amap[ 1 ];
    a_i2_pntr = XA + ldXA * amap[ 0 ];
    a_i3_pntr = XA + ldXA * amap[ 0 ];
    a_i4_pntr = XA + ldXA * amap[ 0 ];
    a_i5_pntr = XA + ldXA * amap[ 0 ];
    a_i6_pntr = XA + ldXA * amap[ 0 ];
    a_i7_pntr = XA + ldXA * amap[ 0 ];
  }
  else {
    a_i0_pntr = XA + ldXA * amap[ 0 ];
    a_i1_pntr = XA + ldXA * amap[ 0 ];
    a_i2_pntr = XA + ldXA * amap[ 0 ];
    a_i3_pntr = XA + ldXA * amap[ 0 ];
    a_i4_pntr = XA + ldXA * amap[ 0 ];
    a_i5_pntr = XA + ldXA * amap[ 0 ];
    a_i6_pntr = XA + ldXA * amap[ 0 ];
    a_i7_pntr = XA + ldXA * amap[ 0 ];
  }


  // loop over rows of XB
  for ( i = 0; i < k; i ++ ) {
    *packA ++ = *a_i0_pntr++;
    *packA ++ = *a_i1_pntr++;
    *packA ++ = *a_i2_pntr++;
    *packA ++ = *a_i3_pntr++;
    *packA ++ = *a_i4_pntr++;
    *packA ++ = *a_i5_pntr++;
    *packA ++ = *a_i6_pntr++;
    *packA ++ = *a_i7_pntr++;
  }
}












// Pack B from 4 different columns of XB
void packB_kcxnc(
    int    n,
    int    k,
    double *XB,
    int    ldXB, // ldXB is the original k
    int    *bmap,
    double *packB
    )
{
  int    i;
  double *b_i0_pntr;
  double *b_i1_pntr;
  double *b_i2_pntr;
  double *b_i3_pntr;
  double *b_i4_pntr;
  double *b_i5_pntr;
  double *b_i6_pntr;
  double *b_i7_pntr;
  double *b_i8_pntr;
  double *b_i9_pntr;
  double *b_i10_pntr;
  double *b_i11_pntr;
  double *b_i12_pntr;
  double *b_i13_pntr;
  double *b_i14_pntr;
  double *b_i15_pntr;
  double *b_i16_pntr;
  double *b_i17_pntr;
  double *b_i18_pntr;
  double *b_i19_pntr;
  double *b_i20_pntr;
  double *b_i21_pntr;
  double *b_i22_pntr;
  double *b_i23_pntr;
  double *b_i24_pntr;
  double *b_i25_pntr;
  double *b_i26_pntr;
  double *b_i27_pntr;
  double *b_i28_pntr;
  double *b_i29_pntr;
  double *packB_check = packB;

  {
    b_i0_pntr = XB + ldXB * bmap[ 0 ];
    b_i1_pntr = XB + ldXB * bmap[ 1 ];
    b_i2_pntr = XB + ldXB * bmap[ 2 ];
    b_i3_pntr = XB + ldXB * bmap[ 3 ];
    b_i4_pntr = XB + ldXB * bmap[ 4 ];
    b_i5_pntr = XB + ldXB * bmap[ 5 ];
    b_i6_pntr = XB + ldXB * bmap[ 6 ];
    b_i7_pntr = XB + ldXB * bmap[ 7 ];
    b_i8_pntr = XB + ldXB * bmap[ 8 ];
    b_i9_pntr = XB + ldXB * bmap[ 9 ];
    b_i10_pntr = XB + ldXB * bmap[ 10 ];
    b_i11_pntr = XB + ldXB * bmap[ 11 ];
    b_i12_pntr = XB + ldXB * bmap[ 12 ];
    b_i13_pntr = XB + ldXB * bmap[ 13 ];
    b_i14_pntr = XB + ldXB * bmap[ 14 ];
    b_i15_pntr = XB + ldXB * bmap[ 15 ];
    b_i16_pntr = XB + ldXB * bmap[ 16 ];
    b_i17_pntr = XB + ldXB * bmap[ 17 ];
    b_i18_pntr = XB + ldXB * bmap[ 18 ];
    b_i19_pntr = XB + ldXB * bmap[ 19 ];
    b_i20_pntr = XB + ldXB * bmap[ 20 ];
    b_i21_pntr = XB + ldXB * bmap[ 21 ];
    b_i22_pntr = XB + ldXB * bmap[ 22 ];
    b_i23_pntr = XB + ldXB * bmap[ 23 ];
    b_i24_pntr = XB + ldXB * bmap[ 24 ];
    b_i25_pntr = XB + ldXB * bmap[ 25 ];
    b_i26_pntr = XB + ldXB * bmap[ 26 ];
    b_i27_pntr = XB + ldXB * bmap[ 27 ];
    b_i28_pntr = XB + ldXB * bmap[ 28 ];
    b_i29_pntr = XB + ldXB * bmap[ 29 ];
  }

  for ( i = 0; i < k; i++ ) {
    *packB ++ = *b_i0_pntr++;
    *packB ++ = *b_i1_pntr++;
    *packB ++ = *b_i2_pntr++;
    *packB ++ = *b_i3_pntr++;
    *packB ++ = *b_i4_pntr++;
    *packB ++ = *b_i5_pntr++;
    *packB ++ = *b_i6_pntr++;
    *packB ++ = *b_i7_pntr++;
    *packB ++ = *b_i8_pntr++;
    *packB ++ = *b_i9_pntr++;
    *packB ++ = *b_i10_pntr++;
    *packB ++ = *b_i11_pntr++;
    *packB ++ = *b_i12_pntr++;
    *packB ++ = *b_i13_pntr++;
    *packB ++ = *b_i14_pntr++;
    *packB ++ = *b_i15_pntr++;
    *packB ++ = *b_i16_pntr++;
    *packB ++ = *b_i17_pntr++;
    *packB ++ = *b_i18_pntr++;
    *packB ++ = *b_i19_pntr++;
    *packB ++ = *b_i20_pntr++;
    *packB ++ = *b_i21_pntr++;
    *packB ++ = *b_i22_pntr++;
    *packB ++ = *b_i23_pntr++;
    *packB ++ = *b_i24_pntr++;
    *packB ++ = *b_i25_pntr++;
    *packB ++ = *b_i26_pntr++;
    *packB ++ = *b_i27_pntr++;
    *packB ++ = *b_i28_pntr++;
    *packB ++ = *b_i29_pntr++;
	packB += 2;
  }


}

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


  for ( j = 0; j < n; j += MIC_DKS_NR ) {
    j_next = j + MIC_DKS_NR;
    for ( i = 0; i < m; i += MIC_DKS_MR ) {
      if ( i + MIC_DKS_MR >= m ) {
        aux.b_next += MIC_DKS_NR * k;
      }

      ks_rank_k_int_d16x14(
          k,
          &packA[ i * k ],
          &packB[ j * k ],
          &packC[ j * ldc + i * MIC_DKS_NR ], // packed
          //&packC[ j * ldc + i ],        // nonpacked
          ldc,
          &aux
          );
    }
  }
}


// This macro kernel is called if k > DKS_KC. 
// packC is required, and it will be discarded after this micro kernel call.
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
    double *packC,
    int    ldc,
    int    pc
    )
{
  int    i, j, j_next;
  aux_t  aux;

  aux.pc = pc;
  aux.b_next = packB;

  switch ( kernel->type ) {
    case KS_GAUSSIAN:
      for ( j = 0; j < n; j += MIC_DKS_NR ) {

        j_next = j + MIC_DKS_NR;

        for ( i = 0; i < m; i += MIC_DKS_MR ) {
          if ( i + DKS_MR >= m ) {
            aux.b_next += MIC_DKS_NR * k;
          }

          ks_gaussian_int_d16x14_var2(
              k,
              kernel->scal,
              packu + i,
              packA2 + i,
              &packA[ i * k ],
              packB2 + j,
              &packB[ j * k ],
              packw + j,
              &packC[ j * ldc + i * MIC_DKS_NR ], // packed
              //&packC[ j * ldc + i ],        // nonpacked
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
    default:
      printf( "Error dgsks_macro_kernel_var2(): illegal kernel type\n" );
      exit( 1 );
  }
}


void dgsks_macro_kernel_var3(
	ks_t   *kernel,
	int    m,
	int    n,
	int    k,
	double *packu,
	double *packA,
	double *packA2,
	double *packB,
	double *packB2,
	double *packw
	)
{
  double c[ 8 * 30 ] __attribute__((aligned(64))) = { { 0.0 } };
  int    i, j, jp, j_next;
  //aux_t  aux[ 60 ];
  aux_t  aux;

  int    tid = omp_get_thread_num();
  int    iid = tid % MIC_KS_IR_NT;



  switch ( kernel->type ) {
	case KS_GAUSSIAN:
      //#pragma omp parallel for num_threads( 60 ) private( i, j, jp )
	  for ( i = iid * MIC_DKS_MR; i < m; i += MIC_KS_IR_NT * MIC_DKS_MR ) {        // 3.rd loop
		
		//aux[ tid ].a_next = packA + ( i + 60 * MIC_DKS_MR ) * k;
		//aux[ tid ].c_buff = &(c[ tid * 8 * 30 ]);

        aux.a_next = packA + ( i + MIC_KS_IR_NT * MIC_DKS_MR ) * k;
		aux.c_buff = c;


		for ( j = 0; j < n; j += MIC_DKS_NR ) {      // 2.nd loop
		  jp = ( j / MIC_DKS_NR ) * MIC_DKS_PACK_NR;

		  //aux[ tid ].b_next = packB + ( jp + MIC_DKS_PACK_NR ) * k;
          aux.b_next = packB + ( jp * MIC_DKS_PACK_NR ) * k;

		  ks_gaussian_asm_d8x30(
			  (unsigned long long)k,
			  kernel->scal,
			  packu + i,
			  packA2 + i,
			  &packA[ i * k ],
			  packB2 + jp,
			  &packB[ jp * k ],
			  packw + jp,
			  //aux + tid
			  &aux
			  );

		  //ks_rank_k_asm_d8x30(
		  //	(unsigned long long)k,
		  //	&packA[ i * k ],
		  //	&packB[ jp * k ],
		  //	c,
		  //	0,
		  //	aux + tid
		  //	);
		}
	  }
	  break;
	case KS_POLYNOMIAL:
	  printf( "Error dgsks_macro_kernel(): polynomial kernel hasn't been implemented.\n" );
	  break;
	case KS_LAPLACE:
	  printf( "Error dgsks_macro_kernel(): laplace kernel hasn't been implemented.\n" );
	  break;
	default:
	  printf( "Error dgsks_macro_kernel(): illegal kernel type\n" );
	  exit( 1 );
  }
}







// This macro kernel is called if k <= DKS_KC 
void dgsks_macro_kernel(
	ks_comm_t *comm,
    ks_t   *kernel,
    int    m,
    int    n,
    int    k,
    double *packu,
    double *packA,
    double *packA2,
    double *packB,
    double *packB2,
    double *packw
    )
{
  int    i, j, j_next, tid, lid;
  aux_t  aux;
  double *u_local;

  tid = omp_get_thread_num();
  lid = tid % 4;

  aux.b_next = packB;

  posix_memalign( (void**)&u_local, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * MIC_DKS_MC );

  for ( i = 0; i < MIC_DKS_MC; i ++ ) {
    u_local[ i ] = 0.0;
  }

  switch ( kernel->type ) {
    case KS_GAUSSIAN:

	  //printf( "in\n" );

	  for ( j = 0; j < n; j += MIC_DKS_NR ) {

		if ( ( j / MIC_DKS_NR ) % 4 == lid ) {

		  j_next = j + MIC_DKS_NR;

		  for ( i = 0; i < m; i += MIC_DKS_MR ) {
			if ( i + MIC_DKS_MR >= m ) {
			  aux.b_next += MIC_DKS_NR * k;
			}
			//ks_gaussian_int_d16x14(
			//ks_gaussian_int_d16x8(
			ks_gaussian_asm_d24x8(
				k,
				kernel->scal,
				//packu + i,
				u_local + i,
				packA2 + i,
				&packA[ i * k ],
				packB2 + j,
				&packB[ j * k ],
				packw + j,
				&aux
				);
		  }
		}
	  }

      


	  // Critical Section: reduction from 4 threads
	  //omp_set_lock( &(comm->lock) );
	  {
		//printf( "update packu: %d\n", tid );
		for ( i = 0; i < m ; i ++ ) {
		  packu[ i ] += u_local[ i ];
		}
	  }
	  //omp_unset_lock( &(comm->lock) );


	  //int my_signal = comm->signal;
	  //int my_arrived;

      //#pragma omp atomic capture
	  //  my_arrived = ++( comm->arrived );

	  //if ( my_arrived == comm->nthd ) {
	  //  comm->arrived = 0;
	  //  comm->signal = !comm->signal;
	  //}
	  //else {
	  //  volatile int *listener = &(comm->signal);
	  //  //printf( "listen\n" );
	  //  while ( *listener == my_signal ) {}
	  //}







	  break;
    case KS_POLYNOMIAL:
      printf( "Error dgsks_macro_kernel(): polynomial kernel hasn't been implemented.\n" );
      break;
    case KS_LAPLACE:
      printf( "Error dgsks_macro_kernel(): laplace kernel hasn't been implemented.\n" );
      break;
    default:
      printf( "Error dgsks_macro_kernel(): illegal kernel type\n" );
      exit( 1 );
  }

  free( u_local );
}

void dgsks_mic(
    ks_t   *kernel,
    int    m,
    int    n,
    int    k,
    double *u,
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
  int    pack_norm;
  int    ldc, padn;
  double *packA, *packB, *packC, *packw, *packu, *packA2, *packB2;


  // Manually allocate aligned memory for packA
  if ( posix_memalign( (void**)&packA, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * MIC_DKS_MC * MIC_DKS_KC * KS_NUM_THD_MC ) ) {
    printf( "dgsks_mic(): posix_memalign() failures" );
    exit( 1 );    
  }

  // Manually allocate aligned memory for packB
  if ( posix_memalign( (void**)&packB, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * MIC_DKS_KC * MIC_DKS_NC ) ) {
    printf( "dgsks_mic(): posix_memalign() failures" );
    exit( 1 );    
  }

  if ( posix_memalign( (void**)&packw, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * MIC_DKS_NC ) ) {
    printf( "dgsks_mic(): posix_memalign() failures" );
    exit( 1 );    
  }

  if ( posix_memalign( (void**)&packu, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * MIC_DKS_MC * KS_NUM_THD_MC ) ) {
    printf( "dgsks_mic(): posix_memalign() failures" );
    exit( 1 );    
  }

  if ( posix_memalign( (void**)&packA2, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * MIC_DKS_MC * KS_NUM_THD_MC ) ) {
    printf( "dgsks_mic(): posix_memalign() failures" );
    exit( 1 );    
  }

  if ( posix_memalign( (void**)&packB2, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * MIC_DKS_NC ) ) {
    printf( "dgsks_mic(): posix_memalign() failures" );
    exit( 1 );    
  }



  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) {
    printf( "dgsks_mic(): early return\n" );
    return;
  }


  switch ( kernel->type ) {
    case KS_GAUSSIAN:
      //printf( "dgsks(): Gaussian kernel\n" );
      pack_norm = 1;
      break;
    case KS_POLYNOMIAL:
      pack_norm = 0;
      break;
    case KS_LAPLACE:
      pack_norm = 1;
      if ( k < 3 ) {
        printf( "Error dgsks_mic(): laplace kernel only supports k > 2.\n" );
      }
      kernel->powe = 0.5 * ( 2.0 - (double)k );
      kernel->scal = tgamma( 0.5 * k + 1.0 ) / 
        ( (double)k * (double)( k - 2 ) * pow( M_PI, 0.5 * k ) );
      break;
    default:
      printf( "Error dgsks_mic(): illegal kernel type\n" );
      exit( 1 );
  }



  // In this case, we will store C with a temporary buffer
  //
  // Observation: PackA and PackB in the 4-th loop is not efficient
  // Solution: (possible) Allocate packC with posix_memalign.
  //           Use the same algorithm as before.
  //
  //           
  if ( k > MIC_DKS_KC ) {

    // Manually allocate aligned memory for packC
    ldc  = ( ( m - 1 ) / MIC_DKS_MR + 1 ) * MIC_DKS_MR;
    padn = MIC_DKS_NC;
    if ( n < MIC_DKS_NC ) {
      padn = ( ( n - 1 ) / MIC_DKS_NR + 1 ) * MIC_DKS_NR;
    }

    if ( posix_memalign( (void**)&packC, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
          sizeof(double) * ldc * padn ) ) {
      printf( "dgsks_mic(): posix_memalign() failures" );
      exit( 1 );    
    }


    for ( jc = 0; jc < n; jc += MIC_DKS_NC ) {           // 6-th loop
      jb = min( n - jc, MIC_DKS_NC );
      for ( pc = 0; pc < k; pc += MIC_DKS_KC ) {         // 5-th loop
        pb = min( k - pc, MIC_DKS_KC );

        // packB, packw, packbb
        for ( j = 0; j < jb; j += MIC_DKS_NR ) {
          
          if ( pc + MIC_DKS_KC >= k ) {
            // Initialize w
            for ( jr = 0; jr < MIC_DKS_NR; jr ++ ) {
              packw[ j + jr ] = 0.0;
            }
            // packw and packB2
            for ( jr = 0; jr < min( jb - j, MIC_DKS_NR ); jr ++ ) {
              packw[ j + jr ] = w[ wmap[ jc + j + jr ] ];
              if ( pack_norm ) {
                packB2[ j + jr ] = XB2[ bmap[ jc + j + jr ] ];
              }
            }
          }

          packB_kcxnc(
              min( jb - j, MIC_DKS_NR ),
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
        
        for ( ic = 0; ic < m; ic += MIC_DKS_MC ) {       // 4-th loop

          // Get the thread id ( 0 ~ 9 )
          int     tid = omp_get_thread_num();

          ib = min( m - ic, MIC_DKS_MC );
          for ( i = 0; i < ib; i += MIC_DKS_MR ) {
            if ( pc + MIC_DKS_KC >= k ) {
              for ( ir = 0; ir < min( ib - i, MIC_DKS_MR ); ir ++ ) {
                packu[ tid * MIC_DKS_MC + i + ir ] = u[ amap[ ic + i + ir ] ];
                //packu[ i + ir ] = u[ amap[ ic + i + ir ] ];
                if ( pack_norm ) {
                  packA2[ tid * MIC_DKS_MC + i + ir ] = XA2[ amap[ ic + i + ir ] ];
                  //packA2[ i + ir ] = XA2[ amap[ ic + i + ir ] ];
                }
              }
            }
            packA_kcxmc(
                min( ib - i, MIC_DKS_MR ),
                pb,
                &XA[ pc ],
                k,
                &amap[ ic + i ],
                &packA[ tid * MIC_DKS_MC * pb + i * pb ]
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
          if ( pc + MIC_DKS_KC < k ) {
            // call the macro kernel
            rank_k_macro_kernel(
                ib,
                jb,
                pb,
                packA + tid * MIC_DKS_MC * pb,
                //packA + tid * DKS_MC * DKS_KC,
                //packA,
                packB,
                &packC[ ic * padn ], // packed
                //&packC[ ic ],        // nonpacked
                ( ( ib - 1 ) / MIC_DKS_MR + 1 ) * MIC_DKS_MR, // packed
                //ldc,                                // nonpacked
                pc
                );
          }
          else {

            printf( "not yet implemented\n" );

            // call the macro kernel
            dgsks_macro_kernel_var2(                      // 1~3 loops
                kernel,
                ib,
                jb,
                pb,
                packu + tid * MIC_DKS_MC,
                //packu,
                packA + tid * MIC_DKS_MC * pb,
                //packA + tid * DKS_MC * DKS_KC,
                //packA,
                packA2 + tid * MIC_DKS_MC,
                //packA2,
                packB,
                packB2,
                packw,
                &packC[ ic * padn ],                // packed
                //&packC[ ic ],                       // nonpacked
                ( ( ib - 1 ) / MIC_DKS_MR + 1 ) * MIC_DKS_MR, // packed
                //ldc,                                // nonpacked
                pc
                );


            //printf( "Packu: %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", 
            //    packu[ 0 ], packu[ 1 ], packu[ 2 ], packu[ 3 ],
            //    packu[ 4 ], packu[ 5 ], packu[ 6 ], packu[ 7 ]
            //    );

            // unpacku
            for ( i = 0; i < ib; i += MIC_DKS_MR ) {
              for ( ir = 0; ir < min( ib - i, MIC_DKS_MR ); ir ++ ) {
                u[ amap[ ic + i + ir ] ] = packu[ tid * MIC_DKS_MC + i + ir ]; // This is possible a concurrent write.
                //u[ amap[ ic + i + ir ] ] = packu[ i + ir ];
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

    ks_comm_t comm[ 60 ];

    for ( i = 0; i < 60; i ++ ) {
	  comm[ i ].obj     = NULL;
	  comm[ i ].nthd    = 4;
	  comm[ i ].signal  = 1;
	  comm[ i ].arrived = 0;
	  omp_init_lock( &(comm[ i ].lock) );
	}



    for ( jc = 0; jc < n; jc += MIC_DKS_NC ) {           // 6-th loop
      jb = min( n - jc, MIC_DKS_NC );
      for ( pc = 0; pc < k; pc += MIC_DKS_KC ) {         // 5-th loop
        pb = min( k - pc, MIC_DKS_KC );

        // packB, packw, packbb
        for ( j = 0; j < jb; j += MIC_DKS_NR ) {
          // Initialize w
          for ( jr = 0; jr < MIC_DKS_NR; jr ++ ) {
            packw[ j + jr ] = 0.0;
          }
          // packw and packB2
          for ( jr = 0; jr < min( jb - j, MIC_DKS_NR ); jr ++ ) {
            packw[ j + jr ] = w[ wmap[ jc + j + jr ] ];
            if ( pack_norm ) {
              packB2[ j + jr ] = XB2[ bmap[ jc + j + jr ] ];
            }
          }
          //printf( "dgsks(): packB, jc = %d, j = %d, jb = %d, k = %d\n", jc, j, jb, k );
          //printf( "bmap pointer: jc + j = %d\n", jc + j );
          //printf( "packB pointer: j * k = %d\n", j * k );

          // packB
          packB_kcxnc(
              min( jb - j, MIC_DKS_NR ),
              pb,
              XB,
              k, // should be ldXB instead
              &bmap[ jc + j ],
              &packB[ j * k ]
              );
        }

        //printf( "dgsks(): 4-th loop, jc = %d, pc = %d\n", jc, pc );

#pragma omp parallel num_threads( 240 ) private( ic, ib, i, ir ) 
		{
		  int     tid = omp_get_thread_num();
		  int     gid = tid / 4;
		  int     lid = tid % 4;

//		  printf( "outside tid=%d\n", tid );

		  for ( ic = 0; ic < m; ic += MIC_DKS_MC ) {       // 4-th loop


			// Check if this mc block is belong to this thread.
			if ( ( ic / MIC_DKS_MC ) % 60 == gid ) {
			  ib = min( m - ic, MIC_DKS_MC );

			  //printf( "outside tid=%d, gid=%d, lid=%d\n", tid, gid, lid );



			  // Each core has 4 threads.
			  // Only the local 0.st thread will have to do the packing
			  if ( lid == 0 ) {
				for ( i = 0; i < ib; i += MIC_DKS_MR ) {
				  for ( ir = 0; ir < min( ib - i, MIC_DKS_MR ); ir ++ ) {
					packu[ gid * MIC_DKS_MC + i + ir ] = u[ amap[ ic + i + ir ] ];
					//packu[ i + ir ] = u[ amap[ ic + i + ir ] ];
					if ( pack_norm ) {
					  packA2[ gid * MIC_DKS_MC + i + ir ] = XA2[ amap[ ic + i + ir ] ];
					  //packA2[ i + ir ] = XA2[ amap[ ic + i + ir ] ];
					}
				  }
				  //printf( "i = %d, ib = %d, min = %d\n", i, ib, min( ib - i, DKS_MR ) );
				  packA_kcxmc(
					  min( ib - i, MIC_DKS_MR ),
					  pb,
					  XA,
					  k,
					  &amap[ ic + i ],
					  &packA[ gid * MIC_DKS_MC * pb + i * pb ]
					  //&packA[ tid * DKS_MC * DKS_KC + i * k ]
					  //&packA[ i * k ]
					  );
				}
				//printf( "PackA: %lf, %lf, %lf, %lf\n", packA[ 8 ], packA[ 9 ], packA[ 10 ], packA[ 11 ] );

			  }

			  // Every group () will wait for the 0.th thread to complete the
			  // packing routine. We set a barrier here.
			  // That is say that every threads with the same gid need to 
			  // sync here. Here we use busy polling to wait for the singal.
			  //if ( lid != 0 ) {
              //  while ( comm[ gid ].signal ) {
			  //    //printf( "%d waiting, signal = %d\n", gid, comm[ gid ].signal );
			  //  }
			  //}

              int my_signal = comm[ gid ].signal;
			  int my_arrived;

			  #pragma omp atomic capture
				my_arrived = ++( comm[ gid ].arrived );

			  if ( my_arrived == comm[ gid ].nthd ) {
				comm[ gid ].arrived = 0;
				comm[ gid ].signal = !comm[ gid ].signal;
			  }
			  else {
				volatile int *listener = &(comm[ gid ].signal);
				while ( *listener == my_signal ) {}
			  }



			  //printf( "%d\n", tid );

			  dgsks_macro_kernel(                      // 1~3 loops
				  &(comm[ gid ]),
				  kernel,
				  ib,
				  jb,
				  pb,
				  //packu,
				  packu + gid * MIC_DKS_MC,
				  //packA,
				  packA + gid * MIC_DKS_MC * pb,
				  //packA + tid * DKS_MC * DKS_KC,
				  //packA2,
				  packA2 + gid * MIC_DKS_MC,
				  packB,
				  packB2,
				  packw
				  );

			  //printf( "Here %d\n", tid );


			  // The 0.th thread needs to unpack u, but other threads are free to
			  // go to the next block if there is any.
			  if ( lid == 0 ) {
				for ( i = 0; i < ib; i += MIC_DKS_MR ) {
				  for ( ir = 0; ir < min( ib - i, MIC_DKS_MR ); ir ++ ) {
					u[ amap[ ic + i + ir ] ] = packu[ gid * MIC_DKS_MC + i + ir ]; // This is possible a concurrent write.
					//u[ amap[ ic + i + ir ] ] = packu[ i + ir ];
				  }
				}
			  }


			}                                    // End of Checking the coressponding task
		  }                                      // End of the 4.th loop
		}                                        // End of the omp parallel region
	  }                                          // End of the 5.th loop
	}                                            // End of the 6.th loop
  }


  free( packA );
  free( packB );
  free( packu );
  free( packw );
  free( packA2 );
  free( packB2 );
}



void dgsks_mic_var2(
    ks_t   *kernel,
    int    m,
    int    n,
    int    k,
    double *u,
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
  int    ic, ib, jc, jb, pc, pb, jp;
  int    ir, jr;
  int    pack_norm;
  int    ldc, padn;
  double *packA, *packB, *packC, *packw, *packu, *packA2, *packB2;
  double beg;
  double time_setup = 0.0;
  double time_ks    = 0.0;


  beg = omp_get_wtime();

  // Manually allocate aligned memory for packA
  if ( posix_memalign( (void**)&packA, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * MIC_DKS_PACK_MC * MIC_DKS_KC ) ) {
    printf( "dgsks_mic(): posix_memalign() failures" );
    exit( 1 );    
  }

  // Manually allocate aligned memory for packB
  if ( posix_memalign( (void**)&packB, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * MIC_DKS_KC * MIC_DKS_PACK_NC * 4 ) ) {
    printf( "dgsks_mic(): posix_memalign() failures" );
    exit( 1 );    
  }

  if ( posix_memalign( (void**)&packw, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * MIC_DKS_PACK_NC * 4 ) ) {
    printf( "dgsks_mic(): posix_memalign() failures" );
    exit( 1 );    
  }

  if ( posix_memalign( (void**)&packu, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * MIC_DKS_PACK_MC * 4 ) ) {
    printf( "dgsks_mic(): posix_memalign() failures" );
    exit( 1 );    
  }

  if ( posix_memalign( (void**)&packA2, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * MIC_DKS_PACK_MC ) ) {
    printf( "dgsks_mic(): posix_memalign() failures" );
    exit( 1 );    
  }

  if ( posix_memalign( (void**)&packB2, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
        sizeof(double) * MIC_DKS_PACK_NC * 4 ) ) {
    printf( "dgsks_mic(): posix_memalign() failures" );
    exit( 1 );    
  }


  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) {
    printf( "dgsks_mic(): early return\n" );
    return;
  }


  switch ( kernel->type ) {
    case KS_GAUSSIAN:
      //printf( "dgsks(): Gaussian kernel\n" );
      pack_norm = 1;
      break;
    case KS_POLYNOMIAL:
      pack_norm = 0;
      break;
    case KS_LAPLACE:
      pack_norm = 1;
      if ( k < 3 ) {
        printf( "Error dgsks_mic(): laplace kernel only supports k > 2.\n" );
      }
      kernel->powe = 0.5 * ( 2.0 - (double)k );
      kernel->scal = tgamma( 0.5 * k + 1.0 ) / 
        ( (double)k * (double)( k - 2 ) * pow( M_PI, 0.5 * k ) );
      break;
    default:
      printf( "Error dgsks_mic(): illegal kernel type\n" );
      exit( 1 );
  }


  time_setup += ( omp_get_wtime() - beg );

  beg = omp_get_wtime();
  {
	for ( ic = 0; ic < m; ic += MIC_DKS_MC ) {           // 6-th loop
	  ib = min( m - ic, MIC_DKS_MC );
	  for ( pc = 0; pc < k; pc += MIC_DKS_KC ) {         // 5-th loop
		pb = min( k - pc, MIC_DKS_KC );

		for ( i = 0; i < ib; i += MIC_DKS_MR ) {
		  for ( ir = 0; ir < min( ib - i, MIC_DKS_MR ); ir ++ ) {
			packu[ i + ir ] = u[ amap[ ic + i + ir ] ];
			if ( pack_norm ) {
			  packA2[ i + ir ] = XA2[ amap[ ic + i + ir ] ];
			}
		  }
		  packA_kcxmc(
			  min( ib - i, MIC_DKS_MR ),
			  pb,
			  XA,
			  k,
			  &amap[ ic + i ],
			  &packA[ i * pb ]
			  );
		}

        //#pragma omp parallel for num_threads( 4 ) private( jb, j, jp, jr, i, ir )
		for ( jc = 0; jc < n; jc += MIC_DKS_NC ) {       // 4-th loop
		  jb = min( n - jc, MIC_DKS_NC );

		  for ( j = 0; j < jb; j += MIC_DKS_NR ) {       // PackB, bb, w
			jp = ( j / MIC_DKS_NR ) * MIC_DKS_PACK_NR;
			for ( jr = 0; jr < MIC_DKS_NR; jr ++ ) {
			  packw[ jp + jr ] = 0.0;
			}
			for ( jr = 0; jr < min( jb - j, MIC_DKS_NR ); jr ++ ) {
			  packw[ jp + jr ] = w[ wmap[ jc + j + jr ] ];
			  if ( pack_norm ) {
				packB2[ jp + jr ] = XB2[ bmap[ jc + j + jr ] ];
			  }
			}
			packB_kcxnc(
				min( jb - j, MIC_DKS_NR ),
				pb,
				XB,
				k, // should be ldXB instead
				&bmap[ jc + j ],
				//&packB[ jp * k + gid * MIC_DKS_PACK_NC * pb ]
				&packB[ jp * k ]
				);
		  }


		  // We assume that we have private packu
		  dgsks_macro_kernel_var3(                       // 1~3 loops
			  kernel,
			  ib,
			  jb,
			  pb,
			  packu,
			  //packu + gid * MIC_DKS_MC,
			  packA,
			  //packA + gid * MIC_DKS_MC * pb,
			  //packA + tid * DKS_MC * DKS_KC,
			  packA2,
			  //packA2 + gid * MIC_DKS_MC,
			  packB,
			  //packB + gid * MIC_DKS_PACK_NC * pb,
			  packB2,
			  //packB2 + gid * MIC_DKS_PACK_NC,
			  packw
			  //packw + gid * MIC_DKS_PACK_NC
			  );
		}                                                // End of the 4.th loop (jc)

		for ( i = 0; i < ib; i += MIC_DKS_MR ) {         // Unpack u
		  for ( ir = 0; ir < min( ib - i, MIC_DKS_MR ); ir ++ ) {
			u[ amap[ ic + i + ir ] ] = packu[ i + ir ];
		  }
		}
	  }                                                  // End of the 5.th loop (pc)
	}                                                    // End of the 6.th loop (ic)
  }
  time_ks = omp_get_wtime() - beg;

  beg = omp_get_wtime();
  free( packA );
  free( packB );
  free( packu );
  free( packw );
  free( packA2 );
  free( packB2 );
  time_setup += ( omp_get_wtime() - beg );

  printf( "%5.3lf, %5.3lf sec\n", time_setup, time_ks );
}








void dgsks_mic_var3(
	ks_comm_t *comm,
    ks_t   *kernel,
    int    m,
    int    n,
    int    k,
    double *u,
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
  int    ic, ib, jc, jb, pc, pb, ip, jp;
  int    ir, jr;
  int    pack_norm;
  int    ldc, padn;
  double *packA, *packB, *packC, *packw, *packu, *packA2, *packB2;
  double beg;
  double time_setup = 0.0;
  double time_ks    = 0.0;

  int    tid, iid, jid, my_signal, my_arrive;

  //ks_comm_t comm;

  tid = omp_get_thread_num();
  iid = tid % MIC_KS_IR_NT;
  jid = tid / MIC_KS_IR_NT;


  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) {
	printf( "dgsks_mic(): early return\n" );
	return;
  }



  beg = omp_get_wtime();

  if ( iid == 0 ) {
	//printf( "master allocate %d, %d\n", iid, jid );

	// Manually allocate aligned memory for packA
	if ( posix_memalign( (void**)&comm->packA, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
		  sizeof(double) * MIC_DKS_PACK_MC * MIC_DKS_KC ) ) {
	  printf( "dgsks_mic(): posix_memalign() failures" );
	  exit( 1 );    
	}

	// Manually allocate aligned memory for packB
	if ( posix_memalign( (void**)&comm->packB, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
		  sizeof(double) * MIC_DKS_KC * MIC_DKS_PACK_NC ) ) {
	  printf( "dgsks_mic(): posix_memalign() failures" );
	  exit( 1 );    
	}

	if ( posix_memalign( (void**)&comm->packw, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
		  sizeof(double) * MIC_DKS_PACK_NC ) ) {
	  printf( "dgsks_mic(): posix_memalign() failures" );
	  exit( 1 );    
	}

	if ( posix_memalign( (void**)&comm->packu, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
		  sizeof(double) * MIC_DKS_PACK_MC ) ) {
	  printf( "dgsks_mic(): posix_memalign() failures" );
	  exit( 1 );    
	}

	if ( posix_memalign( (void**)&comm->packA2, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
		  sizeof(double) * MIC_DKS_PACK_MC ) ) {
	  printf( "dgsks_mic(): posix_memalign() failures" );
	  exit( 1 );    
	}

	if ( posix_memalign( (void**)&comm->packB2, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
		  sizeof(double) * MIC_DKS_PACK_NC ) ) {
	  printf( "dgsks_mic(): posix_memalign() failures" );
	  exit( 1 );    
	}

	//printf( "master allocate done %d, %d\n", iid, jid );
  }

  // =======================================================================
  // We need a barrier here
  // =======================================================================
  my_signal = comm->signal;

  #pragma omp atomic capture
    my_arrive = ++( comm->arrived );

  if ( my_arrive == comm->nthd ) {
	comm->arrived = 0;
	comm->signal = !comm->signal;
  }
  else {
	volatile int *listener = &(comm->signal);
	while ( *listener == my_signal ) {
	  //printf( "%d, %d, my_arrive = %d waiting\n", iid, jid, my_arrive );
	}
  }
  // =======================================================================



  // =======================================================================
  // Get pointers of the packing buffers.
  // =======================================================================
  packu  = comm->packu;
  packA  = comm->packA;
  packA2 = comm->packA2;
  packw  = comm->packw;
  packB  = comm->packB;
  packB2 = comm->packB2;
  // =======================================================================




  switch ( kernel->type ) {
	case KS_GAUSSIAN:
	  //printf( "dgsks(): Gaussian kernel\n" );
	  pack_norm = 1;
	  break;
	case KS_POLYNOMIAL:
	  pack_norm = 0;
	  break;
	case KS_LAPLACE:
	  pack_norm = 1;
	  if ( k < 3 ) {
		printf( "Error dgsks_mic(): laplace kernel only supports k > 2.\n" );
	  }
	  kernel->powe = 0.5 * ( 2.0 - (double)k );
	  kernel->scal = tgamma( 0.5 * k + 1.0 ) / 
		( (double)k * (double)( k - 2 ) * pow( M_PI, 0.5 * k ) );
	  break;
	default:
	  printf( "Error dgsks_mic(): illegal kernel type\n" );
	  exit( 1 );
  }
  time_setup += ( omp_get_wtime() - beg );



  




  // There are MIC_KS_JR_NT threads will execute this loop in total.
  for ( ic = 0; ic < m; ic += MIC_DKS_MC ) {           // 6-th loop
	ib = min( m - ic, MIC_DKS_MC );
	for ( pc = 0; pc < k; pc += MIC_DKS_KC ) {         // 5-th loop
	  pb = min( k - pc, MIC_DKS_KC );

	  // Parallel packing (every thread iid will pack a part of packA.
	  //
	  // e.g. 4-way parallelism
	  //
	  // iid = 0, i = 0,          4 * MIC_DKS_MR, 8 * MIC_DKS_MR, ...
	  // iid = 1, i = MIC_DKS_MR, 5 * MI _DKS_MR, 9 * MIC_DKS_MR, ...
	  // 
	  // ...
	  //
	  for ( i = iid * MIC_DKS_MR; i < ib; i += MIC_KS_IR_NT * MIC_DKS_MR ) {

		// Recompute the packing offset, since MIC_DKS_MR may be different from
		// MIC_DKS_PACK_MR.
		ip = ( i / MIC_DKS_MR ) * MIC_DKS_PACK_MR;

		for ( ir = 0; ir < min( ib - i, MIC_DKS_MR ); ir ++ ) {
		  // Use ip offset for the packing buffer, but use i and ic for
		  // asscessing the original memory.
		  packu[ ip + ir ] = u[ ic + i + ir ];
		  //packu[ ip + ir ] = u[ amap[ ic + i + ir ] ];
		  if ( pack_norm ) {
			packA2[ ip + ir ] = XA2[ amap[ ic + i + ir ] ];
		  }
		}

		packA_kcxmc(
			min( ib - i, MIC_DKS_MR ),
			pb,
			XA,
			k,
			&amap[ ic + i ],
			&packA[ ip * pb ]
			);
	  }


	  //// =======================================================================
	  //my_signal = comm->signal;

      //#pragma omp atomic capture
	  //  my_arrive = ++( comm->arrived );

	  //if ( my_arrive == comm->nthd ) {
	  //  comm->arrived = 0;
	  //  comm->signal = !comm->signal;
	  //}
	  //else {
	  //  volatile int *listener = &(comm->signal);
	  //  while ( *listener == my_signal ) {
	  //    //printf( "%d, %d, my_arrive = %d waiting\n", iid, jid, my_arrive );
	  //  }
	  //}
	  //// =======================================================================



	  for ( jc = 0; jc < n; jc += MIC_DKS_NC ) {       // 4-th loop
		jb = min( n - jc, MIC_DKS_NC );
	    // Only the master ( iid == 0 ) will need to packB.
	    if ( iid == 0 ) {
	      for ( j = 0; j < jb; j += MIC_DKS_NR ) {       // PackB, bb, w

	    	jp = ( j / MIC_DKS_NR ) * MIC_DKS_PACK_NR;

	    	for ( jr = 0; jr < MIC_DKS_PACK_NR; jr ++ ) {
	    	  packw[ jp + jr ] = 0.0;
	    	}

	    	for ( jr = 0; jr < min( jb - j, MIC_DKS_NR ); jr ++ ) {
	    	  packw[ jp + jr ] = w[ wmap[ jc + j + jr ] ];
	    	  if ( pack_norm ) {
	    		packB2[ jp + jr ] = XB2[ bmap[ jc + j + jr ] ];
	    	  }
	    	}

	    	packB_kcxnc(
	    		min( jb - j, MIC_DKS_NR ),
	    		pb,
	    		XB,
	    		k, // should be ldXB instead
	    		&bmap[ jc + j ],
	    		&packB[ jp * pb ]
	    		);
	      }
	    }

	    // =======================================================================
	    // We need a barrier here
	    // =======================================================================
	    my_signal = comm->signal;

        #pragma omp atomic capture
	      my_arrive = ++( comm->arrived );

	    if ( my_arrive == comm->nthd ) {
	      comm->arrived = 0;
	      comm->signal = !comm->signal;
	    }
	    else {
	      volatile int *listener = &(comm->signal);
	      while ( *listener == my_signal ) {
	    	//printf( "%d, %d, my_arrive = %d waiting\n", iid, jid, my_arrive );
	      }
	    }
	    // =======================================================================


	    dgsks_macro_kernel_var3(                       // 1~3 loops
	    	kernel,
	    	ib,
	    	jb,
	    	pb,
	    	packu,
	    	packA,
	    	packA2,
	    	packB,
	    	packB2,
	    	packw
	    	);

	    // =======================================================================
	    my_signal = comm->signal;

        #pragma omp atomic capture
	      my_arrive = ++( comm->arrived );

	    if ( my_arrive == comm->nthd ) {
	      comm->arrived = 0;
	      comm->signal = !comm->signal;
	    }
	    else {
	      volatile int *listener = &(comm->signal);
	      while ( *listener == my_signal ) {
	    	//printf( "%d, %d, my_arrive = %d waiting\n", iid, jid, my_arrive );
	      }
	    }
	    // =======================================================================



	  }                                                // End of 4.th loop 


	  // Parallel unpack u
	  for ( i = iid * MIC_DKS_MR; i < ib; i += MIC_KS_IR_NT * MIC_DKS_MR ) {
		ip = ( i / MIC_DKS_MR ) * MIC_DKS_PACK_MR;
		for ( ir = 0; ir < min( ib - i, MIC_DKS_MR ); ir ++ ) {
		  u[ ic + i + ir ] = packu[ ip + ir ];
		}
	  }


	}                                                  // End of 5.th loop
  }                                                    // End of 6.th loop


  // =======================================================================
  // We need a barrier here
  // =======================================================================
  my_signal = comm->signal;

  #pragma omp atomic capture
    my_arrive = ++( comm->arrived );

  if ( my_arrive == comm->nthd ) {
	comm->arrived = 0;
	comm->signal = !comm->signal;
  }
  else {
	volatile int *listener = &(comm->signal);
	while ( *listener == my_signal ) {
	  //printf( "%d, %d, my_arrive = %d waiting\n", iid, jid, my_arrive );
	}
  }
  // =======================================================================


  beg = omp_get_wtime();
  if ( iid == 0 ) {
	//printf( "master free %d, %d\n", iid, jid );
	free( comm->packA );
	free( comm->packB );
	free( comm->packu );
	free( comm->packw );
	free( comm->packA2 );
	free( comm->packB2 );
  }
  time_setup += ( omp_get_wtime() - beg );
  printf( "setup: %lf sec\n", time_setup  );
}




void dgsks_mic_parallel(
    ks_t   *kernel,
    int    m,
    int    n,
    int    k,
    double *u,
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
  int    i, j;
  double *u_local;
  ks_comm_t comm[ MIC_KS_JC_NT ];
  double beg, setup_time, kernel_time;

  setup_time = 0.0;
  kernel_time = 0.0;


  beg = omp_get_wtime();
  if ( posix_memalign( (void**)&u_local, (size_t)MIC_DKS_SIMD_ALIGN_SIZE, 
		sizeof(double) * m * MIC_KS_JC_NT ) ) {
	printf( "dgsks_mic(): posix_memalign() failures" );
	exit( 1 );    
  }
  setup_time += ( omp_get_wtime() - beg );

  // Zero out the u_local for each thread.
  #pragma omp parallel for num_threads( MIC_KS_JC_NT ) private( j, i )
  for ( j = 0; j < MIC_KS_JC_NT; j ++ ) {
    for ( i = 0; i < m; i++ ) {
	  u_local[ j * m + i ] = 0.0;
	}
    comm[ j ].signal  = 0;
    comm[ j ].arrived = 0;
	comm[ j ].nthd    = MIC_KS_IR_NT;
  }


  beg = omp_get_wtime();
  // Divide and conquare n.
  #pragma omp parallel num_threads( MIC_KS_IR_NT * MIC_KS_JC_NT )
  {
	int    tid = omp_get_thread_num();
	int    iid = tid % MIC_KS_IR_NT;
	int    jid = tid / MIC_KS_IR_NT;

	// Divide n into MIC_KS_JC_NT part.
    int    myn = ( n - 1 ) / MIC_KS_JC_NT + 1;
    int    n0  = jid * myn;
	int    n1  = n0 + myn;

	if ( jid == MIC_KS_JC_NT - 1 ) {
	  myn = n - ( MIC_KS_JC_NT - 1 ) * myn;
	  n1  = n0 + myn;
	}

    //printf( "%d, i = %d, j = %d, n0 = %d, n1 = %d, myn = %d\n", tid, iid, jid, n0, n1, myn );


	dgsks_mic_var3(
		comm + jid,
		kernel,
		m,
		myn,
		k,
		u_local + jid * m,
		XA,
		XA2,
		amap,
		XB,
		XB2,
		bmap + n0,
		w,
		wmap + n0
		);
  } // End of parallel region
  //kernel_time = ( omp_get_wtime() - beg );

  // Currently sequential
  #pragma omp parallel for num_threads( MIC_KS_IR_NT * MIC_KS_JC_NT ) private( i, j )
  for ( i = 0; i < m; i ++ ) {
    for ( j = 0; j < MIC_KS_JC_NT; j ++ ) {
	  u[ amap[ i ] ] += u_local[ j * m + i ];
    }
  }
  kernel_time = ( omp_get_wtime() - beg );

  beg = omp_get_wtime();
  free( u_local );
  setup_time += ( omp_get_wtime() - beg );

  printf( "kernel: %lf, setup: %lf\n", kernel_time, setup_time );
}
