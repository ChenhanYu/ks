/*
 * --------------------------------------------------------------------------
 * GSKS (General Stride Kernel Summation)
 * --------------------------------------------------------------------------
 * Copyright (C) 2014, The University of Texas at Austin
 *
 * ks_util.c
 *
 * Chenhan D. Yu - Department of Computer Science, 
 *                 The University of Texas at Austin
 *
 *
 * Purpose: 
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
#include <ks.h>
#include <gsks_config.h>

#ifdef GSKS_MIC_AVX512
#include <hbwmalloc.h>
#endif

double *ks_malloc_aligned(
    int    m,
    int    n,
    int    size
    )
{
  double *ptr;
  int    err;

#ifdef GSKS_MIC_AVX512
  err = hbw_posix_memalign( (void**)&ptr, (size_t)DKS_SIMD_ALIGN_SIZE, size * m * n );
#else
  err = posix_memalign(     (void**)&ptr, (size_t)DKS_SIMD_ALIGN_SIZE, size * m * n );
#endif

  if ( err ) {
    printf( "ks_malloc_aligned(): posix_memalign() failures" );
    exit( 1 );
  }

  return ptr;
}
