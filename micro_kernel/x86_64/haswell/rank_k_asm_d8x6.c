/*
 * This file is modified and redistribued from 
 * 
 * BLIS
 * An object-based framework for developing high-performance BLAS-like
 * libraries.
 *
 * Copyright (C) 2014, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *  - Neither the name of The University of Texas at Austin nor the names
 *    of its contributors may be used to endorse or promote products
 *    derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 *
 * rank_k_asm_d8x6.c
 * 
 * Mofidifier:
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
 *
 * */


#include <immintrin.h> // AVX
#include <ks.h>

void rank_k_asm_s16x6(
    int    k,
    float  *a,
    float  *b,
    float  *c,
    int    ldc,
    aux_t  *aux
    )
{
  printf( "rank_k_asm_s16x6 not yet implemented.\n" );
}


void rank_k_asm_d8x6(
    int    k,
    double *a,
    double *b,
    double *c,
    int    ldc,
    aux_t  *aux
    )
{
  int    i, j, p;

  if ( aux->pc == 0 ) {
	for ( j = 0; j < 6; j ++ ) {
	  for ( i = 0; i < 8; i ++ ) {
		c[ j * 8 + i ] = 0.0;
	  }
	}
  }
  for ( p = 0; p < k; p ++ ) {
	for ( j = 0; j < 6; j ++ ) {
	  for ( i = 0; i < 8; i ++ ) {
		c[ j * 8 + i ] += a[ i ] * b [ j ];
	  }
	}
	a += 8;
	b += 8;
  }
}
