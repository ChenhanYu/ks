#include <immintrin.h> // AVX
#include <ks.h>


const int    perm[ 16 ] __attribute__((aligned(64))) = { 8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7 };
const int    mask[ 16 ] __attribute__((aligned(64)))  
	       = { 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 
	           0, 0, 0, 0, 0, 0, 0, 0 };
//double c[ 24 * 8 ] __attribute__((aligned(64)));

void ks_gaussian_asm_d24x8(
    int    k,
    double alpha,
    double *u,
    double *aa,
    double *a,
    double *bb,
    double *b,
    double *w,
    aux_t  *aux
    )
{

//  int    perm[ 16 ] __attribute__((aligned(64))) 
//	       = { 8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7 };
//  int    mask[ 16 ] __attribute__((aligned(64)))  
//	       = { 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 
//	           0, 0, 0, 0, 0, 0, 0, 0 };

  //printf( "inside micro\n" );

  int    i;
  int    izero = 0;
  const double neg2   = -2.0;
  const double dzero  =  0.0;
  const double dmone  = -1.0;
  const double log2e  =  1.4426950408889634073599;
  const double maxlog =  7.09782712893383996843e2; // log( 2**1024 )
  const double minlog = -7.08396418532264106224e2; // log( 2**-1024 )
  const double c1     = -6.93145751953125E-1;
  const double c2     = -1.42860682030941723212E-6;

  // Original Remez Order 11 coefficients
  const double w11    =  3.5524625185478232665958141148891055719216674475023e-8;
  const double w10    =  2.5535368519306500343384723775435166753084614063349e-7;
  const double w9     =  2.77750562801295315877005242757916081614772210463065e-6;
  const double w8     =  2.47868893393199945541176652007657202642495832996107e-5;
  const double w7     =  1.98419213985637881240770890090795533564573406893163e-4;
  const double w6     =  1.3888869684178659239014256260881685824525255547326e-3;
  const double w5     =  8.3333337052009872221152811550156335074160546333973e-3;
  const double w4     =  4.1666666621080810610346717440523105184720007971655e-2;
  const double w3     =  0.166666666669960803484477734308515404418108830469798;
  const double w2     =  0.499999999999877094481580370323249951329122224389189;
  const double w1     =  1.0000000000000017952745258419615282194236357388884;
  const double w0     =  0.99999999999999999566016490920259318691496540598896;



  double *ptr_alpha  = &alpha;
  const double *ptr_neg2   = &neg2;
  const double *ptr_minlog = &minlog;
  const double *ptr_log2e  = &log2e;
  const double *ptr_c1     = &c1;
  const double *ptr_c2     = &c2;
  const double *ptr_dmone  = &dmone;
  const double *ptr_w11    = &w11;
  const double *ptr_w10    = &w10;
  const double *ptr_w9     = &w9;
  const double *ptr_w8     = &w8;
  const double *ptr_w7     = &w7;
  const double *ptr_w6     = &w6;
  const double *ptr_w5     = &w5;
  const double *ptr_w4     = &w4;
  const double *ptr_w3     = &w3;
  const double *ptr_w2     = &w2;
  const double *ptr_w1     = &w1;
  const double *ptr_w0     = &w0;

  double *ptr_perm = (double*) perm;
  double *ptr_mask = (double*) mask;
  //double *ptr_c    = (double*) c;

  unsigned long long k_iter = k;

  /*
   * zmm0  ~ zmm7  are c007, 0:7
   * zmm8  ~ zmm15 are c815, 0:7
   * zmm16 ~ zmm23 are c623, 0:7
   *
   * zmm24 ~ zmm26 are a007, a815, a623 ( aa007, aa815, aa623 )
   * zmm27 ~ zmm29 are A007, A815, A623 ( neg2, l2e, alpha, dzero, izero )
   * zmm30 ~ zmm31 are b03, b47
   *
   * zmm24 ~ zmm27 are a0, a1, a2, a3
   * */




  __asm
  {
	vpxord  zmm0,  zmm0, zmm0
	vmovaps zmm1,  zmm0  //clear out registers
	vmovaps zmm2,  zmm0 
	mov     rsi,   k_iter         //loop index
	vmovaps zmm3,  zmm0 
	vmovaps zmm4,  zmm0 
	vmovaps zmm5,  zmm0 
	mov     rax,   a              //load address of a
	vmovaps zmm6,  zmm0 
	mov     rbx,   b              //load address of b
	vmovaps zmm7,  zmm0 
    vmovaps zmm8,  zmm0 
    vprefetch0     0[rax]         // prefetch a
    vmovaps zmm9,  zmm0 
    vmovaps zmm10, zmm0 
    vmovaps zmm11, zmm0 
    vmovaps zmm12, zmm0 
    vprefetch0     0[rbx]         // prefetch b
    vmovaps zmm13, zmm0 
    vmovaps zmm14, zmm0 
    vmovaps zmm15, zmm0 
    vmovaps zmm16, zmm0 
    vmovaps zmm17, zmm0 
    vmovaps zmm18, zmm0 
    vmovaps zmm19, zmm0 
    vmovaps zmm20, zmm0 
    vmovaps zmm21, zmm0 
	cmp     rsi,   0              // Check if we can return earlier.
    vmovaps zmm22, zmm0 
    vmovaps zmm23, zmm0 

    vprefetch0     64[rax]         // prefetch a
    vbroadcastf64x4    zmm30,  0[rbx]          // b0303
    vmovapd            zmm24,  0[rax]          // a007

	je END



    LOOPMAIN:
    vprefetch1         3*64[rax]               // prefetch a
    vbroadcastf64x4    zmm31, 32[rbx]          // b4747
    vfmadd231pd zmm0,  zmm24, zmm30{aaaa}      // c007_0
    vfmadd231pd zmm1,  zmm24, zmm30{bbbb}      // c007_1
    vfmadd231pd zmm2,  zmm24, zmm30{cccc}      // c007_2
    vfmadd231pd zmm3,  zmm24, zmm30{dddd}      // c007_3
    vmovapd            zmm25, 64[rax]          // a815
    vprefetch0         2*64[rax]               // prefetch a
    vfmadd231pd zmm4,  zmm24, zmm31{aaaa}      // c007_4
    vfmadd231pd zmm5,  zmm24, zmm31{bbbb}      // c007_5
    vfmadd231pd zmm6,  zmm24, zmm31{cccc}      // c007_6
    vfmadd231pd zmm7,  zmm24, zmm31{dddd}      // c007_7

    vprefetch1         4*64[rax]               // prefetch a
    vmovapd            zmm26, 128[rax]         // a623
    vfmadd231pd zmm8,  zmm25, zmm30{aaaa}      // c815_0
    vfmadd231pd zmm9,  zmm25, zmm30{bbbb}      // c815_1
    vfmadd231pd zmm10, zmm25, zmm30{cccc}      // c815_2
    vfmadd231pd zmm11, zmm25, zmm30{dddd}      // c815_3
    vmovapd            zmm24, 192[rax]         // a623
    vfmadd231pd zmm16, zmm26, zmm30{aaaa}      // c623_0
    vfmadd231pd zmm17, zmm26, zmm30{bbbb}      // c623_1
    add         rax,   192                     // a += 24
    vprefetch0         0[rax]                  // prefetch a
    vfmadd231pd zmm18, zmm26, zmm30{cccc}      // c623_2
    vfmadd231pd zmm19, zmm26, zmm30{dddd}      // c623_3

    vprefetch1         5*64[rax]               // prefetch a
    vbroadcastf64x4    zmm30, 64[rbx]          // preload bext b0303
    vfmadd231pd zmm12, zmm25, zmm31{aaaa}      // c815_4
    vfmadd231pd zmm13, zmm25, zmm31{bbbb}      // c815_5
	add         rbx,   64                      // b += 8
    vprefetch0         64[rax]                 // prefetch a
    vfmadd231pd zmm14, zmm25, zmm31{cccc}      // c815_6
    vfmadd231pd zmm15, zmm25, zmm31{dddd}      // c815_7
	dec         rsi                            // k -= 1
    vfmadd231pd zmm20, zmm26, zmm31{aaaa}      // c623_4
    vfmadd231pd zmm21, zmm26, zmm31{bbbb}      // c623_5
	cmp         rsi,   0                       // check if the last iteration 
    vfmadd231pd zmm22, zmm26, zmm31{cccc}      // c623_6
    vfmadd231pd zmm23, zmm26, zmm31{dddd}      // c623_7

    jne LOOPMAIN     

	mov     r15,   ptr_neg2                    //load address of neg2
	mov     rax,   aa                          //load address of aa
	mov     rbx,   bb                          //load address of bb

    vbroadcastsd       zmm27,  0[r15]          // neg2
    vmovapd            zmm24,  0[rax]          // aa007
    vfmadd213pd zmm0,  zmm27, zmm24            // c007_0 = c007_0 * neg2 + aa007
    vfmadd213pd zmm1,  zmm27, zmm24            // 
    vfmadd213pd zmm2,  zmm27, zmm24            // 
    vfmadd213pd zmm3,  zmm27, zmm24            // 
    vmovapd            zmm25, 64[rax]          // aa815
    vfmadd213pd zmm4,  zmm27, zmm24            // 
    vfmadd213pd zmm5,  zmm27, zmm24            // 
    vfmadd213pd zmm6,  zmm27, zmm24            // 
    vfmadd213pd zmm7,  zmm27, zmm24            // c007_7 = c007_7 * neg2 + aa007

    vfmadd213pd zmm8,  zmm27, zmm25            // c815_0 = c815_0 * neg2 + aa815
    vfmadd213pd zmm9,  zmm27, zmm25            // 
    vfmadd213pd zmm10, zmm27, zmm25            // 
    vfmadd213pd zmm11, zmm27, zmm25            // 
    vmovapd            zmm26, 128[rax]         // aa623
    vfmadd213pd zmm12, zmm27, zmm25            // 
    vfmadd213pd zmm13, zmm27, zmm25            // 
    vfmadd213pd zmm14, zmm27, zmm25            // 
    vfmadd213pd zmm15, zmm27, zmm25            // c815_7 = c815_7 * neg2 + aa815

    vbroadcastf64x4    zmm30,  0[rbx]          // bb0303
    vfmadd213pd zmm16, zmm27, zmm26            // c623_0 = c623_0 * neg2 + aa623
    vfmadd213pd zmm17, zmm27, zmm26            // 
    vfmadd213pd zmm18, zmm27, zmm26            // 
    vfmadd213pd zmm19, zmm27, zmm26            // 
    vbroadcastf64x4    zmm31, 32[rbx]          // bb4747
    vfmadd213pd zmm20, zmm27, zmm26            // 
    vfmadd213pd zmm21, zmm27, zmm26            // 
    vfmadd213pd zmm22, zmm27, zmm26            // 
    vfmadd213pd zmm23, zmm27, zmm26            // c623_7 = c623_7 * neg2 + aa623
	
	mov         r15,   ptr_alpha               //load address of alpha

    vaddpd      zmm0,  zmm0,  zmm30{aaaa}      // c007_0 += bb0
    vaddpd      zmm1,  zmm1,  zmm30{bbbb}      // c007_1 += bb1
    vaddpd      zmm2,  zmm2,  zmm30{cccc}      // c007_2 += bb2
    vaddpd      zmm3,  zmm3,  zmm30{dddd}      // c007_3 += bb3
    vaddpd      zmm4,  zmm4,  zmm31{aaaa}      // c007_4 += bb4
    vaddpd      zmm5,  zmm5,  zmm31{bbbb}      // c007_5 += bb5
    vaddpd      zmm6,  zmm6,  zmm31{cccc}      // c007_6 += bb6
    vaddpd      zmm7,  zmm7,  zmm31{dddd}      // c007_7 += bb7

    vbroadcastsd       zmm27,  alpha//0[r15]          // alpha

    vaddpd      zmm8,  zmm8,  zmm30{aaaa}      // c815_0 += bb0
    vaddpd      zmm9,  zmm9,  zmm30{bbbb}      // c815_1 += bb1
    vaddpd      zmm10, zmm10, zmm30{cccc}      // c815_2 += bb2
    vaddpd      zmm11, zmm11, zmm30{dddd}      // c815_3 += bb3
    vaddpd      zmm12, zmm12, zmm31{aaaa}      // c815_4 += bb4
    vaddpd      zmm13, zmm13, zmm31{bbbb}      // c815_5 += bb5
    vaddpd      zmm14, zmm14, zmm31{cccc}      // c815_6 += bb6
    vaddpd      zmm15, zmm15, zmm31{dddd}      // c815_7 += bb7

	mov         r15,   ptr_minlog              //load address of alpha

    vaddpd      zmm16, zmm16, zmm30{aaaa}      // c623_0 += bb0
    vaddpd      zmm17, zmm17, zmm30{bbbb}      // c623_1 += bb1
    vaddpd      zmm18, zmm18, zmm30{cccc}      // c623_2 += bb2
    vaddpd      zmm19, zmm19, zmm30{dddd}      // c623_3 += bb3
    vaddpd      zmm20, zmm20, zmm31{aaaa}      // c623_4 += bb4
    vaddpd      zmm21, zmm21, zmm31{bbbb}      // c623_5 += bb5
    vaddpd      zmm22, zmm22, zmm31{cccc}      // c623_6 += bb6
    vaddpd      zmm23, zmm23, zmm31{dddd}      // c623_7 += bb7

	vpxord      zmm28, zmm28, zmm28            // dzero
    vbroadcastsd       zmm29,  0[r15]          // minlog

    vmulpd      zmm0,  zmm0,  zmm27            // c007_0 *= alpha 
    vmulpd      zmm1,  zmm1,  zmm27            // c007_1 *= alpha 
    vmulpd      zmm2,  zmm2,  zmm27            // c007_2 *= alpha 
    vmulpd      zmm3,  zmm3,  zmm27            // c007_3 *= alpha 
    vmulpd      zmm4,  zmm4,  zmm27            // c007_4 *= alpha 
    vmulpd      zmm5,  zmm5,  zmm27            // c007_5 *= alpha 
    vmulpd      zmm6,  zmm6,  zmm27            // c007_6 *= alpha 
    vmulpd      zmm7,  zmm7,  zmm27            // c007_7 *= alpha 

	mov         r15,   ptr_dmone               //load address of -1

    vmulpd      zmm8,  zmm8,  zmm27            // c815_0 *= alpha 
    vmulpd      zmm9,  zmm9,  zmm27            // c815_1 *= alpha 
    vmulpd      zmm10, zmm10, zmm27            // c815_2 *= alpha 
    vmulpd      zmm11, zmm11, zmm27            // c815_3 *= alpha 
    vmulpd      zmm12, zmm12, zmm27            // c815_4 *= alpha 
    vmulpd      zmm13, zmm13, zmm27            // c815_5 *= alpha 
    vmulpd      zmm14, zmm14, zmm27            // c815_6 *= alpha 
    vmulpd      zmm15, zmm15, zmm27            // c815_7 *= alpha 

    vbroadcastsd       zmm24,  0[r15]          // a0 = -1

    vmulpd      zmm16, zmm16, zmm27            // c623_0 *= alpha 
    vmulpd      zmm17, zmm17, zmm27            // c623_1 *= alpha 
    vmulpd      zmm18, zmm18, zmm27            // c623_2 *= alpha 
    vmulpd      zmm19, zmm19, zmm27            // c623_3 *= alpha 
    vmulpd      zmm20, zmm20, zmm27            // c623_4 *= alpha 
    vmulpd      zmm21, zmm21, zmm27            // c623_5 *= alpha 
    vmulpd      zmm22, zmm22, zmm27            // c623_6 *= alpha 
    vmulpd      zmm23, zmm23, zmm27            // c623_7 *= alpha 

    vmovapd     zmm25, zmm24                   // a1 = -1

    vgminpd     zmm0,  zmm0,  zmm28            // min( c, 0.0 ) 
    vgminpd     zmm1,  zmm1,  zmm28            // min( c, 0.0 )  
    vgminpd     zmm2,  zmm2,  zmm28            // min( c, 0.0 )  
    vgminpd     zmm3,  zmm3,  zmm28            // min( c, 0.0 )  
    vgminpd     zmm4,  zmm4,  zmm28            // min( c, 0.0 )  
    vgminpd     zmm5,  zmm5,  zmm28            // min( c, 0.0 )  
    vgminpd     zmm6,  zmm6,  zmm28            // min( c, 0.0 )  
    vgminpd     zmm7,  zmm7,  zmm28            // min( c, 0.0 )  

    vmovapd     zmm26, zmm24                   // a2 = -1

    vgminpd     zmm8,  zmm8,  zmm28            //  
    vgminpd     zmm9,  zmm9,  zmm28            //  
    vgminpd     zmm10, zmm10, zmm28            //  
    vgminpd     zmm11, zmm11, zmm28            //  
    vgminpd     zmm12, zmm12, zmm28            //  
    vgminpd     zmm13, zmm13, zmm28            //  
    vgminpd     zmm14, zmm14, zmm28            //  
    vgminpd     zmm15, zmm15, zmm28            //  

    vmovapd     zmm27, zmm24                   // a3 = -1

    vgminpd     zmm16, zmm16, zmm28            //  
    vgminpd     zmm17, zmm17, zmm28            //  
    vgminpd     zmm18, zmm18, zmm28            //  
    vgminpd     zmm19, zmm19, zmm28            //  
    vgminpd     zmm20, zmm20, zmm28            //  
    vgminpd     zmm21, zmm21, zmm28            //  
    vgminpd     zmm22, zmm22, zmm28            //  
    vgminpd     zmm23, zmm23, zmm28            // min( c, 0.0 )

	mov         r15,   ptr_log2e               //load address of log2e

    vgmaxpd     zmm0,  zmm0,  zmm29            // max( c, minlog ) 
    vgmaxpd     zmm1,  zmm1,  zmm29            // max( c, minlog )  
    vgmaxpd     zmm2,  zmm2,  zmm29            // max( c, minlog )  
    vgmaxpd     zmm3,  zmm3,  zmm29            // max( c, minlog )  
    vgmaxpd     zmm4,  zmm4,  zmm29            // max( c, minlog )  
    vgmaxpd     zmm5,  zmm5,  zmm29            // max( c, minlog )  
    vgmaxpd     zmm6,  zmm6,  zmm29            // max( c, minlog )  
    vgmaxpd     zmm7,  zmm7,  zmm29            // max( c, minlog )  

    vgmaxpd     zmm8,  zmm8,  zmm29            //  
    vgmaxpd     zmm9,  zmm9,  zmm29            //  
    vgmaxpd     zmm10, zmm10, zmm29            //  
    vgmaxpd     zmm11, zmm11, zmm29            //  
    vgmaxpd     zmm12, zmm12, zmm29            //  
    vgmaxpd     zmm13, zmm13, zmm29            //  
    vgmaxpd     zmm14, zmm14, zmm29            //  
    vgmaxpd     zmm15, zmm15, zmm29            //  

    vgmaxpd     zmm16, zmm16, zmm29            //  
    vgmaxpd     zmm17, zmm17, zmm29            //  
    vgmaxpd     zmm18, zmm18, zmm29            //  
    vgmaxpd     zmm19, zmm19, zmm29            //  
    vgmaxpd     zmm20, zmm20, zmm29            //  
    vgmaxpd     zmm21, zmm21, zmm29            //  
    vgmaxpd     zmm22, zmm22, zmm29            //  
    vgmaxpd     zmm23, zmm23, zmm29            // max( c, minlog )

    vfmadd231pd zmm24, zmm0,  0[r15]{1to8}     // a0 = c007_0 * log2e - 1
    vfmadd231pd zmm25, zmm1,  0[r15]{1to8}     // a1 = c007_1 * log2e - 1
    vfmadd231pd zmm26, zmm2,  0[r15]{1to8}     // a2 = c007_2 * log2e - 1
    vfmadd231pd zmm27, zmm3,  0[r15]{1to8}     // a3 = c007_3 * log2e - 1

	mov         r15,   ptr_c1                  //load address of c1

	vcvtfxpntpd2dq     zmm28, zmm24, 0x2       // k0 = double2int( a0 )
	vcvtfxpntpd2dq     zmm29, zmm25, 0x2       // k1 = double2int( a1 )
	vcvtfxpntpd2dq     zmm30, zmm26, 0x2       // k2 = double2int( a2 )
	vcvtfxpntpd2dq     zmm31, zmm27, 0x2       // k3 = double2int( a3 )
    
	vcvtdq2pd   zmm24, zmm28                   // p0 = int2double( k0 ) 
	vcvtdq2pd   zmm25, zmm29                   // p1 = int2double( k1 )
	vcvtdq2pd   zmm26, zmm30                   // p2 = int2double( k2 )
	vcvtdq2pd   zmm27, zmm31                   // p3 = int2double( k3 )

    vfmadd231pd zmm0,  zmm24,  0[r15]{1to8}    // c007_0 += p0 * c1 
    vfmadd231pd zmm1,  zmm25,  0[r15]{1to8}    // c007_1 += p1 * c1 
    vfmadd231pd zmm2,  zmm26,  0[r15]{1to8}    // c007_2 += p2 * c1 
    vfmadd231pd zmm3,  zmm27,  0[r15]{1to8}    // c007_3 += p3 * c1 

	mov         r15,   ptr_c2                  //load address of c2

    vfmadd231pd zmm0,  zmm24,  0[r15]{1to8}    // c007_0 += p0 * c2 
    vfmadd231pd zmm1,  zmm25,  0[r15]{1to8}    // c007_1 += p1 * c2 
    vfmadd231pd zmm2,  zmm26,  0[r15]{1to8}    // c007_2 += p2 * c2 
    vfmadd231pd zmm3,  zmm27,  0[r15]{1to8}    // c007_3 += p3 * c2 

	mov         r15,   ptr_mask                //load address of mask
    vmovdqa32          zmm24,  0[r15]          // zmm24 = perm 

	vpaddd      zmm28, zmm28,  zmm24           // k0 += mask
	vpaddd      zmm29, zmm29,  zmm24           // k1 += mask
	vpaddd      zmm30, zmm30,  zmm24           // k2 += mask
	vpaddd      zmm31, zmm31,  zmm24           // k3 += mask

	mov         r15,   ptr_perm                //load address of perm
    vmovdqa32          zmm24,  0[r15]          // zmm24 = perm

	vpermd      zmm28, zmm24,  zmm28           // permute( k0 )
	vpermd      zmm29, zmm24,  zmm29           // permute( k1 )
	vpermd      zmm30, zmm24,  zmm30           // permute( k2 )
	vpermd      zmm31, zmm24,  zmm31           // permute( k3 )

	mov         r15,   ptr_w11                 //load address of perm

	vpslld      zmm28, zmm28,  20              // shift k0<<20
	vpslld      zmm29, zmm29,  20              // shift k0<<20
	vpslld      zmm30, zmm30,  20              // shift k0<<20
	vpslld      zmm31, zmm31,  20              // shift k0<<20

    vbroadcastsd       zmm24,  w10             // a0 = w10
    vmovapd            zmm25,  zmm24           // a1 = w10
    vmovapd            zmm26,  zmm24           // a2 = w10
    vmovapd            zmm27,  zmm24           // a3 = w10

    vfmadd231pd zmm24, zmm0,   0[r15]{1to8}    // a0 += c007_0 * w11 
    vfmadd231pd zmm25, zmm1,   0[r15]{1to8}    // a1 += c007_1 * w11 
    vfmadd231pd zmm26, zmm2,   0[r15]{1to8}    // a2 += c007_2 * w11 
    vfmadd231pd zmm27, zmm3,   0[r15]{1to8}    // a3 += c007_3 * w11 

	mov         r15,   ptr_w9                  //load address of perm

    vfmadd213pd zmm24, zmm0,   0[r15]{1to8}    // a0 += c007_0 * w9 
    vfmadd213pd zmm25, zmm1,   0[r15]{1to8}    // a1 += c007_1 * w9 
    vfmadd213pd zmm26, zmm2,   0[r15]{1to8}    // a2 += c007_2 * w9 
    vfmadd213pd zmm27, zmm3,   0[r15]{1to8}    // a3 += c007_3 * w9 

	mov         r15,   ptr_w8                  //load address of perm

    vfmadd213pd zmm24, zmm0,   0[r15]{1to8}    // a0 += c007_0 * w8 
    vfmadd213pd zmm25, zmm1,   0[r15]{1to8}    // a1 += c007_1 * w8 
    vfmadd213pd zmm26, zmm2,   0[r15]{1to8}    // a2 += c007_2 * w8 
    vfmadd213pd zmm27, zmm3,   0[r15]{1to8}    // a3 += c007_3 * w8 

	mov         r15,   ptr_w7                  //load address of perm

    vfmadd213pd zmm24, zmm0,   0[r15]{1to8}    // a0 += c007_0 * w7 
    vfmadd213pd zmm25, zmm1,   0[r15]{1to8}    // a1 += c007_1 * w7 
    vfmadd213pd zmm26, zmm2,   0[r15]{1to8}    // a2 += c007_2 * w7 
    vfmadd213pd zmm27, zmm3,   0[r15]{1to8}    // a3 += c007_3 * w7 

	mov         r15,   ptr_w6                  //load address of perm

    vfmadd213pd zmm24, zmm0,   0[r15]{1to8}    // a0 += c007_0 * w6 
    vfmadd213pd zmm25, zmm1,   0[r15]{1to8}    // a1 += c007_1 * w6 
    vfmadd213pd zmm26, zmm2,   0[r15]{1to8}    // a2 += c007_2 * w6 
    vfmadd213pd zmm27, zmm3,   0[r15]{1to8}    // a3 += c007_3 * w6 

	mov         r15,   ptr_w5                  //load address of perm

    vfmadd213pd zmm24, zmm0,   0[r15]{1to8}    // a0 += c007_0 * w5 
    vfmadd213pd zmm25, zmm1,   0[r15]{1to8}    // a1 += c007_1 * w5 
    vfmadd213pd zmm26, zmm2,   0[r15]{1to8}    // a2 += c007_2 * w5 
    vfmadd213pd zmm27, zmm3,   0[r15]{1to8}    // a3 += c007_3 * w5 

	mov         r15,   ptr_w4                  //load address of perm

    vfmadd213pd zmm24, zmm0,   0[r15]{1to8}    // a0 += c007_0 * w4 
    vfmadd213pd zmm25, zmm1,   0[r15]{1to8}    // a1 += c007_1 * w4 
    vfmadd213pd zmm26, zmm2,   0[r15]{1to8}    // a2 += c007_2 * w4 
    vfmadd213pd zmm27, zmm3,   0[r15]{1to8}    // a3 += c007_3 * w4 

	mov         r15,   ptr_w3                  //load address of perm

    vfmadd213pd zmm24, zmm0,   0[r15]{1to8}    // a0 += c007_0 * w3 
    vfmadd213pd zmm25, zmm1,   0[r15]{1to8}    // a1 += c007_1 * w3 
    vfmadd213pd zmm26, zmm2,   0[r15]{1to8}    // a2 += c007_2 * w3 
    vfmadd213pd zmm27, zmm3,   0[r15]{1to8}    // a3 += c007_3 * w3 

	mov         r15,   ptr_w2                  //load address of perm

    vfmadd213pd zmm24, zmm0,   0[r15]{1to8}    // a0 += c007_0 * w2 
    vfmadd213pd zmm25, zmm1,   0[r15]{1to8}    // a1 += c007_1 * w2 
    vfmadd213pd zmm26, zmm2,   0[r15]{1to8}    // a2 += c007_2 * w2 
    vfmadd213pd zmm27, zmm3,   0[r15]{1to8}    // a3 += c007_3 * w2 

	mov         r15,   ptr_w1                  //load address of perm

    vfmadd213pd zmm24, zmm0,   0[r15]{1to8}    // a0 += c007_0 * w1 
    vfmadd213pd zmm25, zmm1,   0[r15]{1to8}    // a1 += c007_1 * w1 
    vfmadd213pd zmm26, zmm2,   0[r15]{1to8}    // a2 += c007_2 * w1 
    vfmadd213pd zmm27, zmm3,   0[r15]{1to8}    // a3 += c007_3 * w1 

	mov         r15,   ptr_w0                  //load address of perm

    vfmadd213pd zmm24, zmm0,   0[r15]{1to8}    // a0 += c007_0 * w0 
    vfmadd213pd zmm25, zmm1,   0[r15]{1to8}    // a1 += c007_1 * w0 
    vfmadd213pd zmm26, zmm2,   0[r15]{1to8}    // a2 += c007_2 * w0 
    vfmadd213pd zmm27, zmm3,   0[r15]{1to8}    // a3 += c007_3 * w0 

	mov         r15,   ptr_dmone               //load address of -1, iteration #1

	vmulpd      zmm0,  zmm24, zmm28            // c007_0 = a0 * p0 
	vmulpd      zmm1,  zmm25, zmm29            // c007_1 = a1 * p1
	vmulpd      zmm2,  zmm26, zmm30            // c007_2 = a2 * p2 
	vmulpd      zmm3,  zmm27, zmm31            // c007_3 = a3 * p3 

    vbroadcastsd       zmm24,  0[r15]          // a0 = -1
	mov         r15,   ptr_log2e               //load address of log2e

    vfmadd231pd zmm24, zmm4,  0[r15]{1to8}     // a0 = c007_4 * log2e - 1
    vmovapd     zmm25, zmm24                   // a1 = -1
    vfmadd231pd zmm25, zmm5,  0[r15]{1to8}     // a1 = c007_5 * log2e - 1
    vmovapd     zmm26, zmm24                   // a2 = -1
    vfmadd231pd zmm26, zmm6,  0[r15]{1to8}     // a2 = c007_6 * log2e - 1
    vmovapd     zmm27, zmm24                   // a3 = -1
    vfmadd231pd zmm27, zmm7,  0[r15]{1to8}     // a3 = c007_7 * log2e - 1

	mov         r15,   ptr_c1                  //load address of c1

	vcvtfxpntpd2dq     zmm28, zmm24, 0x2       // k0 = double2int( a0 )
	vcvtfxpntpd2dq     zmm29, zmm25, 0x2       // k1 = double2int( a1 )
	vcvtfxpntpd2dq     zmm30, zmm26, 0x2       // k2 = double2int( a2 )
	vcvtfxpntpd2dq     zmm31, zmm27, 0x2       // k3 = double2int( a3 )

	vcvtdq2pd   zmm24, zmm28                   // p0 = int2double( k0 ) 
	vcvtdq2pd   zmm25, zmm29                   // p1 = int2double( k1 )
	vcvtdq2pd   zmm26, zmm30                   // p2 = int2double( k2 )
	vcvtdq2pd   zmm27, zmm31                   // p3 = int2double( k3 )

    vfmadd231pd zmm4,  zmm24,  0[r15]{1to8}    // c007_4 += p0 * c1 
    vfmadd231pd zmm5,  zmm25,  0[r15]{1to8}    // c007_5 += p1 * c1 
    vfmadd231pd zmm6,  zmm26,  0[r15]{1to8}    // c007_6 += p2 * c1 
    vfmadd231pd zmm7,  zmm27,  0[r15]{1to8}    // c007_7 += p3 * c1 

	mov         r15,   ptr_c2                  //load address of c2

    vfmadd231pd zmm4,  zmm24,  0[r15]{1to8}    // c007_4 += p0 * c2 
    vfmadd231pd zmm5,  zmm25,  0[r15]{1to8}    // c007_5 += p1 * c2 
    vfmadd231pd zmm6,  zmm26,  0[r15]{1to8}    // c007_6 += p2 * c2 
    vfmadd231pd zmm7,  zmm27,  0[r15]{1to8}    // c007_7 += p3 * c2 

	mov         r15,   ptr_mask                //load address of mask
    vmovdqa32          zmm24,  0[r15]          // zmm24 = perm 

	vpaddd      zmm28, zmm28,  zmm24           // k0 += mask
	vpaddd      zmm29, zmm29,  zmm24           // k1 += mask
	vpaddd      zmm30, zmm30,  zmm24           // k2 += mask
	vpaddd      zmm31, zmm31,  zmm24           // k3 += mask

	mov         r15,   ptr_perm                //load address of perm
    vmovdqa32          zmm24,  0[r15]          // zmm24 = perm

	vpermd      zmm28, zmm24,  zmm28           // permute( k0 )
	vpermd      zmm29, zmm24,  zmm29           // permute( k1 )
	vpermd      zmm30, zmm24,  zmm30           // permute( k2 )
	vpermd      zmm31, zmm24,  zmm31           // permute( k3 )

	mov         r15,   ptr_w11                 //load address of perm

	vpslld      zmm28, zmm28,  20              // shift k0<<20
	vpslld      zmm29, zmm29,  20              // shift k0<<20
	vpslld      zmm30, zmm30,  20              // shift k0<<20
	vpslld      zmm31, zmm31,  20              // shift k0<<20

    vbroadcastsd       zmm24,  w10             // a0 = w10
    vmovapd            zmm25,  zmm24           // a1 = w10
    vmovapd            zmm26,  zmm24           // a2 = w10
    vmovapd            zmm27,  zmm24           // a3 = w10

    vfmadd231pd zmm24, zmm4,   0[r15]{1to8}    // a0 += c007_4 * w11 
    vfmadd231pd zmm25, zmm5,   0[r15]{1to8}    // a1 += c007_5 * w11 
    vfmadd231pd zmm26, zmm6,   0[r15]{1to8}    // a2 += c007_6 * w11 
    vfmadd231pd zmm27, zmm7,   0[r15]{1to8}    // a3 += c007_7 * w11 

	mov         r15,   ptr_w9                  //load address of perm

    vfmadd213pd zmm24, zmm4,   0[r15]{1to8}    // a0 += c007_4 * w9 
    vfmadd213pd zmm25, zmm5,   0[r15]{1to8}    // a1 += c007_5 * w9 
    vfmadd213pd zmm26, zmm6,   0[r15]{1to8}    // a2 += c007_6 * w9 
    vfmadd213pd zmm27, zmm7,   0[r15]{1to8}    // a3 += c007_7 * w9 

	mov         r15,   ptr_w8                  //load address of perm

    vfmadd213pd zmm24, zmm4,   0[r15]{1to8}    // a0 += c007_4 * w8 
    vfmadd213pd zmm25, zmm5,   0[r15]{1to8}    // a1 += c007_5 * w8 
    vfmadd213pd zmm26, zmm6,   0[r15]{1to8}    // a2 += c007_6 * w8 
    vfmadd213pd zmm27, zmm7,   0[r15]{1to8}    // a3 += c007_7 * w8 

	mov         r15,   ptr_w7                  //load address of perm

    vfmadd213pd zmm24, zmm4,   0[r15]{1to8}    // a0 += c007_4 * w7 
    vfmadd213pd zmm25, zmm5,   0[r15]{1to8}    // a1 += c007_5 * w7 
    vfmadd213pd zmm26, zmm6,   0[r15]{1to8}    // a2 += c007_6 * w7 
    vfmadd213pd zmm27, zmm7,   0[r15]{1to8}    // a3 += c007_7 * w7 

	mov         r15,   ptr_w6                  //load address of perm

    vfmadd213pd zmm24, zmm4,   0[r15]{1to8}    // a0 += c007_4 * w6 
    vfmadd213pd zmm25, zmm5,   0[r15]{1to8}    // a1 += c007_5 * w6 
    vfmadd213pd zmm26, zmm6,   0[r15]{1to8}    // a2 += c007_6 * w6 
    vfmadd213pd zmm27, zmm7,   0[r15]{1to8}    // a3 += c007_7 * w6 

	mov         r15,   ptr_w5                  //load address of perm

    vfmadd213pd zmm24, zmm4,   0[r15]{1to8}    // a0 += c007_4 * w5 
    vfmadd213pd zmm25, zmm5,   0[r15]{1to8}    // a1 += c007_5 * w5 
    vfmadd213pd zmm26, zmm6,   0[r15]{1to8}    // a2 += c007_6 * w5 
    vfmadd213pd zmm27, zmm7,   0[r15]{1to8}    // a3 += c007_7 * w5 

	mov         r15,   ptr_w4                  //load address of perm

    vfmadd213pd zmm24, zmm4,   0[r15]{1to8}    // a0 += c007_4 * w4 
    vfmadd213pd zmm25, zmm5,   0[r15]{1to8}    // a1 += c007_5 * w4 
    vfmadd213pd zmm26, zmm6,   0[r15]{1to8}    // a2 += c007_6 * w4 
    vfmadd213pd zmm27, zmm7,   0[r15]{1to8}    // a3 += c007_7 * w4 

	mov         r15,   ptr_w3                  //load address of perm

    vfmadd213pd zmm24, zmm4,   0[r15]{1to8}    // a0 += c007_4 * w3 
    vfmadd213pd zmm25, zmm5,   0[r15]{1to8}    // a1 += c007_5 * w3 
    vfmadd213pd zmm26, zmm6,   0[r15]{1to8}    // a2 += c007_6 * w3 
    vfmadd213pd zmm27, zmm7,   0[r15]{1to8}    // a3 += c007_7 * w3 

	mov         r15,   ptr_w2                  //load address of perm

    vfmadd213pd zmm24, zmm4,   0[r15]{1to8}    // a0 += c007_4 * w2 
    vfmadd213pd zmm25, zmm5,   0[r15]{1to8}    // a1 += c007_5 * w2 
    vfmadd213pd zmm26, zmm6,   0[r15]{1to8}    // a2 += c007_6 * w2 
    vfmadd213pd zmm27, zmm7,   0[r15]{1to8}    // a3 += c007_7 * w2 

	mov         r15,   ptr_w1                  //load address of perm

    vfmadd213pd zmm24, zmm4,   0[r15]{1to8}    // a0 += c007_4 * w1 
    vfmadd213pd zmm25, zmm5,   0[r15]{1to8}    // a1 += c007_5 * w1 
    vfmadd213pd zmm26, zmm6,   0[r15]{1to8}    // a2 += c007_6 * w1 
    vfmadd213pd zmm27, zmm7,   0[r15]{1to8}    // a3 += c007_7 * w1 

	mov         r15,   ptr_w0                  //load address of perm

    vfmadd213pd zmm24, zmm4,   0[r15]{1to8}    // a0 += c007_4 * w0 
    vfmadd213pd zmm25, zmm5,   0[r15]{1to8}    // a1 += c007_5 * w0 
    vfmadd213pd zmm26, zmm6,   0[r15]{1to8}    // a2 += c007_6 * w0 
    vfmadd213pd zmm27, zmm7,   0[r15]{1to8}    // a3 += c007_7 * w0 

	mov         r15,   ptr_dmone               //load address of -1, iteration #2

	vmulpd      zmm4,  zmm24, zmm28            // c007_4 = a0 * p0 
	vmulpd      zmm5,  zmm25, zmm29            // c007_5 = a1 * p1
	vmulpd      zmm6,  zmm26, zmm30            // c007_6 = a2 * p2 
	vmulpd      zmm7,  zmm27, zmm31            // c007_7 = a3 * p3 

    vbroadcastsd       zmm24,  0[r15]          // a0 = -1
	mov         r15,   ptr_log2e               //load address of log2e

    vfmadd231pd zmm24, zmm8,  0[r15]{1to8}     // a0 = c815_0 * log2e - 1
    vmovapd     zmm25, zmm24                   //
    vfmadd231pd zmm25, zmm9,  0[r15]{1to8}     // a1 = c815_1 * log2e - 1
    vmovapd     zmm26, zmm24                   //
    vfmadd231pd zmm26, zmm10, 0[r15]{1to8}     // a2 = c815_2 * log2e - 1
    vmovapd     zmm27, zmm24                   //
    vfmadd231pd zmm27, zmm11, 0[r15]{1to8}     // a3 = c815_3 * log2e - 1

	mov         r15,   ptr_c1                  //load address of c1

	vcvtfxpntpd2dq     zmm28, zmm24, 0x2       // k0 = double2int( a0 )
	vcvtfxpntpd2dq     zmm29, zmm25, 0x2       // k1 = double2int( a1 )
	vcvtfxpntpd2dq     zmm30, zmm26, 0x2       // k2 = double2int( a2 )
	vcvtfxpntpd2dq     zmm31, zmm27, 0x2       // k3 = double2int( a3 )

	vcvtdq2pd   zmm24, zmm28                   // p0 = int2double( k0 ) 
	vcvtdq2pd   zmm25, zmm29                   // p1 = int2double( k1 )
	vcvtdq2pd   zmm26, zmm30                   // p2 = int2double( k2 )
	vcvtdq2pd   zmm27, zmm31                   // p3 = int2double( k3 )

    vfmadd231pd zmm8,  zmm24,  0[r15]{1to8}    // c815_0 += p0 * c1 
    vfmadd231pd zmm9,  zmm25,  0[r15]{1to8}    // c815_1 += p1 * c1 
    vfmadd231pd zmm10, zmm26,  0[r15]{1to8}    // c815_2 += p2 * c1 
    vfmadd231pd zmm11, zmm27,  0[r15]{1to8}    // c815_3 += p3 * c1 

	mov         r15,   ptr_c2                  //load address of c2

    vfmadd231pd zmm8,  zmm24,  0[r15]{1to8}    // c815_0 += p0 * c2 
    vfmadd231pd zmm9,  zmm25,  0[r15]{1to8}    // c815_1 += p1 * c2 
    vfmadd231pd zmm10, zmm26,  0[r15]{1to8}    // c815_2 += p2 * c2 
    vfmadd231pd zmm11, zmm27,  0[r15]{1to8}    // c815_3 += p3 * c2 

	mov         r15,   ptr_mask                //load address of mask
    vmovdqa32          zmm24,  0[r15]          // zmm24 = perm 

	vpaddd      zmm28, zmm28,  zmm24           // k0 += mask
	vpaddd      zmm29, zmm29,  zmm24           // k1 += mask
	vpaddd      zmm30, zmm30,  zmm24           // k2 += mask
	vpaddd      zmm31, zmm31,  zmm24           // k3 += mask

	mov         r15,   ptr_perm                //load address of perm
    vmovdqa32          zmm24,  0[r15]          // zmm24 = perm

	vpermd      zmm28, zmm24,  zmm28           // permute( k0 )
	vpermd      zmm29, zmm24,  zmm29           // permute( k1 )
	vpermd      zmm30, zmm24,  zmm30           // permute( k2 )
	vpermd      zmm31, zmm24,  zmm31           // permute( k3 )

	mov         r15,   ptr_w11                 //load address of perm

	vpslld      zmm28, zmm28,  20              // shift k0<<20
	vpslld      zmm29, zmm29,  20              // shift k0<<20
	vpslld      zmm30, zmm30,  20              // shift k0<<20
	vpslld      zmm31, zmm31,  20              // shift k0<<20

    vbroadcastsd       zmm24,  w10             // a0 = w10
    vmovapd            zmm25,  zmm24           // a1 = w10
    vmovapd            zmm26,  zmm24           // a2 = w10
    vmovapd            zmm27,  zmm24           // a3 = w10

    vfmadd231pd zmm24, zmm8,   0[r15]{1to8}    // a0 += c815_0 * w11 
    vfmadd231pd zmm25, zmm9,   0[r15]{1to8}    // a1 += c815_1 * w11 
    vfmadd231pd zmm26, zmm10,  0[r15]{1to8}    // a2 += c815_2 * w11 
    vfmadd231pd zmm27, zmm11,  0[r15]{1to8}    // a3 += c815_3 * w11 

	mov         r15,   ptr_w9                  //load address of perm

    vfmadd213pd zmm24, zmm8,   0[r15]{1to8}    // a0 += c815_0 * w9 
    vfmadd213pd zmm25, zmm9,   0[r15]{1to8}    // a1 += c815_1 * w9 
    vfmadd213pd zmm26, zmm10,  0[r15]{1to8}    // a2 += c815_2 * w9 
    vfmadd213pd zmm27, zmm11,  0[r15]{1to8}    // a3 += c815_3 * w9 

	mov         r15,   ptr_w8                  //load address of perm

    vfmadd213pd zmm24, zmm8,   0[r15]{1to8}    // a0 += c815_0 * w8 
    vfmadd213pd zmm25, zmm9,   0[r15]{1to8}    // a1 += c815_1 * w8 
    vfmadd213pd zmm26, zmm10,  0[r15]{1to8}    // a2 += c815_2 * w8 
    vfmadd213pd zmm27, zmm11,  0[r15]{1to8}    // a3 += c815_3 * w8 

	mov         r15,   ptr_w7                  //load address of perm

    vfmadd213pd zmm24, zmm8,   0[r15]{1to8}    // a0 += c815_0 * w7 
    vfmadd213pd zmm25, zmm9,   0[r15]{1to8}    // a1 += c815_1 * w7 
    vfmadd213pd zmm26, zmm10,  0[r15]{1to8}    // a2 += c815_2 * w7 
    vfmadd213pd zmm27, zmm11,  0[r15]{1to8}    // a3 += c815_3 * w7 

	mov         r15,   ptr_w6                  //load address of perm

    vfmadd213pd zmm24, zmm8,   0[r15]{1to8}    // a0 += c815_0 * w6 
    vfmadd213pd zmm25, zmm9,   0[r15]{1to8}    // a1 += c815_1 * w6 
    vfmadd213pd zmm26, zmm10,  0[r15]{1to8}    // a2 += c815_2 * w6 
    vfmadd213pd zmm27, zmm11,  0[r15]{1to8}    // a3 += c815_3 * w6 

	mov         r15,   ptr_w5                  //load address of perm

    vfmadd213pd zmm24, zmm8,   0[r15]{1to8}    // a0 += c815_0 * w5 
    vfmadd213pd zmm25, zmm9,   0[r15]{1to8}    // a1 += c815_1 * w5 
    vfmadd213pd zmm26, zmm10,  0[r15]{1to8}    // a2 += c815_2 * w5 
    vfmadd213pd zmm27, zmm11,  0[r15]{1to8}    // a3 += c815_3 * w5 

	mov         r15,   ptr_w4                  //load address of perm

    vfmadd213pd zmm24, zmm8,   0[r15]{1to8}    // a0 += c815_0 * w4 
    vfmadd213pd zmm25, zmm9,   0[r15]{1to8}    // a1 += c815_1 * w4 
    vfmadd213pd zmm26, zmm10,  0[r15]{1to8}    // a2 += c815_2 * w4 
    vfmadd213pd zmm27, zmm11,  0[r15]{1to8}    // a3 += c815_3 * w4 

	mov         r15,   ptr_w3                  //load address of perm

    vfmadd213pd zmm24, zmm8,   0[r15]{1to8}    // a0 += c815_0 * w3 
    vfmadd213pd zmm25, zmm9,   0[r15]{1to8}    // a1 += c815_1 * w3 
    vfmadd213pd zmm26, zmm10,  0[r15]{1to8}    // a2 += c815_2 * w3 
    vfmadd213pd zmm27, zmm11,  0[r15]{1to8}    // a3 += c815_3 * w3 

	mov         r15,   ptr_w2                  //load address of perm

    vfmadd213pd zmm24, zmm8,   0[r15]{1to8}    // a0 += c815_0 * w2 
    vfmadd213pd zmm25, zmm9,   0[r15]{1to8}    // a1 += c815_1 * w2 
    vfmadd213pd zmm26, zmm10,  0[r15]{1to8}    // a2 += c815_2 * w2 
    vfmadd213pd zmm27, zmm11,  0[r15]{1to8}    // a3 += c815_3 * w2 

	mov         r15,   ptr_w1                  //load address of perm

    vfmadd213pd zmm24, zmm8,   0[r15]{1to8}    // a0 += c815_0 * w1 
    vfmadd213pd zmm25, zmm9,   0[r15]{1to8}    // a1 += c815_1 * w1 
    vfmadd213pd zmm26, zmm10,  0[r15]{1to8}    // a2 += c815_2 * w1 
    vfmadd213pd zmm27, zmm11,  0[r15]{1to8}    // a3 += c815_3 * w1 

	mov         r15,   ptr_w0                  //load address of perm

    vfmadd213pd zmm24, zmm8,   0[r15]{1to8}    // a0 += c815_0 * w0 
    vfmadd213pd zmm25, zmm9,   0[r15]{1to8}    // a1 += c815_1 * w0 
    vfmadd213pd zmm26, zmm10,  0[r15]{1to8}    // a2 += c815_2 * w0 
    vfmadd213pd zmm27, zmm11,  0[r15]{1to8}    // a3 += c815_3 * w0 

	mov         r15,   ptr_dmone               //load address of -1, iteration #3

	vmulpd      zmm8,  zmm24, zmm28            // c815_0 = a0 * p0 
	vmulpd      zmm9,  zmm25, zmm29            // c815_1 = a1 * p1
	vmulpd      zmm10, zmm26, zmm30            // c815_2 = a2 * p2 
	vmulpd      zmm11, zmm27, zmm31            // c815_3 = a3 * p3 

    vbroadcastsd       zmm24,  0[r15]          // a0 = -1
	mov         r15,   ptr_log2e               //load address of log2e

    vfmadd231pd zmm24, zmm12, 0[r15]{1to8}     // a0 = c815_4 * log2e - 1
    vmovapd     zmm25, zmm24                   //
    vfmadd231pd zmm25, zmm13, 0[r15]{1to8}     // a1 = c815_5 * log2e - 1
    vmovapd     zmm26, zmm24                   //
    vfmadd231pd zmm26, zmm14, 0[r15]{1to8}     // a2 = c815_6 * log2e - 1
    vmovapd     zmm27, zmm24                   //
    vfmadd231pd zmm27, zmm15, 0[r15]{1to8}     // a3 = c815_7 * log2e - 1

	mov         r15,   ptr_c1                  //load address of c1

	vcvtfxpntpd2dq     zmm28, zmm24, 0x2       // k0 = double2int( a0 )
	vcvtfxpntpd2dq     zmm29, zmm25, 0x2       // k1 = double2int( a1 )
	vcvtfxpntpd2dq     zmm30, zmm26, 0x2       // k2 = double2int( a2 )
	vcvtfxpntpd2dq     zmm31, zmm27, 0x2       // k3 = double2int( a3 )

	vcvtdq2pd   zmm24, zmm28                   // p0 = int2double( k0 ) 
	vcvtdq2pd   zmm25, zmm29                   // p1 = int2double( k1 )
	vcvtdq2pd   zmm26, zmm30                   // p2 = int2double( k2 )
	vcvtdq2pd   zmm27, zmm31                   // p3 = int2double( k3 )

    vfmadd231pd zmm12, zmm24,  0[r15]{1to8}    // c815_4 += p0 * c1 
    vfmadd231pd zmm13, zmm25,  0[r15]{1to8}    // c815_5 += p1 * c1 
    vfmadd231pd zmm14, zmm26,  0[r15]{1to8}    // c815_6 += p2 * c1 
    vfmadd231pd zmm15, zmm27,  0[r15]{1to8}    // c815_7 += p3 * c1 

	mov         r15,   ptr_c2                  //load address of c2

    vfmadd231pd zmm12, zmm24,  0[r15]{1to8}    // c815_4 += p0 * c2 
    vfmadd231pd zmm13, zmm25,  0[r15]{1to8}    // c815_5 += p1 * c2 
    vfmadd231pd zmm14, zmm26,  0[r15]{1to8}    // c815_6 += p2 * c2 
    vfmadd231pd zmm15, zmm27,  0[r15]{1to8}    // c815_7 += p3 * c2 

	mov         r15,   ptr_mask                //load address of mask
    vmovdqa32          zmm24,  0[r15]          // zmm24 = perm 

	vpaddd      zmm28, zmm28,  zmm24           // k0 += mask
	vpaddd      zmm29, zmm29,  zmm24           // k1 += mask
	vpaddd      zmm30, zmm30,  zmm24           // k2 += mask
	vpaddd      zmm31, zmm31,  zmm24           // k3 += mask

	mov         r15,   ptr_perm                //load address of perm
    vmovdqa32          zmm24,  0[r15]          // zmm24 = perm

	vpermd      zmm28, zmm24,  zmm28           // permute( k0 )
	vpermd      zmm29, zmm24,  zmm29           // permute( k1 )
	vpermd      zmm30, zmm24,  zmm30           // permute( k2 )
	vpermd      zmm31, zmm24,  zmm31           // permute( k3 )

	mov         r15,   ptr_w11                 //load address of perm

	vpslld      zmm28, zmm28,  20              // shift k0<<20
	vpslld      zmm29, zmm29,  20              // shift k0<<20
	vpslld      zmm30, zmm30,  20              // shift k0<<20
	vpslld      zmm31, zmm31,  20              // shift k0<<20

    vbroadcastsd       zmm24,  w10             // a0 = w10
    vmovapd            zmm25,  zmm24           // a1 = w10
    vmovapd            zmm26,  zmm24           // a2 = w10
    vmovapd            zmm27,  zmm24           // a3 = w10

    vfmadd231pd zmm24, zmm12,  0[r15]{1to8}    // a0 += c815_4 * w11 
    vfmadd231pd zmm25, zmm13,  0[r15]{1to8}    // a1 += c815_5 * w11 
    vfmadd231pd zmm26, zmm14,  0[r15]{1to8}    // a2 += c815_6 * w11 
    vfmadd231pd zmm27, zmm15,  0[r15]{1to8}    // a3 += c815_7 * w11 

	mov         r15,   ptr_w9                  //load address of perm

    vfmadd213pd zmm24, zmm12,  0[r15]{1to8}    // a0 += c815_4 * w9 
    vfmadd213pd zmm25, zmm13,  0[r15]{1to8}    // a1 += c815_5 * w9 
    vfmadd213pd zmm26, zmm14,  0[r15]{1to8}    // a2 += c815_6 * w9 
    vfmadd213pd zmm27, zmm15,  0[r15]{1to8}    // a3 += c815_7 * w9 

	mov         r15,   ptr_w8                  //load address of perm

    vfmadd213pd zmm24, zmm12,  0[r15]{1to8}    // a0 += c815_4 * w8 
    vfmadd213pd zmm25, zmm13,  0[r15]{1to8}    // a1 += c815_5 * w8 
    vfmadd213pd zmm26, zmm14,  0[r15]{1to8}    // a2 += c815_6 * w8 
    vfmadd213pd zmm27, zmm15,  0[r15]{1to8}    // a3 += c815_7 * w8 

	mov         r15,   ptr_w7                  //load address of perm

    vfmadd213pd zmm24, zmm12,  0[r15]{1to8}    // a0 += c815_4 * w7 
    vfmadd213pd zmm25, zmm13,  0[r15]{1to8}    // a1 += c815_5 * w7 
    vfmadd213pd zmm26, zmm14,  0[r15]{1to8}    // a2 += c815_6 * w7 
    vfmadd213pd zmm27, zmm15,  0[r15]{1to8}    // a3 += c815_7 * w7 

	mov         r15,   ptr_w6                  //load address of perm

    vfmadd213pd zmm24, zmm12,  0[r15]{1to8}    // a0 += c815_4 * w6 
    vfmadd213pd zmm25, zmm13,  0[r15]{1to8}    // a1 += c815_5 * w6 
    vfmadd213pd zmm26, zmm14,  0[r15]{1to8}    // a2 += c815_6 * w6 
    vfmadd213pd zmm27, zmm15,  0[r15]{1to8}    // a3 += c815_7 * w6 

	mov         r15,   ptr_w5                  //load address of perm

    vfmadd213pd zmm24, zmm12,  0[r15]{1to8}    // a0 += c815_4 * w5 
    vfmadd213pd zmm25, zmm13,  0[r15]{1to8}    // a1 += c815_5 * w5 
    vfmadd213pd zmm26, zmm14,  0[r15]{1to8}    // a2 += c815_6 * w5 
    vfmadd213pd zmm27, zmm15,  0[r15]{1to8}    // a3 += c815_7 * w5 

	mov         r15,   ptr_w4                  //load address of perm

    vfmadd213pd zmm24, zmm12,  0[r15]{1to8}    // a0 += c815_4 * w4 
    vfmadd213pd zmm25, zmm13,  0[r15]{1to8}    // a1 += c815_5 * w4 
    vfmadd213pd zmm26, zmm14,  0[r15]{1to8}    // a2 += c815_6 * w4 
    vfmadd213pd zmm27, zmm15,  0[r15]{1to8}    // a3 += c815_7 * w4 

	mov         r15,   ptr_w3                  //load address of perm

    vfmadd213pd zmm24, zmm12,  0[r15]{1to8}    // a0 += c815_4 * w3 
    vfmadd213pd zmm25, zmm13,  0[r15]{1to8}    // a1 += c815_5 * w3 
    vfmadd213pd zmm26, zmm14,  0[r15]{1to8}    // a2 += c815_6 * w3 
    vfmadd213pd zmm27, zmm15,  0[r15]{1to8}    // a3 += c815_7 * w3 

	mov         r15,   ptr_w2                  //load address of perm

    vfmadd213pd zmm24, zmm12,  0[r15]{1to8}    // a0 += c815_4 * w2 
    vfmadd213pd zmm25, zmm13,  0[r15]{1to8}    // a1 += c815_5 * w2 
    vfmadd213pd zmm26, zmm14,  0[r15]{1to8}    // a2 += c815_6 * w2 
    vfmadd213pd zmm27, zmm15,  0[r15]{1to8}    // a3 += c815_7 * w2 

	mov         r15,   ptr_w1                  //load address of perm

    vfmadd213pd zmm24, zmm12,  0[r15]{1to8}    // a0 += c815_4 * w1 
    vfmadd213pd zmm25, zmm13,  0[r15]{1to8}    // a1 += c815_5 * w1 
    vfmadd213pd zmm26, zmm14,  0[r15]{1to8}    // a2 += c815_6 * w1 
    vfmadd213pd zmm27, zmm15,  0[r15]{1to8}    // a3 += c815_7 * w1 

	mov         r15,   ptr_w0                  //load address of perm

    vfmadd213pd zmm24, zmm12,  0[r15]{1to8}    // a0 += c815_4 * w0 
    vfmadd213pd zmm25, zmm13,  0[r15]{1to8}    // a1 += c815_5 * w0 
    vfmadd213pd zmm26, zmm14,  0[r15]{1to8}    // a2 += c815_6 * w0 
    vfmadd213pd zmm27, zmm15,  0[r15]{1to8}    // a3 += c815_7 * w0 

	mov         r15,   ptr_dmone               //load address of -1, iteration #4

	vmulpd      zmm12, zmm24, zmm28            // c815_4 = a0 * p0 
	vmulpd      zmm13, zmm25, zmm29            // c815_5 = a1 * p1
	vmulpd      zmm14, zmm26, zmm30            // c815_6 = a2 * p2 
	vmulpd      zmm15, zmm27, zmm31            // c815_7 = a3 * p3 

    vbroadcastsd       zmm24,  0[r15]          // a0 = -1
	mov         r15,   ptr_log2e               //load address of log2e

    vfmadd231pd zmm24, zmm16, 0[r15]{1to8}     // a0 = c623_0 * log2e - 1
    vmovapd     zmm25, zmm24                   //
    vfmadd231pd zmm25, zmm17, 0[r15]{1to8}     // a1 = c623_1 * log2e - 1
    vmovapd     zmm26, zmm24                   //
    vfmadd231pd zmm26, zmm18, 0[r15]{1to8}     // a2 = c623_2 * log2e - 1
    vmovapd     zmm27, zmm24                   //
    vfmadd231pd zmm27, zmm19, 0[r15]{1to8}     // a3 = c623_3 * log2e - 1

	mov         r15,   ptr_c1                  //load address of c1

	vcvtfxpntpd2dq     zmm28, zmm24, 0x2       // k0 = double2int( a0 )
	vcvtfxpntpd2dq     zmm29, zmm25, 0x2       // k1 = double2int( a1 )
	vcvtfxpntpd2dq     zmm30, zmm26, 0x2       // k2 = double2int( a2 )
	vcvtfxpntpd2dq     zmm31, zmm27, 0x2       // k3 = double2int( a3 )

	vcvtdq2pd   zmm24, zmm28                   // p0 = int2double( k0 ) 
	vcvtdq2pd   zmm25, zmm29                   // p1 = int2double( k1 )
	vcvtdq2pd   zmm26, zmm30                   // p2 = int2double( k2 )
	vcvtdq2pd   zmm27, zmm31                   // p3 = int2double( k3 )

    vfmadd231pd zmm16, zmm24,  0[r15]{1to8}    // c623_0 += p0 * c1 
    vfmadd231pd zmm17, zmm25,  0[r15]{1to8}    // c623_1 += p1 * c1 
    vfmadd231pd zmm18, zmm26,  0[r15]{1to8}    // c623_2 += p2 * c1 
    vfmadd231pd zmm19, zmm27,  0[r15]{1to8}    // c623_3 += p3 * c1 

	mov         r15,   ptr_c2                  //load address of c2

    vfmadd231pd zmm16, zmm24,  0[r15]{1to8}    // c623_0 += p0 * c2 
    vfmadd231pd zmm17, zmm25,  0[r15]{1to8}    // c623_1 += p1 * c2 
    vfmadd231pd zmm18, zmm26,  0[r15]{1to8}    // c623_2 += p2 * c2 
    vfmadd231pd zmm19, zmm27,  0[r15]{1to8}    // c623_3 += p3 * c2 

	mov         r15,   ptr_mask                //load address of mask
    vmovdqa32          zmm24,  0[r15]          // zmm24 = perm 

	vpaddd      zmm28, zmm28,  zmm24           // k0 += mask
	vpaddd      zmm29, zmm29,  zmm24           // k1 += mask
	vpaddd      zmm30, zmm30,  zmm24           // k2 += mask
	vpaddd      zmm31, zmm31,  zmm24           // k3 += mask

	mov         r15,   ptr_perm                //load address of perm
    vmovdqa32          zmm24,  0[r15]          // zmm24 = perm

	vpermd      zmm28, zmm24,  zmm28           // permute( k0 )
	vpermd      zmm29, zmm24,  zmm29           // permute( k1 )
	vpermd      zmm30, zmm24,  zmm30           // permute( k2 )
	vpermd      zmm31, zmm24,  zmm31           // permute( k3 )

	mov         r15,   ptr_w11                 //load address of perm

	vpslld      zmm28, zmm28,  20              // shift k0<<20
	vpslld      zmm29, zmm29,  20              // shift k0<<20
	vpslld      zmm30, zmm30,  20              // shift k0<<20
	vpslld      zmm31, zmm31,  20              // shift k0<<20

    vbroadcastsd       zmm24,  w10             // a0 = w10
    vmovapd            zmm25,  zmm24           // a1 = w10
    vmovapd            zmm26,  zmm24           // a2 = w10
    vmovapd            zmm27,  zmm24           // a3 = w10

    vfmadd231pd zmm24, zmm16,  0[r15]{1to8}    // a0 += c623_0 * w11 
    vfmadd231pd zmm25, zmm17,  0[r15]{1to8}    // a1 += c623_1 * w11 
    vfmadd231pd zmm26, zmm18,  0[r15]{1to8}    // a2 += c623_2 * w11 
    vfmadd231pd zmm27, zmm19,  0[r15]{1to8}    // a3 += c623_3 * w11 

	mov         r15,   ptr_w9                  //load address of perm

    vfmadd213pd zmm24, zmm16,  0[r15]{1to8}    // a0 += c623_0 * w9 
    vfmadd213pd zmm25, zmm17,  0[r15]{1to8}    // a1 += c623_1 * w9 
    vfmadd213pd zmm26, zmm18,  0[r15]{1to8}    // a2 += c623_2 * w9 
    vfmadd213pd zmm27, zmm19,  0[r15]{1to8}    // a3 += c623_3 * w9 

	mov         r15,   ptr_w8                  //load address of perm

    vfmadd213pd zmm24, zmm16,  0[r15]{1to8}    // a0 += c623_0 * w8 
    vfmadd213pd zmm25, zmm17,  0[r15]{1to8}    // a1 += c623_1 * w8 
    vfmadd213pd zmm26, zmm18,  0[r15]{1to8}    // a2 += c623_2 * w8 
    vfmadd213pd zmm27, zmm19,  0[r15]{1to8}    // a3 += c623_3 * w8 

	mov         r15,   ptr_w7                  //load address of perm

    vfmadd213pd zmm24, zmm16,  0[r15]{1to8}    // a0 += c623_0 * w7 
    vfmadd213pd zmm25, zmm17,  0[r15]{1to8}    // a1 += c623_1 * w7 
    vfmadd213pd zmm26, zmm18,  0[r15]{1to8}    // a2 += c623_2 * w7 
    vfmadd213pd zmm27, zmm19,  0[r15]{1to8}    // a3 += c623_3 * w7 

	mov         r15,   ptr_w6                  //load address of perm

    vfmadd213pd zmm24, zmm16,  0[r15]{1to8}    // a0 += c623_0 * w6 
    vfmadd213pd zmm25, zmm17,  0[r15]{1to8}    // a1 += c623_1 * w6 
    vfmadd213pd zmm26, zmm18,  0[r15]{1to8}    // a2 += c623_2 * w6 
    vfmadd213pd zmm27, zmm19,  0[r15]{1to8}    // a3 += c623_3 * w6 

	mov         r15,   ptr_w5                  //load address of perm

    vfmadd213pd zmm24, zmm16,  0[r15]{1to8}    // a0 += c623_0 * w5 
    vfmadd213pd zmm25, zmm17,  0[r15]{1to8}    // a1 += c623_1 * w5 
    vfmadd213pd zmm26, zmm18,  0[r15]{1to8}    // a2 += c623_2 * w5 
    vfmadd213pd zmm27, zmm19,  0[r15]{1to8}    // a3 += c623_3 * w5 

	mov         r15,   ptr_w4                  //load address of perm

    vfmadd213pd zmm24, zmm16,  0[r15]{1to8}    // a0 += c623_0 * w4 
    vfmadd213pd zmm25, zmm17,  0[r15]{1to8}    // a1 += c623_1 * w4 
    vfmadd213pd zmm26, zmm18,  0[r15]{1to8}    // a2 += c623_2 * w4 
    vfmadd213pd zmm27, zmm19,  0[r15]{1to8}    // a3 += c623_3 * w4 

	mov         r15,   ptr_w3                  //load address of perm

    vfmadd213pd zmm24, zmm16,  0[r15]{1to8}    // a0 += c623_0 * w3 
    vfmadd213pd zmm25, zmm17,  0[r15]{1to8}    // a1 += c623_1 * w3 
    vfmadd213pd zmm26, zmm18,  0[r15]{1to8}    // a2 += c623_2 * w3 
    vfmadd213pd zmm27, zmm19,  0[r15]{1to8}    // a3 += c623_3 * w3 

	mov         r15,   ptr_w2                  //load address of perm

    vfmadd213pd zmm24, zmm16,  0[r15]{1to8}    // a0 += c623_0 * w2 
    vfmadd213pd zmm25, zmm17,  0[r15]{1to8}    // a1 += c623_1 * w2 
    vfmadd213pd zmm26, zmm18,  0[r15]{1to8}    // a2 += c623_2 * w2 
    vfmadd213pd zmm27, zmm19,  0[r15]{1to8}    // a3 += c623_3 * w2 

	mov         r15,   ptr_w1                  //load address of perm

    vfmadd213pd zmm24, zmm16,  0[r15]{1to8}    // a0 += c623_0 * w1 
    vfmadd213pd zmm25, zmm17,  0[r15]{1to8}    // a1 += c623_1 * w1 
    vfmadd213pd zmm26, zmm18,  0[r15]{1to8}    // a2 += c623_2 * w1 
    vfmadd213pd zmm27, zmm19,  0[r15]{1to8}    // a3 += c623_3 * w1 

	mov         r15,   ptr_w0                  //load address of perm

    vfmadd213pd zmm24, zmm16,  0[r15]{1to8}    // a0 += c623_0 * w0 
    vfmadd213pd zmm25, zmm17,  0[r15]{1to8}    // a1 += c623_1 * w0 
    vfmadd213pd zmm26, zmm18,  0[r15]{1to8}    // a2 += c623_2 * w0 
    vfmadd213pd zmm27, zmm19,  0[r15]{1to8}    // a3 += c623_3 * w0 

	mov         r15,   ptr_dmone               //load address of -1, iteration #5

	vmulpd      zmm16, zmm24, zmm28            // c623_0 = a0 * p0 
	vmulpd      zmm17, zmm25, zmm29            // c623_1 = a1 * p1
	vmulpd      zmm18, zmm26, zmm30            // c623_2 = a2 * p2 
	vmulpd      zmm19, zmm27, zmm31            // c623_3 = a3 * p3 

    vbroadcastsd       zmm24,  0[r15]          // a0 = -1
	mov         r15,   ptr_log2e               //load address of log2e

    vfmadd231pd zmm24, zmm20, 0[r15]{1to8}     // a0 = c623_4 * log2e - 1
    vmovapd     zmm25, zmm24                   //
    vfmadd231pd zmm25, zmm21, 0[r15]{1to8}     // a1 = c623_5 * log2e - 1
    vmovapd     zmm26, zmm24                   //
    vfmadd231pd zmm26, zmm22, 0[r15]{1to8}     // a2 = c623_6 * log2e - 1
    vmovapd     zmm27, zmm24                   //
    vfmadd231pd zmm27, zmm23, 0[r15]{1to8}     // a3 = c623_7 * log2e - 1

	mov         r15,   ptr_c1                  //load address of c1

	vcvtfxpntpd2dq     zmm28, zmm24, 0x2       // k0 = double2int( a0 )
	vcvtfxpntpd2dq     zmm29, zmm25, 0x2       // k1 = double2int( a1 )
	vcvtfxpntpd2dq     zmm30, zmm26, 0x2       // k2 = double2int( a2 )
	vcvtfxpntpd2dq     zmm31, zmm27, 0x2       // k3 = double2int( a3 )

	vcvtdq2pd   zmm24, zmm28                   // p0 = int2double( k0 ) 
	vcvtdq2pd   zmm25, zmm29                   // p1 = int2double( k1 )
	vcvtdq2pd   zmm26, zmm30                   // p2 = int2double( k2 )
	vcvtdq2pd   zmm27, zmm31                   // p3 = int2double( k3 )

    vfmadd231pd zmm20, zmm24,  0[r15]{1to8}    // c623_4 += p0 * c1 
    vfmadd231pd zmm21, zmm25,  0[r15]{1to8}    // c623_5 += p1 * c1 
    vfmadd231pd zmm22, zmm26,  0[r15]{1to8}    // c623_6 += p2 * c1 
    vfmadd231pd zmm23, zmm27,  0[r15]{1to8}    // c623_7 += p3 * c1 

	mov         r15,   ptr_c2                  //load address of c2

    vfmadd231pd zmm20, zmm24,  0[r15]{1to8}    // c623_4 += p0 * c2 
    vfmadd231pd zmm21, zmm25,  0[r15]{1to8}    // c623_5 += p1 * c2 
    vfmadd231pd zmm22, zmm26,  0[r15]{1to8}    // c623_6 += p2 * c2 
    vfmadd231pd zmm23, zmm27,  0[r15]{1to8}    // c623_7 += p3 * c2 

	mov         r15,   ptr_mask                //load address of mask
    vmovdqa32          zmm24,  0[r15]          // zmm24 = perm 

	vpaddd      zmm28, zmm28,  zmm24           // k0 += mask
	vpaddd      zmm29, zmm29,  zmm24           // k1 += mask
	vpaddd      zmm30, zmm30,  zmm24           // k2 += mask
	vpaddd      zmm31, zmm31,  zmm24           // k3 += mask

	mov         r15,   ptr_perm                //load address of perm
    vmovdqa32          zmm24,  0[r15]          // zmm24 = perm

	vpermd      zmm28, zmm24,  zmm28           // permute( k0 )
	vpermd      zmm29, zmm24,  zmm29           // permute( k1 )
	vpermd      zmm30, zmm24,  zmm30           // permute( k2 )
	vpermd      zmm31, zmm24,  zmm31           // permute( k3 )

	mov         r15,   ptr_w11                 //load address of perm

	vpslld      zmm28, zmm28,  20              // shift k0<<20
	vpslld      zmm29, zmm29,  20              // shift k0<<20
	vpslld      zmm30, zmm30,  20              // shift k0<<20
	vpslld      zmm31, zmm31,  20              // shift k0<<20

    vbroadcastsd       zmm24,  w10             // a0 = w10
    vmovapd            zmm25,  zmm24           // a1 = w10
    vmovapd            zmm26,  zmm24           // a2 = w10
    vmovapd            zmm27,  zmm24           // a3 = w10

    vfmadd231pd zmm24, zmm20,  0[r15]{1to8}    // a0 += c623_4 * w11 
    vfmadd231pd zmm25, zmm21,  0[r15]{1to8}    // a1 += c623_5 * w11 
    vfmadd231pd zmm26, zmm22,  0[r15]{1to8}    // a2 += c623_6 * w11 
    vfmadd231pd zmm27, zmm23,  0[r15]{1to8}    // a3 += c623_7 * w11 

	mov         r15,   ptr_w9                  //load address of perm

    vfmadd213pd zmm24, zmm20,  0[r15]{1to8}    // a0 += c623_4 * w9 
    vfmadd213pd zmm25, zmm21,  0[r15]{1to8}    // a1 += c623_5 * w9 
    vfmadd213pd zmm26, zmm22,  0[r15]{1to8}    // a2 += c623_6 * w9 
    vfmadd213pd zmm27, zmm23,  0[r15]{1to8}    // a3 += c623_7 * w9 

	mov         r15,   ptr_w8                  //load address of perm

    vfmadd213pd zmm24, zmm20,  0[r15]{1to8}    // a0 += c623_4 * w8 
    vfmadd213pd zmm25, zmm21,  0[r15]{1to8}    // a1 += c623_5 * w8 
    vfmadd213pd zmm26, zmm22,  0[r15]{1to8}    // a2 += c623_6 * w8 
    vfmadd213pd zmm27, zmm23,  0[r15]{1to8}    // a3 += c623_7 * w8 

	mov         r15,   ptr_w7                  //load address of perm

    vfmadd213pd zmm24, zmm20,  0[r15]{1to8}    // a0 += c623_4 * w7 
    vfmadd213pd zmm25, zmm21,  0[r15]{1to8}    // a1 += c623_5 * w7 
    vfmadd213pd zmm26, zmm22,  0[r15]{1to8}    // a2 += c623_6 * w7 
    vfmadd213pd zmm27, zmm23,  0[r15]{1to8}    // a3 += c623_7 * w7 

	mov         r15,   ptr_w6                  //load address of perm

    vfmadd213pd zmm24, zmm20,  0[r15]{1to8}    // a0 += c623_4 * w6 
    vfmadd213pd zmm25, zmm21,  0[r15]{1to8}    // a1 += c623_5 * w6 
    vfmadd213pd zmm26, zmm22,  0[r15]{1to8}    // a2 += c623_6 * w6 
    vfmadd213pd zmm27, zmm23,  0[r15]{1to8}    // a3 += c623_7 * w6 

	mov         r15,   ptr_w5                  //load address of perm

    vfmadd213pd zmm24, zmm20,  0[r15]{1to8}    // a0 += c623_4 * w5 
    vfmadd213pd zmm25, zmm21,  0[r15]{1to8}    // a1 += c623_5 * w5 
    vfmadd213pd zmm26, zmm22,  0[r15]{1to8}    // a2 += c623_6 * w5 
    vfmadd213pd zmm27, zmm23,  0[r15]{1to8}    // a3 += c623_7 * w5 

	mov         r15,   ptr_w4                  //load address of perm

    vfmadd213pd zmm24, zmm20,  0[r15]{1to8}    // a0 += c623_4 * w4 
    vfmadd213pd zmm25, zmm21,  0[r15]{1to8}    // a1 += c623_5 * w4 
    vfmadd213pd zmm26, zmm22,  0[r15]{1to8}    // a2 += c623_6 * w4 
    vfmadd213pd zmm27, zmm23,  0[r15]{1to8}    // a3 += c623_7 * w4 

	mov         r15,   ptr_w3                  //load address of perm

    vfmadd213pd zmm24, zmm20,  0[r15]{1to8}    // a0 += c623_4 * w3 
    vfmadd213pd zmm25, zmm21,  0[r15]{1to8}    // a1 += c623_5 * w3 
    vfmadd213pd zmm26, zmm22,  0[r15]{1to8}    // a2 += c623_6 * w3 
    vfmadd213pd zmm27, zmm23,  0[r15]{1to8}    // a3 += c623_7 * w3 

	mov         r15,   ptr_w2                  //load address of perm

    vfmadd213pd zmm24, zmm20,  0[r15]{1to8}    // a0 += c623_4 * w2 
    vfmadd213pd zmm25, zmm21,  0[r15]{1to8}    // a1 += c623_5 * w2 
    vfmadd213pd zmm26, zmm22,  0[r15]{1to8}    // a2 += c623_6 * w2 
    vfmadd213pd zmm27, zmm23,  0[r15]{1to8}    // a3 += c623_7 * w2 

	mov         r15,   ptr_w1                  //load address of perm

    vfmadd213pd zmm24, zmm20,  0[r15]{1to8}    // a0 += c623_4 * w1 
    vfmadd213pd zmm25, zmm21,  0[r15]{1to8}    // a1 += c623_5 * w1 
    vfmadd213pd zmm26, zmm22,  0[r15]{1to8}    // a2 += c623_6 * w1 
    vfmadd213pd zmm27, zmm23,  0[r15]{1to8}    // a3 += c623_7 * w1 

	mov         r15,   ptr_w0                  //load address of perm

    vfmadd213pd zmm24, zmm20,  0[r15]{1to8}    // a0 += c623_4 * w0 
    vfmadd213pd zmm25, zmm21,  0[r15]{1to8}    // a1 += c623_5 * w0 
    vfmadd213pd zmm26, zmm22,  0[r15]{1to8}    // a2 += c623_6 * w0 
    vfmadd213pd zmm27, zmm23,  0[r15]{1to8}    // a3 += c623_7 * w0 

	vmulpd      zmm20, zmm24, zmm28            // c623_4 = a0 * p0 
	vmulpd      zmm21, zmm25, zmm29            // c623_5 = a1 * p1
	vmulpd      zmm22, zmm26, zmm30            // c623_6 = a2 * p2 
	vmulpd      zmm23, zmm27, zmm31            // c623_7 = a3 * p3 

	mov         rbx,   w                       //load address of w
	mov         rax,   u                       //load address of u

    vmovapd            zmm24,  0[rax]          // u007
    vbroadcastf64x4    zmm30,  0[rbx]          // w0303

    vfmadd231pd zmm24, zmm0,  zmm30{aaaa}      // u007 += c007_0 * w0
    vfmadd231pd zmm24, zmm1,  zmm30{bbbb}      // u007 += c007_1 * w1
    vmovapd            zmm25, 64[rax]          // u815
    vfmadd231pd zmm24, zmm2,  zmm30{cccc}      // u007 += c007_2 * w2
    vfmadd231pd zmm24, zmm3,  zmm30{dddd}      // u007 += c007_3 * w3

    vfmadd231pd zmm25, zmm8,  zmm30{aaaa}      // u815 += c815_0 * w0
    vfmadd231pd zmm25, zmm9,  zmm30{bbbb}      // u815 += c815_1 * w1
    vbroadcastf64x4    zmm31, 32[rbx]          // w4747
    vfmadd231pd zmm25, zmm10, zmm30{bbbb}      // u815 += c815_2 * w2
    vfmadd231pd zmm25, zmm11, zmm30{bbbb}      // u815 += c815_3 * w3

    vfmadd231pd zmm24, zmm4,  zmm31{aaaa}      // u007 += c007_4 * w4
    vfmadd231pd zmm24, zmm5,  zmm31{bbbb}      // u007 += c007_5 * w5
    vmovapd            zmm26, 128[rax]         // u623
    vfmadd231pd zmm24, zmm6,  zmm31{cccc}      // u007 += c007_6 * w6
    vfmadd231pd zmm24, zmm7,  zmm31{dddd}      // u007 += c007_7 * w7

	vmovapd     0[rax], zmm24                  // store u007

    vfmadd231pd zmm25, zmm12, zmm31{aaaa}      // u815 += c815_4 * w4
    vfmadd231pd zmm25, zmm13, zmm31{bbbb}      // u815 += c815_5 * w5
    vfmadd231pd zmm25, zmm14, zmm31{cccc}      // u815 += c815_6 * w6
    vfmadd231pd zmm25, zmm15, zmm31{dddd}      // u815 += c815_7 * w7

	vmovapd     64[rax], zmm25                  // store u815

    vfmadd231pd zmm26, zmm16, zmm30{aaaa}      // u623 += c623_0 * w0
    vfmadd231pd zmm26, zmm17, zmm30{bbbb}      // u623 += c623_1 * w1
    vfmadd231pd zmm26, zmm18, zmm30{cccc}      // u623 += c623_2 * w2
    vfmadd231pd zmm26, zmm19, zmm30{dddd}      // u623 += c623_3 * w3

    vfmadd231pd zmm26, zmm20, zmm31{aaaa}      // u623 += c623_4 * w4
    vfmadd231pd zmm26, zmm21, zmm31{bbbb}      // u623 += c623_5 * w5
    vfmadd231pd zmm26, zmm22, zmm31{cccc}      // u623 += c623_6 * w6
    vfmadd231pd zmm26, zmm23, zmm31{dddd}      // u623 += c623_7 * w7

	vmovapd     128[rax], zmm26                  // store u623

	END:

	//mov         rcx,    ptr_c                    //load address of c
	//vmovapd     0[rcx],   zmm24
	//vmovapd     64[rcx],  zmm25
	//vmovapd     128[rcx], zmm26
	//vmovapd     192[rcx], zmm23
	//vmovapd     256[rcx], zmm4
	//vmovapd     320[rcx], zmm5

  }

//   printf( "c007_0\n" );
//
//   printf( "%E, %E, %E, %E, %E, %E, %E, %E\n",
//	   c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
//   printf( "c815_0\n" );
//   printf( "%E, %E, %E, %E, %E, %E, %E, %E\n",
//	   c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15]);
//   printf( "c623_0\n" );
//   printf( "%E, %E, %E, %E, %E, %E, %E, %E\n",
//	   c[16], c[17], c[18], c[19], c[20], c[21], c[22], c[23]);
//

//   int *cint = (int*)c;
//
//   printf( "%d, %d, %d, %d, %d, %d, %d, %d\n",
//	   cint[0], cint[1], cint[2], cint[3], cint[4], cint[5], cint[6], cint[7]);
//   printf( "%d, %d, %d, %d, %d, %d, %d, %d\n",
//	   cint[8], cint[9], cint[10], cint[11], cint[12], cint[13], cint[14], cint[15]);



}
