#include <immintrin.h>
#include <assert.h>
#include <ks.h>

typedef unsigned long long dim_t;
typedef unsigned long long inc_t;


#define A_L1_PREFETCH_DIST 4
#define B_L1_PREFETCH_DIST 2
#define L2_PREFETCH_DIST  16 // Must be greater than 10, because of the way the loop is constructed.

//One iteration of the k_r loop.
//Each iteration, we prefetch A into L1 and into L2
#define ONE_ITER_MAIN_LOOP(C_ADDR, COUNTER) \
{\
        __asm vbroadcastf64x4   zmm30, 0[r15]           \
        __asm vmovapd zmm31, 0[rax]                     \
                                                        \
        __asm vfmadd231pd zmm0, zmm31, zmm30{aaaa}      \
        __asm vfmadd231pd zmm4, zmm31,  4*8[r15]{1to8}  \
        __asm vprefetch0 A_L1_PREFETCH_DIST*256[r15]    \
        __asm vfmadd231pd zmm5, zmm31,  5*8[r15]{1to8}  \
        __asm vprefetch0 A_L1_PREFETCH_DIST*256+64[r15] \
        __asm vfmadd231pd zmm6, zmm31,  6*8[r15]{1to8}  \
        __asm vprefetch0 A_L1_PREFETCH_DIST*256+128[r15]\
        __asm vfmadd231pd zmm7, zmm31,  7*8[r15]{1to8}  \
        __asm vprefetch0 A_L1_PREFETCH_DIST*256+192[r15]\
        __asm vfmadd231pd zmm8, zmm31,  8*8[r15]{1to8}  \
                                                        \
        __asm vprefetch1 0[r15 + r14]                   \
        __asm vfmadd231pd zmm9, zmm31,  9*8[r15]{1to8}  \
        __asm vfmadd231pd zmm1, zmm31, zmm30{bbbb}      \
        __asm vfmadd231pd zmm2, zmm31, zmm30{cccc}      \
        __asm vfmadd231pd zmm3, zmm31, zmm30{dddd}      \
        __asm vfmadd231pd zmm10, zmm31, 10*8[r15]{1to8} \
                                                        \
        __asm vprefetch1 64[r15 + r14]                  \
        __asm vfmadd231pd zmm11, zmm31, 11*8[r15]{1to8} \
        __asm vfmadd231pd zmm12, zmm31, 12*8[r15]{1to8} \
        __asm vfmadd231pd zmm13, zmm31, 13*8[r15]{1to8} \
        __asm vfmadd231pd zmm14, zmm31, 14*8[r15]{1to8} \
        __asm vfmadd231pd zmm15, zmm31, 15*8[r15]{1to8} \
                                                        \
        __asm vprefetch1 2*64[r15 + r14]                \
        __asm vfmadd231pd zmm16, zmm31, 16*8[r15]{1to8} \
        __asm vfmadd231pd zmm17, zmm31, 17*8[r15]{1to8} \
        __asm vfmadd231pd zmm18, zmm31, 18*8[r15]{1to8} \
        __asm vfmadd231pd zmm19, zmm31, 19*8[r15]{1to8} \
        __asm vfmadd231pd zmm20, zmm31, 20*8[r15]{1to8} \
                                                        \
        __asm vprefetch1 3*64[r15 + r14]                \
        __asm vfmadd231pd zmm21, zmm31, 21*8[r15]{1to8} \
        __asm add r15, r12                              \
        __asm vfmadd231pd zmm22, zmm31, -10*8[r15]{1to8}\
        __asm vfmadd231pd zmm23, zmm31, -9*8[r15]{1to8} \
        __asm vfmadd231pd zmm24, zmm31, -8*8[r15]{1to8} \
        __asm dec COUNTER                               \
        __asm vfmadd231pd zmm25, zmm31, -7*8[r15]{1to8} \
                                                        \
                                                        \
        __asm vprefetch1 0[rax + r13]                   \
        __asm vfmadd231pd zmm26, zmm31, -6*8[r15]{1to8} \
        __asm vprefetch0 B_L1_PREFETCH_DIST*8*8[rax]    \
        __asm vfmadd231pd zmm27, zmm31, -5*8[r15]{1to8} \
        __asm add rax, r9                               \
        __asm vfmadd231pd zmm28, zmm31, -4*8[r15]{1to8} \
        __asm cmp COUNTER, 0                            \
        __asm vfmadd231pd zmm29, zmm31, -3*8[r15]{1to8} \
}

//One iteration of the k_r loop.
//Same as ONE_ITER_MAIN_LOOP, but additionally, we prefetch one line of C into the L2 cache
//Current placement of this prefetch instruction is somewhat arbitrary.
#define ONE_ITER_PC_L2(C_ADDR) \
{\
        __asm vbroadcastf64x4   zmm30, 0[r15]           \
        __asm vmovapd zmm31, 0[rax]                     \
                                                        \
        __asm vfmadd231pd zmm0, zmm31, zmm30{aaaa}      \
        __asm vfmadd231pd zmm4, zmm31,  4*8[r15]{1to8}  \
        __asm vprefetch0 A_L1_PREFETCH_DIST*256[r15]    \
        __asm vfmadd231pd zmm5, zmm31,  5*8[r15]{1to8}  \
        __asm vprefetch0 A_L1_PREFETCH_DIST*256+64[r15] \
        __asm vfmadd231pd zmm6, zmm31,  6*8[r15]{1to8}  \
        __asm vprefetch0 A_L1_PREFETCH_DIST*256+128[r15]\
        __asm vfmadd231pd zmm7, zmm31,  7*8[r15]{1to8}  \
        __asm vprefetch0 A_L1_PREFETCH_DIST*256+192[r15]\
        __asm vfmadd231pd zmm8, zmm31,  8*8[r15]{1to8}  \
                                                        \
        __asm vprefetch1 0[r15 + r14]                   \
        __asm vfmadd231pd zmm9, zmm31,  9*8[r15]{1to8}  \
        __asm vfmadd231pd zmm1, zmm31, zmm30{bbbb}      \
        __asm vfmadd231pd zmm2, zmm31, zmm30{cccc}      \
        __asm vfmadd231pd zmm3, zmm31, zmm30{dddd}      \
        __asm vfmadd231pd zmm10, zmm31, 10*8[r15]{1to8} \
                                                        \
        __asm vprefetch1 64[r15 + r14]                  \
        __asm vfmadd231pd zmm11, zmm31, 11*8[r15]{1to8} \
        __asm vprefetch1 0[C_ADDR]                      \
        __asm vfmadd231pd zmm12, zmm31, 12*8[r15]{1to8} \
        __asm vfmadd231pd zmm13, zmm31, 13*8[r15]{1to8} \
        __asm vfmadd231pd zmm14, zmm31, 14*8[r15]{1to8} \
        __asm vfmadd231pd zmm15, zmm31, 15*8[r15]{1to8} \
                                                        \
        __asm vprefetch1 2*64[r15 + r14]                \
        __asm vfmadd231pd zmm16, zmm31, 16*8[r15]{1to8} \
        __asm vfmadd231pd zmm17, zmm31, 17*8[r15]{1to8} \
        __asm vfmadd231pd zmm18, zmm31, 18*8[r15]{1to8} \
        __asm vfmadd231pd zmm19, zmm31, 19*8[r15]{1to8} \
        __asm vfmadd231pd zmm20, zmm31, 20*8[r15]{1to8} \
                                                        \
        __asm vprefetch1 3*64[r15 + r14]                \
        __asm vfmadd231pd zmm21, zmm31, 21*8[r15]{1to8} \
        __asm add r15, r12                              \
        __asm vfmadd231pd zmm22, zmm31, -10*8[r15]{1to8}\
        __asm vfmadd231pd zmm23, zmm31, -9*8[r15]{1to8} \
        __asm add C_ADDR, r11                           \
        __asm vfmadd231pd zmm24, zmm31, -8*8[r15]{1to8} \
        __asm dec r8                                    \
        __asm vfmadd231pd zmm25, zmm31, -7*8[r15]{1to8} \
                                                        \
                                                        \
        __asm vprefetch1 0[rax + r13]                   \
        __asm vfmadd231pd zmm26, zmm31, -6*8[r15]{1to8} \
        __asm vprefetch0 B_L1_PREFETCH_DIST*8*8[rax]    \
        __asm vfmadd231pd zmm27, zmm31, -5*8[r15]{1to8} \
        __asm add rax, r9                               \
        __asm vfmadd231pd zmm28, zmm31, -4*8[r15]{1to8} \
        __asm cmp r8, 0                                 \
        __asm vfmadd231pd zmm29, zmm31, -3*8[r15]{1to8} \
\
}

//One iteration of the k_r loop.
//Same as ONE_ITER_MAIN_LOOP, but additionally, we prefetch 3 cache lines of C into the L1 cache
//Current placement of these prefetch instructions is somewhat arbitrary.
#define ONE_ITER_PC_L1(C_ADDR) \
{\
        __asm vbroadcastf64x4   zmm30, 0[r15]           \
        __asm vmovapd zmm31, 0[rax]                     \
                                                        \
        __asm vfmadd231pd zmm0, zmm31, zmm30{aaaa}      \
        __asm vfmadd231pd zmm4, zmm31,  4*8[r15]{1to8}  \
        __asm vprefetch0 A_L1_PREFETCH_DIST*256[r15]    \
        __asm vfmadd231pd zmm5, zmm31,  5*8[r15]{1to8}  \
        __asm vprefetch0 A_L1_PREFETCH_DIST*256+64[r15] \
        __asm vfmadd231pd zmm6, zmm31,  6*8[r15]{1to8}  \
        __asm vprefetch0 A_L1_PREFETCH_DIST*256+128[r15]\
        __asm vfmadd231pd zmm7, zmm31,  7*8[r15]{1to8}  \
        __asm vprefetch0 A_L1_PREFETCH_DIST*256+192[r15]\
        __asm vfmadd231pd zmm8, zmm31,  8*8[r15]{1to8}  \
                                                        \
        __asm vprefetch1 0[r15 + r14]                   \
        __asm vfmadd231pd zmm9, zmm31,  9*8[r15]{1to8}  \
        __asm vprefetch0 0[C_ADDR]                      \
        __asm vfmadd231pd zmm1, zmm31, zmm30{bbbb}      \
        __asm add C_ADDR, r11 \
        __asm vfmadd231pd zmm2, zmm31, zmm30{cccc}      \
        __asm vfmadd231pd zmm3, zmm31, zmm30{dddd}      \
        __asm vfmadd231pd zmm10, zmm31, 10*8[r15]{1to8} \
                                                        \
        __asm vprefetch1 64[r15 + r14]                  \
        __asm vfmadd231pd zmm11, zmm31, 11*8[r15]{1to8} \
        __asm vprefetch0 0[C_ADDR]                      \
        __asm vfmadd231pd zmm12, zmm31, 12*8[r15]{1to8} \
        __asm add C_ADDR, r11 \
        __asm vfmadd231pd zmm13, zmm31, 13*8[r15]{1to8} \
        __asm vfmadd231pd zmm14, zmm31, 14*8[r15]{1to8} \
        __asm vfmadd231pd zmm15, zmm31, 15*8[r15]{1to8} \
                                                        \
        __asm vprefetch1 2*64[r15 + r14]                \
        __asm vfmadd231pd zmm16, zmm31, 16*8[r15]{1to8} \
        __asm vprefetch0 0[C_ADDR]                      \
        __asm vfmadd231pd zmm17, zmm31, 17*8[r15]{1to8} \
        __asm add C_ADDR, r11                           \
        __asm vfmadd231pd zmm18, zmm31, 18*8[r15]{1to8} \
        __asm vfmadd231pd zmm19, zmm31, 19*8[r15]{1to8} \
        __asm vfmadd231pd zmm20, zmm31, 20*8[r15]{1to8} \
                                                        \
        __asm vprefetch1 3*64[r15 + r14]                \
        __asm vfmadd231pd zmm21, zmm31, 21*8[r15]{1to8} \
        __asm add r15, r12                              \
        __asm vfmadd231pd zmm22, zmm31, -10*8[r15]{1to8}\
        __asm vfmadd231pd zmm23, zmm31, -9*8[r15]{1to8} \
        __asm vfmadd231pd zmm24, zmm31, -8*8[r15]{1to8} \
        __asm dec r8                                    \
        __asm vfmadd231pd zmm25, zmm31, -7*8[r15]{1to8} \
                                                        \
                                                        \
        __asm vprefetch1 0[rax + r13]                   \
        __asm vfmadd231pd zmm26, zmm31, -6*8[r15]{1to8} \
        __asm vprefetch0 B_L1_PREFETCH_DIST*8*8[rax]    \
        __asm vfmadd231pd zmm27, zmm31, -5*8[r15]{1to8} \
        __asm add rax, r9                               \
        __asm vfmadd231pd zmm28, zmm31, -4*8[r15]{1to8} \
        __asm cmp r8, 0                                 \
        __asm vfmadd231pd zmm29, zmm31, -3*8[r15]{1to8} \
\
}


//#define MONITORS
//#define LOOPMON



const int iperm[ 16 ] __attribute__((aligned(64))) = { 8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7 };
const int imask[ 16 ] __attribute__((aligned(64)))  
	       = { 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 
	           0, 0, 0, 0, 0, 0, 0, 0 };

void ks_gaussian_asm_d8x30(
    dim_t  k,
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
  //double c[ 8 * 32 ] __attribute__((aligned(64)));

  double *a_next = aux->a_next;
  double *b_next = aux->b_next;
  double *c_buff = aux->c_buff;

  const double dmone  = -1.0;
  const double dmtwo  = -2.0;
  const double dzero  =  0.0;
  const double log2e  =  1.4426950408889634073599;
  const double maxlog =  7.09782712893383996843e2; // log( 2**1024 )
  const double minlog = -7.08396418532264106224e2; // log( 2**-1024 )
  const double c1     = -6.93145751953125E-1;
  const double c2     = -1.42860682030941723212E-6;
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

  const double *ptr_mask   = (double*) imask;
  const double *ptr_perm   = (double*) iperm;
  const double *m2_ptr     = &dmtwo;
  const double *ptr_dzero  = &dzero;
  const double *ptr_dmone  = &dmone;
  const double *ptr_minlog = &minlog;
  const double *ptr_log2e  = &log2e;
  const double *ptr_alpha  = &alpha;
  const double *ptr_c1     = &c1;
  const double *ptr_c2     = &c2;
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


  //printf( "%E, %E, %E, %E, %E, %E, %E, %E\n",
	 //  w[ 0 ], w[ 1 ], w[ 2 ], w[ 3 ], w[ 4 ], w[ 5 ], w[ 6 ], w[ 7 ]);




    __asm
    {
        vpxord  zmm0,  zmm0, zmm0
        vmovaps zmm1,  zmm0  //clear out registers
        vmovaps zmm2,  zmm0 
        mov rsi, k    //loop index
        vmovaps zmm3,  zmm0 

//        mov r11, rs_c           //load row stride
        vmovaps zmm4,  zmm0 
//        sal r11, 3              //scale row stride
        vmovaps zmm5,  zmm0 
        mov rax, a              //load address of a
        vmovaps zmm6,  zmm0 
        mov r15, b              //load address of b
        vmovaps zmm7,  zmm0 

        vmovaps zmm8,  zmm0 
//        lea r10, [r11 + 2*r11 + 0] //r10 has 3 * r11
        vmovaps zmm9,  zmm0
        vmovaps zmm10, zmm0 
//        mov rdi, r11    
        vmovaps zmm11, zmm0 
//        sal rdi, 2              //rdi has 4*r11

        vmovaps zmm12, zmm0 
        mov rcx, c_buff           //load address of c for prefetching
        vmovaps zmm13, zmm0 
        vmovaps zmm14, zmm0 
        mov r8, k 
        vmovaps zmm15, zmm0 

        vmovaps zmm16, zmm0
        vmovaps zmm17, zmm0
        mov r13, L2_PREFETCH_DIST*8*8
        vmovaps zmm18, zmm0 
        mov r14, L2_PREFETCH_DIST*8*32
        vmovaps zmm19, zmm0 
        vmovaps zmm20, zmm0 
        vmovaps zmm21, zmm0 
        vmovaps zmm22, zmm0 

        vmovaps zmm23, zmm0 
        sub r8, 30 + L2_PREFETCH_DIST       //Check if we have over 40 operations to do.
        vmovaps zmm24, zmm0 
        mov r8, 30
        vmovaps zmm25, zmm0 
        mov r9, 8*8                         //amount to increment b* by each iteration
        vmovaps zmm26, zmm0 
        mov r12, 32*8                       //amount to increment a* by each iteration
        vmovaps zmm27, zmm0 
        vmovaps zmm28, zmm0 
        vmovaps zmm29, zmm0 

        jle CONSIDER_UNDER_40
        sub rsi, 30 + L2_PREFETCH_DIST
        
        //First 30 iterations
        LOOPREFECHCL2:
            ONE_ITER_PC_L2(rcx)
        jne LOOPREFECHCL2
        mov rcx, c_buff

        //Main Loop.
        LOOPMAIN:
            ONE_ITER_MAIN_LOOP(rcx, rsi)
        jne LOOPMAIN
        
        //Penultimate 22 iterations.
        //Break these off from the main loop to avoid prefetching extra shit.
        mov r14, b_next
        mov r13, a_next
        sub r14, r15
        sub r13, rax
        
        mov rsi, L2_PREFETCH_DIST-10
        LOOPMAIN2:
            ONE_ITER_MAIN_LOOP(rcx, rsi)
        jne LOOPMAIN2
        
        
        //Last 10 iterations
        mov r8, 10
        LOOPREFETCHCL1:
            ONE_ITER_PC_L1(rcx)
        jne LOOPREFETCHCL1
       

        jmp POSTACCUM

        //Alternate main loop, with no prefetching of C
        //Used when <= 40 iterations
        CONSIDER_UNDER_40:
        mov rsi, k
        test rsi, rsi
        je POSTACCUM
        LOOP_UNDER_40:
            ONE_ITER_MAIN_LOOP(rcx, rsi)
        jne LOOP_UNDER_40



    POSTACCUM:

    mov r9, c_buff                             //load address of c for update
    mov r12, m2_ptr                            //load address of -2
	mov rax, aa                                //load address of aa
	mov rbx, bb                                //load address of bb

    vbroadcastsd       zmm30,  0[r12]          // neg2
    vmovapd            zmm31,  0[rax]          // aa007

    vfmadd213pd zmm0,  zmm30, zmm31            // c007_0 = c007_0 * neg2 + aa007
    vfmadd213pd zmm1,  zmm30, zmm31            // 
    vfmadd213pd zmm2,  zmm30, zmm31            // 
    vfmadd213pd zmm3,  zmm30, zmm31            // 
    vfmadd213pd zmm4,  zmm30, zmm31            // 
    vfmadd213pd zmm5,  zmm30, zmm31            // 
    vfmadd213pd zmm6,  zmm30, zmm31            // 
    vfmadd213pd zmm7,  zmm30, zmm31            // 
    vfmadd213pd zmm8,  zmm30, zmm31            // 
    vfmadd213pd zmm9,  zmm30, zmm31            // 
    vfmadd213pd zmm10, zmm30, zmm31            // 
    vfmadd213pd zmm11, zmm30, zmm31            // 
    vfmadd213pd zmm12, zmm30, zmm31            // 
    vfmadd213pd zmm13, zmm30, zmm31            // 
    vfmadd213pd zmm14, zmm30, zmm31            // 
    vfmadd213pd zmm15, zmm30, zmm31            // 
    vfmadd213pd zmm16, zmm30, zmm31            // 
    vfmadd213pd zmm17, zmm30, zmm31            // 
    vfmadd213pd zmm18, zmm30, zmm31            // 
    vfmadd213pd zmm19, zmm30, zmm31            // 
    vfmadd213pd zmm20, zmm30, zmm31            // 
    vfmadd213pd zmm21, zmm30, zmm31            // 
    vfmadd213pd zmm22, zmm30, zmm31            // 
    vfmadd213pd zmm23, zmm30, zmm31            // 
    vfmadd213pd zmm24, zmm30, zmm31            // 
    vfmadd213pd zmm25, zmm30, zmm31            // 
    vfmadd213pd zmm26, zmm30, zmm31            // 
    vfmadd213pd zmm27, zmm30, zmm31            // 
    vfmadd213pd zmm28, zmm30, zmm31            // 
    vfmadd213pd zmm29, zmm30, zmm31            // c007_29 = c007_29 * neg2 + aa007 

    vbroadcastf64x4    zmm30,  0[rbx]          // bb0303
    vbroadcastf64x4    zmm31, 32[rbx]          // bb4747

    vaddpd      zmm0,  zmm0,  zmm30{aaaa}      // c007_0 += bb0
    vaddpd      zmm1,  zmm1,  zmm30{bbbb}      // 
    vaddpd      zmm2,  zmm2,  zmm30{cccc}      // 
    vaddpd      zmm3,  zmm3,  zmm30{dddd}      // 

    vbroadcastf64x4    zmm30, 64[rbx]          // bb8181

    vaddpd      zmm4,  zmm4,  zmm31{aaaa}      // c007_4 += bb4
    vaddpd      zmm5,  zmm5,  zmm31{bbbb}      // c007_5 += bb5
    vaddpd      zmm6,  zmm6,  zmm31{cccc}      // c007_6 += bb6
    vaddpd      zmm7,  zmm7,  zmm31{dddd}      // c007_7 += bb7

    vbroadcastf64x4    zmm31, 96[rbx]          // bb2525

    vaddpd      zmm8,  zmm8,  zmm30{aaaa}      // c007_8 += bb8
    vaddpd      zmm9,  zmm9,  zmm30{bbbb}      // 
    vaddpd      zmm10, zmm10, zmm30{cccc}      // 
    vaddpd      zmm11, zmm11, zmm30{dddd}      // 

    vbroadcastf64x4    zmm30, 128[rbx]         // bb6969

    vaddpd      zmm12, zmm12, zmm31{aaaa}      // c007_12 += bb12
    vaddpd      zmm13, zmm13, zmm31{bbbb}      // 
    vaddpd      zmm14, zmm14, zmm31{cccc}      // 
    vaddpd      zmm15, zmm15, zmm31{dddd}      // 

    vbroadcastf64x4    zmm31, 160[rbx]         // bb0303

    vaddpd      zmm16, zmm16, zmm30{aaaa}      // c007_16 += bb16
    vaddpd      zmm17, zmm17, zmm30{bbbb}      // 
    vaddpd      zmm18, zmm18, zmm30{cccc}      // 
    vaddpd      zmm19, zmm19, zmm30{dddd}      // 

    vbroadcastf64x4    zmm30, 192[rbx]         // bb4747

    vaddpd      zmm20, zmm20, zmm31{aaaa}      // c007_20 += bb20
    vaddpd      zmm21, zmm21, zmm31{bbbb}      // 
    vaddpd      zmm22, zmm22, zmm31{cccc}      // 
    vaddpd      zmm23, zmm23, zmm31{dddd}      // 

    mov r15, ptr_alpha                 //load address of alpha

    vaddpd      zmm24, zmm24, zmm30{aaaa}      // c007_24 += bb24
    vaddpd      zmm25, zmm25, zmm30{bbbb}      // 
    vaddpd      zmm26, zmm26, zmm30{cccc}      // 
    vaddpd      zmm27, zmm27, zmm30{dddd}      // 


    vbroadcastsd       zmm30,  0[r15]          // alpha

    vaddpd      zmm28, zmm28, 224[rbx]{1to8}   // c007_28 += bb28
    vaddpd      zmm29, zmm29, 232[rbx]{1to8}   // c007_29 += bb29


	// At this moment,
	//
	// zmm0 ~ zmm29 = square distance
	// zmm30        = alpha
	//

    vmulpd      zmm0,  zmm0,  zmm30            // c007_0 *= alpha 
    vmulpd      zmm1,  zmm1,  zmm30            // 
    vmulpd      zmm2,  zmm2,  zmm30            // 
    vmulpd      zmm3,  zmm3,  zmm30            // 
    vmulpd      zmm4,  zmm4,  zmm30            // 
    vmulpd      zmm5,  zmm5,  zmm30            // 
    vmulpd      zmm6,  zmm6,  zmm30            // 
    vmulpd      zmm7,  zmm7,  zmm30            // 
    vmulpd      zmm8,  zmm8,  zmm30            // 
    vmulpd      zmm9,  zmm9,  zmm30            // 
    vmulpd      zmm10, zmm10, zmm30            // 

	mov         r15,   ptr_minlog              //load address of minlog
    vbroadcastsd       zmm31, 0[r15]           // minlog

    vmulpd      zmm11, zmm11, zmm30            // 
    vmulpd      zmm12, zmm12, zmm30            // 
    vmulpd      zmm13, zmm13, zmm30            // 
    vmulpd      zmm14, zmm14, zmm30            // 
    vmulpd      zmm15, zmm15, zmm30            // 
    vmulpd      zmm16, zmm16, zmm30            // 
    vmulpd      zmm17, zmm17, zmm30            // 
    vmulpd      zmm18, zmm18, zmm30            // 
    vmulpd      zmm19, zmm19, zmm30            // 
    vmulpd      zmm20, zmm20, zmm30            // 

	mov         r12,   ptr_dzero               //load address of 0

    vmulpd      zmm21, zmm21, zmm30            // 
    vmulpd      zmm22, zmm22, zmm30            // 
    vmulpd      zmm23, zmm23, zmm30            // 
    vmulpd      zmm24, zmm24, zmm30            // 
    vmulpd      zmm25, zmm25, zmm30            // 
    vmulpd      zmm26, zmm26, zmm30            // 
    vmulpd      zmm27, zmm27, zmm30            // 
    vmulpd      zmm28, zmm28, zmm30            // 
    vmulpd      zmm29, zmm29, zmm30            // 

	
    vgminpd     zmm0,  zmm0,  0[r12]{1to8}     // min( c, 0.0 ) 
    vgminpd     zmm1,  zmm1,  0[r12]{1to8}     //
    vgminpd     zmm2,  zmm2,  0[r12]{1to8}     //
    vgminpd     zmm3,  zmm3,  0[r12]{1to8}     //
    vgminpd     zmm4,  zmm4,  0[r12]{1to8}     //
    vgminpd     zmm5,  zmm5,  0[r12]{1to8}     //
    vgminpd     zmm6,  zmm6,  0[r12]{1to8}     //
    vgminpd     zmm7,  zmm7,  0[r12]{1to8}     //
    vgminpd     zmm8,  zmm8,  0[r12]{1to8}     //
    vgminpd     zmm9,  zmm9,  0[r12]{1to8}     //
    vgminpd     zmm10, zmm10, 0[r12]{1to8}     //
    vgminpd     zmm11, zmm11, 0[r12]{1to8}     //
    vgminpd     zmm12, zmm12, 0[r12]{1to8}     //
    vgminpd     zmm13, zmm13, 0[r12]{1to8}     //
    vgminpd     zmm14, zmm14, 0[r12]{1to8}     //
    vgminpd     zmm15, zmm15, 0[r12]{1to8}     //
    vgminpd     zmm16, zmm16, 0[r12]{1to8}     //
    vgminpd     zmm17, zmm17, 0[r12]{1to8}     //
    vgminpd     zmm18, zmm18, 0[r12]{1to8}     //
    vgminpd     zmm19, zmm19, 0[r12]{1to8}     //
    vgminpd     zmm20, zmm20, 0[r12]{1to8}     //
    vgminpd     zmm21, zmm21, 0[r12]{1to8}     //
    vgminpd     zmm22, zmm22, 0[r12]{1to8}     //
    vgminpd     zmm23, zmm23, 0[r12]{1to8}     //
    vgminpd     zmm24, zmm24, 0[r12]{1to8}     //
    vgminpd     zmm25, zmm25, 0[r12]{1to8}     //
    vgminpd     zmm26, zmm26, 0[r12]{1to8}     //
    vgminpd     zmm27, zmm27, 0[r12]{1to8}     //
    vgminpd     zmm28, zmm28, 0[r12]{1to8}     //
    vgminpd     zmm29, zmm29, 0[r12]{1to8}     //


	mov         r12,   ptr_dmone               //load address of dmone
	mov         r15,   ptr_log2e               //load address of log2e
    vbroadcastsd       zmm30, 0[r12]           // -1.0


    vgmaxpd     zmm0,  zmm0,  zmm31            // max( c, minlog ) 
    vgmaxpd     zmm1,  zmm1,  zmm31            //
    vgmaxpd     zmm2,  zmm2,  zmm31            //
    vgmaxpd     zmm3,  zmm3,  zmm31            //
    vgmaxpd     zmm4,  zmm4,  zmm31            //
    vgmaxpd     zmm5,  zmm5,  zmm31            //
    vgmaxpd     zmm6,  zmm6,  zmm31            //
    vgmaxpd     zmm7,  zmm7,  zmm31            //
    vgmaxpd     zmm8,  zmm8,  zmm31            //
    vgmaxpd     zmm9,  zmm9,  zmm31            //
    vgmaxpd     zmm10, zmm10, zmm31            //
    vgmaxpd     zmm11, zmm11, zmm31            //
    vgmaxpd     zmm12, zmm12, zmm31            //
    vgmaxpd     zmm13, zmm13, zmm31            //
    vgmaxpd     zmm14, zmm14, zmm31            //
    vgmaxpd     zmm15, zmm15, zmm31            //
    vgmaxpd     zmm16, zmm16, zmm31            //
    vgmaxpd     zmm17, zmm17, zmm31            //
    vgmaxpd     zmm18, zmm18, zmm31            //
    vgmaxpd     zmm19, zmm19, zmm31            //
    vgmaxpd     zmm20, zmm20, zmm31            //
    vgmaxpd     zmm21, zmm21, zmm31            //
    vgmaxpd     zmm22, zmm22, zmm31            //
    vgmaxpd     zmm23, zmm23, zmm31            //
    vgmaxpd     zmm24, zmm24, zmm31            //
    vgmaxpd     zmm25, zmm25, zmm31            //
    vgmaxpd     zmm26, zmm26, zmm31            //
    vgmaxpd     zmm27, zmm27, zmm31            //
    vgmaxpd     zmm28, zmm28, zmm31            //
    vgmaxpd     zmm29, zmm29, zmm31            //


	// At this moment, we are going to store part of c
	// back to the memory.

	vmovapd        0[r9], zmm0                 // storing c007_0 ~ c007_19
    vmovapd     zmm0,  zmm30                    
    vfmadd231pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = c007_0 * log2e - 1
	vmovapd       64[r9], zmm1
    vmovapd     zmm1,  zmm30                    
    vfmadd231pd zmm1,  zmm21,  0[r15]{1to8}  
	vmovapd      128[r9], zmm2
    vmovapd     zmm2,  zmm30                    
    vfmadd231pd zmm2,  zmm22,  0[r15]{1to8}  
	vmovapd      192[r9], zmm3
    vmovapd     zmm3,  zmm30                    
    vfmadd231pd zmm3,  zmm23,  0[r15]{1to8}  
	vmovapd      256[r9], zmm4
    vmovapd     zmm4,  zmm30                    
    vfmadd231pd zmm4,  zmm24,  0[r15]{1to8}  
	vmovapd      320[r9], zmm5
    vmovapd     zmm5,  zmm30                    
    vfmadd231pd zmm5,  zmm25,  0[r15]{1to8}  
	vmovapd      384[r9], zmm6
    vmovapd     zmm6,  zmm30                    
    vfmadd231pd zmm6,  zmm26,  0[r15]{1to8}  
	vmovapd      448[r9], zmm7
    vmovapd     zmm7,  zmm30                    
    vfmadd231pd zmm7,  zmm27,  0[r15]{1to8}  
	vmovapd      512[r9], zmm8
    vmovapd     zmm8,  zmm30                    
    vfmadd231pd zmm8,  zmm28,  0[r15]{1to8}  
	vmovapd      576[r9], zmm9
    vmovapd     zmm9,  zmm30                    
    vfmadd231pd zmm9,  zmm29,  0[r15]{1to8}  
	vmovapd      640[r9], zmm10
	vcvtfxpntpd2dq     zmm10, zmm0, 0x2       // k0 = double2int( a0 )
	vmovapd      704[r9], zmm11
	vcvtfxpntpd2dq     zmm11, zmm1, 0x2       // k1 = double2int( a1 )
	vmovapd      768[r9], zmm12
	vcvtfxpntpd2dq     zmm12, zmm2, 0x2       // k2 = double2int( a2 )
	vmovapd      832[r9], zmm13
	vcvtfxpntpd2dq     zmm13, zmm3, 0x2       // k3 = double2int( a3 )
	vmovapd      896[r9], zmm14
	vcvtfxpntpd2dq     zmm14, zmm4, 0x2       // k4 = double2int( a4 )
	vmovapd      960[r9], zmm15
	vcvtfxpntpd2dq     zmm15, zmm5, 0x2       // k5 = double2int( a5 )
	vmovapd     1024[r9], zmm16
	vcvtfxpntpd2dq     zmm16, zmm6, 0x2       // k6 = double2int( a6 )
	vmovapd     1088[r9], zmm17
	vcvtfxpntpd2dq     zmm17, zmm7, 0x2       // k7 = double2int( a7 )
	vmovapd     1152[r9], zmm18
	vcvtfxpntpd2dq     zmm18, zmm8, 0x2       // k8 = double2int( a8 )
	vmovapd     1216[r9], zmm19
	vcvtfxpntpd2dq     zmm19, zmm9, 0x2       // k9 = double2int( a9 )

	vcvtdq2pd   zmm0,  zmm10                   // p0 = int2double( k0 ) 
	vcvtdq2pd   zmm1,  zmm11                   // p1 = int2double( k1 ) 
	vcvtdq2pd   zmm2,  zmm12                   // p2 = int2double( k2 ) 
	vcvtdq2pd   zmm3,  zmm13                   // p3 = int2double( k3 ) 
	vcvtdq2pd   zmm4,  zmm14                   // p4 = int2double( k4 ) 
	vcvtdq2pd   zmm5,  zmm15                   // p5 = int2double( k5 ) 
	vcvtdq2pd   zmm6,  zmm16                   // p6 = int2double( k6 ) 
	vcvtdq2pd   zmm7,  zmm17                   // p7 = int2double( k7 ) 
	vcvtdq2pd   zmm8,  zmm18                   // p8 = int2double( k8 ) 
	vcvtdq2pd   zmm9,  zmm19                   // p9 = int2double( k9 ) 

	mov         r15,   ptr_c1                  //load address of c1

    vfmadd231pd zmm20, zmm0,   0[r15]{1to8}    // c007_20 += p0 * c1 
    vfmadd231pd zmm21, zmm1,   0[r15]{1to8}    // c007_21 += p1 * c1 
    vfmadd231pd zmm22, zmm2,   0[r15]{1to8}    // c007_22 += p2 * c1 
    vfmadd231pd zmm23, zmm3,   0[r15]{1to8}    // c007_23 += p3 * c1 
    vfmadd231pd zmm24, zmm4,   0[r15]{1to8}    // c007_24 += p4 * c1 
    vfmadd231pd zmm25, zmm5,   0[r15]{1to8}    // c007_25 += p5 * c1 
    vfmadd231pd zmm26, zmm6,   0[r15]{1to8}    // c007_26 += p6 * c1 
    vfmadd231pd zmm27, zmm7,   0[r15]{1to8}    // c007_27 += p7 * c1 
    vfmadd231pd zmm28, zmm8,   0[r15]{1to8}    // c007_28 += p8 * c1 
    vfmadd231pd zmm29, zmm9,   0[r15]{1to8}    // c007_29 += p9 * c1 

	mov         r15,   ptr_c2                  //load address of c2

    vfmadd231pd zmm20, zmm0,   0[r15]{1to8}    // c007_20 += p0 * c2 
    vfmadd231pd zmm21, zmm1,   0[r15]{1to8}    // c007_21 += p1 * c2 
    vfmadd231pd zmm22, zmm2,   0[r15]{1to8}    // c007_22 += p2 * c2 
    vfmadd231pd zmm23, zmm3,   0[r15]{1to8}    // c007_23 += p3 * c2 
    vfmadd231pd zmm24, zmm4,   0[r15]{1to8}    // c007_24 += p4 * c2 
    vfmadd231pd zmm25, zmm5,   0[r15]{1to8}    // c007_25 += p5 * c2 
    vfmadd231pd zmm26, zmm6,   0[r15]{1to8}    // c007_26 += p6 * c2 
    vfmadd231pd zmm27, zmm7,   0[r15]{1to8}    // c007_27 += p7 * c2 
    vfmadd231pd zmm28, zmm8,   0[r15]{1to8}    // c007_28 += p8 * c2 
    vfmadd231pd zmm29, zmm9,   0[r15]{1to8}    // c007_29 += p9 * c2 

	mov         r15,   ptr_mask                //load address of mask
    vmovdqa32          zmm30,  0[r15]          // zmm30 = perm 

	vpaddd      zmm10, zmm10,  zmm30           // k0 += mask
	vpaddd      zmm11, zmm11,  zmm30           // k1 += mask
	vpaddd      zmm12, zmm12,  zmm30           // k2 += mask
	vpaddd      zmm13, zmm13,  zmm30           // k3 += mask
	vpaddd      zmm14, zmm14,  zmm30           // k4 += mask
	vpaddd      zmm15, zmm15,  zmm30           // k5 += mask
	vpaddd      zmm16, zmm16,  zmm30           // k6 += mask
	vpaddd      zmm17, zmm17,  zmm30           // k7 += mask
	vpaddd      zmm18, zmm18,  zmm30           // k8 += mask
	vpaddd      zmm19, zmm19,  zmm30           // k9 += mask

	mov         r15,   ptr_perm                //load address of perm
    vmovdqa32          zmm31,  0[r15]          // zmm31 = perm

	vpermd      zmm10, zmm31,  zmm10           // permute( k0 )
	vpermd      zmm11, zmm31,  zmm11           // permute( k1 )
	vpermd      zmm12, zmm31,  zmm12           // permute( k2 )
	vpermd      zmm13, zmm31,  zmm13           // permute( k3 )
	vpermd      zmm14, zmm31,  zmm14           // permute( k4 )
	vpermd      zmm15, zmm31,  zmm15           // permute( k5 )
	vpermd      zmm16, zmm31,  zmm16           // permute( k6 )
	vpermd      zmm17, zmm31,  zmm17           // permute( k7 )
	vpermd      zmm18, zmm31,  zmm18           // permute( k8 )
	vpermd      zmm19, zmm31,  zmm19           // permute( k9 )

	mov         r12,   ptr_w10                 //load address of perm

    vbroadcastsd       zmm0,   0[r12]          // zmm0 = w10
	vpslld      zmm10, zmm10,  20              // shift k0<<20
    vmovapd            zmm1,   zmm0           
	vpslld      zmm11, zmm11,  20              // shift k1<<20
    vmovapd            zmm2,   zmm0           
	vpslld      zmm12, zmm12,  20              // shift k2<<20
    vmovapd            zmm3,   zmm0           
	vpslld      zmm13, zmm13,  20              // shift k3<<20
    vmovapd            zmm4,   zmm0           
	vpslld      zmm14, zmm14,  20              // shift k4<<20
    vmovapd            zmm5,   zmm0           
	vpslld      zmm15, zmm15,  20              // shift k5<<20
    vmovapd            zmm6,   zmm0           
	vpslld      zmm16, zmm16,  20              // shift k6<<20
    vmovapd            zmm7,   zmm0           
	vpslld      zmm17, zmm17,  20              // shift k7<<20
    vmovapd            zmm8,   zmm0           
	vpslld      zmm18, zmm18,  20              // shift k8<<20
    vmovapd            zmm9,   zmm0           
	vpslld      zmm19, zmm19,  20              // shift k9<<20

	mov         r15,   ptr_w11                 //load address of w11

    vfmadd231pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w10 + c007_20 * w11 
    vfmadd231pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w10 + c007_21 * w11 
    vfmadd231pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w10 + c007_22 * w11 
    vfmadd231pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w10 + c007_23 * w11 
    vfmadd231pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w10 + c007_24 * w11 
    vfmadd231pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w10 + c007_25 * w11 
    vfmadd231pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w10 + c007_26 * w11 
    vfmadd231pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w10 + c007_27 * w11 
    vfmadd231pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w10 + c007_28 * w11 
    vfmadd231pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w10 + c007_29 * w11 

	mov         r15,   ptr_w9                  //load address of w9

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w9 + ( c007_20 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w9 + ( c007_21 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w9 + ( c007_22 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w9 + ( c007_23 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w9 + ( c007_24 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w9 + ( c007_25 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w9 + ( c007_26 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w9 + ( c007_27 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w9 + ( c007_28 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w9 + ( c007_29 * a9 )

	mov         r15,   ptr_w8                  //load address of w8

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w8 + ( c007_20 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w8 + ( c007_21 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w8 + ( c007_22 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w8 + ( c007_23 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w8 + ( c007_24 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w8 + ( c007_25 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w8 + ( c007_26 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w8 + ( c007_27 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w8 + ( c007_28 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w8 + ( c007_29 * a9 )

	mov         r15,   ptr_w7                  //load address of w7

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w7 + ( c007_20 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w7 + ( c007_21 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w7 + ( c007_22 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w7 + ( c007_23 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w7 + ( c007_24 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w7 + ( c007_25 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w7 + ( c007_26 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w7 + ( c007_27 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w7 + ( c007_28 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w7 + ( c007_29 * a9 )

	mov         r15,   ptr_w6                  //load address of w6

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w6 + ( c007_20 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w6 + ( c007_21 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w6 + ( c007_22 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w6 + ( c007_23 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w6 + ( c007_24 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w6 + ( c007_25 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w6 + ( c007_26 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w6 + ( c007_27 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w6 + ( c007_28 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w6 + ( c007_29 * a9 )

	mov         r15,   ptr_w5                  //load address of w5

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w5 + ( c007_20 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w5 + ( c007_21 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w5 + ( c007_22 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w5 + ( c007_23 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w5 + ( c007_24 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w5 + ( c007_25 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w5 + ( c007_26 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w5 + ( c007_27 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w5 + ( c007_28 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w5 + ( c007_29 * a9 )

	mov         r15,   ptr_w4                  //load address of w4

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w4 + ( c007_20 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w4 + ( c007_21 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w4 + ( c007_22 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w4 + ( c007_23 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w4 + ( c007_24 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w4 + ( c007_25 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w4 + ( c007_26 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w4 + ( c007_27 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w4 + ( c007_28 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w4 + ( c007_29 * a9 )

	mov         r15,   ptr_w3                  //load address of w3

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w3 + ( c007_20 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w3 + ( c007_21 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w3 + ( c007_22 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w3 + ( c007_23 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w3 + ( c007_24 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w3 + ( c007_25 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w3 + ( c007_26 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w3 + ( c007_27 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w3 + ( c007_28 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w3 + ( c007_29 * a9 )

	mov         r15,   ptr_w2                  //load address of w2

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w2 + ( c007_20 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w2 + ( c007_21 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w2 + ( c007_22 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w2 + ( c007_23 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w2 + ( c007_24 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w2 + ( c007_25 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w2 + ( c007_26 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w2 + ( c007_27 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w2 + ( c007_28 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w2 + ( c007_29 * a9 )

	mov         r15,   ptr_w1                  //load address of w1

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w1 + ( c007_20 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w1 + ( c007_21 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w1 + ( c007_22 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w1 + ( c007_23 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w1 + ( c007_24 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w1 + ( c007_25 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w1 + ( c007_26 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w1 + ( c007_27 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w1 + ( c007_28 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w1 + ( c007_29 * a9 )

	mov         r15,   ptr_w0                  //load address of w0

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w0 + ( c007_20 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w0 + ( c007_21 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w0 + ( c007_22 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w0 + ( c007_23 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w0 + ( c007_24 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w0 + ( c007_25 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w0 + ( c007_26 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w0 + ( c007_27 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w0 + ( c007_28 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w0 + ( c007_29 * a9 )

	mov         r15,   w                       // load address of w
	mov         rax,   u                       // load address of u
    vmovapd            zmm30,  0[rax]          // u007

	vmulpd      zmm20, zmm0,  zmm10            // c007_20 = a0 * p0 
	vmulpd      zmm21, zmm1,  zmm11            // c007_21 = a1 * p1
	vmulpd      zmm22, zmm2,  zmm12            // c007_22 = a2 * p2 
	vmulpd      zmm23, zmm3,  zmm13            // c007_23 = a3 * p3 
	vmulpd      zmm24, zmm4,  zmm14            // c007_24 = a4 * p4 
	vmulpd      zmm25, zmm5,  zmm15            // c007_25 = a5 * p5 
	vmulpd      zmm26, zmm6,  zmm16            // c007_26 = a6 * p6 
	vmulpd      zmm27, zmm7,  zmm17            // c007_27 = a7 * p7 
	vmulpd      zmm28, zmm8,  zmm18            // c007_28 = a8 * p8 
	vmulpd      zmm29, zmm9,  zmm19            // c007_29 = a9 * p9 


	// u007 = c007_20:29 * w20:29
    vfmadd231pd zmm30, zmm20, 160[r15]{1to8}   // u007 += c007_20 * w20
    vfmadd231pd zmm30, zmm21, 168[r15]{1to8}   // u007 += c007_21 * w21
    vfmadd231pd zmm30, zmm22, 176[r15]{1to8}   // u007 += c007_22 * w22
    vfmadd231pd zmm30, zmm23, 184[r15]{1to8}   // u007 += c007_23 * w23
    vfmadd231pd zmm30, zmm24, 192[r15]{1to8}   // u007 += c007_24 * w24
    vfmadd231pd zmm30, zmm25, 200[r15]{1to8}   // u007 += c007_25 * w25
    vfmadd231pd zmm30, zmm26, 208[r15]{1to8}   // u007 += c007_26 * w26
    vfmadd231pd zmm30, zmm27, 216[r15]{1to8}   // u007 += c007_27 * w27
    vfmadd231pd zmm30, zmm28, 224[r15]{1to8}   // u007 += c007_28 * w28
    vfmadd231pd zmm30, zmm29, 232[r15]{1to8}   // u007 += c007_29 * w29


	// At this moment c007_20 ~ c007_29 is free to be reused for other purpose.
	//
	// zmm30 is now carrying u007; thus, we can't reuse zmm30.
	//

	mov         r12,   ptr_dmone               //load address of dmone
	mov         r15,   ptr_log2e               //load address of log2e
    vbroadcastsd       zmm31, 0[r12]           // -1.0
	
	vmovapd            zmm20,  640[r9]         // zmm20 = c007_10
    vmovapd     zmm0,  zmm31                   // zmm0  = -1.0
    vfmadd231pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = c007_10 * log2e - 1
	vmovapd            zmm21,  704[r9]
    vmovapd     zmm1,  zmm31                    
    vfmadd231pd zmm1,  zmm21,  0[r15]{1to8}  
	vmovapd            zmm22,  768[r9]
    vmovapd     zmm2,  zmm31                    
    vfmadd231pd zmm2,  zmm22,  0[r15]{1to8}  
	vmovapd            zmm23,  832[r9]
    vmovapd     zmm3,  zmm31                    
    vfmadd231pd zmm3,  zmm23,  0[r15]{1to8}  
	vmovapd            zmm24,  896[r9]
    vmovapd     zmm4,  zmm31                    
    vfmadd231pd zmm4,  zmm24,  0[r15]{1to8}  
	vmovapd            zmm25,  960[r9]
    vmovapd     zmm5,  zmm31                    
    vfmadd231pd zmm5,  zmm25,  0[r15]{1to8}  
	vmovapd            zmm26, 1024[r9]
    vmovapd     zmm6,  zmm31                    
    vfmadd231pd zmm6,  zmm26,  0[r15]{1to8}  
	vmovapd            zmm27, 1088[r9]
    vmovapd     zmm7,  zmm31                    
    vfmadd231pd zmm7,  zmm27,  0[r15]{1to8}  
	vmovapd            zmm28, 1152[r9]
    vmovapd     zmm8,  zmm31                    
    vfmadd231pd zmm8,  zmm28,  0[r15]{1to8}  
	vmovapd            zmm29, 1216[r9]         // zmm29 = c007_19
    vmovapd     zmm9,  zmm31                    
    vfmadd231pd zmm9,  zmm29,  0[r15]{1to8}  


	// At this moment,
	//
	// zmm0:9   = a0:a9
	// zmm10:19 = nothing
	// zmm20:29 = c007_10:19
	// zmm30    = u007
	// zmm31    = -1

	vcvtfxpntpd2dq     zmm10, zmm0, 0x2       // k0 = double2int( a0 )
	vcvtfxpntpd2dq     zmm11, zmm1, 0x2       // k1 = double2int( a1 )
	vcvtfxpntpd2dq     zmm12, zmm2, 0x2       // k2 = double2int( a2 )
	vcvtfxpntpd2dq     zmm13, zmm3, 0x2       // k3 = double2int( a3 )
	vcvtfxpntpd2dq     zmm14, zmm4, 0x2       // k4 = double2int( a4 )
	vcvtfxpntpd2dq     zmm15, zmm5, 0x2       // k5 = double2int( a5 )
	vcvtfxpntpd2dq     zmm16, zmm6, 0x2       // k6 = double2int( a6 )
	vcvtfxpntpd2dq     zmm17, zmm7, 0x2       // k7 = double2int( a7 )
	vcvtfxpntpd2dq     zmm18, zmm8, 0x2       // k8 = double2int( a8 )
	vcvtfxpntpd2dq     zmm19, zmm9, 0x2       // k9 = double2int( a9 )

	vcvtdq2pd   zmm0,  zmm10                   // p0 = int2double( k0 ) 
	vcvtdq2pd   zmm1,  zmm11                   // p1 = int2double( k1 ) 
	vcvtdq2pd   zmm2,  zmm12                   // p2 = int2double( k2 ) 
	vcvtdq2pd   zmm3,  zmm13                   // p3 = int2double( k3 ) 
	vcvtdq2pd   zmm4,  zmm14                   // p4 = int2double( k4 ) 
	vcvtdq2pd   zmm5,  zmm15                   // p5 = int2double( k5 ) 
	vcvtdq2pd   zmm6,  zmm16                   // p6 = int2double( k6 ) 
	vcvtdq2pd   zmm7,  zmm17                   // p7 = int2double( k7 ) 
	vcvtdq2pd   zmm8,  zmm18                   // p8 = int2double( k8 ) 
	vcvtdq2pd   zmm9,  zmm19                   // p9 = int2double( k9 ) 

	mov         r15,   ptr_c1                  //load address of c1

    vfmadd231pd zmm20, zmm0,   0[r15]{1to8}    // c007_20 += p0 * c1 
    vfmadd231pd zmm21, zmm1,   0[r15]{1to8}    // c007_21 += p1 * c1 
    vfmadd231pd zmm22, zmm2,   0[r15]{1to8}    // c007_22 += p2 * c1 
    vfmadd231pd zmm23, zmm3,   0[r15]{1to8}    // c007_23 += p3 * c1 
    vfmadd231pd zmm24, zmm4,   0[r15]{1to8}    // c007_24 += p4 * c1 
    vfmadd231pd zmm25, zmm5,   0[r15]{1to8}    // c007_25 += p5 * c1 
    vfmadd231pd zmm26, zmm6,   0[r15]{1to8}    // c007_26 += p6 * c1 
    vfmadd231pd zmm27, zmm7,   0[r15]{1to8}    // c007_27 += p7 * c1 
    vfmadd231pd zmm28, zmm8,   0[r15]{1to8}    // c007_28 += p8 * c1 
    vfmadd231pd zmm29, zmm9,   0[r15]{1to8}    // c007_29 += p9 * c1 

	mov         r15,   ptr_c2                  //load address of c2

    vfmadd231pd zmm20, zmm0,   0[r15]{1to8}    // c007_20 += p0 * c2 
    vfmadd231pd zmm21, zmm1,   0[r15]{1to8}    // c007_21 += p1 * c2 
    vfmadd231pd zmm22, zmm2,   0[r15]{1to8}    // c007_22 += p2 * c2 
    vfmadd231pd zmm23, zmm3,   0[r15]{1to8}    // c007_23 += p3 * c2 
    vfmadd231pd zmm24, zmm4,   0[r15]{1to8}    // c007_24 += p4 * c2 
    vfmadd231pd zmm25, zmm5,   0[r15]{1to8}    // c007_25 += p5 * c2 
    vfmadd231pd zmm26, zmm6,   0[r15]{1to8}    // c007_26 += p6 * c2 
    vfmadd231pd zmm27, zmm7,   0[r15]{1to8}    // c007_27 += p7 * c2 
    vfmadd231pd zmm28, zmm8,   0[r15]{1to8}    // c007_28 += p8 * c2 
    vfmadd231pd zmm29, zmm9,   0[r15]{1to8}    // c007_29 += p9 * c2 

	mov         r15,   ptr_mask                // load address of mask
    vmovdqa32          zmm31,  0[r15]          // zmm31 = perm 

	vpaddd      zmm10, zmm10,  zmm31           // k0 += mask
	vpaddd      zmm11, zmm11,  zmm31           // k1 += mask
	vpaddd      zmm12, zmm12,  zmm31           // k2 += mask
	vpaddd      zmm13, zmm13,  zmm31           // k3 += mask
	vpaddd      zmm14, zmm14,  zmm31           // k4 += mask
	vpaddd      zmm15, zmm15,  zmm31           // k5 += mask
	vpaddd      zmm16, zmm16,  zmm31           // k6 += mask
	vpaddd      zmm17, zmm17,  zmm31           // k7 += mask
	vpaddd      zmm18, zmm18,  zmm31           // k8 += mask
	vpaddd      zmm19, zmm19,  zmm31           // k9 += mask

	mov         r15,   ptr_perm                // load address of perm
    vmovdqa32          zmm31,  0[r15]          // zmm31 = perm

	vpermd      zmm10, zmm31,  zmm10           // permute( k0 )
	vpermd      zmm11, zmm31,  zmm11           // permute( k1 )
	vpermd      zmm12, zmm31,  zmm12           // permute( k2 )
	vpermd      zmm13, zmm31,  zmm13           // permute( k3 )
	vpermd      zmm14, zmm31,  zmm14           // permute( k4 )
	vpermd      zmm15, zmm31,  zmm15           // permute( k5 )
	vpermd      zmm16, zmm31,  zmm16           // permute( k6 )
	vpermd      zmm17, zmm31,  zmm17           // permute( k7 )
	vpermd      zmm18, zmm31,  zmm18           // permute( k8 )
	vpermd      zmm19, zmm31,  zmm19           // permute( k9 )

	mov         r12,   ptr_w10                 // load address of w10

    vbroadcastsd       zmm0,   0[r12]          // zmm0 = w10
	vpslld      zmm10, zmm10,  20              // shift k0<<20
    vmovapd            zmm1,   zmm0           
	vpslld      zmm11, zmm11,  20              // shift k1<<20
    vmovapd            zmm2,   zmm0           
	vpslld      zmm12, zmm12,  20              // shift k2<<20
    vmovapd            zmm3,   zmm0           
	vpslld      zmm13, zmm13,  20              // shift k3<<20
    vmovapd            zmm4,   zmm0           
	vpslld      zmm14, zmm14,  20              // shift k4<<20
    vmovapd            zmm5,   zmm0           
	vpslld      zmm15, zmm15,  20              // shift k5<<20
    vmovapd            zmm6,   zmm0           
	vpslld      zmm16, zmm16,  20              // shift k6<<20
    vmovapd            zmm7,   zmm0           
	vpslld      zmm17, zmm17,  20              // shift k7<<20
    vmovapd            zmm8,   zmm0           
	vpslld      zmm18, zmm18,  20              // shift k8<<20
    vmovapd            zmm9,   zmm0           
	vpslld      zmm19, zmm19,  20              // shift k9<<20

	mov         r15,   ptr_w11                 //load address of w11

    vfmadd231pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w10 + c007_10 * w11 
    vfmadd231pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w10 + c007_11 * w11 
    vfmadd231pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w10 + c007_12 * w11 
    vfmadd231pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w10 + c007_13 * w11 
    vfmadd231pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w10 + c007_14 * w11 
    vfmadd231pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w10 + c007_15 * w11 
    vfmadd231pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w10 + c007_16 * w11 
    vfmadd231pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w10 + c007_17 * w11 
    vfmadd231pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w10 + c007_18 * w11 
    vfmadd231pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w10 + c007_19 * w11 

	mov         r15,   ptr_w9                  //load address of w9

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w9 + ( c007_10 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w9 + ( c007_11 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w9 + ( c007_12 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w9 + ( c007_13 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w9 + ( c007_14 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w9 + ( c007_15 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w9 + ( c007_16 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w9 + ( c007_17 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w9 + ( c007_18 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w9 + ( c007_19 * a9 )

	mov         r15,   ptr_w8                  //load address of w8

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w8 + ( c007_10 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w8 + ( c007_11 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w8 + ( c007_12 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w8 + ( c007_13 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w8 + ( c007_14 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w8 + ( c007_15 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w8 + ( c007_16 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w8 + ( c007_17 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w8 + ( c007_18 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w8 + ( c007_19 * a9 )

	mov         r15,   ptr_w7                  //load address of w7

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w7 + ( c007_10 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w7 + ( c007_11 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w7 + ( c007_12 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w7 + ( c007_13 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w7 + ( c007_14 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w7 + ( c007_15 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w7 + ( c007_16 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w7 + ( c007_17 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w7 + ( c007_18 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w7 + ( c007_19 * a9 )

	mov         r15,   ptr_w6                  //load address of w6

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w6 + ( c007_10 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w6 + ( c007_11 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w6 + ( c007_12 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w6 + ( c007_13 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w6 + ( c007_14 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w6 + ( c007_15 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w6 + ( c007_16 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w6 + ( c007_17 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w6 + ( c007_18 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w6 + ( c007_19 * a9 )

	mov         r15,   ptr_w5                  //load address of w5

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w5 + ( c007_10 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w5 + ( c007_11 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w5 + ( c007_12 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w5 + ( c007_13 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w5 + ( c007_14 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w5 + ( c007_15 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w5 + ( c007_16 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w5 + ( c007_17 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w5 + ( c007_18 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w5 + ( c007_19 * a9 )

	mov         r15,   ptr_w4                  //load address of w4

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w4 + ( c007_10 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w4 + ( c007_11 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w4 + ( c007_12 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w4 + ( c007_13 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w4 + ( c007_14 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w4 + ( c007_15 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w4 + ( c007_16 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w4 + ( c007_17 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w4 + ( c007_18 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w4 + ( c007_19 * a9 )

	mov         r15,   ptr_w3                  //load address of w3

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w3 + ( c007_10 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w3 + ( c007_11 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w3 + ( c007_12 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w3 + ( c007_13 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w3 + ( c007_14 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w3 + ( c007_15 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w3 + ( c007_16 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w3 + ( c007_17 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w3 + ( c007_18 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w3 + ( c007_19 * a9 )

	mov         r15,   ptr_w2                  //load address of w2

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w2 + ( c007_10 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w2 + ( c007_11 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w2 + ( c007_12 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w2 + ( c007_13 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w2 + ( c007_14 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w2 + ( c007_15 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w2 + ( c007_16 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w2 + ( c007_17 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w2 + ( c007_18 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w2 + ( c007_19 * a9 )

	mov         r15,   ptr_w1                  //load address of w1

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w1 + ( c007_10 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w1 + ( c007_11 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w1 + ( c007_12 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w1 + ( c007_13 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w1 + ( c007_14 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w1 + ( c007_15 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w1 + ( c007_16 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w1 + ( c007_17 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w1 + ( c007_18 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w1 + ( c007_19 * a9 )

	mov         r15,   ptr_w0                  //load address of w0

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w0 + ( c007_10 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w0 + ( c007_11 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w0 + ( c007_12 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w0 + ( c007_13 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w0 + ( c007_14 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w0 + ( c007_15 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w0 + ( c007_16 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w0 + ( c007_17 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w0 + ( c007_18 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w0 + ( c007_19 * a9 )

	mov         r15,   w                       // load address of w

	vmulpd      zmm20, zmm0,  zmm10            // c007_10 = a0 * p0 
	vmulpd      zmm21, zmm1,  zmm11            // c007_11 = a1 * p1
	vmulpd      zmm22, zmm2,  zmm12            // c007_12 = a2 * p2 
	vmulpd      zmm23, zmm3,  zmm13            // c007_13 = a3 * p3 
	vmulpd      zmm24, zmm4,  zmm14            // c007_14 = a4 * p4 
	vmulpd      zmm25, zmm5,  zmm15            // c007_15 = a5 * p5 
	vmulpd      zmm26, zmm6,  zmm16            // c007_16 = a6 * p6 
	vmulpd      zmm27, zmm7,  zmm17            // c007_17 = a7 * p7 
	vmulpd      zmm28, zmm8,  zmm18            // c007_18 = a8 * p8 
	vmulpd      zmm29, zmm9,  zmm19            // c007_19 = a9 * p9 


	// u007 = c007_20:29 * w20:29
    vfmadd231pd zmm30, zmm20,  80[r15]{1to8}   // u007 += c007_10 * w10
    vfmadd231pd zmm30, zmm21,  88[r15]{1to8}   // u007 += c007_11 * w11
    vfmadd231pd zmm30, zmm22,  96[r15]{1to8}   // u007 += c007_12 * w12
    vfmadd231pd zmm30, zmm23, 104[r15]{1to8}   // u007 += c007_13 * w13
    vfmadd231pd zmm30, zmm24, 112[r15]{1to8}   // u007 += c007_14 * w14
    vfmadd231pd zmm30, zmm25, 120[r15]{1to8}   // u007 += c007_15 * w15
    vfmadd231pd zmm30, zmm26, 128[r15]{1to8}   // u007 += c007_16 * w16
    vfmadd231pd zmm30, zmm27, 136[r15]{1to8}   // u007 += c007_17 * w17
    vfmadd231pd zmm30, zmm28, 144[r15]{1to8}   // u007 += c007_18 * w18
    vfmadd231pd zmm30, zmm29, 152[r15]{1to8}   // u007 += c007_19 * w19


	// At this moment c007_10 ~ c007_19 is free to be reused for other purpose.
	//
	// zmm30 is now carrying u007; thus, we can't reuse zmm30.
	//

	mov         r12,   ptr_dmone               //load address of dmone
	mov         r15,   ptr_log2e               //load address of log2e
    vbroadcastsd       zmm31, 0[r12]           // -1.0
	
	vmovapd            zmm20,    0[r9]         // zmm20 = c007_0
    vmovapd     zmm0,  zmm31                   // zmm0  = -1.0
    vfmadd231pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = c007_0 * log2e - 1
	vmovapd            zmm21,   64[r9]
    vmovapd     zmm1,  zmm31                    
    vfmadd231pd zmm1,  zmm21,  0[r15]{1to8}  
	vmovapd            zmm22,  128[r9]
    vmovapd     zmm2,  zmm31                    
    vfmadd231pd zmm2,  zmm22,  0[r15]{1to8}  
	vmovapd            zmm23,  192[r9]
    vmovapd     zmm3,  zmm31                    
    vfmadd231pd zmm3,  zmm23,  0[r15]{1to8}  
	vmovapd            zmm24,  256[r9]
    vmovapd     zmm4,  zmm31                    
    vfmadd231pd zmm4,  zmm24,  0[r15]{1to8}  
	vmovapd            zmm25,  320[r9]
    vmovapd     zmm5,  zmm31                    
    vfmadd231pd zmm5,  zmm25,  0[r15]{1to8}  
	vmovapd            zmm26,  384[r9]
    vmovapd     zmm6,  zmm31                    
    vfmadd231pd zmm6,  zmm26,  0[r15]{1to8}  
	vmovapd            zmm27,  448[r9]
    vmovapd     zmm7,  zmm31                    
    vfmadd231pd zmm7,  zmm27,  0[r15]{1to8}  
	vmovapd            zmm28,  512[r9]
    vmovapd     zmm8,  zmm31                    
    vfmadd231pd zmm8,  zmm28,  0[r15]{1to8}  
	vmovapd            zmm29,  576[r9]         // zmm29 = c007_9
    vmovapd     zmm9,  zmm31                    
    vfmadd231pd zmm9,  zmm29,  0[r15]{1to8}  

	// At this moment,
	//
	// zmm0:9   = a0:a9
	// zmm10:19 = nothing
	// zmm20:29 = c007_0:9
	// zmm30    = u007
	// zmm31    = -1

	vcvtfxpntpd2dq     zmm10, zmm0, 0x2       // k0 = double2int( a0 )
	vcvtfxpntpd2dq     zmm11, zmm1, 0x2       // k1 = double2int( a1 )
	vcvtfxpntpd2dq     zmm12, zmm2, 0x2       // k2 = double2int( a2 )
	vcvtfxpntpd2dq     zmm13, zmm3, 0x2       // k3 = double2int( a3 )
	vcvtfxpntpd2dq     zmm14, zmm4, 0x2       // k4 = double2int( a4 )
	vcvtfxpntpd2dq     zmm15, zmm5, 0x2       // k5 = double2int( a5 )
	vcvtfxpntpd2dq     zmm16, zmm6, 0x2       // k6 = double2int( a6 )
	vcvtfxpntpd2dq     zmm17, zmm7, 0x2       // k7 = double2int( a7 )
	vcvtfxpntpd2dq     zmm18, zmm8, 0x2       // k8 = double2int( a8 )
	vcvtfxpntpd2dq     zmm19, zmm9, 0x2       // k9 = double2int( a9 )

	vcvtdq2pd   zmm0,  zmm10                   // p0 = int2double( k0 ) 
	vcvtdq2pd   zmm1,  zmm11                   // p1 = int2double( k1 ) 
	vcvtdq2pd   zmm2,  zmm12                   // p2 = int2double( k2 ) 
	vcvtdq2pd   zmm3,  zmm13                   // p3 = int2double( k3 ) 
	vcvtdq2pd   zmm4,  zmm14                   // p4 = int2double( k4 ) 
	vcvtdq2pd   zmm5,  zmm15                   // p5 = int2double( k5 ) 
	vcvtdq2pd   zmm6,  zmm16                   // p6 = int2double( k6 ) 
	vcvtdq2pd   zmm7,  zmm17                   // p7 = int2double( k7 ) 
	vcvtdq2pd   zmm8,  zmm18                   // p8 = int2double( k8 ) 
	vcvtdq2pd   zmm9,  zmm19                   // p9 = int2double( k9 ) 

	mov         r15,   ptr_c1                  //load address of c1

    vfmadd231pd zmm20, zmm0,   0[r15]{1to8}    // c007_0 += p0 * c1 
    vfmadd231pd zmm21, zmm1,   0[r15]{1to8}    // c007_1 += p1 * c1 
    vfmadd231pd zmm22, zmm2,   0[r15]{1to8}    // c007_2 += p2 * c1 
    vfmadd231pd zmm23, zmm3,   0[r15]{1to8}    // c007_3 += p3 * c1 
    vfmadd231pd zmm24, zmm4,   0[r15]{1to8}    // c007_4 += p4 * c1 
    vfmadd231pd zmm25, zmm5,   0[r15]{1to8}    // c007_5 += p5 * c1 
    vfmadd231pd zmm26, zmm6,   0[r15]{1to8}    // c007_6 += p6 * c1 
    vfmadd231pd zmm27, zmm7,   0[r15]{1to8}    // c007_7 += p7 * c1 
    vfmadd231pd zmm28, zmm8,   0[r15]{1to8}    // c007_8 += p8 * c1 
    vfmadd231pd zmm29, zmm9,   0[r15]{1to8}    // c007_9 += p9 * c1 

	mov         r15,   ptr_c2                  //load address of c2

    vfmadd231pd zmm20, zmm0,   0[r15]{1to8}    // c007_0 += p0 * c2 
    vfmadd231pd zmm21, zmm1,   0[r15]{1to8}    // c007_1 += p1 * c2 
    vfmadd231pd zmm22, zmm2,   0[r15]{1to8}    // c007_2 += p2 * c2 
    vfmadd231pd zmm23, zmm3,   0[r15]{1to8}    // c007_3 += p3 * c2 
    vfmadd231pd zmm24, zmm4,   0[r15]{1to8}    // c007_4 += p4 * c2 
    vfmadd231pd zmm25, zmm5,   0[r15]{1to8}    // c007_5 += p5 * c2 
    vfmadd231pd zmm26, zmm6,   0[r15]{1to8}    // c007_6 += p6 * c2 
    vfmadd231pd zmm27, zmm7,   0[r15]{1to8}    // c007_7 += p7 * c2 
    vfmadd231pd zmm28, zmm8,   0[r15]{1to8}    // c007_8 += p8 * c2 
    vfmadd231pd zmm29, zmm9,   0[r15]{1to8}    // c007_9 += p9 * c2 

	mov         r15,   ptr_mask                // load address of mask
    vmovdqa32          zmm31,  0[r15]          // zmm31 = perm 

	vpaddd      zmm10, zmm10,  zmm31           // k0 += mask
	vpaddd      zmm11, zmm11,  zmm31           // k1 += mask
	vpaddd      zmm12, zmm12,  zmm31           // k2 += mask
	vpaddd      zmm13, zmm13,  zmm31           // k3 += mask
	vpaddd      zmm14, zmm14,  zmm31           // k4 += mask
	vpaddd      zmm15, zmm15,  zmm31           // k5 += mask
	vpaddd      zmm16, zmm16,  zmm31           // k6 += mask
	vpaddd      zmm17, zmm17,  zmm31           // k7 += mask
	vpaddd      zmm18, zmm18,  zmm31           // k8 += mask
	vpaddd      zmm19, zmm19,  zmm31           // k9 += mask

	mov         r15,   ptr_perm                // load address of perm
    vmovdqa32          zmm31,  0[r15]          // zmm31 = perm

	vpermd      zmm10, zmm31,  zmm10           // permute( k0 )
	vpermd      zmm11, zmm31,  zmm11           // permute( k1 )
	vpermd      zmm12, zmm31,  zmm12           // permute( k2 )
	vpermd      zmm13, zmm31,  zmm13           // permute( k3 )
	vpermd      zmm14, zmm31,  zmm14           // permute( k4 )
	vpermd      zmm15, zmm31,  zmm15           // permute( k5 )
	vpermd      zmm16, zmm31,  zmm16           // permute( k6 )
	vpermd      zmm17, zmm31,  zmm17           // permute( k7 )
	vpermd      zmm18, zmm31,  zmm18           // permute( k8 )
	vpermd      zmm19, zmm31,  zmm19           // permute( k9 )

	mov         r12,   ptr_w10                 // load address of w10

    vbroadcastsd       zmm0,   0[r12]          // zmm0 = w10
	vpslld      zmm10, zmm10,  20              // shift k0<<20
    vmovapd            zmm1,   zmm0           
	vpslld      zmm11, zmm11,  20              // shift k1<<20
    vmovapd            zmm2,   zmm0           
	vpslld      zmm12, zmm12,  20              // shift k2<<20
    vmovapd            zmm3,   zmm0           
	vpslld      zmm13, zmm13,  20              // shift k3<<20
    vmovapd            zmm4,   zmm0           
	vpslld      zmm14, zmm14,  20              // shift k4<<20
    vmovapd            zmm5,   zmm0           
	vpslld      zmm15, zmm15,  20              // shift k5<<20
    vmovapd            zmm6,   zmm0           
	vpslld      zmm16, zmm16,  20              // shift k6<<20
    vmovapd            zmm7,   zmm0           
	vpslld      zmm17, zmm17,  20              // shift k7<<20
    vmovapd            zmm8,   zmm0           
	vpslld      zmm18, zmm18,  20              // shift k8<<20
    vmovapd            zmm9,   zmm0           
	vpslld      zmm19, zmm19,  20              // shift k9<<20

	mov         r15,   ptr_w11                 //load address of w11

    vfmadd231pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w10 + c007_0 * w11 
    vfmadd231pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w10 + c007_1 * w11 
    vfmadd231pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w10 + c007_2 * w11 
    vfmadd231pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w10 + c007_3 * w11 
    vfmadd231pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w10 + c007_4 * w11 
    vfmadd231pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w10 + c007_5 * w11 
    vfmadd231pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w10 + c007_6 * w11 
    vfmadd231pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w10 + c007_7 * w11 
    vfmadd231pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w10 + c007_8 * w11 
    vfmadd231pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w10 + c007_9 * w11 

	mov         r15,   ptr_w9                  //load address of w9

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w9 + ( c007_0 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w9 + ( c007_1 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w9 + ( c007_2 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w9 + ( c007_3 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w9 + ( c007_4 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w9 + ( c007_5 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w9 + ( c007_6 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w9 + ( c007_7 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w9 + ( c007_8 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w9 + ( c007_9 * a9 )

	mov         r15,   ptr_w8                  //load address of w8

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w8 + ( c007_0 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w8 + ( c007_1 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w8 + ( c007_2 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w8 + ( c007_3 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w8 + ( c007_4 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w8 + ( c007_5 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w8 + ( c007_6 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w8 + ( c007_7 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w8 + ( c007_8 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w8 + ( c007_9 * a9 )

	mov         r15,   ptr_w7                  //load address of w7

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w7 + ( c007_0 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w7 + ( c007_1 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w7 + ( c007_2 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w7 + ( c007_3 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w7 + ( c007_4 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w7 + ( c007_5 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w7 + ( c007_6 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w7 + ( c007_7 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w7 + ( c007_8 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w7 + ( c007_9 * a9 )

	mov         r15,   ptr_w6                  //load address of w6

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w6 + ( c007_0 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w6 + ( c007_1 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w6 + ( c007_2 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w6 + ( c007_3 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w6 + ( c007_4 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w6 + ( c007_5 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w6 + ( c007_6 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w6 + ( c007_7 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w6 + ( c007_8 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w6 + ( c007_9 * a9 )

	mov         r15,   ptr_w5                  //load address of w5

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w5 + ( c007_0 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w5 + ( c007_1 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w5 + ( c007_2 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w5 + ( c007_3 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w5 + ( c007_4 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w5 + ( c007_5 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w5 + ( c007_6 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w5 + ( c007_7 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w5 + ( c007_8 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w5 + ( c007_9 * a9 )

	mov         r15,   ptr_w4                  //load address of w4

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w4 + ( c007_0 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w4 + ( c007_1 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w4 + ( c007_2 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w4 + ( c007_3 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w4 + ( c007_4 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w4 + ( c007_5 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w4 + ( c007_6 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w4 + ( c007_7 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w4 + ( c007_8 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w4 + ( c007_9 * a9 )

	mov         r15,   ptr_w3                  //load address of w3

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w3 + ( c007_0 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w3 + ( c007_1 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w3 + ( c007_2 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w3 + ( c007_3 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w3 + ( c007_4 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w3 + ( c007_5 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w3 + ( c007_6 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w3 + ( c007_7 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w3 + ( c007_8 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w3 + ( c007_9 * a9 )

	mov         r15,   ptr_w2                  //load address of w2

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w2 + ( c007_0 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w2 + ( c007_1 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w2 + ( c007_2 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w2 + ( c007_3 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w2 + ( c007_4 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w2 + ( c007_5 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w2 + ( c007_6 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w2 + ( c007_7 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w2 + ( c007_8 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w2 + ( c007_9 * a9 )

	mov         r15,   ptr_w1                  //load address of w1

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w1 + ( c007_0 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w1 + ( c007_1 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w1 + ( c007_2 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w1 + ( c007_3 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w1 + ( c007_4 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w1 + ( c007_5 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w1 + ( c007_6 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w1 + ( c007_7 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w1 + ( c007_8 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w1 + ( c007_9 * a9 )

	mov         r15,   ptr_w0                  //load address of w0

    vfmadd213pd zmm0,  zmm20,  0[r15]{1to8}    // a0 = w0 + ( c007_0 * a0 )
    vfmadd213pd zmm1,  zmm21,  0[r15]{1to8}    // a1 = w0 + ( c007_1 * a1 )
    vfmadd213pd zmm2,  zmm22,  0[r15]{1to8}    // a2 = w0 + ( c007_2 * a2 )
    vfmadd213pd zmm3,  zmm23,  0[r15]{1to8}    // a3 = w0 + ( c007_3 * a3 )
    vfmadd213pd zmm4,  zmm24,  0[r15]{1to8}    // a4 = w0 + ( c007_4 * a4 )
    vfmadd213pd zmm5,  zmm25,  0[r15]{1to8}    // a5 = w0 + ( c007_5 * a5 )
    vfmadd213pd zmm6,  zmm26,  0[r15]{1to8}    // a6 = w0 + ( c007_6 * a6 )
    vfmadd213pd zmm7,  zmm27,  0[r15]{1to8}    // a7 = w0 + ( c007_7 * a7 )
    vfmadd213pd zmm8,  zmm28,  0[r15]{1to8}    // a8 = w0 + ( c007_8 * a8 )
    vfmadd213pd zmm9,  zmm29,  0[r15]{1to8}    // a9 = w0 + ( c007_9 * a9 )

	mov         r15,   w                       // load address of w

	vmulpd      zmm20, zmm0,  zmm10            // c007_0 = a0 * p0 
	vmulpd      zmm21, zmm1,  zmm11            // c007_1 = a1 * p1
	vmulpd      zmm22, zmm2,  zmm12            // c007_2 = a2 * p2 
	vmulpd      zmm23, zmm3,  zmm13            // c007_3 = a3 * p3 
	vmulpd      zmm24, zmm4,  zmm14            // c007_4 = a4 * p4 
	vmulpd      zmm25, zmm5,  zmm15            // c007_5 = a5 * p5 
	vmulpd      zmm26, zmm6,  zmm16            // c007_6 = a6 * p6 
	vmulpd      zmm27, zmm7,  zmm17            // c007_7 = a7 * p7 
	vmulpd      zmm28, zmm8,  zmm18            // c007_8 = a8 * p8 
	vmulpd      zmm29, zmm9,  zmm19            // c007_9 = a9 * p9 


	// u007 = c007_20:29 * w20:29
    vfmadd231pd zmm30, zmm20,   0[r15]{1to8}   // u007 += c007_0 * w0
    vfmadd231pd zmm30, zmm21,   8[r15]{1to8}   // u007 += c007_1 * w1
    vfmadd231pd zmm30, zmm22,  16[r15]{1to8}   // u007 += c007_2 * w2
    vfmadd231pd zmm30, zmm23,  24[r15]{1to8}   // u007 += c007_3 * w3
    vfmadd231pd zmm30, zmm24,  32[r15]{1to8}   // u007 += c007_4 * w4
    vfmadd231pd zmm30, zmm25,  40[r15]{1to8}   // u007 += c007_5 * w5
    vfmadd231pd zmm30, zmm26,  48[r15]{1to8}   // u007 += c007_6 * w6
    vfmadd231pd zmm30, zmm27,  56[r15]{1to8}   // u007 += c007_7 * w7
    vfmadd231pd zmm30, zmm28,  64[r15]{1to8}   // u007 += c007_8 * w8
    vfmadd231pd zmm30, zmm29,  72[r15]{1to8}   // u007 += c007_9 * w9

	vmovapd       0[rax], zmm30                 // storing u007




	//vmovapd        0[r9], zmm0                 // storing c007_0 ~ c007_19
	//vmovapd       64[r9], zmm1
	//vmovapd      128[r9], zmm2
	//vmovapd      192[r9], zmm3


	  END:
	}


//   printf( "u007\n" );
//   printf( "%E, %E, %E, %E, %E, %E, %E, %E\n",
//	   u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7]);
//

//   printf( "c007_0\n" );
//   printf( "%E, %E, %E, %E, %E, %E, %E, %E\n",
//	   c_buff[0], c_buff[1], c_buff[2], c_buff[3], c_buff[4], c_buff[5], c_buff[6], c_buff[7]);

   
   //   printf( "c007_1\n" );
//   printf( "%E, %E, %E, %E, %E, %E, %E, %E\n",
//	   c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15]);
//   printf( "c007_28\n" );
//   printf( "%E, %E, %E, %E, %E, %E, %E, %E\n",
//	   c[16], c[17], c[18], c[19], c[20], c[21], c[22], c[23]);
//   printf( "c007_29\n" );
//   printf( "%E, %E, %E, %E, %E, %E, %E, %E\n",
//	   c[24], c[25], c[26], c[27], c[28], c[29], c[30], c[31]);

//   int *cint = (int*)c;

//   printf( "%d, %d, %d, %d, %d, %d, %d, %d\n",
//	   cint[0], cint[1], cint[2], cint[3], cint[4], cint[5], cint[6], cint[7]);
//   printf( "%d, %d, %d, %d, %d, %d, %d, %d\n",
//	   cint[8], cint[9], cint[10], cint[11], cint[12], cint[13], cint[14], cint[15]);


}



