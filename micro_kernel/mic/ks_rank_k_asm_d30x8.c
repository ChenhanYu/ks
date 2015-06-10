#include <immintrin.h>
#include <assert.h>
#include <ks.h>

typedef unsigned long long dim_t;
typedef unsigned long long inc_t;


#define A_L1_PREFETCH_DIST 4
#define B_L1_PREFETCH_DIST 2
#define L2_PREFETCH_DIST  16 // Must be greater than 10, because of the way the loop is constructed.

//Alternate code path uused if C is not row-major
#define UPDATE_C_ROW_SCATTERED(REG1, NUM, BASE_DEST) \
{ \
        __asm kmov k3, ebx \
        __asm GATHER##NUM: \
            __asm vgatherdpd zmm31{k3}, [BASE_DEST + zmm30 * 8] \
            __asm jknzd k3, GATHER##NUM \
        \
        __asm vmulpd REG1, REG1, 0[r12]{1to8} /*scale by alpha*/ \
        __asm vfmadd132pd zmm31, REG1, 0[r13]{1to8} /*scale by beta, add in result*/\
        __asm kmov k3, ebx \
        \
        __asm SCATTER##NUM: \
            __asm vscatterdpd [BASE_DEST + zmm30 * 8]{k3}, zmm31 \
            __asm jknzd k3, SCATTER##NUM \
        __asm add BASE_DEST, r11 \
}


//One iteration of the k_r loop.
//Each iteration, we prefetch A into L1 and into L2
#define ONE_ITER_MAIN_LOOP(C_ADDR, COUNTER) \
{\
        __asm vbroadcastf64x4   zmm30, 0[r15]           \
        __asm vmovapd zmm31, 0[rbx]                     \
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
        __asm vprefetch1 0[rbx + r13]                   \
        __asm vfmadd231pd zmm26, zmm31, -6*8[r15]{1to8} \
        __asm vprefetch0 B_L1_PREFETCH_DIST*8*8[rbx]    \
        __asm vfmadd231pd zmm27, zmm31, -5*8[r15]{1to8} \
        __asm add rbx, r9                               \
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
        __asm vmovapd zmm31, 0[rbx]                     \
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
        __asm vprefetch1 0[rbx + r13]                   \
        __asm vfmadd231pd zmm26, zmm31, -6*8[r15]{1to8} \
        __asm vprefetch0 B_L1_PREFETCH_DIST*8*8[rbx]    \
        __asm vfmadd231pd zmm27, zmm31, -5*8[r15]{1to8} \
        __asm add rbx, r9                               \
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
        __asm vmovapd zmm31, 0[rbx]                     \
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
        __asm vprefetch1 0[rbx + r13]                   \
        __asm vfmadd231pd zmm26, zmm31, -6*8[r15]{1to8} \
        __asm vprefetch0 B_L1_PREFETCH_DIST*8*8[rbx]    \
        __asm vfmadd231pd zmm27, zmm31, -5*8[r15]{1to8} \
        __asm add rbx, r9                               \
        __asm vfmadd231pd zmm28, zmm31, -4*8[r15]{1to8} \
        __asm cmp r8, 0                                 \
        __asm vfmadd231pd zmm29, zmm31, -3*8[r15]{1to8} \
\
}

//This is an array used for the scattter/gather instructions.
extern int offsets[16];


//#define MONITORS
//#define LOOPMON
void ks_rank_k_asm_d30x8(
    dim_t  k,
    double *a,
    double *b,
    double *c,
    int    ldc,
    aux_t  *aux
    )
{
//void bli_dgemm_opt_30x8(
//                    dim_t            k,
//                    double* alpha,
//                    double* a,
//                    double* b,
//                    double* beta,
//                    double* c, inc_t rs_c, inc_t cs_c,
//                    aux_t*       data
//                  )
//{
    //double * a_next = bli_auxinfo_next_a( data );
    //double * b_next = bli_auxinfo_next_b( data );
    double * a_next = a;
    double * b_next = b;

    double dmtwo = -2.0;
    double dzero =  0.0;
	double *alpha = &dmtwo;
	double *beta  = &dzero;


    int * offsetPtr = &offsets[0];

#ifdef MONITORS
    int toph, topl, both, botl, midl, midh, mid2l, mid2h;
#endif
#ifdef LOOPMON
    int tlooph, tloopl, blooph, bloopl;
#endif
    
    __asm
    {
#ifdef MONITORS
        rdtsc
        mov topl, eax
        mov toph, edx 
#endif
        vpxord  zmm0,  zmm0, zmm0
        vmovaps zmm1,  zmm0  //clear out registers
        vmovaps zmm2,  zmm0 
        mov rsi, k    //loop index
        vmovaps zmm3,  zmm0 

//        mov r11, rs_c           //load row stride
        vmovaps zmm4,  zmm0 
//        sal r11, 3              //scale row stride
        vmovaps zmm5,  zmm0 
        mov r15, a              //load address of a
        vmovaps zmm6,  zmm0 
        mov rbx, b              //load address of b
        vmovaps zmm7,  zmm0 

        vmovaps zmm8,  zmm0 
//        lea r10, [r11 + 2*r11 + 0] //r10 has 3 * r11
        vmovaps zmm9,  zmm0
        vmovaps zmm10, zmm0 
//        mov rdi, r11    
        vmovaps zmm11, zmm0 
//        sal rdi, 2              //rdi has 4*r11

        vmovaps zmm12, zmm0 
        mov rcx, c              //load address of c for prefetching
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

#ifdef MONITORS
        rdtsc
        mov midl, eax
        mov midh, edx 
#endif
        jle CONSIDER_UNDER_40
        sub rsi, 30 + L2_PREFETCH_DIST
        
        //First 30 iterations
        LOOPREFECHCL2:
            ONE_ITER_PC_L2(rcx)
        jne LOOPREFECHCL2
        mov rcx, c

        //Main Loop.
        LOOPMAIN:
            ONE_ITER_MAIN_LOOP(rcx, rsi)
        jne LOOPMAIN
        
        //Penultimate 22 iterations.
        //Break these off from the main loop to avoid prefetching extra shit.
        mov r14, a_next
        mov r13, b_next
        sub r14, r15
        sub r13, rbx
        
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

#ifdef MONITORS
        rdtsc
        mov mid2l, eax
        mov mid2h, edx
#endif

        mov r9, c               //load address of c for update
        mov r12, alpha          //load address of alpha

        // Check if C is row stride. If not, jump to the slow scattered update
//        mov r14, cs_c
//        dec r14
//        jne SCATTEREDUPDATE

        mov r14, beta
        vbroadcastsd zmm31, 0[r14] 


        vmulpd zmm0, zmm0, 0[r12]{1to8}
        vmulpd zmm1, zmm1, 0[r12]{1to8}
        vmulpd zmm2, zmm2, 0[r12]{1to8}
        vmulpd zmm3, zmm3, 0[r12]{1to8}
        vfmadd231pd zmm0, zmm31,   0[r9]
        vfmadd231pd zmm1, zmm31,  64[r9]
        vfmadd231pd zmm2, zmm31, 128[r9]
        vfmadd231pd zmm3, zmm31, 192[r9]
        vmovapd   0[r9], zmm0
        vmovapd  64[r9], zmm1
        vmovapd 128[r9], zmm2
        vmovapd 192[r9], zmm3
        add r9, 256

        vmulpd zmm4, zmm4, 0[r12]{1to8}
        vmulpd zmm5, zmm5, 0[r12]{1to8}
        vmulpd zmm6, zmm6, 0[r12]{1to8}
        vmulpd zmm7, zmm7, 0[r12]{1to8}
        vfmadd231pd zmm4, zmm31,   0[r9]
        vfmadd231pd zmm5, zmm31,  64[r9]
        vfmadd231pd zmm6, zmm31, 128[r9]
        vfmadd231pd zmm7, zmm31, 192[r9]
        vmovapd   0[r9], zmm4
        vmovapd  64[r9], zmm5
        vmovapd 128[r9], zmm6
        vmovapd 192[r9], zmm7
        add r9, 256

        vmulpd  zmm8,  zmm8, 0[r12]{1to8}
        vmulpd  zmm9,  zmm9, 0[r12]{1to8}
        vmulpd zmm10, zmm10, 0[r12]{1to8}
        vmulpd zmm11, zmm11, 0[r12]{1to8}
        vfmadd231pd  zmm8, zmm31,   0[r9]
        vfmadd231pd  zmm9, zmm31,  64[r9]
        vfmadd231pd zmm10, zmm31, 128[r9]
        vfmadd231pd zmm11, zmm31, 192[r9]
        vmovapd   0[r9], zmm8
        vmovapd  64[r9], zmm9
        vmovapd 128[r9], zmm10
        vmovapd 192[r9], zmm11
        add r9, 256

        vmulpd zmm12, zmm12, 0[r12]{1to8}
        vmulpd zmm13, zmm13, 0[r12]{1to8}
        vmulpd zmm14, zmm14, 0[r12]{1to8}
        vmulpd zmm15, zmm15, 0[r12]{1to8}
        vfmadd231pd zmm12, zmm31,   0[r9]
        vfmadd231pd zmm13, zmm31,  64[r9]
        vfmadd231pd zmm14, zmm31, 128[r9]
        vfmadd231pd zmm15, zmm31, 192[r9]
        vmovapd   0[r9], zmm12
        vmovapd  64[r9], zmm13
        vmovapd 128[r9], zmm14
        vmovapd 192[r9], zmm15
        add r9, 256
        
        vmulpd zmm16, zmm16, 0[r12]{1to8}
        vmulpd zmm17, zmm17, 0[r12]{1to8}
        vmulpd zmm18, zmm18, 0[r12]{1to8}
        vmulpd zmm19, zmm19, 0[r12]{1to8}
        vfmadd231pd zmm16, zmm31,   0[r9]
        vfmadd231pd zmm17, zmm31,  64[r9]
        vfmadd231pd zmm18, zmm31, 128[r9]
        vfmadd231pd zmm19, zmm31, 192[r9]
        vmovapd   0[r9], zmm16
        vmovapd  64[r9], zmm17
        vmovapd 128[r9], zmm18
        vmovapd 192[r9], zmm19
        add r9, 256

        vmulpd zmm20, zmm20, 0[r12]{1to8}
        vmulpd zmm21, zmm21, 0[r12]{1to8}
        vmulpd zmm22, zmm22, 0[r12]{1to8}
        vmulpd zmm23, zmm23, 0[r12]{1to8}
        vfmadd231pd zmm20, zmm31,   0[r9]
        vfmadd231pd zmm21, zmm31,  64[r9]
        vfmadd231pd zmm22, zmm31, 128[r9]
        vfmadd231pd zmm23, zmm31, 192[r9]
        vmovapd   0[r9], zmm20
        vmovapd  64[r9], zmm21
        vmovapd 128[r9], zmm22
        vmovapd 192[r9], zmm23
        add r9, 256

        vmulpd zmm24, zmm24, 0[r12]{1to8}
        vmulpd zmm25, zmm25, 0[r12]{1to8}
        vmulpd zmm26, zmm26, 0[r12]{1to8}
        vmulpd zmm27, zmm27, 0[r12]{1to8}
        vfmadd231pd zmm24, zmm31,   0[r9]
        vfmadd231pd zmm25, zmm31,  64[r9]
        vfmadd231pd zmm26, zmm31, 128[r9]
        vfmadd231pd zmm27, zmm31, 192[r9]
        vmovapd   0[r9], zmm24
        vmovapd  64[r9], zmm25
        vmovapd 128[r9], zmm26
        vmovapd 192[r9], zmm27
        add r9, 256

        vmulpd zmm28, zmm28, 0[r12]{1to8}
        vmulpd zmm29, zmm29, 0[r12]{1to8}
        vfmadd231pd zmm28, zmm31,   0[r9]
        vfmadd231pd zmm29, zmm31,  64[r9]
        vmovapd   0[r9], zmm28
        vmovapd  64[r9], zmm29
        
        jmp END
        
        SCATTEREDUPDATE:
//        mov r10, offsetPtr 
//        vmovapd zmm31, 0[r10] 
//        vpbroadcastd zmm30, cs_c 
//        mov r13, beta
//        vpmulld zmm30, zmm31, zmm30 
//
//        mov ebx, 255 
//        UPDATE_C_ROW_SCATTERED(zmm0, 0, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm1, 1, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm2, 2, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm3, 3, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm4, 4, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm5, 5, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm6, 6, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm7, 7, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm8, 8, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm9, 9, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm10, 10, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm11, 11, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm12, 12, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm13, 13, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm14, 14, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm15, 15, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm16, 16, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm17, 17, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm18, 18, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm19, 19, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm20, 20, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm21, 21, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm22, 22, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm23, 23, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm24, 24, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm25, 25, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm26, 26, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm27, 27, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm28, 28, r9) 
//        UPDATE_C_ROW_SCATTERED(zmm29, 29, r9)

        END:
#ifdef MONITORS
        rdtsc
        mov botl, eax
        mov both, edx
#endif
    }

#ifdef LOOPMON
    printf("looptime = \t%d\n", bloopl - tloopl);
#endif
#ifdef MONITORS
    dim_t top = ((dim_t)toph << 32) | topl;
    dim_t mid = ((dim_t)midh << 32) | midl;
    dim_t mid2 = ((dim_t)mid2h << 32) | mid2l;
    dim_t bot = ((dim_t)both << 32) | botl;
    printf("setup =\t%u\tmain loop =\t%u\tcleanup=\t%u\ttotal=\t%u\n", mid - top, mid2 - mid, bot - mid2, bot - top);
#endif
}



