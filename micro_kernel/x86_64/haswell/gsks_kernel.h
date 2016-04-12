#ifndef __GSKS_KERNEL_H__
#define __GSKS_KERNEL_H__

#define KERNEL1(name,type) \
  name(                    \
    int    k,              \
    type   *a,             \
    type   *b,             \
    type   *c,             \
    int    ldc,            \
    aux_t  *aux            \
    )

#define KERNEL2(name,type) \
  name(                    \
    int    k,              \
    int    rhs,            \
    type   *u,             \
    type   *a,             \
    type   *aa,            \
    type   *b,             \
    type   *bb,            \
    type   *w,             \
    type   *c,             \
    ks_t   *ker,           \
    aux_t  *aux            \
    )

void KERNEL1(rank_k_int_d8x6,double);
void KERNEL1(rank_k_asm_d8x6,double);
void KERNEL2(gaussian_int_d8x6,double);
void KERNEL2(polynomial_int_d8x6,double);
void KERNEL2(laplace_int_d8x6,double);
void KERNEL2(variable_bandwidth_gaussian_int_d8x6,double);
void KERNEL2(tanh_int_d8x6,double);
void KERNEL2(quartic_int_d8x6,double);
void KERNEL2(multiquadratic_int_d8x6,double);
void KERNEL2(epanechnikov_int_d8x6,double);

void KERNEL1((*rankk),double)  = {
  rank_k_asm_d8x6
  //rank_k_int_d8x6
};

void KERNEL2((*micro[ 8 ]),double) = {
  gaussian_int_d8x6,
  polynomial_int_d8x6,
  laplace_int_d8x6,
  variable_bandwidth_gaussian_int_d8x6,
  tanh_int_d8x6,
  quartic_int_d8x6,
  multiquadratic_int_d8x6,
  epanechnikov_int_d8x6
};

#endif // define __GSKS_KERNEL_H__
