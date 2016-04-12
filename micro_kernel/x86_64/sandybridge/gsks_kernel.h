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

void KERNEL1(ks_rank_k_asm_d8x4,double);
void KERNEL1(ks_rank_k_int_d8x4,double);
void KERNEL2(ks_gaussian_int_d8x4,double);
void KERNEL2(ks_gaussian_svml_d8x4,double);
void KERNEL2(ks_polynomial_int_d8x4,double);
void KERNEL2(ks_laplace3d_int_d8x4,double);
void KERNEL2(ks_variable_bandwidth_gaussian_int_d8x4,double);
void KERNEL2(ks_tanh_int_d8x4,double);
void KERNEL2(ks_quartic_int_d8x4,double);
void KERNEL2(ks_multiquadratic_int_d8x4,double);
void KERNEL2(ks_epanechnikov_int_d8x4,double);

void KERNEL1((*rankk),double)  = {
  ks_rank_k_asm_d8x4
};

void KERNEL2((*micro[ 8 ]),double) = {
  ks_gaussian_int_d8x4,
  ks_polynomial_int_d8x4,
  ks_laplace3d_int_d8x4,
  ks_variable_bandwidth_gaussian_int_d8x4,
  ks_tanh_int_d8x4,
  ks_quartic_int_d8x4,
  ks_multiquadratic_int_d8x4,
  ks_epanechnikov_int_d8x4
};

#endif // define __GSKS_KERNEL_H__
