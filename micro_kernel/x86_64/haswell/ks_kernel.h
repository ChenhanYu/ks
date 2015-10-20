void ks_rank_k_asm_d8x4(
    int    k,
    double *a,
    double *b,
    double *c,
    int    ldc,
    aux_t  *aux
    );

void ks_rank_k_int_d8x4(
    int    k,
    double *a,
    double *b,
    double *c,
    int    ldc,
    aux_t  *aux
    );

// Test d12x4
void ks_rank_k_int_d12x4(
    int    k,
    double *a,
    double *b,
    double *c,
    int    ldc,
    aux_t  *aux
    );

void ks_gaussian_int_d8x4(
    int    k,
    int    rhs,
    double *h,
    double *u,
    double *a,
    double *aa,
    double *b,
    double *bb,
    double *w,
    double *c,
    ks_t   *ker,
    aux_t  *aux
    );

void ks_gaussian_svml_d8x4(
    int    k,
    int    rhs,
    double *h,
    double *u,
    double *a,
    double *aa,
    double *b,
    double *bb,
    double *w,
    double *c,
    ks_t   *ker,
    aux_t  *aux
    );

void ks_polynomial_int_d8x4(
    int    k,
    int    rhs,
    double *h,
    double *u,
    double *a,
    double *aa,
    double *b,
    double *bb,
    double *w,
    double *c,
    ks_t   *ker,
    aux_t  *aux
    );

void ks_laplace3d_int_d8x4(
    int    k,
    int    rhs,
    double *h,
    double *u,
    double *a,
    double *aa,
    double *b,
    double *bb,
    double *w,
    double *c,
    ks_t   *ker,
    aux_t  *aux
    );

void ks_variable_bandwidth_gaussian_int_d8x4(
    int    k,
    int    rhs,
    double *h,
    double *u,
    double *a,
    double *aa,
    double *b,
    double *bb,
    double *w,
    double *c,
    ks_t   *ker,
    aux_t  *aux
    );

void ks_tanh_int_d8x4(
    int    k,
    int    rhs,
    double *h,
    double *u,
    double *a,
    double *aa,
    double *b,
    double *bb,
    double *w,
    double *c,
    ks_t   *ker,
    aux_t  *aux
    );

void ks_quartic_int_d8x4(
    int    k,
    int    rhs,
    double *h,
    double *u,
    double *a,
    double *aa,
    double *b,
    double *bb,
    double *w,
    double *c,
    ks_t   *ker,
    aux_t  *aux
    );

void ks_multiquadratic_int_d8x4(
    int    k,
    int    rhs,
    double *h,
    double *u,
    double *a,
    double *aa,
    double *b,
    double *bb,
    double *w,
    double *c,
    ks_t   *ker,
    aux_t  *aux
    );

// Test d12x4
void ks_multiquadratic_int_d12x4(
    int    k,
    int    rhs,
    double *h,
    double *u,
    double *a,
    double *aa,
    double *b,
    double *bb,
    double *w,
    double *c,
    ks_t   *ker,
    aux_t  *aux
    );

void ks_epanechnikov_int_d8x4(
    int    k,
    int    rhs,
    double *h,
    double *u,
    double *a,
    double *aa,
    double *b,
    double *bb,
    double *w,
    double *c,
    ks_t   *ker,
    aux_t  *aux
    );

void (*micro[ 8 ]) (
    int    k,
    int    rhs,
    double *h,
    double *u,
    double *a,
    double *aa,
    double *b,
    double *bb,
    double *w,
    double *c,
    ks_t   *ker,
    aux_t  *aux
    ) = { 
    //ks_gaussian_int_d8x4,
    ks_gaussian_svml_d8x4,
    ks_polynomial_int_d8x4,
    ks_laplace3d_int_d8x4,
    ks_variable_bandwidth_gaussian_int_d8x4,
    ks_tanh_int_d8x4,
    ks_quartic_int_d8x4,
    ks_multiquadratic_int_d8x4,
    //ks_multiquadratic_int_d12x4,
    ks_epanechnikov_int_d8x4
  };

void (*rankk) (
    int    k,
    double *a,
    double *b,
    double *c,
    int    ldc,
    aux_t  *aux
    ) = {
  ks_rank_k_int_d8x4
  //ks_rank_k_int_d12x4
};
