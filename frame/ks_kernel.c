#include <ks.h>

void ks_gaussian_int_d8x4(
    int    k,
    int    rhs,
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

void (*micro[ 2 ])(
    int    k,
    int    rhs,
    double *u,
    double *a,
    double *aa,
    double *b,
    double *bb,
    double *w,
    double *c,
    ks_t   *ker,
    aux_t  *aux
    ) =
{ ks_gaussian_int_d8x4, ks_polynomial_int_d8x4 };
