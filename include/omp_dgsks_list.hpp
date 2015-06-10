void omp_dgsks_list_unsymmetric(
    ks_t   *kernel,
    int    k,
    std::vector<double> &u,
    int    nxa,
    double *XA,
    std::vector< std::vector<int> > &alist,
    int    nxb,
    double *XB,
    std::vector< std::vector<int> > &blist,
    double *w,
    std::vector< std::vector<int> > &wlist
    );

void omp_dgsks_list_symmetric(
    ks_t   *kernel,
    int    k,
    std::vector<double> &u,
    double *XA,
    int    nxa,
    std::vector< std::vector<int> > &alist,
    std::vector< std::vector<int> > &blist,
    double *w,
    std::vector< std::vector<int> > &wlist
    );

void omp_dgsks_list_separated_u_unsymmetric(
    ks_t   *kernel,
    int    k,
    std::vector<double> &u,
    std::vector< std::vector<int> > &ulist,
    int    nxa,
    double *XA,
    std::vector< std::vector<int> > &alist,
    int    nxb,
    double *XB,
    std::vector< std::vector<int> > &blist,
    double *w,
    std::vector< std::vector<int> > &wlist
    );

void omp_dgsks_list_separated_u_symmetric(
    ks_t   *kernel,
    int    k,
    std::vector<double> &u,
    std::vector< std::vector<int> > &ulist,
    double *XA,
    int    nxa,
    std::vector< std::vector<int> > &alist,
    std::vector< std::vector<int> > &blist,
    double *w,
    std::vector< std::vector<int> > &wlist
    );

void omp_dgsks_list(
    ks_t   *kernel,
    int    k,
    std::vector<double> &u,
    std::vector< std::vector<int> > &ulist,
    double *XA,
    double *XA2,
    std::vector< std::vector<int> > &alist,
    double *XB,
    double *XB2,
    std::vector< std::vector<int> > &blist,
    double *w,
    std::vector< std::vector<int> > &wlist
    );
