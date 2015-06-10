void omp_dgsks_list_unsymmetric_mic(
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

void omp_dgsks_list_symmetric_mic(
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

void omp_dgsks_list_mic(
    ks_t   *kernel,
    int    k,
    std::vector<double> &u,
    double *XA,
    double *XA2,
    std::vector< std::vector<int> > &alist,
    double *XB,
    double *XB2,
    std::vector< std::vector<int> > &blist,
    double *w,
    std::vector< std::vector<int> > &wlist
    );
