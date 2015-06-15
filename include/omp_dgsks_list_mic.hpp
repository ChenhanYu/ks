/*
 * --------------------------------------------------------------------------
 * GSKS (General Stride Kernel Summation)
 * --------------------------------------------------------------------------
 * Copyright (C) 2015, The University of Texas at Austin
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 *
 * omp_dgsks_list_mic.hpp
 *
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
 * */


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
