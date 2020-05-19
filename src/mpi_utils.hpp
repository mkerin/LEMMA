//
// Created by kerin on 2019-05-23.
//

#ifndef LEMMA_MPI_UTILS_HPP
#define LEMMA_MPI_UTILS_HPP

#include "tools/eigen3.3/Dense"
#include "parameters.hpp"

#include <mpi.h>
#include <map>
#include <vector>
#include <iostream>

// #ifdef DATA_AS_FLOAT
// using MPI_EIGEN_TYPE = MPI_FLOAT;
// #else
// using MPI_EIGEN_TYPE = MPI_DOUBLE;
// #endif

namespace mpiUtils {

void sanitise_cout();
std::string currentUsageRAM();

// Partition samples across ranks
void partition_valid_samples_across_ranks(const long& n_samples,
                                          const long &n_var,
                                          const long &n_env,
                                          const parameters &p,
                                          std::map<long, bool>& incomplete_cases,
                                          std::map<long, int>& sample_location);

void mpiReduce_double(void* local, void* global, long size);

double mpiReduce_inplace(double* local);
double mpiReduce_inplace(double local);
long mpiReduce_inplace(long* local);
long long mpiReduce_inplace(long long* local);

Eigen::MatrixXd mpiReduce_inplace(Eigen::Ref<Eigen::MatrixXd> local);

template <typename Derived>
double squaredNorm(const Eigen::DenseBase<Derived>&obj);
}

#endif //LEMMA_MPI_UTILS_HPP
