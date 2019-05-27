//
// Created by kerin on 2019-05-25.
//

#include "mpi_utils.hpp"

#include <mpi.h>
#include <map>
#include <vector>
#include <iostream>
#include <fstream>

#include "tools/eigen3.3/Dense"

void
mpiUtils::partition_valid_samples_across_ranks(const long &n_samples, std::map<std::size_t, bool> &incomplete_cases) {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	std::vector<long> valid_sids, rank_cases;
	for (long ii = 0; ii < n_samples; ii++) {
		if (incomplete_cases.count(ii) == 0) {
			valid_sids.push_back(ii);
		}
	}

	long n_valid_sids = valid_sids.size();
	long samplesPerRank = (n_valid_sids + size - 1) / size;
	for (long ii = 0; ii < n_valid_sids; ii++) {
		if (ii < rank * samplesPerRank || ii >= (rank+1) * samplesPerRank) {
			incomplete_cases[valid_sids[ii]] = true;
		} else {
			rank_cases.push_back(valid_sids[ii]);
		}
	}

	// Check Nlocal sums to expected number of valid samples
	long Nlocal = rank_cases.size();
	long Nglobal;
	MPI_Reduce(&Nlocal, &Nglobal, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if(rank == 0) {
		assert(Nglobal == n_valid_sids);
	}
}

void mpiUtils::mpiReduce_double(void *local, void *global, long size) {
	MPI_Allreduce(local, global, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

double mpiUtils::mpiReduce_inplace(double *local) {
	double global;
	MPI_Allreduce(local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	return global;
}

Eigen::MatrixXd mpiUtils::mpiReduce_inplace(Eigen::Ref<Eigen::MatrixXd> local){
	Eigen::MatrixXd global(local.rows(), local.cols());
	long size = local.rows() * local.cols();
	MPI_Allreduce(local.data(), global.data(), size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	return global;
}

template<typename Derived>
double mpiUtils::squaredNorm(const Eigen::DenseBase<Derived> &obj) {
	double resLocal = obj.squaredNorm();
	double resGlobal;
	MPI_Allreduce(&resLocal, &resGlobal, 1, MPI_DOUBLE, MPI_SUM,
				  MPI_COMM_WORLD);
	return resGlobal;
}

void mpiUtils::sanitise_cout() {
	 int rank;
	 MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	 // Mute std::cout except on rank zero
	 std::ofstream sink("/dev/null");
	 if (rank != 0) {
		 std::cout << "Muting rank " << rank << "..." << std::endl;
		 std::cout.rdbuf(sink.rdbuf());
	 }
}
