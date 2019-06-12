//
// Created by kerin on 2019-05-25.
//

#include "mpi_utils.hpp"
#include "parameters.hpp"

#include <mpi.h>
#include <map>
#include <vector>
#include <iostream>
#include <fstream>

#include "tools/eigen3.3/Dense"

void
mpiUtils::partition_valid_samples_across_ranks(const long &n_samples,
                                               const long &n_var,
                                               const long &n_env,
                                               const parameters &p,
                                               std::map<long, bool> &incomplete_cases) {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	std::vector<long> valid_sids, rank_cases;
	for (long ii = 0; ii < n_samples; ii++) {
		if (incomplete_cases.count(ii) == 0) {
			valid_sids.push_back(ii);
		}
	}

	// dXtEEX_lowertri can be quite large. If really big, then we store fewer
	// samples on rank 0 to avoid going over maxBytesPerRank.
	// WARNING: Need atleast 1 sample on each rank
	long long dXtEEX_bytes = 8 * n_var * n_env * (n_env + 1) / 2;
	if(dXtEEX_bytes >= p.maxBytesPerRank) {
		throw std::runtime_error("Error: will not be able to store dXtEEX on "
		                         "single rank. Either reduce the number of "
		                         "environmental variables, allow more RAM to "
		                         "be used per rank or get in touch to discuss "
		                         "algo implementation changes.");
	}

	long n_valid_sids = valid_sids.size();
	long samplesPerRank = (n_valid_sids + size - 1) / size;
	long long rankZeroBytes = dXtEEX_bytes + n_var * samplesPerRank;
	long rankZeroSamples;
	if(rankZeroBytes > p.maxBytesPerRank) {
		// Predicted to overflow maxBytesPerRank. Adjust accordingly.
		if(p.verbose){
			std::cout << "Reducing no. of samples stored on rank 0 from ";
			std::cout << samplesPerRank << " to ";
		}
		long size1 = size - 1;
		rankZeroSamples = (p.maxBytesPerRank - dXtEEX_bytes) / (long long) n_var;
		samplesPerRank = (n_valid_sids - rankZeroSamples + size1 - 1) / size1;
		if (p.verbose) std::cout << rankZeroSamples << " to allow space for dXtEEX" << std::endl;
		assert(rankZeroSamples > 0);
	} else {
		// No overflow; hence have same number of samples on all ranks.
		rankZeroSamples = samplesPerRank;
	}

	long diff = samplesPerRank - rankZeroSamples;
	for (long ii = 0; ii < n_valid_sids; ii++) {
		long ii1 = ii + diff;
		if (ii1 < rank * samplesPerRank || ii1 >= (rank+1) * samplesPerRank) {
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

long mpiUtils::mpiReduce_inplace(long *local) {
	long global;
	MPI_Allreduce(local, &global, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	return global;
}

long long mpiUtils::mpiReduce_inplace(long long *local) {
	long global;
	MPI_Allreduce(local, &global, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
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
