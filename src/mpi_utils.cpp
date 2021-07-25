//
// Created by kerin on 2019-05-25.
//

#include "mpi_utils.hpp"
#include "parameters.hpp"
#include "file_utils.hpp"

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
                                               std::map<long, bool> &incomplete_cases,
                                               std::map<long, int> &sample_location) {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// dXtEEX_lowertri can be quite large. If really big, then we store fewer
	// samples on rank 0 to avoid going over maxBytesPerRank.
    long long n_valid_sids = n_samples - (long) incomplete_cases.size();
    long long dxteexBytes    = 8 * n_var * n_env * (n_env + 1) / 2;
    long long bytesPerSample = (n_var + 8 * n_env + 8);
    if (p.maxBytesPerRank < (double) (dxteexBytes + bytesPerSample * n_valid_sids) / (double) size){
        throw std::runtime_error("Not possible to load all data into memory across "+std::to_string(size)+
                                 " ranks whilst using less than "+std::to_string(p.maxBytesPerRank)+" bytes per rank.");
    }
    if (p.maxBytesPerRank < dxteexBytes) {
        throw std::runtime_error("dXtEEX is predicted to require "+std::to_string(dxteexBytes)+" bytes and is (currently) stored "
                                 "entirely on 1 rank. This will breach the limit of "+
                                 std::to_string(p.maxBytesPerRank)+" bytes per rank.");
    }
    long samplesPerRank, rankZeroSamples;
    if (size > 1) { // rank0 might be dedicated entirely to dxteex
        rankZeroSamples = std::min(std::max((long long) 0,(p.maxBytesPerRank - dxteexBytes)/bytesPerSample), (n_valid_sids+size-1) / size);
        samplesPerRank = (n_valid_sids-rankZeroSamples+size-2) / (size-1);
    } else {
        samplesPerRank  = n_valid_sids;
        rankZeroSamples = samplesPerRank;
    }

	// store 'rank' that each sample is located in
	// samples excluded due to missing data have location -1
    long diff = samplesPerRank - rankZeroSamples;
	long iiValid = 0;
	for (long ii = 0; ii < n_samples; ii++) {
		if (incomplete_cases.count(ii) == 0) {
		    if (iiValid < rankZeroSamples){
                sample_location[ii] = 0;
		    } else {
                sample_location[ii] = (int) ((iiValid - rankZeroSamples) / samplesPerRank) + 1;
		    }
			iiValid++;
		} else {
			sample_location[ii] = -1;
		}
	}
	assert(iiValid == n_valid_sids);

	if(p.debug) {
		std::vector<long> allii(size, 0);
		for (const auto &kv : sample_location) {
			if (kv.second != -1) {
				int local_rank = kv.second;
				allii[local_rank]++;
			}
		}
		std::cout << "Samples stored on each rank: " << std::endl;
		for (int rr = 0; rr < size; rr++) {
			std::cout << "Rank " << rr << ": " << allii[rr] << std::endl;
		}
	}

	std::vector<long> rank_cases;
	for (long ii = 0; ii < n_samples; ii++) { // keep only the samples for this rank
	    if (sample_location[ii] == rank){
            rank_cases.push_back(ii);
	    } else {
            incomplete_cases[ii] = true;
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

std::string mpiUtils::currentUsageRAM(){
	int world_rank;
	long long kbMax, kbGlobal, kbLocal = fileUtils::getValueRAM();
	long long peakMax, peakLocal = fileUtils::getValueRAM("VmPeak:");
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Allreduce(&kbLocal, &kbMax, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&peakLocal, &peakMax, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&kbLocal, &kbGlobal, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
	double gbGlobal = kbGlobal / 1000.0 / 1000.0;
	double gbMax = kbMax / 1000.0 / 1000.0;
	double gbPeakMax = peakMax / 1000.0 / 1000.0;

	char buffer [200];
	int n;
	n = sprintf(buffer, "RAM usage: %.2f GB in total; max current=%.2f GB and max peak=%.2f GB per rank", gbGlobal, gbMax, gbMax);
	std::string res(buffer);
	return res;
}

void mpiUtils::mpiReduce_double(void *local, void *global, long size) {
	MPI_Allreduce(local, global, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

double mpiUtils::mpiReduce_inplace(double *local) {
	double global;
	MPI_Allreduce(local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	return global;
}

double mpiUtils::mpiReduce_inplace(double local) {
	double global;
	MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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
