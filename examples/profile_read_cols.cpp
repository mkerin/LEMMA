#include <iostream>
#include <random>
#include <chrono>
#include "genotype_matrix_multithread.cpp"
#include "../src/class.h"
#include "../src/tools/eigen3.3/Dense"
#include "../src/tools/eigen3.3/Sparse"
#include "../src/tools/eigen3.3/Eigenvalues"

/*
Comparison of vector vector to vector-matrix computations
*/
const long int N = 2000;
const int P = 400;

int main() {
	int K = 128;

	std::random_device rd;
	std::mt19937 generator(rd());
	std::binomial_distribution<> bin_dist(2, 0.1);

	parameters p;
	p.low_mem = true;
	GenotypeMatrix X(p);
	X.resize(N, P);

	// Fill genotype Matrix
	for (int jj = 0; jj < P; jj++){
		for (std::uint32_t ii = 0; ii < N; ii++){
			double geno = bin_dist(generator);
			X.assign_index(ii, jj, geno);
		}
	}
	X.calc_scaled_values();

	Eigen::MatrixXd D(N, K);
	int ch_start = 0;
	int ch_len = K;
	std::vector< int > chunk_sizes{1, 2, 4, 8, 16, 32, 64, 128};

	// int n = Eigen::nbThreads( );
	std::cout << "Data initialised" << std::endl;
	double res2;

	// // No diagonal
	// res2 = 0;
	// for (int th = 1; th < 17; th++){
	// 	X.params.n_thread = th;
	// 	std::cout << th << " ";
	// 	for (int ss = 0; ss < 20; ss++){
	// 		auto start = std::chrono::system_clock::now();
	// 		for (int kk = 0; kk < 5; kk++){
	// 			D = X.col_block2(ch_start, ch_len);
	// 			res2 += D(0, 0);
	// 		}
	// 		auto end = std::chrono::system_clock::now();
	// 		auto elapsed = end - start;
	// 		std::cout << elapsed.count() << " ";
	// 	}
	// 	std::cout << std::endl;
	// }
	// std::cout << std::endl << res2 << std::endl << std::endl;

	// std::threads
	int nRuns = 10;
	res2 = 0;
	std::cout << "jobs n_threads";
	for (int ss = 0; ss < nRuns; ss++) std::cout << " run" << ss;
	std::cout << std::endl;

	for (int ch_len : chunk_sizes){
		for (int th = 1; th < 17; th++){
			X.params.n_thread = th;
			std::cout << ch_len << " " << th << " ";
			for (int ss = 0; ss < nRuns; ss++){
				auto start = std::chrono::system_clock::now();
				for (int kk = 0; kk < 5; kk++){
					X.col_block3(ch_start, ch_len, D);
					res2 += D(0, 0);
				}
				auto end = std::chrono::system_clock::now();
				auto elapsed = end - start;
				std::cout << elapsed.count() / ch_len << " ";
			}
			std::cout << std::endl;
		}
	}
	std::cout << std::endl << res2 << std::endl << std::endl;

	// Boost multipool
	std::cout << "jobs n_threads";
	for (int ss = 0; ss < nRuns; ss++) std::cout << " run" << ss;
	std::cout << std::endl;
	res2 = 0;
	for (int ch_len : chunk_sizes){
		for (int th = 1; th < 17; th++){
			X.params.n_thread = th;
			std::cout << th << " ";
			for (int ss = 0; ss < nRuns; ss++){
				auto start = std::chrono::system_clock::now();
				for (int kk = 0; kk < 5; kk++){
					X.col_block4(ch_start, ch_len, D);
					res2 += D(0, 0);
				}
				auto end = std::chrono::system_clock::now();
				auto elapsed = end - start;
				std::cout << elapsed.count() / ch_len << " ";
			}
			std::cout << std::endl;
		}
	}
	std::cout << std::endl << res2 << std::endl << std::endl;

	return 0;
}
