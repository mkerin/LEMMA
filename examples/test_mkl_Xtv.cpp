
#define EIGEN_USE_MKL_ALL

#include <iostream>
#include <chrono>
#include "../src/my_timer.hpp"
#include "../src/tools/eigen3.3/Dense"
#include "../src/tools/eigen3.3/Sparse"
#include "../src/tools/eigen3.3/Eigenvalues"

/*
Comparison of vector vector to vector-matrix computations
*/
const long int N = 250000;
const int K = 64;

int main() {
	// Single core version
	Eigen::MatrixXd X  = Eigen::MatrixXd::Random(N, K);
	Eigen::VectorXd vv = Eigen::VectorXd::Random(N);
	Eigen::VectorXd D  = Eigen::VectorXd::Random(N);

	// int n = Eigen::nbThreads( );
	std::cout << "Data initialised" << std::endl;
	double res2 = 0;

	// With diagonal
	std::cout << "DGEMV" << std::endl << std::endl;
	Eigen::VectorXd resv(K);
	for (int th = 1; th < 17; th++){
		mkl_set_num_threads_local(th);
		std::cout << th << " ";
		for (int ss = 0; ss < 20; ss++){
			auto start = std::chrono::system_clock::now();
			for (int kk = 0; kk < 5; kk++){
				resv = X.transpose() * vv;
				res2 += resv(0);
			}
			auto end = std::chrono::system_clock::now();
			auto elapsed = end - start;
			std::cout << elapsed.count() << " ";
		}
		std::cout << std::endl;
	}

	// No diagonal DGEMM
	std::cout << "DGEMM" << std::endl << std::endl;
	Eigen::MatrixXd res(K, K);
	for (int th = 1; th < 16; th++){
		mkl_set_num_threads_local(th);
		std::cout << th << " ";
		// MyTimer tt1("10 repeats takes: %ts \n");
		for (int ss = 0; ss < 10; ss++){
			auto start = std::chrono::system_clock::now();
			for (int kk = 0; kk < 5; kk++){
				res = X.transpose() * X;
				res2 += res(0, 0);
			}
			auto end = std::chrono::system_clock::now();
			auto elapsed = end - start;
			std::cout << elapsed.count() << " ";
		}
		std::cout << std::endl;
	}
	//
	// // With diagonal
	// for (int th = 1; th < 17; th++){
	// 	Eigen::setNbThreads(th);
	// 	std::cout << th << " ";
	// 	// MyTimer tt2("10 repeats takes: %ts \n");
	// 	for (int ss = 0; ss < 20; ss++){
	// 		auto start = std::chrono::system_clock::now();
	// 		for (int kk = 0; kk < 5; kk++){
	// 			res = X.transpose() * D.asDiagonal() * X;
	// 			res2 += res(0, 0);
	// 		}
	// 		auto end = std::chrono::system_clock::now();
	// 		auto elapsed = end - start;
	// 		std::cout << elapsed.count() << " ";
	// 		// tt2.stop();
	// 		// std::cout << tt2.get_lap_seconds() << " ";
	// 		// tt2.resume();
	// 	}
	// 	std::cout << std::endl;
	// }
	//
	// // Upper triangular
	// for (int th = 1; th < 17; th++){
	// 	Eigen::setNbThreads(th);
	// 	std::cout << th << " ";
	// 	// MyTimer tt2("10 repeats takes: %ts \n");
	// 	for (int ss = 0; ss < 20; ss++){
	// 		auto start = std::chrono::system_clock::now();
	// 		for (int kk = 0; kk < 5; kk++){
	// 			res.triangularView<Eigen::Upper>() = X.transpose() * D.asDiagonal() * X;
	// 			res2 += res(0, 0);
	// 		}
	// 		auto end = std::chrono::system_clock::now();
	// 		auto elapsed = end - start;
	// 		std::cout << elapsed.count() << " ";
	// 	}
	// 	std::cout << std::endl;
	// }

	return 0;
}
