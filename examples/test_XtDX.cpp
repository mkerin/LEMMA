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

void single_core_scheme(
	Eigen::VectorXd& ym,
	Eigen::VectorXd& yx,
	Eigen::VectorXd& eta,
	Eigen::VectorXd& Y){

	double A, tot;

	for (int ii = 0; ii < K; ii++){
		Eigen::VectorXd X_kk = Eigen::VectorXd::Random(N);
		A  = (Y - ym - yx.cwiseProduct(eta)).dot(X_kk);
		ym += yx;
		tot += A;
	}
}

void single_core_scheme_sse(
	Eigen::VectorXd& ym,
	Eigen::VectorXd& yx,
	Eigen::VectorXd& eta,
	Eigen::VectorXd& Y){

	double A, tot;

	for (int ii = 0; ii < K; ii++){
		Eigen::VectorXf X_kk = Eigen::VectorXf::Random(N);
		A  = (Y.cast<float>() - ym.cast<float>() - yx.cast<float>().cwiseProduct(eta.cast<float>())).dot(X_kk);
		ym += yx;
		tot += A;
	}
}

int main() {
	// Single core version
	Eigen::MatrixXd X = Eigen::MatrixXd::Random(N, K);
	Eigen::VectorXd D = Eigen::VectorXd::Random(N);

	// int n = Eigen::nbThreads( );
	std::cout << "Data initialised" << std::endl;
	double res2 = 0;

	// No diagonal
	Eigen::MatrixXd res(K, K);
	for (int th = 1; th < 17; th++){
		Eigen::setNbThreads(th);
		std::cout << th << " ";
		// MyTimer tt1("10 repeats takes: %ts \n");
		for (int ss = 0; ss < 20; ss++){
			auto start = std::chrono::system_clock::now();
			for (int kk = 0; kk < 5; kk++){
				res = X.transpose() * X;
				res2 += res(0, 0);
			}
			auto end = std::chrono::system_clock::now();
			auto elapsed = end - start;
			std::cout << elapsed.count() << " ";
			// tt1.stop();
			// std::cout << tt1.get_lap_seconds() << " ";
			// tt1.resume();
		}
		std::cout << std::endl;
	}

	// With diagonal
	for (int th = 1; th < 17; th++){
		Eigen::setNbThreads(th);
		std::cout << th << " ";
		// MyTimer tt2("10 repeats takes: %ts \n");
		for (int ss = 0; ss < 20; ss++){
			auto start = std::chrono::system_clock::now();
			for (int kk = 0; kk < 5; kk++){
				res = X.transpose() * D.asDiagonal() * X;
				res2 += res(0, 0);
			}
			auto end = std::chrono::system_clock::now();
			auto elapsed = end - start;
			std::cout << elapsed.count() << " ";
			// tt2.stop();
			// std::cout << tt2.get_lap_seconds() << " ";
			// tt2.resume();
		}
		std::cout << std::endl;
	}

	return 0;
}
