//
// Created by kerin on 2019-02-26.
//
#include "pve.hpp"

void PVE::he_reg_gxe(){
	// Compute matrix-vector calculations to store
	double P = n_var, N = n_samples;
	EigenDataMatrix zz(n_samples, B), ezz;
	// EigenDataVector zz(n_samples), ezz(n_samples);
	std::mt19937 generator{params.random_seed};
	std::normal_distribution<scalarData> noise_normal(0.0, 1);

	// Compute trace computations
	Eigen::VectorXd eY = Y.cwiseProduct(eta);
	double ytK1y = X.transpose_multiply(Y).squaredNorm() / P;
	double yty = Y.squaredNorm();
	double ytK2y = X.transpose_multiply(eY).squaredNorm() / P;

	Eigen::Vector3d bb;
	bb << ytK1y, ytK2y, yty;
	std::cout << "bb: " << std::endl << bb << std::endl;

	// Randomised trace computations
	Eigen::ArrayXd atrK1(B), atrK2(B), atrK1K1(B), atrK1K2(B), atrK2K2(B);
	for (int bb = 0; bb < B; bb++){
		for (std::size_t ii = 0; ii < n_samples; ii++){
			zz(ii, bb) = noise_normal(generator);
		}
	}
	ezz = (zz.array().colwise() * eta.array()).matrix();
	auto Xtz = X.transpose_multiply(zz);
	auto Xtez = X.transpose_multiply(ezz);
	auto XXtz = X * Xtz;
	auto XXtez = X * Xtez;

	for (int bb = 0; bb < B; bb++){
		atrK1(bb) = Xtz.col(bb).squaredNorm() / P;
		atrK1K1(bb) = XXtz.col(bb).squaredNorm() / P / P;
		atrK2(bb) = Xtez.col(bb).squaredNorm() / P;
		atrK1K2(bb) = (XXtez.col(bb).cwiseProduct(eta).transpose() * XXtz.col(bb))(0,0) / P / P;
		atrK2K2(bb) = XXtez.col(bb).cwiseProduct(eta).squaredNorm() / P / P;

		to_interim_results(atrK1(bb), atrK1K1(bb), atrK2(bb), atrK1K2(bb), atrK2K2(bb));
	}

	// // mean of traces
	double mtrK1   = atrK1.mean();
	double mtrK2   = atrK2.mean();
	double mtrK1K1 = atrK1K1.mean();
	double mtrK1K2 = atrK1K2.mean();
	double mtrK2K2 = atrK2K2.mean();

	// solve HE regression with two components
	Eigen::Matrix3d A;
	A << mtrK1K1, mtrK1K2, mtrK1,
			mtrK1K2, mtrK2K2, mtrK2,
			mtrK1,   mtrK2,   N;

	std::cout << "A: " << std::endl << A << std::endl;

	sigmas = A.colPivHouseholderQr().solve(bb);
}

