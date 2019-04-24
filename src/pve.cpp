//
// Created by kerin on 2019-02-26.
//
#include "pve.hpp"
#include "genotype_matrix.hpp"
#include "typedefs.hpp"
#include "file_streaming.hpp"
#include "parameters.hpp"

#include "tools/Eigen/Dense"

#include <random>


void PVE::set_mog_weights(Eigen::VectorXd weights_beta, Eigen::VectorXd weights_gam) {
	assert(weights_beta.rows() == n_var);
	assert(weights_gam.rows() == n_var);
	mog_beta = true;
	mog_gam = true;
	U.diagonal() = weights_beta;
	V.diagonal() = weights_gam;
	uu = weights_beta;
	vv = weights_gam;
	usum = weights_beta.sum();
	vsum = weights_gam.sum();
}

void PVE::fill_gaussian_noise(unsigned int seed, Eigen::Ref<Eigen::MatrixXd> zz, long nn, long pp) {
	assert(zz.rows() == nn);
	assert(zz.cols() == pp);

	std::mt19937 generator{seed};
	std::normal_distribution<scalarData> noise_normal(0.0, 1);

	for (int bb = 0; bb < pp; bb++) {
		for (std::size_t ii = 0; ii < nn; ii++) {
			zz(ii, bb) = noise_normal(generator);
		}
	}
}

void PVE::he_reg_single_component_mog() {
	// Compute matrix-vector calculations to store
	double P = n_var, N = n_samples;
	// Eigen::DiagonalMatrix<double, Eigen::Dynamic> Uinv;
	// Uinv = Eigen::MatrixXd::Identity(n_samples, n_samples) - U;
	Eigen::VectorXd uuinv = (1 - uu.array()).matrix();
	EigenDataMatrix zz(n_samples, B);

	// Compute trace computations
	Eigen::Vector3d bb;
	double ytK1y = (U * X.transpose_multiply(Y)).squaredNorm() / usum;
	double ytK2y = (uuinv.asDiagonal() * X.transpose_multiply(Y)).squaredNorm() / (P - usum);
	double yty = Y.squaredNorm();
	bb << ytK1y, ytK2y, yty;
	std::cout << "bb: " << std::endl << bb << std::endl;

	// Randomised trace computations
	fill_gaussian_noise(params.random_seed, zz, n_samples, B);
	auto Xtz = X.transpose_multiply(zz);
	Eigen::MatrixXd UUXtz = uu.cwiseProduct(uu).asDiagonal() * Xtz;
	Eigen::MatrixXd UinvUinvXtz = uuinv.cwiseProduct(uuinv).asDiagonal() * Xtz;
	auto XuXtz = X * UUXtz;
	auto XuinvXtz = X * UinvUinvXtz;

	// mean of traces
	Eigen::ArrayXd atrK1(B), atrK1K1(B), atrK2(B), atrK2K2(B), atrK1K2(B);
	for (int bb = 0; bb < B; bb++) {
		atrK1(bb) = (U * Xtz.col(bb)).squaredNorm() / usum;
		atrK1K1(bb) = XuXtz.col(bb).squaredNorm() / usum / usum;
		atrK2(bb) = (uuinv.asDiagonal() * Xtz.col(bb)).squaredNorm() / (P - usum);
		atrK2K2(bb) = XuinvXtz.col(bb).squaredNorm() / (P - usum) / (P - usum);
		atrK1K2(bb) = (XuinvXtz.col(bb).transpose() * XuXtz.col(bb))(0,0) / usum / (P - usum);
		// to_interim_results(atrK1(bb), atrK1K1(bb));
	}
	double mtrK1   = atrK1.mean();
	double mtrK1K1 = atrK1K1.mean();
	double mtrK2   = atrK2.mean();
	double mtrK2K2 = atrK2K2.mean();
	double mtrK1K2 = atrK1K2.mean();

	// solve HE regression with two components
	Eigen::Matrix3d A;
	A << mtrK1K1, mtrK1K2, mtrK1,
	    mtrK1K2, mtrK2K2, mtrK2,
	    mtrK1,   mtrK2,   N;
	std::cout << "A: " << std::endl << A << std::endl;
	sigmas = A.colPivHouseholderQr().solve(bb);
}

void PVE::he_reg_single_component() {
	// Compute matrix-vector calculations to store
	double P = n_var, N = n_samples;
	EigenDataMatrix zz(n_samples, B);

	// Compute trace computations
	Eigen::Vector2d bb;
	double ytK1y = X.transpose_multiply(Y).squaredNorm() / P;
	double yty = Y.squaredNorm();
	bb << ytK1y, yty;
	std::cout << "bb: " << std::endl << bb << std::endl;

	// Randomised trace computations
	fill_gaussian_noise(params.random_seed, zz, n_samples, B);
	auto Xtz = X.transpose_multiply(zz);
	auto XXtz = X * Xtz;

	// mean of traces
	Eigen::ArrayXd atrK1(B), atrK1K1(B);
	for (int bb = 0; bb < B; bb++) {
		atrK1(bb) = Xtz.col(bb).squaredNorm() / P;
		atrK1K1(bb) = XXtz.col(bb).squaredNorm() / P / P;
		to_interim_results(atrK1(bb), atrK1K1(bb));
	}
	double mtrK1   = atrK1.mean();
	double mtrK1K1 = atrK1K1.mean();

	// solve HE regression with two components
	Eigen::Matrix2d A;
	A << mtrK1K1, mtrK1,
	    mtrK1,   N;
	std::cout << "A: " << std::endl << A << std::endl;
	sigmas = A.colPivHouseholderQr().solve(bb);
}

void PVE::he_reg_gxe() {
	// Compute matrix-vector calculations to store
	double P = n_var, N = n_samples;
	EigenDataMatrix zz(n_samples, B), ezz;

	// Compute trace computations
	Eigen::VectorXd eY = Y.cwiseProduct(eta);
	double ytK1y = X.transpose_multiply(Y).squaredNorm() / P;
	double yty = Y.squaredNorm();
	double ytV1y = X.transpose_multiply(eY).squaredNorm() / P;

	Eigen::Vector3d bb;
	bb << ytK1y, ytV1y, yty;
	std::cout << "bb: " << std::endl << bb << std::endl;

	// Randomised trace computations
	fill_gaussian_noise(params.random_seed, zz, n_samples, B);
	ezz = (zz.array().colwise() * eta.array()).matrix();
	auto Xtz = X.transpose_multiply(zz);
	auto Xtez = X.transpose_multiply(ezz);
	auto XXtz = X * Xtz;
	auto XXtez = X * Xtez;

	// // mean of traces
	Eigen::ArrayXd atrK1(B), atrK1K1(B), atrV1(B), atrV1V1(B), atrK1V1(B);
	for (int bb = 0; bb < B; bb++) {
		atrK1(bb) = Xtz.col(bb).squaredNorm() / P;
		atrK1K1(bb) = XXtz.col(bb).squaredNorm() / P / P;
		atrV1(bb) = Xtez.col(bb).squaredNorm() / P;
		atrK1V1(bb) = (XXtez.col(bb).cwiseProduct(eta).transpose() * XXtz.col(bb))(0,0) / P / P;
		atrV1V1(bb) = XXtez.col(bb).cwiseProduct(eta).squaredNorm() / P / P;

		to_interim_results(atrK1(bb), atrK1K1(bb), atrV1(bb), atrK1V1(bb), atrV1V1(bb));
	}
	double mtrK1   = atrK1.mean();
	double mtrV1   = atrV1.mean();
	double mtrK1K1 = atrK1K1.mean();
	double mtrK1V1 = atrK1V1.mean();
	double mtrV1V1 = atrV1V1.mean();

	// solve HE regression with two components
	Eigen::Matrix3d A;
	A << mtrK1K1, mtrK1V1, mtrK1,
	    mtrK1V1, mtrV1V1, mtrV1,
	    mtrK1,   mtrV1,   N;

	std::cout << "A: " << std::endl << A << std::endl;

	sigmas = A.colPivHouseholderQr().solve(bb);
}

void PVE::run(const std::string &file) {
	// Filepath to write interim results to
	init_interim_results(file);
	if(mode_gxe) {
		std::cout << "G+GxE effects model (gaussian prior)" << std::endl;
		he_reg_gxe();
	} else if(mog_beta) {
		std::cout << "Main effects model (MoG prior)" << std::endl;
		he_reg_single_component_mog();
	} else {
		std::cout << "Main effects model (gaussian prior)" << std::endl;
		he_reg_single_component();
	}
boost_io: close(outf);

	std::cout << "Variance components estimates" << std::endl;
	std::cout << sigmas << std::endl;

	h2 = sigmas / sigmas.sum();
	std::cout << "PVE estimates" << std::endl;
	std::cout << h2 << std::endl;
}
