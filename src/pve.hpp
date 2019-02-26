//
// Created by kerin on 2019-01-08.
//

#ifndef BGEN_PROG_PVE_HPP
#define BGEN_PROG_PVE_HPP

#include "genotype_matrix.hpp"
#include "utils.hpp"
#include "file_streaming.hpp"
#include "parameters.hpp"

#include <boost/iostreams/filtering_stream.hpp>

namespace boost_io = boost::iostreams;

class PVE {
public:
	// constants
	long B;
	long n_samples; // number of samples
	long n_var;
	const bool mode_gxe;
	const int n_components;
	int n_covar;
	bool mog_beta, mog_gam;

	parameters params;

	GenotypeMatrix& X;

	Eigen::VectorXd eta;
	Eigen::VectorXd& Y;
	Eigen::MatrixXd& C;
	Eigen::VectorXd sigmas, h2;
	Eigen::DiagonalMatrix<double, Eigen::Dynamic> U, V;
	Eigen::VectorXd uu, vv;
	double usum, vsum;

	std::vector<std::string> components;

	// Interim results
	boost_io::filtering_ostream outf;

	PVE(const parameters& myparams,
			GenotypeMatrix& myX,
			Eigen::VectorXd& myY,
			Eigen::MatrixXd& myC,
			Eigen::VectorXd& myeta) : params(myparams), X(myX), eta(myeta), Y(myY), C(myC), mode_gxe(true), n_components(3) {
		n_samples = X.nn;
		n_var = X.pp;
		B = params.n_pve_samples;
		std::vector<std::string> my_components = {"G", "GxE", "noise"};
		components = my_components;
		mog_beta = false;
		mog_gam = false;

		n_covar = C.cols();
		std::cout << "N-covars: " << n_covar << std::endl;
	}

	PVE(const parameters& myparams,
			GenotypeMatrix& myX,
			Eigen::VectorXd& myY,
			Eigen::MatrixXd& myC) : params(myparams), X(myX), Y(myY), C(myC), mode_gxe(false), n_components(2) {
		n_samples = X.nn;
		n_var = X.pp;
		B = params.n_pve_samples;
		std::vector<std::string> my_components = {"G", "noise"};
		components = my_components;
		mog_beta = false;
		mog_gam = false;

		n_covar = C.cols();
		std::cout << "N-covars: " << n_covar << std::endl;
	}

	void run(const std::string& file){
		// Filepath to write interim results to
		init_interim_results(file);
		if(mode_gxe){
			std::cout << "G+GxE effects model (gaussian prior)" << std::endl;
			he_reg_gxe();
		} else if(mog_beta){
			std::cout << "Main effects model (MoG prior)" << std::endl;
			he_reg_single_component_mog();
		} else {
			std::cout << "Main effects model (gaussian prior)" << std::endl;
			he_reg_single_component();
		}
		boost_io:close(outf);

		std::cout << "Variance components estimates" << std::endl;
		std::cout << sigmas << std::endl;

		h2 = sigmas / sigmas.sum();
		std::cout << "PVE estimates" << std::endl;
		std::cout << h2 << std::endl;
	}

	void set_mog_weights(Eigen::VectorXd weights_beta,
						 Eigen::VectorXd weights_gam){
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

	void fill_gaussian_noise(unsigned int seed,
							 Eigen::Ref<Eigen::MatrixXd> zz,
							 long nn,
							 long pp){
		assert(zz.rows() == nn);
		assert(zz.cols() == pp);

		std::mt19937 generator{seed};
		std::normal_distribution<scalarData> noise_normal(0.0, 1);

		for (int bb = 0; bb < pp; bb++){
			for (std::size_t ii = 0; ii < nn; ii++){
				zz(ii, bb) = noise_normal(generator);
			}
		}
	}

	void he_reg_single_component_mog(){
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
		for (int bb = 0; bb < B; bb++){
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

	void he_reg_single_component(){
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
		for (int bb = 0; bb < B; bb++){
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

	void he_reg_gxe(){
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
		for (int bb = 0; bb < B; bb++){
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

	void to_file(const std::string& file){
		boost_io::filtering_ostream outf;
		auto filename = fstream_init(outf, file, "", "_pve");

		std::cout << "Writing PVE results to " << filename << std::endl;
		outf << "component sigmas h2" << std::endl;

		for (int ii = 0; ii < n_components; ii++){
			outf << components[ii] << " ";
			outf << sigmas[ii] << " ";
			outf << h2[ii] << std::endl;
		}
		boost_io::close(outf);
	}

	void init_interim_results(const std::string& file){
		auto filename = fstream_init(outf, file, "pve_interim/", "");
		std::cout << "Writing interim results to " << filename << std::endl;

		if(mode_gxe){
			outf << "trK1 trK1K1 trV1 trK1V1 trV1V1" << std::endl;
		} else {
			outf << "trK1 trK1K1" << std::endl;
		}
	}

	void to_interim_results(double atrK1, double atrK1K1){
		assert(!mode_gxe);
		outf << atrK1 << " " << atrK1K1 << std::endl;
	}

	void to_interim_results(double atrK1, double atrK1K1, double atrV1, double atrK1V1, double atrV1V1){
		assert(mode_gxe);
		outf << atrK1 << " " << atrK1K1 << " " << atrV1 << " ";
		outf << atrK1V1 << " " << atrV1V1 << std::endl;
	}
};

#endif //BGEN_PROG_PVE_HPP
