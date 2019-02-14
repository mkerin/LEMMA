//
// Created by kerin on 2019-01-08.
//

#ifndef BGEN_PROG_PVE_HPP
#define BGEN_PROG_PVE_HPP

#include "genotype_matrix.hpp"
#include "utils.hpp"
#include "file_streaming.hpp"
#include "class.h"

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

	std::vector<std::string> components;

	parameters params;

	GenotypeMatrix& X;
	Eigen::VectorXd eta;
	Eigen::VectorXd& Y;
	Eigen::MatrixXd& C;

	Eigen::VectorXd sigmas, h2;

	// variance estimates
	double sigma, sigma_g, sigma_gxe;

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

		n_covar = C.cols();
		std::cout << "N-covars: " << n_covar << std::endl;
	}

	void run(const std::string& file){
		// Filepath to write interim results to
		init_interim_results(file);
		if(!mode_gxe){
			he_reg_single_component();
		} else {
			he_reg_gxe();
		}
		boost_io:close(outf);

		std::cout << "HE Regression estimates" << std::endl;
		std::cout << sigmas << std::endl;

		h2 = sigmas / sigmas.sum();
		std::cout << "PVE estimates" << std::endl;
		std::cout << h2 << std::endl;
	}

	void he_reg_single_component(){
		// Compute matrix-vector calculations to store
		double P = n_var, N = n_samples;
		EigenDataVector zz(n_samples), ezz(n_samples);
		std::mt19937 generator{params.random_seed};
		std::normal_distribution<scalarData> noise_normal(0.0, 1);

		// Compute trace computations
		double ytK1y = X.transpose_multiply(Y).squaredNorm() / P;
		double yty = Y.squaredNorm();

		Eigen::Vector2d bb;
		bb << ytK1y, yty;
 		std::cout << "bb: " << std::endl << bb << std::endl;

		// Randomised trace computations
		Eigen::ArrayXd atrK1(B), atrK1K1(B);
		for (int bb = 0; bb < B; bb++){
			// gen z
			for (std::size_t ii = 0; ii < n_samples; ii++){
				zz(ii) = noise_normal(generator);
			}
			auto Xtz = X.transpose_multiply(zz);
			auto XXtz = X * Xtz;

			atrK1(bb) = Xtz.squaredNorm() / P;
			atrK1K1(bb) = XXtz.squaredNorm() / P / P;

			to_interim_results(atrK1(bb), atrK1K1(bb));
		}

		// mean of traces
		double mtrK1   = atrK1.mean();
		double mtrK1K1 = atrK1K1.mean();

		// solve HE regression with two components
		Eigen::Matrix2d A;
		bb << ytK1y, yty;
		A << mtrK1K1, mtrK1,
			 mtrK1,   N;
		std::cout << "A: " << std::endl << A << std::endl;

		sigmas = A.colPivHouseholderQr().solve(bb);
	}

	void he_reg_gxe(){
		// Compute matrix-vector calculations to store
		double P = n_var, N = n_samples;
		EigenDataVector zz(n_samples), ezz(n_samples);
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
			// gen z
			for (std::size_t ii = 0; ii < n_samples; ii++){
				zz(ii) = noise_normal(generator);
			}
			ezz = zz.cwiseProduct(eta);
			auto Xtz = X.transpose_multiply(zz);
			auto Xtez = X.transpose_multiply(ezz);
			auto XXtz = X * Xtz;
			auto XXtez = X * Xtez;

			atrK1(bb) = Xtz.squaredNorm() / P;
			atrK1K1(bb) = XXtz.squaredNorm() / P / P;
			atrK2(bb) = Xtez.squaredNorm() / P;
			atrK1K2(bb) = (XXtez.cwiseProduct(eta).transpose() *XXtz)(0,0) / P / P;
			atrK2K2(bb) = XXtez.cwiseProduct(eta).squaredNorm() / P / P;

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

	void to_file(const std::string& file){
		boost_io::filtering_ostream outf;
		auto filename = fstream_init(outf, file, "", "");

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
			outf << "trK1 trK1K1 trK2 trK1K2 trK2K2" << std::endl;
		} else {
			outf << "trK1 trK1K1" << std::endl;
		}
	}

	void to_interim_results(double atrK1, double atrK1K1){
		assert(!mode_gxe);
		outf << atrK1 << " " << atrK1K1 << std::endl;
	}

	void to_interim_results(double atrK1, double atrK1K1, double atrK2, double atrK1K2, double atrK2K2){
		assert(mode_gxe);
		outf << atrK1 << " " << atrK1K1 << " " << atrK2 << " ";
		outf << atrK1K2 << " " << atrK2K2 << std::endl;
	}
};

#endif //BGEN_PROG_PVE_HPP
