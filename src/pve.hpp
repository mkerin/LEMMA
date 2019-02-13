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

	parameters params;

	GenotypeMatrix& X;
	Eigen::VectorXd eta;
	Eigen::VectorXd& Y;

	Eigen::VectorXd sigmas, h2;

	// variance estimates
	double sigma, sigma_g, sigma_gxe;

	PVE( const parameters& myparams,
			GenotypeMatrix& myX,
			Eigen::VectorXd& myeta,
			Eigen::VectorXd& myY ) : params(myparams), X(myX), eta(myeta), Y(myY) {
		n_samples = X.nn;
		n_var = X.pp;
		B = 200;
	}

	PVE( const parameters& myparams,
			GenotypeMatrix& myX,
			Eigen::VectorXd& myY ) : params(myparams), X(myX), Y(myY) {
		n_samples = X.nn;
		n_var = X.pp;
		B = 200;
	}

	void run(){

		// Compute matrix-vector calculations to store
		EigenDataVector zz(n_samples), ezz(n_samples);
		std::mt19937 generator{params.random_seed};
		std::normal_distribution<scalarData> noise_normal(0.0, 1);

		// Randomised trace computations
		double P = n_var, N = n_samples;
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
		}

		// Compute trace computations
		double ytK1y = X.transpose_multiply(Y).squaredNorm() / P;
		double yty = Y.dot(Y);

		// mean of traces
		double mtrK1   = atrK1.mean();
		double mtrK1K1 = atrK1K1.mean();

		// solve HE regression with two components
		Eigen::Vector2d bb;
		Eigen::Matrix2d A;
 		bb << ytK1y, yty;
		A << mtrK1K1, mtrK1,
			 mtrK1,   N;

		sigmas = A.colPivHouseholderQr().solve(bb);
		std::cout << "HE Regression estimates" << std::endl;
		std::cout << sigmas << std::endl;

		h2 = sigmas / sigmas.sum();
		std::cout << "PVE estimates" << std::endl;
		std::cout << h2 << std::endl;
	}

	void to_file(const std::string& file){
		boost_io::filtering_ostream outf;
		auto filename = fstream_init(outf, file, "", "");

		std::cout << "Writing PVE results to " << filename << std::endl;
		std::vector<std::string> components = {"G", "noise"};
		outf << "component sigmas h2" << std::endl;

		for (int ii = 0; ii < 2; ii++){
			std::cout << components[ii] << std::endl;
			outf << components[ii] << " ";
			outf << sigmas[ii] << " ";
			outf << h2[ii] << std::endl;
		}
		boost_io::close(outf);
	}
};


		// // Compute matrix-vector calculations to store
		// EigenDataVector zz(n_samples), ezz(n_samples);
		// std::mt19937 generator{params.random_seed};
		// std::normal_distribution<scalarData> noise_normal(0.0, 1);
		//
		// // Randomised trace computations
		// double P = n_var, N = n_samples;
		// Eigen::ArrayXd atrK1(B), atrK2(B), atrK1K1(B), atrK1K2(B), atrK2K2(B);
		// for (int bb = 0; bb < B; bb++){
		// 	// gen z
		// 	for (std::size_t ii = 0; ii < n_samples; ii++){
		// 		zz(ii) = noise_normal(generator);
		// 	}
		// 	ezz = zz.cwiseProduct(eta);
		// 	auto Xtz = X.transpose_multiply(zz);
		// 	auto Xtez = X.transpose_multiply(ezz);
		// 	auto XXtz = X * Xtz;
		// 	auto XXtez = X * Xtez;
		//
		// 	atrK1(bb) = Xtz.squaredNorm() / P;
		// 	atrK2(bb) = Xtez.squaredNorm() / P;
		// 	atrK1K1(bb) = XXtz.squaredNorm() / P / P;
		// 	atrK2K2(bb) = XXtez.cwiseProduct(eta).squaredNorm() / P / P;
		// 	atrK1K2(bb) = XXtez.squaredNorm() / P / P;
		// }
		//
		// // Compute trace computations
		// Eigen::VectorXd eY = Y.cwiseProduct(eta);
		// double ytK1y = X.transpose_multiply(Y).squaredNorm() / P;
		// double ytK2y = X.transpose_multiply(eY).squaredNorm() / P;
		// double yty = Y.dot(Y);
		//
		// // mean of traces
		// double mtrK1   = atrK1.mean();
		// double mtrK2   = atrK2.mean();
		// double mtrK1K1 = atrK1K1.mean();
		// double mtrK1K2 = atrK1K2.mean();
		// double mtrK2K2 = atrK2K2.mean();
		//
		// // solve HE regression with two components
		// Eigen::Vector3d bb;
		// Eigen::Matrix3d A;
 		// bb << ytK1y, ytK2y, yty;
		// A << mtrK1K1, mtrK1K2, mtrK1,
		// 	 mtrK1K2, mtrK2K2, mtrK2,
		// 	 mtrK1,   mtrK2,   N;
		//
		// Eigen::Vector3d xx = A.colPivHouseholderQr().solve(bb);
		// std::cout << "HE Regression estimates" << std::endl;
		// std::cout << xx << std::endl;


#endif //BGEN_PROG_PVE_HPP
