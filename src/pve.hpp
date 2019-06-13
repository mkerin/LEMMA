//
// Created by kerin on 2019-01-08.
//

#ifndef BGEN_PROG_PVE_HPP
#define BGEN_PROG_PVE_HPP

#include "genotype_matrix.hpp"
#include "file_utils.hpp"
#include "parameters.hpp"
#include "eigen_utils.hpp"
#include "mpi_utils.hpp"

#include <boost/iostreams/filtering_stream.hpp>

#include <random>

namespace boost_io = boost::iostreams;

class PVE {
public:
	// constants
	long nDraws;
	long n_samples;                                             // number of samples
	long n_var;
	const bool mode_gxe;
	const int n_components;
	long n_covar;
	long n_env;
	bool mog_beta, mog_gam;
	double N, P;
	int world_rank;

	parameters params;

	GenotypeMatrix& X;

	Eigen::VectorXd eta;
	Eigen::VectorXd Y;
	Eigen::MatrixXd& C;
	Eigen::VectorXd sigmas, h2;
	Eigen::VectorXd uu, vv;
	Eigen::MatrixXd CtC_inv;
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
		N = X.nn;
		P = X.pp;
		nDraws = params.n_pve_samples;
		std::vector<std::string> my_components = {"G", "GxE", "noise"};
		components = my_components;
		mog_beta = false;
		mog_gam = false;

		n_covar = C.cols();
		n_env = 1;
		std::cout << "N-covars: " << n_covar << std::endl;

		// Center and scale eta
		EigenUtils::center_matrix(eta);
		EigenUtils::scale_matrix_and_remove_constant_cols(eta);

		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	}

	PVE(const parameters& myparams,
	    GenotypeMatrix& myX,
	    Eigen::VectorXd& myY,
	    Eigen::MatrixXd& myC) : params(myparams), X(myX), Y(myY), C(myC), mode_gxe(false), n_components(2) {
		n_samples = X.nn;
		n_var = X.pp;
		N = X.nn;
		P = X.pp;
		nDraws = params.n_pve_samples;
		std::vector<std::string> my_components = {"G", "noise"};
		components = my_components;
		mog_beta = false;
		mog_gam = false;

		n_covar = C.cols();
		n_env = 0;
		std::cout << "N-covars: " << n_covar << std::endl;

		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	}

	void calc_sigmas(){
		// New general implementation
		struct indexes_t {
			indexes_t() : main(0), noise(1) {
			}
			long main, main2, gxe, gxe2, noise;
		} ind;
		long n_components = 2;
		if(n_env == 1) {
			n_components += 1;
			ind.gxe = ind.noise;
			ind.noise += 1;
		}
		if(mog_beta) {
			n_components += 1;
			ind.main2 = ind.gxe;
			ind.gxe += 1;
			ind.noise += 1;
		}
		if(mog_gam) {
			n_components += 1;
			ind.gxe2 = ind.noise;
			ind.noise += 1;
		}

		/*** trace computations for RHS ***/
		long index = 0;
		Y = project_out_covars(Y);
		Eigen::VectorXd eY, bb(n_components);

		bb(ind.main) = X.transpose_multiply(Y).squaredNorm() / P;
		bb(ind.noise) = Y.squaredNorm();
		if(n_env == 1) {
			eY = Y.cwiseProduct(eta);
			bb(ind.gxe) = X.transpose_multiply(eY).squaredNorm() / P;
		}

		bb = mpiUtils::mpiReduce_inplace(bb);
		std::cout << "b: " << std::endl << bb << std::endl;

		/*** trace computations for LHS ***/
		Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_components, n_components);
		Eigen::VectorXd zz(n_samples);
		std::mt19937 generator{params.random_seed};
		std::normal_distribution<scalarData> noise_normal(0.0, 1);
		for (int rr = 0; rr < nDraws; rr++) {

			// fill gaussian noise
			for (long ii = 0; ii < n_samples; ii++) {
				zz(ii) = noise_normal(generator);
			}
			Eigen::VectorXd Wzz = project_out_covars(zz);

			Eigen::VectorXd Xtz = X.transpose_multiply(zz);
			Xtz = mpiUtils::mpiReduce_inplace(Xtz);
			Eigen::VectorXd XXtz = X * Xtz;
			Eigen::VectorXd WXXtz = project_out_covars(XXtz);
			Eigen::VectorXd XtWz = X.transpose_multiply(Wzz);
			XtWz = mpiUtils::mpiReduce_inplace(XtWz);
			Eigen::VectorXd XXtWz = X * XtWz;
			A(ind.main, ind.main) += mpiUtils::mpiReduce_inplace(WXXtz.dot(XXtWz)) / P / P;
			A(ind.main, ind.noise) += mpiUtils::mpiReduce_inplace(XXtz.dot(Wzz)) / P;

			if(n_env == 1) {
				Eigen::VectorXd ezz = eta.cwiseProduct(zz);
				Eigen::VectorXd eWzz = eta.cwiseProduct(Wzz);

				Eigen::VectorXd Xtez = X.transpose_multiply(ezz);
				Xtez = mpiUtils::mpiReduce_inplace(Xtez);
				Eigen::VectorXd XXtez = X * Xtez;
				Eigen::VectorXd eXXtez = eta.cwiseProduct(XXtez);
				Eigen::VectorXd WeXXtez = project_out_covars(eXXtez);
				Eigen::VectorXd XteWz = X.transpose_multiply(eWzz);
				XteWz = mpiUtils::mpiReduce_inplace(XteWz);
				Eigen::VectorXd XXteWz = X * XteWz;
				Eigen::VectorXd eXXteWz = eta.cwiseProduct(XXteWz);
				A(ind.gxe, ind.gxe) += mpiUtils::mpiReduce_inplace(eXXteWz.dot(WeXXtez)) / P / P;
				A(ind.main, ind.gxe) += mpiUtils::mpiReduce_inplace(WeXXtez.dot(XXtWz)) / P / P;
				A(ind.gxe, ind.noise) += mpiUtils::mpiReduce_inplace(XXteWz.dot(ezz)) / P;
			}

			A(ind.noise, ind.noise) += mpiUtils::mpiReduce_inplace(N) - n_covar;
		}
		A.array() /= nDraws;
//		A = mpiUtils::mpiReduce_inplace(A);
//		A(ind.noise, ind.noise) -= n_covar;
		A = A.selfadjointView<Eigen::Upper>();
		std::cout << "A: " << std::endl << A << std::endl;

		sigmas = A.colPivHouseholderQr().solve(bb);
	}

	Eigen::VectorXd calc_h2(){
		// convert sigmas to h2
		// Trivial if genotype matrices have colmean zero and colvariance one
		return sigmas / sigmas.sum();
	}

	Eigen::MatrixXd project_out_covars(Eigen::Ref<Eigen::MatrixXd> rhs){
		if(n_covar > 0) {
			if (CtC_inv.rows() != n_covar) {
				Eigen::MatrixXd CtC = C.transpose() * C;
				CtC = mpiUtils::mpiReduce_inplace(CtC);
				CtC_inv = CtC.inverse();
			}
			Eigen::MatrixXd res, yhat;
			Eigen::MatrixXd CtRHS = C.transpose() * rhs;
			CtRHS = mpiUtils::mpiReduce_inplace(CtRHS);
			yhat = C * CtC_inv * CtRHS;
			res = rhs - yhat;
			return res;
		} else {
			return rhs;
		}
	}

	void run(const std::string& file);

	void set_mog_weights(Eigen::VectorXd weights_beta,
	                     Eigen::VectorXd weights_gam);

	void fill_gaussian_noise(unsigned int seed,
	                         Eigen::Ref<Eigen::MatrixXd> zz,
	                         long nn,
	                         long n_repeats);

	void he_reg_single_component_mog();

	void to_file(const std::string& file){
		boost_io::filtering_ostream outf;
		auto filename = fileUtils::fstream_init(outf, file, "", "_pve");

		std::cout << "Writing PVE results to " << filename << std::endl;
		outf << "component sigmas h2" << std::endl;

		for (int ii = 0; ii < n_components; ii++) {
			outf << components[ii] << " ";
			outf << sigmas[ii] << " ";
			outf << h2[ii] << std::endl;
		}
		boost_io::close(outf);
	}

	void init_interim_results(const std::string& file){
		auto filename = fileUtils::fstream_init(outf, file, "pve_interim/", "");
		std::cout << "Writing interim results to " << filename << std::endl;

		if(mode_gxe) {
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

void PVE::set_mog_weights(Eigen::VectorXd weights_beta, Eigen::VectorXd weights_gam) {
	assert(weights_beta.rows() == n_var);
	assert(weights_gam.rows() == n_var);
	mog_beta = true;
	mog_gam = true;
	uu = weights_beta;
	vv = weights_gam;
	usum = weights_beta.sum();
	vsum = weights_gam.sum();

	if(mode_gxe) {
		std::vector<std::string> my_components = {"G1", "G2", "GxE1", "GxE2", "noise"};
		components = my_components;
	} else {
		std::vector<std::string> my_components = {"G1", "G2", "noise"};
		components = my_components;
	}
}

void PVE::fill_gaussian_noise(unsigned int seed, Eigen::Ref<Eigen::MatrixXd> zz, long nn, long n_repeats) {
	assert(zz.rows() == nn);
	assert(zz.cols() == n_repeats);

	std::mt19937 generator{seed};
	std::normal_distribution<scalarData> noise_normal(0.0, 1);

	for (int bb = 0; bb < n_repeats; bb++) {
		for (std::size_t ii = 0; ii < nn; ii++) {
			zz(ii, bb) = noise_normal(generator);
		}
	}
}

void PVE::he_reg_single_component_mog() {
	// Compute matrix-vector calculations to store
	double P = n_var, N = n_samples;
	Eigen::VectorXd uuinv = (1 - uu.array()).matrix();
	EigenDataMatrix zz(n_samples, nDraws);

	// Compute trace computations
	Eigen::Vector3d bb;
	double ytK1y = (uu.asDiagonal() * X.transpose_multiply(Y)).squaredNorm() / usum;
	double ytK2y = (uuinv.asDiagonal() * X.transpose_multiply(Y)).squaredNorm() / (P - usum);
	double yty = Y.squaredNorm();
	bb << ytK1y, ytK2y, yty;
	std::cout << "bb: " << std::endl << bb << std::endl;

	// Randomised trace computations
	fill_gaussian_noise(params.random_seed, zz, n_samples, nDraws);
	auto Xtz = X.transpose_multiply(zz);
	Eigen::MatrixXd UUXtz = uu.cwiseProduct(uu).asDiagonal() * Xtz;
	Eigen::MatrixXd UinvUinvXtz = uuinv.cwiseProduct(uuinv).asDiagonal() * Xtz;
	auto XuXtz = X * UUXtz;
	auto XuinvXtz = X * UinvUinvXtz;

	// mean of traces
	Eigen::ArrayXd atrK1(nDraws), atrK1K1(nDraws), atrK2(nDraws), atrK2K2(nDraws), atrK1K2(nDraws);
	for (int bb = 0; bb < nDraws; bb++) {
		atrK1(bb) = (uu.asDiagonal() * Xtz.col(bb)).squaredNorm() / usum;
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

void PVE::run(const std::string &file) {
	// Filepath to write interim results to
	if(world_rank == 0) {
		init_interim_results(file);
	}

	if(mode_gxe) {
		std::cout << "G+GxE effects model (gaussian prior)" << std::endl;
		calc_sigmas();
	} else if(mog_beta) {
		std::cout << "Main effects model (MoG prior)" << std::endl;
		he_reg_single_component_mog();
	} else {
		std::cout << "Main effects model (gaussian prior)" << std::endl;
		calc_sigmas();
	}

	if(world_rank == 0) {
		boost_io::close(outf);
	}

	std::cout << "Variance components estimates" << std::endl;
	std::cout << sigmas << std::endl;

	h2 = calc_h2();
	std::cout << "PVE estimates" << std::endl;
	std::cout << h2 << std::endl;
}

#endif //BGEN_PROG_PVE_HPP
