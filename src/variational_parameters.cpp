//
// Created by kerin on 2019-03-01.
//

#include "variational_parameters.hpp"
#include "hyps.hpp"
#include "parameters.hpp"
#include "tools/eigen3.3/Dense"
#include "mpi_utils.hpp"
#include "file_utils.hpp"

#include <iostream>
#include <limits>

void VariationalParamsBase::resize(std::int32_t n_samples, std::int32_t n_var, long n_covar, long n_env) {
	s1_beta_sq.resize(n_var);
	if(p.mode_mog_prior_beta) {
		s2_beta_sq.resize(n_var);
	}

	if(n_env > 0) {
		// For EdZtZ
		double eps = std::numeric_limits<double>::min();
		sw_sq.resize(n_env);
		sw_sq = eps;

		s1_gam_sq.resize(n_var);
		if (p.mode_mog_prior_gam) {
			s2_gam_sq.resize(n_var);
		}
	}

	// for covars
	if(n_covar > 0) {
		sc_sq.resize(n_covar);
	}
}

void VariationalParamsBase::run_default_init(long n_var, long n_covar, long n_env){
	/* - snp latent variables initialised from zero
	 * - covar latent variables initialised from zero
	 * - env latent variables initialised from uniform
	 */
	alpha_beta    = Eigen::ArrayXd::Zero(n_var);
	mu1_beta       = Eigen::ArrayXd::Zero(n_var);
	s1_beta_sq     = Eigen::ArrayXd::Zero(n_var);
	if(p.mode_mog_prior_beta) {
		mu2_beta   = Eigen::ArrayXd::Zero(n_var);
		s2_beta_sq = Eigen::ArrayXd::Zero(n_var);
	}

	if(n_env > 0) {
		alpha_gam = Eigen::ArrayXd::Zero(n_var);
		mu1_gam   = Eigen::ArrayXd::Zero(n_var);
		s1_gam_sq     = Eigen::ArrayXd::Zero(n_var);
		if (p.mode_mog_prior_gam) {
			mu2_gam   = Eigen::ArrayXd::Zero(n_var);
			s2_gam_sq = Eigen::ArrayXd::Zero(n_var);
		}
		double eps = std::numeric_limits<double>::min();
		muw = Eigen::ArrayXd::Constant(n_env, 1.0 / n_env);
		sw_sq = Eigen::ArrayXd::Constant(n_env, eps);
	}

	if(n_covar > 0) {
		muc   = Eigen::ArrayXd::Zero(n_covar);
		sc_sq = Eigen::ArrayXd::Zero(n_covar);
	}
}

void VariationalParamsBase::snps_to_file(const std::string& path,
										 const GenotypeMatrix &X, long n_env) const {

	boost_io::filtering_ostream outf;
	fileUtils::fstream_init(outf, path);

	outf << "SNPID alpha_beta mu1_beta s1_beta";
	if(p.mode_mog_prior_beta) outf << " mu2_beta s2_beta";
	if(n_env > 0) {
		outf << " alpha_gam mu1_gam s1_gam";
		if (p.mode_mog_prior_gam) outf << " mu2_gam s2_gam";
	}
	outf << std::endl;
	outf << std::scientific << std::setprecision(8);
	long n_var = alpha_beta.rows();
	for (long ii = 0; ii < n_var; ii++) {
		outf << X.SNPID[ii];
		outf << " " << alpha_beta(ii);
		outf << " " << mu1_beta(ii);
		outf << " " << s1_beta_sq(ii);
		if(p.mode_mog_prior_beta) {
			outf << " " << mu2_beta(ii);
			outf << " " << s2_beta_sq(ii);
		}
		if(n_env > 0) {
			outf << " " << alpha_gam(ii);
			outf << " " << mu1_gam(ii);
			outf << " " << s1_gam_sq(ii);
			if (p.mode_mog_prior_gam) {
				outf << " " << mu2_gam(ii);
				outf << " " << s2_gam_sq(ii);
			}
		}
		outf << std::endl;
	}

	boost_io::close(outf);
}

void VariationalParamsBase::env_to_file(const std::string& path,
		const std::vector<std::string>& env_names) const {
	boost_io::filtering_ostream outf;
	fileUtils::fstream_init(outf, path);

	long n_env = env_names.size();
	outf << std::scientific << std::setprecision(7);
	outf << "env mean variance" << std::endl;
	for (int cc = 0; cc < n_env; cc++) {
		outf << env_names[cc] << " ";
		outf << muw(cc) << " ";
		outf << sw_sq(cc) << std::endl;
	}

	boost_io::close(outf);
}

void VariationalParamsBase::covar_to_file(const std::string& path,
		const std::vector<std::string>& covar_names) const {
	boost_io::filtering_ostream outf;
	fileUtils::fstream_init(outf, path);

	long n_covar = covar_names.size();
	outf << std::scientific << std::setprecision(7);
	outf << "covar mean variance" << std::endl;
	for (int cc = 0; cc < n_covar; cc++) {
		outf << covar_names[cc] << " ";
		outf << muc(cc) << " ";
		outf << sc_sq(cc) << std::endl;
	}

	boost_io::close(outf);
}

Eigen::VectorXd VariationalParamsBase::mean_covars() const {
	return muc;
}

Eigen::VectorXd VariationalParamsBase::mean_beta() const {
	Eigen::VectorXd rr_beta;
	if(p.mode_mog_prior_beta) {
		rr_beta = alpha_beta * (mu1_beta - mu2_beta) + mu2_beta;
	} else {
		rr_beta = alpha_beta * mu1_beta;
	}
	return rr_beta;
}

Eigen::VectorXd VariationalParamsBase::mean_gam() const {
	Eigen::VectorXd rr_gam;
	if(p.mode_mog_prior_gam) {
		rr_gam = alpha_gam * (mu1_gam - mu2_gam) + mu2_gam;
	} else {
		rr_gam = alpha_gam * mu1_gam;
	}
	return rr_gam;
}

Eigen::VectorXd VariationalParamsBase::mean_weights() const {
	return muw;
}

double VariationalParamsBase::mean_beta(std::uint32_t jj) const {
	double rr_beta = alpha_beta(jj) * mu1_beta(jj);
	if(p.mode_mog_prior_beta) {
		rr_beta += (1.0 - alpha_beta(jj)) * mu2_beta(jj);
	}
	return rr_beta;
}

double VariationalParamsBase::mean_gam(std::uint32_t jj) const {
	double rr_gam = alpha_gam(jj) * mu1_gam(jj);
	if(p.mode_mog_prior_gam) {
		rr_gam += (1.0 - alpha_gam(jj)) * mu2_gam(jj);
	}
	return rr_gam;
}

Eigen::ArrayXd VariationalParamsBase::var_beta() const {
	Eigen::ArrayXd varB = alpha_beta * (s1_beta_sq + (1.0 - alpha_beta) * mu1_beta.square());
	if(p.mode_mog_prior_beta) {
		varB += (1.0 - alpha_beta) * (s2_beta_sq + (alpha_beta) * mu2_beta.square());
		varB -= 2.0 * alpha_beta * (1.0 - alpha_beta) * mu1_beta * mu2_beta;
	}
	return varB;
}

Eigen::ArrayXd VariationalParamsBase::var_gam() const {
	Eigen::ArrayXd varG = alpha_gam * (s1_gam_sq + (1.0 - alpha_gam) * mu1_gam.square());
	if (p.mode_mog_prior_gam) {
		varG += (1.0 - alpha_gam) * (s2_gam_sq + (alpha_gam) * mu2_gam.square());
		varG -= 2.0 * alpha_gam * (1.0 - alpha_gam) * mu1_gam * mu2_gam;
	}
	return varG;
}

Eigen::ArrayXd VariationalParamsBase::mean_beta_sq(int u0) const {
	// if u0 == 1; E[u \beta^2]
	// if u0 == 0; E[1 - u \beta^2]
	Eigen::ArrayXd res;
	assert(u0 == 1 || u0 == 2);
	if(u0 == 1)
		res = alpha_beta * (s1_beta_sq + mu1_beta.square());
	else if (u0 == 2 && p.mode_mog_prior_beta) {
		res = (1 - alpha_beta) * (s2_beta_sq + mu2_beta.square());
	}
	return res;
}

Eigen::ArrayXd VariationalParamsBase::mean_gam_sq(int u0) const {
	// if u0 == 1; E[u \beta^2]
	// if u0 == 0; E[1 - u \beta^2]
	Eigen::ArrayXd res;
	assert(u0 == 1 || u0 == 2);
	if(u0 == 1)
		res = alpha_gam * (s1_gam_sq + mu1_gam.square());
	else if (u0 == 2 && p.mode_mog_prior_gam) {
		res = (1 - alpha_gam) * (s2_gam_sq + mu2_gam.square());
	}
	return res;
}

void VariationalParameters::init_from_lite(const VariationalParametersLite &init) {
	// yx and ym set to point to appropriate col of VBayesX2::YM and VBayesX2::YX in constructor

	alpha_beta = init.alpha_beta;
	mu1_beta   = init.mu1_beta;
	mu2_beta   = init.mu2_beta;
	alpha_gam  = init.alpha_gam;
	mu1_gam    = init.mu1_gam;
	mu2_gam    = init.mu2_gam;
	s1_beta_sq = init.s1_beta_sq;
	s2_beta_sq = init.s2_beta_sq;
	s1_gam_sq = init.s1_gam_sq;
	s2_gam_sq = init.s2_gam_sq;

	muc   = init.muc;
	muw   = init.muw;
	sc_sq   = init.sc_sq;
	sw_sq   = init.sw_sq;
}

VariationalParametersLite VariationalParameters::convert_to_lite() const {
	VariationalParametersLite vplite(p);
	vplite.ym         = ym;
	vplite.yx         = yx;
	vplite.alpha_beta = alpha_beta;
	vplite.alpha_gam  = alpha_gam;
	vplite.mu1_beta   = mu1_beta;
	vplite.mu1_gam    = mu1_gam;
	vplite.mu2_beta   = mu2_beta;
	vplite.mu2_gam    = mu2_gam;
	vplite.s1_beta_sq = s1_beta_sq;
	vplite.s1_gam_sq  = s1_gam_sq;
	vplite.s2_beta_sq = s2_beta_sq;
	vplite.s2_gam_sq  = s2_gam_sq;

	vplite.muc   = muc;
	vplite.sc_sq   = sc_sq;
	vplite.muw   = muw;
	vplite.sw_sq   = sw_sq;
	vplite.eta   = eta;
	return vplite;
}

void VariationalParameters::calcEdZtZ(const Eigen::Ref<const Eigen::ArrayXXd> &dXtEEX_lowertri, const long &n_env) {
	Eigen::ArrayXd EdZtZlocal = Eigen::ArrayXd::Zero(alpha_beta.rows());

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	if(world_rank == 0) {
		Eigen::ArrayXd muw_sq_combos(n_env * (n_env + 1) / 2);
		for (int ll = 0; ll < n_env; ll++) {
			for (int mm = 0; mm < n_env; mm++) {
				muw_sq_combos(dXtEEX_col_ind(ll, mm, n_env)) = muw(mm) * muw(ll);
			}
		}

		EdZtZlocal = 2 * (dXtEEX_lowertri.rowwise() * muw_sq_combos.transpose()).rowwise().sum();
		if(n_env > 1) {
			for (int ll = 0; ll < n_env; ll++) {
				EdZtZlocal += dXtEEX_lowertri.col(dXtEEX_col_ind(ll, ll, n_env)) * sw_sq(ll);
				EdZtZlocal -= dXtEEX_lowertri.col(dXtEEX_col_ind(ll, ll, n_env)) * muw(ll) * muw(ll);
			}
		}
	}
	EdZtZ.resize(EdZtZlocal.rows(), EdZtZlocal.cols());
	mpiUtils::mpiReduce_double(EdZtZlocal.data(), EdZtZ.data(), EdZtZlocal.size());
}

long dXtEEX_col_ind(long kk, long jj, long n_env) {
	long x_min = std::min(kk, jj);
	long x_diff = std::abs(kk - jj);
	long index = x_min * n_env - ((x_min - 1) * x_min) / 2 + x_diff;
	return index;
}
