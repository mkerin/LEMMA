//
// Created by kerin on 2019-03-01.
//

#include "variational_parameters.hpp"
#include <iostream>
#include <limits>
#include "hyps.hpp"
#include "parameters.hpp"
#include "tools/Eigen/Dense"
#include "Prior.hpp"

void VariationalParameters::run_default_init(long n_var, long n_covar, long n_env){
	/* - snp latent variables initialised from zero
	 * - covar latent variables initialised from zero
	 * - env latent variables initialised from uniform
	 */
	betas->resize(n_var);
	if(n_env > 0) {
		gammas->resize(n_var);
		weights.resize(n_env);

		Eigen::ArrayXd w_init = Eigen::ArrayXd::Constant(n_env, 1.0 / n_env);
		weights.set_mean(w_init);
		pve.resize(2);
		pve << -1, -1;
		s_x.resize(2);
		s_x << -1, -1;
	} else {
		pve.resize(1);
		pve << -1;
		s_x.resize(1);
		s_x << -1;
	}

	if(n_covar > 0) {
		covars.resize(n_covar);
	}
}

void VariationalParameters::dump_snps_to_file(boost_io::filtering_ostream& outf,
                                              const GenotypeMatrix& X, long n_env) const {
	long n_var = betas->size();
	outf << "SNPID " << betas->header("beta");
	if(n_env > 0) {
		outf << " " << gammas->header("gam");
	}
	outf << std::endl;
	outf << std::scientific << std::setprecision(8);
	for (long ii = 0; ii < n_var; ii++) {
		outf << X.SNPID[ii];
		outf << " ";
		write_ith_beta_to_stream(ii, outf);
		if(n_env > 0) {
			outf << " ";
			write_ith_beta_to_stream(ii, outf);
		}
		outf << std::endl;
	}
}

/*** Get and set properties of latent variables ***/

Eigen::VectorXd VariationalParameters::mean_beta() const {
	return betas->mean();
}

Eigen::VectorXd VariationalParameters::mean_gam() const {
	return gammas->mean();
}

Eigen::VectorXd VariationalParameters::mean_weights() const {
	return weights.mean();
}

Eigen::VectorXd VariationalParameters::mean_covar() const {
	return covars.mean();
}

double VariationalParameters::mean_weights(long ll) const {
	return weights.mean(ll);
}

double VariationalParameters::mean_covar(long cc) const {
	return covars.mean(cc);
}

double VariationalParameters::mean_beta(std::uint32_t jj) const {
	return betas->mean(jj);
}

double VariationalParameters::mean_gam(std::uint32_t jj) const {
	return gammas->mean(jj);
}

Eigen::ArrayXd VariationalParameters::var_beta() const {
	return betas->var();
}

Eigen::ArrayXd VariationalParameters::var_gam() const {
	return gammas->var();
}

Eigen::ArrayXd VariationalParameters::var_weights() const {
	return weights.var();
}

Eigen::ArrayXd VariationalParameters::var_covar() const {
	return covars.var();
}

double VariationalParameters::var_weights(long jj) const {
	return weights.var(jj);
}

double VariationalParameters::var_covar(long jj) const {
	return covars.var(jj);
}

void VariationalParameters::check_nan(const double& alpha,
                                      const std::uint32_t& ii){
	// check for NaNs and spit out diagnostics if so.
	if(std::isnan(alpha)) {
		// TODO: dump snpstats to file
		std::cout << "NaN detected at SNP index: ";
		std::cout << ii << std::endl;
		throw std::runtime_error("NaN detected");
	}
}

void VariationalParameters::set_hyps(Hyps hyps){
	sigma = hyps.sigma;
	Eigen::ArrayXd tmp(3);
	tmp << hyps.lambda(0), hyps.slab_var(0), hyps.spike_var(0);
	betas->set_hyps(tmp);

	if(hyps.lambda.size() > 1) {
		tmp << hyps.lambda(1), hyps.slab_var(1), hyps.spike_var(1);
		gammas->set_hyps(tmp);

		tmp.resize(1);
		tmp << hyps.sigma_w;
		weights.set_hyps(tmp);
	}

	tmp.resize(1);
	tmp << hyps.sigma * hyps.sigma_c;
	covars.set_hyps(tmp);
}

void VariationalParameters::calcEdZtZ(const Eigen::Ref<const Eigen::ArrayXXd> &dXtEEX, const int &n_env) {
	Eigen::ArrayXd muw_sq(n_env * n_env);
	for (int ll = 0; ll < n_env; ll++) {
		for (int mm = 0; mm < n_env; mm++) {
			muw_sq(mm*n_env + ll) = weights.mean(mm) * weights.mean(ll);
		}
	}

	EdZtZ = (dXtEEX.rowwise() * muw_sq.transpose()).rowwise().sum();
	if(n_env > 1) {
		for (int ll = 0; ll < n_env; ll++) {
			EdZtZ += dXtEEX.col(ll * n_env + ll) * weights.var(ll);
		}
	}
}
