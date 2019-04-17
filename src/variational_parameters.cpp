//
// Created by kerin on 2019-03-01.
//

#include "variational_parameters.hpp"
#include <iostream>
#include <limits>
#include "hyps.hpp"
#include "parameters.hpp"
#include "tools/eigen3.3/Dense"
#include "Prior.hpp"

void VariationalParamsBase::run_default_init(long n_var, long n_covar, long n_env){
	/* - snp latent variables initialised from zero
	 * - covar latent variables initialised from zero
	 * - env latent variables initialised from uniform
	 */
	betas.resize(n_var);
	gammas.resize(n_var);
	weights.resize(n_env);
	covars.resize(n_covar);

	alpha_beta    = Eigen::ArrayXd::Zero(n_var);
	mu1_beta       = Eigen::ArrayXd::Zero(n_var);
	s1_beta_sq     = Eigen::ArrayXd::Constant(n_var, 0.01);
	if(p.mode_mog_prior_beta) {
		mu2_beta   = Eigen::ArrayXd::Zero(n_var);
		s2_beta_sq = Eigen::ArrayXd::Constant(n_var, 0.01);
	}

	if(n_env > 0) {
		alpha_gam = Eigen::ArrayXd::Zero(n_var);
		mu1_gam   = Eigen::ArrayXd::Zero(n_var);
		s1_gam_sq     = Eigen::ArrayXd::Constant(n_var, 0.01);
		if (p.mode_mog_prior_gam) {
			mu2_gam   = Eigen::ArrayXd::Zero(n_var);
			s2_gam_sq = Eigen::ArrayXd::Constant(n_var, 0.01);
		}
		Eigen::ArrayXd w_init = Eigen::ArrayXd::Constant(n_env, 1.0 / n_env);
		muw = w_init;
		sw_sq = Eigen::ArrayXd::Constant(n_env, 0.01);

		weights.set_mean(w_init);
	}

	if(n_covar > 0) {
		muc   = Eigen::ArrayXd::Zero(n_covar);
		sc_sq = Eigen::ArrayXd::Constant(n_covar, 0.01);
	}
}

void VariationalParamsBase::dump_snps_to_file(boost_io::filtering_ostream& outf,
                                              const GenotypeMatrix& X, long n_env) const {
	long n_var = betas.size();
	 outf << "SNPID " << betas.header("beta");
	 if(n_env > 0){
	  outf << " " << gammas.header("gam");
	 }
	 outf << std::endl;
	 outf << std::scientific << std::setprecision(8);
	 for (long ii = 0; ii < n_var; ii++) {
	  outf << X.SNPID[ii];
	  outf << " " << betas.get_ith_distn(ii);
	  if(n_env > 0) {
	      outf << " " << gammas.get_ith_distn(ii);
	  }
	  outf << std::endl;
	 }
}

/*** Get and set properties of latent variables ***/

Eigen::VectorXd VariationalParamsBase::mean_beta() const {
	 return betas.mean();

//	Eigen::VectorXd rr_beta;
//	if(p.mode_mog_prior_beta) {
//		rr_beta = alpha_beta * (mu1_beta - mu2_beta) + mu2_beta;
//	} else {
//		rr_beta = alpha_beta * mu1_beta;
//	}
//	return rr_beta;
}

Eigen::VectorXd VariationalParamsBase::mean_gam() const {
	 return gammas.mean();
}

Eigen::VectorXd VariationalParamsBase::mean_weights() const {
	 return weights.mean();
}

Eigen::VectorXd VariationalParamsBase::mean_covar() const {
	 return covars.mean();
}

double VariationalParamsBase::mean_weights(long ll) const {
	 return weights.mean(ll);
}

double VariationalParamsBase::mean_covar(long cc) const {
	 return covars.mean(cc);
}

double VariationalParamsBase::mean_beta(std::uint32_t jj) const {
	 return betas.mean(jj);
}

double VariationalParamsBase::mean_gam(std::uint32_t jj) const {
	 return gammas.mean(jj);
}

Eigen::ArrayXd VariationalParamsBase::var_beta() const {
	 return betas.var();
}

Eigen::ArrayXd VariationalParamsBase::var_gam() const {
	 return gammas.var();
}

Eigen::ArrayXd VariationalParamsBase::var_weights() const {
	 return weights.var();
}

/*** KL Divergence ***/
double VariationalParamsBase::kl_div_gamma(const Hyps& hyps) const {
	 double lambda = hyps.lambda[1];
	 double sigma_slab = hyps.slab_var(1);
	 double sigma_spike = hyps.spike_var(1);
	 double res = gammas.kl_div(lambda, sigma_slab, sigma_spike);
	return res;
};

double VariationalParamsBase::kl_div_beta(const Hyps& hyps) const {
	 double lambda = hyps.lambda[0];
	 double sigma_slab = hyps.slab_var(0);
	 double sigma_spike = hyps.spike_var(0);
	 double res = betas.kl_div(lambda, sigma_slab, sigma_spike);
	return res;
};

double VariationalParamsBase::kl_div_covars(const Hyps& hyps) const {
	 double sigma = hyps.sigma * hyps.sigma_c;
	 double res = covars.kl_div(sigma);
	return res;
};

double VariationalParamsBase::kl_div_weights(const Hyps& hyps) const {
	 double sigma = hyps.sigma_w;
	 double res = weights.kl_div(sigma);
	return res;
};

MoGaussian VariationalParamsBase::beta_j_step(long jj, double EXty, double EXtX, Hyps hyps) {
	double old = mean_beta(jj);
	double lambda = hyps.lambda[0];
	double sigma0 = hyps.slab_var[0];                 // slab
	double sigma1 = hyps.spike_var[0];


	double m1 = (EXty + old * EXtX) / hyps.sigma;
	double m2 = -EXtX / 2.0 / hyps.sigma;
	double nat1 = m1;
	double nat2_slab = m2 - 1.0 / 2 / sigma0;
	double nat2_spike = m2 - 1.0 / 2 / sigma1;
	Gaussian slab_g, spike_g;

	slab_g.nats << nat1, nat2_slab;
	spike_g.nats << nat1, nat2_spike;

	MoGaussian gg;
	gg.slab = slab_g;
	gg.spike = spike_g;
	gg.maximise_mix_coeff(lambda, sigma0, sigma1);

	alpha_beta(jj) = gg.mix;
	mu1_beta(jj) = slab_g.mean();
	s1_beta_sq(jj) = slab_g.var();
	mu2_beta(jj) = spike_g.mean();
	s2_beta_sq(jj) = spike_g.var();
	return gg;
}

MoGaussian VariationalParamsBase::gamma_j_step(long jj, double EXty, double EXtX, Hyps hyps) {
	double old = mean_gam(jj);
	double lambda = hyps.lambda[1];
	double sigma0 = hyps.slab_var[1];                 // slab
	double sigma1 = hyps.spike_var[1];


	double m1 = (EXty + old * EXtX) / hyps.sigma;
	double m2 = -EXtX / 2.0 / hyps.sigma;
	double nat1 = m1;
	double nat2_slab = m2 - 1.0 / 2 / sigma0;
	double nat2_spike = m2 - 1.0 / 2 / sigma1;
	Gaussian slab_g, spike_g;

	slab_g.nats << nat1, nat2_slab;
	spike_g.nats << nat1, nat2_spike;

	MoGaussian gg;
	gg.slab = slab_g;
	gg.spike = spike_g;
	gg.maximise_mix_coeff(lambda, sigma0, sigma1);
	check_nan(gg.mix, jj);

	alpha_gam(jj) = gg.mix;
	mu1_gam(jj) = slab_g.mean();
	s1_gam_sq(jj) = slab_g.var();
	mu2_gam(jj) = spike_g.mean();
	s2_gam_sq(jj) = spike_g.var();
	return gg;
}

Gaussian VariationalParamsBase::weights_l_step(long ll, double EXty, double EXtX, Hyps hyps) const {
	double old = mean_weights(ll);

	double m1 = (EXty + old * EXtX) / hyps.sigma;
	double m2 = -EXtX / 2.0 / hyps.sigma;
	double nat1 = m1;
	double nat2 = m2 - 1.0 / 2 / hyps.sigma_w;
	Gaussian w;
	w.nats << nat1, nat2;
	return w;
}

Gaussian VariationalParamsBase::covar_c_step(long cc, double EXty, double EXtX, Hyps hyps) const {
	double old = mean_covar(cc);

	double m1 = (EXty + old * EXtX) / hyps.sigma;
	double m2 = -EXtX / 2.0 / hyps.sigma;
	double nat1 = m1;
	double nat2 = m2 - 1.0 / 2 / hyps.sigma_c;
	Gaussian w;
	w.nats << nat1, nat2;
	return w;
}

void VariationalParamsBase::check_nan(const double& alpha,
                                      const std::uint32_t& ii){
	// check for NaNs and spit out diagnostics if so.
	if(std::isnan(alpha)) {
		// TODO: dump snpstats to file
		std::cout << "NaN detected at SNP index: ";
		std::cout << ii << std::endl;
		throw std::runtime_error("NaN detected");
	}
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

	betas = init.betas;
	gammas = init.gammas;
	weights = init.weights;
	covars = init.covars;
}

VariationalParametersLite VariationalParameters::convert_to_lite() {
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
	vplite.muw   = muw;
	vplite.eta   = eta;

	vplite.betas = betas;
	vplite.gammas = gammas;
	vplite.weights = weights;
	vplite.covars = covars;
	return vplite;
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
