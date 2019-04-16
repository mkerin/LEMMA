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
		double eps = std::numeric_limits<double>::min();
		muw = Eigen::ArrayXd::Constant(n_env, 1.0 / n_env);
		sw_sq = Eigen::ArrayXd::Constant(n_env, 0.01);
	}

	if(n_covar > 0) {
		muc   = Eigen::ArrayXd::Zero(n_covar);
		sc_sq = Eigen::ArrayXd::Constant(n_covar, 0.01);
	}
}

void VariationalParamsBase::dump_snps_to_file(boost_io::filtering_ostream& outf,
                                              const GenotypeMatrix& X, long n_env) const {
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
}

/*** Get and set properties of latent variables ***/

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

Eigen::VectorXd VariationalParamsBase::mean_covar() const {
	return muc;
}

double VariationalParamsBase::mean_weights(long ll) const {
	return muw[ll];
}

double VariationalParamsBase::mean_covar(long cc) const {
//	return covars[cc].mean();
	return muc[cc];
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

Eigen::ArrayXd VariationalParamsBase::var_weights() const {
	return sw_sq;
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

void VariationalParamsBase::set_mean_covar(Eigen::MatrixXd mu){
	long nn = mu.rows();

	covars.resize(nn);
	for (long ii = 0; ii < nn; ii++){
		covars[ii].set_mean_var(mu(ii), 0.5);
	}
}

/*** KL Divergence ***/
double VariationalParamsBase::kl_div_gamma(const Hyps& hyps) const {
	double res = 0;
	double lambda = hyps.lambda[1];
	double sigma_slab = hyps.slab_var(1);
	double sigma_spike = hyps.spike_var(1);
	long n_var = alpha_gam.rows();

	res += std::log(lambda + eps) * alpha_gam.sum();
	res += std::log(1.0 - lambda + eps) * ((double) n_var - alpha_gam.sum());

	for (std::uint32_t kk = 0; kk < n_var; kk++) {
		res -= alpha_gam(kk) * std::log(alpha_gam(kk) + eps);
		res -= (1 - alpha_gam(kk)) * std::log(1 - alpha_gam(kk) + eps);
	}

	if(p.mode_mog_prior_gam) {
		res += n_var / 2.0;

		res -= mean_gam_sq(1).sum() / 2.0 / sigma_slab;
		res -= mean_gam_sq(2).sum() / 2.0 / sigma_spike;

		res += (alpha_gam * s1_gam_sq.log()).sum() / 2.0;
		res += ((1.0 - alpha_gam) * s2_gam_sq.log()).sum() / 2.0;

		res -= std::log(sigma_slab)  * alpha_gam.sum() / 2.0;
		res -= std::log(sigma_spike) * (n_var - alpha_gam.sum()) / 2.0;
	} else {
		res += (alpha_gam * s1_gam_sq.log()).sum() / 2.0;
		res -= (alpha_gam * (mu1_gam.square() + s1_gam_sq)).sum() / 2.0 / sigma_slab;

		res += (1 - std::log(sigma_slab)) * alpha_gam.sum() / 2.0;
	}
	return res;
};

double VariationalParamsBase::kl_div_beta(const Hyps& hyps) const {
	double res = 0;
	double lambda = hyps.lambda[0];
	double sigma_slab = hyps.slab_var(0);
	double sigma_spike = hyps.spike_var(0);
	long n_var = alpha_beta.rows();

	res += std::log(lambda + eps) * alpha_beta.sum();
	res += std::log(1.0 - lambda + eps) * ((double) n_var - alpha_beta.sum());

	for (std::uint32_t kk = 0; kk < n_var; kk++) {
		res -= alpha_beta(kk) * std::log(alpha_beta(kk) + eps);
		res -= (1 - alpha_beta(kk)) * std::log(1 - alpha_beta(kk) + eps);
	}

	if(p.mode_mog_prior_beta) {
		res += n_var / 2.0;

		res -= mean_beta_sq(1).sum() / 2.0 / sigma_slab;
		res -= mean_beta_sq(2).sum() / 2.0 / sigma_spike;

		res += (alpha_beta * s1_beta_sq.log()).sum() / 2.0;
		res += ((1.0 - alpha_beta) * s2_beta_sq.log()).sum() / 2.0;

		res -= std::log(sigma_slab)  * alpha_beta.sum() / 2.0;
		res -= std::log(sigma_spike) * (n_var - alpha_beta.sum()) / 2.0;
	} else {
		res += (alpha_beta * s1_beta_sq.log()).sum() / 2.0;
		res -= (alpha_beta * (mu1_beta.square() + s1_beta_sq)).sum() / 2.0 / sigma_slab;

		res += (1 - std::log(sigma_slab)) * alpha_beta.sum() / 2.0;
	}
	return res;
};

double VariationalParamsBase::kl_div_covars(const Hyps& hyps) const {
	long n_covar = muc.rows();

	double res = 0;
	res += (double) n_covar * (1.0 - std::log(hyps.sigma * hyps.sigma_c)) / 2.0;
	res += sc_sq.log().sum() / 2.0;
	res -= sc_sq.sum() / 2.0 / hyps.sigma / hyps.sigma_c;
	res -= muc.square().sum() / 2.0 / hyps.sigma / hyps.sigma_c;

//	double prior_var = hyps.sigma * hyps.sigma_c;
//	for (auto distn: covars){
//		res += distn.kl_div(prior_var);
//	}

	return res;
};

double VariationalParamsBase::kl_div_weights(const Hyps& hyps) const {
	long n_env = muw.rows();

	double res = 0;
	res += (double) n_env / 2.0;
	res += sw_sq.log().sum() / 2.0;
	res -= sw_sq.sum() / 2.0;
	res -= mean_weights().array().square().sum() / 2.0;

//	double prior_var = 1;
//	for (auto distn: weights){
//		res += distn.kl_div(prior_var);
//	}

	return res;
};

MoGaussian VariationalParamsBase::beta_j_step(long jj, double EXty, double EXtX, Hyps hyps) {
	double old = mean_beta(jj);
	double lambda = hyps.lambda[0];
	double sigma0 = hyps.slab_var[0]; // slab
	double sigma1 = hyps.spike_var[0];


	double m1 = (EXty + old * EXtX) / hyps.sigma;
	double m2 = - EXtX / 2.0 / hyps.sigma;
	double nat1 = m1;
	double nat2_slab = m2 - 1.0 / 2 / sigma0;
	double nat2_spike = m2 - 1.0 / 2 / sigma1;
	Gaussian slab_g, spike_g;

	slab_g.nat_params << nat1, nat2_slab;
	spike_g.nat_params << nat1, nat2_spike;

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
	double sigma0 = hyps.slab_var[1]; // slab
	double sigma1 = hyps.spike_var[1];


	double m1 = (EXty + old * EXtX) / hyps.sigma;
	double m2 = - EXtX / 2.0 / hyps.sigma;
	double nat1 = m1;
	double nat2_slab = m2 - 1.0 / 2 / sigma0;
	double nat2_spike = m2 - 1.0 / 2 / sigma1;
	Gaussian slab_g, spike_g;

	slab_g.nat_params << nat1, nat2_slab;
	spike_g.nat_params << nat1, nat2_spike;

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
	double m2 = - EXtX / 2.0 / hyps.sigma;
	double nat1 = m1;
	double nat2 = m2 - 1.0 / 2 / hyps.sigma_w;
	Gaussian w;
	w.nat_params << nat1, nat2;
	return w;
}

Gaussian VariationalParamsBase::covar_c_step(long cc, double EXty, double EXtX, Hyps hyps) const {
	double old = mean_covar(cc);

	double m1 = (EXty + old * EXtX) / hyps.sigma;
	double m2 = - EXtX / 2.0 / hyps.sigma;
	double nat1 = m1;
	double nat2 = m2 - 1.0 / 2 / hyps.sigma_c;
	Gaussian w;
	w.nat_params << nat1, nat2;
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
	return vplite;
}

void VariationalParameters::calcEdZtZ(const Eigen::Ref<const Eigen::ArrayXXd> &dXtEEX, const int &n_env) {
	Eigen::ArrayXd muw_sq(n_env * n_env);
	for (int ll = 0; ll < n_env; ll++) {
		for (int mm = 0; mm < n_env; mm++) {
			muw_sq(mm*n_env + ll) = muw(mm) * muw(ll);
		}
	}

	EdZtZ = (dXtEEX.rowwise() * muw_sq.transpose()).rowwise().sum();
	if(n_env > 1) {
		for (int ll = 0; ll < n_env; ll++) {
			EdZtZ += dXtEEX.col(ll * n_env + ll) * sw_sq(ll);
		}
	}
}
