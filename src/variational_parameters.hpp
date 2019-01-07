#ifndef VARIATIONAL_PARAMETERS_HPP
#define VARIATIONAL_PARAMETERS_HPP

#include <iostream>
#include <limits>
#include "hyps.hpp"
#include "utils.hpp"
#include "class.h"
#include "tools/eigen3.3/Dense"

class VariationalParamsBase {
public:
	// This stores parameters used in VB and some summary quantities that
	// depend on those parameters.
	parameters p;

	// Variational parameters for slab (params 1)
	Eigen::ArrayXd alpha_beta; // P x (E+1)
	Eigen::ArrayXd mu1_beta;    // P x (E+1)
	Eigen::ArrayXd s1_beta_sq;  // P x (E+1)
	Eigen::ArrayXd mu2_beta;    // P x (E+1)
	Eigen::ArrayXd s2_beta_sq;  // P x (E+1)

	// Variational parameters for spike (MoG prior mode; params 2)
	Eigen::ArrayXd alpha_gam; // P x (E+1)
	Eigen::ArrayXd mu1_gam;    // P x (E+1)
	Eigen::ArrayXd s1_gam_sq;  // P x (E+1)
	Eigen::ArrayXd mu2_gam;    // P x (E+1)
	Eigen::ArrayXd s2_gam_sq;  // P x (E+1)

	// Variational parameters for covariate main effects
	Eigen::ArrayXd muc;    // C x 1
	Eigen::ArrayXd sc_sq;  // C x 1

	// Variational params for weights
	Eigen::ArrayXd muw;      // n_env x 1
	Eigen::ArrayXd sw_sq;    // n_env x 1

	VariationalParamsBase(parameters my_params) : p(my_params){};

	/*** utility functions ***/
	void resize(std::int32_t n_samples, std::int32_t n_var, long n_covar, long n_env){
		s1_beta_sq.resize(n_var);
		if(p.mode_mog_prior_beta) {
			s2_beta_sq.resize(n_var);
		}

		if(n_env > 0) {
			double eps = std::numeric_limits<double>::min();
			sw_sq.resize(n_env);
			sw_sq = eps; // For EdZtZ

			s1_gam_sq.resize(n_var);
			if (p.mode_mog_prior_gam) {
				s2_gam_sq.resize(n_var);
			}
		}

		// for covars
		if(p.use_vb_on_covars){
			sc_sq.resize(n_covar);
		}
	}

	/*** mean of latent variables ***/
	Eigen::VectorXd mean_beta() const {
		Eigen::VectorXd rr_beta;
		if(p.mode_mog_prior_beta){
			rr_beta = alpha_beta * (mu1_beta - mu2_beta) + mu2_beta;
		} else {
			rr_beta = alpha_beta * mu1_beta;
		}
		return rr_beta;
	}

	Eigen::VectorXd mean_gam() const {
		Eigen::VectorXd rr_gam;
		if(p.mode_mog_prior_gam){
			rr_gam = alpha_gam * (mu1_gam - mu2_gam) + mu2_gam;
		} else {
			rr_gam = alpha_gam * mu1_gam;
		}
		return rr_gam;
	}

	double mean_beta(std::uint32_t jj) const {
		double rr_beta = alpha_beta(jj) * mu1_beta(jj);
		if(p.mode_mog_prior_beta){
			rr_beta += (1.0 - alpha_beta(jj)) * mu2_beta(jj);
		}
		return rr_beta;
	}

	double mean_gam(std::uint32_t jj) const {
		double rr_gam = alpha_gam(jj) * mu1_gam(jj);
		if(p.mode_mog_prior_gam){
			rr_gam += (1.0 - alpha_gam(jj)) * mu2_gam(jj);
		}
		return rr_gam;
	}

	/*** variance of latent variables ***/
	Eigen::ArrayXd var_beta() const {
		Eigen::ArrayXd varB = alpha_beta * (s1_beta_sq + (1.0 - alpha_beta) * mu1_beta.square());
		if(p.mode_mog_prior_beta){
			varB += (1.0 - alpha_beta) * (s2_beta_sq + (alpha_beta) * mu2_beta.square());
			varB -= 2.0 * alpha_beta * (1.0 - alpha_beta) * mu1_beta * mu2_beta;
		}
		return varB;
	}

	Eigen::ArrayXd var_gam() const {
		Eigen::ArrayXd varG = alpha_gam * (s1_gam_sq + (1.0 - alpha_gam) * mu1_gam.square());
		if (p.mode_mog_prior_gam) {
			varG += (1.0 - alpha_gam) * (s2_gam_sq + (alpha_gam) * mu2_gam.square());
			varG -= 2.0 * alpha_gam * (1.0 - alpha_gam) * mu1_gam * mu2_gam;
		}
		return varG;
	}

	Eigen::ArrayXd mean_beta_sq(int u0) const {
		// if u0 == 1; E[u \beta^2]
		// if u0 == 0; E[1 - u \beta^2]
		Eigen::ArrayXd res;
		assert(u0 == 1 || u0 == 2);
		if(u0 == 1)
			res = alpha_beta * (s1_beta_sq + mu1_beta.square());
		else if (u0 == 2 && p.mode_mog_prior_beta){
			res = (1 - alpha_beta) * (s2_beta_sq + mu2_beta.square());
		}
		return res;
	}

	Eigen::ArrayXd mean_gam_sq(int u0) const {
		// if u0 == 1; E[u \beta^2]
		// if u0 == 0; E[1 - u \beta^2]
		Eigen::ArrayXd res;
		assert(u0 == 1 || u0 == 2);
		if(u0 == 1)
			res = alpha_gam * (s1_gam_sq + mu1_gam.square());
		else if (u0 == 2 && p.mode_mog_prior_gam){
			res = (1 - alpha_gam) * (s2_gam_sq + mu2_gam.square());
		}
		return res;
	}
};

// Store subset of Variational Parameters to be RAM efficient
class VariationalParametersLite : public VariationalParamsBase {
public:
	// Other quantities to track
	EigenDataVector yx;    // N x 1
	EigenDataVector ym;    // N x 1
	EigenDataVector eta;
	EigenDataVector eta_sq;

	VariationalParametersLite(parameters my_params) : VariationalParamsBase(my_params) {};
};

class VariationalParameters : public VariationalParamsBase {
public:
	// This stores parameters used in VB and some summary quantities that
	// depend on those parameters.

	// Summary quantities
	EigenRefDataVector yx;      // N x 1
	EigenRefDataVector ym;      // N x 1
	EigenRefDataVector eta;     // expected value of matrix product E x w
	EigenRefDataVector eta_sq;  // expected value (E x w) cdot (E x w)

	Eigen::ArrayXd EdZtZ;   // expectation of the diagonal of Z^t Z


	VariationalParameters(parameters my_params,
						  EigenRefDataVector my_ym,
						  EigenRefDataVector my_yx,
						  EigenRefDataVector my_eta,
						  EigenRefDataVector my_eta_sq) : VariationalParamsBase(my_params), yx(my_yx), ym(my_ym),
																   eta(my_eta), eta_sq(my_eta_sq){};

	~VariationalParameters(){};

	void init_from_lite(const VariationalParametersLite& init) {
		// yx and ym set to point to appropriate col of VBayesX2::YM and VBayesX2::YX in constructor

		alpha_beta = init.alpha_beta;
		mu1_beta   = init.mu1_beta;
		mu2_beta   = init.mu2_beta;
		alpha_gam  = init.alpha_gam;
		mu1_gam    = init.mu1_gam;
		mu2_gam    = init.mu2_gam;

		muc   = init.muc;
		muw   = init.muw;
	}

	VariationalParametersLite convert_to_lite(){
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

	void calcEdZtZ(const Eigen::Ref<const Eigen::ArrayXXd>& dXtEEX, const int& n_env){
		Eigen::ArrayXd muw_sq(n_env * n_env);
		for (int ll = 0; ll < n_env; ll++){
			for (int mm = 0; mm < n_env; mm++){
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
};

#endif
