#ifndef VARIATIONAL_PARAMETERS_HPP
#define VARIATIONAL_PARAMETERS_HPP

#include <iostream>
#include "class.h"
#include "tools/eigen3.3/Dense"

// Store subset of Variational Parameters to be RAM efficient
struct VariationalParametersLite {
	// Other quantities to track
	Eigen::VectorXd yx;    // N x 1
	Eigen::VectorXd ym;    // N x 1
	Eigen::VectorXd eta;
	Eigen::VectorXd eta_sq;

	// Variational parameters for slab
	Eigen::ArrayXXd alpha; // P x (E+1)
	Eigen::ArrayXXd mu1;    // P x (E+1)

	// Variational parameters for spike (MoG prior mode)
	Eigen::ArrayXXd mu2;    // P x (E+1)

	// Variational parameters for covariate main effects
	Eigen::ArrayXd  muc;    // C x 1

	// Variational params for weights
	Eigen::ArrayXd  muw;    // n_env x 1
};

class VariationalParameters {
public:
	// This stores parameters used in VB and some summary quantities that
	// depend on those parameters.

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
	Eigen::ArrayXd  muc;    // C x 1
	Eigen::ArrayXd  sc_sq;  // C x 1

	// Variational params for weights
	Eigen::ArrayXd  muw;      // n_env x 1
	Eigen::ArrayXd  sw_sq;    // n_env x 1

	// Summary quantities
	Eigen::VectorXd yx;      // N x 1
	Eigen::VectorXd ym;      // N x 1
	Eigen::VectorXd eta;     // expected value of matrix product E x w
	Eigen::VectorXd eta_sq;  // expected value (E x w) cdot (E x w)
	Eigen::ArrayXd  EdZtZ;   // expectation of the diagonal of Z^t Z
	Eigen::ArrayXd varB;    // variance of beta, gamma under approximating distn
	Eigen::ArrayXd varG;    // variance of beta, gamma under approximating distn


	// sgd
	long int count;

	VariationalParameters(){};
	~VariationalParameters(){};

	void init_from_lite(const VariationalParametersLite& init,
			const parameters& p) {
		ym    = init.ym;
		yx    = init.yx;
		alpha_beta = init.alpha.col(0);
		mu1_beta   = init.mu1.col(0);
		if(p.mode_mog_prior) {
			mu2_beta = init.mu2.col(0);
		}
		if(init.alpha.cols() > 1) {
			alpha_gam = init.alpha.col(1);
			mu1_gam = init.mu1.col(1);
			if(p.mode_mog_prior) {
				mu2_gam = init.mu2.col(1);
			}
		}
		muc   = init.muc;

		count = 0;
		muw   = init.muw;

		eta    = init.eta;
		eta_sq = init.eta_sq;
	}

	VariationalParametersLite convert_to_lite(int n_effects,
			const parameters& p){
		VariationalParametersLite vplite;
		vplite.ym    = ym;
		vplite.yx    = yx;
		if(n_effects > 1) {
			vplite.alpha.resize(alpha_beta.rows(), 2);
			vplite.mu1.resize(alpha_beta.rows(), 2);
			vplite.mu2.resize(alpha_beta.rows(), 2);

			vplite.alpha.col(0) = alpha_beta;
			vplite.mu1.col(0)   = mu1_beta;
			vplite.alpha.col(1) = alpha_gam;
			vplite.mu1.col(1)   = mu1_gam;
			if(p.mode_mog_prior) {
				vplite.mu2.col(0) = mu2_beta;
				vplite.mu2.col(1) = mu2_gam;
			}
		} else {
			vplite.alpha = alpha_beta;
			vplite.mu1    = mu1_beta;
			if(p.mode_mog_prior) {
				vplite.mu2 = mu2_beta;
			}
		}
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
