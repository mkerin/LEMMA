#ifndef VARIATIONAL_PARAMETERS_HPP
#define VARIATIONAL_PARAMETERS_HPP

#include <iostream>
#include "tools/eigen3.3/Dense"

// Store subset of Variational Parameters to be RAM efficient
struct VariationalParametersLite {
	// Other quantities to track
	Eigen::VectorXd yx;    // N x 1
	Eigen::VectorXd ym;    // N x 1
	Eigen::VectorXd eta;
	Eigen::VectorXd eta_sq;

	// Variational parameters for slab
	Eigen::ArrayXd alpha_beta; // P x (E+1)
	Eigen::ArrayXd mu1_beta;    // P x (E+1)
	Eigen::ArrayXd mu2_beta;    // P x (E+1)
	Eigen::ArrayXd s1_beta_sq;    // P x (E+1)
	Eigen::ArrayXd s2_beta_sq;    // P x (E+1)

	// Variational parameters for spike (MoG prior mode)
	Eigen::ArrayXd alpha_gam; // P x (E+1)
	Eigen::ArrayXd mu1_gam;    // P x (E+1)
	Eigen::ArrayXd mu2_gam;    // P x (E+1)
	Eigen::ArrayXd s1_gam_sq;    // P x (E+1)
	Eigen::ArrayXd s2_gam_sq;    // P x (E+1)

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
	Eigen::Ref<Eigen::VectorXd> yx;      // N x 1
	Eigen::Ref<Eigen::VectorXd> ym;      // N x 1
	Eigen::VectorXd eta;     // expected value of matrix product E x w
	Eigen::VectorXd eta_sq;  // expected value (E x w) cdot (E x w)
	Eigen::ArrayXd  EdZtZ;   // expectation of the diagonal of Z^t Z
	Eigen::ArrayXd varB;    // variance of beta, gamma under approximating distn
	Eigen::ArrayXd varG;    // variance of beta, gamma under approximating distn


	// sgd
	long int count;

	VariationalParameters(Eigen::Ref<Eigen::VectorXd> my_ym,
			Eigen::Ref<Eigen::VectorXd> my_yx) : yx(my_yx), ym(my_ym){};
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

		count = 0;
		muw   = init.muw;

		eta    = init.eta;
		eta_sq = init.eta_sq;
	}

	VariationalParametersLite convert_to_lite(){
		VariationalParametersLite vplite;
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
