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
	Eigen::ArrayXXd alpha; // P x (E+1)
	Eigen::ArrayXXd mu;    // P x (E+1)

	// Variational parameters for spike (MoG prior mode)
	Eigen::ArrayXXd mup;    // P x (E+1)

	// Variational parameters for covariate main effects
	Eigen::ArrayXd  muc;    // C x 1

	// Variational params for weights
	Eigen::ArrayXd  muw;    // n_env x 1
};

class VariationalParameters {
public:
	// This stores parameters used in VB and some summary quantities that
	// depend on those parameters.

	// Variational parameters for slab
	Eigen::ArrayXXd alpha; // P x (E+1)
	Eigen::ArrayXXd mu;    // P x (E+1)
	Eigen::ArrayXXd s_sq;  // P x (E+1)

	// Variational parameters for spike (MoG prior mode)
	Eigen::ArrayXXd mup;    // P x (E+1)
	Eigen::ArrayXXd sp_sq;  // P x (E+1)

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
	Eigen::ArrayXXd varB;    // variance of beta, gamma under approximating distn


	// sgd
	long int count;

	VariationalParameters(){};
	~VariationalParameters(){};

	void init_from_lite(const VariationalParametersLite& init) {
		ym    = init.ym;
		yx    = init.yx;
		alpha = init.alpha;
		mu    = init.mu;
		mup   = init.mup;
		muc   = init.muc;

		count = 0;
		muw   = init.muw;

		eta    = init.eta;
		eta_sq = init.eta_sq;
	}

	VariationalParametersLite convert_to_lite(){
		VariationalParametersLite vplite;
		vplite.ym    = ym;
		vplite.yx    = yx;
		vplite.alpha = alpha;
		vplite.mu    = mu;
		vplite.mup   = mup;
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
