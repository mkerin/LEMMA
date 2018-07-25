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
	Eigen::ArrayXXd muc;    // C x 1

	// Variational params for weights
	Eigen::ArrayXd muw;    // n_env x 1
};

class VariationalParameters {
public:
	Eigen::VectorXd yx;    // N x 1
	Eigen::VectorXd ym;    // N x 1

	Eigen::VectorXd yx_hat;    // N x 1
	Eigen::VectorXd ym_hat;    // N x 1

	// Variational parameters for slab
	Eigen::ArrayXXd s_sq;  // P x (E+1)
	Eigen::ArrayXXd alpha; // P x (E+1)
	Eigen::ArrayXXd mu;    // P x (E+1)

	// Variational parameters for spike (MoG prior mode)
	Eigen::ArrayXXd sp_sq;  // P x (E+1)
	Eigen::ArrayXXd mup;    // P x (E+1)

	// Variational parameters for covariate main effects
	Eigen::ArrayXXd muc;    // C x 1
	Eigen::ArrayXXd sc_sq;  // C x 1

	// Variational params for weights
	Eigen::ArrayXd w; // weights
	Eigen::VectorXd eta; // expected value of linear combo
	Eigen::VectorXd eta_sq; // expected values of linear combo squared (elementwise)
	Eigen::ArrayXd muw;    // n_env x 1
	Eigen::ArrayXd sw_sq;    // n_env x 1
	Eigen::ArrayXd exp_dZtZ; // expectation of the diagonal of Z^T Z

	// sgd
	long int count;

	VariationalParameters(){};
	~VariationalParameters(){};

	void init_from_lite(const VariationalParametersLite& init,
                        const Eigen::Ref<const Eigen::ArrayXXd>& dZtZ, const int& n_env) {
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

		// Explicit zero values from initial variance
		sw_sq.resize(n_env);
 		sw_sq = 0.0;

		calcExpDZtZ(dZtZ, n_env);
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

	void calcExpDZtZ(const Eigen::Ref<const Eigen::ArrayXXd>& dZtZ, const int& n_env){
		Eigen::ArrayXd muw_sq(n_env * n_env);
		for (int ll = 0; ll < n_env; ll++){
			for (int mm = 0; mm < n_env; mm++){
				muw_sq(mm*n_env + ll) = muw(mm) * muw(ll);
			}
		}

		exp_dZtZ = (dZtZ.rowwise() * muw_sq.transpose()).rowwise().sum();
		for (int ll = 0; ll < n_env; ll++){
			exp_dZtZ += dZtZ.col(ll * n_env + ll) * sw_sq(ll);
		}
	}
};

#endif
