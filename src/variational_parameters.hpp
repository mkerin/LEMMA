#ifndef VARIATIONAL_PARAMETERS_HPP
#define VARIATIONAL_PARAMETERS_HPP

#include <iostream>
#include "tools/eigen3.3/Dense"

// Store subset of Variational Parameters to be RAM efficient
struct VariationalParametersLite {
	// Store vector of predicted phenotypes Hr only for instance vp_init.
	// In general don't bother storing this.
	Eigen::VectorXd Hr;    // N x 1

	// Variational parameters for slab
	Eigen::ArrayXXd alpha; // P x (E+1)
	Eigen::ArrayXXd mu;    // P x (E+1)

	// Variational parameters for spike (MoG prior mode)
	Eigen::ArrayXXd mup;    // P x (E+1)

	// Variational parameters for covariate main effects
	Eigen::ArrayXXd muc;    // C x 1
};

class VariationalParameters {
public:
	Eigen::VectorXd Hr;    // N x 1

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

	VariationalParameters(){};
	~VariationalParameters(){};

	void init_from_lite(const VariationalParametersLite& init) {
		Hr    = init.Hr;
		alpha = init.alpha;
		mu    = init.mu;
		mup   = init.mup;
		muc   = init.muc;
	}

	VariationalParametersLite convert_to_lite(){
		VariationalParametersLite vplite;
		vplite.Hr    = Hr;
		vplite.alpha = alpha;
		vplite.mu    = mu;
		vplite.mup   = mup;
		vplite.muc   = muc;
		return vplite;
	}
};

#endif
