#ifndef VARIATIONAL_PARAMETERS_HPP
#define VARIATIONAL_PARAMETERS_HPP

#include "genotype_matrix.hpp"
#include "parameters.hpp"
#include "typedefs.hpp"

#include "tools/eigen3.3/Dense"

#include <iostream>
#include <limits>
#include "hyps.hpp"

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
	void resize(std::int32_t n_samples, std::int32_t n_var, long n_covar, long n_env);

	void dump_snps_to_file(boost_io::filtering_ostream& my_outf, const GenotypeMatrix& X, long n_env) const;

	/*** mean of latent variables ***/
	Eigen::VectorXd mean_beta() const;

	Eigen::VectorXd mean_gam() const;

	double mean_beta(std::uint32_t jj) const;

	double mean_gam(std::uint32_t jj) const;

	/*** variance of latent variables ***/
	Eigen::ArrayXd var_beta() const;

	Eigen::ArrayXd var_gam() const;

	Eigen::ArrayXd mean_beta_sq(int u0) const;

	Eigen::ArrayXd mean_gam_sq(int u0) const;
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

	void init_from_lite(const VariationalParametersLite& init);

	VariationalParametersLite convert_to_lite();

	void calcEdZtZ(const Eigen::Ref<const Eigen::ArrayXXd>& dXtEEX, const int& n_env);
};

#endif
