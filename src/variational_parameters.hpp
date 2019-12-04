#ifndef VARIATIONAL_PARAMETERS_HPP
#define VARIATIONAL_PARAMETERS_HPP

#include "genotype_matrix.hpp"
#include "parameters.hpp"
#include "typedefs.hpp"
#include "hyps.hpp"

#include "tools/eigen3.3/Dense"
#include "file_utils.hpp"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/filesystem.hpp>

#include <iostream>
#include <limits>

namespace boost_io = boost::iostreams;

long dXtEEX_col_ind(long kk, long jj, long n_env);

class VariationalParamsBase {
public:
	// This stores parameters used in VB and some summary quantities that
	// depend on those parameters.
	parameters p;

	// Variational parameters for slab (params 1)
	// P x (E+1)
	Eigen::ArrayXd alpha_beta;
	Eigen::ArrayXd mu1_beta;
	Eigen::ArrayXd s1_beta_sq;
	Eigen::ArrayXd mu2_beta;
	Eigen::ArrayXd s2_beta_sq;

	// Variational parameters for spike (MoG prior mode; params 2)
	// P x (E+1)
	Eigen::ArrayXd alpha_gam;
	Eigen::ArrayXd mu1_gam;
	Eigen::ArrayXd s1_gam_sq;
	Eigen::ArrayXd mu2_gam;
	Eigen::ArrayXd s2_gam_sq;

	// Variational parameters for covariate main effects
	// n_covar x 1
	Eigen::ArrayXd muc;
	Eigen::ArrayXd sc_sq;

	// Variational params for weights
	// n_env x 1
	Eigen::ArrayXd muw;
	Eigen::ArrayXd sw_sq;

	VariationalParamsBase(const parameters& my_params) : p(my_params){
	};

	/*** utility functions ***/
	void resize(std::int32_t n_samples, std::int32_t n_var, long n_covar, long n_env);
	void run_default_init(long n_var, long n_covar, long n_env);

	/*** mean of latent variables ***/
	Eigen::VectorXd mean_covars() const;
	Eigen::VectorXd mean_beta() const;
	Eigen::VectorXd mean_gam() const;
	Eigen::VectorXd mean_weights() const;

	double mean_beta(std::uint32_t jj) const;
	double mean_gam(std::uint32_t jj) const;

	/*** variance of latent variables ***/
	Eigen::ArrayXd var_beta() const;
	Eigen::ArrayXd var_gam() const;
	Eigen::ArrayXd mean_beta_sq(int u0) const;
	Eigen::ArrayXd mean_gam_sq(int u0) const;

	/*** IO ***/
	void env_to_file(const std::string& path, const std::vector<std::string>& env_names) const;
	void covar_to_file(const std::string& path, const std::vector<std::string>& covar_names) const;
	void snps_to_file(const std::string& path, const GenotypeMatrix &X, long n_env) const;
};

// Store subset of Variational Parameters to be RAM efficient
class VariationalParametersLite : public VariationalParamsBase {
public:
// Other quantities to track
	EigenDataVector yx;
	EigenDataVector ym;
	EigenDataVector eta;
	EigenDataVector eta_sq;

	VariationalParametersLite(const parameters& my_params) : VariationalParamsBase(my_params) {
	};
};

class VariationalParameters : public VariationalParamsBase {
public:
// This stores parameters used in VB and some summary quantities that
// depend on those parameters.

// Summary quantities
	EigenRefDataVector yx;
	EigenRefDataVector ym;
	EigenRefDataVector eta;
	EigenRefDataVector eta_sq;

	Eigen::ArrayXd EdZtZ;


	VariationalParameters(const parameters my_params,
	                      EigenRefDataVector my_ym,
	                      EigenRefDataVector my_yx,
	                      EigenRefDataVector my_eta,
	                      EigenRefDataVector my_eta_sq) : VariationalParamsBase(my_params), yx(my_yx), ym(my_ym),
		eta(my_eta), eta_sq(my_eta_sq){
	};

	~VariationalParameters(){
	};

	void init_from_lite(const VariationalParametersLite& init);

	VariationalParametersLite convert_to_lite() const;

	void calcEdZtZ(const Eigen::Ref<const Eigen::ArrayXXd>& dXtEEX, const long& n_env);
};

#endif
