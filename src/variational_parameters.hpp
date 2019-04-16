#ifndef VARIATIONAL_PARAMETERS_HPP
#define VARIATIONAL_PARAMETERS_HPP

#include "genotype_matrix.hpp"
#include "parameters.hpp"
#include "typedefs.hpp"
#include "hyps.hpp"
#include "Prior.hpp"

#include "tools/eigen3.3/Dense"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/filesystem.hpp>

#include <iostream>
#include <limits>

namespace boost_io = boost::iostreams;

class VariationalParamsBase {
public:
// This stores parameters used in VB and some summary quantities that
// depend on those parameters.
parameters p;
	double eps = std::numeric_limits<double>::min();


std::vector<MoGaussian> betas;
std::vector<MoGaussian> gammas;
std::vector<Gaussian> weights;
std::vector<Gaussian> covars;

// Variational parameters for slab (params 1)
Eigen::ArrayXd alpha_beta;         // P x (E+1)
Eigen::ArrayXd mu1_beta;            // P x (E+1)
Eigen::ArrayXd s1_beta_sq;          // P x (E+1)
Eigen::ArrayXd mu2_beta;            // P x (E+1)
Eigen::ArrayXd s2_beta_sq;          // P x (E+1)

// Variational parameters for spike (MoG prior mode; params 2)
Eigen::ArrayXd alpha_gam;         // P x (E+1)
Eigen::ArrayXd mu1_gam;            // P x (E+1)
Eigen::ArrayXd s1_gam_sq;          // P x (E+1)
Eigen::ArrayXd mu2_gam;            // P x (E+1)
Eigen::ArrayXd s2_gam_sq;          // P x (E+1)

// Variational parameters for covariate main effects
Eigen::ArrayXd muc;            // C x 1
Eigen::ArrayXd sc_sq;          // C x 1

// Variational params for weights
Eigen::ArrayXd muw;              // n_env x 1
Eigen::ArrayXd sw_sq;            // n_env x 1

VariationalParamsBase(parameters my_params) : p(my_params){
};

/*** utility functions ***/
void run_default_init(long n_var, long n_covar, long n_env);

void dump_snps_to_file(boost_io::filtering_ostream& my_outf, const GenotypeMatrix& X, long n_env) const;

/*** Get and set properties of latent variables ***/
Eigen::VectorXd mean_beta() const;
Eigen::VectorXd mean_gam() const;
Eigen::VectorXd mean_weights() const;
Eigen::VectorXd mean_covar() const;

double mean_beta(std::uint32_t jj) const;
double mean_gam(std::uint32_t jj) const;
double mean_weights(long ll) const;
double mean_covar(long cc) const;

void set_mean_covar(Eigen::MatrixXd mu);
void set_mean_weights(Eigen::MatrixXd mu);

/*** variance of latent variables ***/
Eigen::ArrayXd var_beta() const;
Eigen::ArrayXd var_gam() const;
Eigen::ArrayXd var_weights() const;
Eigen::ArrayXd mean_beta_sq(int u0) const;
Eigen::ArrayXd mean_gam_sq(int u0) const;

/*** Updates ***/
Gaussian weights_l_step(long ll, double EXty, double EXtX, Hyps hyps) const;
Gaussian covar_c_step(long cc, double EXty, double EXtX, Hyps hyps) const;
MoGaussian gamma_j_step(long jj, double EXty, double EXtX, Hyps hyps);
MoGaussian beta_j_step(long jj, double EXty, double EXtX, Hyps hyps);

/*** KL divergence ***/
double kl_div_beta(const Hyps& hyps) const;
double kl_div_gamma(const Hyps& hyps) const;
double kl_div_covars(const Hyps& hyps) const;
double kl_div_weights(const Hyps& hyps) const;

/*** Misc ***/
void check_nan(const double& alpha, const std::uint32_t& ii);
};

// Store subset of Variational Parameters to be RAM efficient
class VariationalParametersLite : public VariationalParamsBase {
public:
// Other quantities to track
EigenDataVector yx;            // N x 1
EigenDataVector ym;            // N x 1
EigenDataVector eta;
EigenDataVector eta_sq;

VariationalParametersLite(parameters my_params) : VariationalParamsBase(my_params) {
};
};

class VariationalParameters : public VariationalParamsBase {
public:
// This stores parameters used in VB and some summary quantities that
// depend on those parameters.

// Summary quantities
EigenRefDataVector yx;              // N x 1
EigenRefDataVector ym;              // N x 1
EigenRefDataVector eta;             // expected value of matrix product E x w
EigenRefDataVector eta_sq;          // expected value (E x w) cdot (E x w)

Eigen::ArrayXd EdZtZ;           // expectation of the diagonal of Z^t Z


VariationalParameters(parameters my_params,
                      EigenRefDataVector my_ym,
                      EigenRefDataVector my_yx,
                      EigenRefDataVector my_eta,
                      EigenRefDataVector my_eta_sq) : VariationalParamsBase(my_params), yx(my_yx), ym(my_ym),
	eta(my_eta), eta_sq(my_eta_sq){
};

~VariationalParameters(){
};

void init_from_lite(const VariationalParametersLite& init);

VariationalParametersLite convert_to_lite();

void calcEdZtZ(const Eigen::Ref<const Eigen::ArrayXXd>& dXtEEX, const int& n_env);
};

#endif
