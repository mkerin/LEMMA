#ifndef VARIATIONAL_PARAMETERS_HPP
#define VARIATIONAL_PARAMETERS_HPP

#include "genotype_matrix.hpp"
#include "parameters.hpp"
#include "typedefs.hpp"
#include "hyps.hpp"
#include "Prior.hpp"

#include "tools/Eigen/Dense"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/filesystem.hpp>

#include <iostream>
#include <limits>
#include <memory>

namespace boost_io = boost::iostreams;

class VariationalParameters {

public:
	// This stores parameters used in VB and some summary quantities that
	// depend on those parameters.
	double eps = std::numeric_limits<double>::min();

	GaussianVec weights;
	GaussianVec weights_momentum;
	GaussianVec covars;
	parameters p;

	std::unique_ptr<ExponentialFamVec> betas;
	std::unique_ptr<ExponentialFamVec> gammas;



	// Estimated variance of phenotype.
	// Not sure where else to put.
	double sigma;
	Eigen::ArrayXd pve;
	Eigen::ArrayXd s_x;
	Eigen::ArrayXd EdZtZ;

	EigenArrayXl current_minibatch_index;
	EigenDataVector y;
	EigenDataMatrix E, C;

	// Other quantities to track
	EigenDataVector ym;                        // N x 1
	EigenDataVector yx;                        // N x 1
	EigenDataVector eta;
	EigenDataVector eta_sq;

	VariationalParameters(parameters my_params) : p(my_params) {
		if (p.beta_prior == "MixOfGaussian") {
			betas.reset(new MoGaussianVec);
		} else if (p.beta_prior == "Gaussian") {
			betas.reset(new GaussianVec);
		} else {
			throw std::runtime_error("Unrecognised beta prior: " + p.beta_prior);
		}
		if (p.beta_prior == "MixOfGaussian") {
			gammas.reset(new MoGaussianVec);
		} else if (p.beta_prior == "Gaussian") {
			gammas.reset(new GaussianVec);
		} else {
			throw std::runtime_error("Unrecognised beta prior: " + p.beta_prior);
		}
	};
	VariationalParameters(const VariationalParameters &obj) : p(obj.p) {
		y      = obj.y;
		ym     = obj.ym;
		yx     = obj.yx;
		eta    = obj.eta;
		eta_sq = obj.eta_sq;

		sigma  = obj.sigma;
		pve    = obj.pve;
		s_x    = obj.s_x;
		EdZtZ = obj.EdZtZ;

		betas.reset(obj.betas->clone());
		gammas.reset(obj.gammas->clone());
		weights = obj.weights;
		covars  = obj.covars;
		weights_momentum = obj.weights_momentum;
	}
	VariationalParameters& operator = (const VariationalParameters &obj) {
		p      = obj.p;
		ym     = obj.ym;
		yx     = obj.yx;
		y      = obj.y;
		eta    = obj.eta;
		eta_sq = obj.eta_sq;

		sigma  = obj.sigma;
		pve    = obj.pve;
		s_x    = obj.s_x;
		EdZtZ = obj.EdZtZ;

		betas.reset(obj.betas->clone());
		gammas.reset(obj.gammas->clone());
		weights = obj.weights;
		covars  = obj.covars;
		weights_momentum = obj.weights_momentum;
		return *this;
	}

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

	/*** variance of latent variables ***/
	Eigen::ArrayXd var_beta() const;
	Eigen::ArrayXd var_gam() const;
	Eigen::ArrayXd var_weights() const;
	Eigen::ArrayXd var_covar() const;

	double var_weights(long jj) const;
	double var_covar(long jj) const;

	/*** Misc ***/
	void update_summary_vars(const EigenArrayXl index, EigenRefDataVector my_y, EigenRefDataMatrix my_C,
			EigenDataMatrix my_E, const GenotypeMatrix& X){
		assert(p.mode_svi);
		assert(X.minibatch_index_set);
		current_minibatch_index = index;

		// Eigen flexi indexing quite sensitive...
		Eigen::VectorXd mean_vec;

		y = my_y(index);
		E = my_E(index, Eigen::all);


		mean_vec = betas->mean();
		ym = X * mean_vec;
		if(covars.size() > 0){
			mean_vec = covars.mean();
			ym += my_C(index, ":") * covars.mean();
		}

		if(gammas->size() > 0) {
			mean_vec = gammas->mean();
			yx = X * mean_vec;
		}
		if(weights.size() > 0){
			mean_vec = weights.mean();
			Eigen::VectorXd eta = my_E(index, ":") * mean_vec;
			eta_sq  = eta.array().square().matrix();
			eta_sq += my_E(index, ":").array().square().matrix() * weights.var().matrix();
		}



	}
	void set_hyps(Hyps hyps);
	void check_nan(const double& alpha, const std::uint32_t& ii);
	void write_ith_beta_to_stream(long ii, std::ostream& outf) const {
		betas->write_ith_distn_to_stream(ii, outf);
	}
	void write_ith_gamma_to_stream(long ii, std::ostream& outf) const {
		gammas->write_ith_distn_to_stream(ii, outf);
	}
	std::string betas_header(std::string prefix = "") const {
		return betas->header(prefix);
	}
	std::string gammas_header(std::string prefix = "") const {
		return gammas->header(prefix);
	}
	void update_pve(long n_env){
		pve[0] = betas->get_hyps_var() * s_x[0];
		if (n_env > 0) {
			pve[1] = gammas->get_hyps_var() * s_x[1];
		}
		pve /= (pve.sum() + sigma);
	}

	void calcEdZtZ(const Eigen::Ref<const Eigen::ArrayXXd>& dXtEEX, const int& n_env);
};

//// Store subset of Variational Parameters to be RAM efficient
//class VariationalParameters : public VariationalParameters {
//public:
//
//	explicit VariationalParameters(parameters my_params) : VariationalParameters(my_params) {};
//
//};

#endif
