//
// Created by kerin on 2019-09-30.
//

#ifndef BGEN_PROG_RHE_REG_COMPONENT_HPP
#define BGEN_PROG_RHE_REG_COMPONENT_HPP

#include "genotype_matrix.hpp"
#include "file_utils.hpp"
#include "parameters.hpp"
#include "eigen_utils.hpp"
#include "mpi_utils.hpp"
#include "data.hpp"
#include "eigen_utils.hpp"

#include "tools/eigen3.3/Dense"

#include <boost/iostreams/filtering_stream.hpp>

#include <random>

class RHEreg_Component {
public:
	Eigen::MatrixXd _XXtWz;
	double ytXXty;
	long n_covar;
	long n_samples;
	long n_draws;
	long n_jacknife_local;
	long rm_jacknife_block;
	double n_var_local;
	parameters params;

	// Identifiers
	std::string label;
	bool is_gxe;
	bool is_active;
	bool is_finalised;
	std::string group;
	std::string effect_type;
	long env_var_index;

	// Storage for jacknife blocks
	std::vector<Eigen::MatrixXd> _XXtzs;
	std::vector<double> n_vars_local;
	std::vector<double> ytXXtys;

	Eigen::MatrixXd C, CtC_inv;
	Eigen::MatrixXd zz;
	Eigen::VectorXd Y;
	Eigen::VectorXd env_var;
	RHEreg_Component(const parameters& myparams,
	                 const Eigen::VectorXd& myY,
	                 const Eigen::MatrixXd& myWzz,
	                 const Eigen::MatrixXd& myC,
	                 const Eigen::MatrixXd& myCtC_inv,
	                 const long& myNJacknifeLocal);

	RHEreg_Component(const parameters& myparams,
	                 const Eigen::VectorXd& myY,
	                 const Eigen::MatrixXd& myC,
	                 const Eigen::MatrixXd& myCtC_inv,
	                 Eigen::MatrixXd XXtWz) : params(myparams), Y(myY),
		C(myC), CtC_inv(myCtC_inv),
		_XXtWz(XXtWz){
		// For aggregating components after genotypes have been streamed
		// ie no need for any of the zz data fields.
		n_jacknife_local = 1;
		n_draws = XXtWz.cols();
		n_samples = XXtWz.rows();
		n_covar = C.cols();

		ytXXty = 0;
		n_var_local = 0;
		label = "";
		is_active = true;
		is_gxe = false;
		is_finalised = false;
		rm_jacknife_block = -1;
	}

	void set_env_var(const Eigen::Ref<const Eigen::VectorXd>& my_env_var);

	void change_env_var(const Eigen::Ref<const Eigen::VectorXd>& my_env_var);

	void set_inactive();

	void add_to_trace_estimator(Eigen::Ref<Eigen::MatrixXd> X,
	                            long jacknife_index = 0);

	void finalise();

	Eigen::MatrixXd getXXtz() const;

	double get_bb_trace() const;

	double get_n_var_local() const;

	double operator*(const RHEreg_Component& other) const;

	Eigen::MatrixXd project_out_covars(Eigen::Ref<Eigen::MatrixXd> rhs) const;
};

void aggregate_GxE_components(const std::vector<RHEreg_Component>& vec_of_components,
                              RHEreg_Component& new_comp,
                              const Eigen::Ref<const Eigen::MatrixXd>& E,
                              const Eigen::Ref<const Eigen::VectorXd>& env_weights,
                              const Eigen::Ref<const Eigen::MatrixXd>& ytEXXtEy);


#endif
