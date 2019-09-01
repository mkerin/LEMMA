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
	Eigen::MatrixXd _XXtz, _XXtWz;
	double ytXXty;
	std::string label;
	long n_env;
	long n_covar;
	long n_samples;
	long n_draws;
	long n_jacknife_local;
	long rm_jacknife_block;
	double n_var_local;
	parameters params;
	bool is_active;
	std::string group;
	std::string effect_type;

	// Storage for jacknife blocks
	std::vector<Eigen::MatrixXd> _XXtzs, _XXtWzs;
	std::vector<double> n_vars_local;
	std::vector<double> ytXXtys;

	Eigen::MatrixXd& C, CtC_inv, zz, Wzz;
	Eigen::VectorXd Y;
	Eigen::VectorXd eta;
	RHEreg_Component(const parameters& myparams,
	                 Eigen::VectorXd& myY,
	                 Eigen::MatrixXd& myzz,
	                 Eigen::MatrixXd& myWzz,
	                 Eigen::MatrixXd& myC,
	                 Eigen::MatrixXd& myCtC_inv,
	                 const long& myNJacknifeLocal);

	void set_eta(Eigen::Ref<Eigen::VectorXd> myeta);

	void set_inactive();

	void add_to_trace_estimator(Eigen::Ref<Eigen::MatrixXd> X,
	                            long jacknife_index = 0);

	void finalise();

	Eigen::MatrixXd getXXtz() const;

	Eigen::MatrixXd getXXtWz() const;

	double get_bb_trace() const;

	double get_n_var_local() const;

	double operator*(const RHEreg_Component& other) const;

	Eigen::MatrixXd project_out_covars(Eigen::Ref<Eigen::MatrixXd> rhs) const;
};

#endif
