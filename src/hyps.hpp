//
// Created by kerin on 13/11/2018.
//

#ifndef BGEN_PROG_HYPS_HPP
#define BGEN_PROG_HYPS_HPP

#include <iostream>
#include "tools/eigen3.3/Dense"
#include "parameters.hpp"

#include <boost/iostreams/filtering_stream.hpp>

namespace boost_io = boost::iostreams;

class Hyps{
	int sigma_ind   = 0;
	int sigma_b_ind = 1;
	int sigma_g_ind = 2;
	int lam_b_ind   = 3;
	int lam_g_ind   = 4;

public:
	double sigma;
	Eigen::ArrayXd slab_var;
	Eigen::ArrayXd spike_var;
	Eigen::ArrayXd slab_relative_var;
	Eigen::ArrayXd spike_relative_var;
	Eigen::ArrayXd lambda;

	// Not hyperparameters, but things that depend on them
	Eigen::ArrayXd s_x;
	Eigen::ArrayXd pve;
	Eigen::ArrayXd pve_large;

	parameters p;

	Hyps(parameters my_params) : p(my_params) {};

	void init_from_grid(int n_effects,
						int ii,
						int n_var,
						const Eigen::Ref<const Eigen::MatrixXd>& hyps_grid,
						const double& my_s_z){
		// Implicit that n_effects > 1

		// Unpack
		double my_sigma = hyps_grid(ii, sigma_ind);
		double my_sigma_b = hyps_grid(ii, sigma_b_ind);
		double my_sigma_g = hyps_grid(ii, sigma_g_ind);
		double my_lam_b = hyps_grid(ii, lam_b_ind);
		double my_lam_g = hyps_grid(ii, lam_g_ind);

		// Resize
		slab_var.resize(n_effects);
		spike_var.resize(n_effects);
		slab_relative_var.resize(n_effects);
		spike_relative_var.resize(n_effects);
		lambda.resize(n_effects);
		s_x.resize(n_effects);

		// Assign initial hyps
		sigma = my_sigma;
		slab_var << my_sigma * my_sigma_b, my_sigma * my_sigma_g;
		spike_var << my_sigma * my_sigma_b / p.spike_diff_factor, my_sigma * my_sigma_g / p.spike_diff_factor;
		slab_relative_var << my_sigma_b, my_sigma_g;
		spike_relative_var << my_sigma_b / p.spike_diff_factor, my_sigma_g / p.spike_diff_factor;
		lambda << my_lam_b, my_lam_g;
		s_x << n_var, my_s_z;
	}

	void init_from_grid(int n_effects,
						int ii,
						int n_var,
						const Eigen::Ref<const Eigen::MatrixXd>& hyps_grid){
		/*** Implicit that n_effects == 1 ***/

		// Unpack
		double my_sigma = hyps_grid(ii, sigma_ind);
		double my_sigma_b = hyps_grid(ii, sigma_b_ind);
		double my_sigma_g = hyps_grid(ii, sigma_g_ind);
		double my_lam_b = hyps_grid(ii, lam_b_ind);
		double my_lam_g = hyps_grid(ii, lam_g_ind);

		// Resize
		slab_var.resize(n_effects);
		spike_var.resize(n_effects);
		slab_relative_var.resize(n_effects);
		spike_relative_var.resize(n_effects);
		lambda.resize(n_effects);
		s_x.resize(n_effects);

		// Assign initial hyps
		sigma = my_sigma;
		slab_var << my_sigma * my_sigma_b;
		spike_var << my_sigma * my_sigma_b / p.spike_diff_factor;
		slab_relative_var << my_sigma_b;
		spike_relative_var << my_sigma_b / p.spike_diff_factor;
		lambda << my_lam_b;
		s_x << n_var;
	}

	friend std::ostream& operator<< (std::ostream &os, const Hyps& hyps);
	friend boost_io::filtering_ostream& operator<< (boost_io::filtering_ostream &os, const Hyps& hyps);
};

std::ostream& operator<<(std::ostream& os, const Hyps& hyps){
	os << hyps.sigma << std::endl;
	os << hyps.lambda << std::endl;
	os << hyps.slab_var << std::endl;
	os << hyps.spike_var << std::endl;
	return os;
}

boost_io::filtering_ostream& operator<<(boost_io::filtering_ostream& os, const Hyps& hyps){
	std::vector<std::string> effects = {"_beta", "_gam"};
	int n_effects = hyps.lambda.rows();
	os << std::scientific << std::setprecision(7);
	os << "hyp value" << std::endl;
	os << "sigma " << hyps.sigma << std::endl;
	for (int ii = 0; ii < n_effects; ii++){
		os << "lambda" << ii+1 << " " << hyps.lambda[ii] << std::endl;
	}
	for (int ii = 0; ii < n_effects; ii++){
		os << "sigma" << effects[ii] << "0 " << hyps.slab_relative_var[ii] << std::endl;
	}
	for (int ii = 0; ii < n_effects; ii++) {
		if((ii == 0 && hyps.p.mode_mog_prior_beta) || (ii == 1 && hyps.p.mode_mog_prior_gam)) {
			os << "sigma" << effects[ii] << "1 " << hyps.spike_relative_var[ii] << std::endl;
		}
	}
	return os;
}

#endif //BGEN_PROG_HYPS_HPP
