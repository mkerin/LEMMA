//
// Created by kerin on 13/11/2018.
//

#ifndef BGEN_PROG_HYPS_HPP
#define BGEN_PROG_HYPS_HPP

#include "tools/eigen3.3/Dense"
#include "parameters.hpp"

#include <boost/iostreams/filtering_stream.hpp>

#include <iostream>
#include <iomanip>
#include <random>

namespace boost_io = boost::iostreams;

class Hyps {
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
	int n_effects;

	Hyps(const parameters& my_params) : p(my_params) {
	};

	void resize(int my_n_effects){
		n_effects = my_n_effects;
		slab_var.resize(n_effects);
		spike_var.resize(n_effects);
		slab_relative_var.resize(n_effects);
		spike_relative_var.resize(n_effects);
		lambda.resize(n_effects);
		s_x.resize(n_effects);
	}
	void init_from_grid(const int& n_effects,
	                    const int& ii,
	                    const long& n_var,
	                    const Eigen::Ref<const Eigen::MatrixXd>& hyps_grid);
	void update_pve();
	double normL2() const;
	bool domain_is_valid() const;
	void random_init(int n_effects, long n_var);

	/*** IO ***/
	void to_file(const std::string &path) const;
	void from_file(const std::string &filename);
	friend std::ostream& operator<< (std::ostream &os, const Hyps& hyps);

	/*** Deprecated ***/
	Eigen::VectorXd get_sigmas(double h_b, double h_g, double lam_b, double lam_g,
							   double f1_b, double f1_g, long n_var) const;
	Eigen::VectorXd get_sigmas(double h_b, double lam_b, double f1_b, long n_var) const;
};

Hyps operator+(const Hyps &h1, const Hyps &h2);
Hyps operator-(const Hyps &h1, const Hyps &h2);
Hyps operator*(const double &scalar, const Hyps &h1);

#endif //BGEN_PROG_HYPS_HPP
