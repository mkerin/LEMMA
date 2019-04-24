//
// Created by kerin on 13/11/2018.
//

#ifndef BGEN_PROG_HYPS_HPP
#define BGEN_PROG_HYPS_HPP

#include <iostream>
#include "tools/Eigen/Dense"
#include "parameters.hpp"

#include <boost/iostreams/filtering_stream.hpp>

#include <iomanip>

namespace boost_io = boost::iostreams;

class Hyps {
	int sigma_ind   = 0;
	int sigma_b_ind = 1;
	int sigma_g_ind = 2;
	int lam_b_ind   = 3;
	int lam_g_ind   = 4;

public:
	double sigma, sigma_w, sigma_c;
	Eigen::ArrayXd slab_var;
	Eigen::ArrayXd spike_var;
	Eigen::ArrayXd slab_relative_var;
	Eigen::ArrayXd spike_relative_var;
	Eigen::ArrayXd lambda;

// Not hyperparameters, but things that depend on them
	Eigen::ArrayXd s_x;
	Eigen::ArrayXd pve;

	parameters p;
	int n_effects;

	Hyps(parameters my_params) : p(my_params) {
		sigma_w = 1.0;
		sigma_c = 10000.0;
	};

	void init_from_grid(int n_effects,
	                    int ii,
	                    int n_var,
	                    const Eigen::Ref<const Eigen::MatrixXd>& hyps_grid);
//void update_pve();
	void read_from_dump(const std::string& filename);
	double normL2() const;
	bool domain_is_valid() const;

	friend std::ostream& operator<< (std::ostream &os, const Hyps& hyps);
	friend boost_io::filtering_ostream& operator<< (boost_io::filtering_ostream &os, const Hyps& hyps);
};

Hyps operator+(const Hyps &h1, const Hyps &h2);
Hyps operator-(const Hyps &h1, const Hyps &h2);
Hyps operator*(const double &scalar, const Hyps &h1);

#endif //BGEN_PROG_HYPS_HPP
