//
// Created by kerin on 13/11/2018.
//

#ifndef BGEN_PROG_HYPS_HPP
#define BGEN_PROG_HYPS_HPP

#include <iostream>
#include "tools/eigen3.3/Dense"
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

	Hyps(parameters my_params) : p(my_params) {
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
	void read_from_dump(const std::string& filename);
	double normL2() const;
	bool domain_is_valid() const;
	Eigen::VectorXd get_sigmas(double h_b, double h_g, double lam_b, double lam_g,
							   double f1_b, double f1_g, long n_var) const {
		Eigen::MatrixXd tmp(4, 4);
		Eigen::VectorXd rhs(4);

		double P = n_var;
		tmp << P * lam_b * (1 - h_b), P * (1 - lam_b) * (1 - h_b), -P * h_b * lam_g, -P * h_b * (1 - lam_g),
				lam_b * (f1_b - 1), f1_b * (1 - lam_b), 0, 0,
				-P * h_g * lam_b, -P * h_g * (1 - lam_b), P * lam_g * (1 - h_g), P * (1 - lam_g) * (1 - h_g),
				0, 0, lam_g * (f1_g - 1), f1_g * (1 - lam_g);
		rhs << h_b, 0, h_g, 0;
		Eigen::VectorXd soln = tmp.inverse() * rhs;
		return soln;
	}
	Eigen::VectorXd get_sigmas(double h_b, double lam_b, double f1_b, long n_var) const {
		Eigen::MatrixXd tmp(2, 2);
		Eigen::VectorXd rhs(2);

		double P = n_var;
		tmp << P * lam_b * (1 - h_b), P * (1 - lam_b) * (1 - h_b),
				lam_b * (f1_b - 1), f1_b * (1 - lam_b);
		rhs << h_b, 0;
		Eigen::VectorXd soln = tmp.inverse() * rhs;
		return soln;
	}
	void use_default_init(int n_effects, long n_var){
		double lam_b = 0.01;
		double lam_g = 0.01;
		double f1_b = 0.05;
		double f1_g = 0.05;
		double h_b = 0.1;
		double h_g = 0.05;

		resize(n_effects);
		if(n_effects == 1) {
			Eigen::VectorXd soln = get_sigmas(h_b, lam_b, f1_b, n_var);
			double my_sigma = 1.0 - h_b - h_g;
			sigma = my_sigma;
			slab_var << my_sigma * soln[0];
			spike_var << my_sigma * soln[1];
			slab_relative_var << soln[0];
			spike_relative_var << soln[1];
			lambda << lam_b;
			s_x << n_var;
		} else {
			Eigen::VectorXd soln = get_sigmas(h_b, h_g, lam_b, lam_g, f1_b, f1_g, n_var);
			double my_sigma = 1.0 - h_b - h_g;
			sigma = my_sigma;
			slab_var << my_sigma * soln[0], my_sigma * soln[2];
			spike_var << my_sigma * soln[1], my_sigma * soln[3];
			slab_relative_var << soln[0], soln[2];
			spike_relative_var << soln[1], soln[3];
			lambda << lam_b, lam_g;
			s_x << n_var, n_var;
		}
	}

	friend std::ostream& operator<< (std::ostream &os, const Hyps& hyps);
	friend boost_io::filtering_ostream& operator<< (boost_io::filtering_ostream &os, const Hyps& hyps);
};

Hyps operator+(const Hyps &h1, const Hyps &h2);
Hyps operator-(const Hyps &h1, const Hyps &h2);
Hyps operator*(const double &scalar, const Hyps &h1);

#endif //BGEN_PROG_HYPS_HPP
