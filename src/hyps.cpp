//
// Created by kerin on 2019-03-01.
//
#include "hyps.hpp"

#include <cmath>

double Hyps::normL2() const {
	double res = sigma * sigma;
	res += slab_relative_var.square().sum();
	res += spike_relative_var.square().sum();
	res += lambda.square().sum();
	return std::sqrt(res);
}

bool Hyps::check_valid_domain() const {
	bool res = true;
	if (sigma <= 0) res = false;
	if (slab_relative_var.minCoeff() <= 0) res = false;
	if (slab_var.minCoeff() <= 0) res = false;
	if (lambda.minCoeff() <= 0) res = false;
	if (lambda.maxCoeff() >= 1) res = false;
	if(p.mode_mog_prior_beta || p.mode_mog_prior_gam){
		if (spike_var.minCoeff() <= 0) res = false;
		if (spike_relative_var.minCoeff() <= 0) res = false;
	}
	return res;
}

// Note: Should this return class hyps!? s_x not defined?
Hyps operator+(const Hyps &h1, const Hyps &h2){
	Hyps hyps(h1.p);
	hyps.sigma = h1.sigma + h2.sigma;
	hyps.slab_var = h1.slab_var + h2.slab_var;
	hyps.slab_relative_var = h1.slab_relative_var + h2.slab_relative_var;
	hyps.lambda = h1.lambda + h2.lambda;
	if(h1.p.mode_mog_prior_beta || h1.p.mode_mog_prior_gam){
		hyps.spike_var = h1.spike_var + h2.spike_var;
		hyps.spike_relative_var = h1.spike_relative_var + h2.spike_relative_var;
	}
	return hyps;
}

Hyps operator-(const Hyps &h1, const Hyps &h2){
	Hyps hyps(h1.p);
	hyps.sigma = h1.sigma - h2.sigma;
	hyps.slab_var = h1.slab_var - h2.slab_var;
	hyps.spike_var = h1.spike_var - h2.spike_var;
	hyps.slab_relative_var = h1.slab_relative_var - h2.slab_relative_var;
	hyps.spike_relative_var = h1.spike_relative_var - h2.spike_relative_var;
	hyps.lambda = h1.lambda - h2.lambda;
	return hyps;
}

Hyps operator*(const double &scalar, const Hyps &h1){
	Hyps hyps(h1.p);
	hyps.sigma = scalar * h1.sigma;
	hyps.slab_var = scalar * h1.slab_var;
	hyps.spike_var = scalar * h1.spike_var;
	hyps.slab_relative_var = scalar * h1.slab_relative_var;
	hyps.spike_relative_var = scalar * h1.spike_relative_var;
	hyps.lambda = scalar * h1.lambda;
	return hyps;
}

void Hyps::init_from_grid(int n_effects, int ii, int n_var, const Eigen::Ref<const Eigen::MatrixXd> &hyps_grid) {
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
	spike_var << my_sigma * my_sigma_b / p.beta_spike_diff_factor;
	slab_relative_var << my_sigma_b;
	spike_relative_var << my_sigma_b / p.beta_spike_diff_factor;
	lambda << my_lam_b;
	s_x << n_var;
}

void Hyps::init_from_grid(int my_n_effects, int ii, int n_var, const Eigen::Ref<const Eigen::MatrixXd> &hyps_grid,
						  const double &my_s_z) {
	n_effects = my_n_effects;

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

	pve.resize(n_effects);
	pve_large.resize(n_effects);

	// Assign initial hyps
	sigma = my_sigma;
	slab_var << my_sigma * my_sigma_b, my_sigma * my_sigma_g;
	spike_var << my_sigma * my_sigma_b / p.beta_spike_diff_factor, my_sigma * my_sigma_g / p.gam_spike_diff_factor;
	slab_relative_var << my_sigma_b, my_sigma_g;
	spike_relative_var << my_sigma_b / p.beta_spike_diff_factor, my_sigma_g / p.gam_spike_diff_factor;
	lambda << my_lam_b, my_lam_g;
	s_x << n_var, my_s_z;
}

void Hyps::update_pve(){
	// Compute heritability

	pve = lambda * slab_relative_var * s_x;
	if(p.mode_mog_prior_beta){
		int ee = 0;
		pve_large[ee] = pve[ee];
		pve[ee] += (1 - lambda[ee]) * spike_relative_var[ee] * s_x[ee];

		if (p.mode_mog_prior_gam && n_effects > 1){
			int ee = 1;
			pve_large[ee] = pve[ee];
			pve[ee] += (1 - lambda[ee]) * spike_relative_var[ee] * s_x[ee];
		}

		pve_large /= (pve.sum() + 1.0);
	}
	pve /= (pve.sum() + 1.0);
}

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
