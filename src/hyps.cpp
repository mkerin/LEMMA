//
// Created by kerin on 2019-03-01.
//
#include "hyps.hpp"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include <cmath>
#include <string>
#include <vector>

namespace boost_io = boost::iostreams;

double Hyps::normL2() const {
	double res = sigma * sigma;
	res += slab_relative_var.square().sum();
	res += spike_relative_var.square().sum();
	res += lambda.square().sum();
	return std::sqrt(res);
}

bool Hyps::domain_is_valid() const {
	bool res = true;
	if (sigma <= 0) res = false;
	if (slab_relative_var.minCoeff() <= 0) res = false;
	if (slab_var.minCoeff() <= 0) res = false;
	if (lambda.minCoeff() <= 0) res = false;
	if (lambda.maxCoeff() >= 1) res = false;
	if(p.mode_mog_prior_beta || p.mode_mog_prior_gam) {
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
	if(h1.p.mode_mog_prior_beta || h1.p.mode_mog_prior_gam) {
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

void Hyps::init_from_grid(int my_n_effects, int ii, int n_var, const Eigen::Ref<const Eigen::MatrixXd> &hyps_grid) {
	// Unpack
	double my_sigma = hyps_grid(ii, sigma_ind);
	double my_sigma_b = hyps_grid(ii, sigma_b_ind);
	double my_sigma_g = hyps_grid(ii, sigma_g_ind);
	double my_lam_b = hyps_grid(ii, lam_b_ind);
	double my_lam_g = hyps_grid(ii, lam_g_ind);

	// Assign initial hyps
	resize(my_n_effects);
	if(n_effects == 1) {
		sigma = my_sigma;
		slab_var << my_sigma * my_sigma_b;
		spike_var << my_sigma * my_sigma_b / p.beta_spike_diff_factor;
		slab_relative_var << my_sigma_b;
		spike_relative_var << my_sigma_b / p.beta_spike_diff_factor;
		lambda << my_lam_b;
		s_x << n_var;
	} else if(n_effects == 2) {
		sigma = my_sigma;
		slab_var << my_sigma * my_sigma_b, my_sigma * my_sigma_g;
		spike_var << my_sigma * my_sigma_b / p.beta_spike_diff_factor, my_sigma * my_sigma_g / p.gam_spike_diff_factor;
		slab_relative_var << my_sigma_b, my_sigma_g;
		spike_relative_var << my_sigma_b / p.beta_spike_diff_factor, my_sigma_g / p.gam_spike_diff_factor;
		lambda << my_lam_b, my_lam_g;
		// Intentionally not setting s_x here
		// Will get updated when updates env-weights
	} else {
		// Not implemented
		assert(false);
	}
}

void Hyps::update_pve(){
	// Compute heritability

	pve = lambda * slab_relative_var * s_x;
	pve_large = pve;
	if(p.mode_mog_prior_beta) {
		int ee = 0;
		pve[ee] += (1 - lambda[ee]) * spike_relative_var[ee] * s_x[ee];
	}

	if (p.mode_mog_prior_gam && n_effects > 1) {
		int ee = 1;
		pve[ee] += (1 - lambda[ee]) * spike_relative_var[ee] * s_x[ee];
	}

	pve_large /= (pve.sum() + 1.0);
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
	for (int ii = 0; ii < n_effects; ii++) {
		os << "lambda" << ii+1 << " " << hyps.lambda[ii] << std::endl;
	}
	for (int ii = 0; ii < n_effects; ii++) {
		os << "sigma" << effects[ii] << "0 " << hyps.slab_relative_var[ii] << std::endl;
	}
	for (int ii = 0; ii < n_effects; ii++) {
		if((ii == 0 && hyps.p.mode_mog_prior_beta) || (ii == 1 && hyps.p.mode_mog_prior_gam)) {
			os << "sigma" << effects[ii] << "1 " << hyps.spike_relative_var[ii] << std::endl;
		}
	}
	return os;
}

void Hyps::read_from_dump(const std::string& filename){
	// TODO: Better to match label to value rather than assuming order

	boost_io::filtering_istream fg;
	std::string gz_str = ".gz";
	if (filename.find(gz_str) != std::string::npos) {
		fg.push(boost_io::gzip_decompressor());
	}
	fg.push(boost_io::file_source(filename));
	if (!fg) {
		std::cout << "ERROR: " << filename << " not opened." << std::endl;
		std::exit(EXIT_FAILURE);
	}

	// Read file twice to acertain number of lines
	std::string line, header, case1 = "hyp value";
	std::vector<std::string> variables;
	std::vector<double> values;

	getline(fg, header);
	assert(header == case1);
	std::stringstream ss;
	while (getline(fg, line)) {
		std::string s1, s2;
		ss.clear();
		ss.str(line);
		// skip variable name
		ss >> s1;
		ss >> s2;

		if(s1 != "") {
			variables.push_back(s1);
			try {
				values.push_back(stod(s2));
			} catch(...) {
				std::cout << "Caught exception from Hyps::read_from_dump" << std::endl;
				throw std::invalid_argument("stod failed");
			}
		}
	}

	std::vector< std::string > case_g = {"sigma", "lambda1", "sigma_beta0", "sigma_beta1"};
	std::vector< std::string > case_gxe = {"sigma", "lambda1", "lambda2", "sigma_beta0", "sigma_gam0",
		                                   "sigma_beta1", "sigma_gam1"};

	if(variables == case_gxe) {
		resize(2);

		sigma = values[0];
		lambda[0] = values[1];
		lambda[1] = values[2];
		slab_relative_var[0] = values[3];
		slab_relative_var[1] = values[4];
		spike_relative_var[0] = values[5];
		spike_relative_var[1] = values[6];

		slab_var = slab_relative_var * sigma;
		spike_var = spike_relative_var * sigma;
	} else if (variables == case_g) {
		resize(1);

		sigma = values[0];
		lambda[0] = values[1];
		slab_relative_var[0] = values[2];
		spike_relative_var[0] = values[3];
	} else {
		std::cout << "Unrecognised hyps header:" << std::endl;
		for (auto sss : variables) {
			std::cout << sss << std::endl;
		}
		throw std::runtime_error("Unrecognised hyps header");
	}
}
