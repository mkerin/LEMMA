//
// Created by kerin on 2019-03-01.
//
#include "hyps.hpp"
#include "file_utils.hpp"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace boost_io = boost::iostreams;

uint getPos(std::vector<std::string> vars, const std::string& var){
	auto it = std::find(vars.begin(), vars.end(), var);
	return it - vars.begin();
}

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

void Hyps::init_from_grid(const int& my_n_effects,
                          const int& ii,
                          const long& n_var,
                          const Eigen::Ref<const Eigen::MatrixXd> &hyps_grid) {
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

void Hyps::to_file(const std::string &path) const {
	boost_io::filtering_ostream outf;
	fileUtils::fstream_init(outf, path);

	std::vector<std::string> effects = {"_beta", "_gam"};
	int n_effects = lambda.rows();
	outf << std::scientific << std::setprecision(7);
	outf << "hyp value" << std::endl;
	outf << "sigma " << sigma << std::endl;
	for (int ii = 0; ii < n_effects; ii++) {
		outf << "lambda" << ii+1 << " " << lambda[ii] << std::endl;
	}
	for (int ii = 0; ii < n_effects; ii++) {
		outf << "sigma" << effects[ii] << "0 " << slab_relative_var[ii] << std::endl;
	}
	for (int ii = 0; ii < n_effects; ii++) {
		if((ii == 0 && p.mode_mog_prior_beta) || (ii == 1 && p.mode_mog_prior_gam)) {
			outf << "sigma" << effects[ii] << "1 " << spike_relative_var[ii] << std::endl;
		}
	}
	boost_io::close(outf);
}

void Hyps::from_file(const std::string &filename){
	boost_io::filtering_istream fg;
	std::string gz_str = ".gz";
	if (filename.find(gz_str) != std::string::npos) {
		fg.push(boost_io::gzip_decompressor());
	}
	fg.push(boost_io::file_source(filename));
	if (!fg) {
		throw std::runtime_error(filename+" could not be opened");
	}

	// Read file twice to acertain number of lines
	std::string line, header, case1 = "hyp value";
	getline(fg, header);
	if (header != case1) {
		throw std::runtime_error("Unexpected header from "+filename);
	}

	std::vector<std::string> variables;
	std::vector<double> values;
	std::string s1, s2;
	while (getline(fg, line)) {
		std::stringstream ss(line);
		ss >> s1;
		ss >> s2;

		if(!s1.empty()) {
			variables.push_back(s1);
			try {
				values.push_back(stod(s2));
			} catch(...) {
				throw std::invalid_argument("stod failed");
			}
		}
	}

	std::vector< std::string > case_g = {"sigma", "lambda1", "sigma_beta0", "sigma_beta1"};
	std::vector< std::string > case_gxe = {"sigma", "lambda1", "lambda2", "sigma_beta0", "sigma_gam0",
		                                   "sigma_beta1", "sigma_gam1"};

	if(variables == case_gxe) {
		resize(2);
		sigma                 = values[getPos(case_gxe,"sigma")];
		lambda[0]             = values[getPos(case_gxe,"lambda1")];
		lambda[1]             = values[getPos(case_gxe,"lambda2")];
		slab_relative_var[0]  = values[getPos(case_gxe,"sigma_beta0")];
		slab_relative_var[1]  = values[getPos(case_gxe,"sigma_gam0")];
		spike_relative_var[0] = values[getPos(case_gxe,"sigma_beta1")];
		spike_relative_var[1] = values[getPos(case_gxe,"sigma_gam1")];
	} else if (variables == case_g) {
		resize(1);
		sigma                 = values[getPos(case_g,"sigma")];
		lambda[0]             = values[getPos(case_g,"lambda1")];
		slab_relative_var[0]  = values[getPos(case_g,"sigma_beta0")];
		spike_relative_var[0] = values[getPos(case_g,"sigma_beta1")];
	} else {
		std::cout << "Unrecognised hyps header:" << std::endl;
		for (auto sss : variables) {
			std::cout << sss << std::endl;
		}
		throw std::runtime_error("Unrecognised variable names in "+filename);
	}
	slab_var = slab_relative_var * sigma;
	spike_var = spike_relative_var * sigma;
}

void Hyps::random_init(int n_effects, long n_var) {
	std::default_random_engine generator(p.random_seed);
	std::uniform_real_distribution<double> unif_hb(0,0.5);
	std::uniform_real_distribution<double> unif_hg(0,0.1);
	std::uniform_real_distribution<double> unif_loglambda(2,std::max(3.0,1-std::log10(n_var)));

	double h_b   = unif_hb(generator);
	double h_g   = unif_hg(generator);
	double lam_b = std::pow(10,-unif_loglambda(generator));
	double lam_g = std::pow(10,-unif_loglambda(generator));

	resize(n_effects);
	if(n_effects == 1) {
		slab_relative_var << h_b / (1 - h_b) / n_var / lam_b;
		spike_relative_var << slab_relative_var / p.beta_spike_diff_factor;
		lambda << lam_b;
		s_x << n_var;
	} else {
		slab_relative_var << h_b / (1-h_b-h_g) / n_var / lam_b, h_b / (1-h_b-h_g) / n_var / lam_g;
		spike_relative_var << slab_relative_var[0] / p.beta_spike_diff_factor, slab_relative_var[1] / p.gam_spike_diff_factor;
		lambda << lam_b, lam_g;
		s_x << n_var, n_var;
	}
	sigma     = 1;
	slab_var << sigma * slab_relative_var;
	spike_var << sigma * spike_relative_var;
}

Eigen::VectorXd Hyps::get_sigmas(double h_b, double lam_b, double f1_b, long n_var) const {
	Eigen::MatrixXd tmp(2, 2);
	Eigen::VectorXd rhs(2);

	double P = n_var;
	tmp << P * lam_b * (1 - h_b), P * (1 - lam_b) * (1 - h_b),
	    lam_b * (f1_b - 1), f1_b * (1 - lam_b);
	rhs << h_b, 0;
	Eigen::VectorXd soln = tmp.inverse() * rhs;
	return soln;
}

Eigen::VectorXd
Hyps::get_sigmas(double h_b, double h_g, double lam_b, double lam_g, double f1_b, double f1_g, long n_var) const {
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
