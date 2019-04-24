//
// Created by kerin on 2019-04-17.
//
#include "Prior.hpp"

#include "tools/Eigen/Dense"

#include <cmath>
#include <stdexcept>
#include <vector>

/*** Vectorised exponential family ***/

Eigen::ArrayXd ExponentialFamVec::nats_to_mean(const Eigen::ArrayXXd& nats) const {
	return -nats.col(0) / nats.col(1) / 2;
}

Eigen::ArrayXd ExponentialFamVec::nats_to_var(const Eigen::ArrayXXd& nats) const {
	return -0.5 / nats.col(1);
}

Eigen::ArrayXXd ExponentialFamVec::mean_var_to_nats(const Eigen::ArrayXd& mu, const Eigen::ArrayXd& sig) {
	assert(mu.rows() == sig.rows());
	Eigen::ArrayXXd res(mu.rows(), 2);
	res.col(1) = -0.5 * sig.inverse();
	res.col(0) = -2.0 * res.col(1) * mu;
	return res;
}

Eigen::ArrayXd ExponentialFamVec::nats_to_kl_div(const Eigen::ArrayXXd& nats, double prior_var) const {
	Eigen::ArrayXd res;
	res = nats_to_var(nats).log();
	res -= (nats_to_var(nats) + nats_to_mean(nats).square()) / prior_var;
	res += (1.0 - std::log(prior_var));
	res *= 0.5;
	return res;
}

double ExponentialFamVec::nats_to_mean(long ii, const Eigen::ArrayXXd& nats) const {
	return -nats(ii, 0) / nats(ii, 1) / 2;
}

double ExponentialFamVec::nats_to_var(long ii, const Eigen::ArrayXXd& nats) const {
	return -0.5 / nats(ii, 1);
}

/*** GaussianVec ***/
void GaussianVec::resize(long n_vars) {
	// Initialise all gaussians at mean 0, var 0.01
	nn = n_vars;
	nats.resize(nn, 2);
	nats.col(0) = Eigen::ArrayXd::Zero(nn);
	nats.col(1) = Eigen::ArrayXd::Constant(nn, -50);
}

std::string GaussianVec::header(std::string prefix) const {
	std::vector<std::string> vv = {"mu", "s_sq"};
	if(prefix.length() > 0) {
		for (auto& ss: vv) {
			ss += "_" + prefix;
		}
	}
	std::string res;
	for (int ii = 0; ii < vv.size(); ii++) {
		res += vv[ii];
		if(ii != vv.size() - 1) {
			res += " ";
		}
	}
	return res;
}

void GaussianVec::read_from_grid(const Eigen::MatrixXd& grid) {
	assert(grid.cols() == 2);
	nn = grid.rows();
	nats = mean_var_to_nats(grid.col(0), grid.col(1));
}
