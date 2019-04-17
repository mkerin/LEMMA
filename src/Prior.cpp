//
// Created by kerin on 2019-04-17.
//
#include "Prior.hpp"

#include "tools/eigen3.3/Dense"

#include <cmath>
#include <stdexcept>
#include <vector>

/*** Prior ***/

double Prior::nats_to_mean(Eigen::ArrayXd nats) const {
	return -1.0 * nats[0] / nats[1] / 2;
}

double Prior::nats_to_var(Eigen::ArrayXd nats) const {
	assert(nats[1] < 0);
	return -1.0 / nats[1] / 2;
}

double Prior::sigmoid(const double &x) {
	return 1.0 / (1.0 + std::exp(-x));
}

/*** Gaussian ***/

std::ostream& operator<< (std::ostream &out, const Gaussian &dist){
	out << dist.mean() << " " << dist.var();
	// out << dist.nats_to_mean(dist.nats) << " " << dist.nats_to_var(dist.nats);
	return out;
}

double Gaussian::mean() const {
	return -1.0 * nats[0] / nats[1] / 2;
}

double Gaussian::var() const {
	assert(nats[1] < 0);
	return -1.0 / nats[1] / 2;
}

/*** Mixture of Gaussians ***/

std::ostream& operator<< (std::ostream &out, const MoGaussian &dist){
	out << dist.mix;
	out << " " << dist.slab << " " << dist.spike;
	// out << " " << dist.nats_to_mean(dist.slab_nats) << " " << dist.nats_to_var(dist.slab_nats);
	// out << " " << dist.nats_to_mean(dist.spike_nats) << " " << dist.nats_to_var(dist.spike_nats);
	return out;
}

void MoGaussian::maximise_mix_coeff(const double &lambda, const double &sigma0, const double &sigma1) {
	// index 0; slab
	// index 1; spike
	double alpha_cnst;
	alpha_cnst  = std::log(lambda / (1.0 - lambda) + eps);
	alpha_cnst -= (std::log(sigma0) - std::log(sigma1)) / 2.0;

	double mu0 = slab.mean();
	double mu1 = spike.mean();
	double s0 = slab.var();
	double s1 = spike.var();
	double ff_k;
	ff_k = mu0 * mu0 / s0;
	ff_k += std::log(s0);
	ff_k -= mu1 * mu1 / s1;
	ff_k -= std::log(s1);
	mix  = sigmoid(ff_k / 2.0 + alpha_cnst);
}

MoGaussian &MoGaussian::operator=(const MoGaussian &gg) {
	// Copy Assignment operator
	// https://www.geeksforgeeks.org/copy-constructor-vs-assignment-operator-in-c/
	slab = gg.slab;
	spike = gg.spike;
	mix = gg.mix;
	return *this;
}

/*** Vectorised exponential family ***/

Eigen::ArrayXd ExponentialFamVec::nats_to_mean(Eigen::ArrayXXd nats) const {
	return -nats.col(0) / nats.col(1) / 2;
}

Eigen::ArrayXd ExponentialFamVec::nats_to_var(Eigen::ArrayXXd nats) const {
	return -0.5 / nats.col(1);
}

Eigen::ArrayXd ExponentialFamVec::nats_to_mean_sq(Eigen::ArrayXXd nats) const {
	return nats_to_mean(nats).square() + nats_to_var(nats);
}

Eigen::ArrayXXd ExponentialFamVec::mean_var_to_nats(Eigen::ArrayXd mu, Eigen::ArrayXd sig) {
	assert(mu.rows() == sig.rows());
	Eigen::ArrayXXd res(mu.rows(), 2);
	res.col(1) = -0.5 * sig.inverse();
	res.col(0) = -2.0 * res.col(1) * mu;
	return res;
}

Eigen::ArrayXd ExponentialFamVec::nats_to_kl_div(Eigen::ArrayXXd nats, double prior_var) const {
	Eigen::ArrayXd res;
	res = nats_to_var(nats).log();
	res -= (nats_to_var(nats) + nats_to_mean(nats).square()) / prior_var;
	res += (1.0 - std::log(prior_var));
	res *= 0.5;
	return res;
}

double ExponentialFamVec::nats_to_mean(long ii, Eigen::ArrayXXd nats) const {
	return -nats(ii, 0) / nats(ii, 1) / 2;
}

double ExponentialFamVec::nats_to_var(long ii, Eigen::ArrayXXd nats) const {
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

Gaussian GaussianVec::get_ith_distn(long ii) const {
	Gaussian dist;
	dist.nats << nats(ii, 0), nats(ii, 1);
	return dist;
}

void GaussianVec::set_ith_distn(long ii, Gaussian new_dist) {
	nats(ii, 0) = new_dist.nats(0);
	nats(ii, 1) = new_dist.nats(1);
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

void GaussianVec::read_from_grid(Eigen::MatrixXd grid) {
	assert(grid.cols() == 2);
	nn = grid.rows();
	nats = mean_var_to_nats(grid.col(0), grid.col(1));
}
