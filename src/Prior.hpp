//
// Created by kerin on 2019-04-16.
//
// https://oopscenities.net/2012/08/09/reference_wrapper/

#ifndef LEMMA_PRIOR_HPP
#define LEMMA_PRIOR_HPP

#include "tools/eigen3.3/Dense"

#include <cmath>
#include <stdexcept>
#include <vector>

class Prior {
public:
	double mix;
	double eps = std::numeric_limits<double>::min();
	Eigen::ArrayXd nats, nats_slab, nats_spike;

	double sigmoid(const double& x);
	double nats_to_mean(Eigen::ArrayXd nats) const;
	double nats_to_var(Eigen::ArrayXd nats) const;
};

class Gaussian : public Prior {
public:
	Eigen::ArrayXd nats;
	Gaussian(){
		// Deliberately setting nats[1] to positive value
		// Initialisation should happen elsewhere
		nats.resize(2);
		nats << 0, 1;
	}
	double mean() const;
	double var() const;
	friend std::ostream& operator<< (std::ostream &out, const Gaussian &dist);
};

class MoGaussian : public Prior {
public:
	Gaussian slab, spike;
	Eigen::ArrayXd slab_nats, spike_nats;
	double mix;
	double eps = std::numeric_limits<double>::min();

	MoGaussian(){
		// Deliberately setting mix to invalid value
		// Initialisation should happen elsewhere
		mix = -1;
		slab_nats.resize(2);
		spike_nats.resize(2);
	}
	MoGaussian& operator = (const MoGaussian &gg);
	void maximise_mix_coeff(const double& lambda, const double& sigma0, const double& sigma1);
	friend std::ostream& operator<< (std::ostream &out, const MoGaussian &dist);
};


// Base class for vectors of exponential family distributions
class ExponentialFamVec {
public:
// no. of distributions in vec
	long nn;
	double eps = std::numeric_limits<double>::min();

	virtual long size() const {
		return nn;
	}
	Eigen::ArrayXd nats_to_mean(Eigen::ArrayXXd nats) const;
	Eigen::ArrayXd nats_to_var(Eigen::ArrayXXd nats) const;
	Eigen::ArrayXd nats_to_mean_sq(Eigen::ArrayXXd nats) const;
	Eigen::ArrayXXd mean_var_to_nats(Eigen::ArrayXd mu, Eigen::ArrayXd sig);
	Eigen::ArrayXd nats_to_kl_div(Eigen::ArrayXXd nats, double prior_var) const;
	double nats_to_mean(long ii, Eigen::ArrayXXd nats) const;
	double nats_to_var(long ii, Eigen::ArrayXXd nats) const;

/*** Basic properties must be defined ***/
	virtual Eigen::VectorXd mean() const = 0;
	virtual Eigen::ArrayXd var() const = 0;
	virtual double mean(long ii) const = 0;
	virtual double var(long ii) const = 0;

/*** Basic access ***/
	virtual void set_mean(Eigen::ArrayXd) = 0;

/*** Each prior structure should store its own hyps ***/
	virtual std::string get_hyps_header(std::string prefix = "") const = 0;
	virtual Eigen::ArrayXXd get_hyps() const = 0;
	virtual void set_hyps(Eigen::ArrayXd vec) = 0;
};

class GaussianVec : public ExponentialFamVec {
public:
	Eigen::ArrayXXd nats;
	double sigma;

	GaussianVec() = default;
	GaussianVec(long n_vars){
		// Initialise all gaussians at mean 0, var 0.01
		nn = n_vars;
		nats.resize(nn, 2);
		nats.col(0) = Eigen::ArrayXd::Zero(nn);
		nats.col(1) = Eigen::ArrayXd::Constant(nn, -50);
	}
	void resize(long n_vars);

/*** Basic properties ***/
	Eigen::VectorXd mean() const {
		return nats_to_mean(nats);
	}
	Eigen::ArrayXd mean_sq() const {
		return nats_to_mean_sq(nats);
	}
	Eigen::ArrayXd var() const {
		return nats_to_var(nats);
	}
	double mean(long ii) const {
		return nats_to_mean(ii, nats);
	}
	double var(long ii) const {
		return nats_to_var(ii, nats);
	}
	double kl_div() const {
		return nats_to_kl_div(nats, sigma).sum();
	}
	void set_mean(Eigen::ArrayXd mu){
		assert(mu.rows() == nn);
		nats.col(0) = -2.0 * nats.col(1) * mu;
	}
	void set_mean(long ii, double mu){
		assert(ii < nn);
		nats(ii, 0) = -2.0 * nats(ii, 1) * mu;
	}

/*** Allow coordinate updates ***/
	Gaussian get_ith_distn(long ii) const;
	void set_ith_distn(long ii, Gaussian new_dist);
	Eigen::ArrayXd get_opt_hyps() const {
		Eigen::ArrayXd res(1);
		double sigma = nats_to_mean_sq(nats).sum() / nn;
		res << sigma;
		return res;
	}

/*** File IO ***/
	std::string header(std::string prefix = "") const;
	void read_from_grid(Eigen::MatrixXd grid);
	std::string get_hyps_header(std::string prefix = "") const {
		std::vector<std::string> vv = {"sigma"};
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
	Eigen::ArrayXXd get_hyps() const {
		Eigen::ArrayXXd res(1, 1);
		res << sigma;
		return res;
	}
	void set_hyps(Eigen::ArrayXd vec){
		sigma = vec[0];
	}
};



class MoGaussianVec : public ExponentialFamVec {
public:
	Eigen::ArrayXXd slab_nats;
	Eigen::ArrayXXd spike_nats;
	Eigen::ArrayXd mix;
	double lambda, sigma_spike, sigma_slab;

	MoGaussianVec() = default;
	MoGaussianVec(long n_vars){
		// Initialise all gaussians at mean 0, var 0.01
		nn = n_vars;
		mix = Eigen::ArrayXd::Constant(nn, 0.1);
		spike_nats.resize(nn, 2);
		slab_nats.resize(nn, 2);
		spike_nats.col(0) = Eigen::ArrayXd::Zero(nn);
		slab_nats.col(0) = Eigen::ArrayXd::Zero(nn);
		spike_nats.col(1) = Eigen::ArrayXd::Constant(nn, -50);
		slab_nats.col(1) = Eigen::ArrayXd::Constant(nn, -50);
	}
	void resize(long n_vars){
		// Initialise all gaussians at mean 0, var 0.01
		nn = n_vars;
		mix = Eigen::ArrayXd::Constant(nn, 0.1);
		spike_nats.resize(nn, 2);
		slab_nats.resize(nn, 2);
		spike_nats.col(0) = Eigen::ArrayXd::Zero(nn);
		slab_nats.col(0) = Eigen::ArrayXd::Zero(nn);
		spike_nats.col(1) = Eigen::ArrayXd::Constant(nn, -50);
		slab_nats.col(1) = Eigen::ArrayXd::Constant(nn, -50);
	}

/*** Basic properties ***/
	Eigen::VectorXd mean() const {
		return nats_to_mean(spike_nats) + mix * (nats_to_mean(slab_nats) - nats_to_mean(spike_nats));
	}
	Eigen::ArrayXd var() const {
		Eigen::ArrayXd res;
		Eigen::ArrayXd mu1 = nats_to_mean(slab_nats), mu2 = nats_to_mean(spike_nats);
		Eigen::ArrayXd s1 = nats_to_var(slab_nats), s2 = nats_to_var(spike_nats);
		res = mix * s1 + (1.0 - mix) * s2;
		res += mix * mu1.square() + (1.0 - mix) * mu2.square();
		res -= (mix * mu1 + (1.0 - mix) * mu2).square();
		return res;
	}
	double mean(long ii) const {
		return nats_to_mean(ii, spike_nats) + mix(ii) * (nats_to_mean(ii, slab_nats) - nats_to_mean(ii, spike_nats));
	}
	double var(long ii) const {
		double res;
		double mu1 = nats_to_mean(ii, slab_nats), mu2 = nats_to_mean(ii, spike_nats);
		double s1 = nats_to_var(ii, slab_nats), s2 = nats_to_var(ii, spike_nats);
		res = mix(ii) * s1 + (1.0 - mix(ii)) * s2;
		res += mix(ii) * mu1 * mu1 + (1.0 - mix(ii)) * mu2 * mu2;
		double tmp = mix(ii) * mu1 + (1.0 - mix(ii)) * mu2;
		res -= tmp * tmp;
		return res;
	}
	double kl_div() const {
		Eigen::ArrayXd res;
		res = mix * nats_to_kl_div(slab_nats, sigma_slab);
		res += (1 - mix) * nats_to_kl_div(spike_nats, sigma_spike);
		res += mix * std::log(lambda + eps);
		res += (1 - mix) * std::log(1 - lambda + eps);

		for (long ii = 0; ii < nn; ii++) {
			res(ii) -= mix(ii) * std::log(mix(ii) + eps);
			res(ii) -= (1 - mix(ii)) * std::log(1 - mix(ii) + eps);
		}
		return res.sum();
	}
	void set_mean(Eigen::ArrayXd mu){
		assert(mu.rows() == nn);
		slab_nats.col(0) = -2.0 * slab_nats.col(1) * mu;
		spike_nats.col(0) = -2.0 * spike_nats.col(1) * mu;
	}
	void set_mean(long ii, double mu){
		assert(ii < nn);
		slab_nats(ii, 0) = -2.0 * slab_nats(ii, 1) * mu;
		spike_nats(ii, 0) = -2.0 * spike_nats(ii, 1) * mu;
	}

/*** Allow coordinate updates ***/
	MoGaussian get_ith_distn(long ii) const {
		MoGaussian dist;
		dist.mix = mix(ii);
		dist.slab.nats << slab_nats(ii, 0), slab_nats(ii, 1);
		dist.spike.nats << spike_nats(ii, 0), spike_nats(ii, 1);
		return dist;
	}
	void set_ith_distn(long ii, MoGaussian new_dist) {
		mix(ii) = new_dist.mix;
		slab_nats(ii, 0) = new_dist.slab.nats(0);
		slab_nats(ii, 1) = new_dist.slab.nats(1);
		spike_nats(ii, 0) = new_dist.spike.nats(0);
		spike_nats(ii, 1) = new_dist.spike.nats(1);
	}
	Eigen::ArrayXd get_opt_hyps() const {
		Eigen::ArrayXd res(3);
		double lambda = mix.sum() / nn;
		double sigma_slab = (nats_to_mean_sq(slab_nats) * mix).sum() / (lambda * nn);
		double sigma_spike = (nats_to_mean_sq(spike_nats) * (1 - mix)).sum() / (nn - lambda * nn);
		res << lambda, sigma_slab, sigma_spike;
		return res;
	}

/*** File IO ***/
	std::string header(std::string prefix = "") const {
		std::vector<std::string> vv = {"alpha", "mu1", "s1", "mu2", "s2"};
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
	void read_from_grid(Eigen::MatrixXd grid){
		assert(grid.cols() == 5);
		nn = grid.rows();
		mix = grid.col(0);
		slab_nats = mean_var_to_nats(grid.col(1), grid.col(2));
		spike_nats = mean_var_to_nats(grid.col(3), grid.col(4));
	}
	std::string get_hyps_header(std::string prefix = "") const {
		std::vector<std::string> vv = {"lambda", "sigma_slab", "sigma_spike", "spike_dilution"};
		if(prefix.length() > 0) {
			for (auto& ss: vv) {
				ss += prefix;
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
	Eigen::ArrayXXd get_hyps() const {
		Eigen::ArrayXXd res(1, 4);
		res << lambda, sigma_slab, sigma_spike, sigma_slab / sigma_spike;
		return res;
	}
	void set_hyps(Eigen::ArrayXd vec){
		lambda = vec[0];
		sigma_slab = vec[1];
		sigma_spike = vec[2];
	}
};




#endif //LEMMA_PRIOR_HPP
