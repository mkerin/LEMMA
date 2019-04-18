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



// Base class for vectors of exponential family distributions
// https://www.geeksforgeeks.org/advanced-c-virtual-copy-constructor/
class ExponentialFamVec {
public:
// no. of distributions in vec
	long nn;
	double eps = std::numeric_limits<double>::min();

	virtual ExponentialFamVec *clone() = 0;
	virtual ~ExponentialFamVec() = default;

	virtual void resize(long n_vars) = 0;
	virtual long size() const {
		return nn;
	}
	Eigen::ArrayXd nats_to_mean(const Eigen::ArrayXXd& nats) const;
	Eigen::ArrayXd nats_to_var(const Eigen::ArrayXXd& nats) const;
	Eigen::ArrayXXd mean_var_to_nats(const Eigen::ArrayXd& mu, const Eigen::ArrayXd& sig);
	Eigen::ArrayXd nats_to_kl_div(const Eigen::ArrayXXd& nats, double prior_var) const;
	double nats_to_mean(long ii, const Eigen::ArrayXXd& nats) const;
	double nats_to_var(long ii, const Eigen::ArrayXXd& nats) const;

	/*** Basic properties must be defined ***/
	virtual Eigen::VectorXd mean() const = 0;
	virtual Eigen::ArrayXd var() const = 0;
	virtual double kl_div() const = 0;
	virtual double mean(long ii) const = 0;
	virtual double var(long ii) const = 0;
	Eigen::ArrayXd sigmoid(const Eigen::ArrayXd& x){
		return 1.0 / (1.0 + Eigen::exp(-x));
	}

	/*** Basic access ***/
	virtual void set_mean(const Eigen::ArrayXd& mu) = 0;
	virtual void set_mean(long ii, double mu) = 0;

	/*** Updates ***/
	virtual void cavi_update_ith_var(long ii, double EXty, double EXtX, double pheno_sigma) = 0;
	virtual void scavi_update_ith_var(long ii, double EXty, double EXtX, double pheno_sigma, double stepsize){
		throw std::logic_error("Not yet implemented");
	}
	virtual Eigen::ArrayXd maximise_mix_coeff(const Eigen::ArrayXXd& slab_nats, const Eigen::ArrayXXd& spike_nats){
		throw std::logic_error("Not yet implemented");
	}

	/*** Each prior structure should store its own hyps ***/
	virtual std::string get_hyps_header(std::string prefix) const = 0;
	virtual Eigen::ArrayXXd get_hyps() const = 0;
	virtual void set_hyps(const Eigen::ArrayXd& vec) = 0;
	virtual Eigen::ArrayXd get_opt_hyps() const = 0;
	virtual double get_hyps_var() const = 0;

	/*** File IO ***/
	virtual void write_ith_distn_to_stream(long ii, std::ostream& outf) const = 0;
	virtual std::string header(std::string prefix) const = 0;
	virtual void read_from_grid(const Eigen::MatrixXd& grid) = 0;
};

class GaussianVec : public ExponentialFamVec {
public:
	Eigen::ArrayXXd nats;
	double sigma;

	GaussianVec() = default;
	~GaussianVec() = default;
	GaussianVec(long n_vars) {
		// Initialise all gaussians at mean 0, var 0.01
		nn = n_vars;
		nats.resize(nn, 2);
		nats.col(0) = Eigen::ArrayXd::Zero(nn);
		nats.col(1) = Eigen::ArrayXd::Constant(nn, -50);
	}

	ExponentialFamVec *clone() override {
		return new GaussianVec(*this);
	}

	/*** Basic properties ***/
	void resize(long n_vars) override;
	Eigen::VectorXd mean() const override {
		return nats_to_mean(nats);
	}
	Eigen::ArrayXd var() const override {
		return nats_to_var(nats);
	}
	double mean(long ii) const override {
		return nats_to_mean(ii, nats);
	}
	double var(long ii) const override  {
		return nats_to_var(ii, nats);
	}
	double kl_div() const override {
		return nats_to_kl_div(nats, sigma).sum();
	}
	void set_mean(const Eigen::ArrayXd& mu) override {
		assert(mu.rows() == nn);
		nats.col(0) = -2.0 * nats.col(1) * mu;
	}
	void set_mean(long ii, double mu) override {
		assert(ii < nn);
		nats(ii, 0) = -2.0 * nats(ii, 1) * mu;
	}

	/*** Allow coordinate updates ***/
	Eigen::ArrayXd get_opt_hyps() const override {
		Eigen::ArrayXd res(1);
		double sigma = (nats_to_mean(nats).array().square() + nats_to_var(nats)).sum() / nn;
		res << sigma;
		return res;
	}
	void cavi_update_ith_var(long ii, double EXty, double EXtX, double pheno_sigma) override {
		assert(ii < nn);
		double old = mean(ii);

		double m1 = (EXty + old * EXtX) / pheno_sigma;
		double m2 = -EXtX / 2.0 / pheno_sigma;
		double nat1 = m1;
		double nat2 = m2 - 1.0 / 2 / sigma;

		// Perform update
		nats(ii, 0) = nat1;
		nats(ii, 1) = nat2;
	}
	void scavi_update_ith_var(long ii, double EXty, double EXtX, double pheno_sigma, double stepsize) override {
		double old = mean(ii);

		double m1 = (EXty + old * EXtX) / pheno_sigma;
		double m2 = -EXtX / 2.0 / pheno_sigma;
		double nat1 = m1;
		double nat2 = m2 - 1.0 / 2 / sigma;

		// Perform update
		nats(ii, 0) = (1 - stepsize) * nats(ii, 0) + stepsize * nat1;
		nats(ii, 1) = (1 - stepsize) * nats(ii, 1) + stepsize * nat2;
	}


	/*** File IO ***/
	void write_ith_distn_to_stream(long ii, std::ostream& outf) const override {
		outf << " " << nats_to_mean(ii, nats);
		outf << " " << nats_to_var(ii, nats);
	}
	std::string header(std::string prefix) const override;
	void read_from_grid(const Eigen::MatrixXd& grid) override ;
	std::string get_hyps_header(std::string prefix) const override {
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
	Eigen::ArrayXXd get_hyps() const override {
		Eigen::ArrayXXd res(1, 1);
		res << sigma;
		return res;
	}
	void set_hyps(const Eigen::ArrayXd& vec) override {
		sigma = vec[0];
	}
	double get_hyps_var() const override {
			return sigma;
	}
};



class MoGaussianVec : public ExponentialFamVec {
public:
	Eigen::ArrayXXd slab_nats;
	Eigen::ArrayXXd spike_nats;
	Eigen::ArrayXd mix;
	double lambda, sigma_spike, sigma_slab;

	MoGaussianVec() = default;
	~MoGaussianVec() = default;
	explicit MoGaussianVec(long n_vars) {
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

	ExponentialFamVec *clone() override {
		return new MoGaussianVec(*this);
	}

	/*** Basic properties ***/
	void resize(long n_vars) override {
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
	Eigen::VectorXd mean() const override {
		return nats_to_mean(spike_nats) + mix * (nats_to_mean(slab_nats) - nats_to_mean(spike_nats));
	}
	Eigen::ArrayXd var() const override {
		Eigen::ArrayXd res;
		Eigen::ArrayXd mu1 = nats_to_mean(slab_nats), mu2 = nats_to_mean(spike_nats);
		Eigen::ArrayXd s1 = nats_to_var(slab_nats), s2 = nats_to_var(spike_nats);
		res = mix * s1 + (1.0 - mix) * s2;
		res += mix * mu1.square() + (1.0 - mix) * mu2.square();
		res -= (mix * mu1 + (1.0 - mix) * mu2).square();
		return res;
	}
	double mean(long ii) const override {
		return nats_to_mean(ii, spike_nats) + mix(ii) * (nats_to_mean(ii, slab_nats) - nats_to_mean(ii, spike_nats));
	}
	double var(long ii) const override {
		double res;
		double mu1 = nats_to_mean(ii, slab_nats), mu2 = nats_to_mean(ii, spike_nats);
		double s1 = nats_to_var(ii, slab_nats), s2 = nats_to_var(ii, spike_nats);
		res = mix(ii) * s1 + (1.0 - mix(ii)) * s2;
		res += mix(ii) * mu1 * mu1 + (1.0 - mix(ii)) * mu2 * mu2;
		double tmp = mix(ii) * mu1 + (1.0 - mix(ii)) * mu2;
		res -= tmp * tmp;
		return res;
	}
	double kl_div() const override {
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
	void set_mean(const Eigen::ArrayXd& mu) override {
		assert(mu.rows() == nn);
		slab_nats.col(0) = -2.0 * slab_nats.col(1) * mu;
		spike_nats.col(0) = -2.0 * spike_nats.col(1) * mu;
	}
	void set_mean(long ii, double mu) override {
		assert(ii < nn);
		slab_nats(ii, 0) = -2.0 * slab_nats(ii, 1) * mu;
		spike_nats(ii, 0) = -2.0 * spike_nats(ii, 1) * mu;
	}

	/*** Allow coordinate updates ***/
	void cavi_update_ith_var(long ii, double EXty, double EXtX, double pheno_sigma) override {
		assert(ii < nn);
		double old = mean(ii);

		double m1 = (EXty + old * EXtX) / pheno_sigma;
		double m2 = -EXtX / 2.0 / pheno_sigma;
		double nat1 = m1;
		double nat2_slab = m2 - 1.0 / 2 / sigma_slab;
		double nat2_spike = m2 - 1.0 / 2 / sigma_spike;

		// Perform update
		slab_nats(ii, 0) = nat1;
		slab_nats(ii, 1) = nat2_slab;
		spike_nats(ii, 0) = nat1;
		spike_nats(ii, 1) = nat2_spike;

		Eigen::ArrayXd tmp = maximise_mix_coeff(slab_nats.row(ii), spike_nats.row(ii));
		assert(tmp.rows() == 1);
		mix(ii) = tmp(0);
	}
	void scavi_update_ith_var(long ii, double EXty, double EXtX, double pheno_sigma, double stepsize) override {
		assert(ii < nn);
		double old = mean(ii);

		double m1 = (EXty + old * EXtX) / pheno_sigma;
		double m2 = -EXtX / 2.0 / pheno_sigma;
		double nat1 = m1;
		double nat2_slab = m2 - 1.0 / 2 / sigma_slab;
		double nat2_spike = m2 - 1.0 / 2 / sigma_spike;

		// Perform update
		slab_nats(ii, 0) = (1 - stepsize) * slab_nats(ii, 0) + stepsize * nat1;
		slab_nats(ii, 1) = (1 - stepsize) * slab_nats(ii, 1) + stepsize * nat2_slab;
		spike_nats(ii, 0) = (1 - stepsize) * spike_nats(ii, 0) + stepsize * nat1;
		spike_nats(ii, 1) = (1 - stepsize) * spike_nats(ii, 1) + stepsize * nat2_spike;

		Eigen::ArrayXd tmp = maximise_mix_coeff(slab_nats.row(ii), spike_nats.row(ii));
		assert(tmp.rows() == 1);
		mix(ii) = tmp(0);
	}
	Eigen::ArrayXd maximise_mix_coeff(const Eigen::ArrayXXd& slab_nats, const Eigen::ArrayXXd& spike_nats) override {
		// index 0; slab
		// index 1; spike
		double alpha_cnst;
		alpha_cnst  = std::log(lambda / (1.0 - lambda) + eps);
		alpha_cnst -= (std::log(sigma_slab) - std::log(sigma_spike)) / 2.0;

		Eigen::ArrayXd mu0 = nats_to_mean(slab_nats);
		Eigen::ArrayXd mu1 = nats_to_mean(spike_nats);
		Eigen::ArrayXd s0 = nats_to_var(slab_nats);
		Eigen::ArrayXd s1 = nats_to_var(spike_nats);
		Eigen::ArrayXd ff_k;
		ff_k = mu0 * mu0 / s0;
		ff_k += s0.log();
		ff_k -= mu1 * mu1 / s1;
		ff_k -= s1.log();
		return sigmoid(ff_k / 2.0 + alpha_cnst);
	}

	/*** File IO ***/
	void write_ith_distn_to_stream(long ii, std::ostream& outf) const override {
		outf << mix(ii);
		outf << " " << nats_to_mean(ii, slab_nats);
		outf << " " << nats_to_var(ii, slab_nats);
		outf << " " << nats_to_mean(ii, spike_nats);
		outf << " " << nats_to_var(ii, spike_nats);
	}
	std::string header(std::string prefix) const override {
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
	void read_from_grid(const Eigen::MatrixXd& grid) override {
		assert(grid.cols() == 5);
		nn = grid.rows();
		mix = grid.col(0);
		slab_nats = mean_var_to_nats(grid.col(1), grid.col(2));
		spike_nats = mean_var_to_nats(grid.col(3), grid.col(4));
	}
	Eigen::ArrayXd get_opt_hyps() const override {
		Eigen::ArrayXd res(3);
		double lambda = mix.sum() / nn;
		double sigma_slab = (mix * (nats_to_mean(slab_nats).array().square() + nats_to_var(slab_nats))).sum() / (lambda * nn);
		double sigma_spike = ((1 - mix) * (nats_to_mean(spike_nats).array().square() + nats_to_var(spike_nats))).sum() / (nn - lambda * nn);
		res << lambda, sigma_slab, sigma_spike;
		return res;
	}
	std::string get_hyps_header(std::string prefix) const override {
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
	Eigen::ArrayXXd get_hyps() const override {
		Eigen::ArrayXXd res(1, 4);
		res << lambda, sigma_slab, sigma_spike, sigma_slab / sigma_spike;
		return res;
	}
	void set_hyps(const Eigen::ArrayXd& vec) override {
		lambda = vec[0];
		sigma_slab = vec[1];
		sigma_spike = vec[2];
	}
	double get_hyps_var() const override {
		return lambda * sigma_slab + (1 - lambda) * sigma_spike;
	}
};




#endif //LEMMA_PRIOR_HPP
