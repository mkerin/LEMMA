// class for implementation of variational bayes algorithm
#ifndef VBAYES_X_HPP
#define VBAYES_X_HPP

#include <iostream>
#include <cmath>
#include <limits>
#include <random>
#include "class.h"
#include "data.hpp"
#include "utils.hpp"
#include "tools/eigen3.3/Dense"
#include "tools/eigen3.3/Sparse"
#include "tools/eigen3.3/Eigenvalues"
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>


// inline double sigmoid(double x);
inline double int_klbeta_x(const Eigen::Ref<const Eigen::VectorXd>& alpha,
						 const Eigen::Ref<const Eigen::VectorXd>& mu, 
						 const std::vector< double >& s_sq,
						 double var_b,
						 double var_g,
						 int n_var,
						 double eps);


// TODO: Minimise deep copy by passing arguments by reference


class vbayes_x {
	// Niave implementation of vbayes algorithm as defined here:
	// https://projecteuclid.org/euclid.ba/1339616726

	// Assumes gxage implementation of the model

	public:
	parameters p;

	int iter_max = 1000;
	double PI = 3.1415926535897;
	double diff_tol = 1e-4;
	double eps = std::numeric_limits<double>::min();

	int n_grid;            // size of hyperparameter grid
	double n_samples;
	std::size_t n_var;
	std::size_t n_var2;
	int sigma_ind;
	int sigma_b_ind;
	int lam_b_ind;
	int sigma_g_ind;
	int lam_g_ind;


	Eigen::MatrixXd X;          // dosage matrix
	Eigen::MatrixXd Y;          // residual phenotype matrix
	Eigen::VectorXd dHtH;       // diagonal of H^T * H where H = (X, Z)
	Eigen::VectorXd Hty;
	Eigen::VectorXd rr;         // column vector of elements rr[kk] = alpha[kk]mu[kk]
	Eigen::MatrixXd hyps_grid;
	Eigen::MatrixXd probs_grid; // prob of each point in grid under hyps
	Eigen::VectorXd aa;         // column vector of participant ages

	std::vector< int > fwd_pass;
	std::vector< int > back_pass;
	bool initialize_params_default;

	// Optimal alpha & mu for each point in importance sampling grid
	std::vector< Eigen::VectorXd > alpha_i;
	std::vector< Eigen::VectorXd > mu_i;

	// Optimal init of alpha & mu for variational approximation
	Eigen::VectorXd alpha1, mu1;

	// posteriors
	std::vector< double > weights;
	std::vector< double > logw;
	std::vector< double > alpha_av, mu_av, beta_av;

	// output filepaths
	boost_io::filtering_ostream outf, outf_hyps, outf_logw, outf_inits;
	boost_io::filtering_ostream outf_mu_updates, outf_alpha_updates;
	std::string ofile_mu_updates, ofile_alpha_updates;

	vbayes_x( data& dat ) : X( dat.G ),
							Y( dat.Y ), 
							p( dat.params ) {
		sigma_ind = 0;
		sigma_b_ind = 1;
		sigma_g_ind = 2;
		lam_b_ind = 3;
		lam_g_ind = 4;
		std::vector< std::string > hyps_names;
		hyps_names.push_back("sigma");
		hyps_names.push_back("sigma_b");
		hyps_names.push_back("sigma_g");
		hyps_names.push_back("lambda_b");
		hyps_names.push_back("lambda_g");
		assert(dat.hyps_names == hyps_names);

		// Data size params
		n_var = dat.n_var;
		n_var2 = 2 * dat.n_var;
		n_samples = (double) dat.n_samples;
		n_grid = dat.hyps_grid.rows();

		// non random initialisation
		if(p.vb_init_file != "NULL"){
			alpha1 = dat.alpha_init;
			mu1 = dat.mu_init;
			initialize_params_default = false;
		} else {
			initialize_params_default = true;
		}

		// Check for interaction flag
		std::size_t x_col;
		if(p.interaction_analysis){
			if(p.x_param_name != "NULL"){
				x_col = find_covar_index(p.x_param_name, dat.covar_names);
			} else {
				x_col = 0;
			}
			aa = dat.W.col(x_col);
		} else {
			throw std::runtime_error("VBAYES_X class called when p.interaction_analysis is false");
		}

		// Pass parameter & genetic matrices
		probs_grid = dat.imprt_grid;
		hyps_grid = dat.hyps_grid;
		Eigen::VectorXd dXtX, dZtZ;
		Eigen::MatrixXd I_a_sq;

		// TODO: Double matrix algebra here functions as expected
		// TODO: RAM is going to explode when creating Z. Do I actually need this?
		// TODO: Maybe we can just use a for loop for dHtH?
		I_a_sq = aa.cwiseProduct(aa).asDiagonal();
		dXtX = (X.transpose() * X).diagonal();
		dZtZ = (X.transpose() * I_a_sq * X).diagonal();
		dHtH.resize(n_var2);
		dHtH << dXtX, dZtZ;
		Hty.resize(n_var2);
 		Hty << (X.transpose() * Y), (X.transpose() * (Y.cwiseProduct(aa)));

	}

	// For use in unit testing.
	vbayes_x(Eigen::MatrixXd myX, Eigen::MatrixXd myY) : X( myX ), Y( myY ){
		// dXtX = (X.transpose() * X).diagonal();
		// Xty = X.transpose() * Y;
		// n_samples = X.rows();
		// n_var = X.cols();
	}

	~vbayes_x(){
	}

	void output_init(){
		std::string gz_str = ".gz";
		std::string ofile, ofile_hyps, ofile_logw, ofile_s_updates, ofile_inits;
		ofile = p.out_file;
		std::size_t pos = ofile.rfind(".");

		ofile_hyps = ofile.substr(0, pos) + "_hyps." + ofile.substr(pos+1, ofile.length());
		ofile_inits = ofile.substr(0, pos) + "_inits." + ofile.substr(pos+1, ofile.length());

		ofile_logw = ofile.substr(0, pos) + "_elbo." + ofile.substr(pos+1, ofile.length());
		ofile_alpha_updates = ofile.substr(0, pos) + "_last_alpha." + ofile.substr(pos+1, ofile.length());
		ofile_mu_updates = ofile.substr(0, pos) + "_last_mu." + ofile.substr(pos+1, ofile.length());

		if (ofile.find(gz_str) != std::string::npos) {
			outf.push(boost_io::gzip_compressor());
			outf_hyps.push(boost_io::gzip_compressor());
			outf_inits.push(boost_io::gzip_compressor());
			outf_logw.push(boost_io::gzip_compressor());
			outf_alpha_updates.push(boost_io::gzip_compressor());
			outf_mu_updates.push(boost_io::gzip_compressor());
		}
		outf.push(boost_io::file_sink(ofile.c_str()));
		outf_hyps.push(boost_io::file_sink(ofile_hyps.c_str()));
		outf_inits.push(boost_io::file_sink(ofile_inits.c_str()));
		outf_logw.push(boost_io::file_sink(ofile_logw.c_str()));
		outf_alpha_updates.push(boost_io::file_sink(ofile_alpha_updates.c_str()));
		outf_mu_updates.push(boost_io::file_sink(ofile_mu_updates.c_str()));

		std::cout << "Writing posterior PIP and beta probabilities to " << ofile << std::endl;
		std::cout << "Writing posterior hyperparameter probabilities to " << ofile_hyps << std::endl;

		if(initialize_params_default){
			std::cout << "Write start points for alpha and mu to " << ofile_inits << std::endl;
		}

		if(p.verbose){
			std::cout << "Writing ELBO from each VB iteration to " << ofile_logw << std::endl;
		}

		outf << "post_alpha post_mu post_beta alpha_init mu_init" << std::endl;
		outf_hyps << "weights logw log_prior" << std::endl;
	}

	std::size_t find_covar_index( std::string colname, std::vector< std::string > col_names ){
		std::size_t x_col;
		std::vector<std::string>::iterator it;
		it = std::find(col_names.begin(), col_names.end(), colname);
		if (it == col_names.end()){
			throw std::invalid_argument("Can't locate parameter " + colname);
		}
		x_col = it - col_names.begin();
		return x_col;
	}

	void check_inputs(){
		assert(Y.rows() == n_samples);
		assert(X.rows() == n_samples);

		for (int ii = 0; ii < n_grid; ii++){
			assert(hyps_grid(ii, sigma_ind) > 0.0);
			assert(hyps_grid(ii, sigma_b_ind) > 0.0);
			assert(hyps_grid(ii, sigma_g_ind) > 0.0);
			assert(hyps_grid(ii, lam_b_ind) > 0.0);
			assert(hyps_grid(ii, lam_b_ind) < 1.0);
			assert(hyps_grid(ii, lam_g_ind) < 1.0);
			assert(hyps_grid(ii, lam_g_ind) > 0.0);
		}
	}

	// Resizes and writes to alpha & mu
	void random_alpha_mu(Eigen::Ref<Eigen::VectorXd> alpha,
						 Eigen::Ref<Eigen::VectorXd> mu){
		// Writes to alpha & mu
		// NB: Also resizes alpha & mu

		std::default_random_engine gen_gauss, gen_unif;
		std::normal_distribution<double> gaussian(0.0,1.0);
		std::uniform_real_distribution<double> uniform(0.0,1.0);
		double my_sum = 0;

		// Check alpha / mu are correct size
		assert(alpha.rows() == n_var2);
		assert(mu.rows() == n_var2);

		// Random initialisation of alpha, mu
		for (int kk = 0; kk < n_var; kk++){
			alpha(kk) = uniform(gen_unif);
			mu(kk) = gaussian(gen_gauss);
			my_sum += alpha(kk);
		}

		// Convert alpha to simplex. Not sure why this is a good idea
		for (int kk = 0; kk < n_var; kk++){
			alpha(kk) /= my_sum;
		}

		// Random initialisation of alpha, mu
		my_sum = 0;
		for (int kk = n_var; kk < n_var2; kk++){
			alpha(kk) = uniform(gen_unif);
			mu(kk) = gaussian(gen_gauss);
			my_sum += alpha(kk);
		}

		// Convert alpha to simplex. Not sure why this is a good idea
		for (int kk = n_var; kk < n_var2; kk++){
			alpha(kk) /= my_sum;
		}
	}

	// Writes to alpha, mu and Xr
	void inner_loop_update(Eigen::Ref<Eigen::VectorXd> alpha,
						   Eigen::Ref<Eigen::VectorXd> mu,
						   Eigen::Ref<Eigen::VectorXd> Xr,
						   Eigen::Ref<Eigen::VectorXd> aZr,
						   const Eigen::Ref<const Eigen::RowVectorXd> hyps, 
						   const std::vector< int >& iter){
		// Inner loop as described in Carbonetto & Stephens Fig 1
		// alpha & mu passed by reference.
		// 
		// Return number of loops until convergence
		double sigma, sigma_b, sigma_g, lam_b, lam_g, ff, rr_k, s_sq;
		int kk;
		// std::cout << " Starting update" << std::endl;

		sigma = hyps(sigma_ind);
		sigma_b = hyps(sigma_b_ind);
		lam_b = hyps(lam_b_ind);
		sigma_g = hyps(sigma_g_ind);
		lam_g = hyps(lam_g_ind);

		for (int jj = 0; jj < n_var2; jj++){

			kk = iter[jj];

			rr_k = alpha(kk) * mu(kk);
			if (kk < n_var){
				s_sq = sigma_g * sigma / (sigma_b * dHtH(kk) + 1.0);
			} else {
				s_sq = sigma_g * sigma / (sigma_g * dHtH(kk) + 1.0);
			}

		// std::cout << " Starting update mu" << std::endl;
			// Update mu (eq 9)
			mu(kk) = s_sq / sigma;
			if (kk < n_var){
				mu(kk) *= (Hty(kk) - Xr.dot(X.col(kk)) + dHtH(kk) * rr_k);
			} else {
				mu(kk) *= (Hty(kk) - aZr.dot(X.col(kk-n_var)) + dHtH(kk) * rr_k);
			}

		// std::cout << " Starting update alpha" << std::endl;
			// Update alpha (eq 10)
			if (kk < n_var){
				ff = std::log(lam_b / (1.0 - lam_b) + eps) + std::log(s_sq / sigma_b / sigma + eps);
				ff += mu(kk) * mu(kk) / s_sq / 2.0;
			} else {
				ff = std::log(lam_g / (1.0 - lam_g) + eps) + std::log(s_sq / sigma_g / sigma + eps);
				ff += mu(kk) * mu(kk) / s_sq / 2.0;
			}
			alpha(kk) = sigmoid(ff);

			if (kk < n_var){
				Xr = Xr + (alpha(kk)*mu(kk) - rr_k) * X.col(kk);
			} else {
				aZr = aZr + (alpha(kk)*mu(kk) - rr_k) * X.col(kk - n_var).cwiseProduct(aa).cwiseProduct(aa);
			}
		}
		// std::cout << " finished update" << std::endl;
	}

	void nan_check(double val, int linenum){
		// if(std::isnan(val)){
		// 	std::string s;
		// 	s = "Error: NaN produced at line " + std::to_string(linenum);
		// 	throw std::runtime_error(s);
		// }
	}

	// Takes only const parameters
	double calc_logw(const Eigen::Ref<const Eigen::RowVectorXd> hyps,
					 const Eigen::Ref<const Eigen::VectorXd>& Xr,
					 const Eigen::Ref<const Eigen::VectorXd>& aZr,
					 const std::vector< double >& s_sq,
					 const Eigen::Ref<const Eigen::VectorXd>& alpha,
					 const Eigen::Ref<const Eigen::VectorXd>& mu){
		// Uses dHtH, X and Y from class namespace
		double res = 0.0;
		assert(mu.rows() == n_var2);
		assert(alpha.rows() == n_var2);
		assert(s_sq.size() == n_var2);

		// Unpack hyperparameters
		double sigma, sigma_b, sigma_g, lam_b, lam_g;
		sigma = hyps(sigma_ind);
		sigma_b = hyps(sigma_b_ind);
		sigma_g = hyps(sigma_g_ind);
		lam_b = hyps(lam_b_ind);
		lam_g = hyps(lam_g_ind);

		// gen Var[B_k]
		Eigen::VectorXd varB(n_var2);
		for (int kk = 0; kk < n_var2; kk++){
			double mu_sq = mu(kk) * mu(kk);
			varB(kk) = alpha(kk)*(s_sq[kk] + (1 - alpha(kk)) * mu_sq);
		nan_check(varB(kk), kk);
		}

		// gen rr
		rr = (alpha.array() * mu.array()).matrix();
		Eigen::VectorXd Zr = ((X * rr.segment(n_var, n_var)).array().colwise() * aa.array()).matrix();

		// Expectation of linear regression log-likelihood
		res -= n_samples * std::log(2.0 * PI * sigma + eps) / 2.0;
		nan_check(res, 310);
		res -= (Y - Xr - Zr).squaredNorm() / 2.0 / sigma;
		nan_check(res, 312);
		res -= 0.5 * (dHtH.dot(varB)) / sigma;
		nan_check(res, 314);

		// Expectation of prior inclusion probabilities
		for (int kk = 0; kk < n_var; kk++){
			res += alpha(kk) * std::log(lam_b + eps);
			res += (1.0 - alpha(kk)) * std::log(1.0 - lam_b + eps);
		}
		for (int kk = n_var; kk < n_var2; kk++){
			res += alpha(kk) * std::log(lam_g + eps);
			res += (1.0 - alpha(kk)) * std::log(1.0 - lam_g + eps);
		}
		nan_check(res, 325);

		// Negative KL divergaence between priors and approx distribution
		double var_b = sigma * sigma_b;
		double var_g = sigma * sigma_g;
		res += int_klbeta_x(alpha, mu, s_sq, var_b, var_g, n_var, eps);
		nan_check(res, 331);

		return res;
	}

	// Returns logw and writes to alpha & mu
	double outer_loop(const Eigen::Ref<const Eigen::RowVectorXd> hyps,
					  Eigen::Ref<Eigen::VectorXd> alpha,
					  Eigen::Ref<Eigen::VectorXd> mu){
		int cnt;
		std::vector< double > s_sq(n_var2, 0);
		std::vector< int > iter;
		Eigen::VectorXd alpha0, mu0, Xr, aZr;
		double diff, sigma, sigma_b, sigma_g, lam_b, lam_g, logw_i;
		std::vector< Eigen::VectorXd > alpha_updates, mu_updates;

		sigma = hyps(sigma_ind);
		sigma_b = hyps(sigma_b_ind);
		sigma_g = hyps(sigma_g_ind);
		lam_b = hyps(lam_b_ind);
		lam_g = hyps(lam_g_ind);

		// Useful quantities
		rr = alpha.cwiseProduct(mu);
		Xr = X * rr.segment(0, n_var);
		aZr = ((X * rr.segment(n_var, n_var)).array().colwise() * aa.cwiseProduct(aa).array()).matrix();

		// solve for s_sq (eq 8)
		for (int kk = 0; kk < n_var; kk++){
			s_sq[kk] = sigma_b * sigma / (sigma_b * dHtH(kk) + 1.0);
		}
		for (int kk = n_var; kk < n_var2; kk++){
			s_sq[kk] = sigma_g * sigma / (sigma_g * dHtH(kk) + 1.0);
		}

		// Start inner loop
		if(p.verbose){
			alpha_updates.push_back(alpha);
			mu_updates.push_back(mu);
		}
		int count = 0;
		bool converged = false;
		while(count < iter_max && !converged){
			alpha0 = alpha;
			mu0 = mu;

			// Alternate between forward & backward passes
			if(count % 2 == 0){
				iter = fwd_pass;
			} else {
				iter = back_pass;
			}

			// Variational inference; update alpha, beta, Xr
			inner_loop_update(alpha, mu, Xr, aZr, hyps, iter);

			if(p.verbose){
				outf_logw << calc_logw(hyps, Xr, aZr, s_sq, alpha, mu) << " ";
				alpha_updates.push_back(alpha);
				mu_updates.push_back(mu);
			}

			// Convergence if maximum diff in mixing coefficients < diff_tol
			count++;
			diff = (alpha0 - alpha).cwiseAbs().maxCoeff();
			if(diff > diff_tol){
				converged = true;
			}
		}

		// log-lik lower bound logw_i for i'th hyperparam gridpoint (eq 14)
		logw_i = calc_logw(hyps, Xr, aZr, s_sq, alpha, mu);

		if(p.verbose){
			outf_logw << std::endl;
			bool alpha_nan, mu_nan, logw_nan;
			alpha_nan = std::isnan(alpha.sum());
			mu_nan = std::isnan(mu.sum());
			logw_nan = std::isnan(logw_i);
			if(alpha_nan || mu_nan || logw_nan){
				// Output iterations of alpha & mu from last round of
				// optimisation
				for (int kk = 0; kk < n_var2; kk++){
					for (int ll = 0; ll < count; ll++){
						outf_alpha_updates << alpha_updates[ll][kk];
						outf_mu_updates << mu_updates[ll][kk];

						if(kk+1 < n_var2){
							outf_alpha_updates << " ";
							outf_mu_updates << " ";
						}
					}
					outf_alpha_updates << std::endl;
					outf_mu_updates << std::endl;
				}

				std::cout << "ERROR: NaN discovered in alpha/mu values after ";
				std::cout << "inference at grid point:" << std::endl << "(";
				std::cout << sigma << ", " << sigma_b << ", " << sigma_g << ", ";
				std::cout << lam_b << ", " << lam_g << ")" << std::endl;
				std::cout << "Dumping alpha/mu updates to:" << std::endl;
				std::cout << ofile_alpha_updates << std::endl;
				std::cout << ofile_mu_updates << std::endl;

				if(alpha_nan){
					throw std::runtime_error("ERROR: NaNs detected in alphas");
				} else if(mu_nan){
					throw std::runtime_error("ERROR: NaNs detected in mus");
				} else {
					throw std::runtime_error("ERROR: NaNs detected in ELBO computation");
				}
			}
		}

		return logw_i;
	}

	void run(){
		Eigen::RowVectorXd hyps;
		Eigen::VectorXd alpha, mu, Xr, rr;
		double logw_i, logw1;
		int cnt;
		bool check = false;

		logw1 = -1.0 * std::numeric_limits<double>::max();

		// Allocate memory
		weights.resize(n_grid);
		logw.resize(n_grid);
		alpha_i.resize(n_grid);
		mu_i.resize(n_grid);
		rr.resize(n_var2);
		alpha.resize(n_var2);
		mu.resize(n_var2);

		// Initialise forward & backward pass vectors
		for(int kk = 0; kk < n_var2; kk++){
			fwd_pass.push_back(kk);
			back_pass.push_back(n_var2 - kk - 1);
		}

		// First run with random alpha / mu
		if(initialize_params_default){
			for (int ii = 0; ii < n_grid; ii++){
				std::cout << "\rRound 1: grid point " << ii+1 << "/" << n_grid;
				hyps = hyps_grid.row(ii);

				// Initialise alpha and mu randomly
				random_alpha_mu(alpha, mu);

				// std::cout << "Starting outer loop" << std::endl;
				logw_i = outer_loop(hyps, alpha, mu);
				if (logw_i > logw1){
					check = true;
					logw1 = logw_i;
					alpha1 = alpha;
					mu1 = mu;
				}
			}
		}
		std::cout << std::endl;

		if(initialize_params_default && !check){
			throw std::runtime_error("ERROR: No valid common starting points found.");
		}

		if(initialize_params_default){
			outf_inits << "alpha mu" << std::endl;
			for(int kk = 0; kk < n_var2; kk++){
				outf_inits << alpha1[kk] << " " << mu1[kk] << std::endl;
			}
		}

		// Second run with best guess alpha / mu
		for (int ii = 0; ii < n_grid; ii++){
			std::cout << "\rRound 2: grid point " << ii+1 << "/" << n_grid;
			hyps = hyps_grid.row(ii);

			// Choose best init for alpha and mu
			alpha = alpha1;
			mu = mu1;

			// Compute unnormalised importance weights
			logw_i = outer_loop(hyps, alpha, mu);
			logw[ii] = logw_i;

			// Save optimised alpha / mu
			alpha_i[ii] = alpha;
			mu_i[ii] = mu;
		}
		std::cout << std::endl;

		// Normalise importance weights
		double max_elem, my_sum = 0;

		for (int ii = 0; ii < n_grid; ii++){
			weights[ii] = logw[ii] + std::log(probs_grid(ii,0) + eps);
		}

		max_elem = *std::max_element(weights.begin(), weights.end());

		for (int ii = 0; ii < n_grid; ii++){
			weights[ii] = std::exp(weights[ii] - max_elem);
		}

		for (int ii = 0; ii < n_grid; ii++){
			my_sum += weights[ii];
		}

		for (int ii = 0; ii < n_grid; ii++){
			weights[ii] /= my_sum;
		}

		// Average alpha + mu over hyperparams
		alpha_av.resize(n_var2, 0.0);
		mu_av.resize(n_var2, 0.0);
		beta_av.resize(n_var2, 0.0);
		for (int ii = 0; ii < n_grid; ii++){
			for (int kk = 0; kk < n_var2; kk++){
				alpha_av[kk] += weights[ii] * alpha_i[ii](kk);
				mu_av[kk] += weights[ii] * mu_i[ii](kk);
				beta_av[kk] += weights[ii] * mu_i[ii](kk) * alpha_i[ii](kk);
			}
		}
	}

	void output_results(){
		// Write results of main inference to file
		for (int kk = 0; kk < n_var; kk++){
			outf << alpha_av[kk] << " " << mu_av[kk] << " ";
 			outf << beta_av[kk] << " " << alpha1[kk] << " ";
 			outf << mu1[kk] << std::endl;
		}

		// Write hyperparams weights to file
		for (int ii = 0; ii < n_grid; ii++){
			outf_hyps << weights[ii] << " " << logw[ii] << " ";
			outf_hyps << std::log(probs_grid(ii,0) + eps) << std::endl;
		}
	}
};

// inline double sigmoid(double x){
// 	return 1.0 / (1.0 + std::exp(-x));
// }

inline double int_klbeta_x(const Eigen::Ref<const Eigen::VectorXd>& alpha,
						 const Eigen::Ref<const Eigen::VectorXd>& mu, 
						 const std::vector< double >& s_sq,
						 double var_b,
						 double var_g,
						 int n_var,
						 double eps){
	double res = 0;
	int n_var2 = 2 * n_var;
	for (int kk = 0; kk < n_var; kk++){
		res += alpha(kk) * (1.0 + std::log(s_sq[kk] / var_b + eps) -
							(s_sq[kk] + mu(kk) * mu(kk)) / var_b) / 2.0;
	}
	for (int kk = n_var; kk < n_var2; kk++){
		res += alpha(kk) * (1.0 + std::log(s_sq[kk] / var_g + eps) -
							(s_sq[kk] + mu(kk) * mu(kk)) / var_g) / 2.0;
	}
	for (int kk = 0; kk < n_var2; kk++){
		res -= alpha[kk] * std::log(alpha[kk] + eps);
		res -= (1 - alpha[kk]) * std::log(1 - alpha[kk] + eps);
	}
	return res;
}

#endif
