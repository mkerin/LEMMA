// re-implementation of variational bayes algorithm for 1D GxE
#ifndef VBAYES_X2_HPP
#define VBAYES_X2_HPP

#include <algorithm>
#include <cstdint>    // uint32_t
#include <iostream>
#include <limits>
#include "class.h"
#include "utils.hpp"  // sigmoid
#include "tools/eigen3.3/Dense"

inline std::size_t find_covar_index( std::string colname, std::vector< std::string > col_names );

class VBayesX2 {
public:
	// Constants
	const int iter_max = 100;
	const double PI = 3.1415926535897;
	const double diff_tol = 1e-4;
	const double eps = std::numeric_limits<double>::min();
	const double logw_tol = 10;
	int print_interval;              // print time every x grid points

	// Column order of hyperparameters in grid
	const int sigma_ind   = 0;
	const int sigma_b_ind = 1;
	const int sigma_g_ind = 2;
	const int lam_b_ind   = 3;
	const int lam_g_ind   = 4;
	const std::vector< std::string > hyps_names = {"sigma", "sigma_b", "sigma_g",
												   "lambda_b", "lambda_g"};

	// sizes
	int           n_grid;            // size of hyperparameter grid
	std::uint32_t n_samples;
	std::uint32_t n_var;
	std::uint32_t n_var2;
	bool          random_params_init;

	// 
	parameters p;
	std::vector< std::uint32_t > fwd_pass;
	std::vector< std::uint32_t > back_pass;

	// Data
	Eigen::MatrixXd X;          // dosage matrix
	Eigen::MatrixXd Y;          // residual phenotype matrix
	Eigen::VectorXd dHtH;       // diagonal of H^T x H where H = (X, Z)
	Eigen::VectorXd Hty;		// vector of H^T x y where H = (X, Z)
	Eigen::VectorXd aa;         // column vector of participant ages
	Eigen::MatrixXd hyps_grid;
	Eigen::MatrixXd probs_grid; // prob of each point in grid under hyps

	// Init points
	Eigen::VectorXd alpha_init;         // column vector of participant ages
	Eigen::VectorXd mu_init;         // column vector of participant ages
	Eigen::VectorXd Hr_init;         // column vector of participant ages

	// Loop variables - don't have to worry about reference parameters
	double                i_sigma;
	double                i_sigma_b;
	double                i_sigma_g;
	double                i_lam_b;
	double                i_lam_g;
	double                i_logw;
	std::vector< double > i_s_sq;
	Eigen::VectorXd       i_alpha;
	Eigen::VectorXd       i_mu;
	Eigen::VectorXd       i_Hr;

	// Things to track from each interaction
	std::vector< int >             counts_list;              // Number of iterations to convergence at each step
	std::vector< std::vector< double > > logw_updates_list;  // elbo updates at each ii
	std::vector< Eigen::VectorXd > mu_list;                  // best mu at each ii
	std::vector< Eigen::VectorXd > alpha_list;               // best alpha at each ii
	std::vector< double >          logw_list;                // best logw at each ii
	std::vector< double >          weights;             // best logw weighted by prior

	// results
	std::vector< double > post_alpha;
	std::vector< double > post_mu;
	std::vector< double > post_beta;

	// boost fstreams
	boost_io::filtering_ostream outf, outf_weights, outf_elbo, outf_inits;
	boost_io::filtering_ostream outf_mus, outf_alphas;

	// time monitoring
	std::chrono::system_clock::time_point time_check;

	VBayesX2( data& dat ) : X( dat.G ),
							Y( dat.Y ), 
							p( dat.params ) {
		assert(std::includes(dat.hyps_names.begin(), dat.hyps_names.end(), hyps_names.begin(), hyps_names.end()));
		assert(p.interaction_analysis);

		// Data size params
		n_var =          dat.n_var;
		n_var2 =         2 * dat.n_var;
		n_samples =      dat.n_samples;
		n_grid =         dat.hyps_grid.rows();
		print_interval = std::max(1, n_grid / 10);

		// non random initialisation
		if(p.vb_init_file != "NULL"){
			alpha_init         = dat.alpha_init;
			mu_init            = dat.mu_init;
			random_params_init = false;
		} else {
			random_params_init = true;
		}

		// Set covariable vector aa
		if(p.x_param_name != "NULL"){
			std::size_t x_col = find_covar_index(p.x_param_name, dat.covar_names);
			aa                = dat.W.col(x_col);
		} else {
			aa                = dat.W.col(0);
		}

		// Allocate memory - vb
		alpha_init.resize(n_var2);
		mu_init.resize(n_var2);
		Hr_init.resize(n_samples);
		i_s_sq.resize(n_var2);
		for(std::uint32_t kk = 0; kk < n_var2; kk++){
			fwd_pass.push_back(kk);
			back_pass.push_back(n_var2 - kk - 1);
		}

		// Reserve memory for trackers
		counts_list.reserve(n_grid);              // Number of iterations to convergence at each step
		logw_updates_list.reserve(n_grid);        // elbo updates at each ii
		mu_list.reserve(n_grid);                  // best mu at each ii
		alpha_list.reserve(n_grid);               // best alpha at each ii
		logw_list.reserve(n_grid);                // best logw at each ii

		// Allocate memory - genetic
		Hty.resize(n_var2);
		dHtH.resize(n_var2);

		// Assign data - genetic
		probs_grid             = dat.imprt_grid;
		hyps_grid              = dat.hyps_grid;

		Eigen::MatrixXd I_a_sq = aa.cwiseProduct(aa).asDiagonal();
		Eigen::VectorXd dXtX   = (X.transpose() * X).diagonal();
		Eigen::VectorXd dZtZ   = (X.transpose() * I_a_sq * X).diagonal();
		dHtH                   << dXtX, dZtZ;
 		Hty                    << (X.transpose() * Y), (X.transpose() * (Y.cwiseProduct(aa)));
	}

	~VBayesX2(){
	}

	void print_time_check(){
		auto now = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = now-time_check;;
		std::cout << " (" << elapsed_seconds.count() << "seconds since last timecheck)" << std::endl;
		time_check = now;
	}

	void run(){
		std::cout << "Starting variational inference" << std::endl;
		time_check = std::chrono::system_clock::now();

		// Round 1; looking for best start point
		if(random_params_init){
			double logw_best = -std::numeric_limits<double>::max();
			bool init_not_set = true;
			Eigen::VectorXd mu1, alpha1;
			for (int ii = 0; ii < n_grid; ii++){
				if(ii % print_interval == 0){
					std::cout << "\rRound 1: grid point " << ii+1 << "/" << n_grid;
					print_time_check();
				}

				// Unpack hyperparams
				i_sigma   = hyps_grid(ii, sigma_ind);
				i_sigma_b = hyps_grid(ii, sigma_b_ind);
				i_sigma_g = hyps_grid(ii, sigma_g_ind);
				i_lam_b   = hyps_grid(ii, lam_b_ind);
				i_lam_g   = hyps_grid(ii, lam_g_ind);

				// random start point
				random_alpha_mu_init();

				// Run outer loop - don't update trackers
				runOuterLoop(false);

				if(std::isfinite(i_logw) && i_logw > logw_best){
					alpha1    = alpha_init;
					mu1       = mu_init;
					logw_best = i_logw;
					init_not_set = false;
				}
			}
			alpha_init = alpha1;
			mu_init = mu1;

			if(init_not_set){
				throw std::runtime_error("No valid start points found (elbo estimates all non-finite?).");
			}

			// Write inits to file
			for (std::uint32_t kk = 0; kk < n_var2; kk++){
				outf_inits << alpha_init[kk] << " " << mu_init[kk] << std::endl;
			}
		}

		// Round 2; initial values already assigned to alpha_init, mu_init
		for (int ii = 0; ii < n_grid; ii++){
			if(ii % print_interval == 0){
				std::cout << "\rRound 2: grid point " << ii+1 << "/" << n_grid;
				print_time_check();
			}

			// Unpack hyperparams
			i_sigma   = hyps_grid(ii, sigma_ind);
			i_sigma_b = hyps_grid(ii, sigma_b_ind);
			i_sigma_g = hyps_grid(ii, sigma_g_ind);
			i_lam_b   = hyps_grid(ii, lam_b_ind);
			i_lam_g   = hyps_grid(ii, lam_g_ind);

			// Run outer loop - update trackers
			runOuterLoop(true);
		}

		// Compute normalised weights using finite elbo
		weights.resize(n_grid);
		if(n_grid > 1){
			for (int ii = 0; ii < n_grid; ii++){
				weights[ii] = logw_list[ii] + std::log(probs_grid(ii,0) + eps);
			}
			weights = normaliseLogWeights(weights);
		} else {
			weights[0] = 1;
		}

		// Average alpha + mu over finite weights
		int nonfinite_count = 0;
		post_alpha.resize(n_var2);
		post_mu.resize(n_var2);
		post_beta.resize(n_var2);
		for (int ii = 0; ii < n_grid; ii++){
			if(std::isfinite(weights[ii])){
				for (std::uint32_t kk = 0; kk < n_var2; kk++){
					post_alpha[kk] += weights[ii] * alpha_list[ii](kk);
					post_mu[kk] += weights[ii] * mu_list[ii](kk);
					post_beta[kk] += weights[ii] * mu_list[ii](kk) * alpha_list[ii](kk);
				}
			} else {
				nonfinite_count++;
			}
		}

		if(nonfinite_count > 0){
			std::cout << "WARNING: " << nonfinite_count << " grid points returned non-finite ELBO.";
			std::cout << "Skipping these when producing posterior estimates.";
		}
	}

	void runOuterLoop(bool update_trackers){
		// minimise KL Divergence and assign elbo estimate
		// Assumes alpha_init, mu_init and Hr_init already exist

		// Assign initial values
		i_alpha = alpha_init;
		i_mu = mu_init;
		i_Hr = Hr_init;

		// Update s_sq
		for (std::uint32_t kk = 0; kk < n_var; kk++){
			i_s_sq[kk] = i_sigma_b * i_sigma / (i_sigma_b * dHtH(kk) + 1.0);
		}
		for (std::uint32_t kk = n_var; kk < n_var2; kk++){
			i_s_sq[kk] = i_sigma_g * i_sigma / (i_sigma_g * dHtH(kk) + 1.0);
		}

		// Run inner loop until convergence
		int count = 0;
		bool converged = false;
		double diff;
		Eigen::VectorXd alpha_prev;
		std::vector< std::uint32_t > iter;
		std::vector< double > logw_updates;
		while(!converged){
			alpha_prev = i_alpha;

			if(count % 2 == 0){
				iter = fwd_pass;
			} else {
				iter = back_pass;
			}

			// log elbo from each iteration, starting from init
			if(p.verbose && update_trackers){
				logw_updates.push_back(calc_logw());
			}

			// Update i_mum i_alpha, i_Hr
			updateAlphaMu(iter);
			count++;

			// Diagnose convergence
			diff = (alpha_prev - i_alpha).cwiseAbs().maxCoeff();
			if(diff < diff_tol){
				converged = true;
			}
		}

		i_logw = calc_logw();
		if(!std::isfinite(i_logw)){
			std::cout << "WARNING: non-finite elbo estimate produced" << std::endl;
		}

		// Log all things that we want to track
		if(update_trackers){
			logw_list.push_back(i_logw);
			counts_list.push_back(count);
			alpha_list.push_back(i_alpha);
			mu_list.push_back(i_mu);
			if(p.verbose){
				logw_updates.push_back(i_logw);  // adding converged estimate
				logw_updates_list.push_back(logw_updates);
			}
		}
	}

	void updateAlphaMu(std::vector< std::uint32_t > iter){
		std::uint32_t kk;
		double rr_k, ff_k;
		for(std::uint32_t jj = 0; jj < n_var2; jj++){
			kk = iter[jj];

			rr_k = i_alpha(kk) * i_mu(kk);

			// Update mu (eq 9)
			i_mu(kk) = i_s_sq[kk] / i_sigma;
			if (kk < n_var){
				i_mu(kk) *= (Hty(kk) - i_Hr.dot(X.col(kk)) + dHtH(kk) * rr_k);
			} else {
				i_mu(kk) *= (Hty(kk) - i_Hr.dot(X.col(kk - n_var).cwiseProduct(aa)) + dHtH(kk) * rr_k);
			}

			// Update alpha (eq 10)  TODO: check syntax / i_  / sigmoid here!
			if (kk < n_var){
				ff_k = std::log(i_lam_b / (1.0 - i_lam_b) + eps) + std::log(i_s_sq[kk] / i_sigma_b / i_sigma + eps);
				ff_k += i_mu(kk) * i_mu(kk) / i_s_sq[kk] / 2.0;
			} else {
				ff_k = std::log(i_lam_g / (1.0 - i_lam_g) + eps) + std::log(i_s_sq[kk] / i_sigma_g / i_sigma + eps);
				ff_k += i_mu(kk) * i_mu(kk) / i_s_sq[kk] / 2.0;
			}
			i_alpha(kk) = sigmoid(ff_k);

			// Update i_Hr
			if (kk < n_var){
				i_Hr = i_Hr + (i_alpha(kk)*i_mu(kk) - rr_k) * X.col(kk);
			} else {
				i_Hr = i_Hr + (i_alpha(kk)*i_mu(kk) - rr_k) * X.col(kk - n_var).cwiseProduct(aa);
			}
		}
	}

	std::vector< double > normaliseLogWeights(std::vector< double > my_weights){
		// Safer to normalise log-weights than niavely convert to weights
		// Skip non-finite values!
		double max_elem = *std::max_element(my_weights.begin(), my_weights.end());
		for (int ii = 0; ii < n_grid; ii++){
			my_weights[ii] = std::exp(my_weights[ii] - max_elem);
		}

		double my_sum = 0.0;
		for (int ii = 0; ii < n_grid; ii++){
			if(std::isfinite(weights[ii])){
				my_sum += my_weights[ii];
			}
		}

		for (int ii = 0; ii < n_grid; ii++){
			my_weights[ii] /= my_sum;
		}
		return my_weights;
	}

	void random_alpha_mu_init(){
		// alpha_init a uniform simplex, mu_init standard gaussian
		// Also sets Hr_init
		std::default_random_engine gen_gauss, gen_unif;
		std::normal_distribution<double> gaussian(0.0,1.0);
		std::uniform_real_distribution<double> uniform(0.0,1.0);
		double my_sum = 0;

		// Check alpha / mu are correct size
		assert(alpha_init.rows() == n_var2);
		assert(mu_init.rows() == n_var2);

		// Random initialisation of alpha, mu
		for (std::uint32_t kk = 0; kk < n_var2; kk++){
			alpha_init(kk) = uniform(gen_unif);
			mu_init(kk) = gaussian(gen_gauss);
			my_sum += alpha_init(kk);
		}

		// Convert alpha to simplex. Why?
		for (std::uint32_t kk = 0; kk < n_var2; kk++){
			alpha_init(kk) /= my_sum;
		}

		// Could reduce matrix multiplication by making alpha and mu inits symmetric.
		Eigen::VectorXd rr = alpha_init.cwiseProduct(mu_init);
		Hr_init << (X * rr.segment(0, n_var) + (X * rr.segment(n_var, n_var)).cwiseProduct(aa));
	}

	double calc_logw(){
		// Using dHtH, Y, Hr and i_* variables
		double res, int_linear = 0, int_gamma = 0, int_klbeta = 0;

		// gen Var[B_k]
		Eigen::VectorXd varB(n_var2);
		for (std::uint32_t kk = 0; kk < n_var2; kk++){
			varB(kk) = i_alpha(kk)*(i_s_sq[kk] + (1 - i_alpha(kk)) * i_mu(kk) * i_mu(kk));
		}

		// Expectation of linear regression log-likelihood
		int_linear -= ((double) n_samples) * std::log(2.0 * PI * i_sigma + eps) / 2.0;
		int_linear -= (Y - i_Hr).squaredNorm() / 2.0 / i_sigma;
		int_linear -= 0.5 * (dHtH.dot(varB)) / i_sigma;

		// gamma
		for (std::uint32_t kk = 0; kk < n_var; kk++){
			int_gamma += i_alpha(kk) * std::log(i_lam_b + eps);
			int_gamma += (1.0 - i_alpha(kk)) * std::log(1.0 - i_lam_b + eps);
		}
		for (std::uint32_t kk = n_var; kk < n_var2; kk++){
			int_gamma += i_alpha(kk) * std::log(i_lam_g + eps);
			int_gamma += (1.0 - i_alpha(kk)) * std::log(1.0 - i_lam_g + eps);
		}

		// kl-beta
		double var_b = i_sigma * i_sigma_g, var_g = i_sigma * i_sigma_g;
		for (std::uint32_t kk = 0; kk < n_var; kk++){
			int_klbeta += i_alpha(kk) * (1.0 + std::log(i_s_sq[kk] / var_b + eps) -
								(i_s_sq[kk] + i_mu(kk) * i_mu(kk)) / var_b) / 2.0;
		}
		for (std::uint32_t kk = n_var; kk < n_var2; kk++){
			int_klbeta += i_alpha(kk) * (1.0 + std::log(i_s_sq[kk] / var_g + eps) -
								(i_s_sq[kk] + i_mu(kk) * i_mu(kk)) / var_g) / 2.0;
		}
		for (std::uint32_t kk = 0; kk < n_var2; kk++){
			int_klbeta -= i_alpha[kk] * std::log(i_alpha[kk] + eps);
			int_klbeta -= (1 - i_alpha[kk]) * std::log(1 - i_alpha[kk] + eps);
		}

		res = int_linear + int_gamma + int_klbeta;
		return res;
	}

	void output_init(){
		// Initialise files ready to write;
		// posteriors to ofile
		// weights and logw weights to ofile_weights
		// (verbose) elbo updates to ofile_elbo
		// (random_params_init) alpha_init/mu_init to ofile_init
		std::size_t pos = p.out_file.rfind(".");

		std::string ofile = fstream_init(outf, ".");
		std::string ofile_weights = fstream_init(outf_weights, "_hyps.");
		std::cout << "Writing posterior PIP and beta probabilities to " << ofile << std::endl;
		std::cout << "Writing posterior hyperparameter probabilities to " << ofile_weights << std::endl;

		if(random_params_init){
			std::string ofile_inits = fstream_init(outf_inits, "_inits.");
			std::cout << "Write start points for alpha and mu to " << ofile_inits << std::endl;
		}
		if(p.verbose){
			std::string ofile_elbo = fstream_init(outf_elbo, "_elbo.");
			std::cout << "Writing ELBO from each VB iteration to " << ofile_elbo << std::endl;

			std::string ofile_alphas = fstream_init(outf_alphas, "_alphas.");
			std::cout << "Writing optimsed alpha from each grid point to " << ofile_alphas << std::endl;

			std::string ofile_mus = fstream_init(outf_mus, "_mus.");
			std::cout << "Writing optimsed alpha from each grid point to " << ofile_mus << std::endl;
		}

		// Headers
		outf << "post_alpha post_mu post_beta" << std::endl;
		outf_weights << "weights logw log_prior count" << std::endl;
		if(random_params_init){
			outf_inits << "alpha mu" << std::endl;
		}
	}

	void output_results(){
		// Write;
		// posteriors to ofile
		// weights / logw / log_priors / counts to ofile_weights
		// (verbose) elbo updates to ofile_elbo

		// Write results of main inference to file
		for (std::uint32_t kk = 0; kk < n_var2; kk++){
			outf << post_alpha[kk] << " " << post_mu[kk] << " ";
 			outf << post_beta[kk] << std::endl;
		}

		// Write hyperparams weights to file
		for (int ii = 0; ii < n_grid; ii++){
			outf_weights << weights[ii] << " " << logw_list[ii] << " ";
			outf_weights << std::log(probs_grid(ii,0) + eps) << " ";
			outf_weights << counts_list[ii] << std::endl;
		}

		if(p.verbose){
			for (int ii = 0; ii < n_grid; ii++){
				for (int cc = 0; cc < counts_list[ii]; cc++){
					outf_elbo << logw_updates_list[ii][cc] << " ";
				}
				outf_elbo << std::endl;
			}

			// Writing optimised alpha and mu from each grid point to file
			// 1 col per gridpoint
			for (std::uint32_t kk = 0; kk < n_var2; kk++){
				for (int ii = 0; ii < n_grid; ii++){
					outf_alphas << alpha_list[ii][kk] << " ";
					outf_mus << mu_list[ii][kk] << " ";
				}
				outf_alphas << std::endl;
				outf_mus << std::endl;
			}
		}
	}

	std::string fstream_init(boost_io::filtering_ostream& my_outf, std::string extra){
		std::string gz_str = ".gz";
		std::size_t pos = p.out_file.rfind(".");

		std::string ofile = p.out_file.substr(0, pos) + extra + p.out_file.substr(pos+1, p.out_file.length());
		if (p.out_file.find(gz_str) != std::string::npos) {
			my_outf.push(boost_io::gzip_compressor());
		}
		my_outf.push(boost_io::file_sink(ofile.c_str()));
		return ofile;
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
};

inline std::size_t find_covar_index( std::string colname, std::vector< std::string > col_names ){
	std::size_t x_col;
	std::vector<std::string>::iterator it;
	it = std::find(col_names.begin(), col_names.end(), colname);
	if (it == col_names.end()){
		throw std::invalid_argument("Can't locate parameter " + colname);
	}
	x_col = it - col_names.begin();
	return x_col;
}
#endif
