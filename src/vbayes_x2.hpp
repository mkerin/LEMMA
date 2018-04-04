// re-implementation of variational bayes algorithm for 1D GxE
#ifndef VBAYES_X2_HPP
#define VBAYES_X2_HPP

#include <algorithm>
#include <cstdint>    // uint32_t
#include <iostream>
#include <limits>
#include <random>
#include <thread> 
#include "sys/types.h"
#include "sys/sysinfo.h"
#include "class.h"
#include "data.hpp"
#include "utils.hpp"  // sigmoid
#include "tools/eigen3.3/Dense"
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>

inline std::size_t find_covar_index( std::string colname, std::vector< std::string > col_names );

struct Hyps{
	double sigma;
	double sigma_b;
	double sigma_g;
	double lam_b;
	double lam_g;
};

struct FreeParameters {
	std::vector< double > s_sq;
	Eigen::VectorXd       alpha;
	Eigen::VectorXd       mu;
	Eigen::VectorXd       Hr;
}; 

class VbTracker {
public:
	std::vector< int >             counts_list;              // Number of iterations to convergence at each step
	std::vector< std::vector< double > > logw_updates_list;  // elbo updates at each ii
	std::vector< Eigen::VectorXd > mu_list;                  // best mu at each ii
	std::vector< Eigen::VectorXd > alpha_list;               // best alpha at each ii
	std::vector< double >          logw_list;                // best logw at each ii

	VbTracker(){
	}

	VbTracker(int n_list){
		counts_list.resize(n_list);
		mu_list.resize(n_list);
		alpha_list.resize(n_list);
		logw_list.resize(n_list);
		logw_updates_list.resize(n_list);
	}

	~VbTracker() {
	}

	void resize(int n_list){
		counts_list.resize(n_list);
		mu_list.resize(n_list);
		alpha_list.resize(n_list);
		logw_list.resize(n_list);
		logw_updates_list.resize(n_list);
	}

	void clear(){
		counts_list.clear();
		mu_list.clear();
		alpha_list.clear();
		logw_list.clear();
		logw_updates_list.clear();
	}

	void copy_ith_element(int ii, VbTracker& other_tracker){
		counts_list[ii]       = other_tracker.counts_list[ii];
		mu_list[ii]           = other_tracker.mu_list[ii];
		alpha_list[ii]        = other_tracker.alpha_list[ii];
		logw_list[ii]         = other_tracker.logw_list[ii];
		logw_updates_list[ii] = other_tracker.logw_updates_list[ii];
	}
};

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
	std::vector< std::uint32_t > back_pass_short;

	// Data
	Eigen::MatrixXd X;          // dosage matrix
	Eigen::MatrixXd Y;          // residual phenotype matrix
	Eigen::VectorXd dXtX;       // diagonal of X^T x X
	Eigen::VectorXd dHtH;       // diagonal of H^T x H where H = (X, Z)
	Eigen::VectorXd Hty;		// vector of H^T x y where H = (X, Z)
	Eigen::VectorXd aa;         // column vector of participant ages
	Eigen::MatrixXd hyps_grid;
	Eigen::MatrixXd probs_grid; // prob of each point in grid under hyps

	// Init points
	Eigen::VectorXd alpha_init; // column vector of participant ages
	Eigen::VectorXd mu_init;    // column vector of participant ages
	Eigen::VectorXd Hr_init;    // column vector of participant ages
	Eigen::VectorXd Xr_init;    // column vector of participant ages

	// Things to track from each interaction
	VbTracker stitched_tracker;
	std::vector< double > weights;             // best logw weighted by prior

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

		// Allocate memory - vb
		alpha_init.resize(n_var2);
		mu_init.resize(n_var2);
		Hr_init.resize(n_samples);
		Xr_init.resize(n_samples);
		for(std::uint32_t kk = 0; kk < n_var2; kk++){
			fwd_pass.push_back(kk);
			back_pass.push_back(n_var2 - kk - 1);
		}
		for(std::uint32_t kk = 0; kk < n_var; kk++){
			back_pass_short.push_back(n_var - kk - 1);
		}

		// Allocate memory - genetic
		Hty.resize(n_var2);
		dHtH.resize(n_var2);
		dXtX.resize(n_var);

		// Set covariable vector aa
		if(p.x_param_name != "NULL"){
			std::size_t x_col = find_covar_index(p.x_param_name, dat.covar_names);
			aa                = dat.W.col(x_col);
		} else {
			aa                = dat.W.col(0);
		}

		// non random initialisation
		if(p.vb_init_file != "NULL"){
			alpha_init         = dat.alpha_init;
			mu_init            = dat.mu_init;
			// Gen Hr_init
			Eigen::VectorXd rr = alpha_init.cwiseProduct(mu_init);
			Hr_init << (X * rr.segment(0, n_var) + (X * rr.segment(n_var, n_var)).cwiseProduct(aa));
			Xr_init << X * rr.segment(0, n_var);
			random_params_init = false;
		} else {
			random_params_init = true;
		}

		// Assign data - genetic
		probs_grid             = dat.imprt_grid;
		hyps_grid              = dat.hyps_grid;

		Eigen::MatrixXd I_a_sq = aa.cwiseProduct(aa).asDiagonal();

		// Avoid calling X.tranpose()
		// Eigen::VectorXd dXtX(n_var), dZtZ(n_var);
		// for(std::std::uint32_t kk = 0; kk < n_var; kk++){
		// 	dXtX(kk, kk) = X.col(kk).squaredNorm();
		// 	dZtZ(kk, kk) = (X.col(kk).cwiseProduct(aa)).squaredNorm();
		// }
		Hty       << (Y.transpose() * X).transpose(), (Y.cwiseProduct(aa).transpose() * X).transpose();

		Eigen::VectorXd dXtX   = (X.transpose() * X).diagonal();
		Eigen::VectorXd dZtZ   = (X.transpose() * I_a_sq * X).diagonal();
		// Hty                    << (X.transpose() * Y), (X.transpose() * (Y.cwiseProduct(aa)));

		dHtH                   << dXtX, dZtZ;
	}

	~VBayesX2(){
	}

	void print_time_check(){
		auto now = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = now-time_check;;
		std::cout << " (" << elapsed_seconds.count();
 		std::cout << " seconds since last timecheck, estimated RAM usage = ";
		std::cout << getValueRAM() << "KB)" << std::endl;
		time_check = now;
	}

	void run(){
		std::cout << "Starting variational inference" << std::endl;
		time_check = std::chrono::system_clock::now();
		if(p.n_thread > 1){
			std::cout << "Running on " << p.n_thread << " threads" << std::endl;
		}

		// Divide grid of hyperparameters into chunks for multithreading
		std::vector< std::vector< int > > chunks(p.n_thread);
		for (int ii = 0; ii < n_grid; ii++){
			int ch_index = (ii % p.n_thread);
			chunks[ch_index].push_back(ii);
		}

		// Allocate memory for trackers
		std::vector< VbTracker > trackers(p.n_thread);
		for (int ch = 0; ch < p.n_thread; ch++){
			trackers[ch].resize(n_grid);
		}

		// Round 1; looking for best start point
		if(random_params_init){
			std::thread t1[p.n_thread];

			for (int ch = 1; ch < p.n_thread; ch++){
				t1[ch] = std::thread( [this, chunks, ch, &trackers] {
					runOuterLoop(chunks[ch], false, trackers[ch]); 
				} );
			}
			runOuterLoop(chunks[0], false, trackers[0]);

			for (int ch = 1; ch < p.n_thread; ch++){
				t1[ch].join();
			}

			// Find best init
			double logw_best = -std::numeric_limits<double>::max();
			bool init_not_set = true;
			for (int ii = 0; ii < n_grid; ii++){
				int tr_index     = (ii % p.n_thread);
				double logw      = trackers[tr_index].logw_list[ii];
				double sigma_g   = hyps_grid(ii, sigma_g_ind);
				if(std::isfinite(logw) && logw > logw_best && sigma_g > 0){
					alpha_init   = trackers[tr_index].alpha_list[ii];
					mu_init      = trackers[tr_index].mu_list[ii];
					logw_best    = logw;
					init_not_set = false;
				}
			}

			if(init_not_set){
				throw std::runtime_error("No valid start points found (elbo estimates all non-finite?).");
			}

			// gen Hr_init
			Eigen::VectorXd rr = alpha_init.cwiseProduct(mu_init);
			Hr_init << (X * rr.segment(0, n_var) + (X * rr.segment(n_var, n_var)).cwiseProduct(aa));
			Xr_init << (X * rr.segment(0, n_var));

			// Write inits to file
			std::string ofile_inits = fstream_init(outf_inits, "_inits.");
			std::cout << "Writing start points for alpha and mu to " << ofile_inits << std::endl;
			outf_inits << "alpha mu" << std::endl;
			for (std::uint32_t kk = 0; kk < n_var2; kk++){
				outf_inits << alpha_init[kk] << " " << mu_init[kk] << std::endl;
			}

			// Clear trackers between rounds
			// This may actually be un-necessary?
			for (int ch = 0; ch < p.n_thread; ch++){
				trackers[ch].clear();
				trackers[ch].resize(n_grid);
			}
		}


		// Round 2; initial values already assigned to alpha_init, mu_init
		std::thread t2[p.n_thread];
		for (int ch = 1; ch < p.n_thread; ch++){
			t2[ch] = std::thread( [this, chunks, ch, &trackers] {
				runOuterLoop(chunks[ch], true, trackers[ch]); 
			} );
		}
		runOuterLoop(chunks[0], true, trackers[0]);
		for (int ch = 1; ch < p.n_thread; ch++){
			t2[ch].join();
		}

		// Stitch trackers back together if using multithreading
		stitched_tracker.resize(n_grid);
		for (int ii = 0; ii < n_grid; ii++){
			int tr = (ii % p.n_thread);  // tracker index
			stitched_tracker.copy_ith_element(ii, trackers[tr]);
		}

		// Compute normalised weights using finite elbo
		weights.resize(n_grid);
		if(n_grid > 1){
			for (int ii = 0; ii < n_grid; ii++){
				weights[ii] = stitched_tracker.logw_list[ii] + std::log(probs_grid(ii,0) + eps);
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
					post_alpha[kk] += weights[ii] * stitched_tracker.alpha_list[ii](kk);
					post_mu[kk] += weights[ii] * stitched_tracker.mu_list[ii](kk);
					post_beta[kk] += weights[ii] * stitched_tracker.mu_list[ii](kk) * stitched_tracker.alpha_list[ii](kk);
				}
			} else {
				nonfinite_count++;
			}
		}

		if(nonfinite_count > 0){
			std::cout << "WARNING: " << nonfinite_count << " grid points returned non-finite ELBO.";
			std::cout << "Skipping these when producing posterior estimates.";
		}

		std::cout << "Variational inference finished" << std::endl;
	}

	void runOuterLoop(std::vector<int> grid_index_list,
                      const bool main_loop,
                      VbTracker& tracker){
		Hyps i_hyps;
		std::uint32_t L;

		for (auto ii : grid_index_list){
			if((ii + 1) % print_interval == 0){
				if(!main_loop){
					std::cout << "\rRound 1: grid point " << ii+1 << "/" << n_grid;
				} else {
					std::cout << "\rRound 2: grid point " << ii+1 << "/" << n_grid;
				}
				print_time_check();
			}

			// Unpack hyperparams
			i_hyps.sigma   = hyps_grid(ii, sigma_ind);
			i_hyps.sigma_b = hyps_grid(ii, sigma_b_ind);
			i_hyps.sigma_g = hyps_grid(ii, sigma_g_ind);
			i_hyps.lam_b   = hyps_grid(ii, lam_b_ind);
			i_hyps.lam_g   = hyps_grid(ii, lam_g_ind);

			// If h_g == 0 then set free parameters approximating interaction
			// coefficients to zero.
			if (i_hyps.sigma_g < 0.000000001){
				L = n_var;
			} else {
				L = n_var2;
			}

			// Run outer loop - don't update trackers
			runInnerLoop(ii, main_loop, L, i_hyps, tracker);
		}
	}

	void runInnerLoop(const int ii, bool main_loop, std::uint32_t L,
                      Hyps i_hyps, VbTracker& tracker){
		// minimise KL Divergence and assign elbo estimate
		// Assumes alpha_init, mu_init and Hr_init already exist
		FreeParameters i_par;

		// Assign initial values
		if (!main_loop) {
			random_alpha_mu_init(L, i_par);
		} else {
			i_par.alpha = alpha_init;
			i_par.mu = mu_init;
			if(L == n_var2){
				i_par.Hr = Hr_init;
			} else {
				i_par.Hr = Xr_init;
				for(std::uint32_t kk = L; kk < n_var2; kk++){
					i_par.alpha[kk] = 0.0;
					i_par.mu[kk] = 0.0;
				}
			}
		}

		// Update s_sq
		i_par.s_sq.resize(n_var2);
		for (std::uint32_t kk = 0; kk < n_var; kk++){
			i_par.s_sq[kk] = i_hyps.sigma_b * i_hyps.sigma / (i_hyps.sigma_b * dHtH(kk) + 1.0);
		}
		for (std::uint32_t kk = n_var; kk < L; kk++){
			i_par.s_sq[kk] = i_hyps.sigma_g * i_hyps.sigma / (i_hyps.sigma_g * dHtH(kk) + 1.0);
		}
		for (std::uint32_t kk = L; kk < n_var2; kk++){
			i_par.s_sq[kk] = 0.0;
		}

		// Run inner loop until convergence
		int count = 0;
		bool converged = false;
		double diff;
		Eigen::VectorXd alpha_prev;
		std::vector< std::uint32_t > iter;
		std::vector< double > logw_updates;
		while(!converged){
			alpha_prev = i_par.alpha;

			if(count % 2 == 0){
				iter = fwd_pass;
			} else if(L == n_var){
				iter = back_pass_short;
			} else {
				iter = back_pass;
			}

			// log elbo from each iteration, starting from init
			if(p.verbose && main_loop){
				logw_updates.push_back(calc_logw(L, i_hyps, i_par));
			}

			// Update i_mum i_alpha, i_Hr
			updateAlphaMu(iter, L, i_hyps, i_par);
			count++;

			// Diagnose convergence
			diff = (alpha_prev - i_par.alpha).cwiseAbs().maxCoeff();
			if(diff < diff_tol){
				converged = true;
			}
		}

		double i_logw = calc_logw(L, i_hyps, i_par);
		if(!std::isfinite(i_logw)){
			std::cout << "WARNING: non-finite elbo estimate produced" << std::endl;
		}

		// Log all things that we want to track
		tracker.logw_list[ii] = i_logw;
		tracker.counts_list[ii] = count;
		tracker.alpha_list[ii] = i_par.alpha;
		tracker.mu_list[ii] = i_par.mu;
		if(p.verbose){
			logw_updates.push_back(i_logw);  // adding converged estimate
			tracker.logw_updates_list[ii] = logw_updates;
		}
	}

	void updateAlphaMu(std::vector< std::uint32_t > iter, std::uint32_t L,
                       Hyps i_hyps, FreeParameters& i_par){
		std::uint32_t kk;
		double rr_k, ff_k;
		for(std::uint32_t jj = 0; jj < L; jj++){
			kk = iter[jj];
			assert(kk < L);

			rr_k = i_par.alpha(kk) * i_par.mu(kk);

			// Update mu (eq 9)
			i_par.mu(kk) = i_par.s_sq[kk] / i_hyps.sigma;
			if (kk < n_var){
				i_par.mu(kk) *= (Hty(kk) - i_par.Hr.dot(X.col(kk)) + dHtH(kk) * rr_k);
			} else {
				i_par.mu(kk) *= (Hty(kk) - i_par.Hr.dot(X.col(kk - n_var).cwiseProduct(aa)) + dHtH(kk) * rr_k);
			}

			// Update alpha (eq 10)  TODO: check syntax / i_  / sigmoid here!
			if (kk < n_var){
				ff_k = std::log(i_hyps.lam_b / (1.0 - i_hyps.lam_b) + eps) + std::log(i_par.s_sq[kk] / i_hyps.sigma_b / i_hyps.sigma + eps) / 2.0;
				ff_k += i_par.mu(kk) * i_par.mu(kk) / i_par.s_sq[kk] / 2.0;
			} else {
				ff_k = std::log(i_hyps.lam_g / (1.0 - i_hyps.lam_g) + eps) + std::log(i_par.s_sq[kk] / i_hyps.sigma_g / i_hyps.sigma + eps) / 2.0;
				ff_k += i_par.mu(kk) * i_par.mu(kk) / i_par.s_sq[kk] / 2.0;
			}
			i_par.alpha(kk) = sigmoid(ff_k);

			// Update i_Hr
			if (kk < n_var){
				i_par.Hr = i_par.Hr + (i_par.alpha(kk)*i_par.mu(kk) - rr_k) * X.col(kk);
			} else {
				i_par.Hr = i_par.Hr + (i_par.alpha(kk)*i_par.mu(kk) - rr_k) * (X.col(kk - n_var).cwiseProduct(aa));
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

	void random_alpha_mu_init(std::uint32_t L,
                              FreeParameters& par){
		// par.alpha a uniform simplex, par.mu standard gaussian
		// Also sets par.Hr
		std::default_random_engine gen_gauss, gen_unif;
		std::normal_distribution<double> gaussian(0.0,1.0);
		std::uniform_real_distribution<double> uniform(0.0,1.0);
		double my_sum = 0;

		// Allocate memory
		par.alpha.resize(n_var2);
		par.mu.resize(n_var2);
		par.Hr.resize(n_samples);

		// Random initialisation of alpha, mu
		for (std::uint32_t kk = 0; kk < L; kk++){
			par.alpha(kk) = uniform(gen_unif);
			par.mu(kk) = gaussian(gen_gauss);
			my_sum += par.alpha(kk);
		}
		for (std::uint32_t kk = L; kk < n_var2; kk++){
			par.alpha(kk) = 0.0;
			par.mu(kk) = 0.0;
		}

		// Convert alpha to simplex. Why?
		for (std::uint32_t kk = 0; kk < L; kk++){
			par.alpha(kk) /= my_sum;
		}

		// Gen Hr; Could reduce matrix multiplication by making alpha and mu inits symmetric.
		Eigen::VectorXd rr = par.alpha.cwiseProduct(par.mu);
		if(L == n_var2){
			par.Hr << (X * rr.segment(0, n_var) + (X * rr.segment(n_var, n_var)).cwiseProduct(aa));
		} else {
			par.Hr << (X * rr.segment(0, n_var));
		}
	}

	double calc_logw(std::uint32_t L,
                     Hyps i_hyps, FreeParameters i_par){
		// Using dHtH, Y, Hr and i_* variables
		double res, int_linear = 0, int_gamma = 0, int_klbeta = 0;

		// gen Var[B_k]
		Eigen::VectorXd varB(n_var2);
		for (std::uint32_t kk = 0; kk < L; kk++){
			varB(kk) = i_par.alpha(kk)*(i_par.s_sq[kk] + (1 - i_par.alpha(kk)) * i_par.mu(kk) * i_par.mu(kk));
		}
		for (std::uint32_t kk = L; kk < n_var2; kk++){
			varB(kk) = 0.0;
		}

		// Expectation of linear regression log-likelihood
		int_linear -= ((double) n_samples) * std::log(2.0 * PI * i_hyps.sigma + eps) / 2.0;
		int_linear -= (Y - i_par.Hr).squaredNorm() / 2.0 / i_hyps.sigma;
		int_linear -= 0.5 * (dHtH.dot(varB)) / i_hyps.sigma;

		// gamma
		for (std::uint32_t kk = 0; kk < n_var; kk++){
			int_gamma += i_par.alpha(kk) * std::log(i_hyps.lam_b + eps);
			int_gamma += (1.0 - i_par.alpha(kk)) * std::log(1.0 - i_hyps.lam_b + eps);
		}
		for (std::uint32_t kk = n_var; kk < L; kk++){
			int_gamma += i_par.alpha(kk) * std::log(i_hyps.lam_g + eps);
			int_gamma += (1.0 - i_par.alpha(kk)) * std::log(1.0 - i_hyps.lam_g + eps);
		}

		// kl-beta
		double var_b = i_hyps.sigma * i_hyps.sigma_b, var_g = i_hyps.sigma * i_hyps.sigma_g;
		for (std::uint32_t kk = 0; kk < n_var; kk++){
			int_klbeta += i_par.alpha(kk) * (1.0 + std::log(i_par.s_sq[kk] / var_b + eps) -
								(i_par.s_sq[kk] + i_par.mu(kk) * i_par.mu(kk)) / var_b) / 2.0;
		}
		for (std::uint32_t kk = n_var; kk < L; kk++){
			int_klbeta += i_par.alpha(kk) * (1.0 + std::log(i_par.s_sq[kk] / var_g + eps) -
								(i_par.s_sq[kk] + i_par.mu(kk) * i_par.mu(kk)) / var_g) / 2.0;
		}
		for (std::uint32_t kk = 0; kk < L; kk++){
			int_klbeta -= i_par.alpha[kk] * std::log(i_par.alpha[kk] + eps);
			int_klbeta -= (1 - i_par.alpha[kk]) * std::log(1 - i_par.alpha[kk] + eps);
		}

		res = int_linear + int_gamma + int_klbeta;
		return res;
	}

	void dump_calc_logw(double int_linear, double int_gamma, double int_klbeta){
		// For use in debugging only
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

		// if(random_params_init){
		// 	std::string ofile_inits = fstream_init(outf_inits, "_inits.");
		// 	std::cout << "Write start points for alpha and mu to " << ofile_inits << std::endl;
		// }
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
		// if(random_params_init){
		// 	outf_inits << "alpha mu" << std::endl;
		// }
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
			outf_weights << weights[ii] << " " << stitched_tracker.logw_list[ii] << " ";
			outf_weights << std::log(probs_grid(ii,0) + eps) << " ";
			outf_weights << stitched_tracker.counts_list[ii] << std::endl;
		}

		if(p.verbose){
			for (int ii = 0; ii < n_grid; ii++){
				for (int cc = 0; cc < stitched_tracker.counts_list[ii]; cc++){
					outf_elbo << stitched_tracker.logw_updates_list[ii][cc] << " ";
				}
				outf_elbo << std::endl;
			}

			// Writing optimised alpha and mu from each grid point to file
			// 1 col per gridpoint
			for (std::uint32_t kk = 0; kk < n_var2; kk++){
				for (int ii = 0; ii < n_grid; ii++){
					outf_alphas << stitched_tracker.alpha_list[ii][kk] << " ";
					outf_mus << stitched_tracker.mu_list[ii][kk] << " ";
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
			assert(hyps_grid(ii, sigma_g_ind) >= 0.0);
			assert(hyps_grid(ii, lam_b_ind) > 0.0);
			assert(hyps_grid(ii, lam_b_ind) < 1.0);
			assert(hyps_grid(ii, lam_g_ind) < 1.0);
			assert(hyps_grid(ii, lam_g_ind) > 0.0);
		}
	}

	int parseLineRAM(char* line){
		// This assumes that a digit will be found and the line ends in " Kb".
		int i = strlen(line);
		const char* p = line;
		while (*p <'0' || *p > '9') p++;
		line[i-3] = '\0';
		i = atoi(p);
		return i;
	}

	int getValueRAM(){ //Note: this value is in KB!
		FILE* file = fopen("/proc/self/status", "r");
		int result = -1;
		char line[128];

		while (fgets(line, 128, file) != NULL){
			if (strncmp(line, "VmRSS:", 6) == 0){
				result = parseLineRAM(line);
				break;
			}
		}
		fclose(file);
		return result;
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
