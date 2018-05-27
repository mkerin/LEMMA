// re-implementation of variational bayes algorithm for 1D GxE
#ifndef VBAYES_X2_HPP
#define VBAYES_X2_HPP

#include <algorithm>
#include <chrono>      // start/end time info
#include <ctime>       // start/end time info
#include <cstdint>    // uint32_t
#include <iostream>
#include <limits>
#include <random>
#include <thread> 
#include "sys/types.h"
#include "sys/sysinfo.h"
#include "class.h"
#include "vbayes_tracker.hpp"
#include "data.hpp"
#include "utils.hpp"  // sigmoid
#include "tools/eigen3.3/Dense"
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/filesystem.hpp>

template <typename T>
inline std::vector<int> validate_grid(const Eigen::MatrixXd &grid, const T n_var);
inline Eigen::MatrixXd subset_matrix(const Eigen::MatrixXd &orig, const std::vector<int> &valid_points);
inline std::size_t find_covar_index( std::string colname, std::vector< std::string > col_names );

struct FreeParameters {
	std::vector< double > s_sq;
	Eigen::VectorXd       alpha;
	Eigen::VectorXd       mu;
	Eigen::VectorXd       Hr;
}; 

class VBayesX2 {
public:
	// Constants
	const int iter_max = 200;
	const double PI = 3.1415926535897;
	const double eps = std::numeric_limits<double>::min();
	const double alpha_tol = 1e-4;
	const double logw_tol = 1e-2;
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
	GenotypeMatrix X;
	Eigen::MatrixXd Y;          // residual phenotype matrix
	Eigen::VectorXd dXtX;       // diagonal of X^T x X
	Eigen::VectorXd dHtH;       // diagonal of H^T x H where H = (X, Z)
	Eigen::VectorXd Hty;		// vector of H^T x y where H = (X, Z)
	Eigen::VectorXd aa;         // column vector of participant ages
	Eigen::MatrixXd r1_hyps_grid;
	Eigen::MatrixXd r1_probs_grid;
	Eigen::MatrixXd hyps_grid;
	Eigen::MatrixXd probs_grid; // prob of each point in grid under hyps

	// Init points
	Eigen::VectorXd alpha_init; // column vector of participant ages
	Eigen::VectorXd mu_init;    // column vector of participant ages
	Eigen::VectorXd Hr_init;    // column vector of participant ages
	Eigen::VectorXd Xr_init;    // column vector of participant ages

	// boost fstreams
	boost_io::filtering_ostream outf, outf_weights, outf_elbo, outf_alpha_diff, outf_inits;
	boost_io::filtering_ostream outf_mus, outf_alphas;

	// time monitoring
	std::chrono::system_clock::time_point time_check;

	VBayesX2( Data& dat ) : X( dat.G ),
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
		X.aa = aa;  // WARNING: Required to be able to call X.col(jj) with jj > P

		// non random initialisation
		if(p.vb_init_file != "NULL"){
			alpha_init         = dat.alpha_init;
			mu_init            = dat.mu_init;

			assert(alpha_init.rows() == n_var2);
			assert(mu_init.rows() == n_var2);
			// Gen Hr_init
			Eigen::VectorXd rr = alpha_init.cwiseProduct(mu_init);
			Hr_init << (X * rr.segment(0, n_var) + (X * rr.segment(n_var, n_var)).cwiseProduct(aa));
			Xr_init << X * rr.segment(0, n_var);
			random_params_init = false;
		} else {
			random_params_init = true;
		}

		// Assign data - hyperparameters
		probs_grid          = dat.imprt_grid;
		hyps_grid           = dat.hyps_grid;

		if(p.r1_hyps_grid_file == "NULL"){
			r1_hyps_grid    = hyps_grid;
			r1_probs_grid    = probs_grid;
		} else {
			r1_hyps_grid    = dat.r1_hyps_grid;
			r1_probs_grid    = dat.r1_probs_grid;
		}

		// Assign data - genetic
		Eigen::VectorXd a_sq = aa.cwiseProduct(aa);
		Hty       << (X.transpose_vector_multiply(Y)), (X.transpose_vector_multiply(Y.cwiseProduct(aa)));

		Eigen::VectorXd dXtX(n_var), dZtZ(n_var), col_j;
		for (std::size_t jj; jj < n_var; jj++){
			col_j = X.col(jj);
			dXtX[jj] = col_j.dot(col_j);
			dZtZ[jj] = col_j.cwiseProduct(a_sq).dot(col_j);
		}
		dHtH                   << dXtX, dZtZ;
	}

	~VBayesX2(){
	}

	void print_time_check(){
		auto now = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = now-time_check;
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

		// Round 1; looking for best start point
		if(random_params_init){
			int r1_n_grid = r1_hyps_grid.rows();

			// Round 1; Divide grid of hyperparameters into chunks for multithreading
			std::vector< std::vector< int > > chunks(p.n_thread);
			for (int ii = 0; ii < r1_n_grid; ii++){
				int ch_index = (ii % p.n_thread);
				chunks[ch_index].push_back(ii);
			}

			// Round 1; Allocate memory for trackers
			std::vector< VbTracker > trackers(p.n_thread);
			for (int ch = 0; ch < p.n_thread; ch++){
				trackers[ch].resize(r1_n_grid);
				trackers[ch].set_main_filepath(p.out_file);
			}

			std::thread t1[p.n_thread];
			for (int ch = 1; ch < p.n_thread; ch++){
				t1[ch] = std::thread( [this, r1_n_grid, chunks, ch, &trackers] {
					runOuterLoop(r1_hyps_grid, r1_n_grid, chunks[ch], false, trackers[ch]); 
				} );
			}
			runOuterLoop(r1_hyps_grid, r1_n_grid, chunks[0], false, trackers[0]);

			for (int ch = 1; ch < p.n_thread; ch++){
				t1[ch].join();
			}

			if(p.verbose){
				write_trackers_to_file("round1_", trackers, r1_n_grid, r1_probs_grid);
			}

			// Find best init
			double logw_best = -std::numeric_limits<double>::max();
			bool init_not_set = true;
			for (int ii = 0; ii < r1_n_grid; ii++){
				int tr_index     = (ii % p.n_thread);
				double logw      = trackers[tr_index].logw_list[ii];
				double sigma_g   = r1_hyps_grid(ii, sigma_g_ind);
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
			std::string ofile_inits = fstream_init(outf_inits, "", "_inits");
			std::cout << "Writing start points for alpha and mu to " << ofile_inits << std::endl;
			outf_inits << "alpha mu" << std::endl;
			for (std::uint32_t kk = 0; kk < n_var2; kk++){
				outf_inits << alpha_init[kk] << " " << mu_init[kk] << std::endl;
			}
		}

		// Round 2; Divide grid of hyperparameters into chunks for multithreading
		std::vector< std::vector< int > > chunks(p.n_thread);
		for (int ii = 0; ii < n_grid; ii++){
			int ch_index = (ii % p.n_thread);
			chunks[ch_index].push_back(ii);
		}

		// Round 2; Allocate memory for trackers
		std::vector< VbTracker > trackers(p.n_thread);
		for (int ch = 0; ch < p.n_thread; ch++){
			trackers[ch].resize(n_grid);
			trackers[ch].set_main_filepath(p.out_file);
		}

		// Round 2; initial values already assigned to alpha_init, mu_init
		std::cout << "Starting Round 2 (resetting timecheck)" << std::endl;
		time_check = std::chrono::system_clock::now();
		std::thread t2[p.n_thread];
		for (int ch = 1; ch < p.n_thread; ch++){
			t2[ch] = std::thread( [this, chunks, ch, &trackers] {
				runOuterLoop(hyps_grid, n_grid, chunks[ch], true, trackers[ch]); 
			} );
		}
		runOuterLoop(hyps_grid, n_grid, chunks[0], true, trackers[0]);
		for (int ch = 1; ch < p.n_thread; ch++){
			t2[ch].join();
		}

		write_trackers_to_file("", trackers, n_grid, probs_grid);

		std::cout << "Variational inference finished" << std::endl;
	}

	void runOuterLoop(const Eigen::Ref<const Eigen::MatrixXd>& outer_hyps_grid,
                      const int outer_n_grid,
                      std::vector<int> grid_index_list,
                      const bool main_loop,
                      VbTracker& tracker){
		Hyps i_hyps;
		std::uint32_t L;

		for (auto ii : grid_index_list){
			if((ii + 1) % print_interval == 0){
				if(!main_loop){
					std::cout << "\rRound 1: grid point " << ii+1 << "/" << outer_n_grid;
				} else {
					std::cout << "\rRound 2: grid point " << ii+1 << "/" << outer_n_grid;
				}
				print_time_check();
			}

			// Unpack hyperparams
			i_hyps.sigma   = outer_hyps_grid(ii, sigma_ind);
			i_hyps.sigma_b = outer_hyps_grid(ii, sigma_b_ind);
			i_hyps.sigma_g = outer_hyps_grid(ii, sigma_g_ind);
			i_hyps.lam_b   = outer_hyps_grid(ii, lam_b_ind);
			i_hyps.lam_g   = outer_hyps_grid(ii, lam_g_ind);

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
		auto inner_start = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed;

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
		double alpha_diff, logw_diff, logw_prev;
		Eigen::VectorXd alpha_prev;
		std::vector< std::uint32_t > iter;
		double i_logw = calc_logw(L, i_hyps, i_par);
		std::vector< double > logw_updates, alpha_diff_updates;
		logw_updates.push_back(i_logw);
		tracker.interim_output_init(ii, main_loop);
		while(!converged  && count < iter_max){
			alpha_prev = i_par.alpha;
			logw_prev = i_logw;

			// Alternate between back and fwd passes
			if(count % 2 == 0){
				iter = fwd_pass;
			} else if(L == n_var){
				iter = back_pass_short;
			} else {
				iter = back_pass;
			}

			// Update i_mu, i_alpha, i_Hr
			updateAlphaMu(iter, L, i_hyps, i_par);
			if(main_loop && p.mode_empirical_bayes){
				// auto update_end = std::chrono::system_clock::now();
				// elapsed = update_end - inner_start;
				// i_logw     = calc_logw(L, i_hyps, i_par);
				// tracker.push_interim_iter_update(count, i_hyps, i_logw, alpha_diff, elapsed);
				updateHyps(i_hyps, i_par, L);
			}

			// Diagnose convergence
			count++;
			i_logw     = calc_logw(L, i_hyps, i_par);
			logw_diff  = i_logw - logw_prev;
			alpha_diff = (alpha_prev - i_par.alpha).cwiseAbs().maxCoeff();
			logw_updates.push_back(i_logw);
			alpha_diff_updates.push_back(alpha_diff);

			if(p.alpha_tol_set_by_user && p.elbo_tol_set_by_user){
				if(alpha_diff < p.alpha_tol || logw_diff < p.elbo_tol){
					converged = true;
				}
			} else if(p.alpha_tol_set_by_user){
				if(alpha_diff < p.alpha_tol){
					converged = true;
				}
			} else if(p.elbo_tol_set_by_user){
				if(logw_diff < p.elbo_tol){
					converged = true;
				}
			} else if(p.mode_empirical_bayes && logw_diff < 0) {
				//  Monotnic trajectory no longer required under EB?
				converged = true;
			} else {
				if(alpha_diff < alpha_tol && logw_diff < logw_tol){
					converged = true;
				}
			}

			// Log updates
			auto update_end = std::chrono::system_clock::now();
			elapsed = update_end - inner_start;
			tracker.push_interim_iter_update(count, i_hyps, i_logw, alpha_diff, elapsed);
		}

		if(!std::isfinite(i_logw)){
			std::cout << "WARNING: non-finite elbo estimate produced" << std::endl;
		}

		// Log all things that we want to track
		auto inner_end = std::chrono::system_clock::now();
		elapsed = inner_end - inner_start;

		tracker.logw_list[ii] = i_logw;
		tracker.counts_list[ii] = count;
		tracker.alpha_list[ii] = i_par.alpha;
		tracker.mu_list[ii] = i_par.mu;
		tracker.elapsed_time_list[ii] = elapsed.count();
		tracker.hyps_list[ii] = i_hyps;
		if(p.verbose){
			logw_updates.push_back(i_logw);  // adding converged estimate
			tracker.logw_updates_list[ii] = logw_updates;
			tracker.alpha_diff_list[ii] = alpha_diff_updates;
		}
		tracker.push_interim_output(ii, main_loop);
	}

	void updateHyps(Hyps& i_hyps, FreeParameters& i_par, std::uint32_t L){
		i_hyps.lam_b   = 0.0;
		i_hyps.lam_g   = 0.0;
		i_hyps.sigma_b = 0.0;
		i_hyps.sigma_g = 0.0;
		i_hyps.sigma   = (Y - i_par.Hr).squaredNorm();

		// max sigma
		double k_varB;
		for (std::uint32_t kk = 0; kk < L; kk++){
			k_varB = i_par.alpha(kk)*(i_par.s_sq[kk] + (1 - i_par.alpha(kk)) * i_par.mu(kk) * i_par.mu(kk));
			i_hyps.sigma += dHtH[kk] * k_varB;
		}
		i_hyps.sigma /= n_samples;

		// max lambda_b & sigma_b
		for (std::uint32_t kk = 0; kk < n_var; kk++){
			i_hyps.lam_b += i_par.alpha[kk];
			i_hyps.sigma_b += i_par.alpha[kk] * (i_par.s_sq[kk] + i_par.mu[kk] * i_par.mu[kk]);
		}
		i_hyps.sigma_b /= i_hyps.lam_b;
		i_hyps.sigma_b /= i_hyps.sigma;
		i_hyps.lam_b /= n_var;

		// max lambda_b & sigma_b
		for (std::uint32_t kk = n_var; kk < L; kk++){
			i_hyps.lam_g += i_par.alpha[kk];
			i_hyps.sigma_g += i_par.alpha[kk] * (i_par.s_sq[kk] + i_par.mu[kk] * i_par.mu[kk]);
		}
		if(L == n_var2){
			i_hyps.sigma_g /= i_hyps.lam_g;
			i_hyps.sigma_g /= i_hyps.sigma;
			i_hyps.lam_g /= n_var;
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
			// i_par.mu(kk) = i_par.s_sq[kk] / i_hyps.sigma;
			// if (kk < n_var){
			// 	i_par.mu(kk) *= (Hty(kk) - i_par.Hr.dot(X.col(kk)) + dHtH(kk) * rr_k);
			// } else {
			// 	i_par.mu(kk) *= (Hty(kk) - i_par.Hr.dot(X.col(kk - n_var).cwiseProduct(aa)) + dHtH(kk) * rr_k);
			// }
			// i_par.mu(kk) *= (Hty(kk) - i_par.Hr.dot(X.col(kk)) + dHtH(kk) * rr_k);
			i_par.mu(kk) = i_par.s_sq[kk] * (Hty(kk) - i_par.Hr.dot(X.col(kk)) + dHtH(kk) * rr_k) / i_hyps.sigma;

			// Update alpha (eq 10)  TODO: check syntax / i_  / sigmoid here!
			if (kk < n_var){
				ff_k = std::log(i_hyps.lam_b / (1.0 - i_hyps.lam_b) + eps) + std::log(i_par.s_sq[kk] / i_hyps.sigma_b / i_hyps.sigma + eps) / 2.0;
				ff_k += i_par.mu(kk) * i_par.mu(kk) / i_par.s_sq[kk] / 2.0;
			} else {
				ff_k = std::log(i_hyps.lam_g / (1.0 - i_hyps.lam_g) + eps) + std::log(i_par.s_sq[kk] / i_hyps.sigma_g / i_hyps.sigma + eps) / 2.0;
				ff_k += i_par.mu(kk) * i_par.mu(kk) / i_par.s_sq[kk] / 2.0;
			}
			i_par.alpha(kk) = sigmoid(ff_k);

			// Update i_Hr; faster to take schur product with aa inside genotype_matrix
			i_par.Hr = i_par.Hr + (i_par.alpha(kk)*i_par.mu(kk) - rr_k) * (X.col(kk));
		}
	}

	void normaliseLogWeights(std::vector< double >& my_weights){
		// Safer to normalise log-weights than niavely convert to weights
		// Skip non-finite values!
		int nn = my_weights.size();
		double max_elem = *std::max_element(my_weights.begin(), my_weights.end());
		for (int ii = 0; ii < nn; ii++){
			my_weights[ii] = std::exp(my_weights[ii] - max_elem);
		}

		double my_sum = 0.0;
		for (int ii = 0; ii < nn; ii++){
			if(std::isfinite(my_weights[ii])){
				my_sum += my_weights[ii];
			}
		}

		for (int ii = 0; ii < nn; ii++){
			my_weights[ii] /= my_sum;
		}
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

	void write_trackers_to_file(const std::string& file_prefix,
                                const std::vector< VbTracker >& trackers,
                                const int my_n_grid,
                                const Eigen::Ref<const Eigen::VectorXd>& my_probs_grid){
		// Stitch trackers back together if using multithreading
		VbTracker stitched_tracker;
		stitched_tracker.resize(my_n_grid);
		for (int ii = 0; ii < my_n_grid; ii++){
			int tr = (ii % p.n_thread);  // tracker index
			stitched_tracker.copy_ith_element(ii, trackers[tr]);
		}

		output_init(file_prefix);
		output_results(stitched_tracker, my_n_grid, my_probs_grid);
	}

	void output_init(const std::string& file_prefix){
		// Initialise files ready to write;
		// posteriors to ofile
		// weights and logw weights to ofile_weights
		// (verbose) elbo updates to ofile_elbo
		// (random_params_init) alpha_init/mu_init to ofile_init
		std::string ofile, ofile_weights;

		ofile = fstream_init(outf, file_prefix, "");
		ofile_weights = fstream_init(outf_weights, file_prefix, "_hyps");
		std::cout << "Writing posterior PIP and beta probabilities to " << ofile << std::endl;
		std::cout << "Writing posterior hyperparameter probabilities to " << ofile_weights << std::endl;

		if(p.verbose){
			std::string ofile_elbo, ofile_alphas, ofile_mus, ofile_alpha_diff;
			ofile_elbo = fstream_init(outf_elbo, file_prefix, "_elbo");
			std::cout << "Writing ELBO from each VB iteration to " << ofile_elbo << std::endl;

			ofile_alpha_diff = fstream_init(outf_alpha_diff, file_prefix, "_alpha_diff");
			std::cout << "Writing max change in alpha from each VB iteration to " << ofile_alpha_diff << std::endl;

			ofile_alphas = fstream_init(outf_alphas, file_prefix, "_alphas");
			std::cout << "Writing optimised alpha from each grid point to " << ofile_alphas << std::endl;

			ofile_mus = fstream_init(outf_mus, file_prefix, "_mus");
			std::cout << "Writing optimised mu from each grid point to " << ofile_mus << std::endl;
		}

		// Headers
		outf << "post_alpha post_mu post_beta" << std::endl;
		outf_weights << "weights logw log_prior count time sigma sigma_b ";
		outf_weights << "sigma_g lambda_b lambda_g" << std::endl;
	}

	void output_results(const VbTracker& stitched_tracker, const int my_n_grid,
						const Eigen::Ref<const Eigen::VectorXd>& my_probs_grid){
		// Write;
		// posteriors to ofile
		// weights / logw / log_priors / counts to ofile_weights
		// (verbose) elbo updates to ofile_elbo

		// Compute normalised weights using finite elbo
		std::vector< double > weights(my_n_grid);
		if(n_grid > 1){
			for (int ii = 0; ii < my_n_grid; ii++){
				weights[ii] = stitched_tracker.logw_list[ii] + std::log(my_probs_grid(ii,0) + eps);
			}
			normaliseLogWeights(weights);
		} else {
			weights[0] = 1;
		}

		// Extract posterior values from tracker
		int nonfinite_count = 0;
		std::vector< double > post_alpha(n_var2);
		std::vector< double > post_mu(n_var2);
		std::vector< double > post_beta(n_var2);
		for (int ii = 0; ii < my_n_grid; ii++){
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

		// Set Precision
		outf_elbo << std::setprecision(4) << std::fixed;
		outf_alpha_diff << std::setprecision(4) << std::fixed;

		// Write results of main inference to file
		for (std::uint32_t kk = 0; kk < n_var2; kk++){
			outf << post_alpha[kk] << " " << post_mu[kk] << " ";
			outf << post_beta[kk] << std::endl;
		}

		// Write hyperparams weights to file
		for (int ii = 0; ii < my_n_grid; ii++){
			outf_weights << weights[ii] << " " << stitched_tracker.logw_list[ii] << " ";
			outf_weights << std::log(my_probs_grid(ii,0) + eps) << " ";
			outf_weights << stitched_tracker.counts_list[ii] << " ";
			outf_weights << stitched_tracker.elapsed_time_list[ii] <<  " ";
			outf_weights << stitched_tracker.hyps_list[ii].sigma << " ";
			outf_weights << stitched_tracker.hyps_list[ii].sigma_b << " ";
			outf_weights << stitched_tracker.hyps_list[ii].sigma_g << " ";
			outf_weights << stitched_tracker.hyps_list[ii].lam_b << " ";
			outf_weights << stitched_tracker.hyps_list[ii].lam_g << std::endl;
		}

		if(p.verbose){
			for (int ii = 0; ii < my_n_grid; ii++){
				for (int cc = 0; cc < stitched_tracker.logw_updates_list[ii].size(); cc++){
					outf_elbo << stitched_tracker.logw_updates_list[ii][cc] << " ";
				}
				outf_elbo << std::endl;
			}

			for (int ii = 0; ii < my_n_grid; ii++){
				for (int cc = 0; cc < stitched_tracker.alpha_diff_list[ii].size(); cc++){
					outf_alpha_diff << stitched_tracker.alpha_diff_list[ii][cc] << " ";
				}
				outf_alpha_diff << std::endl;
			}

			// Writing optimised alpha and mu from each grid point to file
			// 1 col per gridpoint
			for (std::uint32_t kk = 0; kk < n_var2; kk++){
				for (int ii = 0; ii < my_n_grid; ii++){
					outf_alphas << stitched_tracker.alpha_list[ii][kk] << " ";
					outf_mus << stitched_tracker.mu_list[ii][kk] << " ";
				}
				outf_alphas << std::endl;
				outf_mus << std::endl;
			}
		}
	}

	std::string fstream_init(boost_io::filtering_ostream& my_outf,
                             const std::string& file_prefix,
                             const std::string& file_suffix){

		// boost::filesystem::path main_path(p.out_file);
		// boost::filesystem::path dir = main_path.parent_path();
		// boost::filesystem::path file_ext = main_path.extension();
		// boost::filesystem::path filename = main_path.replace_extension().filename();

		std::string filepath   = p.out_file;
		std::string dir        = filepath.substr(0, filepath.rfind("/")+1);
		std::string stem_w_dir = filepath.substr(0, filepath.find("."));
		std::string stem       = stem_w_dir.substr(stem_w_dir.rfind("/")+1, stem_w_dir.size());
		std::string ext        = filepath.substr(filepath.find("."), filepath.size());

		std::string ofile      = dir + file_prefix + stem + file_suffix + ext;

		my_outf.reset();
		std::string gz_str = ".gz";
		if (p.out_file.find(gz_str) != std::string::npos) {
			my_outf.push(boost_io::gzip_compressor());
		}
		my_outf.push(boost_io::file_sink(ofile.c_str()));
		return ofile;
	}

	void check_inputs(){
		// If grid contains hyperparameter values that aren't sensible then we exclude
		assert(Y.rows() == n_samples);
		assert(X.rows() == n_samples);

		std::vector<int> valid_points, r1_valid_points;
		valid_points        = validate_grid(hyps_grid, n_var);
		probs_grid          = subset_matrix(probs_grid, valid_points);
		hyps_grid           = subset_matrix(hyps_grid, valid_points);

		if(valid_points.size() == 0){
			throw std::runtime_error("No valid grid points in hyps_grid.");
		} else if(n_grid > valid_points.size()){
			std::cout << "WARNING: " << n_grid - valid_points.size();
			std::cout << " invalid grid points removed from hyps_grid." << std::endl;
			n_grid = valid_points.size();

			// update print interval
			print_interval = std::max(1, n_grid / 10);
		}

		// r1_hyps_grid assigned during constructor (ie before this function call)
		int r1_n_grid   = r1_hyps_grid.rows();
		r1_valid_points = validate_grid(r1_hyps_grid, n_var);
		r1_hyps_grid    = subset_matrix(r1_hyps_grid, r1_valid_points);

		if(r1_valid_points.size() == 0){
			throw std::runtime_error("No valid grid points in r1_hyps_grid.");
		} else if(r1_n_grid > r1_valid_points.size()){
			std::cout << "WARNING: " << r1_n_grid - r1_valid_points.size();
			std::cout << " invalid grid points removed from r1_hyps_grid." << std::endl;
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

template <typename T>
inline std::vector<int> validate_grid(const Eigen::MatrixXd &grid, const T n_var){
	const int sigma_ind   = 0;
	const int sigma_b_ind = 1;
	const int sigma_g_ind = 2;
	const int lam_b_ind   = 3;
	const int lam_g_ind   = 4;
	double lam_b, lam_g;
	bool chck_lam_b, chck_lam_g, chck_sigma_b, chck_sigma_g, chck_sigma;

	std::vector<int> valid_points;
	for (int ii = 0; ii < grid.rows(); ii++){
		lam_b = grid(ii, lam_b_ind);
		lam_g = grid(ii, lam_g_ind);

		chck_sigma   = (grid(ii, sigma_ind)   >  0.0);
		chck_sigma_b = (grid(ii, sigma_b_ind) >  0.0);
		chck_sigma_g = (grid(ii, sigma_g_ind) >= 0.0);
		chck_lam_b   = (lam_b >= 1.0 / (double) n_var) && (lam_b < 1.0);
		chck_lam_g   = (lam_g >= 1.0 / (double) n_var) && (lam_g < 1.0);
		if(chck_lam_b && chck_lam_g && chck_sigma && chck_sigma_g && chck_sigma_b){
			valid_points.push_back(ii);
		}
	}
	return valid_points;
}

inline Eigen::MatrixXd subset_matrix(const Eigen::MatrixXd &orig, const std::vector<int> &valid_points){
	int n_cols = orig.cols(), n_rows = valid_points.size();
	Eigen::MatrixXd subset(n_rows, n_cols);

	for(int kk = 0; kk < n_rows; kk++){
		for(int jj = 0; jj < n_cols; jj++){
			subset(kk, jj) = orig(valid_points[kk], jj);
		}
	}
	return subset;
}
#endif
