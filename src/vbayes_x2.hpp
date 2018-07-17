// re-implementation of variational bayes algorithm for 1D GxE
// https://stackoverflow.com/questions/21132538/correct-usage-of-the-eigenref-class
#ifndef VBAYES_X2_HPP
#define VBAYES_X2_HPP

#include <algorithm>
#include <chrono>      // start/end time info
#include <ctime>       // start/end time info
#include <cstdint>    // uint32_t
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <thread>
#include "sys/types.h"
#include "sys/sysinfo.h"
#include "class.h"
#include "vbayes_tracker.hpp"
#include "data.hpp"
#include "utils.hpp"  // sigmoid
#include "my_timer.hpp"
#include "variational_parameters.hpp"
#include "tools/eigen3.3/Dense"
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/filesystem.hpp>

template <typename T>
inline std::vector<int> validate_grid(const Eigen::MatrixXd &grid, const T n_var);
inline Eigen::MatrixXd subset_matrix(const Eigen::MatrixXd &orig, const std::vector<int> &valid_points);
inline std::size_t find_covar_index( std::string colname, std::vector< std::string > col_names );

class VBayesX2 {
public:
	// Constants
	const int    iter_max = 500;
	const double PI = 3.1415926535897;
	const double eps = std::numeric_limits<double>::min();
	const double alpha_tol = 1e-4;
	const double logw_tol = 1e-2;
	const double sigma_c = 10000;
	int print_interval;              // print time every x grid points
	std::vector< std::string > covar_names;

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
	int           n_effects;             // no. interaction variables + 1
	std::uint32_t n_samples;
	std::uint32_t n_covar;
	std::uint32_t n_var;
	std::uint32_t n_var2;
	bool          random_params_init;
	bool          run_round1;

	//
	parameters& p;
	std::vector< std::uint32_t > fwd_pass;
	std::vector< std::uint32_t > back_pass;

	// Data
	GenotypeMatrix& X;
	Eigen::MatrixXd& Y;          // residual phenotype matrix
	Eigen::ArrayXXd dXtX;       // diagonal of X^T x X
	Eigen::ArrayXXd dHtH;       // P x (E+1); dHtH_ij = X_i^T * diag(e_j) * X_i
	Eigen::ArrayXXd Hty;         // vector of H^T x y where H = (X, Z)
	Eigen::ArrayXd  Wty;         // vector of W^T x y where W the matrix of covariates
	Eigen::MatrixXd E;          // matrix of variables used for GxE interactions
	Eigen::MatrixXd& W;          // matrix of covariates (superset of GxE variables)
	Eigen::MatrixXd r1_hyps_grid;
	Eigen::MatrixXd r1_probs_grid;
	Eigen::MatrixXd hyps_grid;
	Eigen::MatrixXd probs_grid; // prob of each point in grid under hyps


	// Init points
	VariationalParametersLite vp_init;

	// boost fstreams
	boost_io::filtering_ostream outf, outf_map, outf_wmean, outf_nmean, outf_inits;
	boost_io::filtering_ostream outf_elbo, outf_alpha_diff, outf_map_pred;

	// Monitoring
	std::chrono::system_clock::time_point time_check;

	MyTimer t_updateAlphaMu;
	MyTimer t_elbo;
	MyTimer t_readXk;
	MyTimer t_updateHyps;
	MyTimer t_InnerLoop;

	explicit VBayesX2( Data& dat ) : X( dat.G ),
                            Y( dat.Y ),
                            W( dat.W ),
                            p( dat.params ),
                            t_updateAlphaMu("updateAlphaMu: %ts \n"),
                            t_elbo("calcElbo: %ts \n"),
                            t_readXk("read_X_kk: %ts \n"),
                            t_updateHyps("updateHyps: %ts \n"),
                            t_InnerLoop("runInnerLoop: %ts \n") {
		assert(std::includes(dat.hyps_names.begin(), dat.hyps_names.end(), hyps_names.begin(), hyps_names.end()));
		assert(p.interaction_analysis);

		// Data size params
		n_effects      = dat.n_effects;
		n_var          = dat.n_var;
		n_var2         = n_effects * dat.n_var;
		n_samples      = dat.n_samples;
		n_covar        = dat.n_covar;
		n_grid         = dat.hyps_grid.rows();
		print_interval = std::max(1, n_grid / 10);
		covar_names = dat.covar_names;

		// Allocate memory - fwd/back pass vectors
		std::uint32_t L;
		if(p.mode_alternating_updates){
			L = n_var;
		} else {
			L = n_var * n_effects;
		}
		for(std::uint32_t kk = 0; kk < L; kk++){
			fwd_pass.push_back(kk);
			back_pass.push_back(n_var2 - kk - 1);
		}

		// Read environmental variables
		if(p.x_param_name != "NULL"){
			std::size_t x_col = find_covar_index(p.x_param_name, dat.covar_names);
			E                = dat.W.col(x_col);
		} else {
			E                = dat.W.col(0);
		}
		X.E = E;  // WARNING: Required to be able to call X.col(jj) with jj > P

		// non random initialisation
		if(p.vb_init_file != "NULL"){
			vp_init.alpha         = dat.alpha_init;
			vp_init.mu            = dat.mu_init;
			if(p.mode_mog_prior){
				vp_init.mup   = Eigen::ArrayXXd::Zero(n_var, n_effects);
			}
			if(p.use_vb_on_covars){
				vp_init.muc = Eigen::ArrayXd::Zero(n_covar);
			}

			// Gen Hr_init
			calcHr(vp_init);

			random_params_init = false;
			run_round1         = false;
			if(p.user_requests_round1){
				run_round1     = true;
			}
		} else {
			random_params_init = true;
			run_round1         = true;
		}

		// Assign data - hyperparameters
		probs_grid          = dat.imprt_grid;
		hyps_grid           = dat.hyps_grid;

		if(p.r1_hyps_grid_file == "NULL"){
			r1_hyps_grid    = hyps_grid;
			r1_probs_grid   = probs_grid;
		} else {
			r1_hyps_grid    = dat.r1_hyps_grid;
			r1_probs_grid   = dat.r1_probs_grid;
		}

		// Assign data - genetic
		Hty.matrix().resize(n_var, n_effects);
		Hty.col(0) = X.transpose_vector_multiply(Y).array();
		for (int ee = 1; ee < n_effects; ee++){
			Hty.col(ee) = X.transpose_vector_multiply(Y.cwiseProduct(E.col(ee-1))).array();
		}

		Wty = W.transpose() * Y;

		//
		// Eigen::VectorXd dXtX(n_var), dZtZ(n_var), col_j;
		// for (std::size_t jj = 0; jj < n_var; jj++){
		// 	col_j = X.col(jj);
		// 	dXtX[jj] = col_j.dot(col_j);
		// 	dZtZ[jj] = col_j.cwiseProduct(a_sq).dot(col_j);
		// }
		// dHtH                   << dXtX, dZtZ;

		dHtH.resize(n_var, n_effects);
		Eigen::VectorXd e_k_sq;
		Eigen::VectorXd col_j;
		for (std::size_t jj = 0; jj < n_var; jj++){
			col_j = X.col(jj);
			dHtH(jj, 0) = col_j.dot(col_j);
		}
		for (int ee = 1; ee < n_effects; ee++){
			e_k_sq = E.col(ee-1).cwiseProduct(E.col(ee-1));
			for (std::size_t jj = 0; jj < n_var; jj++){
				col_j = X.col(jj);
				dHtH(jj, ee) = col_j.cwiseProduct(e_k_sq).dot(col_j);
			}
		}
	}

	~VBayesX2(){
		// Close all ostreams
		io::close(outf);
		io::close(outf_map);
		io::close(outf_wmean);
		io::close(outf_nmean);
		io::close(outf_elbo);
		io::close(outf_alpha_diff);
		io::close(outf_inits);

		// Report all timers
		// t_updateAlphaMu.report();
		// t_elbo.report();
		// t_readXk.report();
		// t_updateHyps.report();
		// t_InnerLoop.report();
	}

	void print_time_check(){
		auto now = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = now-time_check;
		std::cout << " (" << elapsed_seconds.count();
		std::cout << " seconds since last timecheck, estimated RAM usage = ";
		std::cout << getValueRAM() << "KB)" << std::endl;
		time_check = now;
	}

	void run_inference(const Eigen::Ref<const Eigen::MatrixXd>& hyps_grid,
                     const bool random_init,
                     const int round_index,
                     std::vector<VbTracker>& trackers){
		// Writes results from inference to trackers

		int n_grid = hyps_grid.rows();

		// Divide grid of hyperparameters into chunks for multithreading
		std::vector< std::vector< int > > chunks(p.n_thread);
		for (int ii = 0; ii < n_grid; ii++){
			int ch_index = (ii % p.n_thread);
			chunks[ch_index].push_back(ii);
		}

		// Allocate memory for trackers
		for (int ch = 0; ch < p.n_thread; ch++){
			trackers[ch].resize(n_grid);
			trackers[ch].set_main_filepath(p.out_file);
			trackers[ch].p = p;
		}

		// Assign set of start points to each thread & run
		std::thread t2[p.n_thread];
		for (int ch = 1; ch < p.n_thread; ch++){
			t2[ch] = std::thread( [this, round_index, hyps_grid, n_grid, chunks, ch, random_init, &trackers] {
				runOuterLoop(round_index, hyps_grid, n_grid, chunks[ch], random_init, trackers[ch]);
			} );
		}
		runOuterLoop(round_index, hyps_grid, n_grid, chunks[0], random_init, trackers[0]);
		for (int ch = 1; ch < p.n_thread; ch++){
			t2[ch].join();
		}
	}

	void run(){
		std::cout << "Starting variational inference" << std::endl;
		time_check = std::chrono::system_clock::now();
		if(p.n_thread > 1){
			std::cout << "Running on " << p.n_thread << " threads" << std::endl;
		}

		// Round 1; looking for best start point
		if(run_round1){
			std::vector< VbTracker > trackers(p.n_thread);
			int r1_n_grid = r1_hyps_grid.rows();
			run_inference(r1_hyps_grid, true, 1, trackers);

			if(p.verbose){
				write_trackers_to_file("round1_", trackers, r1_hyps_grid, r1_probs_grid);
			}

			// Find best init
			double logw_best = -std::numeric_limits<double>::max();
			bool init_not_set = true;
			for (int ii = 0; ii < r1_n_grid; ii++){
				int tr_index     = (ii % p.n_thread);
				double logw      = trackers[tr_index].logw_list[ii];
				if(std::isfinite(logw) && logw > logw_best){
					vp_init      = trackers[tr_index].vp_list[ii];
					logw_best    = logw;
					init_not_set = false;
				}
			}

			if(init_not_set){
				throw std::runtime_error("No valid start points found (elbo estimates all non-finite?).");
			}

			// gen Hr_init
			calcHr(vp_init);

			// Write inits to file
			std::string ofile_inits = fstream_init(outf_inits, "", "_inits");
			std::cout << "Writing start points for alpha and mu to " << ofile_inits << std::endl;
			// outf_inits << "chr rsid pos a0 a1 alpha mu" << std::endl;
			// for (std::uint32_t kk = 0; kk < n_var2; kk++){
			// 	std::uint32_t kk1 = kk % n_var;
			// 	outf_inits << X.chromosome[kk1] << " " << X.rsid[kk1]<< " " << X.position[kk1];
			// 	outf_inits << " " << X.al_0[kk1] << " " << X.al_1[kk1] << " ";
			// 	outf_inits << vp_init.alpha[kk] << " " << vp_init.mu[kk] << std::endl;
			// }

			outf_inits << "chr rsid pos a0 a1";
			for (int ee = 0; ee < n_effects; ee++){
				outf_inits << " alpha" << ee << " mu" << ee;
			}
			outf_inits << std::endl;
			for (std::uint32_t kk = 0; kk < n_var; kk++){
				outf_inits << X.chromosome[kk] << " " << X.rsid[kk] << " " << X.position[kk];
				outf_inits << " " << X.al_0[kk] << " " << X.al_1[kk];
				for (int ee = 0; ee < n_effects; ee++){
					outf_inits << " " << vp_init.alpha(kk, ee);
					outf_inits << " " << vp_init.mu(kk, ee);
				}
				outf_inits << std::endl;
			}

			print_time_check();
		}

		std::vector< VbTracker > trackers(p.n_thread);
		run_inference(hyps_grid, false, 2, trackers);

		write_trackers_to_file("", trackers, hyps_grid, probs_grid);

		std::cout << "Variational inference finished" << std::endl;
	}

	void calcHr(VariationalParameters& vp){
		Eigen::ArrayXXd rr;
		if(p.mode_mog_prior){
			rr = vp.alpha * (vp.mu - vp.mup) + vp.mup;
		} else {
			rr = vp.alpha * vp.mu;
		}

		vp.Hr = X * rr.col(0).matrix();
		for (int ee = 1; ee < n_effects; ee++){
			vp.Hr += (X * rr.col(ee).matrix()).cwiseProduct(E.col(ee-1));
		}
	}

	void calcHr(VariationalParametersLite& vp){
		Eigen::ArrayXXd rr;
		if(p.mode_mog_prior){
			rr = vp.alpha * (vp.mu - vp.mup) + vp.mup;
		} else {
			rr = vp.alpha * vp.mu;
		}

		vp.Hr = X * rr.col(0).matrix();
		for (int ee = 1; ee < n_effects; ee++){
			vp.Hr += (X * rr.col(ee).matrix()).cwiseProduct(E.col(ee-1));
		}
	}

	void runOuterLoop(const int round_index,
                      const Eigen::Ref<const Eigen::MatrixXd>& outer_hyps_grid,
                      const int outer_n_grid,
                      std::vector<int> grid_index_list,
                      const bool random_init,
                      VbTracker& tracker){

		for (auto ii : grid_index_list){
			if((ii + 1) % print_interval == 0){
				std::cout << "\rRound " << round_index << ": grid point " << ii+1 << "/" << outer_n_grid;
				print_time_check();
			}

			// Unpack hyperparams
			// Hyps i_hyps(n_effects,
			// 	outer_hyps_grid(ii, sigma_ind),
			// 	outer_hyps_grid(ii, sigma_b_ind),
			// 	outer_hyps_grid(ii, sigma_g_ind),
			// 	outer_hyps_grid(ii, lam_b_ind),
			// 	outer_hyps_grid(ii, lam_g_ind));

			double sigma = outer_hyps_grid(ii, sigma_ind);
			double sigma_b = outer_hyps_grid(ii, sigma_b_ind);
			double sigma_g = outer_hyps_grid(ii, sigma_g_ind);
			double lam_b = outer_hyps_grid(ii, lam_b_ind);
			double lam_g = outer_hyps_grid(ii, lam_g_ind);

				// Hyps(int n_effects, double my_sigma, double sigma_b, double sigma_g, double lam_b, double lam_g){
			Hyps i_hyps;
			i_hyps.slab_var.resize(n_effects);
			i_hyps.spike_var.resize(n_effects);
			i_hyps.slab_relative_var.resize(n_effects);
			i_hyps.spike_relative_var.resize(n_effects);
			i_hyps.lambda.resize(n_effects);
				//
			i_hyps.sigma = sigma;
			i_hyps.slab_var           << sigma * sigma_b, sigma * sigma_g;
			i_hyps.spike_var          << sigma * sigma_b / 100.0, sigma * sigma_g / 100.0;
			i_hyps.slab_relative_var  << sigma_b, sigma_g;
			i_hyps.spike_relative_var << sigma_b / 100.0, sigma_g / 100.0;
			i_hyps.lambda             << lam_b, lam_g;
				// }

			// Run outer loop - don't update trackers
			runInnerLoop(ii, random_init, round_index, i_hyps, tracker);
		}
	}

	void runInnerLoop(const int ii,
                      const bool random_init,
                      const int round_index,
                      Hyps hyps,
                      VbTracker& tracker){
		t_InnerLoop.resume();
		// minimise KL Divergence and assign elbo estimate
		// Assumes vp_init already exist
		VariationalParameters vp;

		// Assign initial values
		if (random_init) {
			initRandomAlphaMu(vp);
		} else {
			vp.init_from_lite(vp_init);
		}

		// Update s_sq
		updateSSq(hyps, vp);

		// Run inner loop until convergence
		int count = 0;
		bool converged = false;
		Eigen::ArrayXXd alpha_prev;
		std::vector< std::uint32_t > iter;
		double i_logw = calc_logw(hyps, vp);
		std::vector< double > logw_updates, alpha_diff_updates;
		logw_updates.push_back(i_logw);
		tracker.interim_output_init(ii, round_index, n_effects);
		while(!converged  && count < iter_max){
			alpha_prev = vp.alpha;
			double logw_prev = i_logw;
			long int hty_update_counter = 0;

			// Alternate between back and fwd passes
			if(count % 2 == 0){
				iter = fwd_pass;
			} else {
				iter = back_pass;
			}

			if(p.use_vb_on_covars){
				updateCovarEffects(vp, hyps, hty_update_counter);
			}

			// Update variational_parameters alpha, mu
			count++;
			updateAlphaMu(iter, hyps, vp, hty_update_counter);

			// Log updates
			i_logw     = calc_logw(hyps, vp);

			double alpha_diff = (alpha_prev - vp.alpha).abs().maxCoeff();
			alpha_diff_updates.push_back(alpha_diff);

			tracker.push_interim_iter_update(count, hyps, i_logw, alpha_diff,
				t_updateAlphaMu.get_lap_seconds(), hty_update_counter, n_effects, n_var, vp);

			// Update hyps
			if(round_index > 1 && p.mode_empirical_bayes){
				t_updateHyps.resume();
				updateHyps(hyps, vp);
				updateSSq(hyps, vp);

				i_logw     = calc_logw(hyps, vp);
				t_updateHyps.stop();

				tracker.push_interim_iter_update(count, hyps, i_logw, 0.0,
					t_updateHyps.get_lap_seconds(), -1, n_effects, n_var, vp);
			}
			logw_updates.push_back(i_logw);

			// Diagnose convergence
			double logw_diff  = i_logw - logw_prev;
			if(p.alpha_tol_set_by_user && p.elbo_tol_set_by_user){
				if(alpha_diff < p.alpha_tol && logw_diff < p.elbo_tol){
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
				//  Monotonic trajectory no longer required under EB?
				converged = true;
			} else {
				if(alpha_diff < alpha_tol && logw_diff < logw_tol){
					converged = true;
				}
			}
		}

		if(!std::isfinite(i_logw)){
			std::cout << "WARNING: non-finite elbo estimate produced" << std::endl;
		}

		// Log all things that we want to track
		t_InnerLoop.stop();
		tracker.logw_list[ii] = i_logw;
		tracker.counts_list[ii] = count;
		tracker.vp_list[ii] = vp.convert_to_lite();
		tracker.elapsed_time_list[ii] = t_InnerLoop.get_lap_seconds();
		tracker.hyps_list[ii] = hyps;
		if(p.verbose){
			logw_updates.push_back(i_logw);  // adding converged estimate
			tracker.logw_updates_list[ii] = logw_updates;
			tracker.alpha_diff_list[ii] = alpha_diff_updates;
		}
		tracker.push_interim_output(ii, X.chromosome, X.rsid, X.position, X.al_0, X.al_1, n_var, n_effects);
	}

	void updateSSq(const Hyps& hyps,
                   VariationalParameters& vp){
		// Update s_sq
		vp.s_sq.resize(n_var, n_effects);
		if(p.mode_mog_prior){
			vp.sp_sq.resize(n_var, n_effects);
		}

		for (int ee = 0; ee < n_effects; ee++){
			for (std::uint32_t kk = 0; kk < n_var; kk++){
				vp.s_sq(kk, ee)  = hyps.slab_var(ee);
				vp.s_sq(kk, ee) /= (hyps.slab_relative_var(ee) * dHtH(kk, ee) + 1.0);
			}
		}

		if(p.mode_mog_prior){
			for (int ee = 0; ee < n_effects; ee++){
				for (std::uint32_t kk = 0; kk < n_var; kk++){
					vp.sp_sq(kk, ee)  = hyps.spike_var(ee);
					vp.sp_sq(kk, ee) /= (hyps.spike_relative_var(ee) * dHtH(kk, ee) + 1.0);
				}
			}
		}
	}

	void updateHyps(Hyps& hyps,
                    const VariationalParameters& vp){
		// max sigma
		Eigen::ArrayXXd varB(n_var, n_effects);
		calcVarqBeta(hyps, vp, varB);
		hyps.sigma  = (Y - vp.Hr).squaredNorm();
		hyps.sigma += (dHtH * varB).sum();
		hyps.sigma /= n_samples;

		// max lambda
		hyps.lambda = vp.alpha.colwise().sum();

		// max spike & slab variances
		hyps.slab_var  = (vp.alpha * (vp.s_sq + vp.mu.square())).colwise().sum();
		hyps.slab_var /= hyps.lambda;
		hyps.slab_relative_var = hyps.slab_var / hyps.sigma;
		if(p.mode_mog_prior){
			hyps.spike_var  = ((1.0 - vp.alpha) * (vp.sp_sq + vp.mup.square())).colwise().sum();
			hyps.spike_var /= ( (double)n_var - hyps.lambda);
			hyps.spike_relative_var = hyps.spike_var / hyps.sigma;
		}

		// finish max lambda
		hyps.lambda /= n_var;

		// hyps.lam_b         = hyps.lambda(0);
		// hyps.lam_g         = hyps.lambda(1);
		// hyps.sigma_b       = hyps.slab_relative_var(0);
		// hyps.sigma_g       = hyps.slab_relative_var(1);
		// hyps.sigma_g_spike = hyps.spike_relative_var(0);
		// hyps.sigma_g_spike = hyps.spike_relative_var(1);
	}

	void updateAlphaMu(const std::vector< std::uint32_t >& iter,
                       const Hyps& hyps,
                       VariationalParameters& vp,
                       long int& hty_updates){
		t_updateAlphaMu.resume();
		Eigen::VectorXd X_kk(n_samples);
		Eigen::VectorXd Z_kk(n_samples);

		Eigen::ArrayXd alpha_cnst;
		if(p.mode_mog_prior){
			alpha_cnst  = (hyps.lambda / (1.0 - hyps.lambda) + eps).log();
			alpha_cnst -= (hyps.slab_var.log() - hyps.spike_var.log()) / 2.0;
		} else {
			alpha_cnst = (hyps.lambda / (1.0 - hyps.lambda) + eps).log() - hyps.slab_var.log() / 2.0;
		}

		for(std::uint32_t kk : iter ){
			int ee            = kk / n_var;
			std::uint32_t jj = (kk % n_var);

			t_readXk.resume();
			X_kk = X.col(kk);
			t_readXk.stop();

			_internal_updateAlphaMu(X_kk, ee, jj, hty_updates, vp, hyps, alpha_cnst);

			if(p.mode_alternating_updates){
				for(int ee = 1; ee < n_effects; ee++){
					Z_kk = X_kk.cwiseProduct(E.col(ee-1));
					_internal_updateAlphaMu(Z_kk, ee, jj, hty_updates, vp, hyps, alpha_cnst);
				}
			}
		}
		t_updateAlphaMu.stop();
	}

	void _internal_updateAlphaMu(const Eigen::Ref<const Eigen::VectorXd>& H_kk,
								 const int& ee, std::uint32_t jj, long int& hty_updates,
								 VariationalParameters& vp,
								 const Hyps& hyps,
								 const Eigen::Ref<const Eigen::ArrayXd>& alpha_cnst) __attribute__ ((hot)){
		//
		double rr_k_diff;

		if(p.mode_mog_prior){
			double rr_k = vp.alpha(jj, ee) * (vp.mu(jj, ee) - vp.mup(jj, ee)) + vp.mup(jj, ee);

			// Update mu (eq 9); faster to take schur product inside genotype_matrix
			double A       = Hty(jj, ee) - vp.Hr.dot(H_kk) + dHtH(jj, ee) * rr_k;
			vp.mu(jj, ee)  = vp.s_sq(jj, ee) * A / hyps.sigma;
			vp.mup(jj, ee) = vp.sp_sq(jj, ee) * A / hyps.sigma;

			// Update alpha (eq 10)  TODO: check syntax / i_  / sigmoid here!
			double ff_k      = vp.mu(jj, ee) * vp.mu(jj, ee) / vp.s_sq(jj, ee)  / 2.0;
			ff_k            -= vp.mup(jj, ee) * vp.mup(jj, ee) / vp.sp_sq(jj, ee) / 2.0;
			ff_k            += (std::log(vp.s_sq(jj, ee)) - std::log(vp.sp_sq(jj, ee))) / 2.0;
			vp.alpha(jj, ee) = sigmoid(ff_k + alpha_cnst(ee));

			// Update i_Hr; only if coeff is large enough to matter
			double rr_k_new  = vp.alpha(jj, ee) * (vp.mu(jj, ee) - vp.mup(jj, ee)) + vp.mup(jj, ee);
			rr_k_diff        = rr_k_new - rr_k;
		} else {

			double rr_k = vp.alpha(jj, ee) * vp.mu(jj, ee);

			// Update mu (eq 9); faster to take schur product inside genotype_matrix
			vp.mu(jj, ee) = vp.s_sq(jj, ee) * (Hty(jj, ee) - vp.Hr.dot(H_kk) + dHtH(jj, ee) * rr_k) / hyps.sigma;

			// Update alpha (eq 10)
			double ff_k      = (std::log(vp.s_sq(jj, ee)) + vp.mu(jj, ee) * vp.mu(jj, ee) / vp.s_sq(jj, ee)) / 2.0;
			vp.alpha(jj, ee) = sigmoid(ff_k + alpha_cnst(ee));

			// Update i_Hr; only if coeff is large enough to matter
			rr_k_diff        = vp.alpha(jj, ee) * vp.mu(jj, ee) - rr_k;
		}
		if(!p.mode_approximate_residuals || std::abs(rr_k_diff) > p.min_residuals_diff){
			hty_updates++;
			vp.Hr += rr_k_diff * H_kk;
		}
	}

	void updateCovarEffects(VariationalParameters& vp,
                            const Hyps& hyps,
                            long int& hty_updates) __attribute__ ((hot)){
		//
		double N = n_samples;
		for (int ww = 0; ww < n_covar; ww++){
			double rr_k = vp.muc(ww);
			double sc_sq  = hyps.sigma * sigma_c / (sigma_c * ((double) n_samples - 1.0) + 1.0);

			// Update mu (eq 9); faster to take schur product inside genotype_matrix
			vp.muc(ww) = sc_sq * (Wty(ww) - vp.Hr.dot(W.col(ww)) + (N - 1.0) * rr_k) / hyps.sigma;

			// Update i_Hr; only if coeff is large enough to matter
			double rr_k_diff     = vp.muc(ww) - rr_k;
			if(!p.mode_approximate_residuals || std::abs(rr_k_diff) > p.min_residuals_diff){
				hty_updates++;
				vp.Hr += rr_k_diff * W.col(ww);
			}
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

		int nonfinite_count = 0;
		for (int ii = 0; ii < nn; ii++){
			if(!std::isfinite(my_weights[ii])){
				nonfinite_count++;
			}
		}

		if(nonfinite_count > 0){
			std::cout << "WARNING: " << nonfinite_count << " grid points returned non-finite ELBO.";
			std::cout << "Skipping these when producing posterior estimates.";
		}
	}

	void initRandomAlphaMu(VariationalParameters& vp){
		// vp.alpha a uniform simplex, vp.mu standard gaussian
		// Also sets vp.Hr
		std::default_random_engine gen_gauss, gen_unif;
		std::normal_distribution<double> gaussian(0.0,1.0);
		std::uniform_real_distribution<double> uniform(0.0,1.0);

		// Allocate memory
		vp.mu.resize(n_var, n_effects);
		vp.alpha.resize(n_var, n_effects);
		if(p.mode_mog_prior){
			vp.mup = Eigen::ArrayXXd::Zero(n_var, n_effects);
		}

		// Random initialisation of alpha, mu
		for (int ee = 0; ee < n_effects; ee++){
			for (std::uint32_t kk = 0; kk < n_var; 	kk++){
				vp.alpha(kk, ee) = uniform(gen_unif);
				vp.mu(kk, ee)    = gaussian(gen_gauss);
			}
		}

		// Convert alpha to simplex. Why?
		vp.alpha.rowwise() /= vp.alpha.colwise().sum();

		if(p.use_vb_on_covars){
			vp.muc = Eigen::ArrayXd::Zero(n_covar);
		}

		// Gen Hr.
		calcHr(vp);
	}

	double calc_logw(const Hyps& hyps,
                     const VariationalParameters& vp){
		t_elbo.resume();

		// Expectation of linear regression log-likelihood
		double int_linear = calcIntLinear(hyps, vp);

		// gamma
		Eigen::ArrayXd col_sums = vp.alpha.colwise().sum();
		double int_gamma = 0;
		for (int ee = 0; ee < n_effects; ee++){
			int_gamma += col_sums(ee) * std::log(hyps.lambda(ee) + eps);
			int_gamma -= col_sums(ee) * std::log(1.0 - hyps.lambda(ee) + eps);
			int_gamma += (double) n_var *  std::log(1.0 - hyps.lambda(ee) + eps);
		}

		// kl-beta
		double int_klbeta = calcIntKLBeta(hyps, vp);

		// covariates
		if(p.use_vb_on_covars){
			// TODO: elbo should reflect covar estimates
		}

		double res = int_linear + int_gamma + int_klbeta;

		t_elbo.stop();
		return res;
	}

	void calcVarqBeta(const Hyps& hyps,
                      const VariationalParameters& vp,
                      Eigen::Ref<Eigen::ArrayXXd> varB){
		// Variance of effect size beta under approximating distribution q(u, beta)
		assert(varB.rows() == n_var);
		assert(varB.cols() == n_effects);

		varB = vp.alpha * (vp.s_sq + (1.0 - vp.alpha) * vp.mu.square());
		if(p.mode_mog_prior){
			varB += (1.0 - vp.alpha) * (vp.sp_sq + (vp.alpha) * vp.mup.square());
			varB -= 2.0 * vp.alpha * (1.0 - vp.alpha) * vp.mu * vp.mup;
		}
	}

	double calcIntLinear(const Hyps& hyps,
                         const VariationalParameters& vp){
		// Expectation of linear regression log-likelihood
		double int_linear = 0;
		Eigen::ArrayXXd varB(n_var, n_effects);
		calcVarqBeta(hyps, vp, varB);

		// Expectation of linear regression log-likelihood
		int_linear -= ((double) n_samples) * std::log(2.0 * PI * hyps.sigma) / 2.0;
		int_linear -= (Y - vp.Hr).squaredNorm() / 2.0 / hyps.sigma;
		int_linear -= 0.5 * (dHtH * varB).sum() / hyps.sigma;

		return int_linear;
	}

	double calcIntKLBeta(const Hyps& hyps,
                         const VariationalParameters& vp){
		// KL Divergence of log[ p(beta | u, theta) / q(u, beta) ]
		double col_sum, int_klbeta;

		if(p.mode_mog_prior){
			int_klbeta  = n_var * n_effects / 2.0;

			int_klbeta -= ((vp.alpha * (vp.mu.square() + vp.s_sq)).colwise().sum().transpose() / 2.0 / hyps.slab_var).sum();
			int_klbeta += (vp.alpha * vp.s_sq.log()).sum() / 2.0;

			int_klbeta -= (((1.0 - vp.alpha) * (vp.mup.square() + vp.sp_sq)).colwise().sum().transpose() / 2.0 / hyps.spike_var).sum();
			int_klbeta += ((1.0 - vp.alpha) * vp.sp_sq.log()).sum() / 2.0;

			for (int ee = 0; ee < n_effects; ee++){
				col_sum = vp.alpha.col(ee).sum();
				int_klbeta -= std::log(hyps.slab_var(ee))  * col_sum / 2.0;
				int_klbeta -= std::log(hyps.spike_var(ee)) * (n_var - col_sum) / 2.0;
			}
		} else {
			int_klbeta  = (vp.alpha * vp.s_sq.log()).sum() / 2.0;
			int_klbeta -= ((vp.alpha * (vp.mu.square() + vp.s_sq)).colwise().sum().transpose() / 2.0 / hyps.slab_var).sum();

			for (int ee = 0; ee < n_effects; ee++){
				col_sum = vp.alpha.col(ee).sum();
				int_klbeta += col_sum * (1 - std::log(hyps.slab_var(ee))) / 2.0;
			}
		}

		for (int ee = 0; ee < n_effects; ee++){
			for (std::uint32_t kk = 0; kk < n_var; kk++){
				int_klbeta -= vp.alpha(kk, ee) * std::log(vp.alpha(kk, ee) + eps);
				int_klbeta -= (1 - vp.alpha(kk, ee)) * std::log(1 - vp.alpha(kk, ee) + eps);
			}
		}
		return int_klbeta;
	}

	void write_trackers_to_file(const std::string& file_prefix,
                                const std::vector< VbTracker >& trackers,
                                const Eigen::Ref<const Eigen::MatrixXd>& hyps_grid,
                                const Eigen::Ref<const Eigen::VectorXd>& my_probs_grid){
		// Stitch trackers back together if using multithreading
		int my_n_grid = hyps_grid.rows();
		assert(my_n_grid == my_probs_grid.rows());
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

		std::string ofile       = fstream_init(outf, file_prefix, "");
		std::string ofile_map   = fstream_init(outf_map, file_prefix, "_map_snp_stats");
		std::string ofile_wmean = fstream_init(outf_wmean, file_prefix, "_weighted_mean_snp_stats");
		std::string ofile_nmean = fstream_init(outf_nmean, file_prefix, "_niave_mean_snp_stats");
		std::string ofile_map_yhat = fstream_init(outf_map_pred, file_prefix, "_map_yhat");
		std::cout << "Writing converged hyperparameter values to " << ofile << std::endl;
		std::cout << "Writing MAP snp stats to " << ofile_map << std::endl;
		std::cout << "Writing (weighted) average snp stats to " << ofile_wmean << std::endl;
		std::cout << "Writing (niave) average snp stats to " << ofile_nmean << std::endl;
		std::cout << "Writing yhat from map to " << ofile_map_yhat << std::endl;

		if(p.verbose){
			std::string ofile_elbo = fstream_init(outf_elbo, file_prefix, "_elbo");
			std::cout << "Writing ELBO from each VB iteration to " << ofile_elbo << std::endl;

			std::string ofile_alpha_diff = fstream_init(outf_alpha_diff, file_prefix, "_alpha_diff");
			std::cout << "Writing max change in alpha from each VB iteration to " << ofile_alpha_diff << std::endl;
		}
	}

	void output_results(const VbTracker& stitched_tracker, const int my_n_grid,
						const Eigen::Ref<const Eigen::VectorXd>& my_probs_grid){
		// Write;
		// main output; weights logw converged_hyps counts time (currently no prior)
		// snps;
		// - map (outf_map) / elbo weighted mean (outf_wmean) / niave mean + sds (outf_nmean)
		// (verbose);
		// - elbo trajectories (inside the interim files?)
		// - hyp trajectories (inside the interim files)

		// Compute normalised weights using finite elbo
		std::vector< double > weights(my_n_grid);
		if(n_grid > 1){
			for (int ii = 0; ii < my_n_grid; ii++){
				if(p.mode_empirical_bayes){
					weights[ii] = stitched_tracker.logw_list[ii];
				} else {
					weights[ii] = stitched_tracker.logw_list[ii] + std::log(my_probs_grid(ii,0) + eps);
				}
			}
			normaliseLogWeights(weights);
		} else {
			weights[0] = 1;
		}

		// Compute heritability
		Eigen::ArrayXXd pve(my_n_grid, n_effects);
		Eigen::ArrayXXd pve_large(my_n_grid, n_effects);
		Eigen::ArrayXd s_x = dHtH.colwise().sum() / ((double) n_samples - 1.0);

		for (int ii = 0; ii < my_n_grid; ii++){
			Hyps hyps = stitched_tracker.hyps_list[ii];
			Eigen::ArrayXd pve_i(n_effects);
			pve_i = hyps.lambda * hyps.slab_relative_var * s_x;
			if(p.mode_mog_prior){
				pve_large.row(ii) = pve_i;
				pve_i += (1 - hyps.lambda) * hyps.spike_relative_var * s_x;
				pve_large.row(ii) /= (pve_i.sum() + 1.0);
			}
			pve_i /= (pve_i.sum() + 1.0);
			pve.row(ii) = pve_i;
		}

		// Write hyperparams weights to file
		outf << "weight logw";
		if(!p.mode_empirical_bayes){
			outf << " log_prior";
		}
		outf << " count time sigma";
		for (int ee = 0; ee < n_effects; ee++){
			outf << " pve" << ee;
			if(p.mode_mog_prior){
				outf << " pve_large" << ee;
 			}
			outf << " sigma" << ee;
			if(p.mode_mog_prior){
				outf << " sigma_spike" << ee;
			}
			outf << " lambda" << ee;
		}
		outf << std::endl;

		for (int ii = 0; ii < my_n_grid; ii++){
			outf << std::setprecision(4) << weights[ii] << " ";
			outf << stitched_tracker.logw_list[ii] << " ";
			if(!p.mode_empirical_bayes){
				outf << std::log(my_probs_grid(ii,0) + eps) << " ";
			}
			outf << stitched_tracker.counts_list[ii] << " ";
			outf << stitched_tracker.elapsed_time_list[ii] <<  " ";
			outf << stitched_tracker.hyps_list[ii].sigma;

			outf << std::setprecision(8) << std::fixed;
			for (int ee = 0; ee < n_effects; ee++){
				outf << " " << pve(ii, ee);
				if(p.mode_mog_prior){
					outf << " " << pve_large(ii, ee);
				}
				outf << " " << stitched_tracker.hyps_list[ii].slab_relative_var(ee);
				if(p.mode_mog_prior){
					outf << " " << stitched_tracker.hyps_list[ii].spike_relative_var(ee);
				}
				outf << " " << stitched_tracker.hyps_list[ii].lambda(ee);
			}
			outf << std::endl;
		}

		// Extract snp-stats averaged over runs
		Eigen::ArrayXXd wmean_alpha = Eigen::ArrayXXd::Zero(n_var, n_effects);
		Eigen::ArrayXXd wmean_beta  = Eigen::ArrayXXd::Zero(n_var, n_effects);
		Eigen::ArrayXXd nmean_alpha = Eigen::ArrayXXd::Zero(n_var, n_effects);
		Eigen::ArrayXXd nmean_beta  = Eigen::ArrayXXd::Zero(n_var, n_effects);
		for (int ii = 0; ii < my_n_grid; ii++){
			if(std::isfinite(weights[ii])){
				wmean_alpha += weights[ii] * stitched_tracker.vp_list[ii].alpha;
				wmean_beta  += weights[ii] * stitched_tracker.vp_list[ii].alpha * stitched_tracker.vp_list[ii].mu;
				nmean_alpha += stitched_tracker.vp_list[ii].alpha;
				nmean_beta  += stitched_tracker.vp_list[ii].alpha * stitched_tracker.vp_list[ii].mu;
			}
		}
		nmean_alpha /= (double) my_n_grid;
		nmean_beta  /= (double) my_n_grid;

		Eigen::ArrayXXd nmean_alpha_sd = Eigen::ArrayXXd::Zero(n_var, n_effects);
		Eigen::ArrayXXd nmean_beta_sd  = Eigen::ArrayXXd::Zero(n_var, n_effects);
		for (int ii = 0; ii < my_n_grid; ii++){
			if(std::isfinite(weights[ii])){
				nmean_alpha_sd += (stitched_tracker.vp_list[ii].alpha - nmean_alpha).square();
				nmean_beta_sd  += (stitched_tracker.vp_list[ii].alpha * stitched_tracker.vp_list[ii].mu - nmean_beta).square();
			}
		}
		nmean_alpha_sd /= (double) (my_n_grid - 1);
		nmean_beta_sd  /= (double) (my_n_grid - 1);

		// MAP snp-stats to file
		outf_map << "chr rsid pos a0 a1";
		for (int ee = 0; ee < n_effects; ee++){
			outf_map << " alpha" << ee << " beta" << ee;
		}
		outf_map << std::endl;

		outf_map << std::setprecision(9) << std::fixed;
		int ii_map = std::distance(weights.begin(), std::max_element(weights.begin(), weights.end()));
		if(p.use_vb_on_covars){
			for (int cc = 0; cc < n_covar; cc++){
				outf_map << "NA " << covar_names[cc] << " NA NA NA " << 1;
 				outf_map <<  " " << stitched_tracker.vp_list[ii_map].muc(cc);
				for (int ee = 1; ee < n_effects; ee++){
					outf_map << " NA NA";
				}
				outf_map << std::endl;
			}
		}
		for (std::uint32_t kk = 0; kk < n_var; kk++){
			outf_map << X.chromosome[kk] << " " << X.rsid[kk] << " " << X.position[kk];
			outf_map << " " << X.al_0[kk] << " " << X.al_1[kk];
			for (int ee = 0; ee < n_effects; ee++){
				outf_map << " " << stitched_tracker.vp_list[ii_map].alpha(kk, ee);
				outf_map << " " << stitched_tracker.vp_list[ii_map].alpha(kk, ee) * stitched_tracker.vp_list[ii_map].mu(kk, ee);
			}
			outf_map << std::endl;
		}

		// Weighted mean snp-stats to file
		outf_wmean << "chr rsid pos a0 a1";
		for (int ee = 0; ee < n_effects; ee++){
			outf_wmean << " alpha" << ee << " beta" << ee;
		}
		outf_wmean << std::endl;

		outf_wmean << std::setprecision(9) << std::fixed;
		for (std::uint32_t kk = 0; kk < n_var; kk++){
			outf_wmean << X.chromosome[kk] << " " << X.rsid[kk] << " " << X.position[kk];
			outf_wmean << " " << X.al_0[kk] << " " << X.al_1[kk];
			for (int ee = 0; ee < n_effects; ee++){
				outf_wmean << " " << wmean_alpha(kk, ee);
				outf_wmean << " " << wmean_beta(kk, ee);
			}
			outf_wmean << std::endl;
		}

		// Niave mean snp-stats to file
		outf_nmean << "chr rsid pos a0 a1";
		for (int ee = 0; ee < n_effects; ee++){
			outf_nmean << " alpha" << ee << " alpha" << ee << "_sd";
			outf_nmean << " beta" << ee << " beta" << ee << "_sd";
		}
		outf_nmean << std::endl;

		outf_nmean << std::setprecision(9) << std::fixed;
		for (std::uint32_t kk = 0; kk < n_var; kk++){
			outf_nmean << X.chromosome[kk] << " " << X.rsid[kk] << " " << X.position[kk];
			outf_nmean << " " << X.al_0[kk] << " " << X.al_1[kk];
			for (int ee = 0; ee < n_effects; ee++){
				outf_nmean << " " << nmean_alpha(kk, ee);
				outf_nmean << " " << nmean_alpha_sd(kk, ee);
				outf_nmean << " " << nmean_beta(kk, ee);
				outf_nmean << " " << nmean_beta_sd(kk, ee);
			}
			outf_nmean << std::endl;
		}

		// Predicted effects to file
		outf_map_pred << "pred" << std::endl;
		VariationalParametersLite vp_map = stitched_tracker.vp_list[ii_map];
		calcHr(vp_map);
		for (std::uint32_t ii = 0; ii < n_samples; ii++ ){
			outf_map_pred << -(vp_map.Hr(ii) - Y(ii, 0)) << std::endl;
		}

		if(p.verbose){
			outf_elbo << std::setprecision(4) << std::fixed;
			for (int ii = 0; ii < my_n_grid; ii++){
				for (int cc = 0; cc < stitched_tracker.logw_updates_list[ii].size(); cc++){
					outf_elbo << stitched_tracker.logw_updates_list[ii][cc] << " ";
				}
				outf_elbo << std::endl;
			}

			outf_alpha_diff << std::setprecision(4) << std::fixed;
			for (int ii = 0; ii < my_n_grid; ii++){
				for (int cc = 0; cc < stitched_tracker.alpha_diff_list[ii].size(); cc++){
					outf_alpha_diff << stitched_tracker.alpha_diff_list[ii][cc] << " ";
				}
				outf_alpha_diff << std::endl;
			}
		}
	}

	std::string fstream_init(boost_io::filtering_ostream& my_outf,
                             const std::string& file_prefix,
                             const std::string& file_suffix){

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

	std::vector<int> valid_points;
	for (int ii = 0; ii < grid.rows(); ii++){
		double lam_b = grid(ii, lam_b_ind);
		double lam_g = grid(ii, lam_g_ind);

		bool chck_sigma   = (grid(ii, sigma_ind)   >  0.0);
		bool chck_sigma_b = (grid(ii, sigma_b_ind) >  0.0);
		bool chck_sigma_g = (grid(ii, sigma_g_ind) >= 0.0);
		bool chck_lam_b   = (lam_b >= 1.0 / (double) n_var) && (lam_b < 1.0);
		bool chck_lam_g   = (lam_g >= 1.0 / (double) n_var) && (lam_g < 1.0);
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
