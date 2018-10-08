/* re-implementation of variational bayes algorithm for 1D GxE

How to use Eigen::Ref:
https://stackoverflow.com/questions/21132538/correct-usage-of-the-eigenref-class

Building static executable:
https://stackoverflow.com/questions/3283021/compile-a-standalone-static-executable
*/
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
#include "class.h"
#include "vbayes_tracker.hpp"
#include "data.hpp"
#include "utils.hpp"  // sigmoid
#include "my_timer.hpp"
#include "variational_parameters.hpp"
#include "tools/eigen3.3/Dense"
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/filesystem.hpp>

namespace boost_m = boost::math;

template <typename T>
inline std::vector<int> validate_grid(const Eigen::MatrixXd &grid, const T n_var);
inline Eigen::MatrixXd subset_matrix(const Eigen::MatrixXd &orig, const std::vector<int> &valid_points);

class VBayesX2 {
public:
	// Constants
	const double PI = 3.1415926535897;
	const double eps = std::numeric_limits<double>::min();
	const double alpha_tol = 1e-4;
	const double logw_tol = 1e-2;
	const double sigma_c = 10000;
	const double spike_diff_factor = 1000000.0; // Initial diff in variance of spike & slab
	int print_interval;              // print time every x grid points
	std::vector< std::string > covar_names;
	std::vector< std::string > env_names;

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
	std::uint32_t n_env;
	std::uint32_t n_var;
	std::uint32_t n_var2;
	bool          random_params_init;
	bool          run_round1;
	double N; // (double) n_samples


	//
	parameters& p;
	std::vector< std::uint32_t > fwd_pass;
	std::vector< std::uint32_t > back_pass;
	std::vector< std::vector < std::uint32_t >> fwd_pass_chunks;
	std::vector< std::vector < std::uint32_t >> back_pass_chunks;
	std::vector< int > env_fwd_pass;
	std::vector< int > env_back_pass;
	std::map<long int, Eigen::MatrixXd> D_correlations;

	// Data
	GenotypeMatrix& X;
	Eigen::VectorXd Y;          // residual phenotype matrix
	Eigen::ArrayXXd dXtX;       // diagonal of X^T x X
	Eigen::ArrayXXd& dXtEEX;     // P x n_env^2; col (l * n_env + m) is the diagonal of X^T * diag(E_l * E_m) * X
	Eigen::ArrayXd  Cty;         // vector of W^T x y where C the matrix of covariates
	Eigen::ArrayXXd E;          // matrix of variables used for GxE interactions
	Eigen::MatrixXd& C;          // matrix of covariates (superset of GxE variables)
	Eigen::MatrixXd r1_hyps_grid;
	Eigen::MatrixXd r1_probs_grid;
	Eigen::MatrixXd hyps_grid;
	Eigen::MatrixXd probs_grid; // prob of each point in grid under hyps

	// genome wide scan computed upstream
	Eigen::ArrayXXd& snpstats;

	// Init points
	VariationalParametersLite vp_init;

	// boost fstreams
	boost_io::filtering_ostream outf, outf_map, outf_wmean, outf_nmean, outf_inits;
	boost_io::filtering_ostream outf_elbo, outf_alpha_diff, outf_map_pred, outf_w;
	boost_io::filtering_ostream outf_scan;

	// Monitoring
	std::chrono::system_clock::time_point time_check;

	MyTimer t_updateAlphaMu;
	MyTimer t_elbo;
	MyTimer t_maximiseHyps;
	MyTimer t_InnerLoop;
	MyTimer t_snpwise_regression;
	MyTimer t_readXk;

	// sgd
	double minibatch_adjust;

	explicit VBayesX2( Data& dat ) : X( dat.G ),
                            Y(Eigen::Map<Eigen::VectorXd>(dat.Y.data(), dat.Y.rows())),
                            C( dat.W ),
                            dXtEEX( dat.dXtEEX ),
                            snpstats( dat.snpstats ),
                            p( dat.params ),
                            t_updateAlphaMu("updateAlphaMu: %ts \n"),
                            t_elbo("calcElbo: %ts \n"),
                            t_maximiseHyps("maximiseHyps: %ts \n"),
                            t_InnerLoop("runInnerLoop: %ts \n"),
                            t_readXk("read_X_kk: %ts \n"),
                            t_snpwise_regression("calc_snpwise_regression: %ts \n") {
		assert(std::includes(dat.hyps_names.begin(), dat.hyps_names.end(), hyps_names.begin(), hyps_names.end()));
		assert(p.interaction_analysis);
		std::cout << "Initialising vbayes object" << std::endl;

		// Data size params
		n_effects      = dat.n_effects;
		n_var          = dat.n_var;
		n_env          = dat.n_env;
		n_var2         = n_effects * dat.n_var;
		n_samples      = dat.n_samples;
		n_covar        = dat.n_covar;
		n_grid         = dat.hyps_grid.rows();
		print_interval = std::max(1, n_grid / 10);
		covar_names    = dat.covar_names;
		env_names      = dat.env_names;
		N              = (double) n_samples;

		p.vb_chunk_size = (int) std::min((long int) p.vb_chunk_size, (long int) n_samples);

		// Read environmental variables
		E = dat.E;

		// Allocate memory - fwd/back pass vectors
		std::cout << "Allocating indices for fwd/back passes" << std::endl;

		for(std::uint32_t kk = 0; kk < n_var * n_effects; kk++){
			fwd_pass.push_back(kk);
			back_pass.push_back(n_var2 - kk - 1);
		}

		for(int ll = 0; ll < n_env; ll++){
			env_fwd_pass.push_back(ll);
			env_back_pass.push_back(n_env - ll - 1);
		}

		int n_segs = (n_var + p.vb_chunk_size - 1) / p.vb_chunk_size; // ceiling of n_var / chunk size
		int n_chunks = n_segs * n_effects;
		fwd_pass_chunks.resize(n_chunks);
		back_pass_chunks.resize(n_chunks);
		for(std::uint32_t kk = 0; kk < n_effects * n_var; kk++){
			std::uint32_t ch_index = ((kk % n_var)/ p.vb_chunk_size) + (kk / n_var) * n_segs;
			fwd_pass_chunks[ch_index].push_back(kk);

			std::uint32_t kk_bck = n_effects * n_var - 1 - kk;
			std::uint32_t ch_bck_index = n_chunks - 1 - ch_index;
			back_pass_chunks[ch_index].push_back(kk_bck);
		}


//		for (auto chunk : fwd_pass_chunks){
//			for (auto kk : chunk){
//				std::cout << kk << " ";
//			}
//			std::cout << std::endl;
//		}
//
//		for (auto chunk : back_pass_chunks){
//			for (auto kk : chunk){
//				std::cout << kk << " ";
//			}
//			std::cout << std::endl;
//		}


		// non random initialisation
		if(p.vb_init_file != "NULL"){
			std::cout << "Initialisation - set from file" << std::endl;
			vp_init.alpha         = dat.alpha_init;
			vp_init.mu1            = dat.mu_init;
			if(p.mode_mog_prior_beta || p.mode_mog_prior_gam){
				vp_init.mu2   = Eigen::ArrayXXd::Zero(n_var, n_effects);
			}
			if(p.use_vb_on_covars){
				vp_init.muc = Eigen::ArrayXd::Zero(n_covar);
			}

			if(p.env_weights_file != "NULL"){
				vp_init.muw     = dat.E_weights.col(0);
			} else if (p.init_weights_with_snpwise_scan){
				calc_snpwise_regression(vp_init);
			} else {
				vp_init.muw.resize(n_env);
				vp_init.muw     = 1.0 / (double) n_env;
			}
			vp_init.eta     = E.matrix() * vp_init.muw.matrix();
			vp_init.eta_sq  = vp_init.eta.array().square();

			// Gen initial predicted effects
			calcPredEffects(vp_init);

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
		std::cout << "Setting hyps grid" << std::endl;
		probs_grid          = dat.imprt_grid;
		hyps_grid           = dat.hyps_grid;

		if(p.r1_hyps_grid_file == "NULL"){
			r1_hyps_grid    = hyps_grid;
			r1_probs_grid   = probs_grid;
		} else {
			r1_hyps_grid    = dat.r1_hyps_grid;
			r1_probs_grid   = dat.r1_probs_grid;
		}

		Cty = C.transpose() * Y;

		// sgd
		if(p.mode_sgd){
			minibatch_adjust = N / p.sgd_minibatch_size;
			std::cout << "Using SVI with:" << std::endl;
			std::cout << "delay: " << p.sgd_delay << std::endl;
			std::cout << "minibatch size:" << p.sgd_minibatch_size << std::endl;
			std::cout << "decay:" << p.sgd_forgetting_rate << std::endl;
		} else {
			minibatch_adjust = 1.0;
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
	}

	void run(){
		std::cout << "Starting variational inference" << std::endl;
		time_check = std::chrono::system_clock::now();
		int n_thread = 1; // Parrallel starts swapped for multithreaded inference

		// Round 1; looking for best start point
		if(run_round1){

			std::vector< VbTracker > trackers(n_thread);
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

			// gen initial predicted effects
			calcPredEffects(vp_init);

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
					outf_inits << " " << vp_init.mu1(kk, ee);
				}
				outf_inits << std::endl;
			}

			print_time_check();
		}

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
				outf_inits << " " << vp_init.mu1(kk, ee);
			}
			outf_inits << std::endl;
		}

		std::vector< VbTracker > trackers(n_grid);
		run_inference(hyps_grid, false, 2, trackers);

		write_trackers_to_file("", trackers, hyps_grid, probs_grid);

		std::cout << "Variational inference finished" << std::endl;
	}

	void run_inference(const Eigen::Ref<const Eigen::MatrixXd>& hyps_grid,
                     const bool random_init,
                     const int round_index,
                     std::vector<VbTracker>& trackers){
		// Writes results from inference to trackers

		int n_grid = hyps_grid.rows();
		int n_thread = 1; // Parrallel starts swapped for multithreaded inference

		// Divide grid of hyperparameters into chunks for multithreading
		std::vector< std::vector< int > > chunks(n_thread);
		for (int ii = 0; ii < n_grid; ii++){
			int ch_index = (ii % n_thread);
			chunks[ch_index].push_back(ii);
		}

		// Allocate memory for trackers
		for (int nn = 0; nn < n_grid; nn++){
			trackers[nn].resize(1);
			trackers[nn].set_main_filepath(p.out_file);
			trackers[nn].p = p;
		}

		// Assign set of start points to each thread & run
		// std::thread t2[p.n_thread];
		// for (int ch = 1; ch < p.n_thread; ch++){
		// 	t2[ch] = std::thread( [this, round_index, hyps_grid, n_grid, chunks, ch, random_init, &trackers] {
		// 		runOuterLoop(round_index, hyps_grid, n_grid, chunks[ch], random_init, trackers[ch]);
		// 	} );
		// }
		runOuterLoop(round_index, hyps_grid, n_grid, chunks[0], random_init, trackers);
		// for (int ch = 1; ch < p.n_thread; ch++){
		// 	t2[ch].join();
		// }
	}

	void runOuterLoop(const int round_index,
                      const Eigen::Ref<const Eigen::MatrixXd>& outer_hyps_grid,
                      const int outer_n_grid,
                      std::vector<int> grid_index_list,
                      const bool random_init,
                      std::vector<VbTracker>& all_tracker){

		std::vector<Hyps> all_hyps(n_grid);
		for (auto ii : grid_index_list) {
			// Unpack hyperparams
			// Hyps all_hyps[ii](n_effects,
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
			all_hyps[ii].slab_var.resize(n_effects);
			all_hyps[ii].spike_var.resize(n_effects);
			all_hyps[ii].slab_relative_var.resize(n_effects);
			all_hyps[ii].spike_relative_var.resize(n_effects);
			all_hyps[ii].lambda.resize(n_effects);
			all_hyps[ii].s_x.resize(2);

			Eigen::ArrayXd muw_sq(n_env * n_env);
			for (int ll = 0; ll < n_env; ll++) {
				for (int mm = 0; mm < n_env; mm++) {
					muw_sq(mm * n_env + ll) = vp_init.muw(mm) * vp_init.muw(ll);
				}
			}
			//
			all_hyps[ii].sigma = sigma;
			all_hyps[ii].slab_var << sigma * sigma_b, sigma * sigma_g;
			all_hyps[ii].spike_var << sigma * sigma_b / spike_diff_factor, sigma * sigma_g / spike_diff_factor;
			all_hyps[ii].slab_relative_var << sigma_b, sigma_g;
			all_hyps[ii].spike_relative_var << sigma_b / spike_diff_factor, sigma_g / spike_diff_factor;
			all_hyps[ii].lambda << lam_b, lam_g;
			all_hyps[ii].s_x << n_var, (dXtEEX.rowwise() * muw_sq.transpose()).sum() / (N - 1.0);
			// }
		}

		// Run outer loop - don't update trackers
		runInnerLoop(random_init, round_index, all_hyps, all_tracker);
	}

	void runInnerLoop(const bool random_init,
                      const int round_index,
                      std::vector<Hyps>& all_hyps,
                      std::vector<VbTracker>& all_tracker){
		t_InnerLoop.resume();
		// minimise KL Divergence and assign elbo estimate
		// Assumes vp_init already exist
		std::vector<VariationalParameters> all_vp(n_grid);

		// Assign initial values
		for (int nn = 0; nn < n_grid; nn++) {
			if (random_init) {
				initRandomAlphaMu(all_vp[nn]);
			} else {
				all_vp[nn].init_from_lite(vp_init, p);
			}
			updateSSq(all_hyps[nn], all_vp[nn]);
			all_vp[nn].calcEdZtZ(dXtEEX, n_env);
		}

		// Run inner loop until convergence
		int count = 0;
		std::vector<int> converged(n_grid);
		bool all_converged = false;
		std::vector<Eigen::ArrayXd> alpha_prev(n_grid);
		std::vector<double> i_logw(n_grid);
		std::vector<std::vector< double >> logw_updates(n_grid), alpha_diff_updates(n_grid);

		for (int nn = 0; nn < n_grid; nn++){
			i_logw[nn] = calc_logw(all_hyps[nn], all_vp[nn]);
			logw_updates[nn].push_back(i_logw[nn]);
			converged[nn] = 0;

			all_tracker[nn].interim_output_init(nn, round_index, n_effects, n_env, env_names, all_vp[nn]);
		}

		while(!all_converged && count < p.vb_iter_max){
			for (int nn = 0; nn < n_grid; nn++){
				alpha_prev[nn] = all_vp[nn].alpha_beta;
			}
			std::vector<double> logw_prev = i_logw;

			updateAllParams(count, round_index, all_vp, all_hyps, logw_prev, logw_updates);
			std::vector<double> alpha_diff(n_grid);
			for (int nn = 0; nn < n_grid; nn++){
				i_logw[nn]     = calc_logw(all_hyps[nn], all_vp[nn]);
				alpha_diff[nn] = (alpha_prev[nn] - all_vp[nn].alpha_beta).abs().maxCoeff();
				alpha_diff_updates[nn].push_back(alpha_diff[nn]);
			}

			// Interim output
			long int hty_update_counter = 0;
			for (int nn = 0; nn < n_grid; nn++) {
				if (p.use_vb_on_covars) {
					all_tracker[nn].push_interim_covar_values(count, n_covar, all_vp[nn], covar_names);
				}
				if (p.xtra_verbose && count % 5 == 0) {
					all_tracker[nn].push_interim_param_values(count, n_effects, n_var, all_vp[nn],
													  X.chromosome, X.rsid, X.al_0, X.al_1, X.position);
				}
				all_tracker[nn].push_interim_iter_update(count, all_hyps[nn], i_logw[nn], alpha_diff[nn],
												 t_updateAlphaMu.get_lap_seconds(), hty_update_counter, n_effects,
												 n_var, n_env, all_vp[nn]);
			}

			// Diagnose convergence
			for (int nn = 0; nn < n_grid; nn++) {
				double logw_diff = i_logw[nn] - logw_prev[nn];
				if (p.alpha_tol_set_by_user && p.elbo_tol_set_by_user) {
					if (alpha_diff[nn] < p.alpha_tol && logw_diff < p.elbo_tol) {
						converged[nn] = 1;
					}
				} else if (p.alpha_tol_set_by_user) {
					if (alpha_diff[nn] < p.alpha_tol) {
						converged[nn] = 1;
					}
				} else if (p.elbo_tol_set_by_user) {
					if (logw_diff < p.elbo_tol) {
						converged[nn] = 1;
					}
				} else {
					if (alpha_diff[nn] < alpha_tol && logw_diff < logw_tol) {
						converged[nn] = 1;
					}
				}
			}
			if (std::all_of(converged.begin(), converged.end(), [](int i){return i == 1;})){
				all_converged = true;
			}
			count++;
		}

		if(any_of(i_logw.begin(), i_logw.end(), [](double x) {return !std::isfinite(x);})){
			std::cout << "WARNING: non-finite elbo estimate produced" << std::endl;
		}

		// Log all things that we want to track
		t_InnerLoop.stop();
		for (int nn = 0; nn < n_grid; nn++) {
			all_tracker[nn].logw_list[0] = i_logw[nn];
			all_tracker[nn].counts_list[0] = count;
			all_tracker[nn].vp_list[0] = all_vp[nn].convert_to_lite(n_effects, p);
			all_tracker[nn].elapsed_time_list[0] = t_InnerLoop.get_lap_seconds();
			all_tracker[nn].hyps_list[0] = all_hyps[nn];
			if (p.verbose) {
				logw_updates.push_back(i_logw);  // adding converged estimate
				all_tracker[nn].logw_updates_list[0] = logw_updates[nn];
				all_tracker[nn].alpha_diff_list[0] = alpha_diff_updates[nn];
			}
			all_tracker[nn].push_interim_output(0, X.chromosome, X.rsid, X.position, X.al_0, X.al_1, n_var, n_effects);
		}
	}

	/********** VB update functions ************/
	void updateAllParams(const int& count,
			             const int& round_index,
			             std::vector<VariationalParameters>& all_vp,
						 std::vector<Hyps>& all_hyps,
						 std::vector<double>& logw_prev,
						 std::vector<std::vector< double >>& logw_updates){
		std::vector< std::uint32_t > iter;
		std::vector< std::vector< std::uint32_t >> iter_chunks;
		std::vector<double> i_logw(n_grid);
		long int hty_update_counter = 0;

		// Alternate between back and fwd passes
		bool is_fwd_pass = (count % 2 == 0);
		if(is_fwd_pass){
			iter = fwd_pass;
			iter_chunks = fwd_pass_chunks;
		} else {
			iter = back_pass;
			iter_chunks = back_pass_chunks;
		}

		// Update covar main effects
		for (int nn = 0; nn < n_grid; nn++) {
			if (p.use_vb_on_covars) {
				updateCovarEffects(all_vp[nn], all_hyps[nn], hty_update_counter);
				check_monotonic_elbo(all_hyps[nn], all_vp[nn], count, logw_prev[nn], "updateCovarEffects");
			}
		}

		// Update main & interaction effects
		updateAlphaMu(iter_chunks, all_hyps, all_vp, hty_update_counter, is_fwd_pass);

		for (int nn = 0; nn < n_grid; nn++) {
			check_monotonic_elbo(all_hyps[nn], all_vp[nn], count, logw_prev[nn], "updateAlphaMu");

			// Update env-weights
			if (n_env > 1) {
				for (int uu = 0; uu < p.env_update_repeats; uu++) {
					updateEnvWeights(env_fwd_pass, all_hyps[nn], all_vp[nn]);
					updateEnvWeights(env_back_pass, all_hyps[nn], all_vp[nn]);
				}
				check_monotonic_elbo(all_hyps[nn], all_vp[nn], count, logw_prev[nn], "updateEnvWeights");
			}

			// Log updates
			i_logw[nn] = calc_logw(all_hyps[nn], all_vp[nn]);
			double alpha_diff = 0;

			compute_pve(all_hyps[nn]);

			// Maximise hyps
			if (round_index > 1 && p.mode_empirical_bayes) {
				if (count >= p.burnin_maxhyps) wrapMaximiseHyps(all_hyps[nn], all_vp[nn]);

				i_logw[nn] = calc_logw(all_hyps[nn], all_vp[nn]);

				compute_pve(all_hyps[nn]);
			}
			logw_updates[nn].push_back(i_logw[nn]);
		}
	}

	void updateCovarEffects(VariationalParameters& vp,
                            const Hyps& hyps,
                            long int& hty_updates) __attribute__ ((hot)){
		//
		for (int cc = 0; cc < n_covar; cc++){
			double rr_k = vp.muc(cc);

			// Update s_sq
			vp.sc_sq(cc) = hyps.sigma * sigma_c / (sigma_c * (N - 1.0) + 1.0);

			// Update mu
			vp.muc(cc) = vp.sc_sq(cc) * (Cty(cc) - (vp.ym + vp.yx.cwiseProduct(vp.eta)).dot(C.col(cc)) + rr_k * (N - 1.0)) / hyps.sigma;

			// Update predicted effects
			double rr_k_diff     = vp.muc(cc) - rr_k;
			if(!p.mode_approximate_residuals || std::abs(rr_k_diff) > p.min_residuals_diff){
				hty_updates++;
				vp.ym += rr_k_diff * C.col(cc);
			}
		}
	}

	void updateAlphaMu(const std::vector< std::vector< std::uint32_t >>& iter_chunks,
                       const std::vector<Hyps>& all_hyps,
                       std::vector<VariationalParameters>& all_vp,
                       long int& hty_updates,
                       const bool& is_fwd_pass){
		// Divide updates into chunks
		// Partition chunks amongst available threads
		t_updateAlphaMu.resume();

		for (std::uint32_t ch = 0; ch < iter_chunks.size(); ch++){
			std::vector< std::uint32_t > chunk = iter_chunks[ch];
			int ee                 = chunk[0] / n_var;
			int ch_len             = chunk.size();
			std::uint32_t ch_start = *std::min_element(chunk.begin(), chunk.end()) % n_var;

			t_readXk.resume();
			Eigen::MatrixXd D = X.col_block2(ch_start, ch_len);
			t_readXk.stop();

			// variant correlations with residuals
			Eigen::MatrixXd residual(n_samples, n_grid);
			if (ee == 0){
				for(int nn = 0; nn < n_grid; nn++){
					residual.col(nn) = Y - all_vp[nn].ym - all_vp[nn].yx.cwiseProduct(all_vp[nn].eta);
				}
			} else {
				for (int nn = 0; nn < n_grid; nn++){
					residual.col(nn) = (Y - all_vp[nn].ym).cwiseProduct(all_vp[nn].eta) - all_vp[nn].yx.cwiseProduct(all_vp[nn].eta_sq);
				}
			}
			// Want A as updates x runs
			Eigen::MatrixXd AA = residual.transpose() * D;
			AA.transposeInPlace();

			for (int nn = 0; nn < n_grid; nn++) {
				Eigen::Ref<Eigen::VectorXd> A = AA.col(nn);
				// compute variant correlations within chunk
				Eigen::MatrixXd D_corr;
				if (ee == 0) {
					if (D_correlations.count(ch) == 0) {
						D_corr = D.transpose() * D;
						D_correlations[ch] = D_corr;
					} else {
						D_corr = D_correlations[ch];
					}
				} else {
					D_corr = D.transpose() * all_vp[nn].eta_sq.asDiagonal() * D;
				}

				// Cols in D always read in in same order
				// Hence during back pass we reverse A
				if (!is_fwd_pass) {
					Eigen::VectorXd tmp = A.reverse();
					A = tmp;

					Eigen::MatrixXd tmp_mat = D_corr.reverse();
					D_corr = tmp_mat;
				}

				// compute VB updates for alpha, mu, s_sq
				if (ee == 0) {
					_internal_updateAlphaMu_beta(chunk, A, D_corr, D, is_fwd_pass, all_hyps[nn], all_vp[nn]);
				} else {
					_internal_updateAlphaMu_gam(chunk, A, D_corr, D, is_fwd_pass, all_hyps[nn], all_vp[nn]);
				}
			}
		}

		for (int nn = 0; nn < n_grid; nn++){
			// update summary quantity
			calcVarqBeta(all_hyps[nn], all_vp[nn], all_vp[nn].varB, all_vp[nn].varG);
		}


		t_updateAlphaMu.stop();
	}

	void _internal_updateAlphaMu_beta(const std::vector< std::uint32_t >& iter_chunk,
									  const Eigen::Ref<const Eigen::VectorXd>& A,
									  const Eigen::Ref<const Eigen::MatrixXd>& D_corr,
									  const Eigen::Ref<const Eigen::MatrixXd>& D,
									  const bool& is_fwd_pass,
									  const Hyps& hyps,
									  VariationalParameters& vp){

		int ch_len = iter_chunk.size();
		int ee     = iter_chunk[0] / n_var; // Flag; ee = 0 ? beta : gamma

		Eigen::ArrayXd alpha_cnst;
		if(p.mode_mog_prior_beta){
			alpha_cnst  = (hyps.lambda / (1.0 - hyps.lambda) + eps).log();
			alpha_cnst -= (hyps.slab_var.log() - hyps.spike_var.log()) / 2.0;
		} else {
			alpha_cnst = (hyps.lambda / (1.0 - hyps.lambda) + eps).log() - hyps.slab_var.log() / 2.0;
		}

		// Vector of previous values
		Eigen::VectorXd rr_k(ch_len);
		for (int ii = 0; ii < ch_len; ii++){
			std::uint32_t jj = iter_chunk[ii] % n_var;
			rr_k(ii)                       = vp.alpha_beta(jj) * vp.mu1_beta(jj);
			if(p.mode_mog_prior_beta) rr_k(ii) += (1.0 - vp.alpha_beta(jj)) * vp.mu2_beta(jj);
		}

		// adjust updates within chunk
		// Need to be able to go backwards during a back_pass
		Eigen::VectorXd rr_k_diff(ch_len);
		for (int ii = 0; ii < ch_len; ii++){
			int ee           = iter_chunk[ii] / n_var;   // 0 -> main effect
			std::uint32_t jj = (iter_chunk[ii] % n_var); // variant index

			// Update s_sq
			vp.s1_beta_sq(jj)                        = hyps.slab_var(ee);
			vp.s1_beta_sq(jj)                       /= (hyps.slab_relative_var(ee) * (N-1) + 1);
			if(p.mode_mog_prior_beta) vp.s2_beta_sq(jj)  = hyps.spike_var(ee);
			if(p.mode_mog_prior_beta) vp.s2_beta_sq(jj) /= (hyps.spike_relative_var(ee) * (N-1) + 1);

			// Update mu
			double offset = rr_k(ii) * D_corr(ii, ii);
			for (int mm = 0; mm < ii; mm++){
				offset -= rr_k_diff(mm) * D_corr(mm, ii);
			}
			double AA = A(ii) + offset;
			vp.mu1_beta(jj)                       = vp.s1_beta_sq(jj) * AA / hyps.sigma;
			if (p.mode_mog_prior_beta) vp.mu2_beta(jj) = vp.s2_beta_sq(jj) * AA / hyps.sigma;


			// Update alpha
			double ff_k;
			ff_k                        = vp.mu1_beta(jj) * vp.mu1_beta(jj) / vp.s1_beta_sq(jj);
			ff_k                       += std::log(vp.s1_beta_sq(jj));
			if (p.mode_mog_prior_beta) ff_k -= vp.mu2_beta(jj) * vp.mu2_beta(jj) / vp.s2_beta_sq(jj);
			if (p.mode_mog_prior_beta) ff_k -= std::log(vp.s2_beta_sq(jj));
			vp.alpha_beta(jj)           = sigmoid(ff_k / 2.0 + alpha_cnst(ee));

			rr_k_diff(ii)                       = vp.alpha_beta(jj) * vp.mu1_beta(jj) - rr_k(ii);
			if(p.mode_mog_prior_beta) rr_k_diff(ii) += (1.0 - vp.alpha_beta(jj)) * vp.mu2_beta(jj);
		}

		// Because data is still arranged as per fwd pass
		if (!is_fwd_pass){
			Eigen::VectorXd tmp = rr_k_diff.reverse();
			rr_k_diff = tmp;
		}

		// update residuals
		vp.ym += D * rr_k_diff;
	}

	void _internal_updateAlphaMu_gam(const std::vector< std::uint32_t >& iter_chunk,
									 const Eigen::Ref<const Eigen::VectorXd>& A,
									 const Eigen::Ref<const Eigen::MatrixXd>& D_corr,
									 const Eigen::Ref<const Eigen::MatrixXd>& D,
									 const bool& is_fwd_pass,
									 const Hyps& hyps,
									 VariationalParameters& vp){

		int ch_len = iter_chunk.size();
		int ee     = iter_chunk[0] / n_var; // Flag; ee = 0 ? beta : gamma

		Eigen::ArrayXd alpha_cnst;
		if(p.mode_mog_prior_gam){
			alpha_cnst  = (hyps.lambda / (1.0 - hyps.lambda) + eps).log();
			alpha_cnst -= (hyps.slab_var.log() - hyps.spike_var.log()) / 2.0;
		} else {
			alpha_cnst = (hyps.lambda / (1.0 - hyps.lambda) + eps).log() - hyps.slab_var.log() / 2.0;
		}

		// Vector of previous values
		Eigen::VectorXd rr_k(ch_len);
		for (int ii = 0; ii < ch_len; ii++){
			std::uint32_t jj = iter_chunk[ii] % n_var;
			rr_k(ii)                       = vp.alpha_gam(jj) * vp.mu1_gam(jj);
			if(p.mode_mog_prior_gam) rr_k(ii) += (1.0 - vp.alpha_gam(jj)) * vp.mu2_gam(jj);
		}

		// adjust updates within chunk
		// Need to be able to go backwards during a back_pass
		Eigen::VectorXd rr_k_diff(ch_len);
		for (int ii = 0; ii < ch_len; ii++){
			int ee           = iter_chunk[ii] / n_var;   // 0 -> main effect
			std::uint32_t jj = (iter_chunk[ii] % n_var); // variant index

			// Update s_sq
			vp.s1_gam_sq(jj)                        = hyps.slab_var(ee);
			vp.s1_gam_sq(jj)                       /= (hyps.slab_relative_var(ee) * vp.EdZtZ(jj) + 1);
			if(p.mode_mog_prior_gam) vp.s2_gam_sq(jj)  = hyps.spike_var(ee);
			if(p.mode_mog_prior_gam) vp.s2_gam_sq(jj) /= (hyps.spike_relative_var(ee) * vp.EdZtZ(jj) + 1);

			// Update mu
			double offset = rr_k(ii) * D_corr(ii, ii);
			for (int mm = 0; mm < ii; mm++){
				offset -= rr_k_diff(mm) * D_corr(mm, ii);
			}
			double AA = A(ii) + offset;
			vp.mu1_gam(jj)                       = vp.s1_gam_sq(jj) * AA / hyps.sigma;
			if (p.mode_mog_prior_gam) vp.mu2_gam(jj) = vp.s2_gam_sq(jj) * AA / hyps.sigma;


			// Update alpha
			double ff_k;
			ff_k                        = vp.mu1_gam(jj) * vp.mu1_gam(jj) / vp.s1_gam_sq(jj);
			ff_k                       += std::log(vp.s1_gam_sq(jj));
			if (p.mode_mog_prior_gam) ff_k -= vp.mu2_gam(jj) * vp.mu2_gam(jj) / vp.s2_gam_sq(jj);
			if (p.mode_mog_prior_gam) ff_k -= std::log(vp.s2_gam_sq(jj));
			vp.alpha_gam(jj)           = sigmoid(ff_k / 2.0 + alpha_cnst(ee));

			rr_k_diff(ii)                       = vp.alpha_gam(jj) * vp.mu1_gam(jj) - rr_k(ii);
			if(p.mode_mog_prior_gam) rr_k_diff(ii) += (1.0 - vp.alpha_gam(jj)) * vp.mu2_gam(jj);
		}

		// Because data is still arranged as per fwd pass
		if (!is_fwd_pass){
			Eigen::VectorXd tmp = rr_k_diff.reverse();
			rr_k_diff = tmp;
		}

		// update residuals
		vp.yx += D * rr_k_diff;
	}

	void updateSSq(const Hyps& hyps,
                   VariationalParameters& vp){
		// Used only when initialising VB.
		// We compute elbo on starting point hence need variance estimates
		// Also resizes all variance arrays.

		// Would need vp.s_sq to compute properly, which in turn needs vp.sw_sq...
		vp.sw_sq.resize(n_env);
		vp.sw_sq = eps;
		vp.calcEdZtZ(dXtEEX, n_env);

		// Update beta ssq
		int ee = 0;
		vp.s1_beta_sq.resize(n_var);
		vp.s1_beta_sq  = hyps.slab_var(ee);
		vp.s1_beta_sq /= hyps.slab_relative_var(ee) * (N - 1.0) + 1.0;

		if(p.mode_mog_prior_beta) {
			vp.s2_beta_sq.resize(n_var);
			vp.s2_beta_sq = hyps.spike_var(ee);
			vp.s2_beta_sq /= (hyps.spike_relative_var(ee) * (N - 1.0) + 1.0);
		}

		// Update gamma ssq
		ee = 1;
		vp.s1_gam_sq.resize(n_var);
		vp.s1_gam_sq  = hyps.slab_var(ee);
		vp.s1_gam_sq /= (hyps.slab_relative_var(ee) * (N - 1.0) + 1.0);

		if(p.mode_mog_prior_gam) {
			vp.s2_gam_sq.resize(n_var);
			vp.s2_gam_sq = hyps.spike_var(ee);
			vp.s2_gam_sq /= (hyps.spike_relative_var(ee) * (N - 1.0) + 1.0);
		}

		vp.varB.resize(n_var);
		vp.varG.resize(n_var);
		calcVarqBeta(hyps, vp, vp.varB, vp.varG);

		// for covars
		if(p.use_vb_on_covars){
			vp.sc_sq.resize(n_covar);
			for (int cc = 0; cc < n_covar; cc++){
				vp.sc_sq(cc) = hyps.sigma * sigma_c / (sigma_c * (N - 1.0) + 1.0);
			}
		}
	}

	void maximiseHyps(Hyps& hyps,
                    const VariationalParameters& vp){
		t_maximiseHyps.resume();

		// max sigma
		hyps.sigma  = calcExpLinear(hyps, vp);
		if (p.use_vb_on_covars){
			hyps.sigma += (vp.sc_sq + vp.muc.square()).sum() / sigma_c;
			hyps.sigma /= (N + (double) n_covar);
		} else {
			hyps.sigma /= N;
		}

		// beta - max lambda
		int ee = 0;
		hyps.lambda[ee] = vp.alpha_beta.sum();

		// beta - max spike & slab variances
		hyps.slab_var[ee]  = (vp.alpha_beta * (vp.s1_beta_sq + vp.mu1_beta.square())).sum();
		hyps.slab_var[ee] /= hyps.lambda[ee];
		hyps.slab_relative_var[ee] = hyps.slab_var[ee] / hyps.sigma;
		if(p.mode_mog_prior_beta){
			hyps.spike_var[ee]  = ((1.0 - vp.alpha_beta) * (vp.s2_beta_sq + vp.mu2_beta.square())).sum();
			hyps.spike_var[ee] /= ( (double)n_var - hyps.lambda[ee]);
			hyps.spike_relative_var[ee] = hyps.spike_var[ee] / hyps.sigma;
		}

		// beta - finish max lambda
		hyps.lambda[ee] /= n_var;

		// gamma
		if(n_effects > 1){
			ee = 1;
			hyps.lambda[ee] = vp.alpha_gam.sum();

			// max spike & slab variances
			hyps.slab_var[ee]  = (vp.alpha_gam * (vp.s1_gam_sq + vp.mu1_gam.square())).sum();
			hyps.slab_var[ee] /= hyps.lambda[ee];
			hyps.slab_relative_var[ee] = hyps.slab_var[ee] / hyps.sigma;
			if(p.mode_mog_prior_gam){
				hyps.spike_var[ee]  = ((1.0 - vp.alpha_gam) * (vp.s2_gam_sq + vp.mu2_gam.square())).sum();
				hyps.spike_var[ee] /= ( (double)n_var - hyps.lambda[ee]);
				hyps.spike_relative_var[ee] = hyps.spike_var[ee] / hyps.sigma;
			}

			// finish max lambda
			hyps.lambda[ee] /= n_var;
		}

		// hyps.lam_b         = hyps.lambda(0);
		// hyps.lam_g         = hyps.lambda(1);
		// hyps.sigma_b       = hyps.slab_relative_var(0);
		// hyps.sigma_g       = hyps.slab_relative_var(1);
		// hyps.sigma_g_spike = hyps.spike_relative_var(0);
		// hyps.sigma_g_spike = hyps.spike_relative_var(1);

		t_maximiseHyps.stop();
	}

	void updateEnvWeights(const std::vector< int >& iter,
                          Hyps& hyps,
                          VariationalParameters& vp){

		for (int ll : iter){
			// Log previous mean weight
			double r_ll = vp.muw(ll);

			// Update s_sq
			double denom = hyps.sigma;
			denom       += (vp.yx.array() * E.col(ll)).square().sum();
			denom       += (vp.varG * dXtEEX.col(ll*n_env + ll)).sum();
			vp.sw_sq(ll) = hyps.sigma / denom;

			// Remove dependance on current weight
			vp.eta -= (r_ll * E.col(ll)).matrix();

			// Update mu
			Eigen::ArrayXd env_vars = Eigen::ArrayXd::Zero(n_var);
			for (int mm = 0; mm < n_env; mm++){
				if(mm != ll){
					env_vars += vp.muw(mm) * dXtEEX.col(ll*n_env + mm);
				}
			}

			double eff = ((Y - vp.ym).array() * E.col(ll) * vp.yx.array()).sum();
			eff       -= (vp.yx.array() * E.col(ll) * vp.eta.array() * vp.yx.array()).sum();
			eff       -= (vp.varG * env_vars).sum();
			vp.muw(ll) = vp.sw_sq(ll) * eff / hyps.sigma;

			// Update eta
			vp.eta += (vp.muw(ll) * E.col(ll)).matrix();
		}

		// rescale weights such that vector eta has variance 1
		if(p.rescale_eta){
			double sigma = (vp.eta.array() - (vp.eta.array().sum() / N)).matrix().squaredNorm() / (N-1);
			vp.muw /= std::sqrt(sigma);
			vp.eta = E.matrix() * vp.muw.matrix();
			double sigma2 = (vp.eta.array() - (vp.eta.array().sum() / N)).matrix().squaredNorm() / (N-1);
			std::cout << "Eta rescaled; variance " << sigma << " -> variance " << sigma2 <<std::endl;
		}

		// Recompute eta_sq
		vp.eta_sq  = vp.eta.array().square();
		vp.eta_sq += E.square().matrix() * vp.sw_sq.matrix();

		// Recompute expected value of diagonal of ZtZ
		vp.calcEdZtZ(dXtEEX, n_env);

		// Compute s_x; sum of column variances of Z
		Eigen::ArrayXd muw_sq(n_env * n_env);
		for (int ll = 0; ll < n_env; ll++){
			for (int mm = 0; mm < n_env; mm++){
				muw_sq(mm*n_env + ll) = vp.muw(mm) * vp.muw(ll);
			}
		}
		// WARNING: Hard coded limit!
		// WARNING: Updates S_x in hyps
		hyps.s_x(0) = (double) n_var;
		hyps.s_x(1) = (dXtEEX.rowwise() * muw_sq.transpose()).sum() / (N - 1.0);
	}

	double calc_logw(const Hyps& hyps,
                     const VariationalParameters& vp){
		t_elbo.resume();

		// Expectation of linear regression log-likelihood
		double int_linear = -1.0 * calcExpLinear(hyps, vp) / 2.0 / hyps.sigma;
		int_linear -= N * std::log(2.0 * PI * hyps.sigma) / 2.0;

		// gamma
		int ee;
		double col_sum, int_gamma = 0;
		ee = 0;
		col_sum = vp.alpha_beta.sum();
		int_gamma += col_sum * std::log(hyps.lambda(ee) + eps);
		int_gamma -= col_sum * std::log(1.0 - hyps.lambda(ee) + eps);
		int_gamma += (double) n_var *  std::log(1.0 - hyps.lambda(ee) + eps);

		if(n_effects > 1) {
			ee = 1;
			col_sum = vp.alpha_gam.sum();
			int_gamma += col_sum * std::log(hyps.lambda(ee) + eps);
			int_gamma -= col_sum * std::log(1.0 - hyps.lambda(ee) + eps);
			int_gamma += (double) n_var * std::log(1.0 - hyps.lambda(ee) + eps);
		}

		// kl-beta
		double int_klbeta = calcIntKLBeta(hyps, vp);
		if(n_effects > 1) {
			int_klbeta += calcIntKLGamma(hyps, vp);
		}

		// covariates
		double kl_covar = 0.0;
		if(p.use_vb_on_covars){
			kl_covar += (double) n_covar * (1.0 - hyps.sigma * sigma_c) / 2.0;
			kl_covar += vp.sc_sq.log().sum() / 2.0;
			kl_covar -= vp.sc_sq.sum() / 2.0 / hyps.sigma / sigma_c;
			kl_covar -= vp.muc.square().sum() / 2.0 / hyps.sigma / sigma_c;
		}

		// weights
		double kl_weights = 0.0;
		if(n_env > 1) {
			kl_weights += (double) n_env / 2.0;
			kl_weights += vp.sw_sq.log().sum() / 2.0;
			kl_weights -= vp.sw_sq.sum() / 2.0;
			kl_weights -= vp.muw.square().sum() / 2.0;
		}

		double res = int_linear + int_gamma + int_klbeta + kl_covar + kl_weights;

		t_elbo.stop();
		return res;
	}

	void calc_snpwise_regression(VariationalParametersLite& vp){
		/*
		Genome-wide gxe scan now computed upstream
		Eigen::ArrayXXd snpstats contains results

		cols:
		neglogp-main, neglogp-gxe, coeff-gxe-main, coeff-gxe-env..
		*/
		t_snpwise_regression.resume();

		// Keep values from point with highest p-val
		double vv = 0.0;
		for (long int jj = 0; jj < n_var; jj++){
			if (snpstats(jj, 1) > vv){
				vv = snpstats(jj, 1);
				vp.muw = snpstats.block(jj, 3, 1, n_env);

				std::cout << "neglogp at variant " << jj << ": " << vv;
				std::cout << std::endl << vp.muw.transpose() << std::endl;
			}
		}
		t_snpwise_regression.stop();
	}

	/********** Helper functions ************/
	void print_time_check(){
		auto now = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = now-time_check;
		std::cout << " (" << elapsed_seconds.count();
		std::cout << " seconds since last timecheck, estimated RAM usage = ";
		std::cout << getValueRAM() << "KB)" << std::endl;
		time_check = now;
	}

	void initRandomAlphaMu(VariationalParameters& vp){
		// vp.alpha a uniform simplex, vp.mu standard gaussian
		// Also sets predicted effects
		std::default_random_engine gen_gauss, gen_unif;
		std::normal_distribution<double> gaussian(0.0,1.0);
		std::uniform_real_distribution<double> uniform(0.0,1.0);

		// Beta
		vp.mu1_beta.resize(n_var);
		vp.alpha_beta.resize(n_var);
		if(p.mode_mog_prior_beta){
			vp.mu2_beta = Eigen::ArrayXd::Zero(n_var);
		}

		for (std::uint32_t kk = 0; kk < n_var; 	kk++){
			vp.alpha_beta(kk) = uniform(gen_unif);
			vp.mu1_beta(kk)    = gaussian(gen_gauss);
		}
		vp.alpha_beta /= vp.alpha_beta.sum();

		// Gamma
		if(n_effects > 1){
			vp.mu1_gam.resize(n_var);
			vp.alpha_gam.resize(n_var);
			if (p.mode_mog_prior_gam) {
				vp.mu2_gam = Eigen::ArrayXd::Zero(n_var);
			}

			for (std::uint32_t kk = 0; kk < n_var; kk++) {
				vp.alpha_gam(kk) = uniform(gen_unif);
				vp.mu1_gam(kk) = gaussian(gen_gauss);
			}
			vp.alpha_gam /= vp.alpha_gam.sum();
		}

		if(p.use_vb_on_covars){
			vp.muc = Eigen::ArrayXd::Zero(n_covar);
		}

		// Gen predicted effects.
		calcPredEffects(vp);

		// Env weights
		vp.muw     = 1.0 / (double) n_env;
		vp.eta     = E.matrix() * vp.muw.matrix();
		vp.eta_sq  = vp.eta.array().square();
		vp.calcEdZtZ(dXtEEX, n_env);
	}

	void calcPredEffects(VariationalParameters& vp){
		Eigen::VectorXd rr_beta, rr_gam;
		if(p.mode_mog_prior_beta){
			rr_beta = vp.alpha_beta * (vp.mu1_beta - vp.mu2_beta) + vp.mu2_beta;
		} else {
			rr_beta = vp.alpha_beta * vp.mu1_beta;
		}
		if(p.mode_mog_prior_gam){
			rr_gam = vp.alpha_gam * (vp.mu1_gam - vp.mu2_gam) + vp.mu2_gam;
		} else {
			rr_gam = vp.alpha_gam * vp.mu1_gam;
		}

		vp.ym = X * rr_beta;
		if(p.use_vb_on_covars){
			vp.ym += C * vp.muc.matrix();
		}

		vp.yx = X * rr_gam;
	}

	void calcPredEffects(VariationalParametersLite& vp){
		Eigen::MatrixXd rr;
		if(p.mode_mog_prior_beta && p.mode_mog_prior_gam){
			rr = vp.alpha * (vp.mu1 - vp.mu2) + vp.mu2;
		} else {
			rr = vp.alpha * vp.mu1;
		}
		assert(rr.cols() == 2);

		vp.ym = X * rr.col(0);
		if(p.use_vb_on_covars){
			vp.ym += C * vp.muc.matrix();
		}

		vp.yx = X * rr.col(1);
	}

	void check_monotonic_elbo(const Hyps& hyps,
                   VariationalParameters& vp,
                   const int count,
                   double& logw_prev,
                   const std::string& prev_function){
		double i_logw     = calc_logw(hyps, vp);
		if(i_logw < logw_prev){
			std::cout << count << ": " << prev_function;
 			std::cout << " " << logw_prev << " -> " << i_logw;
 			std::cout << " (difference of " << i_logw - logw_prev << ")"<< std::endl;
		}
		logw_prev = i_logw;
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

	void calcVarqBeta(const Hyps& hyps,
                      const VariationalParameters& vp,
                      Eigen::Ref<Eigen::ArrayXd> varB,
					  Eigen::Ref<Eigen::ArrayXd> varG){
		// Variance of effect size beta under approximating distribution q(u, beta)
		assert(varB.rows() == n_var);
		assert(varG.rows() == n_var);

		varB = vp.alpha_beta * (vp.s1_beta_sq + (1.0 - vp.alpha_beta) * vp.mu1_beta.square());
		if(p.mode_mog_prior_beta){
			varB += (1.0 - vp.alpha_beta) * (vp.s2_beta_sq + (vp.alpha_beta) * vp.mu2_beta.square());
			varB -= 2.0 * vp.alpha_beta * (1.0 - vp.alpha_beta) * vp.mu1_beta * vp.mu2_beta;
		}

		varG = vp.alpha_gam * (vp.s1_gam_sq + (1.0 - vp.alpha_gam) * vp.mu1_gam.square());
		if(p.mode_mog_prior_gam){
			varG += (1.0 - vp.alpha_gam) * (vp.s2_gam_sq + (vp.alpha_gam) * vp.mu2_gam.square());
			varG -= 2.0 * vp.alpha_gam * (1.0 - vp.alpha_gam) * vp.mu1_gam * vp.mu2_gam;
		}
	}

	double calcExpLinear(const Hyps& hyps,
                         const VariationalParameters& vp){
		// Expectation of ||Y - C tau - X beta - Z gamma||^2
		double int_linear = 0;

		// Expectation of linear regression log-likelihood
		int_linear  = (Y - vp.ym).squaredNorm();
		int_linear -= 2.0 * (Y - vp.ym).cwiseProduct(vp.eta).dot(vp.yx);
		if(n_env > 1) {
			int_linear += vp.yx.cwiseProduct(vp.eta_sq).dot(vp.yx);
		} else {
			int_linear += vp.yx.cwiseProduct(vp.eta).squaredNorm();
		}

		// variances
		if(p.use_vb_on_covars){
			int_linear += (N - 1.0) * vp.sc_sq.sum(); // covar main
		}
		int_linear += (N - 1.0) * vp.varB.sum();  // beta
		int_linear += (vp.EdZtZ * vp.varG).sum(); // gamma

		return int_linear;
	}

	double calcIntKLBeta(const Hyps& hyps,
						 const VariationalParameters& vp){
		// KL Divergence of log[ p(beta | u, theta) / q(u, beta) ]
		double col_sum, res;
		int ee = 0;

		// beta
		if(p.mode_mog_prior_beta){
			res  = n_var / 2.0;

			res -= (vp.alpha_beta * (vp.mu1_beta.square() + vp.s1_beta_sq)).sum() / 2.0 / hyps.slab_var(ee);
			res += (vp.alpha_beta * vp.s1_beta_sq.log()).sum() / 2.0;

			res -= ((1.0 - vp.alpha_beta) * (vp.mu2_beta.square() + vp.s2_beta_sq)).sum() / 2.0 / hyps.spike_var(ee);
			res += ((1.0 - vp.alpha_beta) * vp.s2_beta_sq.log()).sum() / 2.0;

			col_sum = vp.alpha_beta.sum();
			res -= std::log(hyps.slab_var(ee))  * col_sum / 2.0;
			res -= std::log(hyps.spike_var(ee)) * (n_var - col_sum) / 2.0;
		} else {
			res  = (vp.alpha_beta * vp.s1_beta_sq.log()).sum() / 2.0;
			res -= (vp.alpha_beta * (vp.mu1_beta.square() + vp.s1_beta_sq)).sum() / 2.0 / hyps.slab_var(ee);

			col_sum = vp.alpha_beta.sum();
			res += col_sum * (1 - std::log(hyps.slab_var(ee))) / 2.0;
		}

		for (std::uint32_t kk = 0; kk < n_var; kk++){
			res -= vp.alpha_beta(kk) * std::log(vp.alpha_beta(kk) + eps);
			res -= (1 - vp.alpha_beta(kk)) * std::log(1 - vp.alpha_beta(kk) + eps);
		}
		return res;
	}

	double calcIntKLGamma(const Hyps& hyps,
						  const VariationalParameters& vp){
		// KL Divergence of log[ p(beta | u, theta) / q(u, beta) ]
		double col_sum, res;
		int ee = 1;

		// beta
		if(p.mode_mog_prior_gam){
			res  = n_var / 2.0;

			res -= (vp.alpha_gam * (vp.mu1_gam.square() + vp.s1_gam_sq)).sum() / 2.0 / hyps.slab_var(ee);
			res += (vp.alpha_gam * vp.s1_gam_sq.log()).sum() / 2.0;

			res -= ((1.0 - vp.alpha_gam) * (vp.mu2_gam.square() + vp.s2_gam_sq)).sum() / 2.0 / hyps.spike_var(ee);
			res += ((1.0 - vp.alpha_gam) * vp.s2_gam_sq.log()).sum() / 2.0;

			col_sum = vp.alpha_gam.sum();
			res -= std::log(hyps.slab_var(ee))  * col_sum / 2.0;
			res -= std::log(hyps.spike_var(ee)) * (n_var - col_sum) / 2.0;
		} else {
			res  = (vp.alpha_gam * vp.s1_gam_sq.log()).sum() / 2.0;
			res -= (vp.alpha_gam * (vp.mu1_gam.square() + vp.s1_gam_sq)).sum() / 2.0 / hyps.slab_var(ee);

			col_sum = vp.alpha_gam.sum();
			res += col_sum * (1 - std::log(hyps.slab_var(ee))) / 2.0;
		}

		for (std::uint32_t kk = 0; kk < n_var; kk++){
			res -= vp.alpha_gam(kk) * std::log(vp.alpha_gam(kk) + eps);
			res -= (1 - vp.alpha_gam(kk)) * std::log(1 - vp.alpha_gam(kk) + eps);
		}
		return res;
	}

	void compute_pve(Hyps& hyps){
		// Compute heritability
		hyps.pve.resize(n_effects);
		hyps.pve_large.resize(n_effects);

		hyps.pve = hyps.lambda * hyps.slab_relative_var * hyps.s_x;
		if(p.mode_mog_prior_beta){
			int ee = 0;
			hyps.pve_large[ee] = hyps.pve[ee];
			hyps.pve[ee] += (1 - hyps.lambda[ee]) * hyps.spike_relative_var[ee] * hyps.s_x[ee];

			if (p.mode_mog_prior_gam && n_effects > 1){
				int ee = 1;
				hyps.pve_large[ee] = hyps.pve[ee];
				hyps.pve[ee] += (1 - hyps.lambda[ee]) * hyps.spike_relative_var[ee] * hyps.s_x[ee];
			}

			hyps.pve_large[ee] /= (hyps.pve.sum() + 1.0);
		}
		hyps.pve /= (hyps.pve.sum() + 1.0);
	}

	/********** Output functions ************/
	void write_trackers_to_file(const std::string& file_prefix,
                                const std::vector< VbTracker >& trackers,
                                const Eigen::Ref<const Eigen::MatrixXd>& hyps_grid,
                                const Eigen::Ref<const Eigen::VectorXd>& my_probs_grid){
		// Stitch trackers back together if using multithreading
		int n_thread = 1; // Parrallel starts swapped for multithreaded inference
		int my_n_grid = hyps_grid.rows();
		assert(my_n_grid == my_probs_grid.rows());
		VbTracker stitched_tracker;
		stitched_tracker.resize(my_n_grid);
		for (int ii = 0; ii < my_n_grid; ii++){
			stitched_tracker.copy_ith_element(ii, 0, trackers[ii]);
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
		std::string ofile_w = fstream_init(outf_w, file_prefix, "_env_weights");
		std::cout << "Writing converged hyperparameter values to " << ofile << std::endl;
		std::cout << "Writing MAP snp stats to " << ofile_map << std::endl;
		std::cout << "Writing (weighted) average snp stats to " << ofile_wmean << std::endl;
		std::cout << "Writing (niave) average snp stats to " << ofile_nmean << std::endl;
		std::cout << "Writing yhat from map to " << ofile_map_yhat << std::endl;
		std::cout << "Writing env weights to " << ofile_w << std::endl;

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

		// Write hyperparams weights to file
		outf << "weight logw";
		if(!p.mode_empirical_bayes){
			outf << " log_prior";
		}
		outf << " count time sigma";
		for (int ee = 0; ee < n_effects; ee++){
			outf << " pve" << ee;
			if((ee == 0 && p.mode_mog_prior_beta) || (ee == 1 && p.mode_mog_prior_gam)){
				outf << " pve_large" << ee;
 			}
			outf << " sigma" << ee;
			if((ee == 0 && p.mode_mog_prior_beta) || (ee == 1 && p.mode_mog_prior_gam)){
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
				outf << " " << stitched_tracker.hyps_list[ii].pve(ee);
				if((ee == 0 && p.mode_mog_prior_beta) || (ee == 1 && p.mode_mog_prior_gam)){
					outf << " " << stitched_tracker.hyps_list[ii].pve_large(ee);
				}
				outf << " " << stitched_tracker.hyps_list[ii].slab_relative_var(ee);
				if((ee == 0 && p.mode_mog_prior_beta) || (ee == 1 && p.mode_mog_prior_gam)){
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
				wmean_beta  += weights[ii] * stitched_tracker.vp_list[ii].alpha * stitched_tracker.vp_list[ii].mu1;
				nmean_alpha += stitched_tracker.vp_list[ii].alpha;
				nmean_beta  += stitched_tracker.vp_list[ii].alpha * stitched_tracker.vp_list[ii].mu1;
			}
		}
		nmean_alpha /= (double) my_n_grid;
		nmean_beta  /= (double) my_n_grid;

		Eigen::ArrayXXd nmean_alpha_sd = Eigen::ArrayXXd::Zero(n_var, n_effects);
		Eigen::ArrayXXd nmean_beta_sd  = Eigen::ArrayXXd::Zero(n_var, n_effects);
		for (int ii = 0; ii < my_n_grid; ii++){
			if(std::isfinite(weights[ii])){
				nmean_alpha_sd += (stitched_tracker.vp_list[ii].alpha - nmean_alpha).square();
				nmean_beta_sd  += (stitched_tracker.vp_list[ii].alpha * stitched_tracker.vp_list[ii].mu1 - nmean_beta).square();
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
				outf_map << " " << stitched_tracker.vp_list[ii_map].alpha(kk, ee) * stitched_tracker.vp_list[ii_map].mu1(kk, ee);
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
		calcPredEffects(vp_map);
		for (std::uint32_t ii = 0; ii < n_samples; ii++ ){
			outf_map_pred << vp_map.ym(ii) + vp_map.eta(ii) * vp_map.yx(ii) << std::endl;
		}

		// weights to file
		for (int ll = 0; ll < n_env; ll++){
			outf_w << env_names[ll];
			if(ll + 1 < n_env) outf_w << " ";
		}
		outf_w << std::endl;
		for (int ll = 0; ll < n_env; ll++){
			outf_w << vp_map.muw(ll);
			if(ll + 1 < n_env) outf_w << " ";
		}
		outf_w << std::endl;

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
#ifndef OSX
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
#else
		return -1;
#endif
	}

	/********** SGD stuff; unfinished ************/
	void wrapMaximiseHyps(Hyps& hyps, const VariationalParameters& vp){
		// if(p.mode_sgd){
		// 	Hyps t_hyps;
		// 	maximiseHyps(t_hyps, vp);
		// 	hyps.sigma = _update_param(hyps.sigma, t_hyps.sigma, vp);
		// 	hyps.lambda = _update_param(hyps.lambda, t_hyps.lambda, vp);
		// 	hyps.spike_var = _update_param(hyps.spike_var, t_hyps.spike_var, vp);
		// 	hyps.slab_var = _update_param(hyps.slab_var, t_hyps.slab_var, vp);
		// 	hyps.spike_relative_var = _update_param(hyps.spike_relative_var, t_hyps.spike_relative_var, vp);
		// 	hyps.slab_relative_var = _update_param(hyps.slab_relative_var, t_hyps.slab_relative_var, vp);
		// } else {
			maximiseHyps(hyps, vp);
		// }
	}

	double _update_param(double p_old, double p_new, const VariationalParameters& vp){
		double res, rho = std::pow(vp.count + p.sgd_delay, -p.sgd_forgetting_rate);
		if(p.mode_sgd){
			res = (1 - rho) * p_old + rho * p_new;
		} else {
			res = p_new;
		}
		return(res);
	}

	Eigen::ArrayXd _update_param(Eigen::ArrayXd p_old, Eigen::ArrayXd p_new, const VariationalParameters& vp){
		Eigen::ArrayXd res;
		double rho = std::pow(vp.count + p.sgd_delay, -p.sgd_forgetting_rate);
		if(p.mode_sgd){
			res = (1 - rho) * p_old + rho * p_new;
		} else {
			res = p_new;
		}
		return(res);
	}
};

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
