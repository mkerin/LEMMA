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
	int n_covar;
	int n_env;
	std::uint32_t n_var;
	std::uint32_t n_var2;
	bool          random_params_init;
	bool          run_round1;
	double N; // (double) n_samples


	//
	parameters& p;
	std::vector< std::uint32_t > fwd_pass;
	std::vector< std::uint32_t > back_pass;
	std::vector< int > env_fwd_pass;
	std::vector< int > env_back_pass;

	// Data
	GenotypeMatrix& X;
	Eigen::VectorXd Y;          // residual phenotype matrix
	Eigen::ArrayXXd& dXtEEX;     // P x n_env^2; col (l * n_env + m) is the diagonal of X^T * diag(E_l * E_m) * X
	Eigen::ArrayXd  Cty;         // vector of W^T x y where C the matrix of covariates
	Eigen::ArrayXXd E;          // matrix of variables used for GxE interactions
	Eigen::MatrixXd& C;          // matrix of covariates (superset of GxE variables)
	Eigen::MatrixXd r1_hyps_grid;
	Eigen::MatrixXd hyps_grid;

	// genome wide scan computed upstream
	Eigen::ArrayXXd& snpstats;

	// Init points
	VariationalParametersLite vp_init;

	// boost fstreams
	boost_io::filtering_ostream outf, outf_map, outf_wmean, outf_nmean, outf_inits;
	boost_io::filtering_ostream outf_elbo, outf_alpha_diff, outf_map_pred, outf_w;
	boost_io::filtering_ostream outf_rescan;

	// Monitoring
	std::chrono::system_clock::time_point time_check;

	MyTimer t_updateAlphaMu;
	MyTimer t_elbo;
	MyTimer t_maximiseHyps;
	MyTimer t_InnerLoop;
	MyTimer t_snpwise_regression;

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
                            t_snpwise_regression("calc_snpwise_regression: %ts \n") {
		assert(std::includes(dat.hyps_names.begin(), dat.hyps_names.end(), hyps_names.begin(), hyps_names.end()));
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

		// Read environmental variables
		E = dat.E;

		// Allocate memory - fwd/back pass vectors
		for(std::uint32_t kk = 0; kk < n_var * n_effects; kk++){
			fwd_pass.push_back(kk);
			back_pass.push_back(n_var2 - kk - 1);
		}
		for(int ll = 0; ll < n_env; ll++){
			env_fwd_pass.push_back(ll);
			env_back_pass.push_back(n_env - ll - 1);
		}

		// non random initialisation
		if(p.vb_init_file != "NULL"){
			vp_init.alpha         = dat.alpha_init;
			vp_init.mu            = dat.mu_init;
			vp_init.s_sq          = Eigen::ArrayXXd::Zero(n_var, n_effects);
			if(p.mode_mog_prior){
				vp_init.mup   = Eigen::ArrayXXd::Zero(n_var, n_effects);
				vp_init.sp_sq = Eigen::ArrayXXd::Zero(n_var, n_effects);
			}
			if(p.use_vb_on_covars){
				vp_init.muc   = Eigen::ArrayXd::Zero(n_covar);
			}

			if(p.env_weights_file != "NULL"){
				vp_init.muw     = dat.E_weights.col(0);
			} else if (n_env > 1 && p.init_weights_with_snpwise_scan){
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
		hyps_grid           = dat.hyps_grid;

		if(p.r1_hyps_grid_file == "NULL"){
			r1_hyps_grid    = hyps_grid;
		} else {
			r1_hyps_grid    = dat.r1_hyps_grid;
		}

		if(p.use_vb_on_covars){
			Cty = C.transpose() * Y;
		}

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
		io::close(outf_rescan);
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
				write_trackers_to_file("round1_", trackers, r1_hyps_grid);
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

			print_time_check();
		}

		// Write inits to file - exclude covar values
		std::string ofile_inits = fstream_init(outf_inits, "", "_inits");
		std::cout << "Writing start points for alpha and mu to " << ofile_inits << std::endl;
		write_snp_stats_to_file(outf_inits, vp_init, false, false);
		io::close(outf_inits);

		std::vector< VbTracker > trackers(p.n_thread);
		run_inference(hyps_grid, false, 2, trackers);

		write_trackers_to_file("", trackers, hyps_grid);

		std::cout << "Variational inference finished" << std::endl;
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

	void runOuterLoop(const int round_index,
                      const Eigen::Ref<const Eigen::MatrixXd>& outer_hyps_grid,
                      const int outer_n_grid,
                      std::vector<int> grid_index_list,
                      const bool random_init,
                      VbTracker& tracker){

		for (auto ii : grid_index_list){
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
			i_hyps.s_x.resize(n_effects);

			Eigen::ArrayXd muw_sq(n_env * n_env);
			for (int ll = 0; ll < n_env; ll++){
				for (int mm = 0; mm < n_env; mm++){
					muw_sq(mm*n_env + ll) = vp_init.muw(mm) * vp_init.muw(ll);
				}
			}

			if (n_effects == 2) {
				i_hyps.sigma = sigma;
				i_hyps.slab_var << sigma * sigma_b, sigma * sigma_g;
				i_hyps.spike_var << sigma * sigma_b / p.spike_diff_factor, sigma * sigma_g / p.spike_diff_factor;
				i_hyps.slab_relative_var << sigma_b, sigma_g;
				i_hyps.spike_relative_var << sigma_b / p.spike_diff_factor, sigma_g / p.spike_diff_factor;
				i_hyps.lambda << lam_b, lam_g;
				i_hyps.s_x << n_var, (dXtEEX.rowwise() * muw_sq.transpose()).sum() / (N - 1.0);
			} else if (n_effects == 1){
				i_hyps.sigma = sigma;
				i_hyps.slab_var << sigma * sigma_b;
				i_hyps.spike_var << sigma * sigma_b / p.spike_diff_factor;
				i_hyps.slab_relative_var << sigma_b;
				i_hyps.spike_relative_var << sigma_b / p.spike_diff_factor;
				i_hyps.lambda << lam_b;
				i_hyps.s_x << n_var;
			}

			// Run outer loop - don't update trackers
			runInnerLoop(ii, random_init, round_index, i_hyps, tracker);

			// 'rescan' GWAS of Z on y-ym
			if(n_effects > 1) {
				Eigen::VectorXd gam_neglogp(n_var);
				rescanGWAS(tracker.vp_list[ii], gam_neglogp);
				tracker.push_rescan_gwas(X, n_var, gam_neglogp);
			}

			if((ii + 1) % print_interval == 0){
				std::cout << "\rRound " << round_index << ": grid point " << ii+1 << "/" << outer_n_grid;
				print_time_check();
			}
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
		updateSSq(hyps, vp);
		vp.calcEdZtZ(dXtEEX, n_env);

		// Initial s_x


		// Run inner loop until convergence
		int count = 0;
		bool converged = false;
		Eigen::ArrayXXd alpha_prev;
		std::vector< std::uint32_t > iter;
		double i_logw = calc_logw(hyps, vp);
		std::vector< double > logw_updates, alpha_diff_updates;
		logw_updates.push_back(i_logw);

		tracker.interim_output_init(ii, round_index, n_effects, n_env, env_names, vp);
		while(!converged && count < p.vb_iter_max){
			alpha_prev = vp.alpha;
			double logw_prev = i_logw;

			updateAllParams(count, round_index, vp, hyps, logw_prev, logw_updates);
			i_logw     = calc_logw(hyps, vp);
			double alpha_diff = (alpha_prev - vp.alpha).abs().maxCoeff();
			alpha_diff_updates.push_back(alpha_diff);

			// Interim output
			if(p.use_vb_on_covars){
				tracker.push_interim_covar_values(count, n_covar, vp,covar_names);
			}
			if(p.xtra_verbose && count % 5 == 0){
				tracker.push_interim_param_values(count, n_effects, n_var, vp,
												  X.chromosome, X.rsid, X.al_0, X.al_1, X.position);
			}
			tracker.push_interim_iter_update(count, hyps, i_logw, alpha_diff,
											 t_updateAlphaMu.get_lap_seconds(), n_effects, n_var, n_env, vp);

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
			} else {
				if(alpha_diff < alpha_tol && logw_diff < logw_tol){
					converged = true;
				}
			}
			count++;
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

	/********** VB update functions ************/
	void rescanGWAS(const VariationalParametersLite& vp,
			Eigen::Ref<Eigen::VectorXd> neglogp){
		Eigen::VectorXd pheno = Y - vp.ym;
		Eigen::VectorXd Z_kk(n_samples);

		for(std::uint32_t jj = 0; jj < n_var; jj++ ){
			Z_kk = X.col(jj).cwiseProduct(vp.eta);
			double ztz_inv = 1.0 / Z_kk.dot(Z_kk);
			double gam = Z_kk.dot(pheno) * ztz_inv;
			double rss_null = (pheno - Z_kk * gam).squaredNorm();

			// T-test of variant j
			boost_m::students_t t_dist(n_samples - 1);
			double main_se_j    = std::sqrt(rss_null / (N - 1.0) * ztz_inv);
			double main_tstat_j = gam / main_se_j;
			double main_pval_j  = 2 * boost_m::cdf(boost_m::complement(t_dist, fabs(main_tstat_j)));

			neglogp(jj) = -1 * std::log10(main_pval_j);
		}
	}

	void updateAllParams(const int& count,
			             const int& round_index,
			             VariationalParameters& vp,
						 Hyps& hyps,
						 double& logw_prev,
						 std::vector< double >& logw_updates){
		std::vector< std::uint32_t > iter;
		double i_logw;

		// Alternate between back and fwd passes
		if(count % 2 == 0){
			iter = fwd_pass;
		} else {
			iter = back_pass;
		}

		// Update covar main effects
		if(p.use_vb_on_covars){
			updateCovarEffects(vp, hyps);
			check_monotonic_elbo(hyps, vp, count, logw_prev, "updateCovarEffects");
		}

		// Update main & interaction effects
		updateAlphaMu(iter, hyps, vp);
		check_monotonic_elbo(hyps, vp, count, logw_prev, "updateAlphaMu");

		// Update env-weights
		if( n_env > 1) {
			for (int uu = 0; uu < p.env_update_repeats; uu++) {
				updateEnvWeights(env_fwd_pass, hyps, vp);
				updateEnvWeights(env_back_pass, hyps, vp);
			}
			check_monotonic_elbo(hyps, vp, count, logw_prev, "updateEnvWeights");
		}

		// Log updates
		i_logw     = calc_logw(hyps, vp);
		double alpha_diff = 0;

		compute_pve(vp, hyps);

		// Maximise hyps
		if(round_index > 1 && p.mode_empirical_bayes){
			if (count >= p.burnin_maxhyps) wrapMaximiseHyps(hyps, vp);

			i_logw     = calc_logw(hyps, vp);

			compute_pve(vp, hyps);
		}
		logw_updates.push_back(i_logw);
	}

	void updateCovarEffects(VariationalParameters& vp,
                            const Hyps& hyps) __attribute__ ((hot)){
		//
		for (int cc = 0; cc < n_covar; cc++){
			double rr_k = vp.muc(cc);

			// Update s_sq
			vp.sc_sq(cc) = hyps.sigma * sigma_c / (sigma_c * (N - 1.0) + 1.0);

			// Update mu
			vp.muc(cc) = vp.sc_sq(cc) * (Cty(cc) - (vp.ym + vp.yx.cwiseProduct(vp.eta)).dot(C.col(cc)) + rr_k * (N - 1.0)) / hyps.sigma;

			// Update predicted effects
			double rr_k_diff     = vp.muc(cc) - rr_k;
			vp.ym += rr_k_diff * C.col(cc);
		}
	}

	void updateAlphaMu(const std::vector< std::uint32_t >& iter,
                       const Hyps& hyps,
                       VariationalParameters& vp){
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

			// Skip interaction updates for variants w/o main effect
			if(p.restrict_gamma_updates && ee == 1 && vp.alpha(jj, 0) < p.gamma_updates_thresh){
				// remove previous residual if necessary
				if (vp.alpha(jj, 1) > 1e-6){
					X_kk = X.col(jj);
					vp.yx -= vp.alpha(jj, 1) * vp.mu(jj, 1) * X_kk;
					vp.alpha(jj, 1) = 0.0;
					vp.mu(jj, 1) = 0.0;
				}
				continue;
			}

			X_kk = X.col(jj); // Only read normalised genotypes!

			_internal_updateAlphaMu(X_kk, ee, jj, vp, hyps, alpha_cnst);

			if(p.mode_alternating_updates){
				for(int eee = 1; eee < n_effects; eee++){
					_internal_updateAlphaMu(X_kk, eee, jj, vp, hyps, alpha_cnst);
				}
			}
		}

		// update summary quantity
		calcVarqBeta(hyps, vp, vp.varB);

		t_updateAlphaMu.stop();
	}

	void _internal_updateAlphaMu(const Eigen::Ref<const Eigen::VectorXd>& X_kk,
								 const int& ee, std::uint32_t jj,
								 VariationalParameters& vp,
								 const Hyps& hyps,
								 const Eigen::Ref<const Eigen::ArrayXd>& alpha_cnst) __attribute__ ((hot)){
		//
		double rr_k_diff;

		double rr_k                = vp.alpha(jj, ee) * vp.mu(jj, ee);
		if(p.mode_mog_prior) rr_k += (1.0 - vp.alpha(jj, ee)) * vp.mup(jj, ee);

		// Update s_sq
		// Strictly speaking; only need to do every iter if maximising hyps
		// or updating env-weights.
		if (ee == 0){
			vp.s_sq(jj, ee)                        = hyps.slab_var(ee);
			vp.s_sq(jj, ee)                       /= (hyps.slab_relative_var(ee) * (N-1) + 1);
			if(p.mode_mog_prior) vp.sp_sq(jj, ee)  = hyps.spike_var(ee);
			if(p.mode_mog_prior) vp.sp_sq(jj, ee) /= (hyps.spike_relative_var(ee) * (N-1) + 1);
		} else {
			vp.s_sq(jj, ee)                        = hyps.slab_var(ee);
			vp.s_sq(jj, ee)                       /= (hyps.slab_relative_var(ee) * vp.EdZtZ(jj) + 1);
			if(p.mode_mog_prior) vp.sp_sq(jj, ee)  = hyps.spike_var(ee);
			if(p.mode_mog_prior) vp.sp_sq(jj, ee) /= (hyps.spike_relative_var(ee) * vp.EdZtZ(jj) + 1);
		}

		// Update mu
		double A;
		if(n_effects == 1){
			// Main effects update in main effects only model
			A  = (Y - vp.ym).dot(X_kk) + rr_k * (N - 1.0);
		} else if(ee == 0){
			// Main effects update in interaction model
			A  = (Y - vp.ym - vp.yx.cwiseProduct(vp.eta)).dot(X_kk) + rr_k * (N - 1.0);
		} else {
			// Interaction effects update in interaction model
			A  = (Y - vp.ym).cwiseProduct(vp.eta).dot(X_kk);
			A -= (vp.yx.cwiseProduct(vp.eta_sq).dot(X_kk) - rr_k * vp.EdZtZ(jj));
		}

		vp.mu(jj, ee)                       = vp.s_sq(jj, ee)  * A / hyps.sigma;
		if(p.mode_mog_prior) vp.mup(jj, ee) = vp.sp_sq(jj, ee) * A / hyps.sigma;

		// Update alpha
		double ff_k;
		ff_k                       = vp.mu(jj, ee) * vp.mu(jj, ee) / vp.s_sq(jj, ee);
		ff_k                      += std::log(vp.s_sq(jj, ee));
		if(p.mode_mog_prior) ff_k -= vp.mup(jj, ee) * vp.mup(jj, ee) / vp.sp_sq(jj, ee);
		if(p.mode_mog_prior) ff_k -= std::log(vp.sp_sq(jj, ee));

		vp.alpha(jj, ee)           = sigmoid(ff_k / 2.0 + alpha_cnst(ee));

		// Update residuals only if coeff is large enough to matter
		rr_k_diff                       = vp.alpha(jj, ee) * vp.mu(jj, ee) - rr_k;
		if(p.mode_mog_prior) rr_k_diff += (1.0 - vp.alpha(jj, ee)) * vp.mup(jj, ee);

		if(ee == 0){
			vp.ym += rr_k_diff * X_kk;
		} else {
			vp.yx += rr_k_diff * X_kk;
		}
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

		// Update main
		vp.s_sq.resize(n_var, n_effects);
		vp.s_sq.col(0)  = hyps.slab_var(0);
		vp.s_sq.col(0) /= (hyps.slab_relative_var(0) * (N - 1.0) + 1.0);

		for (int ee = 1; ee < n_effects; ee++){
			for (std::uint32_t kk = 0; kk < n_var; kk++){
				vp.s_sq(kk, ee)  = hyps.slab_var(ee);
				vp.s_sq(kk, ee) /= (hyps.slab_relative_var(ee) * vp.EdZtZ(kk, ee-1) + 1.0);
			}
		}

		if(p.mode_mog_prior){
			vp.sp_sq.resize(n_var, n_effects);
			vp.sp_sq.col(0)  =  hyps.spike_var(0);
			vp.sp_sq.col(0) /= (hyps.spike_relative_var(0) * (N - 1.0) + 1.0);

			for (int ee = 1; ee < n_effects; ee++){
				for (std::uint32_t kk = 0; kk < n_var; kk++){
					vp.sp_sq(kk, ee)  = hyps.spike_var(ee);
					vp.sp_sq(kk, ee) /= (hyps.spike_relative_var(ee) * vp.EdZtZ(kk, ee-1) + 1.0);
				}
			}
		}
		vp.varB.resize(n_var, n_effects);
		calcVarqBeta(hyps, vp, vp.varB);

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

		// max lambda
		hyps.lambda = vp.alpha.colwise().sum();

		// max spike & slab variances
		hyps.slab_var.resize(n_effects);
		hyps.spike_var.resize(n_effects);
		for (int ee = 0; ee < n_effects; ee++){

			// Initial unconstrained max
			hyps.slab_var[ee]  = (vp.alpha.col(ee) * (vp.s_sq.col(ee) + vp.mu.col(ee).square())).sum();
			hyps.slab_var[ee] /= hyps.lambda[ee];
			if(p.mode_mog_prior){
				hyps.spike_var[ee]  = ((1.0 - vp.alpha.col(ee)) * (vp.sp_sq.col(ee) + vp.mup.col(ee).square())).sum();
				hyps.spike_var[ee] /= ( (double)n_var - hyps.lambda[ee]);
			}

			// Remaxise while maintaining same diff in MoG variances if getting too close
			if(p.mode_mog_prior && hyps.slab_var[ee] < p.min_spike_diff_factor * hyps.spike_var[ee]){
				hyps.slab_var[ee]  = (vp.alpha.col(ee) * (vp.s_sq.col(ee) + vp.mu.col(ee).square())).sum();
				hyps.slab_var[ee] += p.min_spike_diff_factor * ((1.0 - vp.alpha.col(ee)) * (vp.sp_sq.col(ee) + vp.mup.col(ee).square())).sum();
				hyps.slab_var[ee] /= (double) n_var;
				hyps.spike_var[ee] = hyps.slab_var[ee] / p.min_spike_diff_factor;
			}
		}

		hyps.slab_relative_var = hyps.slab_var / hyps.sigma;
		if(p.mode_mog_prior){
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
			denom       += (vp.varB.col(1) * dXtEEX.col(ll*n_env + ll)).sum();
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
			eff       -= (vp.varB.col(1) * env_vars).sum();
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
		vp.muw = Eigen::ArrayXd::Zero(n_env);
		double vv = 0.0;
		for (long int jj = 0; jj < n_var; jj++){
			if (snpstats(jj, 1) > vv){
				vv = snpstats(jj, 1);
				vp.muw = snpstats.block(jj, 2, 1, n_env).transpose();

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

		// Gen predicted effects.
		calcPredEffects(vp);

		// Env weights
		vp.muw     = 1.0 / (double) n_env;
		vp.eta     = E.matrix() * vp.muw.matrix();
		vp.eta_sq  = vp.eta.array().square();
		vp.calcEdZtZ(dXtEEX, n_env);
	}

	void calcPredEffects(VariationalParameters& vp){
		Eigen::MatrixXd rr;
		if(p.mode_mog_prior){
			rr = vp.alpha * (vp.mu - vp.mup) + vp.mup;
		} else {
			rr = vp.alpha * vp.mu;
		}

		vp.ym = X * rr.col(0);
		if(p.use_vb_on_covars){
			vp.ym += C * vp.muc.matrix();
		}
		if(n_effects > 1) {
			vp.yx = X * rr.col(1);
		}
	}

	void calcPredEffects(VariationalParametersLite& vp) {
		Eigen::MatrixXd rr;
		if (p.mode_mog_prior) {
			rr = vp.alpha * (vp.mu - vp.mup) + vp.mup;
		} else {
			rr = vp.alpha * vp.mu;
		}

		vp.ym = X * rr.col(0);
		if (p.use_vb_on_covars) {
			vp.ym += C * vp.muc.matrix();
		}
		if (n_effects > 1){
			vp.yx = X * rr.col(1);
		}
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

	double calcExpLinear(const Hyps& hyps,
                         const VariationalParameters& vp){
		// Expectation of ||Y - C tau - X beta - Z gamma||^2
		double int_linear = 0;

		// Expectation of linear regression log-likelihood
		int_linear  = (Y - vp.ym).squaredNorm();
		if(n_effects > 1) {
			int_linear -= 2.0 * (Y - vp.ym).cwiseProduct(vp.eta).dot(vp.yx);
			if (n_env > 1) {
				int_linear += vp.yx.cwiseProduct(vp.eta_sq).dot(vp.yx);
			} else {
				int_linear += vp.yx.cwiseProduct(vp.eta).squaredNorm();
			}
		}

		// variances
		if(p.use_vb_on_covars){
			int_linear += (N - 1.0) * vp.sc_sq.sum(); // covar main
		}
		int_linear += (N - 1.0) * vp.varB.col(0).sum();  // beta
		std::cout << int_linear << std::endl;
		std::cout << vp.varB.col(1).sum() << std::endl;
		std::cout << vp.EdZtZ.sum() << std::endl;
		if(n_effects > 1) {
			int_linear += (vp.EdZtZ * vp.varB.col(1)).sum(); // gamma
		}

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

	void compute_pve(const VariationalParameters& vp, Hyps& hyps){
		// Compute heritability
		hyps.pve.resize(n_effects);
		hyps.pve_large.resize(n_effects);

		hyps.pve = hyps.lambda * hyps.slab_relative_var * hyps.s_x;
		if(p.mode_mog_prior){
			hyps.pve_large = hyps.pve;
			hyps.pve += (1 - hyps.lambda) * hyps.spike_relative_var * hyps.s_x;
			hyps.pve_large /= (hyps.pve.sum() + 1.0);
		}
		hyps.pve /= (hyps.pve.sum() + 1.0);
//
//		// The second version of computing heritability
//		hyps.pve2.resize(n_effects);
//
//		int ee = 0;
//		hyps.pve2[ee] = (vp.ym.array() - vp.ym.mean()).square().sum() / (N-1) + vp.varB.col(ee).sum();
//		if(n_effects > 1) {
//			ee = 1;
//			Eigen::ArrayXd eta_yx = vp.yx.cwiseProduct(vp.eta).array();
//			hyps.pve2[ee] =
//					(eta_yx - eta_yx.mean()).square().sum() / (N - 1) + (vp.EdZtZ * vp.varB.col(ee)).sum() / (N - 1);
//		}
//		hyps.pve2 /= (hyps.pve2.sum() + hyps.sigma);
	}

	/********** Output functions ************/
	void write_trackers_to_file(const std::string& file_prefix,
                                const std::vector< VbTracker >& trackers,
                                const Eigen::Ref<const Eigen::MatrixXd>& hyps_grid){
		// Stitch trackers back together if using multithreading
		int my_n_grid = hyps_grid.rows();
		VbTracker stitched_tracker;
		stitched_tracker.resize(my_n_grid);
		for (int ii = 0; ii < my_n_grid; ii++){
			int tr = (ii % p.n_thread);  // tracker index
			stitched_tracker.copy_ith_element(ii, trackers[tr]);
		}

		output_init(file_prefix);
		output_results(stitched_tracker, my_n_grid);
	}

	void output_init(const std::string& file_prefix){
		// Initialise files ready to write;

		std::string ofile       = fstream_init(outf, file_prefix, "");
		std::string ofile_map   = fstream_init(outf_map, file_prefix, "_map_snp_stats");
		std::string ofile_wmean = fstream_init(outf_wmean, file_prefix, "_weighted_mean_snp_stats");
		std::string ofile_nmean = fstream_init(outf_nmean, file_prefix, "_niave_mean_snp_stats");
		std::string ofile_map_yhat = fstream_init(outf_map_pred, file_prefix, "_map_yhat");
		std::string ofile_w = fstream_init(outf_w, file_prefix, "_env_weights");
		std::string ofile_rescan = fstream_init(outf_rescan, file_prefix, "_map_rescan");
		std::cout << "Writing converged hyperparameter values to " << ofile << std::endl;
		std::cout << "Writing MAP snp stats to " << ofile_map << std::endl;
		std::cout << "Writing (weighted) average snp stats to " << ofile_wmean << std::endl;
		std::cout << "Writing (niave) average snp stats to " << ofile_nmean << std::endl;
		std::cout << "Writing yhat from map to " << ofile_map_yhat << std::endl;
		std::cout << "Writing env weights to " << ofile_w << std::endl;
		std::cout << "Writing 'rescan' p-values of MAP to " << ofile_rescan << std::endl;

		if(p.verbose){
			std::string ofile_elbo = fstream_init(outf_elbo, file_prefix, "_elbo");
			std::cout << "Writing ELBO from each VB iteration to " << ofile_elbo << std::endl;

			std::string ofile_alpha_diff = fstream_init(outf_alpha_diff, file_prefix, "_alpha_diff");
			std::cout << "Writing max change in alpha from each VB iteration to " << ofile_alpha_diff << std::endl;
		}
	}

	void output_results(const VbTracker& tracker, const int my_n_grid){
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
					weights[ii] = tracker.logw_list[ii];
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
			if(p.mode_mog_prior){
				outf << " pve_large" << ee;
 			}
			outf << " sigma" << ee;
			if(p.mode_mog_prior){
				outf << " sigma_spike" << ee;
				outf << " sigma_spike_dilution" << ee;
			}
			outf << " lambda" << ee;
		}
		outf << std::endl;

		for (int ii = 0; ii < my_n_grid; ii++){
			outf << std::setprecision(4) << weights[ii] << " ";
			outf << tracker.logw_list[ii] << " ";
			outf << tracker.counts_list[ii] << " ";
			outf << tracker.elapsed_time_list[ii] <<  " ";
			outf << tracker.hyps_list[ii].sigma;
//			for (int ee = 0; ee < n_effects; ee++) {
//				outf << " " << tracker.hyps_list[ii].pve2[ee];
//			}

			outf << std::setprecision(8) << std::fixed;
			for (int ee = 0; ee < n_effects; ee++){
				outf << " " << tracker.hyps_list[ii].pve(ee);
				if(p.mode_mog_prior){
					outf << " " << tracker.hyps_list[ii].pve_large(ee);
				}
				outf << " " << tracker.hyps_list[ii].slab_relative_var(ee);
				if(p.mode_mog_prior){
					outf << std::setprecision(2);
					outf << " " << tracker.hyps_list[ii].spike_relative_var(ee);
					outf << " " << tracker.hyps_list[ii].slab_relative_var(ee) / tracker.hyps_list[ii].spike_relative_var(ee);
					outf << std::setprecision(4);
				}
				outf << " " << tracker.hyps_list[ii].lambda(ee);
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
				wmean_alpha += weights[ii] * tracker.vp_list[ii].alpha;
				wmean_beta  += weights[ii] * tracker.vp_list[ii].alpha * tracker.vp_list[ii].mu;
				nmean_alpha += tracker.vp_list[ii].alpha;
				nmean_beta  += tracker.vp_list[ii].alpha * tracker.vp_list[ii].mu;
			}
		}
		nmean_alpha /= (double) my_n_grid;
		nmean_beta  /= (double) my_n_grid;

		Eigen::ArrayXXd nmean_alpha_sd = Eigen::ArrayXXd::Zero(n_var, n_effects);
		Eigen::ArrayXXd nmean_beta_sd  = Eigen::ArrayXXd::Zero(n_var, n_effects);
		for (int ii = 0; ii < my_n_grid; ii++){
			if(std::isfinite(weights[ii])){
				nmean_alpha_sd += (tracker.vp_list[ii].alpha - nmean_alpha).square();
				nmean_beta_sd  += (tracker.vp_list[ii].alpha * tracker.vp_list[ii].mu - nmean_beta).square();
			}
		}
		nmean_alpha_sd /= (double) (my_n_grid - 1);
		nmean_beta_sd  /= (double) (my_n_grid - 1);

		// MAP snp-stats to file (include covars)
		long int ii_map = std::distance(weights.begin(), std::max_element(weights.begin(), weights.end()));
		write_snp_stats_to_file(outf_map, tracker.vp_list[ii_map], true, true);


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
		VariationalParametersLite vp_map = tracker.vp_list[ii_map];
		calcPredEffects(vp_map);
		if(n_effects == 1) {
			outf_map_pred << "Xbeta" << std::endl;
			for (std::uint32_t ii = 0; ii < n_samples; ii++) {
				outf_map_pred << vp_map.ym(ii) << std::endl;
			}
		} else {
			outf_map_pred << "Xbeta eta Xgamma" << std::endl;
			for (std::uint32_t ii = 0; ii < n_samples; ii++) {
				outf_map_pred << vp_map.ym(ii) << " " << vp_map.eta(ii) << " " << vp_map.yx(ii) << std::endl;
			}
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

		// Rescan of map
		if(n_env > 1) {
			Eigen::VectorXd gam_neglogp(n_var);
			rescanGWAS(tracker.vp_list[ii_map], gam_neglogp);
			outf_rescan << "chr rsid pos a0 a1 maf info neglogp" << std::endl;
			for (std::uint32_t kk = 0; kk < n_var; kk++) {
				outf_rescan << X.chromosome[kk] << " " << X.rsid[kk] << " " << X.position[kk];
				outf_rescan << " " << X.al_0[kk] << " " << X.al_1[kk] << " ";
				outf_rescan << X.maf[kk] << " " << X.info[kk] << " " << gam_neglogp(kk);
				outf_rescan << std::endl;
			}
		}


		if(p.verbose){
			outf_elbo << std::setprecision(4) << std::fixed;
			for (int ii = 0; ii < my_n_grid; ii++){
				for (int cc = 0; cc < tracker.logw_updates_list[ii].size(); cc++){
					outf_elbo << tracker.logw_updates_list[ii][cc] << " ";
				}
				outf_elbo << std::endl;
			}

			outf_alpha_diff << std::setprecision(4) << std::fixed;
			for (int ii = 0; ii < my_n_grid; ii++){
				for (int cc = 0; cc < tracker.alpha_diff_list[ii].size(); cc++){
					outf_alpha_diff << tracker.alpha_diff_list[ii][cc] << " ";
				}
				outf_alpha_diff << std::endl;
			}
		}
	}

	void write_snp_stats_to_file(boost_io::filtering_ostream& ofile,
			VariationalParametersLite vp,
			const bool& write_covars,
			const bool& write_mog){
		// Assumes ofile has been initialised.

		// Header
		ofile << "chr rsid pos a0 a1 maf info";
		for (int ee = 0; ee < n_effects; ee++){
			ofile << " beta" << ee << " alpha" << ee << " mu" << ee << " s_sq" << ee;
			if(write_mog && p.mode_mog_prior){
				ofile << " mu_spike" << ee << " s_sq_spike" << ee;
			}
		}
		ofile << std::endl;

		ofile << std::setprecision(9) << std::fixed;

		// Covars if appropriate
		if(write_covars && p.use_vb_on_covars){
			for (int cc = 0; cc < n_covar; cc++){
				ofile << "NA " << covar_names[cc] << " NA NA NA " << 1;
				ofile <<  " " << vp.muc(cc);
				for (int ee = 1; ee < n_effects; ee++){
					ofile << " NA NA";
				}
				ofile << std::endl;
			}
		}

		// Genetic params
		Eigen::ArrayXXd       beta_vec  = vp.alpha * vp.mu;
		if(p.mode_mog_prior) beta_vec += (1 - vp.alpha) * vp.mup;

		outf_inits << std::endl;
		for (std::uint32_t kk = 0; kk < n_var; kk++) {
			ofile << X.chromosome[kk] << " " << X.rsid[kk] << " " << X.position[kk];
			ofile << " " << X.al_0[kk] << " " << X.al_1[kk] << " " << X.maf[kk] << " " << X.info[kk];
			for (int ee = 0; ee < n_effects; ee++) {
				ofile << " " << beta_vec(kk, ee);
				ofile << " " << vp.alpha(kk, ee);
				ofile << " " << vp.mu(kk, ee);
				ofile << " " << vp.s_sq(kk, ee);
				if (write_mog && p.mode_mog_prior) {
					ofile << " " << vp.mup(kk, ee);
					ofile << " " << vp.sp_sq(kk, ee);
				}
			}
			ofile << std::endl;
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
		hyps_grid           = subset_matrix(hyps_grid, valid_points);

		if(valid_points.empty()){
			throw std::runtime_error("No valid grid points in hyps_grid.");
		} else if(n_grid > valid_points.size()){
			std::cout << "WARNING: " << n_grid - valid_points.size();
			std::cout << " invalid grid points removed from hyps_grid." << std::endl;
			n_grid = (int) valid_points.size();

			// update print interval
			print_interval = std::max(1, n_grid / 10);
		}

		// r1_hyps_grid assigned during constructor (ie before this function call)
		int r1_n_grid   = r1_hyps_grid.rows();
		r1_valid_points = validate_grid(r1_hyps_grid, n_var);
		r1_hyps_grid    = subset_matrix(r1_hyps_grid, r1_valid_points);

		if(r1_valid_points.empty()){
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

		bool chck_sigma   = (grid(ii, sigma_ind)   >  0.0 && std::isfinite(grid(ii, sigma_ind)));
		bool chck_sigma_b = (grid(ii, sigma_b_ind) >  0.0 && std::isfinite(grid(ii, sigma_b_ind)));
		bool chck_sigma_g = (grid(ii, sigma_g_ind) >= 0.0 && std::isfinite(grid(ii, sigma_g_ind)));
		bool chck_lam_b   = (lam_b >= 1.0 / (double) n_var) && (lam_b < 1.0) && std::isfinite(lam_b);
		bool chck_lam_g   = (lam_g >= 0) && (lam_g < 1.0) && std::isfinite(lam_g);
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
