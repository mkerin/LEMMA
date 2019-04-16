/* re-implementation of variational bayes algorithm for 1D GxE

   How to use Eigen::Ref:
   https://stackoverflow.com/questions/21132538/correct-usage-of-the-eigenref-class

   Building static executable:
   https://stackoverflow.com/questions/3283021/compile-a-standalone-static-executable
 */
#ifndef VBAYES_X2_HPP
#define VBAYES_X2_HPP

#include "parameters.hpp"
#include "data.hpp"
#include "eigen_utils.hpp"
#include "my_timer.hpp"
#include "typedefs.hpp"
#include "file_streaming.hpp"
#include "variational_parameters.hpp"
#include "vbayes_tracker.hpp"
#include "hyps.hpp"

#include <algorithm>
#include <chrono>     // start/end time info
#include <cmath>      // isnan
#include <ctime>      // start/end time info
#include <cstdint>    // uint32_t
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <thread>
#include <set>
#include "sys/types.h"
#include "tools/eigen3.3/Dense"
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/filesystem.hpp>

namespace boost_m = boost::math;
namespace boost_io = boost::iostreams;

inline double sigmoid(double x){
	return 1.0 / (1.0 + std::exp(-x));
}

class VBayesX2 {
public:
// Constants
const double PI = 3.1415926535897;
const double eps = std::numeric_limits<double>::min();
const double alpha_tol = 1e-4;
const double logw_tol = 1e-2;
const double sigma_c = 10000;
std::vector< std::string > covar_names;
std::vector< std::string > env_names;

// Column order of hyperparameters in grid
const std::vector< std::string > hyps_names = {"sigma", "sigma_b", "sigma_g",
	                                           "lambda_b", "lambda_g"};

// sizes
int n_effects;        // no. interaction variables + 1
std::uint32_t n_samples;
unsigned long n_covar;
unsigned long n_env;
std::uint32_t n_var;
std::uint32_t n_var2;
bool random_params_init;
bool run_round1;
double N;         // (double) n_samples

// Chromosomes in data
int n_chrs;
std::vector<int> chrs_present;
std::vector<int> chrs_index;


//
parameters& p;
std::vector< std::uint32_t > fwd_pass;
std::vector< std::uint32_t > back_pass;
//	std::vector< std::vector < std::uint32_t >> fwd_pass_chunks;
//	std::vector< std::vector < std::uint32_t >> back_pass_chunks;
std::vector< std::vector < std::uint32_t > > main_fwd_pass_chunks, gxe_fwd_pass_chunks;
std::vector< std::vector < std::uint32_t > > main_back_pass_chunks, gxe_back_pass_chunks;
std::vector< int > env_fwd_pass;
std::vector< int > env_back_pass;
std::map<unsigned long, Eigen::MatrixXd> D_correlations;         //We can keep D^t D for the main effects

// Data
GenotypeMatrix&  X;
EigenDataVector Y;                   // residual phenotype matrix
EigenDataArrayX Cty;                  // vector of C^T x y where C the matrix of covariates
EigenDataArrayXX E;                  // matrix of variables used for GxE interactions
EigenDataMatrix& C;                  // matrix of covariates (superset of GxE variables)
Eigen::MatrixXd XtE;                  // matrix of covariates (superset of GxE variables)

Eigen::ArrayXXd& dXtEEX;             // P x n_env^2; col (l * n_env + m) is the diagonal of X^T * diag(E_l * E_m) * X

// Global location of y_m = E[X beta] and y_x = E[X gamma]
EigenDataMatrix YY, YX, YM, ETA, ETA_SQ;

// genome wide scan computed upstream
Eigen::ArrayXXd& snpstats;

// Init points
VariationalParametersLite vp_init;
std::vector<Hyps> hyps_inits;

// boost fstreams
boost_io::filtering_ostream outf, outf_map, outf_wmean, outf_nmean, outf_inits;
boost_io::filtering_ostream outf_elbo, outf_alpha_diff, outf_map_pred, outf_weights;
boost_io::filtering_ostream outf_rescan, outf_map_covar;

// Monitoring
std::chrono::system_clock::time_point time_check;
std::chrono::duration<double> elapsed_innerLoop;
VariationalParametersLite GLOBAL_map_vp;

explicit VBayesX2(Data& dat) : X(dat.G),
	Y(Eigen::Map<EigenDataVector>(dat.Y.data(), dat.Y.rows())),
	C(dat.C),
	dXtEEX(dat.dXtEEX),
	snpstats(dat.snpstats),
	p(dat.params),
	GLOBAL_map_vp(dat.params),
	hyps_inits(dat.hyps_inits),
	vp_init(dat.vp_init){
	std::cout << "Initialising vbayes object" << std::endl;
	mkl_set_num_threads_local(p.n_thread);

	// Data size params
	n_effects      = dat.n_effects;
	n_var          = dat.n_var;
	n_env          = dat.n_env;
	n_var2         = n_effects * dat.n_var;
	n_samples      = dat.n_samples;
	n_covar        = dat.n_covar;
	covar_names    = dat.covar_names;
	env_names      = dat.env_names;
	N              = (double) n_samples;
	E = dat.E;

	assert(Y.rows() == n_samples);
	assert(X.rows() == n_samples);
	assert(n_covar == C.cols());
	assert(n_covar == covar_names.size());
	assert(n_env == E.cols());
	assert(n_env == env_names.size());


	random_params_init = false;
	if(p.user_requests_round1) {
		run_round1     = true;
	} else {
		run_round1     = false;
	}

	std::set<int> tmp(X.chromosome.begin(), X.chromosome.end());
	chrs_present.assign(tmp.begin(), tmp.end());
	n_chrs = chrs_present.size();
	for (int cc = 0; cc < n_chrs; cc++) {
		chrs_index.push_back(cc);
	}

	p.main_chunk_size = (unsigned int) std::min((long int) p.main_chunk_size, (long int) n_var);
	p.gxe_chunk_size = (unsigned int) std::min((long int) p.gxe_chunk_size, (long int) n_var);

	// Read environmental variables
	if(n_env > 0) {
		std::cout << "Computing XtE" << std::endl;
		XtE = X.transpose_multiply(E);
		std::cout << "XtE computed" << std::endl;
	}

	// Allocate memory - fwd/back pass vectors
	for(std::uint32_t kk = 0; kk < n_var * n_effects; kk++) {
		fwd_pass.push_back(kk);
		back_pass.push_back(n_var2 - kk - 1);
	}

	for(int ll = 0; ll < n_env; ll++) {
		env_fwd_pass.push_back(ll);
		env_back_pass.push_back(n_env - ll - 1);
	}

	unsigned long n_main_segs, n_gxe_segs, n_chunks;
	// ceiling of n_var / chunk size
	n_main_segs = (n_var + p.main_chunk_size - 1) / p.main_chunk_size;
	// ceiling of n_var / chunk size
	n_gxe_segs = (n_var + p.gxe_chunk_size - 1) / p.gxe_chunk_size;
	n_chunks = n_main_segs;
	if(n_effects > 1) {
		n_chunks += n_gxe_segs;
	}
//
//		fwd_pass_chunks.resize(n_chunks);
//		back_pass_chunks.resize(n_chunks);
//		for(std::uint32_t kk = 0; kk < n_effects * n_var; kk++){
//			std::uint32_t ch_index = (kk < n_var ? kk / p.main_chunk_size : n_main_segs + (kk % n_var) / p.gxe_chunk_size);
//			fwd_pass_chunks[ch_index].push_back(kk);
//			back_pass_chunks[n_chunks - 1 - ch_index].push_back(kk);
//		}

	main_fwd_pass_chunks.resize(n_main_segs);
	main_back_pass_chunks.resize(n_main_segs);
	gxe_fwd_pass_chunks.resize(n_gxe_segs);
	gxe_back_pass_chunks.resize(n_gxe_segs);
	for(std::uint32_t kk = 0; kk < n_var; kk++) {
		std::uint32_t main_ch_index = kk / p.main_chunk_size;
		std::uint32_t gxe_ch_index = kk / p.gxe_chunk_size;
		main_fwd_pass_chunks[main_ch_index].push_back(kk);
		main_back_pass_chunks[n_main_segs - 1 - main_ch_index].push_back(kk);
		gxe_fwd_pass_chunks[gxe_ch_index].push_back(kk + n_var);
		gxe_back_pass_chunks[n_gxe_segs - 1 - gxe_ch_index].push_back(kk + n_var);
	}


//		for (long ii = 0; ii < n_chunks; ii++){
//			std::reverse(back_pass_chunks[ii].begin(), back_pass_chunks[ii].end());
//		}
	for (long ii = 0; ii < n_main_segs; ii++) {
		std::reverse(main_back_pass_chunks[ii].begin(), main_back_pass_chunks[ii].end());
	}
	for (long ii = 0; ii < n_gxe_segs; ii++) {
		std::reverse(gxe_back_pass_chunks[ii].begin(), gxe_back_pass_chunks[ii].end());
	}

//		for (auto chunk : fwd_pass_chunks){
//			for (auto ii : chunk){
//				std::cout << ii << " ";
//			}
//			std::cout << std::endl;
//		}
//
//		for (auto chunk : back_pass_chunks){
//			for (auto ii : chunk){
//				std::cout << ii << " ";
//			}
//			std::cout << std::endl;
//		}
//
//		for (auto chunk : gxe_back_pass_chunks){
//			for (auto ii : chunk){
//				std::cout << ii << " ";
//			}
//			std::cout << std::endl;
//		}
//
//		for (auto chunk : main_back_pass_chunks){
//			for (auto ii : chunk){
//				std::cout << ii << " ";
//			}
//			std::cout << std::endl;
//		}

	if(n_effects == 1) {
		gxe_back_pass_chunks.clear();
		gxe_fwd_pass_chunks.clear();
	}

	// Generate initial values for each run
	random_params_init = false;
	if(p.mode_random_start) {
		std::cout << "Beta and gamma initialised with random draws" << std::endl;
		random_params_init = true;
	}

	if(n_env > 0) {
		// cast used if DATA_AS_FLOAT
		vp_init.eta     = E.matrix() * vp_init.mean_weights().matrix().cast<scalarData>();
		vp_init.eta_sq  = vp_init.eta.array().square().matrix();
		vp_init.eta_sq += E.square().matrix() * vp_init.sw_sq.matrix().template cast<scalarData>();
	}
	calcPredEffects(vp_init);

	if(n_covar > 0) {
		Cty = C.transpose() * Y;
	}
}

~VBayesX2(){
	// Close all ostreams
	boost_io::close(outf);
	boost_io::close(outf_map);
	boost_io::close(outf_wmean);
	boost_io::close(outf_nmean);
	boost_io::close(outf_elbo);
	boost_io::close(outf_alpha_diff);
	boost_io::close(outf_inits);
	boost_io::close(outf_rescan);
	boost_io::close(outf_map_covar);
}

void run(){
	std::cout << "Starting variational inference" << std::endl;
	time_check = std::chrono::system_clock::now();
	// Parrallel starts swapped for multithreaded inference
	int n_thread = 1;

	// Round 1; looking for best start point
	if(run_round1) {

		std::vector< VbTracker > trackers(n_thread, p);
		long n_grid = hyps_inits.size();
		run_inference(hyps_inits, true, 1, trackers);

		if(p.verbose) {
			write_trackers_to_file("round1_", trackers, n_grid);
		}

		// Find best init
		double logw_best = -std::numeric_limits<double>::max();
		bool init_not_set = true;
		for (int ii = 0; ii < n_grid; ii++) {
			double logw      = trackers[ii].logw;
			if(std::isfinite(logw) && logw > logw_best) {
				vp_init      = trackers[ii].vp;
				logw_best    = logw;
				init_not_set = false;
			}
		}

		if(init_not_set) {
			throw std::runtime_error("No valid start points found (elbo estimates all non-finite?).");
		}

		// gen initial predicted effects
		calcPredEffects(vp_init);

		print_time_check();
	}

	// Write inits to file - exclude covar values
	std::string ofile_inits = fstream_init(outf_inits, "", "_inits");
	std::cout << "Writing start points for alpha and mu to " << ofile_inits << std::endl;
	write_snp_stats_to_file(outf_inits, n_effects, n_var, vp_init, X, p, false);
	boost_io::close(outf_inits);

	long n_grid = hyps_inits.size();
	std::vector< VbTracker > trackers(n_grid, p);
	run_inference(hyps_inits, false, 2, trackers);

	write_trackers_to_file("", trackers, n_grid);
	std::cout << "Variational inference finished" << std::endl;
}

void run_inference(const std::vector<Hyps>& hyps_inits,
                   const bool random_init,
                   const int round_index,
                   std::vector<VbTracker>& trackers){
	// Writes results from inference to trackers

	long n_grid = hyps_inits.size();
	// Parrallel starts swapped for multithreaded inference
	int n_thread = 1;

	// Divide grid of hyperparameters into chunks for multithreading
	std::vector< std::vector< int > > chunks(n_thread);
	for (int ii = 0; ii < n_grid; ii++) {
		int ch_index = (ii % n_thread);
		chunks[ch_index].push_back(ii);
	}

	runOuterLoop(round_index, hyps_inits, n_grid, chunks[0], random_init, trackers);
}

void runOuterLoop(const int round_index,
                  std::vector<Hyps> all_hyps,
                  const unsigned long n_grid,
                  const std::vector<int>& grid_index_list,
                  const bool random_init,
                  std::vector<VbTracker>& all_tracker){

	// Run outer loop - don't update trackers
	auto innerLoop_start = std::chrono::system_clock::now();
	runInnerLoop(random_init, round_index, all_hyps, all_tracker);
	auto innerLoop_end = std::chrono::system_clock::now();
	elapsed_innerLoop = innerLoop_end - innerLoop_start;

	// 'rescan' GWAS of Z on y-ym
	if(n_effects > 1) {
		for (int nn = 0; nn < n_grid; nn++) {
			Eigen::VectorXd gam_neglogp(n_var);
			rescanGWAS(all_tracker[nn].vp, gam_neglogp);
			all_tracker[nn].push_rescan_gwas(X, n_var, gam_neglogp);
		}
	}
}

void runInnerLoop(const bool random_init,
                  const int round_index,
                  std::vector<Hyps>& all_hyps,
                  std::vector<VbTracker>& all_tracker){
	// minimise KL Divergence and assign elbo estimate
	// Assumes vp_init already exist
	// TODO: re intergrate random starts
	int print_interval = 25;
	if(random_init) {
		throw std::logic_error("Random starts no longer implemented");
	}
	unsigned long n_grid = all_hyps.size();

	std::vector<VariationalParameters> all_vp;
	setup_variational_params(all_hyps, all_vp);

	// Run inner loop until convergence
	std::vector<int> converged(n_grid, 0);
	bool all_converged = false;
	std::vector<Eigen::ArrayXd> alpha_prev(n_grid);
	std::vector<double> i_logw(n_grid, -1*std::numeric_limits<double>::max());

	for (int nn = 0; nn < n_grid; nn++) {
		all_tracker[nn].init_interim_output(nn, round_index, n_effects, n_env, env_names, all_vp[nn]);
	}

	// SQUAREM objects
	std::vector<Hyps> theta0 = all_hyps;
	std::vector<Hyps> theta1 = all_hyps;
	std::vector<Hyps> theta2 = all_hyps;

	// Allow more flexible start point so that we can resume previous inference run
	int count = p.vb_iter_start;
	while(!all_converged && count < p.vb_iter_max) {
		for (int nn = 0; nn < n_grid; nn++) {
			alpha_prev[nn] = all_vp[nn].alpha_beta;
		}
		std::vector<double> logw_prev = i_logw;
		std::vector<double> alpha_diff(n_grid);

		updateAllParams(count, round_index, all_vp, all_hyps, logw_prev);

		// SQUAREM
		if (p.mode_squarem) {
			if(count % 3 == 0) {
				theta0 = all_hyps;
			} else if (count % 3 == 1) {
				theta1 = all_hyps;
			} else if (count >= p.vb_iter_start + 2) {
				theta2 = all_hyps;
				for (int nn = 0; nn < n_grid; nn++) {
					Hyps rr = theta1[nn] - theta0[nn];
					Hyps vv = (theta2[nn] - theta1[nn]) - rr;
					double step = std::min(-rr.normL2() / vv.normL2(), -1.0);
					Hyps theta = theta0[nn] - 2 * step * rr + step * step * vv;

					// check all hyps in theta remain in valid domain
					while(!theta.domain_is_valid()) {
						step = std::min(step * 0.5, -1.0);
						theta = theta0[nn] - 2 * step * rr + step * step * vv;
					}

					// Copy s_x and pve from theta2
					theta.s_x = theta2[nn].s_x;
					theta.pve = theta2[nn].pve;
					theta.pve_large = theta2[nn].pve_large;

					all_hyps[nn] = theta;
				}
			}
		}

		// update elbo
		for (int nn = 0; nn < n_grid; nn++) {
			i_logw[nn]     = calc_logw(all_hyps[nn], all_vp[nn]);
			alpha_diff[nn] = (alpha_prev[nn] - all_vp[nn].alpha_beta).abs().maxCoeff();
		}

		// Interim output
		for (int nn = 0; nn < n_grid; nn++) {
			all_hyps[nn].update_pve();
			all_tracker[nn].push_interim_hyps(count, all_hyps[nn], i_logw[nn], alpha_diff[nn], n_effects,
			                                  n_var, n_env, all_vp[nn]);
			if (p.param_dump_interval > 0 && count % p.param_dump_interval == 0) {
				all_tracker[nn].dump_state(count, n_samples, n_covar, n_var,
				                           n_env, n_effects,
				                           all_vp[nn], all_hyps[nn], Y, C,
				                           X, covar_names, env_names);
			}
		}

		// Diagnose convergence
		// Disallow convergence on squarem iter
		if(!p.mode_squarem || count % 3 != 2) {
			for (int nn = 0; nn < n_grid; nn++) {
				double logw_diff = std::abs(i_logw[nn] - logw_prev[nn]);
				if (p.alpha_tol_set_by_user && p.elbo_tol_set_by_user) {
					if (alpha_diff[nn] < p.alpha_tol && logw_diff < p.elbo_tol) {
						converged[nn] = 1;
						all_tracker[nn].dump_state(count, n_samples, n_covar, n_var,
						                           n_env, n_effects,
						                           all_vp[nn], all_hyps[nn], Y, C,
						                           X, covar_names, env_names);
					}
				} else if (p.alpha_tol_set_by_user) {
					if (alpha_diff[nn] < p.alpha_tol) {
						converged[nn] = 1;
						all_tracker[nn].dump_state(count, n_samples, n_covar, n_var,
						                           n_env, n_effects,
						                           all_vp[nn], all_hyps[nn], Y, C,
						                           X, covar_names, env_names);
					}
				} else if (p.elbo_tol_set_by_user) {
					if (logw_diff < p.elbo_tol) {
						converged[nn] = 1;
						all_tracker[nn].dump_state(count, n_samples, n_covar, n_var,
						                           n_env, n_effects,
						                           all_vp[nn], all_hyps[nn], Y, C,
						                           X, covar_names, env_names);
					}
				} else {
					if (alpha_diff[nn] < alpha_tol && logw_diff < logw_tol) {
						converged[nn] = 1;
						all_tracker[nn].dump_state(count, n_samples, n_covar, n_var,
						                           n_env, n_effects,
						                           all_vp[nn], all_hyps[nn], Y, C,
						                           X, covar_names, env_names);
					}
				}
			}
		}
		if (std::all_of(converged.begin(), converged.end(), [](int i){
				return i == 1;
			})) {
			all_converged = true;
		}
		count++;

		// Report progress to std::cout
		if((count + 1) % print_interval == 0) {
			int n_converged = 0;
			for (auto& n : converged) {
				n_converged += n;
			}

			std::cout << "Completed " << count+1 << " iterations, " << n_converged << " runs converged";
			print_time_check();
		}
	}

	if(any_of(i_logw.begin(), i_logw.end(), [](double x) {
			return !std::isfinite(x);
		})) {
		std::cout << "WARNING: non-finite elbo estimate produced" << std::endl;
	}

	// Log all things that we want to track
	for (int nn = 0; nn < n_grid; nn++) {
		all_tracker[nn].logw = i_logw[nn];
		all_tracker[nn].count = count;
		all_tracker[nn].vp = all_vp[nn].convert_to_lite();
		all_tracker[nn].hyps = all_hyps[nn];
		all_tracker[nn].push_vp_converged(X, n_var, n_effects);
	}
}

void setup_variational_params(const std::vector<Hyps>& all_hyps,
                              std::vector<VariationalParameters>& all_vp){
	unsigned long n_grid = all_hyps.size();

	// Init global locations YM YX
	YY.resize(n_samples, n_grid);
	YM.resize(n_samples, n_grid);
	for (int nn = 0; nn < n_grid; nn++) {
		YM.col(nn) = vp_init.ym;
		YY.col(nn) = Y;
	}
	YX.resize(n_samples, n_grid);
	ETA.resize(n_samples, n_grid);
	ETA_SQ.resize(n_samples, n_grid);
	if (n_effects > 1) {
		for (int nn = 0; nn < n_grid; nn++) {
			YX.col(nn) = vp_init.yx;
			ETA.col(nn) = vp_init.eta;
			ETA_SQ.col(nn) = vp_init.eta_sq;
		}
	}

	// Init variational params
	for (int nn = 0; nn < n_grid; nn++) {
		VariationalParameters vp(p, YM.col(nn), YX.col(nn), ETA.col(nn), ETA_SQ.col(nn));
		vp.init_from_lite(vp_init);
		if(n_effects > 1) {
			vp.calcEdZtZ(dXtEEX, n_env);
		}
		all_vp.push_back(vp);
	}
}

/********** VB update functions ************/
void updateAllParams(const int& count,
                     const int& round_index,
                     std::vector<VariationalParameters>& all_vp,
                     std::vector<Hyps>& all_hyps,
                     std::vector<double> logw_prev){
	std::vector< std::uint32_t > iter;
	std::vector< std::vector< std::uint32_t > > iter_chunks;
	unsigned long n_grid = all_hyps.size();
	std::vector<double> i_logw(n_grid);

	// Update covar main effects
	if (n_covar > 0) {
		for (int nn = 0; nn < n_grid; nn++) {
			for (int cc = 0; cc < n_covar; cc++){
				_update_covar(cc, all_hyps[nn], all_vp[nn]);
			}
		}
		check_monotonic_elbo(all_hyps, all_vp, count, logw_prev, "updateCovarEffects");
	}

	// Update main & interaction effects
	bool is_fwd_pass = (count % 2 == 0);
	if(is_fwd_pass) {
		for (auto chunk: main_fwd_pass_chunks){
			_update_beta(chunk, all_vp, all_hyps);
		}
		check_monotonic_elbo(all_hyps, all_vp, count, logw_prev, "updateAlphaMu_fwd_main");

		for (auto chunk: gxe_fwd_pass_chunks){
			_update_gamma(chunk, all_vp, all_hyps);
		}
		check_monotonic_elbo(all_hyps, all_vp, count, logw_prev, "updateAlphaMu_fwd_gxe");
	} else {
		for (auto chunk: gxe_back_pass_chunks){
			_update_gamma(chunk, all_vp, all_hyps);
		}
		check_monotonic_elbo(all_hyps, all_vp, count, logw_prev, "updateAlphaMu_back_gxe");

		for (auto chunk: main_back_pass_chunks){
			_update_beta(chunk, all_vp, all_hyps);
		}
		check_monotonic_elbo(all_hyps, all_vp, count, logw_prev, "updateAlphaMu_back_main");
	}

	// Update env-weights
	if (n_effects > 1 && n_env > 1) {
		for (int nn = 0; nn < n_grid; nn++) {
			for (int uu = 0; uu < p.env_update_repeats; uu++) {
				for (auto ll : env_fwd_pass){
					_updates_weights(ll, all_hyps[nn], all_vp[nn]);
				}
				for (auto ll : env_back_pass){
					_updates_weights(ll, all_hyps[nn], all_vp[nn]);
				}
			}

			// Recompute eta_sq
			all_vp[nn].eta_sq  = all_vp[nn].eta.array().square().matrix();
			all_vp[nn].eta_sq += E.square().matrix() * all_vp[nn].var_weights().matrix().cast<scalarData>();

			// Recompute expected value of diagonal of ZtZ
			all_vp[nn].calcEdZtZ(dXtEEX, n_env);

			// Compute s_x; sum of column variances of Z
			Eigen::ArrayXd muw_sq(n_env * n_env);
			for (int ll = 0; ll < n_env; ll++) {
				for (int mm = 0; mm < n_env; mm++) {
					muw_sq(mm*n_env + ll) = all_vp[nn].mean_weights(mm) * all_vp[nn].mean_weights(ll);
				}
			}
			// WARNING: Hard coded limit!
			// WARNING: Updates S_x in hyps
			all_hyps[nn].s_x(0) = (double) n_var;
			all_hyps[nn].s_x(1) = (dXtEEX.rowwise() * muw_sq.transpose()).sum() / (N - 1.0);
			all_hyps[nn].s_x(1) -= (XtE * all_vp[nn].mean_weights().matrix()).array().square().sum() / N / (N - 1.0);

			check_monotonic_elbo(all_hyps[nn], all_vp[nn], count, logw_prev[nn], "updateEnvWeights");
		}
	}

	// Maximise hyps
	if (round_index > 1 && p.mode_empirical_bayes) {
		for (int nn = 0; nn < n_grid; nn++) {
			maximiseHyps(all_hyps[nn], all_vp[nn]);
			check_monotonic_elbo(all_hyps[nn], all_vp[nn], count, logw_prev[nn], "maxHyps");
		}
	}

	// Update PVE
	for (int nn = 0; nn < n_grid; nn++) {
		all_hyps[nn].update_pve();
	}
}

void _update_covar(long cc, const Hyps& hyps, VariationalParameters& vp){
	double old = vp.mean_covar(cc);

	double EXty, EXtX;
	EXty = Cty(cc) - (vp.ym + vp.yx.cwiseProduct(vp.eta)).dot(C.col(cc));
	EXtX = (N-1.0);

	Gaussian w = vp.covar_c_step(cc, EXty, EXtX, hyps);

	vp.muc(cc) = w.mean();
	vp.sc_sq(cc) = w.var();

	vp.ym += (vp.mean_covar(cc) - old) * C.col(cc).matrix();
}

	void _updates_weights(long ll, const Hyps& hyps,
						  VariationalParameters& vp){
		double old = vp.mean_weights(ll);

		Eigen::MatrixXd tmp;
		tmp = vp.var_gam().matrix().transpose() * dXtEEX.block(0, ll * n_env, n_var, n_env).matrix() *  vp.mean_weights();

		double EXty, EXtX;
		EXty       = ((Y - vp.ym).array() * E.col(ll) * vp.yx.array()).sum();
		EXty       -= (vp.yx.array() * vp.yx.array() * E.col(ll) * vp.eta.array()).sum();
		EXty       -= tmp(0,0);

		EXtX = (vp.yx.array().square() * E.col(ll).square()).sum();
		EXtX += (vp.var_gam() * dXtEEX.col(ll*n_env + ll)).sum();

		Gaussian w = vp.weights_l_step(ll, EXty, EXtX, hyps);

		vp.muw(ll) = w.mean();
		vp.sw_sq(ll) = w.var();

		vp.eta += (vp.mean_weights(ll) - old) * E.col(ll).matrix();

	}

void _update_beta(const std::vector<std::uint32_t>& iter,
				   std::vector<VariationalParameters>& all_vp,
				   const std::vector<Hyps>& all_hyps){
	long ch_len   = iter.size();

	EigenDataMatrix D, EXty;

	// D is n_samples x snp_batch
	if(D.cols() != ch_len) {
		D.resize(n_samples, ch_len);
	}
	X.col_block3(iter, D);
	EXty = D.transpose() * (YY - YM - YX.cwiseProduct(ETA));

	for (int nn = 0; nn < all_hyps.size(); nn++) {
		EigenDataMatrix D_corr = (D.transpose() * D);
		EigenDataMatrix EXtX = D_corr.diagonal();
		EigenDataVector rr_k_old(ch_len), rr_k_new(ch_len);

		for (int ii = 0; ii < ch_len; ii++) {
			long jj = (iter[ii] % n_var);
			double old = all_vp[nn].mean_beta(jj);
			rr_k_old(ii) = old;

			// Get param updates
			all_vp[nn].beta_j_step(jj, EXty(ii, nn), EXtX(ii), all_hyps[nn]);


			rr_k_new(ii) = all_vp[nn].mean_beta(jj);
			EXty -= (all_vp[nn].mean_beta(jj) - old) * D_corr.col(ii);
		}
		all_vp[nn].ym += D * (rr_k_new - rr_k_old);
	}
}

	void _update_gamma(const std::vector<std::uint32_t>& iter,
					   std::vector<VariationalParameters>& all_vp,
					   const std::vector<Hyps>& all_hyps){
		long ch_len   = iter.size();

		EigenDataMatrix D, EXty;

		// D is n_samples x snp_batch
		if(D.cols() != ch_len) {
			D.resize(n_samples, ch_len);
		}
		X.col_block3(iter, D);

		EXty = D.transpose() * ((YY - YM).cwiseProduct(ETA) - YX.cwiseProduct(ETA_SQ));

		for (int nn = 0; nn < all_hyps.size(); nn++) {
			EigenDataMatrix D_corr = (D.transpose() * all_vp[nn].eta_sq.asDiagonal() * D);
			EigenDataMatrix EXtX = D_corr.diagonal();
			EigenDataVector rr_k_old(ch_len), rr_k_new(ch_len);

			for (int ii = 0; ii < ch_len; ii++) {
				long jj = (iter[ii] % n_var);
				double old = all_vp[nn].mean_gam(jj);
				rr_k_old(ii) = old;

				// Get param updates
				all_vp[nn].gamma_j_step(jj, EXty(ii, nn), EXtX(ii), all_hyps[nn]);

				rr_k_new(ii) = all_vp[nn].mean_gam(jj);
				EXty -= (all_vp[nn].mean_gam(jj) - old) * D_corr.col(ii);
			}
			all_vp[nn].yx += D * (rr_k_new - rr_k_old);
		}
	}

void maximiseHyps(Hyps& hyps,
                  const VariationalParameters& vp){

	// max sigma
	hyps.sigma  = calcExpLinear(hyps, vp);
	if (n_covar > 0) {
		hyps.sigma += (vp.sc_sq + vp.mean_covar().array().square()).sum() / sigma_c;
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
	if(p.mode_mog_prior_beta) {
		hyps.spike_var[ee]  = ((1.0 - vp.alpha_beta) * (vp.s2_beta_sq + vp.mu2_beta.square())).sum();
		hyps.spike_var[ee] /= ( (double)n_var - hyps.lambda[ee]);
		hyps.spike_relative_var[ee] = hyps.spike_var[ee] / hyps.sigma;
	}

	// beta - finish max lambda
	hyps.lambda[ee] /= n_var;

	// gamma
	if(n_effects > 1) {
		ee = 1;
		hyps.lambda[ee] = vp.alpha_gam.sum();

		// max spike & slab variances
		hyps.slab_var[ee]  = (vp.alpha_gam * (vp.s1_gam_sq + vp.mu1_gam.square())).sum();
		hyps.slab_var[ee] /= hyps.lambda[ee];
		hyps.slab_relative_var[ee] = hyps.slab_var[ee] / hyps.sigma;
		if(p.mode_mog_prior_gam) {
			hyps.spike_var[ee]  = ((1.0 - vp.alpha_gam) * (vp.s2_gam_sq + vp.mu2_gam.square())).sum();
			hyps.spike_var[ee] /= ( (double)n_var - hyps.lambda[ee]);
			hyps.spike_relative_var[ee] = hyps.spike_var[ee] / hyps.sigma;
		}

		// finish max lambda
		hyps.lambda[ee] /= n_var;
	}

//		// max spike & slab variances
//		hyps.slab_var.resize(n_effects);
//		hyps.spike_var.resize(n_effects);
//		for (int ee = 0; ee < n_effects; ee++){
//
//			// Initial unconstrained max
//			hyps.slab_var[ee]  = (vp.alpha.col(ee) * (vp.s_sq.col(ee) + vp.mu.col(ee).square())).sum();
//			hyps.slab_var[ee] /= hyps.lambda[ee];
//			if(p.mode_mog_prior){
//				hyps.spike_var[ee]  = ((1.0 - vp.alpha.col(ee)) * (vp.sp_sq.col(ee) + vp.mup.col(ee).square())).sum();
//				hyps.spike_var[ee] /= ( (double)n_var - hyps.lambda[ee]);
//			}
//
//			// Remaxise while maintaining same diff in MoG variances if getting too close
//			if(p.mode_mog_prior && hyps.slab_var[ee] < p.min_spike_diff_factor * hyps.spike_var[ee]){
//				hyps.slab_var[ee]  = (vp.alpha.col(ee) * (vp.s_sq.col(ee) + vp.mu.col(ee).square())).sum();
//				hyps.slab_var[ee] += p.min_spike_diff_factor * ((1.0 - vp.alpha.col(ee)) * (vp.sp_sq.col(ee) + vp.mup.col(ee).square())).sum();
//				hyps.slab_var[ee] /= (double) n_var;
//				hyps.spike_var[ee] = hyps.slab_var[ee] / p.min_spike_diff_factor;
//			}
//		}
//
//		hyps.slab_relative_var = hyps.slab_var / hyps.sigma;
//		if(p.mode_mog_prior){
//			hyps.spike_relative_var = hyps.spike_var / hyps.sigma;
//		}

	// hyps.lam_b         = hyps.lambda(0);
	// hyps.lam_g         = hyps.lambda(1);
	// hyps.sigma_b       = hyps.slab_relative_var(0);
	// hyps.sigma_g       = hyps.slab_relative_var(1);
	// hyps.sigma_g_spike = hyps.spike_relative_var(0);
	// hyps.sigma_g_spike = hyps.spike_relative_var(1);
}

double calc_logw(const Hyps& hyps,
                 const VariationalParameters& vp){

	// Expectation of linear regression log-likelihood
	double int_linear = -1.0 * calcExpLinear(hyps, vp) / 2.0 / hyps.sigma;
	int_linear -= N * std::log(2.0 * PI * hyps.sigma) / 2.0;

	// kl-beta
	double kl_beta = vp.kl_div_beta(hyps);

	double kl_gamma = 0;
	if(n_effects > 1) {
		kl_gamma += vp.kl_div_gamma(hyps);
	}

	// covariates
	double kl_covar = 0.0;
	if(n_covar > 0) {
		kl_covar = vp.kl_div_covars(hyps);
	}

	// weights
	double kl_weights = 0.0;
	if(n_effects > 1 && n_env > 1) {
		kl_weights = vp.kl_div_weights(hyps);
	}

	double res = int_linear + kl_beta + kl_gamma + kl_covar + kl_weights;

	return res;
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

void calcPredEffects(VariationalParameters& vp){
	Eigen::VectorXd rr_beta = vp.mean_beta();

	vp.ym = X * rr_beta;
	if(n_covar > 0) {
		vp.ym += C * vp.mean_covar().matrix().cast<scalarData>();
	}

	if(n_effects > 1) {
		Eigen::VectorXd rr_gam = vp.mean_gam();
		vp.yx = X * rr_gam;
	}
}

void calcPredEffects(VariationalParametersLite& vp) {
	Eigen::VectorXd rr_beta = vp.mean_beta();

	vp.ym = X * rr_beta;
	if(n_covar > 0) {
		vp.ym += C * vp.mean_covar().matrix().cast<scalarData>();
	}

	if(n_effects > 1) {
		Eigen::VectorXd rr_gam = vp.mean_gam();
		vp.yx = X * rr_gam;
	}
}

void check_monotonic_elbo(const Hyps& hyps,
                          VariationalParameters& vp,
                          const int count,
                          double& logw_prev,
                          const std::string& prev_function){
	double i_logw     = calc_logw(hyps, vp);
	if(i_logw < logw_prev) {
		std::cout << count << ": " << prev_function;
		std::cout << " " << logw_prev << " -> " << i_logw;
		std::cout << " (difference of " << i_logw - logw_prev << ")"<< std::endl;
	}
	logw_prev = i_logw;
}

	void check_monotonic_elbo(const std::vector<Hyps>& all_hyps,
							  std::vector<VariationalParameters>& all_vp,
							  const int count,
							  std::vector<double>& logw_prev,
							  const std::string& prev_function){

		for (int nn = 0; nn < all_hyps.size(); nn++) {
			double i_logw = calc_logw(all_hyps[nn], all_vp[nn]);
			if (i_logw < logw_prev[nn]) {
				std::cout << count << ": " << prev_function;
				std::cout << " " << logw_prev[nn] << " -> " << i_logw;
				std::cout << " (difference of " << i_logw - logw_prev[nn] << ")" << std::endl;
			}
			logw_prev[nn] = i_logw;
		}
	}

void normaliseLogWeights(std::vector< double >& my_weights){
	// Safer to normalise log-weights than niavely convert to weights
	// Skip non-finite values!
	int nn = my_weights.size();
	double max_elem = *std::max_element(my_weights.begin(), my_weights.end());
	for (int ii = 0; ii < nn; ii++) {
		my_weights[ii] = std::exp(my_weights[ii] - max_elem);
	}

	double my_sum = 0.0;
	for (int ii = 0; ii < nn; ii++) {
		if(std::isfinite(my_weights[ii])) {
			my_sum += my_weights[ii];
		}
	}

	for (int ii = 0; ii < nn; ii++) {
		my_weights[ii] /= my_sum;
	}

	int nonfinite_count = 0;
	for (int ii = 0; ii < nn; ii++) {
		if(!std::isfinite(my_weights[ii])) {
			nonfinite_count++;
		}
	}

	if(nonfinite_count > 0) {
		std::cout << "WARNING: " << nonfinite_count << " grid points returned non-finite ELBO.";
		std::cout << "Skipping these when producing posterior estimates.";
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
	if(n_covar > 0) {
		int_linear += (N - 1.0) * vp.sc_sq.sum();
	}
	int_linear += (N - 1.0) * vp.var_beta().sum();
	if(n_effects > 1) {
		int_linear += (vp.EdZtZ * vp.var_gam()).sum();
	}

	return int_linear;
}

double calcKLBeta(const Hyps& hyps,
                  const VariationalParameters& vp){
	// KL Divergence of log[ p(beta | u, theta) / q(u, beta) ]
	double res = 0;
	int ee = 0;

	res += std::log(hyps.lambda(ee) + eps) * vp.alpha_beta.sum();
	res += std::log(1.0 - hyps.lambda(ee) + eps) * ((double) n_var - vp.alpha_beta.sum());

	// Need std::log for eps guard to work
	for (std::uint32_t kk = 0; kk < n_var; kk++) {
		res -= vp.alpha_beta(kk) * std::log(vp.alpha_beta(kk) + eps);
		res -= (1 - vp.alpha_beta(kk)) * std::log(1 - vp.alpha_beta(kk) + eps);
	}

	// beta
	if(p.mode_mog_prior_beta) {
		res += n_var / 2.0;

		res -= vp.mean_beta_sq(1).sum() / 2.0 / hyps.slab_var(ee);
		res -= vp.mean_beta_sq(2).sum() / 2.0 / hyps.spike_var(ee);

		res += (vp.alpha_beta * vp.s1_beta_sq.log()).sum() / 2.0;
		res += ((1.0 - vp.alpha_beta) * vp.s2_beta_sq.log()).sum() / 2.0;

		res -= std::log(hyps.slab_var(ee))  * vp.alpha_beta.sum() / 2.0;
		res -= std::log(hyps.spike_var(ee)) * (n_var - vp.alpha_beta.sum()) / 2.0;
	} else {
		res += (vp.alpha_beta * vp.s1_beta_sq.log()).sum() / 2.0;
		res -= (vp.alpha_beta * (vp.mu1_beta.square() + vp.s1_beta_sq)).sum() / 2.0 / hyps.slab_var(ee);

		res += (1 - std::log(hyps.slab_var(ee))) * vp.alpha_beta.sum() / 2.0;
	}
	return res;
}

double calcKLGamma(const Hyps& hyps,
                   const VariationalParameters& vp){
	// KL Divergence of log[ p(beta | u, theta) / q(u, beta) ]
	double res = 0;
	int ee = 1;

	res += std::log(hyps.lambda(ee) + eps) * vp.alpha_gam.sum();
	res += std::log(1.0 - hyps.lambda(ee) + eps) * ((double) n_var - vp.alpha_gam.sum());

	// Need std::log for eps guard to work
	for (std::uint32_t kk = 0; kk < n_var; kk++) {
		res -= vp.alpha_gam(kk) * std::log(vp.alpha_gam(kk) + eps);
		res -= (1 - vp.alpha_gam(kk)) * std::log(1 - vp.alpha_gam(kk) + eps);
	}

	// beta
	if(p.mode_mog_prior_gam) {
		res += n_var / 2.0;

		res -= vp.mean_gam_sq(1).sum() / 2.0 / hyps.slab_var(ee);
		res -= vp.mean_gam_sq(2).sum() / 2.0 / hyps.spike_var(ee);

		res += (vp.alpha_gam * vp.s1_gam_sq.log()).sum() / 2.0;
		res += ((1.0 - vp.alpha_gam) * vp.s2_gam_sq.log()).sum() / 2.0;

		res -= std::log(hyps.slab_var(ee))  * vp.alpha_gam.sum() / 2.0;
		res -= std::log(hyps.spike_var(ee)) * (n_var - vp.alpha_gam.sum()) / 2.0;
	} else {
		res += (vp.alpha_gam * vp.s1_gam_sq.log()).sum() / 2.0;
		res -= (vp.alpha_gam * (vp.mu1_gam.square() + vp.s1_gam_sq)).sum() / 2.0 / hyps.slab_var(ee);

		res += (1 - std::log(hyps.slab_var(ee))) * vp.alpha_gam.sum() / 2.0;
	}
	return res;
}

void rescanGWAS(const VariationalParametersLite& vp,
                Eigen::Ref<Eigen::VectorXd> neglogp){
	// casts used is DATA_AS_FLOAT
	Eigen::VectorXd pheno = (Y.cast<double>() - vp.ym.cast<double>());
	Eigen::VectorXd Z_kk(n_samples);

	for(std::uint32_t jj = 0; jj < n_var; jj++ ) {
		Z_kk = (X.col(jj).cast<double>().cwiseProduct(vp.eta.cast<double>()));
		double ztz_inv = 1.0 / Z_kk.dot(Z_kk);
		double gam = Z_kk.dot(pheno) * ztz_inv;
		double rss_null = (pheno - Z_kk * gam).squaredNorm();

		// T-test of variant j
		boost_m::students_t t_dist(n_samples - 1);
		double main_se_j    = std::sqrt(rss_null / (N - 1.0) * ztz_inv);
		double main_tstat_j = gam / main_se_j;
		if(!std::isfinite(main_tstat_j)) {
			std::cout << main_se_j << std::endl;
			std::cout << main_tstat_j << std::endl;
			std::cout << rss_null << std::endl;
			std::cout << gam << std::endl;
			std::cout << ztz_inv << std::endl;
			std::cout << "eta" << std::endl;
			std::cout << vp.eta << std::endl;
			std::cout << "col" << std::endl;
			std::cout << X.col(jj) << std::endl;
		}
		double main_pval_j  = 2 * boost_m::cdf(boost_m::complement(t_dist, fabs(main_tstat_j)));

		neglogp(jj) = -1 * std::log10(main_pval_j);
	}
}

void compute_residuals_per_chr(const VariationalParametersLite& vp,
                               std::vector<Eigen::VectorXd>& pred_main,
                               std::vector<Eigen::VectorXd>& pred_int,
                               std::vector<Eigen::VectorXd>& chr_residuals){

//		std::set<int> chrs(X.chromosome.begin(), X.chromosome.end());
	assert(pred_main.size() == n_chrs);
	assert(pred_int.size() == n_chrs);
	assert(chr_residuals.size() == n_chrs);

	// casts used if DATA_AS_FLOAT
	Eigen::VectorXd map_residuals;
	if (n_effects > 1) {
		map_residuals = (Y - vp.ym - vp.yx.cwiseProduct(vp.eta)).cast<double>();
	} else {
		map_residuals = (Y - vp.ym).cast<double>();
	}

	// Compute predicted effects from each chromosome
	Eigen::VectorXd Eq_beta, Eq_gam;

	Eq_beta = vp.mean_beta();
	for (auto cc : chrs_index) {
		pred_main[cc] = X.mult_vector_by_chr(chrs_present[cc], Eq_beta);
	}

	if (n_effects > 1) {
		Eq_gam = vp.mean_gam();
		for (auto cc : chrs_index) {
			pred_int[cc]  = X.mult_vector_by_chr(chrs_present[cc], Eq_gam);
		}
	}

	// Compute mean-centered residuals for each chromosome
	for (auto cc : chrs_index) {
		if (n_effects > 1) {
			chr_residuals[cc] = map_residuals + pred_main[cc] + pred_int[cc].cwiseProduct(vp.eta.cast<double>());
		} else {
			chr_residuals[cc] = map_residuals + pred_main[cc];
		}
		chr_residuals[cc].array() -= chr_residuals[cc].mean();
	}
}

void LOCO_pvals(const VariationalParametersLite& vp,
                const std::vector<Eigen::VectorXd>& chr_residuals,
                Eigen::Ref<Eigen::VectorXd> neglogp_beta,
                Eigen::Ref<Eigen::VectorXd> neglogp_gam,
                Eigen::Ref<Eigen::VectorXd> neglogp_gam_robust,
                Eigen::Ref<Eigen::VectorXd> neglogp_joint,
                Eigen::Ref<Eigen::VectorXd> test_stat_beta,
                Eigen::Ref<Eigen::VectorXd> test_stat_gam,
                Eigen::Ref<Eigen::VectorXd> test_stat_gam_robust,
                Eigen::Ref<Eigen::VectorXd> test_stat_joint){
	assert(neglogp_beta.rows()  == n_var);
	assert(neglogp_gam.rows()   == n_var);
	assert(neglogp_joint.rows() == n_var);
	assert(test_stat_beta.rows()  == n_var);
	assert(test_stat_gam.rows()   == n_var);
	assert(test_stat_joint.rows() == n_var);
	assert(n_effects == 1 || n_effects == 2);

//		std::set<int> chrs(X.chromosome.begin(), X.chromosome.end());
	assert(chr_residuals.size() == n_chrs);

	// Compute p-vals per variant (p=3 as residuals mean centered)
	Eigen::MatrixXd H(n_samples, n_effects);
	boost_m::students_t t_dist(n_samples - n_effects - 1);
	boost_m::fisher_f f_dist(n_effects, n_samples - n_effects - 1);
	for(std::uint32_t jj = 0; jj < n_var; jj++ ) {
		int chr1 = X.chromosome[jj];
		int cc = std::find(chrs_present.begin(), chrs_present.end(), chr1) - chrs_present.begin();
		H.col(0) = X.col(jj).cast<double>();

		if(n_effects == 1) {
			double ztz_inv = 1.0 / H.squaredNorm();
			double tau = (H.transpose() * chr_residuals[cc])(0,0) * ztz_inv;
			double rss_null = (chr_residuals[cc] - H * tau).squaredNorm();
			// T-test of variant j
			double main_se_j    = std::sqrt(rss_null / (N - 1.0) * ztz_inv);
			double main_tstat_j = tau / main_se_j;
			double main_pval_j  = 2 * boost_m::cdf(boost_m::complement(t_dist, fabs(main_tstat_j)));

			neglogp_beta(jj) = -1 * std::log10(main_pval_j);
			test_stat_beta(jj) = main_tstat_j;
		} else if (n_effects > 1) {
			H.col(1) = H.col(0).cwiseProduct(vp.eta.cast<double>());

			// Model Fitting
			Eigen::Matrix2d HtH     = H.transpose() * H;
			Eigen::Matrix2d HtH_inv = HtH.inverse();
			Eigen::Vector2d Hty     = H.transpose() * chr_residuals[cc];
			Eigen::Vector2d tau     = HtH_inv * Hty;

			Eigen::VectorXd resid_alt = chr_residuals[cc] - H * tau;
			double rss_alt  = resid_alt.squaredNorm();
			double rss_null = chr_residuals[cc].squaredNorm();

			// Single-var tests
			double beta_tstat, gam_tstat, rgam_stat, beta_pval, gam_pval, rgam_pval;
			Eigen::Matrix2d HtVH = H.transpose() * resid_alt.cwiseProduct(resid_alt).asDiagonal() * H;
			hetero_chi_sq(HtH_inv, Hty, HtVH, 1, rgam_stat, rgam_pval);
			student_t_test(n_samples, HtH_inv, Hty, rss_alt, 1, gam_tstat, gam_pval);
			student_t_test(n_samples, HtH_inv, Hty, rss_alt, 0, beta_tstat, beta_pval);

			// // T-test on beta
			// double beta_tstat = tau[0] / sqrt(rss_alt * HtH_inv(0, 0) / (N - 3.0));
			// double beta_pval  = 2 * boost_m::cdf(boost_m::complement(t_dist, fabs(beta_tstat)));
			//
			// // T-test on gamma
			// double gam_tstat  = tau[1] / sqrt(rss_alt * HtH_inv(1, 1) / (N - 3.0));
			// double gam_pval   = 2 * boost_m::cdf(boost_m::complement(t_dist, fabs(gam_tstat)));

			// F-test over main+int effects of snp_j
			double joint_fstat, joint_pval;
			joint_fstat       = (rss_null - rss_alt) / 2.0;
			joint_fstat      /= rss_alt / (N - 3.0);
			joint_pval        = 1.0 - boost_m::cdf(f_dist, joint_fstat);

			neglogp_beta[jj]  = -1 * std::log10(beta_pval);
			neglogp_gam[jj]   = -1 * std::log10(gam_pval);
			neglogp_gam_robust[jj]   = -1 * std::log10(rgam_pval);
			neglogp_joint[jj] = -1 * std::log10(joint_pval);
			test_stat_beta[jj]  = beta_tstat;
			test_stat_gam[jj]   = gam_tstat;
			test_stat_gam_robust[jj]   = rgam_stat;
			test_stat_joint[jj] = joint_fstat;
		}
	}
}

/********** Output functions ************/
void write_trackers_to_file(const std::string& file_prefix,
                            const std::vector< VbTracker >& trackers,
                            const long& my_n_grid){
	// Stitch trackers back together if using multithreading
	// Parrallel starts swapped for multithreaded inference
	output_init(file_prefix);
	output_results(trackers, my_n_grid);
}

void output_init(const std::string& file_prefix){
	// Initialise files ready to write;

	std::string ofile       = fstream_init(outf, file_prefix, "");
	std::string ofile_map   = fstream_init(outf_map, file_prefix, "_map_snp_stats");
	std::string ofile_wmean = fstream_init(outf_wmean, file_prefix, "_weighted_mean_snp_stats");
	std::string ofile_nmean = fstream_init(outf_nmean, file_prefix, "_niave_mean_snp_stats");
	std::string ofile_map_yhat = fstream_init(outf_map_pred, file_prefix, "_map_yhat");
	std::string ofile_w = fstream_init(outf_weights, file_prefix, "_env_weights");
	std::string ofile_rescan = fstream_init(outf_rescan, file_prefix, "_map_rescan");
	std::string ofile_map_covar = fstream_init(outf_map_covar, file_prefix, "_map_covar");
	std::cout << "Writing converged hyperparameter values to " << ofile << std::endl;
	std::cout << "Writing MAP snp stats to " << ofile_map << std::endl;
	std::cout << "Writing MAP covar coefficients to " << ofile_map_covar << std::endl;
	std::cout << "Writing (weighted) average snp stats to " << ofile_wmean << std::endl;
	std::cout << "Writing (niave) average snp stats to " << ofile_nmean << std::endl;
	std::cout << "Writing yhat from map to " << ofile_map_yhat << std::endl;
	std::cout << "Writing env weights to " << ofile_w << std::endl;
	std::cout << "Writing 'rescan' p-values of MAP to " << ofile_rescan << std::endl;

	if(p.verbose) {
		std::string ofile_elbo = fstream_init(outf_elbo, file_prefix, "_elbo");
		std::cout << "Writing ELBO from each VB iteration to " << ofile_elbo << std::endl;

		std::string ofile_alpha_diff = fstream_init(outf_alpha_diff, file_prefix, "_alpha_diff");
		std::cout << "Writing max change in alpha from each VB iteration to " << ofile_alpha_diff << std::endl;
	}
}

void output_results(const std::vector<VbTracker>& trackers, const long& my_n_grid){
	// Write;
	// main output; weights logw converged_hyps counts time (currently no prior)
	// snps;
	// - map (outf_map) / elbo weighted mean (outf_wmean) / niave mean + sds (outf_nmean)
	// (verbose);
	// - elbo trajectories (inside the interim files?)
	// - hyp trajectories (inside the interim files)

	// Compute normalised weights using finite elbo
	std::vector< double > weights(my_n_grid);
	if(my_n_grid > 1) {
		for (int ii = 0; ii < my_n_grid; ii++) {
			weights[ii] = trackers[ii].logw;
		}
		normaliseLogWeights(weights);
	} else {
		weights[0] = 1;
	}

	/*** Hyps - header ***/
	outf << "weight elbo count sigma";

	for (int ee = 0; ee < n_effects; ee++) {
		outf << " pve" << ee;
		if((ee == 0 && p.mode_mog_prior_beta) || (ee == 1 && p.mode_mog_prior_gam)) {
			outf << " pve_large" << ee;
		}
		outf << " sigma" << ee;
		if((ee == 0 && p.mode_mog_prior_beta) || (ee == 1 && p.mode_mog_prior_gam)) {
			outf << " sigma_spike" << ee;
			outf << " sigma_spike_dilution" << ee;
		}
		outf << " lambda" << ee;
	}
	outf << std::endl;

	/*** Hyps - converged ***/
	for (int ii = 0; ii < my_n_grid; ii++) {
		outf << std::setprecision(8) << std::fixed;
		outf << weights[ii];
		outf << " " << trackers[ii].logw;
		outf << std::setprecision(6) << std::fixed;
		outf << " " << trackers[ii].count;
		outf << " " << trackers[ii].hyps.sigma;

		outf << std::setprecision(8) << std::fixed;
		for (int ee = 0; ee < n_effects; ee++) {

			// PVE
			outf << " " << trackers[ii].hyps.pve(ee);
			if((ee == 0 && p.mode_mog_prior_beta) || (ee == 1 && p.mode_mog_prior_gam)) {
				outf << " " << trackers[ii].hyps.pve_large(ee);
			}

			// MoG variances
			outf << std::scientific << std::setprecision(5);
			outf << " " << trackers[ii].hyps.slab_relative_var(ee);
			outf << std::setprecision(8) << std::fixed;

			if((ee == 0 && p.mode_mog_prior_beta) || (ee == 1 && p.mode_mog_prior_gam)) {
				outf << std::scientific << std::setprecision(5);
				outf << " " << trackers[ii].hyps.spike_relative_var(ee);

				outf << std::setprecision(3) << std::fixed;
				outf << " " << trackers[ii].hyps.slab_relative_var(ee) / trackers[ii].hyps.spike_relative_var(ee);
				outf << std::setprecision(8) << std::fixed;
			}

			// Lambda
			outf << " " << trackers[ii].hyps.lambda(ee);
		}
		outf << std::endl;
	}

	/*********** Stats from MAP to file ************/
	std::vector<Eigen::VectorXd> map_residuals_by_chr(n_chrs), pred_main(n_chrs), pred_int(n_chrs);
	long int ii_map = std::distance(weights.begin(), std::max_element(weights.begin(), weights.end()));
	VariationalParametersLite vp_map = trackers[ii_map].vp;
	GLOBAL_map_vp = trackers[ii_map].vp;

	// Predicted effects to file
	calcPredEffects(vp_map);
	compute_residuals_per_chr(vp_map, pred_main, pred_int, map_residuals_by_chr);
	Eigen::VectorXd Ealpha = Eigen::VectorXd::Zero(n_samples);
	if(n_covar > 0) {
		Ealpha += (C * vp_map.mean_covar().matrix().cast<scalarData>()).cast<double>();
	}
	if(n_effects == 1) {
		outf_map_pred << "Y Ealpha Xbeta";
		for(auto cc : chrs_index) {
			outf_map_pred << " residuals_excl_chr" << chrs_present[cc];
		}
		outf_map_pred << std::endl;

		for (std::uint32_t ii = 0; ii < n_samples; ii++) {
			outf_map_pred << Y(ii) << " " << Ealpha(ii) << " " << vp_map.ym(ii) - Ealpha(ii);
			for(auto cc : chrs_index) {
				outf_map_pred << " " << map_residuals_by_chr[cc](ii);
			}
			outf_map_pred << std::endl;
		}
	} else {
		outf_map_pred << "Y Ealpha Xbeta eta Xgamma";
		for(auto cc : chrs_index) {
			outf_map_pred << " residuals_excl_chr" << chrs_present[cc];
		}
		outf_map_pred << std::endl;
		for (std::uint32_t ii = 0; ii < n_samples; ii++) {
			outf_map_pred << Y(ii) << " " << Ealpha(ii) << " " << vp_map.ym(ii) - Ealpha(ii);
			outf_map_pred << " " << vp_map.eta(ii) << " " << vp_map.yx(ii);
			for(auto cc : chrs_index) {
				outf_map_pred << " " << map_residuals_by_chr[cc](ii);
			}
			outf_map_pred << std::endl;
		}
	}

	// weights to file
	if(n_effects > 1) {
		for (int ll = 0; ll < n_env; ll++) {
			outf_weights << env_names[ll];
			if (ll + 1 < n_env) outf_weights << " ";
		}
		outf_weights << std::endl;
		for (int ll = 0; ll < n_env; ll++) {
			outf_weights << vp_map.mean_weights(ll);
			if (ll + 1 < n_env) outf_weights << " ";
		}
		outf_weights << std::endl;
	}

	// Compute LOCO p-values
	Eigen::VectorXd neglogp_beta(n_var), neglogp_gam(n_var), neglogp_rgam(n_var), neglogp_joint(n_var);
	Eigen::VectorXd test_stat_beta(n_var), test_stat_gam(n_var), test_stat_rgam(n_var), test_stat_joint(n_var);
	LOCO_pvals(vp_map, map_residuals_by_chr, neglogp_beta, neglogp_gam, neglogp_rgam, neglogp_joint, test_stat_beta, test_stat_gam, test_stat_rgam, test_stat_joint);

	// MAP snp-stats to file
	write_snp_stats_to_file(outf_map, n_effects, n_var, vp_map, X, p, true, neglogp_beta, neglogp_gam, neglogp_rgam, neglogp_joint, test_stat_beta, test_stat_gam, test_stat_rgam, test_stat_joint);
	if(n_covar > 0) {
		write_covars_to_file(outf_map_covar, vp_map);
	}

	// Rescan of map
	if(n_effects > 1) {
		Eigen::VectorXd gam_neglogp(n_var);
		rescanGWAS(trackers[ii_map].vp, gam_neglogp);
		outf_rescan << "chr rsid pos a0 a1 maf info neglogp" << std::endl;
		for (std::uint32_t kk = 0; kk < n_var; kk++) {
			outf_rescan << X.chromosome[kk] << " " << X.rsid[kk] << " " << X.position[kk];
			outf_rescan << " " << X.al_0[kk] << " " << X.al_1[kk] << " ";
			outf_rescan << X.maf[kk] << " " << X.info[kk] << " " << gam_neglogp(kk);
			outf_rescan << std::endl;
		}
	}
}

void write_covars_to_file(boost_io::filtering_ostream& ofile,
                          VariationalParametersLite vp) {
	// Assumes ofile has been initialised.

	// Header
	ofile << "covar beta" << std::endl;

	ofile << std::setprecision(9) << std::fixed;
	for (int cc = 0; cc < n_covar; cc++) {
		ofile << covar_names[cc] << " " << vp.mean_covar(cc) << std::endl;
	}
}

std::string fstream_init(boost_io::filtering_ostream& my_outf,
                         const std::string& file_prefix,
                         const std::string& file_suffix){

	std::string filepath   = p.out_file;
	std::string dir        = filepath.substr(0, filepath.rfind('/')+1);
	std::string stem_w_dir = filepath.substr(0, filepath.find('.'));
	std::string stem       = stem_w_dir.substr(stem_w_dir.rfind('/')+1, stem_w_dir.size());
	std::string ext        = filepath.substr(filepath.find('.'), filepath.size());

	std::string ofile      = dir + file_prefix + stem + file_suffix + ext;

	my_outf.reset();
	std::string gz_str = ".gz";
	if (p.out_file.find(gz_str) != std::string::npos) {
		my_outf.push(boost_io::gzip_compressor());
	}
	my_outf.push(boost_io::file_sink(ofile));
	return ofile;
}

int parseLineRAM(char* line){
	// This assumes that a digit will be found and the line ends in " Kb".
	std::size_t i = strlen(line);
	const char* p = line;
	while (*p <'0' || *p > '9') p++;
	line[i-3] = '\0';
	char* s_end;
	int res = atoi(p);
	return res;
}

int getValueRAM(){         //Note: this value is in KB!
#ifndef OSX
	FILE* file = fopen("/proc/self/status", "r");
	int result = -1;
	char line[128];

	while (fgets(line, 128, file) != NULL) {
		if (strncmp(line, "VmRSS:", 6) == 0) {
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
};

#endif
