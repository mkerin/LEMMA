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
#include "typedefs.hpp"
#include "file_utils.hpp"
#include "variational_parameters.hpp"
#include "vbayes_tracker.hpp"
#include "hyps.hpp"
#include "mpi_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <limits>
#include <queue>
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

inline double get_corr(Eigen::VectorXd v1, Eigen::VectorXd v2){
	v1.array() -= v1.mean();
	v2.array() -= v2.mean();
	double res = v1.dot(v2) / v1.norm() / v2.norm();
	return res;
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

// sizes
	int n_effects;
	long n_samples;
	long n_covar;
	long n_env;
	long n_var;
	long n_var2;
	double Nglobal;
	int world_rank;
	bool first_covar_update;

// Chromosomes in data
	long n_chrs;
	std::vector<long> chrs_present;
	std::vector<long> chrs_index;

	parameters p;
	std::vector<long> fwd_pass;
	std::vector<long> back_pass;
	std::vector< std::vector <long> > main_fwd_pass_chunks, gxe_fwd_pass_chunks;
	std::vector< std::vector <long> > main_back_pass_chunks, gxe_back_pass_chunks;
	std::vector<long> env_fwd_pass, covar_fwd_pass;
	std::vector<long> env_back_pass, covar_back_pass;
	std::map<long, Eigen::MatrixXd> XtX_block_cache, ZtZ_block_cache;

// Data
	GenotypeMatrix&  X;
	EigenDataMatrix& Y;
	EigenDataArrayX Cty;
	EigenDataMatrix& E;
	EigenDataMatrix& C;
	Eigen::MatrixXd CtCRidgeInv;

	Eigen::ArrayXXd& dXtEEX_lowertri;
	std::unordered_map<long, bool> sample_is_invalid;
	std::map<long, int> sample_location;

// Global location of y_m = E[X beta] and y_x = E[X gamma]
	EigenDataMatrix YY, YX, YM, ETA, ETA_SQ;

// genome wide scan computed upstream
	Eigen::ArrayXXd& snpstats;

// Init points
	VariationalParametersLite vp_init;
	std::vector<Hyps> hyps_inits;

// Monitoring
	std::chrono::system_clock::time_point time_check;
	std::chrono::duration<double> elapsed_innerLoop;

	std::vector<Eigen::VectorXd> resid_loco, ym_per_chr, yx_per_chr;

	explicit VBayesX2(Data& dat) : X(dat.G),
		Y(dat.Y),
		C(dat.C),
		E(dat.E),
		dXtEEX_lowertri(dat.dXtEEX_lowertri),
		snpstats(dat.snpstats),
		p(dat.p),
		hyps_inits(dat.hyps_inits),
		sample_location(dat.sample_location),
		sample_is_invalid(dat.sample_is_invalid),
		vp_init(dat.vp_init){
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
#ifdef EIGEN_USE_MKL_ALL
		mkl_set_num_threads_local(p.n_thread);
#endif
		// Data size params
		n_effects      = dat.n_effects;
		n_var          = dat.n_var;
		n_env          = dat.n_env;
		n_var2         = n_effects * dat.n_var;
		n_samples      = dat.n_samples;
		n_covar        = dat.n_covar;
		covar_names    = dat.covar_names;
		env_names      = dat.env_names;
		double Nlocal  = (double) n_samples;
		Nglobal        = mpiUtils::mpiReduce_inplace(&Nlocal);
		// E = dat.E;
		first_covar_update = true;

		assert(Y.rows() == n_samples);
		assert(X.rows() == n_samples);
		assert(n_covar == C.cols());
		assert(n_covar == covar_names.size());
		assert(n_env == E.cols());
		assert(n_env == env_names.size());

		std::set<int> tmp(X.chromosome.begin(), X.chromosome.end());
		chrs_present.assign(tmp.begin(), tmp.end());
		n_chrs = chrs_present.size();
		for (int cc = 0; cc < n_chrs; cc++) {
			chrs_index.push_back(cc);
		}

		p.main_chunk_size = (unsigned int) std::min((long int) p.main_chunk_size, (long int) n_var);
		p.gxe_chunk_size = (unsigned int) std::min((long int) p.gxe_chunk_size, (long int) n_var);

		// When n_env > 1 this gets set when in updateEnvWeights
		if(n_env == 0) {
			for (auto &hyps : hyps_inits) {
				hyps.s_x << n_var;
			}
		} else {
			for (auto &hyps : hyps_inits) {
				hyps.s_x << n_var, n_var;
			}
		}

		// Allocate memory - fwd/back pass vectors
		for(long kk = 0; kk < n_var * n_effects; kk++) {
			fwd_pass.push_back(kk);
			back_pass.push_back(n_var2 - kk - 1);
		}

		for(long ll = 0; ll < n_env; ll++) {
			env_fwd_pass.push_back(ll);
			env_back_pass.push_back(n_env - ll - 1);
		}

		for(long ll = 0; ll < n_covar; ll++) {
			covar_fwd_pass.push_back(ll);
			covar_back_pass.push_back(n_covar - ll - 1);
		}

		// ceiling of n_var / chunk size
		long n_main_segs, n_gxe_segs;
		n_main_segs = (n_var + p.main_chunk_size - 1) / p.main_chunk_size;
		n_gxe_segs = (n_var + p.gxe_chunk_size - 1) / p.gxe_chunk_size;

		main_fwd_pass_chunks.resize(n_main_segs);
		main_back_pass_chunks.resize(n_main_segs);
		gxe_fwd_pass_chunks.resize(n_gxe_segs);
		gxe_back_pass_chunks.resize(n_gxe_segs);
		for(long kk = 0; kk < n_var; kk++) {
			long main_ch_index = kk / p.main_chunk_size;
			long gxe_ch_index = kk / p.gxe_chunk_size;
			main_fwd_pass_chunks[main_ch_index].push_back(kk);
			main_back_pass_chunks[n_main_segs - 1 - main_ch_index].push_back(kk);
			gxe_fwd_pass_chunks[gxe_ch_index].push_back(kk + n_var);
			gxe_back_pass_chunks[n_gxe_segs - 1 - gxe_ch_index].push_back(kk + n_var);
		}

		for (long ii = 0; ii < n_main_segs; ii++) {
			std::reverse(main_back_pass_chunks[ii].begin(), main_back_pass_chunks[ii].end());
		}
		for (long ii = 0; ii < n_gxe_segs; ii++) {
			std::reverse(gxe_back_pass_chunks[ii].begin(), gxe_back_pass_chunks[ii].end());
		}

		if(n_effects == 1) {
			gxe_back_pass_chunks.clear();
			gxe_fwd_pass_chunks.clear();
		}

		// Generate initial values for each run
		if(p.mode_random_start) {
			std::cout << "Beta and gamma initialised with random draws" << std::endl;
		}

		// Initialise summary vars for vp_init
		if(n_env > 0) {
			vp_init.eta     = E * vp_init.muw.matrix().cast<scalarData>();
			vp_init.eta_sq  = vp_init.eta.array().square().matrix();
			vp_init.eta_sq += E.cwiseProduct(E) * vp_init.sw_sq.matrix().template cast<scalarData>();
		}
		calcPredEffects(vp_init);

		// Cache Cty
		if(n_covar > 0) {
			if (p.debug) std::cout << "Caching Cty" << std::endl;
			EigenDataArrayX Ctylocal;
			Ctylocal = C.transpose() * Y;
			Cty.resize(Ctylocal.rows(), Ctylocal.cols());
			mpiUtils::mpiReduce_double(Ctylocal.data(), Cty.data(), Ctylocal.size());
		}

		// Update main effects
		cache_local_ldblocks(main_fwd_pass_chunks, true);
		cache_local_ldblocks(main_back_pass_chunks, false);
	}

	void cache_local_ldblocks(std::vector<std::vector<long> >iter_chunks, bool is_fwd_pass){
		EigenDataMatrix D;
		for (std::uint32_t ch = 0; ch < iter_chunks.size(); ch++) {
			std::vector<long> chunk = iter_chunks[ch];
			int ee = chunk[0] / n_var;
			long ch_len = chunk.size();
			if (D.cols() != ch_len) {
				D.resize(n_samples, ch_len);
			}
			X.col_block3(chunk, D);

			unsigned long memoize_id = ((is_fwd_pass) ? ch : ch + iter_chunks.size());
			if (XtX_block_cache.count(memoize_id) == 0) {
				if (p.n_thread == 1) {
					Eigen::MatrixXd D_corr(ch_len, ch_len);
					D_corr.triangularView<Eigen::StrictlyUpper>() = (D.transpose() * D).template cast<double>();
					XtX_block_cache[memoize_id] = D_corr;
				} else {
					XtX_block_cache[memoize_id] = (D.transpose() * D).template cast<double>();
				}
				XtX_block_cache[memoize_id] = mpiUtils::mpiReduce_inplace(XtX_block_cache[memoize_id]);
			}
		}
	}

	void run(){
		std::cout << std::endl << "Starting variational inference";
#ifndef OSX
		long long kbMax, kbGlobal, kbLocal = fileUtils::getValueRAM();
		MPI_Allreduce(&kbLocal, &kbMax, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
		MPI_Allreduce(&kbLocal, &kbGlobal, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		double gbGlobal = kbGlobal / 1000.0 / 1000.0;
		double gbMax = kbMax / 1000.0 / 1000.0;
		if(world_rank == 0) {
			printf(" (RAM usage: %.2f GB in total; max of %.2f GB per rank)", gbGlobal, gbMax);
		}
#endif
		std::cout << std::endl;

		time_check = std::chrono::system_clock::now();
		long n_grid = hyps_inits.size();
		std::vector< VbTracker > trackers(n_grid, p);
		run_inference(hyps_inits, false, 2, trackers);
		write_converged_hyperparams_to_file("", trackers, n_grid);
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

		// Set run with best ELBO to vp_init
		std::vector< double > weights(n_grid);
		if(n_grid > 1) {
			for (int ii = 0; ii <n_grid; ii++) {
				weights[ii] = trackers[ii].logw;
			}
			normaliseLogWeights(weights);
		} else {
			weights[0] = 1;
		}

		long ii_map = std::distance(weights.begin(), std::max_element(weights.begin(), weights.end()));
		vp_init = trackers[ii_map].vp;

		// Compute residual phenotypes
		compute_residuals_per_chr(vp_init, ym_per_chr, yx_per_chr, resid_loco);
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

		long max_count = 0;
		for (const auto& tracker : all_tracker) {
			max_count = std::max(max_count, tracker.count);
		}
		std::cout << "Variational inference finished after " << max_count;
		std::cout << " iterations and " << elapsed_innerLoop.count() << " seconds.";
		std::cout << std::endl << std::endl;
	}

	void runInnerLoop(const bool random_init,
	                  const int round_index,
	                  std::vector<Hyps>& all_hyps,
	                  std::vector<VbTracker>& all_tracker){
		// minimise KL Divergence and assign elbo estimate
		if (p.xtra_verbose) std::cout << "Starting inner loop" << std::endl;
		int print_interval = 25;
		if(random_init) {
			throw std::logic_error("Random start depreciated.");
		}
		unsigned long n_grid = all_hyps.size();

		std::vector<VariationalParameters> all_vp;
		setup_variational_params(all_hyps, all_vp);

		// Run inner loop until convergence
		std::vector<int> converged(n_grid, 0);
		bool all_converged = false;
		std::vector<Eigen::ArrayXd> w_prev(n_grid), beta_prev(n_grid), gam_prev(n_grid), covar_prev(n_grid);
		std::vector<double> i_logw(n_grid, -1*std::numeric_limits<double>::max());

		for (int nn = 0; nn < n_grid; nn++) {
			if(world_rank == 0) {
				all_tracker[nn].init_interim_output(nn, round_index, n_effects, n_covar, n_env, env_names, all_vp[nn]);
			}
		}

		// SQUAREM objects
		std::vector<Hyps> theta0 = all_hyps;
		std::vector<Hyps> theta1 = all_hyps;
		std::vector<Hyps> theta2 = all_hyps;

		// Allow more flexible start point so that we can resume previous inference run
		long count = p.vb_iter_start;
		while(!all_converged && count < p.vb_iter_max) {
			if (p.debug) std::cout << "Iter count: " << count << std::endl;
			for (int nn = 0; nn < n_grid; nn++) {
				if(n_covar > 0) covar_prev[nn] = all_vp[nn].mean_covars().array();
				beta_prev[nn] = all_vp[nn].mean_beta().array();
				if(n_env > 0) gam_prev[nn] = all_vp[nn].mean_gam().array();
				if(n_env > 1) w_prev[nn] = all_vp[nn].mean_weights().array();
			}
			std::vector<double> logw_prev = i_logw;

			if (p.debug) std::cout << " - update params" << std::endl;
			updateAllParams(count, round_index, all_vp, all_hyps, logw_prev);

			// SQUAREM
			if (p.mode_squarem) {
				if (p.debug) std::cout << " - SQUAREM accelerator" << std::endl;
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
						while(!theta.domain_is_valid() || std::abs(theta.sigma - theta2[nn].sigma) >= 0.05) {
							step = std::min(step * 0.5, -1.0);
							theta = theta0[nn] - 2 * step * rr + step * step * vv;
						}

						// Copy s_x and pve from theta2
						theta.s_x = theta2[nn].s_x;
						theta.pve = theta2[nn].pve;
						theta.n_effects = theta2[nn].n_effects;
						theta.pve_large = theta2[nn].pve_large;

						all_hyps[nn] = theta;
					}
				}
			}

			// update elbo
			std::vector<double> alpha_diff(n_grid), beta_diff(n_grid), gam_diff(n_grid), covar_diff(n_grid), w_diff(n_grid);;
			for (int nn = 0; nn < n_grid; nn++) {
				i_logw[nn]     = calc_logw(all_hyps[nn], all_vp[nn]);
				if(n_covar > 0) covar_diff[nn] = (covar_prev[nn] - all_vp[nn].mean_covars().array()).abs().maxCoeff();
				beta_diff[nn] = (beta_prev[nn] - all_vp[nn].mean_beta().array()).abs().maxCoeff();
				if(n_env > 0) gam_diff[nn] = (gam_prev[nn] - all_vp[nn].mean_gam().array()).abs().maxCoeff();
				if(n_env > 1) w_diff[nn] = (w_prev[nn] - all_vp[nn].mean_weights().array()).abs().maxCoeff();
			}

			// Interim output
			for (int nn = 0; nn < n_grid; nn++) {
				all_hyps[nn].update_pve();
				if(world_rank == 0) {
					all_tracker[nn].push_interim_hyps(count, all_hyps[nn], i_logw[nn],
					                                  covar_diff[nn], beta_diff[nn], gam_diff[nn], w_diff[nn],
					                                  n_effects,
					                                  n_var, n_covar, n_env, all_vp[nn]);
				}
				if (p.param_dump_interval > 0 && count % p.param_dump_interval == 0) {
					all_tracker[nn].dump_state(std::to_string(count), n_samples, n_covar, n_var,
					                           n_env, n_effects,
					                           all_vp[nn], all_hyps[nn], Y, C,
					                           X, covar_names, env_names, sample_is_invalid,
					                           sample_location);
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
			}
			if (std::all_of(converged.begin(), converged.end(), [](int i){
				return i == 1;
			})) {
				all_converged = true;
			}
			count++;
			for (int nn = 0; nn < n_grid; nn++) {
				if(!converged[nn]) {
					all_tracker[nn].count_to_convergence++;
				}
			}

			// Report progress to std::cout
			if((count + 1) % print_interval == 0) {
				int n_converged = 0;
				for (auto& n : converged) {
					n_converged += n;
				}

				std::cout << "Completed " << count+1 << " iterations ";
				print_time_check();
			}
		}

		if(any_of(i_logw.begin(), i_logw.end(), [](double x) {
			return !std::isfinite(x);
		})) {
			std::cout << "WARNING: non-finite elbo estimate produced" << std::endl;
		}

		// Dump converged state
		if (p.xtra_verbose) std::cout << "Dumping converged params" << std::endl;
		for (int nn = 0; nn < n_grid; nn++) {
			all_tracker[nn].dump_state("_converged", n_samples, n_covar, n_var,
			                           n_env, n_effects,
			                           all_vp[nn], all_hyps[nn], Y, C,
			                           X, covar_names, env_names, sample_is_invalid, sample_location);
		}

		// Log all things that we want to track
		for (int nn = 0; nn < n_grid; nn++) {
			all_tracker[nn].logw = i_logw[nn];
			all_tracker[nn].count = count;
			all_tracker[nn].vp = all_vp[nn].convert_to_lite();
			all_tracker[nn].hyps = all_hyps[nn];
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
				vp.calcEdZtZ(dXtEEX_lowertri, n_env);
			}
			all_vp.push_back(vp);
		}
	}

/********** VB update functions ************/
	void updateAllParams(const long& count,
	                     const int& round_index,
	                     std::vector<VariationalParameters>& all_vp,
	                     std::vector<Hyps>& all_hyps,
	                     std::vector<double> logw_prev){
		std::vector< std::uint32_t > iter;
		std::vector< std::vector< std::uint32_t > > iter_chunks;
		unsigned long n_grid = all_hyps.size();
		std::vector<double> i_logw(n_grid);

		// Update covar main effects
		for (int nn = 0; nn < n_grid; nn++) {
			if (n_covar > 0) {
				updateCovarEffects(covar_fwd_pass, all_vp[nn], all_hyps[nn]);
				updateCovarEffects(covar_back_pass, all_vp[nn], all_hyps[nn]);
				check_monotonic_elbo(all_hyps[nn], all_vp[nn], count, logw_prev[nn], "updateCovarEffects");
			}
		}

		// Update main & interaction effects
		std::string ms;
		bool is_fwd_pass = (count % 2 == 0);
		if(is_fwd_pass) {
			ms = "updateAlphaMu_fwd_main";
			updateAlphaMu(main_fwd_pass_chunks, all_hyps, all_vp, is_fwd_pass, logw_prev, count);
		} else {
			ms = "updateAlphaMu_back_gxe";
			updateAlphaMu(gxe_back_pass_chunks, all_hyps, all_vp, is_fwd_pass, logw_prev, count);
		}
		for (int nn = 0; nn < n_grid; nn++) {
			check_monotonic_elbo(all_hyps[nn], all_vp[nn], count, logw_prev[nn], ms);
		}

		if(is_fwd_pass) {
			ms = "updateAlphaMu_fwd_gxe";
			updateAlphaMu(gxe_fwd_pass_chunks, all_hyps, all_vp, is_fwd_pass, logw_prev, count);
		} else {
			ms = "updateAlphaMu_back_main";
			updateAlphaMu(main_back_pass_chunks, all_hyps, all_vp, is_fwd_pass, logw_prev, count);
		}
		for (int nn = 0; nn < n_grid; nn++) {
			check_monotonic_elbo(all_hyps[nn], all_vp[nn], count, logw_prev[nn], ms);
		}

		// Update env-weights
		if (p.debug) std::cout << " - update env weights" << std::endl;
		if (n_effects > 1 && n_env > 1) {
			for (int nn = 0; nn < n_grid; nn++) {
				for (int uu = 0; uu < p.env_update_repeats; uu++) {
					updateEnvWeights(env_fwd_pass, all_hyps[nn], all_vp[nn]);
					updateEnvWeights(env_back_pass, all_hyps[nn], all_vp[nn]);
				}
				check_monotonic_elbo(all_hyps[nn], all_vp[nn], count, logw_prev[nn], "updateEnvWeights");
			}
		}

		// Maximise hyps
		if (round_index > 1 && p.mode_empirical_bayes) {
			if (p.debug) std::cout << " - max hyps" << std::endl;
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

	void updateCovarEffects(const std::vector<long>& iter,
	                        VariationalParameters& vp,
	                        const Hyps& hyps){
		//
		if(p.joint_covar_update) {
			if(first_covar_update) {
				if(p.xtra_verbose) {
					std::cout << "Performing first VB update of covar effects" << std::endl;
				}
				Eigen::MatrixXd CtCRidge = C.transpose() * C;
				CtCRidge = mpiUtils::mpiReduce_inplace(CtCRidge);
				CtCRidge += Eigen::MatrixXd::Identity(n_covar, n_covar) / sigma_c;
				CtCRidgeInv = CtCRidge.inverse();
				first_covar_update = false;
			}

			Eigen::VectorXd rr_k = vp.muc;
			auto Calpha = C * vp.muc.matrix();
			Eigen::MatrixXd A = C.transpose() * (Y - (vp.ym + vp.yx.cwiseProduct(vp.eta)) + Calpha);
			A = mpiUtils::mpiReduce_inplace(A);
			Eigen::VectorXd muc = CtCRidgeInv * A;
			for (int cc = 0; cc < n_covar; cc++) {
				vp.sc_sq(cc) = hyps.sigma * CtCRidgeInv(cc, cc);
				vp.muc(cc) = muc(cc);
			}
			vp.ym += C * (vp.muc.matrix() - rr_k);
		} else {
			for (const auto& cc : iter) {
				double rr_k = vp.muc(cc);

				// Update s_sq
				vp.sc_sq(cc) = hyps.sigma * sigma_c / (sigma_c * (Nglobal - 1.0) + 1.0);

				// Update mu
				double Alocal = (vp.ym + vp.yx.cwiseProduct(vp.eta)).dot(C.col(cc));
				auto A = Cty(cc) - mpiUtils::mpiReduce_inplace(&Alocal);
				vp.muc(cc) = vp.sc_sq(cc) * ( (double) A + rr_k * (Nglobal - 1.0)) / hyps.sigma;

				// Update predicted effects
				double rr_k_diff     = vp.muc(cc) - rr_k;
				vp.ym += rr_k_diff * C.col(cc);
			}
		}
	}

	void updateAlphaMu(const std::vector< std::vector<long> >& iter_chunks,
	                   const std::vector<Hyps>& all_hyps,
	                   std::vector<VariationalParameters>& all_vp,
	                   const bool& is_fwd_pass,
	                   std::vector<double> logw_prev,
	                   const long& count){
		// Divide updates into chunks
		// Partition chunks amongst available threads
		unsigned long n_grid = all_hyps.size();
		EigenDataMatrix D;
		// snp_batch x n_grid
		Eigen::MatrixXd AA;
		// snp_batch x n_grid
		Eigen::MatrixXd rr_diff;

		for (std::uint32_t ch = 0; ch < iter_chunks.size(); ch++) {
			std::vector<long> chunk = iter_chunks[ch];
			int ee                 = chunk[0] / n_var;
			long ch_len   = chunk.size();

			// D is n_samples x snp_batch
			if(D.cols() != ch_len) {
				D.resize(n_samples, ch_len);
			}
			if(rr_diff.rows() != ch_len) {
				rr_diff.resize(ch_len, n_grid);
			}
			if(AA.rows() != ch_len) {
				AA.resize(ch_len, n_grid);
			}
			X.col_block3(chunk, D);

			// Most work done here
			// AA is snp_batch x n_grid
			AA = computeGeneResidualCorrelation(D, ee);

			// Update parameters based on AA
			for (int nn = 0; nn < n_grid; nn++) {
				Eigen::Ref<Eigen::VectorXd> A = AA.col(nn);

				unsigned long memoize_id = ((is_fwd_pass) ? ch : ch + iter_chunks.size());
				adjustParams(nn, memoize_id, chunk, D, A, all_hyps, all_vp, rr_diff);
			}

			// Update residuals
			if(ee == 0) {
				YM.noalias() += D * rr_diff.cast<scalarData>();
			} else {
				YX.noalias() += D * rr_diff.cast<scalarData>();
			}

			if(p.debug) {
				if(ee == 0) {
					for (int nn = 0; nn < n_grid; nn++) {
						check_monotonic_elbo(all_hyps[nn], all_vp[nn], count, logw_prev[nn], "updateAlphaMu_main_internal");
					}
				} else {
					for (int nn = 0; nn < n_grid; nn++) {
						check_monotonic_elbo(all_hyps[nn], all_vp[nn], count, logw_prev[nn], "updateAlphaMu_gxe_internal");
					}
				}
			}
		}
	}

	void adjustParams(const int& nn, const unsigned long& memoize_id,
	                  const std::vector<long>& chunk,
	                  const EigenDataMatrix& D,
	                  const Eigen::Ref<const Eigen::VectorXd>& A,
	                  const std::vector<Hyps>& all_hyps,
	                  std::vector<VariationalParameters>& all_vp,
	                  Eigen::Ref<Eigen::MatrixXd> rr_diff){

		int ee                 = chunk[0] / n_var;
		unsigned long ch_len   = chunk.size();
		Eigen::MatrixXd Dlocal(ch_len, ch_len), Dglobal(ch_len, ch_len);
		if (ee == 0) {
			if (XtX_block_cache.count(memoize_id) == 0) {
				if(p.n_thread == 1) {
					Dlocal.triangularView<Eigen::StrictlyUpper>() = (D.transpose() * D).template cast<double>();
				} else {
					Dlocal = (D.transpose() * D).template cast<double>();
				}
				MPI_Allreduce(Dlocal.data(), Dglobal.data(), Dlocal.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
				XtX_block_cache[memoize_id] = Dglobal;
			}
			_internal_updateAlphaMu_beta(chunk, A, XtX_block_cache[memoize_id], all_hyps[nn], all_vp[nn], rr_diff.col(nn));
		} else {
			if(p.gxe_chunk_size > 1) {
				auto it = ZtZ_block_cache.find(memoize_id);
				if (n_env == 1 && it != ZtZ_block_cache.end()) {
					Dglobal = ZtZ_block_cache[memoize_id];
				} else if(p.n_thread == 1) {
					Dlocal.triangularView<Eigen::StrictlyUpper>() = (D.transpose() * all_vp[nn].eta_sq.asDiagonal() * D).template cast<double>();
					MPI_Allreduce(Dlocal.data(), Dglobal.data(), Dlocal.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
				} else {
					Dlocal = (D.transpose() * all_vp[nn].eta_sq.asDiagonal() * D).template cast<double>();
					MPI_Allreduce(Dlocal.data(), Dglobal.data(), Dlocal.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
				}

				if(n_env == 1 && it == ZtZ_block_cache.end()) {
					ZtZ_block_cache[memoize_id] = Dglobal;
				}
			}
			_internal_updateAlphaMu_gam(chunk, A, Dglobal, all_hyps[nn], all_vp[nn], rr_diff.col(nn));
		}
	}

	template <typename EigenMat>
	Eigen::MatrixXd computeGeneResidualCorrelation(const EigenMat& D,
	                                               const int& ee){
		// Most work done here
		// variant correlations with residuals
		EigenMat resLocal;
		if(n_effects == 1) {
			// Main effects update in main effects only model
			resLocal.noalias() = (YY - YM).transpose() * D;
			resLocal.transposeInPlace();
		} else if (ee == 0) {
			// Main effects update in interaction model
			resLocal.noalias() = (YY - YM - YX.cwiseProduct(ETA)).transpose() * D;
			resLocal.transposeInPlace();
		} else {
			// Interaction effects
			resLocal.noalias() = D.transpose() * ((YY - YM).cwiseProduct(ETA) - YX.cwiseProduct(ETA_SQ));
		}
		EigenMat resGlobal(resLocal.rows(), resLocal.cols());
		// long P = resLocal.rows() * resLocal.cols();
		mpiUtils::mpiReduce_double(resLocal.data(), resGlobal.data(), resLocal.size());
		return(resGlobal.template cast<double>());
	}

	void _internal_updateAlphaMu_beta(const std::vector<long>& iter_chunk,
	                                  const Eigen::Ref<const Eigen::VectorXd>& A,
	                                  const Eigen::Ref<const Eigen::MatrixXd>& D_corr,
	                                  const Hyps& hyps,
	                                  VariationalParameters& vp,
	                                  Eigen::Ref<Eigen::MatrixXd> rr_k_diff){

		unsigned long ch_len = iter_chunk.size();
		int ee = 0;

		Eigen::ArrayXd alpha_cnst;
		if(p.mode_mog_prior_beta) {
			alpha_cnst  = (hyps.lambda / (1.0 - hyps.lambda) + eps).log();
			alpha_cnst -= (hyps.slab_var.log() - hyps.spike_var.log()) / 2.0;
		} else {
			alpha_cnst = (hyps.lambda / (1.0 - hyps.lambda) + eps).log() - hyps.slab_var.log() / 2.0;
		}

		// adjust updates within chunk
		Eigen::VectorXd rr_k(ch_len);
		assert(rr_k_diff.rows() == ch_len);
		for (int ii = 0; ii < ch_len; ii++) {
			long jj = iter_chunk[ii];

			// Log prev value
			rr_k(ii) = vp.mean_beta(jj);

			// Update s_sq
			vp.s1_beta_sq(jj)                        = hyps.slab_var(ee);
			vp.s1_beta_sq(jj)                       /= (hyps.slab_relative_var(ee) * (Nglobal-1) + 1);
			if(p.mode_mog_prior_beta) vp.s2_beta_sq(jj)  = hyps.spike_var(ee);
			if(p.mode_mog_prior_beta) vp.s2_beta_sq(jj) /= (hyps.spike_relative_var(ee) * (Nglobal-1) + 1);

			// Update mu
			double offset = rr_k(ii) * (Nglobal-1.0);
			for (int mm = 0; mm < ii; mm++) {
				offset -= rr_k_diff(mm, 0) * D_corr(mm, ii);
			}
			double AA = A(ii) + offset;
			vp.mu1_beta(jj)                            = vp.s1_beta_sq(jj) * AA / hyps.sigma;
			if (p.mode_mog_prior_beta) vp.mu2_beta(jj) = vp.s2_beta_sq(jj) * AA / hyps.sigma;

			// Update alpha
			double ff_k;
			ff_k                        = vp.mu1_beta(jj) * vp.mu1_beta(jj) / vp.s1_beta_sq(jj);
			ff_k                       += std::log(vp.s1_beta_sq(jj));
			if (p.mode_mog_prior_beta) ff_k -= vp.mu2_beta(jj) * vp.mu2_beta(jj) / vp.s2_beta_sq(jj);
			if (p.mode_mog_prior_beta) ff_k -= std::log(vp.s2_beta_sq(jj));
			vp.alpha_beta(jj)           = sigmoid(ff_k / 2.0 + alpha_cnst(ee));

			rr_k_diff(ii, 0) = vp.mean_beta(jj) - rr_k(ii);

			check_nan(vp.alpha_beta(jj), ff_k, offset, hyps, iter_chunk[ii], rr_k_diff, A, D_corr, vp, alpha_cnst);
		}
	}

	void check_nan(const double& alpha,
	               const double& ff_k,
	               const double& offset,
	               const Hyps& hyps,
	               const long& ii,
	               const Eigen::Ref<const Eigen::MatrixXd>& rr_k_diff,
	               const Eigen::Ref<const Eigen::VectorXd>& A,
	               const Eigen::Ref<const Eigen::MatrixXd>& D_corr,
	               VariationalParameters& vp,
	               const Eigen::Ref<const Eigen::ArrayXd>& alpha_cnst){
		// check for NaNs and spit out diagnostics if so.

		if(std::isnan(alpha)) {
			// TODO: print diagnostics to cout
			// TODO: write all snpstats to file
			std::cout << "NaN detected at SNP index: (";
			std::cout << ii % n_var << ", " << ii / n_var << ")" << std::endl;
			std::cout << "alpha_cnst" << std::endl << alpha_cnst << std::endl << std::endl;
			std::cout << "offset" << std::endl << offset << std::endl << std::endl;
			std::cout << "hyps" << std::endl << hyps << std::endl << std::endl;
			std::cout << "rr_k_diff" << std::endl << rr_k_diff << std::endl << std::endl;
			std::cout << "A" << std::endl << A << std::endl << std::endl;
			std::cout << "D_corr" << std::endl << D_corr << std::endl << std::endl;
			throw std::runtime_error("NaN detected");
		}
	}

	void _internal_updateAlphaMu_gam(const std::vector<long>& iter_chunk,
	                                 const Eigen::Ref<const Eigen::VectorXd>& A,
	                                 const Eigen::Ref<const Eigen::MatrixXd>& D_corr,
	                                 const Hyps& hyps,
	                                 VariationalParameters& vp,
	                                 Eigen::Ref<Eigen::MatrixXd> rr_k_diff){

		long ch_len = iter_chunk.size();
		int ee     = 1;

		Eigen::ArrayXd alpha_cnst;
		if(p.mode_mog_prior_gam) {
			alpha_cnst  = (hyps.lambda / (1.0 - hyps.lambda) + eps).log();
			alpha_cnst -= (hyps.slab_var.log() - hyps.spike_var.log()) / 2.0;
		} else {
			alpha_cnst = (hyps.lambda / (1.0 - hyps.lambda) + eps).log() - hyps.slab_var.log() / 2.0;
		}

		// adjust updates within chunk
		// Need to be able to go backwards during a back_pass
		Eigen::VectorXd rr_k(ch_len);
		assert(rr_k_diff.rows() == ch_len);
		for (int ii = 0; ii < ch_len; ii++) {
			// variant index
			long jj = (iter_chunk[ii] % n_var);

			// Log prev value
			rr_k(ii) = vp.mean_gam(jj);

			// Update s_sq
			double tmp = (p.gxe_chunk_size > 1 && p.n_thread == 1) ? vp.EdZtZ(jj) : D_corr(ii, ii);
			vp.s1_gam_sq(jj)                        = hyps.slab_var(ee);
			vp.s1_gam_sq(jj)                       /= (hyps.slab_relative_var(ee) * vp.EdZtZ(jj) + 1);
			if(p.mode_mog_prior_gam) vp.s2_gam_sq(jj)  = hyps.spike_var(ee);
			if(p.mode_mog_prior_gam) vp.s2_gam_sq(jj) /= (hyps.spike_relative_var(ee) * vp.EdZtZ(jj) + 1);

			// Update mu
			double offset = rr_k(ii) * vp.EdZtZ(jj);
			for (int mm = 0; mm < ii; mm++) {
				offset -= rr_k_diff(mm, 0) * D_corr(mm, ii);
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

			rr_k_diff(ii, 0) = vp.mean_gam(jj) - rr_k(ii);

			check_nan(vp.alpha_gam(jj), ff_k, offset, hyps, iter_chunk[ii], rr_k_diff, A, D_corr, vp, alpha_cnst);
		}
	}

	void maximiseHyps(Hyps& hyps,
	                  const VariationalParameters& vp){

		// max sigma
		hyps.sigma  = calcExpLinear(hyps, vp);
		if (n_covar > 0) {
			hyps.sigma += (vp.sc_sq + vp.muc.square()).sum() / sigma_c;
			hyps.sigma /= (Nglobal + (double) n_covar);
		} else {
			hyps.sigma /= Nglobal;
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

	void updateEnvWeights(const std::vector<long>& iter,
	                      Hyps& hyps,
	                      VariationalParameters& vp){

		Eigen::ArrayXd varG = vp.var_gam();
		for (int ll : iter) {
			// Log previous mean weight
			double r_ll = vp.muw(ll);

			// Update s_sq
			double denom;
			denom  = vp.yx.cwiseProduct(E.col(ll)).squaredNorm();
			if(world_rank == 0) {
				denom += (varG * dXtEEX_lowertri.col(dXtEEX_col_ind(ll, ll, n_env))).sum();
			}
			denom  = mpiUtils::mpiReduce_inplace(&denom);
			denom += hyps.sigma;
			vp.sw_sq(ll) = hyps.sigma / denom;

			// Remove dependance on current weight
			vp.eta -= (r_ll * E.col(ll)).matrix();

			// Update mu
			Eigen::ArrayXd env_vars = Eigen::ArrayXd::Zero(n_var);
			if(world_rank == 0) {
				for (int mm = 0; mm < n_env; mm++) {
					if(mm != ll) {
						env_vars += vp.muw(mm) * dXtEEX_lowertri.col(dXtEEX_col_ind(ll, mm, n_env));
					}
				}
				// MPI_Bcast(env_vars.data(), env_vars.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
			}
			env_vars  = mpiUtils::mpiReduce_inplace(env_vars);

			double eff = ((Y - vp.ym).array() * E.col(ll).array() * vp.yx.array()).sum();
			eff       -= (vp.yx.array() * E.col(ll).array() * vp.eta.array() * vp.yx.array()).sum();
			eff        = mpiUtils::mpiReduce_inplace(&eff);
			eff       -= (varG * env_vars).sum();
			vp.muw(ll) = vp.sw_sq(ll) * eff / hyps.sigma;

			// Update eta
			vp.eta += (vp.muw(ll) * E.col(ll)).matrix();
		}

		// Recompute eta_sq
		vp.eta_sq  = vp.eta.array().square().matrix();
#ifdef DATA_AS_FLOAT
		vp.eta_sq += E.cwiseProduct(E) * vp.sw_sq.matrix().cast<float>();
#else
		vp.eta_sq += E.cwiseProduct(E) * vp.sw_sq.matrix();
#endif

		// Recompute expected value of diagonal of ZtZ
		vp.calcEdZtZ(dXtEEX_lowertri, n_env);

		double colVarZ = 0;
		if(world_rank == 0) {
			Eigen::ArrayXd muw_sq_combos(n_env * (n_env + 1) / 2);
			for (int ll = 0; ll < n_env; ll++) {
				for (int mm = 0; mm < n_env; mm++) {
					muw_sq_combos(dXtEEX_col_ind(ll, mm, n_env)) = vp.muw(mm) * vp.muw(ll);
				}
			}

			colVarZ = 2 * (dXtEEX_lowertri.rowwise() * muw_sq_combos.transpose()).sum();
			for (int ll = 0; ll < n_env; ll++) {
				colVarZ -= (vp.muw(ll) * vp.muw(ll) * dXtEEX_lowertri.col(dXtEEX_col_ind(ll, ll, n_env))).sum();
			}
			colVarZ /= (Nglobal - 1.0);
		}
		colVarZ = mpiUtils::mpiReduce_inplace(&colVarZ);

		// WARNING: Hard coded index
		// WARNING: Updates S_x in hyps
		hyps.s_x(0) = (double) n_var;
		hyps.s_x(1) = colVarZ;
	}

	double calc_logw(const Hyps& hyps,
	                 const VariationalParameters& vp){

		// Expectation of linear regression log-likelihood
		double int_linear = -1.0 * calcExpLinear(hyps, vp) / 2.0 / hyps.sigma;
		int_linear -= Nglobal * std::log(2.0 * PI * hyps.sigma) / 2.0;

		// kl-beta
		double kl_beta = calcKLBeta(hyps, vp);

		double kl_gamma = 0;
		if(n_effects > 1) {
			kl_gamma += calcKLGamma(hyps, vp);
		}

		// covariates
		double kl_covar = 0.0;
		if(n_covar > 0) {
			kl_covar += (double) n_covar * (1.0 - std::log(hyps.sigma * sigma_c)) / 2.0;
			kl_covar += vp.sc_sq.log().sum() / 2.0;
			kl_covar -= vp.sc_sq.sum() / 2.0 / hyps.sigma / sigma_c;
			kl_covar -= vp.muc.square().sum() / 2.0 / hyps.sigma / sigma_c;
		}

		// weights
		double kl_weights = 0.0;
		if(n_effects > 1 && n_env > 1) {
			kl_weights += (double) n_env / 2.0;
			kl_weights += vp.sw_sq.log().sum() / 2.0;
			kl_weights -= vp.sw_sq.sum() / 2.0;
			kl_weights -= vp.muw.square().sum() / 2.0;
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
		std::cout << fileUtils::getValueRAM() << "KB)" << std::endl;
		time_check = now;
	}

	void initRandomAlphaMu(VariationalParameters& vp){
		// vp.alpha a uniform simplex, vp.mu standard gaussian
		// Also sets predicted effects
		std::default_random_engine gen_gauss(p.random_seed), gen_unif(p.random_seed);
		std::normal_distribution<double> gaussian(0.0,1.0);
		std::uniform_real_distribution<double> uniform(0.0,1.0);

		// Beta
		vp.mu1_beta.resize(n_var);
		vp.alpha_beta.resize(n_var);
		if(p.mode_mog_prior_beta) {
			vp.mu2_beta = Eigen::ArrayXd::Zero(n_var);
		}

		for (std::uint32_t kk = 0; kk < n_var; kk++) {
			vp.alpha_beta(kk) = uniform(gen_unif);
			vp.mu1_beta(kk)    = gaussian(gen_gauss);
		}
		vp.alpha_beta /= vp.alpha_beta.sum();

		// Gamma
		if(n_effects > 1) {
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

		if(n_covar > 0) {
			vp.muc = Eigen::ArrayXd::Zero(n_covar);
		}

		// Gen predicted effects.
		calcPredEffects(vp);

		// Env weights - cast if DATA_AS_FLOAT
		vp.muw     = 1.0 / (double) n_env;
		vp.eta     = E * vp.muw.matrix().cast<scalarData>();
		vp.eta_sq  = vp.eta.array().square().matrix();
		vp.calcEdZtZ(dXtEEX_lowertri, n_env);
	}

	void calcPredEffects(VariationalParameters& vp){
		Eigen::VectorXd rr_beta = vp.mean_beta();

		vp.ym = X * rr_beta;
		if(n_covar > 0) {
			vp.ym += C * vp.muc.matrix().cast<scalarData>();
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
			vp.ym += C * vp.muc.matrix().cast<scalarData>();
		}

		if(n_effects > 1) {
			Eigen::VectorXd rr_gam = vp.mean_gam();
			vp.yx = X * rr_gam;
		}
	}

	void check_monotonic_elbo(const Hyps& hyps,
	                          VariationalParameters& vp,
	                          const long& count,
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

	void normaliseLogWeights(std::vector< double >& my_weights){
		// Safer to normalise log-weights than niavely convert to weights
		// Skip non-finite values!
		long nn = my_weights.size();
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
		double int_linear = 0, resLocal = 0;

		// Expectation of linear regression log-likelihood
		resLocal  = (Y - vp.ym).squaredNorm();
		if(n_effects > 1) {
			resLocal -= 2.0 * (Y - vp.ym).cwiseProduct(vp.eta).cwiseProduct(vp.yx).sum();
			if (n_env > 1) {
				resLocal += vp.yx.cwiseProduct(vp.eta_sq).dot(vp.yx);
			} else {
				resLocal += vp.yx.cwiseProduct(vp.eta).squaredNorm();
			}
		}
		MPI_Allreduce(&resLocal, &int_linear, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// variances
		if(n_covar > 0) {
			// covar main
			int_linear += (Nglobal - 1.0) * vp.sc_sq.sum();
		}
		// beta
		int_linear += (Nglobal - 1.0) * vp.var_beta().sum();
		if(n_effects > 1) {
			// gamma
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

	void compute_residuals_per_chr(const VariationalParametersLite& vp,
	                               std::vector<Eigen::VectorXd>& pred_main,
	                               std::vector<Eigen::VectorXd>& pred_int,
	                               std::vector<Eigen::VectorXd>& chr_residuals) const {
		pred_main.resize(n_chrs);
		pred_int.resize(n_chrs);
		chr_residuals.resize(n_chrs);

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
		if(!p.drop_loco) {
			for (auto cc : chrs_index) {
				if (n_effects > 1) {
					chr_residuals[cc] = map_residuals + pred_main[cc] + pred_int[cc].cwiseProduct(vp.eta.cast<double>());
				} else {
					chr_residuals[cc] = map_residuals + pred_main[cc];
				}
				EigenUtils::center_matrix(chr_residuals[cc]);
			}
		} else {
			for (auto cc : chrs_index) {
				chr_residuals[cc] = map_residuals;
				EigenUtils::center_matrix(chr_residuals[cc]);
			}
		}
	}

	void my_LOCO_pvals(const VariationalParametersLite& vp,
	                   const std::vector<Eigen::VectorXd>& chr_residuals,
	                   Eigen::MatrixXd& neglogPvals,
	                   Eigen::MatrixXd& testStats){
		assert(chr_residuals.size() == n_chrs);
		neglogPvals.resize(n_var, (n_env > 0 ? 4 : 1));
		testStats.resize(n_var, (n_env > 0 ? 4 : 1));

		std::map<long,long> chr_changes;
		long chr = X.chromosome[0];
		for (long jj = 0; jj < n_var; jj++) {
			if (chr != X.chromosome[jj]) {
				chr_changes[chr] = jj;
				chr = X.chromosome[jj];
			}
		}
		chr_changes[chr] = n_var;

		EigenDataMatrix D;
		long max_chunk_size = 256, start = 0;
		bool append = false;
		Eigen::MatrixXd chunk_neglogPvals, chunk_testStats;
		while (start < n_var) {
			long chr = X.chromosome[start];
			long cc = std::find(chrs_present.begin(), chrs_present.end(), chr) - chrs_present.begin();

			long chunkSize = std::min(max_chunk_size,chr_changes[chr] - start);
			std::vector<long> chunk(chunkSize);
			std::iota(chunk.begin(),chunk.end(),start);

			if (D.cols() != chunkSize) {
				D.resize(n_samples, chunkSize);
			}
			X.col_block3(chunk, D);
			compute_LOCO_pvals(chr_residuals[cc], D, vp, chunk_neglogPvals, chunk_testStats);

			neglogPvals.block(start,0,chunkSize,(n_env > 0 ? 4 : 1)) = chunk_neglogPvals;
			testStats.block(start,0,chunkSize,(n_env > 0 ? 4 : 1)) = chunk_testStats;
			start += chunkSize;
		}
	}

	void LOCO_pvals_v2(GenotypeMatrix &Xtest,
	                   const VariationalParametersLite &vp,
	                   const long &LOSO_window,
	                   Eigen::Ref<Eigen::VectorXd> neglogp_beta,
	                   Eigen::Ref<Eigen::VectorXd> neglogp_gam_robust,
	                   Eigen::Ref<Eigen::VectorXd> neglogp_joint,
	                   Eigen::Ref<Eigen::VectorXd> test_stat_beta,
	                   Eigen::Ref<Eigen::VectorXd> test_stat_gam_robust,
	                   Eigen::Ref<Eigen::VectorXd> test_stat_joint) const {
		assert(neglogp_beta.rows()  == n_var);
		assert(neglogp_joint.rows() == n_var);
		assert(test_stat_beta.rows()  == n_var);
		assert(test_stat_joint.rows() == n_var);
		assert(LOSO_window >= 0);

		int n_effects = (n_env > 0) ? 2 : 1;
		double N = n_samples;

		Eigen::VectorXd y_resid = (Y - vp.ym).cast<double>();
		if(n_env > 0) {
			y_resid -= vp.yx.cwiseProduct(vp.eta).cast<double>();
		}
		long front = 0, back = 0;

		// Compute p-vals per variant (p=3 as residuals mean centered)
		Eigen::MatrixXd H(n_samples, n_effects);
		boost::math::students_t t_dist(n_samples - n_effects);
		boost::math::fisher_f f_dist(n_effects, n_samples - n_effects);
		for(uint32_t jj = 0; jj < n_var; jj++ ) {
			int chr_test = Xtest.chromosome[jj];
			long pos_test = Xtest.position[jj];

			while (front < n_var && X.position[front] < pos_test + LOSO_window && X.chromosome[front] == chr_test) {
				y_resid += vp.mean_beta(front) * X.col(front);
				if (n_env > 0) y_resid += vp.mean_gam(front) * vp.eta.cwiseProduct(X.col(front));
				front++;
			}

			while (X.position[back] < pos_test - LOSO_window || X.chromosome[back] != chr_test) {
				y_resid -= vp.mean_beta(back) * X.col(back);
				if (n_env > 0) y_resid -= vp.mean_gam(back) * vp.eta.cwiseProduct(X.col(back));
				back++;
			}

			H.col(0) = Xtest.col(jj).cast<double>();

			double rss_alt, rss_null;
			Eigen::MatrixXd HtH(H.cols(), H.cols()), Hty(H.cols(), 1);
			Eigen::MatrixXd HtH_inv(H.cols(), H.cols()), HtVH(H.cols(), H.cols());
			if(n_env == 0) {

				prep_lm(H, y_resid, HtH, HtH_inv, Hty, rss_alt);

				double beta_tstat, beta_pval;
				student_t_test(n_samples, HtH_inv, Hty, rss_alt, 0, beta_tstat, beta_pval);

				neglogp_beta(jj) = -1 * log10(beta_pval);
				test_stat_beta(jj) = beta_tstat;
			} else if (n_env > 0) {
				H.col(1) = H.col(0).cwiseProduct(vp.eta.cast<double>());

				prep_lm(H, y_resid, HtH, HtH_inv, Hty, rss_alt, HtVH);
				rss_null = y_resid.squaredNorm();
				rss_null = mpiUtils::mpiReduce_inplace(&rss_null);

				// Single-var tests
				double beta_tstat, gam_tstat, rgam_stat, beta_pval, gam_pval, rgam_pval;
				hetero_chi_sq(HtH_inv, Hty, HtVH, 1, rgam_stat, rgam_pval);
				student_t_test(n_samples, HtH_inv, Hty, rss_alt, 1, gam_tstat, gam_pval);
				student_t_test(n_samples, HtH_inv, Hty, rss_alt, 0, beta_tstat, beta_pval);

				// F-test over main+int effects of snp_j
				double joint_fstat, joint_pval;
				joint_fstat       = (rss_null - rss_alt) / 2.0;
				joint_fstat      /= rss_alt / (N - 3.0);
				joint_pval        = 1.0 - cdf(f_dist, joint_fstat);

				neglogp_beta[jj]  = -1 * log10(beta_pval);
				neglogp_gam_robust[jj]   = -1 * log10(rgam_pval);
				neglogp_joint[jj] = -1 * log10(joint_pval);
				test_stat_beta[jj]  = beta_tstat;
				test_stat_gam_robust[jj]   = rgam_stat;
				test_stat_joint[jj] = joint_fstat;
			}
		}
	}

	/********** Output functions ************/
	void write_converged_hyperparams_to_file(const std::string& file_prefix,
	                                         const std::vector< VbTracker >& trackers,
	                                         const long& my_n_grid){
		boost_io::filtering_ostream outf;
		std::string ofile       = fileUtils::fstream_init(outf, p.out_file, file_prefix, "");
		std::cout << "Writing converged hyperparameter values to " << ofile << std::endl;

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
		outf << "count elbo sigma";

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
			outf << " " << trackers[ii].count;
			outf << " " << trackers[ii].logw;
			outf << std::setprecision(6) << std::fixed;
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
		boost_io::close(outf);
	}

	void output_vb_results() {
		if(world_rank == 0) {
			std::string format = fileUtils::filepath_format(p.out_file, "", "_converged_vparams_*");
			std::cout << "Writing variational parameters to " << format << std::endl;

			std::string path = fileUtils::filepath_format(p.out_file, "", "_converged_vparams");
			vp_init.dump_to_prefix(path, X, env_names, covar_names);
		}

		if (n_env > 0) {
			std::string path = fileUtils::filepath_format(p.out_file, "", "_converged_eta");
			std::cout << "Writing eta to " << path << std::endl;
			fileUtils::dump_predicted_vec_to_file(vp_init.eta, path, "eta", sample_location);
		}

		/*********** Stats from MAP to file ************/
		// Predicted effects to file
		calcPredEffects(vp_init);
		compute_residuals_per_chr(vp_init, ym_per_chr, yx_per_chr, resid_loco);
		Eigen::VectorXd Ealpha = Eigen::VectorXd::Zero(n_samples);
		if(n_covar > 0) {
			Ealpha += (C * vp_init.muc.matrix().cast<scalarData>()).cast<double>();
		}

		std::string header = "Y";
		long n_cols = 1;
		if (n_covar > 0) {
			header += " Ealpha"; n_cols += 1;
		}
		header += " Xbeta"; n_cols += 1;
		if (n_env > 0) {
			header += " eta Xgamma"; n_cols += 2;
		}
		if(n_chrs > 0) {
			for(auto cc : chrs_present) {
				header += " resid_excl_chr" + std::to_string(cc);
			}
			n_cols += n_chrs;
		}

		Eigen::MatrixXd tmp(n_samples, n_cols);
		int cc = 0;
		tmp.col(cc) = Y; cc++;
		if (n_covar > 0) {
			tmp.col(cc) = Ealpha; cc++;
			tmp.col(cc) = vp_init.ym - Ealpha; cc++;
		} else {
			tmp.col(cc) = vp_init.ym; cc++;
		}
		if (n_env > 0) {
			tmp.col(cc) = vp_init.eta; cc++;
			tmp.col(cc) = vp_init.yx; cc++;
		}
		for(int cc1 = 0; cc1 < n_chrs; cc1++) {
			tmp.col(cc) = resid_loco[cc1]; cc++;
		}
		assert(cc == n_cols);

		std::string path = fileUtils::filepath_format(p.out_file, "", "_converged_yhat");
		std::cout << "Writing predicted and residualised phenotypes to " << path << std::endl;
		fileUtils::dump_predicted_vec_to_file(tmp, path, header, sample_location);

		// Save eta & residual phenotypes to separate files
		boost_io::filtering_ostream tmp_outf;

		for (long cc = 0; cc < n_chrs; cc++) {
			std::string path = fileUtils::filepath_format(p.out_file, "",
			                                              "_converged_resid_pheno_chr" + std::to_string(chrs_present[cc]));
			std::cout << "Writing residualised pheno to " << path << std::endl;
			fileUtils::dump_predicted_vec_to_file(resid_loco[cc], path,
			                                      "chr" + std::to_string(chrs_present[cc]),
			                                      sample_location);
		}
		std::cout << std::endl;
	}

	void my_compute_LOCO_pvals(VariationalParametersLite vp){
		calcPredEffects(vp);
		compute_residuals_per_chr(vp, ym_per_chr, yx_per_chr, resid_loco);

		// Compute LOCO p-values
		Eigen::MatrixXd neglogPvals, testStats;
		if(p.drop_loco) {
			if (p.debug) {
				std::cout << "Computing single-snp hypothesis tests while excluding SNPs within ";
				std::cout << p.LOSO_window << " of the test SNP" << std::endl;
			}
			Eigen::VectorXd neglogp_beta(n_var), neglogp_gam(n_var), neglogp_rgam(n_var), neglogp_joint(n_var);
			Eigen::VectorXd test_stat_beta(n_var), test_stat_gam(n_var), test_stat_rgam(n_var), test_stat_joint(n_var);
			LOCO_pvals_v2(X, vp, p.LOSO_window,
			              neglogp_beta, neglogp_rgam, neglogp_joint,
			              test_stat_beta, test_stat_rgam, test_stat_joint);
			neglogPvals.resize(neglogp_beta.rows(), 3);
			neglogPvals.col(0) = neglogp_beta;
			neglogPvals.col(1) = Eigen::VectorXd::Constant(neglogp_beta.rows(), -1);
			neglogPvals.col(2) = neglogp_rgam;
			testStats.resize(neglogp_beta.rows(), 3);
			testStats.col(0) = test_stat_beta;
			testStats.col(1) = Eigen::VectorXd::Constant(neglogp_beta.rows(), -1);
			testStats.col(2) = test_stat_rgam;
		} else {
			if (p.debug) std::cout << "Computing single-SNP hypothesis tests with LOCO strategy" << std::endl;
			my_LOCO_pvals(vp, resid_loco, neglogPvals, testStats);
		}

		if(world_rank == 0) {
			boost_io::filtering_ostream outf;
			std::string ofile = fileUtils::fstream_init(outf, p.out_file, "", "_loco_pvals");
			std::cout << "Writing single-SNP hypothesis test results ";
			if(p.drop_loco) {
				std::cout << "(leave out SNPs within " << p.LOSO_window << "bp)";
			} else {
				std::cout << "(computed with LOCO strategy)";
			}
			std::cout << " to " << ofile << std::endl;
			bool append = false;
			fileUtils::write_snp_stats_to_file(outf, n_effects, X, append, neglogPvals, testStats);
			boost_io::close(outf);
		}
	}
};

#endif
