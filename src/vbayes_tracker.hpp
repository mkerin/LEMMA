//
// Created by kerin on 13/11/2018.
//
// Holds variational_parameters + other diagnostic values that we might want to track
// Also other auxilliary functions for writing interim output to file
// One instance per 'run'

#ifndef VBAYES_TRACKER_HPP
#define VBAYES_TRACKER_HPP

#include "parameters.hpp"
#include "genotype_matrix.hpp"
#include "hyps.hpp"
#include "file_utils.hpp"
#include "variational_parameters.hpp"
#include "tools/eigen3.3/Dense"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/filesystem.hpp>

#include <chrono>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <limits>
#include <vector>


namespace boost_io = boost::iostreams;

class VbTracker {
public:
	long count;
	VariationalParametersLite vp;
	double logw;
	Hyps hyps;

	parameters p;

// For writing interim output
	boost::filesystem::path dir, subdir;
	boost_io::filtering_ostream outf_elbo, outf_alpha_diff, outf_inits, outf_iter, outf_alpha;
	boost_io::filtering_ostream outf_weights;
	std::string main_out_file;

	std::chrono::system_clock::time_point time_check;
	long count_to_convergence;

	VbTracker(const parameters& my_params) : p(my_params), vp(my_params), hyps(my_params) {
		main_out_file = p.out_file;
		count_to_convergence = 0;
	}

	VbTracker(const VbTracker& tracker) : p(tracker.p), vp(tracker.p), hyps(tracker.p) {
		main_out_file = p.out_file;
		count_to_convergence = 0;
	}

	~VbTracker(){
		boost_io::close(outf_elbo);
		boost_io::close(outf_alpha_diff);
		boost_io::close(outf_inits);
		boost_io::close(outf_iter);
		boost_io::close(outf_alpha);
		boost_io::close(outf_weights);
	};

	void init_interim_output(const long& ii,
	                         const int& round_index,
	                         const int& n_effects,
	                         const long& n_covar,
	                         const long& n_env,
	                         std::vector< std::string > env_names,
	                         const VariationalParameters& vp){
		time_check = std::chrono::system_clock::now();

		// Create directories
		std::string ss = "lemma_interim_files";
		boost::filesystem::path interim_ext(ss), path(main_out_file);
		dir = path.parent_path() / interim_ext;
		subdir = interim_ext;
		boost::filesystem::create_directories(dir);

		// Initialise fstreams
		fileUtils::fstream_init(outf_iter, p.out_file, subdir.string() + "/", "_iter_updates", false);

		// Diagnostics - add header
		outf_iter << "count sigma";
		for (int ee = 0; ee < n_effects; ee++) {
			outf_iter << " pve" << ee;
			if((ee == 0 && p.mode_mog_prior_beta) || (ee == 1 && p.mode_mog_prior_gam)) {
				outf_iter << " pve_large" << ee;
			}
			outf_iter << " sigma" << ee;
			if((ee == 0 && p.mode_mog_prior_beta) || (ee == 1 && p.mode_mog_prior_gam)) {
				outf_iter << " sigma_spike" << ee;
			}
			outf_iter << " lambda" << ee;
		}
		outf_iter << " s_x";
		if(n_env > 0) outf_iter << " s_z";
		outf_iter << " var_covar elbo";
		if (n_covar > 0) outf_iter << " max_covar_diff";
		outf_iter << " max_beta_diff";
		if (n_env > 0) outf_iter << " max_gam_diff";
		if (n_env > 1) outf_iter << " max_w_diff";
		outf_iter << " secs" << std::endl;

		// Weights - add header + initial values
		if(n_env > 0) {
			fileUtils::fstream_init(outf_weights, p.out_file, subdir.string() + "/", "_env_weights", false);
			outf_weights << "count ";
			for (int ll = 0; ll < n_env; ll++) {
				outf_weights << env_names[ll];
				if (ll + 1 < n_env) outf_weights << " ";
			}
			outf_weights << std::endl;
			outf_weights << "-1 ";
			for (int ll = 0; ll < n_env; ll++) {
				outf_weights << vp.muw(ll);
				if (ll + 1 < n_env) outf_weights << " ";
			}
			outf_weights << std::endl;
		}
	}

	void push_interim_hyps(const long& cnt,
	                       const Hyps& i_hyps,
	                       const double& c_logw,
	                       const double& covar_diff,
	                       const double& beta_diff,
	                       const double& gam_diff,
	                       const double& w_diff,
	                       const int& n_effects,
	                       const long& n_var,
	                       const long& n_covar,
	                       const long& n_env,
	                       const VariationalParameters& vp) {
		// Diagnostics + env-weights from latest vb iteration
		std::chrono::duration<double> lapsecs = std::chrono::system_clock::now() - time_check;

		outf_iter << cnt << " ";
		outf_iter << std::setprecision(6) << std::fixed;
		outf_iter << i_hyps.sigma << " ";

		for (int ee = 0; ee < n_effects; ee++) {
			// PVE
			outf_iter << std::setprecision(6) << std::fixed;
			outf_iter << i_hyps.pve(ee) << " ";
			if ((ee == 0 && p.mode_mog_prior_beta) || (ee == 1 && p.mode_mog_prior_gam)) {
				outf_iter << i_hyps.pve_large(ee) << " ";
			}

			// Relative variance
			outf_iter << std::setprecision(12) << std::fixed;
			outf_iter << i_hyps.slab_relative_var(ee) << " ";
			if ((ee == 0 && p.mode_mog_prior_beta) || (ee == 1 && p.mode_mog_prior_gam)) {
				outf_iter << i_hyps.spike_relative_var(ee) << " ";
			}

			// Lambda
			outf_iter << i_hyps.lambda(ee) << " ";
		}

		outf_iter << std::setprecision(6) << std::scientific;
		for (int ee = 0; ee < n_effects; ee++) {
			outf_iter << i_hyps.s_x(ee) << " ";
		}
		outf_iter << vp.muc.square().sum() << " ";
		outf_iter << std::setprecision(6) << std::fixed;
		outf_iter << c_logw << " ";
		outf_iter << std::setprecision(6) << std::scientific;
		if (n_covar > 0) outf_iter << covar_diff << " ";
		outf_iter << beta_diff << " ";
		if (n_env > 0) outf_iter << gam_diff << " ";
		if (n_env > 1) outf_iter << w_diff << " ";
		outf_iter << std::setprecision(6) << std::fixed;
		outf_iter << lapsecs.count() << std::endl;

		if(n_effects > 1) {
			outf_weights << cnt << " ";
			for (int ll = 0; ll < n_env; ll++) {
				outf_weights << vp.muw(ll);
				if (ll < n_env - 1) outf_weights << " ";
			}
			outf_weights << std::endl;
		}
	}

	void dump_state(const std::string& count,
	                const long& n_samples,
	                const long& n_covar,
	                const long& n_var,
	                const long& n_env,
	                const int& n_effects,
	                const VariationalParameters& vp,
	                const Hyps& hyps,
	                const EigenDataVector& Y,
	                const EigenDataArrayXX& C,
	                const GenotypeMatrix& X,
	                const std::vector< std::string >& covar_names,
	                const std::vector< std::string >& env_names,
	                const std::unordered_map<long, bool>& sample_is_invalid,
	                const std::map<long, int>& sample_location){
		std::string path;

		// Aggregate effects
		Eigen::VectorXd Ealpha = Eigen::VectorXd::Zero(n_samples);
		if(n_covar > 0) {
			Ealpha += (C.matrix() * vp.muc.matrix().cast<scalarData>()).cast<double>();
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

		Eigen::MatrixXd tmp(n_samples, n_cols);
		int cc = 0;
		tmp.col(cc) = Y; cc++;
		if (n_covar > 0) {
			tmp.col(cc) = Ealpha; cc++;
		}
		tmp.col(cc) = vp.ym; cc++;
		if (n_env > 0) {
			tmp.col(cc) = vp.eta; cc++;
			tmp.col(cc) = vp.yx; cc++;
		}
		assert(cc == n_cols);

		path = fileUtils::filepath_format(p.out_file, subdir.string() + "/", "_dump_it" + count + "_aggregate");
		fileUtils::dump_predicted_vec_to_file(tmp, path, header, sample_location);

		int world_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		if(world_rank == 0) {
			path = fileUtils::filepath_format(p.out_file, subdir.string() + "/", "_dump_it" + count + "_latent_snps");
			vp.snps_to_file(path, X, n_env);

			if(n_covar > 0) {
				path = fileUtils::filepath_format(p.out_file, subdir.string() + "/", "_dump_it" + count + "_covars");
				vp.covar_to_file(path, covar_names);
			}

			if(n_env > 0) {
				path = fileUtils::filepath_format(p.out_file, subdir.string() + "/", "_dump_it" + count + "_env");
				vp.env_to_file(path, env_names);
			}

			path = fileUtils::filepath_format(p.out_file, subdir.string() + "/", "_dump_it" + count + "_hyps");
			hyps.to_file(path);
		}
	}

};

#endif
