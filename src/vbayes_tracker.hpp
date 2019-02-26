//
// Created by kerin on 13/11/2018.
//
// Holds variational_parameters + other diagnostic values that we might want to track
// Also other auxilliary functions for writing interim output to file
// One instance per 'run'

#ifndef VBAYES_TRACKER_HPP
#define VBAYES_TRACKER_HPP

#include <chrono>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <limits>
#include <vector>


#include "parameters.hpp"
#include "genotype_matrix.hpp"
#include "hyps.hpp"
#include "utils.hpp"
#include "file_streaming.hpp"
#include "variational_parameters.hpp"
#include "tools/eigen3.3/Dense"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/filesystem.hpp>


namespace boost_io = boost::iostreams;

class VbTracker {
public:
	int           count;              // Number of iterations to convergence at each step
	std::vector< double > logw_updates;  // elbo updates at each ii
	std::vector< double > alpha_diffs;  // elbo updates at each ii
	VariationalParametersLite vp;         // best mu at each ii
	double          logw;                // best logw at each ii
	Hyps           hyps;                // hyps values at end of VB inference.

	parameters p;

	// For writing interim output
	boost::filesystem::path dir;
	boost_io::filtering_ostream outf_elbo, outf_alpha_diff, outf_inits, outf_iter, outf_alpha;
	boost_io::filtering_ostream outf_weights, outf_rescan;
	std::string main_out_file;

	std::chrono::system_clock::time_point time_check;

	VbTracker(parameters my_params) : p(my_params), vp(my_params), hyps(my_params) {
		main_out_file = p.out_file;
	}

	VbTracker(const VbTracker& tracker) : p(tracker.p), vp(tracker.p), hyps(tracker.p) {
		main_out_file = p.out_file;
	}

	~VbTracker(){
		boost_io::close(outf_elbo);
		boost_io::close(outf_alpha_diff);
		boost_io::close(outf_inits);
		boost_io::close(outf_iter);
		boost_io::close(outf_alpha);
		boost_io::close(outf_weights);
		boost_io::close(outf_rescan);
	};

	void init_interim_output(const int ii,
							 const int round_index,
							 const int n_effects,
							 const int n_env,
							 std::vector< std::string > env_names,
							 const VariationalParameters& vp){
		time_check = std::chrono::system_clock::now();

		// Create directories
		std::string ss = "interim_files/grid_point_" + std::to_string(ii);
		ss = "r" + std::to_string(round_index) + "_" + ss;
		boost::filesystem::path interim_ext(ss), path(main_out_file);
		dir = path.parent_path() / interim_ext;
		boost::filesystem::create_directories(dir);

		// Initialise fstreams
		fstream_init(outf_iter, dir, "_iter_updates", false);

		// Weights - add header + initial values
		if(n_effects > 1) {
			fstream_init(outf_weights, dir, "_env_weights", false);
			for (int ll = 0; ll < n_env; ll++) {
				outf_weights << env_names[ll];
				if (ll + 1 < n_env) outf_weights << " ";
			}
			outf_weights << std::endl;
			for (int ll = 0; ll < n_env; ll++) {
				outf_weights << vp.muw(ll);
				if (ll + 1 < n_env) outf_weights << " ";
			}
			outf_weights << std::endl;
		}

		// Diagnostics - add header
		outf_iter    << "count\tsigma";
		for (int ee = 0; ee < n_effects; ee++){
			outf_iter << "\tpve" << ee;
			if((ee == 0 && p.mode_mog_prior_beta) || (ee == 1 && p.mode_mog_prior_gam)){
				outf_iter << "\tpve_large" << ee;
			}
			outf_iter << "\tsigma" << ee;
			if((ee == 0 && p.mode_mog_prior_beta) || (ee == 1 && p.mode_mog_prior_gam)){
				outf_iter << "\tsigma_spike" << ee;
			}
			outf_iter << "\tlambda" << ee;
		}
		outf_iter << "\ts_x";
		if(n_effects > 1) outf_iter << "\ts_z";
		outf_iter << "\tvar_covar";
		outf_iter << "\telbo\tmax_alpha_diff\tsecs" << std::endl;
	}

	void push_interim_hyps(const int& cnt,
                                  const Hyps& i_hyps,
                                  const double& c_logw,
                                  const double& c_alpha_diff,
                                  const int& n_effects,
                                  const unsigned long& n_var,
                                  const unsigned long& n_env,
                                  const VariationalParameters& vp){
		// Diagnostics + env-weights from latest vb iteration
		std::chrono::duration<double> lapsecs = std::chrono::system_clock::now() - time_check;

		outf_iter << cnt << "\t";
		outf_iter << std::setprecision(3) << std::fixed;
		outf_iter << i_hyps.sigma << "\t";

		for (int ee = 0; ee < n_effects; ee++) {
			// PVE
			outf_iter << std::setprecision(6) << std::fixed;
			outf_iter << i_hyps.pve(ee) << "\t";
			if ((ee == 0 && p.mode_mog_prior_beta) || (ee == 1 && p.mode_mog_prior_gam)) {
				outf_iter << i_hyps.pve_large(ee) << "\t";
			}

			// Relative variance
			outf_iter << std::setprecision(12) << std::fixed;
			outf_iter << i_hyps.slab_relative_var(ee) << "\t";
			if((ee == 0 && p.mode_mog_prior_beta) || (ee == 1 && p.mode_mog_prior_gam)){
				outf_iter << i_hyps.spike_relative_var(ee) << "\t";
			}

			// Lambda
			outf_iter << i_hyps.lambda(ee) << "\t";
		}

		outf_iter << std::setprecision(3) << std::fixed;
		for( int ee = 0; ee < n_effects; ee++) {
			outf_iter << i_hyps.s_x(ee) << "\t";
		}
		outf_iter << vp.muc.square().sum() << "\t";
		outf_iter << c_logw << "\t";
		outf_iter << c_alpha_diff << "\t";
		outf_iter << lapsecs.count() << std::endl;


		// Weights
		if(n_effects > 1) {
			for (int ll = 0; ll < n_env; ll++) {
				outf_weights << vp.muw(ll);
				if (ll < n_env - 1) outf_weights << "\t";
			}
			outf_weights << std::endl;
		}

		time_check = std::chrono::system_clock::now();
	}

	void push_vp_converged(const GenotypeMatrix& X,
                             const std::uint32_t n_var,
							 const int& n_effects){
		// Assumes that information for all measures that we track have between
		// added to VbTracker at index ii.

		// Converged snp-stats to file
		fstream_init(outf_inits, dir, "_converged", true);
		write_snp_stats_to_file(outf_inits, n_effects, n_var, vp, X, p, true);
		boost_io::close(outf_inits);
	}


	void push_interim_param_values(const int& cnt,
								   const int& n_effects,
								   const unsigned long& n_var,
								   const VariationalParameters& vp,
								   const GenotypeMatrix& X){
		fstream_init(outf_inits, dir, "_params_iter" + std::to_string(cnt), true);
		write_snp_stats_to_file(outf_inits, n_effects, n_var, vp, X, p, true);
		boost_io::close(outf_inits);
	}

	void dump_state(const int& cnt,
					const long& n_covar,
					const long& n_var,
					const int& n_env,
					const int& n_effects,
					const VariationalParameters& vp,
					const Hyps& hyps,
					const GenotypeMatrix& X,
					const std::vector< std::string >& covar_names,
					const std::vector< std::string >& env_names){

		// Snp effects
		fstream_init(outf_inits, dir, "_dump_it" + std::to_string(cnt) + "_snps", true);
		write_snp_stats_to_file(outf_inits, n_effects, n_var, vp, X, p, true);
		boost_io::close(outf_inits);

		// Covars
		if(n_covar > 0) {
			fstream_init(outf_inits, dir, "_dump_it" + std::to_string(cnt) + "_covars", true);
			outf_inits << std::scientific << std::setprecision(7);
			outf_inits << "covar mu s_sq" << std::endl;
			for (int cc = 0; cc < n_covar; cc++) {
				outf_inits << covar_names[cc] << " ";
				outf_inits << vp.muc(cc) << " ";
				outf_inits << vp.sc_sq(cc) << std::endl;
			}
			boost_io::close(outf_inits);
		}

		// Env-weights
		if(n_env > 0) {
			fstream_init(outf_inits, dir, "_dump_it" + std::to_string(cnt) + "_env", true);
			outf_inits << std::scientific << std::setprecision(7);
			outf_inits << "env mu s_sq" << std::endl;
			for (int ll = 0; ll < n_env; ll++) {
				outf_inits << env_names[ll] << " ";
				outf_inits << vp.muw(ll) << " ";
				outf_inits << vp.sw_sq(ll) << std::endl;
			}
			boost_io::close(outf_inits);
		}

		// Hyps
		fstream_init(outf_inits, dir, "_dump_it" + std::to_string(cnt) + "_hyps", true);
		outf_inits << hyps << std::endl;
		boost_io::close(outf_inits);
	}

	void push_interim_covar_values(const int& cnt,
								   const long& n_covar,
								   const VariationalParameters& vp,
								   const std::vector< std::string >& covar_names){
		fstream_init(outf_inits, dir, "_covars_iter" + std::to_string(cnt), true);

		outf_inits << "covar_name mu_covar s_sq_covar";
		outf_inits << std::endl;
		for (int cc = 0; cc < n_covar; cc++){
			outf_inits << covar_names[cc] << " " << vp.muc(cc);
			outf_inits << " " << vp.sc_sq(cc);
			outf_inits << std::endl;
		}
		boost_io::close(outf_inits);
	}

	void push_rescan_gwas(const GenotypeMatrix& X,
							 const std::uint32_t n_var,
							 const Eigen::Ref<const Eigen::VectorXd>& neglogp){
		// Assumes that information for all measures that we track have between
		// added to VbTracker at index ii.

		fstream_init(outf_rescan, dir, "_rescan", true);
		outf_rescan << "chr rsid pos a0 a1 maf info neglogp" << std::endl;
		for (std::uint32_t kk = 0; kk < n_var; kk++){
			outf_rescan << X.chromosome[kk] << " " << X.rsid[kk]<< " " << X.position[kk];
			outf_rescan << " " << X.al_0[kk] << " " << X.al_1[kk] << " ";
			outf_rescan << X.maf[kk] << " " << X.info[kk] << " " << neglogp(kk);
			outf_rescan << std::endl;
		}
	}

	void fstream_init(boost_io::filtering_ostream& my_outf,
                             const boost::filesystem::path& dir,
                             const std::string& file_suffix,
                             const bool& allow_gzip){
		my_outf.reset();

		std::string filepath   = main_out_file;
		std::string stem_w_dir = filepath.substr(0, filepath.find('.'));
		std::string stem       = stem_w_dir.substr(stem_w_dir.rfind('/')+1, stem_w_dir.size());
		std::string ext        = filepath.substr(filepath.find('.'), filepath.size());

		if(!allow_gzip){
			ext = ext.substr(0, ext.find(".gz"));
		}
		std::string ofile      = dir.string() + "/" + stem + file_suffix + ext;

		if (ext.find(".gz") != std::string::npos) {
			my_outf.push(boost_io::gzip_compressor());
		}
		my_outf.push(boost_io::file_sink(ofile));
	}
};

#endif
