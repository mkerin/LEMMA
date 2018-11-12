#ifndef VBAYES_TRACKER_HPP
#define VBAYES_TRACKER_HPP

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <limits>
#include <vector>
#include "variational_parameters.hpp"
#include "tools/eigen3.3/Dense"
#include "my_timer.hpp"
#include "misc_utils.hpp"
#include "genotype_matrix.hpp"
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/filesystem.hpp>
#include "class.h"


namespace boost_io = boost::iostreams;


class Hyps{
	int sigma_ind   = 0;
	int sigma_b_ind = 1;
	int sigma_g_ind = 2;
	int lam_b_ind   = 3;
	int lam_g_ind   = 4;

public:
	double sigma;
	Eigen::ArrayXd slab_var;
	Eigen::ArrayXd spike_var;
	Eigen::ArrayXd slab_relative_var;
	Eigen::ArrayXd spike_relative_var;
	Eigen::ArrayXd lambda;

	// Not hyperparameters, but things that depend on them
	Eigen::ArrayXd s_x;
	Eigen::ArrayXd pve;
	Eigen::ArrayXd pve_large;

	Hyps(){};

	void init_from_grid(int n_effects,
			int ii,
			int n_var,
			const Eigen::Ref<const Eigen::MatrixXd>& hyps_grid,
			const parameters& p,
			const double& my_s_z){
		// Implicit that n_effects > 1

		// Unpack
		double my_sigma = hyps_grid(ii, sigma_ind);
		double my_sigma_b = hyps_grid(ii, sigma_b_ind);
		double my_sigma_g = hyps_grid(ii, sigma_g_ind);
		double my_lam_b = hyps_grid(ii, lam_b_ind);
		double my_lam_g = hyps_grid(ii, lam_g_ind);

		// Resize
		slab_var.resize(n_effects);
		spike_var.resize(n_effects);
		slab_relative_var.resize(n_effects);
		spike_relative_var.resize(n_effects);
		lambda.resize(n_effects);
		s_x.resize(n_effects);

		// Assign initial hyps
		sigma = my_sigma;
		slab_var << my_sigma * my_sigma_b, my_sigma * my_sigma_g;
		spike_var << my_sigma * my_sigma_b / p.spike_diff_factor, my_sigma * my_sigma_g / p.spike_diff_factor;
		slab_relative_var << my_sigma_b, my_sigma_g;
		spike_relative_var << my_sigma_b / p.spike_diff_factor, my_sigma_g / p.spike_diff_factor;
		lambda << my_lam_b, my_lam_g;
		s_x << n_var, my_s_z;
	}

	void init_from_grid(int n_effects,
			int ii,
			int n_var,
			const Eigen::Ref<const Eigen::MatrixXd>& hyps_grid,
			const parameters& p){
		/*** Implicit that n_effects == 1 ***/

		// Unpack
		double my_sigma = hyps_grid(ii, sigma_ind);
		double my_sigma_b = hyps_grid(ii, sigma_b_ind);
		double my_sigma_g = hyps_grid(ii, sigma_g_ind);
		double my_lam_b = hyps_grid(ii, lam_b_ind);
		double my_lam_g = hyps_grid(ii, lam_g_ind);

		// Resize
		slab_var.resize(n_effects);
		spike_var.resize(n_effects);
		slab_relative_var.resize(n_effects);
		spike_relative_var.resize(n_effects);
		lambda.resize(n_effects);
		s_x.resize(n_effects);

		// Assign initial hyps
		sigma = my_sigma;
		slab_var << my_sigma * my_sigma_b;
		spike_var << my_sigma * my_sigma_b / p.spike_diff_factor;
		slab_relative_var << my_sigma_b;
		spike_relative_var << my_sigma_b / p.spike_diff_factor;
		lambda << my_lam_b;
		s_x << n_var;
	}
};


class VbTracker {
public:
	std::vector< int >             counts_list;              // Number of iterations to convergence at each step
	std::vector< std::vector< double > > logw_updates_list;  // elbo updates at each ii
	std::vector< std::vector< double > > alpha_diff_list;  // elbo updates at each ii
	std::vector< VariationalParametersLite > vp_list;                  // best mu at each ii
	std::vector< double >          logw_list;                // best logw at each ii
	std::vector< double >          elapsed_time_list;        // time to compute grid point
	std::vector< Hyps >            hyps_list;                // hyps values at end of VB inference.

	parameters p;

	// For writing interim output
	boost::filesystem::path dir;
	boost_io::filtering_ostream outf_elbo, outf_alpha_diff, outf_inits, outf_iter, outf_alpha;
	boost_io::filtering_ostream outf_weights, outf_rescan;
	std::string main_out_file;
	bool allow_interim_push;

	// Timing
	MyTimer t_interimOutput;

	VbTracker(): t_interimOutput("interimOutput: %ts \n"){
		allow_interim_push = false;
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

	void set_main_filepath(const std::string &ofile){
		main_out_file = ofile;
		allow_interim_push = true;
	}

	void push_interim_param_values(const int& cnt,
                                  const int& n_effects,
                                  const int& n_var,
                                  const VariationalParameters& vp,
                                  const GenotypeMatrix& X){
		t_interimOutput.resume();

		fstream_init(outf_inits, dir, "_params_iter" + std::to_string(cnt), true);
		write_snp_stats_to_file(outf_inits, n_effects, n_var, vp, X, p, true);
		boost_io::close(outf_inits);

		t_interimOutput.stop();
	}

	void push_interim_covar_values(const int& cnt,
                                  const int& n_covar,
                                  const VariationalParameters& vp,
                                  const std::vector< std::string >& covar_names){
		t_interimOutput.resume();
		fstream_init(outf_inits, dir, "_covars_iter" + std::to_string(cnt), true);

		outf_inits << "covar_name mu_covar s_sq_covar";
		outf_inits << std::endl;
		for (int cc = 0; cc < n_covar; cc++){
			outf_inits << covar_names[cc] << " " << vp.muc(cc);
			outf_inits << " " << vp.sc_sq(cc);
			outf_inits << std::endl;
		}
		t_interimOutput.stop();
	}

	void push_interim_iter_update(const int& cnt,
                                  const Hyps& i_hyps,
                                  const double& c_logw,
                                  const double& c_alpha_diff,
                                  const double& lap_seconds,
                                  const int& n_effects,
                                  const int& n_var,
                                  const int& n_env,
                                  const VariationalParameters& vp){
		// Diagnostics + env-weights from latest vb iteration
		t_interimOutput.resume();

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
		outf_iter << c_logw << "\t";
		outf_iter << c_alpha_diff << "\t";
		outf_iter << lap_seconds << std::endl;


		// Weights
		for (int ll = 0; ll < n_env; ll++){
			outf_weights << vp.muw(ll);
			if(ll < n_env - 1) outf_weights << "\t";
		}
		outf_weights << std::endl;

		t_interimOutput.stop();
	}

	void push_interim_output(int ii,
                             const GenotypeMatrix& X,
                             const std::uint32_t n_var,
							 const std::uint32_t n_effects){
		// Assumes that information for all measures that we track have between
		// added to VbTracker at index ii.
		t_interimOutput.resume();

		// TODO: Add converged covar values?

		// Converged snp-stats to file
		fstream_init(outf_inits, dir, "_converged", true);
		write_snp_stats_to_file(outf_inits, n_effects, n_var, vp_list[ii], X, p, true);
		boost_io::close(outf_inits);

		t_interimOutput.stop();
	}

	void push_rescan_gwas(const GenotypeMatrix& X,
							 const std::uint32_t n_var,
							 const Eigen::Ref<const Eigen::VectorXd>& neglogp){
		// Assumes that information for all measures that we track have between
		// added to VbTracker at index ii.
		t_interimOutput.resume();

		fstream_init(outf_rescan, dir, "_rescan", true);
		outf_rescan << "chr rsid pos a0 a1 maf info neglogp" << std::endl;
		for (std::uint32_t kk = 0; kk < n_var; kk++){
			outf_rescan << X.chromosome[kk] << " " << X.rsid[kk]<< " " << X.position[kk];
			outf_rescan << " " << X.al_0[kk] << " " << X.al_1[kk] << " ";
			outf_rescan << X.maf[kk] << " " << X.info[kk] << " " << neglogp(kk);
			outf_rescan << std::endl;
		}

		t_interimOutput.stop();
	}

	void interim_output_init(const int ii,
                             const int round_index,
                             const int n_effects,
                             const int n_env,
                             std::vector< std::string > env_names,
                             const VariationalParameters& vp){
		if(!allow_interim_push){
			throw std::runtime_error("Internal error; interim push not expected");
		}

		// Create directories
		std::string ss = "interim_files/grid_point_" + std::to_string(ii);
		ss = "r" + std::to_string(round_index) + "_" + ss;
		boost::filesystem::path interim_ext(ss), path(main_out_file);
		dir = path.parent_path() / interim_ext;
		boost::filesystem::create_directories(dir);

		// Initialise fstreams
		fstream_init(outf_iter, dir, "_iter_updates", false);
		fstream_init(outf_weights, dir, "_env_weights", false);

		// Weights - add header + initial values
		for (int ll = 0; ll < n_env; ll++){
			outf_weights << env_names[ll];
			if(ll + 1 < n_env) outf_weights << " ";
		}
		outf_weights << std::endl;
		for (int ll = 0; ll < n_env; ll++){
			outf_weights << vp.muw(ll);
			if(ll + 1 < n_env) outf_weights << " ";
		}
		outf_weights << std::endl;

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
		outf_iter << "\ts_x" << "\ts_z";
		outf_iter << "\telbo\tmax_alpha_diff\tseconds" << std::endl;

	}

	void fstream_init(boost_io::filtering_ostream& my_outf,
                             const boost::filesystem::path& dir,
                             const std::string& file_suffix,
                             const bool& allow_gzip){
		my_outf.reset();

		std::string filepath   = main_out_file;
		std::string stem_w_dir = filepath.substr(0, filepath.find("."));
		std::string stem       = stem_w_dir.substr(stem_w_dir.rfind("/")+1, stem_w_dir.size());
		std::string ext        = filepath.substr(filepath.find("."), filepath.size());

		if(!allow_gzip){
			ext = ext.substr(0, ext.find(".gz"));
		}
		std::string ofile      = dir.string() + "/" + stem + file_suffix + ext;

		if (ext.find(".gz") != std::string::npos) {
			my_outf.push(boost_io::gzip_compressor());
		}
		my_outf.push(boost_io::file_sink(ofile));
	}

	void resize(int n_list){
		counts_list.resize(n_list);
		vp_list.resize(n_list);
		logw_updates_list.resize(n_list);
		alpha_diff_list.resize(n_list);
		logw_list.resize(n_list);
		elapsed_time_list.resize(n_list);
		hyps_list.resize(n_list);
		for (int ll = 0; ll < n_list; ll++){
			logw_list[ll] = -std::numeric_limits<double>::max();
		}
	}

	void copy_ith_element(int jj, int ii, const VbTracker& other_tracker){
		counts_list[jj]       = other_tracker.counts_list[ii];
		vp_list[jj]           = other_tracker.vp_list[ii];
		logw_list[jj]         = other_tracker.logw_list[ii];
		logw_updates_list[jj] = other_tracker.logw_updates_list[ii];
		alpha_diff_list[jj]   = other_tracker.alpha_diff_list[ii];
		elapsed_time_list[jj] = other_tracker.elapsed_time_list[ii];
		hyps_list[jj]         = other_tracker.hyps_list[ii];
	}
};

#endif
