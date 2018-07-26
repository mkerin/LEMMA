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
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/filesystem.hpp>
#include "class.h"


namespace io = boost::iostreams;


struct Hyps{
// public:
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

	// Hyps();

	// Hyps(int n_effects, double my_sigma, double sigma_b, double sigma_g, double lam_b, double lam_g){
	// 	slab_var.resize(n_effects);
	// 	spike_var.resize(n_effects);
	// 	slab_relative_var.resize(n_effects);
	// 	spike_relative_var.resize(n_effects);
	// 	lambda.resize(n_effects);
	//
	// 	sigma = my_sigma;
	// 	slab_var           << sigma * sigma_b, sigma * sigma_g;
	// 	spike_var          << sigma * sigma_b / 100.0, sigma * sigma_g / 100.0;
	// 	slab_relative_var  << sigma_b, sigma_g;
	// 	spike_relative_var << sigma_b, sigma_g;
	// 	lambda             << lam_b, lam_g;
	// }
};


class VbTracker {
public:
	std::vector< int >             counts_list;              // Number of iterations to convergence at each step
	std::vector< std::vector< double > > logw_updates_list;  // elbo updates at each ii
	std::vector< std::vector< double > > alpha_diff_list;  // elbo updates at each ii
	std::vector< VariationalParametersLite > vp_list;                  // best mu at each ii
	// std::vector< Eigen::VectorXd > mu_list;                  // best mu at each ii
	// std::vector< Eigen::VectorXd > alpha_list;               // best alpha at each ii
	std::vector< double >          logw_list;                // best logw at each ii
	std::vector< double >          elapsed_time_list;        // time to compute grid point
	std::vector< Hyps >            hyps_list;                // hyps values at end of VB inference.

	parameters p;

	// For writing interim output
	io::filtering_ostream outf_elbo, outf_alpha_diff, outf_weights, outf_inits, outf_iter, outf_alpha, outf_w;
	std::string main_out_file;
	bool allow_interim_push;

	VbTracker(){
		allow_interim_push = false;
	}

	VbTracker(const std::string& ofile) : main_out_file(ofile){
		allow_interim_push = true;
	}

	VbTracker(int n_list, const std::string& ofile) : main_out_file(ofile){
		counts_list.resize(n_list);
		vp_list.resize(n_list);
		// mu_list.resize(n_list);
		// alpha_list.resize(n_list);
		logw_list.resize(n_list);
		logw_updates_list.resize(n_list);
		alpha_diff_list.resize(n_list);
		elapsed_time_list.resize(n_list);
		hyps_list.resize(n_list);

		allow_interim_push = true;
	}

	~VbTracker(){
		io::close(outf_elbo);
		io::close(outf_alpha_diff);
		io::close(outf_weights);
		io::close(outf_inits);
		io::close(outf_iter);
		io::close(outf_alpha);
		io::close(outf_w);
	};

	void set_main_filepath(const std::string &ofile){
		main_out_file = ofile;
		allow_interim_push = true;
	}

	void push_interim_iter_update(const int& cnt,
                                  const Hyps& i_hyps,
                                  const double& c_logw,
                                  const double& c_alpha_diff,
                                  const double& lap_seconds,
                                  const long int hty_counter,
                                  const int& n_effects,
                                  const int& n_var,
                                  const int& n_env,
                                  const VariationalParameters& vp){
		outf_iter << cnt << "\t" << std::setprecision(3) << std::fixed;
		outf_iter << i_hyps.sigma << "\t" << std::setprecision(8) << std::fixed;
		for (int ee = 0; ee < n_effects; ee++){
			outf_iter << i_hyps.pve(ee) << "\t";
			if(p.mode_mog_prior){
				outf_iter << i_hyps.pve_large(ee) << "\t";
			}
			outf_iter << i_hyps.slab_relative_var(ee) << "\t";
			if(p.mode_mog_prior){
				outf_iter << i_hyps.spike_relative_var(ee) << "\t";
			}
			outf_iter << i_hyps.lambda(ee) << "\t";
		}
		outf_iter << std::setprecision(3) << std::fixed << c_logw << "\t";
		outf_iter << c_alpha_diff << "\t";
		outf_iter << lap_seconds << "\t";
		outf_iter << hty_counter << std::endl;

		for (int ll = 0; ll < n_env; ll++){
			outf_w << vp.muw(ll);
			if(ll < n_env - 1) outf_w << "\t";
		}
		outf_w << std::endl;

		if(p.xtra_verbose){
			for (int ee = 0; ee < n_effects; ee++){
				for (std::uint32_t kk = 0; kk < n_var; kk++){
					outf_alpha << vp.alpha(kk, ee);
 					if(!(ee == n_effects-1 && kk == n_var-1)){
						outf_alpha << " ";
					}
				}
			}
			outf_alpha << std::endl;
		}
	}

	void push_interim_output(int ii,
                             const std::vector< int >& chromosome,
                             const std::vector< std::string >& rsid,
                             const std::vector< std::uint32_t >& position,
                             const std::vector< std::string >& al_0,
                             const std::vector< std::string >& al_1,
                             const std::uint32_t n_var,
													   const std::uint32_t n_effects){
		// Assumes that information for all measures that we track have between
		// added to VbTracker at index ii.

		// Write output to file
		outf_weights << "NA" << " " << logw_list[ii] << " ";
		outf_weights << "NA" << " ";
		outf_weights << counts_list[ii] << " ";
		outf_weights << elapsed_time_list[ii] << std::endl;

		for (std::uint32_t kk = 0; kk < n_var; kk++){
			outf_inits << chromosome[kk] << " " << rsid[kk]<< " " << position[kk];
			outf_inits << " " << al_0[kk] << " " << al_1[kk];
			for (int ee = 0; ee < n_effects; ee++){
				outf_inits << " " << vp_list[ii].alpha(kk, ee);
				outf_inits << " " << vp_list[ii].mu(kk, ee);
			}
 			outf_inits << std::endl;
		}
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
		boost::filesystem::path interim_ext(ss), path(main_out_file), dir;
		dir = path.parent_path() / interim_ext;
		boost::filesystem::create_directories(dir);

		// Initialise fstreams
		fstream_init(outf_weights, dir, "_hyps", false);
		fstream_init(outf_iter, dir, "_iter_updates", false);
		fstream_init(outf_w, dir, "_env_weights", false);
		fstream_init(outf_inits, dir, "_inits", true);
		if(p.xtra_verbose){
			fstream_init(outf_alpha, dir, "_alpha", true);
		}

		for (int ll = 0; ll < n_env; ll++){
			outf_w << env_names[ll];
			if(ll + 1 < n_env) outf_w << " ";
		}
		outf_w << std::endl;
		for (int ll = 0; ll < n_env; ll++){
			outf_w << vp.muw(ll);
			if(ll + 1 < n_env) outf_w << " ";
		}
		outf_w << std::endl;

		outf_weights << "weights logw log_prior count time" << std::endl;
		outf_iter    << "count\tsigma";
		for (int ee = 0; ee < n_effects; ee++){
			outf_iter << "\tpve" << ee;
			if(p.mode_mog_prior){
				outf_iter << "\tpve_large" << ee;
 			}
			outf_iter << "\tsigma" << ee;
			if(p.mode_mog_prior){
				outf_iter << "\tsigma_spike" << ee;
			}
			outf_iter << "\tlambda" << ee;
		}
		outf_iter    << "\telbo\tmax_alpha_diff\tseconds\tHty_hits" << std::endl;

		outf_inits << "chr rsid pos a0 a1";
		for(int ee = 0; ee < n_effects; ee++){
			outf_inits << " alpha" << ee << " mu" << ee;
		}
		outf_inits << std::endl;
	}

	void fstream_init(io::filtering_ostream& my_outf,
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
			my_outf.push(io::gzip_compressor());
		}
		my_outf.push(io::file_sink(ofile.c_str()));
	}

	void resize(int n_list){
		counts_list.resize(n_list);
		vp_list.resize(n_list);
		// mu_list.resize(n_list);
		// alpha_list.resize(n_list);
		logw_updates_list.resize(n_list);
		alpha_diff_list.resize(n_list);
		logw_list.resize(n_list);
		elapsed_time_list.resize(n_list);
		hyps_list.resize(n_list);
		for (int ll = 0; ll < n_list; ll++){
			logw_list[ll] = -std::numeric_limits<double>::max();
		}
	}

	void clear(){
		counts_list.clear();
		vp_list.clear();
		// mu_list.clear();
		// alpha_list.clear();
		logw_list.clear();
		logw_updates_list.clear();
		alpha_diff_list.clear();
		elapsed_time_list.clear();
		hyps_list.clear();
	}

	void copy_ith_element(int ii, const VbTracker& other_tracker){
		counts_list[ii]       = other_tracker.counts_list[ii];
		vp_list[ii]           = other_tracker.vp_list[ii];
		// mu_list[ii]           = other_tracker.mu_list[ii];
		// alpha_list[ii]        = other_tracker.alpha_list[ii];
		logw_list[ii]         = other_tracker.logw_list[ii];
		logw_updates_list[ii] = other_tracker.logw_updates_list[ii];
		alpha_diff_list[ii]   = other_tracker.alpha_diff_list[ii];
		elapsed_time_list[ii] = other_tracker.elapsed_time_list[ii];
		hyps_list[ii]         = other_tracker.hyps_list[ii];
	}
};

#endif
