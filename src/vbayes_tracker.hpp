
#include <chrono>      // start/end time info
#include <ctime>       // start/end time info
#include <iostream>
#include <stdexcept>
#include <limits>
#include <vector>
#include "tools/eigen3.3/Dense"
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/filesystem.hpp>

class VbTracker {
public:
	std::vector< int >             counts_list;              // Number of iterations to convergence at each step
	std::vector< std::vector< double > > logw_updates_list;  // elbo updates at each ii
	std::vector< std::vector< double > > alpha_diff_list;  // elbo updates at each ii
	std::vector< Eigen::VectorXd > mu_list;                  // best mu at each ii
	std::vector< Eigen::VectorXd > alpha_list;               // best alpha at each ii
	std::vector< double >          logw_list;                // best logw at each ii
	std::vector< double >          elapsed_time_list;        // time to compute grid point

	// For writing interim output
	boost_io::filtering_ostream outf_elbo, outf_alpha_diff, outf_weights, outf_inits, outf_iter;
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
		mu_list.resize(n_list);
		alpha_list.resize(n_list);
		logw_list.resize(n_list);
		logw_updates_list.resize(n_list);
		alpha_diff_list.resize(n_list);
		elapsed_time_list.resize(n_list);

		allow_interim_push = true;
	}

	~VbTracker() {
	}

	void set_main_filepath(std::string ofile){
		main_out_file = ofile;
		allow_interim_push = true;
	}

	void push_interim_iter_update(const int cnt,
                                  const double c_logw,
                                  const double c_alpha_diff,
                                  const std::chrono::duration<double> elapsed){
		outf_iter << cnt << "\t";
		outf_iter << c_logw << "\t";
		outf_iter << c_alpha_diff << "\t";
		outf_iter << elapsed.count() << std::endl;
	}

	void push_interim_output(int ii, bool main_loop){
		// Assumes that infomration for all measures that we track have between
		// added to VbTracker at index ii.

		// Write output to file
		outf_weights << "NA" << " " << logw_list[ii] << " ";
		outf_weights << "NA" << " ";
		outf_weights << counts_list[ii] << " ";
		outf_weights << elapsed_time_list[ii] << std::endl;

		if(!main_loop){
			for (std::uint32_t kk = 0; kk < alpha_list[ii].size(); kk++){
				outf_inits << alpha_list[ii][kk] << " " << mu_list[ii][kk] << std::endl;
			}
		}
	}

	void interim_output_init(const int ii,
                             const bool main_loop){
		if(!allow_interim_push){
			throw std::runtime_error("Internal error; interim push not expected");
		}

		// Create directories
		std::string ss = "interim_files/grid_point_" + std::to_string(ii);
		if(!main_loop){
			ss = "r1_" + ss;
		}
		boost::filesystem::path interim_ext(ss), p(main_out_file), dir;
		dir = p.parent_path() / interim_ext;
		boost::filesystem::create_directories(dir);

		std::string ofile_weights, ofile_inits, ofile_iter;
		ofile_weights    = fstream_init(outf_weights, dir, "_hyps", false);
		ofile_iter       = fstream_init(outf_iter, dir, "_iter_updates", false);
		outf_weights << "weights logw log_prior count time" << std::endl;
		outf_iter    << "count\telbo\talpha_diff\tseconds" << std::endl;

		// Precision
		outf_iter       << std::setprecision(4) << std::fixed;

		if(!main_loop){
			ofile_inits     = fstream_init(outf_inits, dir, "_inits", true);
			outf_inits << "alpha mu" << std::endl;
		}
	}

	std::string fstream_init(boost_io::filtering_ostream& my_outf,
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
		my_outf.push(boost_io::file_sink(ofile.c_str()));
		return ofile;
	}


	void resize(int n_list){
		counts_list.resize(n_list);
		mu_list.resize(n_list);
		alpha_list.resize(n_list);
		logw_updates_list.resize(n_list);
		alpha_diff_list.resize(n_list);
		logw_list.resize(n_list);
		elapsed_time_list.resize(n_list);
		for (int ll = 0; ll < n_list; ll++){
			logw_list[ll] = -std::numeric_limits<double>::max();
		}
	}

	void clear(){
		counts_list.clear();
		mu_list.clear();
		alpha_list.clear();
		logw_list.clear();
		logw_updates_list.clear();
		alpha_diff_list.clear();
		elapsed_time_list.clear();
	}

	void copy_ith_element(int ii, const VbTracker& other_tracker){
		counts_list[ii]       = other_tracker.counts_list[ii];
		mu_list[ii]           = other_tracker.mu_list[ii];
		alpha_list[ii]        = other_tracker.alpha_list[ii];
		logw_list[ii]         = other_tracker.logw_list[ii];
		logw_updates_list[ii] = other_tracker.logw_updates_list[ii];
		alpha_diff_list[ii] = other_tracker.alpha_diff_list[ii];
		elapsed_time_list[ii] = other_tracker.elapsed_time_list[ii];
	}
};
