// File of Data class for use with src/bgen_prog.cpp
#ifndef DATA_H
#define DATA_H

#include "genotype_matrix.hpp"
#include "parameters.hpp"
#include "typedefs.hpp"
#include "stats_tests.hpp"
#include "eigen_utils.hpp"
#include "variational_parameters.hpp"
#include "file_utils.hpp"
#include "mpi_utils.hpp"

#include "tools/eigen3.3/Dense"
#include "tools/eigen3.3/Eigenvalues"
#include "genfile/bgen/bgen.hpp"
#include "genfile/bgen/View.hpp"

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <chrono>
#include <ctime>
#include <map>
#include <mutex>
#include <regex>
#include <vector>
#include <string>
#include <set>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/complement.hpp> // complements
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>

namespace boost_io = boost::iostreams;
namespace boost_m = boost::math;

inline void read_file_header(const std::string& filename,
                             std::vector<std::string>& col_names);

class Data
{
public:
	parameters p;

	long n_pheno;
	long n_covar;
	long n_env;
	int n_effects;
	long n_samples;
	long Nglobal;
	long n_var;
	long n_var_parsed;
	long n_dxteex_computed;
	long n_snpstats_computed;

	bool Y_reduced;
	bool W_reduced;
	bool E_reduced;

	std::vector< std::string > rsid_list;

	std::map<long, bool> missing_envs;
	std::map<long, bool> missing_covars;
	std::map<long, bool> missing_phenos;
	std::map<long, bool> incomplete_cases;
	std::map<long, int> sample_location;

	std::vector< std::string > pheno_names;
	std::vector< std::string > covar_names;
	std::vector< std::string > env_names;
	std::vector<std::string> extra_covar_pve_names;

	GenotypeMatrix G;
	EigenDataMatrix Y, Y2;
	EigenDataMatrix C;
	EigenDataMatrix E;
	Eigen::ArrayXXd dXtEEX_lowertri;
	EigenDataMatrix C_extra_pve;

	Eigen::MatrixXd resid_loco;
	std::vector<long> loco_chrs;
	std::map<long, bool> missing_resid_loco;

// Init points
	VariationalParametersLite vp_init;
	std::vector<Hyps> hyps_inits;

// For gxe genome-wide scan
	Eigen::ArrayXXd snpstats;
	Eigen::ArrayXXd external_snpstats;
	std::vector< std::string > external_snpstats_SNPID;
	bool bgen_pass;

	boost_io::filtering_ostream outf_scan;
	genfile::bgen::View::UniquePtr bgenView;
	std::vector<genfile::bgen::View::UniquePtr> streamBgenViews;

	bool filters_applied;
	std::unordered_map<long, bool> sample_is_invalid;

// grids for vbayes
	std::vector< std::string > hyps_names;
	Eigen::MatrixXd hyps_grid;
	Eigen::ArrayXXd alpha_init, mu_init;

	explicit Data(parameters& p) : p(p), G(p), vp_init(p) {
		Eigen::setNbThreads(p.n_thread);

		// Create vector of bgen views for mutlithreading
		n_var = 0;
		if(p.bgen_file != "NULL") {
			bgenView = genfile::bgen::View::create(p.bgen_file);
			n_samples = (long) bgenView->number_of_samples();
			n_var     = bgenView->number_of_variants();
		}
		for (auto streamBgenFile : p.streamBgenFiles) {
			genfile::bgen::View::UniquePtr view = genfile::bgen::View::create(streamBgenFile);
			n_samples = (long) view->number_of_samples();
			streamBgenViews.push_back(move(view));
		}
		// Check all bgen files have the same number of samples
		if(p.bgen_file != "NULL" && !p.streamBgenFiles.empty()) {
			assert(n_samples == bgenView->number_of_samples());
			for (int ii = 0; ii < p.streamBgenFiles.size(); ii++) {
				assert(n_samples == streamBgenViews[ii]->number_of_samples());
			}
		}

		n_var_parsed = 0;
		filters_applied = false;
		bgen_pass = true;

		// Explicit initial values to eliminate linter errors..
		n_pheno   = 0;
		n_effects = -1;
		n_covar   = 0;
		n_env     = 0;
		Y_reduced = false;
		W_reduced = false;
	}

	void apply_filters(){
		// filter - incl sample ids
		if(p.incl_sids_file != "NULL") {
			read_incl_sids();
		}

		// filter - range
		if (p.range) {
			std::cout << "Selecting snps in range " << p.range_chr << ":" << p.range_start << "-" << p.range_end << std::endl;
		}

		// filter - incl rsids
		if(p.incl_rsids_file != "NULL") {
			read_incl_rsids();
			std::cout << "Including SNPs from file: " << p.incl_rsids_file << std::endl;
		}

		// filter - select single rsid
		if(p.select_rsid) {
			std::sort(p.rsid.begin(), p.rsid.end());
			std::cout << "Filtering to rsids:" << std::endl;
			long int n_rsids = p.rsid.size();
			for (long int kk = 0; kk < n_rsids; kk++) {
				std::cout << p.rsid[kk]<< std::endl;
			}
		}

		if(p.bgen_file != "NULL") {
			genfile::bgen::IndexQuery::UniquePtr query = genfile::bgen::IndexQuery::create(p.bgi_file);
			if (p.range) {
				genfile::bgen::IndexQuery::GenomicRange rr1(p.range_chr, p.range_start, p.range_end);
				query->include_range( rr1 );
			}
			if(p.incl_rsids_file != "NULL") {
				query->include_rsids( rsid_list );
			}
			if(p.select_rsid) {
				query->include_rsids( p.rsid );
			}

			// filter - apply queries
			query->initialise();
			bgenView->set_query(query);
		}

		for (int ii = 0; ii < p.streamBgenFiles.size(); ii++) {
			genfile::bgen::IndexQuery::UniquePtr query = genfile::bgen::IndexQuery::create(p.streamBgiFiles[ii]);
			if (p.range) {
				genfile::bgen::IndexQuery::GenomicRange rr1(p.range_chr, p.range_start, p.range_end);
				query->include_range( rr1 );
			}
			if(p.incl_rsids_file != "NULL") {
				query->include_rsids( rsid_list );
			}
			if(p.select_rsid) {
				query->include_rsids( p.rsid );
			}
			query->initialise();
			streamBgenViews[ii]->set_query(query);
		}

		if (p.bgen_file != "NULL"){
			bgenView->summarise(std::cout);
		}

		filters_applied = true;
	}

	void read_non_genetic_data(){
		// Apply sample / rsid / range filters if applicable
		if(!filters_applied) {
			apply_filters();
		}

		// Read in phenotypes
		if (p.pheno_file != "NULL") {
			read_pheno();
		}

		// Read in covariates if present
		if(p.covar_file != "NULL") {
			read_covar();
		}

		if (p.extra_pve_covar_file != "NULL" && p.mode_RHE) {
			read_extra_pve_covar();
		}

		// Environmental vars - subset of covars
		if(p.env_file != "NULL") {
			read_environment();
			p.interaction_analysis = true;
		}

		if(!p.mode_no_gxe && p.interaction_analysis) {
			n_effects = 2;
		} else {
			n_effects = 1;
		}

		// Exclude samples with missing values in phenos / covars / filters
		reduce_to_complete_cases();

		// Read in hyperparameter values
		if(p.hyps_grid_file != "NULL") {
			read_hyps();
		}

		if(p.snpstats_file != "NULL") {
			read_external_snpstats();
		}

		if(p.resid_loco_file != "NULL") {
			read_resid_loco();
		}
	}

	void standardise_non_genetic_data(){
		// Step 3; Center phenos, normalise covars
		if(n_covar > 0) {
			EigenUtils::center_matrix(C);
			EigenUtils::scale_matrix_and_remove_constant_cols(C, n_covar, covar_names);
		}
		if(n_env > 0) {
			EigenUtils::center_matrix(E);
			EigenUtils::scale_matrix_and_remove_constant_cols(E, n_env, env_names);
		}

		/* Regression rules
		   1 - if use_VB_on_covars then:
		        - E main effects kept in model
		        - C regressed from Y
		   2 - if !use_VB_on_covars then:
		        - [C, E] regressed from Y

		   if any of E have squared dependance on Y then remove
		 */

		if(n_pheno > 0) {
			EigenUtils::center_matrix(Y);
			EigenUtils::scale_matrix_and_remove_constant_cols(Y, n_pheno, pheno_names);
			if (n_env > 0) {
				if (n_covar == 0) {
					C = E;
					covar_names = env_names;
					n_covar = n_env;
				} else if (p.use_vb_on_covars) {
					EigenDataMatrix tmp(E.rows(), C.cols() + E.cols());
					tmp << C, E;
					C = tmp;
					covar_names.insert(covar_names.end(), env_names.begin(), env_names.end());
					n_covar += n_env;
				} else if (n_covar > 0) {
					regress_first_mat_from_second(C, "covars", covar_names, Y, "pheno");
					regress_first_mat_from_second(C, "covars", covar_names, E, "env");

					C = E;
					covar_names = env_names;
					n_covar = n_env;
				}

				// Removing squared dependance
				std::vector<int> cols_to_remove;
				Eigen::MatrixXd H;
				Eigen::VectorXd tmp = Eigen::VectorXd::Zero(n_samples);
				if (p.exclude_ones_from_env_sq) {
					H.resize(n_samples, n_covar + 1);
					H << tmp, C;
				} else {
					Eigen::MatrixXd ones = Eigen::MatrixXd::Constant(n_samples, 1, 1.0);
					H.resize(n_samples, n_covar + 2);
					H << tmp, ones, C;
				}

				if (p.verbose) std::cout << "Checking for squared dependance: " << std::endl;
				if (p.verbose) std::cout << "Name\t-log10(p-val)" << std::endl;
				long n_signif_envs_sq = 0;
				for (int ee = 0; ee < n_env; ee++) {
					// H.col(0) = E.col(ee);
					double rss_alt, tstat, pval;
					try {
						H.col(0) = E.col(ee).array().square().matrix();
						Eigen::MatrixXd HtH(H.cols(), H.cols()), Hty(H.cols(), 1), HtH_inv(H.cols(), H.cols());
						prep_lm(H, Y, HtH, HtH_inv, Hty, rss_alt);
						student_t_test(n_samples, HtH_inv, Hty, rss_alt, 0, tstat, pval);

						if (p.verbose) std::cout << env_names[ee] << "\t";
						if (p.verbose) std::cout << -1 * std::log10(pval) << std::endl;

						if (pval < 0.01 / (double) n_env) {
							cols_to_remove.push_back(ee);
							n_signif_envs_sq++;
						}
					} catch (const std::exception &e) {
						if (p.xtra_verbose) {
							std::cout << "Error when checking for significant squared envs ";
							std::cout << "(I believe this is due to attempting linear regression ";
							std::cout << "with a singular matrix)" << std::endl;
							std::cout << e.what() << std::endl;
						}
					}
				}
				if (n_signif_envs_sq > 0) {
					Eigen::MatrixXd E_sq(n_samples, n_signif_envs_sq);
					std::vector<std::string> env_sq_names;
					for (int nn = 0; nn < n_signif_envs_sq; nn++) {
						E_sq.col(nn) = E.col(cols_to_remove[nn]).array().square();
						env_sq_names.push_back(env_names[cols_to_remove[nn]] + "_sq");
					}

					if (p.mode_incl_squared_envs) {
						std::cout << "Including the squared effects from " << cols_to_remove.size();
						std::cout << " environmental variables:" << std::endl;
						for (int ee : cols_to_remove) {
							std::cout << env_names[ee] << std::endl;
						}

						EigenUtils::center_matrix(E_sq);
						EigenUtils::scale_matrix_and_remove_constant_cols(E_sq, n_signif_envs_sq, env_sq_names);

						EigenDataMatrix tmp(n_samples, n_covar + n_signif_envs_sq);
						tmp << C, E_sq;
						C = tmp;
						covar_names.insert(covar_names.end(), env_sq_names.begin(), env_sq_names.end());
						n_covar += n_signif_envs_sq;

						assert(n_covar == covar_names.size());
						assert(n_covar == C.cols());
					} else if (p.mode_remove_squared_envs) {
						std::cout << "Projecting out squared dependance from: " << std::endl;
						for (int ee : cols_to_remove) {
							std::cout << env_names[ee] << std::endl;
						}

						Eigen::MatrixXd H(n_samples, n_covar + n_signif_envs_sq);
						H << E_sq, C;

						Eigen::MatrixXd HtH, Hty;
						HtH = H.transpose() * H;
						HtH = mpiUtils::mpiReduce_inplace(HtH);
						Hty = H.transpose() * Y;
						Hty = mpiUtils::mpiReduce_inplace(Hty);
						Eigen::MatrixXd beta = HtH.colPivHouseholderQr().solve(Hty);

						Y -= E_sq * beta.block(0, 0, n_signif_envs_sq, 1);
					} else {
						std::cout << "Warning: Environments with significant squared effects detected (";
						for (const auto &env_sq_name : env_sq_names) {
							std::cout << env_sq_name << ", ";
						}
						std::cout << ") but mitigation suppressed" << std::endl;
					}
				}

				// For internal use
				Y2 = Y;
				regress_first_mat_from_second(C, Y2);
			} else {
				Y2 = Y;
			}
		}
	}

	void read_full_bgen(){
		if(p.bgen_file != "NULL") {
			std::cout << "Reading in BGEN data" << std::endl;
			if (p.flip_high_maf_variants) {
				std::cout << " - Flipping variants with MAF > 0.5" << std::endl;
			}

			auto start = std::chrono::system_clock::now();
			p.chunk_size = bgenView->number_of_variants();
			fileUtils::read_bgen_chunk(bgenView, G, sample_is_invalid, n_samples, p.chunk_size, p, bgen_pass,
			                           n_var_parsed);
			auto end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed = end - start;
			n_var = G.cols();
			std::cout << " - BGEN file contained " << n_var << " valid variants." << std::endl;
			std::cout << " - BGEN file parsed in " << elapsed.count() << "s" << std::endl;

			G.calc_scaled_values();
			G.compute_cumulative_pos();
			if (p.debug) std::cout << " - Computed colwise mean and sd of genetic data" << std::endl << std::endl;

			// Set default hyper-parameters if not read from file
			// Run read_hyps twice as some settings depend on n_var
			// which changes depending on how many variants excluded due to maf etc.
			if (p.hyps_grid_file != "NULL") {
				read_hyps();
			} else if (p.hyps_grid_file == "NULL") {
				if (p.debug) std::cout << "Initialising hyper-parameters with default settings" << std::endl;
				Hyps hyps(p);
				hyps.use_default_init(n_effects, n_var);
				hyps_inits.push_back(hyps);
			}
		}
	}

	void read_incl_rsids(){
		boost_io::filtering_istream fg;
		std::string gz_str = ".gz";
		if (p.incl_rsids_file.find(gz_str) != std::string::npos) {
			fg.push(boost_io::gzip_decompressor());
		}
		fg.push(boost_io::file_source(p.incl_rsids_file));
		if (!fg) {
			std::cout << "ERROR: " << p.incl_rsids_file << " not opened." << std::endl;
			std::exit(EXIT_FAILURE);
		}

		std::stringstream ss;
		std::string line;
		while (getline(fg, line)) {
			ss.clear();
			ss.str(line);
			std::string s;
			while(ss >> s) {
				rsid_list.push_back(s);
			}
		}

		std::sort(rsid_list.begin(), rsid_list.end());
		rsid_list.erase(std::unique(rsid_list.begin(), rsid_list.end()), rsid_list.end());
	}

	void read_incl_sids(){
		boost_io::filtering_istream fg;
		std::string gz_str = ".gz";
		if (p.incl_sids_file.find(gz_str) != std::string::npos) {
			fg.push(boost_io::gzip_decompressor());
		}
		fg.push(boost_io::file_source(p.incl_sids_file));
		if (!fg) {
			std::cout << "ERROR: " << p.incl_sids_file << " not opened." << std::endl;
			std::exit(EXIT_FAILURE);
		}

		std::vector<std::string> bgen_ids;
		if(p.bgen_file != "NULL") {
			bgenView->get_sample_ids(
				[&]( std::string const& id ) {
				bgen_ids.push_back(id);
			}
				);
		} else if (!p.streamBgenFiles.empty()) {
			streamBgenViews[0]->get_sample_ids(
				[&]( std::string const& id ) {
				bgen_ids.push_back(id);
			}
				);
		} else {
			std::runtime_error("No valid bgen file found.");
		}

		std::stringstream ss;
		std::string line;
		std::set<std::string> user_sample_ids;
		while (getline(fg, line)) {
			ss.clear();
			ss.str(line);
			std::string s;
			ss >> s;
			user_sample_ids.insert(s);
		}

		std::set<std::string>::iterator it;
		for (long ii = 0; ii < n_samples; ii++) {
			it = user_sample_ids.find(bgen_ids[ii]);
			if (it == user_sample_ids.end()) {
				incomplete_cases[ii] = true;
			}
		}
		std::cout << "Subsetted down to " << user_sample_ids.size() << " ids from file: " << p.incl_sids_file;
		std::cout << std::endl;
	}

	void read_vb_init_file(const std::string& filename,
	                       Eigen::MatrixXd& M,
	                       std::vector< std::string >& col_names,
	                       std::vector< std::string >& init_key){
		// Need custom function to deal with variable input. Sometimes
		// we have string columns with rsid / a0 etc
		// init_chr, init_pos, init_a0, init_a1;

		boost_io::filtering_istream fg;
		std::string gz_str = ".gz";
		if (filename.find(gz_str) != std::string::npos) {
			fg.push(boost_io::gzip_decompressor());
		}
		fg.push(boost_io::file_source(filename));
		if (!fg) {
			std::cout << "ERROR: " << filename << " not opened." << std::endl;
			std::exit(EXIT_FAILURE);
		}

		// Read file twice to acertain number of lines
		std::string line;
		int n_grid = 0;
		getline(fg, line);
		while (getline(fg, line)) {
			n_grid++;
		}
		fg.reset();
		if (filename.find(gz_str) != std::string::npos) {
			fg.push(boost_io::gzip_decompressor());
		}
		fg.push(boost_io::file_source(filename));

		// Reading column names
		col_names.clear();
		if (!getline(fg, line)) {
			std::cout << "ERROR: " << filename << " contains zero lines." << std::endl;
			std::exit(EXIT_FAILURE);
		}
		std::stringstream ss;
		std::string s;
		int n_cols = 0;
		ss.clear();
		ss.str(line);
		while (ss >> s) {
			++n_cols;
			col_names.push_back(s);
		}
		std::cout << " Reading matrix of size " << n_grid << " x " << n_cols;
		std::cout << " from " << filename << std::endl;

		// Write remainder of file to Eigen matrix M
		M.resize(n_grid, n_cols);
		int i = 0;
		double tmp_d;
		std::string key_i;
		assert(n_cols == 7);
		try {
			while (getline(fg, line)) {
				if (i >= n_grid) {
					throw std::runtime_error("ERROR: could not convert txt file (too many lines).");
				}
				ss.clear();
				ss.str(line);
				key_i = "";
				for (int k = 0; k < n_cols; k++) {
					std::string sss;
					ss >> sss;
					if(k == 0 || k == 2 || k == 3) {
						key_i += sss + "~";
					} else if (k == 4) {
						key_i += sss;
						init_key.push_back(key_i);
					} else if(k >= 5) {
						M(i, k) = stod(sss);
					}
				}
				// loop should end at i == n_grid
				i++;
			}
			if (i < n_grid) {
				throw std::runtime_error("ERROR: could not convert txt file (too few lines).");
			}
		} catch (const std::exception &exc) {
			throw;
		}
	}

	void read_pheno( ){
		Eigen::MatrixXd tmpY;
		EigenUtils::read_matrix(p.pheno_file, tmpY, pheno_names, missing_phenos);
		if (tmpY.cols() > 1){
			if (p.pheno_col_num == -1) {
				throw std::runtime_error("Multiple phenotypes detected; specify one with --pheno-col-num");
			} else {
				Y = tmpY.col(p.pheno_col_num);
			}
		} else {
			Y = tmpY;
		}
		assert(Y.cols() == 1);
		assert(Y.rows() == n_samples);
		n_pheno = Y.cols();
		Y_reduced = false;
	}

	void read_resid_loco(){
		std::vector<std::string> resid_loco_names;
		EigenUtils::read_matrix(p.resid_loco_file, resid_loco, resid_loco_names, missing_resid_loco);
		loco_chrs.clear();
		std::vector<int> keep_cols;
		for (long cc = 0; cc < resid_loco_names.size(); cc++){
			try {
				std::string chr = std::regex_replace(resid_loco_names[cc], std::regex("resid_excl_chr([0-9]+).*"), std::string("$1"));
				loco_chrs.push_back(std::stoi(chr));
				keep_cols.push_back(cc);
			} catch (...) {}
		}
		for (long cc = 0; cc < keep_cols.size(); cc++){
			resid_loco.col(cc) = resid_loco.col(keep_cols[cc]);
		}
		std::cout << " - detected residualised phenotype with " << keep_cols.size() << " LOCO partitions" << std::endl;
	}

	void read_covar( ){
		// Read covariates to Eigen matrix C
		EigenUtils::read_matrix(p.covar_file, C, covar_names, missing_covars);
		assert(C.rows() == n_samples);
		n_covar = C.cols();
		W_reduced = false;
	}

	void read_extra_pve_covar( ){
		// Read covariates to Eigen matrix C
		EigenUtils::read_matrix(p.extra_pve_covar_file, C_extra_pve, extra_covar_pve_names);
	}

	void read_environment( ){
		// Read covariates to Eigen matrix C
		EigenUtils::read_matrix(p.env_file, E, env_names, missing_envs);
		assert(E.rows() == n_samples);
		n_env = E.cols();
		E_reduced = false;
	}

	void read_external_snpstats( ){
		std::vector< std::string > col_names;
		read_txt_file_w_context( p.snpstats_file, 8, external_snpstats,
		                         external_snpstats_SNPID, col_names);

		if(external_snpstats.cols() != n_env + 3) {
			std::cout << "Expecting columns in order: " << std::endl;
			std::cout << "SNPID, chr, rsid, pos, allele0, allele1, maf, info, snpstats.." << std::endl;
			throw std::runtime_error("Unexpected number of columns");
		}
	}

	void read_txt_file_w_context( const std::string& filename,
	                              const int& col_offset,
	                              Eigen::ArrayXXd& M,
	                              std::vector<std::string>& M_snpids,
	                              std::vector<std::string>& col_names){
		/*
		 * Txt file where the first column is snp ids, then x-1 contextual,
		 * then a matrix to be read into memory.
		 * Reads file twice to ascertain number of lines.
		 * col_offset - how many contextual columns to skip
		 */

		// Reading from file
		boost_io::filtering_istream fg;
		std::string gz_str = ".gz";
		if (filename.find(gz_str) != std::string::npos) {
			fg.push(boost_io::gzip_decompressor());
		}
		fg.push(boost_io::file_source(filename));
		if (!fg) {
			std::cout << "ERROR: " << filename << " not opened." << std::endl;
			std::exit(EXIT_FAILURE);
		}

		// Read file twice to ascertain number of lines
		int n_lines = 0;
		std::string line;
		// skip header
		getline(fg, line);
		while (getline(fg, line)) {
			n_lines++;
		}
		fg.reset();
		if (filename.find(gz_str) != std::string::npos) {
			fg.push(boost_io::gzip_decompressor());
		}
		fg.push(boost_io::file_source(filename));

		// Reading column names
		col_names.clear();
		if (!getline(fg, line)) {
			std::cout << "ERROR: " << filename << " contains zero lines" << std::endl;
			std::exit(EXIT_FAILURE);
		}
		std::string s;
		int n_cols = 0;
		col_names.clear();
		std::stringstream ss(line);
		while (ss >> s) {
			++n_cols;
			col_names.push_back(s);
		}
		assert(n_cols > col_offset);

		// Write remainder of file to Eigen matrix M
		M.resize(n_lines, n_cols - col_offset);
		int i = 0;
		double tmp_d;
		while (getline(fg, line)) {
			ss.clear();
			ss.str(line);
			for (int k = 0; k < n_cols; k++) {
				std::string sss;
				ss >> sss;
				if (k == 0) {
					M_snpids.push_back(sss);
				}
				if (k >= col_offset) {
					try{
						M(i, k-col_offset) = stod(sss);
					} catch (const std::invalid_argument &exc) {
						std::cout << "Found value " << sss << " on line " << i;
						std::cout << " of file " << filename << std::endl;
						throw std::runtime_error("Unexpected value");
					}
				}
			}
			// loop should end at i == n_samples
			i++;
		}
		std::cout << n_lines << " rows found in " << filename << std::endl;
	}

	void parse_dxteex_header(const int& col_offset,
	                         boost_io::filtering_istream& fg,
	                         std::vector<std::string>& col_names){

		// Reading from file
		std::string gz_str = ".gz";
		if (p.dxteex_file.find(gz_str) != std::string::npos) {
			fg.push(boost_io::gzip_decompressor());
		}
		fg.push(boost_io::file_source(p.dxteex_file));
		if (!fg) {
			std::cout << "ERROR: " << p.dxteex_file << " not opened." << std::endl;
			std::exit(EXIT_FAILURE);
		}

		// Reading column names
		std::string s, line;
		if (!getline(fg, line)) {
			std::cout << "ERROR: " << p.dxteex_file << " contains zero lines" << std::endl;
			std::exit(EXIT_FAILURE);
		}
		int n_cols = 0;
		col_names.clear();
		std::stringstream ss(line);
		while (ss >> s) {
			++n_cols;
			col_names.push_back(s);
		}
		assert(n_cols > col_offset);
	}

	bool read_dxteex_line(const int& col_offset,
	                      boost_io::filtering_istream& fg,
	                      Eigen::ArrayXd& M,
	                      const long n_cols,
	                      std::string& snpid,
	                      long& ii){

		std::string line;
		bool file_ongoing = (bool) getline(fg, line);
		if(file_ongoing) {
			std::stringstream ss(line);
			for (int k = 0; k < n_cols; k++) {
				std::string sss;
				ss >> sss;
				if (k == 0) {
					snpid = sss;
				}
				if (k >= col_offset) {
					try {
						M(k - col_offset) = stod(sss);
					} catch (const std::invalid_argument &exc) {
						std::cout << "Found value " << sss << " on line " << ii;
						std::cout << " of file " << p.dxteex_file << std::endl;
						throw std::runtime_error("Unexpected value");
					}
				}
			}
			ii++;
		}
		return file_ongoing;
	}

	void calc_snpstats(){
		std::cout << "Reordering/computing snpwise scan...";
		auto start = std::chrono::system_clock::now();
		snpstats.resize(n_var, n_env + 3);
		std::vector<std::string>::iterator it;
		n_snpstats_computed = 0;
		auto N = (double) n_samples;
		boost_m::fisher_f f_dist(n_env, n_samples - n_env - 1);
		for (std::size_t jj = 0; jj < n_var; jj++) {
			it = std::find(external_snpstats_SNPID.begin(), external_snpstats_SNPID.end(), G.SNPID[jj]);
			if (it == external_snpstats_SNPID.end()) {
				n_snpstats_computed++;
				EigenDataVector X_kk = G.col(jj);
				EigenDataMatrix H(n_samples, 1 + n_env);
				H << X_kk, (E.array().colwise() * X_kk.array()).matrix();

				Eigen::MatrixXd HtH = (H.transpose() * H).cast<double>();
				Eigen::MatrixXd Hty = (H.transpose() * Y2).cast<double>();

				// Fitting regression models
				EigenDataMatrix tau1_j = X_kk.transpose() * Y2 / (N-1.0);
				Eigen::MatrixXd tau2_j = EigenUtils::solve(HtH, Hty);
				double rss_null = (Y2 - X_kk * tau1_j).squaredNorm();
#ifdef DATA_AS_FLOAT
				double rss_alt  = (Y2 - H * tau2_j.cast<float>()).squaredNorm();
#else
				double rss_alt  = (Y2 - H * tau2_j).squaredNorm();
#endif
				// T-test; main effect of variant j
				boost_m::students_t t_dist(n_samples - 1);
				double main_se_j    = std::sqrt(rss_null) / (N - 1.0);
				double main_tstat_j = tau1_j(0, 0) / main_se_j;
				double main_pval_j  = 2 * boost_m::cdf(boost_m::complement(t_dist, fabs(main_tstat_j)));

				// F-test; joint interaction effect of variant j
				double f_stat        = (rss_null - rss_alt) / (double) n_env;
				f_stat              /= rss_alt / (double) (n_samples - n_env - 1);
				double gxe_pval_j    = boost_m::cdf(boost_m::complement(f_dist, fabs(f_stat)));
				double gxe_neglogp_j = -1 * std::log10(gxe_pval_j);
				if(!std::isfinite(gxe_neglogp_j) || f_stat < 0) {
					std::cout << "Warning: neglog-p = " << gxe_neglogp_j << std::endl;
					std::cout << "Warning: p-val = "    << gxe_pval_j << std::endl;
					std::cout << "Warning: rss_null = " << rss_null << std::endl;
					std::cout << "Warning: rss_alt = "  << rss_alt << std::endl;
					std::cout << "Warning: f_stat = "   << f_stat << std::endl;
				}

				// Log relevant stats
				snpstats(jj, 1)    = -1 * std::log10(gxe_pval_j);
				snpstats(jj, 0)    = -1 * std::log10(main_pval_j);
				for (int ee = 0; ee < n_env + 1; ee++) {
					snpstats(jj, ee + 2) = tau2_j(ee);
				}
			} else {
				snpstats.row(jj) = external_snpstats.row(it - external_snpstats_SNPID.begin());
			}
		}
		std::cout << " (" << n_snpstats_computed << " computed from raw data, ";
		std::cout << n_var - n_snpstats_computed << " read from file)" << std::endl;

		if(p.snpstats_file == "NULL") {
			std::string ofile_scan = fstream_init(outf_scan, "", "_snpwise_scan");
			std::cout << "Writing snp-wise scan to file " << ofile_scan << std::endl;

			outf_scan << "chr rsid pos a0 a1 af info neglogp_main neglogp_gxe";
			outf_scan << std::endl;
			for (long kk = 0; kk < n_var; kk++) {
				outf_scan << G.chromosome[kk] << " " << G.rsid[kk] << " " << G.position[kk];
				outf_scan << " " << G.al_0[kk] << " " << G.al_1[kk];
				outf_scan << " " << G.maf[kk] << " " << G.info[kk];
				outf_scan << " " << snpstats(kk, 0) << " " << snpstats(kk, 1);
				outf_scan << std::endl;
			}
			boost_io::close(outf_scan);
		}

		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		std::cout << "snpwise scan constructed in " << elapsed.count() << " seconds" << std::endl;
	}

	void calc_dxteex(){
		int world_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		std::cout << "Building dXtEEX array" << std::endl;
		auto start = std::chrono::system_clock::now();
		dXtEEX_lowertri.resize(n_var, n_env * (n_env + 1) / 2);

		// Unfilled snp indexes;
		std::unordered_set<long> unfilled_indexes;
		for (long jj = 0; jj < n_var; jj++) {
			unfilled_indexes.insert(jj);
		}

		if(p.dxteex_file != "NULL") {
			long nNotFound = 0;
			boost_io::filtering_istream fg;
			std::vector< std::string > col_names;
			parse_dxteex_header(6, fg, col_names);
			long n_cols = col_names.size();

			std::string snpid;
			Eigen::ArrayXd dxteex_row(n_env * n_env);
			int dxteex_check = 0, se_cnt = 0;
			double mean_ae = 0, max_ae = 0;
			long ii = 0;
			std::cout << " - processing precomputed entries from " << p.dxteex_file << std::endl;
			while(read_dxteex_line(6, fg, dxteex_row, n_cols, snpid, ii)) {
				auto it = std::find(G.SNPID.begin(), G.SNPID.end(), snpid);
				if(it == G.SNPID.end()) {
					nNotFound++;
				} else {
					long jj = it - G.SNPID.begin();
					for (int ll = 0; ll < n_env; ll++) {
						for (int mm = 0; mm <= ll; mm++) {
							dXtEEX_lowertri(jj, dXtEEX_col_ind(ll, mm, n_env)) = dxteex_row(ll * n_env + mm);
						}
					}
					unfilled_indexes.erase(jj);

					// Double check external dXtEEX
					if(dxteex_check < 100) {
						dxteex_check++;
						EigenDataArrayX cl_j = G.col(jj);
						for (int ll = 0; ll < n_env; ll++) {
							for (int mm = 0; mm <= ll; mm++) {
								double dztz_lmj = (cl_j * E.array().col(ll) * E.array().col(mm) * cl_j).sum();
								dztz_lmj = mpiUtils::mpiReduce_inplace(&dztz_lmj);
								// if(world_rank == 0) {
								double x1 = std::abs(dXtEEX_lowertri(jj, dXtEEX_col_ind(ll, mm, n_env)) - dztz_lmj);
								max_ae = std::max(x1, max_ae);
								mean_ae += x1;
								// }
								se_cnt++;
							}
						}
					} else if (dxteex_check == 100) {
						dxteex_check++;
						mean_ae /= (double) se_cnt;
						if (p.debug) {
							std::cout << " -- Double checking data for the first 100 SNPs suggests ";
							std::cout << "max absolute error = " << max_ae << ", ";
							std::cout << "mean absolute error = " << mean_ae << std::endl;
						}
					}
				}
			}
			if (nNotFound > 0 && p.verbose) {
				std::cout << " -- " << nNotFound << " SNPIDs from " << p.dxteex_file;
				std::cout << " not present in bgen file" << std::endl;

			}
		}

		// Compute correlations not available in file
		n_dxteex_computed = 0;
		for (auto jj : unfilled_indexes) {
			n_dxteex_computed++;
			EigenDataArrayX cl_j = G.col(jj);
			for (int ll = 0; ll < n_env; ll++) {
				for (int mm = 0; mm <= ll; mm++) {
					double dztz_lmj = (cl_j * E.array().col(ll) * E.array().col(mm) * cl_j).sum();
					dztz_lmj = mpiUtils::mpiReduce_inplace(&dztz_lmj);
					// if(world_rank == 0) {
					dXtEEX_lowertri(jj, dXtEEX_col_ind(ll, mm, n_env)) = dztz_lmj;
					// }
				}
			}
		}

		std::cout << " - entries for " << n_dxteex_computed << " variants computed from raw data" << std::endl;
		std::cout << " - entries for " << n_var - n_dxteex_computed << " variants read from file" << std::endl;
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		std::cout << " - dXtEEX array constructed in " << elapsed.count() << " seconds" << std::endl;
	}

	void read_hyps(){
		// For use in vbayes object
		hyps_inits.clear();
		hyps_names.clear();

		std::vector< std::string > case1 = {"sigma", "sigma_b", "lambda_b"};
		std::vector< std::string > case2 = {"sigma", "sigma_b", "sigma_g", "lambda_b", "lambda_g"};
		std::vector< std::string > case3 = {"hyp", "value"};
		std::vector< std::string > case4c = {"h_b", "lambda_b", "f1_b"};
		std::vector< std::string > case4d = {"h_b", "lambda_b", "f1_b", "h_g", "lambda_g", "f1_g"};

		read_file_header(p.hyps_grid_file, hyps_names);
		int n_hyps_removed = 0;

		if(std::includes(hyps_names.begin(), hyps_names.end(), case1.begin(), case1.end())) {
			assert(p.interaction_analysis);
			EigenUtils::read_matrix(p.hyps_grid_file, hyps_grid, hyps_names);
			long n_grid = hyps_grid.rows();

			// Unpack from grid to hyps object
			for (int ii = 0; ii < n_grid; ii++) {
				Hyps hyps(p);
				hyps.init_from_grid(n_effects, ii, n_var, hyps_grid);
				if (hyps.domain_is_valid()) {
					hyps_inits.push_back(hyps);
				} else {
					n_hyps_removed++;
				}
			}
		} else if(std::includes(hyps_names.begin(), hyps_names.end(), case2.begin(), case2.end())) {
			// If not interaction analysis check interaction hyps are zero
			EigenUtils::read_matrix(p.hyps_grid_file, hyps_grid, hyps_names);
			long n_grid = hyps_grid.rows();

			if(!p.interaction_analysis) {
				double sigma_g_sum  = hyps_grid.col(2).array().abs().sum();
				double lambda_g_sum = hyps_grid.col(4).array().abs().sum();
				if (sigma_g_sum > 1e-6 || lambda_g_sum > 1e-6) {
					std::cout << "WARNING: You have non-zero hyperparameters for interaction effects,";
					std::cout << " but no environmental variables provided." << std::endl;
				}
			}

			// Unpack from grid to hyps object
			for (int ii = 0; ii < n_grid; ii++) {
				Hyps hyps(p);
				hyps.init_from_grid(n_effects, ii, n_var, hyps_grid);
				if (hyps.domain_is_valid()) {
					hyps_inits.push_back(hyps);
				} else {
					n_hyps_removed++;
				}
			}
		} else if (hyps_names == case3) {
			// Reading from hyps dump
			Hyps hyps(p);
			hyps.read_from_dump(p.hyps_grid_file);
			assert(hyps.n_effects == n_effects);
			hyps_inits.push_back(hyps);
		} else if(hyps_names == case4c) {
			// h_b lambda_b f1_b
			EigenUtils::read_matrix(p.hyps_grid_file, hyps_grid, hyps_names);
			long n_grid = hyps_grid.rows();

			for (int ii = 0; ii < n_grid; ii++) {
				Hyps hyps(p);
				hyps.resize(1);
				hyps.sigma = 1 - hyps_grid(ii, 0);

				Eigen::VectorXd soln = hyps.get_sigmas(hyps_grid(ii, 0), hyps_grid(ii, 1), hyps_grid(ii, 2), n_var);
				hyps.slab_var << hyps.sigma * soln[0];
				hyps.spike_var << hyps.sigma * soln[1];
				hyps.lambda << hyps_grid(ii, 1);
				hyps.s_x << n_var;

				hyps.slab_relative_var << hyps.slab_var / hyps.sigma;
				hyps.spike_relative_var << hyps.spike_var / hyps.sigma;
				if(hyps.domain_is_valid()) {
					hyps_inits.push_back(hyps);
				} else {
					n_hyps_removed++;
				}
			}
		} else if(hyps_names == case4d) {
			// h_b lambda_b f1_b h_g lambda_g f1_g
			assert(p.interaction_analysis);
			EigenUtils::read_matrix(p.hyps_grid_file, hyps_grid, hyps_names);
			long n_grid = hyps_grid.rows();

			for (int ii = 0; ii < n_grid; ii++) {
				Hyps hyps(p);
				hyps.resize(2);
				hyps.sigma = 1 - hyps_grid(ii, 0) - hyps_grid(ii, 3);

				Eigen::VectorXd soln = hyps.get_sigmas(hyps_grid(ii, 0), hyps_grid(ii, 3), hyps_grid(ii, 1),
				                                       hyps_grid(ii, 4), hyps_grid(ii, 2), hyps_grid(ii, 5), n_var);
				hyps.slab_var << hyps.sigma * soln[0], hyps.sigma * soln[2];
				hyps.spike_var << hyps.sigma * soln[1], hyps.sigma * soln[3];
				hyps.lambda << hyps_grid(ii, 1), hyps_grid(ii, 4);
				hyps.s_x << n_var, n_var;

				hyps.slab_relative_var << hyps.slab_var / hyps.sigma;
				hyps.spike_relative_var << hyps.spike_var / hyps.sigma;
				if(hyps.domain_is_valid()) {
					hyps_inits.push_back(hyps);
				} else {
					n_hyps_removed++;
				}
			}
		}

		if(n_hyps_removed > 0) {
			std::cout << "WARNING: " << n_hyps_removed;
			std::cout << " invalid grid points removed from hyps_grid." << std::endl;
		}

		// // Verify header of grid file as expected
		// if(params.interaction_analysis) {
		//  if(!std::includes(hyps_names.begin(), hyps_names.end(), case2.begin(), case2.end())) {
		//      throw std::runtime_error("Column names of --hyps_grid must be sigma sigma_b sigma_g lambda_b lambda_g");
		//  }
		// } else {
		//  if(hyps_names != case1) {
		//
		//      if(std::includes(hyps_names.begin(), hyps_names.end(), case2.begin(), case2.end())) {
		//          double sigma_g_sum  = hyps_grid.col(2).array().abs().sum();
		//          double lambda_g_sum = hyps_grid.col(4).array().abs().sum();
		//          if(sigma_g_sum > 1e-6 || lambda_g_sum > 1e-6) {
		//              std::cout << "WARNING: You have non-zero hyperparameters for interaction effects,";
		//              std::cout << " but no environmental variables provided." << std::endl;
		//          }
		//      } else {
		//          throw std::runtime_error("Column names of --hyps_grid must be sigma sigma_b lambda_b or sigma sigma_b sigma_g lambda_b lambda_g");
		//      }
		//  }
		// }
		//
		// // Option to provide separate grid to evaluate in round 1
		// std::vector< std::string > r1_hyps_names, r1_probs_names;
		// if ( params.r1_hyps_grid_file != "NULL" ) {
		//  EigenUtils::read_matrix(params.r1_hyps_grid_file, r1_hyps_grid, r1_hyps_names);
		//  if(hyps_names != r1_hyps_names) {
		//      throw std::invalid_argument( "Header of --r1_hyps_grid must match --hyps_grid." );
		//  }
		// }
	}

	void assign_vb_init_from_file(VariationalParametersLite& vp_init){
		// Different filetypes
		std::vector<std::string> case1 = {"alpha", "mu"};
		std::vector<std::string> case2 = {"chr", "rsid", "pos", "a0", "a1", "beta", "gamma"};
		std::vector<std::string> latents1 = {"alpha_beta", "mu1_beta", "s1_beta", "mu2_beta", "s2_beta"};
		std::vector<std::string> latents2 = {"alpha_gam", "mu1_gam", "s1_gam", "mu2_gam", "s2_gam"};
		std::vector<std::string> case3a = {"SNPID"}, case3b = {"SNPID"};
		case3b.insert(case3b.end(),latents1.begin(),latents1.end());
		case3b.insert(case3b.end(),latents2.begin(),latents2.end());
		case3a.insert(case3a.end(),latents1.begin(),latents1.end());

		std::vector<std::string> vb_init_colnames;
		read_file_header(p.vb_init_file, vb_init_colnames);

		Eigen::MatrixXd vb_init_mat;
		std::vector< std::string > init_key;
		if(vb_init_colnames == case1) {
			EigenUtils::read_matrix(p.vb_init_file, vb_init_mat, vb_init_colnames);
			assert(p.bgen_file == "NULL" || vb_init_mat.rows() == 2 * n_var);

			alpha_init = Eigen::Map<Eigen::ArrayXXd>(vb_init_mat.col(0).data(), n_var, n_effects);
			mu_init = Eigen::Map<Eigen::ArrayXXd>(vb_init_mat.col(1).data(), n_var, n_effects);

			vp_init.alpha_beta     = alpha_init.col(0);
			vp_init.mu1_beta       = mu_init.col(0);
			if(n_env > 0) {
				vp_init.alpha_gam     = alpha_init.col(1);
				vp_init.mu1_gam       = mu_init.col(1);
			}
		} else if (vb_init_colnames == case2) {
			read_vb_init_file(p.vb_init_file, vb_init_mat, vb_init_colnames,
			                  init_key);
			std::cout << "--vb_init file with contextual information detected" << std::endl;
			std::cout << "Warning: This will be O(PL) where L = " << vb_init_mat.rows();
			std::cout << " is the number of lines in file given to --vb_init." << std::endl;

			std::vector<std::string>::iterator it;
			unsigned long index_kk;
			for(int kk = 0; kk < vb_init_mat.rows(); kk++) {
				it = std::find(G.SNPKEY.begin(), G.SNPKEY.end(), init_key[kk]);
				if (it == G.SNPKEY.end()) {
					std::cout << "WARNING: Can't locate variant with key: ";
					std::cout << init_key[kk] << std::endl;
				} else {
					index_kk = it - G.SNPKEY.begin();
					vp_init.alpha_beta(index_kk)    = 1.0;
					vp_init.mu1_beta(index_kk)      = vb_init_mat(kk, 5);
					if(n_effects > 1) {
						vp_init.alpha_gam(index_kk) = 1.0;
						vp_init.mu1_gam(index_kk)   = vb_init_mat(kk, 6);
					}
				}
			}
		} else if(vb_init_colnames == case3a) {
			EigenUtils::read_matrix_and_skip_cols(p.vb_init_file, 1, vb_init_mat, vb_init_colnames);
			assert(p.bgen_file == "NULL" || vb_init_mat.rows() == n_var);

			vp_init.alpha_beta = vb_init_mat.col(0);
			vp_init.mu1_beta = vb_init_mat.col(1);
			vp_init.s1_beta_sq = vb_init_mat.col(2);
			vp_init.mu2_beta = vb_init_mat.col(3);
			vp_init.s2_beta_sq = vb_init_mat.col(4);
		} else if(vb_init_colnames == case3b) {
			EigenUtils::read_matrix_and_skip_cols(p.vb_init_file, 1, vb_init_mat, vb_init_colnames);
			assert(p.bgen_file == "NULL" || vb_init_mat.rows() == n_var);

			vp_init.alpha_beta = vb_init_mat.col(0);
			vp_init.mu1_beta = vb_init_mat.col(1);
			vp_init.s1_beta_sq = vb_init_mat.col(2);
			vp_init.mu2_beta = vb_init_mat.col(3);
			vp_init.s2_beta_sq = vb_init_mat.col(4);
			vp_init.alpha_gam = vb_init_mat.col(5);
			vp_init.mu1_gam = vb_init_mat.col(6);
			vp_init.s1_gam_sq = vb_init_mat.col(7);
			vp_init.mu2_gam = vb_init_mat.col(8);
			vp_init.s2_gam_sq = vb_init_mat.col(9);
		} else {
			// Unknown header
			std::cout << "Unknown header in " << p.vb_init_file << std::endl;
			std::cout << "Expected headers are:" << std::endl;
			for (const auto& ss : case1) std::cout << ss << " ";
			std::cout << std::endl;
			for (const auto& ss : case2) std::cout << ss << " ";
			std::cout << std::endl;
			for (const auto& ss : case3b) std::cout << ss << " ";
			std::cout << std::endl;
			throw std::runtime_error("Unexpected header");
		}

		assert(p.bgen_file == "NULL" || vp_init.alpha_beta.rows() == n_var);
		if(n_env > 0) {
			assert(p.bgen_file == "NULL" || vp_init.alpha_gam.rows() == n_var);
		}
	}

	void set_vb_init(){
		if (p.resume_prefix != "NULL") {
			std::cout << "Reading in previous parameter state" << std::endl;
		}
		vp_init.run_default_init(n_var, n_covar, n_env);

		// Manually set env coeffs
		if(p.env_coeffs_file != "NULL") {
			std::vector<std::string> col_names;
			std::vector<std::string> case1 = {"env", "mu", "s_sq"};
			std::vector<std::string> case2 = {"env", "mu"};
			read_file_header(p.env_coeffs_file, col_names);

			Eigen::MatrixXd coeffs;
			if(col_names == case1) {
				EigenUtils::read_matrix_and_skip_cols(p.env_coeffs_file, 1, coeffs, col_names);
				vp_init.muw = coeffs.col(0);
				vp_init.sw_sq = coeffs.col(1);
			} else if(col_names == case2) {
				EigenUtils::read_matrix_and_skip_cols(p.env_coeffs_file, 1, coeffs, col_names);
				vp_init.muw = coeffs.col(0);
			} else if(col_names.size() == 1) {
				EigenUtils::read_matrix(p.env_coeffs_file, coeffs, col_names);
				vp_init.muw = coeffs.col(0);
			} else {
				throw std::runtime_error("Unexpected file to --environment_weights");
			}
			assert(vp_init.muw.rows() == n_env);
		} else if (n_env > 1 && p.init_weights_with_snpwise_scan) {
			assert(false);
			// calc_snpwise_regression(vp_init);
		}
		if(n_env > 0) {
			vp_init.eta = E * vp_init.muw.matrix();
		}

		// Manually set covar coeffs
		if(p.covar_coeffs_file != "NULL") {
			std::vector<std::string> col_names;
			std::vector<std::string> case1 = {"covar", "mu", "s_sq"};
			read_file_header(p.covar_coeffs_file, col_names);

			Eigen::MatrixXd coeffs;
			if(col_names == case1) {
				EigenUtils::read_matrix_and_skip_cols(p.covar_coeffs_file, 1, coeffs, col_names);
				vp_init.muc = coeffs.col(0);
				vp_init.sc_sq = coeffs.col(1);
			} else if(col_names.size() == 1) {
				EigenUtils::read_matrix(p.covar_coeffs_file, coeffs, col_names);
				vp_init.muc = coeffs.col(0);
			} else {
				throw std::runtime_error("Unexpected file to --environment_weights");
			}
			assert(vp_init.muc.rows() == n_covar);
		} else if (n_covar > 0) {
			// Start covars at least squared solution
			if(p.debug) std::cout << "Starting covars at least squares fit" << std::endl;
			Eigen::MatrixXd CtC = C.transpose() * C;
			CtC = mpiUtils::mpiReduce_inplace(CtC);
			Eigen::MatrixXd Cty = C.transpose() * Y;
			Cty = mpiUtils::mpiReduce_inplace(Cty);
			vp_init.muc = CtC.colPivHouseholderQr().solve(Cty);
		}

		// Manually set coeffs for SNP latent variables
		if(p.vb_init_file != "NULL") {
			assign_vb_init_from_file(vp_init);
		}
		if (p.debug) std::cout << "Done vb init" << std::endl;
		if (p.resume_prefix != "NULL") {
			std::cout << std::endl;
		}
	}

	void dump_processed_data(){
		std::string path, header;
		// std::cout << "Dumping processed data" << std::endl;
		path = fileUtils::filepath_format(p.out_file, "", "_debug_processed_env");
		std::cout << "Dumping processed E matrix to: " << path << std::endl;
		fileUtils::dump_predicted_vec_to_file(E, path, env_names, sample_location);

		path = fileUtils::filepath_format(p.out_file, "", "_debug_processed_pheno");
		std::cout << "Dumping processed Y matrix to: " << path << std::endl;
		fileUtils::dump_predicted_vec_to_file(Y, path, pheno_names, sample_location);

		path = fileUtils::filepath_format(p.out_file, "", "_debug_processed_covar");
		std::cout << "Dumping processed C matrix to: " << path << std::endl;
		fileUtils::dump_predicted_vec_to_file(C, path, covar_names, sample_location);
	}

	void read_mog_weights(const std::string& filename,
	                      Eigen::VectorXd& alpha_beta,
	                      Eigen::VectorXd& alpha_gam){
		// TODO: move into set_vb_init
		Eigen::MatrixXd tmp;
		std::vector<std::string> colnames;
		EigenUtils::read_matrix(filename, tmp, colnames);
		alpha_beta = tmp.col(0);
		alpha_gam = tmp.col(1);

		assert(tmp.cols() == 2);
		assert(tmp.rows() == G.cols());
		assert(tmp.minCoeff() >= 0);
		assert(tmp.maxCoeff() <= 1);
	}

	template <typename EigenMat>
	EigenMat reduce_mat_to_complete_cases( EigenMat& M,
	                                       bool& matrix_reduced,
	                                       const long& n_cols,
	                                       const std::map<long, bool>& incomplete_cases ) {
		// Remove rows contained in incomplete_cases
		EigenMat M_tmp;
		if (matrix_reduced) {
			throw std::runtime_error("ERROR: Trying to remove incomplete cases twice...");
		}

		// Create temporary matrix of complete cases
		unsigned long n_incomplete = incomplete_cases.size();
		M_tmp.resize(n_samples - n_incomplete, n_cols);

		// Fill M_tmp with non-missing entries of M
		int ii_tmp = 0;
		for (std::size_t ii = 0; ii < n_samples; ii++) {
			if (incomplete_cases.count(ii) == 0) {
				for (int kk = 0; kk < n_cols; kk++) {
					M_tmp(ii_tmp, kk) = M(ii, kk);
				}
				ii_tmp++;
			}
		}

		// Assign new values to reference variables
		matrix_reduced = true;
		return M_tmp;
	}

	void regress_first_mat_from_second(const EigenDataMatrix& A,
	                                   EigenDataMatrix& yy){

		Eigen::MatrixXd AtA, Aty;
		AtA = (A.transpose() * A).cast<double>();
		AtA = mpiUtils::mpiReduce_inplace(AtA);
		Aty = (A.transpose() * yy).cast<double>();
		Aty = mpiUtils::mpiReduce_inplace(Aty);

		Eigen::MatrixXd bb = EigenUtils::solve(AtA, Aty);
		yy -= A * bb.cast<scalarData>();
	}

	void regress_first_mat_from_second(const EigenDataMatrix& A,
	                                   const std::string& Astring,
	                                   const std::vector<std::string>& A_names,
	                                   EigenDataMatrix& yy,
	                                   const std::string& yy_string){
		//
		std::cout << "Regressing " << Astring << " from " << yy_string << ":" << std::endl;
		unsigned long nnn = A_names.size();
		for(int cc = 0; cc < std::min(nnn, (unsigned long) 10); cc++) {
			std::cout << ( cc > 0 ? ", " : "" ) << A_names[cc];
		}
		if (nnn > 10) {
			std::cout << "... (" << nnn << " variables)";
		}
		std::cout << std::endl;

		Eigen::MatrixXd AtA, Aty;
		AtA = (A.transpose() * A).cast<double>();
		AtA = mpiUtils::mpiReduce_inplace(AtA);
		Aty = (A.transpose() * yy).cast<double>();
		Aty = mpiUtils::mpiReduce_inplace(Aty);

		Eigen::MatrixXd bb = EigenUtils::solve(AtA, Aty);
		yy -= A * bb.cast<scalarData>();
	}

	void reduce_to_complete_cases() {
		// Remove any samples with incomplete covariates or phenotypes from
		// Y and C.
		// Note; other functions (eg. read_incl_sids) may add to incomplete_cases
		// Note; during unit testing sometimes only phenos or covars present.

		incomplete_cases.insert(missing_covars.begin(), missing_covars.end());
		incomplete_cases.insert(missing_phenos.begin(), missing_phenos.end());
		incomplete_cases.insert(missing_envs.begin(), missing_envs.end());
		incomplete_cases.insert(missing_resid_loco.begin(), missing_resid_loco.end());

		mpiUtils::partition_valid_samples_across_ranks(n_samples, n_var, n_env, p, incomplete_cases, sample_location);

		sample_is_invalid.clear();
		for (long ii = 0; ii < n_samples; ii++) {
			if (incomplete_cases.find(ii) == incomplete_cases.end()) {
				sample_is_invalid[ii] = false;
			} else {
				sample_is_invalid[ii] = true;
			}
		}

		if(n_pheno > 0) {
			Y = reduce_mat_to_complete_cases(Y, Y_reduced, n_pheno, incomplete_cases);
		}
		if(n_covar > 0) {
			C = reduce_mat_to_complete_cases(C, W_reduced, n_covar, incomplete_cases);
		}
		if(n_env > 0) {
			E = reduce_mat_to_complete_cases(E, E_reduced, n_env, incomplete_cases);
		}
		if(p.extra_pve_covar_file != "NULL" && p.mode_RHE) {
			long n_cols = C_extra_pve.cols();
			bool placeholder = false;
			C_extra_pve = reduce_mat_to_complete_cases(C_extra_pve, placeholder, n_cols, incomplete_cases);
		}
		if (p.resid_loco_file != "NULL"){
			long n_cols = resid_loco.cols();
			bool placeholder = false;
			resid_loco = reduce_mat_to_complete_cases(resid_loco, placeholder, n_cols, incomplete_cases);
		}
		n_samples -= incomplete_cases.size();
		missing_phenos.clear();
		missing_covars.clear();
		missing_envs.clear();

		Nglobal = mpiUtils::mpiReduce_inplace(&n_samples);

		int world_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

		std::cout << "Reduced to " << Nglobal;
		std::cout << " samples with complete data across covariates";
		if(p.env_file != "NULL") std::cout << ", env-variables";
		std::cout << " and phenotype";
		std::cout << " (" << n_samples << " on rank " << world_rank << ")." << std::endl << std::endl;
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
};


inline void read_file_header(const std::string& filename,
                             std::vector<std::string>& col_names){

	// Get colnames from file
	boost_io::filtering_istream fg;
	std::string gz_str = ".gz";
	if (filename.find(gz_str) != std::string::npos) {
		fg.push(boost_io::gzip_decompressor());
	}
	fg.push(boost_io::file_source(filename));
	if (!fg) {
		std::cout << "ERROR: " << filename << " not opened." << std::endl;
		std::exit(EXIT_FAILURE);
	}

	// Reading column names
	col_names.clear();
	std::string line;
	if (!getline(fg, line)) {
		std::cout << "ERROR: " << filename << " contains zero lines" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	std::stringstream ss;
	std::string s;
	col_names.clear();
	ss.clear();
	ss.str(line);
	while (ss >> s) {
		col_names.push_back(s);
	}
}

#endif
