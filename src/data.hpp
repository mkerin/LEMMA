// File of Data class for use with src/bgen_prog.cpp
#ifndef DATA_H
#define DATA_H

#include "bgen_parser.hpp"
#include "genotype_matrix.hpp"
#include "my_timer.hpp"
#include "parameters.hpp"
#include "typedefs.hpp"
#include "stats_tests.hpp"
#include "eigen_utils.hpp"
#include "variational_parameters.hpp"

#include "tools/Eigen/Dense"
#include "tools/Eigen/Eigenvalues"
#include "genfile/bgen/bgen.hpp"
#include "genfile/bgen/View.hpp"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstddef>     // for ptrdiff_t class
#include <chrono>      // start/end time info
#include <ctime>       // start/end time info
#include <map>
#include <mutex>
#include <vector>
#include <string>
#include <set>
#include <stdexcept>
#include <thread>

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
	parameters params;


	unsigned long n_pheno;                                 // number of phenotypes
	unsigned long n_covar;                                 // number of covariates
	unsigned long n_env;                                 // number of env variables
	int n_effects;                                   // number of environmental interactions
	long n_samples;                                 // number of samples
	long n_var;
	long n_var_parsed;                                 // Track progress through IndexQuery
	long int n_dxteex_computed;
	long int n_snpstats_computed;

	bool Y_reduced;                                   // Variables to track whether we have already
	bool W_reduced;                                   // reduced to complete cases or not.
	bool E_reduced;

	std::vector< std::string > external_dXtEEX_SNPID;
	std::vector< std::string > rsid_list;

	std::map<int, bool> missing_envs;                                   // set of subjects missing >= 1 env variables
	std::map<int, bool> missing_covars;                                 // set of subjects missing >= 1 covariate
	std::map<int, bool> missing_phenos;                                 // set of subjects missing >= phenotype
	std::map< std::size_t, bool > incomplete_cases;                                 // union of samples missing data

	std::vector< std::string > pheno_names;
	std::vector< std::string > covar_names;
	std::vector< std::string > env_names;

	GenotypeMatrix G;
	EigenDataMatrix Y, Y2;                                 // phenotype matrix (#2 always has covars regressed)
	EigenDataMatrix C;                                 // covariate matrix
	EigenDataMatrix E;                                 // env matrix
	Eigen::ArrayXXd dXtEEX;
	Eigen::ArrayXXd external_dXtEEX;

// Init points
	VariationalParameters vp_init;
	std::vector<Hyps> hyps_inits;

	genfile::bgen::View::UniquePtr bgenView;
	std::vector<genfile::bgen::View::UniquePtr> bgenViews;

// For gxe genome-wide scan
// cols neglogp-main, neglogp-gxe, coeff-gxe-main, coeff-gxe-env..
	Eigen::ArrayXXd snpstats;
	Eigen::ArrayXXd external_snpstats;
	std::vector< std::string > external_snpstats_SNPID;
	bool bgen_pass;

	boost_io::filtering_ostream outf_scan;

	bool filters_applied;
	std::mutex mtx;

// grid things for vbayes
	std::vector< std::string > hyps_names;
	Eigen::MatrixXd r1_hyps_grid, hyps_grid;
	Eigen::ArrayXXd alpha_init, mu_init;

	explicit Data( const parameters& p ) : params(p), G(p), vp_init(p) {
		Eigen::setNbThreads(params.n_thread);
		int n = Eigen::nbThreads( );
		std::cout << "Threads used by eigen: " << n << std::endl;


		// Create vector of bgen views for mutlithreading
		bgenView = genfile::bgen::View::create(p.bgen_file);
		for (int nn = 0; nn < params.n_bgen_thread; nn++) {
			genfile::bgen::View::UniquePtr bgenView = genfile::bgen::View::create(p.bgen_file);
			bgenViews.push_back(std::move(bgenView));
		}

		n_samples = (long) bgenView->number_of_samples();
		n_var_parsed = 0;
		filters_applied = false;
		bgen_pass = true;

		// Explicit initial values to eliminate linter errors..
		n_pheno   = 0;
		n_effects = -1;
		n_covar   = 0;
		n_env     = 0;
		n_var     = 0;
		Y_reduced = false;
		W_reduced = false;
	}

	~Data() {
		// system time at end
	}

	void apply_filters(){
		// filter - incl sample ids
		if(params.incl_sids_file != "NULL") {
			read_incl_sids();
		}

		// filter - init queries
		genfile::bgen::IndexQuery::UniquePtr query = genfile::bgen::IndexQuery::create(params.bgi_file);
		std::vector<genfile::bgen::IndexQuery::UniquePtr> queries;
		for (int nn = 0; nn < params.n_bgen_thread; nn++) {
			genfile::bgen::IndexQuery::UniquePtr my_query = genfile::bgen::IndexQuery::create(params.bgi_file);
			queries.push_back(move(my_query));
		}

		// filter - range
		if (params.range) {
			std::cout << "Selecting range..." << std::endl;
			genfile::bgen::IndexQuery::GenomicRange rr1(params.chr, params.range_start, params.range_end);
			query->include_range( rr1 );

			for (int nn = 0; nn < params.n_bgen_thread; nn++) {
				queries[nn]->include_range( rr1 );
			}
		}

		// filter - incl rsids
		if(params.select_snps) {
			read_incl_rsids();
			std::cout << "Filtering SNPs by rsid..." << std::endl;
			query->include_rsids( rsid_list );

			for (int nn = 0; nn < params.n_bgen_thread; nn++) {
				queries[nn]->include_rsids( rsid_list );
			}
		}

		// filter - select single rsid
		if(params.select_rsid) {
			std::sort(params.rsid.begin(), params.rsid.end());
			std::cout << "Filtering to rsids:" << std::endl;
			long int n_rsids = params.rsid.size();
			for (long int kk = 0; kk < n_rsids; kk++) {
				std::cout << params.rsid[kk]<< std::endl;
			}
			query->include_rsids( params.rsid );

			for (int nn = 0; nn < params.n_bgen_thread; nn++) {
				queries[nn]->include_rsids( params.rsid );
			}
		}

		// filter - apply queries
		query->initialise();
		for (int nn = 0; nn < params.n_bgen_thread; nn++) {
			queries[nn]->initialise();
		}

		bgenView->set_query( query );
		for (int nn = 0; nn < params.n_bgen_thread; nn++) {
			bgenViews[nn]->set_query( queries[nn] );
		}

		// print summaries
		bgenView->summarise(std::cout);

		filters_applied = true;
	}

	void read_non_genetic_data(){
		// Apply sample / rsid / range filters if applicable
		if(!filters_applied) {
			apply_filters();
		}

		// Read in phenotypes
		read_pheno();

		// Read in covariates if present
		if(params.covar_file != "NULL") {
			read_covar();
		}

		// Environmental vars - subset of covars
		if(params.env_file != "NULL") {
			read_environment();
		}

		if(!params.mode_no_gxe && params.interaction_analysis) {
			n_effects = 2;
		} else {
			n_effects = 1;
		}

		// Exclude samples with missing values in phenos / covars / filters
		reduce_to_complete_cases();

		// Read in hyperparameter values
		if(params.hyps_grid_file != "NULL") {
			read_hyps();
		}

		if(n_env > 1 && params.dxteex_file != "NULL") {
			read_external_dxteex();
		}

		if(params.snpstats_file != "NULL") {
			read_external_snpstats();
		}
	}

	void standardise_non_genetic_data(){
		// Step 3; Center phenos, normalise covars
		EigenUtils::center_matrix(Y);
		if(params.scale_pheno) {
			std::cout << "Scaling phenotype" << std::endl;
			EigenUtils::scale_matrix_and_remove_constant_cols( Y, n_pheno, pheno_names );
		}
		if(n_covar > 0) {
			EigenUtils::center_matrix(C);
			EigenUtils::scale_matrix_and_remove_constant_cols( C, n_covar, covar_names );
		}
		if(n_env > 0) {
			EigenUtils::center_matrix(E);
			EigenUtils::scale_matrix_and_remove_constant_cols( E, n_env, env_names );
		}

		/* Regression rules
		   1 - if use_VB_on_covars then:
		        - E main effects kept in model
		        - C regressed from Y
		   2 - if !use_VB_on_covars then:
		        - [C, E] regressed from Y

		   if any of E have squared dependance on Y then remove
		 */

		if (n_env > 0) {
			if (params.use_vb_on_covars) {
				if (n_covar > 0) {
					regress_first_mat_from_second(C, "covars", covar_names, Y, "pheno");
					regress_first_mat_from_second(C, "covars", covar_names, E, "env");
				}

				C = E;
				covar_names = env_names;
				n_covar = n_env;
			} else if (n_covar > 0) {
				EigenDataMatrix tmp(C.rows(), C.cols() + E.cols());
				tmp << C, E;
				C = tmp;
				covar_names.insert(covar_names.end(), env_names.begin(), env_names.end());
				n_covar += n_env;
			} else {
				C = E;
				covar_names = env_names;
				n_covar = n_env;
			}

			// Removing squared dependance
			std::vector<int> cols_to_remove;
			Eigen::MatrixXd H(n_samples, n_covar + 1);
			Eigen::VectorXd tmp = Eigen::VectorXd::Zero(n_samples);

			H << tmp, C;
			std::cout << "Checking for squared dependance: " << std::endl;
			std::cout << "Name\t-log10(p-val)" << std::endl;
			long n_signif_envs_sq = 0;
			for (int ee = 0; ee < n_env; ee++) {
				// H.col(0) = E.col(ee);
				H.col(0) = E.col(ee).array().square().matrix();

				Eigen::MatrixXd HtH_inv = (H.transpose() * H).inverse();
				Eigen::MatrixXd Hty = H.transpose() * Y;
				double rss = (Y - H * HtH_inv * Hty).squaredNorm();

				double pval = student_t_test(n_samples, HtH_inv, Hty, rss, 0);

				std::cout << env_names[ee] << "\t";
				std::cout << -1 * std::log10(pval) << std::endl;

				if (pval < 0.01 / (double) n_env) {
					cols_to_remove.push_back(ee);
					n_signif_envs_sq++;
				}
			}
			if (n_signif_envs_sq > 0) {
				Eigen::MatrixXd E_sq(n_samples, n_signif_envs_sq);
				std::vector<std::string> env_sq_names;
				for (int nn = 0; nn < n_signif_envs_sq; nn++) {
					E_sq.col(nn) = E.col(cols_to_remove[nn]).array().square();
					env_sq_names.push_back(env_names[cols_to_remove[nn]] + "_sq");
				}

				if (params.mode_incl_squared_envs) {
					std::cout << "Including squared env effects from: " << std::endl;
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
				} else if (params.mode_remove_squared_envs) {
					std::cout << "Projecting out squared dependance from: " << std::endl;
					for (int ee : cols_to_remove) {
						std::cout << env_names[ee] << std::endl;
					}

					Eigen::MatrixXd H(n_samples, n_covar + n_signif_envs_sq);
					H << E_sq, C;

					Eigen::MatrixXd HtH = H.transpose() * H;
					Eigen::MatrixXd Hty = H.transpose() * Y;
					Eigen::MatrixXd beta = HtH.colPivHouseholderQr().solve(Hty);

					Y -= E_sq * beta.block(0, 0, n_signif_envs_sq, 1);
				} else {
					std::cout << "Warning: Projection of significant square envs (";
					for (int ee : cols_to_remove) {
						std::cout << env_names[ee] << ", ";
					}
					std::cout << ") suppressed" << std::endl;
				}
			}

			// For internal use
			Y2 = Y;
			regress_first_mat_from_second(C, Y2);
		} else {
			Y2 = Y;
		}
	}

	void read_full_bgen(){
		std::cout << "Reading in BGEN" << std::endl;
		params.chunk_size = bgenView->number_of_variants();
		MyTimer t_readFullBgen("BGEN parsed in %ts \n");

		if(params.flip_high_maf_variants) {
			std::cout << "Flipping variants with MAF > 0.5" << std::endl;
		}
		t_readFullBgen.resume();
		read_bgen_chunk();
		t_readFullBgen.stop();
		std::cout << "BGEN contained " << n_var << " variants." << std::endl;
	}

	bool read_bgen_chunk() {
		// Wrapper around BgenView to read in a 'chunk' of data. Remembers
		// if last call hit the EOF, and returns false if so.
		// Assumed that:
		// - commandline args parsed and passed to params
		// - bgenView initialised with correct filename
		// - scale + centering happening internally

		// Exit function if last call hit EOF.
		if (!bgen_pass) return false;

		// Temporary variables to store info from read_variant()
		std::string chr_j;
		std::uint32_t pos_j;
		std::string rsid_j;
		std::vector< std::string > alleles_j;
		// read but ignored
		std::string SNPID_j;
		std::vector< std::vector< double > > probs;
		ProbSetter setter( &probs );

		double x, dosage, check, info_j, f1, chunk_missingness;
		double dosage_mean, dosage_sigma, missing_calls = 0.0;
		int n_var_incomplete = 0;

		// Resize genotype matrix
		G.resize(n_samples, params.chunk_size);

		long int n_constant_variance = 0;
		std::uint32_t jj = 0;
		while ( jj < params.chunk_size && bgen_pass ) {
			bgen_pass = bgenView->read_variant( &SNPID_j, &rsid_j, &chr_j, &pos_j, &alleles_j );
			if (!bgen_pass) break;
			n_var_parsed++;

			// Read probs + check maf filter
			bgenView->read_genotype_data_block( setter );

			// maf + info filters; computed on valid sample_ids & variants whose alleles
			// sum to 1
			std::map<int, bool> missing_genos;
			Eigen::ArrayXd dosage_j(n_samples);
			double f2 = 0.0, valid_count = 0.0;
			std::uint32_t ii_obs = 0;
			for( std::size_t ii = 0; ii < probs.size(); ii++ ) {
				if (incomplete_cases.count(ii) == 0) {
					f1 = dosage = check = 0.0;
					for( std::size_t kk = 0; kk < probs[ii].size(); kk++ ) {
						x = probs[ii][kk];
						dosage += x * kk;
						f1 += x * kk * kk;
						check += x;
					}
					if(check > 0.9999 && check < 1.0001) {
						dosage_j[ii_obs] = dosage;
						f2 += (f1 - dosage * dosage);
						valid_count++;
					} else {
						missing_genos[ii_obs] = true;
						dosage_j[ii_obs] = 0.0;
					}
					ii_obs++;
				}
			}
			assert(ii_obs == n_samples);

			double d1    = dosage_j.sum();
			double maf_j = d1 / (2.0 * valid_count);

			// Flip dosage vector if maf > 0.5
			if(params.flip_high_maf_variants && maf_j > 0.5) {
				dosage_j = (2.0 - dosage_j);
				for (std::uint32_t ii = 0; ii < n_samples; ii++) {
					if (missing_genos.count(ii) > 0) {
						dosage_j[ii] = 0.0;
					}
				}

				f2       = 4.0 * valid_count - 4.0 * d1 + f2;
				d1       = dosage_j.sum();
				maf_j    = d1 / (2.0 * valid_count);
			}

			double mu    = d1 / valid_count;
			info_j       = 1.0;
			if(maf_j > 1e-10 && maf_j < 0.9999999999) {
				info_j -= f2 / (2.0 * valid_count * maf_j * (1.0 - maf_j));
			}

			// Compute sd
			double sigma = (dosage_j - mu).square().sum();
			sigma = std::sqrt(sigma/(valid_count - 1.0));

			// Filters
			if (params.maf_lim && (maf_j < params.min_maf || maf_j > 1 - params.min_maf)) {
				continue;
			}
			if (params.info_lim && info_j < params.min_info) {
				continue;
			}
			if(!params.keep_constant_variants && d1 < 5.0) {
				n_constant_variance++;
				continue;
			}
			if(!params.keep_constant_variants && sigma <= 1e-12) {
				n_constant_variance++;
				continue;
			}

			// filters passed; write contextual info
			G.al_0[jj]     = alleles_j[0];
			G.al_1[jj]     = alleles_j[1];
			G.maf[jj]      = maf_j;
			G.info[jj]     = info_j;
			G.rsid[jj]     = rsid_j;
			G.chromosome[jj] = std::stoi(chr_j);
			G.position[jj] = pos_j;
			std::string key_j = chr_j + "~" + std::to_string(pos_j) + "~" + alleles_j[0] + "~" + alleles_j[1];
			G.SNPKEY[jj]   = key_j;
			G.SNPID[jj] = SNPID_j;

			// filters passed; write dosage to G
			// Note that we only write dosage for valid sample ids
			// Scale + center dosage, set missing to mean
			if(!missing_genos.empty()) {
				n_var_incomplete += 1;
				missing_calls += (double) missing_genos.size();
				for (std::uint32_t ii = 0; ii < n_samples; ii++) {
					if (missing_genos.count(ii) > 0) {
						dosage_j[ii] = mu;
					}
				}
			}

			for (std::uint32_t ii = 0; ii < n_samples; ii++) {
				G.assign_index(ii, jj, dosage_j[ii]);
			}
			// G.compressed_dosage_sds[jj] = sigma;
			// G.col_means[jj] = mu;

			jj++;
		}

		// need to resize G whilst retaining existing coefficients if while
		// loop exits early due to EOF.
		G.conservativeResize(n_samples, jj);
		assert( G.rsid.size() == jj );
		n_var = jj;

		chunk_missingness = missing_calls / (double) (n_var * n_samples);
		if(chunk_missingness > 0.0) {
			std::cout << "Chunk missingness " << chunk_missingness << "(";
			std::cout << n_var_incomplete << "/" << n_var;
			std::cout << " variants incomplete)" << std::endl;
		}

		if(n_constant_variance > 0) {
			std::cout << n_constant_variance << " variants removed due to ";
			std::cout << "constant variance" << std::endl;
		}

		if(jj == 0) {
			// Immediate EOF
			return false;
		} else {
			return true;
		}
	}

	void read_incl_rsids(){
		boost_io::filtering_istream fg;
		std::string gz_str = ".gz";
		if (params.incl_rsids_file.find(gz_str) != std::string::npos) {
			fg.push(boost_io::gzip_decompressor());
		}
		fg.push(boost_io::file_source(params.incl_rsids_file));
		if (!fg) {
			std::cout << "ERROR: " << params.incl_rsids_file << " not opened." << std::endl;
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
		if (params.incl_sids_file.find(gz_str) != std::string::npos) {
			fg.push(boost_io::gzip_decompressor());
		}
		fg.push(boost_io::file_source(params.incl_sids_file));
		if (!fg) {
			std::cout << "ERROR: " << params.incl_sids_file << " not opened." << std::endl;
			std::exit(EXIT_FAILURE);
		}

		std::vector<std::string> bgen_ids;
		bgenView->get_sample_ids(
			[&]( std::string const& id ) {
			bgen_ids.push_back(id);
		}
			);

		// Want to read in a sid to be included, and skip along bgen_ids until
		// we find it.
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
		std::cout << "Subsetted down to " << user_sample_ids.size() << " ids from --incl_sample_ids";
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
		// Read phenotypes to Eigen matrix Y
		EigenUtils::read_matrix(params.pheno_file, n_samples, Y, pheno_names, missing_phenos);
		n_pheno = Y.cols();
		if(n_pheno != 1) {
			std::cout << "ERROR: Only expecting one phenotype at a time." << std::endl;
		}
		Y_reduced = false;
	}

	void read_covar( ){
		// Read covariates to Eigen matrix C
		EigenUtils::read_matrix(params.covar_file, n_samples, C, covar_names, missing_covars);
		n_covar = C.cols();
		W_reduced = false;
	}

	void read_environment( ){
		// Read covariates to Eigen matrix C
		EigenUtils::read_matrix(params.env_file, n_samples, E, env_names, missing_envs);
		n_env = E.cols();
		E_reduced = false;
	}

	void read_external_dxteex( ){
		std::vector< std::string > col_names;
		read_txt_file_w_context( params.dxteex_file, 6, external_dXtEEX,
		                         external_dXtEEX_SNPID, col_names);

		if(external_dXtEEX.cols() != n_env * n_env) {
			std::cout << "Expecting columns in order: " << std::endl;
			std::cout << "SNPID, chr, rsid, pos, allele0, allele1, env-snp covariances.." << std::endl;
			throw std::runtime_error("Unexpected number of columns");
		}
	}

	void read_external_snpstats( ){
		std::vector< std::string > col_names;
		read_txt_file_w_context( params.snpstats_file, 8, external_snpstats,
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
		std::stringstream ss;
		std::string s;
		int n_cols = 0;
		col_names.clear();
		ss.clear();
		ss.str(line);
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

	void calc_snpstats(){
		std::cout << "Reordering/computing snpwise scan...";
		MyTimer my_timer("snpwise scan constructed in %ts \n");
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

		if(params.snpstats_file == "NULL") {
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
	}

	void calc_dxteex(){
		std::cout << "Reordering/building dXtEEX array...";
		MyTimer t_calcDXtEEX("dXtEEX array constructed in %ts \n");
		t_calcDXtEEX.resume();
		EigenDataArrayX cl_j;
		scalarData dztz_lmj;
		dXtEEX.resize(n_var, n_env * n_env);
		n_dxteex_computed = 0;
		int dxteex_check = 0, se_cnt = 0;
		double mean_ae = 0, max_ae = 0;
		for (std::size_t jj = 0; jj < n_var; jj++) {
			auto it = std::find(external_dXtEEX_SNPID.begin(), external_dXtEEX_SNPID.end(), G.SNPID[jj]);
			if (it == external_dXtEEX_SNPID.end()) {
				n_dxteex_computed++;
				cl_j = G.col(jj);
				for (int ll = 0; ll < n_env; ll++) {
					for (int mm = 0; mm <= ll; mm++) {
						dztz_lmj = (cl_j * E.array().col(ll) * E.array().col(mm) * cl_j).sum();
						dXtEEX(jj, ll*n_env + mm) = dztz_lmj;
						dXtEEX(jj, mm*n_env + ll) = dztz_lmj;
					}
				}
			} else {
				dXtEEX.row(jj) = external_dXtEEX.row(it - external_dXtEEX_SNPID.begin());
				if(dxteex_check < 100) {
					dxteex_check++;
					cl_j = G.col(jj);
					for (int ll = 0; ll < n_env; ll++) {
						for (int mm = 0; mm <= ll; mm++) {
							dztz_lmj = (cl_j * E.array().col(ll) * E.array().col(mm) * cl_j).sum();
							double x1 = std::abs(dXtEEX(jj, ll*n_env + mm) - dztz_lmj);
							double x2 = std::abs(dXtEEX(jj, mm*n_env + ll) - dztz_lmj);
							max_ae = std::max(x1, max_ae);
							max_ae = std::max(x2, max_ae);
							mean_ae += x1;
							mean_ae += x2;
							se_cnt++;
						}
					}
				}
			}
		}
		if(params.dxteex_file != "NULL" && n_dxteex_computed < n_var) {
			mean_ae /= (double) se_cnt;
			std::cout << "Checking correlations from first 100 SNPs" << std::endl;
			std::cout << " - max absolute error = " << max_ae << std::endl;
			std::cout << " - mean absolute error = " << mean_ae << std::endl;
		}
		std::cout << " (" << n_dxteex_computed << " computed from raw data, " << n_var - n_dxteex_computed << " read from file)" << std::endl;
	}

	void read_hyps(){
		// For use in vbayes object

		std::vector< std::string > case1 = {"sigma", "sigma_b", "lambda_b"};
		std::vector< std::string > case2 = {"sigma", "sigma_b", "sigma_g", "lambda_b", "lambda_g"};
		std::vector< std::string > case3 = {"hyp", "value"};

		read_file_header(params.hyps_grid_file, hyps_names);
		int n_hyps_removed = 0;

		if(std::includes(hyps_names.begin(), hyps_names.end(), case1.begin(), case1.end())) {
			assert(params.interaction_analysis);
			EigenUtils::read_matrix(params.hyps_grid_file, hyps_grid, hyps_names);

			// Unpack from grid to hyps object
			long n_grid = hyps_grid.rows();
			for (int ii = 0; ii < n_grid; ii++) {
				Hyps hyps(params);
				hyps.init_from_grid(n_effects, ii, n_var, hyps_grid);
				if(hyps.domain_is_valid()) {
					hyps_inits.push_back(hyps);
				} else {
					n_hyps_removed++;
				}
			}
		} else if(std::includes(hyps_names.begin(), hyps_names.end(), case2.begin(), case2.end())) {
			EigenUtils::read_matrix(params.hyps_grid_file, hyps_grid, hyps_names);
			// If not interaction analysis check interaction hyps are zero
			if(!params.interaction_analysis) {
				double sigma_g_sum  = hyps_grid.col(2).array().abs().sum();
				double lambda_g_sum = hyps_grid.col(4).array().abs().sum();
				if(sigma_g_sum > 1e-6 || lambda_g_sum > 1e-6) {
					std::cout << "WARNING: You have non-zero hyperparameters for interaction effects,";
					std::cout << " but no environmental variables provided." << std::endl;
				}
			}

			// Unpack from grid to hyps object
			long n_grid = hyps_grid.rows();
			for (int ii = 0; ii < n_grid; ii++) {
				Hyps hyps(params);
				hyps.init_from_grid(n_effects, ii, n_var, hyps_grid);
				if(hyps.domain_is_valid()) {
					hyps_inits.push_back(hyps);
				} else {
					n_hyps_removed++;
				}
			}
		} else if(hyps_names == case3) {
			// Reading from hyps dump
			Hyps hyps(params);
			hyps.read_from_dump(params.hyps_grid_file);
			hyps_inits.push_back(hyps);
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
		//      // Allow gxe params if coeffs set to zero
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

	void assign_vb_init_from_file(VariationalParameters& vp_init){
		// Set initial conditions for VB
		std::cout << "Reading initialisation for alpha from file" << std::endl;

		// Different filetypes
		std::vector<std::string> case1 = {"alpha", "mu"};
		std::vector<std::string> case2 = {"chr", "rsid", "pos", "a0", "a1", "beta", "gamma"};
		std::vector<std::string> latents1 = {"alpha_beta", "mu1_beta", "s1_beta", "mu2_beta", "s2_beta"};
		std::vector<std::string> latents2 = {"alpha_gam", "mu1_gam", "s1_gam", "mu2_gam", "s2_gam"};
		std::vector<std::string> case3 = {"SNPID"};
		case3.insert(case3.end(),latents1.begin(),latents1.end());
		case3.insert(case3.end(),latents2.begin(),latents2.end());

		std::vector<std::string> vb_init_colnames;
		read_file_header(params.vb_init_file, vb_init_colnames);

		Eigen::MatrixXd vb_init_mat;
		std::vector< std::string > init_key;
		if(vb_init_colnames == case1) {
			EigenUtils::read_matrix(params.vb_init_file, vb_init_mat, vb_init_colnames);
			assert(vb_init_mat.rows() == 2 * n_var);

			alpha_init = Eigen::Map<Eigen::ArrayXXd>(vb_init_mat.col(0).data(), n_var, n_effects);
			mu_init = Eigen::Map<Eigen::ArrayXXd>(vb_init_mat.col(1).data(), n_var, n_effects);

			vp_init.betas->set_mean(mu_init.col(0));
//		vp_init.alpha_beta     = alpha_init.col(0);
//		vp_init.mu1_beta       = mu_init.col(0);
			if(n_env > 0) {
				vp_init.gammas->set_mean(mu_init.col(1));
//			vp_init.alpha_gam     = alpha_init.col(1);
//			vp_init.mu1_gam       = mu_init.col(1);
			}
		} else if (vb_init_colnames == case2) {
			read_vb_init_file(params.vb_init_file, vb_init_mat, vb_init_colnames,
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
//				vp_init.alpha_beta(index_kk)    = 1.0;
//				vp_init.mu1_beta(index_kk)      = vb_init_mat(kk, 5);
					vp_init.betas->set_mean(index_kk, vb_init_mat(kk, 5));
					if(n_effects > 1) {
//					vp_init.alpha_gam(index_kk) = 1.0;
//					vp_init.mu1_gam(index_kk)   = vb_init_mat(kk, 6);
						vp_init.gammas->set_mean(index_kk, vb_init_mat(kk, 6));
					}
				}
			}
		} else if(vb_init_colnames == case3) {
			EigenUtils::read_matrix_and_skip_cols(params.vb_init_file, 1, vb_init_mat, vb_init_colnames);
			assert(vb_init_mat.rows() == n_var);

			vp_init.betas->read_from_grid(vb_init_mat.block(0, 0, n_var, 5));
			vp_init.gammas->read_from_grid(vb_init_mat.block(0, 5, n_var, 5));

//		vp_init.alpha_beta = vb_init_mat.col(0);
//		vp_init.mu1_beta = vb_init_mat.col(1);
//		vp_init.s1_beta_sq = vb_init_mat.col(2);
//		vp_init.mu2_beta = vb_init_mat.col(3);
//		vp_init.s2_beta_sq = vb_init_mat.col(4);
//		vp_init.alpha_gam = vb_init_mat.col(5);
//		vp_init.mu1_gam = vb_init_mat.col(6);
//		vp_init.s1_gam_sq = vb_init_mat.col(7);
//		vp_init.mu2_gam = vb_init_mat.col(8);
//		vp_init.s2_gam_sq = vb_init_mat.col(9);
		} else {
			// Unknown header
			std::cout << "Unknown header in " << params.vb_init_file << std::endl;
			std::cout << "Expected headers are:" << std::endl;
			for (auto ss : case1) std::cout << ss << " ";
			std::cout << std::endl;
			for (auto ss : case2) std::cout << ss << " ";
			std::cout << std::endl;
			for (auto ss : case3) std::cout << ss << " ";
			std::cout << std::endl;
			throw std::runtime_error("Unexpected header");
		}
	}

	void set_vb_init(){
		vp_init.run_default_init(n_var, n_covar, n_env);

		// Manually set env coeffs
		if(params.env_coeffs_file != "NULL") {
			std::vector<std::string> col_names;
			std::vector<std::string> case1 = {"env", "mu", "s_sq"};
			read_file_header(params.env_coeffs_file, col_names);

			Eigen::MatrixXd coeffs;
			if(col_names == case1) {
				EigenUtils::read_matrix_and_skip_cols(params.env_coeffs_file, 1, coeffs, col_names);
				vp_init.weights.read_from_grid(coeffs);
			} else if(col_names.size() == 1) {
				EigenUtils::read_matrix(params.env_coeffs_file, coeffs, col_names);
				vp_init.weights.set_mean(coeffs.col(0));
			} else {
				throw std::runtime_error("Unexpected file to --environment_weights");
			}
		} else if (n_env > 1 && params.init_weights_with_snpwise_scan) {
			assert(false);
			// calc_snpwise_regression(vp_init);
		}

		// Manually set covar coeffs
		if(params.covar_coeffs_file != "NULL") {
			std::vector<std::string> col_names;
			std::vector<std::string> case1 = {"covar", "mu", "s_sq"};
			read_file_header(params.covar_coeffs_file, col_names);

			Eigen::MatrixXd coeffs;
			if(col_names == case1) {
				EigenUtils::read_matrix_and_skip_cols(params.covar_coeffs_file, 1, coeffs, col_names);
				vp_init.covars.read_from_grid(coeffs);
			} else if(col_names.size() == 1) {
				EigenUtils::read_matrix(params.covar_coeffs_file, coeffs, col_names);
				vp_init.covars.set_mean(coeffs.col(0));
			} else {
				throw std::runtime_error("Unexpected file to --environment_weights");
			}
		} else if (n_covar > 0) {
			// Start covars at least squared solution
			std::cout << "Starting covars at least squares fit" << std::endl;
			Eigen::MatrixXd CtC = C.transpose() * C;
			Eigen::MatrixXd Cty = C.transpose() * Y;
			Eigen::MatrixXd muc = CtC.colPivHouseholderQr().solve(Cty);

			vp_init.covars.set_mean(muc);
		}

		// Manually set coeffs for SNP latent variables
		if(params.vb_init_file != "NULL") {
			assign_vb_init_from_file(vp_init);
		}
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
	                                       const unsigned long& n_cols,
	                                       const std::map< std::size_t, bool >& incomplete_cases ) {
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

		Eigen::MatrixXd AtA = (A.transpose() * A).cast<double>();
		Eigen::MatrixXd Aty = (A.transpose() * yy).cast<double>();

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

		Eigen::MatrixXd AtA = (A.transpose() * A).cast<double>();
		Eigen::MatrixXd Aty = (A.transpose() * yy).cast<double>();

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

		if(n_pheno > 0) {
			Y = reduce_mat_to_complete_cases( Y, Y_reduced, n_pheno, incomplete_cases );
		}
		if(n_covar > 0) {
			C = reduce_mat_to_complete_cases( C, W_reduced, n_covar, incomplete_cases );
		}
		if(n_env > 0) {
			E = reduce_mat_to_complete_cases( E, E_reduced, n_env, incomplete_cases );
		}
		n_samples -= incomplete_cases.size();
		missing_phenos.clear();
		missing_covars.clear();
		missing_envs.clear();

		std::cout << "Reduced to " << n_samples << " samples with complete data";
		std::cout << " across covariates";
		if(params.env_file != "NULL") std::cout << ", env-variables" << std::endl;
		std::cout << " and phenotype." << std::endl;
	}

	std::string fstream_init(boost_io::filtering_ostream& my_outf,
	                         const std::string& file_prefix,
	                         const std::string& file_suffix){

		std::string filepath   = params.out_file;
		std::string dir        = filepath.substr(0, filepath.rfind('/')+1);
		std::string stem_w_dir = filepath.substr(0, filepath.find('.'));
		std::string stem       = stem_w_dir.substr(stem_w_dir.rfind('/')+1, stem_w_dir.size());
		std::string ext        = filepath.substr(filepath.find('.'), filepath.size());

		std::string ofile      = dir + file_prefix + stem + file_suffix + ext;

		my_outf.reset();
		std::string gz_str = ".gz";
		if (params.out_file.find(gz_str) != std::string::npos) {
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
