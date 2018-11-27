// File of Data class for use with src/bgen_prog.cpp
#ifndef DATA_H
#define DATA_H

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstddef>     // for ptrdiff_t class
#include <chrono>      // start/end time info
#include <ctime>       // start/end time info
#include <map>
#include <vector>
#include <string>
#include <string>
#include <stdexcept>
#include "class.h"
#include "utils.hpp"
#include "my_timer.hpp"
#include "tools/eigen3.3/Dense"
#include "tools/eigen3.3/Sparse"
#include "tools/eigen3.3/Eigenvalues"

#include "genotype_matrix.hpp"

#include "bgen_parser.hpp"
#include "genfile/bgen/bgen.hpp"
#include "genfile/bgen/View.hpp"

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/complement.hpp> // complements
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>

namespace boost_io = boost::iostreams;
namespace boost_m = boost::math;

inline std::size_t find_covar_index( const std::string& colname, std::vector< std::string > col_names );


class Data
{
	public :
	parameters params;


	unsigned long n_pheno; // number of phenotypes
	unsigned long n_covar; // number of covariates
	unsigned long n_env; // number of env variables
	int n_effects;   // number of environmental interactions
	std::uint32_t n_samples; // number of samples
	bool bgen_pass;
	std::uint32_t n_var;
	std::size_t n_var_parsed; // Track progress through IndexQuery
	long int n_dxteex_computed;
	long int n_snpstats_computed;

	bool Y_reduced;   // Variables to track whether we have already
	bool W_reduced;   // reduced to complete cases or not.
	bool E_reduced;

	std::vector< std::string > external_dXtEEX_SNPID;
	std::vector< std::string > rsid_list;

	std::map<int, bool> missing_envs;   // set of subjects missing >= 1 env variables
	std::map<int, bool> missing_covars; // set of subjects missing >= 1 covariate
	std::map<int, bool> missing_phenos; // set of subjects missing >= phenotype
	std::map< std::size_t, bool > incomplete_cases; // union of samples missing data

	std::vector< std::string > pheno_names;
	std::vector< std::string > covar_names;
	std::vector< std::string > env_names;

	GenotypeMatrix G;
	EigenDataMatrix Y, Y2; // phenotype matrix (#2 always has covars regressed)
	EigenDataMatrix W; // covariate matrix
	EigenDataMatrix E; // env matrix
	Eigen::ArrayXXd dXtEEX;
	Eigen::ArrayXXd external_dXtEEX;
	Eigen::MatrixXd E_weights;
	genfile::bgen::View::UniquePtr bgenView;

	// For gxe genome-wide scan
	// cols neglogp-main, neglogp-gxe, coeff-gxe-main, coeff-gxe-env..
	Eigen::ArrayXXd snpstats;
	Eigen::ArrayXXd external_snpstats;
	std::vector< std::string > external_snpstats_SNPID;

	boost_io::filtering_ostream outf_scan;

	std::chrono::system_clock::time_point start;
	bool filters_applied;

	// grid things for vbayes
	std::vector< std::string > hyps_names;
	Eigen::MatrixXd r1_hyps_grid, hyps_grid;
	Eigen::ArrayXXd alpha_init, mu_init;

	explicit Data( const parameters& p ) : params(p), G(p) {
		// system time at start
		start = std::chrono::system_clock::now();
		std::time_t start_time = std::chrono::system_clock::to_time_t(start);
		std::cout << "Starting analysis at " << std::ctime(&start_time) << std::endl;
		std::cout << "Compiled from git branch: master" << std::endl;

		Eigen::setNbThreads(params.n_thread);
		int n = Eigen::nbThreads( );
		std::cout << "Threads used by eigen: " << n << std::endl;

		bgenView = genfile::bgen::View::create(p.bgen_file);
		bgen_pass = true;
		n_samples = (std::uint32_t) bgenView->number_of_samples();
		n_var_parsed = 0;
		filters_applied = false;

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
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end-start;
		std::time_t end_time = std::chrono::system_clock::to_time_t(end);
		std::cout << "Analysis finished at " << std::ctime(&end_time);
		std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;
	}

	void apply_filters(){
		// filter - incl sample ids
		if(params.incl_sids_file != "NULL"){
			read_incl_sids();
		}

		// filter - range
		if (params.range){
			std::cout << "Selecting range..." << std::endl;
			genfile::bgen::IndexQuery::UniquePtr query = genfile::bgen::IndexQuery::create(params.bgi_file);
			genfile::bgen::IndexQuery::GenomicRange rr1(params.chr, params.range_start, params.range_end);
			query->include_range( rr1 ).initialise();
			bgenView->set_query( query );
		}

		// filter - incl rsids
		if(params.select_snps){
			read_incl_rsids();
			std::cout << "Filtering SNPs by rsid..." << std::endl;
			genfile::bgen::IndexQuery::UniquePtr query = genfile::bgen::IndexQuery::create(params.bgi_file);
			query->include_rsids( rsid_list ).initialise();
			bgenView->set_query( query );
		}

		// filter - select single rsid
		if(params.select_rsid){
			std::sort(params.rsid.begin(), params.rsid.end());
			std::cout << "Filtering to rsids:" << std::endl;
			long int n_rsids = params.rsid.size();
			for (long int kk = 0; kk < n_rsids; kk++){
				std::cout << params.rsid[kk]<< std::endl;
			}
			genfile::bgen::IndexQuery::UniquePtr query = genfile::bgen::IndexQuery::create(params.bgi_file);
			query->include_rsids( params.rsid ).initialise();
			bgenView->set_query( query );
		}

		filters_applied = true;
	}

	void read_non_genetic_data(){
		// Apply sample / rsid / range filters if applicable
		if(!filters_applied){
			apply_filters();
		}

		// Read in phenotypes
		read_pheno();

		// Read in covariates if present
		if(params.covar_file != "NULL"){
			read_covar();
		}

		// Environmental vars - subset of covars
		if(params.env_file != "NULL"){
			read_environment();
        } else if(params.x_param_name != "NULL"){
            std::size_t x_col = find_covar_index(params.x_param_name, covar_names);
            E                = W.col(x_col);
            n_env            = 1;
            E_reduced        = false;
            env_names.push_back(params.x_param_name);
        } else if(params.interaction_analysis){
            E                = W.col(0);
            n_env            = 1;
            E_reduced        = false;
            env_names.push_back(covar_names[0]);
        } else {
		    n_env = 0;
		}

		if(params.env_weights_file != "NULL" && n_env > 1 && params.env_file != "NULL"){
			read_environment_weights();
		}

		if(params.interaction_analysis){
			n_effects = 2;  // 1 more than actually present... n_effects?
		} else {
			n_effects = 1;
			assert(n_env == 0);
		}

		// Exclude samples with missing values in phenos / covars / filters
		reduce_to_complete_cases();

		// Read in grids for importance sampling
		if (params.mode_vb) {
			read_grids();
		}

		if(n_env > 1 && params.dxteex_file != "NULL"){
			read_external_dxteex();
		}

		if(params.snpstats_file != "NULL"){
			read_external_snpstats();
		}

		// Read starting point for VB approximation if provided
		if(params.mode_vb && params.vb_init_file != "NULL"){
			// read_alpha_mu();
		}
	}

	void standardise_non_genetic_data(){
		// Step 3; Center phenos, normalise covars
		center_matrix( Y, n_pheno );
		if(params.scale_pheno){
			std::cout << "Scaling phenotype" << std::endl;
			scale_matrix( Y, n_pheno, pheno_names );
		}
		if(params.covar_file != "NULL"){
			center_matrix( W, n_covar );
			scale_matrix( W, n_covar, covar_names );
		}
		if(n_env > 0){
			center_matrix( E, n_env );
			scale_matrix( E, n_env, env_names );
		}

		// Y2 always contains the controlled phenotype
		if(n_covar > 0 && params.use_vb_on_covars){
			Y2 = Y;
			regress_out_covars(Y2);
		} else if (n_covar > 0){
			regress_out_covars(Y);
			Y2 = Y;
		}
	}

	void read_full_bgen(){
		std::cout << "Reading in BGEN" << std::endl;
		params.chunk_size = bgenView->number_of_variants();
		MyTimer t_readFullBgen("BGEN parsed in %ts \n");

		if(params.flip_high_maf_variants){
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
		std::string chr_j ;
		std::uint32_t pos_j ;
		std::string rsid_j ;
		std::vector< std::string > alleles_j ;
		std::string SNPID_j ; // read but ignored
		std::vector< std::vector< double > > probs ;
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
					if(check > 0.9999 && check < 1.0001){
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
			if(params.flip_high_maf_variants && maf_j > 0.5){
				dosage_j = (2.0 - dosage_j);
				for (std::uint32_t ii = 0; ii < n_samples; ii++){
					if (missing_genos.count(ii) > 0){
						dosage_j[ii] = 0.0;
					}
				}

				f2       = 4.0 * valid_count - 4.0 * d1 + f2;
				d1       = dosage_j.sum();
				maf_j    = d1 / (2.0 * valid_count);
			}

			double mu    = d1 / valid_count;
			info_j       = 1.0;
			if(maf_j > 1e-10 && maf_j < 0.9999999999){
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
			if(!params.keep_constant_variants && d1 < 5.0){
				n_constant_variance++;
				continue;
			}
			if(!params.keep_constant_variants && sigma <= 1e-12){
				n_constant_variance++;
				continue;
			}

			// filters passed; write contextual info
			G.al_0.push_back(alleles_j[0]);
			G.al_1.push_back(alleles_j[1]);
			G.maf.push_back(maf_j);
			G.info.push_back(info_j);
			G.rsid.push_back(rsid_j);
			G.chromosome.push_back(std::stoi(chr_j));
			G.position.push_back(pos_j);
			std::string key_j = chr_j + "~" + std::to_string(pos_j) + "~" + alleles_j[0] + "~" + alleles_j[1];
			G.SNPKEY.push_back(key_j);
			G.SNPID.push_back(SNPID_j);

			// filters passed; write dosage to G
			// Note that we only write dosage for valid sample ids
			// Scale + center dosage, set missing to mean
			if(!missing_genos.empty()){
				n_var_incomplete += 1;
				missing_calls += (double) missing_genos.size();
				for (std::uint32_t ii = 0; ii < n_samples; ii++){
					if (missing_genos.count(ii) > 0){
						dosage_j[ii] = mu;
					}
				}
			}

			for (std::uint32_t ii = 0; ii < n_samples; ii++){
				G.assign_index(ii, jj, dosage_j[ii]);
			}
			// G.compressed_dosage_sds[jj] = sigma;
			// G.compressed_dosage_means[jj] = mu;

			jj++;
		}

		// need to resize G whilst retaining existing coefficients if while
		// loop exits early due to EOF.
		G.conservativeResize(n_samples, jj);
		assert( G.rsid.size() == jj );
		n_var = jj;

		chunk_missingness = missing_calls / (double) (n_var * n_samples);
		if(chunk_missingness > 0.0){
			std::cout << "Chunk missingness " << chunk_missingness << "(";
 			std::cout << n_var_incomplete << "/" << n_var;
			std::cout << " variants incomplete)" << std::endl;
		}

		if(n_constant_variance > 0){
			std::cout << n_constant_variance << " variants removed due to ";
			std::cout << "constant variance" << std::endl;
		}

		if(jj == 0){
			// Immediate EOF
			return false;
		} else {
			return true;
		}
	}

	void read_incl_rsids(){
		boost_io::filtering_istream fg;
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
		fg.push(boost_io::file_source(params.incl_sids_file));
		if (!fg) {
			std::cout << "ERROR: " << params.incl_sids_file << " not opened." << std::endl;
			std::exit(EXIT_FAILURE);
		}

		std::vector<std::string> bgen_ids;
		bgenView->get_sample_ids(
			[&]( std::string const& id ) { bgen_ids.push_back(id); }
		);

		// Want to read in a sid to be included, and skip along bgen_ids until
		// we find it.
		int ii = 0, bb = 0;
		std::stringstream ss;
		std::string line;
		try {
			while (getline(fg, line)) {
				ss.clear();
				ss.str(line);
				std::string s;
				ss >> s;
				if (bb >= n_samples){
					throw std::logic_error("ERROR: Either you have tried "
					"to include an id not present in the BGEN file, or the "
					"the provided ids are in the wrong order");
				}
				while(s != bgen_ids[bb]) {
					incomplete_cases[bb] = true;
					bb++;
					if (bb >= n_samples){
						std::cout << "Failed to find a match for sample_id:";
						std::cout << "The first 10 bgen ids are:" << std::endl;
						for(int iii = 0; iii < 10; iii++){
							std::cout << bgen_ids[iii] << std::endl;
						}
						std::cout << s << std::endl;
						throw std::logic_error("ERROR: Either you have tried "
						"to include an id not present in the BGEN file, or the "
						"the provided ids are in the wrong order");
					}
				}

				// bgen_ids[bb] == s
				bb++;
				ii++;
			}

			for (int jj = bb; jj < n_samples; jj++){
				incomplete_cases[jj] = true;
			}
		} catch (const std::exception &exc) {
			// throw std::runtime_error("ERROR: problem converting incl_sample_ids.");
			throw;
		}
		// n_samples = ii;
		std::cout << "Subsetted down to " << ii << " ids from --incl_sample_ids";
		std::cout << std::endl;
	}

	void read_grid_file( const std::string& filename,
						 Eigen::MatrixXd& M,
						 std::vector< std::string >& col_names){
		// Used in mode_vb only.
		// Slightly different from read_txt_file in that I don't know
		// how many rows there will be and we can assume no missing values.

		boost_io::filtering_istream fg;
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
		fg.push(boost_io::file_source(filename));

		// Reading column names
		if (!getline(fg, line)) {
			std::cout << "ERROR: " << filename << " not read." << std::endl;
			std::exit(EXIT_FAILURE);
		}
		std::stringstream ss;
		std::string s1;
		int n_cols = 0;
		ss.clear();
		ss.str(line);
		while (ss >> s1) {
			++n_cols;
			col_names.push_back(s1);
		}
		std::cout << n_grid << " x " << n_cols << " matrix read from " << filename << std::endl;

		// Write remainder of file to Eigen matrix M
		M.resize(n_grid, n_cols);
		int i = 0;
		double tmp_d;
		try {
			while (getline(fg, line)) {
				if (i >= n_grid) {
					throw std::runtime_error("ERROR: could not convert txt file (too many lines).");
				}
				ss.clear();
				ss.str(line);
				for (int k = 0; k < n_cols; k++) {
					std::string s;
					ss >> s;
					try{
						tmp_d = stod(s);
					} catch (const std::invalid_argument &exc){
						std::cout << s << " on line " << i << std::endl;
						throw;
					}

					M(i, k) = tmp_d;
				}
				i++; // loop should end at i == n_grid
			}
			if (i < n_grid) {
				throw std::runtime_error("ERROR: could not convert txt file (too few lines).");
			}
		} catch (const std::exception &exc) {
			throw;
		}
	}


	void read_vb_init_file(const std::string& filename,
                           Eigen::MatrixXd& M,
                           std::vector< std::string >& col_names,
                           std::vector< std::string >& init_key){
		// Used in mode_vb only.
		// Need custom function to deal with variable input. Sometimes
		// we have string columns with rsid / a0 etc
		// init_chr, init_pos, init_a0, init_a1;

		boost_io::filtering_istream fg;
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
		fg.push(boost_io::file_source(filename));

		// Reading column names
		if (!getline(fg, line)) {
			std::cout << "ERROR: " << filename << " not read." << std::endl;
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
		std::cout << " Detected " << n_cols << " column(s) from " << filename << std::endl;

		// Write remainder of file to Eigen matrix M
		M.resize(n_grid, n_cols);
		int i = 0;
		double tmp_d;
		std::string key_i;
		try {
			while (getline(fg, line)) {
				if (i >= n_grid) {
					throw std::runtime_error("ERROR: could not convert txt file (too many lines).");
				}
				ss.clear();
				ss.str(line);
				if(n_cols < 7){
					for (int k = 0; k < n_cols; k++) {
						std::string sss;
						ss >> sss;
						try{
							tmp_d = stod(sss);
						} catch (const std::invalid_argument &exc){
							std::cout << sss << " on line " << i << std::endl;
							throw;
						}
						M(i, k) = tmp_d;
					}
				} else if(n_cols == 7){
					key_i = "";
					for (int k = 0; k < n_cols; k++) {
						std::string sss;
						ss >> sss;
						if(k == 0){
							key_i += sss + "~";
						} else if (k == 2){
							key_i += sss + "~";
							// init_pos.push_back(std::stoull(s));
						} else if (k == 3){
							key_i += sss + "~";
							// init_a0.push_back(s);
						} else if (k == 4){
							key_i += sss;
							init_key.push_back(key_i);
							// init_a1.push_back(s);
						} else if(k >= 5){
							M(i, k) = stod(sss);
						}
					}
				} else {
					throw std::runtime_error("Unexpected number of columns.");
				}
				i++; // loop should end at i == n_grid
			}
			if (i < n_grid) {
				throw std::runtime_error("ERROR: could not convert txt file (too few lines).");
			}
		} catch (const std::exception &exc) {
			throw;
		}
	}

	void read_txt_file( const std::string& filename,
						Eigen::MatrixXd& M,
						unsigned long& n_cols,
						std::vector< std::string >& col_names,
						std::map< int, bool >& incomplete_row ){
		// pass top line of txt file filename to col_names, and body to M.
		// TODO: Implement how to deal with missing values.

		boost_io::filtering_istream fg;
		fg.push(boost_io::file_source(filename));
		if (!fg) {
			std::cout << "ERROR: " << filename << " not opened." << std::endl;
			std::exit(EXIT_FAILURE);
		}

		// Reading column names
		std::string line;
		if (!getline(fg, line)) {
			std::cout << "ERROR: " << filename << " not read." << std::endl;
			std::exit(EXIT_FAILURE);
		}
		std::stringstream ss;
		std::string s;
		n_cols = 0;
		ss.clear();
		ss.str(line);
		while (ss >> s) {
			++n_cols;
			col_names.push_back(s);
		}
		std::cout << " Detected " << n_cols << " column(s) from " << filename << std::endl;

		// Write remainder of file to Eigen matrix M
		incomplete_row.clear();
		M.resize(n_samples, n_cols);
		int i = 0;
		double tmp_d;
		try {
			while (getline(fg, line)) {
				if (i >= n_samples) {
					throw std::runtime_error("ERROR: could not convert txt file (too many lines).");
				}
				ss.clear();
				ss.str(line);
				for (int k = 0; k < n_cols; k++) {
					std::string sss;
					ss >> sss;
					/// NA
					if (sss == "NA" || sss == "NAN" || sss == "NaN" || sss == "nan") {
						tmp_d = params.missing_code;
					} else {
						try{
							tmp_d = stod(sss);
						} catch (const std::invalid_argument &exc){
							std::cout << sss << " on line " << i << std::endl;
							throw;
						}
					}

					if(tmp_d != params.missing_code) {
						M(i, k) = tmp_d;
					} else {
						M(i, k) = params.missing_code;
						incomplete_row[i] = true;
					}
				}
				i++; // loop should end at i == n_samples
			}
			if (i < n_samples) {
				throw std::runtime_error("ERROR: could not convert txt file (too few lines).");
			}
		} catch (const std::exception &exc) {
			throw;
		}
	}

	void read_txt_file( const std::string& filename,
						Eigen::MatrixXf& M,
						unsigned long& n_cols,
						std::vector< std::string >& col_names,
						std::map< int, bool >& incomplete_row ){
		// pass top line of txt file filename to col_names, and body to M.
		// TODO: Implement how to deal with missing values.

		boost_io::filtering_istream fg;
		fg.push(boost_io::file_source(filename));
		if (!fg) {
			std::cout << "ERROR: " << filename << " not opened." << std::endl;
			std::exit(EXIT_FAILURE);
		}

		// Reading column names
		std::string line;
		if (!getline(fg, line)) {
			std::cout << "ERROR: " << filename << " not read." << std::endl;
			std::exit(EXIT_FAILURE);
		}
		std::stringstream ss;
		std::string s;
		n_cols = 0;
		ss.clear();
		ss.str(line);
		while (ss >> s) {
			++n_cols;
			col_names.push_back(s);
		}
		std::cout << " Detected " << n_cols << " column(s) from " << filename << std::endl;

		// Write remainder of file to Eigen matrix M
		incomplete_row.clear();
		M.resize(n_samples, n_cols);
		int i = 0;
		float tmp_d;
		try {
			while (getline(fg, line)) {
				if (i >= n_samples) {
					throw std::runtime_error("ERROR: could not convert txt file (too many lines).");
				}
				ss.clear();
				ss.str(line);
				for (int k = 0; k < n_cols; k++) {
					std::string sss;
					ss >> sss;
					/// NA
					if (sss == "NA" || sss == "NAN" || sss == "NaN" || sss == "nan") {
						M(i, k) = params.missing_code;
						incomplete_row[i] = true;
					} else {
						try{
							tmp_d = std::stof(sss);
							M(i, k) = tmp_d;
						} catch (const std::invalid_argument &exc){
							std::cout << sss << " on line " << i << std::endl;
							throw;
						}
					}
				}
				i++; // loop should end at i == n_samples
			}
			if (i < n_samples) {
				throw std::runtime_error("ERROR: could not convert txt file (too few lines).");
			}
		} catch (const std::exception &exc) {
			throw;
		}
	}

	template <typename Derived>
	void center_matrix( Eigen::MatrixBase<Derived>& M,
						unsigned long& n_cols ){
		// Center eigen matrix passed by reference.
		// Only call on matrixes which have been reduced to complete cases,
		// as no check for incomplete rows.

		for (int k = 0; k < n_cols; k++) {
			double mu = 0.0;
			double count = 0;
			for (int i = 0; i < n_samples; i++) {
				mu += M(i, k);
				count += 1;
			}

			mu = mu / count;
			for (int i = 0; i < n_samples; i++) {
				M(i, k) -= mu;
			}
			// std::cout << "Mean centered matrix:" << std::endl << M << std::endl;
		}
	}

	template <typename EigenMat>
	void scale_matrix( EigenMat& M,
						unsigned long& n_cols,
 						std::vector< std::string >& col_names){
		// Scale eigen matrix passed by reference.
		// Removes columns with zero variance + updates col_names.
		// Only call on matrixes which have been reduced to complete cases,
		// as no check for incomplete rows.

		std::vector<std::size_t> keep;
		std::vector<std::string> keep_names;
		std::vector<std::string> reject_names;
		for (std::size_t k = 0; k < n_cols; k++) {
			double sigma = 0.0;
			double count = 0;
			for (int i = 0; i < n_samples; i++) {
				double val = M(i, k);
				sigma += val * val;
				count += 1;
			}

			sigma = sqrt(sigma/(count - 1));
			if (sigma > 1e-12) {
				for (int i = 0; i < n_samples; i++) {
					M(i, k) /= sigma;
				}
				keep.push_back(k);
				keep_names.push_back(col_names[k]);
			} else {
				reject_names.push_back(col_names[k]);
			}
		}

		if (keep.size() != n_cols) {
			std::cout << " Removing " << (n_cols - keep.size())  << " column(s) with zero variance:" << std::endl;
			for(int kk = 0; kk < (n_cols - keep.size()); kk++){
				std::cout << reject_names[kk] << std::endl;
			}
			// subset cols
			for (std::size_t i = 0; i < keep.size(); i++) {
				M.col(i) = M.col(keep[i]);
			}
			M.conservativeResize(M.rows(), keep.size());

			n_cols = keep.size();
			col_names = keep_names;
		}

		if (n_cols == 0) {
			throw std::runtime_error("ERROR: No columns left with nonzero variance after scale_matrix()");
		}
	}

	void read_pheno( ){
		// Read phenotypes to Eigen matrix Y
		if ( params.pheno_file != "NULL" ) {
			read_txt_file( params.pheno_file, Y, n_pheno, pheno_names, missing_phenos );
		} else {
			throw std::invalid_argument( "Tried to read NULL pheno file." );
		}
		if(n_pheno != 1){
			std::cout << "ERROR: Only expecting one phenotype at a time." << std::endl;
		}
		Y_reduced = false;
	}

	void read_covar( ){
		// Read covariates to Eigen matrix W
		if ( params.covar_file != "NULL" ) {
			read_txt_file( params.covar_file, W, n_covar, covar_names, missing_covars );
		} else {
			throw std::logic_error( "Tried to read NULL covar file." );
		}
		W_reduced = false;
	}

	void read_environment( ){
		// Read covariates to Eigen matrix W
		if ( params.env_file != "NULL" ) {
			read_txt_file( params.env_file, E, n_env, env_names, missing_envs );
		} else {
			throw std::logic_error( "Tried to read NULL env file." );
		}
		E_reduced = false;
	}

	void read_environment_weights( ){
		unsigned long n_cols;
		std::vector< std::string > col_names;
		std::map<int, bool> missing_rows;
		// read_txt_file( params.env_weights_file, E_weights, n_cols, col_names, missing_rows );
		read_grid_file(params.env_weights_file, E_weights, col_names);

		assert(E_weights.rows() == n_env);
		assert(missing_rows.empty());
	}

	void read_external_dxteex( ){
		std::vector< std::string > col_names;
		read_txt_file_w_context( params.dxteex_file, 6, external_dXtEEX,
                                 external_dXtEEX_SNPID, col_names);

		if(external_dXtEEX.cols() != n_env * n_env){
			std::cout << "Expecting columns in order: " << std::endl;
			std::cout << "SNPID, chr, rsid, pos, allele0, allele1, env-snp covariances.." << std::endl;
			throw std::runtime_error("Unexpected number of columns");
		}
	}

	void read_external_snpstats( ){
		std::vector< std::string > col_names;
		read_txt_file_w_context( params.snpstats_file, 8, external_snpstats,
								 external_snpstats_SNPID, col_names);

		if(external_snpstats.cols() != n_env + 3){
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
		Txt file where the first column is snp ids, then x-1 contextual,
		then a matrix to be read into memory.

		Reads file twice to ascertain number of lines.

		col_offset - how many contextual columns to skip
		*/

		// Reading from file
		boost_io::filtering_istream fg;
		fg.push(boost_io::file_source(filename));
		if (!fg) {
			std::cout << "ERROR: " << filename << " not opened." << std::endl;
			std::exit(EXIT_FAILURE);
		}

		// Read file twice to ascertain number of lines
		int n_lines = 0;
		std::string line;
		getline(fg, line); // skip header
		while (getline(fg, line)) {
			n_lines++;
		}
		fg.reset();
		fg.push(boost_io::file_source(filename));

		// Reading column names
		if (!getline(fg, line)) {
			std::cout << "ERROR: " << filename << " not read." << std::endl;
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
				if (k == 0){
					M_snpids.push_back(sss);
				}
				if (k >= col_offset){
					try{
						M(i, k-col_offset) = stod(sss);
					} catch (const std::invalid_argument &exc){
						std::cout << "Found value " << sss << " on line " << i;
	 					std::cout << " of file " << filename << std::endl;
						throw std::runtime_error("Unexpected value");
					}
				}
			}
			i++; // loop should end at i == n_samples
		}
		std::cout << n_lines << " rows found in " << filename << std::endl;
	}

	void calc_snpstats(){
		std::cout << "Reordering/computing results for snpwise scan" << std::endl;
		MyTimer my_timer("snpwise scan constructed in %ts \n");
		snpstats.resize(n_var, n_env + 3);
		std::vector<std::string>::iterator it;
		n_snpstats_computed = 0;
		auto N = (double) n_samples;
		boost_m::fisher_f f_dist(n_env, n_samples - n_env - 1);
		for (std::size_t jj = 0; jj < n_var; jj++){
			it = std::find(external_snpstats_SNPID.begin(), external_snpstats_SNPID.end(), G.SNPID[jj]);
			if (it == external_snpstats_SNPID.end()){
				n_snpstats_computed++;
				EigenDataVector X_kk = G.col(jj);
				EigenDataMatrix H(n_samples, 1 + n_env);
				H << X_kk, (E.array().colwise() * X_kk.array()).matrix();

				Eigen::MatrixXd HtH = (H.transpose() * H).cast<double>();
				Eigen::MatrixXd Hty = (H.transpose() * Y2).cast<double>();

				// Fitting regression models
				EigenDataMatrix tau1_j = X_kk.transpose() * Y2 / (N-1.0);
				Eigen::MatrixXd tau2_j = solve(HtH, Hty);
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
				if(!std::isfinite(gxe_neglogp_j) || f_stat < 0){
					std::cout << "Warning: neglog-p = " << gxe_neglogp_j << std::endl;
					std::cout << "Warning: p-val = "    << gxe_pval_j << std::endl;
					std::cout << "Warning: rss_null = " << rss_null << std::endl;
					std::cout << "Warning: rss_alt = "  << rss_alt << std::endl;
					std::cout << "Warning: f_stat = "   << f_stat << std::endl;
				}

				// Log relevant stats
				snpstats(jj, 1)    = -1 * std::log10(gxe_pval_j);
				snpstats(jj, 0)    = -1 * std::log10(main_pval_j);
				for (int ee = 0; ee < n_env + 1; ee++){
					snpstats(jj, ee + 2) = tau2_j(ee);
				}
			} else {
				snpstats.row(jj) = external_snpstats.row(it - external_snpstats_SNPID.begin());
			}
		}
		std::cout << n_snpstats_computed << " computed from raw data, " << n_var - n_snpstats_computed << " read from file" << std::endl;

		if(params.snpstats_file == "NULL"){
			std::string ofile_scan = fstream_init(outf_scan, "", "_snpwise_scan");
			std::cout << "Writing snp-wise scan to file " << ofile_scan << std::endl;

			outf_scan << "chr rsid pos a0 a1 af info neglogp_main neglogp_gxe";
			outf_scan << std::endl;
			for (std::uint32_t kk = 0; kk < n_var; kk++){
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
		std::cout << "Reordering/building dXtEEX array" << std::endl;
		MyTimer t_calcDXtEEX("dXtEEX array constructed in %ts \n");
		t_calcDXtEEX.resume();
		EigenDataArrayX cl_j;
		scalarData dztz_lmj;
		dXtEEX.resize(n_var, n_env * n_env);
		std::vector<std::string>::iterator it;
		n_dxteex_computed = 0;
		for (std::size_t jj = 0; jj < n_var; jj++){
			it = std::find(external_dXtEEX_SNPID.begin(), external_dXtEEX_SNPID.end(), G.SNPID[jj]);
			if (it == external_dXtEEX_SNPID.end()){
				n_dxteex_computed++;
				cl_j = G.col(jj);
				for (int ll = 0; ll < n_env; ll++){
					for (int mm = 0; mm <= ll; mm++){
						dztz_lmj = (cl_j * E.array().col(ll) * E.array().col(mm) * cl_j).sum();
						dXtEEX(jj, ll*n_env + mm) = dztz_lmj;
						dXtEEX(jj, mm*n_env + ll) = dztz_lmj;
					}
				}
			} else {
				dXtEEX.row(jj) = external_dXtEEX.row(it - external_dXtEEX_SNPID.begin());
			}
		}
		std::cout << n_dxteex_computed << " computed from raw data, " << n_var - n_dxteex_computed << " read from file" << std::endl;
	}

	void read_grids(){
		// For use in vbayes object

		std::vector< std::string > true_fixed_names = {"sigma", "sigma_b", "lambda_b"};
		std::vector< std::string > true_gxage_names = {"sigma", "sigma_b", "sigma_g", "lambda_b", "lambda_g"};


		read_grid_file( params.hyps_grid_file, hyps_grid, hyps_names );
//		read_grid_file( params.hyps_probs_file, imprt_grid, imprt_names );

		// Verify header of grid file as expected
		if(params.interaction_analysis){
			if(!std::includes(hyps_names.begin(), hyps_names.end(), true_gxage_names.begin(), true_gxage_names.end())){
				throw std::runtime_error("Column names of --hyps_grid must be sigma sigma_b sigma_g lambda_b lambda_g");
			}
		} else {
			if(hyps_names != true_fixed_names){
				// Allow gxe params if coeffs set to zero
				if(std::includes(hyps_names.begin(), hyps_names.end(), true_gxage_names.begin(), true_gxage_names.end())){
					double sigma_g_sum  = hyps_grid.col(2).array().abs().sum();
					double lambda_g_sum = hyps_grid.col(4).array().abs().sum();
					if(sigma_g_sum > 1e-6 || lambda_g_sum > 1e-6){
						throw std::runtime_error("In a main effects only analysis columns for sigma_g and lambda_g should either be absent or contain zeros");
					}
				} else {
					throw std::runtime_error("Column names of --hyps_grid must be sigma sigma_b lambda_b or sigma sigma_b sigma_g lambda_b lambda_g");
				}
			}
		}

		// If

		// Option to provide separate grid to evaluate in round 1
		std::vector< std::string > r1_hyps_names, r1_probs_names;
		if ( params.r1_hyps_grid_file != "NULL" ) {
			read_grid_file( params.r1_hyps_grid_file, r1_hyps_grid, r1_hyps_names );
			if(hyps_names != r1_hyps_names){
				throw std::invalid_argument( "Header of --r1_hyps_grid must match --hyps_grid." );
			}
//			read_grid_file( params.r1_probs_grid_file, r1_probs_grid, r1_probs_names );
		}
	}

	void read_alpha_mu(){
		// For use in vbayes object
		Eigen::MatrixXd vb_init_mat;
		// std::vector< int > init_chr;
		// std::vector< uint32_t > init_pos;
		// chr~pos~a0~a1
		std::vector< std::string > init_key;
		// std::vector< std::string > init_a1;

		std::vector< std::string > vb_init_colnames;
		std::vector< std::string > cols_check1 = {"alpha", "mu"};
		std::vector<std::string> cols_check2 = {"chr", "rsid", "pos", "a0", "a1", "beta", "gamma"};

		if ( params.vb_init_file != "NULL" ) {
			std::cout << "Reading initialisation for alpha from file" << std::endl;
			read_vb_init_file(params.vb_init_file, vb_init_mat, vb_init_colnames,
                              init_key);

			if(vb_init_mat.cols() < 7){
				if(!std::includes(vb_init_colnames.begin(), vb_init_colnames.end(), cols_check1.begin(), cols_check1.end())){
					throw std::runtime_error("First 2 columns of --vb_init should be alpha mu ");
				}
				alpha_init = Eigen::Map<Eigen::ArrayXXd>(vb_init_mat.col(0).data(), n_var, n_effects);
				mu_init = Eigen::Map<Eigen::ArrayXXd>(vb_init_mat.col(1).data(), n_var, n_effects);
			} else {
				for(int aa = 0; aa < 7; aa++){
					std::cout << vb_init_colnames[aa] << std::endl;
				}
				assert(vb_init_colnames == cols_check2);
				std::cout << "--vb_init file with contextual information detected" << std::endl;
				std::cout << "Warning: This will be O(PL) where L = " << vb_init_mat.rows();
				std::cout << " is the number of lines in file given to --vb_init." << std::endl;
				alpha_init = Eigen::MatrixXd::Zero(n_var, n_effects);
				mu_init    = Eigen::MatrixXd::Zero(n_var, n_effects);

				std::vector<std::string>::iterator it;
				unsigned long index_kk;
				for(int kk = 0; kk < vb_init_mat.rows(); kk++){
					it = std::find(G.SNPKEY.begin(), G.SNPKEY.end(), init_key[kk]);
					if (it == G.SNPKEY.end()){
						std::cout << "WARNING: Can't locate variant with key: ";
						std::cout << init_key[kk] << std::endl;
					} else {
						index_kk = it - G.SNPKEY.begin();

						alpha_init(index_kk, 0)      = 1.0;
						mu_init(index_kk, 0)         = vb_init_mat(kk, 5);
						if(n_effects > 1) {
							alpha_init(index_kk, 1) = 1.0;
							mu_init(index_kk, 1) = vb_init_mat(kk, 6);
						}
					}
				}
			}

		} else {
			throw std::invalid_argument( "Tried to read NULL --vb_init file." );
		}

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

	void regress_out_covars(EigenDataMatrix& yy){
		std::cout << "Regressing out covars:" << std::endl;
		for(int cc = 0; cc < std::min(n_covar, (unsigned long) 10); cc++){
			std::cout << ( cc > 0 ? ", " : "" ) << covar_names[cc];
		}
		if (n_covar > 10){
			std::cout << "... (" << n_covar << " variables)";
		}
		std::cout << std::endl;

		Eigen::MatrixXd WtW = (W.transpose() * W).cast<double>();
		Eigen::MatrixXd Wty = (W.transpose() * yy).cast<double>();

		Eigen::MatrixXd bb = solve(WtW, Wty);
#ifdef DATA_AS_FLOAT
		yy -= W * bb.cast<float>();
#else
		yy -= W * bb;
#endif
	}

	void reduce_to_complete_cases() {
		// Remove any samples with incomplete covariates or phenotypes from
		// Y and W.
		// Note; other functions (eg. read_incl_sids) may add to incomplete_cases
		// Note; during unit testing sometimes only phenos or covars present.

		incomplete_cases.insert(missing_covars.begin(), missing_covars.end());
		incomplete_cases.insert(missing_phenos.begin(), missing_phenos.end());
		if(params.env_file != "NULL"){
			incomplete_cases.insert(missing_envs.begin(), missing_envs.end());
		}

		if(n_pheno > 0){
			Y = reduce_mat_to_complete_cases( Y, Y_reduced, n_pheno, incomplete_cases );
		}
		if(n_covar > 0){
			W = reduce_mat_to_complete_cases( W, W_reduced, n_covar, incomplete_cases );
		}
		if(n_env > 0){
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

inline std::size_t find_covar_index(const std::string& colname, std::vector< std::string > col_names ){
	std::size_t x_col;
	std::vector<std::string>::iterator it;
	it = std::find(col_names.begin(), col_names.end(), colname);
	if (it == col_names.end()){
		throw std::invalid_argument("Can't locate parameter " + colname);
	}
	x_col = it - col_names.begin();
	return x_col;
}

#endif
