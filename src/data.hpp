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

inline Eigen::MatrixXd getCols(const Eigen::MatrixXd &X, const std::vector<size_t> &cols);
inline void setCols(Eigen::MatrixXd &X, const std::vector<size_t> &cols, const Eigen::MatrixXd &values);
inline size_t numRows(const Eigen::MatrixXd &A);
inline size_t numCols(const Eigen::MatrixXd &A);
inline void setCol(Eigen::MatrixXd &A, const Eigen::VectorXd &v, size_t col);
inline Eigen::VectorXd getCol(const Eigen::MatrixXd &A, size_t col);
inline Eigen::MatrixXd solve(const Eigen::MatrixXd &A, const Eigen::MatrixXd &b);


class Data 
{
	public :
	parameters params;

	std::vector< std::string > chromosome, rsid;
	std::vector< uint32_t > position;
	std::vector< std::vector< std::string > > alleles;
	
	int n_pheno; // number of phenotypes
	int n_covar; // number of covariates
	int n_samples; // number of samples
	long int n_snps; // number of snps
	bool bgen_pass;
	int n_var;
	std::size_t n_var_parsed; // Track progress through IndexQuery

	bool Y_reduced;   // Variables to track whether we have already
	bool W_reduced;   // reduced to complete cases or not.

	std::vector< double > info;
	std::vector< double > maf;
	std::vector< std::string > rsid_list;

	std::map<int, bool> missing_covars; // set of subjects missing >= 1 covariate
	std::map<int, bool> missing_phenos; // set of subjects missing >= phenotype
	std::map< int, bool > incomplete_cases; // union of samples missing data

	std::vector< std::string > pheno_names;
	std::vector< std::string > covar_names;

	GenotypeMatrix G;
	Eigen::MatrixXd Y; // phenotype matrix
	Eigen::MatrixXd W; // covariate matrix
	Eigen::VectorXd Z; // interaction vector
	genfile::bgen::View::UniquePtr bgenView;
	std::vector< double > beta, tau, neglogP, neglogP_2dof;
	std::vector< std::vector< double > > gamma;

	boost_io::filtering_ostream outf;

	std::chrono::system_clock::time_point start;
	bool filters_applied;

	// grid things for vbayes
	std::vector< std::string > hyps_names, imprt_names;
	Eigen::MatrixXd r1_hyps_grid, r1_probs_grid, hyps_grid, imprt_grid;
	Eigen::VectorXd alpha_init, mu_init;

	
	// constructors/destructors
	// data() : bgenView( "NULL" ) {
	// 	bgen_pass = false; // No bgen file set; read_bgen_chunk won't run.
	// }

	// Data( const std::string& filename ) : G(false) {
	// 	// system time at start
	// 	start = std::chrono::system_clock::now();
	// 	std::time_t start_time = std::chrono::system_clock::to_time_t(start);
	// 	std::cout << "Starting analysis at " << std::ctime(&start_time) << std::endl;
	// 	std::cout << "Compiled from git branch: master" << std::endl;
	// 
	// 	bgenView = genfile::bgen::View::create(filename);
	// 	bgen_pass = true;
	// 	n_samples = bgenView->number_of_samples();
	// 	n_var_parsed = 0;
	// 	filters_applied = false;
	// }

	Data( const parameters& p ) : params(p), G(p.low_mem) {
		// system time at start
		start = std::chrono::system_clock::now();
		std::time_t start_time = std::chrono::system_clock::to_time_t(start);
		std::cout << "Starting analysis at " << std::ctime(&start_time) << std::endl;
		std::cout << "Compiled from git branch: master" << std::endl;

		bgenView = genfile::bgen::View::create(p.bgen_file);
		bgen_pass = true;
		n_samples = bgenView->number_of_samples();
		n_var_parsed = 0;
		filters_applied = false;
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
			genfile::bgen::IndexQuery::GenomicRange rr1(params.chr, params.start, params.end);
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
			for (int kk = 0; kk < params.rsid.size(); kk++) std::cout << params.rsid[kk]<< std::endl;
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

		// Exclude samples with missing values in phenos / covars / filters
		reduce_to_complete_cases();

		// Read in grids for importance sampling
		if (params.mode_vb) {
			read_grids();
		}

		// Read starting point for VB approximation if provided
		if(params.mode_vb && params.vb_init_file != "NULL"){
			// read_alpha_mu();
		}
	}

	void standardise_non_genetic_data(){
		// Step 3; Center phenos, normalise covars
		center_matrix( Y, n_pheno );
		// scale_matrix( Y, n_pheno, pheno_names );
		if(params.covar_file != "NULL"){
			center_matrix( W, n_covar );
			scale_matrix( W, n_covar, covar_names );
		}
	}

	void read_full_bgen(){
		params.chunk_size = bgenView->number_of_variants();
		read_bgen_chunk();
	}

	bool read_bgen_chunk() {
		// Wrapper around BgenView to read in a 'chunk' of data. Remembers
		// if last call hit the EOF, and returns false if so.
		// Assumed that:
		// - commandline args parsed and passed to params
		// - bgenView initialised with correct filename

		// Exit function if last call hit EOF.
		if (!bgen_pass) return false;

		// Temporary variables to store info from read_variant()
		std::string chr_j ;
		uint32_t pos_j ;
		std::string rsid_j ;
		std::vector< std::string > alleles_j ;
		std::string SNPID ; // read but ignored
		std::vector< std::vector< double > > probs ;
		ProbSetter setter( &probs );
		std::map<int, bool> missing_genos;

		double d1, theta, x, dosage, check, info_j, f1, f2, chunk_missingness;
		double dosage_mean, dosage_sigma, missing_calls = 0.0;
		int n_var_incomplete = 0;

		// Wipe variant context from last chunk
		maf.clear();
		info.clear();
		rsid.clear();
		chromosome.clear();
		position.clear();
		alleles.clear();

		// Resize genotype matrix
		G.resize(n_samples, params.chunk_size);
		// Eigen::VectorXd dosage_j(n_samples);

		long int n_constant_variance = 0;
		double valid_count;
		std::size_t jj = 0;
		while ( jj < params.chunk_size && bgen_pass ) {
			bgen_pass = bgenView->read_variant( &SNPID, &rsid_j, &chr_j, &pos_j, &alleles_j );
			n_var_parsed++;
			if (!bgen_pass) break;
			assert( alleles_j.size() > 0 );

			// Read probs + check maf filter
			bgenView->read_genotype_data_block( setter );

			// maf + info filters; computed on valid sample_ids & variants whose alleles
			// sum to 1
			std::map< std::size_t, bool > invalid_count;
			d1 = f2 = valid_count = 0.0;
			for( std::size_t ii = 0; ii < probs.size(); ++ii ) {
				if (incomplete_cases.count(ii) == 0) {
					f1 = dosage = check = 0.0;
					for( std::size_t kk = 0; kk < probs[ii].size(); ++kk ) {
						x = probs[ii][kk];
						dosage += x * kk;
						f1 += x * kk * kk;
						check += x;
					}
					if(check > 0.9999 && check < 1.0001){
						d1 += dosage;   // dosage mean
						f2 += (f1 - dosage * dosage);
						valid_count++;
					} else {
						invalid_count[ii] = 1;
					}
				}
			}
			theta = d1 / (2.0 * valid_count);
			info_j = 1.0;
			if(theta > 1e-10 && theta < 0.9999999999){
				info_j -= f2 / (2.0 * valid_count * theta * (1.0 - theta));
			}
			if (params.maf_lim && (theta < params.min_maf || theta > 1 - params.min_maf)) {
				continue;
			}
			if (params.info_lim && info_j < params.min_info) {
				continue;
			}

			// check non-zero variance
			double mu = d1 / valid_count;
			if(!params.keep_constant_variants){
				double val, sigma = 0;
				for(std::size_t ii = 0; ii < probs.size(); ++ii){
					if ((incomplete_cases.count(ii) == 0) && (invalid_count.count(ii) == 0)) {
						val = 0.0;
						for( std::size_t kk = 0; kk < probs[ii].size(); ++kk ) {
							val += x * probs[ii][kk];
						}
						val -= mu;
						sigma += val * val;
					}
				}
				sigma = std::sqrt(sigma/(valid_count - 1.0));
				std::cout << "d1: " << d1 << std::endl;
				std::cout << "valid_count: " << valid_count << std::endl;
				std::cout << "mean: " << mu << std::endl;
				std::cout << "sigma: " << sigma << std::endl << std::endl;
				if(sigma < 1e-9){
					n_constant_variance++;
					continue;
				}
			}

			// filters passed; write contextual info
			maf.push_back(theta);
			info.push_back(info_j);
			rsid.push_back(rsid_j);
			chromosome.push_back(chr_j);
			position.push_back(pos_j);
			alleles.push_back(alleles_j);
			G.al_0.push_back(alleles_j[0]);
			G.al_1.push_back(alleles_j[1]);
			G.rsid.push_back(rsid_j);
			G.chromosome.push_back(std::stoi(chr_j));
			G.position.push_back(pos_j);
			std::string key_j = chr_j + "~" + std::to_string(pos_j) + "~" + alleles_j[0] + "~" + alleles_j[1];
			G.SNPKEY.push_back(key_j);
			
			// filters passed; write dosage to G
			// Note that we only write dosage for valid sample ids
			std::size_t ii_obs = 0;
			missing_genos.clear();
			for( std::size_t ii = 0; ii < probs.size(); ++ii ) {
				if (incomplete_cases.count(ii) == 0) {
					dosage = 0.0;

					for( std::size_t kk = 0; kk < probs[ii].size(); ++kk ) {
						x = probs[ii][kk];
						dosage += x * kk;
					}

					if(params.geno_check){
						if(invalid_count.count(ii) == 0){
							G.assign_index(ii_obs, jj, dosage);
						} else {
							missing_genos[ii_obs] = 1;
							G.assign_index(ii_obs, jj, std::nan(""));
						}
					} else {
						G.assign_index(ii_obs, jj, dosage);
					}

					ii_obs++; // loop should end at ii_obs == n_samples
				}
			}
			// G.compressed_dosage_sds[jj] = sigma;
			// G.compressed_dosage_means[jj] = d1;

			if (ii_obs < n_samples) {
				throw std::logic_error("ERROR: Fewer non-missing genotypes than expected");
			}

			// Log number of missing entries
			if(missing_genos.size() > 0){
				n_var_incomplete += 1;
				missing_calls += (double) missing_genos.size();
			}
			jj++;
		}

		// need to resize G whilst retaining existing coefficients if while
		// loop exits early due to EOF.
		G.conservativeResize(n_samples, jj);
		assert( rsid.size() == jj );
		assert( chromosome.size() == jj );
		assert( position.size() == jj );
		assert( alleles.size() == jj );
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
		fg.push(boost_io::file_source(params.incl_rsids_file.c_str()));
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
		fg.push(boost_io::file_source(params.incl_sids_file.c_str()));
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
				while(s.compare(bgen_ids[bb]) != 0) {
					incomplete_cases[bb] = 1;
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
				incomplete_cases[jj] = 1;
			}
		} catch (const std::exception &exc) {
			// throw std::runtime_error("ERROR: problem converting incl_sample_ids.");
			throw;
		}
		// n_samples = ii;
		std::cout << "Subsetted down to " << ii << " ids from --incl_sample_ids";
		std::cout << std::endl;
	}

	void read_grid_file( std::string filename,
						 Eigen::MatrixXd& M,
						 std::vector< std::string >& col_names){
		// Used in mode_vb only.
		// Slightly different from read_txt_file in that I don't know
		// how many rows there will be and we can assume no missing values.

		boost_io::filtering_istream fg;
		fg.push(boost_io::file_source(filename.c_str()));
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
		fg.push(boost_io::file_source(filename.c_str()));

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

	void read_vb_init_file(std::string filename,
                           Eigen::MatrixXd& M,
                           std::vector< std::string >& col_names,
                        //    std::vector< int >& init_chr,
                        //    std::vector< uint32_t >& init_pos,
                        //    std::vector< std::string >& init_a0,
                           std::vector< std::string >& init_key){
		// Used in mode_vb only.
		// Need custom function to deal with variable input. Sometimes
		// we have string columns with rsid / a_0 etc
		// init_chr, init_pos, init_a0, init_a1;

		boost_io::filtering_istream fg;
		fg.push(boost_io::file_source(filename.c_str()));
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
		fg.push(boost_io::file_source(filename.c_str()));

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
				if(n_cols == 2){
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
				} else if(n_cols == 7){
					key_i = "";
					for (int k = 0; k < n_cols; k++) {
						std::string s;
						ss >> s;
						if(k == 0){
							key_i += s + "~";
						} else if (k == 1){
							key_i += s + "~";
							// init_pos.push_back(std::stoull(s));
						} else if (k == 3){
							key_i += s + "~";
							// init_a0.push_back(s);
						} else if (k == 4){
							key_i += s;
							init_key.push_back(key_i);
							// init_a1.push_back(s);
						} else if(k >= 5){
							M(i, k) = stod(s);
						}
					}
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

	void read_txt_file( std::string filename,
						Eigen::MatrixXd& M,
						int& n_cols,
						std::vector< std::string >& col_names,
						std::map< int, bool >& incomplete_row ){
		// pass top line of txt file filename to col_names, and body to M.
		// TODO: Implement how to deal with missing values.

		boost_io::filtering_istream fg;
		fg.push(boost_io::file_source(filename.c_str()));
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
					std::string s;
					ss >> s;
					/// NA
					if (s == "NA" || s == "NAN" || s == "NaN" || s == "nan") {
						tmp_d = params.missing_code;
					} else {
						try{
							tmp_d = stod(s);
						} catch (const std::invalid_argument &exc){
							std::cout << s << " on line " << i << std::endl;
							throw;
						}
					}

					if(tmp_d != params.missing_code) {
						M(i, k) = tmp_d;
					} else {
						M(i, k) = params.missing_code;
						incomplete_row[i] = 1;
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

	void center_matrix( Eigen::MatrixXd& M,
						int& n_cols ){
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

	void scale_matrix( Eigen::MatrixXd& M,
						int& n_cols,
 						std::vector< std::string >& col_names){
		// Scale eigen matrix passed by reference.
		// Removes columns with zero variance + updates col_names.
		// Only call on matrixes which have been reduced to complete cases,
		// as no check for incomplete rows.

		std::vector<size_t> keep;
		std::vector<std::string> keep_names;
		std::vector<std::string> reject_names;
		for (int k = 0; k < n_cols; k++) {
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
			M = getCols(M, keep);
			
			n_cols = keep.size();
			col_names = keep_names;
		}

		if (n_cols == 0) {
			throw std::runtime_error("ERROR: No columns left with nonzero variance after scale_matrix()");
		}
	}

	void scale_matrix( Eigen::MatrixXd& M,
						int& n_cols){
		// Scale eigen matrix passed by reference.
		// Removes columns with zero variance.
		// Only call on matrixes which have been reduced to complete cases,
		// as no check for incomplete rows.

		std::vector<size_t> keep;
		for (int k = 0; k < n_cols; k++) {
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
			}
		}

		if (keep.size() != n_cols) {
			std::cout << " Removing " << (n_cols - keep.size())  << " columns with zero variance." << std::endl;
			M = getCols(M, keep);
			
			n_cols = keep.size();
		}

		if (n_cols == 0) {
			throw std::runtime_error("ERROR: No columns left with nonzero variance after scale_matrix()");
		}
	}

	void scale_matrix_conserved( Eigen::MatrixXd& M,
						int& n_cols){
		// Scale eigen matrix passed by reference.
		// Does not remove columns with zero variance.
		// Only call on matrixes which have been reduced to complete cases,
		// as no check for incomplete rows.

		for (int k = 0; k < n_cols; k++) {
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
			}
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

	void read_grids(){
		// For use in vbayes object

		std::vector< std::string > true_fixed_names = {"sigma", "sigma_b", "lambda_b"};
		std::vector< std::string > true_gxage_names = {"sigma", "sigma_b", "sigma_g", "lambda_b", "lambda_g"};

		if ( params.hyps_grid_file == "NULL" ) {
			throw std::invalid_argument( "Must provide hyperparameter grid file." );
		}
		if ( params.hyps_probs_file == "NULL" ) {
			throw std::invalid_argument( "Must provide hyperparameter probabilities file." );
		}


		read_grid_file( params.hyps_grid_file, hyps_grid, hyps_names );
		read_grid_file( params.hyps_probs_file, imprt_grid, imprt_names );

		assert(imprt_grid.cols() == 1);
		assert(hyps_grid.rows() == imprt_grid.rows());

		// Verify header of grid file as expected
		if(params.interaction_analysis){
			if(!std::includes(hyps_names.begin(), hyps_names.end(), true_gxage_names.begin(), true_gxage_names.end())){
				throw std::runtime_error("Column names of --hyps_grid must be sigma sigma_b sigma_g lambda_b lambda_g");
			}
		} else {
			if(hyps_names != true_fixed_names){
				throw std::runtime_error("Column names of --hyps_grid must be sigma sigma_b lambda");
			}
		}

		// Option to provide separate grid to evaluate in round 1
		std::vector< std::string > r1_hyps_names, r1_probs_names;
		if ( params.r1_hyps_grid_file != "NULL" ) {
			read_grid_file( params.r1_hyps_grid_file, r1_hyps_grid, r1_hyps_names );
			if(hyps_names != r1_hyps_names){
				throw std::invalid_argument( "Header of --r1_hyps_grid must match --hyps_grid." );
			}
			read_grid_file( params.r1_probs_grid_file, r1_probs_grid, r1_probs_names );
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
		std::vector<std::string> cols_check2 = {"chr", "pos", "rsid", "a0", "a1", "beta", "gamma"};

		if ( params.vb_init_file != "NULL" ) {
			std::cout << "Reading initialisation for alpha from file" << std::endl;
			read_vb_init_file(params.vb_init_file, vb_init_mat, vb_init_colnames,
                              init_key);

			assert(vb_init_mat.cols() == 2 || vb_init_mat.cols() == 7);
			assert(vb_init_colnames == cols_check1 || vb_init_colnames == cols_check2);
			if(vb_init_mat.cols() == 2){
				alpha_init = Eigen::Map<Eigen::VectorXd>(vb_init_mat.col(0).data(), vb_init_mat.rows());
				mu_init = Eigen::Map<Eigen::VectorXd>(vb_init_mat.col(1).data(), vb_init_mat.rows());
			} else {
				std::cout << "--vb_init file with contextual information detected" << std::endl;
				std::cout << "Warning: This will be O(PL) where L = " << vb_init_mat.rows();
				std::cout << " is the number of lines in file given to --vb_init." << std::endl;
				alpha_init = Eigen::VectorXd::Zero(2*n_var);
				mu_init    = Eigen::VectorXd::Zero(2*n_var);

				std::vector<std::string>::iterator it;
				std::uint32_t index_kk;
				for(int kk = 0; kk < vb_init_mat.rows(); kk++){
					it = std::find(G.SNPKEY.begin(), G.SNPKEY.end(), init_key[kk]);
					if (it == G.SNPKEY.end()){
						std::cout << "WARNING: Can't locate variant with key: ";
						std::cout << init_key[kk] << std::endl;
					} else {
						index_kk = it - G.SNPKEY.begin();

						alpha_init[index_kk]         = 1.0;
						alpha_init[index_kk + n_var] = 1.0;
						mu_init[index_kk]            = vb_init_mat(kk, 0);
						mu_init[index_kk + n_var]    = vb_init_mat(kk, 1);
					}
				}
			}

		} else {
			throw std::invalid_argument( "Tried to read NULL --vb_init file." );
		}

	}

	void reduce_mat_to_complete_cases( Eigen::Ref<Eigen::MatrixXd> M, 
								   bool& matrix_reduced,
								   int n_cols,
								   std::map< int, bool > incomplete_cases ) {
		// Remove rows contained in incomplete_cases
		Eigen::MatrixXd M_tmp;
		if (matrix_reduced) {
			throw std::runtime_error("ERROR: Trying to remove incomplete cases twice...");
		}

		// Create temporary matrix of complete cases
		int n_incomplete = incomplete_cases.size();
		M_tmp.resize(n_samples - n_incomplete, n_cols);

		// Fill M_tmp with non-missing entries of M
		int ii_tmp = 0;
		for (int ii = 0; ii < n_samples; ii++) {
			if (incomplete_cases.count(ii) == 0) {
				for (int kk = 0; kk < n_cols; kk++) {
					M_tmp(ii_tmp, kk) = M(ii, kk);
				}
				ii_tmp++;
			} else {
				// std::cout << "Deleting row " << ii << std::endl;
			}
		}

		// Assign new values to reference variables
		M = M_tmp;
		matrix_reduced = true;
	}

	void regress_out_covars() {
		std::cout << "Regressing out covars:" << std::endl;
		for(int cc = 0; cc < n_covar; cc++){
			std::cout << ( cc > 0 ? ", " : "" ) << covar_names[cc]; 
		}
		std::cout << std::endl;

		Eigen::MatrixXd ww = W.rowwise() - W.colwise().mean(); //not needed probably
		Eigen::MatrixXd bb = solve(ww.transpose() * ww, ww.transpose() * Y);
		Y = Y - ww * bb;
	}

	void reduce_to_complete_cases() {
		// Remove any samples with incomplete covariates or phenotypes from
		// Y and W.
		// Note; other functions (eg. read_incl_sids) may add to incomplete_cases
		// Note; during unit testing sometimes only phenos or covars present.

		incomplete_cases.insert(missing_covars.begin(), missing_covars.end());
		incomplete_cases.insert(missing_phenos.begin(), missing_phenos.end());
		if(params.pheno_file != "NULL"){
			reduce_mat_to_complete_cases( Y, Y_reduced, n_pheno, incomplete_cases );
		}
		if(params.covar_file != "NULL"){
			reduce_mat_to_complete_cases( W, W_reduced, n_covar, incomplete_cases );
		}
		n_samples -= incomplete_cases.size();
		missing_phenos.clear();
		missing_covars.clear();

		std::cout << "Reduced to " << n_samples << " samples with complete data";
 		std::cout << " across covariates and phenotype." << std::endl;
	}
};


inline size_t numRows(const Eigen::MatrixXd &A) {
	return A.rows();
}

inline size_t numCols(const Eigen::MatrixXd &A) {
	return A.cols();
}

inline void setCol(Eigen::MatrixXd &A, const Eigen::VectorXd &v, size_t col) {
	assert(numRows(v) == numRows(A));
	A.col(col) = v;
}

inline Eigen::VectorXd getCol(const Eigen::MatrixXd &A, size_t col) {
	return A.col(col);
}

inline void setCols(Eigen::MatrixXd &X, const std::vector<size_t> &cols, const Eigen::MatrixXd &values) {
	assert(cols.size() == numCols(values));
	assert(numRows(X) == numRows(values));

	if (cols.size() == 0) {
		return;
	}

	for (size_t i = 0; i < cols.size(); i++) {
		setCol(X, getCol(values, i), cols[i]);
	}
}

inline Eigen::MatrixXd getCols(const Eigen::MatrixXd &X, const std::vector<size_t> &cols) {
	Eigen::MatrixXd result(numRows(X), cols.size());
	assert(cols.size() == numCols(result));
	assert(numRows(X) == numRows(result));

	if (cols.size() == 0) {
		return result;
	}

	for (size_t i = 0; i < cols.size(); i++) {
		setCol(result, getCol(X, cols[i]), i);
	}

	return result;
}

inline Eigen::MatrixXd solve(const Eigen::MatrixXd &A, const Eigen::MatrixXd &b) {
	Eigen::MatrixXd x = A.colPivHouseholderQr().solve(b);
	if (fabs((double)((A * x - b).norm()/b.norm())) > 1e-8) {
		std::cout << "ERROR: could not solve covariate scatter matrix." << std::endl;
		std::exit(EXIT_FAILURE);
	}
	return x;
}

#endif
