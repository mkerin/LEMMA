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


class data 
{
	public :
	parameters params;

	std::vector< std::string > chromosome, rsid, SNPID;
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

	Eigen::MatrixXd G; // probabilistic genotype matrix
	Eigen::MatrixXi GG; // rounded genotype matrix
	Eigen::MatrixXd Y; // phenotype matrix
	Eigen::MatrixXd W; // covariate matrix
	Eigen::VectorXd Z; // interaction vector
	genfile::bgen::View::UniquePtr bgenView;
	std::vector< double > beta, tau, neglogP, neglogP_2dof;
	std::vector< std::vector< double > > gamma;

	boost_io::filtering_ostream outf;

	std::chrono::system_clock::time_point start;

	// grid things for vbayes
	std::vector< std::string > hyps_names, imprt_names;
	Eigen::MatrixXd r1_hyps_grid, hyps_grid, imprt_grid;
	Eigen::VectorXd alpha_init, mu_init;

	
	// constructors/destructors
	// data() : bgenView( "NULL" ) {
	// 	bgen_pass = false; // No bgen file set; read_bgen_chunk won't run.
	// }

	data( std::string filename ) {
		// system time at start
		start = std::chrono::system_clock::now();
		std::time_t start_time = std::chrono::system_clock::to_time_t(start);
		std::cout << "Starting analysis at " << std::ctime(&start_time) << std::endl;
		std::cout << "Compiled from git branch: master" << std::endl;

		bgenView = genfile::bgen::View::create(filename);
		bgen_pass = true;
		n_samples = bgenView->number_of_samples();
		n_var_parsed = 0;
	}
	
	~data() {
		// system time at end
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end-start;
		std::time_t end_time = std::chrono::system_clock::to_time_t(end);
		std::cout << "Analysis finished at " << std::ctime(&end_time);
		std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;
	}

	void output_init() {
		// open output file
		std::string ofile, gz_str = ".gz";

		ofile = params.out_file;
		if (params.out_file.find(gz_str) != std::string::npos) {
			outf.push(boost_io::gzip_compressor());
		}
		outf.push(boost_io::file_sink(ofile.c_str()));

		if(params.mode_vcf){
			// Output header for vcf file
			outf << "##fileformat=VCFv4.2\n"
				<< "FORMAT=<ID=GP,Type=Float,Number=G,Description=\"Genotype call probabilities\">\n"
				<< "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT" ;
			bgenView->get_sample_ids(
				[&]( std::string const& id ) { outf << "\t" << id ; }
			) ;
			outf << "\n" ;
		}

		if(params.mode_lm){
			// Output header for vcf file
			outf << "chr\trsid\tpos\ta_0\ta_1\taf\tinfo\tbeta\ttau";
			outf << "\tneglogP_1dof\tgamma1\tgamma2\tgamma3\tneglogP_2dof" << std::endl;
		}

		if(params.mode_lm2){
			outf << "chr\trsid\tpos\ta_0\ta_1\taf\tinfo";
			for (int kk = 0; kk < n_covar; kk++){
				outf << "\tbeta_" << covar_names[kk];
			}
			outf << "\tbeta_var_j\tbeta_gx" << params.x_param_name;
			for (int kk = 0; kk < params.n_gconf; kk++){
				outf << "\tbeta_gx" << params.gconf[kk];
			}
			outf << "\tneglogP_gx" << params.x_param_name << std::endl;
		}

		if(params.mode_joint_model){
			outf << "chr\trsid\tpos\ta_0\ta_1\taf\tinfo\ttau\t1dof_neglogP";
 			outf << std::endl;
		}
	}

	void output_results() {
		if(params.mode_lm){
			for (int s = 0; s < n_var; s++){
				outf << chromosome[s] << "\t" << rsid[s] << "\t" << position[s] << "\t";
				outf << alleles[s][0] << "\t" << alleles[s][1] << "\t" << maf[s] << "\t";
				outf << info[s] << "\t" << beta[s] << "\t" << tau[s] << "\t";
	 			outf << neglogP[s] << "\t" << gamma[s][0] << "\t" << gamma[s][1];
	 			outf << "\t" << gamma[s][2] << "\t" << neglogP_2dof[s] << std::endl;
			}
		}

		if(params.mode_joint_model){
			for (int s = 0; s < n_var; s++){
				outf << chromosome[s] << "\t" << rsid[s] << "\t" << position[s] << "\t";
				outf << alleles[s][0] << "\t" << alleles[s][1] << "\t" << maf[s] << "\t";
				outf << info[s] << "\t" << tau[s] << "\t" << neglogP[s] << std::endl;
			}
		}
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
		Eigen::VectorXd variant_j(n_samples);
		std::string chr_j ;
		uint32_t pos_j ;
		std::string rsid_j ;
		std::vector< std::string > alleles_j ;
		std::string SNPID ; // read but ignored
		std::vector< std::vector< double > > probs ;
		ProbSetter setter( &probs );
		std::map<int, bool> missing_genos;

		double d1, theta, x, dosage, check, info_j, f1, f2, chunk_missingness;
		double missing_calls = 0.0;
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

		std::size_t valid_count, jj = 0;
		while ( jj < params.chunk_size && bgen_pass ) {
			bgen_pass = bgenView->read_variant( &SNPID, &rsid_j, &chr_j, &pos_j, &alleles_j );
			n_var_parsed++;
			if (!bgen_pass) break;
			assert( alleles_j.size() > 0 );

			// // range filter
			// if (params.range && (pos_j < params.start || pos_j > params.end)){
			// 	bgenView->ignore_genotype_data_block();
			// 	continue;
			// }

			// Read probs + check maf filter
			bgenView->read_genotype_data_block( setter );

			// maf + info filters; computed on valid sample_ids & variants whose alleles
			// sum to 1
			d1 = f2 = 0.0;
			valid_count = 0;
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
						d1 += dosage;
						f2 += (f1 - dosage * dosage);
						valid_count++;
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

			// filters passed; write contextual info
			maf.push_back(theta);
			info.push_back(info_j);
			rsid.push_back(rsid_j);
			chromosome.push_back(chr_j);
			position.push_back(pos_j);
			alleles.push_back(alleles_j);
			
			// filters passed; write dosage to variant_j
			// Track mu + sd on first pass, then standardise before writing to G
			// Note that we only write dosage for valid sample ids
			std::size_t ii_obs = 0;
			double mu = 0.0;
			double count = 0;
			missing_genos.clear();
			for( std::size_t ii = 0; ii < probs.size(); ++ii ) {
				if (incomplete_cases.count(ii) == 0) {
					dosage = 0.0;
					check = 0.0;

					// Check for overflow
					if(ii_obs >= n_samples){
						throw std::logic_error("INTERNAL ERROR: More genetic samples than expected");
					}

					for( std::size_t kk = 0; kk < probs[ii].size(); ++kk ) {
						x = probs[ii][kk];
						dosage += x * kk;
						check += x;
					}

					if(params.geno_check){
						if(check > 0.9999 && check < 1.0001){
							variant_j[ii_obs] = dosage;
							mu += dosage;
							count += 1;
						} else if(check > 0){
							std::cout << "Unexpected sum of allele probs: ";
		 					std::cout << check << " at sample=" << ii;
		 					std::cout << ", variant=" << jj << std::endl;
							throw std::logic_error("Allele probs expected to sum to 1 or 0");
						} else {
							missing_genos[ii_obs] = 1;
						}
					} else {
						variant_j[ii_obs] = dosage;
						mu += dosage;
						count += 1;
					}

					ii_obs++; // loop should end at ii_obs == n_samples
				}
			}

			if (ii_obs < n_samples) {
				throw std::logic_error("ERROR: Fewer non-missing genotypes than expected");
			}

			// Center entries and set missing to zero
			std::cout << "Starting to standardise vector" << std::endl;
			if(missing_genos.size() > 0){
				n_var_incomplete += 1;
				missing_calls += (double) missing_genos.size();
			}

			mu = mu / count;
			for (std::size_t ii = 0; ii < n_samples; ii++) {
				if (missing_genos.count(ii) == 0) {
					variant_j[ii] -= mu;
				} else {
					variant_j[ii] = 0.0;
				}
			}

			double sigma = variant_j.squaredNorm();
			sigma = std::sqrt(sigma/(count - 1.0));

			if (sigma > 1e-12){
				for (std::size_t ii = 0; ii < n_samples; ii++){
					G(ii, jj) = variant_j[ii] / sigma;
				}
			} else {
				for (std::size_t ii = 0; ii < n_samples; ii++){
					G(ii, jj) = variant_j[ii];
				}	
			}
			std::cout << variant_j << std::endl;

			// // Set missing entries to mean
			// // Could mean center here, but still want to write to VCF.
			// if(missing_genos.size() > 0){
			// 	n_var_incomplete += 1;
			// 	missing_calls += (double) missing_genos.size();
			// 	mu = mu / count;
			// 	for (int ii = 0; ii < n_samples; ii++) {
			// 		if (missing_genos.count(ii) != 0) {
			// 			G(ii, jj) = mu;
			// 		}
			// 	}
			// }
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

		// keep integer copy of genotypes for 2 dof test
		if(params.test_2dof){
			Eigen::MatrixXd tmp;
			tmp = (G.array() + 0.5).matrix();
			GG = tmp.cast <int> ();
		}

		chunk_missingness = missing_calls / (double) (n_var * n_samples);
		if(chunk_missingness > 0.0){
			std::cout << "Chunk missingness " << chunk_missingness << "(";
 			std::cout << n_var_incomplete << "/" << n_var;
			std::cout << " variants incomplete)" << std::endl;
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
				incomplete_cases[bb] = 1;
			}
		} catch (const std::exception &exc) {
			// throw std::runtime_error("ERROR: problem converting incl_sample_ids.");
			throw;
		}
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

		std::vector<size_t> keep;
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
		std::vector< std::string > r1_hyps_names;
		if ( params.r1_hyps_grid_file != "NULL" ) {
			read_grid_file( params.r1_hyps_grid_file, r1_hyps_grid, r1_hyps_names );
			if(hyps_names != r1_hyps_names){
				throw std::invalid_argument( "Header of --r1_hyps_grid must match --hyps_grid." );
			}
		}
	}

	void read_alpha_mu(){
		// For use in vbayes object
		Eigen::MatrixXd vb_init_mat;

		std::vector< std::string > vb_init_colnames;
		std::vector< std::string > cols_check = {"alpha", "mu"};

		if ( params.vb_init_file != "NULL" ) {
			std::cout << "Reading initialisation for alpha from file" << std::endl;
			read_grid_file( params.vb_init_file, vb_init_mat, vb_init_colnames );

			assert(vb_init_mat.cols() == 2);
			assert(vb_init_mat.rows() == n_var || vb_init_mat.rows() == 2*n_var);
			assert(vb_init_colnames == cols_check);

			alpha_init = Eigen::Map<Eigen::VectorXd>(vb_init_mat.col(0).data(), vb_init_mat.rows());
			mu_init = Eigen::Map<Eigen::VectorXd>(vb_init_mat.col(1).data(), vb_init_mat.rows());
		} else {
			throw std::invalid_argument( "Tried to read NULL --vb_init file." );
		}

	}

	void reduce_mat_to_complete_cases( Eigen::MatrixXd& M, 
								   bool& matrix_reduced,
								   int n_cols,
								   std::map< int, bool > incomplete_cases ) {
		// Remove rows contained in incomplete_cases
		int n_incomplete;
		Eigen::MatrixXd M_tmp;
		if (matrix_reduced) {
			throw std::runtime_error("ERROR: Trying to remove incomplete cases twice...");
		}

		// Create temporary matrix of complete cases
		n_incomplete = incomplete_cases.size();
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

	void regress_covars() {
		std::cout << "Regressing out covars:" << std::endl;
		for(int cc = 0; cc < n_covar; cc++){
			std::cout << ( cc > 0 ? ", " : "" ) << covar_names[cc]; 
		}
		std::cout << std::endl;

		Eigen::MatrixXd ww = W.rowwise() - W.colwise().mean(); //not needed probably
		Eigen::MatrixXd bb = solve(ww.transpose() * ww, ww.transpose() * Y);
		Y = Y - ww * bb;
	}

	std::size_t find_covar_index( std::string colname ){
		std::size_t x_col;
		std::vector<std::string>::iterator it;
		it = std::find(covar_names.begin(), covar_names.end(), colname);
		if (it == covar_names.end()){
			throw std::invalid_argument("Can't locate parameter " + colname);
		}
		x_col = it - covar_names.begin();
		return x_col;
	}

	void calc_lrts() {
		// For-loop through variants and compute interaction models.
		// Save to
		// data.tau, data.beta
		// Y is a matrix of dimension n_samples x 1
		Eigen::VectorXd e_j, f_j, g_j, gamma_j;
		double beta_j, tau_j, xtx_inv, loglik_null, loglik_alt, chi_stat;
		long double pval;

		// Determine which covar to use in interaction
		std::ptrdiff_t x_col;
		if( params.x_param_name != "NULL"){
			x_col = find_covar_index(params.x_param_name);
		} else {
			x_col = 0;
		}

		Eigen::VectorXd vv(Eigen::Map<Eigen::VectorXd>(W.col(x_col).data(), n_samples));
		Eigen::MatrixXd Z = G.array().colwise() * vv.array();

		beta.clear();
		tau.clear();
		gamma.clear();
		neglogP.clear();
		neglogP_2dof.clear();
		xtx_inv = 1.0 / (n_samples - 1.0);

		for (int jj = 0; jj < n_var; jj++){
			Eigen::Map<Eigen::VectorXd> G_j(G.col(jj).data(), n_samples);
			Eigen::Map<Eigen::VectorXd> Z_j(Z.col(jj).data(), n_samples);

			// null
			beta_j = xtx_inv * (G_j.transpose() * Y)(0,0);
			e_j = Y - G_j * beta_j;

			// alt - 1dof
			tau_j = (Z_j.transpose() * e_j)(0,0) / (Z_j.transpose() * Z_j)(0,0);
			f_j = e_j - Z_j * tau_j;

			// Saving variables
			beta.push_back(beta_j);
			tau.push_back(tau_j);
			neglogP.push_back(lrt(e_j, f_j, 1));

			// 2 dof stuff
			Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_samples, 3);
			Eigen::MatrixXd AA = Eigen::MatrixXd::Zero(n_samples, 2);
			Eigen::MatrixXd D;
			std::vector<double> nn(3, 0);
			std::vector< double > gamma_vec(3, std::nan(""));
			int kk;
			for (int ii = 0; ii < n_samples; ii++){
				kk = GG(ii,jj);
				nn[kk] += 1.0;
			}

			if(std::all_of(nn.begin(), nn.end(), [](int i){return i>0.0;})){
				for (int ii = 0; ii < n_samples; ii++){
					kk = GG(ii,jj);
					// A(ii, kk) = vv(ii);
					if(kk == 0){
						AA(ii, 0) -= nn[1] * vv(ii) / nn[0];
						AA(ii, 1) -= nn[2] * vv(ii) / nn[0];
					} else {
						AA(ii, kk-1) = vv(ii);
					}
				}

				D = (AA.transpose() * AA);
				gamma_j = D.ldlt().solve(AA.transpose() * e_j);
				g_j = e_j - AA * gamma_j;

				gamma_vec[1] = gamma_j(0, 0);
				gamma_vec[2] = gamma_j(1, 0);
				gamma_vec[0] = -(nn[1]*gamma_vec[1] + nn[2]*gamma_vec[2]) / nn[0];
				gamma.push_back(gamma_vec);
				neglogP_2dof.push_back(lrt(e_j, g_j, 2));
			} else {
				gamma.push_back(gamma_vec);
				neglogP_2dof.push_back(std::nan(""));
			}
		}
	}

	void calc_lrts2() {
		// Determine column indexs to include in X_g
		std::vector< std::size_t > col_indexes;
		if( params.x_param_name != "NULL"){
			col_indexes.push_back(find_covar_index(params.x_param_name));
		} else {
			col_indexes.push_back(0);
		}
		for (int ii = 0; ii < params.n_gconf; ii++){
			col_indexes.push_back(find_covar_index(params.gconf[ii]));
		}

		for (int jj = 0; jj < n_var; jj++){
			// Build matrix X_g
			Eigen::MatrixXd tmp;
			tmp = (getCols(W, col_indexes)).array().colwise() * G.col(jj).array();
			int tmp_ncol = params.n_gconf + 1;
			center_matrix( tmp, tmp_ncol );
			scale_matrix( tmp, tmp_ncol );

			// Combine to matrix X_comb = (X_c, X_g)
			int comb_col = n_covar + 2 + params.n_gconf;
			Eigen::MatrixXd X_comb(n_samples, comb_col);
			X_comb << W, G.col(jj), tmp;

			// Fit joint regression Y = X_comb beta_comb + epsilon
			Eigen::MatrixXd XtX_comb = X_comb.transpose() * X_comb;
			Eigen::MatrixXd XtX_comb_inv = XtX_comb.inverse();
			Eigen::MatrixXd beta_comb_j = XtX_comb_inv * X_comb.transpose() * Y;

			// Create vector of standard errors
			double s_var = (Y - X_comb * beta_comb_j).norm() / std::sqrt(n_samples - comb_col);
			std::vector< double > se;
			for (int kk = 0; kk < comb_col; kk++){
				se.push_back(s_var * std::sqrt(XtX_comb_inv(kk, kk)));
			}

			// Compute t-test on GxE param
			double n_dof;
			n_dof = (double) (n_samples - (n_covar + 2 + params.n_gconf));
			boost::math::students_t t_dist(n_dof);
			double t_stat = beta_comb_j(n_covar + 1, 0) / se[n_covar + 1];

			// p-value from 2 tailed t-test
			double q = 2 * boost::math::cdf(boost::math::complement(t_dist, fabs(t_stat)));

 			// Output stats
			outf << chromosome[jj] << "\t" << rsid[jj] << "\t" << position[jj];
			outf << "\t" << alleles[jj][0] << "\t" << alleles[jj][1] << "\t";
			outf << maf[jj] << "\t" << info[jj];

			for (int kk = 0; kk < n_covar + 2 + params.n_gconf; kk++){
				outf << "\t" << beta_comb_j(kk, 0);
			}

			outf << "\t" << -1 * std::log10(q) << std::endl;
		}
	}

	void calc_joint_model() {
		// Y = G beta
		// vs
		// Y = G beta + Z tau
		// Want:
		// - coefficients
		// - variance explained
		// - F test comparing model fit (just need residuals)
		// - AIC?
		// Common sense checks; n < p
		if (n_var > n_samples){
			throw std::logic_error("n_samples must be less than n_var for the joint model");
		}

		// Determine which covar to use in interaction
		std::ptrdiff_t x_col;
		if( params.x_param_name != "NULL"){
			std::vector<std::string>::iterator it;
			it = std::find(covar_names.begin(), covar_names.end(), params.x_param_name);
			if (it == covar_names.end()){
				throw std::invalid_argument("Can't locate --interaction parameter");
			}
			x_col = it - covar_names.begin();
		} else {
			x_col = 0;
		}

		Eigen::VectorXd vv(Eigen::Map<Eigen::VectorXd>(W.col(x_col).data(), n_samples));
		std::cout << "Initialising matrix W (" << G.rows() << "," << 2*G.cols() << ")" << std::endl;
		Eigen::MatrixXd W(G.rows(), G.cols() + G.cols());
		W << G, (G.array().colwise() * vv.array()).matrix();
		std::cout << W << std::endl;
		// G should be centered and scaled by now
		std::cout << "Fitting polygenic model" << std::endl;
		Eigen::MatrixXd beta = solve(G.transpose() * G, G.transpose() * Y);
		Eigen::VectorXd e_null = Y - G * beta;

		// eta; vector of coefficients c(beta, tau)
		std::cout << "Fitting joint interaction model" << std::endl;
		Eigen::MatrixXd eta = solve(W.transpose() * W, W.transpose() * Y);
		Eigen::VectorXd e_alt = Y - W * eta;
		std::cout << eta << std::endl;
		std::cout << e_alt << std::endl;

		boost::math::students_t t_dist(n_samples - n_var - 1);
		double pval_j, eta_j;
		for(int jj = 0; jj < n_var; jj++){
			eta_j = eta(n_var + jj, 0);
			tau.push_back(eta_j);
			pval_j = 1.0 - boost::math::cdf(t_dist, eta_j);
			neglogP.push_back(-1*std::log10(pval_j));
		}

		// F test
		double f_stat, rss_null, rss_alt, pval, neglogp_joint;
		rss_null = e_null.dot(e_null);
		rss_alt = e_alt.dot(e_alt);
		f_stat = (rss_null - rss_alt) / (double) n_var;
		f_stat /= rss_alt / (double) (n_samples - n_var - 1);
		boost::math::fisher_f f_dist(n_var, n_samples - n_var);
		pval = 1.0 - boost::math::cdf(f_dist, f_stat);
		neglogp_joint = -1 * std::log10(pval);

		// Output
		std::cout << "F-test comparing models " << std::endl;
		std::cout << "H0: Y = G x beta" << std::endl;
		std::cout << "vs" << std::endl;
		std::cout << "H0: Y = G x beta + Z x tau" << std::endl;
		std::cout << "F-stat = " << f_stat << ", neglogP = " << neglogp_joint << std::endl;

		std::cout << "Variance explained by null: ";
		std::cout << 100 * (1.0 - rss_null / (Y.transpose() * Y)(0,0)) << std::endl;
		std::cout << "Variance explained by alt: ";
		std::cout << 100 * (1.0 - rss_alt / (Y.transpose() * Y)(0,0)) << std::endl;
	}

	double lrt(Eigen::VectorXd null, Eigen::VectorXd alt, int df){
		// Logliks correct up to ignoreable constant
		boost::math::chi_squared chi_dist_1(1), chi_dist_2(2);
		double loglik_null, loglik_alt, chi_stat, neglogp;

		loglik_null = std::log(n_samples) - std::log(null.dot(null));
		loglik_null *= n_samples/2.0;
		loglik_alt = std::log(n_samples) - std::log(alt.dot(alt));
		loglik_alt *= n_samples/2.0;

		chi_stat = 2*(loglik_alt - loglik_null);
		if(chi_stat < 0 && chi_stat > -0.0000001){
			chi_stat = 0.0; // Sometimes we get -1e-09
		}
		if (df == 1){
			neglogp = std::log10(boost::math::cdf(boost::math::complement(chi_dist_1, chi_stat)));
		} else {
			neglogp = std::log10(boost::math::cdf(boost::math::complement(chi_dist_2, chi_stat)));
		}
		neglogp *= -1.0;
		return neglogp;
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

	void run() {
		int ch = 0;

		// Step 1; Read in raw covariates and phenotypes
		// - also makes a record of missing values
		read_covar();
		read_pheno();

		// Step 2; Reduce raw covariates and phenotypes to complete cases
		// - may change value of n_samples
		// - will also skip these cases when reading bgen later
		reduce_to_complete_cases();

		// Step 3; Center phenos, genotypes, normalise covars
		center_matrix( Y, n_pheno );
		center_matrix( W, n_covar );
		scale_matrix( W, n_covar, covar_names );

		// Step 4; Regress covars out of phenos
		if(!params.mode_lm2){
			std::cout << " Regressing out covariates" << std::endl;
			regress_covars();
		}

		// TODO: Move UI messages to coherent place?
		if( params.x_param_name != "NULL"){
			std::cout << "Searching for --interaction param ";
			std::cout << params.x_param_name << std::endl;
		} else {
			std::cout << "Choosing first covar to use as interaction term (default)" << std::endl;
		}

		// Write headers to output file
		output_init();

		while (read_bgen_chunk()) {
			// Raw dosage read in to G
			std::cout << "Chunk " << ch+1 << " read (size " << n_var;
			std::cout << ", " << n_var_parsed-1 << "/" << bgenView->number_of_variants();
			std::cout << " variants parsed)" << std::endl;

			// Normalise genotypes
			// center_matrix( G, n_var );
			// scale_matrix( G, n_var );
			std::cout << G << std::endl;

			// Actually compute models
			if(params.mode_lm){
				calc_lrts();
				output_results();
			} else if(params.mode_lm2){
				calc_lrts2();
			} else if(params.mode_joint_model){
				calc_joint_model();
				output_results();
			}
			ch++;
		}
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
