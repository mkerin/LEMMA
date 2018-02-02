// File of Data class for use with src/bgen_prog.cpp
#ifndef DATA_H
#define DATA_H

#include <iostream>
#include <cmath>
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

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>

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

	bool G_reduced;   // Variables to track whether we have already
	bool Y_reduced;   // reduced to complete cases or not.
	bool W_reduced;

	std::vector< double > info;
	std::vector< double > maf;

	std::map<int, bool> missing_covars; // set of subjects missing >= 1 covariate
	std::map<int, bool> missing_phenos; // set of subjects missing >= phenotype
	std::map< int, bool > incomplete_cases; // union of samples missing data

	std::vector< std::string > pheno_names;
	std::vector< std::string > covar_names;

	Eigen::MatrixXd G; // genotype matrix
	Eigen::MatrixXd Y; // phenotype matrix
	Eigen::MatrixXd W; // covariate matrix
	Eigen::VectorXd Z; // interaction vector
	BgenParser bgenParser; // hopefully initialised appropriately from
						   // initialisation list in Data constructor
	std::vector< double > beta, tau, neglogP;

	boost_io::filtering_ostream outf;
	
	// constructors/destructors
	// data() : bgenParser( "NULL" ) {
	// 	bgen_pass = false; // No bgen file set; read_bgen_chunk won't run.
	// }

	data( std::string filename ) : bgenParser( filename ){
		bgen_pass = true;
		n_samples = bgenParser.number_of_samples();
	}
	
	~data() {
	}

	void output_init() {
		// open output file
		std::string ofile, gz_str = ".gz";
		std::cout << "Opening output files ..." << std::endl;

		ofile = params.out_file;
		outf.push(boost_io::file_sink(ofile.c_str()));
		// outf << "chr rsid pos a_0 a_1 af info";

		if(params.mode_vcf){
			// Output header for vcf file
			outf << "##fileformat=VCFv4.2\n"
				<< "FORMAT=<ID=GP,Type=Float,Number=G,Description=\"Genotype call probabilities\">\n"
				<< "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT" ;
			bgenParser.get_sample_ids(
				[&]( std::string const& id ) { outf << "\t" << id ; }
			) ;
			outf << "\n" ;
		}

		if(params.mode_lm){
			// Output header for vcf file
			outf << "chr rsid pos a_0 a_1 af beta tau neglogP" << std::endl;
		}
	}

	bool read_bgen_chunk() {
		// Wrapper around BgenParser to read in a 'chunk' of data. Remembers
		// if last call hit the EOF, and returns false if so.
		// Assumed that:
		// - commandline args parsed and passed to params
		// - bgenParser initialised with correct filename
		std::cout << "Entering read_bgen_chunk()" << std::endl;
		// Exit function if last call hit EOF.
		if (!bgen_pass) return false;

		// Update n_samples (only needed on 1st run, but not sure where to put this)
		// n_samples = bgenParser.number_of_samples(); // TODO: move this line.

		// Temporary variables to store info from read_variant()
		std::string my_chr ;
		uint32_t my_pos ;
		std::string my_rsid ;
		std::vector< std::string > my_alleles ;
		std::vector< std::vector< double > > probs ;
		std::map<int, bool> missing_genos;

		double d1, af, x, dosage, check;

		// Wipe variant info from last chunk
		maf.clear();
		rsid.clear();
		chromosome.clear();
		position.clear();
		alleles.clear();

		// Resize genotype matrix
		G.resize(n_samples, params.chunk_size);

		std::size_t valid_count, jj = 0;
		while ( jj < params.chunk_size && bgen_pass ) {
			bgen_pass = bgenParser.read_variant( &my_chr, &my_pos, &my_rsid, &my_alleles );
			if (!bgen_pass) break;
			assert( my_alleles.size() > 0 );

			// range filter
			if (params.range && (my_pos < params.start || my_pos > params.end)){
				bgenParser.ignore_probs();
				continue;
			}

			// Read probs + check maf filter
			bgenParser.read_probs( &probs );

			// maf filter; computed on valid sample_ids & variants whose alleles
			// sum to 1
			d1 = 0.0;
			valid_count = 0;
			for( std::size_t ii = 0; ii < probs.size(); ++ii ) {
				if (incomplete_cases.count(ii) == 0) {
					check = 0.0;
					dosage = 0.0;
					for( std::size_t kk = 0; kk < probs[ii].size(); ++kk ) {
						x = probs[ii][kk];
						dosage += x * kk;
						check += x;
					}
					if(check > 0.9999 && check < 1.0001){
						d1 += dosage;
						valid_count++;
					}
				}
			}
			af = d1 / (2.0 * valid_count);
			if (params.maf_lim && af < params.min_maf) {
				continue;
			}

			// filters passed; write contextual info
			maf.push_back(af);
			rsid.push_back(my_rsid);
			chromosome.push_back(my_chr);
			position.push_back(my_pos);
			alleles.push_back(my_alleles);
			
			// filters passed; write dosage to G
			// Note that we only write dosage for valid sample ids
			std::size_t ii_obs = 0;
			double mu = 0.0;
			double count = 0;
			missing_genos.clear();
			for( std::size_t ii = 0; ii < probs.size(); ++ii ) {
				if (incomplete_cases.count(ii) == 0) {
					dosage = 0.0;
					check = 0.0;

					for( std::size_t kk = 0; kk < probs[ii].size(); ++kk ) {
						x = probs[ii][kk];
						dosage += x * kk;
						check += x;
					}

					if(check > 0.9999 && check < 1.0001){
						G(ii_obs,jj) = dosage;
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

					ii_obs++; // loop should end at ii_obs == n_samples
				}
			}

			if (ii_obs < n_samples) {
				throw std::logic_error("ERROR: Fewer non-missing genotypes than expected");
			}

			// Set missing entries to mean
			// Could mean center here, but still want to write to VCF.
			mu = mu / count;
			for (int ii = 0; ii < n_samples; ii++) {
				if (missing_genos.count(ii) != 0) {
					G(ii, jj) = mu;
				}
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
		G_reduced = false;
		std::cout << "Chunk size: " << jj << std::endl;

		if(jj == 0){
			// Immediate EOF
			return false;
		} else {
			return true;
		}
	}

	void read_incl_sids(){
		boost_io::filtering_istream fg;
		fg.push(boost_io::file_source(params.incl_sids_file.c_str()));
		if (!fg) {
			std::cout << "ERROR: " << params.incl_sids_file << " not opened." << std::endl;
			std::exit(EXIT_FAILURE);
		}

		std::vector<std::string> bgen_ids;
		bgenParser.get_sample_ids(
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
				// TODO: This comparison is not yet working.
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
		std::cout << " Detected " << n_cols << " columns from " << filename << std::endl;

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
						tmp_d = stod(s);
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
						int& n_cols,
						std::map< int, bool > incomplete_row ){
		// Center eigen matrix passed by reference.

		std::vector<size_t> keep;
		for (int k = 0; k < n_cols; k++) {
			double mu = 0.0;
			double count = 0;
			for (int i = 0; i < n_samples; i++) {
				if (incomplete_row.count(i) == 0) {
					mu += M(i, k);
					count += 1;
				}
			}

			mu = mu / count;
			for (int i = 0; i < n_samples; i++) {
				if (incomplete_row.count(i) == 0) {
					M(i, k) -= mu;
				} else {
					M(i, k) = 0.0;
				}
			}
			// std::cout << "Mean centered matrix:" << std::endl << M << std::endl;
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
						int& n_cols ){
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

	void scale_matrix( Eigen::MatrixXd& M,
						int& n_cols,
						std::map< int, bool > incomplete_row ){
		// Scale eigen matrix passed by reference.
		// Removes columns with zero variance.

		std::vector<size_t> keep;
		for (int k = 0; k < n_cols; k++) {
			double sigma = 0.0;
			double count = 0;
			for (int i = 0; i < n_samples; i++) {
				if (incomplete_row.count(i) == 0) {
					double val = M(i, k);
					sigma += val * val;
					count += 1;
				}
			}

			sigma = sqrt(sigma/(count - 1));
			if (sigma > 1e-12) {  
				for (int i = 0; i < n_samples; i++) {
					if (incomplete_row.count(i) == 0) {
						M(i, k) /= sigma;
					}
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
			throw std::invalid_argument( "Tried to read NULL covar file." );
		}
		W_reduced = false;
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
		Eigen::MatrixXd ww = W.rowwise() - W.colwise().mean(); //not needed probably
		Eigen::MatrixXd bb = solve(ww.transpose() * ww, ww.transpose() * Y);
		Y = Y - ww * bb;
	}

	void calc_lrts() {
		// For-loop through variants and compute interaction models.
		// Save to
		// data.tau, data.beta
		// Y is a matrix of dimension n_samples x 1
		std::cout << "Entering calc_lrts()" << std::endl;
		// Eigen::MatrixXd e_j, f_j, G_j, Z_j;
		Eigen::VectorXd e_j, f_j;
		double beta_j, tau_j, xtx_inv, loglik_null, loglik_alt, chi_stat;
		long double pval;
		boost::math::chi_squared chi_dist(1);

		Eigen::VectorXd vv(Eigen::Map<Eigen::VectorXd>(W.col(0).data(), n_samples));
		Eigen::MatrixXd Z = G.array().colwise() * vv.array();

		// Output stats
		// std::cout << "W is " << W.rows() << "x" << W.cols() << std::endl;
		// std::cout << "G is " << G.rows() << "x" << G.cols() << std::endl;
		// std::cout << "Y\tG\tW\tZ" << std::endl;
		// for (int ii = 0; ii < n_samples; ii++){
		// 	std::cout << Y(ii,0) << "\t" << G(ii,0) << "\t" << W(ii,0) << "\t";
		// 	std::cout << Z(ii,0) << "\t" << std::endl;
		// }

		beta.clear();
		tau.clear();
		neglogP.clear();
		xtx_inv = 1.0 / (n_samples - 1.0);

		for (int jj = 0; jj < n_var; jj++){
			std::cout << "Computing lrts for variant: " << jj << std::endl;
			// G_j = G.col(jj);
			// Z_j = Z.col(jj);
			Eigen::Map<Eigen::VectorXd> G_j(G.col(jj).data(), n_samples);
			Eigen::Map<Eigen::VectorXd> Z_j(Z.col(jj).data(), n_samples);
			
			beta_j = xtx_inv * (G_j.transpose() * Y)(0,0);
			e_j = Y - G_j * beta_j;
			loglik_null = std::log(n_samples) - std::log(e_j.dot(e_j));
			loglik_null *= n_samples/2.0;

			tau_j = (Z_j.transpose() * e_j)(0,0) / (Z_j.transpose() * Z_j)(0,0);
			f_j = e_j - Z_j * tau_j;
			loglik_alt = std::log(n_samples) - std::log(f_j.dot(f_j));
			loglik_alt *= n_samples/2.0;

			std::cout << "Null loglik: " << loglik_null << std::endl;
			std::cout << "Alt loglik: " << loglik_alt << std::endl;

			chi_stat = 2*(loglik_alt - loglik_null);
			std::cout << "Test statistic: " << chi_stat << std::endl;
			pval = 1 - boost::math::cdf(chi_dist, chi_stat);
			

			// Saving variables
			beta.push_back(beta_j);
			tau.push_back(tau_j);
			neglogP.push_back(-1 * std::log10(pval));
		}
	}

	void output_lm() {
		for (int s = 0; s < n_var; s++){
			outf << chromosome[s] << " " << rsid[s] << " " << position[s] << " ";
			outf << alleles[s][0] << " " << alleles[s][1] << " " << maf[s] << " ";
			outf << beta[s] << " " << tau[s] << " " << neglogP[s] << std::endl;
		}
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
		scale_matrix( W, n_covar );

		// Step 4; Regress covars out of phenos
		regress_covars();

		while (read_bgen_chunk()) {
			// Raw dosage read in to G
			std::cout << "Raw G is " << G.rows() << "x" << G.cols() << std::endl;
			std::cout << G << std::endl;

			// Normalise genotypes
			center_matrix( G, n_var );
			scale_matrix( G, n_var );
			std::cout << "Normalised G is " << G.rows() << "x" << G.cols() << std::endl;
			std::cout << G << std::endl;

			// Actually compute models
			calc_lrts();
			output_lm();
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
