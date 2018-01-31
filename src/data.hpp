// File of Data class for use with src/bgen_prog.cpp
#ifndef DATA_H
#define DATA_H

#include <iostream>
#include <cmath>
#include <map>
#include <vector>
#include <string>
#include "class.h"
#include "tools/eigen3.3/Dense"
#include "tools/eigen3.3/Sparse"
#include "tools/eigen3.3/Eigenvalues"

#include "bgen_parser.hpp"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>

namespace boost_io = boost::iostreams;

inline Eigen::MatrixXd getCols(const Eigen::MatrixXd &X, const std::vector<size_t> &cols);
inline void setCols(Eigen::MatrixXd &X, const std::vector<size_t> &cols, const Eigen::MatrixXd &values);
inline size_t numRows(const Eigen::MatrixXd &A);
inline size_t numCols(const Eigen::MatrixXd &A);
inline void setCol(Eigen::MatrixXd &A, const Eigen::VectorXd &v, size_t col);
inline Eigen::VectorXd getCol(const Eigen::MatrixXd &A, size_t col);


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
	int G_ncol;

	bool G_reduced;   // Variables to track whether we have already
	bool Y_reduced;   // reduced to complete cases or not.
	bool W_reduced;

	std::vector< double > info;
	std::vector< double > maf;

	std::map<int, bool> missing_covars; // set of subjects missing >= 1 covariate
	std::map<int, bool> missing_phenos; // set of subjects missing >= phenotype

	std::vector< std::string > pheno_names;
	std::vector< std::string > covar_names;

	Eigen::MatrixXd G; // genotype matrix
	Eigen::MatrixXd Y; // phenotype matrix
	Eigen::MatrixXd W; // covariate matrix
	Eigen::VectorXd Z; // interaction vector
	BgenParser bgenParser; // hopefully initialised appropriately from
						   // initialisation list in Data constructor

	boost_io::filtering_ostream outf;
	
	// constructors/destructors
	// data() : bgenParser( "NULL" ) {
	// 	bgen_pass = false; // No bgen file set; read_bgen_chunk won't run.
	// }

	data( std::string filename ) : bgenParser( filename ){
		bgen_pass = true;
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
	}

	bool read_bgen_chunk() {
		// Wrapper around BgenParser to read in a 'chunk' of data. Remembers
		// if last call hit the EOF, and returns false if so.
		// Assumed that:
		// - commandline args parsed and passed to params
		// - bgenParser initialised with correct filename
		std::cout << "Entering read_bgen_chunk()" << std::endl;
		// Exit function if last call ran out of variants to read in.
		if (!bgen_pass) return false;

		// Update n_samples (only needed on 1st run, but not sure where to put this)
		n_samples = bgenParser.number_of_samples();

		// Temporary variables to store info from read_variant()
		std::string my_chr ;
		uint32_t my_pos ;
		std::string my_rsid ;
		std::vector< std::string > my_alleles ;
		std::vector< std::vector< double > > probs ;

		double d1, af, x, dosage;

		// Wipe variant info from last chunk
		maf.clear();
		rsid.clear();
		chromosome.clear();
		position.clear();
		alleles.clear();

		// Resize genotype matrix
		G.resize(n_samples, params.chunk_size);

		int jj = 0;
		while ( jj < params.chunk_size && bgen_pass ) {
			bgen_pass = bgenParser.read_variant( &my_chr, &my_pos, &my_rsid, &my_alleles );
			if (!bgen_pass) break;
			assert( my_alleles.size() > 0 );

			// range filter
			if (params.range && (my_pos < params.start || my_pos > params.end)){
				bgenParser.ignore_probs();
				continue;
			}

			// maf + info filters
			d1 = 0.0;
			for( std::size_t i = 0; i < probs.size(); ++i ) {
				for( std::size_t j = 0; j < probs[i].size(); ++j ) {
					x = probs[i][j];
					d1 += x * j;
				}
			}
			af = d1 / (2.0 * n_samples);
			if (params.maf_lim && af < params.min_maf) {
				bgenParser.ignore_probs();
				continue;
			}

			// filters passed; write dosage to G
			maf.push_back(af);
			rsid.push_back(my_rsid);
			chromosome.push_back(my_chr);
			position.push_back(my_pos);
			alleles.push_back(my_alleles);
			
			bgenParser.read_probs( &probs ) ;
			for( std::size_t ii = 0; ii < probs.size(); ++ii ) {
				dosage = 0.0;
				for( std::size_t kk = 0; kk < probs[ii].size(); ++kk ) {
					x = probs[ii][kk];
					dosage += x * kk;
				}
				G(ii,jj) = dosage;
			}
			jj++;
		}

		// if while loop exits early due to EOF,
		// we need to resize G whilst retaining existing coefficients.
		G.conservativeResize(n_samples, jj);
		assert( rsid.size() == jj );
		assert( chromosome.size() == jj );
		assert( position.size() == jj );
		assert( alleles.size() == jj );
		G_ncol = jj;

		return true;
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
			exit(-1);
		}

		// Reading column names
		std::string line;
		if (!getline(fg, line)) {
			std::cout << "ERROR: " << filename << " not read." << std::endl;
			exit(-1);
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
			throw std::runtime_error("ERROR: could not convert txt file.");
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
			for (int i = 0; i < n_samples; i++) {
				if (incomplete_row.count(i) == 0) {
					double val = M(i, k);
					sigma += val * val;
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
		// Reduce phenos, covars, genotypes to the set of complete cases across covar/pheno?
		// Center the columns (inc. set non-complete cases to 0?)
		// Compute residual phenotypes after regressing out covariates
		// Then scale columns.
		if ( params.pheno_file != "NULL" ) {
			read_txt_file( params.pheno_file, Y, n_pheno, pheno_names, missing_phenos );
			// center_matrix( Y, n_pheno, missing_phenos );
		}
	}

	void read_covar( ){
		// Read covariates to Eigen matrix W, then center and scale the columns
		if ( params.covar_file != "NULL" ) {
			read_txt_file( params.covar_file, W, n_covar, covar_names, missing_covars );
			// center_matrix( W, n_covar, missing_covars );
			// scale_matrix( W, n_covar, missing_covars );
		}
	}

	void reduce_to_complete_cases( Eigen::MatrixXd& M, 
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
		M_tmp.resize(n_cols, n_samples - n_incomplete);

		// Fill M_tmp with non-missing entries of M
		int ii_tmp = 0;
		for (int ii = 0; ii < n_samples; ii++) {
			if (incomplete_row.count(ii) == 0) {
				for (int kk = 0; kk < n_cols; kk++) {
					M_tmp(ii_tmp, kk) = M(ii, kk);
				}
				ii_tmp++;
			}
		}

		// Assign new values to reference variables
		M = M_tmp;
		matrix_reduced = true;
	}

	void regress_covars() {
		Eigen::MatrixXd ww = W.rowwise() - W.colwise().mean(); //not needed probably
		Eigen::MatrixXd bb = solve(ww.transpose() * ww, ww.transpose() * y);
		Y = Y - ww * bb;
	}

	void calc_lrts() {
		// Need to think a little about how to implement this.
		double xtx_inv = 1.0 / (n_samples - 1.0);

		// Null model
		// beta;         n_pheno x n_var
		// G;            n_samples x n_var
		// Y.tranpose(); n_pheno x n_samples
		// ee;           n_samples x n_pheno <- Nope, thats not what we want.
		// 
		Eigen::MatrixXd beta, ee;
		beta = xtx_inv * Y.transpose() * G;
		ee = Y - G * beta.transpose();
		loglik_null = - N * std::log( ee.transpose() * ee ) / 2.0;

		// 1 Dof interction model; niave implementation
		// tbc

		// 1 Dof interaction model; 2 step regression version
		// X_1dof;       n_samples x n_var
		// beta_1dof;    n_pheno x n_var
		// ee_1dof;      n_samples x n_pheno
		// loglik_1dof;  n_pheno x n_var
		Eigen::MatrixXd X_1dof, beta_1dof;
		Z = W.col(1);
		X_1dof = G.colwise() * Z;
		center_matrix( X_1dof, n_samples, missing_covars ); // missing_covars should be empty
		scale_matrix( X_1dof, n_samples, missing_covars );
		beta_1dof = xtx_inv * ee.transpose() * G;
		ee_1dof = Y - G * beta.transpose();
		loglik_1dof = - N * std::log( ee_1dof.transpose() * ee_1dof ) / 2.0;
		
		
		
	}

	void run() {
		int ch = 0;
		while (read_bgen_chunk()) {
			if (ch == 0) {
				read_covar();
				read_pheno();

				// Step 2; Reduce to complete cases
				std::map< int, bool > incomplete_cases;
				incomplete_cases.insert(missing_covars.begin(), missing_covars.end());
				incomplete_cases.insert(missing_phenos.begin(), missing_phenos.end());
				reduce_to_complete_cases( G, G_ncol, G_reduced, incomplete_cases ); 
				reduce_to_complete_cases( Y, n_pheno, Y_reduced, incomplete_cases );
				reduce_to_complete_cases( W, n_covar, W_reduced, incomplete_cases );
				n_samples = n_samples - incomplete_cases.size();
				missing_phenos.clear();
				missing_covars.clear();

				// Step 3; Center phenos, normalise covars
				center_matrix( Y, n_pheno, missing_phenos );
				center_matrix( W, n_covar, missing_covars );
				scale_matrix( W, n_covar, missing_covars );

				// Step 4; Regress covars out of phenos
				regress_covars();

				// Step 5; Scale phenos
				scale_matrix( Y, n_pheno, missing_phenos );
			} else {
				reduce_to_complete_cases( G, incomplete_cases );
			}
			calc_lrts();
			output_pvals();

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

#endif
