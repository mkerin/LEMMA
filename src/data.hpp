// File of Data class for use with src/bgen_prog.cpp
#ifndef DATA_H
#define DATA_H

#include <iostream>
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

	std::vector< double > info;
	std::vector< double > maf;

	std::map<int, bool> missing_covars; // set of subjects missing >= 1 covariate
	std::map<int, bool> missing_phenos; // set of subjects missing >= phenotype

	std::vector< std::string > pheno_names;
	std::vector< std::string > covar_names;

	Eigen::MatrixXd G; // genotype matrix
	Eigen::MatrixXd Y; // phenotype matrix
	Eigen::MatrixXd W; // covariate matrix
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
		// Center + scale eigen matrix passed by reference.

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
						M(i, k) = M(i, k) / sigma;
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
	}

	void read_pheno( ){
		if ( params.pheno_file != "NULL" ) {
			read_txt_file( params.pheno_file, Y, n_pheno, pheno_names, missing_phenos );
			center_matrix( Y, n_pheno, missing_phenos );
		}

		if (n_pheno == 0) {
			throw std::runtime_error("ERROR: No pheno's with nonzero variance");
		}
	}

	void read_covar( ){
		if ( params.covar_file != "NULL" ) {
			read_txt_file( params.covar_file, W, n_covar, covar_names, missing_covars );
			center_matrix( W, n_covar, missing_covars );
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
