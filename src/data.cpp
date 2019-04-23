//
// Created by kerin on 2019-02-26.
//
#include <map>
#include <string>
#include "data.hpp"
#include "tools/Eigen/Dense"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>


void Data::read_txt_file(const std::string &filename, Eigen::MatrixXd &M, unsigned long &n_cols,
                         std::vector<std::string> &col_names, std::map<int, bool> &incomplete_row) {
	// pass top line of txt file filename to col_names, and body to M.
	// TODO: Implement how to deal with missing values.

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
	std::string line;
	if (!getline(fg, line)) {
		std::cout << "ERROR: " << filename << " contains zero lines." << std::endl;
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
	std::cout << " Reading matrix of size " << n_samples << " x " << n_cols << " from " << filename << std::endl;

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
					} catch (const std::invalid_argument &exc) {
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
			i++;             // loop should end at i == n_samples
		}
		if (i < n_samples) {
			throw std::runtime_error("ERROR: could not convert txt file (too few lines).");
		}
	} catch (const std::exception &exc) {
		throw;
	}
}

template<typename EigenMat>
EigenMat reduce_mat_to_complete_cases(EigenMat &M, bool &matrix_reduced, const unsigned long &n_cols,
                                      const std::map<std::size_t, bool> &incomplete_cases) {
	// Remove rows contained in incomplete_cases
	long nn = M.rows();
	EigenMat M_tmp;
	if (matrix_reduced) {
		throw std::runtime_error("ERROR: Trying to remove incomplete cases twice...");
	}

	// Create temporary matrix of complete cases
	unsigned long n_incomplete = incomplete_cases.size();
	M_tmp.resize(nn - n_incomplete, n_cols);

	// Fill M_tmp with non-missing entries of M
	int ii_tmp = 0;
	for (std::size_t ii = 0; ii < nn; ii++) {
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
