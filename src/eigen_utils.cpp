//
// Created by kerin on 2019-02-28.
//

#include "eigen_utils.hpp"

#include "tools/eigen3.3/Dense"
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include <iostream>
#include <cmath>
#include <map>
#include <vector>
#include <string>
#include <set>

namespace boost_io = boost::iostreams;

Eigen::MatrixXd EigenUtils::subset_matrix(const Eigen::MatrixXd &orig, const std::vector<int> &valid_points) {
	long n_cols = orig.cols(), n_rows = valid_points.size();
	Eigen::MatrixXd subset(n_rows, n_cols);

	for(int kk = 0; kk < n_rows; kk++){
		for(int jj = 0; jj < n_cols; jj++){
			subset(kk, jj) = orig(valid_points[kk], jj);
		}
	}
	return subset;
}

template <typename EigenMat>
void EigenUtils::read_matrix(const std::string &filename, const long &n_rows, EigenMat &M, std::vector<std::string> &col_names,
				 std::map<int, bool> &incomplete_row) {
	/* Assumptions:
	- n_rows constant (number of samples constant across files)
	*/

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
	int n_cols = 0;
	ss.clear();
	ss.str(line);
	while (ss >> s) {
		++n_cols;
		col_names.push_back(s);
	}
	std::cout << " Reading matrix of size " << n_rows << " x " << n_cols << " from " << filename << std::endl;

	// Write remainder of file to Eigen matrix M
	incomplete_row.clear();
	M.resize(n_rows, n_cols);
	int i = 0;
	double tmp_d;
	while (getline(fg, line)) {
		if (i >= n_rows) {
			throw std::runtime_error("ERROR: could not convert txt file (too many lines).");
		}
		ss.clear();
		ss.str(line);
		for (int k = 0; k < n_cols; k++) {
			std::string sss;
			ss >> sss;
			/// NA
			if (sss == "NA" || sss == "NAN" || sss == "NaN" || sss == "nan") {
				tmp_d = 0; // Will skip over this value in future
				incomplete_row[i] = true;
			} else {
				try{
					tmp_d = stod(sss);
				} catch (const std::invalid_argument &exc){
					std::cout << sss << " on line " << i << std::endl;
					throw;
				}
			}
			M(i, k) = tmp_d;
		}
		i++; // loop should end at i == n_samples
	}
	if (i < n_rows) {
		throw std::runtime_error("ERROR: could not convert txt file (too few lines).");
	}
}

void EigenUtils::read_matrix(const std::string &filename, Eigen::MatrixXd &M, std::vector <std::string> &col_names) {
	/* Assumptions:
	- dimensions unknown
	- assume no missing values
	*/

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
	int n_rows = 0;
	getline(fg, line);
	while (getline(fg, line)) {
		n_rows++;
	}
	fg.reset();
	if (filename.find(gz_str) != std::string::npos) {
		fg.push(boost_io::gzip_decompressor());
	}
	fg.push(boost_io::file_source(filename));

	// Reading column names
	if (!getline(fg, line)) {
		std::cout << "ERROR: " << filename << " contains zero lines." << std::endl;
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
	std::cout << " Reading matrix of size " << n_rows << " x " << n_cols << " from " << filename << std::endl;

	// Write remainder of file to Eigen matrix M
	M.resize(n_rows, n_cols);
	int i = 0;
	double tmp_d;
	while (getline(fg, line)){
		if (i >= n_rows) {
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
		i++; // loop should end at i == n_rows
	}
	if (i < n_rows) {
		throw std::runtime_error("ERROR: could not convert txt file (too few lines).");
	}
}

Eigen::MatrixXf EigenUtils::solve(const Eigen::MatrixXf &A, const Eigen::MatrixXf &b) {
	Eigen::MatrixXf x = A.colPivHouseholderQr().solve(b);
	double check = fabs((double)((A * x - b).norm()/b.norm()));
	if (check > 1e-6) {
		// std::string ms = "ERROR: could not solve covariate scatter matrix (Check = " +
		// std::to_string(check) + ").";
		// throw std::runtime_error(ms);
		std::cout << "WARNING: Error in solving covariate scatter matrix is: " << check << std::endl;
	}
	return x;
}

Eigen::MatrixXd EigenUtils::solve(const Eigen::MatrixXd &A, const Eigen::MatrixXd &b) {
	Eigen::MatrixXd x = A.colPivHouseholderQr().solve(b);
	double check = fabs((double)((A * x - b).norm()/b.norm()));
	if (check > 1e-8) {
		std::string ms = "ERROR: could not solve covariate scatter matrix (Check = " +
						 std::to_string(check) + ").";
		throw std::runtime_error(ms);
	}
	return x;
}

// No need to call this TemporaryFunction() function,
// it's just to avoid link error.
void TemporaryFunctionEigenUtils (){
	std::string filename;
	long rows;
	std::vector<std::string> names;
	std::map<int, bool> incomplete;
	Eigen::MatrixXd matXd;
	Eigen::MatrixXf matXf;

	EigenUtils::read_matrix(filename, rows, matXd, names, incomplete);
	EigenUtils::read_matrix(filename, rows, matXf, names, incomplete);
}
