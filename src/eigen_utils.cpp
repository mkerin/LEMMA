//
// Created by kerin on 2019-02-28.
//

#include "eigen_utils.hpp"
#include "mpi_utils.hpp"
#include "typedefs.hpp"

#include "tools/eigen3.3/Dense"
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include <iostream>
#include <cmath>
#include <map>
#include <vector>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace boost_io = boost::iostreams;

Eigen::MatrixXd EigenUtils::subset_matrix(const Eigen::MatrixXd &orig,
                                          const std::vector<int> &valid_points) {
	long n_cols = orig.cols(), n_rows = valid_points.size();
	Eigen::MatrixXd subset(n_rows, n_cols);

	for(int kk = 0; kk < n_rows; kk++) {
		for(int jj = 0; jj < n_cols; jj++) {
			subset(kk, jj) = orig(valid_points[kk], jj);
		}
	}
	return subset;
}

template <typename EigenMat, typename Map>
EigenMat EigenUtils::remove_rows( EigenMat& M,
	                              const Map& incomplete_cases ) {
		// Remove rows contained in incomplete_cases
		EigenMat M_tmp;
		long n_cols = M.cols(), n_rows = M.rows();
		long n_incomplete = 0;
		for (const auto& kv : incomplete_cases) {
			n_incomplete += kv.second;
		}
		M_tmp.resize(n_rows - n_incomplete, n_cols);

		// Fill M_tmp with non-missing entries of M
		int ii_tmp = 0;
		for (std::size_t ii = 0; ii < n_rows; ii++) {
			if ((incomplete_cases.find(ii) == incomplete_cases.end()) || (!incomplete_cases.at(ii))) {
				for (int kk = 0; kk < n_cols; kk++) {
					M_tmp(ii_tmp, kk) = M(ii, kk);
				}
				ii_tmp++;
			}
		}
		return M_tmp;
	}

template <typename EigenMat>
void EigenUtils::write_matrix(boost_io::filtering_ostream& outf,
                              EigenMat& M,
                              std::vector<std::string >& col_names,
                              std::vector<std::string>& row_names){
	if(!col_names.empty()) {
		if (row_names.size() > 0) assert(col_names.size() == M.cols() + 1);
		if (row_names.size() == 0) assert(col_names.size() == M.cols());
		for (int jj = 0; jj < col_names.size(); jj++) {
			outf << col_names[jj];
			if (jj < col_names.size() - 1) {
				outf << " ";
			}
		}
		outf << std::endl;
	}

	if(!row_names.empty()) {
		assert(row_names.size() == M.rows());
		for (long ii = 0; ii < M.rows(); ii++) {
			outf << row_names[ii] << " ";
			for (long jj = 0; jj < M.cols(); jj++) {
				outf << M(ii, jj);
				if (jj < M.cols() - 1) {
					outf << " ";
				}
			}
			outf << std::endl;
		}
	} else {
		outf << M;
	}
}

template <typename EigenMat>
void EigenUtils::read_matrix( const std::string& filename,
                              EigenMat& M){
	std::vector<std::string> placeholder;
	EigenUtils::read_matrix(filename, M, placeholder);
}

template <typename EigenMat>
void EigenUtils::read_matrix(const std::string &filename,
                             EigenMat &M,
                             std::vector <std::string> &col_names){
	std::map<long, bool> incomplete_row;
	EigenUtils::read_matrix(filename, M, col_names, incomplete_row);
}

template <typename EigenMat>
void EigenUtils::read_matrix(const std::string &filename,
                             EigenMat &M,
                             std::vector <std::string> &col_names,
                             std::map<long, bool> &incomplete_row) {
	/* Read txt file into martix. Files can be gzipped. */

	boost_io::filtering_istream fg;
	std::string gz_str = ".gz";
	if (filename.find(gz_str) != std::string::npos) {
		fg.push(boost_io::gzip_decompressor());
	}
	fg.push(boost_io::file_source(filename));
	if (!fg) {
		throw std::runtime_error(filename+" could not be opened.");
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
		throw std::runtime_error(filename+" contains zero lines.");
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
	std::cout << "Reading matrix of size " << n_rows << " x " << n_cols << " from " << filename << std::endl;

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
			if (sss == "NA" || sss == "NAN" || sss == "NaN" || sss == "nan") {
				tmp_d = 0;
				incomplete_row[i] = true;
			} else {
				try {
					tmp_d = stod(sss);
				} catch (const std::invalid_argument &exc) {
					std::cout << sss << " on line " << i << std::endl;
					throw;
				}
			}
			M(i, k) = tmp_d;
		}
		i++;
	}
}

void EigenUtils::read_matrix_and_skip_cols(const std::string &filename,
                                           const int& n_skip_cols,
                                           Eigen::MatrixXd &M,
                                           std::vector <std::string> &col_names) {
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
		throw std::runtime_error(filename+" could not be opened.");
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
		throw std::runtime_error(filename+" contains zero lines.");
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
	assert(n_skip_cols < n_cols);

	// Write remainder of file to Eigen matrix M
	M.resize(n_rows, n_cols - n_skip_cols);
	int i = 0;
	double tmp_d;
	while (getline(fg, line)) {
		if (i >= n_rows) {
			throw std::runtime_error("ERROR: could not convert txt file (too many lines).");
		}
		ss.clear();
		ss.str(line);
		for (int k = 0; k < n_cols; k++) {
			std::string s;
			ss >> s;
			if (k >= n_skip_cols) {
				try {
					tmp_d = stod(s);
				} catch (const std::invalid_argument &exc) {
					std::cout << s << " on line " << i << std::endl;
					throw;
				}
				M(i, k - n_skip_cols) = tmp_d;
			}
		}
		i++;
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

template <typename EigenMat>
void EigenUtils::scale_matrix_and_remove_constant_cols(EigenMat &M,
                                                       long &n_cols,
                                                       std::vector<std::string> &col_names){
	// Scale eigen matrix passed by reference.
	// Removes columns with zero variance + updates col_names.
	// Only call on matrixes which have been reduced to complete cases,
	// as no check for incomplete rows.
	double Nlocal = M.rows();
	double Nglobal = mpiUtils::mpiReduce_inplace(&Nlocal);

	std::vector<std::size_t> keep;
	std::vector<std::string> keep_names;
	std::vector<std::string> reject_names;
	for (std::size_t k = 0; k < n_cols; k++) {
		double sum_mik2 = M.col(k).array().square().sum();
		sum_mik2 = mpiUtils::mpiReduce_inplace(&sum_mik2);
//		double sigma = 0.0;
//		double count = 0;
//		for (int i = 0; i < n_rows; i++) {
//			double val = M(i, k);
//			sigma += val * val;
//			count += 1;
//		}
		double sigma = std::sqrt(sum_mik2 / (Nglobal - 1));

//		sigma = sqrt(sigma/(count - 1));
		if (sigma > 1e-12) {
			M.col(k).array() /= sigma;
//			for (int i = 0; i < n_rows; i++) {
//				M(i, k) /= sigma;
//			}
			keep.push_back(k);
			keep_names.push_back(col_names[k]);
		} else {
			reject_names.push_back(col_names[k]);
		}
	}

	if (keep.size() != n_cols) {
		std::cout << " Removing " << (n_cols - keep.size())  << " column(s) with zero variance:" << std::endl;
		for(auto name : reject_names) {
			std::cout << name << std::endl;
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

template <typename EigenMat>
void EigenUtils::center_matrix(EigenMat& M){
	// Center eigen matrix passed by reference.
	// Only call on matrixes which have been reduced to complete cases,
	// as no check for incomplete rows.
	long n_cols = M.cols();
	long n_rows = M.rows();
	double Nlocal = M.rows();
	double Nglobal = mpiUtils::mpiReduce_inplace(&Nlocal);

	for (int k = 0; k < n_cols; k++) {
		double sum_mik = 0.0;
		for (int i = 0; i < n_rows; i++) {
			sum_mik += M(i, k);
		}

		sum_mik = mpiUtils::mpiReduce_inplace(&sum_mik);
		double mu = sum_mik / Nglobal;
		for (int i = 0; i < n_rows; i++) {
			M(i, k) -= mu;
		}
	}
}

Eigen::MatrixXd EigenUtils::project_out_covars(Eigen::Ref<Eigen::MatrixXd> rhs, const Eigen::Ref<const Eigen::MatrixXd> &C,
                                               const Eigen::Ref<const Eigen::MatrixXd> &CtC_inv) {
	assert(CtC_inv.cols() == C.cols());
	assert(CtC_inv.rows() == C.cols());
	assert(C.rows() == rhs.rows());
	Eigen::MatrixXd CtRHS = C.transpose() * rhs;
	CtRHS = mpiUtils::mpiReduce_inplace(CtRHS);
	Eigen::MatrixXd beta = CtC_inv * CtRHS;
	Eigen::MatrixXd yhat = C * beta;
	Eigen::MatrixXd res = rhs - yhat;
	return res;
}

// Explicit instantiation
// https://stackoverflow.com/questions/2152002/how-do-i-force-a-particular-instance-of-a-c-template-to-instantiate

template void EigenUtils::read_matrix(const std::string&,
                                      Eigen::MatrixXf&, std::vector<std::string>&, std::map<long, bool>&);
template void EigenUtils::read_matrix(const std::string&,
                                      Eigen::MatrixXd&, std::vector<std::string>&, std::map<long, bool>&);
template void EigenUtils::read_matrix(const std::string&,
                                      Eigen::MatrixXf&, std::vector<std::string>&);
template void EigenUtils::read_matrix(const std::string&,
                                      Eigen::MatrixXd&, std::vector<std::string>&);
template void EigenUtils::read_matrix(const std::string&, Eigen::MatrixXf&);
template void EigenUtils::read_matrix(const std::string&, Eigen::MatrixXd&);
template void EigenUtils::write_matrix(boost_io::filtering_ostream&,
                                       Eigen::VectorXd&,
                                       std::vector<std::string>&,
                                       std::vector<std::string>&);
template void EigenUtils::write_matrix(boost_io::filtering_ostream&,
                                       Eigen::MatrixXd&,
                                       std::vector<std::string>&,
                                       std::vector<std::string>&);
template void EigenUtils::center_matrix(Eigen::MatrixXd&);
template void EigenUtils::center_matrix(Eigen::MatrixXf&);
template void EigenUtils::center_matrix(Eigen::VectorXd&);
template void EigenUtils::center_matrix(Eigen::VectorXf&);
template void EigenUtils::scale_matrix_and_remove_constant_cols(Eigen::MatrixXf&,
                                                                long&, std::vector<std::string>&);
template void EigenUtils::scale_matrix_and_remove_constant_cols(Eigen::MatrixXd&,
                                                                long&, std::vector<std::string>&);
template void EigenUtils::scale_matrix_and_remove_constant_cols(Eigen::VectorXd&,
                                                                long&, std::vector<std::string>&);
template EigenDataMatrix EigenUtils::remove_rows( EigenDataMatrix&,
	                                       const std::map<long, bool>&);
template EigenDataMatrix EigenUtils::remove_rows( EigenDataMatrix&,
	                                       const std::unordered_map<long, bool>&);
