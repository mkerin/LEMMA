//
// Created by kerin on 2019-02-28.
//

#ifndef LEMMA_EIGEN_UTILS_HPP
#define LEMMA_EIGEN_UTILS_HPP

#include "tools/eigen3.3/Dense"

#include <iostream>
#include <cmath>
#include <map>
#include <vector>
#include <string>
#include <set>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>

namespace boost_io = boost::iostreams;

namespace EigenUtils {

template <typename EigenMat>
void center_matrix(EigenMat& M);

template <typename EigenMat>
void scale_matrix_and_remove_constant_cols(EigenMat &M,
                                           long &n_cols,
                                           std::vector<std::string> &col_names);

template <typename EigenMat>
void scale_matrix_and_remove_constant_cols(EigenMat &M){
	long n_cols = M.cols();
	std::vector<std::string> placeholder;
	for (long ii = 0; ii < n_cols; ii++) {
		placeholder.emplace_back("tmp");
	}
	scale_matrix_and_remove_constant_cols(M, n_cols, placeholder);
}

Eigen::MatrixXf solve(const Eigen::MatrixXf &A, const Eigen::MatrixXf &b);

Eigen::MatrixXd solve(const Eigen::MatrixXd &A, const Eigen::MatrixXd &b);

Eigen::MatrixXd subset_matrix(const Eigen::MatrixXd &orig, const std::vector<int> &valid_points);

template <typename EigenMat>
void read_matrix(const std::string& filename,
                 const long& n_rows,
                 EigenMat& M,
                 std::vector< std::string >& col_names,
                 std::map<long, bool>& incomplete_row );


void read_matrix(const std::string& filename,
                 Eigen::MatrixXd& M,
                 std::vector< std::string >& col_names);

void read_matrix(const std::string& filename,
                 Eigen::MatrixXd& M);

void read_matrix_and_skip_cols(const std::string &filename,
                               const int& n_skip_cols,
                               Eigen::MatrixXd &M,
                               std::vector <std::string> &col_names);

template <typename EigenMat>
void write_matrix(boost_io::filtering_ostream& outf,
                  EigenMat& M,
                  std::vector<std::string>& col_names,
                  std::vector<std::string>& row_names);

Eigen::MatrixXd project_out_covars(Eigen::Ref<Eigen::MatrixXd> rhs,
                                   const Eigen::Ref<const Eigen::MatrixXd> &C,
                                   const Eigen::Ref<const Eigen::MatrixXd> &CtC_inv);
}

#endif //LEMMA_EIGEN_UTILS_HPP
