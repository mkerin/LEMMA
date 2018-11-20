// class for implementation of variational bayes algorithm
#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <stdexcept>
#include <cmath>
#include "tools/eigen3.3/Dense"
#include "tools/eigen3.3/Sparse"
#include "tools/eigen3.3/Eigenvalues"

/***************** Typedefs *****************/
#ifdef DATA_AS_FLOAT
using scalarData          = float;
using EigenDataMatrix     = Eigen::MatrixXf;
using EigenDataVector     = Eigen::VectorXf;
using EigenDataArrayXX    = Eigen::ArrayXXf;
using EigenDataArrayX     = Eigen::ArrayXf;
using EigenRefDataMatrix  = Eigen::Ref<Eigen::MatrixXf>;
using EigenRefDataVector  = Eigen::Ref<Eigen::VectorXf>;
using EigenRefDataArrayXX = Eigen::Ref<Eigen::ArrayXXf>;
using EigenRefDataArrayX  = Eigen::Ref<Eigen::ArrayXf>;
#else
using scalarData          = double;
using EigenDataMatrix     = Eigen::MatrixXd;
using EigenDataVector     = Eigen::VectorXd;
using EigenDataArrayXX    = Eigen::ArrayXXd;
using EigenDataArrayX     = Eigen::ArrayXd;
using EigenRefDataMatrix  = Eigen::Ref<Eigen::MatrixXd>;
using EigenRefDataVector  = Eigen::Ref<Eigen::VectorXd>;
using EigenRefDataArrayXX = Eigen::Ref<Eigen::ArrayXXd>;
using EigenRefDataArrayX  = Eigen::Ref<Eigen::ArrayXd>;
#endif

/***************** Math *****************/

inline double sigmoid(double x){
	return 1.0 / (1.0 + std::exp(-x));
}

inline Eigen::MatrixXf solve(const Eigen::MatrixXf &A, const Eigen::MatrixXf &b) {
	Eigen::MatrixXf x = A.colPivHouseholderQr().solve(b);
	double check = fabs((double)((A * x - b).norm()/b.norm()));
	if (check > 1e-6) {
		std::string ms = "ERROR: could not solve covariate scatter matrix (Check = " +
						std::to_string(check) + ").";
		throw std::runtime_error(ms);
	}
	return x;
}

inline Eigen::MatrixXd solve(const Eigen::MatrixXd &A, const Eigen::MatrixXd &b) {
	Eigen::MatrixXd x = A.colPivHouseholderQr().solve(b);
	double check = fabs((double)((A * x - b).norm()/b.norm()));
	if (check > 1e-8) {
		std::string ms = "ERROR: could not solve covariate scatter matrix (Check = " +
						std::to_string(check) + ").";
		throw std::runtime_error(ms);
	}
	return x;
}

#endif
