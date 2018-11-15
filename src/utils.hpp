// class for implementation of variational bayes algorithm
#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <cmath>
#include "tools/eigen3.3/Dense"
#include "tools/eigen3.3/Sparse"
#include "tools/eigen3.3/Eigenvalues"

inline double sigmoid(double x){
	return 1.0 / (1.0 + std::exp(-x));
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
