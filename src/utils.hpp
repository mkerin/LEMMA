// class for implementation of variational bayes algorithm
#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <cmath>

inline double sigmoid(double x){
	return 1.0 / (1.0 + std::exp(-x));
}

#endif
