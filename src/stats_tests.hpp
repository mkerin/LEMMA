// parse_arguments
#ifndef STATS_TESTS_HPP
#define STATS_TESTS_HPP
#define CATCH_CONFIG_MAIN

#include <cmath>

#include "tools/eigen3.3/Dense"
#include "tools/eigen3.3/Sparse"
#include "tools/eigen3.3/Eigenvalues"

#include <boost/math/distributions/non_central_chi_squared.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/complement.hpp> // complements

namespace boost_m  = boost::math;

template <typename Derived>
double student_t_test(long nn,
					  int jj,
					  const Eigen::MatrixBase<Derived>& HtH_inv,
					  const Eigen::MatrixBase<Derived>& Hty,
					  double rss){
	/* 2-sided Student t-test on regression output
	H0: beta[jj] != 0
 	*/
	int pp = HtH_inv.rows();
	assert(jj <= pp);

	auto beta = HtH_inv * Hty;
	double tstat = beta(jj, 0);
	tstat /= std::sqrt(rss * HtH_inv(jj, jj) / (double) (nn - pp));

	boost_m::students_t t_dist(nn - pp);
	double pval  = 2 * boost_m::cdf(boost_m::complement(t_dist, fabs(tstat)));
	return pval;
}

#endif
