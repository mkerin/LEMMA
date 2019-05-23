// parse_arguments
#ifndef STATS_TESTS_HPP
#define STATS_TESTS_HPP
#define CATCH_CONFIG_MAIN

#include <cmath>

#include "tools/eigen3.3/Dense"
#include "tools/eigen3.3/Eigenvalues"

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/complement.hpp> // complements

namespace boost_m  = boost::math;

template <typename Derived1, typename Derived2>
void prep_lm(const Eigen::MatrixBase<Derived1>& H,
			 const Eigen::MatrixBase<Derived2>& y,
			 EigenRefDataMatrix HtH,
			 EigenRefDataMatrix HtH_inv,
			 EigenRefDataMatrix tau,
			 EigenRefDataMatrix Hty,
			 double& rss,
			 EigenRefDataMatrix HtVH){
	/*** All of the heavy lifting for linear hypothesis tests.
	 * Easier to have in one place if we go down the MPI route.
	 */

	HtH     = H.transpose() * H;
	Hty     = H.transpose() * y;
	tau     = HtH_inv * Hty;

	EigenDataVector resid = y - H * tau;
	HtVH = H.transpose() * resid.cwiseProduct(resid).asDiagonal() * H;
	HtH_inv = HtH.inverse();
	rss = resid.squaredNorm();
}

template <typename Derived1, typename Derived2>
void prep_lm(const Eigen::MatrixBase<Derived1>& H,
			 const Eigen::MatrixBase<Derived2>& y,
			 EigenRefDataMatrix HtH,
			 EigenRefDataMatrix HtH_inv,
			 EigenRefDataMatrix tau,
			 EigenRefDataMatrix Hty,
			 double& rss){
	/*** All of the heavy lifting for linear hypothesis tests.
	 * Easier to have in one place if we go down the MPI route.
	 */

	HtH     = H.transpose() * H;
	Hty     = H.transpose() * y;
	tau     = HtH_inv * Hty;

	EigenDataVector resid = y - H * tau;
	HtH_inv = HtH.inverse();
	rss = resid.squaredNorm();
}

template <typename Derived1, typename Derived2>
void student_t_test(long nn,
                      const Eigen::MatrixBase<Derived1>& HtH_inv,
                      const Eigen::MatrixBase<Derived2>& Hty,
                      double rss,
                      int jj,
                      double& tstat,
                      double& pval){
	/* 2-sided Student t-test on regression output
	H0: beta[jj] != 0
 	*/
	int pp = HtH_inv.rows();
	assert(jj <= pp);

	auto beta = HtH_inv * Hty;
	tstat = beta(jj, 0);
	tstat /= std::sqrt(rss * HtH_inv(jj, jj) / (double) (nn - pp));

	boost_m::students_t t_dist(nn - pp);
	pval  = 2 * boost_m::cdf(boost_m::complement(t_dist, fabs(tstat)));
}

template <typename Derived1, typename Derived2>
double student_t_test(long nn,
                      const Eigen::MatrixBase<Derived1>& HtH_inv,
                      const Eigen::MatrixBase<Derived2>& Hty,
                      double rss,
                      int jj){
	double tstat, pval;
	student_t_test(nn, HtH_inv, Hty, rss, jj, tstat, pval);
	return pval;
}

template <typename Derived1, typename Derived2>
void hetero_chi_sq(const Eigen::MatrixBase<Derived1>& HtH_inv,
                      const Eigen::MatrixBase<Derived2>& Hty,
                      const Eigen::MatrixBase<Derived1>& HtVH,
                      int jj,
                      double& chi_stat,
                      double& pval){
	/* Standard errors adjusted for Heteroscedasticity
	https://en.wikipedia.org/wiki/Heteroscedasticity-consistent_standard_errors
	HtVH = (H.transpose() * resid_sq.asDiagonal() * H)
	*/
	int pp = HtH_inv.rows();
	assert(jj <= pp);

	auto beta = HtH_inv * Hty;
	auto var_beta = HtH_inv * HtVH * HtH_inv;
	chi_stat = beta(jj, 0) * beta(jj, 0);
	chi_stat /= var_beta(jj, jj);

	boost_m::chi_squared chi_dist(1);
	pval = boost_m::cdf(boost_m::complement(chi_dist, chi_stat));
}

template <typename Derived1, typename Derived2>
double hetero_chi_sq(const Eigen::MatrixBase<Derived1>& HtH_inv,
                      const Eigen::MatrixBase<Derived2>& Hty,
                      const Eigen::MatrixBase<Derived1>& HtVH,
                      int jj){
	double tstat, pval;
	hetero_chi_sq(HtH_inv, Hty, HtVH, jj, tstat, pval);
	return pval;
}

#endif
