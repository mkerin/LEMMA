//
// Created by kerin on 2019-12-01.
//
#include "mpi_utils.hpp"
#include "typedefs.hpp"

#include "tools/eigen3.3/Dense"
#include "tools/eigen3.3/Eigenvalues"

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/complement.hpp> // complements

#include <cmath>

namespace boost_m  = boost::math;

void prep_lm(const Eigen::MatrixXd &H,
             const Eigen::MatrixXd &y,
             EigenRefDataMatrix HtH,
             EigenRefDataMatrix HtH_inv,
             EigenRefDataMatrix Hty,
             double &rss,
             EigenRefDataMatrix HtVH) {
	/*** All of the heavy lifting for linear hypothesis tests.
	 * Easier to have in one place if we go down the MPI route.
	 */

	HtH     = H.transpose() * H;
	HtH     = mpiUtils::mpiReduce_inplace(HtH);
	Hty     = H.transpose() * y;
	Hty     = mpiUtils::mpiReduce_inplace(Hty);
	HtH_inv = HtH.inverse();

	EigenDataVector resid = y - H * HtH_inv * Hty;
	HtVH = H.transpose() * resid.cwiseProduct(resid).asDiagonal() * H;
	HtVH = mpiUtils::mpiReduce_inplace(HtVH);

	rss = resid.squaredNorm();
	rss = mpiUtils::mpiReduce_inplace(&rss);
}

void prep_lm(const Eigen::MatrixXd &H,
             const Eigen::MatrixXd &y,
             EigenRefDataMatrix HtH,
             EigenRefDataMatrix HtH_inv,
             EigenRefDataMatrix Hty,
             double &rss) {
	/*** All of the heavy lifting for linear hypothesis tests.
	 * Easier to have in one place if we go down the MPI route.
	 */

	HtH     = H.transpose() * H;
	HtH     = mpiUtils::mpiReduce_inplace(HtH);
	Hty     = H.transpose() * y;
	Hty     = mpiUtils::mpiReduce_inplace(Hty);
	HtH_inv = HtH.inverse();

	EigenDataVector resid = y - H * HtH_inv * Hty;
	rss = resid.squaredNorm();
	rss = mpiUtils::mpiReduce_inplace(&rss);
}

void student_t_test(long nn,
                    const Eigen::MatrixXd &HtH_inv,
                    const Eigen::MatrixXd &Hty,
                    double rss,
                    int jj,
                    double &tstat,
                    double &pval) {
	/* 2-sided Student t-test on regression output
	   H0: beta[jj] != 0
	 */
	long pp = HtH_inv.rows();
	assert(jj <= pp);
	nn = mpiUtils::mpiReduce_inplace(&nn);

	auto beta = HtH_inv * Hty;
	tstat = beta(jj, 0);
	tstat /= std::sqrt(rss * HtH_inv(jj, jj) / (double) (nn - pp));

	boost_m::students_t t_dist(nn - pp);
	pval  = 2 * boost_m::cdf(boost_m::complement(t_dist, fabs(tstat)));
}

void hetero_chi_sq(const Eigen::MatrixXd &HtH_inv,
                   const Eigen::MatrixXd &Hty,
                   const Eigen::MatrixXd &HtVH,
                   int jj,
                   double &chi_stat,
                   double &pval) {
	/* Standard errors adjusted for Heteroscedasticity
	   https://en.wikipedia.org/wiki/Heteroscedasticity-consistent_standard_errors
	   HtVH = (H.transpose() * resid_sq.asDiagonal() * H)
	 */
	long pp = HtH_inv.rows();
	assert(jj <= pp);

	auto beta = HtH_inv * Hty;
	auto var_beta = HtH_inv * HtVH * HtH_inv;
	chi_stat = beta(jj, 0) * beta(jj, 0);
	chi_stat /= var_beta(jj, jj);

	boost_m::chi_squared chi_dist(1);
	pval = boost_m::cdf(boost_m::complement(chi_dist, chi_stat));
}

void homo_chi_sq(long nn,
                 const Eigen::MatrixXd &HtH_inv,
                 const Eigen::MatrixXd &Hty,
                 const double rss,
                 const int jj,
                 double &chi_stat,
                 double &pval) {
	/* Essentially the square of the t-test from regression
	 */
	long pp = HtH_inv.rows();
	assert(jj <= pp);
	nn = mpiUtils::mpiReduce_inplace(&nn);

	auto beta = HtH_inv * Hty;
	chi_stat = beta(jj, 0) * beta(jj, 0);
	chi_stat /= rss * HtH_inv(jj, jj) / (double) (nn - pp);

	boost_m::chi_squared chi_dist(1);
	pval = boost_m::cdf(boost_m::complement(chi_dist, chi_stat));
}

void computeSingleSnpTests(EigenRefDataMatrix Xtest,
                           EigenRefDataMatrix neglogPvals,
                           EigenRefDataMatrix chiSqStats,
                           EigenRefDataMatrix pheno_resid) {
	/* Computes
	 * main effects chisq
	 */
	assert(neglogPvals.rows() == Xtest.cols());
	assert(chiSqStats.rows() == Xtest.cols());
	assert(neglogPvals.cols() == 1);
	assert(chiSqStats.cols() == 1);

	long n_samples = pheno_resid.rows();
	long n_var = Xtest.cols();
	long n_effects = 1;

	// Compute p-vals per variant (p=3 as residuals mean centered)
	EigenDataMatrix H(n_samples, n_effects);
	EigenDataMatrix Hty(n_effects, 1);
	EigenDataMatrix HtH(n_effects, n_effects);
	EigenDataMatrix HtH_inv(n_effects, n_effects);
	EigenDataMatrix HtVH(n_effects, n_effects);
	for(std::uint32_t jj = 0; jj < n_var; jj++ ) {
		H.col(0) = Xtest.col(jj);

		// Single-var tests
		double rss_alt, rss_null;
		double beta_stat, gam_stat, rgam_stat, beta_pval, gam_pval, rgam_pval;
		prep_lm(H, pheno_resid, HtH, HtH_inv, Hty, rss_alt, HtVH);
		homo_chi_sq(n_samples, HtH_inv, Hty, rss_alt, 0, beta_stat, beta_pval);

		neglogPvals(jj, 0) = -1 * std::log10(beta_pval);

		chiSqStats(jj, 0) = beta_stat;
	}
}

void computeSingleSnpTests(EigenRefDataMatrix Xtest,
                           EigenRefDataMatrix neglogPvals,
                           EigenRefDataMatrix chiSqStats,
                           EigenRefDataMatrix pheno_resid,
                           EigenRefDataVector eta) {
	/* Computes
	 * main effects chisq
	 * gxe chisq (homoskedastic)
	 * gxe chisq (heteroskedastic)
	 */
	assert(neglogPvals.rows() == Xtest.cols());
	assert(chiSqStats.rows() == Xtest.cols());
	assert(neglogPvals.cols() == 3);
	assert(chiSqStats.cols() == 3);

	long n_samples = pheno_resid.rows();
	long n_var = Xtest.cols();
	long n_effects = 2;

	// Compute p-vals per variant (p=3 as residuals mean centered)
	EigenDataMatrix H(n_samples, n_effects);
	EigenDataMatrix Hty(n_effects, 1);
	EigenDataMatrix HtH(n_effects, n_effects);
	EigenDataMatrix HtH_inv(n_effects, n_effects);
	EigenDataMatrix HtVH(n_effects, n_effects);
	for(std::uint32_t jj = 0; jj < n_var; jj++ ) {
		H.col(0) = Xtest.col(jj);
		H.col(1) = H.col(0).cwiseProduct(eta);

		// Single-var tests
		double rss_alt, rss_null;
		double beta_stat, gam_stat, rgam_stat, beta_pval, gam_pval, rgam_pval;
		prep_lm(H, pheno_resid, HtH, HtH_inv, Hty, rss_alt, HtVH);
		homo_chi_sq(n_samples, HtH_inv, Hty, rss_alt, 0, beta_stat, beta_pval);
		homo_chi_sq(n_samples, HtH_inv, Hty, rss_alt, 1, gam_stat, gam_pval);
		hetero_chi_sq(HtH_inv, Hty, HtVH, 1, rgam_stat, rgam_pval);

		neglogPvals(jj, 0) = -1 * std::log10(beta_pval);
		neglogPvals(jj, 1) = -1 * std::log10(gam_pval);
		neglogPvals(jj, 2) = -1 * std::log10(rgam_pval);

		chiSqStats(jj, 0) = beta_stat;
		chiSqStats(jj, 1) = gam_stat;
		chiSqStats(jj, 2) = rgam_stat;
	}
}

double homo_chi_sq(const long nn,
                   const Eigen::MatrixXd &HtH_inv,
                   const Eigen::MatrixXd &Hty,
                   const double rss,
                   const int jj) {
	double tstat, pval;
	homo_chi_sq(nn, HtH_inv, Hty, rss, jj, tstat, pval);
	return pval;
}

double hetero_chi_sq(const Eigen::MatrixXd &HtH_inv,
                     const Eigen::MatrixXd &Hty,
                     const Eigen::MatrixXd &HtVH,
                     int jj) {
	double tstat, pval;
	hetero_chi_sq(HtH_inv, Hty, HtVH, jj, tstat, pval);
	return pval;
}

double student_t_test(long nn,
                      const Eigen::MatrixXd &HtH_inv,
                      const Eigen::MatrixXd &Hty,
                      double rss,
                      int jj) {
	double tstat, pval;
	student_t_test(nn, HtH_inv, Hty, rss, jj, tstat, pval);
	return pval;
}
