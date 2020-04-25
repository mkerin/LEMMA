//
// Created by kerin on 2019-12-01.
//
#include "mpi_utils.hpp"
#include "typedefs.hpp"
#include "variational_parameters.hpp"

#include "tools/eigen3.3/Dense"
#include "tools/eigen3.3/Eigenvalues"

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/complement.hpp>
#include <boost/math/distributions/fisher_f.hpp>

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
                    double &stat,
                    double &pval) {
	/* 2-sided Student t-test on regression output
	   H0: beta[jj] != 0
	 */
	long pp = HtH_inv.rows();
	assert(jj <= pp);
	nn = mpiUtils::mpiReduce_inplace(&nn);

	auto beta = HtH_inv * Hty;
	stat = beta(jj, 0);
	stat /= std::sqrt(rss * HtH_inv(jj, jj) / (double) (nn - pp));
//	if (std::isnan(stat)){
//		std::cout << "est = " << beta(jj, 0) << std::endl;
//		std::cout << "sd(est) = " << std::sqrt(rss * HtH_inv(jj, jj) / (double) (nn - pp)) << std::endl;
//		std::cout << "rss = " << rss << std::endl;
//	}

	boost_m::students_t t_dist(nn - pp);
	pval  = 2 * boost_m::cdf(boost_m::complement(t_dist, fabs(stat)));
}

void hetero_chi_sq(const Eigen::MatrixXd &HtH_inv,
                   const Eigen::MatrixXd &Hty,
                   const Eigen::MatrixXd &HtVH,
                   int jj,
                   double &stat,
                   double &pval) {
	/* Standard errors adjusted for Heteroscedasticity
	   https://en.wikipedia.org/wiki/Heteroscedasticity-consistent_standard_errors
	   HtVH = (H.transpose() * resid_sq.asDiagonal() * H)
	 */
	long pp = HtH_inv.rows();
	assert(jj <= pp);

	auto beta = HtH_inv * Hty;
	auto var_beta = HtH_inv * HtVH * HtH_inv;
	stat = beta(jj, 0) * beta(jj, 0);
	stat /= var_beta(jj, jj);
	stat = std::abs(stat);
//	if (std::isnan(stat)){
//		std::cout << "est_sq = " << beta(jj, 0) * beta(jj, 0) << std::endl;
//		std::cout << "var(est) = " << var_beta(jj, jj) << std::endl;
//	}

	boost_m::chi_squared chi_dist(1);
	pval = boost_m::cdf(boost_m::complement(chi_dist, stat));
}

void homo_chi_sq(long nn,
                 const Eigen::MatrixXd &HtH_inv,
                 const Eigen::MatrixXd &Hty,
                 const double rss,
                 const int jj,
                 double &stat,
                 double &pval) {
	/* Essentially the square of the t-test from regression
	 */
	long pp = HtH_inv.rows();
	assert(jj <= pp);
	nn = mpiUtils::mpiReduce_inplace(&nn);

	auto beta = HtH_inv * Hty;
	stat = beta(jj, 0) * beta(jj, 0);
	stat /= rss * HtH_inv(jj, jj) / (double) (nn - pp);
	stat = std::abs(stat);
//	if (std::isnan(stat)){
//		std::cout << "est_sq = " << beta(jj, 0) * beta(jj, 0) << std::endl;
//		std::cout << "var(est) = " << rss * HtH_inv(jj, jj) / (double) (nn - pp) << std::endl;
//	}

	boost_m::chi_squared chi_dist(1);
	pval = boost_m::cdf(boost_m::complement(chi_dist, stat));
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

template <typename GenoMat>
void compute_LOCO_pvals(const EigenDataVector &resid_pheno,
		const GenoMat &Xtest,
		Eigen::MatrixXd &neglogPvals,
						Eigen::MatrixXd &testStats,
						const EigenDataVector &eta) {
	bool isGxE     = eta.rows() > 0;
	long n_var     = Xtest.cols();
	long n_samples = Xtest.rows();
	long n_effects = (isGxE ? 2 : 1);
	double Nlocal  = n_samples;
	double Nglobal = mpiUtils::mpiReduce_inplace(&Nlocal);

	neglogPvals.resize(n_var, (isGxE ? 4 : 1));
	testStats.resize(n_var, (isGxE ? 4 : 1));

	// Compute p-vals per variant (p=3 as residuals mean centered)
	Eigen::MatrixXd H(n_samples, 2 + 2 * (isGxE ? 1 : 0));
	H.col(0) = Eigen::VectorXd::Constant(n_samples, 1.0);
	if (isGxE) H.col(3) = eta.cast<double>();
	boost_m::students_t t_dist(n_samples - H.cols() - 1);
	boost_m::fisher_f f_dist(n_effects, n_samples - H.cols() - 1);
	for(std::uint32_t jj = 0; jj < n_var; jj++ ) {
		H.col(1) = Xtest.col(jj);

		double rss_alt, rss_null;
		Eigen::MatrixXd HtH(H.cols(), H.cols()), Hty(H.cols(), 1);
		Eigen::MatrixXd HtH_inv(H.cols(), H.cols()), HtVH(H.cols(), H.cols());
		if(!isGxE) {
			double beta_tstat, beta_pval;
			prep_lm(H, resid_pheno, HtH, HtH_inv, Hty, rss_alt);
			student_t_test(n_samples, HtH_inv, Hty, rss_alt, 1, beta_tstat, beta_pval);

			neglogPvals(jj,0) = -1 * log10(beta_pval);
			testStats(jj,0)   = beta_tstat;
		} else {
			H.col(2) = H.col(1).cwiseProduct(eta.cast<double>());
			try {
				// Single-var tests
				double beta_tstat, gam_tstat, rgam_stat, beta_pval, gam_pval, rgam_pval;
				prep_lm(H, resid_pheno, HtH, HtH_inv, Hty, rss_alt, HtVH);
				hetero_chi_sq(HtH_inv, Hty, HtVH, 2, rgam_stat, rgam_pval);
				student_t_test(n_samples, HtH_inv, Hty, rss_alt, 2, gam_tstat, gam_pval);
				student_t_test(n_samples, HtH_inv, Hty, rss_alt, 1, beta_tstat, beta_pval);

				// F-test over main+int effects of snp_j
				double joint_fstat, joint_pval;
				rss_null = resid_pheno.squaredNorm();
				rss_null = mpiUtils::mpiReduce_inplace(&rss_null);
				joint_fstat = (rss_null - rss_alt) / 2.0;
				joint_fstat /= rss_alt / (Nglobal - 3.0);
				joint_pval = 1.0 - boost_m::cdf(f_dist, joint_fstat);

				neglogPvals(jj, 0) = -1 * std::log10(beta_pval);
				neglogPvals(jj, 1) = -1 * std::log10(gam_pval);
				neglogPvals(jj, 2) = -1 * std::log10(rgam_pval);
				neglogPvals(jj, 3) = -1 * std::log10(joint_pval);
				testStats(jj, 0) = beta_tstat;
				testStats(jj, 1) = gam_tstat;
				testStats(jj, 2) = rgam_stat;
				testStats(jj, 3) = joint_fstat;
			} catch (...) {
				neglogPvals(jj, 0) = std::numeric_limits<double>::quiet_NaN();
				neglogPvals(jj, 1) = std::numeric_limits<double>::quiet_NaN();
				neglogPvals(jj, 2) = std::numeric_limits<double>::quiet_NaN();
				neglogPvals(jj, 3) = std::numeric_limits<double>::quiet_NaN();
				testStats(jj, 0) = std::numeric_limits<double>::quiet_NaN();
				testStats(jj, 1) = std::numeric_limits<double>::quiet_NaN();
				testStats(jj, 2) = std::numeric_limits<double>::quiet_NaN();
				testStats(jj, 3) = std::numeric_limits<double>::quiet_NaN();
			}
		}
	}
}

// Explicit instantiation
// https://stackoverflow.com/questions/2152002/how-do-i-force-a-particular-instance-of-a-c-template-to-instantiate
template void compute_LOCO_pvals(const EigenDataVector&, const EigenDataMatrix&,
                                 Eigen::MatrixXd&, Eigen::MatrixXd&,const EigenDataVector&);
template void compute_LOCO_pvals(const EigenDataVector&, const GenotypeMatrix&,
                                 Eigen::MatrixXd&, Eigen::MatrixXd&,const EigenDataVector&);
