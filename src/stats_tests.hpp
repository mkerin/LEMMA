// parse_arguments
#ifndef STATS_TESTS_HPP
#define STATS_TESTS_HPP

#include "typedefs.hpp"
#include "tools/eigen3.3/Dense"
#include "variational_parameters.hpp"

void prep_lm(const Eigen::MatrixXd& H,
             const Eigen::MatrixXd& y,
             EigenRefDataMatrix HtH,
             EigenRefDataMatrix HtH_inv,
             EigenRefDataMatrix Hty,
             double& rss,
             EigenRefDataMatrix HtVH);

void prep_lm(const Eigen::MatrixXd& H,
             const Eigen::MatrixXd& y,
             EigenRefDataMatrix HtH,
             EigenRefDataMatrix HtH_inv,
             EigenRefDataMatrix Hty,
             double& rss);

void student_t_test(long nn,
                    const Eigen::MatrixXd& HtH_inv,
                    const Eigen::MatrixXd& Hty,
                    double rss,
                    int jj,
                    double& stat,
                    double& pval);

double student_t_test(long nn,
                      const Eigen::MatrixXd& HtH_inv,
                      const Eigen::MatrixXd& Hty,
                      double rss,
                      int jj);

void hetero_chi_sq(const Eigen::MatrixXd& HtH_inv,
                   const Eigen::MatrixXd& Hty,
                   const Eigen::MatrixXd& HtVH,
                   int jj,
                   double& stat,
                   double& pval);

double hetero_chi_sq(const Eigen::MatrixXd& HtH_inv,
                     const Eigen::MatrixXd& Hty,
                     const Eigen::MatrixXd& HtVH,
                     int jj);

void homo_chi_sq(long nn,
                 const Eigen::MatrixXd& HtH_inv,
                 const Eigen::MatrixXd& Hty,
                 const double rss,
                 const int jj,
                 double& stat,
                 double& pval);

double homo_chi_sq(const long nn,
                   const Eigen::MatrixXd& HtH_inv,
                   const Eigen::MatrixXd& Hty,
                   const double rss,
                   const int jj);

template <typename GenoMat>
void compute_LOCO_pvals(const EigenDataVector& resid_pheno,
                        const GenoMat& Xtest,
                        const VariationalParametersLite& vp,
                        Eigen::MatrixXd& neglogPvals,
                        Eigen::MatrixXd& testStats);

#endif
