// parse_arguments
#ifndef STATS_TESTS_HPP
#define STATS_TESTS_HPP
#define CATCH_CONFIG_MAIN

#include "typedefs.hpp"
#include "tools/eigen3.3/Dense"

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
                    double& tstat,
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
                   double& chi_stat,
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
                 double& chi_stat,
                 double& pval);

double homo_chi_sq(const long nn,
                   const Eigen::MatrixXd& HtH_inv,
                   const Eigen::MatrixXd& Hty,
                   const double rss,
                   const int jj);

void computeSingleSnpTests(EigenRefDataMatrix Xtest,
                           EigenRefDataMatrix neglogPvals,
                           EigenRefDataMatrix chiSqStats,
                           EigenRefDataMatrix pheno_resid);

void computeSingleSnpTests(EigenRefDataMatrix Xtest,
                           EigenRefDataMatrix neglogPvals,
                           EigenRefDataMatrix chiSqStats,
                           EigenRefDataMatrix pheno_resid,
                           EigenRefDataVector eta);

#endif
