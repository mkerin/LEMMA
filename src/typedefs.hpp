//
// Created by kerin on 2019-03-01.
//

#ifndef LEMMA_TYPEDEFS_HPP
#define LEMMA_TYPEDEFS_HPP

#include "tools/eigen3.3/Dense"

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

#endif //LEMMA_TYPEDEFS_HPP
