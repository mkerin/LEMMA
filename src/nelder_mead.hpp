//
// Created by kerin on 2019-09-01.
//

#ifndef LEMMA_NELDER_MEAD_HPP
#define LEMMA_NELDER_MEAD_HPP

/*################################################################################
##
##   Copyright (C) 2016-2018 Keith O'Hara
##
##   This file is part of the OptimLib C++ library.
##
##   Licensed under the Apache License, Version 2.0 (the "License");
##   you may not use this file except in compliance with the License.
##   You may obtain a copy of the License at
##
##       http://www.apache.org/licenses/LICENSE-2.0
##
##   Unless required by applicable law or agreed to in writing, software
##   distributed under the License is distributed on an "AS IS" BASIS,
##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##   See the License for the specific language governing permissions and
##   limitations under the License.
##
################################################################################*/
#include "parameters.hpp"

#include "tools/eigen3.3/Dense"

/*
 * Nelder-Mead
 */

bool optimNelderMead(Eigen::Ref<Eigen::VectorXd> init_out_vals,
                     std::function<double (Eigen::VectorXd parameters, void* grad_out)> opt_objfn,
                     parameters p,
                     const long& iter_max);

void setupNelderMead(Eigen::VectorXd init_out_vals,
                     std::function<double(Eigen::VectorXd vals_inp, void *grad_out)> opt_objfn,
                     Eigen::MatrixXd& simplex_points,
                     Eigen::VectorXd& simplex_fn_vals);

void iterNelderMead(Eigen::Ref<Eigen::MatrixXd> simplex_points,
                    Eigen::Ref<Eigen::VectorXd> simplex_fn_vals,
                    std::function<double(Eigen::VectorXd vals_inp, void *grad_out)> opt_objfn,
                    parameters p);

long get_index_min(Eigen::Ref<Eigen::VectorXd> vec);

#endif //LEMMA_NELDER_MEAD_HPP
