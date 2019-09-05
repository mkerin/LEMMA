//
// Created by kerin on 2019-09-01.
//
#include "nelder_mead.hpp"
#include "parameters.hpp"

#include "tools/eigen3.3/Dense"

#include <algorithm>
#include <functional>
#include <vector>
#include <utility>

Eigen::VectorXd unit_vec(const size_t j, const size_t n) {
	Eigen::VectorXd ret =  Eigen::VectorXd::Zero(n);
	ret(j) = 1;
	return ret;
}

long get_index_min(Eigen::Ref<Eigen::VectorXd> vec){
	long res = 0;
	double best = 100000000000000;
	for (long ii = 0; ii < vec.rows(); ii++) {
		if(vec[ii] < best) {
			res = ii;
		}
	}
	return res;
}

bool sortbyfirst(const std::pair<double, long> &a,
                 const std::pair<double, long> &b){
	return (a.first < b.first);
}

bool optimNelderMead(Eigen::Ref<Eigen::VectorXd> init_out_vals,
                     std::function<double(Eigen::VectorXd vals_inp, void *grad_out)> opt_objfn,
                     parameters p,
                     const long& iter_max) {
	bool success = false;

	const long n_vals = init_out_vals.rows();
	Eigen::VectorXd simplex_fn_vals(n_vals+1);
	Eigen::MatrixXd simplex_points(n_vals+1,n_vals);

	// Create simplex
	setupNelderMead(init_out_vals, opt_objfn, simplex_points, simplex_fn_vals);

	//
	// begin loop
	std::cout << "\nNelder-Mead: beginning search..." << std::endl;
	std::cout << "  - Initialization Phase:" << std::endl;
	std::cout << "    Objective function value at each vertex:" << std::endl << simplex_fn_vals.transpose() << std::endl;
	std::cout << "    Simplex matrix:" << std::endl << simplex_points << std::endl;

	const double err_tol = 1E-08;
	long iter = 0;
	double err = 2*err_tol;
	double min_val = simplex_fn_vals.minCoeff();

	while (err > err_tol && iter < iter_max) {
		iter++;

		iterNelderMead(simplex_points, simplex_fn_vals, opt_objfn, p);

		// check change in fn_val
		for (size_t i=0; i < n_vals + 1; i++) {
			simplex_fn_vals(i) = opt_objfn(simplex_points.row(i).transpose(),nullptr);
		}

		err = std::abs(min_val - simplex_fn_vals.maxCoeff());
		min_val = simplex_fn_vals.minCoeff();

		long index_min = get_index_min(simplex_fn_vals);
		init_out_vals = simplex_points.row(index_min);

		// printing
		std::cout << "  - Iteration: " << iter << std::endl;
		std::cout << "    min_val:   " << min_val << std::endl;

		std::cout << "    Current optimal input values:" << std::endl;
		std::cout << simplex_points.row(index_min) << std::endl;

		std::cout << "    Objective function value at each vertex:" << std::endl << simplex_fn_vals.transpose() << std::endl;
		std::cout << "    Simplex matrix:" << std::endl << simplex_points << std::endl;
	}

	return success;
}


void setupNelderMead(Eigen::VectorXd init_out_vals,
                     std::function<double(Eigen::VectorXd vals_inp, void *grad_out)> opt_objfn,
                     Eigen::MatrixXd& simplex_points,
                     Eigen::VectorXd& simplex_fn_vals){
	// Create simplex
	const long n_vals = init_out_vals.rows();
	simplex_fn_vals(0) = opt_objfn(init_out_vals,nullptr);
	simplex_points.row(0) = init_out_vals;

	double setup_stepsize = 0.05;
	for (size_t i=1; i < n_vals + 1; i++) {
		simplex_points.row(i) = init_out_vals;
		if (init_out_vals(i-1) != 0.0) {
			simplex_points.row(i) += setup_stepsize * init_out_vals(i-1) * unit_vec(i-1,n_vals);
		} else {
			simplex_points.row(i) += setup_stepsize * setup_stepsize * unit_vec(i-1,n_vals);
		}

		simplex_fn_vals(i) = opt_objfn(simplex_points.row(i),nullptr);
	}
}


void iterNelderMead(Eigen::Ref<Eigen::MatrixXd> simplex_points,
                    Eigen::Ref<Eigen::VectorXd> simplex_fn_vals,
                    std::function<double(Eigen::VectorXd vals_inp, void *grad_out)> opt_objfn,
                    parameters p){
	const long n_vals = simplex_points.cols();
	bool next_iter = false;

	// expansion / contraction parameters
	const double par_alpha = 1.0;
	const double par_beta  = 0.75 - 1.0 / (2.0*n_vals);
	const double par_gamma = 1.0 + 2.0 / n_vals;
	const double par_delta = 1.0 - 1.0 / n_vals;
	// double par_alpha = 1.0;     // reflection parameter
	// double par_beta  = 0.5;     // contraction parameter
	// double par_gamma = 2.0;     // expansion parameter
	// double par_delta = 0.5;     // shrinkage parameter


	if(p.mode_debug) {
		std::cout << "Sorting function values (lowest to highest)" << std::endl;
		std::cout << simplex_fn_vals.transpose() << std::endl;
	}
	std::vector<std::pair<double, long> > sorter;
	for (long ii = 0; ii < n_vals + 1; ii++) {
		sorter.emplace_back(std::make_pair(simplex_fn_vals(ii), ii));
	}
	std::sort(sorter.begin(), sorter.end(), sortbyfirst);

	Eigen::VectorXd simplex_fn_vals_bak = simplex_fn_vals;
	Eigen::MatrixXd simplex_points_bak = simplex_points;
	for (long ii = 0; ii < n_vals + 1; ii++) {
		simplex_fn_vals(ii) = simplex_fn_vals_bak(sorter[ii].second);
		simplex_points.row(ii) = simplex_points_bak.row(sorter[ii].second);
	}
	if(p.mode_debug) {
		std::cout << simplex_fn_vals.transpose() << std::endl;
	}

	// step 2
	Eigen::VectorXd centroid = simplex_points.block(0, 0, n_vals, n_vals).colwise().mean();

	Eigen::VectorXd x_r = centroid;
	x_r += par_alpha * (centroid - simplex_points.row(n_vals).transpose());

	double f_r = opt_objfn(x_r,nullptr);

	if (f_r >= simplex_fn_vals(0) && f_r < simplex_fn_vals(n_vals-1)) {
		// reflected point is neither best nor worst in the new simplex
		simplex_points.row(n_vals) = x_r.transpose();
		next_iter = true;
	}

	// step 3

	if (!next_iter && f_r < simplex_fn_vals(0)) {
		// reflected point is better than the current best; try to go farther along this direction
		Eigen::VectorXd x_e = centroid + par_gamma*(x_r - centroid);
		double f_e = opt_objfn(x_e,nullptr);

		if (f_e < f_r) {
			simplex_points.row(n_vals) = x_e.transpose();
		} else {
			simplex_points.row(n_vals) = x_r.transpose();
		}

		next_iter = true;
	}

	// steps 4, 5, 6

	if (!next_iter && f_r >= simplex_fn_vals(n_vals-1)) {
		// reflected point is still worse than x_n; contract

		// steps 4 and 5
		if (f_r < simplex_fn_vals(n_vals)) {
			// outside contraction
			Eigen::VectorXd x_oc = centroid + par_beta*(x_r - centroid);

			double f_oc = opt_objfn(x_oc,nullptr);

			if (f_oc <= f_r) {
				simplex_points.row(n_vals) = x_oc.transpose();
				next_iter = true;
			}
		} else{
			// inside contraction: f_r >= simplex_fn_vals(n_vals)

			// x_ic = centroid - par_beta*(x_r - centroid);
			Eigen::VectorXd x_ic = centroid + par_beta*(simplex_points.row(n_vals).transpose() - centroid);

			double f_ic = opt_objfn(x_ic,nullptr);

			if (f_ic < simplex_fn_vals(n_vals)) {
				simplex_points.row(n_vals) = x_ic.transpose();
				next_iter = true;
			}
		}
	}

	// step 6
	if (!next_iter) {
		// neither outside nor inside contraction was acceptable; shrink the simplex toward x(0)
		for (size_t i=1; i < n_vals + 1; i++) {
			simplex_points.row(i) = simplex_points.row(0) + par_delta*(simplex_points.row(i) - simplex_points.row(0));
		}
	}
}
