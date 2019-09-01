//
// Created by kerin on 2019-09-01.
//
#define EIGEN_USE_MKL_ALL
#include "catch.hpp"

#include "../src/nelder_mead.hpp"

double example_surface_1(Eigen::VectorXd vals_inp, void* grad_out) {
	// https://www.benfrederickson.com/numerical-optimization/
	const double x = vals_inp(0);
	const double y = vals_inp(1);

	double obj_val = x * x + y * y + x * std::sin(y) + y * std::sin(x);
	return obj_val;
}

double example_surface_2(Eigen::VectorXd vals_inp, void* grad_out) {
	// https://www.benfrederickson.com/numerical-optimization/
	const double x = vals_inp(0);
	const double y = vals_inp(1);

	double obj_val = std::pow(x * x + y - 11, 2) + std::pow(x + y * y - 7, 2);
	return obj_val;
}


TEST_CASE("NelderMead") {
	// https://codesachin.wordpress.com/2016/01/16/nelder-mead-optimization/
	// https://www.benfrederickson.com/numerical-optimization/
	SECTION("Example 1: min at zero") {
		parameters p;
		Eigen::VectorXd x(2);
		x(0) = -4;
		x(1) = 1;
		bool success = optimNelderMead(x, example_surface_1, p);
		CHECK(x(0) == Approx(-0.0050957943));
		CHECK(x(1) == Approx(0.0051339533));
	}

	SECTION("Example 2: multi-modal") {
		parameters p;
		Eigen::VectorXd x(2);
		x(0) = 1;
		x(1) = 1;
		bool success = optimNelderMead(x, example_surface_2, p);
		CHECK(x(0) == Approx(3));
		CHECK(x(1) == Approx(2));
	}
}
