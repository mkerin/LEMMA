//
// Created by kerin on 12/11/2018.
//

// tests-main.cpp
#include <iostream>
#include "../src/tools/Eigen/Dense"
#include "catch.hpp"

TEST_CASE("Algebra in Eigen3") {

	Eigen::MatrixXd X(3, 3), X2;
	Eigen::VectorXd v1(3), v2(3);
	X << 1, 2, 3,
			4, 5, 6,
			7, 8, 9;
	v1 << 1, 1, 1;
	v2 << 1, 2, 3;
	X2 = X.rowwise().reverse();

	SECTION("dot product of vector with col vector"){
		CHECK((v1.dot(X.col(0))) == 12.0);
	}

	SECTION("Eigen reverses columns as expected"){
		Eigen::MatrixXd res(3, 3);
		res << 3, 2, 1,
				6, 5, 4,
				9, 8, 7;
		CHECK(X2 == res);
	}

	SECTION("coefficient-wise product between vectors"){
		Eigen::VectorXd res(3);
		res << 1, 2, 3;
		CHECK((v1.array() * v2.array()).matrix() == res);
		CHECK(v1.cwiseProduct(v2) == res);
	}

	SECTION("coefficient-wise subtraction between vectors"){
		Eigen::VectorXd res(3);
		res << 0, 1, 2;
		CHECK((v2 - v1) == res);
	}

	SECTION("Check .sum() function"){
		Eigen::VectorXd res(3);
		res << 1, 2, 3;
		CHECK(res.sum() == 6);
	}

	SECTION("Sum of NaN returns NaN"){
		Eigen::VectorXd res(3);
		res << 1, std::numeric_limits<double>::quiet_NaN(), 3;
		CHECK(std::isnan(res.sum()));
	}

	SECTION("Ref of columns working correctly"){
		Eigen::Ref<Eigen::VectorXd> y1 = X.col(0);
		CHECK(y1(0) == 1);
		CHECK(y1(1) == 4);
		CHECK(y1(2) == 7);
		X = X + X;
		CHECK(y1(0) == 2);
		CHECK(y1(1) == 8);
		CHECK(y1(2) == 14);
	}

	SECTION("Refs still work after resizing underlying object"){
		// This is not possible
		Eigen::MatrixXd tmp = Eigen::MatrixXd::Random(3, 3);
		Eigen::Ref<Eigen::VectorXd> ym = tmp.col(0);

		// CHECK(ym.squaredNorm() == ym.dot(tmp.col(0)));
		// CHECK(ym.rows() == 3);
		// CHECK(ym(2) == 3);

		// tmp.conservativeResize(2, 2);
		// CHECK(tmp.rows() == 2);
		// CHECK(ym.rows() == 2);
	}

	SECTION("Check Eigen flexi indexing"){
		int N = 4;
		int P = 3;
		int L = 2;

		Eigen::MatrixXd M = Eigen::MatrixXd::Random(N, P);
		Eigen::MatrixXd lhs_m = Eigen::MatrixXd::Random(N, L);
		Eigen::VectorXd lhs_v = Eigen::VectorXd::Random(N);
		Eigen::VectorXd lhs2_v = Eigen::VectorXd::Random(N / 2);
		Eigen::VectorXd dd = Eigen::VectorXd::Random(N);
		Eigen::MatrixXd rhs_m = Eigen::MatrixXd::Random(P, L);
		Eigen::VectorXd rhs_v = Eigen::VectorXd::Random(P);


		Eigen::Array<long, Eigen::Dynamic, 1> eii(4); eii << 3, 1, 6, 5;

		Eigen::MatrixXd res_m;
		Eigen::VectorXd res_v;


		res_m = M(Eigen::seq(0,Eigen::last,2), Eigen::all).template cast<double>() * rhs_v.asDiagonal();
		res_m = M(Eigen::seq(0,Eigen::last,2), Eigen::all).template cast<double>() * rhs_v.asDiagonal();



		res_m = lhs_v.transpose() * M.cast<double>();
		res_m = lhs2_v.transpose() * M(Eigen::seq(0,Eigen::last,2), ":").template cast<double>();
		res_m = M(Eigen::seq(0,Eigen::last,2), Eigen::all).template cast<double>() * rhs_v.asDiagonal();
		res_m = lhs_v(eii).transpose() * M(eii, Eigen::all).template cast<double>();
	}

	SECTION("Conservative Resize"){
		std::vector<int> keep;
		keep.push_back(1);
		for (std::size_t i = 0; i < keep.size(); i++) {
			X.col(i) = X.col(keep[i]);
		}
		X.conservativeResize(X.rows(), keep.size());

		CHECK(X.rows() == 3);
		CHECK(X.cols() == 1);
		CHECK(X(0, 0) == 2);
	}

	SECTION("selfAdjoit views"){
		Eigen::MatrixXd m3(3, 3);
		m3.triangularView<Eigen::StrictlyUpper>() = X.transpose() * X;
		CHECK(m3(0, 1) == 78);
	}

	SECTION("colwise subtraction between vector and matrix"){
		Eigen::MatrixXd res;
		res = -1*(X.colwise() - v1);
		CHECK(res(0, 0) == 0);
		CHECK(res.rows() == 3);
		CHECK(res.cols() == 3);
	}
}
