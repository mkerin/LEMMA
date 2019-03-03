//
// Created by kerin on 2019-03-02.
//

// tests-main.cpp
#include "../src/hyps.hpp"

#include "catch.hpp"

#include <vector>

TEST_CASE("Hyps"){
	SECTION("Copy constructor in vectors"){

		parameters p;
		Hyps hyps(p);
		hyps.sigma = 1;
		hyps.pve.resize(1);
		hyps.pve(0) = 1;

		std::vector<Hyps> all_hyps(2, hyps);
		std::vector<Hyps> theta0 = all_hyps;

		theta0[0].sigma = 2;
		theta0[0].pve(0) = 2;

		CHECK(all_hyps[0].sigma == 1);
		CHECK(all_hyps[0].pve(0) == 1);
	}
}
