//
// Created by kerin on 2019-03-02.
//

// tests-main.cpp
#include "../src/hyps.hpp"

#include "catch.hpp"

#include <vector>

TEST_CASE("Hyps"){
	parameters p;
	SECTION("Copy constructor in vectors"){

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

	SECTION("Default initialisation"){
		Hyps hyps(p);
		Eigen::VectorXd soln;

		soln = hyps.get_sigmas(0.1, 0.01, 0.05, 1000);
		CHECK(soln[0] == Approx(0.0005555556));
		CHECK(soln[1] == Approx(0.0001066218));

		soln = hyps.get_sigmas(0.1, 0, 0.01, 0.01, 0.05, 0.05, 1000);
		CHECK(soln[0] == Approx(0.0005555556));
		CHECK(soln[1] == Approx(0.0001066218));
		CHECK(soln[2] == Approx(0));
		CHECK(soln[3] == Approx(0));

		soln = hyps.get_sigmas(0, 0.1, 0.01, 0.01, 0.05, 0.05, 1000);
		CHECK(soln[0] == Approx(0));
		CHECK(soln[1] == Approx(0));
		CHECK(soln[2] == Approx(0.0005555556));
		CHECK(soln[3] == Approx(0.0001066218));

		hyps.use_default_init(2, 1000);
		CHECK(hyps.slab_relative_var[0] == Approx(0.0005882353));
		CHECK(hyps.spike_relative_var[0] == Approx(0.0001128936));
		CHECK(hyps.slab_relative_var[1] == Approx(0.0002941176));
		CHECK(hyps.spike_relative_var[1] == Approx(0.0000564468));
	}
}
