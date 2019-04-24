//
// Created by kerin on 2019-04-16.
//

#include "../src/Prior.hpp"
#include "../src/eigen_utils.hpp"

#include "catch.hpp"

#include <string>
#include <vector>
#include <functional>

TEST_CASE("Priors"){
//	SECTION("Resolve virtual functions at runtime"){
//		std::vector<std::reference_wrapper<Prior> > vec;
//
//		Gaussian gg, gg1;
//		gg1.set_mean_var(1, 1);
//		CHECK(gg.mean() == 0);
//		CHECK(gg.var() == 1);
//		CHECK(gg.kl_div(1) == 0);
//
//		vec.push_back(gg);
//		CHECK(vec[0].get().mean() == 0);
//		CHECK(vec[0].get().var() == 1);
//		CHECK(vec[0].get().kl_div(1) == 0);
//
//		vec.push_back(gg1);
//		CHECK(vec[1].get().mean() == 1);
//		CHECK(vec[1].get().var() == 1);
//		CHECK(vec[1].get().kl_div(1) == -0.5);
//	}
//
//	SECTION("Set mean var for vector"){
//		std::vector<Gaussian> vec(10);
//
//		for (auto &vv : vec) {
//			vv.set_mean_var(1, 2);
//		}
//		CHECK(vec[0].var() == 2.0);
//	}

	SECTION("Vectorised mixture of gaussians"){
		MoGaussianVec mog_vec(2);
		SECTION("Check header"){
			std::string q1 = mog_vec.header("");
			std::string a1 ("alpha mu1 s1 mu2 s2");
			CHECK(q1 == a1);

			std::string q2 = mog_vec.header("gam");
			std::string a2 ("alpha_gam mu1_gam s1_gam mu2_gam s2_gam");
			CHECK(q2 == a2);
		}

		SECTION("Write to / read from file"){
			// Create output file
			std::string filename ("data/io_test/test_mog_vec.txt");
			boost_io::filtering_ostream outf;
			EigenUtils::fstream_init(outf, filename, false);

			// Change mog from default
			mog_vec.mix(0) = 0.9;
			mog_vec.slab_nats(0, 0) = 10;
			CHECK(mog_vec.slab_nats(0, 0) == 10);

			// Write to file
			outf << mog_vec.header("") << std::endl;
			for (long ii = 0; ii < mog_vec.size(); ii++) {
				mog_vec.write_ith_distn_to_stream(ii, outf);
				outf << std::endl;
			}

			// Read from file
			Eigen::MatrixXd grid;
			std::vector<std::string> cols;
			EigenUtils::read_matrix(filename, grid, cols);
			CHECK(grid(0, 1) == Approx(0.1));
			MoGaussianVec mog_vec2(2);
			mog_vec2.read_from_grid(grid);

			// Check same
			CHECK(mog_vec.mean(0) == mog_vec2.mean(0));
		}

		// // Get i'th distribution
		// MoGaussian distn1 = mog_vec.get_ith_distn(0);
		//
		// CHECK(mog_vec.mean(0) == 0);
		//
		// distn1.set_mix_mean_var(1, 0.5, 1.0, 0, 1.0);
		// CHECK(distn1.slab.mean() == 0.5);
		//
		// // Set i'th distribution
		// mog_vec.set_ith_distn(0, distn1);
		// CHECK(mog_vec.mean(0) == 0.5);


	}
}
