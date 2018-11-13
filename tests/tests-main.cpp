// tests-main.cpp
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sys/stat.h>
#include "../src/tools/eigen3.3/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/vbayes_x2.hpp"
#include "../src/data.hpp"
#include "../src/hyps.hpp"
#include "../src/genotype_matrix.hpp"


TEST_CASE( "Algebra in Eigen3" ) {

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
}

TEST_CASE( "Example 1: single-env" ){
	parameters p;

	SECTION("Ex1. No filters applied, low mem mode"){
		char* argv[] = { (char*) "bin/bgen_prog", (char*) "--mode_vb", (char*) "--low_mem",
						 (char*) "--interaction", (char*) "x",
						 (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
						 (char*) "--out", (char*) "data/io_test/fake_age.out",
						 (char*) "--pheno", (char*) "data/io_test/pheno.txt",
						 (char*) "--hyps_grid", (char*) "data/io_test/hyperpriors_gxage.txt",
						 (char*) "--hyps_probs", (char*) "data/io_test/hyperpriors_gxage_probs.txt",
						 (char*) "--vb_init", (char*) "data/io_test/answer_init.txt",
						 (char*) "--covar", (char*) "data/io_test/age.txt"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		Data data( p );

		std::cout << "Data initialised" << std::endl;
		data.read_non_genetic_data();
		SECTION( "Ex1. Raw non genetic data read in accurately"){
            CHECK(data.n_covar == 1);
            CHECK(data.n_env == 1);
			CHECK(data.n_pheno == 1);
			CHECK(data.n_samples == 50);
			CHECK(data.Y(0,0) == Approx(-1.18865038973338));
			CHECK(data.W(0,0) == Approx(-0.33472645347487201));
			CHECK(data.E(0, 0) == Approx(-0.33472645347487201));
			CHECK(data.hyps_grid(0,1) == Approx(0.317067781333932));
		}

		data.standardise_non_genetic_data();
		SECTION( "Ex1. Non genetic data standardised + covars regressed"){
			CHECK(data.params.scale_pheno == true);
			CHECK(data.params.use_vb_on_covars == false);
			CHECK(data.params.covar_file != "NULL");
//			CHECK(data.Y(0,0) == Approx(-3.6676363273605137)); Centered
//			CHECK(data.Y(0,0) == Approx(-1.5800573524786081)); Scaled
			CHECK(data.Y(0,0) == Approx(-1.262491384814441));
			CHECK(data.Y2(0,0) == Approx(-1.262491384814441));
			CHECK(data.W(0,0) == Approx(-0.58947939694779772));
			CHECK(data.E(0,0) == Approx(-0.58947939694779772));
		}

		data.read_full_bgen();
		SECTION( "Ex1. bgen read in & standardised correctly"){
			CHECK(data.G.low_mem);
			CHECK(data.params.low_mem);
            CHECK(data.params.flip_high_maf_variants);
			CHECK(data.G(0, 0) == Approx(1.8570984229));
		}

		SECTION( "Ex1. Confirm calc_dxteex() reorders properly"){
		    data.params.dxteex_file = "data/io_test/inputs/dxteex_mixed.txt";
			data.read_external_dxteex();
            data.calc_dxteex();
            CHECK(data.dXtEEX(0, 0) == Approx(87.204591182113916));
            CHECK(data.n_dxteex_computed == 1);
		}

		data.calc_dxteex();
		if(p.vb_init_file != "NULL"){
			data.read_alpha_mu();
		}
		VBayesX2 VB(data);
		VB.check_inputs();
		SECTION("Ex1. Vbayes_X2 initialised correctly"){
			CHECK(VB.n_samples == 50);
			CHECK(VB.N == 50.0);
			CHECK(VB.n_env == 1);
			CHECK(VB.n_covar == 1);
			CHECK(VB.n_effects == 2);
			CHECK(VB.vp_init.muw(0) == 1.0);
			CHECK(!VB.p.init_weights_with_snpwise_scan);
			CHECK(VB.dXtEEX(0, 0) == Approx(87.1907593967));
		}

        SECTION("Ex1. Explicitly checking updates"){
			// Initialisation
			CHECK(VB.vp_init.ym(0) == Approx(0.0003200476));
			CHECK(VB.vp_init.yx(0) == Approx(0.0081544079));
			CHECK(VB.vp_init.eta(0) == Approx(-0.5894793969));

			// Set up for RunInnerLoop
			long n_grid = VB.hyps_grid.rows();
			long n_samples = VB.n_samples;
			std::vector<Hyps> all_hyps;
			VB.unpack_hyps(VB.hyps_grid, all_hyps);

			// Set up for updateAllParams
			std::vector<VariationalParameters> all_vp;
			VB.setup_variational_params(all_hyps, all_vp);
			VariationalParameters& vp = all_vp[0];
			Hyps& hyps = all_hyps[0];

			int round_index = 2;
			std::vector<double> logw_prev(n_grid, -1);
			std::vector<std::vector< double >> logw_updates(n_grid);

			// Ground zero as expected
			CHECK(vp.alpha_beta(0) * vp.mu1_beta(0) == Approx(-0.00015854116408000002));
			CHECK(data.Y(0,0) == Approx(-1.262491384814441));
			CHECK(vp.ym(0) == Approx(0.0003200476));
			CHECK(vp.yx(0) == Approx(0.0081544079));
			CHECK(vp.eta(0) == Approx(-0.5894793969));

			VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev, logw_updates);

			SECTION("A computed correctly"){
				int ee = 0;
				int n_effects = 2;
				int n_grid = 2;
				Eigen::MatrixXd D;
				Eigen::Ref<Eigen::MatrixXd> Y = VB.Y;
				std::vector< std::uint32_t > chunk = VB.fwd_pass_chunks[0];
				unsigned long ch_len   = chunk.size();

				// D is n_samples x snp_batch
				if(D.cols() != ch_len){
					D.resize(n_samples, ch_len);
				}
				VB.X.col_block3(chunk, D);

				CHECK(D(0, 0) == Approx(1.8570984229));

				// Most work done here
				// variant correlations with residuals
				Eigen::MatrixXd residual(n_samples, n_grid);
				if(n_effects == 1){
					// Main effects update in main effects only model
					for(int nn = 0; nn < n_grid; nn++) {
						residual.col(nn) = Y - all_vp[nn].ym;
					}
				} else if (ee == 0){
					// Main effects update in interaction model
					for(int nn = 0; nn < n_grid; nn++){
						residual.col(nn) = Y - all_vp[nn].ym - all_vp[nn].yx.cwiseProduct(all_vp[nn].eta);
					}
				} else {
					// Interaction effects
					for (int nn = 0; nn < n_grid; nn++){
						residual.col(nn) = (Y - all_vp[nn].ym).cwiseProduct(all_vp[nn].eta) - all_vp[nn].yx.cwiseProduct(all_vp[nn].eta_sq);
					}
				}
				Eigen::MatrixXd AA = residual.transpose() * D; // n_grid x snp_batch
				AA.transposeInPlace();                         // convert to snp_batch x n_grid

//				CHECK(residual(0, 0) == Approx(-1.2580045769624202));
//				CHECK(AA(0, 0) == Approx(-9.76793));
			}

			CHECK(VB.X.col(0)(0) == Approx(1.8570984229));
			CHECK(vp.s1_beta_sq(0) == Approx(0.0031087381));
			CHECK(vp.mu1_beta(0) == Approx(-0.0303900712));
			CHECK(vp.alpha_beta(0) == Approx(0.1447783263));
			CHECK(vp.alpha_beta(1) == Approx(0.1517251004));
			CHECK(vp.mu1_beta(1) == Approx(-0.0355760798));
			CHECK(vp.alpha_beta(63) == Approx(0.1784518373));
			CHECK(VB.calc_logw(hyps, vp) == Approx(-60.983398393));

			VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev, logw_updates);

			CHECK(vp.alpha_beta(0) == Approx(0.1350711123));
			CHECK(vp.mu1_beta(0) == Approx(-0.0205395866));
			CHECK(vp.alpha_beta(1) == Approx(0.1400764528));
			CHECK(vp.alpha_beta(63) == Approx(0.1769882239));
			CHECK(VB.calc_logw(hyps, vp) == Approx(-60.606081598));
		}

		std::vector< VbTracker > trackers(VB.hyps_grid.rows());
		VB.run_inference(VB.hyps_grid, false, 2, trackers);
		SECTION("Ex1. Vbayes_X2 inference correct") {
			CHECK(trackers[0].count == 33);
			CHECK(trackers[3].count == 33);
			CHECK(trackers[0].logw == Approx(-60.522210486));
			CHECK(trackers[1].logw == Approx(-59.9696083263));
			CHECK(trackers[2].logw == Approx(-60.30658117));
			CHECK(trackers[3].logw == Approx(-61.0687573393));
		}
	}
}

TEST_CASE( "Example 2: multi-env" ){
	parameters p;

	SECTION("Ex2. No filters applied, high mem mode"){
		char* argv[] = { (char*) "bin/bgen_prog", (char*) "--mode_vb",
						 (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
						 (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
						 (char*) "--out", (char*) "data/io_test/fake_env.out",
						 (char*) "--pheno", (char*) "data/io_test/pheno.txt",
						 (char*) "--hyps_grid", (char*) "data/io_test/hyperpriors_gxage.txt",
						 (char*) "--hyps_probs", (char*) "data/io_test/hyperpriors_gxage_probs.txt",
						 (char*) "--vb_init", (char*) "data/io_test/answer_init.txt",
						 (char*) "--covar", (char*) "data/io_test/n50_p100_env.txt"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		Data data( p );

		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		data.calc_dxteex();
        data.calc_snpstats();
		if(p.vb_init_file != "NULL"){
			data.read_alpha_mu();
		}
		VBayesX2 VB(data);
		VB.check_inputs();
		SECTION("Ex2. Vbayes_X2 initialised correctly"){
			CHECK(VB.n_samples == 50);
			CHECK(VB.N == 50.0);
			CHECK(VB.n_env == 4);
			CHECK(VB.n_covar == 4);
			CHECK(VB.n_effects == 2);
			CHECK(VB.vp_init.muw(0) == 0.25);
			CHECK(VB.p.init_weights_with_snpwise_scan == false);
			CHECK(VB.dXtEEX(0, 0) == Approx(38.9390135703));
		}

		SECTION("Ex2. Explicitly checking updates"){
			// Set up for RunInnerLoop
			long n_grid = VB.hyps_grid.rows();
			long n_samples = VB.n_samples;
			std::vector<Hyps> all_hyps;
			VB.unpack_hyps(VB.hyps_grid, all_hyps);

			// Set up for updateAllParams
			std::vector<VariationalParameters> all_vp;
			VB.setup_variational_params(all_hyps, all_vp);
			VariationalParameters& vp = all_vp[0];
			Hyps& hyps = all_hyps[0];

			int round_index = 2;
			std::vector<double> logw_prev(n_grid, -1);
			std::vector<std::vector< double >> logw_updates(n_grid);

			VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev, logw_updates);

			CHECK(vp.alpha_beta(0) == Approx(0.1339907047));
			CHECK(vp.alpha_beta(1) == Approx(0.1393645403 ));
			CHECK(vp.alpha_beta(63) == Approx(0.1700976171));
			CHECK(vp.muw(0) == Approx(0.1096760209));
			CHECK(VB.calc_logw(hyps, vp) == Approx(-68.2656816517));

			VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev, logw_updates);

			CHECK(vp.alpha_beta(0) == Approx(0.1292192489));
			CHECK(vp.alpha_beta(1) == Approx(0.1326174323));
			CHECK(vp.alpha_beta(63) == Approx(0.1704601589));
			CHECK(vp.muw(0) == Approx(0.0455626691));
			CHECK(VB.calc_logw(hyps, vp) == Approx(-67.6870841008));
		}

		std::vector< VbTracker > trackers(VB.hyps_grid.rows());
		VB.run_inference(VB.hyps_grid, false, 2, trackers);
		SECTION("Ex2. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 35);
			CHECK(trackers[3].count == 35);
			CHECK(trackers[0].logw == Approx(-67.6055600008));
			CHECK(trackers[1].logw == Approx(-67.3497693394));
			CHECK(trackers[2].logw == Approx(-67.757622793));
			CHECK(trackers[3].logw == Approx(-68.5048150566));
		}
	}
}

TEST_CASE( "Example 3: multi-env w/ covars" ){
	parameters p;

	SECTION("Ex3. No filters applied, high mem mode"){
		char* argv[] = { (char*) "bin/bgen_prog", (char*) "--mode_vb",
						 (char*) "--use_vb_on_covars",
						 (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
						 (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
						 (char*) "--out", (char*) "data/io_test/fake_env.out",
						 (char*) "--pheno", (char*) "data/io_test/pheno.txt",
						 (char*) "--hyps_grid", (char*) "data/io_test/hyperpriors_gxage.txt",
						 (char*) "--hyps_probs", (char*) "data/io_test/hyperpriors_gxage_probs.txt",
						 (char*) "--vb_init", (char*) "data/io_test/answer_init.txt",
						 (char*) "--covar", (char*) "data/io_test/n50_p100_env.txt"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		Data data( p );

		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		SECTION( "Ex3. Non genetic data standardised + covars regressed"){
			CHECK(data.params.scale_pheno == true);
			CHECK(data.params.use_vb_on_covars == true);
			CHECK(data.params.covar_file != "NULL");
//			CHECK(data.Y(0,0) == Approx(-3.6676363273605137)); Centered
			CHECK(data.Y(0,0) == Approx(-1.5800573524786081)); // Scaled
			CHECK(data.Y2(0,0) == Approx(-1.5567970303));
			CHECK(data.W(0,0) == Approx(0.8957059881));
			CHECK(data.E(0,0) == Approx(0.8957059881));
		}
		data.read_full_bgen();

		data.calc_dxteex();
		data.calc_snpstats();
		if(p.vb_init_file != "NULL"){
			data.read_alpha_mu();
		}
		VBayesX2 VB(data);
		VB.check_inputs();
		SECTION("Ex3. Vbayes_X2 initialised correctly"){
			CHECK(VB.n_samples == 50);
			CHECK(VB.N == 50.0);
			CHECK(VB.n_env == 4);
			CHECK(VB.n_covar == 4);
			CHECK(VB.n_effects == 2);
			CHECK(VB.vp_init.muw(0) == 0.25);
			CHECK(VB.p.init_weights_with_snpwise_scan == false);
			CHECK(VB.dXtEEX(0, 0) == Approx(38.9390135703));
		}

		SECTION("Ex3. Explicitly checking updates"){
			// Set up for RunInnerLoop
			long n_grid = VB.hyps_grid.rows();
			std::vector<Hyps> all_hyps;
			VB.unpack_hyps(VB.hyps_grid, all_hyps);

			// Set up for updateAllParams
			std::vector<VariationalParameters> all_vp;
			VB.setup_variational_params(all_hyps, all_vp);

			int round_index = 2;
			std::vector<double> logw_prev(n_grid, -1);
			std::vector<std::vector< double >> logw_updates(n_grid);

			VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev, logw_updates);
			VariationalParameters& vp = all_vp[0];
			Hyps& hyps = all_hyps[0];

			CHECK(vp.muc(0) == Approx(0.1221946024));
			CHECK(vp.muc(3) == Approx(-0.1595909887));
			CHECK(vp.alpha_beta(0) == Approx(0.1339235799));
			CHECK(vp.alpha_beta(1) == Approx(0.1415361555));
			CHECK(vp.alpha_beta(63) == Approx(0.1724736345));
			CHECK(vp.muw(0, 0) == Approx(0.1127445891));
			CHECK(VB.calc_logw(all_hyps[0], vp) == Approx(-20076.0449393003));

			VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev, logw_updates);

			CHECK(vp.muc(0) == Approx(0.1463805515));
			CHECK(vp.muc(3) == Approx(-0.1128544804));
			CHECK(vp.alpha_beta(0) == Approx(0.1292056073));
			CHECK(vp.alpha_beta(1) == Approx(0.1338797264));
			CHECK(vp.alpha_beta(63) == Approx(0.1730150924));
			CHECK(vp.muw(0, 0) == Approx(0.0460748751));
			CHECK(VB.calc_logw(all_hyps[0], vp) == Approx(-20075.3681431899));
		}

		std::vector< VbTracker > trackers(VB.hyps_grid.rows());
		VB.run_inference(VB.hyps_grid, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 33);
			CHECK(trackers[3].count == 33);
			CHECK(trackers[0].logw == Approx(-20075.279701215));
			CHECK(trackers[1].logw == Approx(-20074.9040629751));
			CHECK(trackers[2].logw == Approx(-20075.2341616388));
			CHECK(trackers[3].logw == Approx(-20075.9304539824));
		}
	}
}

TEST_CASE( "Example 6: single-env w MoG + hyps max" ){
	parameters p;

	SECTION("Ex6. No filters applied, high mem mode"){
		char* argv[] = { (char*) "bin/bgen_prog", (char*) "--mode_vb", (char*) "--effects_prior_mog",
						 (char*) "--interaction", (char*) "x", (char*) "--vb_iter_max", (char*) "20",
						 (char*) "--mode_empirical_bayes",
						 (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
						 (char*) "--out", (char*) "data/io_test/fake_age.out",
						 (char*) "--pheno", (char*) "data/io_test/pheno.txt",
						 (char*) "--hyps_grid", (char*) "data/io_test/hyperpriors_gxage.txt",
						 (char*) "--hyps_probs", (char*) "data/io_test/hyperpriors_gxage_probs.txt",
						 (char*) "--vb_init", (char*) "data/io_test/answer_init.txt",
						 (char*) "--covar", (char*) "data/io_test/age.txt",
						 (char*) "--spike_diff_factor", (char*) "100"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		Data data( p );

		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		SECTION( "Ex6. Non genetic data standardised + covars regressed"){
			CHECK(data.params.scale_pheno == true);
			CHECK(data.params.use_vb_on_covars == false);
			CHECK(data.params.covar_file != "NULL");
//			CHECK(data.Y(0,0) == Approx(-3.6676363273605137)); Centered
//			CHECK(data.Y(0,0) == Approx(-1.5800573524786081)); Scaled
			CHECK(data.Y(0,0) == Approx(-1.262491384814441));
			CHECK(data.Y2(0,0) == Approx(-1.262491384814441));
			CHECK(data.W(0,0) == Approx(-0.58947939694779772));
			CHECK(data.E(0,0) == Approx(-0.58947939694779772));
		}

		data.read_full_bgen();
		SECTION( "Ex6. bgen read in & standardised correctly"){
			CHECK(data.G.low_mem == false);
			CHECK(data.params.low_mem == false);
			CHECK(data.params.flip_high_maf_variants == true);
			CHECK(data.G(0, 0) == Approx(1.8604233373));
		}

		SECTION( "Ex6. Confirm calc_dxteex() reorders properly"){
			data.params.dxteex_file = "data/io_test/inputs/dxteex_mixed.txt";
			data.read_external_dxteex();
			data.calc_dxteex();
			CHECK(data.dXtEEX(0, 0) == Approx(87.204591182113916));
			CHECK(data.n_dxteex_computed == 1);
		}

		data.calc_dxteex();
		if(p.vb_init_file != "NULL"){
			data.read_alpha_mu();
		}
		VBayesX2 VB(data);
		VB.check_inputs();
		SECTION("Ex6. Vbayes_X2 initialised correctly"){
			CHECK(VB.n_samples == 50);
			CHECK(VB.N == 50.0);
			CHECK(VB.n_env == 1);
			CHECK(VB.n_covar == 1);
			CHECK(VB.n_effects == 2);
			CHECK(VB.vp_init.muw(0) == 1.0);
			CHECK(VB.p.init_weights_with_snpwise_scan == false);
			CHECK(VB.dXtEEX(0, 0) == Approx(87.204591182113916));
		}


		SECTION("Ex6. Explicitly checking updates"){
			// Set up for RunInnerLoop
			long n_grid = VB.hyps_grid.rows();
			std::vector<Hyps> all_hyps;
			VB.unpack_hyps(VB.hyps_grid, all_hyps);

			// Set up for updateAllParams
			std::vector<VariationalParameters> all_vp;
			VB.setup_variational_params(all_hyps, all_vp);

			int round_index = 2;
			std::vector<double> logw_prev(n_grid, -1);
			std::vector<std::vector< double >> logw_updates(n_grid);

			VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev, logw_updates);
			VariationalParameters& vp = all_vp[0];
			Hyps& hyps = all_hyps[0];

			CHECK(vp.alpha_beta(0) == Approx(0.1447525646));
			CHECK(vp.mu1_beta(0) == Approx(-0.0304566021));
			CHECK(vp.mu2_beta(0) == Approx(-0.0003586526));
			CHECK(vp.alpha_beta(1) == Approx(0.1515936892));
			CHECK(vp.mu1_beta(1) == Approx(-0.0356183259));
			CHECK(vp.mu2_beta(1) == Approx(-0.0004194363));
			CHECK(vp.alpha_beta(63) == Approx(0.1762251019));
			CHECK(hyps.sigma == Approx(0.3994029731));
			CHECK(hyps.lambda(0) == Approx(0.1693099847));
			CHECK(hyps.slab_var(0) == Approx(0.0056085838));
			CHECK(hyps.spike_var(0) == Approx(0.0000368515));
			CHECK(VB.calc_logw(hyps, vp) == Approx(-52.129381445));

			VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev, logw_updates);

			CHECK(vp.alpha_beta(0) == Approx(0.1428104733));
			CHECK(vp.mu1_beta(0) == Approx(-0.01972825));
			CHECK(vp.mu2_beta(0) == Approx(-0.0002178332));
			CHECK(vp.alpha_beta(1) == Approx(0.1580997887));
			CHECK(vp.alpha_beta(63) == Approx(0.6342565543));
			CHECK(hyps.sigma == Approx(0.2888497603));
			CHECK(hyps.lambda(0) == Approx(0.2065007836));
			CHECK(hyps.slab_var(0) == Approx(0.0077922078));
			CHECK(hyps.spike_var(0) == Approx(0.0000369985));
			CHECK(VB.calc_logw(hyps, vp) == Approx(-48.0705874648));
		}

		SECTION("Ex6. Checking rescan") {
			// Set up for RunInnerLoop
			// Set up for RunInnerLoop
			int round_index = 2;
			long n_samples = VB.n_samples;
			long n_grid = VB.hyps_grid.rows();
			std::vector<Hyps> all_hyps;
			VB.unpack_hyps(VB.hyps_grid, all_hyps);

			// Allocate trackers
			std::vector< VbTracker > trackers(n_grid);
			for (int nn = 0; nn < n_grid; nn++){
				trackers[nn].set_main_filepath(p.out_file);
				trackers[nn].p = p;
			}


			VB.runInnerLoop(false, round_index, all_hyps, trackers);


			CHECK(trackers[1].logw == Approx(-45.7823937859));
			CHECK(trackers[1].vp.eta[0] == Approx(-0.8185317198));
			CHECK(trackers[1].vp.ym[0] == Approx(-0.4439596651));

			Eigen::VectorXd gam_neglogp(VB.n_var);
			VB.rescanGWAS(trackers[1].vp, gam_neglogp);

			CHECK(gam_neglogp[1] == Approx(0.3038129038));

			Eigen::VectorXd pheno = VB.Y - trackers[1].vp.ym;
			Eigen::VectorXd Z_kk(n_samples);

			CHECK(pheno[0] == Approx(-0.3288994));


			int jj = 1;
			Z_kk = VB.X.col(jj).cwiseProduct(trackers[1].vp.eta);
			double ztz_inv = 1.0 / Z_kk.dot(Z_kk);
			double gam = Z_kk.dot(pheno) * ztz_inv;
			double rss_null = (pheno - Z_kk * gam).squaredNorm();

			// T-test of variant j
			boost_m::students_t t_dist(n_samples - 1);
			double main_se_j    = std::sqrt(rss_null / (VB.N - 1.0) * ztz_inv);
			double main_tstat_j = gam / main_se_j;
			double main_pval_j  = 2 * boost_m::cdf(boost_m::complement(t_dist, fabs(main_tstat_j)));

			double neglogp_j = -1 * std::log10(main_pval_j);

			CHECK(VB.X.col(jj)[0] == Approx(0.7465835328));
			CHECK(Z_kk[0] == Approx(-0.44009531));
			CHECK(gam == Approx(0.0223947128));
			CHECK(main_pval_j == Approx(0.576447458));
			CHECK(main_tstat_j == Approx(0.5623409325));
			CHECK(main_se_j == Approx(0.0398240845));
			CHECK(rss_null == Approx(7.9181184549));
		}

		std::vector< VbTracker > trackers(VB.hyps_grid.rows());
		VB.run_inference(VB.hyps_grid, false, 2, trackers);
		SECTION("Ex6. Vbayes_X2 inference correct"){
//			CHECK(trackers[0].count == 741);
//			CHECK(trackers[3].count == 71);
//			CHECK(trackers[0].logw == Approx(-45.2036994175));
//			CHECK(trackers[1].logw == Approx(-40.8450319874));
//			CHECK(trackers[2].logw == Approx(-40.960377414));
//			CHECK(trackers[3].logw == Approx(-40.9917439828));
			CHECK(trackers[0].count == 20);
			CHECK(trackers[3].count == 20);
			CHECK(trackers[0].logw == Approx(-45.8542053615));
			CHECK(trackers[1].logw == Approx(-45.7823937859));
			CHECK(trackers[2].logw == Approx(-41.3150655897));
			CHECK(trackers[3].logw == Approx(-41.639981773));
		}
	}
}

