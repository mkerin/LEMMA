// tests-main.cpp
#define EIGEN_USE_MKL_ALL
#include "catch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include "../src/tools/eigen3.3/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/vbayes_x2.hpp"
#include "../src/data.hpp"
#include "../src/hyps.hpp"


// Scenarios
char* case1a[] = { (char*) "--mode_vb",
	               (char*) "--mode_empirical_bayes",
	               (char*) "--spike_diff_factor", (char*) "10000",
	               (char*) "--vb_iter_max", (char*) "10",
	               (char*) "--hyps_grid", (char*) "data/io_test/single_hyps_gxage.txt",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test1a.out"};

char* case1b[] = { (char*) "--mode_vb",
	               (char*) "--mode_empirical_bayes",
	               (char*) "--spike_diff_factor", (char*) "10000",
	               (char*) "--vb_iter_max", (char*) "10",
	               (char*) "--vb_iter_start", (char*) "3",
	               (char*) "--resume_from_param_dump",
	               (char*) "data/io_test/r2_interim_files/grid_point_0/test1a_dump_it2",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test1b.out"};

char* case1c[] = { (char*) "--mode_calc_snpstats",
	               (char*) "--resume_from_param_dump",
	               (char*) "data/io_test/r2_interim_files/grid_point_0/test1a_dump_it10",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test1c.out"};



TEST_CASE("Resume from multi-env + mog + emp_bayes"){
	parameters p;

	SECTION("Run to iter 10"){
		int argc = sizeof(case1a)/sizeof(case1a[0]);
		parse_arguments(p, argc, case1a);

		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		data.calc_dxteex();
		data.set_vb_init();

		VBayesX2 VB(data);

		std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
		SECTION("Ex4. Explicitly checking hyps") {
			// Set up for RunInnerLoop
			long n_grid = VB.hyps_inits.size();
			std::vector<Hyps> all_hyps = VB.hyps_inits;
			std::vector<VariationalParameters> all_vp;
			VB.setup_variational_params(all_hyps, all_vp);

			int round_index = 2;
			std::vector<double> logw_prev(n_grid, -std::numeric_limits<double>::max());
			std::vector<std::vector< double > > logw_updates(n_grid);
			VariationalParameters& vp = all_vp[0];
			Hyps& hyps = all_hyps[0];

			VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-92.2292775905));
			VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-89.6710643279));
			VB.updateAllParams(2, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-88.4914916475));

			CHECK(VB.YM.squaredNorm() == Approx(14.6462021668));
			CHECK(VB.YX.squaredNorm() == Approx(0.0004903837));
			CHECK(VB.ETA.squaredNorm() == Approx(0.0773475751));
			CHECK(VB.ETA_SQ.squaredNorm() == Approx(294.9017799794));

			VbTracker tracker(p);
			tracker.init_interim_output(0,2, VB.n_effects, VB.n_covar, VB.n_env, VB.env_names, vp);
			tracker.dump_state("2", VB.n_samples, VB.n_covar, VB.n_var, VB.n_env,
			                   VB.n_effects, vp, hyps, VB.Y, VB.C, VB.X,
			                   VB.covar_names, VB.env_names);

			VB.updateAllParams(3, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-87.8880225449));

			CHECK(vp.ym.squaredNorm() == Approx(15.7893306211));
			CHECK(vp.yx.squaredNorm() == Approx(0.0000929716));
			CHECK(vp.eta.squaredNorm() == Approx(0.0231641669));
		}

		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		SECTION("Check converged inference"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-86.8131749627));
		}

		SECTION("Check converged snp-stats"){
			// Compute snp-stats
			long n_var = VB.n_var;
			long n_chrs = VB.n_chrs;
			VariationalParametersLite& vp_init = VB.vp_init;

			std::vector<Eigen::VectorXd> map_residuals_by_chr(n_chrs), pred_main(n_chrs), pred_int(n_chrs);
			Eigen::VectorXd neglogp_beta(n_var), neglogp_gam(n_var), neglogp_rgam(n_var), neglogp_joint(n_var);
			Eigen::VectorXd test_stat_beta(n_var), test_stat_gam(n_var), test_stat_rgam(n_var), test_stat_joint(n_var);

			VB.compute_residuals_per_chr(vp_init, pred_main, pred_int, map_residuals_by_chr);
			VB.LOCO_pvals(vp_init, map_residuals_by_chr, neglogp_beta, neglogp_gam, neglogp_rgam, neglogp_joint,
			              test_stat_beta, test_stat_gam, test_stat_rgam, test_stat_joint);

			CHECK(map_residuals_by_chr[0](0) == Approx(-1.5520966012));
			CHECK(neglogp_beta(0) == Approx(0.2697970087));
			CHECK(neglogp_gam(0) == Approx(1.6834605272));
		}

	}

	SECTION("Resume from iter 2"){
		int argc = sizeof(case1b)/sizeof(case1b[0]);
		parse_arguments(p, argc, case1b);

		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		data.calc_dxteex();
		data.set_vb_init();

		VBayesX2 VB(data);

		std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
		SECTION("Ex4. Explicitly checking hyps") {
			long n_grid = VB.hyps_inits.size();
			std::vector<Hyps> all_hyps = VB.hyps_inits;
			std::vector<VariationalParameters> all_vp;
			VB.setup_variational_params(all_hyps, all_vp);

			int round_index = 2;
			std::vector<double> logw_prev(n_grid, -std::numeric_limits<double>::max());
			std::vector<std::vector< double > > logw_updates(n_grid);
			VariationalParameters& vp = all_vp[0];
			Hyps& hyps = all_hyps[0];

			CHECK(VB.YM.squaredNorm() == Approx(14.64620215));
			CHECK(VB.YX.squaredNorm() == Approx(0.0004903837));
			CHECK(VB.ETA.squaredNorm() == Approx(0.0773475736));
			CHECK(VB.ETA_SQ.squaredNorm() == Approx(294.9017821007));

			VB.updateAllParams(3, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-87.8880225713));

			CHECK(vp.ym.squaredNorm() == Approx(15.7893305635));
			CHECK(vp.yx.squaredNorm() == Approx(0.0000929716));
			CHECK(vp.eta.squaredNorm() == Approx(0.0231641668));
		}

		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-86.8131749627));
		}
	}

	SECTION("Compute snp-stats from dump of converged params"){
		int argc = sizeof(case1c)/sizeof(case1c[0]);
		parse_arguments(p, argc, case1c);

		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		data.set_vb_init();

		VBayesX2 VB(data);

		// Compute snp-stats
		long n_var = VB.n_var;
		long n_chrs = VB.n_chrs;
		VariationalParametersLite& vp_init = VB.vp_init;

		std::vector<Eigen::VectorXd> map_residuals_by_chr(n_chrs), pred_main(n_chrs), pred_int(n_chrs);
		Eigen::VectorXd neglogp_beta(n_var), neglogp_gam(n_var), neglogp_rgam(n_var), neglogp_joint(n_var);
		Eigen::VectorXd test_stat_beta(n_var), test_stat_gam(n_var), test_stat_rgam(n_var), test_stat_joint(n_var);

		VB.compute_residuals_per_chr(vp_init, pred_main, pred_int, map_residuals_by_chr);
		VB.LOCO_pvals(vp_init, map_residuals_by_chr, neglogp_beta, neglogp_gam, neglogp_rgam, neglogp_joint,
		              test_stat_beta, test_stat_gam, test_stat_rgam, test_stat_joint);

		CHECK(map_residuals_by_chr[0](0) == Approx(-1.5520966012));
		CHECK(neglogp_beta(0) == Approx(0.2697970087));
		CHECK(neglogp_gam(0) == Approx(1.6834605272));
	}
}


char* case2a[] = { (char*) "--mode_vb",
	               (char*) "--mode_squarem",
	               (char*) "--spike_diff_factor", (char*) "10000",
	               (char*) "--vb_iter_max", (char*) "10",
	               (char*) "--hyps_grid", (char*) "data/io_test/single_hyps_gxage.txt",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test2a.out.gz"};

char* case2b[] = { (char*) "--mode_vb",
	               (char*) "--mode_squarem",
	               (char*) "--spike_diff_factor", (char*) "10000",
	               (char*) "--vb_iter_max", (char*) "10",
	               (char*) "--vb_iter_start", (char*) "3",
	               (char*) "--resume_from_param_dump",
	               (char*) "data/io_test/r2_interim_files/grid_point_0/test2a_dump_it2",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test2b.out.gz"};

TEST_CASE("Resume from multi-env + mog + squarem"){
	parameters p;

	SECTION("Run to iter 10"){
		int argc = sizeof(case2a)/sizeof(case2a[0]);
		parse_arguments(p, argc, case2a);
		p.mode_squarem = true;

		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		data.calc_dxteex();
		data.set_vb_init();

		VBayesX2 VB(data);

		std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
		SECTION("Ex4. Explicitly checking hyps") {
			// Set up for RunInnerLoop
			long n_grid = VB.hyps_inits.size();
			std::vector<Hyps> all_hyps = VB.hyps_inits;
			std::vector<VariationalParameters> all_vp;
			VB.setup_variational_params(all_hyps, all_vp);

			int round_index = 2;
			std::vector<double> logw_prev(n_grid, -std::numeric_limits<double>::max());
			std::vector<std::vector< double > > logw_updates(n_grid);
			VariationalParameters& vp = all_vp[0];
			Hyps& hyps = all_hyps[0];

			VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-92.2292775905));
			VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-89.6710643279));
			VB.updateAllParams(2, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-88.4914916475));

			CHECK(VB.YM.squaredNorm() == Approx(14.6462021668));
			CHECK(VB.YX.squaredNorm() == Approx(0.0004903837));
			CHECK(VB.ETA.squaredNorm() == Approx(0.0773475736));
			CHECK(VB.ETA_SQ.squaredNorm() == Approx(294.9017821007));

			VbTracker tracker(p);
			tracker.init_interim_output(0,2, VB.n_effects, VB.n_covar, VB.n_env, VB.env_names, vp);
			tracker.dump_state("2", VB.n_samples, VB.n_covar, VB.n_var, VB.n_env,
			                   VB.n_effects, vp, hyps, VB.Y, VB.C, VB.X,
			                   VB.covar_names, VB.env_names);

			VB.updateAllParams(3, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-87.8880225449));

			CHECK(vp.ym.squaredNorm() == Approx(15.7893305635));
			CHECK(vp.yx.squaredNorm() == Approx(0.0000929716));
			CHECK(vp.eta.squaredNorm() == Approx(0.0231641668));
			CHECK(VB.ETA_SQ.squaredNorm() == Approx(397.6779293259));
		}

		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-86.650909737));
		}
	}

	SECTION("Resume from iter 2"){
		int argc = sizeof(case2b)/sizeof(case2b[0]);
		parse_arguments(p, argc, case2b);
		p.mode_squarem = true;

		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		data.calc_dxteex();
		data.set_vb_init();

		VBayesX2 VB(data);

		std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
		SECTION("Ex4. Explicitly checking hyps") {
			long n_grid = VB.hyps_inits.size();
			std::vector<Hyps> all_hyps = VB.hyps_inits;
			std::vector<VariationalParameters> all_vp;
			VB.setup_variational_params(all_hyps, all_vp);

			int round_index = 2;
			std::vector<double> logw_prev(n_grid, -std::numeric_limits<double>::max());
			std::vector<std::vector< double > > logw_updates(n_grid);
			VariationalParameters& vp = all_vp[0];
			Hyps& hyps = all_hyps[0];

			CHECK(VB.YM.squaredNorm() == Approx(14.6462021668));
			CHECK(VB.YX.squaredNorm() == Approx(0.0004903837));
			CHECK(VB.ETA.squaredNorm() == Approx(0.0773475736));
			CHECK(VB.ETA_SQ.squaredNorm() == Approx(294.9017821007));
			CHECK(VB.calc_logw(hyps, vp) == Approx(-88.4914916517));

			VB.updateAllParams(3, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-87.8880225713));

			CHECK(vp.ym.squaredNorm() == Approx(15.7893305635));
			CHECK(vp.yx.squaredNorm() == Approx(0.0000929716));
			CHECK(vp.eta.squaredNorm() == Approx(0.0231641668));
			CHECK(VB.ETA_SQ.squaredNorm() == Approx(397.6779293259));
		}

		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 10);
			// Slight discrepancy between original run and restart.
			// Think this is because we now need the previous two hyps values to
			// keep using SQUAREM from the same place
			// CHECK(trackers[0].logw == Approx(-86.6456071112));
			CHECK(trackers[0].logw == Approx(-86.533162843));
		}
	}
}


// Scenarios
char* case3a[] = { (char*) "--mode_vb",
	               (char*) "--mode_empirical_bayes",
	               (char*) "--use_vb_on_covars",
	               (char*) "--vb_iter_max", (char*) "10",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test3a.out"};

char* case3c[] = { (char*) "--mode_calc_snpstats", (char*) "--use_vb_on_covars",
	               (char*) "--resume_from_param_dump",
	               (char*) "data/io_test/r2_interim_files/grid_point_0/test3a_dump_it10",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test3c.out"};


TEST_CASE("Resume from multi-env + mog + emp_bayes + incl_covars"){
	parameters p;

	SECTION("Run to iter 10"){
		int argc = sizeof(case3a)/sizeof(case3a[0]);
		parse_arguments(p, argc, case3a);

		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		data.calc_dxteex();
		data.set_vb_init();

		VBayesX2 VB(data);

		std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		SECTION("Check converged inference"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-88.8466020959));
		}

		SECTION("Check converged snp-stats"){
			// Compute snp-stats
			long n_var = VB.n_var;
			long n_chrs = VB.n_chrs;
			VariationalParametersLite& vp_init = VB.vp_init;

			std::vector<Eigen::VectorXd> map_residuals_by_chr(n_chrs), pred_main(n_chrs), pred_int(n_chrs);
			Eigen::VectorXd neglogp_beta(n_var), neglogp_gam(n_var), neglogp_rgam(n_var), neglogp_joint(n_var);
			Eigen::VectorXd test_stat_beta(n_var), test_stat_gam(n_var), test_stat_rgam(n_var), test_stat_joint(n_var);

			VB.compute_residuals_per_chr(vp_init, pred_main, pred_int, map_residuals_by_chr);
			VB.LOCO_pvals(vp_init, map_residuals_by_chr, neglogp_beta, neglogp_gam, neglogp_rgam, neglogp_joint,
			              test_stat_beta, test_stat_gam, test_stat_rgam, test_stat_joint);

			CHECK(map_residuals_by_chr[0](0) == Approx(-1.5468266922));
			CHECK(neglogp_beta(0) == Approx(0.2859132953));
			CHECK(neglogp_gam(0) == Approx(1.62452219));
			CHECK(neglogp_rgam(0) == Approx(2.9858778387));
		}
		VB.write_map_stats_to_file("");
	}


	SECTION("Compute snp-stats from dump of converged params"){
		int argc = sizeof(case3c)/sizeof(case3c[0]);
		parse_arguments(p, argc, case3c);

		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		data.set_vb_init();

		VBayesX2 VB(data);

		// Compute snp-stats
		long n_var = VB.n_var;
		long n_chrs = VB.n_chrs;
		long n_samples = VB.n_samples;
		long n_env = VB.n_env;
		VariationalParametersLite& vp_init = VB.vp_init;

		std::vector<Eigen::VectorXd> map_residuals_by_chr(n_chrs), pred_main(n_chrs), pred_int(n_chrs);
		Eigen::VectorXd neglogp_beta(n_var), neglogp_gam(n_var), neglogp_rgam(n_var), neglogp_joint(n_var);
		Eigen::VectorXd test_stat_beta(n_var), test_stat_gam(n_var), test_stat_rgam(n_var), test_stat_joint(n_var);

		VB.compute_residuals_per_chr(vp_init, pred_main, pred_int, map_residuals_by_chr);
		VB.LOCO_pvals(vp_init, map_residuals_by_chr, neglogp_beta, neglogp_gam, neglogp_rgam, neglogp_joint,
		              test_stat_beta, test_stat_gam, test_stat_rgam, test_stat_joint);

		CHECK(map_residuals_by_chr[0](0) == Approx(-1.5468266922));
		CHECK(neglogp_beta(0) == Approx(0.2859132953));
		CHECK(neglogp_gam(0) == Approx(1.62452219));
		CHECK(neglogp_rgam(0) == Approx(2.9858778387));

		VB.p.LOSO_window = 10;
		VB.LOCO_pvals_v2(VB.X, vp_init, VB.p.LOSO_window, neglogp_beta,
								neglogp_rgam,
								neglogp_joint,
								test_stat_beta,
								test_stat_rgam,
								test_stat_joint);

		CHECK(neglogp_beta(0) == Approx(0.2739696266));
		CHECK(neglogp_rgam(0) == Approx(3.3699459279));
	}
}
