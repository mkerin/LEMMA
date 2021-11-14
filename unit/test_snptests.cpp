//
// Created by kerin on 2019-12-26.
//
#include "catch.hpp"

#include "../src/tools/eigen3.3/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/vbayes.hpp"
#include "../src/data.hpp"

#include <string>

char* full_run[] = { (char*) "prog",
	                 (char*) "--VB-varEM",
	                 (char*) "--VB-iter-max", (char*) "10",
	                 (char*) "--random-seed", (char*) "1",
	                 (char*) "--pheno", (char*) "unit/data/pheno.txt",
	                 (char*) "--environment", (char*) "unit/data/n50_p100_env.txt",
	                 (char*) "--bgen", (char*) "unit/data/n50_p100.bgen",
	                 (char*) "--out", (char*) "unit/data/test5a.out.gz"};

char* from_resume[] = { (char*) "prog",
	                    (char*) "--bgen", (char*) "unit/data/n50_p100.bgen",
	                    (char*) "--singleSnpStats",
	                    (char*) "--resume-from-state",
	                    (char*) "unit/data/lemma_interim_files/test5a_dump_it_converged",
	                    (char*) "--pheno", (char*) "unit/data/pheno.txt",
	                    (char*) "--environment", (char*) "unit/data/n50_p100_env.txt",
	                    (char*) "--out", (char*) "unit/data/test5b.out.gz"};

char* from_file[] = { (char*) "prog",
	                  (char*) "--streamBgen", (char*) "unit/data/n50_p100.bgen",
	                  (char*) "--singleSnpStats",
	                  (char*) "--resid-pheno", (char*) "unit/data/test5a_converged_yhat.out.gz",
	                  (char*) "--environment", (char*) "unit/data/test5a_converged_eta.out.gz",
	                  (char*) "--out", (char*) "unit/data/test5c.out.gz"};


void my_test(const std::string& mode){
	parameters p;
	if (mode == "after_lemma") {
		int argc = sizeof(full_run) / sizeof(full_run[0]);
		parse_arguments(p, argc, full_run);
	} else if (mode == "from_state") {
		int argc = sizeof(from_resume) / sizeof(from_resume[0]);
		parse_arguments(p, argc, from_resume);
	} else if (mode == "from_file") {
		int argc = sizeof(from_file) / sizeof(from_file[0]);
		parse_arguments(p, argc, from_file);
	}

	Data data(p);
	data.read_non_genetic_data();
	data.standardise_non_genetic_data();
	data.read_full_bgen();
	data.calc_dxteex();
	data.set_vb_init();

	Eigen::MatrixXd neglogPvals, testStats;
	if (!p.streamBgenFiles.empty()) {
		long n_var_parsed = 0;
		GenotypeMatrix Xstream(p);
		bool bgen_pass = true;
		fileUtils::read_bgen_chunk(data.streamBgenViews[0], Xstream, data.sample_is_invalid,
		                           data.n_samples, 128, p, bgen_pass, n_var_parsed);
		Xstream.calc_scaled_values();
		compute_LOCO_pvals(data.resid_loco.col(0), Xstream, neglogPvals, testStats, data.vp_init.eta);
	} else {
		VBayes VB(data);
		if (mode == "after_lemma") {
			std::vector<VbTracker> trackers(VB.hyps_inits.size(), p);
			VB.run_inference(VB.hyps_inits, false, 2, trackers);
			VB.output_vb_results();
		}
		VB.compute_LOCO_pvals(VB.vp_init, neglogPvals, testStats);
	}
	CHECK(neglogPvals(0, 0) == Approx(0.2692832123));
	CHECK(neglogPvals(0, 1) == Approx(1.3696407518));
	CHECK(neglogPvals(0, 2) == Approx(2.185395935));
    CHECK(neglogPvals(0, 3) == Approx(0.9992683491));
}


TEST_CASE("Hypothesis tests invariant to code path"){
	auto mode = GENERATE(as<std::string>{}, "after_lemma","from_file","from_state");
	DYNAMIC_SECTION("Mode: " << mode << ", streamBgen: " << (mode == "after_lemma" ? "true" : "false")) {
		my_test(mode);
	}
}
