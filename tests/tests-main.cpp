// tests-main.cpp
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sys/stat.h>
#include "../src/tools/eigen3.3/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/vbayes.hpp"
#include "../src/vbayes_x2.hpp"
#include "../src/data.hpp"
#include "../src/genotype_matrix.hpp"

TEST_CASE( "Checking parse_arguments", "[io]" ) {
	parameters p;
	
	SECTION( "--convert_to_vcf parsed correctly" ) {
		char* argv[] = { (char*) "bin/bgen_prog",
						 (char*) "--convert_to_vcf", 
						 (char*) "--bgen", 
						 (char*) "data/example/example.v11.bgen",
						 (char*) "--out", 
						 (char*) "data/tmp.out"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		CHECK(p.bgen_file == "data/example/example.v11.bgen");
		CHECK(p.out_file == "data/tmp.out");
		CHECK(p.mode_vcf == true);
	}

	SECTION( "--lm parsed correctly" ) {
		char* argv[] = { (char*) "bin/bgen_prog",
						 (char*) "--lm", 
						 (char*) "--bgen", 
						 (char*) "data/example/example.v11.bgen",
						 (char*) "--out", 
						 (char*) "data/tmp.out",
						 (char*) "--pheno", 
						 (char*) "data/test/empty.pheno",
						 (char*) "--covar", 
						 (char*) "data/test/empty.covar",
						 (char*) "--chunk", 
						 (char*) "100"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		CHECK(p.bgen_file == "data/example/example.v11.bgen");
		CHECK(p.out_file == "data/tmp.out");
		CHECK(p.pheno_file == "data/test/empty.pheno");
		CHECK(p.covar_file == "data/test/empty.covar");
		CHECK(p.mode_lm == true);

		SECTION( "chunk_size correctly assigned" ) {
			CHECK(p.chunk_size == 100);
		}
	}

	SECTION( "reading in genetic confounders" ) {
		std::vector< std::string > true_gconf;
		true_gconf.push_back("gconf1");
		true_gconf.push_back("gconf2");
		true_gconf.push_back("gconf3");
		SECTION( "Parse combo 1" ) {
			char* argv[] = { (char*) "bin/bgen_prog",
							 (char*) "--full_lm", 
							 (char*) "--bgen", 
							 (char*) "data/example/example.v11.bgen",
							 (char*) "--out", 
							 (char*) "data/tmp.out",
							 (char*) "--pheno", 
							 (char*) "data/test/empty.pheno",
							 (char*) "--covar", 
							 (char*) "data/test/empty.covar",
							 (char*) "--genetic_confounders", 
							 (char*) "gconf1",
							 (char*) "gconf2",
							 (char*) "gconf3"};
			int argc = sizeof(argv)/sizeof(argv[0]);
			parse_arguments(p, argc, argv);
			CHECK(p.gconf == true_gconf);
			CHECK(p.n_gconf == 3);
		}
		SECTION( "Parse combo 2" ) {
			char* argv[] = { (char*) "bin/bgen_prog",
							 (char*) "--full_lm", 
							 (char*) "--bgen", 
							 (char*) "data/example/example.v11.bgen",
							 (char*) "--out", 
							 (char*) "data/tmp.out",
							 (char*) "--genetic_confounders", 
							 (char*) "gconf1",
							 (char*) "gconf2",
							 (char*) "gconf3",
							 (char*) "--pheno", 
							 (char*) "data/test/empty.pheno",
							 (char*) "--covar", 
							 (char*) "data/test/empty.covar"};
			int argc = sizeof(argv)/sizeof(argv[0]);
			parse_arguments(p, argc, argv);
			CHECK(p.gconf == true_gconf);
			CHECK(p.n_gconf == 3);
		}
	}
}

TEST_CASE( "Algebra in Eigen3" ) {

	Eigen::MatrixXd X(3, 3);
	Eigen::VectorXd v1(3), v2(3);
	X << 1, 2, 3,
		 4, 5, 6,
		 7, 8, 9;
	v1 << 1, 1, 1;
	v2 << 1, 2, 3;

	SECTION("dot product of vector with col vector"){
		CHECK((v1.dot(X.col(0))) == 12.0);
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
}

TEST_CASE( "GenotypeMatrix Class" ) {

	SECTION("low-mem mode on"){
		GenotypeMatrix GM(true, 3, 2);
		GM.assign_index(0, 0, 0.2);
		GM.assign_index(1, 0, 0.8);
		GM.assign_index(2, 0, 0.2);
		GM.assign_index(0, 1, 0.3);
		GM.assign_index(1, 1, 0.345);
		GM.assign_index(2, 1, 0.213);

		SECTION("Empty constructor"){
			GenotypeMatrix GM2(true);
			GM2.resize(3, 2);
			GM2.assign_index(0, 0, 0.2);
			GM2.assign_index(1, 0, 0.8);
			GM2.assign_index(2, 0, 0.2);
			CHECK(GM2(0,0) == GM(0,0));
		}

		SECTION("Column means and sds computed correctly"){
			GM.calc_scaled_values();
			Eigen::VectorXd true_mean(2), true_sd(2);
			true_mean << 0.3997396, 0.2877604;
			true_sd << 2.879253, 14.846298;   // 0.34731227, 0.06735686;

			CHECK(GM.scaling_performed == true);
			CHECK(Approx(true_mean.sum()) == GM.compressed_dosage_means.sum());
			CHECK(Approx(true_sd.sum()) == GM.compressed_dosage_inv_sds.sum());
		}

		SECTION("Read access returns standardised values"){
			CHECK(Approx(GM(0, 1)) == 0.1933112);
		}

		SECTION("Decompressed matrix multiplication"){
			Eigen::VectorXd vv(2);
			vv << 0.55, 0.676;
			Eigen::VectorXd res = GM * vv;
			Eigen::VectorXd res_truth(3);
			res_truth << -0.1868643, 1.2362057, -1.0493414;
			CHECK(res_truth[0] == Approx(res[0]));
			CHECK(res_truth[1] == Approx(res[1]));
		}

		SECTION("Read column access"){
			Eigen::VectorXd c2a(3), c2b, c2_truth(3);
			c2_truth << 0.1933112, 0.8892314, -1.0825425;
			// CHECK_THROWS(GM.col(1, c2a));
			// GM.col(1, c2a);
			c2b = GM.col(1);

			// CHECK(Approx(c2a[0]) == c2_truth[0]);
			CHECK(Approx(c2b[0]) == c2_truth[0]);
			CHECK(Approx(c2b[1]) == c2_truth[1]);
		}

		SECTION("Conservative resize"){
			GM.conservativeResize(3, 1);
			CHECK(GM.M.cols() == 1);
			CHECK(Approx(GM(1,0)) == 1.1547005);
		}

		SECTION("transpose_vector_multiply "){
			Eigen::VectorXd yy(3), res;
			yy << 0.550, 0.676, -1.226;
 			res = GM.transpose_vector_multiply(yy);

			CHECK(res.rows() == 2);
			CHECK(res.cols() == 1);
			CHECK(Approx(res(0, 0)) == 1.170866);
		}

		SECTION("dot_with_jth_col "){
			Eigen::VectorXd yy(3);
			yy << 10.2, 5.35, 3.1;
 			double res = GM.dot_with_jth_col(yy, 1);

			CHECK(Approx(res) == 3.37328);
		}
	}

	SECTION("Normal functionality when low-mem off"){
		GenotypeMatrix GM(false, 3, 2);
		GM.assign_index(0, 0, 0.2);
		GM.assign_index(1, 0, 0.8);
		GM.assign_index(2, 0, 0.2);
		GM.assign_index(0, 1, 0.3);
		GM.assign_index(1, 1, 0.345);
		GM.assign_index(2, 1, 0.213);

		SECTION("Read access returns standardised values"){
			CHECK(Approx(GM(0, 1)) == 0.2086301);
		}
		SECTION("Matrix multiplication functions on scaled matrix"){
			Eigen::VectorXd vv(2);
			vv << 0.55, 0.676;
			Eigen::VectorXd res = GM * vv;
			Eigen::VectorXd res_truth(3);
			res_truth << -0.1765087, 1.2294428, -1.0529341;
			CHECK(res_truth[0] == Approx(res[0]));
			CHECK(res_truth[1] == Approx(res[1]));
		}

		SECTION("Missing values handled correctly"){
			GM.assign_index(2, 0, std::nan(""));
			CHECK(GM(1, 0) == Approx(0.7071068));
			CHECK(0.0 == Approx(GM(2, 0)));
		}

		SECTION("transpose_vector_multiply "){
			Eigen::VectorXd yy(3), res;
			yy << 0.550, 0.676, -1.226;
 			res = GM.transpose_vector_multiply(yy);

			CHECK(res.rows() == 2);
			CHECK(res.cols() == 1);
			CHECK(Approx(res(0, 0)) == 1.170866);
		}
	}
}

TEST_CASE( "Data Class" ){
	parameters p;

	SECTION("No filters applied, high mem mode"){
		char* argv[] = { (char*) "bin/bgen_prog", (char*) "--mode_vb", 
						 (char*) "--interaction", (char*) "x",
						 (char*) "--bgen", (char*) "tests/data/case6_n50_p100.bgen",
						 (char*) "--out", (char*) "tests/data/fake.out",
						 (char*) "--pheno", (char*) "tests/data/pheno.txt",
						 (char*) "--hyps_grid", (char*) "tests/data/hyperpriors_gxage_v1.txt",
						 (char*) "--hyps_probs", (char*) "tests/data/hyperpriors_gxage_v1_probs.txt",
						 (char*) "--vb_init", (char*) "tests/data/vb_inits.out",
						 (char*) "--covar", (char*) "tests/data/age.txt"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		Data data( p );

		data.read_non_genetic_data();
		SECTION( "Raw non genetic data read in accurately"){
			CHECK(data.Y(0,0) == Approx(-1.18865038973338));
			CHECK(data.W(0,0) == Approx(-0.334726453474872));
			CHECK(data.alpha_init(2,0) == Approx(0.000102891));
			CHECK(data.mu_init(4,0) == Approx(-0.0968564));
			CHECK(data.hyps_grid(0,6) == Approx(0.0153846153846154));
			CHECK(data.imprt_grid(4,0) == Approx(1.0372941821041e-18));
		}

		data.standardise_non_genetic_data();
		SECTION( "Non genetic data standardised"){
			CHECK(data.Y(0,0) == Approx(-1.58005735));
			CHECK(data.W(0,0) == Approx(-0.5894794));
		}

		data.regress_out_covars();
		SECTION( "Covariates regressed out"){
			CHECK(data.Y(0,0) == Approx(-1.2624914));
		}

		data.read_full_bgen();
		SECTION( "bgen read in & standardised correctly"){
			CHECK(data.G(0, 0) == Approx(-0.2220569));
		}
	}
}


TEST_CASE( "vbayes_x2.hpp", "[VBayesX2]" ) {
	parameters p;
	char* argv[] = { (char*) "bin/bgen_prog", (char*) "--mode_vb", 
					 (char*) "--interaction", (char*) "x",
					 (char*) "--bgen", (char*) "tests/data/case6_n50_p100.bgen",
					 (char*) "--out", (char*) "tests/data/fake.out",
					 (char*) "--pheno", (char*) "tests/data/pheno.txt",
					 (char*) "--hyps_grid", (char*) "tests/data/hyperpriors_gxage_v1.txt",
					 (char*) "--hyps_probs", (char*) "tests/data/hyperpriors_gxage_v1_probs.txt",
					 (char*) "--vb_init", (char*) "tests/data/vb_inits.out",
					 (char*) "--covar", (char*) "tests/data/age.txt"};
	int argc = sizeof(argv)/sizeof(argv[0]);
	parse_arguments(p, argc, argv);
	Data data( p );
	data.read_non_genetic_data();
	data.standardise_non_genetic_data();
	data.read_full_bgen();

	// Pass data to VBayes object
	VBayesX2 VB(data);

	// SECTION()
	CHECK(VB.n_grid == 7);
	VB.check_inputs();
	CHECK(VB.n_grid == 6);
	CHECK(VB.hyps_grid.rows() == 6);
	CHECK(VB.probs_grid.rows() == 6);

	SECTION("Function to validate hyperparameter grid"){
			int n_var = 50;
			Eigen::MatrixXd orig(3, 5), attempt, answer(2, 5);
			std::vector<int> attempt_vec, answer_vec;

			// Filling answers
			orig << 1, 0.1, 0.1, 0.1, 0.1,
					1, 0.1, 0.1, 0.001, 0.1, 
					1, 0.1, 0.1, 0.1, 0.1;
			answer << 1, 0.1, 0.1, 0.1, 0.1,
					  1, 0.1, 0.1, 0.1, 0.1;
			answer_vec.push_back(0);
			answer_vec.push_back(2);

			CHECK(validate_grid(orig, n_var) == answer_vec);
			CHECK(subset_matrix(orig, answer_vec) == answer);
	}
}

// TEST_CASE( "Checking data", "[data]" ) {
// 	char* argv[] = { (char*) "bin/bgen_prog",
// 					 (char*) "--bgen", 
// 					 (char*) "data/example/example.v11.bgen"};
// 	int argc = sizeof(argv)/sizeof(argv[0]);
// 	parameters p;
// 	parse_arguments(p, argc, argv);
// 	Data data( p.bgen_file );
// 
// 	SECTION( "read_txt_file behaving sensibly" ) {
// 		std::string filename = "tests/cases/test_covar.txt";
// 		Eigen::MatrixXd M_read, M_ans(3, 3), M_cent(3, 3);
// 		int n_cols;
// 		std::vector< std::string > col_names_read, col_names_ans;
// 		std::map< int, bool > incomplete_row;
// 
// 		data.n_samples = 3;
// 		data.read_txt_file( filename,
// 							M_read,
// 							n_cols,
// 							col_names_read,
// 							incomplete_row );
// 
// 		M_ans << 1, 2, 3, 4, 5, 6, 7, 8, 9;
// 		col_names_ans.push_back("A");
// 		col_names_ans.push_back("B");
// 		col_names_ans.push_back("C");
// 
// 		CHECK( M_read == M_ans );
// 		CHECK( n_cols == 3 );
// 		CHECK( col_names_read == col_names_ans );
// 
// 		SECTION( "Number of covariates greater than n_samples" ) {
// 			data.n_samples = 2;
// 			CHECK_THROWS_AS(data.read_txt_file( filename,
// 								M_read,
// 								n_cols,
// 								col_names_read,
// 								incomplete_row ), std::runtime_error);
// 		}
// 
// 		SECTION( "center_matrix behaving sensibly" ) {
// 			Eigen::MatrixXd M_cent(3, 3);
// 			int n_cols_ans = 3;
// 			M_cent << -1, -1, -1, 0, 0, 0, 1, 1, 1;
// 
// 			data.center_matrix( M_read, n_cols_ans );
// 			data.scale_matrix( M_read, n_cols_ans );
// 			CHECK( M_read == M_cent );
// 		}
// 
// 		// SECTION( "reduce_mat_to_complete_cases behaving sensibly" ) {
// 		// 	Eigen::MatrixXd M_reduc(2, 3);
// 		// 	int n_cols_ans = 3;
// 		// 	bool check = false;
// 		// 	std::map< int, bool > incomplete_cases;
// 		// 	M_reduc << 1, 2, 3, 7, 8, 9;
// 		// 	incomplete_cases[1] = 1;
// 		// 
// 		// 	data.reduce_mat_to_complete_cases(M_read, check, n_cols_ans, incomplete_cases);
// 		// 	// std::cout << "M_read is now " << M_read.rows() << "x" << M_read.cols() << std::endl;
// 		// 	CHECK( M_read == M_reduc );
// 		// }
// 
// 		// SECTION( "scale_matrix reports removal of zero variance cols" ) {
// 		// 	Eigen::MatrixXd W(4, 3);
// 		// 	std::vector<std::string> covar_names, red_covar_names;
// 		// 	
// 		// 	// init
// 		// 	W << 1, 0, 1, 1, 0, 1, -1, 0, -1, -1, 0, -1;
// 		// 	covar_names.push_back("c1").push_back("c2").push_back("c3");
// 		// 	red_covar_names.push_back("c1").push_back("c3");
// 		// 
// 		// 	// test
// 		// 	data.W = W;
// 		// 	data.covar_names = covar_names;
// 		// 	data.n_col = 3;
// 		// 	data.scale_matrix(data.W, data.n_col, data.covar_names);
// 		// 	CHECK( data.covar_names == red_covar_names );
// 		// 	CHECK( data.n_col == 2 );
// 		// }
// 	}
// }
// 
// TEST_CASE( "Check incl_sample_ids", "[data]" ) {
// 	parameters p;
// 
// 	SECTION( "Read in sids correctly and subset covar matrix" ) {
// 		char* argv[] = { (char*) "bin/bgen_prog",
// 						 (char*) "--bgen", 
// 						 (char*) "data/test/example.v11.bgen",
// 						 (char*) "--covar", 
// 						 (char*) "data/test/t1_incl_sample_ids/valid_ids.covar",
// 						 (char*) "--incl_sample_ids", 
// 						 (char*) "data/test/t1_incl_sample_ids/valid_ids.txt"};
// 		int argc = sizeof(argv)/sizeof(argv[0]);
// 		parse_arguments(p, argc, argv);
// 		Data data( p.bgen_file );
// 		data.params = p;
// 		
// 		CHECK(data.params.incl_sids_file == "data/test/t1_incl_sample_ids/valid_ids.txt");
// 		data.read_incl_sids();
// 		CHECK(data.n_samples - data.incomplete_cases.size() == 7);
// 		data.read_covar();
// 		data.reduce_to_complete_cases(); // Also edits n_samples
// 		CHECK(data.n_samples == 7);
// 		Eigen::MatrixXd C(7,1);
// 		C << 1, 3, 17, 21, 22, 25, 500;
// 		CHECK(data.W == C);
// 	}
// 
// 	SECTION( "Empty incl_sample_ids file throws error" ) {
// 		char* argv[] = { (char*) "bin/bgen_prog",
// 						 (char*) "--bgen", 
// 						 (char*) "data/test/example.v11.bgen",
// 						 (char*) "--incl_sample_ids", 
// 						 (char*) "data/test/t1_incl_sample_ids/empty.txt"};
// 		int argc = sizeof(argv)/sizeof(argv[0]);
// 		CHECK_THROWS_AS(parse_arguments(p, argc, argv), std::runtime_error);
// 	}
// 	
// 	SECTION( "Unordered sids throw error" ) {
// 		char* argv[] = { (char*) "bin/bgen_prog",
// 						 (char*) "--bgen", 
// 						 (char*) "data/test/example.v11.bgen",
// 						 (char*) "--incl_sample_ids", 
// 						 (char*) "data/test/t1_incl_sample_ids/invalid_ids1.txt"};
// 		int argc = sizeof(argv)/sizeof(argv[0]);
// 		parse_arguments(p, argc, argv);
// 		Data data( p.bgen_file );
// 		data.params = p;
// 		
// 		CHECK_THROWS_AS(data.read_incl_sids(), std::logic_error);
// 	}
// 
// 	SECTION( "Absent sid throws error" ) {
// 		char* argv[] = { (char*) "bin/bgen_prog",
// 						 (char*) "--bgen", 
// 						 (char*) "data/test/example.v11.bgen",
// 						 (char*) "--incl_sample_ids", 
// 						 (char*) "data/test/t1_incl_sample_ids/invalid_ids2.txt"};
// 		int argc = sizeof(argv)/sizeof(argv[0]);
// 		parse_arguments(p, argc, argv);
// 		Data data( p.bgen_file );
// 		data.params = p;
// 		
// 		CHECK_THROWS_AS(data.read_incl_sids(), std::logic_error);
// 	}
// 
// 	SECTION( "Absent sid throws error including last sid" ) {
// 		char* argv[] = { (char*) "bin/bgen_prog",
// 						 (char*) "--bgen", 
// 						 (char*) "data/test/example.v11.bgen",
// 						 (char*) "--incl_sample_ids", 
// 						 (char*) "data/test/t1_incl_sample_ids/invalid_ids3.txt"};
// 		int argc = sizeof(argv)/sizeof(argv[0]);
// 		parse_arguments(p, argc, argv);
// 		Data data( p.bgen_file );
// 		data.params = p;
// 		
// 		CHECK_THROWS_AS(data.read_incl_sids(), std::logic_error);
// 	}
// 
// }


// TEST_CASE( "Unit test vbayes", "[vbayes]" ) {
// 	double PI = 3.1415926535897;
// 	Eigen::MatrixXd myX(2, 2), myY(2, 1);
// 	myY << 1.0, 2.0;
// 	myX << 1.0, 0.0, 0.0, 1.0;
// 	vbayes VB(myX, myY);
// 	
// 	SECTION( "Check vbayes.returnZ" ) {
// 		// Hardcoded comparison with R
// 		Eigen::VectorXd alpha(2), mu(2);
// 		alpha << 1.0, 0.0;
// 		mu << 0.7, 0.2;
// 		std::vector< double > s_sq;
// 		s_sq.push_back(1.2);
// 		s_sq.push_back(0.8);
// 		double pi = 0.2;
// 		double sigma = 1.0;
// 		double sigmab = 1.0;
// 
// 		double res = -6.569298;
// 		CHECK(Approx(res) == VB.calc_logw(sigma, sigmab, pi, s_sq, alpha, mu));
// 	}
// }
