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
#include "../src/genotype_matrix.hpp"
// #include "../src/data.hpp"

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
	
	// Annoyingly no death test functionality exists within the Catch2 framework
	// SECTION( "Error caused by invalid flag" ) {
	// 	const char* argv[] = { "bin/bgen_prog", "--hello", "you"};
	// 	int argc = sizeof(argv)/sizeof(argv[0]);
	// 	parse_arguments(p, argc, argv);
	// }
}

TEST_CASE( "Unit test vbayes", "[vbayes]" ) {
	double PI = 3.1415926535897;
	Eigen::MatrixXd myX(2, 2), myY(2, 1);
	myY << 1.0, 2.0;
	myX << 1.0, 0.0, 0.0, 1.0;
	vbayes VB(myX, myY);
	
	SECTION( "Check vbayes.returnZ" ) {
		// Hardcoded comparison with R
		Eigen::VectorXd alpha(2), mu(2);
		alpha << 1.0, 0.0;
		mu << 0.7, 0.2;
		std::vector< double > s_sq;
		s_sq.push_back(1.2);
		s_sq.push_back(0.8);
		double pi = 0.2;
		double sigma = 1.0;
		double sigmab = 1.0;

		double res = -6.569298;
		CHECK(Approx(res) == VB.calc_logw(sigma, sigmab, pi, s_sq, alpha, mu));
	}

	// SECTION( "Check vbayes.random_mu_alpha" ) {
	// 	int n_var = 5;
	// 	Eigen::MatrixXd myX(2, n_var), myY(2, 1);
	// 	vbayes VB(myX, myY);
	// 	Eigen::VectorXd alpha, mu;
	// 	VB.random_alpha_mu(alpha, mu);
	// 
	// 	std::cout << "alpha mu" << std::endl;
	// 	for (int kk = 0; kk < n_var; kk++){
	// 		std::cout << alpha(kk) << " " << mu(kk) << std::endl;
	// 	}
	// }

	// SECTION( "Check vbayes.calc_logw" ) {
	// 	std::string dir = "data/io_test/t7_varbvs_sub/";
	// 	data Data( "data/io_test/t7_varbvs_sub/n500_p1000.bgen" );
	// 	Data.params.pheno_file = dir + "pheno.txt";
	// 	Data.params.alpha_file = dir + "alpha_init.txt";
	// 	Data.params.mu_file = dir + "mu_init.txt";
	// 
	// 	Data.read_pheno();
	// 	Data.params.chunk_size = 1100;
	// 	Data.read_bgen_chunk();
	// 	Data.read_alpha_mu();
	// 
	// 	vbayes VB(Data.G, Data.Y);
	// 	double sigma = 1.0;
	// 	double sigmab = 1.0;
	// 	double q = 0.01;
	// 	double obs;
	// 
	// 	std::vector< double > s_sq;
	// 	s_sq.resize(VB.n_var);
	// 	for (int kk = 0; kk < VB.n_var; kk++){
	// 		s_sq[kk] = sigmab * sigma / (sigmab * VB.dXtX(kk) + 1.0);
	// 	}
	// 
	// 	obs = VB.calc_logw(sigma, sigmab, q, s_sq, Data.alpha_init, Data.mu_init);
	// 	CHECK(Approx(obs) == -1169.272);
	// }
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

	GenotypeMatrix GM(2, 3);
	GM.assign_index(0, 0, 0.2);
	GM.assign_index(0, 1, 0.2);
	GM.assign_index(0, 2, 0.345);
	GM.assign_index(1, 0, 0.8);
	GM.assign_index(1, 1, 0.3);
	GM.assign_index(1, 2, 0.213);

	// Eigen::MatrixXd GM_uint(2,3);
	// GM_uint << 51, 51, 8, 204, 76, 54;

	SECTION("Empty constructor"){
		GenotypeMatrix GM2;
		GM2.resize(2, 3);
		GM2.assign_index(0, 0, 0.2);
		CHECK(GM2.read_index(0,0) == GM.read_index(0,0));
	}

	SECTION("Decompressed matrix multiplication"){
		Eigen::VectorXd vv(3);
		vv << 0.3, 0.55, 0.676;
		Eigen::VectorXd res = GM * vv;
		Eigen::VectorXd res_truth(2);
		res_truth << 0.4046914, 0.5479180;
		CHECK(Approx(res.sum()) == res_truth.sum());
	}

	SECTION("Read column access"){
		Eigen::VectorXd c2a, c2b, c2_truth(2);
		c2_truth << 0.201172, 0.298828;
		// CHECK_THROWS(GM.col(1, c2a));
		c2a.resize(2);
		GM.col(1, c2a);
		c2b = GM.col(1);
		CHECK(Approx(c2a.sum()) == c2_truth.sum());
		CHECK(Approx(c2b.sum()) == c2_truth.sum());
	}

	SECTION("Write column access"){
		// write 2nd column to 1st column and check equivalent
		Eigen::VectorXd c2, post_c1;
		c2 = GM.col(1);
		GM.assign_col(0, c2);
		post_c1 = GM.col(0);

		CHECK(post_c1 == c2);
	}
	
}

// TEST_CASE( "Checking data", "[data]" ) {
// 	char* argv[] = { (char*) "bin/bgen_prog",
// 					 (char*) "--bgen", 
// 					 (char*) "data/example/example.v11.bgen"};
// 	int argc = sizeof(argv)/sizeof(argv[0]);
// 	parameters p;
// 	parse_arguments(p, argc, argv);
// 	data Data( p.bgen_file );
// 
// 	SECTION( "read_txt_file behaving sensibly" ) {
// 		std::string filename = "test/cases/test_covar.txt";
// 		Eigen::MatrixXd M_read, M_ans(3, 3), M_cent(3, 3);
// 		int n_cols;
// 		std::vector< std::string > col_names_read, col_names_ans;
// 		std::map< int, bool > incomplete_row;
// 
// 		Data.n_samples = 3;
// 		Data.read_txt_file( filename,
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
// 			Data.n_samples = 2;
// 			CHECK_THROWS_AS(Data.read_txt_file( filename,
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
// 			Data.center_matrix( M_read, n_cols_ans );
// 			Data.scale_matrix( M_read, n_cols_ans );
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
// 		// 	Data.reduce_mat_to_complete_cases(M_read, check, n_cols_ans, incomplete_cases);
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
// 		// 	Data.W = W;
// 		// 	Data.covar_names = covar_names;
// 		// 	Data.n_col = 3;
// 		// 	Data.scale_matrix(Data.W, Data.n_col, Data.covar_names);
// 		// 	CHECK( Data.covar_names == red_covar_names );
// 		// 	CHECK( Data.n_col == 2 );
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
// 		data Data( p.bgen_file );
// 		Data.params = p;
// 		
// 		CHECK(Data.params.incl_sids_file == "data/test/t1_incl_sample_ids/valid_ids.txt");
// 		Data.read_incl_sids();
// 		CHECK(Data.n_samples - Data.incomplete_cases.size() == 7);
// 		Data.read_covar();
// 		Data.reduce_to_complete_cases(); // Also edits n_samples
// 		CHECK(Data.n_samples == 7);
// 		Eigen::MatrixXd C(7,1);
// 		C << 1, 3, 17, 21, 22, 25, 500;
// 		CHECK(Data.W == C);
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
// 		data Data( p.bgen_file );
// 		Data.params = p;
// 		
// 		CHECK_THROWS_AS(Data.read_incl_sids(), std::logic_error);
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
// 		data Data( p.bgen_file );
// 		Data.params = p;
// 		
// 		CHECK_THROWS_AS(Data.read_incl_sids(), std::logic_error);
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
// 		data Data( p.bgen_file );
// 		Data.params = p;
// 		
// 		CHECK_THROWS_AS(Data.read_incl_sids(), std::logic_error);
// 	}
// 
// }
