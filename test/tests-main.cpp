// tests-main.cpp
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
#include <sys/stat.h>
#include "../src/parse_arguments.hpp"
#include "../src/data.hpp"

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
		REQUIRE(p.bgen_file == "data/example/example.v11.bgen");
		REQUIRE(p.out_file == "data/tmp.out");
		REQUIRE(p.mode_vcf == true);
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
						 (char*) "data/test/empty.covar"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		REQUIRE(p.bgen_file == "data/example/example.v11.bgen");
		REQUIRE(p.out_file == "data/tmp.out");
		REQUIRE(p.pheno_file == "data/test/empty.pheno");
		REQUIRE(p.covar_file == "data/test/empty.covar");
		REQUIRE(p.mode_lm == true);
	}

	SECTION( "chunk_size correctly assigned" ) {
		char* argv[] = { (char*) "bin/bgen_prog",
						 (char*) "--chunk", 
						 (char*) "100"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		REQUIRE(p.chunk_size == 100);
	}
	
	// Annoyingly no death test functionality exists within the Catch2 framework
	// SECTION( "Error caused by invalid flag" ) {
	// 	const char* argv[] = { "bin/bgen_prog", "--hello", "you"};
	// 	int argc = sizeof(argv)/sizeof(argv[0]);
	// 	parse_arguments(p, argc, argv);
	// }
}

TEST_CASE( "Checking data", "[data]" ) {
	char* argv[] = { (char*) "bin/bgen_prog",
					 (char*) "--bgen", 
					 (char*) "data/example/example.v11.bgen"};
	int argc = sizeof(argv)/sizeof(argv[0]);
	parameters p;
	parse_arguments(p, argc, argv);
	data Data( p.bgen_file );

	SECTION( "read_txt_file behaving sensibly" ) {
		std::string filename = "test/cases/test_covar.txt";
		Eigen::MatrixXd M_read, M_ans(3, 3), M_cent(3, 3);
		int n_cols;
		std::vector< std::string > col_names_read, col_names_ans;
		std::map< int, bool > incomplete_row;

		Data.n_samples = 3;
		Data.read_txt_file( filename,
							M_read,
							n_cols,
							col_names_read,
							incomplete_row );

		M_ans << 1, 2, 3, 4, 5, 6, 7, 8, 9;
		col_names_ans.push_back("A");
		col_names_ans.push_back("B");
		col_names_ans.push_back("C");

		REQUIRE( M_read == M_ans );
		REQUIRE( n_cols == 3 );
		REQUIRE( col_names_read == col_names_ans );

		SECTION( "Number of covariates greater than n_samples" ) {
			Data.n_samples = 2;
			REQUIRE_THROWS_AS(Data.read_txt_file( filename,
								M_read,
								n_cols,
								col_names_read,
								incomplete_row ), std::runtime_error);
		}

		SECTION( "center_matrix behaving sensibly" ) {
			Eigen::MatrixXd M_cent(3, 3);
			int n_cols_ans = 3;
			M_cent << -1, -1, -1, 0, 0, 0, 1, 1, 1;

			Data.center_matrix( M_read, n_cols_ans, incomplete_row );
			Data.scale_matrix( M_read, n_cols_ans, incomplete_row );
			REQUIRE( M_read == M_cent );
		}

		SECTION( "reduce_mat_to_complete_cases behaving sensibly" ) {
			Eigen::MatrixXd M_reduc(2, 3);
			int n_cols_ans = 3;
			bool check = false;
			std::map< int, bool > incomplete_cases;
			M_reduc << 1, 2, 3, 7, 8, 9;
			incomplete_cases[1] = 1;

			Data.reduce_mat_to_complete_cases(M_read, check, n_cols_ans, incomplete_cases);
			// std::cout << "M_read is now " << M_read.rows() << "x" << M_read.cols() << std::endl;
			REQUIRE( M_read == M_reduc );
		}
	}
}

TEST_CASE( "Check incl_sample_ids", "[data]" ) {
	parameters p;

	SECTION( "Read in sids correctly and subset covar matrix" ) {
		char* argv[] = { (char*) "bin/bgen_prog",
						 (char*) "--bgen", 
						 (char*) "data/test/example.v11.bgen",
						 (char*) "--covar", 
						 (char*) "data/test/t1_incl_sample_ids/valid_ids.covar",
						 (char*) "--incl_sample_ids", 
						 (char*) "data/test/t1_incl_sample_ids/valid_ids.txt"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		data Data( p.bgen_file );
		Data.params = p;
		
		REQUIRE(Data.params.incl_sids_file == "data/test/t1_incl_sample_ids/valid_ids.txt");
		Data.read_incl_sids();
		REQUIRE(Data.n_samples - Data.incomplete_cases.size() == 7);
		Data.read_covar();
		Data.reduce_to_complete_cases(); // Also edits n_samples
		REQUIRE(Data.n_samples == 7);
		Eigen::MatrixXd C(7,1);
		C << 1, 3, 17, 21, 22, 25, 500;
		REQUIRE(Data.W == C);
	}

	SECTION( "Empty incl_sample_ids file throws error" ) {
		char* argv[] = { (char*) "bin/bgen_prog",
						 (char*) "--bgen", 
						 (char*) "data/test/example.v11.bgen",
						 (char*) "--incl_sample_ids", 
						 (char*) "data/test/t1_incl_sample_ids/empty.txt"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		REQUIRE_THROWS_AS(parse_arguments(p, argc, argv), std::runtime_error);
	}
	
	SECTION( "Unordered sids throw error" ) {
		char* argv[] = { (char*) "bin/bgen_prog",
						 (char*) "--bgen", 
						 (char*) "data/test/example.v11.bgen",
						 (char*) "--incl_sample_ids", 
						 (char*) "data/test/t1_incl_sample_ids/invalid_ids1.txt"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		data Data( p.bgen_file );
		Data.params = p;
		
		REQUIRE_THROWS_AS(Data.read_incl_sids(), std::logic_error);
	}

	SECTION( "Absent sid throws error" ) {
		char* argv[] = { (char*) "bin/bgen_prog",
						 (char*) "--bgen", 
						 (char*) "data/test/example.v11.bgen",
						 (char*) "--incl_sample_ids", 
						 (char*) "data/test/t1_incl_sample_ids/invalid_ids2.txt"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		data Data( p.bgen_file );
		Data.params = p;
		
		REQUIRE_THROWS_AS(Data.read_incl_sids(), std::logic_error);
	}

	SECTION( "Absent sid throws error including last sid" ) {
		char* argv[] = { (char*) "bin/bgen_prog",
						 (char*) "--bgen", 
						 (char*) "data/test/example.v11.bgen",
						 (char*) "--incl_sample_ids", 
						 (char*) "data/test/t1_incl_sample_ids/invalid_ids3.txt"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		data Data( p.bgen_file );
		Data.params = p;
		
		REQUIRE_THROWS_AS(Data.read_incl_sids(), std::logic_error);
	}

}
