// tests-main.cpp
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
#include <sys/stat.h>
#include "../src/parse_arguments.hpp"
#include "../src/data.hpp"

TEST_CASE( "Checking parse_arguments", "[io]" ) {
	parameters p;
	
	SECTION( "BGEN filepath correctly assigned" ) {
		char* argv[] = { (char*) "bin/bgen_prog",
						 (char*) "--bgen", 
						 (char*) "data/example/example.v11.bgen"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		REQUIRE(p.bgen_file == "data/example/example.v11.bgen");
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
			REQUIRE( M_read == M_cent );
		}
	}
}
