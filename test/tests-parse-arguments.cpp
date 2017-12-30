// tests-factorial.cpp
#include "catch.hpp"

#include <iostream>
#include "../src/parse_arguments.hpp"

TEST_CASE( "Checking parse_arguments", "[io]" ) {
	parameters p;
	
	SECTION( "BGEN filepath correctly assigned" ) {
		char* argv[] = { (char*) "bin/bgen_prog",
						 (char*) "--bgen", 
						 (char*) "data/raw/example.v11.bgen"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		REQUIRE(p.bgen_file == "data/raw/example.v11.bgen");
	}
	
	// Annoyingly no death test functionality exists within the Catch2 framework
	// SECTION( "Error caused by invalid flag" ) {
	// 	const char* argv[] = { "bin/bgen_prog", "--hello", "you"};
	// 	int argc = sizeof(argv)/sizeof(argv[0]);
	// 	parse_arguments(p, argc, argv);
	// }
}
