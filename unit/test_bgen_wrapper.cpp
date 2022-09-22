// tests-main.cpp
#include "catch.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include "../src/bgen_wrapper.hpp"

void assert_file_exists(const std::string &filename);

TEST_CASE("BgenWrapper"){
	SECTION("Check files exists"){
		std::string bgenFile = "unit/data/n50_p100.bgen";
		assert_file_exists(bgenFile); // this fails if the tests are being run from the wrong directory
		bgenWrapper::check_file_exists(bgenFile,".bgen"); // failure here indicates a problem with boost::filesystem
	}
	SECTION("Create bgenWrapper::View"){
		std::string bgenFile = "unit/data/n50_p100.bgen";
		bgenWrapper::View view(bgenFile);
		view.summarise(std::cout);
		std::size_t n_samples = view.number_of_samples();
		std::uint32_t n_var = view.number_of_variants();
		std::vector<std::string> ids = view.get_sample_ids();
	}

	SECTION("Create IndexQuery"){
		std::string bgiFile = "unit/data/n50_p100.bgen.bgi";
		bgenWrapper::IndexQuery query(bgiFile);
		std::string chr = "1";
		std::uint32_t start = 1, end = 50;
		query.include_range(chr, start, end);
		std::vector< std::string > rsid = {"rs116720794:729632:C:T", "rs3131972:752721:A:G"};
		query.include_rsids( rsid );
		query.initialise();
	}

	SECTION("Read bgen data"){
		std::string chr_j, rsid_j, SNPID_j;
		std::uint32_t pos_j;
		std::vector< std::string > alleles_j;
		std::string bgenFile = "unit/data/n50_p100.bgen";
		bgenWrapper::View view(bgenFile);

		// Initialise dosageSetter
		std::size_t n_samples = view.number_of_samples();
		std::unordered_map<long, bool> sample_is_invalid;
		for (long ii = 0; ii < n_samples; ii++) {
			sample_is_invalid[ii] = false;
		}
		EigenDataVector dosage_j(n_samples);

		// Read data for single variant
		bool success = view.read_variant( &SNPID_j, &rsid_j, &chr_j, &pos_j, &alleles_j );
		CHECK(success);
		auto stats = view.read_genotype_data_block( sample_is_invalid, n_samples, dosage_j );
	}

	SECTION("Create & set IndexQuery"){
		std::string bgenFile = "unit/data/n50_p100.bgen";
		std::string bgiFile = "unit/data/n50_p100.bgen.bgi";
		bgenWrapper::View view(bgenFile);
		bgenWrapper::IndexQuery query(bgiFile);
		std::vector< std::string > rsid = {"rs116720794:729632:C:T", "rs3131972:752721:A:G"};
		query.include_rsids( rsid );
		query.initialise();
		view.set_query(query);
		std::uint32_t n_var = view.number_of_variants();
		CHECK(n_var == 2);
	}
}

void assert_file_exists(const std::string &filename) {
	// Throw error if given file does not exist.
	// NB: Doesn't check if file is empty etc.
   std::ifstream ifile;
   ifile.open(filename);
   if(!ifile) {
      throw std::invalid_argument("File '"+filename+"' does not exist");
   }
}