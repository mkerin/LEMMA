//
// Created by kerin on 2019-06-01.
//
#include "catch.hpp"

#include "../src/tools/eigen3.3/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/vbayes.hpp"
#include "../src/data.hpp"
#include "../src/hyps.hpp"
#include "../src/genotype_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <sys/stat.h>

TEST_CASE("Data") {
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	parameters p;
	p.bgen_file = "unit/data/n50_p100.bgen";
	p.bgi_file = "unit/data/n50_p100.bgen.bgi";
	p.env_file = "unit/data/n50_p100_env.txt";
	p.pheno_file = "unit/data/pheno.txt";
	Data data(p);

	data.read_non_genetic_data();
	SECTION("Ex1. Raw non genetic data read in accurately") {
		CHECK(data.n_env == 4);
		CHECK(data.n_pheno == 1);
		CHECK(data.Nglobal == 50);
		if(world_rank == 0) {
			CHECK(data.Y(0, 0) == Approx(-1.18865038973338));
			CHECK(data.E(0, 0) == Approx(0.785198212));
		}
	}

	data.standardise_non_genetic_data();
	SECTION("Check non genetic data standardised") {
		CHECK(data.p.covar_file == "NULL");
		if(world_rank == 0) {
			CHECK(data.Y(0, 0) == Approx(-1.5800573524786081));
			CHECK(data.Y2(0, 0) == Approx(-1.5567970303));
			CHECK(data.E(0, 0) == Approx(0.8957059881));
		}
	}

	data.read_full_bgen();
	SECTION("bgen read in & standardised correctly") {
		CHECK(data.G.compressed_dosage_means(60) == Approx(1.4796875));
		CHECK(data.G.compressed_dosage_means(61) == Approx(0.3034375));
		CHECK(data.G.compressed_dosage_means(62) == Approx(0.42375));
		CHECK(data.G.compressed_dosage_means(63) == Approx(1.59890625));
		CHECK(data.n_var == 69);
	}
}
