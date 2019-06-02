//
// Created by kerin on 2019-06-01.
//
#define EIGEN_USE_MKL_ALL
#include "catch.hpp"

#include "../src/tools/eigen3.3/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/vbayes_x2.hpp"
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

	p.env_file = "data/io_test/n50_p100_env.txt";
	p.pheno_file = "data/io_test/pheno.txt";

	SECTION("Check envs") {
		p.bgen_file = "data/io_test/n50_p100.bgen";
		p.bgi_file = "data/io_test/n50_p100.bgen.bgi";
		p.low_mem = true;
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
		SECTION("Check non genetic data standardised + covars regressed") {
			CHECK(data.p.scale_pheno);
			CHECK(data.p.use_vb_on_covars);
			CHECK(data.p.covar_file == "NULL");
			if(world_rank == 0) {
				CHECK(data.Y(0, 0) == Approx(-1.5800573524786081));
				CHECK(data.Y2(0, 0) == Approx(-1.5567970303));
				CHECK(data.E(0, 0) == Approx(0.8957059881));
			}
		}
	}

	SECTION("n50_p100.bgen (low mem), covars, sample subset") {
		p.covar_file = "data/io_test/age.txt";
		p.bgen_file = "data/io_test/n50_p100.bgen";
		p.bgi_file = "data/io_test/n50_p100.bgen.bgi";
		p.incl_sids_file = "data/io_test/sample_ids_head28.txt";
		p.low_mem = true;
		Data data(p);

		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		SECTION("Ex1. bgen read in & standardised correctly") {
			data.G.calc_scaled_values();
			CHECK(data.G.low_mem);
			CHECK(data.p.low_mem);
			CHECK(!data.p.flip_high_maf_variants);
			CHECK(data.G.compressed_dosage_means(0) == Approx(1.2979910714));
			CHECK(data.G.compressed_dosage_means(50) == Approx(0.3610491071));
			CHECK(data.n_var == 54);
		}

		SECTION("dXtEEX computed correctly") {
			data.calc_dxteex();
			CHECK(data.dXtEEX_lowertri(0, 0) == Approx(23.2334219303));
			CHECK(data.dXtEEX_lowertri(1, 0) == Approx(27.9920667408));
			CHECK(data.dXtEEX_lowertri(2, 0) == Approx(24.7041225993));
			CHECK(data.dXtEEX_lowertri(3, 0) == Approx(24.2423580715));

			CHECK(data.dXtEEX_lowertri(0, dXtEEX_col_ind(1, 0, data.n_env)) == Approx(-1.056112897));
			CHECK(data.dXtEEX_lowertri(1, dXtEEX_col_ind(1, 0, data.n_env)) == Approx(-8.526431457));
			CHECK(data.dXtEEX_lowertri(2, dXtEEX_col_ind(1, 0, data.n_env)) == Approx(-6.5950206611));
			CHECK(data.dXtEEX_lowertri(3, dXtEEX_col_ind(1, 0, data.n_env)) == Approx(-3.6842212598));
		}
	}

	SECTION("n50_p100.bgen (low mem)") {
		p.bgen_file = "data/io_test/n50_p100.bgen";
		p.bgi_file = "data/io_test/n50_p100.bgen.bgi";
		p.low_mem = true;
		Data data(p);

		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		data.G.calc_scaled_values();
		SECTION("Ex1. bgen read in & standardised correctly") {
			CHECK(data.G.low_mem);
			CHECK(data.p.low_mem);
			CHECK(!data.p.flip_high_maf_variants);
			CHECK(data.G.compressed_dosage_means(60) == Approx(1.18140625));
			CHECK(data.G.compressed_dosage_means(61) == Approx(0.30359375));
			CHECK(data.G.compressed_dosage_means(62) == Approx(0.30390625));
			CHECK(data.G.compressed_dosage_means(63) == Approx(0.715));
			CHECK(data.n_var == 67);
		}

		SECTION("dXtEEX computed correctly") {
			data.calc_dxteex();
			CHECK(data.dXtEEX_lowertri(0, 0) == Approx(38.9610805993));
			CHECK(data.dXtEEX_lowertri(1, 0) == Approx(38.2995451744));
			CHECK(data.dXtEEX_lowertri(2, 0) == Approx(33.7077899144));
			CHECK(data.dXtEEX_lowertri(3, 0) == Approx(35.7391671158));

			CHECK(data.dXtEEX_lowertri(0, dXtEEX_col_ind(1, 0, data.n_env)) == Approx(-2.6239467101));
			CHECK(data.dXtEEX_lowertri(1, dXtEEX_col_ind(1, 0, data.n_env)) == Approx(-13.0001255314));
			CHECK(data.dXtEEX_lowertri(2, dXtEEX_col_ind(1, 0, data.n_env)) == Approx(-11.6635557299));
			CHECK(data.dXtEEX_lowertri(3, dXtEEX_col_ind(1, 0, data.n_env)) == Approx(-7.2154836264));
		}

		SECTION("Ex1. Confirm calc_dxteex() reorders properly") {
			data.p.dxteex_file = "data/io_test/n50_p100_dxteex_low_mem.txt";
			data.calc_dxteex();
			CHECK(data.dXtEEX_lowertri(0, 0) == Approx(38.9610805993));
			CHECK(data.dXtEEX_lowertri(1, 0) == Approx(38.2995451744));
			CHECK(data.dXtEEX_lowertri(2, 0) == Approx(33.7077899144));
			CHECK(data.dXtEEX_lowertri(3, 0) == Approx(35.7391671158));

			CHECK(data.dXtEEX_lowertri(0, dXtEEX_col_ind(1, 0, data.n_env)) == Approx(-2.6239467101));
			CHECK(data.dXtEEX_lowertri(1, dXtEEX_col_ind(1, 0, data.n_env)) == Approx(-13.0001255314));
			CHECK(data.dXtEEX_lowertri(2, dXtEEX_col_ind(1, 0, data.n_env)) == Approx(-11.6635557299));
			CHECK(data.dXtEEX_lowertri(3, dXtEEX_col_ind(1, 0, data.n_env)) == Approx(-7.2154836264));
			CHECK(data.n_dxteex_computed == 1);
		}
	}

	SECTION("n50_p100_chr2.bgen") {
		p.bgen_file = "data/io_test/n50_p100_chr2.bgen";
		p.bgi_file = "data/io_test/n50_p100_chr2.bgen.bgi";
		Data data(p);

		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		data.G.calc_scaled_values();
		SECTION("Ex1. bgen read in & standardised correctly") {
			CHECK(data.G.low_mem);
			CHECK(data.p.low_mem);
			CHECK(!data.p.flip_high_maf_variants);
			CHECK(data.G.compressed_dosage_means(60) == Approx(1.00203125));
			CHECK(data.G.compressed_dosage_means(61) == Approx(0.9821875));
			CHECK(data.G.compressed_dosage_means(62) == Approx(0.10390625));
			CHECK(data.G.compressed_dosage_means(63) == Approx(0.68328125));
			CHECK(data.n_var == 75);
		}
	}

	SECTION("Check mult_vector_by_chr"){
		p.bgen_file = "data/io_test/n50_p100_chr2.bgen";
		p.bgi_file = "data/io_test/n50_p100_chr2.bgen.bgi";
		Data data(p);

		data.read_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd vv = Eigen::VectorXd::Ones(data.G.pp);
		Eigen::VectorXd v1 = data.G.mult_vector_by_chr(1, vv);
		Eigen::VectorXd v2 = data.G.mult_vector_by_chr(22, vv);

		if(world_rank == 0) {
			CHECK(v1(0) == Approx(-0.8981400368));
			CHECK(v1(1) == Approx(-4.9936547948));
			CHECK(v1(2) == Approx(-1.7085924856));
			CHECK(v1(3) == Approx(0.8894016653));

			CHECK(v2(0) == Approx(-10.8022318897));
			CHECK(v2(1) == Approx(11.658910645));
			CHECK(v2(2) == Approx(-16.742754449));
			CHECK(v2(3) == Approx(0.9656298668));
		}
	}
}
