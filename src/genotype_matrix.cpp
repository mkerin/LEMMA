//
// Created by kerin on 2019-03-01.
//

#include "genotype_matrix.hpp"

#include "my_timer.hpp"
#include "parameters.hpp"
#include "typedefs.hpp"
#include "mpi_utils.hpp"

#include "tools/eigen3.3/Dense"
#include <algorithm>
#include <iostream>
#include <limits>
#include <cstdint>               // uint32_t
#include <cmath>
#include <random>
#include <thread>
#include <vector>
#include <map>

void GenotypeMatrix::assign_index(const long &ii, const long &jj, double x) {
	if(low_mem) {
		if(std::isnan(x)) {
			throw std::runtime_error("ERROR: missing values not yet compatible in low-mem mode");
		}
		M(ii, jj) = CompressDosage(x);
	} else {
		if(std::isnan(x)) {
			missing_genos[jj][ii] = 1;
		} else {
			G(ii, jj) = x;
		}
	}
	scaling_performed = false;
}

#ifdef DATA_AS_FLOAT
#else
void GenotypeMatrix::col(long jj, EigenRefDataVector vec) {
	assert(jj < pp);
	if(!scaling_performed) {
		calc_scaled_values();
	}

	if(low_mem) {
		vec = M.cast<double>().col(jj);
		vec *= (intervalWidth * compressed_dosage_inv_sds[jj]);
		vec.array() += (0.5 * intervalWidth - compressed_dosage_means[jj]) * compressed_dosage_inv_sds[jj];
	} else {
		vec = G.col(jj);
	}
}
#endif

Eigen::MatrixXd GenotypeMatrix::transpose_multiply(EigenRefDataMatrix lhs) const {
	// Return G^t lhs
	assert(lhs.rows() == nn);
	assert(scaling_performed);

	if(low_mem) {
		if(params.mode_debug) std::cout << "Starting transpose multiply" << std::endl;
		Eigen::MatrixXd res;
		Eigen::VectorXd colsums = lhs.colwise().sum().matrix().cast<double>();

		// Diagnostic messages as worried about RAM
		Eigen::MatrixXd Mt_lhs(pp, lhs.cols());
		for (int ll = 0; ll < lhs.cols(); ll++) {
			EigenRefDataVector tmp = lhs.col(ll);
			Mt_lhs.col(ll) = tmp.cast<double>().transpose() * M.cast<double>();
		}

		res = intervalWidth * (compressed_dosage_inv_sds.asDiagonal() * Mt_lhs);
		res += 0.5 * intervalWidth * compressed_dosage_inv_sds * colsums.transpose();
		res -= compressed_dosage_inv_sds.cwiseProduct(compressed_dosage_means) * colsums.transpose();
		if(params.mode_debug) std::cout << "Ending transpose multiply" << std::endl;
		return res;
	} else {
		return (G.transpose() * lhs.matrix()).cast<double>();
	}
}

EigenDataMatrix GenotypeMatrix::col_block(const std::uint32_t &ch_start, const int &ch_len) {
	if(!scaling_performed) {
		calc_scaled_values();
	}

	if(low_mem) {
		double ww = intervalWidth;

		EigenDataArrayX E = (0.5 * ww - compressed_dosage_means.segment(ch_start, ch_len).array()).cast<scalarData>();
		EigenDataArrayX S = compressed_dosage_inv_sds.segment(ch_start, ch_len).cast<scalarData>();
		EigenDataArrayXX res;

		res = ww * M.block(0, ch_start, nn, ch_len).cast<scalarData>();
		res.rowwise() += E.transpose();
		res.rowwise() *= S.transpose();
		return res.matrix();
	} else {
		return G.block(0, ch_start, nn, ch_len);
	}
}

Eigen::VectorXd GenotypeMatrix::mult_vector_by_chr(const long& chr, const Eigen::Ref<const Eigen::VectorXd> &rhs) {
	// (y^t G)^t <=> G^t y
	assert(rhs.rows() == pp);
	if(!scaling_performed) {
		calc_scaled_values();
	}

	// Find chr block
	// Chromosomes read via bgen guaranteed to be ordered in blocks
	std::vector<int> tmp;
	tmp.push_back(chr);
	auto it_start = std::find(chromosome.begin(), chromosome.end(), chr);
	auto it_end = std::find_end(chromosome.begin(), chromosome.end(), tmp.begin(), tmp.end());
	long chr_st, chr_en, chr_size;
	chr_st = it_start - chromosome.begin();
	chr_en = it_end - chromosome.begin();
	chr_size = chr_en - chr_st + 1;

	Eigen::VectorXd res;
	if(low_mem) {
		Eigen::VectorXd rhs_trans = rhs.cwiseProduct(compressed_dosage_inv_sds);
		auto offset = rhs_trans.segment(chr_st, chr_size).sum() * intervalWidth * 0.5;
		offset -= compressed_dosage_means.segment(chr_st, chr_size).dot(rhs_trans.segment(chr_st, chr_size));

		res = M.block(0, chr_st, nn, chr_size).cast<double>() * rhs_trans.segment(chr_st, chr_size);
		return (res.array() * intervalWidth + offset).matrix();
	} else {
		return G.block(0, chr_st, nn, chr_size).cast<double>() * rhs.segment(chr_st, chr_size);
	}
}

template<typename Deriv>
void GenotypeMatrix::col_block3(const std::vector<long> &chunk,
                                Eigen::MatrixBase<Deriv> &D) {
	if(!scaling_performed) {
		calc_scaled_values();
	}

	// Partition jobs amongst threads
	long ch_len = chunk.size();
	std::vector<std::vector<long> > indexes(params.n_thread);
	for (long ii = 0; ii < ch_len; ii++) {
		// indexes[ii % params.n_thread].push_back(ii);
		indexes[0].push_back(ii);
	}

	// Decompress char -> double
// #ifdef DEBUG
	get_cols(indexes[0], chunk, D);
	// for (int nn = 1; nn < params.n_thread; nn++){
	//  get_cols(indexes[nn], chunk, D);
	// }
// #else
//      std::thread t1[params.n_thread];
//      for (int nn = 1; nn < params.n_thread; nn++){
//          t1[nn] = std::thread( [this, &indexes, nn, &chunk, &D] {
//              get_cols(indexes[nn], chunk, D);
//          });
//      }
//      get_cols(indexes[0], chunk, D);
//      for (int nn = 1; nn < params.n_thread; nn++){
//          t1[nn].join();
//      }
// #endif
}

template<typename Deriv>
void GenotypeMatrix::get_cols(const std::vector<long> &index,
                              const std::vector<long> &iter_chunk,
                              Eigen::MatrixBase<Deriv> &D) {
	// D.col(ii) = X.col(chunk(ii))
	for(const auto& ii : index ) {
		long jj = (iter_chunk[ii] % pp);
//			D.col(ii) = col(jj);
		col(jj, D.col(ii));
	}
}

void GenotypeMatrix::calc_scaled_values() {
	if (low_mem) {
		compute_means_and_sd();
	} else {
		standardise_matrix();
	}
	scaling_performed = true;
}

void GenotypeMatrix::compute_means_and_sd() {
	double Nlocal = nn;
	double Nglobal = mpiUtils::mpiReduce_inplace(&Nlocal);

	// Column means
	for (Index jj = 0; jj < pp; jj++) {
		double sum_mij = 0;
		compressed_dosage_means[jj] = 0;
		for (Index ii = 0; ii < nn; ii++) {
			sum_mij += DecompressDosage(M(ii, jj));
		}
		sum_mij = mpiUtils::mpiReduce_inplace(&sum_mij);
		compressed_dosage_means[jj] = sum_mij / Nglobal;
	}


	// Column standard deviation
	double val, sigma;
	for (Index jj = 0; jj < pp; jj++) {
		sigma = 0;
		for (Index ii = 0; ii < nn; ii++) {
			val = DecompressDosage(M(ii, jj)) - compressed_dosage_means[jj];
			sigma += val * val;
		}
		sigma = mpiUtils::mpiReduce_inplace(&sigma);
		compressed_dosage_sds[jj] = std::sqrt(sigma / (Nglobal - 1));
	}

//	compressed_dosage_sds /= ((double) nn - 1.0);
//	compressed_dosage_sds = compressed_dosage_sds.array().sqrt().matrix();

	for (Index jj = 0; jj < pp; jj++) {
		sigma = compressed_dosage_sds[jj];
		if (sigma > 1e-9) {
			compressed_dosage_inv_sds[jj] = 1 / sigma;
		} else {
			compressed_dosage_inv_sds[jj] = 0.0;
		}
	}
}

void GenotypeMatrix::standardise_matrix() {
	if(!low_mem) {
		for (std::size_t k = 0; k < pp; k++) {
			double mu = 0.0;
			double count = 0;
			for (std::size_t i = 0; i < nn; i++) {
				if (missing_genos[k].count(i) == 0) {
					mu += G(i, k);
					count += 1;
				}
			}

			mu = mu / count;
			double val, sigma = 0.0;
			for (std::size_t i = 0; i < nn; i++) {
				if (missing_genos[k].count(i) == 0) {
					G(i, k) -= mu;
					val = G(i, k);
					sigma += val * val;
				} else {
					G(i, k) = 0.0;
				}
			}

			sigma = sqrt(sigma/(count - 1));
			if (sigma > 1e-12) {
				for (std::size_t i = 0; i < nn; i++) {
					G(i, k) /= sigma;
				}
			}
		}
	}
}

void GenotypeMatrix::resize(const long &n, const long &p) {
	if(low_mem) {
		M.resize(n, p);
	} else {
		G.resize(n, p);
	}
	compressed_dosage_means.resize(p);
	compressed_dosage_sds.resize(p);
	compressed_dosage_inv_sds.resize(p);
	missing_genos.resize(p);
	nn = n;
	pp = p;

	al_0.resize(p);
	al_1.resize(p);
	maf.resize(p);
	info.resize(p);
	rsid.resize(p);
	chromosome.resize(p);
	position.resize(p);
	SNPKEY.resize(p);
	SNPID.resize(p);
}

void GenotypeMatrix::move_variant(std::uint32_t old_index, std::uint32_t new_index) {
	// If shuffle variants + assoicated information left if we decide to
	// exclude from analysis.
	assert(new_index < old_index);

	if(low_mem) {
		M.col(new_index) = M.col(old_index);
	} else {
		G.col(new_index) = G.col(old_index);
	}

//		std::cout << "Moving " << rsid[old_index] << " from " << old_index << " to " << new_index << std::endl;

	al_0[new_index]       = al_0[old_index];
	al_1[new_index]       = al_1[old_index];
	maf[new_index]        = maf[old_index];
	info[new_index]       = info[old_index];
	rsid[new_index]       = rsid[old_index];
	chromosome[new_index] = chromosome[old_index];
	position[new_index]   = position[new_index];
	SNPKEY[new_index]     = SNPKEY[old_index];
	SNPID[new_index]      = SNPID[old_index];
}

void GenotypeMatrix::conservativeResize(const long &n, const long &p) {
	if(low_mem) {
		M.conservativeResize(n, p);
	} else {
		G.conservativeResize(n, p);
	}
	compressed_dosage_means.conservativeResize(p);
	compressed_dosage_sds.conservativeResize(p);
	compressed_dosage_inv_sds.conservativeResize(p);
	missing_genos.resize(p);
	nn = n;
	pp = p;

	al_0.resize(p);
	al_1.resize(p);
	maf.resize(p);
	info.resize(p);
	rsid.resize(p);
	chromosome.resize(p);
	position.resize(p);
	SNPKEY.resize(p);
	SNPID.resize(p);
}

Eigen::VectorXd GenotypeMatrix::col(long jj) {
	assert(jj < pp);
	Eigen::VectorXd vec(nn);
	if(!scaling_performed) {
		calc_scaled_values();
	}

	if(low_mem) {
		vec = M.cast<double>().col(jj);
		vec *= (intervalWidth * compressed_dosage_inv_sds[jj]);
		vec.array() += (0.5 * intervalWidth - compressed_dosage_means[jj]) * compressed_dosage_inv_sds[jj];
	} else {
		vec = G.col(jj);
	}
	return vec;
}

EigenDataMatrix GenotypeMatrix::operator*(EigenRefDataMatrix rhs) const {
	assert(scaling_performed);
	assert(rhs.rows() == pp);

	if(low_mem) {
		EigenDataMatrix res(nn, rhs.cols());
		for (int ll = 0; ll < rhs.cols(); ll++) {
			EigenRefDataVector tmp = rhs.col(ll);
			res.col(ll) = M.cast<scalarData>() * compressed_dosage_inv_sds.cast<scalarData>().asDiagonal() * tmp.cast<scalarData>();
		}
		res *= intervalWidth;
		// res = M.cast<scalarData>() * compressed_dosage_inv_sds.cast<scalarData>().asDiagonal() * rhs * intervalWidth;
		res.array().rowwise() += (compressed_dosage_inv_sds.cast<scalarData>().asDiagonal() * rhs).array().colwise().sum() * intervalWidth * 0.5;
		res.array().rowwise() -= (compressed_dosage_inv_sds.cast<scalarData>().cwiseProduct(compressed_dosage_means.cast<scalarData>()).asDiagonal() * rhs).array().colwise().sum();
		return res;
	} else {
		return G * rhs;
	}
}


// No need to call this TemporaryFunction() function,
// it's just to avoid link error.
void TemporaryFunctionGenotypeMatrix (){
	std::vector<long> chunk;
	EigenDataMatrix mat;
	GenotypeMatrix X(false);

	X.col_block3(chunk, mat);
}
