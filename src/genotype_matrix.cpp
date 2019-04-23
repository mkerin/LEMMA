//
// Created by kerin on 2019-03-01.
//

#include "genotype_matrix.hpp"

#include "my_timer.hpp"
#include "parameters.hpp"
#include "typedefs.hpp"
#include "tools/Eigen/Dense"
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

Eigen::MatrixXd GenotypeMatrix::transpose_multiply(EigenRefDataArrayXX lhs) {
	// Return G^t lhs
	assert(lhs.rows() == nn);
	if(!scaling_performed) {
		calc_scaled_values();
	}

	if(low_mem) {
		Eigen::MatrixXd res;
		Eigen::VectorXd colsums = lhs.colwise().sum().matrix().cast<double>();

		// Need to do this one column at a time to avoid casting entirety of M to double
		Eigen::MatrixXd Mt_lhs(pp, lhs.cols());
		for (int ll = 0; ll < lhs.cols(); ll++) {
			EigenRefDataVector tmp = lhs.col(ll);
			Mt_lhs.col(ll) = tmp.cast<double>().transpose() * M.cast<double>();
		}

		res = intervalWidth * (col_sds_inv.asDiagonal() * Mt_lhs);
		res += 0.5 * intervalWidth * col_sds_inv * colsums.transpose();
		res -= col_sds_inv.cwiseProduct(col_means) * colsums.transpose();
		return res;
	} else {
		return (G.transpose() * lhs.matrix()).cast<double>();
	}
}

Eigen::VectorXd GenotypeMatrix::mult_vector_by_chr(const int &chr, const Eigen::Ref<const Eigen::VectorXd> &rhs) {
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
		Eigen::VectorXd rhs_trans = rhs.cwiseProduct(col_sds_inv);
		auto offset = rhs_trans.segment(chr_st, chr_size).sum() * intervalWidth * 0.5;
		offset -= col_means.segment(chr_st, chr_size).dot(rhs_trans.segment(chr_st, chr_size));

		res = M.block(0, chr_st, nn, chr_size).cast<double>() * rhs_trans.segment(chr_st, chr_size);
		return (res.array() * intervalWidth + offset).matrix();
	} else {
		return G.block(0, chr_st, nn, chr_size).cast<double>() * rhs.segment(chr_st, chr_size);
	}
}

template<typename Deriv>
void GenotypeMatrix::col_block(const std::vector<std::uint32_t> &chunk, Eigen::MatrixBase<Deriv> &D) {
	if(!scaling_performed) {
		calc_scaled_values();
	}

	// Have tried partitioning this amongst threads
	// Minimal improvement in read time.
	unsigned long ch_len = chunk.size();
	std::vector<std::vector<int> > indexes(p.n_thread);
	for (int ii = 0; ii < ch_len; ii++) {
		indexes[0].push_back(ii);
	}

	// Decompress char -> double
	get_cols(indexes[0], chunk, D);
}

template<typename Deriv>
void GenotypeMatrix::get_cols(const std::vector<int> &index, const std::vector<std::uint32_t> &iter_chunk,
                              Eigen::MatrixBase<Deriv> &D) {
	for(auto ii: index ) {
		long jj = (iter_chunk[ii] % pp);
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
	// Column means
	for (Index jj = 0; jj < pp; jj++) {
		col_means[jj] = 0;
		for (Index ii = 0; ii < nn; ii++) {
			col_means[jj] += DecompressDosage(M(ii, jj));
		}
	}
	col_means /= (double) nn;

	// Column standard deviation
	double val, sigma;
	Eigen::VectorXd compressed_dosage_sds(pp);
	for (Index jj = 0; jj < pp; jj++) {
		sigma = 0;
		for (Index ii = 0; ii < nn; ii++) {
			val = DecompressDosage(M(ii, jj)) - col_means[jj];
			sigma += val * val;
		}
		compressed_dosage_sds[jj] = sigma;
	}

	compressed_dosage_sds /= ((double) nn - 1.0);
	compressed_dosage_sds = compressed_dosage_sds.array().sqrt().matrix();

	for (Index jj = 0; jj < pp; jj++) {
		sigma = compressed_dosage_sds[jj];
		if (sigma > 1e-9) {
			col_sds_inv[jj] = 1 / sigma;
		} else {
			col_sds_inv[jj] = 0.0;
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
	col_means.resize(p);
	col_sds_inv.resize(p);
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
	col_means.conservativeResize(p);
	col_sds_inv.conservativeResize(p);
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
		vec *= (intervalWidth * col_sds_inv[jj]);
		vec.array() += (0.5 * intervalWidth - col_means[jj]) * col_sds_inv[jj];
	} else {
		vec = G.col(jj);
	}
	return vec;
}

void GenotypeMatrix::col(long jj, EigenRefDataVector vec) const {
	assert(jj < pp);
	assert(scaling_performed);

	if (minibatch_index_set) {
		if (low_mem) {
			vec = M(minibatch_index, jj).cast<scalarData>();
			vec *= (intervalWidth * col_sds_inv[jj]);
			vec.array() += (0.5 * intervalWidth - col_means[jj]) * col_sds_inv[jj];
		} else {
			vec = G(minibatch_index, jj);
		}
	} else {
		if (low_mem) {
			vec = M.col(jj).cast<scalarData>();
			vec *= (intervalWidth * col_sds_inv[jj]);
			vec.array() += (0.5 * intervalWidth - col_means[jj]) * col_sds_inv[jj];
		} else {
			vec = G.col(jj);
		}
	}
}

EigenDataMatrix GenotypeMatrix::operator*(EigenRefDataMatrix rhs) const {
	assert(scaling_performed);
	assert(rhs.rows() == pp);

	if (minibatch_index_set) {
		if (low_mem) {
			EigenDataMatrix res(nn, rhs.cols());
			for (int ll = 0; ll < rhs.cols(); ll++) {
				EigenDataVector tmp = col_sds_inv.cast<scalarData>().asDiagonal() * rhs.col(ll).cast<scalarData>();
				// Use of Eigen::all here seems to lead to errors
				res.col(ll) = M(minibatch_index, ":").template cast<scalarData>() * tmp;
			}
			res *= intervalWidth;
			res.array().rowwise() +=
				(col_sds_inv.cast<scalarData>().asDiagonal() * rhs).array().colwise().sum() * intervalWidth * 0.5;
			res.array().rowwise() -= (
				col_sds_inv.cast<scalarData>().cwiseProduct(col_means.cast<scalarData>()).asDiagonal() *
				rhs).array().colwise().sum();
			return res;
		} else {
			// Eigen flexi-indexing doesn't seem to work when used to multiply with matrix.
			// return G(minibatch_index, Eigen::all).template cast<scalarData>() * rhs;
			throw std::logic_error("Not implemented");
		}
	} else {
		if (low_mem) {
			EigenDataMatrix res(nn, rhs.cols());
			for (int ll = 0; ll < rhs.cols(); ll++) {
				EigenRefDataVector tmp = rhs.col(ll);
				res.col(ll) =
					M.cast<scalarData>() * col_sds_inv.cast<scalarData>().asDiagonal() * tmp.cast<scalarData>();
			}
			res *= intervalWidth;
			res.array().rowwise() +=
				(col_sds_inv.cast<scalarData>().asDiagonal() * rhs).array().colwise().sum() * intervalWidth * 0.5;
			res.array().rowwise() -= (
				col_sds_inv.cast<scalarData>().cwiseProduct(col_means.cast<scalarData>()).asDiagonal() *
				rhs).array().colwise().sum();
			return res;
		} else {
			return G * rhs;
		}
	}
}


// No need to call this TemporaryFunction() function,
// it's just to avoid link error.
void TemporaryFunctionGenotypeMatrix (){
	std::vector< std::uint32_t> chunk;
	EigenDataMatrix mat;
	GenotypeMatrix X;

	X.col_block(chunk, mat);
}
