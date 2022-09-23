#ifndef GENOTYPE_MATRIX
#define GENOTYPE_MATRIX

#include "parameters.hpp"
#include "typedefs.hpp"
#include "tools/eigen3.3/Dense"
#include <algorithm>
#include <iostream>
#include <limits>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <random>
#include <thread>
#include <vector>
#include <map>

// Memory efficient class for storing dosage data
// - Use uint instead of double to store dosage probabilities
// - Basically a wrapper around an eigen matrix
class GenotypeMatrix {
	// Compression objects
	const std::uint8_t maxCompressedValue = std::numeric_limits<std::uint8_t>::max();
	const std::uint16_t numCompressedIntervals = static_cast<std::uint16_t> (maxCompressedValue + 1);
	const double L = 2.0;
	const double intervalWidth = L/numCompressedIntervals;
	const double invIntervalWidth = numCompressedIntervals / 2.0;
	long nn, pp;

public:
	const bool low_mem;
	bool scaling_performed;
	parameters params;

	Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> M;
	EigenDataMatrix G;

	std::vector<int> chromosome;
	std::vector<std::string> al_0, al_1, rsid;
	std::vector<std::uint32_t> position;
	std::vector<long> cumulative_pos;
	std::vector<double> maf, info;
	// chr~pos~a0~a1
	std::vector< std::string > SNPKEY;
	std::vector< std::string > SNPID;

	std::vector<std::map<std::size_t, bool> > missing_genos;
	Eigen::VectorXd compressed_dosage_means;
	Eigen::VectorXd compressed_dosage_sds;
	Eigen::VectorXd compressed_dosage_inv_sds;

	// Interface type of Eigen indices -> see eigen3/Eigen/src/Core/EigenBase.h
	typedef Eigen::Index Index;

	// Constructors
	GenotypeMatrix(const parameters& my_params, const bool& use_low_mem) : low_mem(use_low_mem),
		params(my_params){
		scaling_performed = false;
		nn = 0;
		pp = 0;
	};

	explicit GenotypeMatrix(const parameters& my_params) : low_mem(my_params.low_mem),
		params(my_params){
		scaling_performed = false;
		nn = 0;
		pp = 0;
	};

	~GenotypeMatrix() = default;

	/********** Input / Write access methods ************/

	// Eigen element access
	void assign_index(const long& ii, const long& jj, double x);

	/********** Output / Read access methods ************/

	// Eigen element access
	double operator()(const long& ii, const long& jj){
		if(!scaling_performed) {
			calc_scaled_values();
		}

		if(low_mem) {
			return (DecompressDosage(M(ii, jj)) - compressed_dosage_means[jj]) * compressed_dosage_inv_sds[jj];
		} else {
			return G(ii, jj);
		}
	}

#ifdef DATA_AS_FLOAT
	// Eigen read column
	Eigen::VectorXf col(long jj) const {
		assert(jj < pp);
		assert(scaling_performed);
		Eigen::VectorXf vec(nn);

		if(low_mem) {
			vec = M.cast<float>().col(jj);
			vec *= (intervalWidth * compressed_dosage_inv_sds[jj]);
			vec.array() += (0.5 * intervalWidth - compressed_dosage_means[jj]) * compressed_dosage_inv_sds[jj];
		} else {
			vec = G.col(jj);
		}
		return vec;
	}
#else
	// Eigen read column
	Eigen::VectorXd col(long jj) const;
#endif

	// Eigen read column
	void col(long jj, EigenRefDataVector vec) const;

	// Eigen matrix multiplication
#ifdef DATA_AS_FLOAT
	// Eigen matrix multiplication
	EigenDataMatrix operator*(Eigen::Ref<Eigen::MatrixXd> rhs) const {
		assert(scaling_performed);
		assert(rhs.rows() == pp);
		if(low_mem) {
			if(params.mode_debug) std::cout << "Starting low-mem matrix mult" << std::endl;
			EigenDataMatrix res(nn, rhs.cols());
			for (int ll = 0; ll < rhs.cols(); ll++) {
				Eigen::Ref<Eigen::VectorXd> tmp = rhs.col(ll);
				res.col(ll) = M.cast<scalarData>() * compressed_dosage_inv_sds.cast<scalarData>().asDiagonal() * tmp.cast<scalarData>();
			}
			res *= intervalWidth;
			// res = M.cast<scalarData>() * compressed_dosage_inv_sds.cast<scalarData>().asDiagonal() * rhs.cast<scalarData>() * intervalWidth;
			res.array().rowwise() += (compressed_dosage_inv_sds.cast<scalarData>().asDiagonal() * rhs.cast<scalarData>()).array().colwise().sum() * intervalWidth * 0.5;
			res.array().rowwise() -= (compressed_dosage_inv_sds.cast<scalarData>().cwiseProduct(compressed_dosage_means.cast<scalarData>()).asDiagonal() * rhs.cast<scalarData>()).array().colwise().sum();
			if(params.mode_debug) std::cout << "Ending low-mem matrix mult" << std::endl;
			return res;
		} else {
			return G * rhs.cast<scalarData>();
		}
	}
#endif

	// Eigen matrix multiplication
	EigenDataMatrix operator*(EigenRefDataMatrix rhs) const;

	Eigen::MatrixXd transpose_multiply(EigenRefDataMatrix lhs) const;

	// Eigen lhs matrix multiplication
	Eigen::VectorXd mult_vector_by_chr(const long& chr, const Eigen::Ref<const Eigen::VectorXd>& rhs);

	template <typename Deriv>
	void col_block3(const std::vector<long>& chunk,
	                Eigen::MatrixBase<Deriv>& D) const;

	template <typename Deriv>
	void get_cols(const std::vector<long> &index,
	              const std::vector<long> &iter_chunk,
	              Eigen::MatrixBase<Deriv>& D) const;

	/********** Mean center & unit variance; internal use ************/
	void calc_scaled_values();

	void compute_means_and_sd();

	void standardise_matrix();

	/********** Utility functions ************/
	void compute_cumulative_pos(){
		cumulative_pos.resize(pp);
		for (long ii = 0; ii < pp; ii++) {
			if (ii == 0){
				cumulative_pos[0] = position[0];
			} else {
				long diff = position[ii] - position[ii-1];
				if(diff < 0) diff = 1;
				cumulative_pos[ii] = cumulative_pos[ii - 1] + diff;
			}
		}
	}

	inline Index rows() const {
		return nn;
	}

	inline Index cols() const {
		return pp;
	}

	void resize(const long& n, const long& p);

	void move_variant(std::uint32_t old_index, std::uint32_t new_index);

	void conservativeResize(const long& n, const long& p);

	friend std::ostream &operator<<( std::ostream &output, const GenotypeMatrix &gg ) {
		output << gg.M;
		return output;
	}

	/********** Compression/Decompression ************/
	inline std::uint8_t CompressDosage(double dosage){
		assert(dosage <= 2.0);
		if(dosage > 2) {
			std::cout << "WARNING: dosage = " << dosage << std::endl;
		}
		dosage = std::min(dosage, L - 1e-6);
		return static_cast<std::uint8_t>(std::floor(dosage * invIntervalWidth));
	}

	inline double DecompressDosage(std::uint8_t compressed_dosage)
	{
		return (compressed_dosage + 0.5)*intervalWidth;
	}
};

#endif
