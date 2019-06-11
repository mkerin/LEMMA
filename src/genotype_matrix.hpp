/* low-mem genotype matrix
   Useful links:
   - http://www.learncpp.com/cpp-tutorial/131-function-templates/
   - https://www.tutorialspoint.com/cplusplus/cpp_overloading.htm
   - https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
   - https://stackoverflow.com/questions/23841723/eigen-library-assigning-matrixs-elements

   Outline
        Basically a wrapper around an Eigen matrix of unsigned ints.

   To compress dosage entries I split the interval [0,2) into 2^n segments (dosage
   matrix assumed to be standardised). For each dosage value I store the index of
   the segment that it falls into, and return the midpoint of the segment when decompressing.
 */

#ifndef GENOTYPE_MATRIX
#define GENOTYPE_MATRIX

#include "my_timer.hpp"
#include "parameters.hpp"
#include "typedefs.hpp"
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

public:
	const bool low_mem;
	bool scaling_performed;
	parameters params;

	Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> M;     // used in low-mem mode
	EigenDataMatrix G;     // used when not in low-mem node

	std::vector< int > chromosome;
	std::vector< std::string > al_0, al_1, rsid;
	std::vector< std::uint32_t > position;
	std::vector< double > maf, info;
	// chr~pos~a0~a1
	std::vector< std::string > SNPKEY;
	std::vector< std::string > SNPID;

	std::vector<std::map<std::size_t, bool> > missing_genos;
	Eigen::VectorXd compressed_dosage_means;
	Eigen::VectorXd compressed_dosage_sds;
	Eigen::VectorXd compressed_dosage_inv_sds;      // 1 / col-wise sd
	std::size_t nn, pp;

	// Interface type of Eigen indices -> see eigen3/Eigen/src/Core/EigenBase.h
	typedef Eigen::Index Index;

	// Constructors
	GenotypeMatrix(const bool& use_low_mem) : low_mem(use_low_mem){
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

//	// Replacement(s) for write-version of Eigen Method .col()
//	template<typename T>
//	void assign_col(const T& jj, Eigen::Ref<Eigen::VectorXd> vec){
//		assert(vec.rows() == nn);
//
//		if(low_mem){
//			for (Index ii = 0; ii < nn; ii++){
//				M(ii, jj) = CompressDosage(vec[ii]);
//			}
//		} else {
//			for (Index ii = 0; ii < nn; ii++){
//				G(ii, jj) = vec[ii];
//			}
//		}
//
//		scaling_performed = false;
//	}

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
	Eigen::VectorXf col(long jj){
		assert(jj < pp);
		Eigen::VectorXf vec(nn);
		if(!scaling_performed) {
			calc_scaled_values();
		}

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
	Eigen::VectorXd col(long jj);
#endif

	// Eigen read column
	void col(long jj, EigenRefDataVector vec);
//
//	void col(long jj, Eigen::Ref<Eigen::VectorXf> vec){
//		assert(jj < pp);
//		if(!scaling_performed){
//			calc_scaled_values();
//		}
//
//		if(low_mem){
//			vec = M.cast<float>().col(jj);
//			vec *= (intervalWidth * compressed_dosage_inv_sds[jj]);
//			vec.array() += (0.5 * intervalWidth - compressed_dosage_means[jj]) * compressed_dosage_inv_sds[jj];
//		} else {
//			vec = G.col(jj);
//		}
//	}

//	// Dot with jth col - this was actually slower. Oh well.
//	double dot_with_jth_col(const Eigen::Ref<const Eigen::VectorXd>& vec, Index jj){
//		assert(jj < pp);
//		double tmp, offset, res;
//		if(!scaling_performed){
//			calc_scaled_values();
//		}
//
//		tmp = vec.dot(M.col(jj).cast<double>());
//		offset = vec.sum();
//
//		res = intervalWidth * tmp + offset * (intervalWidth * 0.5 - compressed_dosage_means[jj]);
//		res *= compressed_dosage_inv_sds[jj];
//		return res;
//	}

	// Eigen matrix multiplication
#ifdef DATA_AS_FLOAT
	// Eigen matrix multiplication
	EigenDataMatrix operator*(Eigen::Ref<Eigen::MatrixXd> rhs){
		if(!scaling_performed) {
			calc_scaled_values();
		}
		assert(rhs.rows() == pp);
		if(low_mem) {
			EigenDataMatrix res(nn, rhs.cols());
			for (int ll = 0; ll < rhs.cols(); ll++) {
				Eigen::Ref<Eigen::VectorXd> tmp = rhs.col(ll);
				res.col(ll) = M.cast<scalarData>() * compressed_dosage_inv_sds.cast<scalarData>().asDiagonal() * tmp.cast<scalarData>();
			}
			res *= intervalWidth;
			// res = M.cast<scalarData>() * compressed_dosage_inv_sds.cast<scalarData>().asDiagonal() * rhs.cast<scalarData>() * intervalWidth;
			res.array().rowwise() += (compressed_dosage_inv_sds.cast<scalarData>().asDiagonal() * rhs.cast<scalarData>()).array().colwise().sum() * intervalWidth * 0.5;
			res.array().rowwise() -= (compressed_dosage_inv_sds.cast<scalarData>().cwiseProduct(compressed_dosage_means.cast<scalarData>()).asDiagonal() * rhs.cast<scalarData>()).array().colwise().sum();
			return res;
		} else {
			return G * rhs.cast<scalarData>();
		}
	}
#endif

	// Eigen matrix multiplication
	EigenDataMatrix operator*(EigenRefDataMatrix rhs);

	Eigen::MatrixXd transpose_multiply(EigenRefDataMatrix lhs);

	EigenDataMatrix col_block(const std::uint32_t& ch_start,
	                          const int& ch_len);

	// Eigen lhs matrix multiplication
	Eigen::VectorXd mult_vector_by_chr(const long& chr, const Eigen::Ref<const Eigen::VectorXd>& rhs);

	template <typename Deriv>
	void col_block3(const std::vector<long>& chunk,
	                Eigen::MatrixBase<Deriv>& D);

	template <typename Deriv>
	void get_cols(const std::vector<long> &index,
	              const std::vector<long> &iter_chunk,
	              Eigen::MatrixBase<Deriv>& D);

	// // Eigen lhs matrix multiplication
	// Eigen::VectorXd transpose_vector_multiply(const Eigen::Ref<const Eigen::VectorXd>& lhs,
	//                                        bool lhs_centered){
	//  // G.transpose_vector_multiply(y) <=> (y^t G)^t <=> G^t y
	//  // NOTE: assumes that lhs is centered!
	//  if(!scaling_performed){
	//      calc_scaled_values();
	//  }
	//
	//  Eigen::VectorXd res;
	//  if(low_mem){
	//      assert(lhs.rows() == M.rows());
	//      assert(std::abs(lhs.sum()) < 1e-9);
	//
	//      res = lhs.transpose() * M.cast<double>();
	//
	//      return res.cwiseProduct(compressed_dosage_inv_sds) * intervalWidth;
	//  } else {
	//      return lhs.transpose() * G;
	//  }
	// }

	/********** Mean center & unit variance; internal use ************/
	void calc_scaled_values();

	void compute_means_and_sd();

	void standardise_matrix();

	/********** Utility functions ************/
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
