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
	parameters p;

	Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> M;     // used in low-mem mode
	EigenDataMatrix G;     // used when not in low-mem node

	std::vector< int > chromosome;
	std::vector< std::string > al_0, al_1, rsid;
	std::vector< std::uint32_t > position;
	std::vector< double > maf, info;
	std::vector< std::string > SNPKEY;
	std::vector< std::string > SNPID;

	std::vector<std::map<std::size_t, bool> > missing_genos;
	Eigen::VectorXd col_means;
	Eigen::VectorXd col_sds_inv;
	std::size_t nn, pp;

	bool minibatch_index_set;
	EigenArrayXl minibatch_index;

	// Interface type of Eigen indices -> see eigen3/Eigen/src/Core/EigenBase.h
	typedef Eigen::Index Index;

	// Constructors
	GenotypeMatrix() : low_mem(false){
		scaling_performed = false;
		nn = 0;
		pp = 0;
		minibatch_index_set = false;
	};

	explicit GenotypeMatrix(const parameters& my_params) : low_mem(my_params.low_mem),
		p(my_params){
		scaling_performed = false;
		nn = 0;
		pp = 0;
		minibatch_index_set = false;
	};

	~GenotypeMatrix() = default;

	/*** Operations for minibatch subsampling ***/
	void set_minibatch_index(const EigenArrayXl& index){
		assert(p.mode_svi);
		minibatch_index = index;
		minibatch_index_set = true;
		assert(minibatch_index.maxCoeff() < nn);
		assert(minibatch_index.minCoeff() >= 0);
	};
	void col(long jj, EigenRefDataVector vec) const;

	EigenDataMatrix operator*(EigenRefDataMatrix rhs) const;

	/*** Input / Write access methods ***/
	// Eigen element access
	void assign_index(const long& ii, const long& jj, double x);


	/*** Output / Read access methods ***/
	// Eigen element access
	double operator()(const long& ii, const long& jj){
		if(!scaling_performed) {
			calc_scaled_values();
		}

		if(low_mem) {
			return (DecompressDosage(M(ii, jj)) - col_means[jj]) * col_sds_inv[jj];
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
			vec *= (intervalWidth * col_sds_inv[jj]);
			vec.array() += (0.5 * intervalWidth - col_means[jj]) * col_sds_inv[jj];
		} else {
			vec = G.col(jj);
		}
		return vec;
	}

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
				res.col(ll) = M.cast<scalarData>() * col_sds_inv.cast<scalarData>().asDiagonal() * tmp.cast<scalarData>();
			}
			res *= intervalWidth;
			// res = M.cast<scalarData>() * col_sds_inv.cast<scalarData>().asDiagonal() * rhs.cast<scalarData>() * intervalWidth;
			res.array().rowwise() += (col_sds_inv.cast<scalarData>().asDiagonal() * rhs.cast<scalarData>()).array().colwise().sum() * intervalWidth * 0.5;
			res.array().rowwise() -= (col_sds_inv.cast<scalarData>().cwiseProduct(col_means.cast<scalarData>()).asDiagonal() * rhs.cast<scalarData>()).array().colwise().sum();
			return res;
		} else {
			return G * rhs.cast<scalarData>();
		}
	}
#else
	// Eigen read column
	Eigen::VectorXd col(long jj);
#endif

	Eigen::MatrixXd transpose_multiply(EigenRefDataArrayXX lhs);
	Eigen::VectorXd mult_vector_by_chr(const int& chr, const Eigen::Ref<const Eigen::VectorXd>& rhs);

	template <typename Deriv>
	void col_block(const std::vector<std::uint32_t> &chunk,
	               Eigen::MatrixBase<Deriv> &D);

	template <typename Deriv>
	void get_cols(const std::vector<int> &index,
	              const std::vector<std::uint32_t> &iter_chunk,
	              Eigen::MatrixBase<Deriv>& D);

	/*** Mean center & unit variance; internal use ***/
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
	inline std::uint8_t CompressDosage(double dosage)
	{
		dosage = std::min(dosage, L - 1e-6);
		return static_cast<std::uint8_t>(std::floor(dosage * invIntervalWidth));
	}

	inline double DecompressDosage(std::uint8_t compressed_dosage)
	{
		return (compressed_dosage + 0.5)*intervalWidth;
	}
};



#endif
