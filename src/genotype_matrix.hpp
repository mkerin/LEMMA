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

Operations that class GenotypeMatrix should support:
- read/write access to the i,j th entry           operator()
- read/write access to the j th column            col
- matrix multiplication with matrix of doubles.   operator*
- resize()

Questions:
- More intelligent way to overload operators for write access.
	eg. G.assign_col(jj, vec);     => G.col(jj) = vec;
	eg. G.assign_index(ii, jj, x); => G(ii, jj) = x;
	Return reference for a function?

- Glaring inefficiencies
	eg. unnecessary copying of memory.
*/

#ifndef GENOTYPE_MATRIX
#define GENOTYPE_MATRIX

#include <algorithm>
#include <iostream>
#include <limits>
#include <cstdint>               // uint32_t
#include "tools/eigen3.3/Dense"


// Memory efficient class for storing dosage data
// - Use uint instead of double to store dosage probabilities
// - Basically a wrapper around an eigen matrix
class GenotypeMatrix {
	// Compression objects
	const std::uint8_t maxCompressedValue = std::numeric_limits<std::uint8_t>::max();
	const std::uint16_t numCompressedIntervals = static_cast<std::uint16_t> (maxCompressedValue + 1);
	const double intervalWidth = 2.0/numCompressedIntervals;
	const double invIntervalWidth = numCompressedIntervals / 2.0;

public:
	bool means_and_sd_computed;
	Eigen::VectorXd compressed_dosage_means;
	Eigen::VectorXd compressed_dosage_sds;
	Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> M;

	// Interface type of Eigen indices -> see eigen3/Eigen/src/Core/EigenBase.h
	typedef Eigen::Index Index;

	// Constructors
	GenotypeMatrix(){
		means_and_sd_computed = false;
	};
	GenotypeMatrix(int n, int p){
		M.resize(n, p);
		compressed_dosage_means.resize(p);
		compressed_dosage_sds.resize(p);

		means_and_sd_computed = false;
	};
	~GenotypeMatrix(){
	};

	// Replacement(s) for read-version of Eigen element access
	double operator()(Index ii, Index jj){
		return DecompressDosage(M(ii, jj));
	}

	double read_index(Index ii, Index jj){
		return DecompressDosage(M(ii, jj));
	}

	// Replacement(s) for write-version of Eigen element access
	void assign_index(Index ii, Index jj, double x){
		M(ii, jj) = CompressDosage(x);

		means_and_sd_computed = false;
	}


	// Replacement(s) for read-version of Eigen Method .col()
	template<typename T>
	void col(T jj, Eigen::Ref<Eigen::VectorXd> vec){
		assert(vec.rows() == M.rows());
		for (Index ii = 0; ii < M.rows(); ii++){
			vec[ii] = DecompressDosage(M(ii, jj));
		}
	}

	template<typename T>
	Eigen::VectorXd col(T jj){
		Eigen::VectorXd vec(M.rows());
		for (Index ii = 0; ii < M.rows(); ii++){
			vec[ii] = DecompressDosage(M(ii, jj));
		}
		return vec;
	}

	// Replacement(s) for write-version of Eigen Method .col()
	template<typename T>
	void assign_col(T jj, Eigen::Ref<Eigen::VectorXd> vec){
		assert(vec.rows() == M.rows());
		for (Index ii = 0; ii < M.rows(); ii++){
			M(ii, jj) = CompressDosage(vec[ii]);
		}
	}

	// 	Replacement(s) for Eigen Matrix multiplication
	Eigen::VectorXd operator*(const Eigen::Ref<const Eigen::VectorXd> rhs){
		double vec_total = rhs.sum() * intervalWidth * 0.5;

		if(!means_and_sd_computed){
			compute_means_and_sd();
		}

		Eigen::VectorXd rhs_trans = rhs.cwiseProduct(compressed_dosage_sds.cwiseInverse());
		double offset = rhs_trans.sum() * intervalWidth * 0.5;
		offset -= compressed_dosage_means.dot(rhs_trans);

		return ((M.cast<double>() * rhs_trans).array() * intervalWidth + offset).matrix();
	}

	void compute_means_and_sd(){
		// Column means
		for (Index jj = 0; jj < M.cols(); jj++){
			compressed_dosage_means[jj] = 0;
			for (Index ii = 0; ii < M.rows(); ii++){
				compressed_dosage_means[jj] += DecompressDosage(M(ii, jj));
			}
		}
		compressed_dosage_means /= (double) M.rows();

		// Column standard deviation
		double val;
		for (Index jj = 0; jj < M.cols(); jj++){
			compressed_dosage_sds[jj] = 0;
			for (Index ii = 0; ii < M.rows(); ii++){
				val = DecompressDosage(M(ii, jj)) - compressed_dosage_means[jj];
				compressed_dosage_sds[jj] += val * val;
			}
		}
		compressed_dosage_sds /= ((double) M.rows() - 1.0);
		compressed_dosage_sds = compressed_dosage_sds.array().sqrt().matrix();

		means_and_sd_computed = true;
	}

	// Brute force approach
	// template<typename Derived>
	// Eigen::MatrixXd operator*(const Eigen::MatrixBase<Derived>& rhs){
	// 	Eigen::MatrixXd M = Eigen::MatrixXd::Zero(M.rows(), rhs.cols());
	// 	double vecDecompressionConstant = rhs.sum() * intervalWidth;
	// 
	// 	for (Index ii = 0; ii < M.rows(); ii++){
	// 		for (Index kk = 0; kk < rhs.cols(); kk++){
	// 			for (Index jj = 0; jj < M.cols(); jj++){
	// 				M(ii, kk) += M(ii, jj) * rhs(jj, kk);
	// 			}
	// 		}
	// 	}
	// 	return (M.array() * intervalWidth * 2.0 + vecDecompressionConstant).matrix();
	// }

	// Utility functions
	inline Index rows() const { return M.rows(); }

	inline Index cols() const { return M.cols(); }

	template <typename T>
	void resize(const T& n, const T& p){
		M.resize(n, p);
		compressed_dosage_means.resize(p);
		compressed_dosage_sds.resize(p);
	}

	friend std::ostream &operator<<( std::ostream &output, const GenotypeMatrix &gg ) { 
		output << gg.M;
		return output;
	}

	/********** Compression/Decompression ************/
	inline std::uint8_t CompressDosage(double dosage)
	{
		dosage = std::min(dosage, 1.999999);
		return static_cast<std::uint16_t>(std::floor(dosage * invIntervalWidth));
	}

	inline double DecompressDosage(std::uint8_t compressed_dosage)
	{
		return (compressed_dosage + 0.5)*intervalWidth;
	}
};

#endif
