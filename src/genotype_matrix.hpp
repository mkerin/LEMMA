/* low-mem genotype matrix
Useful links:
- http://www.learncpp.com/cpp-tutorial/131-function-templates/
- https://www.tutorialspoint.com/cplusplus/cpp_overloading.htm
- https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
- https://stackoverflow.com/questions/23841723/eigen-library-assigning-matrixs-elements

Outline
	Basically a wrapper around an Eigen matrix of unsigned ints.

To compress dosage entries I split the interval [0,1) into 2^n segments (dosage 
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
	const double sizeCompressedInterval = 1.0/numCompressedIntervals;
	const double sumCompressionConstant = 0.5;

public:
	Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G;

	// Interface type of Eigen indices -> see eigen3/Eigen/src/Core/EigenBase.h
	typedef Eigen::Index Index;

	// Constructors
	GenotypeMatrix(){};
	GenotypeMatrix(int n, int p){
		G.resize(n, p);
		std::cout << "Constructing commpressed dosage matrix with ";
 		std::cout << numCompressedIntervals << " intervals" << std::endl;
	};
	~GenotypeMatrix(){
	};

	// Replacement(s) for read-version of Eigen element access
	double operator()(Index ii, Index jj){
		return DecompressDosage(G(ii, jj));
	}

	double read_index(Index ii, Index jj){
		return DecompressDosage(G(ii, jj));
	}

	// Replacement(s) for write-version of Eigen element access
	void assign_index(Index ii, Index jj, double x){
		G(ii, jj) = CompressDosage(x);
	}


	// Replacement(s) for read-version of Eigen Method .col()
	template<typename T>
	void col(T jj, Eigen::Ref<Eigen::VectorXd> vec){
		assert(vec.rows() == G.rows());
		for (Index ii = 0; ii < G.rows(); ii++){
			vec[ii] = DecompressDosage(G(ii, jj));
		}
	}

	template<typename T>
	Eigen::VectorXd col(T jj){
		Eigen::VectorXd vec(G.rows());
		for (Index ii = 0; ii < G.rows(); ii++){
			vec[ii] = DecompressDosage(G(ii, jj));
		}
		return vec;
	}

	// Replacement(s) for write-version of Eigen Method .col()
	template<typename T>
	void assign_col(T jj, Eigen::Ref<Eigen::VectorXd> vec){
		assert(vec.rows() == G.rows());
		for (Index ii = 0; ii < G.rows(); ii++){
			G(ii, jj) = CompressDosage(vec[ii]);
		}
	}

	// 	Replacement(s) for Eigen Matrix multiplication
	Eigen::VectorXd operator*(const Eigen::Ref<const Eigen::VectorXd> rhs){
		double vec_total = rhs.sum() * sizeCompressedInterval * 0.5;
		return ((G.cast<double>() * rhs).array() * sizeCompressedInterval + vec_total).matrix();
	}

	// Brute force approach
	// template<typename Derived>
	// Eigen::MatrixXd operator*(const Eigen::MatrixBase<Derived>& rhs){
	// 	Eigen::MatrixXd M = Eigen::MatrixXd::Zero(G.rows(), rhs.cols());
	// 	double vecDecompressionConstant = rhs.sum() * sizeCompressedInterval;
	// 
	// 	for (Index ii = 0; ii < G.rows(); ii++){
	// 		for (Index kk = 0; kk < rhs.cols(); kk++){
	// 			for (Index jj = 0; jj < G.cols(); jj++){
	// 				M(ii, kk) += G(ii, jj) * rhs(jj, kk);
	// 			}
	// 		}
	// 	}
	// 	return (M.array() * sizeCompressedInterval * 2.0 + vecDecompressionConstant).matrix();
	// }

	// Utility functions
	inline Index rows() const { return G.rows(); }

	inline Index cols() const { return G.cols(); }

	template <typename T>
	void resize(const T& n, const T& p){
		G.resize(n, p);
	}

	friend std::ostream &operator<<( std::ostream &output, const GenotypeMatrix &gg ) { 
		output << gg.G;
		return output;
	}

	/********** Compression/Decompression ************/
	inline std::uint8_t CompressDosage(double dosage)
	{
		dosage = std::min(dosage, 0.999999);
		return static_cast<std::uint16_t>(std::floor(dosage*numCompressedIntervals));
	}

	inline double DecompressDosage(std::uint8_t compressed_dosage)
	{
		return (compressed_dosage+sumCompressionConstant)*sizeCompressedInterval;
	}
};

#endif
