/* low-mem genotype matrix
Useful links:
- http://www.learncpp.com/cpp-tutorial/131-function-templates/
- https://www.tutorialspoint.com/cplusplus/cpp_overloading.htm



*/
#ifndef GENOTYPE_MATRIX
#define GENOTYPE_MATRIX

#include <algorithm>
#include <iostream>
#include <limits>
#include <cstdint>    // uint32_t
#include "tools/eigen3.3/Dense"


// Memory efficient genotype (or dosage!) matrix class
// - Use uint instead of double to store dosage probabilities
// - Basically a wrapper around an eigen matrix
class GenotypeMatrix {
	// Compression objects
	const std::uint8_t minCompressedValue = std::numeric_limits<std::uint8_t>::min();
	const std::uint8_t maxCompressedValue = std::numeric_limits<std::uint8_t>::max();
	const std::uint16_t numCompressedIntervals = static_cast<std::uint16_t> (maxCompressedValue + 1);
	const double sizeCompressedInterval = 1.0/numCompressedIntervals;
	const double sumCompressionConstant = 0.5;


public:
	Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> G;

	GenotypeMatrix(); // empty constructor
	~GenotypeMatrix(){
	};

	void resize(int n, int p){
		G.resize(n, p);
	}

	void AssignUncompressedProb(int i, int j, double x){
		G(i, j) = GetCompressedProb(x);
	}


	/********** Compression/Decompression ************/
	inline std::uint8_t GetCompressedProb(double prob)
	{
		prob = std::min(prob, 0.999999);
		return static_cast<std::uint16_t>(std::floor(prob*numCompressedIntervals)+minCompressedValue);
	}

	inline double GetUncompressedProb(std::uint8_t compressed_prob)
	{
		return (compressed_prob+sumCompressionConstant)*sizeCompressedInterval;
	}
};

	// template <typename T>
	// Eigen::VectorXd operator*(T other){
	// 	return G * T;
	// }

#endif
