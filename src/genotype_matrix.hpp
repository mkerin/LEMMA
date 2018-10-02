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
#include <cmath>
#include <random>
#include <vector>
#include <map>
#include "my_timer.hpp"
#include "tools/eigen3.3/Dense"

inline Eigen::MatrixXd getCols(const Eigen::MatrixXd &X, const std::vector<size_t> &cols);
inline void setCols(Eigen::MatrixXd &X, const std::vector<size_t> &cols, const Eigen::MatrixXd &values);
inline size_t numRows(const Eigen::MatrixXd &A);
inline size_t numCols(const Eigen::MatrixXd &A);
inline void setCol(Eigen::MatrixXd &A, const Eigen::VectorXd &v, size_t col);
inline Eigen::VectorXd getCol(const Eigen::MatrixXd &A, size_t col);


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

	Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> M; // used in low-mem mode
	Eigen::MatrixXd G; // used when not in low-mem node

	std::vector< int > chromosome;
	std::vector< std::string > al_0, al_1, rsid;
	std::vector< std::uint32_t > position;
	// chr~pos~a0~a1
	std::vector< std::string > SNPKEY;
	std::vector< std::string > SNPID;

	std::vector<std::map<int, bool>> missing_genos;
	Eigen::VectorXd compressed_dosage_means;
	Eigen::VectorXd compressed_dosage_sds;
	Eigen::VectorXd compressed_dosage_inv_sds;  // 1 / col-wise sd
	// Eigen::VectorXd aa;  // vector of ages
	Eigen::MatrixXd E;
	std::size_t NN, PP;
	int n_effects;

	// Interface type of Eigen indices -> see eigen3/Eigen/src/Core/EigenBase.h
	typedef Eigen::Index Index;

	// When using stochastic gradient descent
	bool mode_sgd;
	long int nBatch;
	long int batch_start;
	MyTimer t_readXk;

	// Constructors
	GenotypeMatrix(bool use_compression) : low_mem(use_compression),
                                           t_readXk("read_X_kk: %ts \n"){
		scaling_performed = false;
		mode_sgd = false;
		NN = 0;
		PP = 0;
		n_effects = 0;
	};

	GenotypeMatrix(bool use_compression,
                   const long int n,
                   const long int p,
                   const int my_n_effects) : low_mem(use_compression),
                                             t_readXk("read_X_kk: %ts \n"){
		if(low_mem){
			M.resize(n, p);
		} else {
			G.resize(n, p);
		}
		NN = n;
		PP = p;
		n_effects = my_n_effects;
		mode_sgd = false;

		compressed_dosage_means.resize(p);
		compressed_dosage_sds.resize(p);
		compressed_dosage_inv_sds.resize(p);
		missing_genos.resize(p);
		if(n_effects > 1){
			E.resize(n, n_effects-1);
		}

		scaling_performed = false;
	};

	~GenotypeMatrix(){
	};

	// sgd
	void draw_minibatch(long int my_nBatch){
		std::default_random_engine generator;
		std::uniform_int_distribution<long int> distribution(0,NN - my_nBatch);

		batch_start = distribution(generator);
		nBatch = my_nBatch;
		mode_sgd = true;
	}

	/********** Input / Write access methods ************/

	// Eigen element access
	template <typename T, typename T2>
	void assign_index(const T& ii, const T2& jj, double x){
		if(low_mem){
			if(std::isnan(x)){
				throw std::runtime_error("ERROR: missing values not yet compatible in low-mem mode");
			}
			M(ii, jj) = CompressDosage(x);
		} else {
			if(std::isnan(x)){
				missing_genos[jj][ii] = 1;
			} else {
				G(ii, jj) = x;
			}
		}
		scaling_performed = false;
	}

	// Replacement(s) for write-version of Eigen Method .col()
	template<typename T>
	void assign_col(const T& jj, Eigen::Ref<Eigen::VectorXd> vec){
		assert(vec.rows() == NN);

		if(low_mem){
			for (Index ii = 0; ii < NN; ii++){
				M(ii, jj) = CompressDosage(vec[ii]);
			}
		} else {
			for (Index ii = 0; ii < NN; ii++){
				G(ii, jj) = vec[ii];
			}
		}

		scaling_performed = false;
	}

	/********** Output / Read access methods ************/

	// Eigen element access
	template <typename T, typename T2>
	double operator()(const T& ii, const T2& jj){
		if(!scaling_performed){
			calc_scaled_values();
		}

		if(low_mem){
			return (DecompressDosage(M(ii, jj)) - compressed_dosage_means[jj]) * compressed_dosage_inv_sds[jj];
		} else {
			return G(ii, jj);
		}
	}

	// Eigen read column
	template<typename T>
	Eigen::VectorXd col(T jj){
		Eigen::VectorXd vec(NN);
		if(!scaling_performed){
			calc_scaled_values();
		}

		t_readXk.resume();
		if(low_mem){
			if(jj < PP){
				if(mode_sgd){
					vec = M.cast<double>().block(batch_start, jj, nBatch, 1);
				} else {
					vec = M.cast<double>().col(jj);
				}
				vec *= (intervalWidth * compressed_dosage_inv_sds[jj]);
				vec = vec.array() + (0.5 * intervalWidth - compressed_dosage_means[jj]) * compressed_dosage_inv_sds[jj];

			} else {
				int ee = jj / PP;
				jj %= PP;

				if(mode_sgd){
					vec = M.cast<double>().block(batch_start, jj, nBatch, 1);
				} else {
					vec = M.cast<double>().col(jj);
				}
				vec *= (intervalWidth * compressed_dosage_inv_sds[jj]);
 				vec = vec.array() + (0.5 * intervalWidth - compressed_dosage_means[jj]) * compressed_dosage_inv_sds[jj];

				if(mode_sgd){
					vec = vec.cwiseProduct(E.block(batch_start, ee-1, nBatch, 1));
				} else {
					vec = vec.cwiseProduct(E.col(ee-1));
				}
			}
		} else {
			if(jj < PP){
				if(mode_sgd){
					vec = G.block(batch_start, jj, nBatch, 1);
				} else {
					vec = G.col(jj);
				}
			} else {
				int ee = jj / PP;
				jj %= PP;
				if(mode_sgd){
					vec = G.block(batch_start, jj, nBatch, 1).cwiseProduct(E.block(batch_start, ee-1, nBatch, 1));
				} else {
					vec = G.col(jj).cwiseProduct(E.col(ee-1));
				}
			}
		}
		t_readXk.stop();
		return vec;
	}

	// Eigen read column
	template<typename T>
	void col(T jj, Eigen::Ref<Eigen::VectorXd> vec){
		if(!scaling_performed){
			calc_scaled_values();
		}

		if(low_mem){
			if(jj < PP){
				for (Index ii = 0; ii < NN; ii++){
					vec[ii] = (DecompressDosage(M(ii, jj)) - compressed_dosage_means[jj]) * compressed_dosage_inv_sds[jj];
				}
			} else {
				int ee = jj / PP;
				jj %= PP;
				for (Index ii = 0; ii < NN; ii++){
					vec[ii] = E(ii, ee-1) * (DecompressDosage(M(ii, jj)) - compressed_dosage_means[jj]) * compressed_dosage_inv_sds[jj];
				}
			}
		} else {
			if(jj < PP){
				vec = G.col(jj);
			} else {
				int ee = jj / PP;
				jj %= PP;
				vec = G.col(jj).cwiseProduct(E.col(ee-1));
			}
		}
	}

	// Dot with jth col - this was actually slower. Oh well.
	double dot_with_jth_col(const Eigen::Ref<const Eigen::VectorXd>& vec, Index jj){
		double tmp, offset, res;
		if(!scaling_performed){
			calc_scaled_values();
		}

		if (jj < PP){
			tmp = vec.dot(M.col(jj).cast<double>());
			offset = vec.sum();
		} else {
			int ee = jj / PP;
			jj %= PP;
			Eigen::VectorXd bb(vec.rows());
			bb = bb.cwiseProduct(E.col(ee-1));

			// for(Index ii = 0; ii < NN; ii++){
			// 	tmp += aa[ii] * vec[ii] * M(ii, jj);
			// }
			tmp = bb.dot(M.col(jj).cast<double>());
			offset = bb.sum();
		}

		res = intervalWidth * tmp + offset * (intervalWidth * 0.5 - compressed_dosage_means[jj]);
		res *= compressed_dosage_inv_sds[jj];
		return res;
	}

	// Eigen matrix multiplication
	Eigen::VectorXd operator*(const Eigen::Ref<const Eigen::VectorXd>&  rhs){
		if(!scaling_performed){
			calc_scaled_values();
		}

		if(low_mem){
			double vec_total = rhs.sum() * intervalWidth * 0.5;

			Eigen::VectorXd rhs_trans = rhs.cwiseProduct(compressed_dosage_inv_sds);
			double offset = rhs_trans.sum() * intervalWidth * 0.5;
			offset -= compressed_dosage_means.dot(rhs_trans);

			return ((M.cast<double>() * rhs_trans).array() * intervalWidth + offset).matrix();
		} else {
			return G * rhs;
		}
	}

	// Eigen lhs matrix multiplication
	Eigen::VectorXd transpose_vector_multiply(const Eigen::Ref<const Eigen::VectorXd>& lhs){
		// G.transpose_vector_multiply(y) <=> (y^t G)^t <=> G^t y
		// NOTE: assumes that lhs is centered!
		if(!scaling_performed){
			calc_scaled_values();
		}

		Eigen::VectorXd res;
		if(low_mem){
			assert(lhs.rows() == M.rows());
			double offset = lhs.sum();

			res = (lhs.transpose() * M.cast<double>()).array() + 0.5 * offset;
			res *= intervalWidth;
			res -= (offset * compressed_dosage_means);
			return res.cwiseProduct(compressed_dosage_inv_sds);
		} else {
			return lhs.transpose() * G;
		}
	}

	// // Eigen lhs matrix multiplication
	// Eigen::VectorXd transpose_vector_multiply(const Eigen::Ref<const Eigen::VectorXd>& lhs,
	// 										  bool lhs_centered){
	// 	// G.transpose_vector_multiply(y) <=> (y^t G)^t <=> G^t y
	// 	// NOTE: assumes that lhs is centered!
	// 	if(!scaling_performed){
	// 		calc_scaled_values();
	// 	}
	//
	// 	Eigen::VectorXd res;
	// 	if(low_mem){
	// 		assert(lhs.rows() == M.rows());
	// 		assert(std::abs(lhs.sum()) < 1e-9);
	//
	// 		res = lhs.transpose() * M.cast<double>();
	//
	// 		return res.cwiseProduct(compressed_dosage_inv_sds) * intervalWidth;
	// 	} else {
	// 		return lhs.transpose() * G;
	// 	}
	// }

	/********** Mean center & unit variance; internal use ************/
	void calc_scaled_values(){
		if (low_mem){
			compute_means_and_sd();
		} else {
			standardise_matrix();
		}
		scaling_performed = true;
	}

	void compute_means_and_sd(){
		// Column means
		for (Index jj = 0; jj < PP; jj++){
			compressed_dosage_means[jj] = 0;
			for (Index ii = 0; ii < NN; ii++){
				compressed_dosage_means[jj] += DecompressDosage(M(ii, jj));
			}
		}
		compressed_dosage_means /= (double) NN;

		// Column standard deviation
		double val, sigma;
		Eigen::VectorXd compressed_dosage_sds(PP);
		for (Index jj = 0; jj < PP; jj++){
			sigma = 0;
			for (Index ii = 0; ii < NN; ii++){
				val = DecompressDosage(M(ii, jj)) - compressed_dosage_means[jj];
				sigma += val * val;
			}
			compressed_dosage_sds[jj] = sigma;
		}

		compressed_dosage_sds /= ((double) NN - 1.0);
		compressed_dosage_sds = compressed_dosage_sds.array().sqrt().matrix();

		for (Index jj = 0; jj < PP; jj++){
			sigma = compressed_dosage_sds[jj];
			if (sigma > 1e-9){
				compressed_dosage_inv_sds[jj] = 1 / sigma;
			} else {
				compressed_dosage_inv_sds[jj] = 0.0;
			}
		}
	}

	void standardise_matrix(){
		if(!low_mem){
			for (std::size_t k = 0; k < PP; k++) {
				double mu = 0.0;
				double count = 0;
				for (std::size_t i = 0; i < NN; i++) {
					if (missing_genos[k].count(i) == 0) {
						mu += G(i, k);
						count += 1;
					}
				}

				mu = mu / count;
				double val, sigma = 0.0;
				for (std::size_t i = 0; i < NN; i++) {
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
					for (std::size_t i = 0; i < NN; i++) {
						G(i, k) /= sigma;
					}
				}
			}
		}
	}

	/********** Utility functions ************/
	inline Index rows() const { return NN; }

	inline Index cols() const { return PP; }

	template <typename T, typename T2, typename T3>
	void resize(const T& n, const T2& p, const T3& my_n_effects){
		if(low_mem){
			M.resize(n, p);
		} else {
			G.resize(n, p);
		}
		compressed_dosage_means.resize(p);
		compressed_dosage_sds.resize(p);
		compressed_dosage_inv_sds.resize(p);
		missing_genos.resize(p);
		E.resize(n, my_n_effects);
		NN = n;
		PP = p;
		n_effects = my_n_effects;
	}

	template <typename T, typename T2, typename T3>
	void conservativeResize(const T& n, const T2& p, const T3& my_n_effects){
		if(low_mem){
			M.conservativeResize(n, p);
		} else {
			G.conservativeResize(n, p);
		}
		compressed_dosage_means.conservativeResize(p);
		compressed_dosage_sds.conservativeResize(p);
		compressed_dosage_inv_sds.conservativeResize(p);
		E.conservativeResize(n, my_n_effects);
		missing_genos.resize(p);
		NN = n;
		PP = p;
		n_effects = my_n_effects;
	}

	friend std::ostream &operator<<( std::ostream &output, const GenotypeMatrix &gg ) {
		output << gg.M;
		return output;
	}

	/********** Compression/Decompression ************/
	inline std::uint8_t CompressDosage(double dosage)
	{
		dosage = std::min(dosage, L - 1e-6);
		return static_cast<std::uint16_t>(std::floor(dosage * invIntervalWidth));
	}

	inline double DecompressDosage(std::uint8_t compressed_dosage)
	{
		return (compressed_dosage + 0.5)*intervalWidth;
	}
};



#endif
