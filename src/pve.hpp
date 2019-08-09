//
// Created by kerin on 2019-01-08.
//

#ifndef BGEN_PROG_PVE_HPP
#define BGEN_PROG_PVE_HPP

#include "genotype_matrix.hpp"
#include "file_utils.hpp"
#include "parameters.hpp"
#include "eigen_utils.hpp"
#include "mpi_utils.hpp"
#include "data.hpp"

#include <boost/iostreams/filtering_stream.hpp>

#include <random>

namespace boost_io = boost::iostreams;

struct Index_t {
	Index_t() : main(0), noise(1) {
	}
	long main, gxe, noise;
};

class PVE_Component {
public:
	Eigen::MatrixXd _XXtz, _XXtWz;
	double ytXXty;
	std::string label;
	long n_env;
	long n_covar;
	long n_samples;
	long n_draws;
	long n_jacknife_local;
	long rm_jacknife_block;
	double n_var_local;
	parameters params;
	bool is_active;

	// Storage for jacknife blocks
	std::vector<Eigen::MatrixXd> _XXtzs, _XXtWzs;
	std::vector<double> n_vars_local;
	std::vector<double> ytXXtys;

	Eigen::MatrixXd& C, CtC_inv, zz, Wzz;
	Eigen::VectorXd Y;
	Eigen::VectorXd eta;
	PVE_Component(const parameters& myparams,
	              Eigen::VectorXd& myY,
	              Eigen::MatrixXd& myzz,
	              Eigen::MatrixXd& myWzz,
	              Eigen::MatrixXd& myC,
	              Eigen::MatrixXd& myCtC_inv,
	              const long& myNJacknifeLocal) : params(myparams), Y(myY),
		zz(myzz), Wzz(myWzz), C(myC), CtC_inv(myCtC_inv), n_jacknife_local(myNJacknifeLocal) {
		assert(n_jacknife_local > 0);
		n_covar = C.cols();
		n_samples = zz.rows();
		n_draws = zz.cols();

		n_env = 0;
		label = "";
		is_active = true;
		rm_jacknife_block = -1;

		ytXXty = 0;
		ytXXtys.resize(n_jacknife_local, 0);

		n_var_local = 0;
		n_vars_local.resize(n_jacknife_local, 0);

		_XXtzs.resize(n_jacknife_local);
		_XXtWzs.resize(n_jacknife_local);
		for(long ii = 0; ii < n_jacknife_local; ii++) {
			Eigen::MatrixXd mm = Eigen::MatrixXd::Zero(n_samples, n_draws);
			_XXtzs[ii] = mm;
			_XXtWzs[ii] = mm;
		}

	}

	void set_eta(Eigen::Ref<Eigen::VectorXd> myeta){
		assert(is_active);
		n_env = 1;
		eta = myeta;
		Y.array() *= eta.array();
		zz.array().colwise() *= eta.array();
		Wzz.array().colwise() *= eta.array();
	}

	void set_inactive(){
		// The inactive component corresponds to sigma_e
		// Ie the 'noise' component
		assert(n_env == 0);
		is_active = false;
		_XXtz = zz;
		_XXtWz = Wzz;
		n_var_local = 1;
		ytXXty = Y.squaredNorm();
	}

	void add_to_trace_estimator(Eigen::Ref<Eigen::MatrixXd> X,
	                            long jacknife_index = 0){
		assert(jacknife_index < n_jacknife_local);
		if(is_active) {
			Eigen::MatrixXd Xty = X.transpose() * Y;
			Xty = mpiUtils::mpiReduce_inplace(Xty);
			ytXXtys[jacknife_index] += Xty.squaredNorm();
			Eigen::MatrixXd Xtz = X.transpose() * zz;
			Xtz = mpiUtils::mpiReduce_inplace(Xtz);
			_XXtzs[jacknife_index] += X * Xtz;
			if(n_covar > 0) {
				Eigen::MatrixXd XtWz = X.transpose() * Wzz;
				XtWz = mpiUtils::mpiReduce_inplace(XtWz);
				_XXtWzs[jacknife_index] += X * XtWz;
			}
			n_vars_local[jacknife_index] += X.cols();
		}
	}

	void finalise(){
		// Sum over the different jacknife blocks;
		if(is_active) {
			if(n_env > 0) {
				for (auto& mm : _XXtzs) {
					mm.array().colwise() *= eta.array();
				}
				for (auto& mm : _XXtWzs) {
					mm.array().colwise() *= eta.array();
				}
			}

			_XXtz = Eigen::MatrixXd::Zero(n_samples, n_draws);
			_XXtWz = Eigen::MatrixXd::Zero(n_samples, n_draws);
			for (auto& mm : _XXtzs) {
				_XXtz += mm;
			}
			for (auto& mm : _XXtWzs) {
				_XXtWz += mm;
			}

			n_var_local = std::accumulate(n_vars_local.begin(), n_vars_local.end(), 0.0);
			ytXXty = std::accumulate(ytXXtys.begin(), ytXXtys.end(), 0.0);
		}
	}

	Eigen::MatrixXd getXXtz() const {
		if(rm_jacknife_block >= 0) {
			return (_XXtz - _XXtzs[rm_jacknife_block]);
		} else {
			return _XXtz;
		}
	}

	Eigen::MatrixXd getXXtWz() const {
		if(rm_jacknife_block >= 0) {
			return (_XXtWz - _XXtWzs[rm_jacknife_block]);
		} else {
			return _XXtWz;
		}
	}

	double get_bb_trace() const {
		if(rm_jacknife_block >= 0) {
			return (ytXXty - ytXXtys[rm_jacknife_block]) / get_n_var_local();
		} else {
			return ytXXty / get_n_var_local();
		}
	}

	double get_n_var_local() const {
		if(rm_jacknife_block >= 0) {
			return n_var_local - n_vars_local[rm_jacknife_block];
		} else {
			return n_var_local;
		}
	}

	double operator*(const PVE_Component& other) const {
		double res;
		if(n_covar == 0) {
			res = getXXtz().cwiseProduct(other.getXXtz()).sum();
		} else if (n_covar > 0) {
			if(label == "noise" || other.label == "noise") {
				res = getXXtz().cwiseProduct(other.getXXtWz()).sum();
			} else {
				Eigen::MatrixXd XXtz = getXXtz();
				Eigen::MatrixXd WXXtz = project_out_covars(XXtz);
				res = WXXtz.cwiseProduct(other.getXXtWz()).sum();
			}
		} else {
			throw std::runtime_error("Error in PVE_Component");
		}
		return res / get_n_var_local() / other.get_n_var_local() / (double) n_draws;
	}

	Eigen::MatrixXd project_out_covars(Eigen::Ref<Eigen::MatrixXd> rhs) const {
		return EigenUtils::project_out_covars(rhs, C, CtC_inv, params.mode_debug);
	}
};

class PVE {
public:
	// constants
	long n_draws;
	long n_samples;
	long n_components;
	long n_covar;
	long n_env;
	long n_var;
	double Nglobal;
	int world_rank, world_size;

	std::map<long, int> sample_location;

	parameters p;

	const GenotypeMatrix& X;

	Eigen::VectorXd eta;
	Eigen::VectorXd Y;
	Eigen::MatrixXd& C;
	Eigen::MatrixXd CtC_inv;
	Eigen::ArrayXd sigmas, sigmasb;
	Eigen::ArrayXXd sigmas_jack, h2_jack, h2b_jack;
	Eigen::ArrayXd h2, h2_se_jack, h2_bias_corrected;
	Eigen::ArrayXd h2b, h2b_se_jack, h2b_bias_corrected;
	Eigen::ArrayXd n_var_jack;

	EigenDataMatrix zz;
	EigenDataMatrix Wzz;
	const std::unordered_map<long, bool>& sample_is_invalid;
	Data& data;

	// std::vector<std::string> components;
	std::vector<PVE_Component> components;
	Index_t ind;

	PVE(Data& dat,
	    Eigen::VectorXd& myY,
	    Eigen::MatrixXd& myC,
	    Eigen::VectorXd& myeta) : p(dat.p), X(dat.G), eta(myeta), Y(myY), C(myC),
		sample_is_invalid(dat.sample_is_invalid),
		sample_location(dat.sample_location), data(dat) {
		n_samples = data.n_samples;
		Nglobal = mpiUtils::mpiReduce_inplace(n_samples);
		n_draws = p.n_pve_samples;

		n_covar = C.cols();
		n_env = 1;

		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	}

	PVE(Data& dat,
	    Eigen::VectorXd& myY,
	    Eigen::MatrixXd& myC) : p(dat.p), X(dat.G), Y(myY), C(myC),
		sample_is_invalid(dat.sample_is_invalid),
		sample_location(dat.sample_location), data(dat) {
		n_samples = data.n_samples;
		Nglobal = mpiUtils::mpiReduce_inplace(n_samples);
		n_draws = p.n_pve_samples;

		n_covar = C.cols();
		n_env = 0;

		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	}

	void initialise_components();

	void calc_RHE();

	Eigen::MatrixXd construct_vc_system(const std::vector<PVE_Component>& components);

	Eigen::ArrayXd calc_h2(Eigen::Ref<Eigen::MatrixXd> AA,
	                       Eigen::Ref<Eigen::VectorXd> bb,
	                       const bool& reweight_sigmas = false);

	void process_jacknife_samples();

	Eigen::MatrixXd project_out_covars(Eigen::Ref<Eigen::MatrixXd> rhs);

	void run();

	void fill_gaussian_noise(unsigned int seed,
	                         Eigen::Ref<Eigen::MatrixXd> zz,
	                         long nn,
	                         long n_draws);

	void to_file(const std::string& file);

	double get_jacknife_var(Eigen::ArrayXd jack_estimates);

	double get_jacknife_bias_correct(Eigen::ArrayXd jack_estimates, double full_data_est);
};

#endif //BGEN_PROG_PVE_HPP
