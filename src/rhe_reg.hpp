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
#include "eigen_utils.hpp"
#include "rhe_reg_component.hpp"

#include "tools/eigen3.3/Dense"

#include <boost/iostreams/filtering_stream.hpp>

#include <random>

namespace boost_io = boost::iostreams;


class RHEreg {
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

	const parameters& p;
	const GenotypeMatrix& X;

	Eigen::VectorXd eta;
	Eigen::VectorXd Y;
	Eigen::MatrixXd& C;
	Eigen::MatrixXd CtC_inv;
	Eigen::ArrayXd sigmas, sigmasb;
	Eigen::ArrayXXd sigmas_jack, h2_jack;
	Eigen::ArrayXd h2, h2_se_jack, h2_bias_corrected;
	Eigen::ArrayXd n_var_jack;

	EigenDataMatrix zz;
	EigenDataMatrix Wzz;
	const std::unordered_map<long, bool>& sample_is_invalid;
	Data& data;

	// std::vector<std::string> components;
	std::vector<RHEreg_Component> components;
	std::vector<std::string> SNPGROUPS_snpid, SNPGROUPS_group, all_SNPGROUPS;

	RHEreg(Data& dat,
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

	RHEreg(Data& dat,
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

	void run();

	void initialise_components();

	void calc_RHE();

	void optim_RHE_LEMMA();

	Eigen::MatrixXd construct_vc_system(const std::vector<RHEreg_Component>& components);

	Eigen::ArrayXd calc_h2(Eigen::Ref<Eigen::MatrixXd> AA,
	                       Eigen::Ref<Eigen::VectorXd> bb,
	                       const bool& reweight_sigmas = false);

	void process_jacknife_samples();

	Eigen::MatrixXd project_out_covars(Eigen::Ref<Eigen::MatrixXd> rhs);

	void fill_gaussian_noise(unsigned int seed,
	                         Eigen::Ref<Eigen::MatrixXd> zz,
	                         long nn,
	                         long n_draws);

	void to_file(const std::string& file);

	void read_RHE_groups(const std::string& filename);

	double get_jacknife_var(Eigen::ArrayXd jack_estimates);

	double get_jacknife_bias_correct(Eigen::ArrayXd jack_estimates, double full_data_est);
};

#endif //BGEN_PROG_PVE_HPP
