//
// Created by kerin on 2019-03-01.
//

#include "file_streaming.hpp"
#include "parameters.hpp"
#include "genotype_matrix.hpp"
#include "variational_parameters.hpp"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/filesystem.hpp>

#include <iomanip>
#include <string>
#include <vector>

namespace boost_io = boost::iostreams;

std::string fstream_init(boost_io::filtering_ostream &my_outf, const std::string &file, const std::string &file_prefix,
                         const std::string &file_suffix) {

	std::string filepath   = file;
	std::string dir        = filepath.substr(0, filepath.rfind('/')+1);
	std::string stem_w_dir = filepath.substr(0, filepath.find('.'));
	std::string stem       = stem_w_dir.substr(stem_w_dir.rfind('/')+1, stem_w_dir.size());
	std::string ext        = filepath.substr(filepath.find('.'), filepath.size());

	std::string ofile      = dir + file_prefix + stem + file_suffix + ext;

	// Allows prefix to contain subfolders
	boost::filesystem::path bfilepath(ofile);
	if(!boost::filesystem::exists(bfilepath.parent_path())) {
		boost::filesystem::create_directories(bfilepath.parent_path());
	}

	my_outf.reset();
	std::string gz_str = ".gz";
	if (file.find(gz_str) != std::string::npos) {
		my_outf.push(boost_io::gzip_compressor());
	}
	my_outf.push(boost_io::file_sink(ofile));
	return ofile;
}

void write_snp_stats_to_file(boost_io::filtering_ostream &ofile, const int &n_effects, const int &n_var,
                             const VariationalParameters &vp, const GenotypeMatrix &X, const parameters &p,
                             const bool &write_mog) {
	// Function to write parameter values from genetic effects to file
	// Assumes ofile has been initialised

	ofile << "chr rsid pos a0 a1 maf info";
	ofile << " mean_beta " << vp.betas_header("beta");
	if(n_effects > 1) {
		ofile << " mean_gam " << vp.gammas_header("gam");
	}
	ofile << std::endl;

	ofile << std::scientific << std::setprecision(7);

	for (std::uint32_t kk = 0; kk < n_var; kk++) {
		ofile << X.chromosome[kk] << " " << X.rsid[kk] << " " << X.position[kk];
		ofile << " " << X.al_0[kk] << " " << X.al_1[kk] << " " << X.maf[kk] << " " << X.info[kk];
		ofile << " " << vp.mean_beta(kk) << " ";
		vp.write_ith_beta_to_stream(kk, ofile);
		if(n_effects > 1) {
			ofile << " " << vp.mean_gam(kk) << " ";
			vp.write_ith_gamma_to_stream(kk, ofile);
		}
		ofile << std::endl;
	}
}

void write_snp_stats_to_file(boost_io::filtering_ostream &ofile, const int &n_effects, const int &n_var,
                             const VariationalParametersLite &vp, const GenotypeMatrix &X, const parameters &p,
                             const bool &write_mog) {
	// Function to write parameter values from genetic effects to file
	// Assumes ofile has been initialised

	ofile << "chr rsid pos a0 a1 maf info";
	ofile << " mean_beta " << vp.betas_header("beta");
	if(n_effects > 1) {
		ofile << " mean_gam " << vp.gammas_header("gam");
	}
	ofile << std::endl;
	ofile << std::scientific << std::setprecision(7);

	for (std::uint32_t kk = 0; kk < n_var; kk++) {
		ofile << X.chromosome[kk] << " " << X.rsid[kk] << " " << X.position[kk];
		ofile << " " << X.al_0[kk] << " " << X.al_1[kk] << " " << X.maf[kk] << " " << X.info[kk];
		ofile << " " << vp.mean_beta(kk) << " ";
		vp.write_ith_beta_to_stream(kk, ofile);
		if(n_effects > 1) {
			ofile << " " << vp.mean_gam(kk) << " ";
			vp.write_ith_gamma_to_stream(kk, ofile);
		}
		ofile << std::endl;
	}
}

void write_snp_stats_to_file(boost_io::filtering_ostream &ofile, const int &n_effects, const int &n_var,
                             const VariationalParametersLite &vp, const GenotypeMatrix &X, const parameters &p,
                             const bool &write_mog, const Eigen::Ref<const Eigen::VectorXd> &neglogp_beta,
                             const Eigen::Ref<const Eigen::VectorXd> &neglogp_gam,
                             const Eigen::Ref<const Eigen::VectorXd> &neglogp_rgam,
                             const Eigen::Ref<const Eigen::VectorXd> &neglogp_joint,
                             const Eigen::Ref<const Eigen::VectorXd> &test_stat_beta,
                             const Eigen::Ref<const Eigen::VectorXd> &test_stat_gam,
                             const Eigen::Ref<const Eigen::VectorXd> &test_stat_rgam,
                             const Eigen::Ref<const Eigen::VectorXd> &test_stat_joint) {
	// Function to write parameter values from genetic effects to file
	// Assumes ofile has been initialised

	ofile << "chr rsid pos a0 a1 maf info";
	ofile << " beta0 " << vp.betas_header("beta");
	ofile << " loco_t_stat0 loco_t_neglogp0";
	if(n_effects > 1) {
		ofile << " beta1 " << vp.gammas_header("gam");
		ofile << " loco_t_stat1 loco_t_neglogp1";
		ofile << " loco_chi_stat loco_robust_neglogp";
		ofile << " loco_f_stat loco_f_neglogp";
	}
	ofile << std::endl;

	ofile << std::scientific << std::setprecision(7);
	for (std::uint32_t kk = 0; kk < n_var; kk++) {
		ofile << X.chromosome[kk] << " " << X.rsid[kk] << " " << X.position[kk];
		ofile << " " << X.al_0[kk] << " " << X.al_1[kk] << " " << X.maf[kk] << " " << X.info[kk];

		// main effects
		ofile << " " << vp.mean_beta(kk);
		ofile << " ";
		vp.write_ith_beta_to_stream(kk, ofile);
		ofile << " " << test_stat_beta(kk);
		ofile << " " << neglogp_beta(kk);

		// Interaction effects
		if(n_effects > 1) {
			ofile << " " << vp.mean_gam(kk);
			ofile << " ";
			vp.write_ith_gamma_to_stream(kk, ofile);
			ofile << " " << test_stat_gam(kk);
			ofile << " " << neglogp_gam(kk);
			ofile << " " << test_stat_rgam(kk);
			ofile << " " << neglogp_rgam(kk);
			ofile << " " << test_stat_joint(kk);
			ofile << " " << neglogp_joint(kk);
		}
		ofile << std::endl;
	}
}
