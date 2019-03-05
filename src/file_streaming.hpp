//
// Created by kerin on 12/11/2018.
//

#ifndef BGEN_PROG_MISC_UTILS_HPP
#define BGEN_PROG_MISC_UTILS_HPP

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

/***************** File writing *****************/

std::string fstream_init(boost_io::filtering_ostream& my_outf,
						 const std::string& file,
						 const std::string& file_prefix,
						 const std::string& file_suffix);

void write_snp_stats_to_file(boost_io::filtering_ostream& ofile,
							 const int& n_effects,
							 const int& n_var,
							 const VariationalParameters& vp,
							 const GenotypeMatrix& X,
							 const parameters& p,
							 const bool& write_mog);

void write_snp_stats_to_file(boost_io::filtering_ostream& ofile,
							 const int& n_effects,
							 const int& n_var,
							 const VariationalParametersLite& vp,
							 const GenotypeMatrix& X,
							 const parameters& p,
							 const bool& write_mog);

void write_snp_stats_to_file(boost_io::filtering_ostream& ofile,
							 const int& n_effects,
							 const int& n_var,
							 const VariationalParametersLite& vp,
							 const GenotypeMatrix& X,
							 const parameters& p,
							 const bool& write_mog,
							 const Eigen::Ref<const Eigen::VectorXd>& neglogp_beta,
							 const Eigen::Ref<const Eigen::VectorXd>& neglogp_gam,
							 const Eigen::Ref<const Eigen::VectorXd>& neglogp_rgam,
							 const Eigen::Ref<const Eigen::VectorXd>& neglogp_joint,
							 const Eigen::Ref<const Eigen::VectorXd>& test_stat_beta,
							 const Eigen::Ref<const Eigen::VectorXd>& test_stat_gam,
							 const Eigen::Ref<const Eigen::VectorXd>& test_stat_rgam,
							 const Eigen::Ref<const Eigen::VectorXd>& test_stat_joint);


#endif //BGEN_PROG_MISC_UTILS_HPP
