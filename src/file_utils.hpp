//
// Created by kerin on 12/11/2018.
//

#ifndef FILE_UTILS_HPP
#define FILE_UTILS_HPP

#include "parameters.hpp"
#include "genotype_matrix.hpp"
#include "bgen_wrapper.hpp"

#include "tools/eigen3.3/Dense"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/filesystem.hpp>

#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>

namespace boost_io = boost::iostreams;

/***************** File writing *****************/
namespace fileUtils {
long long getValueRAM(const std::string& field = "VmRSS:");

std::string fstream_init(boost_io::filtering_ostream &my_outf,
                         const std::string &file,
                         const std::string &file_prefix = "",
                         const std::string &file_suffix = "",
                         const bool& allow_gzip = true);

std::string filepath_format(const std::string& orig,
                            const std::string& file_prefix,
                            const std::string& file_suffix);

void dump_predicted_vec_to_file(Eigen::Ref<Eigen::MatrixXd> mat,
                                const std::string& filename,
                                const std::vector<std::string>& header,
                                const std::map<long, int>& sample_location);

void dump_predicted_vec_to_file(Eigen::Ref<Eigen::MatrixXd> mat,
                                const std::string& filename,
                                const std::string& header,
                                const std::map<long, int>& sample_location);

void write_snp_stats_to_file(boost_io::filtering_ostream &ofile,
                             const int &n_effects,
                             const GenotypeMatrix &X,
                             const bool &append,
                             const Eigen::Ref<const Eigen::MatrixXd> &neglogPvals,
                             const Eigen::Ref<const Eigen::MatrixXd> &testStats);

void write_snp_stats_to_file(boost_io::filtering_ostream &ofile, const int &n_effects,
                             const GenotypeMatrix &X,
                             const bool &append,
                             const Eigen::Ref<const Eigen::VectorXd> &neglogp_beta,
                             const Eigen::Ref<const Eigen::VectorXd> &neglogp_gam,
                             const Eigen::Ref<const Eigen::VectorXd> &neglogp_rgam,
                             const Eigen::Ref<const Eigen::VectorXd> &test_stat_beta,
                             const Eigen::Ref<const Eigen::VectorXd> &test_stat_gam,
                             const Eigen::Ref<const Eigen::VectorXd> &test_stat_rgam);

void write_snp_stats_to_file(boost_io::filtering_ostream &ofile, const int &n_effects,
                             const GenotypeMatrix &X,
                             const bool &append,
                             const Eigen::Ref<const Eigen::VectorXd> &neglogp_beta,
                             const Eigen::Ref<const Eigen::VectorXd> &neglogp_gam,
                             const Eigen::Ref<const Eigen::VectorXd> &neglogp_rgam,
                             const Eigen::Ref<const Eigen::VectorXd> &neglogp_joint,
                             const Eigen::Ref<const Eigen::VectorXd> &test_stat_beta,
                             const Eigen::Ref<const Eigen::VectorXd> &test_stat_gam,
                             const Eigen::Ref<const Eigen::VectorXd> &test_stat_rgam,
                             const Eigen::Ref<const Eigen::VectorXd> &test_stat_joint);

void read_bgen_metadata(const std::string& filename,
                        std::vector<int>& chr);

bool read_bgen_chunk(const bgenWrapper::View &bgenView,
                     GenotypeMatrix &G,
                     const std::unordered_map<long, bool> &sample_is_invalid,
                     const long &n_samples,
                     const long &chunk_size,
                     const parameters &p,
                     bool &bgen_pass,
                     long &n_var_parsed);

bool read_bgen_chunk(const bgenWrapper::View &bgenView,
                     Eigen::MatrixXd &G,
                     const std::unordered_map<long, bool> &sample_is_invalid,
                     const long &n_samples,
                     const long &chunk_size,
                     const parameters &p,
                     bool &bgen_pass,
                     long &n_var_parsed,
                     std::vector<std::string> &SNPIDS);
}

#endif //FILE_UTILS_HPP
