//
// Created by kerin on 2019-03-01.
//

#include "file_utils.hpp"
#include "parameters.hpp"
#include "genotype_matrix.hpp"

#include "genfile/bgen/bgen.hpp"
#include "genfile/bgen/View.hpp"
#include "bgen_parser.hpp"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/filesystem.hpp>

#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <set>

namespace boost_io = boost::iostreams;

long long parseLineRAM(char* line){
	// This assumes that a digit will be found and the line ends in " Kb".
	std::size_t i = strlen(line);
	const char* p = line;
	while (*p <'0' || *p > '9') p++;
	line[i-3] = '\0';
	char* s_end;
	long long res = std::stoll(p);
	return res;
}

long long fileUtils::getValueRAM(const std::string& field){
#ifndef OSX
	FILE* file = fopen("/proc/self/status", "r");
	long long result = -1;
	char line[128];

	while (fgets(line, 128, file) != NULL) {
		if (strncmp(line, field.c_str(), 6) == 0) {
			result = parseLineRAM(line);
			break;
		}
	}
	fclose(file);
	return result;
#else
	return -1;
#endif
}

std::string fileUtils::fstream_init(boost_io::filtering_ostream &my_outf,
                                    const std::string &file,
                                    const std::string &file_prefix,
                                    const std::string &file_suffix,
                                    const bool& allow_gzip) {

	std::string filepath   = file;
	std::string dir        = filepath.substr(0, filepath.rfind('/')+1);
	std::string stem_w_dir = filepath.substr(0, filepath.find('.'));
	std::string stem       = stem_w_dir.substr(stem_w_dir.rfind('/')+1, stem_w_dir.size());
	std::string ext        = filepath.substr(filepath.find('.'), filepath.size());
	if(!allow_gzip) {
		ext = ext.substr(0, ext.find(".gz"));
	}

	std::string ofile      = dir + file_prefix + stem + file_suffix + ext;

	// Allows prefix to contain subfolders
	boost::filesystem::path bfilepath(ofile);
	if(bfilepath.parent_path() != "" && !boost::filesystem::exists(bfilepath.parent_path())) {
		std::cout << "Creating parent path " << bfilepath.parent_path() << std::endl;
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

std::string fileUtils::filepath_format(const std::string& orig,
                                       const std::string& file_prefix,
                                       const std::string& file_suffix){
	std::string filepath   = orig;
	std::string dir        = filepath.substr(0, filepath.rfind('/')+1);
	std::string stem_w_dir = filepath.substr(0, filepath.find('.'));
	std::string stem       = stem_w_dir.substr(stem_w_dir.rfind('/')+1, stem_w_dir.size());
	std::string ext        = filepath.substr(filepath.find('.'), filepath.size());

	std::string ofile      = dir + file_prefix + stem + file_suffix + ext;
	return ofile;
}

void fileUtils::dump_predicted_vec_to_file(Eigen::Ref<Eigen::MatrixXd> mat,
                                           const std::string& filename,
                                           const std::vector<std::string>& header,
                                           const std::map<long, int>& sample_location){
	std::string header_string;
	for (int ii = 0; ii < header.size() - 1; ii++) {
		header_string += header[ii] + " ";
	}
	header_string += header[header.size() - 1];

	fileUtils::dump_predicted_vec_to_file(mat, filename, header_string, sample_location);
}

void fileUtils::dump_predicted_vec_to_file(Eigen::Ref<Eigen::MatrixXd> mat,
                                           const std::string& filename,
                                           const std::string& header,
                                           const std::map<long, int>& sample_location){
	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	std::vector<Eigen::MatrixXd> all_mat(world_size);
	long n_cols = mat.cols();

	if(world_rank == 0) {
		std::vector<long> all_n_samples(world_size);
		for (const auto &kv : sample_location) {
			if (kv.second != -1) {
				all_n_samples[kv.second]++;
			}
		}

		for (int rr = 1; rr < world_size; rr++) {
			all_mat[rr].resize(all_n_samples[rr], n_cols);
			MPI_Recv(all_mat[rr].data(), all_n_samples[rr] * n_cols, MPI_DOUBLE, rr, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		all_mat[0] = mat;
	} else {
		MPI_Send(mat.data(), mat.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}

	if(world_rank == 0) {
		boost_io::filtering_ostream outf;
		std::string gz_str = ".gz";
		if (filename.find(gz_str) != std::string::npos) {
			outf.push(boost_io::gzip_compressor());
		}
		outf.push(boost_io::file_sink(filename));

		std::vector<long> all_ii(world_size, 0);
		outf << header << std::endl;
		for (const auto &kv : sample_location) {
			if(kv.second == -1) {
				for (long cc = 0; cc < n_cols; cc++) {
					outf << "NA";
					outf << (cc != n_cols - 1 ? " " : "");
				}
				outf << std::endl;
			} else {
				for (long cc = 0; cc < n_cols; cc++) {
					outf << all_mat[kv.second](all_ii[kv.second], cc);
					outf << (cc != n_cols - 1 ? " " : "");
				}
				outf << std::endl;
				all_ii[kv.second]++;
			}
		}
		outf.pop();
		boost_io::close(outf);
	}
}

void fileUtils::write_snp_stats_to_file(boost_io::filtering_ostream &ofile, const int &n_effects,
                                        const GenotypeMatrix &X,
                                        const bool &append,
                                        const Eigen::Ref<const Eigen::VectorXd> &neglogp_beta,
                                        const Eigen::Ref<const Eigen::VectorXd> &neglogp_gam,
                                        const Eigen::Ref<const Eigen::VectorXd> &neglogp_rgam,
                                        const Eigen::Ref<const Eigen::VectorXd> &neglogp_joint,
                                        const Eigen::Ref<const Eigen::VectorXd> &test_stat_beta,
                                        const Eigen::Ref<const Eigen::VectorXd> &test_stat_gam,
                                        const Eigen::Ref<const Eigen::VectorXd> &test_stat_rgam,
                                        const Eigen::Ref<const Eigen::VectorXd> &test_stat_joint) {
	// Function to write parameter values from genetic effects to file
	// Assumes ofile has been initialised
	long n_var = X.cols();

	if(!append) {
		ofile << "SNPID chr rsid pos a0 a1 maf info";
		ofile << " tStat_main neglogP_main";
		if(n_effects > 1) {
			if (test_stat_gam.size() > 0) ofile << " tStat_gxe_standardSE";
			if (neglogp_gam.size() > 0) ofile << " neglogP_gxe_standardSE";
			ofile << " chiSqStat_gxe_robust" << " neglogP_gxe_robust";
//			if (test_stat_joint.size() > 0) ofile << " loco_f_stat";
//			if (neglogp_joint.size() > 0) ofile << " loco_f_neglogp";
		}
		ofile << std::endl;
	}

	ofile << std::scientific << std::setprecision(7);
	for (std::uint32_t kk = 0; kk < n_var; kk++) {
		ofile << X.SNPID[kk] << " " << X.chromosome[kk] << " " << X.rsid[kk] << " " << X.position[kk];
		ofile << " " << X.al_0[kk] << " " << X.al_1[kk] << " " << X.maf[kk] << " " << X.info[kk];

		// main effects
		ofile << " " << test_stat_beta(kk);
		ofile << " " << neglogp_beta(kk);

		// Interaction effects
		if(n_effects > 1) {
			if (test_stat_gam.size() > 0) ofile << " " << test_stat_gam(kk);
			if (neglogp_gam.size() > 0) ofile << " " << neglogp_gam(kk);
			ofile << " " << test_stat_rgam(kk);
			ofile << " " << neglogp_rgam(kk);
//			if (test_stat_joint.size() > 0) ofile << " " << test_stat_joint(kk);
//			if (neglogp_joint.size() > 0) ofile << " " << neglogp_joint(kk);
		}
		ofile << std::endl;
	}
}

void fileUtils::write_snp_stats_to_file(boost_io::filtering_ostream &ofile, const int &n_effects,
                                        const GenotypeMatrix &X,
                                        const bool &append,
                                        const Eigen::Ref<const Eigen::VectorXd> &neglogp_beta,
                                        const Eigen::Ref<const Eigen::VectorXd> &neglogp_gam,
                                        const Eigen::Ref<const Eigen::VectorXd> &neglogp_rgam,
                                        const Eigen::Ref<const Eigen::VectorXd> &test_stat_beta,
                                        const Eigen::Ref<const Eigen::VectorXd> &test_stat_gam,
                                        const Eigen::Ref<const Eigen::VectorXd> &test_stat_rgam) {
	Eigen::VectorXd emptyVec;
	fileUtils::write_snp_stats_to_file(ofile, n_effects, X, append,
	                                   neglogp_beta,   neglogp_gam,   neglogp_rgam,   emptyVec,
	                                   test_stat_beta, test_stat_gam, test_stat_rgam, emptyVec);
}

void fileUtils::write_snp_stats_to_file(boost_io::filtering_ostream &ofile,
                                        const int &n_effects,
                                        const GenotypeMatrix &X,
                                        const bool &append,
                                        const Eigen::Ref<const Eigen::MatrixXd> &neglogPvals,
                                        const Eigen::Ref<const Eigen::MatrixXd> &testStats){
	Eigen::VectorXd emptyVec;
	if (n_effects > 1) {
		fileUtils::write_snp_stats_to_file(ofile, n_effects, X, append,
										   neglogPvals.col(0), neglogPvals.col(1), neglogPvals.col(2), emptyVec,
										   testStats.col(0),   testStats.col(1),   testStats.col(2),   emptyVec);
	} else {
		fileUtils::write_snp_stats_to_file(ofile, n_effects, X, append,
										   neglogPvals.col(0), emptyVec, emptyVec, emptyVec,
										   testStats.col(0),   emptyVec, emptyVec, emptyVec);
	}
}

void fileUtils::read_bgen_metadata(const std::string& filename,
								   std::vector<int>& chr) {
	genfile::bgen::View::UniquePtr bgenView = genfile::bgen::View::create(filename);

	std::string   chr_j, rsid_j, SNPID_j;
	std::uint32_t pos_j;
	std::vector< std::string > alleles_j;
	bool bgen_pass = true;

	chr.clear();

	while (bgen_pass) {
		bgen_pass = bgenView->read_variant(&SNPID_j, &rsid_j, &chr_j, &pos_j, &alleles_j);
		if (bgen_pass){
			bgenView->ignore_genotype_data_block();
		}
		chr.push_back(std::stoi(chr_j));
	}
}

bool fileUtils::read_bgen_chunk(genfile::bgen::View::UniquePtr &bgenView,
                                GenotypeMatrix &G,
                                const std::unordered_map<long, bool> &sample_is_invalid,
                                const long &n_samples,
                                const long &chunk_size,
                                const parameters &p,
                                bool &bgen_pass,
                                long &n_var_parsed){
	// Wrapper around BgenView to read in a 'chunk' of data. Remembers
	// if last call hit the EOF, and returns false if so.

	// Exit function if last call hit EOF.
	if (!bgen_pass) return false;

	// Temporary variables to store info from read_variant()
	std::string chr_j;
	std::uint32_t pos_j;
	std::string rsid_j;
	std::vector< std::string > alleles_j;
	std::string SNPID_j;

	long nInvalid = sample_is_invalid.size() - n_samples;
	DosageSetter setter_v2(sample_is_invalid, nInvalid);

	double chunk_missingness = 0;
	long n_var_incomplete = 0;

	// Resize genotype matrix
	G.resize(n_samples, chunk_size);

	long int n_constant_variance = 0;
	std::uint32_t jj = 0;
	while ( jj < chunk_size && bgen_pass) {
		bgen_pass = bgenView->read_variant( &SNPID_j, &rsid_j, &chr_j, &pos_j, &alleles_j );
		if (!bgen_pass) break;
		n_var_parsed++;

		// Read dosage
		bgenView->read_genotype_data_block( setter_v2 );

		double d1     = setter_v2.m_sum_eij;
		double maf_j  = setter_v2.m_maf;
		double info_j = setter_v2.m_info;
		double mu     = setter_v2.m_mean;
		double missingness_j    = setter_v2.m_missingness;
		double sigma = std::sqrt(setter_v2.m_sigma2);

		// Filters
		if (p.maf_lim && (maf_j < p.min_maf || maf_j > 1 - p.min_maf)) {
			continue;
		}
		if (p.info_lim && info_j < p.min_info) {
			continue;
		}
		// if (p.missingness_lim && missingness_j > p.max_missingness) {
		//  continue;
		// }
		if(!p.keep_constant_variants && d1 < 5.0) {
			n_constant_variance++;
			continue;
		}
		if(!p.keep_constant_variants && sigma <= 1e-12) {
			n_constant_variance++;
			continue;
		}

		// filters passed; write contextual info
		chunk_missingness += missingness_j;
		if(missingness_j > 0) n_var_incomplete++;

		G.al_0[jj]     = alleles_j[0];
		G.al_1[jj]     = alleles_j[1];
		G.maf[jj]      = maf_j;
		G.info[jj]     = info_j;
		G.rsid[jj]     = rsid_j;
		G.chromosome[jj] = std::stoi(chr_j);
		G.position[jj] = pos_j;
		std::string key_j = chr_j + "~" + std::to_string(pos_j) + "~" + alleles_j[0] + "~" + alleles_j[1];
		G.SNPKEY[jj]   = key_j;
		G.SNPID[jj] = SNPID_j;

		for (std::uint32_t ii = 0; ii < n_samples; ii++) {
			G.assign_index(ii, jj, setter_v2.m_dosage[ii]);
		}
		// G.compressed_dosage_sds[jj] = sigma;
		// G.compressed_dosage_means[jj] = mu;

		jj++;
	}

	// need to resize G whilst retaining existing coefficients if while
	// loop exits early due to EOF.
	G.conservativeResize(n_samples, jj);
	assert( G.rsid.size() == jj );

	chunk_missingness /= jj;
	if(chunk_missingness > 0.0) {
		std::cout << "Average chunk missingness " << chunk_missingness << "(";
		std::cout << n_var_incomplete << "/" << G.cols();
		std::cout << " variants contain >=1 imputed entry)" << std::endl;
	}

	if(n_constant_variance > 0) {
		std::cout << " - " << n_constant_variance << " variants removed due to ";
		std::cout << "constant variance" << std::endl;
	}

	if(jj == 0) {
		// Immediate EOF
		return false;
	} else {
		return true;
	}
}

bool fileUtils::read_bgen_chunk(genfile::bgen::View::UniquePtr &bgenView,
                                Eigen::MatrixXd &G,
                                const std::unordered_map<long, bool> &sample_is_invalid,
                                const long &n_samples,
                                const long &chunk_size,
                                const parameters &p,
                                bool &bgen_pass,
                                long &n_var_parsed,
                                std::vector<std::string>& SNPIDS){
	// Wrapper around BgenView to read in a 'chunk' of data. Remembers
	// if last call hit the EOF, and returns false if so.

	// Exit function if last call hit EOF.
	if (!bgen_pass) return false;

	// Temporary variables to store info from read_variant()
	std::string chr_j;
	std::uint32_t pos_j;
	std::string rsid_j;
	std::vector< std::string > alleles_j;
	std::string SNPID_j;

	long nInvalid = sample_is_invalid.size() - n_samples;
	DosageSetter setter_v2(sample_is_invalid, nInvalid);

	double chunk_missingness = 0;
	long n_var_incomplete = 0;
	SNPIDS.clear();

	// Resize genotype matrix
	G.resize(n_samples, chunk_size);

	long int n_constant_variance = 0;
	std::uint32_t jj = 0;
	while ( jj < chunk_size && bgen_pass) {
		bgen_pass = bgenView->read_variant( &SNPID_j, &rsid_j, &chr_j, &pos_j, &alleles_j );
		if (!bgen_pass) break;
		n_var_parsed++;

		// Read probs + check maf filter
		bgenView->read_genotype_data_block( setter_v2 );

		double d1     = setter_v2.m_sum_eij;
		double maf_j  = setter_v2.m_maf;
		double info_j = setter_v2.m_info;
		double mu     = setter_v2.m_mean;
		double missingness_j    = setter_v2.m_missingness;
		double sigma = std::sqrt(setter_v2.m_sigma2);

		// Filters
		if (p.maf_lim && (maf_j < p.min_maf || maf_j > 1 - p.min_maf)) {
			continue;
		}
		if (p.info_lim && info_j < p.min_info) {
			continue;
		}
		if(!p.keep_constant_variants && d1 < 5.0) {
			n_constant_variance++;
			continue;
		}
		if(!p.keep_constant_variants && sigma <= 1e-12) {
			n_constant_variance++;
			continue;
		}

		SNPIDS.push_back(SNPID_j);

		// filters passed; write contextual info
		chunk_missingness += missingness_j;
		if(missingness_j > 0) n_var_incomplete++;

		for (std::uint32_t ii = 0; ii < n_samples; ii++) {
			G(ii, jj) = setter_v2.m_dosage[ii];
		}
		jj++;
	}

	// need to resize G whilst retaining existing coefficients if while
	// loop exits early due to EOF.
	G.conservativeResize(n_samples, jj);

//	chunk_missingness /= jj;
//	if(chunk_missingness > 0.0) {
//		std::cout << "Average chunk missingness " << chunk_missingness << "(";
//		std::cout << n_var_incomplete << "/" << G.cols();
//		std::cout << " variants contain >=1 imputed entry)" << std::endl;
//	}
//
//	if(n_constant_variance > 0) {
//		std::cout << n_constant_variance << " variants removed due to ";
//		std::cout << "constant variance" << std::endl;
//	}

	if(jj == 0) {
		// Immediate EOF
		return false;
	} else {
		return true;
	}
}
