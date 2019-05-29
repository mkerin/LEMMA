//
// Created by kerin on 2019-03-01.
//

#include "file_utils.hpp"
#include "parameters.hpp"
#include "genotype_matrix.hpp"
#include "variational_parameters.hpp"

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

std::string fileUtils::fstream_init(boost_io::filtering_ostream &my_outf, const std::string &file,
                                    const std::string &file_prefix,
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

//std::string variational_params_header(const parameters& p, const long& n_effects){
//	std::string header = "";
//	header += "beta0 alpha0 mu0 s_sq0";
//	if(p.mode_mog_prior_beta) header += " mu_spike0 s_sq_spike0";
//	if(n_effects > 1){
//		header += "beta1 alpha1 mu1 s_sq1";
//		if(p.mode_mog_prior_gam) header += " mu_spike1 s_sq_spike1";
//	}
//	return header;
//}

std::string variational_params_header(const parameters& p, const int& effect_index){
	std::string header = "";
	assert(effect_index == 0 || effect_index == 1);
	if(effect_index == 0) {
		header += "beta0 alpha0 mu0 s_sq0";
		if(p.mode_mog_prior_beta) header += " mu_spike0 s_sq_spike0";
	} else {
		header += "beta1 alpha1 mu1 s_sq1";
		if(p.mode_mog_prior_gam) header += " mu_spike1 s_sq_spike1";
	}
	return header;
}

void fileUtils::write_snp_stats_to_file(boost_io::filtering_ostream &ofile, const int &n_effects, const int &n_var,
                                        const VariationalParameters &vp, const GenotypeMatrix &X, const parameters &p,
                                        const bool &write_mog) {
	// Function to write parameter values from genetic effects to file
	// Assumes ofile has been initialised

	ofile << "chr rsid pos a0 a1 maf info";
	for (int ee = 0; ee < n_effects; ee++) {
		ofile << " " << variational_params_header(p, ee);
	}
	ofile << std::endl;

	Eigen::ArrayXXd mean_beta  = vp.alpha_beta * vp.mu1_beta;
	if(p.mode_mog_prior_beta) mean_beta += (1 - vp.alpha_beta) * vp.mu2_beta;

	Eigen::ArrayXXd mean_gam  = vp.alpha_gam * vp.mu1_gam;
	if(p.mode_mog_prior_beta) mean_gam += (1 - vp.alpha_gam) * vp.mu2_gam;

	ofile << std::scientific << std::setprecision(7);

	for (std::uint32_t kk = 0; kk < n_var; kk++) {
		ofile << X.chromosome[kk] << " " << X.rsid[kk] << " " << X.position[kk];
		ofile << " " << X.al_0[kk] << " " << X.al_1[kk] << " " << X.maf[kk] << " " << X.info[kk];

		// main effects
		ofile << " " << mean_beta(kk);
		ofile << " " << vp.alpha_beta(kk);
		ofile << " " << vp.mu1_beta(kk);
		ofile << " " << vp.s1_beta_sq(kk);
		if(write_mog && p.mode_mog_prior_beta) {
			ofile << " " << vp.mu2_beta(kk);
			ofile << " " << vp.s2_beta_sq(kk);
		}

		// Interaction effects
		if(n_effects > 1) {
			ofile << " " << mean_gam(kk);
			ofile << " " << vp.alpha_gam(kk);
			ofile << " " << vp.mu1_gam(kk);
			ofile << " " << vp.s1_gam_sq(kk);
			if (write_mog && p.mode_mog_prior_gam) {
				ofile << " " << vp.mu2_gam(kk);
				ofile << " " << vp.s2_gam_sq(kk);
			}
		}
		ofile << std::endl;
	}
}

void fileUtils::write_snp_stats_to_file(boost_io::filtering_ostream &ofile, const int &n_effects, const int &n_var,
                                        const VariationalParametersLite &vp, const GenotypeMatrix &X, const parameters &p,
                                        const bool &write_mog) {
	// Function to write parameter values from genetic effects to file
	// Assumes ofile has been initialised

	ofile << "chr rsid pos a0 a1 maf info";
	for (int ee = 0; ee < n_effects; ee++) {
		ofile << " " << variational_params_header(p, ee);
	}
	ofile << std::endl;

	Eigen::ArrayXXd mean_beta  = vp.alpha_beta * vp.mu1_beta;
	if(p.mode_mog_prior_beta) mean_beta += (1 - vp.alpha_beta) * vp.mu2_beta;

	Eigen::ArrayXXd mean_gam  = vp.alpha_gam * vp.mu1_gam;
	if(p.mode_mog_prior_beta) mean_gam += (1 - vp.alpha_gam) * vp.mu2_gam;

	ofile << std::scientific << std::setprecision(7);

	for (std::uint32_t kk = 0; kk < n_var; kk++) {
		ofile << X.chromosome[kk] << " " << X.rsid[kk] << " " << X.position[kk];
		ofile << " " << X.al_0[kk] << " " << X.al_1[kk] << " " << X.maf[kk] << " " << X.info[kk];

		// main effects
		ofile << " " << mean_beta(kk);
		ofile << " " << vp.alpha_beta(kk);
		ofile << " " << vp.mu1_beta(kk);
		ofile << " " << vp.s1_beta_sq(kk);
		if(write_mog && p.mode_mog_prior_beta) {
			ofile << " " << vp.mu2_beta(kk);
			ofile << " " << vp.s2_beta_sq(kk);
		}

		// Interaction effects
		if(n_effects > 1) {
			ofile << " " << mean_gam(kk);
			ofile << " " << vp.alpha_gam(kk);
			ofile << " " << vp.mu1_gam(kk);
			ofile << " " << vp.s1_gam_sq(kk);
			if (write_mog && p.mode_mog_prior_gam) {
				ofile << " " << vp.mu2_gam(kk);
				ofile << " " << vp.s2_gam_sq(kk);
			}
		}
		ofile << std::endl;
	}
}

void fileUtils::write_snp_stats_to_file(boost_io::filtering_ostream &ofile, const int &n_effects, const int &n_var,
                                        const VariationalParametersLite &vp, const GenotypeMatrix &X,
                                        const parameters &p,
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
	ofile << " " << variational_params_header(p, 0);
	ofile << " loco_t_stat" << 0 << " loco_t_neglogp" << 0;
	if(n_effects > 1) {
		ofile << " " << variational_params_header(p, 1);
		if (test_stat_gam.size() > 0) ofile << " loco_t_stat" << 1;
		if (neglogp_gam.size() > 0) ofile << " loco_t_neglogp" << 1;
		ofile << " loco_chi_stat" << " loco_robust_neglogp" << " loco_f_stat" << " loco_f_neglogp";
	}
	ofile << std::endl;

	Eigen::ArrayXXd mean_beta  = vp.alpha_beta * vp.mu1_beta;
	if(p.mode_mog_prior_beta) mean_beta += (1 - vp.alpha_beta) * vp.mu2_beta;

	Eigen::ArrayXXd mean_gam  = vp.alpha_gam * vp.mu1_gam;
	if(p.mode_mog_prior_beta) mean_gam += (1 - vp.alpha_gam) * vp.mu2_gam;

	ofile << std::scientific << std::setprecision(7);
	for (std::uint32_t kk = 0; kk < n_var; kk++) {
		ofile << X.chromosome[kk] << " " << X.rsid[kk] << " " << X.position[kk];
		ofile << " " << X.al_0[kk] << " " << X.al_1[kk] << " " << X.maf[kk] << " " << X.info[kk];

		// main effects
		ofile << " " << mean_beta(kk);
		ofile << " " << vp.alpha_beta(kk);
		ofile << " " << vp.mu1_beta(kk);
		ofile << " " << vp.s1_beta_sq(kk);
		if(write_mog && p.mode_mog_prior_beta) {
			ofile << " " << vp.mu2_beta(kk);
			ofile << " " << vp.s2_beta_sq(kk);
		}
		ofile << " " << test_stat_beta(kk);
		ofile << " " << neglogp_beta(kk);

		// Interaction effects
		if(n_effects > 1) {
			ofile << " " << mean_gam(kk);
			ofile << " " << vp.alpha_gam(kk);
			ofile << " " << vp.mu1_gam(kk);
			ofile << " " << vp.s1_gam_sq(kk);
			if (write_mog && p.mode_mog_prior_gam) {
				ofile << " " << vp.mu2_gam(kk);
				ofile << " " << vp.s2_gam_sq(kk);
			}
			if (test_stat_gam.size() > 0) ofile << " " << test_stat_gam(kk);
			if (neglogp_gam.size() > 0) ofile << " " << neglogp_gam(kk);
			ofile << " " << test_stat_rgam(kk);
			ofile << " " << neglogp_rgam(kk);
			ofile << " " << test_stat_joint(kk);
			ofile << " " << neglogp_joint(kk);
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
                                        const Eigen::Ref<const Eigen::VectorXd> &neglogp_joint,
                                        const Eigen::Ref<const Eigen::VectorXd> &test_stat_beta,
                                        const Eigen::Ref<const Eigen::VectorXd> &test_stat_gam,
                                        const Eigen::Ref<const Eigen::VectorXd> &test_stat_rgam,
                                        const Eigen::Ref<const Eigen::VectorXd> &test_stat_joint) {
	// Function to write parameter values from genetic effects to file
	// Assumes ofile has been initialised
	long n_var = X.cols();

	if(!append) {
		ofile << "chr rsid pos a0 a1 maf info";
		ofile << " loco_t_stat0 loco_t_neglogp0";
		if(n_effects > 1) {
			if (neglogp_gam.size() > 0) ofile << " loco_t_stat1";
			if (test_stat_gam.size() > 0) ofile << " loco_t_neglogp1";
			ofile << " loco_chi_stat" << " loco_robust_neglogp";
			ofile << " loco_f_stat" << " loco_f_neglogp";
		}
		ofile << std::endl;
	}

	ofile << std::scientific << std::setprecision(7);
	for (std::uint32_t kk = 0; kk < n_var; kk++) {
		ofile << X.chromosome[kk] << " " << X.rsid[kk] << " " << X.position[kk];
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
			ofile << " " << test_stat_joint(kk);
			ofile << " " << neglogp_joint(kk);
		}
		ofile << std::endl;
	}
}

bool fileUtils::read_bgen_chunk(genfile::bgen::View::UniquePtr &bgenView,
                                GenotypeMatrix &G,
                                const std::unordered_map<long, bool> &sample_is_invalid,
                                const long &n_samples,
                                const long &chunk_size,
                                const parameters &p,
                                bool &bgen_pass,
                                long &n_var_parsed) {
	// Wrapper around BgenView to read in a 'chunk' of data. Remembers
	// if last call hit the EOF, and returns false if so.
	// Assumed that:
	// - commandline args parsed and passed to params
	// - bgenView initialised with correct filename
	// - scale + centering happening internally

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
		std::cout << n_constant_variance << " variants removed due to ";
		std::cout << "constant variance" << std::endl;
	}

	if(jj == 0) {
		// Immediate EOF
		return false;
	} else {
		return true;
	}
}

void fileUtils::dump_yhat_to_file(boost_io::filtering_ostream &outf, const long &n_samples, const long &n_covar,
                                  const long &n_var, const int &n_env, const EigenDataVector &Y,
                                  const VariationalParametersLite &vp, const Eigen::Ref<const Eigen::VectorXd> &Ealpha,
                                  std::unordered_map<long, bool> sample_is_invalid){
	std::vector<Eigen::VectorXd> resid_loco;
	std::vector<int> chrs_present;

	fileUtils::dump_yhat_to_file(outf, n_samples, n_covar,
	                             n_var, n_env, Y,
	                             vp, Ealpha,
	                             sample_is_invalid,
	                             resid_loco, chrs_present);
}

void fileUtils::dump_yhat_to_file(boost_io::filtering_ostream &outf, const long &n_samples, const long &n_covar,
                                  const long &n_var, const int &n_env, const EigenDataVector &Y,
                                  const VariationalParametersLite &vp, const Eigen::Ref<const Eigen::VectorXd> &Ealpha,
                                  std::unordered_map<long, bool> sample_is_invalid,
                                  const std::vector<Eigen::VectorXd> &resid_loco, std::vector<int> chrs_present) {
	assert(chrs_present.size() == resid_loco.size() || resid_loco.empty());

	// header
	long n_cols = 0, n_chrs = resid_loco.size();

	outf << "Y"; n_cols++;
	if (n_covar > 0) outf << " Ealpha"; n_cols++;
	outf << " Xbeta"; n_cols++;
	if (n_env > 0) outf << " eta Xgamma"; n_cols += 2;
	if(n_chrs > 0) {
		for(auto cc : chrs_present) {
			outf << " residuals_excl_chr" << cc;
		}
		n_cols += n_chrs;
	}
	outf << std::endl;

	// body
	long iiValid = 0;
	for (long ii = 0; ii < sample_is_invalid.size(); ii++) {
		if(sample_is_invalid[ii]) {
			// NA for incomplete sample
			for (int jj = 0; jj < n_cols; jj++) {
				outf << "NA";
				if(jj < n_cols - 1) outf << " ";
			}
			outf << std::endl;
		} else {
			// sample info
			outf << Y(iiValid);
			if (n_covar > 0) {
				outf << " " << Ealpha(iiValid) << " " << vp.ym(iiValid) - Ealpha(iiValid);
			} else {
				outf << " " << vp.ym(iiValid);
			}
			if (n_env > 0) {
				outf << " " << vp.eta(iiValid) << " " << vp.yx(iiValid);
			}
			if (n_chrs > 0) {
				for (int cc = 0; cc < n_chrs; cc++) {
					outf << " " << resid_loco[cc](iiValid);
				}
			}
			outf << std::endl;
			iiValid++;
		}
	}
	assert(iiValid == n_samples);
	boost_io::close(outf);
}
