// Class to perform snp wise linear tests

#include "data.hpp"

class SNPwiseLM{
public:
	parameters params;

	std::vector< std::string > chromosome, rsid, SNPID;
	std::vector< uint32_t > position;
	std::vector< std::vector< std::string > > alleles;
	
	int n_pheno; // number of phenotypes
	int n_covar; // number of covariates
	int n_samples; // number of samples
	long int n_snps; // number of snps
	bool bgen_pass;
	int n_var;
	std::size_t n_var_parsed; // Track progress through IndexQuery

	bool Y_reduced;   // Variables to track whether we have already
	bool W_reduced;   // reduced to complete cases or not.

	std::vector< double > info;
	std::vector< double > maf;
	std::vector< std::string > rsid_list;

	std::map<int, bool> missing_covars; // set of subjects missing >= 1 covariate
	std::map<int, bool> missing_phenos; // set of subjects missing >= phenotype
	std::map< int, bool > incomplete_cases; // union of samples missing data

	std::vector< std::string > pheno_names;
	std::vector< std::string > covar_names;

	Eigen::MatrixXd G; // probabilistic genotype matrix
	Eigen::MatrixXi GG; // rounded genotype matrix
	Eigen::MatrixXd Y; // phenotype matrix
	Eigen::MatrixXd W; // covariate matrix
	Eigen::VectorXd Z; // interaction vector
	genfile::bgen::View::UniquePtr bgenView;
	std::vector< double > beta, tau, neglogP, neglogP_2dof;
	std::vector< std::vector< double > > gamma;
	std::vector<double> variant_sd;

	boost_io::filtering_ostream outf;

	// Constructors
	SNPwiseLM(Data& data){
		assert(data.GM.low_mem == false);
		G = data.G;


		variant_sd = data.variant_sd;
	};
	~SNPwiseLM(){};

	void run() {
		int ch = 0;

		// Step 1; Read in raw covariates and phenotypes
		// - also makes a record of missing values
		read_covar();
		read_pheno();

		// Step 2; Reduce raw covariates and phenotypes to complete cases
		// - may change value of n_samples
		// - will also skip these cases when reading bgen later
		reduce_to_complete_cases();

		// Step 3; Center phenos, genotypes, normalise covars
		center_matrix( Y, n_pheno );
		center_matrix( W, n_covar );
		scale_matrix( W, n_covar, covar_names );

		// Step 4; Regress covars out of phenos
		if(!params.mode_lm2){
			std::cout << " Regressing out covariates" << std::endl;
			regress_covars();
		}

		// TODO: Move UI messages to coherent place?
		if( params.x_param_name != "NULL"){
			std::cout << "Searching for --interaction param ";
			std::cout << params.x_param_name << std::endl;
		} else {
			std::cout << "Choosing first covar to use as interaction term (default)" << std::endl;
		}

		// Write headers to output file
		output_init();

		while (read_bgen_chunk()) {
			// Raw dosage read in to G
			std::cout << "Chunk " << ch+1 << " read (size " << n_var;
			std::cout << ", " << n_var_parsed-1 << "/" << bgenView->number_of_variants();
			std::cout << " variants parsed)" << std::endl;

			// Normalise genotypes
			// center_matrix( G, n_var );
			// scale_matrix( G, n_var );
			// TODO exclude cols wth var 0
			std::vector<size_t> keep;
			for(std::size_t jj = 0; jj < n_var; jj++){
				if (variant_sd[jj] > 1e-12) {
					keep.push_back(jj);	
				}
			}
			if (keep.size() != n_var) {
				std::cout << " Removing " << (n_var - keep.size())  << " columns with zero variance." << std::endl;
				G = getCols(G, keep);
				
				n_var = keep.size();
			}

			// Actually compute models
			if(params.mode_lm){
				calc_lrts();
				output_results();
			} else if(params.mode_lm2){
				calc_lrts2();
			} else if(params.mode_joint_model){
				calc_joint_model();
				output_results();
			}
			ch++;
		}
	}


	std::size_t find_covar_index( std::string colname ){
		std::size_t x_col;
		std::vector<std::string>::iterator it;
		it = std::find(covar_names.begin(), covar_names.end(), colname);
		if (it == covar_names.end()){
			throw std::invalid_argument("Can't locate parameter " + colname);
		}
		x_col = it - covar_names.begin();
		return x_col;
	}

	void calc_lrts() {
		// For-loop through variants and compute interaction models.
		// Save to
		// data.tau, data.beta
		// Y is a matrix of dimension n_samples x 1
		Eigen::VectorXd e_j, f_j, g_j, gamma_j;
		double beta_j, tau_j, xtx_inv, loglik_null, loglik_alt, chi_stat;
		long double pval;

		// Determine which covar to use in interaction
		std::ptrdiff_t x_col;
		if( params.x_param_name != "NULL"){
			x_col = find_covar_index(params.x_param_name);
		} else {
			x_col = 0;
		}

		Eigen::VectorXd vv(Eigen::Map<Eigen::VectorXd>(W.col(x_col).data(), n_samples));
		Eigen::MatrixXd Z = G.array().colwise() * vv.array();

		beta.clear();
		tau.clear();
		gamma.clear();
		neglogP.clear();
		neglogP_2dof.clear();
		xtx_inv = 1.0 / (n_samples - 1.0);

		for (int jj = 0; jj < n_var; jj++){
			Eigen::Map<Eigen::VectorXd> G_j(G.col(jj).data(), n_samples);
			Eigen::Map<Eigen::VectorXd> Z_j(Z.col(jj).data(), n_samples);

			// null
			beta_j = xtx_inv * (G_j.transpose() * Y)(0,0);
			e_j = Y - G_j * beta_j;

			// alt - 1dof
			tau_j = (Z_j.transpose() * e_j)(0,0) / (Z_j.transpose() * Z_j)(0,0);
			f_j = e_j - Z_j * tau_j;

			// Saving variables
			beta.push_back(beta_j);
			tau.push_back(tau_j);
			neglogP.push_back(lrt(e_j, f_j, 1));

			// 2 dof stuff
			Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_samples, 3);
			Eigen::MatrixXd AA = Eigen::MatrixXd::Zero(n_samples, 2);
			Eigen::MatrixXd D;
			std::vector<double> nn(3, 0);
			std::vector< double > gamma_vec(3, std::nan(""));
			int kk;
			for (int ii = 0; ii < n_samples; ii++){
				kk = GG(ii,jj);
				nn[kk] += 1.0;
			}

			if(std::all_of(nn.begin(), nn.end(), [](int i){return i>0.0;})){
				for (int ii = 0; ii < n_samples; ii++){
					kk = GG(ii,jj);
					// A(ii, kk) = vv(ii);
					if(kk == 0){
						AA(ii, 0) -= nn[1] * vv(ii) / nn[0];
						AA(ii, 1) -= nn[2] * vv(ii) / nn[0];
					} else {
						AA(ii, kk-1) = vv(ii);
					}
				}

				D = (AA.transpose() * AA);
				gamma_j = D.ldlt().solve(AA.transpose() * e_j);
				g_j = e_j - AA * gamma_j;

				gamma_vec[1] = gamma_j(0, 0);
				gamma_vec[2] = gamma_j(1, 0);
				gamma_vec[0] = -(nn[1]*gamma_vec[1] + nn[2]*gamma_vec[2]) / nn[0];
				gamma.push_back(gamma_vec);
				neglogP_2dof.push_back(lrt(e_j, g_j, 2));
			} else {
				gamma.push_back(gamma_vec);
				neglogP_2dof.push_back(std::nan(""));
			}
		}
	}

	void calc_lrts2() {
		// Determine column indexs to include in X_g
		std::vector< std::size_t > col_indexes;
		if( params.x_param_name != "NULL"){
			col_indexes.push_back(find_covar_index(params.x_param_name));
		} else {
			col_indexes.push_back(0);
		}
		for (int ii = 0; ii < params.n_gconf; ii++){
			col_indexes.push_back(find_covar_index(params.gconf[ii]));
		}

		for (int jj = 0; jj < n_var; jj++){
			// Build matrix X_g
			Eigen::MatrixXd tmp;
			tmp = (getCols(W, col_indexes)).array().colwise() * G.col(jj).array();
			int tmp_ncol = params.n_gconf + 1;
			center_matrix( tmp, tmp_ncol );
			scale_matrix( tmp, tmp_ncol );

			// Combine to matrix X_comb = (X_c, X_g)
			int comb_col = n_covar + 2 + params.n_gconf;
			Eigen::MatrixXd X_comb(n_samples, comb_col);
			X_comb << W, G.col(jj), tmp;

			// Fit joint regression Y = X_comb beta_comb + epsilon
			Eigen::MatrixXd XtX_comb = X_comb.transpose() * X_comb;
			Eigen::MatrixXd XtX_comb_inv = XtX_comb.inverse();
			Eigen::MatrixXd beta_comb_j = XtX_comb_inv * X_comb.transpose() * Y;

			// Create vector of standard errors
			double s_var = (Y - X_comb * beta_comb_j).norm() / std::sqrt(n_samples - comb_col);
			std::vector< double > se;
			for (int kk = 0; kk < comb_col; kk++){
				se.push_back(s_var * std::sqrt(XtX_comb_inv(kk, kk)));
			}

			// Compute t-test on GxE param
			double n_dof;
			n_dof = (double) (n_samples - (n_covar + 2 + params.n_gconf));
			boost::math::students_t t_dist(n_dof);
			double t_stat = beta_comb_j(n_covar + 1, 0) / se[n_covar + 1];

			// p-value from 2 tailed t-test
			double q = 2 * boost::math::cdf(boost::math::complement(t_dist, fabs(t_stat)));

 			// Output stats
			outf << chromosome[jj] << "\t" << rsid[jj] << "\t" << position[jj];
			outf << "\t" << alleles[jj][0] << "\t" << alleles[jj][1] << "\t";
			outf << maf[jj] << "\t" << info[jj];

			for (int kk = 0; kk < n_covar + 2 + params.n_gconf; kk++){
				outf << "\t" << beta_comb_j(kk, 0);
			}

			outf << "\t" << -1 * std::log10(q) << std::endl;
		}
	}

	void calc_joint_model() {
		// Y = G beta
		// vs
		// Y = G beta + Z tau
		// Want:
		// - coefficients
		// - variance explained
		// - F test comparing model fit (just need residuals)
		// - AIC?
		// Common sense checks; n < p
		if (n_var > n_samples){
			throw std::logic_error("n_samples must be less than n_var for the joint model");
		}

		// Determine which covar to use in interaction
		std::ptrdiff_t x_col;
		if( params.x_param_name != "NULL"){
			std::vector<std::string>::iterator it;
			it = std::find(covar_names.begin(), covar_names.end(), params.x_param_name);
			if (it == covar_names.end()){
				throw std::invalid_argument("Can't locate --interaction parameter");
			}
			x_col = it - covar_names.begin();
		} else {
			x_col = 0;
		}

		Eigen::VectorXd vv(Eigen::Map<Eigen::VectorXd>(W.col(x_col).data(), n_samples));
		std::cout << "Initialising matrix W (" << G.rows() << "," << 2*G.cols() << ")" << std::endl;
		Eigen::MatrixXd W(G.rows(), G.cols() + G.cols());
		W << G, (G.array().colwise() * vv.array()).matrix();
		std::cout << W << std::endl;
		// G should be centered and scaled by now
		std::cout << "Fitting polygenic model" << std::endl;
		Eigen::MatrixXd beta = solve(G.transpose() * G, G.transpose() * Y);
		Eigen::VectorXd e_null = Y - G * beta;

		// eta; vector of coefficients c(beta, tau)
		std::cout << "Fitting joint interaction model" << std::endl;
		Eigen::MatrixXd eta = solve(W.transpose() * W, W.transpose() * Y);
		Eigen::VectorXd e_alt = Y - W * eta;
		std::cout << eta << std::endl;
		std::cout << e_alt << std::endl;

		boost::math::students_t t_dist(n_samples - n_var - 1);
		double pval_j, eta_j;
		for(int jj = 0; jj < n_var; jj++){
			eta_j = eta(n_var + jj, 0);
			tau.push_back(eta_j);
			pval_j = 1.0 - boost::math::cdf(t_dist, eta_j);
			neglogP.push_back(-1*std::log10(pval_j));
		}

		// F test
		double f_stat, rss_null, rss_alt, pval, neglogp_joint;
		rss_null = e_null.dot(e_null);
		rss_alt = e_alt.dot(e_alt);
		f_stat = (rss_null - rss_alt) / (double) n_var;
		f_stat /= rss_alt / (double) (n_samples - n_var - 1);
		boost::math::fisher_f f_dist(n_var, n_samples - n_var);
		pval = 1.0 - boost::math::cdf(f_dist, f_stat);
		neglogp_joint = -1 * std::log10(pval);

		// Output
		std::cout << "F-test comparing models " << std::endl;
		std::cout << "H0: Y = G x beta" << std::endl;
		std::cout << "vs" << std::endl;
		std::cout << "H0: Y = G x beta + Z x tau" << std::endl;
		std::cout << "F-stat = " << f_stat << ", neglogP = " << neglogp_joint << std::endl;

		std::cout << "Variance explained by null: ";
		std::cout << 100 * (1.0 - rss_null / (Y.transpose() * Y)(0,0)) << std::endl;
		std::cout << "Variance explained by alt: ";
		std::cout << 100 * (1.0 - rss_alt / (Y.transpose() * Y)(0,0)) << std::endl;
	}

	double lrt(Eigen::VectorXd null, Eigen::VectorXd alt, int df){
		// Logliks correct up to ignoreable constant
		boost::math::chi_squared chi_dist_1(1), chi_dist_2(2);
		double loglik_null, loglik_alt, chi_stat, neglogp;

		loglik_null = std::log(n_samples) - std::log(null.dot(null));
		loglik_null *= n_samples/2.0;
		loglik_alt = std::log(n_samples) - std::log(alt.dot(alt));
		loglik_alt *= n_samples/2.0;

		chi_stat = 2*(loglik_alt - loglik_null);
		if(chi_stat < 0 && chi_stat > -0.0000001){
			chi_stat = 0.0; // Sometimes we get -1e-09
		}
		if (df == 1){
			neglogp = std::log10(boost::math::cdf(boost::math::complement(chi_dist_1, chi_stat)));
		} else {
			neglogp = std::log10(boost::math::cdf(boost::math::complement(chi_dist_2, chi_stat)));
		}
		neglogp *= -1.0;
		return neglogp;
	}


	void regress_covars() {
		std::cout << "Regressing out covars:" << std::endl;
		for(int cc = 0; cc < n_covar; cc++){
			std::cout << ( cc > 0 ? ", " : "" ) << covar_names[cc]; 
		}
		std::cout << std::endl;

		Eigen::MatrixXd ww = W.rowwise() - W.colwise().mean(); //not needed probably
		Eigen::MatrixXd bb = solve(ww.transpose() * ww, ww.transpose() * Y);
		Y = Y - ww * bb;
	}

	void output_init() {
		// open output file
		std::string ofile, gz_str = ".gz";

		ofile = params.out_file;
		if (params.out_file.find(gz_str) != std::string::npos) {
			outf.push(boost_io::gzip_compressor());
		}
		outf.push(boost_io::file_sink(ofile.c_str()));

		if(params.mode_vcf){
			// Output header for vcf file
			outf << "##fileformat=VCFv4.2\n"
				<< "FORMAT=<ID=GP,Type=Float,Number=G,Description=\"Genotype call probabilities\">\n"
				<< "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT" ;
			bgenView->get_sample_ids(
				[&]( std::string const& id ) { outf << "\t" << id ; }
			) ;
			outf << "\n" ;
		}

		if(params.mode_lm){
			// Output header for vcf file
			outf << "chr\trsid\tpos\ta_0\ta_1\taf\tinfo\tbeta\ttau";
			outf << "\tneglogP_1dof\tgamma1\tgamma2\tgamma3\tneglogP_2dof" << std::endl;
		}

		if(params.mode_lm2){
			outf << "chr\trsid\tpos\ta_0\ta_1\taf\tinfo";
			for (int kk = 0; kk < n_covar; kk++){
				outf << "\tbeta_" << covar_names[kk];
			}
			outf << "\tbeta_var_j\tbeta_gx" << params.x_param_name;
			for (int kk = 0; kk < params.n_gconf; kk++){
				outf << "\tbeta_gx" << params.gconf[kk];
			}
			outf << "\tneglogP_gx" << params.x_param_name << std::endl;
		}

		if(params.mode_joint_model){
			outf << "chr\trsid\tpos\ta_0\ta_1\taf\tinfo\ttau\t1dof_neglogP";
 			outf << std::endl;
		}
	}

	void output_results() {
		if(params.mode_lm){
			for (int s = 0; s < n_var; s++){
				outf << chromosome[s] << "\t" << rsid[s] << "\t" << position[s] << "\t";
				outf << alleles[s][0] << "\t" << alleles[s][1] << "\t" << maf[s] << "\t";
				outf << info[s] << "\t" << beta[s] << "\t" << tau[s] << "\t";
	 			outf << neglogP[s] << "\t" << gamma[s][0] << "\t" << gamma[s][1];
	 			outf << "\t" << gamma[s][2] << "\t" << neglogP_2dof[s] << std::endl;
			}
		}

		if(params.mode_joint_model){
			for (int s = 0; s < n_var; s++){
				outf << chromosome[s] << "\t" << rsid[s] << "\t" << position[s] << "\t";
				outf << alleles[s][0] << "\t" << alleles[s][1] << "\t" << maf[s] << "\t";
				outf << info[s] << "\t" << tau[s] << "\t" << neglogP[s] << std::endl;
			}
		}
	}

};
