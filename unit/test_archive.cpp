//
// Created by kerin on 2019-03-22.
//


TEST_CASE( "Example 1: single-env" ){
parameters p;

SECTION("Ex1. No filters applied, low mem mode"){
char* argv[] = { (char*) "bin/bgen_prog", (char*) "--VB", (char*) "--low_mem",
				 (char*) "--mode_spike_slab", (char*) "--mode_regress_out_covars",
				 (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
				 (char*) "--out", (char*) "data/io_test/fake_age.out",
				 (char*) "--pheno", (char*) "data/io_test/pheno.txt",
				 (char*) "--hyps_grid", (char*) "data/io_test/hyperpriors_gxage.txt",
				 (char*) "--hyps_probs", (char*) "data/io_test/hyperpriors_gxage_probs.txt",
				 (char*) "--vb_init", (char*) "data/io_test/answer_init.txt",
				 (char*) "--environment", (char*) "data/io_test/age.txt"};
int argc = sizeof(argv)/sizeof(argv[0]);
parse_arguments(p, argc, argv);
Data data( p );

std::cout << "Data initialised" << std::endl;
data.read_non_genetic_data();
SECTION( "Ex1. Raw non genetic data read in accurately"){
//            CHECK(data.n_covar == 1);
CHECK(data.n_env == 1);
CHECK(data.n_pheno == 1);
CHECK(data.n_samples == 50);
CHECK(data.Y(0,0) == Approx(-1.18865038973338));
//CHECK(data.W(0,0) == Approx(-0.33472645347487201));
CHECK(data.E(0, 0) == Approx(-0.33472645347487201));
CHECK(data.hyps_grid(0,1) == Approx(0.317067781333932));
}

data.standardise_non_genetic_data();
SECTION( "Ex1. Non genetic data standardised + covars regressed"){
CHECK(data.params.scale_pheno == true);
CHECK(data.params.use_vb_on_covars == false);
CHECK(data.params.covar_file == "NULL");
//			CHECK(data.Y(0,0) == Approx(-3.6676363273605137)); Centered
//			CHECK(data.Y(0,0) == Approx(-1.5800573524786081)); Scaled
CHECK(data.Y(0,0) == Approx(-1.262491384814441));
CHECK(data.Y2(0,0) == Approx(-1.262491384814441));
//			CHECK(data.W(0,0) == Approx(-0.58947939694779772));
CHECK(data.E(0,0) == Approx(-0.58947939694779772));
}

data.read_full_bgen();
SECTION( "Ex1. bgen read in & standardised correctly"){
CHECK(data.G.low_mem);
CHECK(data.params.low_mem);
CHECK(data.params.flip_high_maf_variants);
CHECK(data.G(0, 0) == Approx(1.8570984229));
}

SECTION( "Ex1. Confirm calc_dxteex() reorders properly"){
data.params.dxteex_file = "data/io_test/inputs/dxteex_mixed.txt";
data.read_external_dxteex();
data.calc_dxteex();
CHECK(data.dXtEEX(0, 0) == Approx(87.204591182113916));
CHECK(data.n_dxteex_computed == 1);
}

data.calc_dxteex();
if(p.vb_init_file != "NULL"){
data.read_alpha_mu();
}
VBayesX2 VB(data);
VB.check_inputs();
SECTION("Ex1. Vbayes_X2 initialised correctly"){
CHECK(VB.n_samples == 50);
CHECK(VB.N == 50.0);
CHECK(VB.n_env == 1);
//			CHECK(VB.n_covar == 1);
CHECK(VB.n_effects == 2);
CHECK(VB.vp_init.muw(0) == 1.0);
CHECK(!VB.p.init_weights_with_snpwise_scan);
CHECK(VB.dXtEEX(0, 0) == Approx(87.1907593967));
}

std::vector< VbTracker > trackers(VB.hyps_grid.rows(), p);
SECTION("Ex1. Explicitly checking updates"){
// Initialisation
#ifdef DATA_AS_FLOAT
CHECK( (double)  VB.vp_init.ym(0) == Approx(0.0003200434));
#else
CHECK(VB.vp_init.ym(0) == Approx(0.0003200476));
#endif
CHECK(VB.vp_init.yx(0) == Approx(0.0081544079));
CHECK(VB.vp_init.eta(0) == Approx(-0.5894793969));

// Set up for RunInnerLoop
long n_grid = VB.hyps_grid.rows();
long n_samples = VB.n_samples;
std::vector<Hyps> all_hyps;
VB.unpack_hyps(VB.hyps_grid, all_hyps);

// Set up for updateAllParams
std::vector<VariationalParameters> all_vp;
VB.setup_variational_params(all_hyps, all_vp);
VariationalParameters& vp = all_vp[0];
Hyps& hyps = all_hyps[0];

int round_index = 2;
std::vector<double> logw_prev(n_grid, -std::numeric_limits<double>::max());
std::vector<std::vector< double >> logw_updates(n_grid);

// Ground zero as expected
CHECK(vp.alpha_beta(0) * vp.mu1_beta(0) == Approx(-0.00015854116408000002));
CHECK(data.Y(0,0) == Approx(-1.262491384814441));
#ifdef DATA_AS_FLOAT
CHECK( (double) vp.ym(0) == Approx( 0.0003200434));
#else
CHECK(vp.ym(0) == Approx(0.0003200476));
#endif
CHECK(vp.yx(0) == Approx(0.0081544079));
CHECK(vp.eta(0) == Approx(-0.5894793969));

VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev);

CHECK(VB.X.col(0)(0) == Approx(1.8570984229));
CHECK(vp.s1_beta_sq(0) == Approx(0.0031087381));
CHECK(vp.mu1_beta(0) == Approx(-0.0303900712));
CHECK(vp.alpha_beta(0) == Approx(0.1447783263));
CHECK(vp.alpha_beta(1) == Approx(0.1517251004));
CHECK(vp.mu1_beta(1) == Approx(-0.0355760798));
CHECK(vp.alpha_beta(63) == Approx(0.1784518373));
CHECK(VB.calc_logw(hyps, vp) == Approx(-60.983398393));

VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev);

CHECK(vp.alpha_beta(0) == Approx(0.1350711123));
CHECK(vp.mu1_beta(0) == Approx(-0.0205395866));
CHECK(vp.alpha_beta(1) == Approx(0.1400764528));
CHECK(vp.alpha_beta(63) == Approx(0.1769882239));
CHECK(VB.calc_logw(hyps, vp) == Approx(-60.606081598));
}

VB.run_inference(VB.hyps_grid, false, 2, trackers);
SECTION("Ex1. Vbayes_X2 inference correct") {
CHECK(trackers[0].count == 33);
CHECK(trackers[3].count == 33);
CHECK(trackers[0].logw == Approx(-60.522210486));
CHECK(trackers[1].logw == Approx(-59.9696083263));
CHECK(trackers[2].logw == Approx(-60.30658117));
CHECK(trackers[3].logw == Approx(-61.0687573393));
}
}
}

TEST_CASE( "Example 2a: multi-env + bgen over 2chr" ){
parameters p;

SECTION("Ex2. No filters applied, high mem mode"){
char* argv[] = { (char*) "bin/bgen_prog", (char*) "--VB", (char*) "--high_mem",
				 (char*) "--mode_spike_slab", (char*) "--mode_regress_out_covars",
				 (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
				 (char*) "--bgen", (char*) "data/io_test/n50_p100_chr2.bgen",
				 (char*) "--out", (char*) "data/io_test/fake_env.out",
				 (char*) "--pheno", (char*) "data/io_test/pheno.txt",
				 (char*) "--hyps_grid", (char*) "data/io_test/hyperpriors_gxage.txt",
				 (char*) "--vb_init", (char*) "data/io_test/answer_init.txt"};
int argc = sizeof(argv)/sizeof(argv[0]);
parse_arguments(p, argc, argv);
Data data( p );

data.read_non_genetic_data();
data.standardise_non_genetic_data();
data.read_full_bgen();

data.calc_dxteex();
data.calc_snpstats();
if(p.vb_init_file != "NULL"){
data.read_alpha_mu();
}
VBayesX2 VB(data);
VB.check_inputs();
SECTION("Ex2. Vbayes_X2 initialised correctly"){
CHECK(VB.n_samples == 50);
CHECK(VB.N == 50.0);
CHECK(VB.n_var == 73);
CHECK(VB.n_env == 4);
// CHECK(VB.n_covar == 4);
//CHECK(VB.n_effects == 2);
CHECK(VB.vp_init.muw(0) == 0.25);
CHECK(VB.p.init_weights_with_snpwise_scan == false);
CHECK(VB.dXtEEX(0, 0) == Approx(44.6629676819));
}

std::vector< VbTracker > trackers(VB.hyps_grid.rows(), p);
SECTION("Ex2. Explicitly checking updates"){
// Set up for RunInnerLoop
long n_grid = VB.hyps_grid.rows();
long n_samples = VB.n_samples;
std::vector<Hyps> all_hyps;
VB.unpack_hyps(VB.hyps_grid, all_hyps);

// Set up for updateAllParams
std::vector<VariationalParameters> all_vp;
VB.setup_variational_params(all_hyps, all_vp);
VariationalParameters& vp = all_vp[0];
Hyps& hyps = all_hyps[0];

int round_index = 2;
std::vector<double> logw_prev(n_grid, -std::numeric_limits<double>::max());
std::vector<std::vector< double >> logw_updates(n_grid);

VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev);

CHECK(vp.alpha_beta(0) == Approx(0.0103168718));
CHECK(vp.alpha_beta(1) == Approx(0.0101560491));
CHECK(vp.alpha_beta(63) == Approx(0.0098492375));
CHECK(vp.alpha_gam(0) == Approx(0.013394603));
CHECK(vp.muw(0) == Approx(0.1593944543));
CHECK(VB.calc_logw(hyps, vp) == Approx(-71.1292851018));

VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev);

CHECK(vp.alpha_beta(0) == Approx(0.0101823562));
CHECK(vp.alpha_beta(1) == Approx(0.0100615294));
CHECK(vp.alpha_beta(63) == Approx(0.0098486026));
CHECK(vp.muw(0) == Approx(0.031997336));
CHECK(VB.calc_logw(hyps, vp) == Approx(-69.8529334166));
}

VB.run_inference(VB.hyps_grid, false, 2, trackers);
SECTION("Ex2. Vbayes_X2 inference correct"){
CHECK(trackers[0].count == 10);
CHECK(trackers[3].count == 10);
CHECK(trackers[0].logw == Approx(-69.7419880272));
CHECK(trackers[1].logw == Approx(-69.9470990972));
CHECK(trackers[2].logw == Approx(-70.1298787803));
CHECK(trackers[3].logw == Approx(-70.2928879787));
}

SECTION("Partition residuals amongst chromosomes"){
int n_chrs = VB.n_chrs;
long n_samples = VB.n_samples;
std::vector<Eigen::VectorXd> map_residuals_by_chr(n_chrs), pred_main(n_chrs), pred_int(n_chrs);
long ii_map = 0;
VariationalParametersLite vp_map = trackers[ii_map].vp;

// Predicted effects to file
VB.calcPredEffects(vp_map);
VB.compute_residuals_per_chr(vp_map, pred_main, pred_int, map_residuals_by_chr);

Eigen::VectorXd check_Xgam = Eigen::VectorXd::Zero(n_samples);
Eigen::VectorXd check_Xbeta = Eigen::VectorXd::Zero(n_samples);
Eigen::VectorXd check_resid = Eigen::VectorXd::Zero(n_samples);
for (int cc = 0; cc < n_chrs; cc++){
check_Xbeta += pred_main[cc];
check_Xgam += pred_int[cc];
check_resid += map_residuals_by_chr[cc];
}
if(VB.p.use_vb_on_covars) {
check_Xbeta += (VB.C * vp_map.muc.matrix().cast<scalarData>()).cast<double>();
check_resid += (VB.C * vp_map.muc.matrix().cast<scalarData>()).cast<double>();
}
check_resid -= VB.Y.cast<double>();
check_resid /= (double) (n_chrs-1);

Eigen::VectorXd resid = (VB.Y - vp_map.ym - vp_map.yx.cwiseProduct(vp_map.eta)).cast<double>();

CHECK(n_chrs == 2);
CHECK(pred_main[0](0) == Approx(0.0275588533));
CHECK(pred_main[1](0) == Approx(-0.0404733278));
CHECK(check_Xbeta(0)  == Approx(vp_map.ym(0)));
CHECK(check_Xgam(0)   == Approx(vp_map.yx(0)));
CHECK(check_resid(0)  == Approx(resid(0)));
}
}
}

TEST_CASE( "Example 3: multi-env w/ covars" ){
parameters p;

SECTION("Ex3. No filters applied, high mem mode"){
char* argv[] = { (char*) "bin/bgen_prog", (char*) "--VB", (char*) "--high_mem",
				 (char*) "--use_vb_on_covars", (char*) "--mode_spike_slab",
				 (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt.gz",
				 (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
				 (char*) "--out", (char*) "data/io_test/fake_env.out",
				 (char*) "--pheno", (char*) "data/io_test/pheno.txt",
				 (char*) "--hyps_grid", (char*) "data/io_test/hyperpriors_gxage.txt",
				 (char*) "--vb_init", (char*) "data/io_test/answer_init.txt"};
int argc = sizeof(argv)/sizeof(argv[0]);
parse_arguments(p, argc, argv);
Data data( p );

data.read_non_genetic_data();
data.standardise_non_genetic_data();
SECTION( "Ex3. Non genetic data standardised + covars regressed"){
CHECK(data.params.scale_pheno == true);
CHECK(data.params.use_vb_on_covars == true);
CHECK(data.params.covar_file == "NULL");
//			CHECK(data.Y(0,0) == Approx(-3.6676363273605137)); Centered
CHECK(data.Y(0,0) == Approx(-1.5800573524786081)); // Scaled
CHECK(data.Y2(0,0) == Approx(-1.5567970303));
//CHECK(data.W(0,0) == Approx(0.8957059881));
CHECK(data.E(0,0) == Approx(0.8957059881));
}
data.read_full_bgen();

data.calc_dxteex();
data.calc_snpstats();
if(p.vb_init_file != "NULL"){
data.read_alpha_mu();
}
VBayesX2 VB(data);
VB.check_inputs();
SECTION("Ex3. Vbayes_X2 initialised correctly"){
CHECK(VB.n_samples == 50);
CHECK(VB.N == 50.0);
CHECK(VB.n_env == 4);
//CHECK(VB.n_covar == 4);
CHECK(VB.n_effects == 2);
CHECK(VB.vp_init.muw(0) == 0.25);
CHECK(VB.p.init_weights_with_snpwise_scan == false);
CHECK(VB.dXtEEX(0, 0) == Approx(38.9390135703));
CHECK(VB.dXtEEX(1, 0) == Approx(38.34695));
CHECK(VB.dXtEEX(2, 0) == Approx(33.7626));
CHECK(VB.dXtEEX(3, 0) == Approx(35.71962));

CHECK(VB.dXtEEX(0, 4) == Approx(-2.58481));
CHECK(VB.dXtEEX(1, 4) == Approx(-13.04073));
CHECK(VB.dXtEEX(2, 4) == Approx(-11.69077));
CHECK(VB.dXtEEX(3, 4) == Approx(-7.17068));
}

std::vector< VbTracker > trackers(VB.hyps_grid.rows(), p);
SECTION("Ex3. Explicitly checking updates"){
// Set up for RunInnerLoop
long n_grid = VB.hyps_grid.rows();
std::vector<Hyps> all_hyps;
VB.unpack_hyps(VB.hyps_grid, all_hyps);

// Set up for updateAllParams
std::vector<VariationalParameters> all_vp;
VB.setup_variational_params(all_hyps, all_vp);

int round_index = 2;
std::vector<double> logw_prev(n_grid, -std::numeric_limits<double>::max());
std::vector<std::vector< double >> logw_updates(n_grid);

VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev);
VariationalParameters& vp = all_vp[0];
Hyps& hyps = all_hyps[0];

CHECK(vp.muc(0) == Approx(0.1221946024));
CHECK(vp.muc(3) == Approx(-0.1595909887));
CHECK(vp.alpha_beta(0) == Approx(0.1339235799));
CHECK(vp.alpha_beta(1) == Approx(0.1415361555));
CHECK(vp.alpha_beta(63) == Approx(0.1724736345));
CHECK(vp.muw(0, 0) == Approx(0.1127445891));
CHECK(VB.calc_logw(all_hyps[0], vp) == Approx(-94.4656200443));

CHECK(vp.alpha_gam(0) == Approx(0.1348765515));
CHECK(vp.alpha_gam(1) == Approx(0.1348843768));
CHECK(vp.alpha_gam(63) == Approx(0.1351395247));
CHECK(vp.mu1_beta(0) == Approx(-0.0189890299));
CHECK(vp.mu1_beta(1) == Approx(-0.0275538256));
CHECK(vp.mu1_beta(63) == Approx(-0.0470801956));
CHECK(vp.mu1_gam(0) == Approx(0.0048445126));
CHECK(vp.mu1_gam(1) == Approx(0.0005509309));
CHECK(vp.mu1_gam(63) == Approx(-0.0040966814));
CHECK(vp.s1_gam_sq(0) == Approx(0.0035251837));
CHECK(vp.s1_gam_sq(1) == Approx(0.0035489038));
CHECK(vp.s1_gam_sq(63) == Approx(0.0035479273));

VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev);

CHECK(vp.muc(0) == Approx(0.1463805515));
CHECK(vp.muc(3) == Approx(-0.1128544804));
CHECK(vp.alpha_beta(0) == Approx(0.1292056073));
CHECK(vp.alpha_beta(1) == Approx(0.1338797264));
CHECK(vp.alpha_beta(63) == Approx(0.1730150924));
CHECK(vp.muw(0, 0) == Approx(0.0460748751));
CHECK(VB.calc_logw(all_hyps[0], vp) == Approx(-93.7888239338));


CHECK(vp.alpha_gam(0) == Approx(0.1228414938));
CHECK(vp.alpha_gam(1) == Approx(0.1244760462));
CHECK(vp.alpha_gam(63) == Approx(0.1240336666));
CHECK(vp.mu1_gam(0) == Approx(-0.0013406961));
CHECK(vp.mu1_gam(1) == Approx(-0.0021107307));
CHECK(vp.mu1_gam(63) == Approx(0.0010160659));
CHECK(vp.s1_gam_sq(0) == Approx(0.0028616572));
CHECK(vp.s1_gam_sq(1) == Approx(0.0029466955));
CHECK(vp.s1_gam_sq(63) == Approx(0.0029262235));

VB.updateAllParams(2, round_index, all_vp, all_hyps, logw_prev);

CHECK(vp.alpha_beta(0) == Approx(0.1291159583));
CHECK(vp.alpha_beta(1) == Approx(0.1337078986));
CHECK(vp.alpha_beta(63) == Approx(0.1846784602));
CHECK(vp.alpha_gam(0) == Approx(0.1205867018));
CHECK(vp.alpha_gam(1) == Approx(0.1223799879));
CHECK(vp.alpha_gam(63) == Approx(0.1219421923));
CHECK(vp.mu1_beta(0) == Approx(-0.0099430405));
CHECK(vp.mu1_beta(1) == Approx(-0.0186819136));
CHECK(vp.mu1_beta(63) == Approx(-0.0522879252));
CHECK(vp.mu1_gam(0) == Approx(-0.0010801898));
CHECK(vp.mu1_gam(1) == Approx(-0.0010635764));
CHECK(vp.mu1_gam(63) == Approx(-0.0006202975));
CHECK(vp.muw(0, 0) == Approx(0.0285866235));
}

VB.run_inference(VB.hyps_grid, false, 2, trackers);
SECTION("Ex3. Vbayes_X2 inference correct"){
CHECK(trackers[0].count == 33);
CHECK(trackers[3].count == 33);
CHECK(trackers[0].logw == Approx(-93.7003814019));
CHECK(trackers[1].logw == Approx(-93.3247434264));
CHECK(trackers[2].logw == Approx(-93.6548417528));
CHECK(trackers[3].logw == Approx(-94.3511347264));
}
}
}


TEST_CASE( "Example 6: single-env w MoG + hyps max" ){
parameters p;

SECTION("Ex6. No filters applied, high mem mode"){
char* argv[] = { (char*) "bin/bgen_prog", (char*) "--VB", (char*) "--effects_prior_mog",
				 (char*) "--VB-iter-max", (char*) "20",  (char*) "--mode_regress_out_covars",
				 (char*) "--VB-varEM", (char*) "--high_mem",
				 (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
				 (char*) "--out", (char*) "data/io_test/fake_age.out",
				 (char*) "--pheno", (char*) "data/io_test/pheno.txt",
				 (char*) "--hyps_grid", (char*) "data/io_test/hyperpriors_gxage.txt",
				 (char*) "--hyps_probs", (char*) "data/io_test/hyperpriors_gxage_probs.txt",
				 (char*) "--vb_init", (char*) "data/io_test/answer_init.txt",
				 (char*) "--environment", (char*) "data/io_test/age.txt",
				 (char*) "--spike_diff_factor", (char*) "100"};
int argc = sizeof(argv)/sizeof(argv[0]);
parse_arguments(p, argc, argv);
Data data( p );

data.read_non_genetic_data();
data.standardise_non_genetic_data();
SECTION( "Ex6. Non genetic data standardised + covars regressed"){
CHECK(data.params.scale_pheno == true);
CHECK(data.params.use_vb_on_covars == false);
CHECK(data.params.covar_file == "NULL");
//			CHECK(data.Y(0,0) == Approx(-3.6676363273605137)); Centered
//			CHECK(data.Y(0,0) == Approx(-1.5800573524786081)); Scaled
CHECK(data.Y(0,0) == Approx(-1.262491384814441));
CHECK(data.Y2(0,0) == Approx(-1.262491384814441));
//CHECK(data.W(0,0) == Approx(-0.58947939694779772));
CHECK(data.E(0,0) == Approx(-0.58947939694779772));
}

data.read_full_bgen();
SECTION( "Ex6. bgen read in & standardised correctly"){
CHECK(data.G.low_mem == false);
CHECK(data.params.low_mem == false);
CHECK(data.params.flip_high_maf_variants == true);
CHECK(data.G(0, 0) == Approx(1.8604233373));
}

SECTION( "Ex6. Confirm calc_dxteex() reorders properly"){
data.params.dxteex_file = "data/io_test/inputs/dxteex_mixed.txt";
data.read_external_dxteex();
data.calc_dxteex();
CHECK(data.dXtEEX(0, 0) == Approx(87.204591182113916));
CHECK(data.n_dxteex_computed == 1);
}

data.calc_dxteex();
if(p.vb_init_file != "NULL"){
data.read_alpha_mu();
}
VBayesX2 VB(data);
VB.check_inputs();
SECTION("Ex6. Vbayes_X2 initialised correctly"){
CHECK(VB.n_samples == 50);
CHECK(VB.N == 50.0);
CHECK(VB.n_env == 1);
//CHECK(VB.n_covar == 1);
CHECK(VB.n_effects == 2);
CHECK(VB.vp_init.muw(0) == 1.0);
CHECK(VB.p.init_weights_with_snpwise_scan == false);
CHECK(VB.dXtEEX(0, 0) == Approx(87.204591182113916));
}

std::vector< VbTracker > trackers(VB.hyps_grid.rows(), p);
SECTION("Ex6. Explicitly checking updates"){
// Set up for RunInnerLoop
long n_grid = VB.hyps_grid.rows();
std::vector<Hyps> all_hyps;
VB.unpack_hyps(VB.hyps_grid, all_hyps);

// Set up for updateAllParams
std::vector<VariationalParameters> all_vp;
VB.setup_variational_params(all_hyps, all_vp);

int round_index = 2;
std::vector<double> logw_prev(n_grid, -std::numeric_limits<double>::max());
std::vector<std::vector< double >> logw_updates(n_grid);

VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev);
VariationalParameters& vp = all_vp[0];
Hyps& hyps = all_hyps[0];

CHECK(vp.alpha_beta(0) == Approx(0.1447525646));
CHECK(vp.mu1_beta(0) == Approx(-0.0304566021));
CHECK(vp.mu2_beta(0) == Approx(-0.0003586526));
CHECK(vp.alpha_beta(1) == Approx(0.1515936892));
CHECK(vp.mu1_beta(1) == Approx(-0.0356183259));
CHECK(vp.mu2_beta(1) == Approx(-0.0004194363));
CHECK(vp.alpha_beta(63) == Approx(0.1762251019));
CHECK(hyps.sigma == Approx(0.3994029731));
CHECK(hyps.lambda(0) == Approx(0.1693099847));
CHECK(hyps.slab_var(0) == Approx(0.0056085838));
CHECK(hyps.spike_var(0) == Approx(0.0000368515));
CHECK(VB.calc_logw(hyps, vp) == Approx(-52.129381445));

VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev);

CHECK(vp.alpha_beta(0) == Approx(0.1428104733));
CHECK(vp.mu1_beta(0) == Approx(-0.01972825));
CHECK(vp.mu2_beta(0) == Approx(-0.0002178332));
CHECK(vp.alpha_beta(1) == Approx(0.1580997887));
CHECK(vp.alpha_beta(63) == Approx(0.6342565543));
CHECK(hyps.sigma == Approx(0.2888497603));
CHECK(hyps.lambda(0) == Approx(0.2065007836));
CHECK(hyps.slab_var(0) == Approx(0.0077922078));
CHECK(hyps.spike_var(0) == Approx(0.0000369985));
CHECK(VB.calc_logw(hyps, vp) == Approx(-48.0705874648));
}

SECTION("Ex6. Checking rescan") {
// Set up for RunInnerLoop
// Set up for RunInnerLoop
int round_index = 2;
long n_samples = VB.n_samples;
long n_grid = VB.hyps_grid.rows();
std::vector<Hyps> all_hyps;
VB.unpack_hyps(VB.hyps_grid, all_hyps);

// Allocate trackers
std::vector< VbTracker > trackers(n_grid, p);

VB.runInnerLoop(false, round_index, all_hyps, trackers);


CHECK(trackers[1].logw == Approx(-45.7823937859));
CHECK(trackers[1].vp.eta[0] == Approx(-0.5894793969));
CHECK(trackers[1].vp.ym[0] == Approx(-0.8185317198));

Eigen::VectorXd gam_neglogp(VB.n_var);
VB.rescanGWAS(trackers[1].vp, gam_neglogp);

// Eigen::VectorXd pheno = VB.Y - trackers[1].vp.ym;
// Eigen::VectorXd Z_kk(n_samples);
// int jj = 1;
// Z_kk = VB.X.col(jj).cwiseProduct(trackers[1].vp.eta);
// double ztz_inv = 1.0 / Z_kk.dot(Z_kk);
// double gam = Z_kk.dot(pheno) * ztz_inv;
// double rss_null = (pheno - Z_kk * gam).squaredNorm();
//
// // T-test of variant j
// boost_m::students_t t_dist(n_samples - 1);
// double main_se_j    = std::sqrt(rss_null / (VB.N - 1.0) * ztz_inv);
// double main_tstat_j = gam / main_se_j;
// double main_pval_j  = 2 * boost_m::cdf(boost_m::complement(t_dist, fabs(main_tstat_j)));
//
// double neglogp_j = -1 * std::log10(main_pval_j);

CHECK(gam_neglogp[1] == Approx(0.2392402716));
// CHECK(pheno[0] == Approx(-0.4439596651));
// CHECK(VB.X.col(jj)[0] == Approx(0.7465835328));
// CHECK(Z_kk[0] == Approx(-0.44009531));
// CHECK(gam == Approx(0.0223947128));
// CHECK(main_pval_j == Approx(0.576447458));
// CHECK(main_tstat_j == Approx(0.5623409325));
// CHECK(main_se_j == Approx(0.0398240845));
// CHECK(rss_null == Approx(7.9181184549));
}

VB.run_inference(VB.hyps_grid, false, 2, trackers);
SECTION("Ex6. Vbayes_X2 inference correct"){
//			CHECK(trackers[0].count == 741);
//			CHECK(trackers[3].count == 71);
//			CHECK(trackers[0].logw == Approx(-45.2036994175));
//			CHECK(trackers[1].logw == Approx(-40.8450319874));
//			CHECK(trackers[2].logw == Approx(-40.960377414));
//			CHECK(trackers[3].logw == Approx(-40.9917439828));
CHECK(trackers[0].count == 20);
CHECK(trackers[3].count == 20);
CHECK(trackers[0].logw == Approx(-45.8542053615));
CHECK(trackers[1].logw == Approx(-45.7823937859));
CHECK(trackers[2].logw == Approx(-41.3150655897));
CHECK(trackers[3].logw == Approx(-41.639981773));
}
}
}
