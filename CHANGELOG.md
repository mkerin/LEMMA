# Version 0.9.3
Bugfix:
- Don't try to regress out covars when not present
- Now able to run only main effects model
- Need to transpose snpstats.row() to assign to Eigen::ArrayXd vp.muw
- Init weights from sumstats if n_env > 1
- read_environment_weights() should read from env_weights_file! (+ use read_grid_file())

Changes:
- Ability to set minimum difference in variances of MoG.
- Start of creating functions to write output to file.
- predicted values stratified into Xbeta and Zgamma.
- 'Rescan' GWAS of Z on y-ym for MAP iteration.
- Removed dependence on probs grid
- All variant metadata stored in genotype matrix object.
- Close outf_scan after writing.
- Check hyp grid values are finite
- Cleaner timing output

# Version 0.9.2
Bugfixes:
- CMakeLists; hardcoding of ${BGEN} + included rescomp path to boost
- E_reduced initialised.
- MAP output reflects if mog prior used.

Changes:
- removed mode_approximate_updates and hty_counter.
- E[beta] and s_sq included in parameter dumps.
- Expansion of Catch2 tests to include param updates in Mog mode.
- Flag to set spike_diff_factor from commandline.

# Version 0.9.1
Minor bugfix in hyps maxizing hyps.

# Version 0.9.0
Each run assigned to a single core. Think this is highly wasteful.

Ordered todo list:
1. Profile code
2. Output runtime of component functions to text file
3. Allow Catch2 tests to be separated into new files
