# Version 0.10.3

Changes:
- DATA_AS_FLOAT implemented. Preliminary testing suggests this is a 2x speedup with small changes in accuracy
- Initialise logw with numeric_limits::min()
- interim outputs include low maintenance measure of wall clock time
- include sum of squares of covar coeffs in interim output

Bugfix:
- removed circular dependancy between header files

# Version 0.10.2

Changes:
- Only a single thread used to decompress cols.
- vb block size for main effects set to 64. gxe effects allowed to be tuned.
- Only compute StrictlyUpper triangular part of snp corr matrix for param adjustment (note with Eigen this is not multithreaded!)

# Version 0.10.1

Changes:
- All threads used to decompress genotype columns.
- Major refactoring: single array for each parameter (in future allows for mutliple runs in XXd array)
- Hyps now a class
- VbTracker stores only a single run
- We now keep running threads until they all converge
- Some tests limited to 20 iterations
- Simplified internal timing

# Version 0.9.3
Bugfix:
- Don't try to regress out covars when not present
- Now able to run only main effects model
- Need to transpose snpstats.row() to assign to Eigen::ArrayXd vp.muw
- Init weights from sumstats if n_env > 1

Changes:
- Ability to set minimum difference in variances of MoG.
- Start of creating functions to write output to file.
- predicted values stratified into Xbeta and Zgamma.
- 'Rescan' GWAS of Z on y-ym for MAP iteration.
- Removed dependence on probs grid
- All variant metadata stored in genotype matrix object.
- Close outf_scan after writing.
- Check hyp grid values are finite

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
