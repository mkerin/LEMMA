# Version 0.10.6

Changes:
- Support for SQUAREM accelerator
- GxE tests done with 'robust' standard errors (to heteroskedacity)
- Don't allow convergence on SQUAREM iteration

Commandline:
- SQUAREM on by default
-- --empirical_bayes turns on niave hyp maximisation
-- --constant_hyps reduces to just variational inference


# Version 0.10.6

Changes:
- bugfix in GenotypeMatrix::transpose_multiply
- Added PVE estimation with HE reg method
- bugfix in computing chromosome-residuals
- Change to dump state; should be possible to fully recover now


# Version 0.10.5

Changes:
- Multithreaded read from bgen
- bugfix in GenotypeMatrix::mult_vector_by_chr
- VbTracker::dump_state
- Refactoring of elbo; confident this is correct
- --mode_debug

# Version 0.10.4

Changes:
- Covars regressed from pheno and env (Y = E alpha + X beta + Z gamma)
- More flexible input of hyps grid in main effects mode.
- Removal of --interaction flag.
- Use --low_mem implementation by default.
- Flag --mode_no_gxe to run Y = E alpha + X beta
- dxteex only computed if n_effects > 1 (interaction mode)
- software version number included in commandline output
- ability to read from gzipped files
- output pvals + test stats computed with LOCO strategy
- print rows x cols of files read in
- monotonic elbo check for main and gxe effects

Bugfix:
- forgot to use log in elbo when use_vb_on_covars
- bugfix in pve_large
- mean center s_z

TODO:
- Stop flipping variants if maf > 0.5 (misleading if triplet snps).
- regress covars from genotypes.

# Version 0.10.3

Changes:
- Now compile with INTEL MKL
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
- read_environment_weights() should read from env_coeffs_file! (+ use read_grid_file())

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
- Flag to set beta_spike_diff_factor from commandline.

# Version 0.9.1
Minor bugfix in hyps maxizing hyps.

# Version 0.9.0
Each run assigned to a single core. Think this is highly wasteful.

Ordered todo list:
1. Profile code
2. Output runtime of component functions to text file
3. Allow Catch2 tests to be separated into new files
