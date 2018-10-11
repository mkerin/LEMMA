# Version 0.9.3
Changes:
- Ability to set minimum difference in variances of MoG.
- Start of creating functions to write output to file.

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
