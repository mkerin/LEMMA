vars=( hyps_grid environment_weights suppress_squared_env_removal incl_squared_envs resume_from_state state_dump_interval incl_rsids incl_sample_ids spike_diff_factor gam_spike_diff_factor beta_spike_diff_factor force_write_vparams covar_init vb_init xtra_verbose snpwise_scan pve_mog_weights rhe_random_vectors use_vb_on_covars keep_constant_variants mode_debug raw_phenotypes high_mem low_mem joint_covar_update min_alpha_diff vb_iter_start effects_prior_mog mode_spike_slab main_chunk_size gxe_chunk_size min_spike_diff_factor mode_regress_out_covars exclude_ones_from_env_sq mode_alternating_updates hyps_probs loso_window_size drop_loco init_weights_with_snpwise_scan mode_pve_est mode_dump_processed_data )

for var in "${vars[@]}"; do
    new=`echo "$var" | sed 's/_/-/g'`
    sed -i "s/--${var}/--${new}/g" unit/*pp src/parse_arguments.cpp README.md
done

for var in "${vars[@]}"; do
    new=`echo "$var" | sed 's/_/-/g'`
    sed -i "s/\"${var}/\"${new}/g" src/parse_arguments.cpp
done

% Convert unit tests TODO dxteex
vars=( n50_p100_env n50_p100.bgen pheno n50_p100_ones n50_p100_nls_env_weights n50_p100_nm_env_weights )
for var in "${vars}"; do
cp data/io_test/${var}* unit/data/
sed -i "s/data\/io_test\/${var}/unit\/data\/${var}/g" unit/*pp
done
