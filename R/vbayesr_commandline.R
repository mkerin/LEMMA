# R script to run interaction VB model with commandline arguments
'vbayesr commandline interface.

Usage:
vbayesr_commandline.R run --vcf=FILE --pheno=FILE --covar=FILE --out=FILE [--vb_init=FILE --hyps_grid=FILE --hyps_probs=FILE -v  --plot_dir=DIR --true_vals=FILE]
vbayesr_commandline.R plot --out=FILE --plot_dir=DIR [--true_vals=FILE]

--vcf=FILE        Filepath to vcf file
--covar=FILE      Filepath to covar file. Header expected.
--pheno=FILE      Filepath to pheno file
--out=FILE        Filepath used to save output to.
--hyps_grid=FILE  Filepath to hyps_grid file
--vb_init=FILE    Filepath to VB inits file
--plot_dir=PATH      Directory to save plots to
-v --verbose      Extra diagnostic messages printed to screen.
--true_vals=FILE  Filepath to file containing true hyperparameters (for simulation). Also the readme.' -> doc
library(vbayesr)
# source('R/vbayes/misc.R')

if(T){
    opts = docopt::docopt(doc)
} else {
    opts = list(run = TRUE,
                plot = FALSE,
                vcf = "data/simulations/genetic/n1000_p2000.vcf.gz",
                covar = "data/simulations/adaptive_grid/case1/age.txt",
                pheno = "data/simulations/adaptive_grid/case1/pheno.txt",
                out = "data/simulations/adaptive_grid/case1/r_adaptive.rds",
                vb_init = "data/simulations/adaptive_grid/case1/cpp_inference_inits.out",
                plot_dir = "results/adaptive_grid/case1",
                true_vals = "data/simulations/adaptive_grid/readme.txt")
    
    opts = list(run = FALSE,
                plot = TRUE,
                out = "data/simulations/adaptive_grid/case1/cpp_inference.out",
                plot_dir = "results/adaptive_grid/case1",
                true_vals = "data/simulations/adaptive_grid/case1/readme.txt")
}
print(opts)
stopifnot(xor(opts$run, opts$plot))

if(opts$run){
    ### Reading rest of data
    print("Reading in genetic data")
    X            = vbayesr::readStandardisedDosage(opts$vcf)
    print("Reading in the rest of the data")
    y            = read.table(opts$pheno, header = T)[,1]
    age          = read.table(opts$covar, header = T)[,1]
    stopifnot(length(y) == nrow(X))
    stopifnot(length(age) == nrow(X))
    
    if(is.null(opts$hyps_grid)){
        grid.hyps     = NULL
    } else {
        grid.hyps     = read.table(opts$hyps_grid, header = T)
    }
    
    # alpha / mu initial values
    if(is.null(opts$vb_init)){
        df.init = NULL
    } else {
        df.init = read.table(opts$vb_init, header = T)
    }
    
    # run program
    out = vbayesr::vbayesr(y, X, age, grid.hyps, vb.init = df.init)
    print(paste0("Saving to: ", opts$out))
    saveRDS(out, opts$out)
}

if (opts$plot) {
    if(grepl("cpp", opts$out) || grepl("\\.out", opts$out)){
        out = vbayesr::readCppOutput(opts$out, my.mode = "brief")
    } else {
        out = readRDS(opts$out)
    }
    
    # library(tidyverse)
    # source('../software/vbayesr_package/R/functions_plotting.R')
    # p.all = PlotGridSearch(out$grid.post, df.true_vals = df.true_vals, default.zoom = F)
    # ggplot2::ggsave(file.path(opts$plot_dir, "search_layer_all.pdf"), p.all, width=11.5, height = 7)
}    

if (opts$plot || (opts$run & !is.null(opts$plot_dir))) {
    if(is.null(opts$true_vals)){
        df.true_vals = NULL
    } else {
        df.true_vals = read.table(opts$true_vals, header = T)
        colnames(df.true_vals) = gsub("\\.100", "", colnames(df.true_vals))
        df.true_vals$h_b = df.true_vals$h_b / 100
        df.true_vals$h_g = df.true_vals$h_g / 100
    }
    
    ### Plotting
    print("Starting to generate plots")
    prefix = file.path(opts$plot_dir, paste0(tools::file_path_sans_ext(basename(opts$out)), "_"))
    
    ## GRIDSEARCH PLOTS
    p.all = plot(out, method = "gridsearch", df.true_vals = df.true_vals)
    ggplot2::ggsave(paste0(prefix, "gridsearch_l_all.pdf"), p.all, width=11.5, height = 7)
    
    p.all = plot(out, method = "gridsearch", df.true_vals = df.true_vals, metric = "logw")
    ggplot2::ggsave(paste0(prefix, "gridsearch_logw_l_all.pdf"), p.all, width=11.5, height = 7)
    
    for (ii in out$grid.post$layer_id){
        p1 = plot(out, method = "gridsearch", layer_id = ii, df.true_vals = df.true_vals)
        ggplot2::ggsave(paste0(prefix, "gridsearch_l", ii, ".pdf"), p1, width=11.5, height = 7)
        
        p1 = plot(out, method = "gridsearch", layer_id = ii, df.true_vals = df.true_vals, metric = "logw")
        ggplot2::ggsave(paste0(prefix, "gridsearch_logw_l", ii, ".pdf"), p1, width=11.5, height = 7)
    }
    
    ## ELBO TRAJECTORIES
    if(!all(is.na(out$logw.updates.list))){
        p.traj = plot(out, method = "trajectories")
        ggplot2::ggsave(paste0(prefix, "trajectories.pdf"), p.traj, width=11.5, height = 7)
        
    }
    
    ## COUNTS TO CONVERGENCE
    p.counts = plot(out, method = "counts")
    ggplot2::ggsave(paste0(prefix, "counts.pdf"), p.counts, width=11.5, height = 7)
    
    ## MARGINAL DISTRIBUTIONS
    p.dash = plot(out, method = "dashboard", df.true_vals = df.true_vals)
    ggplot2::ggsave(paste0(prefix, "marginal_distributions.pdf"), p.dash, width=11.5, height = 7)
    
}


