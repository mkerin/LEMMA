proj_dir = "/Users/kelb4230/Dropbox/MyPCBackUp/Documents/gms/dphil/projects/bgen-prog"
# Creating pheno / covariate file for example 1
N = 500
gg1 = c(NA,
        1.9357,
        1.91553,
        1.01743,
        1.91159,
        1.85892,
        0.926727,
        1.94586,
        1.89142,
        1.87692,
        1.86243)

gg2 = c(0.1604,
        0.0642395,
        0.0844116,
        0.982574,
        0.0884094,
        0.141083,
        1.07333,
        0.0541382,
        0.108582,
        0.123077,
        0.137573)


# Performing test regression
covar.path = file.path(proj_dir, "data", "io_test", "t6_lm2", "t6.covar")
pheno.path = file.path(proj_dir, "data", "io_test", "t6_lm2", "t6.pheno")
daf.pheno = read.table(pheno.path, header=T)
daf.covar = read.table(covar.path, header=T)

# Functions
lrt_from_residuals = function(null, alt, df){
    nn = length(null)
    loglik.null = (log(nn) - log(crossprod(null))) * nn / 2
    loglik.alt = (log(nn) - log(crossprod(alt))) * nn / 2
    test.stat = 2*(loglik.alt - loglik.null)
    print(paste("loglik.null (up to const):", loglik.null))
    print(paste("loglik.alt (up to const):", loglik.alt))
    print(paste("neglogP:", -log10(pchisq(test.stat, df, lower.tail = F))))
}

my_bgen_prog_lm = function(gg){
    # env - covar1
    # gconf - covar2
    # 
    # lm(y ~ covars + env + covars*env + gene + gene*env + gene*covars)
    # lm(y ~ covar1 + covar2 + covar1*covar2 + gg + gg*covar1 + gg*covar2)
    
    # af
    af = mean(gg[!is.na(gg)]) / 2
    
    # Mean imputation
    gg[is.na(gg)] = mean(gg[!is.na(gg)])
    
    N = nrow(daf.covar)
    daf = cbind(cbind(daf.covar, daf.pheno), gg=c(gg, rep(NA, N - length(gg))))
    daf$gg.int = ceiling(daf$gg - 0.5)
    daf = daf[complete.cases(daf),]
    
    # Scale complete samples only
    daf$gg = as.vector(scale(daf$gg, scale = T))
    daf$covar1 = as.vector(scale(daf$covar1, scale = T))
    daf$covar2 = as.vector(scale(daf$covar2, scale = T))
    daf$covar1_covar2 = as.vector(scale(daf$covar1_covar2, scale = T))
    daf$pheno1 = as.vector(scale(daf$pheno1, scale = F))
    
    daf1 = daf[,c("covar1", "covar2", "covar1_covar2", "gg")]
    daf1$gg_covar1 = as.vector(scale(daf1$gg * daf1$covar1, center = T, scale = T))
    daf1$gg_covar2 = as.vector(scale(daf1$gg * daf1$covar2, center = T, scale = T))
    daf1$y = daf$pheno1
    
    # run lm
    ff = lm(y~0+., daf1)
    
    # coefficients
    res = c(summary(ff)$coefficients[, 1])
    res["neglogp"] = -log10(summary(ff)$coefficients["gg_covar1", 4])
    print(res)
}

my_bgen_prog_lm(gg1)
my_bgen_prog_lm(gg2)


# Create example files.
# covar.path = file.path(proj_dir, "data", "io_test", "t6_lm2", "t6.covar")
# pheno.path = file.path(proj_dir, "data", "io_test", "t6_lm2", "t6.pheno")
# 
# cc1 = as.vector(scale(sample(1:3, replace=T, size = length(gg2)), center=T, scale = F))
# cc2 = as.vector(scale(rnorm(length(gg2)), center=T, scale = F))
# # 
# daf.covar = data.frame(covar1 = c(cc1, rep(NA, N - length(gg2))),
#                        covar2 = c(cc2, rep(NA, N - length(gg2))),
#                        covar1_covar2 = c(cc1*cc2, rep(NA, N - length(gg2))))
# daf.pheno = data.frame(pheno1 = c(gg2 + cc1 + gg2*cc1 + cc1*cc2 + rnorm(length(gg2), 0, 20),
#                                   rep(NA, N - length(gg2))))
# write.table(daf.covar, covar.path,
#             row.names = F,
#             col.names = T,
#             quote = F)
# write.table(daf.pheno, pheno.path,
#             row.names = F,
#             col.names = T,
#             quote = F)

