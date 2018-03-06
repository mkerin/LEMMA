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
covar.path = file.path(proj_dir, "data", "io_test", "t2_lm", "t2.covar")
pheno.path = file.path(proj_dir, "data", "io_test", "t2_lm", "t2.pheno")
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

my_bgen_prog_joint = function(gga, ggb){
    # Mean imputation
    gga[is.na(gga)] = mean(gga[!is.na(gga)])
    ggb[is.na(ggb)] = mean(ggb[!is.na(ggb)])
    
    N = nrow(daf.covar)
    daf = cbind(cbind(daf.covar, daf.pheno), 
                gga = c(gga, rep(NA, N - length(gga))),
                ggb = c(ggb, rep(NA, N - length(ggb))))
    daf = daf[complete.cases(daf),]
    
    # Scale complete samples only
    daf$gga = as.vector(scale(daf$gga, scale = T))
    daf$ggb = as.vector(scale(daf$ggb, scale = T))
    daf$covar1 = as.vector(scale(daf$covar1, scale = T))
    daf$pheno1 = as.vector(scale(daf$pheno1, scale = F))
    daf$za = daf$covar1 * daf$gga
    daf$zb = daf$covar1 * daf$ggb
    
    # null
    f0 = lm(pheno1 ~ covar1, daf)
    daf$pheno10 = f0$residuals
    f1 = lm(pheno10 ~ 0 + gga + ggb, daf)
    
    # joint interaction
    f2 = lm(pheno10 ~ 0 + gga + ggb + za + zb, daf)
    test = anova(f1, f2)
    print(paste0("Var-explained null: ", 100 * var(f1$fitted.values) / var(daf$pheno10)))
    print(paste0("Var-explained null: ", 100 * var(f2$fitted.values) / var(daf$pheno10)))
    print(paste0("F-test: ", test$F[2]))
    print(paste0("neglogp: ", -log10(test[2, "Pr(>F)"])))
}

my_bgen_prog_joint(gg1, gg2)


# Create example files.
# covar.path = file.path(proj_dir, "data", "example", "example.covar")
# pheno.path = file.path(proj_dir, "data", "example", "example.pheno")
# 
# daf.covar = data.frame(covar1 = c(sample(1:3, replace=T, size = length(gg)),
#                                   rep(NA, N - length(gg))))
# daf.pheno = data.frame(pheno1 = c(gg + cc + gg*cc + rnorm(length(gg), 0, 20),
#                                   rep(NA, N - length(gg))))
# write.table(daf.covar, covar.path,
#             row.names = F,
#             col.names = T)
# write.table(daf.pheno, pheno.path,
#             row.names = F,
#             col.names = T)

