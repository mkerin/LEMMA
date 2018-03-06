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

my_bgen_prog_lm = function(gg){
    
    # af
    af = mean(gg[!is.na(gg)]) / 2
    
    # info
    
    
    # Mean imputation
    gg[is.na(gg)] = mean(gg[!is.na(gg)])
    
    N = nrow(daf.covar)
    daf = cbind(cbind(daf.covar, daf.pheno), gg=c(gg, rep(NA, N - length(gg))))
    daf$gg.int = ceiling(daf$gg - 0.5)
    daf = daf[complete.cases(daf),]
    
    # Scale complete samples only
    daf$gg = as.vector(scale(daf$gg, scale = T))
    daf$covar1 = as.vector(scale(daf$covar1, scale = T))
    daf$pheno1 = as.vector(scale(daf$pheno1, scale = F))
    daf$z = daf$covar1 * daf$gg
    
    # null
    f0 = lm(pheno1 ~ covar1, daf)
    daf$pheno10 = f0$residuals
    f1 = lm(pheno10 ~ 0 + gg, daf)
    daf$pheno11 = f1$residuals
    
    # 1 dof
    print("1dof test")
    f2 = lm(pheno11 ~ 0 + z, daf)
    print(paste("af:", af))
    print(paste("beta:", f1$coefficients["gg"]))
    print(paste("tau:", f2$coefficients["z"]))
    lrt_from_residuals(f1$residuals, f2$residuals, 1)
    
    # 2dof
    print("2dof test")
    if (length(unique(daf$gg.int)) == 3){
        for(kk in 1:3){
            daf[,paste0("covar1", LETTERS[kk])] = ifelse(daf$gg.int == kk,
                                                         daf$covar1,
                                                         0)
        }
        f3 = lm(pheno11 ~ 0 + covar1A + covar1B + covar1C, daf)
        lrt_from_residuals(f1$residuals, f3$residuals, 2)
    } else {
        print("ERROR - need data covering all 3 genotypes")
    }
}

my_bgen_prog_lm(gg1)
my_bgen_prog_lm(gg2)


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

