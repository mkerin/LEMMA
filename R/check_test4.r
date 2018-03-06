

gg3 = c(1.98779,
        1.97803,
        0.0211182,
        0.0308838,
        1.00125,
        1.00571,
        0.997559,
        0.00964355,
        1.00342,
        1.00409,
        0.00567627,
        0.0130615,
        0.999298,
        0.998688,
        1.00308,
        0.997742,
        0.0604553,
        0.00256348,
        1.00366,
        0.0308838,
        0.987152)

# Performing test regression
proj_dir = "/Users/kelb4230/Dropbox/MyPCBackUp/Documents/gms/dphil/projects/bgen-prog"
covar.path = file.path(proj_dir, "data", "io_test", "t4_lm_2dof", "t4.covar")
pheno.path = file.path(proj_dir, "data", "io_test", "t4_lm_2dof", "t4.pheno")
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
    
    # 2dof w/ constraints
    print("2dof test")
    if (length(unique(daf$gg.int)) == 3){
        for(kk in 1:3){
            col_name = paste0("covar1", LETTERS[kk])
            daf[,col_name] = ifelse(daf$gg.int == kk-1, daf$covar1, 0)
        }
        MM = table(daf$gg.int)
        daf$covar1B_trans = daf$covar1B - daf$covar1A * MM["1"] / MM["0"]
        daf$covar1C_trans = daf$covar1C - daf$covar1A * MM["2"] / MM["0"]
        f3 = lm(pheno11 ~ 0 + covar1B_trans + covar1C_trans, daf)
        
        gam2 = f3$coefficients["covar1B_trans"]
        gam3 = f3$coefficients["covar1C_trans"]
        gam1 = -(gam2 * MM["1"] + gam3 * MM["2"]) / MM["0"]
        print(paste("gamma1:", gam1))
        print(paste("gamma2:", gam2))
        print(paste("gamma3:", gam3))
        lrt_from_residuals(f1$residuals, f3$residuals, 2)
    } else {
        print("ERROR - need data covering all 3 genotypes")
    }
}

my_bgen_prog_lm(gg3)



# # Create example files.
# N = 500
# covar.path = file.path(proj_dir, "data", "io_test", "t4_lm_2dof", "t4.covar")
# pheno.path = file.path(proj_dir, "data", "io_test", "t4_lm_2dof", "t4.pheno")
# cc = sample(1:10, replace=T, size = length(gg3))
# daf.covar = data.frame(covar1 = c(cc, rep(NA, N - length(gg3))))
# daf.pheno = data.frame(pheno1 = c(gg + cc + gg*cc + rnorm(length(gg3), 0, 20),
#                                   rep(NA, N - length(gg3))))
# write.table(daf.covar, covar.path,
#             row.names = F,
#             col.names = T)
# write.table(daf.pheno, pheno.path,
#             row.names = F,
#             col.names = T)
