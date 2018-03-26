# Script to generate hyperparams grid assuming identical priors on
# main and interaction effects.
# Format also compatible for intake to bgen_prog
# Hardcoded to io_test/t7*
# 
source('R/misc.R')


## constants
test.dir   = "data/io_test/t9_varbvs_zero_hg"
n.sigma    = 1
n.lambda_b = 3
n.lambda_g = 3
n.h_b      = 4
n.h_g      = 4


## Data
X          = readStandardisedDosage(file.path(test.dir, "n50_p100.vcf.gz"))
aa         = read.table(file.path(test.dir, "age.txt"), header = T)[,1]
Z          = apply(X, 2, function(col){col * aa})
s_z        = sum(apply(Z, 2, var))  # sum of sample variances over variants
s_x        = sum(apply(X, 2, var))  # == P unless some cols have zero variance


## Functions
priorProb = function(row, my.s_x = s_x, my.s_z = s_z, cols = colnames(df)){
    lam = as.numeric(row[grep("^lambda_b$", cols)]) +
        as.numeric(row[grep("^lambda_g$", cols)])
    probs = data.frame(sigma = 1 / row[grep("^sigma$", cols)],
                       h = 1,
                       lambda = f_y(lam, -2, 0.6))
    return(probs)
}

pve2sigma.singleVar = function(pve, my.s_x, my.s_z, lam_b, lam_g){
    # Calculates effects variance under the assumption that this is identical for main
    # and interaction effects.
    # Assumes pve identical across these two items. Might not have to.
    pve / (1 - pve) / (lam_b * my.s_x + lam_g * my.s_z)
}

pve2sigma = function(lam, sample_var, pve, other_pve){
    pve / (1 - pve - other_pve) / lam / sample_var
}



## Create grid
df = expand.grid(sigma = seq(1, 1, length.out = n.sigma),
                 h_b = seq(0.1, 0.4, length.out = n.h_b),
                 h_g = seq(0.1, 0.4, length.out = n.h_g),
                 logit10_lambda_b = seq(-2.8, -0.8, length.out = n.lambda_b),
                 logit10_lambda_g = seq(-2.8, -0.8, length.out = n.lambda_g))

df$lambda_b = sigmoid10(df$logit10_lambda_b)
df$lambda_g = sigmoid10(df$logit10_lambda_g)
df$sigma_b = pve2sigma(df$lambda_b, s_x, df$h_b, df$h_g)
df$sigma_g = pve2sigma(df$lambda_g, s_z, df$h_g, df$h_b)

df = df[,c("sigma", "sigma_b", "sigma_g", "lambda_b", "lambda_g", "h_b", "h_g")]


## Calc prior probs
df.probs = apply(df, 1, priorProb)
df.probs = do.call(rbind, df.probs)
df.probs = apply(df.probs, 1, function(row, total){ row / total }, colSums(df.probs))
df.probs = t(df.probs)
df.probs = data.frame(prob = apply(df.probs, 1, prod))


## Save output
save.path = file.path(test.dir, "hyperpriors_gxage.txt")
print(paste0("Saving to: ", save.path))
write.table(df, save.path,
            row.names = F,
            col.names = T,
            quote = F)

save.path = file.path(test.dir, "hyperpriors_gxage_probs.txt")
print(paste0("Saving to: ", save.path))
write.table(df.probs, save.path,
            row.names = F,
            col.names = T,
            quote = F)


