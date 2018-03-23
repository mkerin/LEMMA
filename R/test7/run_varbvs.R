# Script to run varbvs and write "correct" output to io_test/t7*
# 
# 
# 
library(varbvs)
source('R/misc.R')


## constants
test.dir   = "data/io_test/t7_varbvs_constrained"


## Data
y          = read.table(file.path(test.dir, "pheno.txt"), header = T)[,1]
age        = read.table(file.path(test.dir, "age.txt"), header = T)[,1]
X          = readStandardisedDosage(file.path(test.dir, "n50_p100.vcf.gz"))
Z          = apply(X, 2, function(col){col * age})
df.init    = read.table(file.path(test.dir, "answer_init.txt"), header = T)
df.grid    = read.table(file.path(test.dir, "hyperpriors_gxage.txt"), header = T)
df.probs   = read.table(file.path(test.dir, "hyperpriors_gxage_probs.txt"), header = T)


## Wrangling
y          = as.vector(scale(y, center = T, scale = F))
age        = as.vector(scale(age, center = T, scale = T))
df.fit     = data.frame(y = y, age = age)
fit        = lm(y ~ 0 + age, data = df.fit)
y          = as.vector(fit$residuals)

stopifnot(all(df.grid$sigma_b == df.grid$sigma_g))
stopifnot(all(df.grid$lambda_b == df.grid$lambda_g))


## Run varbvs
H = cbind(X, Z)
fit = varbvs::varbvs(H, Z              = NULL,
                     y                 = y,
                     family            = "gaussian",
                     sigma             = df.grid$sigma,
                     sa                = df.grid$sigma_b,
                     logodds           = logit10(df.grid$lambda_b),
                     tol               = 1e-4,
                     maxiter           = 100,
                     alpha             = df.init$alpha,
                     mu                = df.init$mu,
                     initialize.params = F)


## convert object to output expected by plots etc

varbvsToROutput = function(vfit, grid.hyps, grid.probs){
    # account for prior probs and renormalise
    weights = vfit$logw + log(grid.probs[,1])
    weights = normalizelogweights(weights)
    
    res = list(weights           = weights,
               mu.post           = NA,
               beta.post         = as.vector(vfit$beta),
               alpha.post        = as.vector(vfit$pip),
               alpha.list        = length(as.list(as.data.frame(vfit$alpha))),
               mu.list           = length(as.list(as.data.frame(vfit$mu))),
               counts.vec        = NA,
               logw.updates.list = NA,
               grid.hyps         = grid.hyps,
               grid.probs        = grid.probs,
               logw.vec          = vfit$logw)
}

res = varbvsToROutput(fit, df.grid, df.probs)
saveRDS(res, file.path(test.dir, "answer.rds"))

