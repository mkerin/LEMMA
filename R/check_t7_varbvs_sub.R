# Script to check inference of varbvs package with bgen_prog
# 
# 
# 
# 
library(VariantAnnotation, quietly = T)
library(varbvs)
library(ggplot2)
library(ggrepel)
source("R/misc.R")

args = list("data/io_test/t7_varbvs_sub/n500_p1000.vcf.gz",
            "data/io_test/t7_varbvs_sub/")

vcf <- readVcf(args[[1]], "hg19")
y = as.matrix(read.table(file.path(args[[2]], "pheno.txt"), header=T))
grid = read.table(file.path(args[[2]], "hyperpriors_grid.txt"), header=T)
alpha.init = read.table(file.path(args[[2]], "alpha_init.txt"), header=T)
mu.init = read.table(file.path(args[[2]], "mu_init.txt"), header=T)

plt.dir = file.path(args[[2]], "plots")
if(!dir.exists(plt.dir)){ dir.create(plt.dir) }

# matrix of dosages
ds = sapply(geno(vcf)$GP, function(vec){vec[2] + 2*vec[3]})
ds = matrix(ds, nrow = nrow(geno(vcf)$GP), ncol = ncol(geno(vcf)$GP))
row.names(ds) = row.names(geno(vcf)$GP)
colnames(ds) = colnames(geno(vcf)$GP)
ds = t(ds)

# Run varbvs
fit.v <- varbvs(ds,NULL,y,family = "gaussian",
                  logodds = c(log10(grid$pi), -3.5), sa = 1, sigma = 1, 
                  alpha = as.matrix(alpha.init), 
                  mu = as.matrix(mu.init))

# NB: error in function with single grid point. Hence run with two and fit from first
fit.v$mu = fit.v$mu[,1]
fit.v$alpha = fit.v$alpha[,1]
fit.v$w = 1
fit.v$pip      <- c(fit.v$alpha)
fit.v$beta <- c(fit.v$alpha *fit.v$mu)
fit.v$logw = fit.v$logw[1]



# Take a look at how accurate the PIP are
true_beta = read.table(file.path(args[[2]], "pheno_true_beta.txt"), header = T)[,1]
true_pip = ifelse(true_beta == 0, 0, 1)
LD.m = calc_ldmap(ds)
p = plotAnnotPIP(fit.v$pip, true_pip, LD.m)
ggsave(file.path(plt.dir, "pip_answer.pdf"), p)

# Write posterior inf file
ans = data.frame(post_alpha = c(fit.v$pip),
                 post_mu = c(fit.v$mu),
                 post_beta = c(fit.v$beta))
write.table(ans, file.path(args[[2]], "varbvs_answer.out"), col.names=T, row.names=F, quote=F)

# Write posterior hyps file
ans = data.frame(weights = c(fit.v$w),
                 logw = c(fit.v$logw))
write.table(ans, file.path(args[[2]], "varbvs_answer_hyps.out"), col.names=T, row.names=F, quote=F)




# # Create hyperprior files
# logodds = seq(-3.5,-1,0.1)
# q  <- sigmoid10(logodds)
# grid = data.frame(sigma_e = 1,
#                   sigma_b = 1,
#                   pi = q)
# write.table(grid, file.path(args[[2]], "hyperpriors_grid.txt"), col.names=T, row.names=F, quote=F)
# 
# probs = data.frame(probs = rep(1, nrow(grid)))
# write.table(probs, file.path(args[[2]], "hyperpriors_probs.txt"), col.names=T, row.names=F, quote=F)
# 
# write.table(head(grid, 1), file.path(args[[2]], "hyperpriors_grid_sub.txt"), col.names=T, row.names=F, quote=F)
# write.table(head(probs, 1), file.path(args[[2]], "hyperpriors_probs_sub.txt"), col.names=T, row.names=F, quote=F)
# 
# # Save sensible start point
# alpha.init = fit.v$alpha[,20]
# mu.init = fit.v$mu[,20]
# write.table(alpha.init, file.path(args[[2]], "alpha_init.txt"), col.names=T, row.names=F, quote=F)
# write.table(mu.init, file.path(args[[2]], "mu_init.txt"), col.names=T, row.names=F, quote=F)

# # Save varbvs_answer.out
# ans = data.frame(post_alpha = c(fit.v$alpha %*% fit.v$w),
#                  post_mu = c(fit.v$mu %*% fit.v$w),
#                  post_beta = c((fit.v$alpha * fit.v$mu) %*% fit.v$w))
# write.table(ans, file.path(args[[2]], "varbvs_answer.out"), col.names=T, row.names=F, quote=F)
# write.table(ans, file.path(args[[2]], "answer.out"), col.names=T, row.names=F, quote=F)
