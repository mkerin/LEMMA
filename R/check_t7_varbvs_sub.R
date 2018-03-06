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
grid = read.table(file.path(args[[2]], "hyperpriors_gridtxt"), header=T)
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
fit <- varbvs(ds,NULL,y,family = "gaussian",
                  logodds = log10(grid$pi), sa = 1, sigma = 1, 
                  alpha = as.matrix(alpha.init), 
                  mu = as.matrix(mu.init))


# Take a look at how accurate the PIP are
true_beta = read.table(file.path(args[[2]], "pheno_true_beta.txt"), header = T)[,1]
true_pip = ifelse(true_beta == 0, 0, 1)
LD.m = calc_ldmap(ds)
p = plotAnnotPIP(fit$pip, true_pip, LD.m)
ggsave(file.path(plt.dir, "pip_answer.pdf"), p)

# Write posterior inf file
ans = data.frame(post_alpha = c(fit$alpha %*% fit$w),
                 post_mu = c(fit$mu %*% fit$w),
                 post_beta = c((fit$alpha * fit$mu) %*% fit$w))
write.table(ans, file.path(args[[2]], "answer.out"), col.names=T, row.names=F, quote=F)

# Write posterior hyps file
ans = data.frame(weights = c(fit$w),
                 logw = c(fit$logw))
write.table(ans, file.path(args[[2]], "answer_hyps.out"), col.names=T, row.names=F, quote=F)



# Extract logodds
logodds = seq(-3.5,-1,0.1)
q  <- sigmoid10(logodds)

grid = data.frame(sigma_e = 1,
                  sigma_b = 1,
                  pi = q)

# Get the normalized importance weights.
w <- fit$w
ggplot(data.frame(x = log10q,y = w), aes(x, y)) + geom_point()


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
# alpha.init = fit$alpha[,20]
# mu.init = fit$mu[,20]
# write.table(alpha.init, file.path(args[[2]], "alpha_init.txt"), col.names=T, row.names=F, quote=F)
# write.table(mu.init, file.path(args[[2]], "mu_init.txt"), col.names=T, row.names=F, quote=F)

# # Save varbvs_answer.out
# ans = data.frame(post_alpha = c(fit$alpha %*% fit$w),
#                  post_mu = c(fit$mu %*% fit$w),
#                  post_beta = c((fit$alpha * fit$mu) %*% fit$w))
# write.table(ans, file.path(args[[2]], "varbvs_answer.out"), col.names=T, row.names=F, quote=F)
# write.table(ans, file.path(args[[2]], "answer.out"), col.names=T, row.names=F, quote=F)
