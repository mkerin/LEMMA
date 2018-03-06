# Script to check inference of varbvs package with bgen_prog
# 
# 
# 
# 
library(VariantAnnotation, quietly = T)
library(varbvs)
library(ggplot2)
library(ggrepel)

args = list("data/io_test/t7_varbvs/n500_p1000.vcf.gz",
            "data/io_test/t7_varbvs/")

vcf <- readVcf(args[[1]], "hg19")
y = as.matrix(read.table(file.path(args[[2]], "pheno.txt"), header=T))
grid = read.table(file.path(args[[2]], "hyperpriors_grid.txt"), header=T)
grid.sub = read.table(file.path(args[[2]], "hyperpriors_grid_sub.txt"), header=T)

alpha.init = read.table(file.path(args[[2]], "alpha_init.txt"), header=T)
mu.init = read.table(file.path(args[[2]], "mu_init.txt"), header=T)

# Functions
sigmoid10 <- function(x){
    1/(1 + 10^(-x))
}
logit10 = function(x){
    log10(x / (1 - x))
}

calc_r2 = function(x, y){
    # calculate r2 statistic between dosage vectors x and y
    # p1 = sum(ceiling(x - 0.5) == 1)
    # p2 = sum(ceiling(x - 0.5) == 2)
    # q1 = sum(ceiling(x - 0.5) == 1)
    # q2 = sum(ceiling(y - 0.5) == 2)
    # 
    # P11 = sum(ceiling(x - 0.5) == 1 && ceiling(y - 0.5) == 1)
    # P12 = sum(ceiling(x - 0.5) == 1 && ceiling(y - 0.5) == 2)
    # P21 = sum(ceiling(x - 0.5) == 2 && ceiling(y - 0.5) == 1)
    # P22 = sum(ceiling(x - 0.5) == 2 && ceiling(y - 0.5) == 2)
    # 
    # D = P11 * P22 - P12 * P21
    # r2 = D^2 / p1 / p2 / q1 / q2
    return(cor(x, y)^2)
}

calc_ldmap = function(X, width = 10){
    # X - N x P dosage matrix
    # width - window for pairwise comparisons
    P = ncol(X)
    heatmat = matrix(NA, P, P)
    for (ii in seq_len(P-1)){
        if(ii == P-1) next
        
        for (jj in seq(ii+1, min(ii+1+width, P))){
            heatmat[ii,jj] = calc_r2(X[,ii], X[,jj])
        }
    }
    
    heatmat.m = reshape2::melt(heatmat)
    heatmat.m = subset(heatmat.m, !is.na(value))
    return(heatmat.m)
}

plotAnnotPIP = function(my.pip, my.true_pip, LD.m, pip.thresh = 0.5, win.size = 10){
    # Return ggplot of Posterior Inclusion Probability, with annotations between
    # variants with high PIP in LD with truely included variants.
    # my.pip; vector of PIP
    # my.true_pip; vector
    # LD.m; melted dataframe of pairwise ld between SNPs
    P = ncol(ds)
    
    df.plot = data.frame(pip = my.pip,
                         true_pip = as.factor(my.true_pip),
                         row_ind = seq_along(my.pip))
    
    # Check for LD between identified SNPs and true SNPs
    pip.snps = which(df.plot$pip > pip.thresh)
    df.labels = list()
    for (snp in pip.snps){
        win.index = seq(max(0, snp - win.size), min(snp + win.size, P))
        window = df.plot$true_pip[win.index]
        win.true_pips = win.index[which(window == 1)]
        
        win.ld = rbind(subset(LD.m, Var1 %in% win.true_pips & Var2 == snp),
                       subset(LD.m, Var2 %in% win.true_pips & Var1 == snp))
        win.ld = win.ld[order(win.ld$value, decreasing = T),]
        df.labels[[snp]] = head(win.ld)
    }
    df.labels = do.call(rbind, df.labels)
    
    # Reformat into labels for ggrepel
    df.labels$r2 = df.labels$value
    df.labels$value=NULL
    df.labels$lbl = paste0("r2=", round(df.labels$r2, 2), " with ",
                           ifelse(df.labels$Var1 %in% pip.snps, df.labels$Var1, df.labels$Var2))
    aa = reshape2::melt(df.labels, measure.vars = c("Var1", "Var2"))
    aa$pip = df.plot$pip[match(aa$value, df.plot$row_ind)]
    aa1 = subset(aa, !value %in% pip.snps)
    
    aa2 = subset(aa, value %in% pip.snps)
    aa2$lbl = paste0("Var ", aa2$value)
    aa2$r2=NULL
    aa2 = aa2[!duplicated(aa2),]
    
    # create plot
    p1 = ggplot(df.plot, aes(x=row_ind, pip)) +
        geom_point(aes(colour = true_pip)) +
        geom_label_repel(aes(x=value, y=pip, label=lbl), box.padding = 2.0, data = aa1) +
        geom_label_repel(aes(x=value, y=pip, label=lbl), box.padding = 1.0, data = aa2)
    return(p1)
}


# matrix of dosages
ds = sapply(geno(vcf)$GP, function(vec){vec[2] + 2*vec[3]})
ds = matrix(ds, nrow = nrow(geno(vcf)$GP), ncol = ncol(geno(vcf)$GP))
row.names(ds) = row.names(geno(vcf)$GP)
colnames(ds) = colnames(geno(vcf)$GP)
ds = t(ds)

# Run varbvs
fit.sub <- varbvs(ds,NULL,y,family = "gaussian",
                  logodds = log10(grid.sub$pi), sa = 1, sigma = 1, 
                  alpha = as.matrix(alpha.init), 
                  mu = as.matrix(mu.init))

fit <- varbvs(ds,NULL,y,family = "gaussian",
              logodds = log10(grid$pi), sa = 1, sigma = 1)


# Take a look at how accurate the PIP are
true_beta = read.table(file.path(args[[2]], "pheno_true_beta.txt"), header = T)[,1]
true_pip = ifelse(true_beta == 0, 0, 1)
LD.m = calc_ldmap(ds)
p = plotAnnotPIP(fit$pip, true_pip, LD.m)
ggsave(file.path(args[[2]], "plots", "pip_answer.pdf"), p)





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
