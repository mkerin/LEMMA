# script of misc functions
# 
# sigmoid10
# logit10
# calc_r2
# calc_ldmap
# plotAnnotPIP
library(ggrepel)

## Functions
sigmoid10 <- function(x){
    1/(1 + 10^(-x))
}

sigmoid <- function(x){
    1/(1 + exp(-x))
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

plotAnnotPIP = function(my.pip, my.true_pip, LD.m, pip.thresh = 0.5, win.size = 10, P = nrow(X), interaction.analysis = T){
    # Return ggplot of Posterior Inclusion Probability, with annotations between
    # variants with high PIP in LD with truely included variants.
    # my.pip; vector of PIP
    # my.true_pip; vector
    # LD.m; melted dataframe of pairwise ld between SNPs
    
    df.plot = data.frame(pip = my.pip,
                         true_pip = as.factor(my.true_pip),
                         row_ind = seq_along(my.pip))
    
    # Create basic plot
    fd2 = subset(df.plot, true_pip == 1)
    df.plot = subset(df.plot, true_pip == 0)
    p1 = ggplot(df.plot, aes(x=row_ind, pip)) +
        geom_point(aes(colour = true_pip)) +
        geom_point(aes(colour = true_pip), data = fd2)  # try to ensure true values overlay noise
    
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
    
    # If SNPs in LD with identified snps exist then add to plot
    if(!is.null(df.labels) && nrow(df.labels) > 0){
        df.labels$r2 = df.labels$value
        df.labels$value=NULL
        df.labels$lbl = paste0("r2=", round(df.labels$r2, 2), " with ",
                               ifelse(df.labels$Var1 %in% pip.snps, df.labels$Var1, df.labels$Var2))
        aa = reshape2::melt(df.labels, measure.vars = c("Var1", "Var2"))
        aa$pip = df.plot$pip[match(aa$value, df.plot$row_ind)]
        aa1 = subset(aa, !value %in% pip.snps)
        aa1 = subset(aa1, r2 > 0.2)  # Only label true SNPs with r2 > 0.2
        
        aa2 = subset(aa, value %in% pip.snps)
        aa2$lbl = paste0("Var ", aa2$value)
        aa2$r2=NULL
        aa2 = aa2[!duplicated(aa2),]
        
        # Add labels
        p1 = p1 + geom_label_repel(aes(x=value, y=pip, label=lbl), box.padding = 2.0, data = aa1) +
            geom_label_repel(aes(x=value, y=pip, label=lbl), box.padding = 1.0, data = aa2)
    }
    
    if(interaction.analysis){
        p1 = p1 + geom_vline(xintercept=P/2, linetype="dotted")
    }
    return(p1)
}

readDosage = function(path.to.vcf){
    vcf <- VariantAnnotation::readVcf(path.to.vcf, "hg19")
    geno_vcf = VariantAnnotation::geno(vcf)$GP
    X = sapply(geno_vcf, function(vec){vec[2] + 2*vec[3]})
    X = matrix(X, nrow = nrow(geno_vcf), ncol = ncol(geno_vcf))
    row.names(X) = row.names(geno_vcf)
    colnames(X) = colnames(geno_vcf)
    X = t(X)
    return(X)
}

readStandardisedDosage = function(path.to.vcf){
    X = readDosage(path.to.vcf)
    X = scale(X, center = T, scale = T)
    if(any(is.na(X))){
        if(sum(is.na(X)) %% nrow(X) == 0){
            X[,which(is.na(colSums(X)))] = 0
        } else {
            stop("Unexplained NAs in genotypes")
        }
    }
    return(X)
}

# ----------------------------------------------------------------------
# normalizelogweights takes as input an array of unnormalized
# log-probabilities logw and returns normalized probabilities such
# that the sum is equal to 1.
normalizelogweights <- function (logw) {
    
    # Guard against underflow or overflow by adjusting the
    # log-probabilities so that the largest probability is 1.
    c <- max(logw)
    w <- exp(logw - c)
    
    # Normalize the probabilities.
    return(w/sum(w))
}

updateAlphaMu = function(updates, alpha, mu, Hr){
    # Calling on global values of hyps, Hr, X, Z
    for (jj in updates) {
        
        # Compute the variational estimate of the posterior variance.
        if(jj <= n_var){
            s <- sigma_b * sigma/(sigma_b * dHtH[jj] + 1)
        } else {
            s <- sigma_g * sigma/(sigma_g * dHtH[jj] + 1)
        }
        
        # Update the variational estimate of the posterior mean.
        r     <- alpha[jj] * mu[jj]
        mu[jj] <- s/sigma * (Hty[jj] + dHtH[jj]*r - sum(H[,jj]*Hr))
        
        # Update the variational estimate of the posterior inclusion
        # probability.
        if(jj <= n_var){
            alpha[jj] <- sigmoid(log(lam_b / (1 - lam_b) + eps) + log(s/(sigma_b*sigma)) + mu[jj]^2/s/2)
        } else {
            alpha[jj] <- sigmoid(log(lam_g / (1 - lam_g) + eps) + log(s/(sigma_g*sigma)) + mu[jj]^2/s/2)
        }
        
        # Update Xr or Zr.
        Hr <- Hr + (alpha[jj]*mu[jj] - r) * H[,jj]
    }
    
    return(list(alpha = alpha,
                mu = mu,
                Hr = Hr))
}

calc_logw = function(alpha, mu, s_sq, Hr){
    # From global namespace; dHtH, Hr, interaction.analysis
    N = nrow(X)
    P = ncol(X)
    mu_sq = mu * mu
    varB = alpha * (s_sq + mu_sq) - (alpha * mu)^2
    
    # linear
    logw = -N * log(2*pi * sigma) / 2
    logw = logw - t(y - Hr) %*% (y - Hr) / 2 / sigma
    logw = logw - sum(dHtH * varB) / 2 / sigma 
    
    # gamma
    int.gamma = sum(alpha[b.index]*log(lam_b + eps)) + 
        sum((1 - alpha[b.index])*log((1 - lam_b) + eps))
    if(interaction.analysis){
        int.gamma = int.gamma + sum(alpha[g.index]*log(lam_g + eps)) + 
            sum((1 - alpha[g.index])*log((1 - lam_g) + eps))
    }
    logw = logw + int.gamma
    
    # kl.beta
    kl.beta = sum(alpha[b.index] / 2 * (1 + log(s_sq[b.index] / sigma_b / sigma) -
                                                (s_sq[b.index] + mu_sq[b.index]) / sigma_b / sigma))
    kl.beta = kl.beta - sum(alpha[b.index]*log(alpha[b.index] + eps))
    kl.beta = kl.beta - sum((1 - alpha[b.index])*log((1 - alpha[b.index]) + eps))
    
    if(interaction.analysis){
        kl.beta = kl.beta + sum(alpha[g.index] / 2 * (1 + log(s_sq[g.index] / sigma_g / sigma) -
                                                (s_sq[g.index] + mu_sq[g.index]) / sigma_g / sigma))
        kl.beta = kl.beta - sum(alpha[g.index]*log(alpha[g.index] + eps))
        kl.beta = kl.beta - sum((1 - alpha[g.index])*log((1 - alpha[g.index]) + eps))
    }
    
    logw = logw + kl.beta
    # browser()
    
    return(logw)
}


runOuterLoop = function(alpha.init, mu.init){
    # From global namespace; hyps, dHtH, b.index, g.index, diff.tol, logw.tol, ii
    
    # Assign initial values
    alpha = alpha.init
    mu = mu.init
    
    # generate s_sq
    if(interaction.analysis){
        s_sq = c(sigma_b * sigma / (sigma_b * dHtH[b.index] + 1), sigma_g * sigma / (sigma_g * dHtH[g.index] + 1))
    } else {
        s_sq = sigma_b * sigma / (sigma_b * dHtH + 1)
    }
    
    # Run outer loop
    converged = FALSE
    count = 0
    logw.ii.vec = c()
    logw_i = -.Machine$double.xmax
    while(!converged && count < iter.max){
        alpha.prev = alpha
        mu.prev = mu
        logw.prev = logw_i

        if(count %% 2 == 0){
            updates = 1:n_var2
        } else {
            updates = seq(n_var2, 1, -1)
        }
        
        # Track elbo updates starting from init
        logw_i = calc_logw(alpha, mu, s_sq, Hr)
        logw.ii.vec = c(logw.ii.vec, logw_i)
        
        res = updateAlphaMu(updates, alpha, mu, Hr)
        alpha = res$alpha
        mu = res$mu
        Hr = res$Hr
        
        count = count + 1
        
        # Diagnose convergence
        if(max(abs(alpha - alpha.prev)) < diff.tol && (logw_i - logw.prev) < logw.tol){
            converged = TRUE
        }
    }
    
    logw_i = calc_logw(alpha, mu, s_sq, Hr)
    logw.ii.vec = c(logw.ii.vec, logw_i)
    if(!is.finite(logw_i)){
        print(paste0("WARNING: non-finite elbo produced at grid point: ", ii))
    }
    
    # Values to return
    out = list(logw.ii = logw_i,
               logw.ii.vec = logw.ii.vec,
               alpha = alpha,
               mu = mu,
               count = count)
}


PlotMarg = function(hyp, daf, true.val = NULL){
    # Returns ggplot of marginal posterior distribution for the given hyperparameter
    # 
    # :str: hyp
    #       in {sigma, sigma_b, sigma_g, lam_b, lam_g}
    # :data.frame: daf
    #       cbind(grid, weights)
    require(ggplot2)
    require(grid)
    require(gridExtra)
    
    if(!(all(c(hyp, "weights") %in% colnames(daf)))){
        print(colnames(daf))
        stop("Not (all(c(hyp, \"weights\") %in% colnames(daf)))")
    }
    
    expected.hyp = sum(daf$weights * daf[,hyp])
    
    vals = unique(daf[,hyp])
    df.plot = data.frame(vals = vals,
                         marg_prob = NA,
                         pi = NA)
    for (ii in seq_along(vals)){
        val = vals[ii]
        df.plot$marg_prob[ii] = sum(daf[which(daf[,hyp] == val), "weights"])
    }
    
    p1 = ggplot(df.plot, aes(x=vals, y = marg_prob)) +
        geom_point() +
        labs(x = paste0(hyp),
             y = "Marginal posterior probability") + 
        geom_vline(xintercept = expected.hyp, colour = "blue", linetype = "dashed")
    
    if(!is.null(true.val)){
        p1 = p1 + geom_vline(xintercept=true.val, colour = "red")
            
    }
    return(p1)
}

dlogit_dx = function(x){
    1 / (x * (1 - x))
}

f_y = function(y, a, b){
    # g = sigmoid10
    g_inv = logit10
    dg_inv_dx = dlogit_dx
    f_x = dnorm
    abs(dg_inv_dx(y)) * f_x(g_inv(y), a, b)
}


readCppOutput = function(inference_path){
    # Amalgamate all of the files dumped by cpp code into single R object
    # (similar to output from R implementation)
    # WARNING: hyps grid filepath hardcoded!!
    
    # Formatting filepaths
    prefix          = tools::file_path_sans_ext(inference_path)
    suffix          = tools::file_ext(inference_path)
    path.weights    = paste0(prefix, "_hyps.", suffix)
    path.elbo       = paste0(prefix, "_elbo.", suffix)
    path.grid.hyps  = paste0(dirname(inference_path), "/hyperpriors_gxage_v1.txt")
    path.grid.probs = paste0(dirname(inference_path), "/hyperpriors_gxage_v1_probs.txt")
    path.mus        = paste0(prefix, "_mus.", suffix)
    path.alphas     = paste0(prefix, "_alphas.", suffix)

    # Read data from file
    df.inf          = read.table(inference_path, header = T)
    df.weights      = read.table(path.weights, header = T)
    df.alphas       = read.table(path.alphas, header = F)
    df.mus          = read.table(path.mus, header = F)
    
    if(file.exists(path.grid.hyps) & file.exists(path.grid.probs)){
        df.grid     = read.table(path.grid.hyps, header = T)
        grid.probs  = read.table(path.grid.probs, header = T)[,1]
    } else {
        df.grid     = NA
        grid.probs  = NA
    }
    
    # Need to get a little more creative to get the elbo updates
    df.elbo = read.table(path.elbo, header = F, sep = "\1", stringsAsFactors = F)
    logw.updates.list = lapply(1:nrow(df.elbo), function(ii, daf){
            as.numeric(unlist(strsplit(as.character(daf[ii,1]), " ")))
        }, df.elbo)
    
    res = list(weights           = df.weights$weights,
               logw.vec          = df.weights$logw,
               mu.post           = df.inf$post_mu,
               beta.post         = df.inf$post_beta,
               alpha.post        = df.inf$post_alpha,
               alpha.list        = as.list(df.alphas),
               mu.list           = as.list(df.mus),
               counts.vec        = df.weights$count,
               logw.updates.list = logw.updates.list,
               grid.hyps         = df.grid,
               grid.probs        = grid.probs)
    return(res)
}

readROutput = function(inference_path){
    # Streamline RDS object saved by R implementation
    
    res = readRDS(inference_path)
    if(!("counts.list" %in% names(res))){
        res$counts.vec = unlist(res$counts.list)
        res$counts.list = NULL
    }
    
    if(!("logw.vec" %in% names(res))){
        res$logw.vec = unlist(res$logw.list)
        res$logw.list = NULL
    }
    
    res$mu.post = as.vector(res$mu.post)
    res$weights = as.vector(res$weights)
    res$beta.post = as.vector(res$beta.post)
    return(res)
}


