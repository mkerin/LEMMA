# script of misc functions
# 
# sigmoid10
# logit10
# calc_r2
# calc_ldmap
# plotAnnotPIP

## Functions
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
