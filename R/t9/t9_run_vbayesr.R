# Script to run varbvs and write "correct" output to test/
# 
# 
# 
source('/well/marchini/kebl4230/software/VBayesR/R/VBayesR_functions.R')

if(T){
    args = R.utils::cmdArgs()
} else {
    args = list("data/io_test/t9_varbvs_zero_hg")
}
print(args)

## constants
data.dir   = args[[1]]
test.dir   = args[[1]]


## Data
y          = read.table(file.path(data.dir, "pheno.txt"), header = T)[,1]
age        = read.table(file.path(data.dir, "age.txt"), header = T)[,1]
X          = readStandardisedDosage(file.path(data.dir, "n50_p100.vcf.gz"))
df.init    = read.table(file.path(data.dir, "answer_init.txt"), header = T)
df.grid    = read.table(file.path(data.dir, "hyperpriors_gxage.txt"), header = T)
df.probs   = read.table(file.path(data.dir, "hyperpriors_gxage_probs.txt"), header = T)


## Wrangling
y          = as.vector(scale(y, center = T, scale = F))
age        = as.vector(scale(age, center = T, scale = T))
df.fit     = data.frame(y = y, age = age)
fit        = lm(y ~ 0 + age, data = df.fit)
y          = as.vector(fit$residuals)
Z          = apply(X, 2, function(col){col * age})

if((!all(df.grid$sigma_b == df.grid$sigma_g)) || !all(df.grid$lambda_b == df.grid$lambda_g)){
    print("WARNING: running different priors for interaction and main effects")
}

cat('
    ############################################################
    ### Expected that h_g == 0, hence run only on X ############
    ############################################################
    ')
pp = ncol(X)
stopifnot(pp %% 2 == 0)
Z = X[,seq(pp/2 + 1,pp)]
X = X[,seq(1, pp/2)]
######

H = cbind(X, Z)
interaction.analysis = T


#### Run VBayesR


# Constats
iter.max        = 100
eps             = .Machine$double.xmin
diff.tol        = 1e-4
n_var           = ncol(X)
n_var2          = 2 * ncol(X)


# Useful quantities 
ns                 = nrow(df.grid)
outer.report       = max(1, floor(ns / 10))
b.index            = seq(1, n_var)
g.index            = seq(n_var+1, n_var2)
dHtH               = c(diag(t(X) %*% X), diag(t(Z) %*% Z))
Hty                = c(t(X) %*% y, t(Z) %*% y)

###
alpha.init         = df.init$alpha[seq_len(pp)]
mu.init            = df.init$mu[seq_len(pp)]
random.params.init = FALSE
Hr                 = X %*% (alpha.init[b.index] * mu.init[b.index]) + Z %*% (alpha.init[g.index] * mu.init[g.index])
###

# Things to track
counts.list       = list()                 # Number of iterations to convergence at each step
logw.updates.list = list()                 # elbo updates at each ii
mu.list           = list()                 # best mu at each ii
alpha.list        = list()                 # best alpha at each ii
logw.list         = list()

start = Sys.time()
for (ii in seq_len(ns)){
    if(ii %% outer.report == 0){
        print(paste0("Round 2: ", ii,  " / " , ns))
        print(Sys.time() - start)
        start = Sys.time()
    }
    
    # Unpack hyps
    sigma   = df.grid[ii, 1]
    sigma_b = df.grid[ii, 2]
    sigma_g = df.grid[ii, 3]
    lam_b   = df.grid[ii, 4]
    lam_g   = df.grid[ii, 5]
    
    ###
    stopifnot(sigma_g < 1e-9)
    sigma_g = sigma_b
    lam_g = lam_b
    ###
    
    out = runOuterLoop(alpha.init, mu.init)
    
    if(is.finite(out$logw.ii)){
        logw.list[[ii]]         = out$logw.ii
        logw.updates.list[[ii]] = out$logw.ii.vec
        alpha.list[[ii]]        = out$alpha
        mu.list[[ii]]           = out$mu
        counts.list[[ii]]       = out$count
    }
}


# Compute normalised weights
logweights = unlist(logw.list) + log(df.probs$prob)
weights = normalizelogweights(logweights)


# Compute posterior alpha, mu, beta
alphas     = do.call(rbind, alpha.list)
mus        = do.call(rbind, mu.list)

alpha.post = t(alphas) %*% weights
mu.post    = t(mus) %*% weights
beta.post  = t(mus * alphas) %*% weights


# Save output to file
res = list(weights           = weights,
           mu.post           = mu.post,
           beta.post         = beta.post,
           alpha.post        = alpha.post,
           alpha.list        = alpha.list,
           mu.list           = mu.list,
           counts.list       = counts.list,
           logw.updates.list = logw.updates.list,
           grid.hyps         = df.grid,
           grid.probs        = df.probs,
           logw.list         = logw.list,
           N                 = nrow(X))

save.path = file.path(test.dir, "answer.rds")
print(paste0("Saving output to: ", save.path))
saveRDS(res, save.path)


