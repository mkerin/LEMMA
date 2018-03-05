# Script to check inference of varbvs package with bgen_prog
# 
# args:
# 1 - path to inference
# 2 - path to save target
library(ggplot2)
library(ggrepel)

if(T){
    args = R.utils::cmdArgs()
} else {
    args = list("data/io_test/t7_varbvs/attempt.out",
                "data/io_test/t7_varbvs/plots/pip_attempt.out")
}
print(args)

my.inf = read.table(args[[1]], header = T)
true_beta = read.table(file.path(dirname(args[[1]]), "pheno_true_beta.txt"), header = T)[,1]
true_pip = ifelse(true_beta == 0, 0, 1)

df.plot = data.frame(pip = my.inf$post_alpha,
                     true_pip = as.factor(true_pip),
                     row_ind = seq_along(true_pip))
ggplot(df.plot, aes(x=row_ind, pip)) +
    geom_point(aes(colour = true_pip))
ggsave(args[[2]])



