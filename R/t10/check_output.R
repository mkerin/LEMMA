# Script to judge if output from two vbayes implementations is identical
# 
# Currently hardcoded to io_test/t7*
# 
# 
source('R/misc.R')

args = list("data/io_test/t10_varbvs_without_init")
args = R.utils::cmdArgs()
print(args)

test.dir = args[[1]]

r.res = readROutput(file.path(test.dir, "answer.rds"))
cpp.res = readCppOutput(file.path(test.dir, "attempt.out"))


df.logw = as.data.frame(cbind(cpp.res$logw.vec, r.res$logw.vec))
colnames(df.logw) = c("cpp", "r")
print(df.logw)

if (sum((df.logw$r - df.logw$cpp)^2) < 0.1){
    print("PASS")
} else {
    print("FAIL")
}