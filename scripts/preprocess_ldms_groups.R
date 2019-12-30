# Script to append the SNPID (from bgenix -list) to LDscore and MAF fields (from GCTA)
# 
# Arguments:
# - bgenix_path : path to bgenix files. Will parse regex.
# - ldscore_path : path to ldscore files. Will parse regex.
# - out : path to save results to.
require("R.utils")
require("data.table")
args = R.utils::cmdArgs(names = c("keys", "ldscore", "out"))
print(args)

freadSafe = function(path, quiet = F, select = NULL, nrows=Inf){
  if(!quiet){
    print(paste0("Reading in ", path))
  }
  if(grepl(".gz$", path)){
    aa = data.table::fread(cmd=paste0("gunzip -c ", path), data.table = F, select=select, nrows=nrows)
  } else {
    aa = data.table::fread(path, data.table = F, select=select, nrows=nrows)
  }
  return(aa)
}

readBGENIX = function(path){
  daf = freadSafe(path, select = c("rsid", "alternate_ids", "chromosome", "position"))
  colnames(daf) = c("rsid", "SNPID", "chr", "pos")
  return(daf)
}

readLDSCORE = function(path){
  daf = freadSafe(ldscore_file, select = c("SNP", "chr", "bp", "MAF", "ldscore"))
  colnames(daf) = c("rsid", "chr", "pos", "MAF", "ldscore")
  return(daf)
}

bgenix_files = list.files(dirname(args$bgenix_path), full.names = T, pattern = basename(args$bgenix_path))
ldscore_files = list.files(dirname(args$ldscore_path), full.names = T, pattern = basename(args$ldscore_path))
print("Detected Bgenix files:")
print(bgenix_files)

print("Detected ldscore files:")
print(ldscore_files)

df.bgenix = do.call(rbind, lapply(bgenix_files, readBGENIX))
daf       = do.call(rbind, lapply(ldscore_files, readLDSCORE))

# Locate the bgenix SNPID of each SNP
map             = match(daf$rsid, df.bgenix$rsid)
map[is.na(map)] = match(daf$rsid, df.bgenix$alternate_ids)[is.na(map)]
map[is.na(map)] = match(daf$key, df.bgenix$key)[is.na(map)]

if(any(is.na(map))){
  print(paste0("Unable to locate SNPID of ", sum(is.na(map)), " SNPs"))
}

daf$SNPID = df.bgenix$SNPID[map]

# Assign groups
breaks = quantile(daf$ldscore, c(0, 0.25, 0.5, 0.75, 1), na.rm=T)

LDgroups = cut(daf$ldscore, breaks = breaks, labels = paste0("LDSCORE", 1:4), include.lowest = T)
MAFgroups = cut(daf$MAF, breaks = c(0.001, 0.1, 0.2, 0.3, 0.4, 0.5), labels = paste0("MAF", 1:5),
                include.lowest = T)

daf$group = paste(MAFgroups, LDgroups, sep = "_")

daf = daf[complete.cases(LDgroups) & complete.cases(MAFgroups), c("SNPID", "group")]
print(nrow(daf))

print(paste0("Saving output to ", args$out))
write.table(daf, args$out, col.names=T, row.names=F, quote=F)

