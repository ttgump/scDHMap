rm(list =ls())
library(ggplot2)
library(ggforce)
library(rhdf5)
library(plyr)
library(RColorBrewer)
library(smoother)

## Divide the latent embedding of scDHMap on Paul cells by Poincare pseudotime
## The branching method was inspired by Haghverdi et al. (2016), Diffusion pseudotime robustly reconstructs branching cellular lineages, Nature Methods

kendall_finite_cor <- function(b1, b2, new1, new2) {
  b11 <- numeric(length(b1))
  b11[b1 >= new1] <- 1
  b11[b1 <  new1] <- -1
  
  b22 <- numeric(length(b2))
  b22[b2 >= new2] <- 1
  b22[b2 <  new2] <- -1
  
  b11 %*% b22
}

branchcut <- function(pt, bid, b) {
  n <- nrow(bid)
  all_branches <- seq_len(3L)
  
  # sanity checks
  stopifnot(b %in% all_branches)
  stopifnot(ncol(pt) == 3L, ncol(bid) == 3L)
  stopifnot(nrow(pt) == n)
#  stopifnot(is.double(pt), is.integer(bid))
  
  # find cell indexes per branch 
  other <- all_branches[all_branches != b]
  b1 <- other[[1L]]
  b2 <- other[[2L]]
  
  # PT for other branches, sorted by b3
  b3_idxs <- bid[, b]
  pt1 <- pt[b3_idxs, b1]
  pt2 <- pt[b3_idxs, b2]
  
  kcor <- vapply(seq_len(n - 1L), function(s1) {
    s2 <- s1 + 1L
    l <- seq_len(s1)
    r <- seq(s2, n)
    
    k_l <- kendall_finite_cor(pt1[l], pt2[l], pt1[[s2]], pt2[[s2]])
    k_r <- kendall_finite_cor(pt1[r], pt2[r], pt1[[s1]], pt2[[s1]])
    
    k_l/s1 - k_r/(n - s1)
  }, double(1))
  
  kcor <- smth.gaussian(kcor, 5L)
  cut <- which.max(kcor)
  
  b3_idxs[seq_len(cut)]
}


find_tips <- function(pt, root) {
  x <- root
  dx <- pt[x,]
  y <- which.max(dx)
  dy <- pt[y,]
  z <- which.max(dx + dy)
  
  c(x, y, z)
}



scDHMap.latent <- read.table("transformed_final_latent.txt", sep=" ")
scDHMap.pt <- as.numeric(readLines("transformed_Poincare_pseudotime.txt"))
paul.cell_type <- readLines("Paul_celltypes.txt")


scDHMap.latent.dist <- read.table("pairwise_distances.txt", sep=" ")
iroot <- 841

tips <- find_tips(scDHMap.latent.dist, iroot)
tips_pt <- scDHMap.latent.dist[, tips]
bid <- apply(tips_pt, 2, function(z) order(z, decreasing = F))

branch_x <- branchcut(tips_pt, bid, 1)
branch_y <- branchcut(tips_pt, bid, 2)
branch_z <- branchcut(tips_pt, bid, 3)

brances <- rep("unassigned", nrow(scDHMap.latent))
brances[branch_x] <- "1"
brances[branch_y] <- "2"
brances[branch_z] <- "3"
brances <- factor(brances, levels=c("unassigned", "1", "2", "3"))

dat <- data.frame(scDHMap.latent, `Poincaré pseudotime`=scDHMap.pt, Branch=brances, 
                  isTips=F, `Cell type`=paul.cell_type, check.names=F)
dat$isTips[tips] <- T

write.csv(dat, "branch_result.csv", row.names=F)

ggplot(dat, aes(x=V1, y=V2, color=Branch)) + geom_point(size=.5) +
  annotate("path", x=0+1*cos(seq(0,2*pi,length.out=100)), y=0+1*sin(seq(0,2*pi,length.out=100))) +
  theme_classic() + theme(axis.title=element_blank(), axis.ticks=element_blank(), axis.text=element_blank(), axis.line=element_blank(),
                          legend.position="right")

