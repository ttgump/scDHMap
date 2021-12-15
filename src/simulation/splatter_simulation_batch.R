rm(list = ls())
library(rhdf5)
library(dyntoy)
library(splatter)
library(scater)
library(dplyr)


all.dropout.rate <- c()
for(i in 1:10) {
  platform_simple <- function(
    n_cells = 4000L,
    n_features = 3000L,
    n_batches = 6L,
    pct_main_features = 0.5,
    dropout_mid = 2.5,
    dropout_shape = -1,
    de_prob = 0.2,
    de_facScale = 0.3,
    seed=99+i
  ) {
    list(
      platform_id = "simple",
      estimate = splatter::newSplatParams(nGenes = n_features, batchCells = c(0.05, 0.1, 0.15, 0.2, 0.25, 0.25)*n_cells,
                                          dropout.type = "experiment", dropout.mid = dropout_mid, dropout.shape = dropout_shape, 
                                          de.prob = de_prob, de.facLoc = de_facScale, seed=seed),
      num_cells = n_cells,
      num_features = n_features,
      pct_main_features = pct_main_features
    )
  }
  
  platform = platform_simple()
  splatter_params <- platform$estimate
  
  # sample parameters
  set.seed(100)
  n_steps_per_length = 100
  path.skew = runif(1, 0, 1)
  path.nonlinearProb = runif(1, 0, 1)
  path.sigmaFac = runif(1, 0, 1)
  bcv.common.factor = runif(1, 10, 200)
  
  # extract path from milestone network
  milestone_network <- dyntoy::generate_milestone_network("tree")
  
  root <- setdiff(milestone_network$from, milestone_network$to)
  path.to <- c(root, milestone_network$to)
  path.from <- as.numeric(factor(milestone_network$from, levels = path.to)) - 1
  
  # factor added to bcv.common, influences how strong the biological effect is
  splatter_params@bcv.common <- splatter_params@bcv.common / bcv.common.factor
  
  sim <- splatter::splatSimulatePaths(
    splatter_params,
    group.prob = milestone_network$length/sum(milestone_network$length),
    path.from = path.from,
    path.length = ceiling(milestone_network$length*n_steps_per_length),
    path.nonlinearProb = path.nonlinearProb,
    path.sigmaFac = path.sigmaFac,
    path.skew = path.skew
  )
  
  counts <- sim@assays@data$counts
  truecounts <- sim@assays@data$TrueCounts
  batch <- sim$Batch
  batch <- as.numeric(factor(batch))
  batch <- model.matrix(~ -1 + factor(batch))
  
  dropout.rate <- (sum(counts==0)-sum(truecounts==0))/sum(truecounts>0)
  all.dropout.rate <- c(all.dropout.rate, dropout.rate)
  
  # gold standard trajectory
  progressions <- milestone_network %>%
    dplyr::slice(as.numeric(gsub("Path([0-9]*)", "\\1", sim$Group))) %>%
    mutate(step = sim$Step, cell_id = as.character(sim$Cell), group = sim$Group) %>%
    group_by(from, to) %>%
    mutate(percentage = pmin(1, (step - 1) / ceiling(length * n_steps_per_length))) %>%
    ungroup()
  #  select(cell_id, from, to, percentage)
  progressions <- progressions %>% filter(cell_id %in% colnames(counts))
  
  
  sim1 <- logNormCounts(sim)
  sim1 <- runPCA(sim1)
  sim.pca <- reducedDim(sim1)
  sim.pca.dat <- data.frame(sim.pca, label=paste(progressions$from, progressions$to, sep=" -> "))
  
  pdf(paste("./sim_data/sim_", i, "_pca.pdf", sep=""), width=5, height=4, onefile=F)
  print(ggplot(sim.pca.dat, aes(x=PC1, y=PC2, color=label)) + geom_point() +
          theme_classic())
  dev.off()
  
  save(milestone_network, progressions, file=paste("./sim_data/True_trajectory_", i, ".RData", sep=""))
  h5write(as.matrix(counts), paste("./sim_data/Splatter_simulate_", i, ".h5", sep=""), "X")
  h5write(as.matrix(truecounts), paste("./sim_data/Splatter_simulate_", i, ".h5", sep=""), "X_true")
  h5write(t(batch), paste("./sim_data/Splatter_simulate_", i, ".h5", sep=""), "Y")
  
}

write.csv(all.dropout.rate, file="./sim_data/simulate_dropout_rates.csv", row.names=F, col.names=F)