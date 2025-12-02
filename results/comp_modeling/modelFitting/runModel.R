args <- commandArgs()
modelName = args[6]
print("runModel.R")
print(modelName)
stanFile=args[7]
dataFile=args[8]
chains=args[9]
parallel_chains=args[10]
iter_warmup=args[11]
iter_sampling=args[12]
thin=args[13]
save_warmup=args[14]
output_dir=args[15]

set.seed(41)
library(dplyr)
library(ggplot2)
library(hBayesDM)
library(pheatmap)
library(truncnorm)
library(rstan)
library(cmdstanr)
library(bayesplot)
library(posterior)
library(loo)
library(rstanarm)
options(device = "png")

stan_data <- readRDS(dataFile)
model <- cmdstan_model(stanFile)
Sys.time()
fit <- model$sample(
  data = stan_data,
  chains = as.numeric(chains),
  parallel_chains = as.numeric(parallel_chains),
  iter_warmup= as.numeric(iter_warmup),
  iter_sampling=as.numeric(iter_sampling),
  thin=as.numeric(thin),
  save_warmup=FALSE,
  output_dir=output_dir,
  output_basename=modelName
)
csvs=fit$output_files()

outputFolder = paste0("modelOutputs", "/", modelName, "/")
saveRDS(csvs, paste0(outputFolder, 'csvpaths_', modelName, '.rds'))

#fitFile=paste0(outputFolder, 'fit_', modelName, ".rds")
#fit$save_object(file =  fitFile)
