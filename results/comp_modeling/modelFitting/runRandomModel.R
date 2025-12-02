Sys.info()["nodename"]
print("Check Model and extract params")
args <- commandArgs()
modelName = args[6]
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

###RUN MODEL
print("Run model")
stan_data <- readRDS(dataFile)
model <- cmdstan_model(stanFile)
Sys.time()
fit <- model$sample(
  data = stan_data,
  chains = as.numeric(chains),
  parallel_chains = as.numeric(parallel_chains),
  iter_warmup= as.numeric(iter_warmup),
  iter_sampling=as.numeric(iter_sampling),
  thin= as.numeric(thin),
  save_warmup=FALSE,
  output_dir=output_dir,
  output_basename=modelName,
  fixed_param=TRUE
)
csvs=fit$output_files()

outputFolder = paste0("modelOutputs", "/", modelName, "/")
saveRDS(csvs, paste0(outputFolder, 'csvpaths_', modelName, '.rds'))

##LOAD MODEL FIT
print("Load model fit")
outputFolder = paste0("modelOutputs", "/", modelName, "/")
csvspaths<-readRDS(paste0(outputFolder, 'csvpaths_', modelName, '.rds'))
fit <- as_cmdstan_fit(csvspaths)

loglik <- posterior::as_draws_matrix(fit$draws(c('log_lik')))

##save out files
#save outputs
print("saving")
saveRDS(loglik, paste0(outputFolder, "loglik_", modelName, ".rds"))