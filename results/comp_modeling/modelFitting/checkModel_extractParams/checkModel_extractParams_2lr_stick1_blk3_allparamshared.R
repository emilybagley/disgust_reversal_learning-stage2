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

print("Load model fit")
outputFolder = paste0("modelOutputs", "/", modelName, "/")
csvspaths<-readRDS(paste0(outputFolder, 'csvpaths_', modelName, '.rds'))
fit <- as_cmdstan_fit(csvspaths)

##########################################################
#extract draws
print("extract draws")
draws <- posterior::as_draws_matrix(fit$draws(c('alpha_rew', 'alpha_pun','beta', 'omega'
                                                 )))
mean_draws <- posterior::as_draws_matrix(fit$draws(c('mu_alpha_rew', 'mu_alpha_pun','mu_beta', 'mu_omega'
                                                       )))
loglik <- posterior::as_draws_matrix(fit$draws(c('log_lik')))

#check model outputs
print("check model")
theme_set(theme_bw())
neff <- mcmc_neff(neff_ratio(fit))
trace <- mcmc_trace(mean_draws)
rhat <- mcmc_rhat(summarise_draws(draws)$rhat)

#extract model parameters
print("extract model params")
indiv_params <-summarise_draws(draws, mean)

alpha_rew <- posterior::as_draws_matrix(fit$draws(c('alpha_rew')))
indiv_alpha_rew <-data.frame(summarise_draws(alpha_rew, mean))

alpha_pun <- posterior::as_draws_matrix(fit$draws(c('alpha_pun')))
indiv_alpha_pun <-data.frame(summarise_draws(alpha_pun, mean))

beta <- posterior::as_draws_matrix(fit$draws(c('beta')))
indiv_beta <-data.frame(summarise_draws(beta, mean))

omega <- posterior::as_draws_matrix(fit$draws(c('omega')))
indiv_omega <-data.frame(summarise_draws(omega, mean))


indiv_alpha_rew$subjID <- as.integer(gsub("alpha_rew\\[|\\]", "", indiv_alpha_rew$variable))
indiv_alpha_pun$subjID <- as.integer(gsub("alpha_pun\\[|\\]", "", indiv_alpha_pun$variable))
indiv_beta$subjID <- as.integer(gsub("beta\\[|\\]", "", indiv_beta$variable))
indiv_omega$subjID <- as.integer(gsub("omega\\[|\\]", "", indiv_omega$variable))


names(indiv_alpha_rew)[names(indiv_alpha_rew) == "mean"] <- "alpha_rew"
names(indiv_alpha_pun)[names(indiv_alpha_pun) == "mean"] <- "alpha_pun"
names(indiv_beta)[names(indiv_beta) == "mean"] <- "beta"
names(indiv_omega)[names(indiv_omega) == "mean"] <- "omega"


model_params <- Reduce(function(x, y) merge(x, y, by = "subjID"), 
                    list(indiv_alpha_rew[, c("subjID", "alpha_rew")],
                         indiv_alpha_pun[, c("subjID", "alpha_pun")],
                         indiv_beta[, c("subjID", "beta")],
                         indiv_omega[, c("subjID", "omega")]
                         ))

#save outputs
print("saving")
ggsave(paste0(outputFolder, "neffplot_", modelName, ".png"), neff)
ggsave(paste0(outputFolder, "traceplot_", modelName, ".png"), trace)
ggsave(paste0(outputFolder, "rhat_", modelName, ".png"), rhat)


saveRDS(model_params, paste0(outputFolder, "modelPars_", modelName, ".rds"))
saveRDS(draws, paste0(outputFolder, "draws_", modelName, ".rds"))
saveRDS(rhat, paste0(outputFolder, "rhat_", modelName, ".rds"))
saveRDS(loglik, paste0(outputFolder, "loglik_", modelName, ".rds"))