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
draws <- posterior::as_draws_matrix(fit$draws(c('p_alpha_rew', 'p_alpha_pun','p_beta', 'p_omega',
                                                 'e_alpha_rew', 'e_alpha_pun','e_beta', 'e_omega'
                                                 )))
mean_draws <- posterior::as_draws_matrix(fit$draws(c('mu_p_alpha_rew', 'mu_p_alpha_pun','mu_p_beta', 'mu_p_omega',
                                                      'mu_e_alpha_rew', 'mu_e_alpha_pun','mu_e_beta', 'mu_e_omega'
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

p_alpha_rew <- posterior::as_draws_matrix(fit$draws(c('p_alpha_rew')))
indiv_p_alpha_rew <-data.frame(summarise_draws(p_alpha_rew, mean))

p_alpha_pun <- posterior::as_draws_matrix(fit$draws(c('p_alpha_pun')))
indiv_p_alpha_pun <-data.frame(summarise_draws(p_alpha_pun, mean))

p_beta <- posterior::as_draws_matrix(fit$draws(c('p_beta')))
indiv_p_beta <-data.frame(summarise_draws(p_beta, mean))

p_omega <- posterior::as_draws_matrix(fit$draws(c('p_omega')))
indiv_p_omega <-data.frame(summarise_draws(p_omega, mean))

e_alpha_rew <- posterior::as_draws_matrix(fit$draws(c('e_alpha_rew')))
indiv_e_alpha_rew <-data.frame(summarise_draws(e_alpha_rew, mean))

e_alpha_pun <- posterior::as_draws_matrix(fit$draws(c('e_alpha_pun')))
indiv_e_alpha_pun <-data.frame(summarise_draws(e_alpha_pun, mean))

e_beta <- posterior::as_draws_matrix(fit$draws(c('e_beta')))
indiv_e_beta <-data.frame(summarise_draws(e_beta, mean))

e_omega <- posterior::as_draws_matrix(fit$draws(c('e_omega')))
indiv_e_omega <-data.frame(summarise_draws(e_omega, mean))



indiv_p_alpha_rew$subjID <- as.integer(gsub("p_alpha_rew\\[|\\]", "", indiv_p_alpha_rew$variable))
indiv_p_alpha_pun$subjID <- as.integer(gsub("p_alpha_pun\\[|\\]", "", indiv_p_alpha_pun$variable))
indiv_p_beta$subjID <- as.integer(gsub("p_beta\\[|\\]", "", indiv_p_beta$variable))
indiv_p_omega$subjID <- as.integer(gsub("p_omega\\[|\\]", "", indiv_p_omega$variable))

indiv_e_alpha_rew$subjID <- as.integer(gsub("e_alpha_rew\\[|\\]", "", indiv_e_alpha_rew$variable))
indiv_e_alpha_pun$subjID <- as.integer(gsub("e_alpha_pun\\[|\\]", "", indiv_e_alpha_pun$variable))
indiv_e_beta$subjID <- as.integer(gsub("e_beta\\[|\\]", "", indiv_e_beta$variable))
indiv_e_omega$subjID <- as.integer(gsub("e_omega\\[|\\]", "", indiv_e_omega$variable))


names(indiv_p_alpha_rew)[names(indiv_p_alpha_rew) == "mean"] <- "p_alpha_rew"
names(indiv_p_alpha_pun)[names(indiv_p_alpha_pun) == "mean"] <- "p_alpha_pun"
names(indiv_p_beta)[names(indiv_p_beta) == "mean"] <- "p_beta"
names(indiv_p_omega)[names(indiv_p_omega) == "mean"] <- "p_omega"

names(indiv_e_alpha_rew)[names(indiv_e_alpha_rew) == "mean"] <- "e_alpha_rew"
names(indiv_e_alpha_pun)[names(indiv_e_alpha_pun) == "mean"] <- "e_alpha_pun"
names(indiv_e_beta)[names(indiv_e_beta) == "mean"] <- "e_beta"
names(indiv_e_omega)[names(indiv_e_omega) == "mean"] <- "e_omega"


model_params <- Reduce(function(x, y) merge(x, y, by = "subjID"), 
                    list(indiv_p_alpha_rew[, c("subjID", "p_alpha_rew")],
                         indiv_p_alpha_pun[, c("subjID", "p_alpha_pun")],
                         indiv_p_beta[, c("subjID", "p_beta")],
                         indiv_p_omega[, c("subjID", "p_omega")],
                         indiv_e_alpha_rew[, c("subjID", "e_alpha_rew")],
                         indiv_e_alpha_pun[, c("subjID", "e_alpha_pun")],
                         indiv_e_beta[, c("subjID", "e_beta")],
                         indiv_e_omega[, c("subjID", "e_omega")]
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