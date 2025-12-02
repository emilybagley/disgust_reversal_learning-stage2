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
                                                 'd_alpha_rew', 'd_alpha_pun','d_beta', 'd_omega',
                                                 'f_alpha_rew', 'f_alpha_pun','f_beta', 'f_omega')))
mean_draws <- posterior::as_draws_matrix(fit$draws(c('mu_p_alpha_rew', 'mu_p_alpha_pun','mu_p_beta', 'mu_p_omega',
                                                      'mu_d_alpha_rew', 'mu_d_alpha_pun','mu_d_beta', 'mu_d_omega',
                                                       'mu_f_alpha_rew', 'mu_f_alpha_pun','mu_f_beta', 'mu_f_omega')))
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

f_alpha_rew <- posterior::as_draws_matrix(fit$draws(c('f_alpha_rew')))
indiv_f_alpha_rew <-data.frame(summarise_draws(f_alpha_rew, mean))

f_alpha_pun <- posterior::as_draws_matrix(fit$draws(c('f_alpha_pun')))
indiv_f_alpha_pun <-data.frame(summarise_draws(f_alpha_pun, mean))

f_beta <- posterior::as_draws_matrix(fit$draws(c('f_beta')))
indiv_f_beta <-data.frame(summarise_draws(f_beta, mean))

f_omega <- posterior::as_draws_matrix(fit$draws(c('f_omega')))
indiv_f_omega <-data.frame(summarise_draws(f_omega, mean))

d_alpha_rew <- posterior::as_draws_matrix(fit$draws(c('d_alpha_rew')))
indiv_d_alpha_rew <-data.frame(summarise_draws(d_alpha_rew, mean))

d_alpha_pun <- posterior::as_draws_matrix(fit$draws(c('d_alpha_pun')))
indiv_d_alpha_pun <-data.frame(summarise_draws(d_alpha_pun, mean))

d_beta <- posterior::as_draws_matrix(fit$draws(c('d_beta')))
indiv_d_beta <-data.frame(summarise_draws(d_beta, mean))

d_omega <- posterior::as_draws_matrix(fit$draws(c('d_omega')))
indiv_d_omega <-data.frame(summarise_draws(d_omega, mean))



indiv_p_alpha_rew$subjID <- as.integer(gsub("p_alpha_rew\\[|\\]", "", indiv_p_alpha_rew$variable))
indiv_p_alpha_pun$subjID <- as.integer(gsub("p_alpha_pun\\[|\\]", "", indiv_p_alpha_pun$variable))
indiv_p_beta$subjID <- as.integer(gsub("p_beta\\[|\\]", "", indiv_p_beta$variable))
indiv_p_omega$subjID <- as.integer(gsub("p_omega\\[|\\]", "", indiv_p_omega$variable))

indiv_f_alpha_rew$subjID <- as.integer(gsub("f_alpha_rew\\[|\\]", "", indiv_f_alpha_rew$variable))
indiv_f_alpha_pun$subjID <- as.integer(gsub("f_alpha_pun\\[|\\]", "", indiv_f_alpha_pun$variable))
indiv_f_beta$subjID <- as.integer(gsub("f_beta\\[|\\]", "", indiv_f_beta$variable))
indiv_f_omega$subjID <- as.integer(gsub("f_omega\\[|\\]", "", indiv_f_omega$variable))

indiv_d_alpha_rew$subjID <- as.integer(gsub("d_alpha_rew\\[|\\]", "", indiv_d_alpha_rew$variable))
indiv_d_alpha_pun$subjID <- as.integer(gsub("d_alpha_pun\\[|\\]", "", indiv_d_alpha_pun$variable))
indiv_d_beta$subjID <- as.integer(gsub("d_beta\\[|\\]", "", indiv_d_beta$variable))
indiv_d_omega$subjID <- as.integer(gsub("d_omega\\[|\\]", "", indiv_d_omega$variable))


names(indiv_p_alpha_rew)[names(indiv_p_alpha_rew) == "mean"] <- "p_alpha_rew"
names(indiv_p_alpha_pun)[names(indiv_p_alpha_pun) == "mean"] <- "p_alpha_pun"
names(indiv_p_beta)[names(indiv_p_beta) == "mean"] <- "p_beta"
names(indiv_p_omega)[names(indiv_p_omega) == "mean"] <- "p_omega"

names(indiv_f_alpha_rew)[names(indiv_f_alpha_rew) == "mean"] <- "f_alpha_rew"
names(indiv_f_alpha_pun)[names(indiv_f_alpha_pun) == "mean"] <- "f_alpha_pun"
names(indiv_f_beta)[names(indiv_f_beta) == "mean"] <- "f_beta"
names(indiv_f_omega)[names(indiv_f_omega) == "mean"] <- "f_omega"

names(indiv_d_alpha_rew)[names(indiv_d_alpha_rew) == "mean"] <- "d_alpha_rew"
names(indiv_d_alpha_pun)[names(indiv_d_alpha_pun) == "mean"] <- "d_alpha_pun"
names(indiv_d_beta)[names(indiv_d_beta) == "mean"] <- "d_beta"
names(indiv_d_omega)[names(indiv_d_omega) == "mean"] <- "d_omega"

model_params <- Reduce(function(x, y) merge(x, y, by = "subjID"), 
                    list(indiv_p_alpha_rew[, c("subjID", "p_alpha_rew")],
                         indiv_p_alpha_pun[, c("subjID", "p_alpha_pun")],
                         indiv_p_beta[, c("subjID", "p_beta")],
                         indiv_p_omega[, c("subjID", "p_omega")],
                         indiv_f_alpha_rew[, c("subjID", "f_alpha_rew")],
                         indiv_f_alpha_pun[, c("subjID", "f_alpha_pun")],
                         indiv_f_beta[, c("subjID", "f_beta")],
                         indiv_f_omega[, c("subjID", "f_omega")],
                         indiv_d_alpha_rew[, c("subjID", "d_alpha_rew")],
                         indiv_d_alpha_pun[, c("subjID", "d_alpha_pun")],
                         indiv_d_beta[, c("subjID", "d_beta")],
                         indiv_d_omega[, c("subjID", "d_omega")]
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