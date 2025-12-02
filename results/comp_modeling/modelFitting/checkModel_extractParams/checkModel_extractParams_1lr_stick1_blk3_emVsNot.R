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
draws <- posterior::as_draws_matrix(fit$draws(c('e_alpha', 'p_alpha','e_beta',  'p_beta', 'e_omega', 'p_omega')))
mean_draws <- posterior::as_draws_matrix(fit$draws(c('mu_e_alpha','mu_p_alpha','mu_e_beta', 'mu_p_beta', 'mu_e_omega', 'mu_p_omega')))
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
e_alpha <- posterior::as_draws_matrix(fit$draws(c('e_alpha')))
indiv_e_alpha <-data.frame(summarise_draws(e_alpha, mean))
p_alpha <- posterior::as_draws_matrix(fit$draws(c('p_alpha')))
indiv_p_alpha <-data.frame(summarise_draws(p_alpha, mean))
e_beta <- posterior::as_draws_matrix(fit$draws(c('e_beta')))
indiv_e_beta <-data.frame(summarise_draws(e_beta, mean))
p_beta <- posterior::as_draws_matrix(fit$draws(c('p_beta')))
indiv_p_beta<-data.frame(summarise_draws(p_beta, mean))
e_omega <- posterior::as_draws_matrix(fit$draws(c('e_omega')))
indiv_e_omega <-data.frame(summarise_draws(e_omega, mean))
p_omega <- posterior::as_draws_matrix(fit$draws(c('p_omega')))
indiv_p_omega <-data.frame(summarise_draws(p_omega, mean))

indiv_e_alpha$subjID <- as.integer(gsub("e_alpha\\[|\\]", "", indiv_e_alpha$variable))
indiv_e_beta$subjID <- as.integer(gsub("e_beta\\[|\\]", "", indiv_e_beta$variable))
indiv_e_omega$subjID <- as.integer(gsub("e_omega\\[|\\]", "", indiv_e_omega$variable))
names(indiv_e_alpha)[names(indiv_e_alpha) == "mean"] <- "e_alpha"
names(indiv_e_beta)[names(indiv_e_beta) == "mean"] <- "e_beta"
names(indiv_e_omega)[names(indiv_e_omega) == "mean"] <- "e_omega"
indiv_p_alpha$subjID <- as.integer(gsub("p_alpha\\[|\\]", "", indiv_p_alpha$variable))
indiv_p_beta$subjID <- as.integer(gsub("p_beta\\[|\\]", "", indiv_p_beta$variable))
indiv_p_omega$subjID <- as.integer(gsub("p_omega\\[|\\]", "", indiv_p_omega$variable))
names(indiv_p_alpha)[names(indiv_p_alpha) == "mean"] <- "p_alpha"
names(indiv_p_beta)[names(indiv_p_beta) == "mean"] <- "p_beta"
names(indiv_p_omega)[names(indiv_p_omega) == "mean"] <- "p_omega"

model_params <- Reduce(function(x, y) merge(x, y, by = "subjID"), 
                    list(indiv_e_alpha[, c("subjID", "e_alpha")],
                         indiv_e_beta[, c("subjID", "e_beta")],
                         indiv_e_omega[, c("subjID", "e_omega")],
                         indiv_p_alpha[, c("subjID", "p_alpha")],
                         indiv_p_beta[, c("subjID", "p_beta")],
                         indiv_p_omega[, c("subjID", "p_omega")]
                         )
                         )

#save outputs
print("saving")
ggsave(paste0(outputFolder, "neffplot_", modelName, ".png"), neff)
ggsave(paste0(outputFolder, "traceplot_", modelName, ".png"), trace)
ggsave(paste0(outputFolder, "rhat_", modelName, ".png"), rhat)


saveRDS(model_params, paste0(outputFolder, "modelPars_", modelName, ".rds"))
saveRDS(draws, paste0(outputFolder, "draws_", modelName, ".rds"))
saveRDS(rhat, paste0(outputFolder, "rhat_", modelName, ".rds"))
saveRDS(loglik, paste0(outputFolder, "loglik_", modelName, ".rds"))