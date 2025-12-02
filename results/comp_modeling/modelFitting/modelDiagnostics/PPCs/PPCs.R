args <- commandArgs()
print("Posterior Predictive Checks")
modelName = args[6]
dataFile=args[7]
outdir=args[8]
print("PPCs.R")
print(modelName)
print(outdir)

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

#load actual data
print("Load data")
stan_data <- readRDS(paste0("../../",dataFile))
true_y <- stan_data$choice
outcome_df <- stan_data$outcome

#load model fit
print("Load model fit")
modelName <- "1lr_stick1_blk3_allparamsep"
modelFolder = paste0("../../modelOutputs", "/", modelName, "/")
print(paste0(modelFolder, "y_pred_", modelName, ".rds"))
csvspaths<-readRDS(paste0(modelFolder, 'csvpaths_', modelName, '.rds'))
fit <- as_cmdstan_fit(csvspaths)

#extract y_pred matrix
print("Extract y_pred matrix")
y_pred_mat <- as_draws_matrix(fit$draws("y_pred"))

#combine actual data and ypred matrix
pp_df <- data.frame()
df <- data.frame()
for(sub_no in 1:dim(true_y)[1]) {
    for(block_no in 1:dim(true_y)[2]) {
        subj_observed=true_y[sub_no, block_no,]
        subj_outcome=outcome_df[sub_no, block_no,]
        subj_cols <- grep(paste0("y_pred\\[", sub_no, ",", block_no, ",.*\\]$"), 
                            colnames(y_pred_mat), value = TRUE)
        subj_y_pred <- y_pred_mat[, subj_cols]
        y_pred_Medians <- apply(subj_y_pred, 2, median)
        y_pred_mean <- apply(subj_y_pred, 2, mean)
        y_pred_LowerQ <- apply(subj_y_pred, 2, quantile, probs = 0.25)
        y_pred_UpperQ <- apply(subj_y_pred, 2, quantile, probs = 0.75)

        trial_idx <- 1:sum(!is.na(subj_observed))
        df <- data.frame(
            subject = sub_no,
            block = block_no,
            trial = trial_idx,
            observed = subj_observed,
            outcome = subj_outcome,
            pred_median = y_pred_Medians,
            pred_lower = y_pred_LowerQ,
            pred_upper = y_pred_UpperQ,
            pred_mean = y_pred_mean
        )
        df <- df[!(df$observed == -1 & df$pred_median == -1), ]
        pp_df <- rbind(pp_df, df)
    }
}

#save out
print("Save out files")
#outputFolder = paste0("modelOutputs/", modelName, "/PPCs/")
saveRDS(y_pred_mat, paste0("y_pred_", modelName, ".rds"))
saveRDS(pp_df, paste0("postpred_alltrials_", modelName, ".rds") )
