#module load R/4.3.1_v2

##load packages
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

raw_data <- read.csv("../csvs/complete_task_excluded.csv")

data<-raw_data[!is.na(raw_data$trial_till_correct), ]
data <- data[, c('participant_no', 'n_trial', 'block_no',"block_type", 'stim_selected', 'feedback')]
#data$block_no <- data$block_no +1
colnames(data) <- c("subjID", "trial_no", "block_no", "block_Type", "choice", "outcome")
data <- data %>%
  mutate(outcome = ifelse(outcome == "correct", 1, -1))
data <- data %>%
  mutate(choice = ifelse(choice == 0, 1, 2))
data$block_no=data$block_no+1
data$block_Type <- tolower(data$block_Type)
data <- as.data.frame(data)

#set up sim data for stan
df <- data %>% arrange(subjID, block_Type) ##make sure blocks go disgust, fear, points
subj_list <- unique(df$subjID)
N <- length(subj_list)
B <- 3 ##max number of blocks
Bsubj <- rep(B, N)
block_list <- 1:B  # assuming 3 blocks per subject

# Create an N x B matrix of trial counts per subject per block
Tsubj <- matrix(0, nrow = length(subj_list), ncol = length(block_list))

for (i in seq_along(subj_list)) {
  subj_data <- df[df$subjID == subj_list[i], ]
  for (j in block_list) {
    if (j==1){
      block_data <- subj_data[subj_data$block_Type == "disgust", ]
    } else if (j==2){
      block_data <- subj_data[subj_data$block_Type == "fear", ]
    } else if (j==3){
      block_data <- subj_data[subj_data$block_Type == "points", ]
    }
    Tsubj[i, j] <- nrow(block_data)
  }
}
T <- max(Tsubj)  # max number of trials across all subj/block combos

choice_array <- array(-1, dim = c(N, B, T))      
outcome_array <- array(0, dim = c(N, B, T))

for (i in seq_along(subj_list)) {
  subj_data <- df[df$subjID == subj_list[i], ]
  for (j in seq_along(block_list)) {
    if (j==1){
      block_data <- subj_data[subj_data$block_Type == "disgust", ]
    } else if (j==2){
      block_data <- subj_data[subj_data$block_Type == "fear", ]
    } else if (j==3){
      block_data <- subj_data[subj_data$block_Type == "points", ]
    }
    n_trials <- nrow(block_data)
    if (n_trials > 0) {
      choice_array[i, j, 1:n_trials] <- block_data$choice
      outcome_array[i, j, 1:n_trials] <- block_data$outcome
    }
  }
}
stan_data <-list(N = N, B = B, Bsubj = Bsubj, T = T, Tsubj = Tsubj, choice = choice_array, outcome=outcome_array)

fileName=paste("../csvs/stanData_", 'excluded_df', ".rds", sep="")
saveRDS(stan_data, file = fileName)
