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

##simulate parameters
num_subjs  <- 100    # Number of subjects

simul_pars <- data.frame(pointsLearningRate = rtruncnorm(num_subjs ,0 ,1, 0.63, 0.44),
                        pointsInverseTemp= rtruncnorm(num_subjs ,0 ,10, 4.22, 4),
                        pointsStickiness = rtruncnorm(num_subjs ,-5 ,5, 0.15, 1.42),
                        fearLearningRate = rtruncnorm(num_subjs ,0 ,1, 0.63, 0.44),
                        fearInverseTemp= rtruncnorm(num_subjs ,0 ,10, 4.22, 4),
                        fearStickiness = rtruncnorm(num_subjs ,-5 ,5, 0.15, 1.42),
                        disgustLearningRate = rtruncnorm(num_subjs ,0 ,1, 0.63, 0.44),
                        disgustInverseTemp= rtruncnorm(num_subjs ,0 ,10, 4.22, 4),
                        disgustStickiness = rtruncnorm(num_subjs ,-5 ,5, 0.15, 1.42),
                        subjID  = 1:num_subjs)


df <- data.frame(
            trial_no = numeric(),
            correct = numeric(),
            choice = numeric(),
            outcome = numeric(),
            stim1Val = numeric(),
            stim2Val = numeric(),
            nextProb1= numeric(),
            nextProb2 = numeric(),
            nextChoice= numeric(),
            learning_Rate = numeric(),
            inverse_Temp = numeric(),
            subjID = numeric()
            )
for (subj in 1:num_subjs){
    block_order <- sample(c('fear', 'disgust', 'points'))
    sub_df <- data.frame(
                trial_no = numeric(),
                block_no = numeric(),
                correct = numeric(),
                choice = numeric(),
                outcome = numeric(),
                stim1Val = numeric(),
                stim2Val = numeric(),
                nextProb1= numeric(),
                nextProb2 = numeric(),
                nextChoice= numeric(),
                learning_Rate = numeric(),
                inverse_Temp = numeric(),
                stickiness = numeric(),
                subjID = numeric()
                )
    #set reversal details
    reversalThresh <- 5 #have a reversal when 5 trials in a row have been correct
    maxReversals <- 8 #will have 7 reversals of contingency in each block - so max is 8
    maxTrial <-200
    n_block <- 3


    ##cycle through the three blocks
    for (block in 1:n_block){
        block_type <- block_order[block]
        block_df <- data.frame(
                trial_no = numeric(),
                block_no = numeric(),
                correct = numeric(),
                choice = numeric(),
                outcome = numeric(),
                stim1Val = numeric(),
                stim2Val = numeric(),
                nextProb1= numeric(),
                nextProb2 = numeric(),
                nextChoice= numeric(),
                learning_Rate = numeric(),
                inverse_Temp = numeric(),
                stickiness = numeric(),
                subjID = numeric()
                )


        #set which stimulus is correct first
        firstCorrect = sample(c(1, 2), size = 1, replace = FALSE, prob = c(0.5, 0.5))
        block_df[1, 'correct'] = firstCorrect

        #set parameters
        if (block_type == "points"){
            learningRate <- simul_pars[simul_pars$subjID==subj, ]$pointsLearningRate
            inverseTemp <- simul_pars[simul_pars$subjID==subj, ]$pointsInverseTemp
            stickiness <- simul_pars[simul_pars$subjID==subj, ]$pointsStickiness
        } else if (block_type == "fear"){
            learningRate <- simul_pars[simul_pars$subjID==subj, ]$fearLearningRate
            inverseTemp <- simul_pars[simul_pars$subjID==subj, ]$fearInverseTemp  
            stickiness <- simul_pars[simul_pars$subjID==subj, ]$fearStickiness          
        } else if (block_type == "disgust"){
            learningRate <- simul_pars[simul_pars$subjID==subj, ]$disgustLearningRate
            inverseTemp <- simul_pars[simul_pars$subjID==subj, ]$disgustInverseTemp
            stickiness <- simul_pars[simul_pars$subjID==subj, ]$disgustStickiness
        } else {
            print("error")
        }

        #run through each trial
        for (trial_no in 1:maxTrial){   
            block_df[trial_no, "trial_no"] <- trial_no
            block_df[trial_no, "subjID"] <- subj     
            block_df[trial_no, 'block_no'] <- block  
            if (trial_no == 1){ #create a random start for trial 1
                choice <- sample(c(1, 2), size = 1, replace = FALSE, prob = c(0.5, 0.5))
                prevStim1Val <- 0 
                prevStim2Val <- 0 
                nCorrect <- 0
                correct <- firstCorrect
                nReversal=0
            } else { #load in relevant params from trial before
                choice <- nextChoice
                prevStim1Val <- stim1Val
                prevStim2Val <- stim2Val
                correct <- nextCorrect
            }
            block_df[block_df$trial_no==trial_no, 'choice'] <- choice

            #create probabilistic outcome (in the task)
            if (choice == correct){
                outcome <- sample(c(1, -1), size = 1, replace = FALSE, prob = c(0.8, 0.2))
                nCorrect= nCorrect +1 
            } else {
                outcome <- sample(c(-1, 1), size = 1, replace = FALSE, prob = c(0.8, 0.2))
                nCorrect=0
            }
            block_df[block_df$trial_no==trial_no, 'outcome'] <- outcome 
            block_df[block_df$trial_no==trial_no, 'nCorrect'] <- nCorrect 

            # update stimulus values using equation 1
            if (choice == 1){ # update stim1Val
                predictionError <- outcome - prevStim1Val
                stim1Val <- prevStim1Val + learningRate*predictionError
                stim2Val <- prevStim2Val
            } else { # update stim2Val
                predictionError <- outcome - prevStim2Val
                stim2Val <- prevStim2Val + learningRate*predictionError
                stim1Val <- prevStim1Val
                }
            block_df[block_df$trial_no==trial_no, 'stim1Val'] <- stim1Val
            block_df[block_df$trial_no==trial_no, 'stim2Val'] <- stim2Val

            # update probability of next choice using the softmax function (equation 2)
            last_choice <- choice 
            if (last_choice == 1){
                nextProb1 <- 1/(1+ exp(-inverseTemp*((stim1Val-stim2Val)+stickiness))) #probability of selecting stim1
                nextProb2 <- 1-nextProb1
            } else {
                nextProb2 <- 1/(1+ exp(-inverseTemp*((stim2Val-stim1Val)+stickiness))) 
                nextProb1 <- 1-nextProb2
            }

            block_df[block_df$trial_no==trial_no, 'nextProb1'] <- nextProb1
            block_df[block_df$trial_no==trial_no, 'nextProb2'] <- nextProb2

            #convert this probability into a choice for the next trial
            nextChoice = sample(c(1, 2), size = 1, replace = FALSE, prob = c(nextProb1, nextProb2))
            block_df[block_df$trial_no==trial_no, 'nextChoice'] <- nextChoice

            #determine what will be the correct option on the next trial
            if(nCorrect < 5){
                nextCorrect <- correct
                nReversal <- nReversal
            } else if(nCorrect==5){
                nReversal = nReversal + 1
                if(correct==1){
                    nextCorrect<-2
                } else{
                    nextCorrect<-1
                }   
            }
            block_df[block_df$trial_no==trial_no, 'nextCorrect'] <- nextCorrect
            block_df[block_df$trial_no==trial_no, 'correct'] <- correct
            block_df[block_df$trial_no==trial_no, 'nReversal'] <- nReversal
            block_df <- block_df %>% mutate(learning_Rate = learningRate)
            block_df <- block_df %>% mutate(inverse_Temp = inverseTemp)
            block_df <- block_df %>% mutate(block_Type = block_type)

            if ((nReversal == maxReversals) & (nCorrect == 5)) {
                break # stop the for-loop if the max number of reversals has been reached
            } else if (nCorrect==5){
                nCorrect<-0
            }
        }
    sub_df =rbind(sub_df, block_df)  
    }
    df = rbind(df, sub_df)
}


#set up sim data for stan
df <- df %>% arrange(subjID, block_Type) ##make sure blocks go disgust, fear, points
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

##save out
fileName="simData.rds"
saveRDS(df, file = fileName)

fileName="simParams.rds"
saveRDS(simul_pars, file = fileName)

fileName="stan_data.rds"
saveRDS(stan_data, file=fileName)