/**
 * Probabilistic Reversal Learning (PRL) Task
 *
 * Reward-Punishment Model by Ouden et al. (2013) Neuron
 */

data {
  int<lower=1> N;                          // Number of subjects
  int<lower=1> B;                          // Max number of blocks across subjects
  array[N] int<lower=1, upper=B> Bsubj;       // Number of blocks for each subject
  int<lower=0> T;                          // Max number of trials across subjects
  array[N, B] int<lower=0, upper=T> Tsubj;  // Number of trials/block for each subject
  array[N, B, T] int<lower=-1, upper=2> choice; // Choice for each subject-block-trial
  array[N, B, T] real outcome;                   // Outcome (reward/loss) for each subject-block-trial
}

transformed data {
  // Default value for (re-)initializing parameter vectors
  vector[2] initV;
  initV = rep_vector(0.0, 2);
}

// Declare all parameters as vectors for vectorizing
parameters {
  // Hyper(group)-parameters
  vector[9] mu_pr;
  vector<lower=0>[9] sigma;

  // Subject-level raw parameters (for Matt trick)
  vector[N] d_alpha_rew_pr;   // pos learning rate
  vector[N] d_alpha_pun_pr;   // neg learning rate
  vector[N] d_beta_pr;   // inverse temperature
  vector[N] f_alpha_rew_pr;   // pos learning rate
  vector[N] f_alpha_pun_pr;   // neg learning rate
  vector[N] f_beta_pr;   // inverse temperature
  vector[N] p_alpha_rew_pr;   // learning rate
  vector[N] p_alpha_pun_pr;   // learning rate
  vector[N] p_beta_pr;   // inverse temperature
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0, upper=1>[N] d_alpha_rew;
  vector<lower=0, upper=1>[N] d_alpha_pun;
  vector<lower=0, upper=10>[N] d_beta;
  vector<lower=0, upper=1>[N] f_alpha_rew;
  vector<lower=0, upper=1>[N] f_alpha_pun;
  vector<lower=0, upper=10>[N] f_beta;
  vector<lower=0, upper=1>[N] p_alpha_rew;
  vector<lower=0, upper=1>[N] p_alpha_pun;
  vector<lower=0, upper=10>[N] p_beta;

  for (i in 1:N) {
    d_alpha_rew[i]  = Phi_approx(mu_pr[1] + sigma[1] * d_alpha_rew_pr[i]);
    d_alpha_pun[i]  = Phi_approx(mu_pr[2] + sigma[2] * d_alpha_pun_pr[i]);
    d_beta[i]  = Phi_approx(mu_pr[3] + sigma[3] * d_beta_pr[i]) * 10;
    f_alpha_rew[i]  = Phi_approx(mu_pr[4] + sigma[4] * f_alpha_rew_pr[i]);
    f_alpha_pun[i]  = Phi_approx(mu_pr[5] + sigma[5] * f_alpha_pun_pr[i]);
    f_beta[i]  = Phi_approx(mu_pr[6] + sigma[6] * f_beta_pr[i]) * 10;
    p_alpha_rew[i]  = Phi_approx(mu_pr[7] + sigma[7] * p_alpha_rew_pr[i]);
    p_alpha_pun[i]  = Phi_approx(mu_pr[8] + sigma[8] * p_alpha_pun_pr[i]);
    p_beta[i]  = Phi_approx(mu_pr[9] + sigma[9] * p_beta_pr[i]) * 10;
  }
}

model {
  // Hyperparameters
  mu_pr  ~ normal(0, 1);
  sigma ~ normal(0, 0.2);

  // individual parameters
  d_alpha_rew_pr ~ normal(0, 1);
  d_alpha_pun_pr ~ normal(0, 1);
  d_beta_pr ~ normal(0, 1);
  f_alpha_rew_pr ~ normal(0, 1);
  f_alpha_pun_pr ~ normal(0, 1);
  f_beta_pr ~ normal(0, 1);
  p_alpha_rew_pr ~ normal(0, 1);
  p_alpha_pun_pr ~ normal(0, 1);
  p_beta_pr ~ normal(0, 1);

  for (i in 1:N) {
    for (bIdx in 1:Bsubj[i]) {  // new
        // Define Values
        vector[2] ev;   // Expected value
        real PE;        // prediction error

        // Initialize values
        ev = initV;     // initial ev values


    if (bIdx == 1) {
      for (t in 1:Tsubj[i, bIdx]) {
        // Softmax choice
        choice[i, bIdx, t] ~ categorical_logit(ev * d_beta[i]);

        // Prediction Error
        PE = outcome[i, bIdx, t] - ev[choice[i, bIdx, t]];

        // Update expected value of chosen stimulus
        if (outcome[i, bIdx, t] > 0)
          ev[choice[i, bIdx, t]] += d_alpha_rew[i] * PE;
        else
          ev[choice[i, bIdx, t]] += d_alpha_pun[i] * PE;
      }
    }
 
    if (bIdx == 2) {
      for (t in 1:Tsubj[i, bIdx]) {
        // Softmax choice
        choice[i, bIdx, t] ~ categorical_logit(ev * f_beta[i]);

        // Prediction Error
        PE = outcome[i, bIdx, t] - ev[choice[i, bIdx, t]];

        // Update expected value of chosen stimulus
        if (outcome[i, bIdx, t] > 0)
          ev[choice[i, bIdx, t]] += f_alpha_rew[i] * PE;
        else
          ev[choice[i, bIdx, t]] += f_alpha_pun[i] * PE;
      }
    }

    if (bIdx == 3) {
      for (t in 1:Tsubj[i, bIdx]) {
        // Softmax choice
        choice[i, bIdx, t] ~ categorical_logit(ev * p_beta[i]);

        // Prediction Error
        PE = outcome[i, bIdx, t] - ev[choice[i, bIdx, t]];

        // Update expected value of chosen stimulus
        if (outcome[i, bIdx, t] > 0)
          ev[choice[i, bIdx, t]] += p_alpha_rew[i] * PE;
        else
          ev[choice[i, bIdx, t]] += p_alpha_pun[i] * PE;
      }
    } 

  }
}
}

generated quantities {
  // For group level parameters
  real<lower=0, upper=1>  mu_d_alpha_rew;
  real<lower=0, upper=1>  mu_d_alpha_pun;
  real<lower=0, upper=10> mu_d_beta;
  real<lower=0, upper=1>  mu_f_alpha_rew;
  real<lower=0, upper=1>  mu_f_alpha_pun;
  real<lower=0, upper=10> mu_f_beta;
  real<lower=0, upper=1>  mu_p_alpha_rew;
  real<lower=0, upper=1>  mu_p_alpha_pun;
  real<lower=0, upper=10> mu_p_beta;

  // For log likelihood calculation
  array[N] real log_lik;

  // For model regressors
  array[N, B, T] real ev_c;   // Expected value of the chosen option
  array[N, B, T] real ev_nc;  // Expected value of the non-chosen option
  array[N, B, T] real pe;     // Prediction error

  // For posterior predictive check
  array[N, B, T] real y_pred;

// Initialize all the variables to avoid NULL values
  for (i in 1:N) {
      for (b in 1:B){
        for (t in 1:T) {
            ev_c[i, b, t]   = 0;
            ev_nc[i, b, t]  = 0;
            pe[i, b, t]     = 0;

            y_pred[i, b, t]    = -1;
      }
    }
  }

  mu_d_alpha_rew = Phi_approx(mu_pr[1]);
  mu_d_alpha_pun = Phi_approx(mu_pr[2]);
  mu_d_beta = Phi_approx(mu_pr[3]) * 10;
  mu_f_alpha_rew = Phi_approx(mu_pr[4]);
  mu_f_alpha_pun = Phi_approx(mu_pr[5]);
  mu_f_beta = Phi_approx(mu_pr[6]) * 10;
  mu_p_alpha_rew = Phi_approx(mu_pr[7]);
  mu_p_alpha_pun = Phi_approx(mu_pr[8]);
  mu_p_beta = Phi_approx(mu_pr[9]) * 10;

  { // local section, this saves time and space
    for (i in 1:N) {
      log_lik[i]=0;
      for (bIdx in 1:Bsubj[i]) {
          // Define values
          vector[2] ev; // Expected value
          real PE;      // Prediction error
          ev = initV; //initial ev values

          if(bIdx == 1){
            for (t in 1:Tsubj[i, bIdx]) {
                // Softmax choice
                log_lik[i]  += categorical_logit_lpmf(choice[i, bIdx, t] | ev * d_beta[i]);

                // generate posterior prediction for current trial
                y_pred[i, bIdx, t] = categorical_rng(softmax(ev * d_beta[i]));

                // Prediction Error
                PE = outcome[i, bIdx, t] - ev[choice[i, bIdx, t]];

                // Store values for model regressors
                ev_c[i, bIdx,  t]   = ev[choice[i, bIdx, t]];
                ev_nc[i, bIdx, t]  = ev[3 - choice[i, bIdx, t]];
                pe[i, bIdx, t]     = PE;

                // Update expected value of chosen stimulus
                if (outcome[i, bIdx, t] > 0)
                  ev[choice[i, bIdx, t]] += d_alpha_rew[i] * PE;
                else
                  ev[choice[i, bIdx, t]] += d_alpha_pun[i] * PE;
              }
          }

          if(bIdx == 2){
            for (t in 1:Tsubj[i, bIdx]) {
                // Softmax choice
                log_lik[i]  += categorical_logit_lpmf(choice[i, bIdx, t] | ev * f_beta[i]);

                // generate posterior prediction for current trial
                y_pred[i, bIdx, t] = categorical_rng(softmax(ev * f_beta[i]));

                // Prediction Error
                PE = outcome[i, bIdx, t] - ev[choice[i, bIdx, t]];

                // Store values for model regressors
                ev_c[i, bIdx,  t]   = ev[choice[i, bIdx, t]];
                ev_nc[i, bIdx, t]  = ev[3 - choice[i, bIdx, t]];
                pe[i, bIdx, t]     = PE;

                // Update expected value of chosen stimulus
                if (outcome[i, bIdx, t] > 0)
                  ev[choice[i, bIdx, t]] += f_alpha_rew[i] * PE;
                else
                  ev[choice[i, bIdx, t]] += f_alpha_pun[i] * PE;
              }
          }

          if(bIdx == 3){
            for (t in 1:Tsubj[i, bIdx]) {
                // Softmax choice
                log_lik[i]  += categorical_logit_lpmf(choice[i, bIdx, t] | ev * p_beta[i]);

                // generate posterior prediction for current trial
                y_pred[i, bIdx, t] = categorical_rng(softmax(ev * p_beta[i]));

                // Prediction Error
                PE = outcome[i, bIdx, t] - ev[choice[i, bIdx, t]];

                // Store values for model regressors
                ev_c[i, bIdx,  t]   = ev[choice[i, bIdx, t]];
                ev_nc[i, bIdx, t]  = ev[3 - choice[i, bIdx, t]];
                pe[i, bIdx, t]     = PE;

                // Update expected value of chosen stimulus
                if (outcome[i, bIdx, t] > 0)
                  ev[choice[i, bIdx, t]] += p_alpha_rew[i] * PE;
                else
                  ev[choice[i, bIdx, t]] += p_alpha_pun[i] * PE;
              }
          }

      }


    }
  }
}

