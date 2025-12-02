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
  vector[2] initsoftMaxIntermed;
  initV = rep_vector(0.0, 2);
  initsoftMaxIntermed = rep_vector(0.0, 2);
}

// Declare all parameters as vectors for vectorizing
parameters {
  // Hyper(group)-parameters
  vector[6] mu_pr;
  vector<lower=0>[6] sigma;

  // Subject-level raw parameters (for Matt trick)
  vector[N] e_alpha_pr;   // learning rate
  vector[N] e_beta_pr;   // inverse temperature
  vector[N] e_omega_pr; //stickiness

  vector[N] p_alpha_pr;   // learning rate
  vector[N] p_beta_pr;   // inverse temperature
  vector[N] p_omega_pr; //stickiness
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0, upper=1>[N] e_alpha;
  vector<lower=0, upper=10>[N] e_beta;
  vector<lower=-5, upper=5>[N] e_omega;

  vector<lower=0, upper=1>[N] p_alpha;
  vector<lower=0, upper=10>[N] p_beta;
  vector<lower=-5, upper=5>[N] p_omega;

  for (i in 1:N) {
    e_alpha[i]  = Phi_approx(mu_pr[1] + sigma[1] * e_alpha_pr[i]);
    e_beta[i]  = Phi_approx(mu_pr[2] + sigma[2] * e_beta_pr[i]) * 10;
    e_omega[i] = (Phi_approx(mu_pr[3] + sigma[3] * e_omega_pr[i])-0.5)*10; 

    p_alpha[i]  = Phi_approx(mu_pr[4] + sigma[4] * p_alpha_pr[i]);
    p_beta[i]  = Phi_approx(mu_pr[5] + sigma[5] * p_beta_pr[i]) * 10;
    p_omega[i] = (Phi_approx(mu_pr[6] + sigma[6] * p_omega_pr[i])-0.5)*10; 
  }
}

model {
  // Hyperparameters
  mu_pr  ~ normal(0, 1);
  sigma ~ normal(0, 0.2);

  // individual parameters
  e_alpha_pr ~ normal(0, 1);
  e_beta_pr ~ normal(0, 1);
  e_omega_pr ~ normal(0, 1);

  p_alpha_pr ~ normal(0, 1);
  p_beta_pr ~ normal(0, 1);
  p_omega_pr ~ normal(0, 1);

  for (i in 1:N) {
    for (bIdx in 1:Bsubj[i]) {  // new
      // Define Values
      vector[2] ev;   // Expected value
      real PE;        // prediction error
      real last_choice;   //choice on last trial
      vector[2] softMaxIntermed;  // V't(c) 
      
      // Initialize values
      ev = initV;     // initial ev values
      softMaxIntermed = initsoftMaxIntermed; //initialise V't(c)
      last_choice = 0; // initial last_choice

      if (bIdx == 1) {
          for (t in 1:(Tsubj[i, bIdx])) {  // new
            //stickiness
            if (last_choice == 1) {
                  softMaxIntermed[1] = ev[1] + e_omega[i];
                  softMaxIntermed[2] = ev[2];}
            else {
                  softMaxIntermed[2] = ev[2] + e_omega[i];
                  softMaxIntermed[1] = ev[1];}

            // Softmax choice
            choice[i, bIdx, t] ~ categorical_logit(softMaxIntermed * e_beta[i]);
            last_choice = choice[i, bIdx, t];

            // Prediction Error
            PE = outcome[i, bIdx, t] - ev[choice[i, bIdx, t]];

            // Update expected value of chosen stimulus
            ev[choice[i, bIdx, t]] += e_alpha[i] * PE;
          }
      }

      if (bIdx == 2) {
          for (t in 1:(Tsubj[i, bIdx])) {  // new
            //stickiness
            if (last_choice == 1) {
                  softMaxIntermed[1] = ev[1] + e_omega[i];
                  softMaxIntermed[2] = ev[2];}
            else {
                  softMaxIntermed[2] = ev[2] + e_omega[i];
                  softMaxIntermed[1] = ev[1];}

            // Softmax choice
            choice[i, bIdx, t] ~ categorical_logit(softMaxIntermed * e_beta[i]);
            last_choice = choice[i, bIdx, t];

            // Prediction Error
            PE = outcome[i, bIdx, t] - ev[choice[i, bIdx, t]];

            // Update expected value of chosen stimulus
            ev[choice[i, bIdx, t]] += e_alpha[i] * PE;
          }
      }

      if (bIdx == 3) {
          for (t in 1:(Tsubj[i, bIdx])) {  // new
            //stickiness
            if (last_choice == 1) {
                  softMaxIntermed[1] = ev[1] + p_omega[i];
                  softMaxIntermed[2] = ev[2];}
            else {
                  softMaxIntermed[2] = ev[2] + p_omega[i];
                  softMaxIntermed[1] = ev[1];}

            // Softmax choice
            choice[i, bIdx, t] ~ categorical_logit(softMaxIntermed * p_beta[i]);
            last_choice = choice[i, bIdx, t];

            // Prediction Error
            PE = outcome[i, bIdx, t] - ev[choice[i, bIdx, t]];

            // Update expected value of chosen stimulus
            ev[choice[i, bIdx, t]] += p_alpha[i] * PE;
          }
      }
    }
  }
}

generated quantities {
  // For group level parameters
  real<lower=0, upper=1>  mu_e_alpha;
  real<lower=0, upper=10> mu_e_beta;
  real<lower=-5, upper=5> mu_e_omega;

  real<lower=0, upper=1>  mu_p_alpha;
  real<lower=0, upper=10> mu_p_beta;
  real<lower=-5, upper=5> mu_p_omega;

  // For log likelihood calculation
  array[N] real log_lik;

  // For model regressors
  array[N, B, T] real ev_c;   // Expected value of the chosen option
  array[N, B, T] real ev_nc;  // Expected value of the non-chosen option
  array[N, B, T] real pe;     // Prediction error
  real last_choice; 

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

  mu_e_alpha = Phi_approx(mu_pr[1]);
  mu_e_beta = Phi_approx(mu_pr[2]) * 10;
  mu_e_omega = (Phi_approx(mu_pr[3])-0.5)*10;

  mu_p_alpha = Phi_approx(mu_pr[4]);
  mu_p_beta = Phi_approx(mu_pr[5]) * 10;
  mu_p_omega = (Phi_approx(mu_pr[6])-0.5)*10;

  { // local section, this saves time and space
    for (i in 1:N) {
      log_lik[i]=0;
      for (bIdx in 1:Bsubj[i]) {
        // Define values
        vector[2] ev; // Expected value
        vector[2] softMaxIntermed; // V't(c)
        real PE;      // Prediction error

        // Initialize values
        ev = initV; // initial ev values
        softMaxIntermed = initsoftMaxIntermed;
        last_choice = 0;

        if (bIdx == 1) {
            for (t in 1:(Tsubj[i, bIdx])) {
              //stickiness
              if (last_choice == 1) {
                  softMaxIntermed[1] = ev[1] + e_omega[i];
                  softMaxIntermed[2] = ev[2];}
              else {
                  softMaxIntermed[2] = ev[2] + e_omega[i];
                  softMaxIntermed[1] = ev[1];}

              // Softmax choice
              log_lik[i]  += categorical_logit_lpmf(choice[i, bIdx, t] | softMaxIntermed * e_beta[i]);

              // generate posterior prediction for current trial
              y_pred[i, bIdx, t] = categorical_rng(softmax(softMaxIntermed * e_beta[i]));

              // Prediction Error
              PE = outcome[i, bIdx, t] - ev[choice[i, bIdx, t]];

              // Store values for model regressors
              ev_c[i, bIdx, t]   = ev[choice[i, bIdx, t]];
              ev_nc[i, bIdx, t]  = ev[3 - choice[i, bIdx, t]];
              pe[i, bIdx, t]     = PE;

              // Update expected value of chosen stimulus
              ev[choice[i, bIdx, t]] += e_alpha[i] * PE;
              last_choice = choice[i, bIdx, t];
            }
        }

        if (bIdx == 2) {
            for (t in 1:(Tsubj[i, bIdx])) {
              //stickiness
              if (last_choice == 1) {
                  softMaxIntermed[1] = ev[1] + e_omega[i];
                  softMaxIntermed[2] = ev[2];}
              else {
                  softMaxIntermed[2] = ev[2] + e_omega[i];
                  softMaxIntermed[1] = ev[1];}

              // Softmax choice
              log_lik[i]  += categorical_logit_lpmf(choice[i, bIdx, t] | softMaxIntermed * e_beta[i]);

              // generate posterior prediction for current trial
              y_pred[i, bIdx, t] = categorical_rng(softmax(softMaxIntermed * e_beta[i]));

              // Prediction Error
              PE = outcome[i, bIdx, t] - ev[choice[i, bIdx, t]];

              // Store values for model regressors
              ev_c[i, bIdx, t]   = ev[choice[i, bIdx, t]];
              ev_nc[i, bIdx, t]  = ev[3 - choice[i, bIdx, t]];
              pe[i, bIdx, t]     = PE;

              // Update expected value of chosen stimulus
              ev[choice[i, bIdx, t]] += e_alpha[i] * PE;
              last_choice = choice[i, bIdx, t];
            }
        }

        if (bIdx == 3) {
            for (t in 1:(Tsubj[i, bIdx])) {
              //stickiness
              if (last_choice == 1) {
                  softMaxIntermed[1] = ev[1] + p_omega[i];
                  softMaxIntermed[2] = ev[2];}
              else {
                  softMaxIntermed[2] = ev[2] + p_omega[i];
                  softMaxIntermed[1] = ev[1];}

              // Softmax choice
              log_lik[i]  += categorical_logit_lpmf(choice[i, bIdx, t] | softMaxIntermed * p_beta[i]);

              // generate posterior prediction for current trial
              y_pred[i, bIdx, t] = categorical_rng(softmax(softMaxIntermed * p_beta[i]));

              // Prediction Error
              PE = outcome[i, bIdx, t] - ev[choice[i, bIdx, t]];

              // Store values for model regressors
              ev_c[i, bIdx, t]   = ev[choice[i, bIdx, t]];
              ev_nc[i, bIdx, t]  = ev[3 - choice[i, bIdx, t]];
              pe[i, bIdx, t]     = PE;

              // Update expected value of chosen stimulus
              ev[choice[i, bIdx, t]] += p_alpha[i] * PE;
              last_choice = choice[i, bIdx, t];
            }
        }

      }
    }
  }
}

