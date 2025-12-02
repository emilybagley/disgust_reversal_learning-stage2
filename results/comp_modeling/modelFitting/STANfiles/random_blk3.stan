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


model {
    for (i in 1:N) {
      for (bIdx in 1:Bsubj[i]) {  // new
        for (t in 1:Tsubj[i, bIdx]) {
        choice[i, bIdx, t] ~ categorical_logit(rep_vector(0.0, 2));
    }
  }
}
}

generated quantities {
  // For log likelihood calculation
  array[N] real log_lik;

  // For posterior predictive check
  array[N, B, T] real y_pred;

  // Initialize all the variables to avoid NULL values
  for (i in 1:N) {
    for (bIdx in 1:Bsubj[i]) {  // new
      for (t in 1:T) {
        y_pred[i, bIdx, t]    = -1;
      }
    }
  }

  { // local section, this saves time and space
    for (i in 1:N) {
      // Initialize values
      log_lik[i] = 0;

      for (bIdx in 1:Bsubj[i]) {
        for (t in 1:Tsubj[i, bIdx]) {
            // Softmax choice
            log_lik[i]  += categorical_logit_lpmf(choice[i, bIdx, t] | rep_vector(0.0, 2));

            // generate posterior prediction for current trial
            y_pred[i, bIdx, t] = categorical_rng(softmax(rep_vector(0.0, 2)));
        }
      }
    }
  }
}

