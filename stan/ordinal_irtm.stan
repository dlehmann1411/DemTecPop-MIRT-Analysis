
// Ordinal MIRT model with shared cutpoints and theory-driven constraints
data {
  int<lower=1> N;                        // Number of respondents
  int<lower=1> K;                        // Number of items
  int<lower=2> C;                        // Number of ordinal response categories
  int<lower=1> D;                        // Number of latent dimensions
  array[N, K] int<lower=1, upper=C> Y;   // Ordinal response matrix
  matrix[K, D] lambda_signs;             // Theoretical loading constraints: (-1, 0, 1)
}

parameters {
  matrix[K, D] lambda_raw;              // Unconstrained loadings
  vector[K] intercepts;                 // Item difficulties
  matrix[N, D] z;                       // Standard-normal base
  vector[C - 1] raw_cutpoints;          // Raw cutpoints (unordered)
  vector<lower=0>[K] discrim;           // Item discriminations
}

transformed parameters {
  matrix[N, D] theta = z;               //sigma_theta = 1.0
  matrix[K, D] lambda;
  ordered[C - 1] cutpoints;


  // Apply theory-driven constraints to lambda_raw
  for (k in 1:K) {
    for (d in 1:D) {
      lambda[k, d] = lambda_signs[k, d] * lambda_raw[k, d];
    }
  }

  // Sort raw_cutpoints to obtain valid cutpoints
  cutpoints = sort_asc(raw_cutpoints);
}

model {
  // --- Priors ---
  to_vector(lambda_raw) ~ normal(0, 1);
  intercepts ~ normal(0, 2);
  discrim ~ lognormal(0, 0.3);
  to_vector(z) ~ std_normal();

  raw_cutpoints ~ normal(0, 2);  // Weakly informative location
  for (c in 1:(C - 2)) {
    target += normal_lpdf(cutpoints[c + 1] - cutpoints[c] | 1.0, 0.5);  // Penalize overly small/large gaps
  }

  // --- Theory-guided priors on constrained loadings ---
  for (k in 1:K) {
    for (d in 1:D) {
      if (lambda_signs[k, d] == 1)
        lambda_raw[k, d] ~ normal(0.5, 0.25);
      else if (lambda_signs[k, d] == -1)
        lambda_raw[k, d] ~ normal(-0.5, 0.25);
      else
        lambda_raw[k, d] ~ normal(0, 0.05);  // Shrink inactive dimensions
    }
  }

  // --- Likelihood ---
    for (n in 1:N) {
    for (k in 1:K) {
      real eta;
      eta = discrim[k] * (dot_product(lambda[k], theta[n]) + intercepts[k]);
      eta = fmin(fmax(eta, -20), 20);  // Constraint for numerical stability
      Y[n, k] ~ ordered_logistic(eta, cutpoints);
    }
  }
} 


