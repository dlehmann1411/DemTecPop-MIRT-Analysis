
// ------------------------------------------------------------
// ORDINAL MIRT MODEL — EXPLORATORY STRUCTURE WITH RHS PRIOR
// ------------------------------------------------------------

// This model estimates a multidimensional IRT model for ordinal responses,
// using a regularized horseshoe prior on item loadings to induce sparsity
// and enable data-driven dimensionality discovery.
// Suitable for exploratory MIRT with unknown or partially known structure.
//
// Key idea: Rather than fixing which item loads on which factor, we estimate
// all loadings and use shrinkage to identify the relevant dimensions.

// ------------------------------------------------------------

data {
  int<lower=1> N;                // Number of respondents
  int<lower=1> K;                // Number of items
  int<lower=1> D;                // Number of latent dimensions (exploratory space)
  int<lower=2> C;                // Number of ordinal response categories (e.g., Likert scale with 7 levels)

  array[N, K] int<lower=1, upper=C> Y;   // Response matrix (ordinal responses coded from 1 to C)
}

parameters {
  matrix[N, D] theta;                   // Latent traits: respondent positions on D latent dimensions
  vector<lower=0>[K] alpha;             // Discrimination parameters: item slopes (positive-only)
  vector[K] intercepts;                // Item intercepts: baseline difficulty/location parameters
  ordered[C - 1] cutpoints;            // Shared category thresholds for ordered logistic responses

  // Exploratory item loading structure with shrinkage priors
  matrix[K, D] lambda_raw;             // Free loadings: each item may load on each dimension
  real<lower=0> tau;                   // Global shrinkage parameter: how aggressively we shrink all loadings
  vector<lower=0>[K] lambda_local;     // Local shrinkage: item-specific relevance weights
}

model {
  // ------------------------------------------------------------
  // PRIORS — Weakly informative priors for stability under VI
  // ------------------------------------------------------------

  // Latent traits (theta): standard normal prior across respondents
  to_vector(theta) ~ normal(0, 1);

  // Discrimination (alpha): weakly informative prior centered at 1
  alpha ~ normal(1, 0.5);

  // Intercepts and cutpoints: weakly centered around 0
  intercepts ~ normal(0, 1);
  cutpoints ~ normal(0, 1);

  // ------------------------------------------------------------
  // REGULARIZED HORSESHOE PRIORS ON LOADINGS (lambda_raw)
  // ------------------------------------------------------------

  // Global and local shrinkage hyperpriors
  tau ~ cauchy(0, 1);                   // Encourages global sparsity
  lambda_local ~ cauchy(0, 1);          // Allows exceptions for each item

  // Hierarchical shrinkage on each item-dimension loading
  for (k in 1:K) {
    for (d in 1:D) {
      lambda_raw[k, d] ~ normal(0, tau * lambda_local[k]);
    }
  }

  // ------------------------------------------------------------
  // LIKELIHOOD — Ordered logistic formulation
  // ------------------------------------------------------------

  matrix[N, K] eta;
  
  // 1. Compute latent predictor: respondent traits × item loadings
  eta = theta * lambda_raw';            // Matrix multiplication: N x D * D x K → N x K

  // 2. Apply item discrimination weights
  eta = eta .* rep_matrix(alpha', N);   // Elementwise multiplication: scale each column/item by alpha[k]

  // 3. Add item-specific intercepts
  eta = eta + rep_matrix(intercepts', N);

  // 4. Likelihood: ordinal response model per item per respondent
  for (n in 1:N) {
    for (k in 1:K) {
      Y[n, k] ~ ordered_logistic(eta[n, k], cutpoints);
    }
  }
}
