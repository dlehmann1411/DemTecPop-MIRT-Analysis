
// ------------------------------------------------------------
// ORDINAL MIRT — EXPLORATORY STRUCTURE WITH REGULARIZED HORSESHOE
// Element-wise RHS on factor loadings (lambda), item-specific thresholds,
// no intercepts and no alpha to avoid scale confounding with lambda.
// ------------------------------------------------------------

data {
  int<lower=1> N;                    // Number of respondents
  int<lower=1> K;                    // Number of items
  int<lower=1> D;                    // Number of latent dimensions (exploratory space, e.g., 15)
  int<lower=2> C;                    // Number of ordered response categories
  array[N, K] int<lower=1, upper=C> Y; // Response matrix (1..C)
}

parameters {
  // Latent traits
  matrix[N, D] theta;                // Person parameters (standard normal prior)

  // Regularized horseshoe for item-by-dimension loadings (element-wise)
  matrix[K, D] z_lambda;             // Non-centered base normals
  vector<lower=0>[D] tau;            // Global (dimension-wise) shrinkage scales
  matrix<lower=0>[K, D] lambda_local;// Local (element-wise) shrinkage scales
  real<lower=0> c2;                  // Slab variance parameter (controls tail heaviness)

  // Item-specific ordered thresholds
  array[K] ordered[C - 1] cutpoints; // Thresholds per item (graded response)
}

transformed parameters {
  matrix[K, D] lambda;               // Actual loadings after shrinkage
  for (k in 1:K) {
    for (d in 1:D) {
      // Regularized horseshoe shrinkage factor (Piironen & Vehtari)
      real lambda_tilde = tau[d] * lambda_local[k, d];
      real shrink = sqrt( c2 * square(lambda_tilde) / (c2 + square(lambda_tilde)) );
      lambda[k, d] = z_lambda[k, d] * shrink;
    }
  }
}

model {
  // -----------------------
  // Priors
  // -----------------------
  to_vector(theta) ~ normal(0, 1);            // Identifies scale of theta
  to_vector(z_lambda) ~ normal(0, 1);         // Non-centered base
  tau ~ cauchy(0, 1);                         // Global (half-Cauchy due to <lower=0>)
  to_vector(lambda_local) ~ cauchy(0, 1);     // Local (half-Cauchy)
  c2 ~ inv_gamma(2, 8);                       // Moderately wide slab

  // Mild priors on item thresholds for stability (optional but recommended)
  for (k in 1:K) cutpoints[k] ~ normal(0, 2);

  // -----------------------
  // Likelihood
  // -----------------------
  {
    matrix[N, K] eta = theta * lambda';       // Person-by-item linear predictor
    for (k in 1:K) {
      for (n in 1:N) {
        Y[n, k] ~ ordered_logistic(eta[n, k], cutpoints[k]);
      }
    }
  }
}
