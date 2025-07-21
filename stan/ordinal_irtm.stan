
// ------------------------------------------------------------
// ORDINAL MIRT MODEL — SIMPLE PRIOR STRUCTURE FOR STABLE VI
// ------------------------------------------------------------

data {
  int<lower=1> N;                // Number of respondents
  int<lower=1> K;                // Number of items
  int<lower=1> D;                // Number of latent dimensions
  int<lower=2> C;                // Number of ordinal response categories

  array[N, K] int<lower=1, upper=C> Y;   // Response matrix (ordinal, 1 to C)
  matrix[K, D] lambda_signs;             // Fixed loading design (-1, 0, +1)
}

parameters {
  // Latent traits: theta_n for each respondent across D dimensions
  matrix[N, D] theta;

  // Discrimination (slope) parameters for each item
  vector<lower=0>[K] alpha;

  // Intercepts (difficulty/location) for each item
  vector[K] intercepts;

  // Ordered cutpoints (shared across items)
  ordered[C - 1] cutpoints;
}

model {
  // ------------------------------------------------------------
  // PRIORS — flat & VI-friendly
  // ------------------------------------------------------------

  // Latent abilities: standard normal priors
  to_vector(theta) ~ normal(0, 1);

  // Discrimination parameters: mildly informative around α = 1
  alpha ~ normal(1, 0.5);

  // Item intercepts: centered at 0, weakly informative
  intercepts ~ normal(0, 1);

  // Thresholds for ordinal response categories
  cutpoints ~ normal(0, 1);

  // ------------------------------------------------------------
  // LIKELIHOOD
  // ------------------------------------------------------------
  for (n in 1:N) {
    for (k in 1:K) {
      real eta = intercepts[k] + alpha[k] * dot_product(lambda_signs[k], theta[n]);
      Y[n, k] ~ ordered_logistic(eta, cutpoints);
    }
  }
}
