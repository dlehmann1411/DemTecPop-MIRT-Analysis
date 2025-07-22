
// ------------------------------------------------------------
// ORDINAL MIRT MODEL — VECTORIZED FOR PERFORMANCE (VI-Optimized)
// ------------------------------------------------------------

data {
  int<lower=1> N;                // Number of respondents
  int<lower=1> K;                // Number of items
  int<lower=1> D;                // Number of latent dimensions
  int<lower=2> C;                // Number of ordinal response categories

  array[N, K] int<lower=1, upper=C> Y;   // Response matrix (ordinal, 1 to C)
  matrix[K, D] lambda_signs;             // Fixed loading structure (-1, 0, +1)
}

parameters {
  // Latent traits: one vector per respondent
  matrix[N, D] theta;

  // Item discrimination (slopes)
  vector<lower=0>[K] alpha;

  // Item intercepts (locations)
  vector[K] intercepts;

  // Ordered category thresholds (shared across items)
  ordered[C - 1] cutpoints;
}

model {
  // ------------------------------------------------------------
  // PRIORS — flat & VI-friendly
  // ------------------------------------------------------------

  // Standard normal priors for latent traits
  to_vector(theta) ~ normal(0, 1);

  // Mildly informative priors for discrimination parameters (centered on 1)
  alpha ~ normal(1, 0.5);

  // Weakly informative priors for item intercepts
  intercepts ~ normal(0, 1);

  // Shared thresholds for ordinal responses
  cutpoints ~ normal(0, 1);

  // ------------------------------------------------------------
  // LIKELIHOOD — fully vectorized eta computation
  // ------------------------------------------------------------

  matrix[N, K] eta;
  eta = theta * lambda_signs';                    // Matrix product: N × D × D × K → N × K
  eta = eta .* rep_matrix(alpha', N);             // Scale each item (column-wise)
  eta = eta + rep_matrix(intercepts', N);         // Add intercepts (broadcast across rows)

  // Apply the ordered logistic model to each response
  for (n in 1:N) {
    for (k in 1:K) {
      Y[n, k] ~ ordered_logistic(eta[n, k], cutpoints);
    }
  }
}
