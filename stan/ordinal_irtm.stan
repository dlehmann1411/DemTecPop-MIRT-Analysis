
data {
  int<lower=1> N;                // Number of respondents
  int<lower=1> K;                // Number of items
  int<lower=1> D;                // Number of latent dimensions
  int<lower=2> C;                // Number of ordinal response categories

  array[N, K] int<lower=1, upper=C> Y;         // Response matrix (ordinal, 1 to C)
  matrix[K, D] lambda_signs;                  // Constraint matrix: fixed design of loadings (-1, 0, +1)
}

parameters {
  // Latent trait estimates (abilities): one D-dimensional vector per respondent
  matrix[N, D] theta;

  // Item discriminations (sensitivity)
  vector[K] alpha;

  // Item difficulty parameters
  vector[K] intercepts;

  // Ordered thresholds shared across all items
  ordered[C - 1] cutpoints;
}

model {
  // ---------------------------------------------------------------
  // PRIORS
  // ---------------------------------------------------------------

  // Standard normal priors on latent abilities
  to_vector(theta) ~ normal(0, 1);

  // Prior for item discriminations: centered at 1
  alpha ~ normal(1, 0.5);

  // Prior for item intercepts
  intercepts ~ normal(0, 1);

  // Prior for cutpoints (ordered thresholds)
  cutpoints ~ normal(0, 1);

  // ---------------------------------------------------------------
  // LIKELIHOOD
  // ---------------------------------------------------------------

  for (n in 1:N) {
    for (k in 1:K) {
      real eta = intercepts[k] + alpha[k] * dot_product(lambda_signs[k], theta[n]);
      Y[n, k] ~ ordered_logistic(eta, cutpoints);
    }
  }
}
