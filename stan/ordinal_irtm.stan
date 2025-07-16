
data {
  int<lower=1> N;                // Number of respondents
  int<lower=1> K;                // Number of items
  int<lower=1> D;                // Number of latent dimensions
  int<lower=2> C;                // Number of ordinal response categories

  array[N, K] int<lower=1, upper=C> Y;         // Response matrix (ordinal, 1 to C)
  matrix[K, D] lambda_signs;                  // Constraint matrix: fixed design of loadings (-1, 0, +1)
}

parameters {
  // [A] Hierarchical structure for latent traits
  vector[D] mu_theta;                         // Trait means for each dimension
  vector<lower=0>[D] sigma_theta;             // Trait standard deviations
  matrix[N, D] theta;                         // Latent trait scores for each person

  // [B] Discrimination parameters (positive, skewed)
  vector<lower=0>[K] alpha;                   // Item discrimination (strength of item sensitivity)

  // Item location parameters
  vector[K] intercepts;                       // Item intercepts (difficulty)

  // [C] Ordered thresholds (cutpoints) shared across all items
  ordered[C - 1] cutpoints;
}

model {
  // ---------------------------------------------------------------
  // PRIOR DISTRIBUTIONS
  // ---------------------------------------------------------------

  // [A] Hierarchical priors for latent traits
  mu_theta ~ normal(0, 1);                    // Prior on trait means (per dimension)
  sigma_theta ~ normal(1, 0.5);               // Prior on trait variability
  for (d in 1:D)
    theta[, d] ~ normal(mu_theta[d], sigma_theta[d]);  // Individual-level abilities

  // [B] Lognormal prior for discrimination parameters (positive, moderately skewed)
  alpha ~ lognormal(0, 0.5);

  // Weakly informative prior for item intercepts
  intercepts ~ normal(0, 1);

  // [C] Prior for cutpoints (ordered thresholds) — regular and wide
  cutpoints ~ normal([-2, -1, 0, 1, 2, 3], 0.5);  // Helps stabilize estimation

  // ---------------------------------------------------------------
  // LIKELIHOOD
  // ---------------------------------------------------------------

  for (n in 1:N) {
    for (k in 1:K) {
      // Linear predictor for person n and item k
      real eta = intercepts[k] + alpha[k] * dot_product(lambda_signs[k], theta[n]);

      // Ordinal logistic response model
      Y[n, k] ~ ordered_logistic(eta, cutpoints);
    }
  }
}
