
// ------------------------------------------------------------
// ORDINAL MIRT MODEL — HIERARCHICAL PRIORS + LOGNORMAL α PRIOR
// Updated to include:
// [A] Hierarchical priors for latent traits (mu_theta, sigma_theta)
// [B] Lognormal prior for item discrimination parameters (alpha)
// [C] Informative prior on cutpoints
// ------------------------------------------------------------

data {
  int<lower=1> N;                // Number of respondents
  int<lower=1> K;                // Number of items
  int<lower=1> D;                // Number of latent dimensions
  int<lower=2> C;                // Number of ordinal response categories (e.g., Likert levels)

  array[N, K] int<lower=1, upper=C> Y;   // Response matrix: ordinal ratings (1 to C)
  matrix[K, D] lambda_signs;             // Fixed design matrix of item loadings (-1, 0, +1)
}

parameters {
  // [A] Hierarchical latent trait structure
  vector[D] mu_theta;                    // Trait means per dimension
  vector<lower=0>[D] sigma_theta;        // Trait standard deviations
  matrix[N, D] theta;                    // Latent traits per person

  // [B] Discrimination parameters (positive-only)
  vector<lower=0>[K] alpha;              // Item discrimination (strength of slope)

  // Item difficulty parameters
  vector[K] intercepts;                  // Item intercepts (location on latent scale)

  // [C] Global cutpoints across items (ordered)
  ordered[C - 1] cutpoints;              // Shared ordinal thresholds
}

model {
  // ------------------------------------------------------------
  // PRIORS
  // ------------------------------------------------------------

  // [A] Hierarchical priors for latent traits
  mu_theta ~ normal(0, 1);                            // Mean trait level per dimension
  sigma_theta ~ normal(1, 0.5);                       // Trait spread per dimension
  for (d in 1:D)
    theta[, d] ~ normal(mu_theta[d], sigma_theta[d]); // Individual-level latent traits

  // [B] Lognormal prior for discrimination (positive, skewed)
  alpha ~ lognormal(0, 0.5);                          // Regularization and positive constraint

  // Priors for item intercepts (weakly informative)
  intercepts ~ normal(0, 1);

  // [C] Prior for cutpoints: informative to stabilize estimation
  cutpoints ~ normal([-2, -1, 0, 1, 2, 3], 0.5);

  // ------------------------------------------------------------
  // LIKELIHOOD
  // ------------------------------------------------------------

  for (n in 1:N) {
    for (k in 1:K) {
      // Linear predictor for person n and item k
      real eta = intercepts[k] + alpha[k] * dot_product(lambda_signs[k], theta[n]);

      // Ordinal response modeled via ordered logistic regression
      Y[n, k] ~ ordered_logistic(eta, cutpoints);
    }
  }
}
