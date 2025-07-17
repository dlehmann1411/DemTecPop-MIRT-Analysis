
// ------------------------------------------------------------
// ORDINAL MIRT MODEL — HIERARCHICAL PRIORS + LOGNORMAL α PRIOR
// ------------------------------------------------------------
// This model estimates a multidimensional IRT model for ordinal data.
// It includes:
// [A] Hierarchical structure on latent traits (theta) via mu and sigma
// [B] Lognormal priors for positive item discrimination parameters (alpha)
// [C] Weakly informative priors on item intercepts (difficulty)
// [D] Informative priors on shared ordered cutpoints (thresholds)
// ------------------------------------------------------------

data {
  int<lower=1> N;                // Number of respondents
  int<lower=1> K;                // Number of items
  int<lower=1> D;                // Number of latent dimensions
  int<lower=2> C;                // Number of ordinal response categories (e.g., 1–7 Likert)

  array[N, K] int<lower=1, upper=C> Y;   // Response matrix: ordinal answers [1,...,C]
  matrix[K, D] lambda_signs;             // Design matrix: fixed loadings per item (values -1, 0, +1)
}

parameters {
  // [A] Hierarchical latent trait structure
  vector[D] mu_theta;                    // Mean of each latent trait (dimension-specific)
  vector<lower=0>[D] sigma_theta;        // Standard deviation of each trait (controls variance)
  matrix[N, D] theta;                    // Latent traits for each person (N x D)

  // [B] Discrimination parameters
  vector<lower=0>[K] alpha;              // Item slopes: sensitivity to underlying traits

  // [C] Item intercepts (difficulty or location on latent scale)
  vector[K] intercepts;

  // [D] Ordered thresholds (cutpoints) shared across all items
  ordered[C - 1] cutpoints;              // Category boundaries (must be increasing)
}

model {
  // ------------------------------------------------------------
  // PRIOR DISTRIBUTIONS
  // ------------------------------------------------------------

  // [A] Priors for trait means and standard deviations
  mu_theta ~ normal(0, 1);                            // Weakly informative prior on latent means
  sigma_theta ~ normal(1, 0.5);                       // Prior for trait spread (positive)
  for (d in 1:D)
    theta[, d] ~ normal(mu_theta[d], sigma_theta[d]); // Individual-level latent traits

  // [B] Prior for item discriminations
  alpha ~ lognormal(0, 0.5);                          // Skewed, positive prior to avoid negative slopes

  // [C] Prior for item difficulties
  intercepts ~ normal(0, 1);                          // Centered around 0, weakly informative

  // [D] Prior for ordered thresholds
  cutpoints ~ normal([-2, -1, 0, 1, 2, 3], 0.5);       // Informative: helps separate response categories

  // ------------------------------------------------------------
  // LIKELIHOOD
  // ------------------------------------------------------------

  for (n in 1:N) {
    for (k in 1:K) {
      // Linear predictor for respondent n on item k:
      // intercept + discrimination-weighted sum over dimension-specific traits
      real eta = intercepts[k] + alpha[k] * dot_product(lambda_signs[k], theta[n]);

      // Observed ordinal response is modeled via ordered logistic regression
      Y[n, k] ~ ordered_logistic(eta, cutpoints);
    }
  }
}
