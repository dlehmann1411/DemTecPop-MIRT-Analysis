




data {
  int<lower=1> N;                // Number of respondents (persons)
  int<lower=1> K;                // Number of items
  int<lower=1> D;                // Number of latent dimensions (traits)
  int<lower=2> C;                // Number of ordinal response categories (e.g., Likert scale levels)
  
  // Response matrix: each entry Y[n,k] is a categorical response in {1, ..., C}
  int<lower=1, upper=C> Y[N, K]; 
  
  // Fixed factor loading design: a matrix with entries in {-1, 0, +1}, 
  // specifying how each item loads on each dimension (Morucci et al. style)
  matrix[K, D] lambda_signs;
}

parameters {
  // Latent trait estimates (abilities): one D-dimensional vector per respondent
  matrix[N, D] theta;

  // Item discriminations: how strongly each item responds to the latent trait(s)
  vector[K] alpha;

  // Item intercepts (difficulty or location parameters)
  vector[K] intercepts;

  // Common threshold vector (cutpoints) used across all items
  // Must be ordered to ensure identifiability of the ordinal scale
  ordered[C - 1] cutpoints;
}

model {
  // ---------------------------------------------------------------
  // PRIOR DISTRIBUTIONS
  // ---------------------------------------------------------------

  // Standard normal prior on abilities: each person has D traits
  to_vector(theta) ~ normal(0, 1);

  // Shrink discriminations to 1: normal prior centered on typical IRT scale
  alpha ~ normal(1, 0.5);

  // Weakly informative prior for item intercepts (difficulty/location)
  intercepts ~ normal(0, 1);

  // Prior on threshold parameters: shared across all items (global scale)
  cutpoints ~ normal(0, 1);

  // ---------------------------------------------------------------
  // LIKELIHOOD
  // ---------------------------------------------------------------

  for (n in 1:N) {         // Loop over persons
    for (k in 1:K) {       // Loop over items

      // Compute linear predictor (eta) for respondent n on item k
      // Using fixed loadings (lambda_signs), alpha[k] as discrimination,
      // and intercepts[k] as item difficulty/location
      real eta = intercepts[k] + alpha[k] * dot_product(lambda_signs[k], theta[n]);

      // Observed response follows ordered logistic likelihood
      Y[n, k] ~ ordered_logistic(eta, cutpoints);
    }
  }
}
