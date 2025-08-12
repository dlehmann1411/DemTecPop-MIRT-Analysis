
// ------------------------------------------------------------
// ORDINAL MIRT — EXPLORATORY WITH ELEMENT-WISE REGULARIZED HS
// c2 (slab variance) is passed as DATA to enable grid sweeps.
// No intercepts and no alphas; item-specific thresholds.
// ------------------------------------------------------------

data {
  int<lower=1> N;                       // respondents
  int<lower=1> K;                       // items
  int<lower=1> D;                       // latent dimensions (exploratory space)
  int<lower=2> C;                       // ordinal categories
  array[N, K] int<lower=1, upper=C> Y;  // responses (1..C)
  real<lower=0> c2;                     // slab variance (fixed per run; grid-swept in R)
}

parameters {
  matrix[N, D] theta;                   // person parameters
  matrix[K, D] z_lambda;                // non-centered base normals
  vector<lower=0>[D] tau;               // global (dimension-wise) shrinkage
  matrix<lower=0>[K, D] lambda_local;   // local (element-wise) shrinkage
  array[K] ordered[C - 1] cutpoints;    // item-specific thresholds
}

transformed parameters {
  matrix[K, D] lambda;                  // factor loadings after shrinkage
  for (k in 1:K) {
    for (d in 1:D) {
      real lambda_tilde = tau[d] * lambda_local[k, d];
      // Regularized horseshoe shrinkage factor (Piironen & Vehtari)
      real shrink = sqrt( c2 * square(lambda_tilde) / (c2 + square(lambda_tilde)) );
      lambda[k, d] = z_lambda[k, d] * shrink;
    }
  }
}

model {
  // Priors
  to_vector(theta) ~ normal(0, 1);
  to_vector(z_lambda) ~ normal(0, 1);
  tau ~ cauchy(0, 1);                    // half-Cauchy via lower bound
  to_vector(lambda_local) ~ cauchy(0, 1);
  for (k in 1:K) cutpoints[k] ~ normal(0, 2);  // mild stabilization

  // Likelihood
  {
    matrix[N, K] eta = theta * lambda';
    for (k in 1:K) {
      for (n in 1:N) {
        Y[n, k] ~ ordered_logistic(eta[n, k], cutpoints[k]);
      }
    }
  }
}
