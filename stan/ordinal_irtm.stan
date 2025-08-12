
// ------------------------------------------------------------
// ORDINAL MIRT — EXPLORATORY WITH ELEMENT-WISE REGULARIZED HS
// - c2 (slab variance) is passed as DATA to enable grid sweeps
// - Element-wise regularized horseshoe on loadings (lambda)
// - No intercepts and no alphas; item-specific thresholds
// - Thresholds reparameterized as base + positive gaps (stable)
// ------------------------------------------------------------

data {
  int<lower=1> N;                       // respondents
  int<lower=1> K;                       // items
  int<lower=1> D;                       // latent dimensions (exploratory space)
  int<lower=2> C;                       // number of ordinal categories
  array[N, K] int<lower=1, upper=C> Y;  // responses (coded 1..C)
  real<lower=0> c2;                     // slab variance (fixed per run; grid-swept in R)
}

parameters {
  // Person and loading parameters
  matrix[N, D] theta;                   // person parameters (latent traits)
  matrix[K, D] z_lambda;                // non-centered base normals for loadings
  vector<lower=0>[D] tau;               // global (dimension-wise) shrinkage
  matrix<lower=0>[K, D] lambda_local;   // local (element-wise) shrinkage

  // Thresholds (reparameterized):
  //   cutpoints_k = [cut_base_k, cut_base_k + gap_k1, cut_base_k + gap_k1 + gap_k2, ...]
  vector[K] cut_base;                              // first threshold per item
  matrix<lower=1e-6>[K, C - 2] gap;                // strictly positive gaps between thresholds
}

transformed parameters {
  // Factor loadings after regularized horseshoe shrinkage
  matrix[K, D] lambda;
  for (k in 1:K) {
    for (d in 1:D) {
      real lambda_tilde = tau[d] * lambda_local[k, d];
      // Regularized horseshoe shrinkage factor (Piironen & Vehtari)
      real shrink = sqrt( c2 * square(lambda_tilde) / (c2 + square(lambda_tilde)) );
      lambda[k, d] = z_lambda[k, d] * shrink;
    }
  }

  // Build ordered thresholds from base + positive gaps
  array[K] vector[C - 1] cutpoints;
  for (k in 1:K) {
    cutpoints[k][1] = cut_base[k];
    for (c in 2:(C - 1)) {                // <-- Stan loop syntax
      // strictly increasing by construction (gap > 0)
      cutpoints[k][c] = cutpoints[k][c - 1] + gap[k, c - 1];
    }
  }
}

model {
  // -----------------------
  // Priors
  // -----------------------
  // Person parameters (identify scale of theta)
  to_vector(theta) ~ normal(0, 1);

  // Non-centered base for loadings
  to_vector(z_lambda) ~ normal(0, 1);

  // Regularized horseshoe scales
  tau ~ cauchy(0, 1);                        // global (half-Cauchy via lower bound)
  to_vector(lambda_local) ~ cauchy(0, 1);    // local (half-Cauchy)

  // Threshold priors (stable, encourage reasonable spacing)
  cut_base ~ normal(0, 1.5);                 // location of first threshold per item
  to_vector(gap) ~ lognormal(log(1.0), 0.35);// positive gaps; median ~ 1

  // -----------------------
  // Likelihood
  // -----------------------
  {
    matrix[N, K] eta = theta * lambda';      // person-by-item linear predictor
    for (k in 1:K) {
      for (n in 1:N) {
        Y[n, k] ~ ordered_logistic(eta[n, k], cutpoints[k]);
      }
    }
  }
}
