
# simulate_and_fit_vi.R
# --------------------------------------------------------------
# Simulates data under an ordinal MIRT model and fits it using
# variational inference with CmdStanR and a custom Stan model.
# --------------------------------------------------------------

# Load required packages
library(tidyverse)
library(cmdstanr)
library(posterior)
library(here)

# --------------------------------------------------------------
# Step 1: Simulation settings
# --------------------------------------------------------------
set.seed(123)

N <- 500     # Number of persons
K <- 10      # Number of items
D <- 3       # Number of latent dimensions
C <- 5       # Number of ordinal response categories

# --------------------------------------------------------------
# Step 2: Simulate fixed model parameters
# --------------------------------------------------------------
# Latent traits (abilities)
theta_true <- matrix(rnorm(N * D, 0, 1), nrow = N, ncol = D)

# Discrimination parameters
alpha_true <- rlnorm(K, log(1), 0.3)  # Log-normal around 1

# Item intercepts (difficulty)
intercepts_true <- rnorm(K, 0, 1)

# Cutpoints (common for all items)
cutpoints_true <- c(-1.5, -0.5, 0.5, 1.5)  # Ordered thresholds

# Lambda loading matrix (fixed design)
lambda_signs <- matrix(0, nrow = K, ncol = D)
for (k in 1:K) {
  lambda_signs[k, ((k - 1) %% D) + 1] <- 1
}

# --------------------------------------------------------------
# Step 3: Generate response data
# --------------------------------------------------------------
Y <- matrix(NA, nrow = N, ncol = K)

for (n in 1:N) {
  for (k in 1:K) {
    eta <- intercepts_true[k] + alpha_true[k] * sum(lambda_signs[k, ] * theta_true[n, ])
    probs <- c(
      plogis(cutpoints_true[1] - eta),
      diff(plogis(cutpoints_true - eta)),
      1 - plogis(cutpoints_true[length(cutpoints_true)] - eta)
    )
    Y[n, k] <- sample(1:C, 1, prob = probs)
  }
}

# --------------------------------------------------------------
# Step 4: Prepare data list for Stan
# --------------------------------------------------------------
stan_data <- list(
  N = N,
  K = K,
  D = D,
  C = C,
  Y = Y,
  lambda_signs = lambda_signs
)

# --------------------------------------------------------------
# Step 5: Compile and run the Stan model using variational inference
# --------------------------------------------------------------
model <- cmdstan_model(here("stan", "ordinal_irtm_vi.stan"))

fit_vi <- model$variational(
  data = stan_data,
  iter = 10000,
  grad_samples = 1,
  elbo_samples = 100,
  output_samples = 1000,
  seed = 123
)

# --------------------------------------------------------------
# Step 6: Extract and inspect posterior samples
# --------------------------------------------------------------
draws <- fit_vi$draws(format = "df")

# Posterior mean estimates of theta
theta_draws <- draws %>% select(starts_with("theta"))
theta_est <- theta_draws %>% summarise(across(everything(), mean))

# View first few estimates
print(head(theta_est))
