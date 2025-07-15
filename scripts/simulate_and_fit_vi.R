
# batch_simulate_and_fit_vi.R
# --------------------------------------------------------------
# BATCHED Simulation + Recovery Evaluation for Ordinal MIRT VI
# --------------------------------------------------------------
# This script performs a systematic batch simulation study of an ordinal MIRT
# (Multidimensional Item Response Theory) model estimated via Variational Inference (VI).
#
# For each combination of the following parameters:
#   - N: Number of respondents (sample size)
#   - K: Number of items
#   - alpha_sd: Standard deviation of item discriminations (α)
#   - cutpoint_sets: Thresholds for ordinal categories (narrow vs. wide)
#   - D: Number of latent dimensions (traits)
#
# it:
#   1. Simulates a dataset with known parameters
#   2. Fits the model using VI
#   3. Computes the recovery correlation between true and estimated theta values
#   4. Saves the results (posterior, plots, summary) for each run
#   5. Logs all results into a master summary CSV
#
# The output is stored in: /results/batch_<timestamp>/
# The key file for analysis is: batch_summary.csv

library(tidyverse)
library(cmdstanr)
library(posterior)
library(here)

# --------------------------------------------------------------
# Step 1: Define Batch Grid of Parameter Settings
# --------------------------------------------------------------
N_vals <- c(500, 1000, 5000, 10000)
K_vals <- c(18)
alpha_sds <- c(0.1, 0.3, 0.5)
cutpoint_sets <- list(
  narrow = c(-1.5, -1, -0.5, 0, 0.5, 1),
  wide = c(-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)
)
D_vals <- c(3, 4)

# Timestamp for tracking
batch_id <- format(Sys.time(), "%Y%m%d_%H%M")

# Output folder
out_dir <- here("results", paste0("batch_", batch_id))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Logging data frame
log_results <- tibble()

# --------------------------------------------------------------
# Step 2: Begin Batch Loop
# --------------------------------------------------------------
for (N in N_vals) {
  for (K in K_vals) {
    for (alpha_sd in alpha_sds) {
      for (cp_name in names(cutpoint_sets)) {
        for (D in D_vals) {
          
          C <- 7
          cutpoints_true <- cutpoint_sets[[cp_name]]
          
          # Simulate true parameters
          theta_true <- matrix(rnorm(N * D), nrow = N, ncol = D)
          alpha_true <- rlnorm(K, log(1), alpha_sd)
          intercepts_true <- rnorm(K, 0, 1)
          
          lambda_signs <- matrix(0, nrow = K, ncol = D)
          for (d in 1:D) {
            row_start <- floor((d - 1) * K / D) + 1
            row_end <- floor(d * K / D)
            lambda_signs[row_start:row_end, d] <- 1
          }
          
          ordered_logistic_probs <- function(eta, cutpoints) {
            logits <- c(-Inf, cutpoints, Inf)
            cdfs <- plogis(logits - eta)
            diff(cdfs)
          }
          
          Y <- matrix(NA, nrow = N, ncol = K)
          for (n in 1:N) {
            for (k in 1:K) {
              eta <- intercepts_true[k] + alpha_true[k] * sum(lambda_signs[k, ] * theta_true[n, ])
              probs <- ordered_logistic_probs(eta, cutpoints_true)
              Y[n, k] <- sample(1:C, 1, prob = probs)
            }
          }
          
          stan_data <- list(N = N, K = K, D = D, C = C, Y = Y, lambda_signs = lambda_signs)
          
          model <- cmdstan_model(here("stan", "ordinal_irtm.stan"))
          
          fit_vi <- model$variational(
            data = stan_data,
            iter = 10000,
            grad_samples = 1,
            elbo_samples = 100,
            output_samples = 1000,
            seed = 123
          )
          
          draws <- fit_vi$draws(format = "df")
          theta_draws <- draws %>% select(starts_with("theta"))
          theta_est <- theta_draws %>% summarise(across(everything(), mean))
          
          theta_est_vector <- as.numeric(theta_est)
          theta_true_vector <- as.numeric(theta_true)
          correlation <- cor(theta_true_vector, theta_est_vector)
          
          sim_id <- paste0("N", N, "_K", K, "_sd", alpha_sd, "_", cp_name, "_D", D)
          
          # Save results
          write_csv(theta_est, file.path(out_dir, paste0("theta_est_", sim_id, ".csv")))
          write_csv(data.frame(correlation = correlation), file.path(out_dir, paste0("summary_", sim_id, ".csv")))
          
          plot <- qplot(theta_true_vector, theta_est_vector) +
            labs(title = paste0("Recovery: ", sim_id),
                 x = "True Theta", y = "Estimated Theta (mean VI)") +
            theme_minimal()
          
          ggsave(file.path(out_dir, paste0("plot_", sim_id, ".png")), plot, bg = "white")
          
          # Append to log
          log_results <- bind_rows(log_results, tibble(
            N = N,
            K = K,
            D = D,
            alpha_sd = alpha_sd,
            cutpoints = cp_name,
            correlation = correlation
          ))
        }
      }
    }
  }
}

# Save batch log
write_csv(log_results, file.path(out_dir, "batch_summary.csv"))
message("\n✅ All simulations completed. Summary saved to:", out_dir)
