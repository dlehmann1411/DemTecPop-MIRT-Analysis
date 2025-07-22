# batch_simulate_and_fit_vi.R
# ------------------------------------------------------------------------------
# Targeted Simulation + Recovery Evaluation for Ordinal MIRT (Variational Inference)
# ------------------------------------------------------------------------------
# This script conducts a focused simulation study for a multidimensional IRT model
# estimated via mean-field Variational Inference (VI). It simulates data with known
# parameters and evaluates how well the model recovers latent traits (theta).
#
# Using simplified flat MIRT model without hierarchical priors (Stan model: ordinal_irtm_flat.stan)


library(tidyverse)
library(cmdstanr)
library(posterior)
library(here)

# -----------------------------
# Simulation grid
# -----------------------------
N_vals <- c(10000, 20000)
K_vals <- c(18)
alpha_sds <- c(0.5, 0.75)
cutpoint_sets <- list(
  narrow    = c(-1.5, -1, -0.5, 0, 0.5, 1),
  empirical = c(-2, -1.5, -0.5, 0.5, 1.5, 2.5)
)
D_vals <- c(3)

# Output directory
batch_id <- format(Sys.time(), "%Y%m%d_%H%M")
out_dir <- here("results", paste0("batch_", batch_id))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

data_log <- tibble()

for (N in N_vals) {
  for (K in K_vals) {
    for (alpha_sd in alpha_sds) {
      for (cp_name in names(cutpoint_sets)) {
        for (D in D_vals) {
          
          C <- 7
          cutpoints_true <- cutpoint_sets[[cp_name]]
          
          # Simulate flat theta (standard normal)
          theta_true <- matrix(rnorm(N * D), nrow = N, ncol = D)
          
          # Simulate item parameters
          alpha_true <- rlnorm(K, log(1), alpha_sd)
          intercepts_true <- rnorm(K, 0, 1.5)
          
          # Loading structure
          lambda_signs <- matrix(0, nrow = K, ncol = D)
          for (d in 1:D) {
            idx <- ((d - 1) * K / D + 1):(d * K / D)
            lambda_signs[idx, d] <- 1
          }
          
          # Calculate eta matrix
          eta_matrix <- theta_true %*% t(lambda_signs)
          eta_matrix <- sweep(eta_matrix, 2, alpha_true, "*")
          eta_matrix <- sweep(eta_matrix, 2, intercepts_true, "+")
          
          # Response generation (ordered logit)
          ordered_logistic_probs <- function(eta, cutpoints) {
            logits <- c(-Inf, cutpoints, Inf)
            cdfs <- plogis(logits - eta)
            diff(cdfs)
          }
          
          Y <- matrix(NA, nrow = N, ncol = K)
          for (n in 1:N) {
            for (k in 1:K) {
              probs <- ordered_logistic_probs(eta_matrix[n, k], cutpoints_true)
              Y[n, k] <- sample(1:C, 1, prob = probs)
            }
          }
          
          # Stan input
          stan_data <- list(N = N, K = K, D = D, C = C, Y = Y, lambda_signs = lambda_signs)
          
          # Compile model (skip if already compiled)
          model <- cmdstan_model(here("stan", "ordinal_irtm.stan"), force_recompile = FALSE)
          
          # VI estimation
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
          
          correlation <- cor(as.numeric(theta_true), as.numeric(theta_est))
          sim_id <- paste0("N", N, "_K", K, "_sd", alpha_sd, "_", cp_name, "_D", D)
          
          write_csv(theta_est, file.path(out_dir, paste0("theta_est_", sim_id, ".csv")))
          write_csv(data.frame(correlation = correlation), file.path(out_dir, paste0("summary_", sim_id, ".csv")))
          
          plot <- qplot(as.numeric(theta_true), as.numeric(theta_est)) +
            labs(title = paste0("Recovery: ", sim_id),
                 x = "True Theta", y = "Estimated Theta (mean VI)") +
            theme_minimal()
          
          ggsave(file.path(out_dir, paste0("plot_", sim_id, ".png")), plot, bg = "white")
          
          data_log <- bind_rows(data_log, tibble(
            N = N, K = K, D = D, alpha_sd = alpha_sd, cutpoints = cp_name, correlation = correlation
          ))
        }
      }
    }
  }
}

write_csv(data_log, file.path(out_dir, "batch_summary.csv"))
message("\n✅ All simulations completed. Summary saved to:", out_dir)



################################################################################


# analyze_batch_vi.R
# ---------------------------------------------------------------------
# Summary analysis script for evaluating performance of ordinal MIRT-VI
# simulation batches. Reads from a single batch_summary.csv file and
# produces aggregated insights and visualizations.
# ---------------------------------------------------------------------

# --------------------------------------------------------------
# Step 1: Load the latest batch summary
# --------------------------------------------------------------
# Adjust this path if you want to analyze an older batch manually
batch_folders <- list.dirs(here("results"), full.names = TRUE, recursive = FALSE)
latest_batch <- batch_folders[which.max(file.info(batch_folders)$mtime)]
summary_path <- file.path(latest_batch, "batch_summary.csv")

summary_data <- read_csv(summary_path)

# --------------------------------------------------------------
# Step 2: Inspect and clean
# --------------------------------------------------------------

# Ensure correct column types
summary_data <- summary_data %>%
  mutate(
    N = as.factor(N),
    D = as.factor(D),
    alpha_sd = as.factor(alpha_sd),
    cutpoints = factor(cutpoints, levels = c("narrow", "wide"))
  )

# --------------------------------------------------------------
# Step 3: Summary statistics
# --------------------------------------------------------------
overall_stats <- summary_data %>%
  group_by(N, D, alpha_sd, cutpoints) %>%
  summarise(
    mean_r = mean(correlation),
    sd_r = sd(correlation),
    .groups = "drop"
  )

print("Mean correlations per condition:")
print(overall_stats, n = 72)

# --------------------------------------------------------------
# Step 4: Visualization: heatmaps or faceted plots
# --------------------------------------------------------------
ggplot(overall_stats, aes(x = alpha_sd, y = N, fill = mean_r)) +
  geom_tile(color = "white") +
  facet_grid(D ~ cutpoints) +
  scale_fill_viridis_c(option = "D", name = "Mean r") +
  labs(
    title = "Recovery Performance by VI (Mean Correlation)",
    x = "Discrimination SD (alpha_sd)",
    y = "Sample Size (N)"
  ) +
  theme_minimal()
