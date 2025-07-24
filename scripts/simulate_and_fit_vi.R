# simulate_and_fit_rhs.R
# ------------------------------------------------------------------------------
# Simulation + Recovery Evaluation for Exploratory Ordinal MIRT (RHS + VI)
# ------------------------------------------------------------------------------
# This script simulates ordinal item responses under a sparse loading structure
# and estimates latent traits using a Regularized Horseshoe (RHS) prior in Stan.

library(tidyverse)
library(cmdstanr)
library(posterior)
library(here)

# -----------------------------
# Simulation grid
# -----------------------------
N_vals <- c(20000)
K_vals <- c(18)
alpha_sds <- c(0.5, 0.75)
cutpoint_sets <- list(
  empirical = c(-2, -1.5, -0.5, 0.5, 1.5, 2.5)
)
D_vals <- c(6)  # Exploratory: allow 6 dimensions, simulate signal on 3 only

batch_id <- format(Sys.time(), "%Y%m%d_%H%M")
out_dir <- here("results_rhs", paste0("batch_", batch_id))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
data_log <- tibble()

for (N in N_vals) {
  for (K in K_vals) {
    for (alpha_sd in alpha_sds) {
      for (cp_name in names(cutpoint_sets)) {
        for (D in D_vals) {
          
          C <- 7
          cutpoints_true <- cutpoint_sets[[cp_name]]
          
          # Simulate theta from 3 overlapping ideological groups
          group_size <- N %/% 3
          remainder <- N %% 3
          
          centers <- list(
            democrat   = c( 1.5,  0.5, -1.0),
            technocrat = c( 0.5,  1.5, -0.5),
            populist   = c(-1.0, -0.5,  1.5)
          )
          
          spread <- 1.0
          theta_dem <- matrix(rnorm(group_size * 3, mean = rep(centers$democrat, each = group_size), sd = spread), ncol = 3, byrow = TRUE)
          theta_tec <- matrix(rnorm(group_size * 3, mean = rep(centers$technocrat, each = group_size), sd = spread), ncol = 3, byrow = TRUE)
          theta_pop <- matrix(rnorm((group_size + remainder) * 3, mean = rep(centers$populist, each = group_size + remainder), sd = spread), ncol = 3, byrow = TRUE)
          
          theta_true_3d <- rbind(theta_dem, theta_tec, theta_pop)
          theta_true <- cbind(theta_true_3d, matrix(0, nrow = N, ncol = D - 3))  # pad remaining dims with 0
          
          group_id <- c(rep("Democrat", group_size), rep("Technocrat", group_size), rep("Populist", group_size + remainder))
          
          # Simulate item parameters
          alpha_true <- rlnorm(K, log(1), alpha_sd)
          intercepts_true <- rnorm(K, 0, 1.5)
          
          # Simulate sparse loading matrix lambda_true (only load on 3 dims)
          lambda_true <- matrix(0, nrow = K, ncol = D)
          lambda_true[1:6, 1] <- runif(6, 0.7, 1.2)
          lambda_true[7:12, 2] <- runif(6, 0.7, 1.2)
          lambda_true[13:18, 3] <- runif(6, 0.7, 1.2)
          
          # Generate linear predictor
          eta_matrix <- theta_true %*% t(lambda_true)
          eta_matrix <- sweep(eta_matrix, 2, alpha_true, "*")
          eta_matrix <- sweep(eta_matrix, 2, intercepts_true, "+")
          
          # Simulate ordinal responses
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
          
          stan_data <- list(N = N, K = K, D = D, C = C, Y = Y)
          
          model <- cmdstan_model(here("stan", "ordinal_irtm_rhs.stan"), force_recompile = TRUE)
          
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
          
          plot_data <- tibble(
            true_theta = as.numeric(theta_true),
            est_theta  = as.numeric(theta_est),
            group      = rep(group_id, times = D)
          )
          
          plot <- ggplot(plot_data, aes(x = true_theta, y = est_theta, color = group)) +
            geom_point(alpha = 0.3, size = 0.5) +
            scale_color_manual(values = c("Democrat" = "#0072B2", 
                                          "Technocrat" = "#009E73", 
                                          "Populist" = "#D55E00")) +
            labs(title = paste0("Recovery by Subtype: ", sim_id),
                 x = "True Theta", y = "Estimated Theta (mean VI)", color = "Group") +
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
message("\n✅ All RHS simulations completed. Summary saved to:", out_dir)


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
