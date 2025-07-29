
# simulate_and_fit_rhs.R
# ------------------------------------------------------------------------------
# Simulation + Recovery Evaluation for Exploratory Ordinal MIRT (RHS + VI)
# ------------------------------------------------------------------------------
# This script simulates ordinal item responses under a sparse loading structure
# and estimates latent traits using a Regularized Horseshoe (RHS) prior in Stan.
# It is designed to test whether the RHS prior can recover the true sparse
# item-factor loading structure and latent traits under a realistic scenario.

library(tidyverse)
library(cmdstanr)
library(posterior)
library(here)
library(reshape2)

# -----------------------------
# Simulation grid (design settings)
# -----------------------------
N_vals <- c(20000)              # Number of respondents
K_vals <- c(18)                 # Number of items
alpha_sds <- c(0.5, 0.75)       # Variation in discrimination parameters
cutpoint_sets <- list(          # Ordinal cutpoints for response categories
  empirical = c(-2, -1.5, -0.5, 0.5, 1.5, 2.5)
)
D_vals <- c(6)                  # Exploratory space: 6 latent dimensions

# Output folder setup
batch_id <- format(Sys.time(), "%Y%m%d_%H%M")
out_dir <- here("results_rhs", paste0("batch_", batch_id))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
data_log <- tibble()

# -----------------------------
# Simulation loop
# -----------------------------
for (N in N_vals) {
  for (K in K_vals) {
    for (alpha_sd in alpha_sds) {
      for (cp_name in names(cutpoint_sets)) {
        for (D in D_vals) {
          
          C <- 7
          cutpoints_true <- cutpoint_sets[[cp_name]]
          
          # Simulate latent traits (theta): 3 ideological profiles
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
          
          # Add small noise to unused dimensions
          theta_noise <- matrix(rnorm(nrow(theta_true_3d) * (D - 3), mean = 0, sd = 0.05), ncol = D - 3)
          theta_true <- cbind(theta_true_3d, theta_noise)
          
          group_id <- c(rep("Democrat", group_size), rep("Technocrat", group_size), rep("Populist", group_size + remainder))
          
          # Simulate item parameters
          alpha_true <- rlnorm(K, log(1), alpha_sd)     # Discrimination
          intercepts_true <- rnorm(K, 0, 1.5)           # Intercepts
          
          # Simulate sparse loading matrix (true lambda): 3 informative dims
          set.seed(42)
          true_dim <- rep(1:3, each = 6)
          lambda_true <- matrix(NA, nrow = K, ncol = D)
          for (k in 1:K) {
            for (d in 1:D) {
              lambda_true[k, d] <- ifelse(d == true_dim[k],
                                          runif(1, 0.7, 1.2),     # true signal
                                          rnorm(1, 0, 0.05))       # weak noise
            }
          }
          
          # Generate linear predictor eta = theta * lambda^T * alpha + intercept
          eta_matrix <- theta_true %*% t(lambda_true)
          eta_matrix <- sweep(eta_matrix, 2, alpha_true, "*")
          eta_matrix <- sweep(eta_matrix, 2, intercepts_true, "+")
          
          # Generate responses from ordered logistic model
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
          
          # Prepare Stan input
          stan_data <- list(N = N, K = K, D = D, C = C, Y = Y)
          
          # Compile and fit Stan model
          model <- cmdstan_model(here("stan", "ordinal_irtm.stan"), force_recompile = TRUE)
          
          fit_vi <- model$variational(
            data = stan_data,
            iter = 10000,
            grad_samples = 1,
            elbo_samples = 100,
            output_samples = 1000,
            seed = 123
          )
          
          # Extract lambda (item loadings)
          draws <- fit_vi$draws(format = "df")
          lambda_draws <- draws %>% select(starts_with("lambda_raw"))
          lambda_means <- lambda_draws %>% summarise(across(everything(), mean))
          lambda_matrix <- matrix(unlist(lambda_means), nrow = K, byrow = TRUE)
          
          sim_id <- paste0("N", N, "_K", K, "_sd", alpha_sd, "_", cp_name, "_D", D)
          
          # Heatmap of estimated loadings
          heatmap_df <- melt(lambda_matrix)
          colnames(heatmap_df) <- c("Item", "Dimension", "Loading")
          
          heatmap_plot <- ggplot(heatmap_df, aes(x = factor(Dimension), y = factor(Item), fill = Loading)) +
            geom_tile() +
            scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
            theme_minimal() +
            labs(title = paste0("Estimated Item Loadings (Lambda): ", sim_id), x = "Dimension", y = "Item")
          ggsave(file.path(out_dir, paste0("heatmap_", sim_id, ".png")), heatmap_plot, bg = "white")
          
          # Barplot of active loadings (|lambda| > threshold)
          active_dims <- apply(abs(lambda_matrix), 2, function(x) sum(x > 0.1))
          barplot(active_dims, main = paste0("Active Loadings per Dimension: ", sim_id),
                  ylab = "# of Items", xlab = "Dimension")
          dev.copy(png, file.path(out_dir, paste0("active_dims_", sim_id, ".png")))
          dev.off()
          
          # Identify strongest dimension for each item
          top_loading <- apply(abs(lambda_matrix), 1, which.max)
          top_df <- tibble(Item = 1:K, Top_Dimension = top_loading)
          write_csv(top_df, file.path(out_dir, paste0("top_loadings_", sim_id, ".csv")))
          
          # Extract and evaluate theta estimates
          theta_draws <- draws %>% select(starts_with("theta"))
          theta_est <- theta_draws %>% summarise(across(everything(), mean))
          theta_matrix <- matrix(unlist(theta_est), ncol = D, byrow = TRUE)
          theta_variances <- apply(theta_matrix, 2, var)
          write_csv(tibble(Dimension = 1:D, Variance = theta_variances),
                    file.path(out_dir, paste0("theta_var_", sim_id, ".csv")))
          
          # Append summary info to log
          data_log <- bind_rows(data_log, tibble(
            N = N, K = K, D = D, alpha_sd = alpha_sd, cutpoints = cp_name
          ))
        }
      }
    }
  }
}

write_csv(data_log, file.path(out_dir, "batch_summary.csv"))
message("\n✅ All RHS simulations completed. Summary saved to:", out_dir)
