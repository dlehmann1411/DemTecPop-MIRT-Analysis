
# ---------------------------------------------------------------
# Exploratory ORDINAL MIRT with element-wise Regularized Horseshoe
# - c2 is treated as DATA (grid-swept here)
# - VI for fast screening (ELBO, sparsity patterns, PIPs)
# - NUTS for reliable posteriors (paper-grade uncertainty)
# ---------------------------------------------------------------

library(tidyverse)
library(cmdstanr)
library(posterior)
library(here)
library(reshape2)
library(yardstick)
library(clue)       # Hungarian matching for label alignment
library(stringr)

# ----------------------- Config --------------------------------
N_vals      <- c(10000, 20000)
c2_vals     <- c(2, 5, 10)             # slab variance grid
reps        <- 1:3
D_fit       <- 15
K           <- 18
C           <- 7
active_dims <- 3                       # true active factors per simulation
EPS        <- 0.075                    # PIP threshold |lambda| > EPS

# Control which engines to run
DO_VI       <- TRUE                    # run VI screening
DO_NUTS     <- TRUE                    # run NUTS confirmation
RUN_NUTS_ON_REP <- 1                   # run NUTS only for rep == this value (adjust if needed)

# Output root
batch_id <- format(Sys.time(), "%Y%m%d_%H%M")
out_root <- here("results_rhs_batch", paste0("batch_", batch_id))
dir.create(out_root, recursive = TRUE)

# ----------------------- Helpers -------------------------------

# Item-specific ordered logit probabilities
ordered_logistic_probs_item <- function(eta, cut_k) {
  logits <- c(-Inf, cut_k, Inf)
  cdfs   <- plogis(logits - eta)
  diff(cdfs)
}

# Extract K x D matrix of lambda statistics from draws_df
# 'fun' is applied element-wise over posterior draws of each lambda[i,j]
collect_lambda_matrix <- function(draws_df, K, D, fun = mean) {
  cols <- grep("^lambda\\[", names(draws_df), value = TRUE)
  if (length(cols) != K * D) {
    stop("Mismatch: found ", length(cols), " lambda columns, expected ", K * D)
  }
  mat <- matrix(NA_real_, nrow = K, ncol = D)
  for (nm in cols) {
    idx <- stringr::str_match(nm, "lambda\\[(\\d+),(\\d+)\\]")
    i <- as.integer(idx[2]); j <- as.integer(idx[3])
    mat[i, j] <- fun(as.numeric(draws_df[[nm]]))
  }
  rownames(mat) <- paste0("item_", 1:K)
  colnames(mat) <- paste0("dim_", 1:D)
  mat
}

# Align estimated dimensions to true dimensions via max |cor| across items
align_dimensions <- function(L_est, L_true) {
  D_hat <- ncol(L_est); D_true <- ncol(L_true)
  S <- matrix(0, nrow = D_hat, ncol = D_true)
  for (dh in 1:D_hat) {
    for (dt in 1:D_true) {
      S[dh, dt] <- suppressWarnings(cor(L_est[, dh], L_true[, dt]))
    }
  }
  S[is.na(S)] <- 0
  # maximize absolute correlation -> minimize negative abs(cor)
  cost <- -abs(S)
  perm <- solve_LSAP(cost)   # mapping est dh -> true dt
  list(L_est_aligned = L_est[, perm, drop = FALSE],
       perm = as.integer(perm),
       score = S)
}

# ----------------------- Compile Stan once ---------------------
stan_file <- here("stan", "ordinal_mirt_rhs_regularized_c2data.stan")
mod <- cmdstan_model(stan_file, force_recompile = FALSE)

# ----------------------- Main loop -----------------------------
for (N in N_vals) {
  for (c2_fix in c2_vals) {
    for (rep in reps) {
      
      set.seed(42 + rep)
      sim_id <- paste0("N", N, "_c2", c2_fix, "_rep", rep)
      out_dir <- file.path(out_root, sim_id)
      dir.create(out_dir, recursive = TRUE)
      
      # 1) Simulate true theta (sparse factor structure across dimensions)
      active_dim_ids <- sample(1:D_fit, active_dims)
      theta_true <- matrix(0, nrow = N, ncol = D_fit)
      for (d in active_dim_ids) {
        theta_true[, d] <- rnorm(N, mean = rnorm(1, 0, 1.5), sd = 1.0)
      }
      
      # 2) Simulate true lambda (one strong loading per item + small noise elsewhere)
      lambda_true <- matrix(rnorm(K * D_fit, 0, 0.02), nrow = K, ncol = D_fit)
      for (k in 1:K) {
        d <- sample(active_dim_ids, 1)
        lambda_true[k, d] <- runif(1, 0.7, 1.2)
      }
      
      # 3) Item-specific thresholds (sorted with mild jitter)
      base_cut <- seq(-2, 2, length.out = C - 1)
      cutpoints_true <- matrix(NA, nrow = K, ncol = C - 1)
      for (k in 1:K) {
        cutpoints_true[k, ] <- sort(base_cut + rnorm(C - 1, 0, 0.25))
      }
      
      # 4) Generate responses
      eta <- theta_true %*% t(lambda_true)
      Y <- matrix(NA_integer_, nrow = N, ncol = K)
      for (n in 1:N) {
        for (k in 1:K) {
          probs <- ordered_logistic_probs_item(eta[n, k], cutpoints_true[k, ])
          Y[n, k] <- sample.int(C, 1, prob = probs)
        }
      }
      
      # 5) Fit(s)
      stan_data <- list(N = N, K = K, D = D_fit, C = C, Y = Y, c2 = c2_fix)
      
      # --- (a) VI: fast screening (do NOT use for init) -------------
      if (DO_VI) {
        vi <- mod$variational(
          data = stan_data, seed = 1000 + rep,
          iter = 3000, output_samples = 500,
          grad_samples = 1, elbo_samples = 100
        )
        vi_draws <- as_draws_df(vi$draws())
        
        Lambda_hat_vi <- collect_lambda_matrix(vi_draws, K, D_fit, fun = mean)
        Lambda_pip_vi <- collect_lambda_matrix(
          vi_draws, K, D_fit,
          fun = function(x) mean(abs(x) > EPS)
        )
        
        write.csv(Lambda_hat_vi, file.path(out_dir, "Lambda_hat_VI.csv"), row.names = FALSE)
        write.csv(Lambda_pip_vi, file.path(out_dir, "Lambda_PIP_VI.csv"), row.names = FALSE)
      }
      
      # --- (b) NUTS: reliable posterior (subset by rep to save time) -
      have_nuts <- FALSE
      if (DO_NUTS && rep == RUN_NUTS_ON_REP) {
        fit <- mod$sample(
          data = stan_data,
          chains = 4, parallel_chains = 4, seed = 2000 + rep,
          iter_warmup = 750, iter_sampling = 750,
          adapt_delta = 0.97, max_treedepth = 12,
          init = 0.1    # small random inits (robust)
        )
        draws_df <- as_draws_df(fit$draws())
        Lambda_hat <- collect_lambda_matrix(draws_df, K, D_fit, fun = mean)
        Lambda_pip <- collect_lambda_matrix(
          draws_df, K, D_fit,
          fun = function(x) mean(abs(x) > EPS)
        )
        write.csv(Lambda_hat, file.path(out_dir, "Lambda_hat_NUTS.csv"), row.names = FALSE)
        write.csv(Lambda_pip, file.path(out_dir, "Lambda_PIP_NUTS.csv"), row.names = FALSE)
        have_nuts <- TRUE
      }
      
      # 6) Choose reference estimates for downstream evaluation
      if (have_nuts) {
        Lambda_ref <- Lambda_hat
        PIP_ref    <- Lambda_pip
        ref_tag    <- "NUTS"
      } else if (DO_VI) {
        Lambda_ref <- Lambda_hat_vi
        PIP_ref    <- Lambda_pip_vi
        ref_tag    <- "VI"
      } else {
        stop("No inference engine ran. Set DO_VI or DO_NUTS to TRUE.")
      }
      
      # 7) Alignment and classification metrics (using reference estimates)
      align <- align_dimensions(Lambda_ref, lambda_true)
      Lambda_al <- align$L_est_aligned
      perm      <- align$perm
      
      true_dim <- apply(abs(lambda_true), 1, which.max)
      pred_dim <- apply(abs(Lambda_al), 1, which.max)
      
      classification <- tibble(
        item = 1:K,
        true_dim = true_dim,
        pred_dim = pred_dim
      )
      
      conf_mat <- table(True = true_dim, Pred = pred_dim)
      write.csv(conf_mat, file.path(out_dir, paste0("confusion_matrix_", ref_tag, ".csv")))
      
      acc <- mean(true_dim == pred_dim)
      writeLines(sprintf("Accuracy (%s): %.3f", ref_tag, acc),
                 file.path(out_dir, paste0("accuracy_", ref_tag, ".txt")))
      
      # 8) Factor relevance via column L2 (reference)
      col_l2 <- sqrt(colSums(Lambda_ref^2))
      write.csv(col_l2, file.path(out_dir, paste0("lambda_col_L2_", ref_tag, ".csv")), row.names = FALSE)
      
      # 9) Heatmaps
      # True
      heat_true <- melt(lambda_true)
      colnames(heat_true) <- c("Item", "Dimension", "Loading")
      p1 <- ggplot(heat_true, aes(x = factor(Dimension), y = factor(Item), fill = Loading)) +
        geom_tile() +
        scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
        labs(title = paste("True Lambda -", sim_id)) +
        theme_minimal()
      ggsave(file.path(out_dir, "lambda_true.png"), p1, bg = "white", width = 7, height = 7, dpi = 150)
      
      # Estimated (aligned, reference engine)
      heat_est <- melt(Lambda_al)
      colnames(heat_est) <- c("Item", "Dimension", "Loading")
      p2 <- ggplot(heat_est, aes(x = factor(Dimension), y = factor(Item), fill = Loading)) +
        geom_tile() +
        scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
        labs(title = paste0("Estimated Lambda (aligned, ", ref_tag, ") - ", sim_id)) +
        theme_minimal()
      ggsave(file.path(out_dir, paste0("lambda_est_", ref_tag, ".png")), p2, bg = "white", width = 7, height = 7, dpi = 150)
      
      # PIP heatmap (reference)
      heat_pip <- melt(PIP_ref)
      colnames(heat_pip) <- c("Item", "Dimension", "PIP")
      p3 <- ggplot(heat_pip, aes(x = factor(Dimension), y = factor(Item), fill = PIP)) +
        geom_tile() +
        scale_fill_gradient(limits = c(0, 1)) +
        labs(title = paste0("PIP(|lambda|>", EPS, ") (", ref_tag, ") - ", sim_id)) +
        theme_minimal()
      ggsave(file.path(out_dir, paste0("lambda_pip_", ref_tag, ".png")), p3, bg = "white", width = 7, height = 7, dpi = 150)
      
      # 10) Agreement VI vs NUTS if both ran
      if (DO_VI && have_nuts) {
        vi_vec   <- as.vector(Lambda_hat_vi)
        nuts_vec <- as.vector(Lambda_hat)
        agree <- suppressWarnings(cor(vi_vec, nuts_vec, use = "pairwise.complete.obs"))
        writeLines(sprintf("VI vs NUTS mean(lambda) corr: %.3f", agree),
                   file.path(out_dir, "vi_nuts_agreement.txt"))
      }
      
      message("✅ Done with: ", sim_id, " [engine=", ref_tag, "]")
    }
  }
}
