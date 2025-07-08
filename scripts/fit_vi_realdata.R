
# fit_vi_realdata.R
# --------------------------------------------------------------
# Fits ordinal MIRT model using VI on real-world data
# (e.g. PreparedConjointData.csv from Dem,Tec,Pop project)
# --------------------------------------------------------------

library(tidyverse)
library(cmdstanr)
library(posterior)
library(here)

# --------------------------------------------------------------
# Step 1: Load and filter data
# --------------------------------------------------------------
data <- read_csv(here("data", "PreparedConjointData.csv"))

# Item identifiers (must match the column names in your dataset)
item_columns <- c(
  paste0("pop", 1:6),
  paste0("tec", 1:6),
  paste0("dem", 1:6)
)

# Clean: remove incomplete or constant response patterns
data_clean <- data %>%
  filter(complete.cases(across(all_of(item_columns)))) %>%
  unite("response_pattern", all_of(item_columns), sep = "", remove = FALSE) %>%
  filter(!response_pattern %in% map_chr(1:7, ~ strrep(as.character(.x), 18)))

# Create response matrix
response_matrix <- data_clean %>%
  select(all_of(item_columns)) %>%
  mutate(across(everything(), as.integer)) %>%
  as.matrix()

# --------------------------------------------------------------
# Step 2: Define constraint matrix (lambda_signs)
# --------------------------------------------------------------
K <- ncol(response_matrix)
D <- 3
lambda_signs <- matrix(0, nrow = K, ncol = D)
lambda_signs[1:6, 1] <- 1     # Populism
lambda_signs[7:12, 2] <- 1    # Technocracy
lambda_signs[13:18, 3] <- 1   # Democracy

# --------------------------------------------------------------
# Step 3: Prepare data list for Stan
# --------------------------------------------------------------
stan_data <- list(
  N = nrow(response_matrix),
  K = K,
  D = D,
  C = 7,
  Y = response_matrix,
  lambda_signs = lambda_signs
)

# --------------------------------------------------------------
# Step 4: Compile and fit the model using VI
# --------------------------------------------------------------
model <- cmdstan_model(here("stan", "ordinal_irtm.stan"), force_recompile = TRUE)

fit_vi <- model$variational(
  data = stan_data,
  iter = 10000,
  grad_samples = 1,
  elbo_samples = 100,
  output_samples = 1000,
  seed = 123
)

# --------------------------------------------------------------
# Step 5: Extract and store theta estimates
# --------------------------------------------------------------
draws <- fit_vi$draws(format = "df")
theta_draws <- draws %>% select(starts_with("theta"))
theta_est <- theta_draws %>% summarise(across(everything(), mean))

write_csv(theta_est, here("results", "theta_vi_realdata.csv"))
message("VI fit completed and latent scores saved.")
