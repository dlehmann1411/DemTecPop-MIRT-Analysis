
###############################################################
#    IRT-M Analysis Project Dem,Tec,Pop                       #
###############################################################
rm(list = ls())

# ---------------------------
# Load packages
# ---------------------------
library(tidyverse)
library(cmdstanr)
library(here)
library(posterior)

# ---------------------------
# Step 1: Load and filter data
# ---------------------------
data <- read.csv(here("data", "PreparedConjointData.csv"))

# Select 18 items and ResponseId
item_columns <- c(
  "ResponseId",
  paste0(rep(c("pop", "tec", "dem"), each = 6), 1:6)
)

# Clean: remove incomplete and invariant responses
data_clean <- data %>%
  filter(complete.cases(across(all_of(item_columns)))) %>%
  unite("response_pattern", all_of(item_columns[-1]), sep = "", remove = FALSE) %>%
  filter(!response_pattern %in% map_chr(1:7, ~ strrep(as.character(.x), 18)))

# Response matrix
response_matrix <- data_clean %>%
  select(all_of(item_columns[-1])) %>%
  mutate(across(everything(), as.integer)) %>%
  as.matrix()

# ---------------------------
# Step 2: Create constraint matrix
# ---------------------------
lambda_signs <- matrix(0, nrow = 18, ncol = 3)  # Default shrinkage
colnames(lambda_signs) <- c("Pop", "Tec", "Dem")
lambda_signs[1:6, 1]   <- 1   # pop1–pop6 → Populism
lambda_signs[7:12, 2]  <- 1   # tec1–tec6 → Technocracy
lambda_signs[13:18, 3] <- 1   # dem1–dem6 → Democracy

# ---------------------------
# Step 3: Prepare Stan data
# ---------------------------
stan_data <- list(
  N = nrow(response_matrix),
  K = ncol(response_matrix),
  C = 7,
  D = 3,
  Y = response_matrix,
  lambda_signs = lambda_signs
)

# ---------------------------
# Step 4: Compile Stan model
# ---------------------------
model <- cmdstan_model(here("stan", "ordinal_irtm.stan"), force_recompile = TRUE)

# ---------------------------
# Step 5: Fit model
# ---------------------------
fit <- model$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 2,
  iter_sampling = 1000,
  iter_warmup = 1000,
  seed = 123,
  refresh = 50
)

# ---------------------------
# Step 6: Extract posterior draws
# ---------------------------
print(fit$summary(variables = c("theta")))

posterior_df <- fit$draws(c("theta", "lambda_raw", "intercepts", "cutpoints")) %>%
  as_draws_df()

write.csv(posterior_df, here("results", "posterior_draws.csv"), row.names = FALSE)

message("Model fitting completed successfully.")
