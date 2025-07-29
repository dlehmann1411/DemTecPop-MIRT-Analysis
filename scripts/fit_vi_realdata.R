
# fit_vi_realdata.R
# --------------------------------------------------------------
# Fits ordinal MIRT model using VI on real-world data
# (e.g. PreparedConjointData.csv from Dem,Tec,Pop project)
# --------------------------------------------------------------

library(tidyverse)
library(cmdstanr)
library(posterior)
library(here)
library(ggplot2)
library(GGally)
library(naniar)
library(psych)

# --------------------------------------------------------------
# Step 1: Load and filter data
# --------------------------------------------------------------
data <- read_csv(here("data", "PreparedConjointData.csv"))

# Define relevant item columns
item_columns <- c(
  paste0("pop", 1:6),
  paste0("tec", 1:6),
  paste0("dem", 1:6)
)

# Clean data: remove incomplete or constant response patterns,
# keep only ResponseId and relevant items
data_clean <- data %>%
  filter(complete.cases(across(all_of(item_columns)))) %>%
  unite("response_pattern", all_of(item_columns), sep = "", remove = FALSE) %>%
  filter(!response_pattern %in% map_chr(1:7, ~ strrep(as.character(.x), 18))) %>%
  select(ResponseId, all_of(item_columns))

# -----------------------------------------------
# Step 2: Prepare response matrix and item-only DataFrame
# -----------------------------------------------
item_data <- data_clean %>%
  select(-ResponseId)

# -----------------------------------------------
# Step 3: Visualize item distributions (ordinal Likert-style)
# -----------------------------------------------
item_data %>%
  pivot_longer(everything(), names_to = "item", values_to = "response") %>%
  ggplot(aes(x = factor(response))) +
  geom_bar(fill = "steelblue") +
  facet_wrap(~ item, scales = "free_y") +
  labs(title = "Response Distributions per Item",
       x = "Response Category", y = "Count") +
  theme_minimal()

response_matrix <- item_data %>%
  mutate(across(everything(), as.integer)) %>%
  as.matrix()

# -----------------------------------------------
# Step 4: Correlation matrix (Pearson; ordinal approx.)
# -----------------------------------------------
cor_matrix <- cor(item_data, use = "pairwise.complete.obs", method = "pearson")


ggcorr(cor_matrix, label = TRUE, label_size = 3, hjust = 0.75,
       layout.exp = 1, name = "Pearson\nCorrelation") +
  labs(title = "Item Correlation Matrix")


# --------------------------------------------------------------
# Step 2: Define constraint matrix (lambda_signs)
# --------------------------------------------------------------

# Number of items and dimensions
K <- ncol(response_matrix)  # Should be 18
D <- 3                      # Latent dimensions: Populism, Technocracy, Democracy

# Initialize empty loading matrix (18 items × 3 dimensions)
lambda_signs <- matrix(0, nrow = K, ncol = D)

# Assign fixed loadings: 6 items per dimension
lambda_signs[1:6,    1] <- 1  # Items 1–6 → Populism
lambda_signs[7:12,   2] <- 1  # Items 7–12 → Technocracy
lambda_signs[13:18,  3] <- 1  # Items 13–18 → Democracy


# --------------------------------------------------------------
# Step 3: Prepare data list for Stan
# --------------------------------------------------------------

stan_data <- list(
  N = nrow(response_matrix),     # Number of respondents
  K = K,                         # Number of items (18)
  D = D,                         # Number of latent dimensions (3)
  C = 7,                         # Number of ordinal response categories (1–7)
  Y = response_matrix,           # Response matrix (N × K)
  lambda_signs = lambda_signs    # Fixed loading structure (K × D)
)


# --------------------------------------------------------------
# Step 4: Compile and fit the model using VI
# --------------------------------------------------------------
model <- cmdstan_model(
  here("stan", "ordinal_irtm.stan"), force_recompile = TRUE)

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

theta_matrix <- matrix(as.numeric(theta_est), ncol = D, byrow = TRUE)
colnames(theta_matrix) <- c("Populism", "Technocracy", "Democracy")
theta_df <- as_tibble(theta_matrix)

theta_df %>%
  pivot_longer(everything()) %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  facet_wrap(~name, scales = "free") +
  theme_minimal()

cor(theta_df)

