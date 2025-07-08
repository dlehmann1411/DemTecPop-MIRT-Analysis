
###############################################################
# Simulationsvergleich: Dichotom vs. Ordinal IRT mit brms     #
#                                                             #
###############################################################

# ---- Pakete laden ----
library(tidyverse)
library(brms)

# ---- Einstellungen ----
set.seed(123)
N <- 500      # Anzahl Personen
K <- 10       # Anzahl Items
C <- 5        # Anzahl ordinaler Kategorien

# ---- Fähigkeiten und Schwierigkeitsgrade simulieren ----
theta <- rnorm(N, 0, 1)  # Personenfähigkeiten
b <- rnorm(K, 0, 1)      # Itemschwierigkeiten
cutpoints <- c(-1.5, -0.5, 0.5, 1.5)  # Cutpoints für ordinal

# ---- Dichotome Daten simulieren ----
response_matrix_bin <- matrix(NA, nrow = N, ncol = K)
for (i in 1:N) {
  for (j in 1:K) {
    eta <- theta[i] - b[j]
    p <- plogis(eta)
    response_matrix_bin[i, j] <- rbinom(1, 1, p)
  }
}

sim_bin_data <- as.data.frame(response_matrix_bin) %>%
  mutate(id = 1:N) %>%
  pivot_longer(cols = starts_with("V"), names_to = "item", values_to = "resp") %>%
  mutate(item = factor(item), id = factor(id))

# ---- Ordinale Daten simulieren ----
response_matrix_ord <- matrix(NA, nrow = N, ncol = K)
for (i in 1:N) {
  for (j in 1:K) {
    eta <- theta[i] - b[j]
    probs <- c(
      plogis(cutpoints[1] - eta),
      diff(plogis(cutpoints - eta)),
      1 - plogis(cutpoints[length(cutpoints)] - eta)
    )
    response_matrix_ord[i, j] <- sample(1:C, 1, prob = probs)
  }
}

sim_ord_data <- as.data.frame(response_matrix_ord) %>%
  mutate(id = 1:N) %>%
  pivot_longer(cols = starts_with("V"), names_to = "item", values_to = "resp") %>%
  mutate(item = factor(item), id = factor(id))

# ---- Modelle mit brms schätzen ----

## Dichotomes Modell
fit_bin <- brm(
  resp ~ 0 + item + (1 | id),
  data = sim_bin_data,
  family = bernoulli("logit"),
  chains = 2, iter = 1000, seed = 123
)

## Ordinales Modell
fit_ord <- brm(
  resp ~ item + (1 | id),
  data = sim_ord_data,
  family = cumulative("logit"),
  chains = 2, iter = 1000, seed = 456
)

# ---- Stan-Code ausgeben ----
writeLines(stancode(fit_bin))
writeLines(stancode(fit_ord))

# ---- Modelle speichern ----
saveRDS(fit_bin, "fit_bin.rds")
saveRDS(fit_ord, "fit_ord.rds")



###############################################


