
###############################################################
#                                                             #
#    MIRT Analysis Project Dem,Tec,Pop                        #
#                                                             #
###############################################################

rm(list = ls())

# Define required packages
required_packages <- c(
  "here", "dplyr", "tidyr", "tidyverse", "forcats", "progress",
  "mirt", "psych", "lavaan", "blavaan", "semPlot",
  "ggplot2", "corrplot", "ggcorrplot", "factoextra", "reshape2", "kableExtra", "stargazer",
  "lme4", "MASS", "sandwich", "ashr", "cjbart", "fastDummies",
  "FactorHet", "Matrix", "mclust", "tgp", "FNN", "lbfgs", "mlr", "mlrMBO", "ParamHelpers", "smoof", "pracma",         
  "data.table", "scales", "tibble", "purrr", "stringr", "e1071"          
)


# Install missing packages
new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if(length(new_packages)) install.packages(new_packages)

# Load all packages
invisible(lapply(required_packages, library, character.only = TRUE))
rm(new_packages, required_packages)



#---------------------
# Main Dataset       #
#---------------------

data <- read.csv(here("data", "PreparedConjointData.csv"))
# Contains the original survey data used for the analysis, including all 18 key items (pop1-pop6, tec1-tec6, dem1-dem6)
# and respondent identifiers.


#------------------------ 
# Datasets Pooled Model #      
#------------------------

# 1. MIRT (4 Dimensions)

exploratory_models <- readRDS(here("data","exploratory_models.rds"))
# Contains the fitted exploratory factor models for dimensions ranging from 1 to 6.
# Each model is stored as an object in the list `exploratory_models`, representing the results of an exploratory
# factor analysis using the `mirt` package.
# Key outputs in each model include:
# - Factor loadings for each item on the specified number of dimensions.
# - Variance explained by the factors (SS loadings).
# - Factor correlations when multiple dimensions are used.
# - Fit statistics for the model (accessible separately, not stored within this object).
# These models are used to assess the dimensionality of the survey data and to evaluate item contributions
# across varying numbers of latent dimensions.


# 2. Item Diagnostics

fit_indices_exploratory <- read.csv(here("data","fit_indices_exploratory.csv"))
# Stores AIC, BIC, and log-likelihood values for exploratory models with 1 to 9 dimensions.
# Useful for comparing the overall model fit across different dimensional solutions. Ideally suited for extracting the factor loadings.

item_fit_exploratory <- readRDS(here("data", "itemfit_results.rds"))
# Contains item-level fit statistics (e.g., S-X², RMSEA, p-values) for each model dimension (1–9).
# Used to assess individual item misfit and identify overfitting or local dependence across dimensional solutions.

results <- list(
  models = exploratory_models,
  itemfit_tables = item_fit_exploratory,
  fit_table = fit_indices_exploratory
)
# Reconstructs the full MIRT analysis output as a structured list.
# Allows convenient access to models, item-level fit diagnostics, and global fit statistics for downstream analysis.
# This replicates the output of run_full_mirt() after restoring saved .rds and .csv files.


item_stats_exploratory <- read.csv(here("data","item_stats_exploratory.csv"))
# Contains detailed fit statistics (e.g., RMSEA, TLI, CFI, SRMR) and item fit statistics for exploratory models with 1-9 dimensions.
# Helps evaluate model fit quality and compare the appropriateness of different dimensional solutions.


# 3. Person Fit Evaluation

person_fit_exploratory <- read.csv(here("data","person_fit_exploratory.csv"))
# Contains person-fit statistics (e.g., Zh, Outfit, Infit) for exploratory MIRT models with 1-6 dimensions.
# Each row represents a respondent's fit statistics for a specific dimension, identified by their RespondentID.
# The dataset is used to assess the quality of responses and detect respondents whose answers may not align well with the model assumptions.


# 4. Latent Scores 


latent_scores_4 <- read.csv(here("data" ,"latent_scores_4_factors.csv"))
# Contains the latent factor scores for a 4-dimensional exploratory MIRT model.
# Each row corresponds to a respondent, identified by their ResponseId.
# The dataset includes:
# - Four latent dimensions, represented as continuous scores.
# - ResponseId to link the latent scores back to the original survey data.

merged_data_latent_scores <- read.csv(here("data", "merged_data_with_latent_scores.csv"))
# Contains the original survey data merged with the latent scores derived from the 3-dimensional, 4-dimensional and 5-dimensional exploratory MIRT model.
# This dataset serves as the foundation for subsequent analyses, including clustering based on latent dimensions.


analysis_data_final <- read.csv(here("data", "Analysis_Data_Final.csv"))
# Contains the cleaned and merged survey data, including clustering results for 3-dimensional, 4-dimensional and 5-dimensional solution, as well as distance-based weights (Inverse Distance & Exponential Decay) for both 4D and 5D cluster solutions.
# Each row represents a respondent, identified by their ResponseId.
# This dataset serves as the basis for all subsequent analyses, especially in the conjoint experiment.  


comparison_first_differences <- read.csv(here("data", "comparison_first_differences.csv"))
# This table aggregates simulation results from all conjoint-based 
# subgroup analyses across different weighting strategies and both 
# dimensional solutions (3D - 5D).

factorhet_results <- readRDS(here("data", "factorhet_results_full.RDS"))


#-----------------------------------       
# Datasets Country-Specific Models #      
#-----------------------------------

country_specific_models <- readRDS(here("data", "country_specific_exploratory_models.rds"))
# Contains exploratory MIRT models (1-9 dimensions) for each of the 16 countries in the survey dataset.
# The dataset is structured as a nested list:
# - Top-level: Each country's name (e.g., "Germany", "France").
# - Second-level: Exploratory models for 1-9 dimensions (e.g., "model_dim1", "model_dim2").

fit_indices_country_specific <- read.csv(here("data","fit_indices_country_specific.csv"))
# This dataset contains the model fit indices for country-specific exploratory MIRT models 
# across dimensions 1 to 9. The indices provide insights into how well each dimensional 
# solution fits the survey data for each country in the analysis´(stores AIC, BIC, and LogLik)
# Purpose:
# The dataset is used to compare model fit across countries and determine whether 
# the optimal number of dimensions varies by country.



#----------------------------------
# Filtering and Omitting Values
#----------------------------------


# Step 1: Define the 18 item columns and ResponseId
item_columns <- c(
  "ResponseId",           # Ensure ResponseId is included
  "pop1", "tec1", "dem1",  
  "pop2", "tec2", "dem2",  
  "pop3", "tec3", "dem3",  
  "pop4", "tec4", "dem4",  
  "pop5", "tec5", "dem5",  
  "pop6", "tec6", "dem6"   
)

# Step 2: Filter out rows with NAs in the 18 items but keep all 120 columns
data_clean <- data %>%
  filter(complete.cases(across(all_of(item_columns))))

# Step 3: Add a response pattern column to detect invariant responses
data_clean <- data_clean %>%
  unite(col = "response_pattern",
        c("pop1", "pop2", "pop3", "pop4", "pop5", "pop6",
          "dem1", "dem2", "dem3", "dem4", "dem5", "dem6",
          "tec1", "tec2", "tec3", "tec4", "tec5", "tec6"),
        sep = "", remove = FALSE)


# Step 4: Define invariant response patterns (all same responses across items)
invariant_patterns <- c(
  "111111111111111111", "222222222222222222", "333333333333333333",
  "444444444444444444", "555555555555555555", "666666666666666666",
  "777777777777777777"
)

data_clean <- data_clean %>%
  filter(!response_pattern %in% invariant_patterns)

response_ids <- data_clean$ResponseId

# Step 5: Create the response_matrix (only 18 items) and ResponseId
response_matrix <- data_clean %>%
  dplyr::select(pop1, pop2, pop3, pop4, pop5, pop6,
                tec1, tec2, tec3, tec4, tec5, tec6,
                dem1, dem2, dem3, dem4, dem5, dem6) %>%
  mutate(across(everything(), as.numeric)) %>%
  as.matrix()

item_names <- colnames(response_matrix)

# Step 6: Clean up unused variables (optional)
rm(item_columns, invariant_patterns)



#################################################################
#                                                               #
#                                                               # 
#                     Exploratory MIRT Model                    #
#                                                               # 
#                                                               #
#################################################################


run_full_mirt <- function(response_matrix, dims = 1:9, output_dir = "data") {
  library(mirt)
  library(tibble)
  library(here)
  
  exploratory_models <- list()
  fit_indices <- data.frame()
  itemfit_results <- list()
  
  for (num_factors in dims) {
    cat("\nFitting model with", num_factors, "dimensions...\n")
    
    tryCatch({
      # Estimate exploratory model
      mod <- mirt(
        data = response_matrix,
        model = num_factors,
        itemtype = "graded",
        exploratory = TRUE,
        method = "MHRM",
        se.type = "MHRM",
        verbose = TRUE,
        calcNull = TRUE
      )
      
      exploratory_models[[as.character(num_factors)]] <- mod
      
      # Extract global fit indices
      aic <- tryCatch(extract.mirt(mod, 'AIC'), error = function(e) NA)
      bic <- tryCatch(extract.mirt(mod, 'BIC'), error = function(e) NA)
      loglik <- tryCatch(extract.mirt(mod, 'logLik'), error = function(e) NA)
      converged <- tryCatch(mod@OptimInfo$converged, error = function(e) NA)
      
      fit_indices <- rbind(fit_indices, data.frame(
        Dimensions = num_factors,
        AIC = aic,
        BIC = bic,
        LogLik = loglik,
        Converged = converged
      ))
      
      # Extract item-level fit statistics
      itemfit_df <- tryCatch({
        itemfit(mod) %>%
          as.data.frame() %>%
          rownames_to_column("Item") %>%
          mutate(Dimensions = num_factors)
      }, error = function(e) {
        warning(paste("Item fit extraction failed for", num_factors, "dimensions:", e$message))
        NULL
      })
      
      itemfit_results[[as.character(num_factors)]] <- itemfit_df
      
    }, error = function(e) {
      cat("Error fitting model with", num_factors, "dimensions:\n", e$message, "\n")
    })
  }
  
  # Print final summary table
  cat("\nModel Fit Indices Summary:\n")
  print(fit_indices)
  
  # Save results
  saveRDS(exploratory_models, here(output_dir, "exploratory_models.rds"))
  saveRDS(itemfit_results, here(output_dir, "itemfit_results.rds"))
  write.csv(fit_indices, here(output_dir, "fit_indices_exploratory.csv"), row.names = FALSE)
  
  # Return as structured list
  invisible(list(
    models = exploratory_models,
    fit_table = fit_indices,
    itemfit_tables = itemfit_results
  ))
}

results <- run_full_mirt(response_matrix = response_matrix)

############################################
#                                          # 
# Dimensions     AIC     BIC    LogLik     #
#          1 1162630 1163627 -581188.9     #
#          2 1140699 1141830 -570206.3     #
#          3 1128596 1129854 -564138.9     #
#          4 1117366 1118742 -558509.0     #
#          5 1112366 1113854 -555995.2     #
#          6 1112013 1113604 -555805.7     #
#          7 1112251 1113936 -555912.7     #
#          8 1112381 1114154 -555966.7     #
#          9 1112945 1114797 -556238.7     #
#                                          #
############################################

#----------------------------------
# Item Fit Indices for 4D/5D & 6D
#----------------------------------

results$itemfit_tables[["4"]]  # For 4D 
results$itemfit_tables[["5"]]  # For 5D
results$itemfit_tables[["6"]]  # For 6D

# ---- 4D Model ----
# The 4-dimensional model shows consistently strong item-level fit:
# - All RMSEA.S_X2 values are well below the commonly used threshold of 0.015.
# - No item exceeds RMSEA ≈ .013, indicating a close fit between the model and observed item responses.
# - While most p-values are statistically significant (due to the large sample size),
#   this is not unusual and does not indicate misfit when RMSEA values remain low.
# → Conclusion: The 4D solution provides the most psychometrically stable structure.

# ---- 5D Model ----
# The 5-dimensional model shows some degradation in item fit:
# - Multiple items (e.g., dem2, dem3, dem5, dem6, tec5, pop5) exhibit RMSEA > 0.015.
# - This suggests overfitting, possible dimension fragmentation, or cross-loading instability.
# - Although global AIC improves slightly, item-level misfit increases substantially.
# → Conclusion: The 5D model adds complexity without improving item-level model fit.

# ---- 6D Model ----
# The 6-dimensional model exhibits severe item misfit:
# - Many items exceed RMSEA values of 0.03–0.07 — a level generally interpreted as poor fit.
# - All p-values are effectively zero, reinforcing the interpretation of structural instability.
# - Despite being the AIC-optimal solution globally, the item-level diagnostics point to major
#   overparameterization and a breakdown of model interpretability.
# → Conclusion: The 6D model is not acceptable for interpretation or scale construction.

#---------------------------------
# Factor Loadings for 4D/5D & 6D
#---------------------------------

summary(results$models[["3"]], rotate = "oblimin")$rotF
summary(results$models[["4"]], rotate = "oblimin")$rotF
summary(results$models[["5"]], rotate = "oblimin")$rotF
summary(results$models[["6"]], rotate = "oblimin")$rotF


#-------------------------------------------------------
# Additional Model Fit Statistics and Combined Item Stats 
#-------------------------------------------------------

extract_mirt_fitstats <- function(models, item_names, dims = 4:6, output_dir = "data") {
  library(mirt)
  library(here)
  library(tibble)
  
  combined_stats <- data.frame()
  
  for (num_factors in dims) {
    cat("\nExtracting fit statistics for", num_factors, "dimensions...\n")
    
    tryCatch({
      # Retrieve model
      mod <- models[[as.character(num_factors)]]
      
      # Compute global fit statistics (M2)
      fit <- M2(mod, type = "C2")
      rmsea <- fit$RMSEA[1]
      tli <- fit$TLI
      cfi <- fit$CFI
      srmr <- fit$SRMSR
      
      # Compute item-level fit statistics
      item_fit <- itemfit(mod)
      
      # Merge global + item-level fit into long format
      model_stats <- data.frame(
        Dimensions = num_factors,
        RMSEA = rmsea,
        TLI = tli,
        CFI = cfi,
        SRMR = srmr,
        Item = item_names,
        S_X2 = item_fit$S_X2,
        df_S_X2 = item_fit$df.S_X2,
        RMSEA_S_X2 = item_fit$RMSEA.S_X2,
        p_S_X2 = item_fit$p.S_X2
      )
      
      combined_stats <- rbind(combined_stats, model_stats)
      
    }, error = function(e) {
      cat("Error processing model with", num_factors, "dimensions:\n", e$message, "\n")
    })
  }
  
  # Save results
  write.csv(combined_stats, here(output_dir, "item_stats_exploratory.csv"), row.names = FALSE)
  
  # Return for immediate inspection
  invisible(combined_stats)
}

combined_stats <- extract_mirt_fitstats(
  models = exploratory_models,
  item_names = item_names,
  dims = 4:6,
  output_dir = "data"
)

#-----------------------------------------------
# Extraction of Item Discrimination Parameters
#-----------------------------------------------

# Define model IDs to evaluate
model_ids <- c("4")

# Initialize list to store evaluation results
evaluation_list <- list()

# Loop through selected models
for (model_id in model_ids) {
  message("Processing Model with ", model_id, " dimensions")
  
  mod <- results$models[[model_id]]
  itemfit <- results$itemfit_tables[[model_id]]
  num_factors <- as.numeric(model_id)
  factor_names <- paste0("F", 1:num_factors)
  
  # 1. Extract rotated factor loadings and communalities
  rotated <- summary(mod, rotate = "oblimin")
  rotated_loadings <- rotated$rotF %>%
    as.data.frame() %>%
    rownames_to_column("Item")
  
  communality <- tibble(
    Item = rownames(rotated$rotF),
    Communality = rotated$h2
  )
  
  loadings <- rotated_loadings %>%
    left_join(communality, by = "Item") %>%
    mutate(
      Dominant_Loading = pmap_dbl(dplyr::select(., all_of(factor_names)), ~ max(abs(c(...)))),
      Dominant_Factor = pmap_chr(dplyr::select(., all_of(factor_names)), function(...) {
        vals <- c(...)
        names(vals) <- factor_names
        names(vals)[which.max(abs(vals))]
      }),
      Cross_Loading = rowSums(abs(dplyr::select(., all_of(factor_names))) >= 0.30) > 1
    ) %>%
    dplyr::select(Item, Dominant_Factor, Dominant_Loading, Cross_Loading, Communality)
  
  # 2. Extract discrimination parameters (a-values)
  a_params <- coef(mod, IRTpars = TRUE, simplify = TRUE)$items %>%
    as.data.frame() %>%
    rownames_to_column("Item") %>%
    rowwise() %>%
    mutate(Max_Discrimination = max(abs(c_across(starts_with("a"))))) %>%
    ungroup()
  
  # 3. Extract RMSEA from item fit
  rmsea_table <- itemfit %>%
    as.data.frame() %>%
    dplyr::select(Item = item, RMSEA.S_X2)
  
  # 4. Combine all diagnostics into one evaluation table
  item_eval <- loadings %>%
    left_join(a_params, by = "Item") %>%
    left_join(rmsea_table, by = "Item") %>%
    mutate(
      Block = case_when(
        str_detect(Item, "pop") ~ "Populism",
        str_detect(Item, "tec") ~ "Technocracy",
        str_detect(Item, "dem") ~ "Democracy",
        TRUE ~ "Other"
      ),
      High_Discrimination = Max_Discrimination > 2.5,
      Recommendation = case_when(
        Dominant_Loading < 0.30 ~ "Exclude: weak loading",
        Communality < 0.30 ~ "Review: low communality",
        Max_Discrimination < 0.50 ~ "Exclude: low discrimination",
        High_Discrimination ~ "Review: extreme discrimination",
        RMSEA.S_X2 > 0.03 ~ "Review: poor fit",
        Cross_Loading ~ "Review: cross-loading",
        TRUE ~ "Retain"
      ),
      Model = paste0(model_id, "D")
    )
  
  # Store the evaluation table
  evaluation_list[[model_id]] <- item_eval
  
  # Combine results across models
  evaluation_results <- bind_rows(evaluation_list)
  
  # Print output
  print(evaluation_results)
}

rm(model_ids,evaluation_list,model_id,mod,itemfit,num_factors,factor_names,rotated,
   rotated_loadings,communality,loadings,a_params,rmsea_table,item_eval,evaluation_results)


# -------------------------------
# Person Fit Evaluation (e.g., 4D Model)
# -------------------------------

# Select the dimensions to be evaluated (can be 3, 4, 5, etc.)
dimensions_to_check <- c(4)

# Compute person-fit statistics for all selected dimensions
person_fit_results <- purrr::map_dfr(dimensions_to_check, function(dim) {
  message("Calculating person fit for ", dim, " dimensions...")
  
  mod <- exploratory_models[[as.character(dim)]]
  
  tryCatch({
    fit <- personfit(mod)
    tibble(
      ResponseId = response_ids,
      Dimension = dim,
      Zh = fit$Zh,
      Infit = fit$infit,
      Outfit = fit$outfit
    )
  }, error = function(e) {
    message("Error in dimension ", dim, ": ", e$message)
    return(NULL)
  })
})

# Save the results to CSV
write.csv(person_fit_results, here("data", "person_fit_exploratory.csv"), row.names = FALSE)

# -------------------------------
# Classification Based on Person-Fit Indices
# Thresholds from Jones et al. (2023) and Felt et al. (2017)
# -------------------------------

infit_upper <- 1.7       # Above this = possible inconsistency
outfit_upper <- 1.6      # Above this = outlier response behavior
zh_upper <- 2            # Overfit threshold
zh_lower <- -2         # Misfit threshold

# Flag respondents based on one or more misfit criteria
person_fit_classified <- person_fit_exploratory %>%
  mutate(fit_status = case_when(
    Zh < zh_lower ~ "Misfit_Zh",
    Zh > zh_upper ~ "Overfit_Zh",
    Infit > infit_upper ~ "Misfit_Infit",
    Outfit > outfit_upper ~ "Misfit_Outfit",
    TRUE ~ "OK"
  )) %>%
  filter(fit_status != "OK")  # Retain only flagged cases

# -------------------------------
# Merge Misfitting Respondents with Their Raw Response Data
# -------------------------------

# Define the 18 item variables
items <- c(paste0("pop", 1:6), paste0("tec", 1:6), paste0("dem", 1:6))

# Select country + item data for each respondent
respondent_info <- data_clean %>%
  dplyr::select(ResponseId, ctry, all_of(items))

# Merge person-fit flags with response patterns
flagged_respondents_df <- person_fit_classified %>%
  left_join(respondent_info, by = "ResponseId") %>%
  dplyr::select(ResponseId, ctry, Zh, Infit, Outfit, fit_status, all_of(items))



#-----------------------------------------------------
# Extraction of Latent Scores for 4-dimensional model
#-----------------------------------------------------

# Extract latent scores using MAP
latent_scores_4 <- fscores(
  exploratory_models[[4]],   # 4-dimensional model
  method = "MAP",            # MAP estimation
  full.scores = TRUE,        # Return scores for each respondent
  theta_lim = c(-4, 4)       # Define limits for the latent trait integral (adjust as needed)
)


# Combine response_ids with latent scores
latent_scores_4_dimensions <- data.frame(ResponseId = response_ids, latent_scores_4)

# Rename the dimensions in the latent_scores_4_with_ids dataset
colnames(latent_scores_4_dimensions)[-1] <- c(
  "pluralist_democracy_4d", 
  "populist_dualism_4d", 
  "competitive_representation_4d", 
  "independent_expertise_4d"
)

# Save the latent_scores for 4D as CSV
write.csv(latent_scores_4_dimensions, here("data","latent_scores_4_factors.csv"), row.names = FALSE)

# Merge the datasets based on ResponseId, keeping only matching observations
merged_data_latent_scores <- merge(data_clean, latent_scores_4_dimensions, by = "ResponseId", all = FALSE)


# Include age_group and edu_group variable for further processing
merged_data_latent_scores <- merged_data_latent_scores %>%
  mutate(age_group = cut(age,
                         breaks = c(15, 29, 44, 59, 74, 100),
                         labels = c("18–29", "30–44", "45–59", "60–74", "75+")))

merged_data_latent_scores <- merged_data_latent_scores %>%
  mutate(edu_group = case_when(
    edulvl %in% 0:5 ~ "Low",
    edulvl %in% 6:10 ~ "Medium",
    edulvl %in% 11:15 ~ "High",
    edulvl >= 16 ~ "Very High",
    TRUE ~ NA_character_
  ))

# Save combined dataset as CSV

write.csv(merged_data_latent_scores, here("data", "merged_data_with_latent_scores.csv"))


#################################################################
#                                                               #
#                                                               # 
#                     Conjoint Analysis                         #
#                                                               # 
#                                                               #
#################################################################


# ------------------------------------------------------------
# Conjoint Analysis 1 – AMCE Estimation (Unweighted Models)
# ------------------------------------------------------------
# This analysis estimates Average Marginal Component Effects (AMCEs) 
# for policy and institutional preferences using a hierarchical linear model. 
# No weights or latent dimensions are applied at this stage.
# ------------------------------------------------------------

# Ensure 'ctry' and 'ResponseId' are factors for hierarchical modeling
ConjointData <-  merged_data_latent_scores%>% 
  mutate(
    ctry = factor(ctry),
    ResponseId = factor(ResponseId)
  )

# Split data into 3 conjoint rounds
round1 <- ConjointData %>%
  dplyr::select(-ends_with(c("round2", "round3")),
                -c("c2me", "c2p", "p2a", "p2b", "p2c", "p2d", "p2e", "p2f", "p2g", "p2temp",
                   "c3me", "c3p", "p3a", "p3b", "p3c", "p3d", "p3e", "p3f", "p3g", "p3temp")) %>%
  mutate(round = 1)

round2 <- ConjointData %>%
  dplyr::select(-ends_with(c("round1", "round3")),
                -c("c1me", "c1p", "p1a", "p1b", "p1c", "p1d", "p1e", "p1f", "p1g", "p1temp",
                   "c3me", "c3p", "p3a", "p3b", "p3c", "p3d", "p3e", "p3f", "p3g", "p3temp")) %>%
  mutate(round = 2)

round3 <- ConjointData %>%
  dplyr::select(-ends_with(c("round1", "round2")),
                -c("c2me", "c2p", "p2a", "p2b", "p2c", "p2d", "p2e", "p2f", "p2g", "p2temp",
                   "c1me", "c1p", "p1a", "p1b", "p1c", "p1d", "p1e", "p1f", "p1g", "p1temp")) %>%
  mutate(round = 3)

# Rename dimension variables uniformly across rounds
round1 <- round1 %>% rename(
  dim1 = instidim1_round1, dim2 = instidim2_round1, dim3 = instidim3_round1, dim4 = instidim4_round1,
  dim5 = policydim1_round1, dim6 = policydim2_round1, dim7 = policydim3_round1, dim8 = policydim4_round1,
  cme = c1me, cp = c1p, pa = p1a, pb = p1b, pc = p1c, pd = p1d, pe = p1e, pf = p1f, pg = p1g, ptemp = p1temp
)

round2 <- round2 %>% rename(
  dim1 = instidim1_round2, dim2 = instidim2_round2, dim3 = instidim3_round2, dim4 = instidim4_round2,
  dim5 = policydim1_round2, dim6 = policydim2_round2, dim7 = policydim3_round2, dim8 = policydim4_round2,
  cme = c2me, cp = c2p, pa = p2a, pb = p2b, pc = p2c, pd = p2d, pe = p2e, pf = p2f, pg = p2g, ptemp = p2temp
)

round3 <- round3 %>% rename(
  dim1 = instidim1_round3, dim2 = instidim2_round3, dim3 = instidim3_round3, dim4 = instidim4_round3,
  dim5 = policydim1_round3, dim6 = policydim2_round3, dim7 = policydim3_round3, dim8 = policydim4_round3,
  cme = c3me, cp = c3p, pa = p3a, pb = p3b, pc = p3c, pd = p3d, pe = p3e, pf = p3f, pg = p3g, ptemp = p3temp
)

# Combine all rounds into one dataset
DataRounds <- rbind(round1, round2, round3)

# Remove invalid observations (tokens outside 0-10 range)
DataRounds <- DataRounds %>% filter(cp >= 0 & cp <= 10)

# Convert relevant variables to factors
DataRounds <- DataRounds %>% 
  mutate(across(starts_with("dim"), as.factor)) %>% 
  mutate(across(ends_with("att_order"), as.factor)) %>% 
  mutate(round = factor(round),
         female = case_when(gender == 1 ~ 1,
                            gender == 0 ~ 0,
                            TRUE ~ NA_real_))

# Create dummy variables for AMCE estimation
DataRoundsDummies <- fastDummies::dummy_cols(DataRounds,
                                             select_columns = c("dim1", "dim2", "dim3", "dim4",
                                                                "dim5", "dim6", "dim7", "dim8"),
                                             remove_first_dummy = FALSE)

# Define plotting function for AMCE estimates
plot_model <- function(model, adjusted = T) {
  # extract coefficients + standard errors
  Coeffs <- model@beta
  VarCov <- vcov(model)
  SEs <- sqrt(diag(VarCov))
  LowCIs <- Coeffs - 1.96 * SEs
  UPCIs <- Coeffs + 1.96 * SEs
  
  # create dataframe for plotting
  PlotData <- data.frame(
    estimate = Coeffs,
    low_ci = LowCIs,
    up_ci = UPCIs
  )
  
  # adjust for multiple hypotheses testing
  PenalizedAMCEs <- ash(Coeffs, SEs)
  PlotData$post_mean <- PenalizedAMCEs$result$PosteriorMean
  PlotData$post_low_ci <- PlotData$post_mean - 1.96 * PenalizedAMCEs$result$PosteriorSD
  PlotData$post_up_ci <- PlotData$post_mean + 1.96 * PenalizedAMCEs$result$PosteriorSD
  
  # remove irrelevant coefficients
  PlotData <- PlotData[-1, ]
  PlotData <- PlotData[1:16, ]
  
  # insert rows with 0s for reference categories
  RefRow <- rep(0, 3)
  PlotData <- rbind(
    PlotData[1, ], RefRow, PlotData[2:3, ], RefRow,
    PlotData[4:5, ], RefRow, PlotData[6:7, ], RefRow,
    PlotData[8:9, ], RefRow, PlotData[10:11, ], RefRow,
    PlotData[12:13, ], RefRow, PlotData[14:15, ], RefRow,
    PlotData[16, ]
  )
  
  # add questions and attributes
  # PlotData$fake_y <- rep(c(-1, 0, 1), 8)
  PlotData$questions <- factor(c(
    rep("Who should make policy decisions?", 3),
    rep("What should be the basis of politicial decisions?", 3),
    rep("What should politics reflect?", 3),
    rep("What is the nature of politics?", 3),
    rep("What should economic policies look like?", 3),
    rep("What should immigration policies look like?", 3),
    rep("What should climate change policies look like?", 3),
    rep("What should cultural policies look like?", 3)
  )) %>%
    fct_inorder(ordered = T)
  PlotData$attribute <- factor(c(
    "The People", "Political Representatives", "Experts",
    "Will of the People", "Compromise", "Independent Expertise",
    "Immediate Demands of the People", "Balance Immediate Demands\nand Long-term Needs",
    "Long-term Needs",
    "A Struggle between Good and Evil", "Competition between Political Ideas",
    "Finding Objective Solutions",
    "No Compensation of Companies\nand Households for Higher Prices",
    "Partial Compensation of Comapanies\nand Households for Higher Prices",
    "Full Compensation of Companies\nand Households for Higher Prices",
    "No Integration Programs\nfor Immigrants", "Optional Integration Programs\nfor Immigrants",
    "Mandatory Integration Programs\nfor Immigrants",
    "No Action to Reduce\nClimate Change", "Moderate Action to Reduce\nClimate Change",
    "Strong Action to Reduce\nClimate Change",
    "No Constraints of Religious Symbols\nin Public Space",
    "Regulation of Religious Symbols\nin Public Space",
    "Ban of Religious Symbols\nfrom Public Space"
  )) %>%
    fct_inorder(ordered = T)
  
  if (adjusted == F) {
    # plot AMCEs (not adjusted)
    ggplot(PlotData) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
      geom_point(aes(x = estimate, y = attribute), size = 2) +
      geom_linerange(aes(xmin = low_ci, xmax = up_ci, y = attribute)) +
      facet_wrap(~questions, ncol = 2, scales = "free_y") +
      ylab("") +
      xlab("Average Marginal Component Effect") +
      theme_minimal()
  } else {
    # plot AMCEs (adjusted)
    ggplot(PlotData) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
      geom_point(aes(x = post_mean, y = attribute), size = 2) +
      geom_linerange(aes(xmin = post_low_ci, xmax = post_up_ci, y = attribute)) +
      facet_wrap(~questions, ncol = 2, scales = "free_y") +
      ylab("") +
      xlab("Average Marginal Component Effect") +
      theme_minimal()
  }
}

# Fit hierarchical linear model without weights (unweighted AMCE estimation)
model_amce_unweighted <- lmer(
  cp ~ dim1_1 + dim1_2 + dim2_1 + dim2_2 +
    dim3_1 + dim3_2 + dim4_1 + dim4_2 +
    dim5_1 + dim5_2 + dim6_1 + dim6_2 +
    dim7_1 + dim7_2 + dim8_1 + dim8_2 +
    female + age + edulvl + lrscale +
    (1 | ctry / ResponseId),
  data = DataRoundsDummies,
  REML = FALSE
)

summary(model_amce_unweighted)


# ------------------------------------------------------------
# Additional Robustness Models for Conjoint Analysis 1
# ------------------------------------------------------------

# Model 2: No Covariates (only profile attributes)
model_amce_nocovariates <- lmer(
  cp ~ dim1_1 + dim1_2 + dim2_1 + dim2_2 +
    dim3_1 + dim3_2 + dim4_1 + dim4_2 +
    dim5_1 + dim5_2 + dim6_1 + dim6_2 +
    dim7_1 + dim7_2 + dim8_1 + dim8_2 +
    (1 | ctry / ResponseId),
  data = DataRoundsDummies,
  REML = FALSE
)

# Model 3: No Country-Level Random Intercept
model_amce_nocountry <- lmer(
  cp ~ dim1_1 + dim1_2 + dim2_1 + dim2_2 +
    dim3_1 + dim3_2 + dim4_1 + dim4_2 +
    dim5_1 + dim5_2 + dim6_1 + dim6_2 +
    dim7_1 + dim7_2 + dim8_1 + dim8_2 +
    female + age + edulvl + lrscale +
    (1 | ResponseId),
  data = DataRoundsDummies,
  REML = FALSE
)

# ------------------------------------------------------------
# Export All Three Models to LaTeX Table
# ------------------------------------------------------------
stargazer(model_amce_unweighted, model_amce_nocovariates, model_amce_nocountry,
          type = "latex",
          title = "AMCE Estimates – Robustness Checks",
          column.labels = c("Full Model", "No Covariates", "No Country-Level RE"),
          covariate.labels = c(
            "Dim 1: The People", "Dim 1: Experts",
            "Dim 2: Will of the People", "Dim 2: Independent Expertise",
            "Dim 3: Immediate Demands", "Dim 3: Long-term Needs",
            "Dim 4: Good and Evil", "Dim 4: Objective Solutions",
            "Dim 5: No Compensation", "Dim 5: Full Compensation",
            "Dim 6: No Integration Programs", "Dim 6: Mandatory Integration Programs",
            "Dim 7: No Climate Action", "Dim 7: Strong Climate Action",
            "Dim 8: No Constraints Religious Symbols", "Dim 8: Ban Religious Symbols",
            "Female", "Age", "Education Level", "Left-Right Scale"
          ),
          dep.var.caption = "Dependent Variable: Number of Tokens Assigned",
          dep.var.labels = "Player 2 Tokens (0–10)",
          omit.stat = c("f", "ser"),  # Omit F-statistics and standard error summary
          digits = 3)

rm(analysis_data_final, ConjointData, DataRounds, round1, round2, round3)

# ----------------------------------------------------------------
# Conjoint Analysis 3 – Simulated First Differences (3D, 4D, 5D)
# This script runs a simulation of first differences in expected 
# token allocation for all cluster types in the 3D, 4D, and 5D solution.
# The results are visualized in separate high-quality plots and saved.
# ----------------------------------------------------------------


# STEP 1: Simulation Function

# write custom function for the simulation
sim_subgroups <- function(subgroup = NULL, seed, adjusted = FALSE, data = DataRoundsDummies, weight_var = "idw_weight_3d") {
  
  # filter possible subgroup
  if (!is.null(subgroup)) {
    data <- data %>% filter(!!rlang::sym(subgroup) == 1)
  }
  
  # estimate model
  HLModel <- lmer(
    cp ~ dim1_1 + dim1_2 + dim2_1 + dim2_2 +
      dim3_1 + dim3_2 + dim4_1 + dim4_2 +
      dim5_1 + dim5_2 + dim6_1 + dim6_2 +
      dim7_1 + dim7_2 + dim8_1 + dim8_2 +
      female + age + edulvl + lrscale +
      (1 | ctry / ResponseId),
    data = data,
    weights = data[[weight_var]],
    REML = TRUE # using REML assures convergence
  )
  
  # get fixed coefficients
  FixedCoeffs <- fixef(HLModel)
  # get fixed variance covariance matrix
  FixedVarCov <- as.matrix(vcov(HLModel))
  
  if (!adjusted) {
    
    # draw from multivariate normal distribution
    set.seed(seed)
    MVNormMat <- MASS::mvrnorm(n = 1000, mu = FixedCoeffs, Sigma = FixedVarCov)
  } else {
    
    # add penalty for multiple testing
    FixedSEs <- sqrt(diag(FixedVarCov))  
    Penalized <- ash(FixedCoeffs, FixedSEs)
    PenalizedMean <- Penalized$result$PosteriorMean
    PenalizedSD <- Penalized$result$PosteriorSD
    PenalizedVarCov <- FixedVarCov
    diag(PenalizedVarCov) <- PenalizedSD^2
    PenalizedVarCov <- as.matrix(nearPD(PenalizedVarCov)$mat)
    
    # draw from multivariate normal distribution
    set.seed(seed)
    MVNormMat <- MASS::mvrnorm(n = 1000, mu = PenalizedMean, Sigma = PenalizedVarCov)
  }
  
  # select independent variables and add
  # country + respondent intercepts
  IVs <- data %>%
    dplyr::select(dim1_1, dim1_2, dim2_1, dim2_2, dim3_1, dim3_2, dim4_1, dim4_2,
           dim5_1, dim5_2, dim6_1, dim6_2, dim7_1, dim7_2, dim8_1, dim8_2,
           female, age, edulvl, lrscale, ctry, ResponseId) %>%
    mutate(ctry_resp_intercept = 1) %>%
    relocate(ctry_resp_intercept) %>%
    drop_na()
  
  # add random effects for countries and respondents
  RanEff <- ranef(HLModel)
  RanEffCtry <- rownames_to_column(RanEff$ctry)
  RanEffResp <- rownames_to_column(RanEff$`ResponseId:ctry`)
  RanEffResp$rowname <- gsub(":.*", "", RanEffResp$rowname)
  
  IVs <- IVs %>%
    left_join(RanEffCtry, by = c("ctry" = "rowname")) %>%
    left_join(RanEffResp, by = c("ResponseId" = "rowname")) %>%
    rowwise() %>%
    mutate(ctry_resp_intercept = ctry_resp_intercept + `(Intercept).x` + `(Intercept).y`) %>%
    ungroup() %>%
    dplyr::select(-`(Intercept).x`, -`(Intercept).y`, -ctry, -ResponseId) %>%
    as.matrix()
  
  # Indexe
  idx_list <- lapply(1:8, function(i) which(str_detect(colnames(IVs), paste0("dim", i, "_"))))
  
  # set up array with adjusted variable values
  AdjIVs <- array(rep(IVs, 24), dim = c(nrow(IVs), ncol(IVs), 24))
  
  for (i in 1:8) {
    a <- (i - 1) * 3
    AdjIVs[, idx_list[[i]][1], a + 1] <- 1
    AdjIVs[, idx_list[[i]][2], a + 1] <- 0
    AdjIVs[, idx_list[[i]],     a + 2] <- 0
    AdjIVs[, idx_list[[i]][1], a + 3] <- 0
    AdjIVs[, idx_list[[i]][2], a + 3] <- 1
  }
  
  # empty matrix to store expected values
  ExpVals <- matrix(NA, 1000, 24)
  for (i in 1:24) {
    ExpVals[, i] <- apply(MVNormMat, 1, function(s) mean(AdjIVs[,,i] %*% s))
  }
  
  # calculate first differences
  diffmat <- cbind(
    tech1_minus_dem1  = ExpVals[,3]  - ExpVals[,2],
    pop1_minus_dem1   = ExpVals[,1]  - ExpVals[,2],
    pop1_minus_tech1  = ExpVals[,1]  - ExpVals[,3],
    tech2_minus_dem2  = ExpVals[,6]  - ExpVals[,5],
    pop2_minus_dem2   = ExpVals[,4]  - ExpVals[,5],
    pop2_minus_tech2  = ExpVals[,4]  - ExpVals[,6],
    tech3_minus_dem3  = ExpVals[,9]  - ExpVals[,8],
    pop3_minus_dem3   = ExpVals[,7]  - ExpVals[,8],
    pop3_minus_tech3  = ExpVals[,7]  - ExpVals[,9],
    tech4_minus_dem4  = ExpVals[,12] - ExpVals[,11],
    pop4_minus_dem4   = ExpVals[,10] - ExpVals[,11],
    pop4_minus_tech4  = ExpVals[,10] - ExpVals[,12],
    strong1_minus_mid = ExpVals[,15] - ExpVals[,14],
    no1_minus_mid     = ExpVals[,13] - ExpVals[,14],
    no1_minus_strong1 = ExpVals[,13] - ExpVals[,15],
    strong2_minus_mid = ExpVals[,18] - ExpVals[,17],
    no2_minus_mid     = ExpVals[,16] - ExpVals[,17],
    no2_minus_strong2 = ExpVals[,16] - ExpVals[,18],
    strong3_minus_mid = ExpVals[,21] - ExpVals[,20],
    no3_minus_mid     = ExpVals[,19] - ExpVals[,20],
    no3_minus_strong3 = ExpVals[,19] - ExpVals[,21],
    strong4_minus_mid = ExpVals[,24] - ExpVals[,23],
    no4_minus_mid     = ExpVals[,22] - ExpVals[,23],
    no4_minus_strong4 = ExpVals[,22] - ExpVals[,24]
  )
  
  # create dataframe and get CIs + mean
  out <- apply(diffmat, 2, function(x) quantile(x, c(.025, .975)))
  mean_diff <- colMeans(diffmat)
  
  # add questions
  result <- tibble(
    lower = out[1,], upper = out[2,], mean = mean_diff,
    question = rep(c(
      "Who should make decisions?",
      "What should be the basis?",
      "What should politics reflect?",
      "What is the nature of politics?",
      "Economic policy",
      "Immigration policy",
      "Climate policy",
      "Cultural policy"
    ), each = 3),
    # add comparisons
    comparison = c(
      "Experts – Political Representatives",
      "The People – Political Representatives",
      "The People – Experts",
      "Independent Expertise – Compromise",
      "Will of the People – Compromise",
      "Will of the People – Independent Expertise",
      "Long-term Needs – Balanced Needs",
      "Immediate Demands – Balanced Needs",
      "Immediate Demands – Long-term Needs",
      "Objective Solutions – Political Competition",
      "Good vs Evil – Political Competition",
      "Good vs Evil – Objective Solutions",
      "Full Compensation – Partial Compensation",
      "No Compensation – Partial Compensation",
      "No Compensation – Full Compensation",
      "Mandatory Programs – Optional Programs",
      "No Programs – Optional Programs",
      "No Programs – Mandatory Programs",
      "Strong Action – Moderate Climate Action",
      "No Action – Moderate Climate Action",
      "No Action – Strong Climate Action",
      "Ban Symbols – Regulate Symbols",
      "No Constraints – Regulate Symbols",
      "No Constraints – Ban Symbols"
    )
  )
  
  return(result)
}


# STEP 2: Plot function
plot_simulated_fd <- function(df, cluster_dim, weight_type, output_path) {
  df$question <- factor(df$question, levels = unique(df$question))
  df$subtype <- fct_relabel(df$subtype, str_wrap, width = 25)
  
  df <- df %>%
    mutate(
      significant = ifelse(lower > 0 | upper < 0, TRUE, FALSE),
      alpha_value = ifelse(significant, 1, 0.4),
      point_size = ifelse(significant, 3.5, 2.5)
    )
  
  p <- ggplot(df, aes(
    x = mean,
    y = fct_reorder(comparison, mean),
    color = subtype,
    alpha = alpha_value
  )) +
    geom_vline(xintercept = 0, color = "grey60", linetype = "dashed", linewidth = 0.6) +
    geom_point(aes(size = point_size), position = position_dodge(width = 0.6)) +
    geom_errorbarh(
      aes(xmin = lower, xmax = upper),
      height = 0.25,
      position = position_dodge(width = 0.6),
      linewidth = 0.9
    ) +
    facet_wrap(~question, scales = "free_y", ncol = 2) +
    scale_color_brewer(palette = "Set2") +
    scale_alpha_identity() +
    scale_size_identity() +
    labs(
      title = paste0("Subgroup Effects – ", toupper(cluster_dim), " Clusters (", weight_type, ")"),
      subtitle = "Simulated First Differences in Expected Token Allocation (95% Confidence Intervals)",
      x = "First Difference (Expected Tokens)",
      y = "Profile Comparison",
      color = "Cluster"
    ) +
    theme_classic(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 17),
      plot.subtitle = element_text(size = 12, margin = margin(b = 8)),
      strip.text = element_text(face = "bold", size = 13),
      axis.title.x = element_text(face = "bold", margin = margin(t = 10)),
      axis.title.y = element_text(face = "bold", margin = margin(r = 10)),
      axis.text.y = element_text(size = 11, hjust = 0),
      legend.position = "bottom",
      legend.title = element_text(face = "bold"),
      legend.text = element_text(size = 11),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA)
    )
  
  filename <- paste0("SimulatedFD_", toupper(cluster_dim), "_", weight_type, ".png")
  ggsave(file.path(output_path, filename), plot = p, width = 14, height = 8, dpi = 300, bg = "white")
}



# STEP 3: Full Loop
run_simulated_fd_loop <- function(cluster_vars, output_path, output_data_path) {
  weight_types <- c("idw", "expdecay")
  all_results <- list()
  
  for (cluster_var in cluster_vars) {
    cluster_labels <- unique(DataRoundsDummies[[cluster_var]])
    cluster_dim <- str_extract(cluster_var, "3d|4d|5d")
    
    for (weight_type in weight_types) {
      weight_var <- paste0(weight_type, "_weight_", cluster_dim)
      
      for (label in cluster_labels) {
        data_with_dummy <- DataRoundsDummies %>%
          mutate(dummy_subgroup = ifelse(!!rlang::sym(cluster_var) == label, 1, 0))
        
        result <- sim_subgroups(
          subgroup = "dummy_subgroup",
          seed = 1000 + which(cluster_labels == label),
          adjusted = TRUE,
          data = data_with_dummy,
          weight_var = weight_var
        ) %>%
          mutate(
            subtype = label,
            cluster_dim = toupper(cluster_dim),
            weight_type = weight_type
          )
        
        all_results[[paste0(cluster_dim, "_", weight_type, "_", label)]] <- result
      }
    }
  }
  
  combined_df <- bind_rows(all_results)
  write.csv(combined_df, file = file.path(output_data_path, "comparison_first_differences.csv"), row.names = FALSE)
  return(combined_df)
}


# STEP 4: Execute
DataRoundsDummies$idw_none <- 1  # fallback weights
output_folder <- here("output")
output_data_folder <- here("data")
cluster_vars <- c("cluster_3d_label", "cluster_4d_label", "cluster_5d_label")

# Run loop and get output lists
sim_output <- run_simulated_fd_loop(cluster_vars, output_folder, output_data_folder)


# STEP 5: Create plots for each cluster dimension with all subtypes included
unique_dims <- unique(sim_output$cluster_dim)
weight_types <- unique(sim_output$weight_type)

for (dim in unique_dims) {
  for (weight in weight_types) {
    plot_data <- sim_output %>%
      filter(cluster_dim == dim, weight_type == weight)
    
    plot_simulated_fd(plot_data, dim, weight, output_folder)
  }
}


####################################
# Heterogeneous Causal Effects via FactorHet (MBO-Optimized Version)
# Using experimentally manipulated treatment dummies (dim1_1 to dim8_2)
# with automatic lambda tuning (resource-aware)
####################################

library(FactorHet)
library(dplyr)
library(here)

# Create binary gender dummy: 1 = female, 0 = male
DataRoundsDummies$female <- NA
DataRoundsDummies$female[DataRoundsDummies$gender == 1] <- 1
DataRoundsDummies$female[DataRoundsDummies$gender == 0] <- 0

# Define treatment variables based on experimental design (factorial dummy-coding)
treatments <- grep("^dim[1-8]_[1-2]$", names(DataRoundsDummies), value = TRUE)

# Define moderator variables (political attitudes + sociodemographics)
moderators <- c(
  "pluralist_democracy_4d", "populist_dualism_4d",
  "competitive_representation_4d", "independent_expertise_4d",
  "female", "age_group", "edu_group"
)

# Ensure proper factor structure
DataRoundsDummies <- DataRoundsDummies %>%
  mutate(
    ResponseId = factor(ResponseId),
    round = factor(round),
    ctry = factor(ctry)
  )

# Create model formulas
treatment_formula <- as.formula(paste("cp_binary ~", paste(treatments, collapse = " + ")))
moderator_formula <- as.formula(paste("~", paste(moderators, collapse = " + ")))

# Define thresholds and latent subgroups to test
thresholds <- 4:6
K_values <- 2:4
factorhet_results <- list()

for (K in K_values) {
  factorhet_results[[paste0("K_", K)]] <- list()
  
  for (threshold in thresholds) {
    
    # Binary outcome based on token threshold
    DataRoundsDummies$cp_binary <- ifelse(DataRoundsDummies$cp >= threshold, 1, 0)
    
    if (length(unique(DataRoundsDummies$cp_binary)) == 2) {
      
      model_data <- DataRoundsDummies %>%
        dplyr::select(cp_binary, ResponseId, round, ctry, all_of(treatments), all_of(moderators)) %>%
        dplyr::filter(complete.cases(.))
      
      if (nrow(model_data) > 100) {
        
        set.seed(123)
        
        fit <- FactorHet::FactorHet_mbo(
          formula = treatment_formula,
          design = model_data,
          K = K,
          moderator = moderator_formula,
          group = ~ ResponseId,
          task = ~ round,
          mbo_control = FactorHet_mbo_control(
            criterion = "BIC",
            iters = 7,
            mbo_range = c(-4.5, -0.5),
            mbo_method = "regr.bgp",
            se_final = TRUE,
            verbose = FALSE
          ),
          control = FactorHet_control(
            iterations = 1000,
            tolerance.logposterior = 1e-5,
            rare_threshold = 0,
            do_SQUAREM = TRUE,
            return_data = TRUE
          ),
          initialize = FactorHet_init(
            short_EM = TRUE,
            short_EM_it = 30,
            nrep = 3,
            short_EM_init = "random_member"
          )
        )

factorhet_results[[paste0("K_", K)]][[paste0("cp_ge_", threshold)]] <- fit

cat("\u2714 Finished: K =", K, "| cp >=", threshold, "| N =", nrow(model_data), "\n")

      } else {
        cat("\u26A0 Skipped: Not enough observations (K =", K, ", cp >=", threshold, ")\n")
      }
    } else {
      cat("\u26A0 Skipped: No variation in cp_binary for threshold >=", threshold, "\n")
    }
  }
}

# Save results safely
saveRDS(factorhet_results, file = here("data", "factorhet_results.RDS"))


meta_summary <- expand.grid(K = K_values, threshold = thresholds)
meta_summary$BIC <- NA
meta_summary$logLik <- NA
meta_summary$n_obs <- NA

for (i in 1:nrow(meta_summary)) {
  kname <- paste0("K_", meta_summary$K[i])
  tname <- paste0("cp_ge_", meta_summary$threshold[i])
  model <- tryCatch(factorhet_results[[kname]][[tname]], error = function(e) NULL)
  
  if (!is.null(model)) {
    meta_summary$BIC[i] <- tryCatch(BIC(model), error = function(e) NA)
    meta_summary$logLik[i] <- tryCatch(logLik(model), error = function(e) NA)
    
    # neue Lösung: Anzahl Beobachtungen aus posterior
    meta_summary$n_obs[i] <- tryCatch(length(model$posterior$membership), error = function(e) NA)
  }
}



library(purrr)

# AMEs extrahieren
all_ames <- list()

for (K in K_values) {
  for (threshold in thresholds) {
    model <- tryCatch(factorhet_results[[paste0("K_", K)]][[paste0("cp_ge_", threshold)]], error = function(e) NULL)
    
    if (!is.null(model)) {
      ame <- tryCatch(AME(model, baseline = NA)$data, error = function(e) NULL)
      
      if (!is.null(ame)) {
        ame$K <- K
        ame$threshold <- threshold
        all_ames[[paste0("K", K, "_T", threshold)]] <- ame
      }
    }
  }
}

ame_combined <- bind_rows(all_ames)


library(dplyr)
library(ggplot2)

# Filter: Nur K = 3, cp ≥ 6
ame_viz <- ame_combined %>%
  filter(K == 3, threshold == 6)

# Faktorlevels sortieren nach Effektgröße für saubere Plots
ame_viz <- ame_viz %>%
  group_by(factor) %>%
  mutate(mean_effect = mean(marginal_effect, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(factor = reorder(factor, mean_effect))

# Plot: AMEs je Faktor, gruppiert nach Subgruppe
ggplot(ame_viz, aes(x = marginal_effect, y = factor, fill = factor(group))) +
  geom_col(position = "dodge") +
  geom_errorbarh(aes(xmin = ll, xmax = ul), height = 0.2, color = "black") +
  facet_wrap(~ group, scales = "free_y") +
  labs(
    title = "Average Marginal Effects by Treatment (K = 3, cp ≥ 6)",
    subtitle = "Facetted by latent subgroup",
    x = "AME (Marginal Effect on Pr(cp ≥ 6))",
    y = "Treatment Factor",
    fill = "Subgroup"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    strip.text = element_text(face = "bold", size = 12),
    legend.position = "none"
  )

















#################################################################
#                                                               #
#                                                               # 
#               Clustering for Conjoint Experiment              #
#                                                               # 
#                                                               #
#################################################################


rm(list = ls())

merged_data <- merged_data_latent_scores

# demean individuals
# (helps getting rid of constantly high/low responding clusters)
# (Demeaned values show how much a person's responses deviate from their own average. 
# Positive values indicate that the person responded higher than their personal average on that specific question,
# while negative values mean they responded lower than their average. By using demeaning, you focus on the relative differences within each person, 
# allowing you to better understand whether they deviate from their own typical response pattern on certain questions.
# This method helps to isolate how individuals vary from their own baseline, 
# giving a clearer picture of their response tendencies in comparison to their overall average.)

demean_latent_scores <- function(latent_df, dimension_names) {
  latent_df %>%
    rowwise() %>%
    mutate(
      ind_mean = mean(c_across(all_of(dimension_names))),
      across(all_of(dimension_names),
             ~ .x - ind_mean,
             .names = "{.col}_demean")
    ) %>%
    ungroup() %>%
    dplyr::select(ends_with("_demean"))
}

perform_kmeans_clustering <- function(demeaned_df, k_values, seed_base = 1700) {
  cluster_results <- list()
  wcss <- numeric(length(k_values))
  
  for (i in seq_along(k_values)) {
    set.seed(seed_base + i * 100)
    k <- k_values[i]
    
    km_result <- kmeans(demeaned_df, centers = k, nstart = 30, iter.max = 100, algorithm = "MacQueen")
    cluster_results[[as.character(k)]] <- km_result
    
    
    wcss[i] <- km_result$betweenss / km_result$totss
    
    # Reporting
    cat("\n--- K =", k, "---\n")
    cat("Cluster Sizes:\n")
    print(km_result$size)
    cat("Variance Explained:", round(wcss[i], 3), "\n")
    cat("Cluster Centers:\n")
    print(round(km_result$centers, 2))
  }
  
  # Elbow-Plot
  plot(k_values, wcss, type = "b", xlab = "Number of Clusters (k)", ylab = "Proportion of Variance Explained")
  
  return(cluster_results)
}

# Dimension Labels for 4D
dims_4d <- c("pluralist_democracy_4d", "populist_dualism_4d", "competitive_representation_4d", "independent_expertise_4d")

Demeaned_4d <- demean_latent_scores(merged_data[, dims_4d], dims_4d)

Scaled_4d <- scale(Demeaned_4d)

k_values <- 4

# Clustering for 4D
clusters_4d <- perform_kmeans_clustering(Scaled_4d, k_values)

# Define cluster names for the four-dimensional solution (k = 3)

labels_4d <- c(
  "Representation-Focused Realist",   # Cluster 1: Low pluralism, moderate populism, high support for representation; pragmatic acceptance of democratic competition
  "Radical Sovereigntist",            # Cluster 2: Strong pluralism, low populism, moderate representation and technocracy; ideal-type democrat
  "Democratic Core Citizen"           # Cluster 3: Strong pluralism, rejection of populist dualism, technocratic openness, moderate support for representative democracy; ideal-type democratic citizen.
)

# Assign numeric cluster IDs (from k-means output) to the dataset
merged_data$cluster_4d_id <- clusters_4d[["3"]]$cluster  # 4D cluster assignment

# Map cluster IDs to descriptive names for 4D

merged_data$cluster_4d_label <- factor(
  merged_data$cluster_4d_id, 
  levels = 1:4, 
  labels = labels_4d
)


# =====================================================
# Cluster Evaluation Matrix (3D–5D, k = 3:7)
# Goal: Visualize how much variance different cluster solutions explain
# =====================================================

run_variance_matrix <- TRUE  # Can also be set to FALSE

if (run_variance_matrix) {
  
  # ---------------------------
  # Demeaning Function (reused)
  # ---------------------------
  demean_latent_scores <- function(latent_df, dimension_names) {
    latent_df %>%
      rowwise() %>%
      mutate(
        ind_mean = mean(c_across(all_of(dimension_names))),
        across(all_of(dimension_names),
               ~ .x - ind_mean,
               .names = "{.col}_demean")
      ) %>%
      ungroup() %>%
      dplyr::select(ends_with("_demean"))
  }
  
  # ---------------------------
  # Main Function: Run K-Means & Calculate Explained Variance
  # ---------------------------
  compute_explained_variance_matrix <- function(data, dimension_sets, k_range = 3:7, seed_base = 1700) {
    results_matrix <- matrix(NA, nrow = length(dimension_sets), ncol = length(k_range))
    rownames(results_matrix) <- names(dimension_sets)
    colnames(results_matrix) <- paste0("k=", k_range)
    
    for (d in seq_along(dimension_sets)) {
      dim_name <- names(dimension_sets)[d]
      dim_cols <- dimension_sets[[d]]
      
      latent_df <- data[, dim_cols]
      demeaned_df <- demean_latent_scores(latent_df, dim_cols)
      
      for (i in seq_along(k_range)) {
        k <- k_range[i]
        set.seed(seed_base + d * 100 + k)
        km_result <- kmeans(demeaned_df, centers = k, nstart = 30, iter.max = 100, algorithm = "MacQueen")
        explained <- km_result$betweenss / km_result$totss
        results_matrix[d, i] <- explained
      }
    }
    
    return(as.data.frame(results_matrix))
  }
  
  # ---------------------------
  # Define Dimension Sets
  # ---------------------------
  dimension_sets <- list(
    "3D" = c("expertise_institutionalism_3d", "people_centrism_3d", "anti_compromise_radicalism_3d"),
    "4D" = c("pluralist_democracy_4d", "populist_dualism_4d", "competitive_representation_4d", "independent_expertise_4d"),
    "5D" = c("pluralist_democracy_5d", "anti_pluralist_conflict_5d", "competitive_representation_5d", "independent_expertise_5d", "popular_sovereignty_5d")
  )
  
  # ---------------------------
  # Compute Explained Variance Matrix
  # ---------------------------
  variance_matrix <- compute_explained_variance_matrix(
    data = merged_data,
    dimension_sets = dimension_sets,
    k_range = 3:7
  )
  
  print("Explained Variance Matrix:")
  print(variance_matrix)
  
  # ---------------------------
  # Visualize as Heatmap
  # ---------------------------
  variance_long <- melt(as.matrix(variance_matrix))
  colnames(variance_long) <- c("Dimension", "k", "ExplainedVariance")
  
  ggplot(variance_long, aes(x = k, y = Dimension, fill = ExplainedVariance)) +
    geom_tile(color = "grey90", size = 0.4) +
    geom_text(aes(label = sprintf("%.2f", ExplainedVariance)), color = "black", size = 4.2) +
    scale_fill_gradientn(
      colours = c("#f7fbff", "#c6dbef", "#6baed6", "#2171b5"),
      limits = c(0, 1),
      name = "Explained Variance",
      guide = guide_colorbar(
        barwidth = 12,
        barheight = 1,
        title.position = "top",
        title.hjust = 0.5
      )
    ) +
    scale_x_discrete(position = "top") +
    labs(
      title = NULL,
      x = "Number of Clusters (k)",
      y = "Number of Dimensions"
    ) +
    theme_minimal(base_size = 13) +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_text(size = 12, face = "bold"),
      axis.text.y = element_text(size = 12, face = "bold"),
      axis.title = element_text(size = 13, face = "bold"),
      legend.position = "bottom",
      legend.title = element_text(size = 11),
      legend.text = element_text(size = 10),
      plot.margin = margin(10, 10, 10, 10)
    )
}

#--------------------------------------------
#Distance Weighting of Clusters
#--------------------------------------------

# We use these weights (IDW and Exponential Decay) to account for the varying 
# proximity of individuals to their respective cluster centers. Rather than
# treating all members of a cluster as equally representative, weighting by
# distance allows us to reflect the degree of alignment an individual has with
# their assigned cluster. This helps capture more nuanced differences within
# clusters—respondents closer to the center are given more influence, while
# those further away are weighted less, ensuring a more accurate representation
# of underlying patterns. This method minimizes error by acknowledging the
# heterogeneity within clusters.


# Function to compute distance-based weights for each respondent
calculate_distance_weights <- function(demeaned_df, cluster_result, prefix = "4d") {
  cluster_centers <- cluster_result$centers
  n_clusters <- nrow(cluster_centers)
  
  # Compute distance to assigned cluster center only
  weight_df <- demeaned_df %>%
    rowwise() %>%
    mutate(
      respondent_vec = list(c_across(everything())),
      cluster_id = cluster_result$cluster[cur_group_id()],
      assigned_center = list(cluster_centers[cluster_id, ]),
      
      # Euclidean distance to assigned cluster center
      distance_to_center = dist(rbind(respondent_vec, assigned_center)),
      
      # Inverse Distance Weighting
      !!paste0("idw_weight_", prefix) := 1 / (1 + distance_to_center),
      
      # Exponential Decay Weighting (λ = 0.5)
      !!paste0("expdecay_weight_", prefix) := exp(-0.5 * distance_to_center)
    ) %>%
    ungroup() %>%
    dplyr::select(starts_with("idw_weight_"), starts_with("expdecay_weight_"))
  
  return(weight_df)
}

# Calculate weights for 3D clusters
distance_weights_3d <- calculate_distance_weights(Demeaned_3d, clusters_3d[["3"]], prefix = "3d")

# Calculate weights for 4D clusters
distance_weights_4d <- calculate_distance_weights(Demeaned_4d, clusters_4d[["3"]], prefix = "4d")

# Calculate weights for 5D clusters
distance_weights_5d <- calculate_distance_weights(Demeaned_5d, clusters_5d[["3"]], prefix = "5d")

# Merge weights into merged_data
merged_data <- cbind(merged_data, distance_weights_3d, distance_weights_4d, distance_weights_5d)

# Save the final dataset
write.csv(merged_data, here("data", "Analysis_Data_Final.csv"), row.names = FALSE)





##################################################################
#                                                                #
# Exploratory MIRT Model (Country-Specific)                      #
#                                                                #
##################################################################

library(mirt)
library(dplyr)

# Listen und Dataframes initialisieren
all_models <- list()
fit_indices_all <- data.frame()

# Hauptschleife durch jedes Land
for (country in unique(data$ctry)) {
  cat("\n🔎 Processing country:", country, "\n")
  
  # Länderspezifische Daten filtern
  country_data <- data %>% filter(ctry == country)
  
  # Items auswählen (Achtung: ResponseId ausschließen!)
  country_items <- country_data %>% select(all_of(selected_items))
  response_matrix <- as.matrix(country_items %>% select(-ResponseId))
  
  # Prüfen der Stichprobengröße
  if (nrow(response_matrix) < 200) {
    cat("  ⚠️ Too few cases (", nrow(response_matrix), ") for stable estimation in", country, "\n")
    next
  }
  
  # Modelle des Landes sammeln
  country_models <- list()
  
  # Explorative Modelle für 1–6 Dimensionen (oder deine Wunsch-Dimensionszahl)
  for (num_factors in 1:6) {
    cat("  ✅ Fitting model with", num_factors, "dimensions...\n")
    
    tryCatch({
      # Modell fitten
      model <- mirt(
        data = response_matrix,
        model = num_factors,
        itemtype = "graded",
        exploratory = TRUE,
        method = "QMCEM",
        rotate = "oblimin"
      )
      
      # Modell speichern
      country_models[[paste0("model_dim", num_factors)]] <- model
      
      # Fit-Indizes extrahieren
      aic <- extract.mirt(model, "AIC")
      bic <- extract.mirt(model, "BIC")
      loglik <- extract.mirt(model, "logLik")
      
      # Ergebnisse direkt in Fit-Tabelle speichern
      fit_indices_all <- rbind(
        fit_indices_all,
        data.frame(
          Country = country,
          Dimensions = num_factors,
          AIC = aic,
          BIC = bic,
          LogLik = loglik
        )
      )
      
    }, error = function(e) {
      cat("  ❌ Error for", country, "with", num_factors, "dimensions:\n   ", e$message, "\n")
    })
  }
  
  # Modelle pro Land speichern
  all_models[[country]] <- country_models
  
  # Modelle in .RDS-Datei speichern
  saveRDS(all_models, "country_specific_exploratory_models.rds")
  
  # Fit-Indizes in CSV-Datei speichern
  write.csv(fit_indices_all, "fit_indices_country_specific.csv", row.names = FALSE)
}




# Prüfe, welche Modelle existieren und lade Meta-Daten
meta_summary <- expand.grid(
  K = 2:4,
  threshold = 1:9,
  stringsAsFactors = FALSE
)

meta_summary$BIC <- NA
meta_summary$logLik <- NA
meta_summary$n_obs <- NA

for (i in 1:nrow(meta_summary)) {
  kname <- paste0("K_", meta_summary$K[i])
  tname <- paste0("cp_ge_", meta_summary$threshold[i])
  model <- tryCatch(factorhet_results[[kname]][[tname]], error = function(e) NULL)
  
  if (!is.null(model)) {
    meta_summary$BIC[i] <- tryCatch(BIC(model), error = function(e) NA)
    meta_summary$logLik[i] <- tryCatch(logLik(model), error = function(e) NA)
    
    # Sicherstellen, dass Design-Objekt vorhanden ist
    if (!is.null(model$design)) {
      meta_summary$n_obs[i] <- nrow(model$design)
    } else {
      meta_summary$n_obs[i] <- NA
    }
  }
}

# Beste Modelle nach BIC filtern
meta_summary %>%
  arrange(BIC) %>%
  head(27)


fit <- factorhet_results[["K_4"]][["cp_ge_5"]]



ame_best <- AME(fit, baseline = NA)
ame_best$plot     # Direkt visualisieren
ame_best$data     # Zahlen für Tabelle






























# Factor Loadings Country-Specific
###################################################

# Manuell festgelegte optimale Dimensionen für jedes Land
optimal_dimensions <- c(
  Argentina = 5, Austria = 5, Brazil = 6, `Czech Republic` = 6, France = 5,
  Germany = 5, Greece = 6, Hungary = 5, Ireland = 5, Italy = 5,
  Poland = 5, Spain = 5, Sweden = 5, Turkey = 5, `United Kingdom` = 5, `United States` = 5
)

# Initialize a list to store results for comparison
factor_loadings_comparison <- list()

# Loop through each country
for (country in names(optimal_dimensions)) {
  cat("\n# Comparing Factor Loadings for Country:", country, "\n")
  
  # Access the models for the current country
  models <- country_specific_models[[country]]
  
  # Extract the optimal number of dimensions for the country
  num_factors_optimal <- optimal_dimensions[country]
  
  # Extract factor loadings for the optimal model
  optimal_loadings <- summary(models[[num_factors_optimal]], rotate = "oblimin")$rotF
  
  # Extract factor loadings for the 4-dimensional model
  if (length(models) >= 4) {
    four_dim_loadings <- summary(models[[4]], rotate = "oblimin")$rotF
  } else {
    four_dim_loadings <- NULL
    cat("4-dimensional model not available for", country, "\n")
  }
  
  # Store the comparison in the results list
  factor_loadings_comparison[[country]] <- list(
    optimal_dimensions = num_factors_optimal,
    optimal_loadings = optimal_loadings,
    four_dim_loadings = four_dim_loadings
  )
  
  # Display results for the current country
  cat("\nOptimal Model (", num_factors_optimal, "dimensions) Loadings:\n")
  print(optimal_loadings)
  
  if (!is.null(four_dim_loadings)) {
    cat("\n4-Dimensional Model Loadings:\n")
    print(four_dim_loadings)
  }
}



# Retrieve factor loadings for a specific country
country <- "United States"

# Optimal dimensions and loadings
optimal_loadings <- factor_loadings_comparison[[country]]$optimal_loadings
print(optimal_loadings)

# 4-dimensional model
four_dim_loadings <- factor_loadings_comparison[[country]]$four_dim_loadings
print(four_dim_loadings)






