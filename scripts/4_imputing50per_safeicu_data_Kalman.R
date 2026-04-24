#!/usr/bin/env Rscript

# ------------------------------------------------------------
# ICU Vitals – Missing Data Imputation with Kalman Filter
#
# - Reads per-patient 24hr-split CSVs from `numeric_data_path`
# - Cleans placeholder values (-, 0-patterns, Inf) to NA
# - Computes Shock Index (SI_by_abp / SI_by_nbp) when possible
# - Saves:
#     * Non-imputed data (for QC)
#     * Kalman-imputed data
#     * Date-wise median / SD / IQR summaries (before & after imputation)
#     * Per-variable stats (mean/median/sd/min/max before & after impute)
#     * Missingness summary per file
# ------------------------------------------------------------

library(gtools)    # smartbind
library(zoo)
library(dplyr)
library(stringr)
library(imputeTS)

# -----------------------------
# User configuration (edit these)
# -----------------------------
numeric_data_path   <- "Gap24"                # input: folder with per-patient folders (e.g., s1234/)
save_path_logi      <- "non_imputed"          # output: raw/logic/non-imputed copies
save_path_impute    <- "data_imputed"         # output: Kalman-imputed files
missing_status_path <- "imputation_summaries" # output: summaries, missingness, stats

dir.create(save_path_logi,      showWarnings = FALSE, recursive = TRUE)
dir.create(save_path_impute,    showWarnings = FALSE, recursive = TRUE)
dir.create(missing_status_path, showWarnings = FALSE, recursive = TRUE)

# -----------------------------
# Helper functions
# -----------------------------

# Replace bad placeholders with NA
fun_99_100 <- function(x) {
  gsub("\\b-\\b|\\b^0$\\b|\\b^0.*0$\\b|Inf", NA, trimws(x))
}

my_median <- function(x) median(x, na.rm = TRUE)
my_sd     <- function(x) sd(x, na.rm = TRUE)
my_iqr    <- function(x) IQR(x, na.rm = TRUE)

# Kalman only if <=10% missing; otherwise leave as-is
my_kalman <- function(x) {
  if ((sum(is.na(x)) / length(x)) <= 0.1) {
    x <- tryCatch(na_kalman(x), error = function(e) x)
  }
  x
}

# -----------------------------
# Containers for summaries
# -----------------------------
stats_before      <- data.frame()
stats_after       <- data.frame()
date_med_noimp    <- data.frame()
date_med_imp      <- data.frame()
date_sd_imp       <- data.frame()
date_iqr_imp      <- data.frame()
missing_summary   <- NULL

# -----------------------------
# Main loop over patient folders
# -----------------------------
folders <- list.files(numeric_data_path)

for (f in seq_along(folders)) {
  folder <- folders[f]
  message("Processing folder ", f, " / ", length(folders), ": ", folder)
  
  UHID <- gsub("^s", "", folder)
  path_in_folder      <- file.path(numeric_data_path, folder)
  file_nam            <- list.files(path_in_folder)
  new_save_path       <- file.path(save_path_impute, folder)
  new_save_path_logi  <- file.path(save_path_logi, folder)
  
  dir.create(new_save_path,      showWarnings = FALSE, recursive = TRUE)
  dir.create(new_save_path_logi, showWarnings = FALSE, recursive = TRUE)
  
  for (nam in file_nam) {
    message("  File: ", nam)
    
    num_dat <- read.csv(file.path(path_in_folder, nam), stringsAsFactors = FALSE)
    subj_id <- gsub("\\.csv$", "", nam)
    
    if ("X" %in% colnames(num_dat)) {
      num_dat$X <- NULL
    }
    
    # Branch 1: time_start present
    if ("time_start" %in% colnames(num_dat)) {
      ind_rm <- grep("time_start|PID|PICU", colnames(num_dat))
      
      # Clean string placeholders and convert to numeric
      num_dat[ , -ind_rm] <- apply(num_dat[ , -ind_rm, drop = FALSE], 2, fun_99_100)
      
      num_dat$time_start <- strptime(num_dat$time_start, "%Y-%m-%d %H:%M:%S", tz = "UTC")
      num_dat$time_start <- as.POSIXct(num_dat$time_start)
      
      time_diff <- num_dat$time_start[2:nrow(num_dat)] - num_dat$time_start[1:(nrow(num_dat) - 1)]
      
      num_dat[ , -ind_rm] <- sapply(sapply(num_dat[ , -ind_rm, drop = FALSE], as.character), as.numeric)
      
      # Shock Index calculations
      if (length(grep("HR", colnames(num_dat))) != 0 &&
          length(grep("ART\\.Sys", colnames(num_dat))) != 0) {
        num_dat$SI_by_abp <- num_dat$HR / num_dat$ART.Sys
      }
      
      if (length(grep("HR", colnames(num_dat))) != 0 &&
          length(grep("Sys", colnames(num_dat))) != 0) {
        num_dat$SI_by_nbp <- num_dat$HR / num_dat$Sys
      }
      
      # Sort and date-wise medians BEFORE imputation
      num_dat <- num_dat[order(num_dat$time_start), ]
      sum_dat <- num_dat[ , -c(2:3), drop = FALSE] %>%
        group_by(Timestamp = cut(time_start, breaks = "1 day")) %>%
        summarise_all(funs(my_median))
      
      sum_dat$UHID <- UHID
      date_med_noimp <- smartbind(date_med_noimp, data.frame(sum_dat))
      
      # Per-variable stats BEFORE imputation
      cols <- colnames(num_dat)[-ind_rm]
      for (col_nam in cols) {
        hht <- num_dat[[col_nam]]
        stats_before <- rbind(
          stats_before,
          data.frame(
            subj_id  = subj_id,
            variable = col_nam,
            mean     = mean(hht, na.rm = TRUE),
            median   = median(hht, na.rm = TRUE),
            sd       = sd(hht, na.rm = TRUE),
            max      = max(hht, na.rm = TRUE),
            min      = min(hht, na.rm = TRUE),
            stringsAsFactors = FALSE
          )
        )
      }
      
      # Temperature cleaning
      if (length(grep("T1", colnames(num_dat))) != 0 &&
          length(grep("T2", colnames(num_dat))) != 0) {
        num_dat$T1[num_dat$T1 < 30] <- NA
        num_dat$T2[num_dat$T2 < 30] <- NA
        num_dat$Td[num_dat$T2 < 30 | num_dat$T1 < 30] <- NA
        num_dat$T1 <- pmax(num_dat$T1, num_dat$T2, na.rm = TRUE)
        num_dat$T2 <- pmin(num_dat$T1, num_dat$T2, na.rm = TRUE)
      }
      
      # Save non-imputed version
      num_dat_logi <- num_dat
      write.csv(num_dat_logi, file.path(new_save_path_logi, nam), row.names = FALSE)
      
      # Imputation
      num_dat_imputed <- num_dat
      
      if (nrow(num_dat_imputed) > 30) {
        imp_cols <- grep("HR|PR|RR|T1|T2|Td|SpO2|ART\\.Sys|ART.Dia|SI_by_abp|Sys|SI_by_nbp",
                         colnames(num_dat_imputed))
        if (length(imp_cols) > 0) {
          num_dat_imputed[ , imp_cols] <- tryCatch(
            sapply(num_dat_imputed[ , imp_cols, drop = FALSE], my_kalman),
            error = function(e) {
              sapply(num_dat_imputed[ , imp_cols, drop = FALSE], my_kalman)
            }
          )
        }
      }
      
      # Date-wise summaries AFTER imputation
      num_dat_imputed <- num_dat_imputed[order(num_dat_imputed$time_start), ]
      
      sum_dat_impute <- num_dat_imputed[ , -c(2:3), drop = FALSE] %>%
        group_by(Timestamp = cut(time_start, breaks = "1 day")) %>%
        summarise_all(funs(my_median))
      sum_dat_impute$UHID <- UHID
      date_med_imp <- smartbind(date_med_imp, data.frame(sum_dat_impute))
      
      sum_dat_impute_sd <- num_dat_imputed[ , -c(2:3), drop = FALSE] %>%
        group_by(Timestamp = cut(time_start, breaks = "1 day")) %>%
        summarise_all(funs(my_sd))
      sum_dat_impute_sd$UHID <- UHID
      date_sd_imp <- smartbind(date_sd_imp, data.frame(sum_dat_impute_sd))
      
      sum_dat_impute_iqr <- num_dat_imputed[ , -c(2:3), drop = FALSE] %>%
        group_by(Timestamp = cut(time_start, breaks = "1 day")) %>%
        summarise_all(funs(my_iqr))
      sum_dat_impute_iqr$UHID <- UHID
      date_iqr_imp <- smartbind(date_iqr_imp, data.frame(sum_dat_impute_iqr))
      
      # Per-variable stats AFTER imputation
      cols_imp <- colnames(num_dat_imputed)[-ind_rm]
      for (col_nam in cols_imp) {
        hht <- num_dat_imputed[[col_nam]]
        stats_after <- rbind(
          stats_after,
          data.frame(
            subj_id  = subj_id,
            variable = col_nam,
            mean     = mean(hht, na.rm = TRUE),
            median   = median(hht, na.rm = TRUE),
            sd       = sd(hht, na.rm = TRUE),
            max      = max(hht, na.rm = TRUE),
            min      = min(hht, na.rm = TRUE),
            stringsAsFactors = FALSE
          )
        )
      }
      
    } else {
      # Branch 2: no time_start column (rare / fallback)
      ind_rm <- grep("PID|PICU", colnames(num_dat))
      
      num_dat[ , -ind_rm] <- apply(num_dat[ , -ind_rm, drop = FALSE], 2, fun_99_100)
      num_dat[ , -ind_rm] <- sapply(sapply(num_dat[ , -ind_rm, drop = FALSE], as.character), as.numeric)
      
      if (all(c("HR", "ART.Sys") %in% colnames(num_dat))) {
        num_dat$SI_by_abp <- num_dat$HR / num_dat$ART.Sys
      }
      if (all(c("HR", "Sys") %in% colnames(num_dat))) {
        num_dat$SI_by_nbp <- num_dat$HR / num_dat$Sys
      }
      
      num_dat_logi <- is.na(num_dat)
      write.csv(num_dat_logi, file.path(new_save_path_logi, nam), row.names = FALSE)
      
      num_dat <- num_dat[order(num_dat$time_start), ]
      sum_dat <- num_dat[ , -c(1:2), drop = FALSE] %>%
        group_by(Timestamp = cut(time_start, breaks = "1 day")) %>%
        summarise_all(funs(my_median))
      sum_dat$UHID <- UHID
      date_med_noimp <- smartbind(date_med_noimp, data.frame(sum_dat))
      
      cols <- colnames(num_dat)[-ind_rm]
      for (col_nam in cols) {
        hht <- num_dat[[col_nam]]
        stats_before <- rbind(
          stats_before,
          data.frame(
            subj_id  = subj_id,
            variable = col_nam,
            mean     = mean(hht, na.rm = TRUE),
            median   = median(hht, na.rm = TRUE),
            sd       = sd(hht, na.rm = TRUE),
            max      = max(hht, na.rm = TRUE),
            min      = min(hht, na.rm = TRUE),
            stringsAsFactors = FALSE
          )
        )
      }
      
      if (length(grep("T1", colnames(num_dat))) != 0 &&
          length(grep("T2", colnames(num_dat))) != 0) {
        num_dat$Td[num_dat$T2 < 30 | num_dat$T1 < 30] <- NA
        num_dat$T1[num_dat$T1 < 30] <- NA
        num_dat$T1[num_dat$T2 < 30] <- NA
        num_dat$T1 <- pmax(num_dat$T1, num_dat$T2, na.rm = TRUE)
        num_dat$T2 <- pmin(num_dat$T1, num_dat$T2, na.rm = TRUE)
      }
      
      num_dat_imputed <- num_dat
      
      if (nrow(num_dat_imputed) > 30) {
        imp_cols <- grep("HR|PR|RR|T1|T2|Td|SpO2|ART\\.Sys|ART.Dia|SI_by_abp|Sys|SI_by_nbp",
                         colnames(num_dat_imputed))
        if (length(imp_cols) > 0) {
          num_dat_imputed[ , imp_cols] <- tryCatch(
            sapply(num_dat_imputed[ , imp_cols, drop = FALSE], my_kalman),
            error = function(e) {
              sapply(num_dat_imputed[ , imp_cols, drop = FALSE], my_kalman)
            }
          )
        }
      }
      
      num_dat_imputed <- num_dat_imputed[order(num_dat_imputed$time_start), ]
      sum_dat_impute <- num_dat_imputed[ , -grep("PICU|PID", colnames(num_dat_imputed)), drop = FALSE] %>%
        group_by(Timestamp = cut(time_start, breaks = "1 day")) %>%
        summarise_all(funs(my_median))
      sum_dat_impute$UHID <- UHID
      date_med_imp <- smartbind(date_med_imp, data.frame(sum_dat_impute))
      
      cols_imp <- colnames(num_dat_imputed)[-ind_rm]
      for (col_nam in cols_imp) {
        hht <- num_dat_imputed[[col_nam]]
        stats_after <- rbind(
          stats_after,
          data.frame(
            subj_id  = subj_id,
            variable = col_nam,
            mean     = mean(hht, na.rm = TRUE),
            median   = median(hht, na.rm = TRUE),
            sd       = sd(hht, na.rm = TRUE),
            max      = max(hht, na.rm = TRUE),
            min      = min(hht, na.rm = TRUE),
            stringsAsFactors = FALSE
          )
        )
      }
    }
    
    # Save imputed file
    write.csv(num_dat_imputed, file.path(new_save_path, nam), row.names = FALSE)
    
    # Missingness summary for this file
    miss_percent <- round((apply(is.na(num_dat_imputed), 2, sum) / nrow(num_dat_imputed)) * 100, 2)
    ins_names    <- c("subj_id", "n_rows", colnames(num_dat_imputed))
    ins_values   <- c(as.character(subj_id), nrow(num_dat_imputed), miss_percent)
    row_df       <- as.data.frame(t(ins_values), stringsAsFactors = FALSE)
    colnames(row_df) <- ins_names
    
    if (is.null(missing_summary)) {
      missing_summary <- row_df
    } else {
      missing_summary <- smartbind(missing_summary, row_df)
    }
  }
}

# -----------------------------
# Save summary outputs
# -----------------------------

write.csv(missing_summary,
          file.path(missing_status_path, "missing_after_impute_summary.csv"),
          row.names = FALSE)

if (nrow(stats_before) > 0) {
  write.csv(stats_before,
            file.path(missing_status_path, "variable_stats_before_impute.csv"),
            row.names = FALSE)
}
if (nrow(stats_after) > 0) {
  write.csv(stats_after,
            file.path(missing_status_path, "variable_stats_after_impute.csv"),
            row.names = FALSE)
}

if (nrow(date_med_noimp) > 0) {
  write.csv(date_med_noimp,
            file.path(missing_status_path, "datewise_median_before_impute.csv"),
            row.names = FALSE)
}
if (nrow(date_med_imp) > 0) {
  write.csv(date_med_imp,
            file.path(missing_status_path, "datewise_median_after_impute.csv"),
            row.names = FALSE)
}
if (nrow(date_sd_imp) > 0) {
  write.csv(date_sd_imp,
            file.path(missing_status_path, "datewise_sd_after_impute.csv"),
            row.names = FALSE)
}
if (nrow(date_iqr_imp) > 0) {
  write.csv(date_iqr_imp,
            file.path(missing_status_path, "datewise_iqr_after_impute.csv"),
            row.names = FALSE)
}

save.image(file.path(missing_status_path, "workspace_missing_statistics.RData"))

message("Imputation pipeline completed.")
