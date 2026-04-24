#!/usr/bin/env Rscript

# ------------------------------------------------------------
# ICU Vitals: Split by 24-hour gaps and generate 15s-resolution time series
#
# - Input: Per-patient folders under `main_path`, each containing CSVs
# - Merges all files for each patient
# - Cleans placeholder values (-99, -100, 0, Inf)
# - Splits data into chunks separated by >= 24 hours
# - Fills gaps > 30 seconds with 15-second timestamps
# - Cleans T1/T2/Td temperature fields
# - Saves:
#   * Per-chunk files into `save_path_matched/<UHID>_k.csv`
#   * Date-wise median summaries into one CSV
# ------------------------------------------------------------

library(gtools)
library(stringr)
library(zoo)
library(dplyr)

# -----------------------------
# User configuration
# -----------------------------
main_path         <- "Matched"    # folder with per-patient subfolders: s<UHID>/
save_path_matched <- "Gap24"      # folder to save processed 15s-resolution chunks
summary_out_path  <- "24hr_gap_date_char_wto_impute_tcorr.csv"

dir.create(save_path_matched, showWarnings = FALSE, recursive = TRUE)

# -----------------------------
# Helper functions
# -----------------------------
my_median <- function(x) median(x, na.rm = TRUE)

fun_99_100 <- function(x) {
  gsub(
    "\\-99|\\-100|\\b-\\b|\\b^0$\\b|\\b^0.*0$\\b|Inf",
    NA,
    trimws(x)
  )
}

# -----------------------------
# Main processing
# -----------------------------

# patient folders of form "s<digits>"
folders <- list.files(main_path, pattern = "^s\\d+")
summary_list <- list()

for (f in seq_along(folders)) {
  folder <- folders[f]
  message("Processing folder ", f, " / ", length(folders), ": ", folder)
  
  UHID <- gsub("^s\\-?", "", folder)
  path_in_folder <- file.path(main_path, folder)
  file_nam <- list.files(path_in_folder)
  
  new_save_path <- file.path(save_path_matched, folder)
  dir.create(new_save_path, showWarnings = FALSE, recursive = TRUE)
  
  # -------------------------
  # Merge all files for this patient
  # -------------------------
  temp <- read.csv(file.path(path_in_folder, file_nam[1]), stringsAsFactors = FALSE)
  if ("X" %in% colnames(temp)) {
    temp$X <- NULL
  }
  temp[ , -1] <- apply(temp[ , -1, drop = FALSE], 2, fun_99_100)
  
  if (length(file_nam) > 1) {
    for (i in 2:length(file_nam)) {
      dat2 <- read.csv(file.path(path_in_folder, file_nam[i]), stringsAsFactors = FALSE)
      if ("X" %in% colnames(dat2)) {
        dat2$X <- NULL
      }
      dat2[ , -1] <- apply(dat2[ , -1, drop = FALSE], 2, fun_99_100)
      temp <- smartbind(temp, dat2)
    }
  }
  
  temp$time_start <- strptime(temp$time_start, "%Y-%m-%d %H:%M:%S", tz = "IST")
  temp$time_start <- as.POSIXct(temp$time_start)
  time_diff <- temp$time_start[2:nrow(temp)] - temp$time_start[1:(nrow(temp) - 1)]
  
  # -------------------------
  # Case 1: One or more 24h gaps -> split into chunks
  # -------------------------
  if (length(which(time_diff >= 24 * 60 * 60)) != 0) {
    
    inds <- which(time_diff >= 24 * 60 * 60)
    inds_all <- c(1, inds, nrow(temp))
    
    for (k in 1:(length(inds_all) - 1)) {
      dat <- temp[(inds_all[k] + 1):(inds_all[k + 1]), ]
      
      if ("X" %in% colnames(dat)) {
        dat$X <- NULL
      }
      
      dat$time_start <- as.POSIXct(dat$time_start)
      
      # remove duplicate timestamps
      if (any(duplicated(dat$time_start))) {
        dat <- dat[!duplicated(dat$time_start), ]
      }
      
      # convert vitals columns to numeric
      if (ncol(dat) > 2) {
        dat[ , -c(1:2)] <- sapply(sapply(dat[ , -c(1:2), drop = FALSE], as.character), as.numeric)
      }
      
      if ("PICU" %in% colnames(dat)) {
        dat$PICU <- as.integer(dat$PICU)
      }
      
      # fill gaps > 30 seconds with 15-second grid
      time_diff <- dat$time_start[2:nrow(dat)] - dat$time_start[1:(nrow(dat) - 1)]
      
      while (length(which(time_diff >= 30)) != 0) {
        ind_to_process <- which(time_diff >= 30)
        ind <- ind_to_process[1]
        
        ts <- seq.POSIXt(dat$time_start[ind], dat$time_start[ind + 1], by = "15 sec")
        df <- data.frame(time_start = ts)
        dat <- full_join(df, dat, by = "time_start")
        dat <- dat[order(dat$time_start), ]
        time_diff <- dat$time_start[2:nrow(dat)] - dat$time_start[1:(nrow(dat) - 1)]
      }
      
      # temperature cleaning if columns exist
      if (length(grep("T1", colnames(dat))) != 0 &&
          length(grep("T2", colnames(dat))) != 0) {
        dat$Td[which(dat$T2 < 30 | dat$T1 < 30)] <- NA
        dat$T1[which(dat$T1 < 30)] <- NA
        dat$T2[which(dat$T2 < 30)] <- NA
        dat$T1 <- pmax(dat$T1, dat$T2, na.rm = TRUE)
        dat$T2 <- pmin(dat$T1, dat$T2, na.rm = TRUE)
      }
      
      # save chunk
      out_file <- file.path(new_save_path, paste0(UHID, "_", k, ".csv"))
      write.csv(dat, out_file, row.names = FALSE)
      
      # date-wise summary
      dat <- dat[order(dat$time_start), ]
      sum_dat <- dat[ , -2, drop = FALSE] %>%
        group_by(Timestamp = cut(time_start, breaks = "1 day")) %>%
        summarise_all(my_median)
      
      sum_dat$UHID <- UHID
      summary_list[[length(summary_list) + 1]] <- data.frame(sum_dat)
    }
    
  } else {
    # -------------------------
    # Case 2: No 24h gap, single continuous segment
    # -------------------------
    
    if ("X" %in% colnames(temp)) {
      temp$X <- NULL
    }
    
    # remove duplicate timestamps if any
    if (any(duplicated(temp$time_start))) {
      temp <- temp[!duplicated(temp$time_start), ]
    }
    
    temp$time_start <- as.POSIXct(temp$time_start)
    
    if (ncol(temp) > 2) {
      temp[ , -c(1:2)] <- sapply(sapply(temp[ , -c(1:2), drop = FALSE], as.character), as.numeric)
    }
    
    time_diff <- temp$time_start[2:nrow(temp)] - temp$time_start[1:(nrow(temp) - 1)]
    
    # fill gaps > 30 seconds with 15-second grid
    while (length(which(time_diff >= 30)) != 0) {
      ind_to_process <- which(time_diff >= 30)
      ind <- ind_to_process[1]
      
      ts <- seq.POSIXt(temp$time_start[ind], temp$time_start[ind + 1], by = "15 sec")
      df <- data.frame(time_start = ts)
      temp <- full_join(df, temp, by = "time_start")
      temp <- temp[order(temp$time_start), ]
      time_diff <- temp$time_start[2:nrow(temp)] - temp$time_start[1:(nrow(temp) - 1)]
    }
    
    temp$PID <- UHID
    
    if (length(grep("T1", colnames(temp))) != 0 &&
        length(grep("T2", colnames(temp))) != 0) {
      temp$Td[which(temp$T2 < 30 | temp$T1 < 30)] <- NA
      temp$T1[which(temp$T1 < 30)] <- NA
      temp$T2[which(temp$T2 < 30)] <- NA
      temp$T1 <- pmax(temp$T1, temp$T2, na.rm = TRUE)
      temp$T2 <- pmin(temp$T1, temp$T2, na.rm = TRUE)
    }
    
    # date-wise summary
    sum_dat <- temp[ , -2, drop = FALSE] %>%
      group_by(Timestamp = cut(time_start, breaks = "1 day")) %>%
      summarise_all(my_median)
    
    sum_dat$UHID <- UHID
    summary_list[[length(summary_list) + 1]] <- data.frame(sum_dat)
    
    # save single continuous file
    out_file <- file.path(new_save_path, paste0(UHID, ".csv"))
    write.csv(temp, out_file, row.names = FALSE)
  }
}

# -----------------------------
# Write combined date-wise summary
# -----------------------------
if (length(summary_list) > 0) {
  temp_date_char <- smartbind(summary_list)
  write.csv(temp_date_char, summary_out_path, row.names = FALSE)
  message("Summary written to: ", summary_out_path)
} else {
  message("No data summarised; summary file not written.")
}
