#!/usr/bin/env Rscript

# ------------------------------------------------------------
# HL7-like ICU Log Cleaner
#
# - Reads raw unsolicited log files from a folder
# - Extracts PID / PV1 / OBX segments between "------" markers
# - Builds a wide table per time window and PID
# - Cleans OBX, PID, PV1 fields
# - Writes cleaned CSVs to an output folder
#
# Dependencies: qdapRegex
# ------------------------------------------------------------

library(qdapRegex)

# -----------------------------
# User configuration
# -----------------------------
data_dir    <- "path/to/raw_logs"          # folder with raw text log files
output_dir  <- "path/to/cleaned_outputs"   # folder where cleaned_*.csv will be written

# ensure output directory exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# -----------------------------
# Helper functions
# -----------------------------

clean_obx <- function(x) {
  gsub(
    "^OBX\\|\\|NM\\|\\b\\d{3}\\b\\^.*\\|\\d{4}\\||" %+%
      "\\|\\|\\|\\|\\|\\|F$|" %+%
      "\\|\\|\\|\\|\\|\\|F\\|\\|APERIODIC\\|.*|" %+%
      "^OBX\\|\\|NM\\|52\\^Height\\|\\||" %+%
      "^OBX\\|\\|NM\\|51\\^Weight\\|\\|",
    "",
    x
  )
}

clean_PID <- function(x) {
  x <- gsub("PID\\|\\|\\|\\|\\||\\|\\|\\|\\^\\^\\^\\^\\|\\|\\|", "", x)
  x <- gsub("\\^\\|\\|\\|", "_", x)
  x
}

clean_PV1 <- function(x) {
  gsub("^PV1\\|\\|I\\|\\^\\^|\\|\\|\\|\\|\\|.*\\|(P|N)\\|.*", "", x)
}

`%+%` <- function(a, b) paste0(a, b)

# -----------------------------
# File lists and selection
# -----------------------------

data_files <- list.files(data_dir, full.names = FALSE)
# exclude partial files if needed
data_files <- data_files[!grepl("filepart", data_files)]

cleaned_files <- list.files(output_dir, full.names = FALSE)
cleaned_files <- cleaned_files[!grepl("filepart", cleaned_files)]

# remove "cleaned_" prefix and ".csv" suffix to match raw filenames
cleaned_base <- gsub("^cleaned_|\\.csv$", "", cleaned_files)

files_to_process <- setdiff(data_files, cleaned_base)

cat("Found", length(data_files), "raw files.\n")
cat("Already cleaned:", length(cleaned_base), "\n")
cat("To process:", length(files_to_process), "\n\n")

# -----------------------------
# Main processing loop
# -----------------------------

for (z in seq_along(files_to_process)) {
  fname <- files_to_process[z]
  message("Processing file: ", fname)
  
  # read raw lines
  file_path <- file.path(data_dir, fname)
  res <- readLines(file_path, warn = FALSE)
  
  # keep only relevant lines: separators, PV1, PID, OBX numeric codes
  res_filtered <- res[grep(
    "------.*|^PV1|^PID|^OBX\\|\\|NM\\|\\b\\d{3}\\b\\^.*F|^OBX\\|\\|NM\\|\\b\\d{2}\\b\\^.*F$",
    res
  )]
  
  # build list of channel names from OBX / PV1 etc.
  res_names <- gsub(
    "^OBX\\|\\|NM\\|\\b\\d{3}\\b\\^|" %+%
      "\\|\\d{4}.*|" %+%
      "^OBX\\|\\|NM\\|\\d{2}\\^|" %+%
      "\\|\\|\\d*\\.\\d*\\|.*F$|" %+%
      "\\|\\|I\\|\\^\\^PICU|" %+%
      "&\\d?&.*&?&?.*|" %+%
      "\\|\\|I\\|\\^\\^|" %+%
      "\\|\\|I\\|?",
    "",
    res_filtered
  )
  
  # remove PID and separator lines from names
  res_names <- res_names[!grepl("PID.*|-----.*", res_names)]
  res_names_unique <- unique(res_names)
  
  # define columns: time_start, PID, all channel names, time_end
  len_col <- length(res_names_unique) + 3
  col_names <- c("time_start", "PID", res_names_unique, "time_end")
  
  # positions of time separators
  time_stamp_idx <- grep("----.*", res_filtered)
  
  # container for each PID block
  logged_data <- list()
  row_counter <- 1
  
  # iterate between consecutive time separators
  if (length(time_stamp_idx) > 1) {
    for (i in seq_len(length(time_stamp_idx) - 1)) {
      segment <- res_filtered[time_stamp_idx[i]:time_stamp_idx[i + 1]]
      
      # find PID lines within this segment
      pid_idx <- grep(".*PID.*", segment)
      
      # current logic: only segments with exactly one PID
      if (length(pid_idx) == 1) {
        # create a one-row data.frame for this PID/time window
        temp_df <- data.frame(
          matrix(-99, ncol = len_col, nrow = 1),
          stringsAsFactors = FALSE
        )
        colnames(temp_df) <- col_names
        
        # fill time_start, PID, time_end
        temp_df$time_start <- segment[1]
        temp_df$PID        <- segment[2]
        temp_df$time_end   <- segment[length(segment)]
        
        # fill OBX / channel fields
        for (n in seq_along(res_names_unique)) {
          channel_name <- res_names_unique[n]
          col_idx <- n + 2  # offset for time_start, PID
          
          # patterns that may contain this channel
          pattern1 <- paste(".*\\^", channel_name, "\\|.*", sep = "")
          pattern2 <- paste(".*\\^", channel_name, "&\\d&.*", sep = "")
          pattern3 <- paste("^", channel_name, sep = "")
          
          # try OBX pattern with ^channel_name|
          matches1 <- segment[grep(pattern1, segment)]
          if (length(matches1) > 0) {
            temp_df[1, col_idx] <- matches1[1]
          }
          
          # try OBX pattern with ^channel_name&digit&
          matches2 <- segment[grep(pattern2, segment)]
          if (length(matches2) > 0) {
            temp_df[1, col_idx] <- matches2[1]
          }
          
          # try lines starting directly with the channel name
          matches3 <- segment[grep(pattern3, segment)]
          if (length(matches3) == 1) {
            temp_df[1, col_idx] <- matches3[1]
          }
        }
        
        logged_data[[row_counter]] <- temp_df
        row_counter <- row_counter + 1
      }
    }
  }
  
  # nothing parsed -> skip file
  if (length(logged_data) == 0) {
    warning("No PID segments extracted for file: ", fname)
    next
  }
  
  final <- do.call(rbind, logged_data)
  
  # -----------------------------
  # Cleaning phase
  # -----------------------------
  
  final_copy <- final
  
  # clean OBX and PV1 columns
  final_copy <- as.data.frame(lapply(final_copy, clean_obx), stringsAsFactors = FALSE)
  final_copy <- as.data.frame(lapply(final_copy, clean_PV1), stringsAsFactors = FALSE)
  
  # clean PID
  final_copy$PID <- clean_PID(final_copy$PID)
  
  # extract PID between "|||" and "||"
  pid_chunks <- rm_between(final_copy$PID, "|||", "||", extract = TRUE)
  final_copy$PID <- vapply(
    pid_chunks,
    function(x) if (length(x) > 0) x[1] else NA_character_,
    FUN.VALUE = character(1)
  )
  
  # extract PV1 location between "PICU&" and "&32"
  if ("PV1" %in% colnames(final_copy)) {
    pv1_chunks <- rm_between(final_copy$PV1, "PICU&", "&32", extract = TRUE)
    final_copy$PV1 <- vapply(
      pv1_chunks,
      function(x) if (length(x) > 0) x[1] else NA_character_,
      FUN.VALUE = character(1)
    )
  }
  
  # parse time_start into date and time
  final_copy$time_start <- gsub("--------- | ----------", "", final_copy$time_start)
  time_split <- strsplit(final_copy$time_start, "\\s{2,}")
  
  # ensure we have exactly 2 parts per row; otherwise set NA
  datestamp <- sapply(time_split, function(x) if (length(x) >= 1) x[1] else NA_character_)
  timestamp <- sapply(time_split, function(x) if (length(x) >= 2) x[2] else NA_character_)
  
  final_copy$datestamp <- format(as.Date(datestamp), "%d-%m-%y")
  final_copy$timestamp <- timestamp
  
  # -----------------------------
  # Write cleaned file
  # -----------------------------
  out_path <- file.path(output_dir, paste0("cleaned_", fname, ".csv"))
  write.csv(final_copy, out_path, row.names = FALSE)
  message("Written: ", out_path)
}

cat("\nProcessing complete.\n")
