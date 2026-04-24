#!/usr/bin/env Rscript

main_path          <- "Cleaned_Unsolicited"   # folder with cleaned input CSVs
save_path_matched  <- "Matched"              # base folder for PID-matched outputs
save_path_unmatched <- "Unmatched"           # base folder for unmatched rows
error_log_path     <- "error_log.csv"        # CSV file for errors

dir.create(save_path_matched,  showWarnings = FALSE, recursive = TRUE)
dir.create(save_path_unmatched, showWarnings = FALSE, recursive = TRUE)

file_nam <- list.files(main_path)

library(stringr)

error_list <- data.frame(
  File_Index    = integer(),
  File_Name     = character(),
  Error_Message = character(),
  stringsAsFactors = FALSE
)

for (i in seq_along(file_nam)) {
  tryCatch({
    input_file <- file.path(main_path, file_nam[i])
    dat <- read.csv(input_file, stringsAsFactors = FALSE)
    
    nam <- gsub("\\.csv$", "", file_nam[i])
    my_date_str <- do.call(rbind, str_split(nam, ",", n = 2))[, 2]
    my_date <- as.Date(trimws(my_date_str), format = "%B %d, %Y")
    
    dat2 <- dat[, grep(
      "time_start|PID|PV1$|\\bPICU$\\b|\\bHR\\b|\\bRR\\b|PR|\\bSpO2$\\b|T1|T2|Td|Dia|Sys|Mean",
      colnames(dat),
      ignore.case = TRUE
    )]
    
    dat2$PID <- as.character(dat2$PID)
    
    if ("PV1" %in% colnames(dat2)) {
      colnames(dat2)[colnames(dat2) == "PV1"] <- "PICU"
    }
    
    pids <- unique(dat2$PID)
    
    if (!all(is.na(pids))) {
      pids <- pids[!is.na(pids)]
      
      for (pid in pids) {
        dat_pids <- dat2[dat2$PID == pid, ]
        new_save_path <- file.path(save_path_matched, paste0("s", pid))
        dir.create(new_save_path, showWarnings = FALSE, recursive = TRUE)
        
        output_file <- file.path(new_save_path, paste0(pid, "_", my_date, ".csv"))
        message("Saving: ", output_file)
        write.csv(dat_pids, output_file, row.names = FALSE)
      }
      
      dat_remained <- dat2[is.na(dat2$PID), ]
      output_unmatched <- file.path(save_path_unmatched, paste0("remained_", file_nam[i]))
      message("Saving unmatched data: ", output_unmatched)
      write.csv(dat_remained, output_unmatched, row.names = FALSE)
      
    } else {
      output_unmatched <- file.path(save_path_unmatched, file_nam[i])
      message("Saving unmatched file: ", output_unmatched)
      write.csv(dat2, output_unmatched, row.names = FALSE)
    }
    
  }, error = function(e) {
    message("Error in processing file: ", file_nam[i], " | Error: ", e$message)
    error_list <<- rbind(
      error_list,
      data.frame(
        File_Index    = i,
        File_Name     = file_nam[i],
        Error_Message = e$message,
        stringsAsFactors = FALSE
      )
    )
  })
}

if (nrow(error_list) > 0) {
  message("Saving error log to ", error_log_path)
  write.csv(error_list, error_log_path, row.names = FALSE)
} else {
  message("No errors encountered.")
}
