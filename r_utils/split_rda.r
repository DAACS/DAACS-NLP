library(readr)
library(arrow)

export_rda_to_parquet <- function(rda_file_path, output_dir) {
  # Load the R data file
  load(rda_file_path)

  # Get all objects of type 'data.frame' in the environment
  data_frames <- Filter(function(x) is.data.frame(get(x)), ls())

  # Ensure output directory exists
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # Iterate over each dataframe and write to Parquet file
  for (df_name in data_frames) {
    df <- get(df_name)
    parquet_path <- file.path(output_dir, paste0(df_name, ".parquet"))
    write_parquet(df, parquet_path)
    cat("Exported", df_name, "to", parquet_path, "\n")
  }
}

# be sure to: setwd("/Users/afraser/Documents/src/daacs-nlp") 
# make sure your wgu.rda file is in a folder called ./nogit/
rda_file_path <- "./nogit/wgu.rda" 
output_dir <- "./nogit/wdu"         # 
export_rda_to_parquet(rda_file_path, output_dir)