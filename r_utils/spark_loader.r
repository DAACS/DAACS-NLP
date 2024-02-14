# Load SparkR library
Sys.setenv(SPARK_HOME = "/opt/homebrew/Cellar/apache-spark/3.5.0")

library(SparkR)

# Initialize Spark session
sparkR.session()
datadir <- "file:///Users/angela.lui/Documents/src/DAACS-NLP/data/wgu/"

# Set DAACS_ID
DAACS_ID <- "daacs_id"

# Load WGU ratings data
ratings_columns <- c("EssayID", "TotalScore1", "TotalScore2", "TotalScore")
wgu_ratings_raw <- read.df(paste0(datadir, "WGU_Ratings.csv"), source = "csv", header = "true") %>%
  select(ratings_columns) %>%
  withColumnRenamed("EssayID", DAACS_ID)

# Count unique EssayIDs
essay_id_counts <- count(wgu_ratings_raw, DAACS_ID)
unique_essay_ids <- filter(essay_id_counts, count == 1) %>% select(DAACS_ID)

# Filter WGU ratings for unique EssayIDs
wgu_ratings <- join(wgu_ratings_raw, unique_essay_ids, DAACS_ID)

# Load essays_human_ratings data
essays_human_rated <- read.df("file:///path/to/essay_human_ratings.parquet", source = "parquet") %>%
  withColumnRenamed("EssayID", DAACS_ID) %>%
  join(unique_essay_ids, DAACS_ID)

# Join essays and grades
essays_and_grades <- join(essays_human_rated, wgu_ratings, DAACS_ID)
