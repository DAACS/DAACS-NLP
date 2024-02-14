import pandas as pd
from daacs.infrastructure.bootstrap import Bootstrap
from daacs.infrastructure.wgu_file import WGU_File
from daacs.infrastructure.essays import Essays
from loguru import logger
from pyspark.sql.functions import desc, lit, udf, corr, when, lower, col

DAACS_ID="daacs_id"
b = Bootstrap()
spark = b.get_spark() 

## These are for test and train!! We hae grades for these!! 
## Load WGU and filter out the records that are in there twice.

## use average of two raters? 
## use the ones where the agree.

ratings_columns = ['EssayID', 'TotalScore1', 'TotalScore2', 'TotalScore']
wgu_ratings_raw = spark.read.option("header", True)\
    .csv(b.file_url(WGU_File.wgu_ratings))\
    .select(ratings_columns)\
    .withColumnRenamed("EssayID", DAACS_ID)
essay_id_counts = wgu_ratings_raw.groupBy(DAACS_ID).count()
unique_essay_ids = essay_id_counts.filter(col("count") == 1).select(DAACS_ID)
wgu_ratings = wgu_ratings_raw.join(unique_essay_ids, [DAACS_ID])

# wgu_ratings.printSchema() 
# root
#  |-- daacs_id: string (nullable = true)
#  |-- TotalScore1: string (nullable = true)
#  |-- TotalScore2: string (nullable = true)
#  |-- TotalScore: string (nullable = true)


essays_human_rated = spark.read.parquet(b.file_url(WGU_File.essay_human_ratings))\
    .withColumnRenamed("EssayID", DAACS_ID)\
    .join(unique_essay_ids, [DAACS_ID])


essays_and_grades = essays_human_rated.join(wgu_ratings, [DAACS_ID])