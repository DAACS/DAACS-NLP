import pandas as pd
from daacs.infrastructure.bootstrap import Bootstrap
from daacs.infrastructure.wgu_file import WGU_File
#from daacs.infrastructure.essays import Essays
from loguru import logger
from pyspark.sql.functions import desc, lit, udf, corr, when, lower, col

b = Bootstrap()
spark = b.get_spark() 

## These are the ones we are going to try to grade!!
# essays = spark.read.parquet(b.file_url(WGU_File.essays))
# 6,000 of these.
# essays.printSchema() 
# root
#  |-- DAACS_ID: integer (nullable = true)
#  |-- attempt: double (nullable = true)
#  |-- essay: string (nullable = true)



## These are for test and train!! We hae grades for these!! 
## Load WGU and filter out the records that are in there twice.
ratings_columns = ['EssayID', 'TotalScore1', 'TotalScore2', 'TotalScore']
wgu_ratings_raw = spark.read.option("header", True)\
    .csv(b.file_url(WGU_File.wgu_ratings))\
    .select(ratings_columns)
essay_id_counts = wgu_ratings_raw.groupBy("EssayId").count()
unique_essay_ids = essay_id_counts.filter(col("count") == 1).select("EssayId")
wgu_ratings = wgu_ratings_raw.join(unique_essay_ids, ["EssayId"])
# wgu_ratings.printSchema() 
# root
#  |-- EssayID: string (nullable = true)
#  |-- TotalScore1: string (nullable = true)
#  |-- TotalScore2: string (nullable = true)
#  |-- TotalScore: string (nullable = true)


essays_human_rated = spark.read.parquet(b.file_url(WGU_File.essay_human_ratings))\
    .distinct() \
    .join(unique_essay_ids, ["EssayId"])

# essays_human_rated.select("EssayId").distinct().count() = 1094
# with the join(unique_essay_ids) = 881

# root
#  |-- EssayID: integer (nullable = true)
#  |-- essay: string (nullable = true)
#  |-- file_name: string (nullable = true)

# : 1094 -> distinct human graded essay ids.
# In [34]: essays_human_rated.printSchema()
# root
#  |-- essay: string (nullable = true)
#  |-- file_name: string (nullable = true)
#  |-- EssayID: integer (nullable = true)


## Read in a few essays
### Split into Test/Train dataframes





## Send them up as "trained essays"

