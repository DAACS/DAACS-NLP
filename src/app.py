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
# ->> We probably need more columns.

essays_human_rated = spark.read.parquet(b.file_url(WGU_File.essay_human_ratings))\
    .withColumnRenamed("EssayID", DAACS_ID)\
    .join(unique_essay_ids, [DAACS_ID])


essays_and_grades = essays_human_rated.join(wgu_ratings, [DAACS_ID])
# essays_and_grades.show(10)
# +--------+-------------+--------------------+-----------+-----------+----------+
# |daacs_id|    file_name|               essay|TotalScore1|TotalScore2|TotalScore|
# +--------+-------------+--------------------+-----------+-----------+----------+
# |     148|Essay0148.txt|\tI feel as thoug...|         21|         NA|        NA|
# |     463|Essay0463.txt|    My SRL result...|         25|         NA|        NA|
# |     471|Essay0471.txt|My self-regulated...|         21|         NA|        NA|
# |     496|Essay0496.txt|     My first tho...|         20|         NA|        NA|
# |     833|Essay0833.txt|In my SRL It stat...|         12|         NA|        NA|
# +--------+-------------+--------------------+-----------+-----------+----------+



essays_non_graded = spark.read.parquet(b.file_url(WGU_File.essays))\
    .withColumnRenamed("DAACS_ID", DAACS_ID )
essays_non_graded.show(10)
# +--------+-------+--------------------+
# |daacs_id|attempt|               essay|
# +--------+-------+--------------------+
# |   10143|    1.0|After taking the ...|
# |    2550|    1.0|\nAbout the SRL: ...|
# |    9496|    1.0|Taking the self-r...|
# |    5072|    1.0|    Going into th...|
# |    7224|    1.0|My results on the...|
# |    3306|    1.0|The self-regulate...|
# |    3539|    1.0|The DAACS self-re...|
# |    3174|    1.0|The Self-Regulate...|
# |    1979|    1.0|My self-regulated...|
# |    9569|    1.0|The results of my...|
# +--------+-------+--------------------+
