import pandas as pd
from daacs.infrastructure.bootstrap import Bootstrap
from daacs.infrastructure.wgu_file import WGU_File
from daacs.infrastructure.essays import Essays
from loguru import logger
from pyspark.sql.functions import desc, lit, udf, corr, when, lower, col

DAACS_ID="daacs_id"
b = Bootstrap()
spark = b.get_spark() 

essays_human_ratings = spark.read.parquet(b.file_url(WGU_File.essay_human_ratings))\
    .withColumnRenamed("EssayID", DAACS_ID)\
    .orderBy(DAACS_ID)

train = essays_human_ratings.filter(col(DAACS_ID) < 400)
test = essays_human_ratings.filter(col(DAACS_ID) > 400)

train_pandas = train.toPandas() 

