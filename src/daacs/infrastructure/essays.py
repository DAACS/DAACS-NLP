from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name
from pyspark.sql.functions import regexp_extract, col


class Essays:

    def __init__(self):
        pass

    def to_parquet(self, src_dir: str, out_dir: str, spark: SparkSession) -> None :
        essays_df = spark.read.text(src_dir) \
                    .withColumn("file_name", input_file_name())\
                    .withColumn("EssayID", regexp_extract(col("file_name"), "Essay(\\d+).txt", 1).cast("int"))\
                    .withColumnRenamed("value", "essay")\
                    .withColumn("file_name", regexp_extract("file_name", "([^/]+)$", 1))
        print(f"{out_dir} is out dir" )
        