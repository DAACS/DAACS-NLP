from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name
from pyspark.sql.functions import regexp_extract, col, concat_ws, collect_list, first
from loguru import logger
import os
from daacs.infrastructure.bootstrap import Bootstrap
from pyspark.sql import SparkSession, Window 

class Essays:
    """
    Essays works with all the essays.
    """

    def __init__(self):
        pass

    def to_parquet(self, src_dir: str, out_dir: str, spark: SparkSession) -> None :
        """
        This reads all the essays in from the essays directory, and then writes out a single parquet
        file with EssayID, essay and file_name. 
        """
        # b = Bootstrap() 
        # essays = Essays() 
        # src_dir = f"{b.DATA_DIR}/Essay_Human_Ratings/WGU-Essays/*.txt" 
        # out_dir = f"{b.DATA_DIR}/wgu_trained" 
        # essays.to_parquet(src_dir=src_dir, out_dir=out_dir, spark=spark)
        # b.rename_parquet_files(out_dir, "essay_human_ratings4")

        essays_df = spark.read.text(src_dir) \
                            .withColumn("file_name", input_file_name())

        # Define a window spec partitioned by file_name
        windowSpec = Window.partitionBy("file_name")

        # Aggregate the text content per file
        essays_aggregated = essays_df.withColumn("essay", concat_ws(" ", collect_list("value").over(windowSpec))) \
                                    .groupBy("file_name").agg(first("essay").alias("essay")) \
                                    .withColumn("EssayID", regexp_extract(col("file_name"), "Essay(\\d+).txt", 1).cast("int")) \
                                    .withColumn("file_name", regexp_extract(col("file_name"), "([^/]+)$", 1))

        essays_aggregated.repartition(1).write.mode("overwrite").parquet(out_dir)

 
        