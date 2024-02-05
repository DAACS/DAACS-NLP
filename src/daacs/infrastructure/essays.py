from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name
from pyspark.sql.functions import regexp_extract, col
from loguru import logger
import os
from daacs.infrastructure.bootstrap import Bootstrap


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
        # b = bootsrap ...  
        # essays = Essays() 
        # src_dir = f"{b.DATA_DIR}/Essay_Human_Ratings/WGU-Essays/*.txt" 
        # out_dir = f"{b.DATA_DIR}/wgu_trained" 
        # # essays.to_parquet(src_dir=src_dir, out_dir=out_dir, spark=spark)
        # b.rename_parquet_files(out_dir, "essay_human_ratings")

        ## Now go move that essays_human_ratings.parquet over to DATA_DIR/wgu
        ## and delete teh wgu_trained directory if you want to.

        logger.info(f"Reading essays from: {src_dir}")
        logger.info(f"Writing parquet file to: {out_dir}")
        essays_df = spark.read.text(src_dir) \
                    .withColumn("file_name", input_file_name())\
                    .withColumn("EssayID", regexp_extract(col("file_name"), "Essay(\\d+).txt", 1).cast("int"))\
                    .withColumnRenamed("value", "essay")\
                    .withColumn("file_name", regexp_extract("file_name", "([^/]+)$", 1))
        essays_df.repartition(1).write.mode("overwrite").parquet(out_dir)
 
        