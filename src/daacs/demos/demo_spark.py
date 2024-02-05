import pandas as pd
from daacs.infrastructure.bootstrap import Bootstrap
from daacs.infrastructure.wgu_file import WGU_File

b = Bootstrap()
spark = b.get_spark() 

essays = spark.read\
    .parquet(b.file_url(WGU_File.essays))\
    .filter("attempt != 1")

essays.show(10, True)

