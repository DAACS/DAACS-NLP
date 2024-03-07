import pandas as pd
from daacs.infrastructure.bootstrap import Bootstrap
from daacs.infrastructure.wgu_file import WGU_File
from daacs.infrastructure.essays import Essays
from loguru import logger
from pyspark.sql.functions import desc, lit, udf, corr, when, lower, col

DAACS_ID="daacs_id"
b = Bootstrap()
essays_and_grades = b.get_essays_and_grades()

print(essays_and_grades.head(10))