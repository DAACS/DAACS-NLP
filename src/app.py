import pandas as pd
from daacs.infrastructure.bootstrap import Bootstrap
from daacs.infrastructure.wgu_file import WGU_File
from daacs.infrastructure.essays import Essays
from loguru import logger

b = Bootstrap()
spark = b.get_spark() 


essays = Essays() 
print(b.DATA_DIR)
src_dir = f"{b.DATA_DIR}/Essay_Human_Ratings/WGU-Essays/*.txt" 
out_dir = f"{b.DATA_DIR}/wgu_trained" 

print(src_dir)
print(out_dir)
essays.to_parquet(src_dir=src_dir, out_dir=out_dir, spark=spark)


## Read in a few essays
### Split into Test/Train dataframes





## Send them up as "trained essays"

