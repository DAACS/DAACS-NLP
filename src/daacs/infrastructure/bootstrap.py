import configparser
import logging
import os
import pandas as pd
from daacs.infrastructure.wgu_file import WGU_File

class Bootstrap:
    PROJ_ROOT = None
    

    """
    Used for loading data directly into a spark dataframe.

    Requires three things being set in .venv/bin/activate.
    export JAVA_HOME="/Users/afraser/.jdk/jdk-11.0.18+10/Contents/Home"
    export SPARK_HOME="/Users/afraser/Documents/src/pyspark-env/.venv/lib/python3.10/site-packages/pyspark"
    export PYTHONPATH="/Users/afraser/Documents/src/pyspark-env/src:/Users/afraser/Documents/src/pyspark-env/.venv/lib/python3.10/site-packages"

    Also requires sh/download_spark_jars.sh to run, with jars moved to $SPARK_HOME/jars.
 
    Use like so:
    In [1]: from daacs.infrastructure.bootstrap import Bootstrap
    In [2]: b = Bootstrap()
    In [3]: spark = b.get_spark()
    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
    24/01/26 00:35:59 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
    In [4]: df = spark.read.option("header", True).csv("s3a://tonyfraser-data/stack/merged_stack_raw.csv")

    """
    def __init__(self):
        self.DAACS_ID = "daacs_id"
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        current_directory = os.path.dirname(os.path.abspath(__file__))
        self.PROJ_ROOT = os.path.abspath(os.path.join(current_directory, '..', '..', '..'))
        self.DATA_DIR = f"{self.PROJ_ROOT}/data"
        self.MODEL_DIR = f"{self.PROJ_ROOT}/model"
        self.TENSOR_LOGS = f"{self.PROJ_ROOT}/tensor_logs"
        self.SIMPLE_DIR = f"{self.PROJ_ROOT}/simple_model"
        self.BERT_DIR = f"{self.PROJ_ROOT}/bert_model"
        self.ENCODED_DATA_DIR = f"{self.DATA_DIR}/encodings/"
        

    def file_url(self, fn: str, prefix="file:///"):
        ## Returns a file url, for spark.read.parquet(file_url) or pd.read_parquet(file_url)
        return f"{prefix}{self.DATA_DIR}/wgu/{fn}"

    def load_config(self, ini_fn: str = f"resources/{os.getenv('MODULE')}.ini") -> configparser.ConfigParser:
        ini_path = self.get_resource(ini_fn)
        config = configparser.ConfigParser()
        logging.info(f"loading config file from:{str(ini_path)}")
        config.read([str(ini_path)])
        return config

    def running_on_ecs(self) -> bool:
        if  os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"):
            return True
        else:
            return False

    def get_spark(self):
        import botocore.session
        from pyspark.sql import SparkSession
        from pyspark import SparkConf

        session = botocore.session.get_session()
        credentials = session.get_credentials()
        conf = SparkConf()
        conf.set('spark.hadoop.fs.s3a.access.key', credentials.access_key)
        conf.set('spark.hadoop.fs.s3a.secret.key', credentials.secret_key)
        conf.set("spark.hadoop.fs.s3a.impl","org.apache.hadoop.fs.s3a.S3AFileSystem")
        conf.set("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        conf.set('spark.driver.memory', '4g')
        conf.set('spark.executor.memory', '4g')

        if self.running_on_ecs():
            #overwrite provider to use session token, doesn't seem to work as a array, event though it is supposed to.
            conf.set("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider")
            conf.set("spark.hadoop.fs.s3a.session.token", credentials.token)
        spark = (
            SparkSession
                .builder
                .config(conf=conf)
                .appName(f"my-executor")
                .getOrCreate()
        )
        self.spark = spark
        self.spark.sparkContext.setLogLevel("ERROR")
        return self.spark

    def kill_spark(self):
        """
        <i>If necessary, used to close spark after job completion.</i>
        """
        self.spark.stop()

    def set_log_level(self, level="warning"):
        """
        A shortcut to help ipython or jupyter users using bootstrap to adjust logging between code blocks.
        Use this to make sure you're updating all the loggers.
        ::
            bootstrap.set_log_level('debug')
        """
        if level == "warning":
            logging.getLogger().setLevel(logging.WARNING)
        if level == "error":
            logging.getLogger().setLevel(logging.ERROR)
        if level == "info":
            logging.getLogger().setLevel(logging.INFO)
        if level == "debug":
            logging.getLogger().setLevel(logging.DEBUG)

    def rename_parquet_files(self, directory: str, new_name: str):
        ## /Users/afraser/Documents/src/daacs-nlp/data/wgu_trained
        ##         contains a file like part-00000 ... .snappy.parquet
        ## you want it to become essays.parquet
    
        ## Bootstrap().rename_parquet_files(out_dir, "essays")
        ## 
        ## Then, move it to data/wgu
        for filename in os.listdir(directory):
            if filename.endswith(".parquet"):
                # Construct the full file path
                old_file = os.path.join(directory, filename)
                # Define the new file name; you might want to add logic to make this unique
                new_file = os.path.join(directory, new_name + ".parquet")
                # Rename the file
                os.rename(old_file, new_file)
                print(f"Renamed {old_file} to {new_file}")
   
    def get_essays_and_grades(self) -> pd.DataFrame :
  #  def get_essays_and_grades(self, ratings_columns = ['EssayID', 'TotalScore1', 'TotalScore2', 'TotalScore']) -> pd.DataFrame :

        # Read the ratings CSV
        wgu_ratings_raw = pd.read_csv(self.file_url(WGU_File.wgu_ratings))
        wgu_ratings_raw = wgu_ratings_raw.rename(columns={"EssayID": self.DAACS_ID})

        # Filter unique essays
        essay_id_counts = wgu_ratings_raw[self.DAACS_ID].value_counts()
        unique_essay_ids = essay_id_counts[essay_id_counts == 1].index
        wgu_ratings = wgu_ratings_raw[wgu_ratings_raw[self.DAACS_ID].isin(unique_essay_ids)]

        # Read the essays human rated Parquet file
        essays_human_rated = pd.read_parquet(self.file_url(WGU_File.essay_human_ratings))
        essays_human_rated = essays_human_rated.rename(columns={"EssayID": self.DAACS_ID})
        essays_human_rated = essays_human_rated[essays_human_rated[self.DAACS_ID].isin(unique_essay_ids)]

        # Merge the datasets
        essays_and_grades = pd.merge(essays_human_rated, wgu_ratings, on=self.DAACS_ID)
        essays_and_grades.set_index(self.DAACS_ID, inplace=True)
        essays_and_grades.index = essays_and_grades.index.astype(int)
        
        return essays_and_grades
    

    def get_essays_and_grades_spark(self):

        spark = self.get_spark() 

        ## These are for test and train!! We hae grades for these!! 
        ## Load WGU and filter out the records that are in there twice.

        ## use average of two raters? 
        ## use the ones where the agree.

        ratings_columns = ['EssayID', 'TotalScore1', 'TotalScore2', 'TotalScore']
        wgu_ratings_raw = spark.read.option("header", True)\
            .csv(self.file_url(WGU_File.wgu_ratings))\
            .select(ratings_columns)\
            .withColumnRenamed("EssayID", self.DAACS_ID)
        essay_id_counts = wgu_ratings_raw.groupBy(self.DAACS_ID).count()
        unique_essay_ids = essay_id_counts.filter(col("count") == 1).select(self.DAACS_ID)
        wgu_ratings = wgu_ratings_raw.join(unique_essay_ids, [self.DAACS_ID])

        # wgu_ratings.printSchema() D
        # root
        #  |-- daacs_id: string (nullable = true)
        #  |-- TotalScore1: string (nullable = true)
        #  |-- TotalScore2: string (nullable = true)
        #  |-- TotalScore: string (nullable = true)


        essays_human_rated = spark.read.parquet(self.file_url(WGU_File.essay_human_ratings))\
            .withColumnRenamed("EssayID", self.DAACS_ID)\
            .join(unique_essay_ids, [self.DAACS_ID])

        essays_and_grades = essays_human_rated.join(wgu_ratings, [self.DAACS_ID])
        return essays_and_grades