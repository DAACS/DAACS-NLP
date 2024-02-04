import configparser
import logging
import os



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
        current_directory = os.path.dirname(os.path.abspath(__file__))
        self.PROJ_ROOT = os.path.abspath(os.path.join(current_directory, '..', '..', '..'))
        self.DATA_DIR = f"{self.PROJ_ROOT}/data"

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
