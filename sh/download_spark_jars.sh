#!/bin/bash

#This script is part of the conda setup, it downloads jars to SPARK_HOME.
#You need the s3 jars for everything you are going to run pyspark with.

#You can run it now, or you can wait until build locally runs it for you.
echo "Downloading Spark Jars!"

rel=$(dirname "$0" )
dir=$( cd "$rel" && pwd )/lib
mkdir  -p $dir

# Don't forget to remove guava10* to use pyspark
#rm  $dir/jars/guava*.jar
curl --silent https://repo1.maven.org/maven2/com/google/guava/guava/23.1-jre/guava-23.1-jre.jar                                 --output "$dir"/guava-23.1-jre.jar
curl --silent https://repo1.maven.org/maven2/org/apache/spark/spark-hadoop-cloud_2.12/3.3.0/spark-hadoop-cloud_2.12-3.3.0.jar   --output "$dir"/spark-hadoop-cloud_2.12-3.3.0.jar
curl --silent https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.1026/aws-java-sdk-bundle-1.11.1026.jar      --output "$dir"/aws-java-sdk-bundle-1.11.1026.jar
curl --silent https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.2/hadoop-aws-3.3.2.jar                            --output "$dir"/hadoop-aws-3.3.2.jar
curl --silent https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-client/3.3.2/hadoop-client-3.3.0.jar                      --output "$dir"/hadoop-client-3.3.0.jar

# Now you have to move them into place manually, from lib into pypsark. 
# It should look like this:
# mv ./lib/*.jar .venv/lib/python3.11/site-packages/pyspark/jars/. 