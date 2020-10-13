import pandas as pd
import numpy as np
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
pd_temp = pd.DataFrame(np.random.random(10))
spark_temp = spark.createDataFrame(pd_temp)

# Add spark_temp to the catalog
spark_temp.createOrReplaceTempView("temp_pandas")

file_path = "C:\\Users\\mpshs\\OneDrive\\PycharmProjects\\DataCamp\\BigData\\data\\airports-data.csv"
# Read in the airports data
airports = spark.read.csv(file_path, header=True)

# dataframes are inmutable, so in order to modify it you have to create a new one
airports = airports.withColumn("test-column", airports.elevation_ft + 100)
airports.createOrReplaceTempView("temp_csv")

# Examine the tables in the catalog again
print(spark.catalog.listTables())
