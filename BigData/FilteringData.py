from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

file_path = "C:\\Users\\mpshs\\OneDrive\\PycharmProjects\\DataCamp\\BigData\\data\\airports-data.csv"
# Read in the airports data
airports = spark.read.csv(file_path, header=True)

small_airport = airports.filter(airports.type.contains("small"))
small_airport.show()

high_elevation = airports.filter("elevation_ft > 3000")
high_elevation.show()

