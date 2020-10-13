from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

file_path = "C:\\Users\\mpshs\\OneDrive\\PycharmProjects\\DataCamp\\BigData\\data\\airports-data.csv"
# Read in the airports data
airports = spark.read.csv(file_path, header=True)


# Select the first set of columns
selected1 = airports.select("type","name","local_code")

# Select the second set of columns
temp = airports.select(airports.type, airports.name, airports.local_code)

# Define first filter
filterA = airports.elevation_ft > 3000

# Define second filter
filterB = airports.iso_country == "US"

# Filter the data, first by filterA then by filterB
selected2 = temp.filter(filterA).filter(filterB)
