+++
date = '2025-05-06T21:17:29+08:00'
draft = false 
title = 'pyspark memo'
+++
# Chapter 1
# Content
- PySpark Datafram
- Reading The Dataset
- Checking the Datatype of the Column(Schema)
- Selecting Columns And indexing
- Check Describe option similar to Pandas
- Adding Columns
- Droping Columns
- Renaming Columns

## Codes:
```python
from pyspark.sql import SparkSession

# To use spark dataframe, we need session first
spark = SparkSession.builder.appName('DataFrame').getOrCreate()
# This read will not infer header and the types
spark.read().csv('test.csv')

#or you can use read option(k, v)
df_spark = spark.read().option('header', 'true').csv('test.csv')
#And if you want to use something like auto casting, you can add inferSchema
df_spark = spark.read().option('header', 'true').csv('test.csv', inferSchema=True)

# check your schema
df_spark.printSchema()
>>> root
 |-- Name: string (nullable = true)
 |-- age: integer (nullable = true)
 |-- Experience: integer (nullable = true)

#You can do the read this way, too.
df_spark = spark.read.csv('test.csv', header=True, inferSchema=True)
df_spark.show()
>>>
+---------+---+----------+
|     Name|age|Experience|
+---------+---+----------+
|    Krish| 31|        10|
|Sudhanshu| 30|         8|
|    Sunny| 29|         4|
+---------+---+----------+

df_spark.head(3)
df_spark.show()

# if you want to select some columns
df_spark.select('Name').show()
df_spark.select(['Name', 'Experience']).show()
#The select will return you dataframe by the way.

# If you want just a column, you can do this.
df_spark['Name']
>>> Column<'Name'>

# and like in pandas, there are 'describe', which can help you to print the statistics of your data. by the way, describe returns you a dataframe. 
df_spark.describe().show()

# So how can you replace, adding or *renaming columns?
# This is adding new columns. And it is not in-place.
df_spark = df_spark.withColumn('Experience After 2 year', df_spark['Experience'] + 2)
# You can drop, not in-place, too.
df_spark = df_spark.drop('Experinece After 2 year')
# You can rename
df_spark = df_spark.withColumnRenamed('Name', 'New Name')
```
