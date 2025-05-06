+++
date = '2025-05-07T01:31:22+08:00'
draft = false 
title = 'PySpark Memo 3'
+++
# Chapter 2
# Content
- Filter Operation
- including: &, |, \=\=, ~

## Codes:
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('DataFrame').getOrCreate()
df_spark = spark.read().csv('test.csv', header=True, inferSchema=True)

# filter first format:
df_spark.filter('Salary<=2000')
# filter second format:
df_spark.filter(df_spark['Salary'] <= 2000)

# filter and
df_spark.filter((df_spark['Salary'] <= 2000) & (df_spark['Salary'] >= 1500))
# you can use or as |
# filter ~ is also the same
```
