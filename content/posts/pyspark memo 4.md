+++
date = '2025-05-07T16:55:33+08:00'
draft = false 
title = 'PySpark Memo 4'
+++
# Chapter 2
# Content
- Aggregate And GroupBy

## Codes:
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('DataFrame').getOrCreate()
df_spark = spark.read().csv('test.csv', header=True, inferSchema=True)

# You can do groupby and then aggregate.
# You get 2 different way to do aggreate.

# where the aggregator, i.e. sum gives you a dataframe.
df_spark.groupBy('Name').sum()

df_spark.groupBy('Name').mean()
df_spark.groupBy('Name').avg()
df_spark.groupBy('Name').count()
...

# Another way to do aggregate:
df_spark.agg({'Salary':'sum'})
```
