+++
date = '2025-05-07T01:16:44+08:00'
draft = false 
title = 'PySpark Memo 2'
+++
# Chapter 2
# Content
- Dropping Columns
- Dropping Rows
- Various Parameter In Dropping functionalities
- Handling Missing values by Mean, Median and Mode

## Codes:
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('DataFrame').getOrCreate()
df_spark = spark.read().csv('test.csv', header=True, inferSchema=True)
df_spark.drop('Name')
df_spark.na.drop()
# This is default way of na.drop, any.
df_spark.na.drop(how='any')
# 'all' will trop those containing every feature as na
df_spark.na.drop(how='all')
# You can add threshold, those get at least 2 non-na will remain.
df_spark.na.drop(how='any', threshold=2)
# You can use subset to limit the view from whole to a little subset, so that you can use threshold or 'how' or flexible
df_spark.na.drop(how='any', subset=['Age'])

# so you can fill in the na, too
# you provide fill(value_to_fill, columns_to_select)
df_spark.na.fill('Missing Values', ['Experience', 'age'])

#You can use MLlib, too
from pyspark.ml.feature import Imputer

imputer = Imputer(
	inputCols=['age', 'Experience', 'Salary'],
	outputCols=["{}_imputed".format(c) for c in ['age', 'Experience', 'Salary']]
).setStrategy('median')
imputer.fit(df_pyspark).transform(df_spark).show()


```
