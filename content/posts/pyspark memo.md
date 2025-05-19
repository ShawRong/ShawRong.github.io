---
title: "pyspark memo"
date: 2025-05-19T09:30:16.345Z
draft: false
tags: []
---

# Classic Staring Code

```python
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName('conf').setMaster("local[*]")
# to use spark, SparkContext is necessary.
sc = SparkContext(conf=conf)

data = [1, 2, 3, 4]
rdd = sc.parallelized(data, 4)
# or you can read from file (hdfs, file, ...)
file = sc.textFile('README.md', 4)
# no materialization so far.
```

# Transformations
## typical one
- map
- filter
- distince
- flatMap
```python
rdd.map(lambda x: x * 2)

rdd.filter(lambda x: x % 2 == 0)

rdd.distinct()
>>> [1, 2, 3, 3] -> [1, 2, 3]

# flatMap will flap list.
rdd.flatMap(lambda x: [x, x+5])
>>> [1, 2, 3] -> [1, 6, 2, 7, 3, 8]
rdd.flatMap(lambda x: list(x))
>>> [{1, 2}, {3}] -> [1, 2, 3]
# transformation, so all not materialized.
```

## key-value transformation
- reduceByKey
- sortByKey
- groupByKey
```python
rdd.reduceByKey(lambda a, b: a + b)
>>> [(1, 2), (3, 4), (3, 6)] -> [(1, 2), (3, 10)]
# sort by keys 
rdd.sortByKey()
>>> [(1, 2), (2, 3), (1, 3)] -> [(1, 2), (1, 3), (2, 3)]
rdd.groupByKey()
>>> [(1, 2), (1, 3), (2, 3)] -> [(1, [2, 3]), (2, [3])]

```
## other
- mapValues
- sortBy
- join
```python
rdd.groupByKey().mapValues(sum)
# False for desending, True for ascending
rdd.sortBy(lambda x:x[1], False)

rdd.join(other_rdd, rdd.id == other_rdd.uid)
```


# Actions
reduce (commutative and associative)
take
collect
takeOrdered(n, key=func)
count
isEmpty
treeReduce(Reduces the elements of this RDD in a multi-level tree pattern. faster than normal one)
```python
rdd = sc.parallelize([1, 2, 3])
rdd.reduce(lambda a, b: a * b)
rdd.take(2)
rdd.collect()

rdd = sc.parallelize([5, 3, 1, 2])
# defualt ascending order, key function can be customized
rdd.takeOrdered(3, lambda s: -1 * s)

rdd = sc.parallelize([1, 2, 3, 4])
# start with $1, using function $2 to merge in partition, use function $3 to merge between partition
rdd.aggregate(set(), lambda x, y: x.add(y), lambda x, y: x.union(y))

# foreach is action, map is transformation
fruits.foreach(lambda x: print("I get a", x))
>>> I get a pen
>>> I get a apple
```


# Accumulator & Broadcast
## Accumulator
read-only value for driver
write-only for task

accumulator can used in actions or transformations:
- actions: each taskps update to accumulator only once
- failed/slow may get rescheduled, no guarantees
```python
accum = sc.accumulator(0)
rdd = sc.parallelize([1, 2, 3, 4])
def f(x):
	global accum
	accum += x #pay attention to '+='
rdd.foreach(f)
accum.value
>>> Value:10
```

```python
file = sc.textFile(inputFile)
blankLines = sc.accumulator(0)
def extractCallSigns(line):
	global blankLines
	if (line == ""):
		blankLines += 1
	return line.split(" ")
callSigns = file.flatMap(extractCallSigns)
print '...'
```

## Broadcast
read only on executors
```python
signPrefixes = sc.boradcast(loadCallSignTable())
def processSignCount(sign_count, signPrefixes):
	country = lookupCountry(sign_count[0], signPrefixes)
	count = sign_count[1]
	return (country, count)
countryContactCounts = (contactCounts.map(processSignCount).reduceByKey(lambda x, y: x + y))
```