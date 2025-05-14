+++
date = '2025-05-14T17:10:29+08:00'
draft = false 
title = '5002 Data Warehouse Memo'
+++

# What's Data warehouse
Suppose there are two queries.
```
select all the UST students coming from HK.
select all the UST students coming from Mainland.
```
They are complicate query, and could cost for one day.
So we can store our pre-computed results.
This is Data Warehouse--pre-computed results.

User ask database, data warehouse could response without querying database. 

# Basic Ideas about how to build warehouse.

We know clearly that we there are query related to grouping or agregating. So we can build our warehouse according to these group. 
Suppose there are table containing three different columns-- p, c, s. 
we can mark a group by with p, c, s as pcs. And a group with c, s as cs, so on so forth.

And we know the result of group c or s be derived from cs. So we can draw a typical picture of this thing.

![[Pasted image 20250514173327.png]]

# Cost calculation
There none means a total count of the rows(no grouping just select from all).


Here the xx M means the cost of answering a specific view(i.e. pc), if this view is materialized. If the one of the parents or parent-parent are not materialzed. It will trace back to the most close materialized parent for answer, and the cost is consistent with it. 
```
Answering s:
If without any thing materialized, the cost could be 6M.
If ps is materialized, the cost could be 0.8M.
If s is materizalized, the cost coulbe be 0.01M.
```
# How to calculate the gain.
We just to calculate the previous cost and current cost. And we compare the them to get the different. 
The difference is the gain.
