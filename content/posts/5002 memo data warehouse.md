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

There none means a tot
