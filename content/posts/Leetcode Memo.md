---
title: "Leetcode Memo"
date: 2025-05-29T09:46:50.443Z
draft: false
tags: []
---

# Tier
[3373. 连接两棵树后最大目标节点数目 II](https://leetcode.cn/problems/maximize-the-number-of-target-nodes-after-connecting-trees-ii/)
The traverse of tier can be viewed as the traverse of graph.
We can do the following to traverse the tier to get depth, from root. This is based on depth first search.
```python
# This is for build children list, for dfs
# edges represents: [[1, 2], [2, 3], [3, 4]], i.e. node 1 links to 2 etc...
def builder(edges):
	n = len(edges) + 1 #how many node
	children = [[] for _ in range(n)]
	for u, v in edges:
		children[u].append(v)
		children[v].append(u) #there are duplicate, so you need to check using parent == child?
	return children

# node for current search for node, just a index for node, like 1
# parent, the parent of current searching node. 
# depth, mark depth here
# children, comes from builder
def dfs(node, parent, depth, children):
	# res update here
	for child in children[node]:
		if child == parent: # for duplicate checking
			continue
		# res accumulate with children
		dfs(child, node, depth + 1, children)
	return res
```