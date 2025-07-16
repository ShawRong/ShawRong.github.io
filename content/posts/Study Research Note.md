---
title: "Study Research Note"
date: 2025-07-16T02:59:48.949Z
draft: false
tags: []
---

# good reading
- llm quantization world -[zhihu](https://zhuanlan.zhihu.com/p/686232369)
	it include:
	- introduction to fix point quantization
	- current study focus on per-channel, token, group fix point quantization
	- per-group refer to we do quantization on a series of continuous element
	- 
# mixture quantization paper

- AWQ and SqueezeLLM  
	-[zhihu](https://zhuanlan.zhihu.com/p/697992170)
	AWQ use linear scale (pretty good)
	Squeeze use dynamic un-uniform scale quantization (not that good)
	key observation: there are salient weight, we can perform quantization according to these salient weight
- QBS(Qptimal Brain Surgeon), classic, need to read
- GPTQ and OBQ
- 


# TODO list
-[TODO list](https://shawrong.github.io/posts/read-todo/)
obsidian: [[Read TODO]]
# Questions:
- what's group lasso, which one is better compared with weight decay? why it's suggested weight decay is not suitable for sparsity? can weight decay therefore be used for mixture precision.
- does optimal brain damage really works? we need to try or research. there are several modern method:
	- Magnitude-based pruning (simpler, often similarly effective)
	- Gradual pruning during training
	- Lottery ticket hypothesis approaches
	- More sophisticated second-order methods like Fisher Information
- what's fisher information?
- In the assumption of OBD, it says: "delta E caused by deleting several parameters is the sum of the delta E's caused by deleting each parameter individually." Does this assumption really work?
# Key findings (now):
- use weight decay, non-proportionate for sparsity or mixture precision
- It omit the cross term, lacking ability of find redundant pattern, and not simpler than magnitude way.

# Links
-[OBD note](https://shawrong.github.io/posts/obd-note-not-work/)
obsidian: [[OBD Note (not work)]]