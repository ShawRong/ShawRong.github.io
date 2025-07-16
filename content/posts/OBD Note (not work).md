---
title: "OBD Note (not work)"
date: 2025-07-16T02:55:21.308Z
draft: false
tags: []
---

# Note of Optimal Brain Damage
**interests**
- Complexity measures: **Vapnik-Chervonenkis dimensionality**, a time-honored(albeit inexact) measure of complexity: simply the number of non-zero free parameters.
- some measure of network complexity: in the statistical inference literature and NN literature.
- How can the author make statement that: automatic network minimization procedure ands as **as an interactive tool to suggest better architectures.**
- One of the main points of this paper is to **move beyond the approximation that magnitude equanls saliency**
- I don't think it works now.

# Key findings:
- use weight decay, non-proportionate for sparsity or mixture precision
- It omit the cross term, lacking ability of find redundant pattern, and not simpler than magnitude way.

# Questions:
- what's group lasso, which one is better compared with weight decay? why it's suggested weight decay is not suitable for sparsity? can weight decay therefore be used for mixture precision.
- does optimal brain damage really works? we need to try or research. there are several modern method:
	- Magnitude-based pruning (simpler, often similarly effective)
	- Gradual pruning during training
	- Lottery ticket hypothesis approaches
	- More sophisticated second-order methods like Fisher Information
- what's fisher information?
- In the assumption of OBD, it says: "delta E caused by deleting several parameters is the sum of the delta E's caused by deleting each parameter individually." Does this assumption really work?

# claude talk
-[claude public](https://claude.ai/public/artifacts/6bc3dc4c-53b7-4548-a9cb-12d1533f3baf)
-[OBD does not work today.](https://claude.ai/public/artifacts/6535b7e0-a73b-49ea-a8e0-4010974251d4)