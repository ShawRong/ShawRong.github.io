---
title: "Systematic Outliers"
date: 2025-09-25T09:49:09.037Z
draft: false
tags: []
---

# Systematic Outliers are Simultaneous and Interconnected
- The three types of outliers are not isolated but occur simultaneously.
- (**Weight**)Weight outliers align perfectly with activation outliers in the **feature dimensions**
- (**Activation and Attention**)95% of the positions in the input sequence that have activation outliers also coincide with positions where attention outliers appear. (activation and attention)
- The outliers of activation and attention concentrated in start tokens like <\s> and \[CLS\] and Weak Semantic tokens.
- Corelation between different types of outliers: 1. weight outliers has 100% consistency with the activation outliers 2. Attentions outliers has 95% consistency with the activation outliers. 3. weight outliers appear at the same feature dimmension with the activation outliers. 
- They got an important observation that: In the process, weight outliers lead to activation outliers, which then influence attention outliers, with this influence extending to non-outlier tokens.
- Just like the Figure 7 shows, we can see a normal activation comes in and amplified by gate and up_proj, and further amplyfied by the down_proj. So they says weight outliers lead to activation outliers.
- The activation outliers can influence the generated Q and K. But the value vectors corresponding to these tokens show comparatively smaller magnitudes.
- Some tokens become **outliers** in activation space. These outlier tokens, via their queries and keys, align in such a way that other tokens strongly “pay attention” to them (i.e., their dot-product attention scores become large). But those tokens’ **values** (the information they carry) are _not_ proportionally large. So, the model might be using these tokens not to contribute large content but as **anchors** — reference tokens that “pull” or arrange the attention structure, indirectly shaping how other tokens’ information gets aggregated.
- Outliers gradually vanish in the final layers by values of opposite signs. You can check the Figure 9 for more infomation. You can see that the activation after down_proj adding with input activation cancel out each other and produce a reasonable fair activation.
- **Summary**: In the lifecycle of systematic outliers, weight outliers drive the emergence of activation outliers, which propagate anomalies into the attention mechanism. This interdependence extends their influence to non-outlier tokens.




## Questions
- [ ] It said that "The systematic outliers exhibit strong correlations across feature and squenece dimensions as well as layers." what's the sequence dimension here?
	It means this sytematic outlier appear at token level in different features but also shows in different token level and layer level. Outliers are rare, but when they occur, they are highly structured. The same feature dimensions light up across token and layers, rather than being random one-offs. That's why they're called systematic outliers.
- [ ] It's said there are three types of outliers, what's them?
- [ ] What's the meaning of  consistency in table 1?
- [ ] Are they analysing training or inference?



# Future Analysis of Systematic Outlers

## Questions
- [ ] It says that "ensuring minimal updates for those tokens", I am not that clear about this so called update things.
- [ ]