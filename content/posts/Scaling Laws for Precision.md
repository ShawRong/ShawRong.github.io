---
title: "Scaling Laws for Precision"
date: 2025-09-25T09:49:38.130Z
draft: false
tags: []
---

Abstract
===========
The main contribution of this paper is devise "precision-aware" scaling laws for training and inference.
It comes from observations that the degradation introduced by post-training quantization increases as models are trained on more data, which demands for scaling law to tell us the data should be used.
For training, scaling laws suggest that training larger models in lower precision may be compute optimal.
They unify the scaling laws for post and pre-training quantization to arrive at a single functional form that predicts degradation from training and inference in varied precisions.


Questions:
===========



Summary:
===========
- The previous scaling law tells us the tradeoffs between model/dataset size to balance performance and compute. However, the precision are becoming an important third factor that contribute to both cost and performance. With the purpose of knowing the tradeoffs between precision, parameters and data, and their relationship with respect to pre-training and inference.
-