---
title: "Contextual Embedding"
date: 2025-07-11T07:21:28.798Z
draft: false
tags: []
---

# Contextual Embeddings - Simple Summary

## What Are They?

Word representations that **change based on context**, unlike static embeddings where each word has a fixed vector.

Example: "bank" gets different embeddings in:

- "river bank" (geographic)
- "savings bank" (financial)

## Where Are They Created?

**Inside the self-attention mechanism** of transformer layers.

## How Do They Work?

### The Process:

1. **Static embeddings** (from lookup table) + **positional encoding**
2. **Self-attention** calculates how much each word should "pay attention" to others
3. **Mix embeddings** based on attention weights → **Contextual embeddings**

### The Formula:

```
For each token i:
contextual_embedding[i] = Σ(attention_weight[i,j] × value_embedding[j])
                         j=0 to sequence_length
```

**Key insight**: Each token's final embedding is a weighted sum of ALL tokens in the sequence (including itself).

## Current Usage in LLMs

### Modern Models:

- **GPT-4, Claude, etc.**: Use 100+ transformer layers
- **Each layer** creates more sophisticated contextual embeddings
- **Context windows**: Up to 2M+ tokens
- **Multi-head attention**: Captures different relationship types

### Architecture:

- **GPT**: Causal (masked) self-attention
- **BERT**: Bidirectional self-attention
- **T5**: Encoder-decoder attention

## Why Important?

- **Polysemy**: Same word, different meanings
- **Context understanding**: "bank" near "river" vs "loan"
- **Long-range dependencies**: Words can influence each other across long distances
- **Foundation of modern NLP**: All current LLMs are built on this principle

## Key Takeaway

Contextual embeddings aren't just used in modern LLMs - **they ARE what makes modern LLMs work**. Every token's representation dynamically incorporates information from the entire context through self-attention.