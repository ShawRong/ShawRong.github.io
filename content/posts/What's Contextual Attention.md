---
title: "What's Contextual Attention"
date: 2025-07-11T07:21:58.192Z
draft: false
tags: []
---

# Self-Attention vs Contextual Attention

## Important Clarification

**"Contextual attention" is not a standard term in the field.** You might be thinking of different types of attention mechanisms. Let me explain the key distinctions:

## Self-Attention (Standard Term)

**Definition**: Each token attends to all tokens in the same sequence (including itself)

**Key Characteristics**:

- Input sequence: ["The", "bank", "river", "flows"]
- Each word looks at ALL words in the same sentence
- "bank" attends to: "The", "bank", "river", "flows"
- Used in: BERT, GPT, most modern transformers

**Formula**:

```
Attention(Q,K,V) = softmax(QK^T/√d)V
where Q, K, V all come from the same input sequence
```

## Cross-Attention (What you might mean by "contextual")

**Definition**: Tokens from one sequence attend to tokens from another sequence

**Key Characteristics**:

- Two sequences: Source and Target
- Example: Translation - English sentence attends to French sentence
- Query comes from target, Key/Value from source
- Used in: Encoder-decoder models, T5, original Transformer

**Formula**:

```
CrossAttention(Q,K,V) = softmax(QK^T/√d)V
where Q comes from sequence A, K,V come from sequence B
```

## Common Confusion Points

### 1. Self-Attention Creates Contextual Embeddings

- **Self-attention mechanism** → produces **contextual embeddings**
- The mechanism is "self-attention"
- The output is "contextual embeddings"

### 2. Types of Self-Attention

|Type|Description|Example|
|---|---|---|
|**Bidirectional**|Can attend to past and future tokens|BERT|
|**Causal/Masked**|Can only attend to past tokens|GPT|
|**Local**|Only attends to nearby tokens|Some efficient transformers|
|**Global**|Attends to all tokens|Standard transformers|

### 3. Attention Variants

|Variant|What it means|
|---|---|
|**Self-Attention**|Same sequence attends to itself|
|**Cross-Attention**|Different sequences attend to each other|
|**Multi-Head**|Multiple attention computations in parallel|
|**Scaled Dot-Product**|Standard attention formula with scaling|

## Visual Comparison

### Self-Attention (BERT/GPT style):

```
Input: "The bank river flows"
       ↓    ↓    ↓     ↓
    [Attend to all tokens in same sequence]
       ↓    ↓    ↓     ↓
Output: Contextual embeddings
```

### Cross-Attention (Translation style):

```
English: "The bank"  →  French: "La banque"
         ↓                      ↑
      [Attend across languages]
```

## In Modern LLMs

### GPT Models:

- Use **causal self-attention**
- Each token attends to previous tokens only
- Creates contextual embeddings

### BERT Models:

- Use **bidirectional self-attention**
- Each token attends to all tokens
- Creates contextual embeddings

### T5 Models:

- **Encoder**: Bidirectional self-attention
- **Decoder**: Causal self-attention + cross-attention to encoder

## Bottom Line

- **Self-attention** = the mechanism
- **Contextual embeddings** = the output
- **Cross-attention** = attention between different sequences
- There's no standard term "contextual attention" - you likely mean one of the above!