---
title: "Bert CLS token Study Notes"
date: 2025-07-11T07:57:38.679Z
draft: false
tags: []
---

# BERT [CLS] Token - Study Notes

## BERT Output Structure

- BERT outputs hidden representations for each input token position
- Example: `[CLS] hello world [SEP]` → 4 hidden vectors (one per token)
- Each position gets a contextual representation

## What is the [CLS] Token?

- **Purpose**: Designated position for sequence-level information aggregation
- **Mechanism**: Uses self-attention to "see" and combine info from all other tokens
- **Design**: Has no inherent meaning, so it's free to learn task-specific representations

## Key Point: Cannot Use [CLS] Directly

❌ **Pre-trained [CLS] won't work for your task**

- Pre-trained [CLS] is optimized for "Next Sentence Prediction"
- NOT optimized for sentiment, classification, or other downstream tasks

## How to Use [CLS] Properly

✅ **Fine-tuning is required**

1. **Add task-specific head**:
    
    - Take [CLS] hidden state → feed to classification layer (linear + softmax)
2. **Fine-tune entire model**:
    
    - New classification head (starts random)
    - Pre-trained BERT parameters (including [CLS] behavior)
3. **Result**: [CLS] learns to aggregate info relevant to YOUR specific task
    

## Summary

- [CLS] = learnable sequence-level aggregation point
- Pre-trained [CLS] ≠ ready for your task
- Fine-tuning required to make [CLS] useful for classification