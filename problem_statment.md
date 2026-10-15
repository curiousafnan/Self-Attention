# Day 41 — Self-Attention

## Problem
Day 39's attention bridged two different sequences — queries from
the decoder, keys and values from the encoder. The recurrence was
still there: the LSTM loop computed one timestep at a time, and
position 50 could not be computed until position 49 finished.

Self-attention applies the same mechanism within a single sequence,
and drops the recurrence entirely. Every position is computed
simultaneously.

## Core Questions
- What is a token computing when it attends to its own sequence?
- Why project the embedding three times instead of using it directly?
- Why divide the scores by √d_k?
- Why must future positions be blocked during training?

## Requirements
- SelfAttention module: three projections, scaled dot-product, softmax
- Optional causal mask
- Heatmap comparing masked and unmasked attention weights