# Day 41 — Self-Attention

Scaled dot-product self-attention with optional causal masking.
No recurrence — every position computed in one pass.

## Core Class
SelfAttention(hidden_states)
  W_q, W_k, W_v : nn.Linear(d, d, bias=False)
  forward(x, mask=None) → (output, weights)

## The Mechanism
    Q, K, V = x @ W_q, x @ W_k, x @ W_v
    scores  = (Q @ Kᵀ) / √d_k
    α       = softmax(scores, dim=-1)
    out     = α @ V

Shapes: x (2,6,128) → out (2,6,128), weights (2,6,6)

## Key Insight
One vector cannot be a question, an answer-label, and content at
once. Three projections let each role be learned separately.

The mask uses -inf, not 0, because the zero must appear AFTER
softmax. It exists to make parallel training honest — otherwise
position 3 reads position 4, which is its own answer.

Untrained attention is uniform. The structure comes from training,
not from the architecture.

## Run
python main.py
python visualize.py