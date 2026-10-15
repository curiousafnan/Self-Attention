# Approach — Day 41

## From Cross-Attention to Self-Attention
Day 39:
    query  ← decoder state s
    keys   ← encoder states h_i
    values ← encoder states h_i
Two sequences, two sources.

Day 41: all three come from the same sequence. A token attends to
the other tokens around it, and the output is a contextual embedding
— "sat" becomes "sat, with a cat doing the sitting."

This is what Word2Vec could not do. There "bank" had one fixed
vector forever. Here it differs depending on whether "river" or
"money" is nearby, because it attends to them.

## Why Three Projections
A single vector would have to play three roles at once:
    asking    — "which tokens here are subjects?"
    matching  — "I am a noun, I could be a subject"
    supplying — the content actually absorbed once selected

Those are different jobs competing for the same dimensions. What
makes a token findable is not necessarily what is worth taking from
it. Three learned projections let each role be shaped independently.

    Q = x @ W_q    the question
    K = x @ W_k    the label matched against it
    V = x @ W_v    the content handed over

Q and K produce the scores. V is not involved in matching at all.

## The Mechanism
    Q, K, V = x @ W_q, x @ W_k, x @ W_v     each (batch, seq, d)
    scores  = (Q @ Kᵀ) / √d_k                (batch, seq, seq)
    α       = softmax(scores, dim=-1)
    out     = α @ V                          (batch, seq, d)

Row i, column j of the score matrix is "how much does token i attend
to token j." Square, because every token scores against every token.

Kᵀ swaps the last two axes only — (batch, d, seq) — so the batch
dimension is carried along by the batched matmul.

## Why √d_k
Each dot product sums d_k terms, so with unit-scale inputs the result
grows as roughly √d_k. With d_k=128 that is a factor of ~11, putting
scores in a range like ±30.

softmax([30, 2, 1]) ≈ [1.0, 0.0, 0.0] — saturated. Near-zero weights
mean near-zero gradient to those positions, and learning stalls
before the model has worked out where to attend.

Dividing by √d_k cancels exactly that growth. It is a fixed constant
known before seeing any data — a scaling correction, NOT
normalization. LayerNorm (Day 43) computes mean and std from the
data; this does not.

## Causal Masking
Build the mask from a matrix of ones, never from the scores:
    mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))

diagonal=1 excludes the diagonal — a token may attend to itself.

-inf, not zero. Zero is an ordinary score that softmax would give a
healthy weight. e^(-inf) = 0 is what produces an actual zero weight,
after which the surviving positions renormalize to sum to 1.

mask=None for encoder self-attention, mask=True for decoder.