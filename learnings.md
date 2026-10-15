# Learnings — Day 41

## What a Token Computes by Attending to Its Own Sequence
In "the cat sat on the mat", the token "sat" attends most strongly
to "cat" — because "cat" is its subject. The output at that position
is then a weighted mixture:

    0.05·V_the + 0.70·V_cat + 0.15·V_sat + ...

The embedding that went in knew only the word "sat". The vector
coming out carries information about what did the sitting. That is a
contextual embedding, and every position produces one simultaneously
in a single matmul.

## Three Roles, Three Projections
One vector cannot be a good question, a good answer-label, and good
content at the same time. Q asks, K is matched against, V is what
gets taken. Separate learned matrices, same input.

V plays no part in the matching. Q·K decides how much; V is what.

## Scaling Is Not Normalization
/ √d_k is a fixed constant computed from the dimension, applied
before softmax to stop it saturating. Nothing is measured from the
data.

LayerNorm — which appears later, between sublayers — computes mean
and std from the actual values. Different operation, different
purpose, both present in a transformer.

## Masking: Why Training Specifically
Training feeds the whole target sequence at once for parallelism.
That means position 3's computation can see positions 4, 5, 6 — the
answers it is meant to predict. Without a mask the model learns
"copy the next slot", loss drops to zero, and inference fails
completely.

At inference there is nothing to cheat with — future tokens do not
exist yet. The mask is kept so the computation matches what the
weights were trained under.

The real point: without masking you would have to generate one
position at a time, sequentially. The mask is what buys parallel
training while preserving the left-to-right constraint. It is the
trick that makes transformers trainable at scale.

## Untrained vs Trained Attention
Untrained weights came out near-uniform, around 0.167 = 1/6. The
projections are random, so no token has reason to prefer any other.

After training the distribution peaks: some positions near 0.8,
most near 0.02. High weight means that token strongly influences
this one; near-zero means almost no influence.

Nothing in the code produces that peaking. The mask creates the
triangle; the structure inside each row comes entirely from
training, as W_q, W_k and W_v get shaped by the loss.

Same as Day 39's anti-diagonal, which emerged with no alignment
labels. Attention patterns are learned, not architectural.

## What the Side-by-Side Heatmap Shows
Unmasked: every cell populated, all around 0.17.
Masked: clean lower triangle. Row 0 is 1.00 on itself — only one
position available, so softmax has no choice.

Two details confirming the mask behaves correctly:
- Row 5 is IDENTICAL in both plots. The last row has nothing above
  the diagonal, so masking is a no-op there.
- Row 1 unmasked has 0.20, 0.10 in its first two columns; masked it
  becomes 0.66, 0.34. Same 2:1 ratio, rescaled. The masked positions
  did not simply vanish — the survivors absorbed their share.

## Bugs Fixed
- super().__init__(self) — takes no arguments
- torch.sqrt on a Python int — needs math.sqrt
- softmax without dim
- V * α element-wise instead of α @ V, and in the wrong order
- Returned the pre-softmax scores instead of the weights
- Built the mask from torch.triu(scores) rather than
  torch.triu(torch.ones(...)). This WORKED, because .bool() on
  nonzero random scores happens to give True — but any score of
  exactly 0.0 would come back False and silently leave one future
  position visible. The mask must not depend on the score values.