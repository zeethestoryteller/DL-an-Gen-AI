# Attention Mechanism

## 1. Why RNNs weren't good enough (the motivation)

In an encoder-decoder RNN (used for translation), the whole input sentence gets squeezed into **one single vector** called the context vector (HC). The decoder then generates the output using only that one vector.

Three problems with this:

**Problem 1 — Single vector bottleneck.** You're trying to compress an entire sentence (maybe 50 words) into one fixed-size vector (say, 200 numbers). There's a limit to how much meaning you can squeeze into that vector, just like human short-term memory can only hold a handful of things at once. Long sentences lose information — that's why RNN-based translation degrades badly after ~30-40 tokens.

**Problem 2 — Vanishing/exploding gradients.** Already discussed in earlier weeks; long sequences make training unstable.

**Problem 3 — Sequential processing.** To compute hidden state h2, you *must* first compute h1. This means RNNs can't be parallelized — you're stuck processing one token at a time, which wastes GPU power. Modern deep learning is about scale (billions/trillions of tokens), and sequential processing is a hard bottleneck against that.

**The fix:** instead of forcing everything through one bottleneck vector, let the decoder look back at *all* the input words directly, and decide dynamically which ones matter most for the word it's generating right now. That's attention.

## 2. The core idea of attention

Key insight: when translating, each output word doesn't come from just *one* input word — it comes from a **mixture (weighted combination)** of input words.

Example: translating "the cat is sleeping on the mat" into Hindi — the word meaning "cat" might come almost entirely from the input word "cat" (weight ≈ 1, everything else ≈ 0). But a word like "rahi" (roughly "is," but carrying gender information) draws a little from "sleeping," a little from "is," and a lot from "cat."

So mathematically: each output token = a **weighted sum** of input embeddings, and those weights ("attention weights") are learned automatically — not hand-coded.

## 3. The Query, Key, Value (Q, K, V) framework

This whole thing is modeled after a database lookup:

- **Query (Q):** the question you're asking right now ("how much did this Aadhaar-numbered person earn?")
- **Key (K):** the thing you compare your question against (the Aadhaar number column)
- **Value (V):** the actual answer you retrieve once you've matched the query to the right key (the earning amount)

For attention: each word's embedding gets passed through three different learned weight matrices to produce its query vector, key vector, and value vector:

```
Q = E · W_Q
K = E · W_K
V = E · W_V
```

`W_Q`, `W_K`, `W_V` are matrices of learnable parameters — they start random and get trained via backpropagation, just like any other neural network weights. Nobody tells the model "this is a query" — that role emerges naturally from training.

<img width="787" height="610" alt="image" src="https://github.com/user-attachments/assets/41e7a1a9-303d-4017-855e-591b5a14116a" />


**Q and K feed the similarity/softmax pipeline, but V skips straight past that and rejoins later** — notice the V line running down the right side and entering the "weighted sum" box directly. That's the detail that trips people up when reading the formula `softmax(QKᵀ/√d)·V` — it looks like V goes through the same pipeline as Q and K, but it doesn't; it's just the thing being blended once the weights are decided.

Everything in the diagram maps directly onto the numbers we computed earlier:
- **Q · Kᵀ** → the raw scores `[[-0.403, 0.097], [0.705, 0.657]]`
- **Softmax** → the α weights `[0.438, 0.562]` and `[0.506, 0.494]`
- **Weighted sum with V** → the final `Ē1`, `Ē2` vectors


## 4. How attention weights are computed

1. **Similarity score:** compare a query vector to every key vector, using some similarity measure (dot product is the standard choice — more on this below).
2. **Normalize with softmax:** raw similarity scores don't sum to 1, so you push them through softmax to turn them into proper weights that are all positive and add up to 1 (like a probability distribution).
3. **Weighted sum of values:** multiply each value vector by its corresponding attention weight and add them up. That sum is your new, context-aware embedding.

```
weight_i = softmax(similarity(Q, K_i))
output = Σ weight_i * V_i
```

**Hard attention** = one weight is exactly 1 and the rest are 0 (like picking exactly one row from a database).
**Soft attention** (the common kind) = weights are spread across several inputs, blending information from multiple words.

## 5. Similarity/scoring functions

Several ways exist to measure how "similar" a query vector is to a key vector (these are called kernels): Gaussian kernel, box-car kernel (hard cutoff), or a linear/ReLU-based kernel. The lecture shows that if you expand out a squared-distance-based similarity measure algebraically, most of the terms end up canceling or becoming constant once normalized — what's actually left doing the real work is the **dot product** between Q and K.

**Dot product intuition:** if two vectors are unit length and aligned, their dot product is 1 (very similar). If they're perpendicular, the dot product is 0 (unrelated / independent concepts).

## 6. Scaled dot-product attention

Plain dot product has an issue: as the vector dimension `d` grows, the dot product's variance grows too (since you're summing more terms), producing very large numbers — bad for numerical stability and gradient descent.

**Fix (from the "Attention Is All You Need" paper, 2017):** divide the dot product by √d before applying softmax:

```
score(Q, K) = (Q · K) / sqrt(d)
attention_weight = softmax(score)
```

This is called **scaled dot-product attention** — the standard formula used almost everywhere today.

## 7. The full matrix formula

Instead of doing this one word at a time, you do it for the whole sentence at once using matrices (this is what enables parallel processing on GPUs — a big improvement over RNNs):

```
Attention(Q, K, V) = softmax(Q · Kᵀ / sqrt(d)) · V
```

Shapes:
- `Q` is `n × d` (n = number of output/query tokens, d = embedding size)
- `K` is `m × d` (m = number of input/key tokens)
- `Q · Kᵀ` gives an `n × m` matrix — a similarity score between every query and every key
- After softmax, multiply by `V` (which is `m × v`) to get an `n × v` output — one context-aware vector per output token

## 8. Self-attention vs cross-attention

- **Self-attention:** Q, K, and V all come from the *same* sequence. Used to build context-aware embeddings within a single sentence — e.g., letting "bank" incorporate meaning from "river" or "money" appearing nearby, even before any translation happens. This is one of the main contributions of the Transformer paper — applying attention *everywhere*, not just during decoding.
- **Cross-attention:** Q comes from one sequence (e.g., the decoder/output), while K and V come from a *different* sequence (e.g., the encoder/input). This is how the decoder looks back at the original input sentence while generating a translation, or how an answer generator looks back at a source question/document.

## 9. Masking (handling padding)

Sentences in a batch have different lengths, so shorter ones get padded with blank/filler tokens to match the longest sentence. Problem: you don't want the model paying attention to meaningless padding tokens.

**Trick:** instead of using slow conditional logic ("if padded, skip"), you assign padded positions a huge negative score (like -10⁶) before the softmax. Since `e^(-1000000)` is essentially 0, softmax automatically assigns those positions ~0 attention weight — no explicit branching needed, which keeps everything fast on GPUs. This is called a **masked softmax**.

## Implementation

Here's scaled dot-product self-attention in PyTorch, following the lecture's structure exactly:

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d = Q.size(-1)
    # similarity scores: Q x K^T, shape (n, m)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)

    if mask is not None:
        # mask == 0 means "padding" -> push score very negative
        scores = scores.masked_fill(mask == 0, -1e9)

    weights = F.softmax(scores, dim=-1)   # normalize so each row sums to 1
    output = torch.matmul(weights, V)      # weighted sum of values
    return output, weights


class SelfAttention(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W_Q = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = torch.nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, embeddings, mask=None):
        # embeddings shape: (batch, seq_len, embed_dim)
        Q = self.W_Q(embeddings)
        K = self.W_K(embeddings)
        V = self.W_V(embeddings)
        output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        return output, attn_weights


# example: "thinking machines" — 2 tokens, embedding size 8
embed_dim = 8
embeddings = torch.randn(1, 2, embed_dim)  # (batch=1, seq_len=2, dim=8)

attn = SelfAttention(embed_dim)
new_embeddings, weights = attn(embeddings)

print(weights)   # how much "thinking" attends to itself vs "machines", and vice versa
print(new_embeddings)  # the new, context-aware embeddings
```

For **cross-attention**, the only change is that Q comes from the decoder sequence while K and V come from the encoder sequence:

```python
class CrossAttention(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W_Q = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = torch.nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, decoder_embeddings, encoder_embeddings, mask=None):
        Q = self.W_Q(decoder_embeddings)   # from output sequence
        K = self.W_K(encoder_embeddings)   # from input sequence
        V = self.W_V(encoder_embeddings)   # from input sequence
        return scaled_dot_product_attention(Q, K, V, mask)
```

For **masking padded tokens**, build a mask tensor where `1` = real token and `0` = padding, and pass it in — the `masked_fill` line handles the rest.

