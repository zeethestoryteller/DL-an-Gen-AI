# word embeddings, the foundation you need before attention and transformers.

## 1. The problem: words as plain numbers don't work

Earlier in the course, each word got turned into a single number (an index) — "the" = 1, "cat" = 45, etc. This worked as *input* to an RNN, but it has a big flaw:

A computer sees "happy" = 37 and "sad" = 812. Those numbers tell it nothing about the fact that happy and sad are opposites, or that "delighted" is basically the same idea as "happy." Meaning (semantics) just isn't there in a single scalar number.

**Core idea:** humans understand words by their *relationships* to other words. Machines need a representation that captures that.

## 2. The fix: represent each word as a vector, not a number

Instead of "cat" = 45, we say "cat" = `[0.2, -0.7, 1.3, ..., 0.05]` — a list of, say, 200 numbers. This is a **word embedding**.

- Every word in the vocabulary gets its own unique vector.
- You choose the length of this vector (the "embedding size" or "embedding dimension") — e.g., 200.
- Think of these 200 numbers as 200 "hidden categories" the model can use to describe meaning — though we never manually define what those categories are. The model figures that out itself during training.

**Why this matters:** with vectors, you can measure *distance* between words. Words with similar meaning end up with similar vectors (small distance between them). Words with unrelated meanings end up far apart.

## 3. Semantic geometry: meaning as space

If embeddings are trained well, something remarkable happens automatically:

- `cat` and `dog` vectors are close together (both are pets/animals).
- `car` is far away from both (unrelated concept).
- Clusters form naturally: animals together, vehicles together, fruits together — without you ever telling the model "these are animals."

Even more strikingly, **vector arithmetic captures relationships**:

```
embedding(king) - embedding(man) + embedding(woman) ≈ embedding(queen)
embedding(India) - embedding(New Delhi) + embedding(France) ≈ embedding(Paris)
```

This is emergent — nobody programs this rule in. It just falls out of training on enough text. This famous demonstration is from the original **word2vec** paper (~2013).

## 4. How embeddings get learned: "a word is known by the company it keeps"

This is the central training philosophy. The model doesn't know what any word *means*. It only looks at **which words tend to appear near each other** in huge amounts of text.

- "My pet is a cat" and "My pet is a dog" occur often → "pet" gets pulled close to both "cat" and "dog" in vector space.
- "My pet is a car" basically never occurs → "car" stays far from "pet."
- Over many billions of sentences, these co-occurrence patterns statistically shape the vectors so that related words end up near each other.

This is done through gradient descent, repeated over a massive text corpus — no one hand-labels any of this.

**Classic algorithms that do this:** Word2Vec, GloVe, FastText. These produce *static* embeddings — one fixed vector per word.

## 5. The limitation of static embeddings: one word, multiple meanings

Problem: "bank" in "river bank" and "bank" in "money bank" are totally different meanings, but a static embedding gives "bank" the exact same vector every time. The model can't tell them apart.

## 6. The fix: contextual embeddings (BERT and GPT)

Modern models generate a **different embedding for the same word depending on its surrounding context** (the sentence it's in).

- **BERT** (bidirectional): reads the whole sentence — both the words before *and* after the target word — before deciding on its embedding. Like a bidirectional RNN/LSTM.
- **GPT** (autoregressive): only reads the words *before* the target word (left-to-right), then decides the embedding. This matters later for why GPT is good at generating text one word at a time.

So "bank" near "river" gets one vector, and "bank" near "money" gets a different vector — even though it's the same word in the dictionary.

## 7. How embeddings are actually trained in practice

1. **Initialize randomly**: for every word in the vocabulary, create a vector of your chosen size (e.g., 200) filled with small random numbers.
2. **Feed into a task**: pass these embedding vectors as input to a neural network that does some task — sentiment analysis, or (more powerful) predicting the next word in a sentence.
3. **Backpropagate**: when the network makes an error, gradients flow all the way back through the network *and into the embedding vectors themselves*. So the embeddings get updated right alongside the network's weights.
4. **Repeat over huge amounts of text**: the more data and the harder the task (e.g., "predict the next word" over the whole internet), the richer and more meaningful the embeddings become.

This is why GPT/Gemini-style models need so much data — they aren't told any meaning directly; they infer everything purely from which words appear near which other words, across billions of examples.

## Implementing this yourself

Here's the minimal, concrete version of what's described, in PyTorch:

```python
import torch
import torch.nn as nn

vocab_size = 10000      # number of unique words
embedding_dim = 200     # size of each word's vector

# Step 1: random initialization of every word's vector
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# Step 2: turn a sentence (as indices) into embeddings
# e.g. "the cat" -> [1, 45]
word_indices = torch.tensor([1, 45])
word_vectors = embedding_layer(word_indices)   # shape: [2, 200]

# Step 3: feed word_vectors into your RNN / next-word-predictor / classifier
# During training.backward(), gradients update embedding_layer.weight too
# — this is how the vectors "learn" meaning over time.
```

- `embedding_layer.weight` is literally the big table of vectors — one row per word.
- If you train this jointly with a next-word-prediction task on a lot of text, you're doing a simplified version of word2vec/GPT-style embedding training.
- If you don't want to train your own, you can just download **pretrained** embeddings (GloVe, word2vec, or a pretrained BERT/GPT model) and use those vectors directly — that's what most real projects do.

To check that your embeddings "worked," measure cosine similarity between vector pairs:

```python
import torch.nn.functional as F

sim = F.cosine_similarity(word_vectors[0], word_vectors[1], dim=0)
# high similarity = words are semantically close
```
---
The idea that once you have these embeddings, you need a mechanism for each word to "look at" other relevant words in the sentence (like GPT looking left, or BERT looking both directions) to build a smarter, context-aware representation. That's the next lecture in this sequence, and it's the direct ancestor of the Transformer architecture ("T" in GPT).
