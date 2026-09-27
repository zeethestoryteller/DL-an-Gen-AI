# Word Embeddings 

---

## 1. The Problem: Words as Plain Numbers

Pehle jo approach thi usme har word ko ek single index diya jata tha:
`the → 1`, `a → 2`, aur aage bhi is tarah.

Yeh kaam to karta hai, lekin **meaning (semantics)** poori tarah missing ho jaati hai:

- Computer ko nahi pata ki "happy" "sad" ka opposite hai, ya "delighted" ke similar hai — uske liye woh sirf ek number hai.
- Insaan meaning ke through sochte hain: synonyms, opposites, categories — scalar (single number) representation isko capture nahi kar sakta.
- Isi missing structure ko **semantics** kehte hain.

---

## 2. Embeddings Aren't Just for Words (Image Analogy)

Yeh concept naya nahi hai — CNNs me bhi hum yehi dekh chuke hain:

```
256 × 256 image  →  [CNN layers]  →  200 × 1 embedding vector
```

Poori image (lakhs of pixels) ko end me ek chhota vector (jaise 200 numbers) me compress kar diya jaata hai — yeh vector image ka "essence" capture karta hai. Word embeddings bhi bilkul yehi cognitive idea replicate karte hain, bas text ke liye.

---

## 3. What is a Word Embedding?

Ek single number ki jagah, har word ko ek **list of numbers (vector)** diya jaata hai.

```
king → [0.62, -0.11, 0.94, ... ]   (e.g. 200 numbers)
```

- Vocabulary ke har word ka apna **unique vector** hota hai — koi do words same vector share nahi karte.
- Isko word ka "address" samajh sakte ho — jaise address me multiple pieces of info hoti hain (city, street, pin code), waise hi vector me multiple features hoti hain.
- Numbers ki count (e.g. 200) ko **embedding size / dimension** kehte hain.

---

## 4. The Key Property: Similarity = Distance

> "Words with similar meanings will have similar vectors, and the distance between two vectors tells us how similar the words are."

**Example (2-D toy embedding):**

| Word | Vector |
|------|--------|
| cat  | (1, 2) |
| dog  | (1.5, 2.5) |
| car  | (-0.3, 1.5) |

Yahan `cat` aur `dog` ke vectors close hain, jabki `car` unse door hai — exactly jaisa expect karte hain.

Yeh structure **hand-designed nahi hota** — training ke through automatically emerge hota hai, jab model ko bohot zyada text diya jaata hai.

---

## 5. The "Magic": Clustering & Vector Arithmetic

**Clustering:** Embeddings ko plot karo to bina kisi explicit instruction ke:
- cat, dog, pet — ek cluster
- car, truck, bus — dusra cluster
- apple, orange, banana — teesra cluster

**Analogies (word2vec, 2013):**

```
king − man + woman ≈ queen
India − New Delhi + France ≈ Paris
```

Agar yehi cheez scalar IDs (jaise `10 − 23 + 45`) se try karo, to kuch meaningful nahi milega. Sirf vectors hi yeh relational structure capture karte hain.

---

## 6. Training: How Are Embeddings Learned?

> "A word is known by the company it keeps."

- Billions of words books, articles, websites se feed karte hain.
- **Co-occurrence** dekha jaata hai — kaunsa word kiske paas aata hai. "Cat" aur "dog" dono often "pet" ke paas aate hain → unke vectors close ho jaate hain training ke dauraan.
- Statistical roots is idea ke **unigram, bigram, trigram** jaisi cheezon me hain (context window me kaunsa word kis word ke saath sath occur karta hai) — lekin deep learning me yeh sirf pure statistics nahi, gradient descent se aur sophisticated tareeke se seekha jaata hai.
- Classic algorithms: **word2vec**, **GloVe**, **fastText**.

**Training kaise hota hai (step by step):**

1. Har word ke liye ek random vector se initialize karo (embedding size, jaise 200, fix karke). E.g. `the → [-0.126, 0.26, ...]`, `a → [-0.6, -0.3, 0.85, ...]`.
2. Yeh embedding ek network me input jaati hai jo koi specific task solve karta hai — jaise sentiment analysis, ya next-word prediction.

```
word → embedding (E) → Neural Network (N1, N2, ...) → prediction (task output)
```

3. Task ka feedback (loss) aata hai, aur **backpropagation** poore path se hoti hai — sirf network weights (N1, N2) hi update nahi hote, balki embedding vector khud bhi update hota hai.
4. Repeat karte raho (gradient descent) — jitna sophisticated task aur jitna wide corpus, utne behtar embeddings emerge karte hain.

**Note on softmax (context ka reference):** Jab bhi hum "context ko dekhna" discuss karte hain, professor ne softmax ka comparison diya — sigmoid sirf apne khud ke input pe depend karta hai, lekin softmax teeno (ya sabhi) values ko dekh ke decide karta hai:

$$
\text{softmax}(z_1) = \frac{e^{z_1}}{e^{z_1} + e^{z_2} + e^{z_3}}
$$

Yehi philosophy contextual embeddings me bhi lagti hai — ek word ka vector sirf usi word pe nahi, balki **surrounding words (context)** pe bhi depend karta hai. Jitna zyada left-right dekh sakte ho, utna bada **context length**.

---

## 7. A Limitation: Contextual Embeddings (BERT vs GPT)

Classic embeddings **fixed** hote hain — "bank" ka ek hi vector hota hai, chahe uska matlab *river bank* ho ya *money bank*. Yeh problem hai.

| Model | Approach |
|-------|----------|
| **BERT** | Bidirectional — poora sentence padhta hai, left aur right dono taraf, embedding banane se pehle (jaise bidirectional LSTM/RNN). |
| **GPT** | Sirf left-to-right padhta hai — jo pehle aaya hai, usी tak se embedding banata hai. |

**Notation (jaisa lecture me use hua):**

```
bank (river context)  → embedding E1
bank (money context)  → embedding E2
```

Agar bank ka context "money" (E3) ho, to uska embedding E2 nahi balki E2′ (E2-prime) hoga — matlab **same word, different sentence → different vector**. Yehi contextual embedding ka core idea hai, aur yeh sabhi modern LLMs (BERT, GPT) me hota hai.

Result: "bank" ka embedding surrounding words (context) ke hisaab se change hota hai.

---

## 8. Recap

- Words ko **vectors** banaya jaata hai, single numbers nahi, taaki meaning encode ho sake.
- Similar words → nearby vectors; distance ≈ semantic similarity.
- Bade text corpora se co-occurrence ke through automatically learn hote hain — kabhi hand-coded nahi hote.
- Emergent behavior: clustering aur analogy arithmetic (king − man + woman ≈ queen).
- Modern models (BERT, GPT) embeddings ko **contextual** banate hain — same word alag sentences me alag vector leta hai.

**Next in course:** Embeddings akele kaafi kyun nahi hain → **Attention mechanism** → **Transformers**.
