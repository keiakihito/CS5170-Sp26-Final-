# InfoGain-RAG Replication Plan

**Paper:** InfoGain-RAG: Boosting Retrieval-Augmented Generation through Document Information Gain-based Reranking and Filtering (Wang et al., EMNLP 2025)

---

## Overview

We replicate the InfoGain-RAG framework, which improves RAG by scoring each retrieved document's contribution to correct answer generation (DIG score), then training a lightweight reranker on those scores.

---

## Input [x]

| Component | Details |
|-----------|---------|
| **Dataset** | TriviaQA (110K queries sampled for training; also evaluated on NaturalQA, PopQA, FM2) |
| **Retrieval Corpus** | December 2018 Wikipedia dump |
| **Retriever** | Contriever (retrieves top-100 candidate documents per query) |
| **Generator LLM** | Qwen2.5-7B — used to compute DIG scores and generate final answers |
| **Reranker Base Model** | RoBERTa-large (355M params) — fine-tuned as the reranker |

---

## Process

### Step 0 — Preprocessing [x]
- Load TriviaQA (110K queries); remove train-test overlap
- For each query, run Contriever against the Wikipedia dump to retrieve top-100 candidate documents
- Form `<query, answer, document>` triplets — up to 100 triplets per query
- Output: triplet dataset ready for DIG scoring in Step 2

### Step 1 — Query Categorization []
- Run Qwen2.5-7B on each query **without** any retrieved documents
- Compute baseline generation confidence `p_φ(y|x)`
- Classify queries as:
  - **Proficient**: LLM answers correctly on its own (high confidence)
  - **Challenging**: LLM struggles without external documents (low confidence)

### Step 2 — DIG Score Computation (Data Collection) []
- For each query, retrieve top-k candidate documents via Contriever
- For each `(query, document)` pair, compute confidence **with** the document: `p_φ(y|x, d)`
- Compute DIG score: `DIG(d|x) = p_φ(y|x, d) − p_φ(y|x)`
- Confidence is estimated using:
  - **Sliding Window Smoothing** (equation 1) — mitigates length bias
  - **Token Importance Weighting** (equation 2) — up-weights first k=3 tokens
- Categorize documents:
  - `DIG > 0.5` → positive (helpful)
  - `DIG < −0.2` → negative (misleading)
  - `−0.05 ~ 0.05` → negligible
- Build training dataset of ~88K `(query, document, DIG score)` triplets

### Step 3 — Multi-task Reranker Training []
- Fine-tune RoBERTa-large on the DIG dataset using two combined losses:
  - **CE Loss** (equation 4): binary classification — helpful vs. noisy documents (68K balanced samples)
  - **Margin Loss** (equation 7): pairwise ranking — ensures negative docs score lower than positives (34K groups of 1 query + 3-5 positive + negative docs)
  - **Combined**: `L_total = β · L_CE + (1 − β) · L_Margin` with β=0.75
- Hyperparameters: lr=5e-6, γ=15, α=0.6, ω_i=0.8 for first k=3 tokens
- Hardware: A800 GPU with Adam optimizer

### Step 4 — Inference with InfoGain-RAG []
- New query → Contriever retrieves top-100 documents
- RoBERTa reranker scores and reorders all documents
- Filter: keep top-4 documents with score > 0.2 (retain at least 2 if fewer pass)
- Pass filtered, reranked documents to Qwen for final answer generation

---

## Output & Evaluation []

| Metric | Details |
|--------|---------|
| **Primary Metric** | Exact Match (EM) accuracy |
| **Evaluated On** | TriviaQA, NaturalQA, PopQA, FM2 |
| **Baselines to Compare** | Naive RAG, BGE-Reranker-Large, GTE-7B reranker |

---

## Component Summary

```
Training pipeline:
TriviaQA (110K queries)
  └─► Step 0: Contriever retrieves top-100 docs → <query, answer, doc> triplets
        └─► Step 1: Qwen2.5-7B baseline confidence → proficient / challenging labels
              └─► Step 2: Qwen2.5-7B DIG scoring → scored triplets (~88K samples)
                    └─► Step 3: RoBERTa-large fine-tuning (CE + Margin loss)

Inference:
Query → Contriever → RoBERTa reranker → filter (threshold 0.2) → Qwen2.5-7B → Answer
```

---

## Key Files / Checkpoints to Download

- `Qwen/Qwen2.5-7B` from Hugging Face
- `FacebookAI/roberta-large` from Hugging Face
- TriviaQA dataset (via HuggingFace `datasets` or official site)
- December 2018 Wikipedia dump (for retrieval corpus)
