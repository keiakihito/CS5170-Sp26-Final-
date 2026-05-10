from dataclasses import dataclass
from enum import Enum
from typing import List

import torch
import torch.nn.functional as F


class DIGCategory(Enum):
    POSITIVE = "positive"      # DIG > b1  — helpful document
    NEGLIGIBLE = "negligible"  # |DIG| ≈ 0 — no meaningful contribution
    NEGATIVE = "negative"      # DIG < b2  — misleading document


@dataclass
class DIGResult:
    query: str
    answer: str
    document: str
    doc_id: str
    confidence_with_doc: float
    confidence_without_doc: float
    dig_score: float
    category: DIGCategory


class DIGScorer:
    """Step 2: Compute Document Information Gain scores for <query, answer, doc> triplets.

    Implements equations 1-3 from the paper:
      Eq 1 — Sliding window smoothing over token probabilities
      Eq 2 — Token importance weighting (first k tokens weighted higher)
      Eq 3 — DIG = p_φ(y|x,d) − p_φ(y|x)
    """

    # Paper hyperparameters (Section 4.1)
    WINDOW_SIZE = 3       # W — sliding window width
    TOP_K_TOKENS = 3      # k — number of initial tokens with higher weight
    ALPHA = 0.6           # α — weight hyper-parameter
    OMEGA = 0.8           # ω_i — importance weight for first k tokens

    # DIG thresholds (Section 3.1.2)
    B1 = 0.5              # positive boundary
    B2 = -0.2             # negative boundary

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    # -- input construction --

    def _build_input(self, query: str, document: str | None) -> str:
        if document:
            return f"{document}\n\n{query}"
        return query

    # -- model execution --

    def _tokenize(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _run_model(self, text: str) -> torch.Tensor:
        inputs = self._tokenize(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits[0]  # (seq_len, vocab_size)

    def _raw_token_probs(self, logits: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
        probs = []
        for i, tid in enumerate(token_ids):
            pos = min(i, logits.shape[0] - 1)
            prob = F.softmax(logits[pos], dim=-1)[tid]
            probs.append(prob)
        return torch.stack(probs)  # (seq_len,)

    def _get_token_probs(self, query: str, answer: str, document: str | None = None) -> torch.Tensor:
        """Run LLM and return raw per-token probabilities for the answer tokens."""
        text = self._build_input(query, document)
        token_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        logits = self._run_model(text)
        return self._raw_token_probs(logits, token_ids)

    # -- eq 1: sliding window smoothing --

    def _window_slice(self, probs: torch.Tensor, i: int) -> torch.Tensor:
        half = self.WINDOW_SIZE // 2
        lo = max(0, i - half)
        hi = min(len(probs), i + half + 1)
        return probs[lo:hi]

    def _sliding_window_smooth(self, token_probs: torch.Tensor) -> torch.Tensor:
        """Apply eq 1: smooth each token prob with a window of size W."""
        smoothed = [self._window_slice(token_probs, i).mean() for i in range(len(token_probs))]
        return torch.stack(smoothed)

    # -- eq 2: token importance weighting --

    def _early_token_term(self, probs: torch.Tensor, k: int) -> torch.Tensor:
        weights = torch.full((k,), self.OMEGA).pow(self.ALPHA)
        return (probs[:k] * weights).prod()

    def _late_token_term(self, probs: torch.Tensor, k: int) -> torch.Tensor:
        if len(probs) <= k:
            return torch.tensor(1.0)
        return probs[k:].pow(1 - self.ALPHA).prod()

    def _token_importance_weight(self, smoothed_probs: torch.Tensor) -> float:
        """Apply eq 2: weight first k tokens by ω^α, rest by (1-α), return scalar confidence."""
        k = min(self.TOP_K_TOKENS, len(smoothed_probs))
        score = self._early_token_term(smoothed_probs, k) * self._late_token_term(smoothed_probs, k)
        return float(score.clamp(0.0, 1.0))

    # -- public API --

    def _is_valid_answer(self, answer: str) -> bool:
        return bool(answer.strip()) and bool(self.tokenizer.encode(answer, add_special_tokens=False))

    def _confidence_pipeline(self, query: str, answer: str, document: str | None) -> float:
        token_probs = self._get_token_probs(query, answer, document)
        smoothed = self._sliding_window_smooth(token_probs)
        return self._token_importance_weight(smoothed)

    def compute_confidence(self, query: str, answer: str, document: str | None = None) -> float:
        """Compute p_φ(y|x) or p_φ(y|x,d) depending on whether document is provided."""
        if not self._is_valid_answer(answer):
            return 0.0
        return self._confidence_pipeline(query, answer, document)

    def _categorize(self, dig_score: float) -> DIGCategory:
        """Assign POSITIVE / NEGLIGIBLE / NEGATIVE label based on DIG thresholds."""
        if dig_score >= self.B1:
            return DIGCategory.POSITIVE
        if dig_score <= self.B2:
            return DIGCategory.NEGATIVE
        return DIGCategory.NEGLIGIBLE

    def _build_dig_result(
        self, query: str, answer: str, document: str, doc_id: str,
        conf_with: float, conf_without: float,
    ) -> DIGResult:
        dig_score = conf_with - conf_without
        return DIGResult(
            query=query,
            answer=answer,
            document=document,
            doc_id=doc_id,
            confidence_with_doc=conf_with,
            confidence_without_doc=conf_without,
            dig_score=dig_score,
            category=self._categorize(dig_score),
        )

    def compute_dig(self, query: str, answer: str, document: str, doc_id: str) -> DIGResult:
        """Compute DIG score for one <query, answer, doc> triplet (eq 3)."""
        conf_with = self.compute_confidence(query, answer, document=document)
        conf_without = self.compute_confidence(query, answer, document=None)
        return self._build_dig_result(query, answer, document, doc_id, conf_with, conf_without)

    def score_triplets(self, triplets: list) -> List[DIGResult]:
        """Score a list of Triplet objects and return DIGResult for each."""
        return [self.compute_dig(t.query, t.answer, t.document, t.doc_id) for t in triplets]
