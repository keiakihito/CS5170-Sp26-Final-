from dataclasses import dataclass
from enum import Enum
from typing import List

import torch
import torch.nn.functional as F


class QueryType(Enum):
    PROFICIENT = "proficient"
    CHALLENGING = "challenging"


@dataclass
class QueryResult:
    query: str
    answer: str
    confidence: float
    query_type: QueryType


class QueryCategorizer:
    """Step 1: Categorize queries by LLM baseline confidence (no documents)."""

    CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    # -- confidence helpers --

    def _tokenize_answer(self, answer: str) -> list[int]:
        return self.tokenizer.encode(answer, add_special_tokens=False)

    def _run_model(self, query: str) -> torch.Tensor:
        inputs = self.tokenizer(query, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits[0]  # (seq_len, vocab_size)

    def _token_log_probs(self, logits: torch.Tensor, token_ids: list[int]) -> list[float]:
        log_probs = []
        for i, token_id in enumerate(token_ids):
            pos = min(i, logits.shape[0] - 1)
            log_prob = F.log_softmax(logits[pos], dim=-1)[token_id].item()
            log_probs.append(log_prob)
        return log_probs

    def _avg_log_prob_to_confidence(self, log_probs: list[float]) -> float:
        avg = sum(log_probs) / len(log_probs)
        return float(torch.exp(torch.tensor(avg)).clamp(0.0, 1.0))

    # -- public API --

    def compute_confidence(self, query: str, answer: str) -> float:
        """Compute p_φ(y|x) — generation confidence without any document."""
        if not answer.strip():
            return 0.0
        token_ids = self._tokenize_answer(answer)
        if not token_ids:
            return 0.0
        logits = self._run_model(query)
        log_probs = self._token_log_probs(logits, token_ids)
        return self._avg_log_prob_to_confidence(log_probs)

    def _classify(self, confidence: float) -> QueryType:
        return QueryType.PROFICIENT if confidence >= self.CONFIDENCE_THRESHOLD else QueryType.CHALLENGING

    def categorize(self, query: str, answer: str) -> QueryResult:
        """Return QueryResult with confidence score and proficient/challenging label."""
        confidence = self.compute_confidence(query, answer)
        return QueryResult(
            query=query,
            answer=answer,
            confidence=confidence,
            query_type=self._classify(confidence),
        )

    def categorize_batch(self, queries: List[str], answers: List[str]) -> List[QueryResult]:
        """Categorize a list of (query, answer) pairs."""
        if len(queries) != len(answers):
            raise ValueError(
                f"queries and answers must have the same length, "
                f"got {len(queries)} and {len(answers)}"
            )
        return [self.categorize(q, a) for q, a in zip(queries, answers)]
