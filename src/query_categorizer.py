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

    # Queries with confidence above this threshold are considered proficient
    CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def compute_confidence(self, query: str, answer: str) -> float:
        """Compute p_φ(y|x) — generation confidence without any document."""
        if not answer.strip():
            return 0.0

        answer_token_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        if not answer_token_ids:
            return 0.0

        inputs = self.tokenizer(query, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # outputs.logits shape: (1, seq_len, vocab_size) — use last position per token
        logits = outputs.logits[0]  # (seq_len, vocab_size)

        # Score each answer token using the logit at the corresponding position
        log_probs = []
        for i, token_id in enumerate(answer_token_ids):
            pos = min(i, logits.shape[0] - 1)
            token_log_prob = F.log_softmax(logits[pos], dim=-1)[token_id].item()
            log_probs.append(token_log_prob)

        # Average log-prob, then map to [0, 1] via exp
        avg_log_prob = sum(log_probs) / len(log_probs)
        return float(torch.exp(torch.tensor(avg_log_prob)).clamp(0.0, 1.0))

    def categorize(self, query: str, answer: str) -> QueryResult:
        """Return QueryResult with confidence score and proficient/challenging label."""
        confidence = self.compute_confidence(query, answer)
        query_type = (
            QueryType.PROFICIENT
            if confidence >= self.CONFIDENCE_THRESHOLD
            else QueryType.CHALLENGING
        )
        return QueryResult(
            query=query,
            answer=answer,
            confidence=confidence,
            query_type=query_type,
        )

    def categorize_batch(self, queries: List[str], answers: List[str]) -> List[QueryResult]:
        """Categorize a list of (query, answer) pairs."""
        if len(queries) != len(answers):
            raise ValueError(
                f"queries and answers must have the same length, "
                f"got {len(queries)} and {len(answers)}"
            )
        return [self.categorize(q, a) for q, a in zip(queries, answers)]
