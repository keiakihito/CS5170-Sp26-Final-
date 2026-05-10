import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.query_categorizer import QueryCategorizer, QueryResult, QueryType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_mock_model_tokenizer(answer_token_ids: list[int], logit_value: float = 5.0):
    """Return a (model, tokenizer) mock pair that produces controlled logits."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    tokenizer.encode.side_effect = lambda text, **kwargs: answer_token_ids
    tokenizer.decode.side_effect = lambda ids, **kwargs: "answer"

    # Build logits so that the answer tokens always have the highest probability
    vocab_size = max(answer_token_ids) + 10
    high_logit = torch.full((1, 1, vocab_size), -100.0)
    for tid in answer_token_ids:
        high_logit[0, 0, tid] = logit_value

    model = MagicMock()
    model.device = torch.device("cpu")
    output = MagicMock()
    output.logits = high_logit.expand(1, len(answer_token_ids), vocab_size)
    model.return_value = output

    return model, tokenizer


@pytest.fixture
def high_confidence_categorizer():
    # answer tokens [10, 11, 12] all get logit=10 → near-certain confidence
    model, tokenizer = make_mock_model_tokenizer([10, 11, 12], logit_value=10.0)
    return QueryCategorizer(model, tokenizer, device="cpu")


@pytest.fixture
def low_confidence_categorizer():
    # answer tokens [10, 11, 12] get logit=0.1 → very low confidence
    model, tokenizer = make_mock_model_tokenizer([10, 11, 12], logit_value=0.1)
    return QueryCategorizer(model, tokenizer, device="cpu")


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_stores_device(self, high_confidence_categorizer):
        # Guarantee: categorizer exposes the device it was constructed with
        assert high_confidence_categorizer.device == "cpu"

    def test_stores_model_and_tokenizer(self):
        # Guarantee: model and tokenizer passed at construction are retained
        model, tokenizer = make_mock_model_tokenizer([10])
        cat = QueryCategorizer(model, tokenizer, device="cpu")
        assert cat.model is model
        assert cat.tokenizer is tokenizer


# ---------------------------------------------------------------------------
# compute_confidence
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    def test_returns_float(self, high_confidence_categorizer):
        # Guarantee: confidence is always a plain Python float
        conf = high_confidence_categorizer.compute_confidence("What is AI?", "intelligence")
        assert isinstance(conf, float)

    def test_confidence_in_zero_one(self, high_confidence_categorizer):
        # Guarantee: confidence is a valid probability in [0, 1]
        conf = high_confidence_categorizer.compute_confidence("What is AI?", "intelligence")
        assert 0.0 <= conf <= 1.0

    def test_high_logit_gives_high_confidence(self, high_confidence_categorizer):
        # Guarantee: when the model strongly favours the answer tokens, confidence > 0.5
        conf = high_confidence_categorizer.compute_confidence("What is AI?", "intelligence")
        assert conf > 0.5

    def test_low_logit_gives_low_confidence(self, low_confidence_categorizer):
        # Guarantee: when the model is uncertain about the answer tokens, confidence < 0.5
        conf = low_confidence_categorizer.compute_confidence("What is AI?", "intelligence")
        assert conf < 0.5

    def test_empty_answer_returns_zero(self, high_confidence_categorizer):
        # Guarantee: an empty answer string yields confidence 0.0 (nothing to score)
        conf = high_confidence_categorizer.compute_confidence("What is AI?", "")
        assert conf == 0.0


# ---------------------------------------------------------------------------
# categorize
# ---------------------------------------------------------------------------

class TestCategorize:
    def test_returns_query_result(self, high_confidence_categorizer):
        # Guarantee: categorize always returns a QueryResult dataclass
        result = high_confidence_categorizer.categorize("What is AI?", "intelligence")
        assert isinstance(result, QueryResult)

    def test_result_preserves_query_and_answer(self, high_confidence_categorizer):
        # Guarantee: QueryResult carries back the original query and answer strings
        result = high_confidence_categorizer.categorize("What is AI?", "intelligence")
        assert result.query == "What is AI?"
        assert result.answer == "intelligence"

    def test_high_confidence_is_proficient(self, high_confidence_categorizer):
        # Guarantee: a query the LLM answers confidently is labelled PROFICIENT
        result = high_confidence_categorizer.categorize("What is AI?", "intelligence")
        assert result.query_type == QueryType.PROFICIENT

    def test_low_confidence_is_challenging(self, low_confidence_categorizer):
        # Guarantee: a query the LLM struggles with is labelled CHALLENGING
        result = low_confidence_categorizer.categorize("What is AI?", "intelligence")
        assert result.query_type == QueryType.CHALLENGING

    def test_confidence_stored_in_result(self, high_confidence_categorizer):
        # Guarantee: the confidence score used for classification is accessible on the result
        result = high_confidence_categorizer.categorize("What is AI?", "intelligence")
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# categorize_batch
# ---------------------------------------------------------------------------

class TestCategorizeBatch:
    def test_returns_list_of_query_results(self, high_confidence_categorizer):
        # Guarantee: batch output is a list of QueryResult objects, one per input pair
        results = high_confidence_categorizer.categorize_batch(
            ["Q1", "Q2"], ["A1", "A2"]
        )
        assert isinstance(results, list)
        assert all(isinstance(r, QueryResult) for r in results)

    def test_batch_length_matches_input(self, high_confidence_categorizer):
        # Guarantee: output list length equals the number of input queries
        results = high_confidence_categorizer.categorize_batch(
            ["Q1", "Q2", "Q3"], ["A1", "A2", "A3"]
        )
        assert len(results) == 3

    def test_empty_batch_returns_empty_list(self, high_confidence_categorizer):
        # Guarantee: an empty input batch returns an empty list without error
        results = high_confidence_categorizer.categorize_batch([], [])
        assert results == []

    def test_batch_mismatched_lengths_raises(self, high_confidence_categorizer):
        # Guarantee: mismatched queries/answers lengths raises ValueError immediately
        with pytest.raises(ValueError):
            high_confidence_categorizer.categorize_batch(["Q1", "Q2"], ["A1"])
