from unittest.mock import MagicMock

import pytest
import torch

from src.dig_scorer import DIGCategory, DIGResult, DIGScorer
from src.preprocessor import Triplet


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_tokenizer(answer_token_ids: list[int]) -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.ones(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    tokenizer.encode.side_effect = lambda text, **kwargs: answer_token_ids
    return tokenizer


def _make_logits(answer_token_ids: list[int], high: bool) -> torch.Tensor:
    vocab_size = max(answer_token_ids) + 50
    seq_len = len(answer_token_ids)
    if high:
        logits = torch.full((1, seq_len, vocab_size), -100.0)
        for pos, tid in enumerate(answer_token_ids):
            logits[0, pos, tid] = 10.0
    else:
        logits = torch.zeros((1, seq_len, vocab_size))
    return logits


def _make_mock_model(logits: torch.Tensor) -> MagicMock:
    output = MagicMock()
    output.logits = logits
    model = MagicMock()
    model.device = torch.device("cpu")
    model.return_value = output
    return model


def make_scorer(answer_token_ids: list[int] = [10, 11, 12], high: bool = True) -> DIGScorer:
    tokenizer = _make_mock_tokenizer(answer_token_ids)
    logits = _make_logits(answer_token_ids, high)
    model = _make_mock_model(logits)
    return DIGScorer(model, tokenizer, device="cpu")


def _logits_for_conf(c: float, answer_token_ids: list[int]) -> torch.Tensor:
    vocab_size = max(answer_token_ids) + 50
    seq_len = len(answer_token_ids)
    if c > 0.5:
        logits = torch.full((1, seq_len, vocab_size), -100.0)
        for pos, tid in enumerate(answer_token_ids):
            logits[0, pos, tid] = 10.0
    else:
        logits = torch.zeros((1, seq_len, vocab_size))
    return logits


def _make_alternating_side_effect(confs: list[float], answer_token_ids: list[int]):
    call_count = {"n": 0}

    def side_effect(**kwargs):
        idx = call_count["n"] % len(confs)
        call_count["n"] += 1
        out = MagicMock()
        out.logits = _logits_for_conf(confs[idx], answer_token_ids)
        return out

    return side_effect


def make_varying_scorer(conf_with: float, conf_without: float) -> DIGScorer:
    """Scorer whose model returns different logits on successive calls."""
    answer_token_ids = [10, 11, 12]
    tokenizer = _make_mock_tokenizer(answer_token_ids)
    model = MagicMock()
    model.device = torch.device("cpu")
    model.side_effect = _make_alternating_side_effect([conf_with, conf_without], answer_token_ids)
    return DIGScorer(model, tokenizer, device="cpu")


SAMPLE_TRIPLET = Triplet(
    query="What is the capital of France?",
    answer="Paris",
    document="France is a country in Europe. Its capital is Paris.",
    doc_id="doc_001",
)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_stores_model_tokenizer_device(self):
        # Guarantee: constructor stores model, tokenizer, and device attributes
        scorer = make_scorer()
        assert scorer.model is not None
        assert scorer.tokenizer is not None
        assert scorer.device == "cpu"

    def test_hyperparams_set_to_paper_values(self):
        # Guarantee: default hyperparameters match paper Section 4.1 values
        scorer = make_scorer()
        assert scorer.WINDOW_SIZE == 3
        assert scorer.TOP_K_TOKENS == 3
        assert scorer.ALPHA == 0.6
        assert scorer.OMEGA == 0.8
        assert scorer.B1 == 0.5
        assert scorer.B2 == -0.2


# ---------------------------------------------------------------------------
# _sliding_window_smooth  (eq 1)
# ---------------------------------------------------------------------------

class TestSlidingWindowSmooth:
    def test_output_length_matches_input(self):
        # Guarantee: smoothed tensor has same length as input token sequence
        scorer = make_scorer()
        probs = torch.tensor([0.1, 0.5, 0.9, 0.3, 0.7])
        smoothed = scorer._sliding_window_smooth(probs)
        assert len(smoothed) == len(probs)

    def test_values_in_zero_one(self):
        # Guarantee: all smoothed probabilities remain in [0, 1]
        scorer = make_scorer()
        probs = torch.rand(10)
        smoothed = scorer._sliding_window_smooth(probs)
        assert (smoothed >= 0).all() and (smoothed <= 1).all()

    def test_uniform_input_unchanged(self):
        # Guarantee: smoothing a uniform sequence leaves values unchanged
        scorer = make_scorer()
        probs = torch.full((6,), 0.5)
        smoothed = scorer._sliding_window_smooth(probs)
        assert torch.allclose(smoothed, probs, atol=1e-5)

    def test_window_averages_neighbours(self):
        # Guarantee: each output is the mean of the W surrounding input values
        scorer = make_scorer()
        probs = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        smoothed = scorer._sliding_window_smooth(probs)
        # With W=3, position 2 (the spike) should average positions 1,2,3 → 1/3
        assert abs(smoothed[2].item() - 1.0 / scorer.WINDOW_SIZE) < 1e-4


# ---------------------------------------------------------------------------
# _token_importance_weight  (eq 2)
# ---------------------------------------------------------------------------

class TestTokenImportanceWeight:
    def test_returns_float(self):
        # Guarantee: token importance weighting always returns a plain Python float
        scorer = make_scorer()
        probs = torch.tensor([0.8, 0.7, 0.9, 0.5, 0.4])
        result = scorer._token_importance_weight(probs)
        assert isinstance(result, float)

    def test_result_in_zero_one(self):
        # Guarantee: weighted confidence is a valid probability in [0, 1]
        scorer = make_scorer()
        probs = torch.rand(5)
        result = scorer._token_importance_weight(probs)
        assert 0.0 <= result <= 1.0

    def test_high_early_tokens_boost_confidence(self):
        # Guarantee: sequence with high early-token probs scores higher than high late-token probs
        scorer = make_scorer()
        early_high = torch.tensor([0.9, 0.9, 0.9, 0.1, 0.1])
        late_high  = torch.tensor([0.1, 0.1, 0.1, 0.9, 0.9])
        assert scorer._token_importance_weight(early_high) > scorer._token_importance_weight(late_high)

    def test_single_token_answer(self):
        # Guarantee: a single-token answer does not raise and returns a valid float
        scorer = make_scorer()
        probs = torch.tensor([0.75])
        result = scorer._token_importance_weight(probs)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# compute_confidence
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    def test_returns_float(self):
        # Guarantee: compute_confidence always returns a plain Python float
        scorer = make_scorer()
        conf = scorer.compute_confidence("What is AI?", "intelligence")
        assert isinstance(conf, float)

    def test_in_zero_one(self):
        # Guarantee: confidence is a valid probability in [0, 1]
        scorer = make_scorer()
        conf = scorer.compute_confidence("What is AI?", "intelligence")
        assert 0.0 <= conf <= 1.0

    def test_high_logits_give_high_confidence(self):
        # Guarantee: strongly favoured answer tokens yield confidence > 0.5
        scorer = make_scorer(high=True)
        conf = scorer.compute_confidence("What is AI?", "intelligence")
        assert conf > 0.5

    def test_uniform_logits_give_low_confidence(self):
        # Guarantee: uniform logits (no advantage) yield confidence < 0.5
        scorer = make_scorer(high=False)
        conf = scorer.compute_confidence("What is AI?", "intelligence")
        assert conf < 0.5

    def test_accepts_document_argument(self):
        # Guarantee: compute_confidence accepts an optional document without error
        scorer = make_scorer()
        conf = scorer.compute_confidence("What is AI?", "intelligence", document="Some doc text.")
        assert isinstance(conf, float)

    def test_empty_answer_returns_zero(self):
        # Guarantee: empty answer string yields confidence 0.0
        scorer = make_scorer()
        assert scorer.compute_confidence("What is AI?", "") == 0.0


# ---------------------------------------------------------------------------
# _categorize
# ---------------------------------------------------------------------------

class TestCategorize:
    def test_above_b1_is_positive(self):
        # Guarantee: DIG score above B1 is labelled POSITIVE
        scorer = make_scorer()
        assert scorer._categorize(0.6) == DIGCategory.POSITIVE

    def test_below_b2_is_negative(self):
        # Guarantee: DIG score below B2 is labelled NEGATIVE
        scorer = make_scorer()
        assert scorer._categorize(-0.3) == DIGCategory.NEGATIVE

    def test_near_zero_is_negligible(self):
        # Guarantee: DIG score near zero (between B2 and B1) is labelled NEGLIGIBLE
        scorer = make_scorer()
        assert scorer._categorize(0.0) == DIGCategory.NEGLIGIBLE

    def test_exactly_b1_is_positive(self):
        # Guarantee: DIG score exactly equal to B1 boundary is labelled POSITIVE
        scorer = make_scorer()
        assert scorer._categorize(0.5) == DIGCategory.POSITIVE

    def test_exactly_b2_is_negative(self):
        # Guarantee: DIG score exactly equal to B2 boundary is labelled NEGATIVE
        scorer = make_scorer()
        assert scorer._categorize(-0.2) == DIGCategory.NEGATIVE


# ---------------------------------------------------------------------------
# compute_dig  (eq 3)
# ---------------------------------------------------------------------------

class TestComputeDIG:
    def test_returns_dig_result(self):
        # Guarantee: compute_dig returns a DIGResult dataclass
        scorer = make_scorer()
        result = scorer.compute_dig(
            SAMPLE_TRIPLET.query, SAMPLE_TRIPLET.answer,
            SAMPLE_TRIPLET.document, SAMPLE_TRIPLET.doc_id,
        )
        assert isinstance(result, DIGResult)

    def test_dig_score_equals_diff_of_confidences(self):
        # Guarantee: dig_score = confidence_with_doc − confidence_without_doc (eq 3)
        scorer = make_scorer()
        result = scorer.compute_dig(
            SAMPLE_TRIPLET.query, SAMPLE_TRIPLET.answer,
            SAMPLE_TRIPLET.document, SAMPLE_TRIPLET.doc_id,
        )
        expected = result.confidence_with_doc - result.confidence_without_doc
        assert abs(result.dig_score - expected) < 1e-6

    def test_preserves_query_answer_doc_id(self):
        # Guarantee: DIGResult carries back original query, answer, and doc_id
        scorer = make_scorer()
        result = scorer.compute_dig(
            SAMPLE_TRIPLET.query, SAMPLE_TRIPLET.answer,
            SAMPLE_TRIPLET.document, SAMPLE_TRIPLET.doc_id,
        )
        assert result.query == SAMPLE_TRIPLET.query
        assert result.answer == SAMPLE_TRIPLET.answer
        assert result.doc_id == SAMPLE_TRIPLET.doc_id

    def test_category_consistent_with_dig_score(self):
        # Guarantee: category label matches what _categorize would assign for the dig_score
        scorer = make_scorer()
        result = scorer.compute_dig(
            SAMPLE_TRIPLET.query, SAMPLE_TRIPLET.answer,
            SAMPLE_TRIPLET.document, SAMPLE_TRIPLET.doc_id,
        )
        assert result.category == scorer._categorize(result.dig_score)


# ---------------------------------------------------------------------------
# score_triplets
# ---------------------------------------------------------------------------

class TestScoreTriplets:
    def test_returns_list_of_dig_results(self):
        # Guarantee: score_triplets returns a list of DIGResult objects
        scorer = make_scorer()
        results = scorer.score_triplets([SAMPLE_TRIPLET])
        assert isinstance(results, list)
        assert all(isinstance(r, DIGResult) for r in results)

    def test_output_length_matches_input(self):
        # Guarantee: one DIGResult is returned per input triplet
        scorer = make_scorer()
        triplets = [SAMPLE_TRIPLET, SAMPLE_TRIPLET]
        results = scorer.score_triplets(triplets)
        assert len(results) == 2

    def test_empty_input_returns_empty_list(self):
        # Guarantee: empty triplet list returns empty list without error
        scorer = make_scorer()
        assert scorer.score_triplets([]) == []
