from unittest.mock import MagicMock

import pytest
import torch

from src.inference import (
    InferencePipeline,
    InferenceResult,
    RankedDocument,
    evaluate_exact_match,
    exact_match,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_retriever(docs: list[dict]) -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve.side_effect = lambda query, top_k=100: docs[:top_k]
    return retriever


def _make_sequential_score_side_effect(scores: list[float]):
    """Return a side-effect function that consumes `scores` sequentially across calls."""
    call_count = {"n": 0}

    def side_effect(input_ids, attention_mask):
        n = input_ids.shape[0]
        start = call_count["n"]
        call_count["n"] += n
        return torch.tensor(scores[start : start + n])

    return side_effect


def _make_mock_reranker(scores: list[float]) -> MagicMock:
    """Reranker whose forward() returns fixed scores in order."""
    reranker = MagicMock()
    reranker.side_effect = _make_sequential_score_side_effect(scores)
    reranker.eval = MagicMock()
    return reranker


def _make_mock_generator(answer: str = "Paris") -> MagicMock:
    generator = MagicMock()
    output_ids = torch.tensor([[1, 2, 3]])
    generator.generate.return_value = output_ids
    return generator


def _make_mock_generator_tokenizer(answer: str = "Paris") -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.ones(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }
    tokenizer.decode.return_value = answer
    return tokenizer


def _make_mock_reranker_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.ones(1, 8, dtype=torch.long),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    }
    return tokenizer


SAMPLE_DOCS = [
    {"doc_id": f"d{i}", "text": f"passage {i}", "score": 1.0 - i * 0.1}
    for i in range(10)
]

SAMPLE_SCORES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1, 0.05, 0.01]


def make_pipeline(
    docs: list[dict] = SAMPLE_DOCS,
    scores: list[float] = SAMPLE_SCORES,
    answer: str = "Paris",
) -> InferencePipeline:
    retriever = _make_mock_retriever(docs)
    reranker = _make_mock_reranker(scores)
    reranker_tokenizer = _make_mock_reranker_tokenizer()
    generator = _make_mock_generator(answer)
    gen_tokenizer = _make_mock_generator_tokenizer(answer)
    return InferencePipeline(
        retriever=retriever,
        reranker=reranker,
        reranker_tokenizer=reranker_tokenizer,
        generator=generator,
        generator_tokenizer=gen_tokenizer,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# InferencePipeline — init
# ---------------------------------------------------------------------------

class TestPipelineInit:
    def test_stores_components(self):
        # Guarantee: pipeline stores all five components passed at construction
        p = make_pipeline()
        assert p.retriever is not None
        assert p.reranker is not None
        assert p.generator is not None
        assert p.generator_tokenizer is not None

    def test_hyperparams_match_paper(self):
        # Guarantee: default thresholds match paper Section 4.1 inference settings
        p = make_pipeline()
        assert p.FILTER_THRESHOLD == 0.2
        assert p.TOP_K_DOCS == 4
        assert p.MIN_DOCS == 2


# ---------------------------------------------------------------------------
# _score_documents
# ---------------------------------------------------------------------------

class TestScoreDocuments:
    def test_returns_ranked_document_list(self):
        # Guarantee: _score_documents returns a list of RankedDocument objects
        p = make_pipeline()
        result = p._score_documents("What is AI?", SAMPLE_DOCS[:3])
        assert isinstance(result, list)
        assert all(isinstance(d, RankedDocument) for d in result)

    def test_length_matches_input(self):
        # Guarantee: one RankedDocument is returned per input document
        p = make_pipeline()
        result = p._score_documents("What is AI?", SAMPLE_DOCS[:4])
        assert len(result) == 4

    def test_preserves_doc_id_and_text(self):
        # Guarantee: doc_id and text from the input doc are carried into RankedDocument
        p = make_pipeline()
        result = p._score_documents("What is AI?", SAMPLE_DOCS[:1])
        assert result[0].doc_id == SAMPLE_DOCS[0]["doc_id"]
        assert result[0].text == SAMPLE_DOCS[0]["text"]

    def test_score_is_float(self):
        # Guarantee: each RankedDocument has a plain Python float score
        p = make_pipeline()
        result = p._score_documents("What is AI?", SAMPLE_DOCS[:2])
        assert all(isinstance(d.score, float) for d in result)


# ---------------------------------------------------------------------------
# _rank_documents
# ---------------------------------------------------------------------------

class TestRankDocuments:
    def test_returns_sorted_descending(self):
        # Guarantee: ranked list is ordered highest score first
        p = make_pipeline()
        docs = [
            RankedDocument("d1", "t1", 0.3),
            RankedDocument("d2", "t2", 0.9),
            RankedDocument("d3", "t3", 0.1),
        ]
        ranked = p._rank_documents(docs)
        scores = [d.score for d in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_length_unchanged(self):
        # Guarantee: ranking does not drop or duplicate documents
        p = make_pipeline()
        docs = [RankedDocument(f"d{i}", f"t{i}", float(i)) for i in range(5)]
        assert len(p._rank_documents(docs)) == 5

    def test_empty_input_returns_empty(self):
        # Guarantee: ranking an empty list returns an empty list without error
        p = make_pipeline()
        assert p._rank_documents([]) == []


# ---------------------------------------------------------------------------
# _filter_documents
# ---------------------------------------------------------------------------

class TestFilterDocuments:
    def test_removes_below_threshold(self):
        # Guarantee: documents with score below FILTER_THRESHOLD are excluded
        p = make_pipeline()
        docs = [
            RankedDocument("d1", "t1", 0.9),
            RankedDocument("d2", "t2", 0.5),
            RankedDocument("d3", "t3", 0.05),  # below 0.2
        ]
        result = p._filter_documents(docs)
        assert all(d.score >= p.FILTER_THRESHOLD for d in result)

    def test_respects_top_k(self):
        # Guarantee: at most TOP_K_DOCS documents are returned
        p = make_pipeline()
        docs = [RankedDocument(f"d{i}", f"t{i}", 0.9 - i * 0.01) for i in range(20)]
        result = p._filter_documents(docs)
        assert len(result) <= p.TOP_K_DOCS

    def test_retains_min_docs_when_all_below_threshold(self):
        # Guarantee: at least MIN_DOCS documents are kept even if all scores are below threshold
        p = make_pipeline()
        docs = [RankedDocument(f"d{i}", f"t{i}", 0.01) for i in range(5)]
        result = p._filter_documents(docs)
        assert len(result) >= p.MIN_DOCS

    def test_retains_min_docs_when_only_one_passes(self):
        # Guarantee: MIN_DOCS is enforced when only one document passes the threshold
        p = make_pipeline()
        docs = [
            RankedDocument("d1", "t1", 0.9),   # passes
            RankedDocument("d2", "t2", 0.01),  # below threshold
            RankedDocument("d3", "t3", 0.01),
        ]
        result = p._filter_documents(docs)
        assert len(result) >= p.MIN_DOCS

    def test_empty_input_returns_empty(self):
        # Guarantee: filtering an empty list returns an empty list without error
        p = make_pipeline()
        assert p._filter_documents([]) == []


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_returns_string(self):
        # Guarantee: _build_prompt always returns a string
        p = make_pipeline()
        docs = [RankedDocument("d1", "Some passage.", 0.9)]
        assert isinstance(p._build_prompt("What is AI?", docs), str)

    def test_prompt_contains_query(self):
        # Guarantee: the query appears somewhere in the constructed prompt
        p = make_pipeline()
        docs = [RankedDocument("d1", "Some passage.", 0.9)]
        prompt = p._build_prompt("What is AI?", docs)
        assert "What is AI?" in prompt

    def test_prompt_contains_document_text(self):
        # Guarantee: each selected document's text appears in the prompt
        p = make_pipeline()
        docs = [RankedDocument("d1", "Unique passage text.", 0.9)]
        prompt = p._build_prompt("What is AI?", docs)
        assert "Unique passage text." in prompt

    def test_multiple_docs_all_included(self):
        # Guarantee: all selected documents are included in the prompt
        p = make_pipeline()
        docs = [RankedDocument(f"d{i}", f"passage_{i}", 0.9) for i in range(3)]
        prompt = p._build_prompt("query", docs)
        for d in docs:
            assert d.text in prompt


# ---------------------------------------------------------------------------
# _generate_answer
# ---------------------------------------------------------------------------

class TestGenerateAnswer:
    def test_returns_string(self):
        # Guarantee: _generate_answer always returns a string
        p = make_pipeline(answer="Paris")
        assert isinstance(p._generate_answer("some prompt"), str)

    def test_returns_expected_answer(self):
        # Guarantee: decoded generator output is returned as the answer
        p = make_pipeline(answer="Paris")
        assert p._generate_answer("What is the capital of France?") == "Paris"


# ---------------------------------------------------------------------------
# run  (full pipeline)
# ---------------------------------------------------------------------------

class TestRun:
    def test_returns_inference_result(self):
        # Guarantee: run() always returns an InferenceResult dataclass
        p = make_pipeline()
        result = p.run("What is the capital of France?")
        assert isinstance(result, InferenceResult)

    def test_result_preserves_query(self):
        # Guarantee: InferenceResult carries back the original query string
        p = make_pipeline()
        result = p.run("What is the capital of France?")
        assert result.query == "What is the capital of France?"

    def test_answer_is_string(self):
        # Guarantee: the generated answer in the result is a non-empty string
        p = make_pipeline(answer="Paris")
        result = p.run("What is the capital of France?")
        assert isinstance(result.answer, str) and result.answer

    def test_selected_docs_within_top_k(self):
        # Guarantee: no more than TOP_K_DOCS documents are passed to the generator
        p = make_pipeline()
        result = p.run("What is the capital of France?")
        assert len(result.selected_docs) <= p.TOP_K_DOCS

    def test_selected_docs_are_subset_of_ranked(self):
        # Guarantee: selected_docs are drawn from the full ranked list
        p = make_pipeline()
        result = p.run("What is the capital of France?")
        ranked_ids = {d.doc_id for d in result.ranked_docs}
        for d in result.selected_docs:
            assert d.doc_id in ranked_ids


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_identical_strings_match(self):
        # Guarantee: identical prediction and ground truth returns True
        assert exact_match("Paris", "Paris") is True

    def test_different_strings_no_match(self):
        # Guarantee: different prediction and ground truth returns False
        assert exact_match("London", "Paris") is False

    def test_case_insensitive(self):
        # Guarantee: match is case-insensitive (normalised before comparison)
        assert exact_match("paris", "Paris") is True

    def test_whitespace_normalised(self):
        # Guarantee: leading/trailing whitespace is stripped before comparison
        assert exact_match("  Paris  ", "Paris") is True

    def test_punctuation_stripped(self):
        # Guarantee: trailing punctuation is stripped before comparison
        assert exact_match("Paris.", "Paris") is True


# ---------------------------------------------------------------------------
# evaluate_exact_match
# ---------------------------------------------------------------------------

class TestEvaluateExactMatch:
    def test_all_correct_returns_one(self):
        # Guarantee: all correct predictions give EM score of 1.0
        assert evaluate_exact_match(["Paris", "Rome"], ["Paris", "Rome"]) == 1.0

    def test_all_wrong_returns_zero(self):
        # Guarantee: all wrong predictions give EM score of 0.0
        assert evaluate_exact_match(["London", "Berlin"], ["Paris", "Rome"]) == 0.0

    def test_half_correct_returns_half(self):
        # Guarantee: half correct predictions give EM score of 0.5
        assert evaluate_exact_match(["Paris", "Berlin"], ["Paris", "Rome"]) == 0.5

    def test_empty_lists_returns_zero(self):
        # Guarantee: empty input lists return 0.0 without error
        assert evaluate_exact_match([], []) == 0.0

    def test_mismatched_lengths_raises(self):
        # Guarantee: mismatched prediction/ground-truth list lengths raise ValueError
        with pytest.raises(ValueError):
            evaluate_exact_match(["Paris"], ["Paris", "Rome"])
