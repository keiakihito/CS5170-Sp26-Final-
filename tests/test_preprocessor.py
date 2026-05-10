import gzip
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.preprocessor import ContrieverRetriever, Preprocessor, Triplet, WikipediaCorpus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tsv_gz(rows: list[tuple[str, str, str]], path: str) -> None:
    """Write a minimal psgs_w100-style TSV.gz: id<TAB>text<TAB>title."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("id\ttext\ttitle\n")
        for doc_id, text, title in rows:
            f.write(f"{doc_id}\t{text}\t{title}\n")


def make_mock_retriever(docs: list[dict]) -> MagicMock:
    """Return a ContrieverRetriever mock whose retrieve() respects top_k."""
    retriever = MagicMock(spec=ContrieverRetriever)
    retriever.retrieve.side_effect = lambda query, top_k=100: docs[:top_k]
    return retriever


SAMPLE_DOCS = [
    {"doc_id": f"d{i}", "text": f"passage {i}", "score": 1.0 - i * 0.01}
    for i in range(5)
]

SAMPLE_QA = [
    {"query": "What is AI?", "answer": "intelligence"},
    {"query": "Who invented Python?", "answer": "Guido van Rossum"},
]


# ---------------------------------------------------------------------------
# WikipediaCorpus
# ---------------------------------------------------------------------------

class TestWikipediaCorpus:
    @pytest.fixture
    def corpus_path(self, tmp_path):
        p = str(tmp_path / "psgs_w100.tsv.gz")
        make_tsv_gz([("1", "passage one", "Title A"), ("2", "passage two", "Title B")], p)
        return p

    def test_len_matches_row_count(self, corpus_path):
        # Guarantee: __len__ returns the exact number of passages in the file
        corpus = WikipediaCorpus(corpus_path)
        assert len(corpus) == 2

    def test_get_passage_returns_text(self, corpus_path):
        # Guarantee: get_passage returns the text column for a known doc_id
        corpus = WikipediaCorpus(corpus_path)
        assert corpus.get_passage("1") == "passage one"

    def test_get_passage_unknown_id_raises(self, corpus_path):
        # Guarantee: requesting a non-existent doc_id raises KeyError
        corpus = WikipediaCorpus(corpus_path)
        with pytest.raises(KeyError):
            corpus.get_passage("999")

    def test_loads_all_rows(self, corpus_path):
        # Guarantee: every row in the file is accessible after loading
        corpus = WikipediaCorpus(corpus_path)
        assert corpus.get_passage("2") == "passage two"


# ---------------------------------------------------------------------------
# ContrieverRetriever
# ---------------------------------------------------------------------------

class TestContrieverRetriever:
    @pytest.fixture
    def retriever(self):
        corpus = MagicMock(spec=WikipediaCorpus)
        corpus.__len__ = MagicMock(return_value=10)

        vocab_size, hidden = 100, 32
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.ones(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }

        model = MagicMock()
        # last_hidden_state shape: (1, seq_len, hidden)
        output = MagicMock()
        output.last_hidden_state = torch.randn(1, 5, hidden)
        model.return_value = output

        # Pre-encode 10 fake passage embeddings into the retriever
        passages = [{"doc_id": str(i), "text": f"passage {i}"} for i in range(10)]
        retriever = ContrieverRetriever(model, tokenizer, corpus, device="cpu")
        retriever._passage_index = torch.randn(10, hidden)
        retriever._passage_ids = [str(i) for i in range(10)]
        retriever._passage_texts = [f"passage {i}" for i in range(10)]
        return retriever

    def test_retrieve_returns_list(self, retriever):
        # Guarantee: retrieve always returns a list
        results = retriever.retrieve("What is AI?", top_k=3)
        assert isinstance(results, list)

    def test_retrieve_respects_top_k(self, retriever):
        # Guarantee: retrieve returns exactly top_k results when corpus is larger
        results = retriever.retrieve("What is AI?", top_k=3)
        assert len(results) == 3

    def test_retrieve_result_has_required_keys(self, retriever):
        # Guarantee: every result dict contains doc_id, text, and score keys
        results = retriever.retrieve("What is AI?", top_k=1)
        assert {"doc_id", "text", "score"} <= set(results[0].keys())

    def test_retrieve_sorted_by_score_descending(self, retriever):
        # Guarantee: results are ordered highest score first
        results = retriever.retrieve("What is AI?", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_top_k_capped_at_corpus_size(self, retriever):
        # Guarantee: top_k larger than corpus size returns at most corpus size results
        results = retriever.retrieve("What is AI?", top_k=1000)
        assert len(results) <= 10


# ---------------------------------------------------------------------------
# Preprocessor.load_triviaqa
# ---------------------------------------------------------------------------

class TestLoadTriviaQA:
    @pytest.fixture
    def preprocessor(self):
        return Preprocessor(make_mock_retriever(SAMPLE_DOCS))

    def test_returns_list_of_dicts(self, preprocessor, tmp_path):
        # Guarantee: load_triviaqa returns a list of dicts
        # Use the real dataset path if available, else skip
        dataset_path = "/Users/keita-katsumi/Dropbox/Academic/CalPolyPomona/2026/CS5170/Final/dataset/TriviaQA"
        if not os.path.exists(dataset_path):
            pytest.skip("TriviaQA dataset not available")
        result = preprocessor.load_triviaqa(dataset_path, split="train")
        assert isinstance(result, list)
        assert isinstance(result[0], dict)

    def test_each_item_has_query_and_answer(self, preprocessor):
        # Guarantee: every item from load_triviaqa has non-empty query and answer fields
        dataset_path = "/Users/keita-katsumi/Dropbox/Academic/CalPolyPomona/2026/CS5170/Final/dataset/TriviaQA"
        if not os.path.exists(dataset_path):
            pytest.skip("TriviaQA dataset not available")
        result = preprocessor.load_triviaqa(dataset_path, split="train")
        for item in result[:10]:
            assert "query" in item and item["query"]
            assert "answer" in item and item["answer"]

    def test_no_train_test_overlap(self, preprocessor):
        # Guarantee: queries in train split do not appear in validation split
        dataset_path = "/Users/keita-katsumi/Dropbox/Academic/CalPolyPomona/2026/CS5170/Final/dataset/TriviaQA"
        if not os.path.exists(dataset_path):
            pytest.skip("TriviaQA dataset not available")
        train = preprocessor.load_triviaqa(dataset_path, split="train")
        val = preprocessor.load_triviaqa(dataset_path, split="validation")
        train_queries = {item["query"] for item in train}
        val_queries = {item["query"] for item in val}
        assert train_queries.isdisjoint(val_queries)


# ---------------------------------------------------------------------------
# Preprocessor.build_triplets
# ---------------------------------------------------------------------------

class TestBuildTriplets:
    @pytest.fixture
    def preprocessor(self):
        return Preprocessor(make_mock_retriever(SAMPLE_DOCS))

    def test_returns_list_of_triplets(self, preprocessor):
        # Guarantee: build_triplets returns a list of Triplet dataclass instances
        result = preprocessor.build_triplets(SAMPLE_QA, top_k=3)
        assert isinstance(result, list)
        assert all(isinstance(t, Triplet) for t in result)

    def test_triplet_count_is_queries_times_top_k(self, preprocessor):
        # Guarantee: total triplets equals number of queries multiplied by top_k
        result = preprocessor.build_triplets(SAMPLE_QA, top_k=3)
        assert len(result) == len(SAMPLE_QA) * 3

    def test_triplet_preserves_query_and_answer(self, preprocessor):
        # Guarantee: each Triplet carries the original query and answer strings
        result = preprocessor.build_triplets([SAMPLE_QA[0]], top_k=2)
        for t in result:
            assert t.query == SAMPLE_QA[0]["query"]
            assert t.answer == SAMPLE_QA[0]["answer"]

    def test_triplet_document_is_non_empty(self, preprocessor):
        # Guarantee: document field of every Triplet is a non-empty string
        result = preprocessor.build_triplets(SAMPLE_QA, top_k=2)
        for t in result:
            assert isinstance(t.document, str) and t.document.strip()

    def test_triplet_has_doc_id(self, preprocessor):
        # Guarantee: every Triplet records the doc_id of the retrieved passage
        result = preprocessor.build_triplets([SAMPLE_QA[0]], top_k=1)
        assert result[0].doc_id == SAMPLE_DOCS[0]["doc_id"]

    def test_empty_qa_returns_empty_list(self, preprocessor):
        # Guarantee: an empty input list produces an empty triplet list without error
        result = preprocessor.build_triplets([], top_k=5)
        assert result == []


# ---------------------------------------------------------------------------
# Preprocessor.iter_triplets
# ---------------------------------------------------------------------------

class TestIterTriplets:
    @pytest.fixture
    def preprocessor(self):
        return Preprocessor(make_mock_retriever(SAMPLE_DOCS))

    def test_iter_yields_triplets(self, preprocessor):
        # Guarantee: iter_triplets yields Triplet objects one by one
        result = list(preprocessor.iter_triplets(SAMPLE_QA, top_k=2))
        assert all(isinstance(t, Triplet) for t in result)

    def test_iter_matches_build_triplets_output(self, preprocessor):
        # Guarantee: iter_triplets and build_triplets produce identical content
        built = preprocessor.build_triplets(SAMPLE_QA, top_k=3)
        iterated = list(preprocessor.iter_triplets(SAMPLE_QA, top_k=3))
        assert built == iterated
