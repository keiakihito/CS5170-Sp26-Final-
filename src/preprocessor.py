import csv
import gzip
import os
from dataclasses import dataclass
from typing import Iterator, List

import torch


@dataclass
class Triplet:
    query: str
    answer: str
    document: str
    doc_id: str


# ---------------------------------------------------------------------------
# WikipediaCorpus
# ---------------------------------------------------------------------------

class WikipediaCorpus:
    """Loads and indexes the psgs_w100.tsv.gz Wikipedia passage corpus."""

    def __init__(self, tsv_gz_path: str):
        self._passages = self._load(tsv_gz_path)

    def _load(self, path: str) -> dict[str, str]:
        passages: dict[str, str] = {}
        with gzip.open(path, "rt", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                passages[row["id"]] = row["text"]
        return passages

    def __len__(self) -> int:
        return len(self._passages)

    def get_passage(self, doc_id: str) -> str:
        if doc_id not in self._passages:
            raise KeyError(f"doc_id '{doc_id}' not found in corpus")
        return self._passages[doc_id]


# ---------------------------------------------------------------------------
# ContrieverRetriever
# ---------------------------------------------------------------------------

class ContrieverRetriever:
    """Wraps facebook/contriever-msmarco to retrieve top-k passages from a corpus."""

    def __init__(self, model, tokenizer, corpus: WikipediaCorpus, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.device = device
        # Pre-populated by build_index() or injected directly in tests
        self._passage_index: torch.Tensor | None = None
        self._passage_ids: list[str] = []
        self._passage_texts: list[str] = []

    def _tokenize(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _mean_pool(self, model_output) -> torch.Tensor:
        return model_output.last_hidden_state.mean(dim=1)  # (1, hidden)

    def _encode(self, text: str) -> torch.Tensor:
        inputs = self._tokenize(text)
        with torch.no_grad():
            output = self.model(**inputs)
        return self._mean_pool(output)

    def _rank(self, query_emb: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        scores = (self._passage_index @ query_emb.T).squeeze(-1)  # (N,)
        k = min(top_k, len(scores))
        return torch.topk(scores, k)

    def retrieve(self, query: str, top_k: int = 100) -> List[dict]:
        """Return list of {doc_id, text, score} dicts ranked by relevance."""
        query_emb = self._encode(query)
        top_scores, top_indices = self._rank(query_emb, top_k)
        return [
            {
                "doc_id": self._passage_ids[i],
                "text": self._passage_texts[i],
                "score": top_scores[idx].item(),
            }
            for idx, i in enumerate(top_indices.tolist())
        ]


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class Preprocessor:
    """Step 0: Load TriviaQA, retrieve top-100 docs per query, produce triplets."""

    def __init__(self, retriever: ContrieverRetriever):
        self.retriever = retriever

    # -- TriviaQA loading helpers --

    def _find_cache_dir(self, dataset_path: str) -> str:
        cache_dirs = [d for d in os.listdir(dataset_path) if d.startswith("datasets--")]
        if not cache_dirs:
            raise FileNotFoundError(f"No HuggingFace dataset cache found in {dataset_path}")
        return cache_dirs[0]

    def _repo_id_from_cache_dir(self, cache_dir_name: str) -> str:
        # datasets--mandarjoshi--trivia_qa  →  mandarjoshi/trivia_qa
        return cache_dir_name.removeprefix("datasets--").replace("--", "/")

    def _extract_answer(self, raw: object) -> str:
        if isinstance(raw, dict):
            return raw.get("value") or (raw.get("aliases") or [""])[0]
        return str(raw)

    def _parse_qa_pairs(self, dataset) -> List[dict]:
        pairs = []
        for item in dataset:
            query = item.get("question") or item.get("query", "")
            answer = self._extract_answer(item.get("answer", ""))
            if query and answer:
                pairs.append({"query": query, "answer": answer})
        return pairs

    def _deduplicate(self, qa_pairs: List[dict]) -> List[dict]:
        seen: set[str] = set()
        unique = []
        for pair in qa_pairs:
            if pair["query"] not in seen:
                seen.add(pair["query"])
                unique.append(pair)
        return unique

    def load_triviaqa(self, dataset_path: str, split: str = "train") -> List[dict]:
        """Load TriviaQA split and return list of {query, answer} dicts."""
        from datasets import load_dataset

        cache_dir_name = self._find_cache_dir(dataset_path)
        repo_id = self._repo_id_from_cache_dir(cache_dir_name)
        ds = load_dataset(repo_id, "rc.wikipedia", cache_dir=dataset_path, split=split)
        return self._deduplicate(self._parse_qa_pairs(ds))

    # -- Triplet building helpers --

    def _docs_to_triplets(self, pair: dict, docs: List[dict]) -> List[Triplet]:
        return [
            Triplet(
                query=pair["query"],
                answer=pair["answer"],
                document=doc["text"],
                doc_id=doc["doc_id"],
            )
            for doc in docs
        ]

    def build_triplets(self, qa_pairs: List[dict], top_k: int = 100) -> List[Triplet]:
        """Retrieve top_k docs per query and return all <query, answer, doc> triplets."""
        return list(self.iter_triplets(qa_pairs, top_k=top_k))

    def iter_triplets(self, qa_pairs: List[dict], top_k: int = 100) -> Iterator[Triplet]:
        """Memory-efficient iterator version of build_triplets."""
        for pair in qa_pairs:
            docs = self.retriever.retrieve(pair["query"], top_k=top_k)
            yield from self._docs_to_triplets(pair, docs)
