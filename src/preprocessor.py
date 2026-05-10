import csv
import gzip
from dataclasses import dataclass
from typing import Iterator, List

import torch
import torch.nn.functional as F


@dataclass
class Triplet:
    query: str
    answer: str
    document: str
    doc_id: str


class WikipediaCorpus:
    """Loads and indexes the psgs_w100.tsv.gz Wikipedia passage corpus."""

    def __init__(self, tsv_gz_path: str):
        self._passages: dict[str, str] = {}
        with gzip.open(tsv_gz_path, "rt", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self._passages[row["id"]] = row["text"]

    def __len__(self) -> int:
        return len(self._passages)

    def get_passage(self, doc_id: str) -> str:
        if doc_id not in self._passages:
            raise KeyError(f"doc_id '{doc_id}' not found in corpus")
        return self._passages[doc_id]


class ContrieverRetriever:
    """Wraps facebook/contriever-msmarco to retrieve top-k passages from a corpus."""

    def __init__(self, model, tokenizer, corpus: WikipediaCorpus, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.device = device
        # These are pre-populated externally (e.g. by an index-building step)
        # or set directly in tests; see build_index() for production use.
        self._passage_index: torch.Tensor | None = None
        self._passage_ids: list[str] = []
        self._passage_texts: list[str] = []

    def _encode(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model(**inputs)
        # Mean-pool over token dimension
        return output.last_hidden_state.mean(dim=1)  # (1, hidden)

    def retrieve(self, query: str, top_k: int = 100) -> List[dict]:
        """Return list of {doc_id, text, score} dicts ranked by relevance."""
        query_emb = self._encode(query)  # (1, hidden)
        # Dot-product similarity against pre-built passage index
        scores = (self._passage_index @ query_emb.T).squeeze(-1)  # (N,)
        k = min(top_k, len(scores))
        top_scores, top_indices = torch.topk(scores, k)
        return [
            {
                "doc_id": self._passage_ids[i],
                "text": self._passage_texts[i],
                "score": top_scores[idx].item(),
            }
            for idx, i in enumerate(top_indices.tolist())
        ]


class Preprocessor:
    """Step 0: Load TriviaQA, retrieve top-100 docs per query, produce triplets."""

    def __init__(self, retriever: ContrieverRetriever):
        self.retriever = retriever

    def load_triviaqa(self, dataset_path: str, split: str = "train") -> List[dict]:
        """Load TriviaQA split and return list of {query, answer} dicts."""
        from datasets import load_dataset
        import os

        # Locate the HuggingFace hub cache dir (named datasets--<org>--<name>)
        cache_dirs = [d for d in os.listdir(dataset_path) if d.startswith("datasets--")]
        if not cache_dirs:
            raise FileNotFoundError(f"No HuggingFace dataset cache found in {dataset_path}")

        cache_dir = os.path.join(dataset_path, cache_dirs[0])
        # Derive repo_id from directory name: datasets--mandarjoshi--trivia_qa → mandarjoshi/trivia_qa
        repo_id = cache_dirs[0].removeprefix("datasets--").replace("--", "/")

        ds = load_dataset(repo_id, "rc.wikipedia", cache_dir=dataset_path, split=split)
        split_ds = ds

        qa_pairs = []
        for item in split_ds:
            query = item.get("question") or item.get("query", "")
            # TriviaQA answers may be a dict with an "aliases" list or a plain string
            raw_answer = item.get("answer", "")
            if isinstance(raw_answer, dict):
                answer = raw_answer.get("value") or (raw_answer.get("aliases") or [""])[0]
            else:
                answer = str(raw_answer)
            if query and answer:
                qa_pairs.append({"query": query, "answer": answer})

        # Remove overlap: deduplicate by query string
        seen: set[str] = set()
        unique = []
        for pair in qa_pairs:
            if pair["query"] not in seen:
                seen.add(pair["query"])
                unique.append(pair)

        return unique

    def build_triplets(
        self, qa_pairs: List[dict], top_k: int = 100
    ) -> List[Triplet]:
        """Retrieve top_k docs per query and return all <query, answer, doc> triplets."""
        return list(self.iter_triplets(qa_pairs, top_k=top_k))

    def iter_triplets(
        self, qa_pairs: List[dict], top_k: int = 100
    ) -> Iterator[Triplet]:
        """Memory-efficient iterator version of build_triplets."""
        for pair in qa_pairs:
            docs = self.retriever.retrieve(pair["query"], top_k=top_k)
            for doc in docs:
                yield Triplet(
                    query=pair["query"],
                    answer=pair["answer"],
                    document=doc["text"],
                    doc_id=doc["doc_id"],
                )
