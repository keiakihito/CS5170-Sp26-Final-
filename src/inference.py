import string
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class RankedDocument:
    doc_id: str
    text: str
    score: float


@dataclass
class InferenceResult:
    query: str
    answer: str
    ranked_docs: List[RankedDocument]   # all docs after reranking
    selected_docs: List[RankedDocument] # docs that passed filter


class InferencePipeline:
    """Step 4: Full InfoGain-RAG inference pipeline.

    Query → Contriever (top-100) → RoBERTa reranker → filter → Qwen → answer
    """

    # Paper Section 4.1 inference settings
    FILTER_THRESHOLD = 0.2   # minimum reranker score to keep a document
    TOP_K_DOCS = 4            # maximum documents passed to the generator
    MIN_DOCS = 2              # minimum documents retained even if below threshold

    def __init__(
        self,
        retriever,
        reranker,
        reranker_tokenizer,
        generator,
        generator_tokenizer,
        device: str = "cpu",
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.reranker_tokenizer = reranker_tokenizer
        self.generator = generator
        self.generator_tokenizer = generator_tokenizer
        self.device = device

    # -- scoring helpers --

    def _tokenize_query_doc(self, query: str, doc_text: str) -> dict:
        inputs = self.reranker_tokenizer(
            query, doc_text, return_tensors="pt", truncation=True, max_length=512
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _reranker_score(self, query: str, doc_text: str) -> float:
        inputs = self._tokenize_query_doc(query, doc_text)
        with torch.no_grad():
            score = self.reranker(inputs["input_ids"], inputs["attention_mask"])
        return float(score[0]) if score.dim() > 0 else float(score)

    def _score_documents(self, query: str, docs: List[dict]) -> List[RankedDocument]:
        """Run reranker on all (query, doc) pairs and return scored RankedDocument list."""
        return [
            RankedDocument(
                doc_id=d["doc_id"],
                text=d["text"],
                score=self._reranker_score(query, d["text"]),
            )
            for d in docs
        ]

    # -- ranking helpers --

    def _rank_documents(self, scored: List[RankedDocument]) -> List[RankedDocument]:
        """Sort documents by score descending."""
        return sorted(scored, key=lambda d: d.score, reverse=True)

    # -- filtering helpers --

    def _docs_above_threshold(self, ranked: List[RankedDocument]) -> List[RankedDocument]:
        return [d for d in ranked if d.score >= self.FILTER_THRESHOLD]

    def _apply_min_docs(
        self, filtered: List[RankedDocument], ranked: List[RankedDocument]
    ) -> List[RankedDocument]:
        if len(filtered) < self.MIN_DOCS:
            return ranked[: self.MIN_DOCS]
        return filtered

    def _filter_documents(self, ranked: List[RankedDocument]) -> List[RankedDocument]:
        """Apply threshold filter, keep top-K, always retain at least MIN_DOCS."""
        if not ranked:
            return []
        filtered = self._docs_above_threshold(ranked)
        selected = self._apply_min_docs(filtered, ranked)
        return selected[: self.TOP_K_DOCS]

    # -- prompt and generation helpers --

    def _format_doc(self, i: int, doc: RankedDocument) -> str:
        return f"Document {i + 1}: {doc.text}"

    def _build_prompt(self, query: str, docs: List[RankedDocument]) -> str:
        """Construct the prompt string: selected documents concatenated with the query."""
        doc_texts = "\n".join(self._format_doc(i, d) for i, d in enumerate(docs))
        return f"{doc_texts}\n\nQuestion: {query}\nAnswer:"

    def _tokenize_prompt(self, prompt: str) -> dict:
        inputs = self.generator_tokenizer(prompt, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _decode_output(self, output_ids: torch.Tensor) -> str:
        return self.generator_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _generate_answer(self, prompt: str) -> str:
        """Run Qwen on the prompt and return the generated answer string."""
        inputs = self._tokenize_prompt(prompt)
        output_ids = self.generator.generate(**inputs)
        return self._decode_output(output_ids)

    # -- public API --

    def run(self, query: str) -> InferenceResult:
        """Run the full pipeline for one query and return an InferenceResult."""
        docs = self.retriever.retrieve(query, top_k=100)
        scored = self._score_documents(query, docs)
        ranked = self._rank_documents(scored)
        selected = self._filter_documents(ranked)
        prompt = self._build_prompt(query, selected)
        answer = self._generate_answer(prompt)
        return InferenceResult(
            query=query,
            answer=answer,
            ranked_docs=ranked,
            selected_docs=selected,
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _lowercase_strip(text: str) -> str:
    return text.lower().strip()


def _remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _normalise(text: str) -> str:
    return _collapse_whitespace(_remove_punctuation(_lowercase_strip(text)))


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Compute Exact Match: True if normalised prediction equals normalised ground truth."""
    return _normalise(prediction) == _normalise(ground_truth)


def evaluate_exact_match(predictions: List[str], ground_truths: List[str]) -> float:
    """Return mean EM accuracy over a list of prediction/ground-truth pairs."""
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"predictions and ground_truths must have the same length, "
            f"got {len(predictions)} and {len(ground_truths)}"
        )
    if not predictions:
        return 0.0
    return sum(exact_match(p, g) for p, g in zip(predictions, ground_truths)) / len(predictions)
