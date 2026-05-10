"""
InfoGain-RAG — Full end-to-end pipeline runner.

Chains all five steps (preprocess → DIG score → train reranker → inference → evaluate)
and writes real EM results to outputs/results.json so main.py can visualize them.

Intermediate outputs are saved after each step so you can resume if the run is interrupted.

Usage:

  Smoke test (CPU, no downloads, ~10 seconds):
    python src/run_pipeline.py --smoke

  Full run (GPU on RunPod):
    python src/run_pipeline.py --data_dir dataset/ --device cuda

Options:
    --smoke         Run end-to-end with tiny fake data and mocked models (no GPU, no downloads)
    --data_dir      Path to the dataset/ folder (default: dataset/)
    --out_dir       Where to write outputs (default: outputs/)
    --checkpoint    Path to save/load the trained reranker (default: outputs/reranker.pt)
    --n_train       Max QA pairs for training  (default: all)
    --n_eval        Max QA pairs for evaluation (default: all)
    --epochs        Training epochs (default: 3)
    --batch_size    Training batch size (default: 16)
    --device        cpu | cuda (default: cpu)
    --resume        Skip steps whose output files already exist
"""

import argparse
import json
import os
import pickle
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from evaluator import EvalRecord, EvalSuite, ResultsVisualizer
from inference import InferencePipeline, evaluate_exact_match
from preprocessor import ContrieverRetriever, Preprocessor, Triplet, WikipediaCorpus
from dig_scorer import DIGResult, DIGScorer
from reranker import MultiTaskReranker, RerankerTrainer, TrainingBatch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="InfoGain-RAG full pipeline")
    p.add_argument("--smoke",       action="store_true",
                   help="Run with tiny fake data and mocked models — no GPU or downloads needed")
    p.add_argument("--data_dir",    default="dataset/")
    p.add_argument("--out_dir",     default="outputs/")
    p.add_argument("--checkpoint",  default="outputs/reranker.pt")
    p.add_argument("--n_train",     type=int, default=None)
    p.add_argument("--n_eval",      type=int, default=None)
    p.add_argument("--epochs",      type=int, default=3)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--device",      default="cpu")
    p.add_argument("--resume",      action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _pkl_path(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, f"{name}.pkl")


def _save_pkl(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved → {path}")


def _load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _should_skip(path: str, resume: bool) -> bool:
    if resume and os.path.exists(path):
        print(f"  Resuming — found {path}, skipping this step.")
        return True
    return False


# ---------------------------------------------------------------------------
# Smoke-test helpers — tiny fake data + mocked models, no downloads needed
# ---------------------------------------------------------------------------

def _make_smoke_qa_pairs() -> list:
    return [
        {"question": f"What is question {i}?", "answer": f"answer{i}"}
        for i in range(10)
    ]


def _make_smoke_triplets() -> list:
    from preprocessor import Triplet
    return [
        Triplet(
            query=f"What is question {i}?",
            answer=f"answer{i}",
            document=f"This passage talks about answer{i} in detail.",
            doc_id=f"d{i}",
        )
        for i in range(20)
    ]


def _make_smoke_dig_results() -> list:
    from dig_scorer import DIGCategory, DIGResult
    results = []
    for i in range(20):
        category = DIGCategory.POSITIVE if i % 2 == 0 else DIGCategory.NEGATIVE
        dig_score = 0.6 if category == DIGCategory.POSITIVE else -0.3
        results.append(DIGResult(
            query=f"What is question {i % 10}?",
            answer=f"answer{i % 10}",
            document=f"This passage talks about answer{i % 10} in detail.",
            doc_id=f"d{i}",
            confidence_with_doc=0.8 if category == DIGCategory.POSITIVE else 0.2,
            confidence_without_doc=0.2,
            dig_score=dig_score,
            category=category,
        ))
    return results


def _make_smoke_retriever(qa_pairs: list):
    from unittest.mock import MagicMock
    retriever = MagicMock()
    retriever.retrieve.side_effect = lambda query, top_k=100: [
        {"doc_id": f"d{j}", "text": f"passage about {query} number {j}", "score": 1.0 - j * 0.1}
        for j in range(min(top_k, 5))
    ]
    return retriever


def _make_smoke_reranker_components(device: str):
    """Return a tiny real MultiTaskReranker with a small random encoder — no download."""
    import torch.nn as nn
    from unittest.mock import MagicMock

    hidden = 32

    # minimal encoder mock that returns pooler_output of the right shape
    class _TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, hidden)
            self.config = MagicMock()
            self.config.hidden_size = hidden

        def forward(self, input_ids, attention_mask=None, **kwargs):
            batch = input_ids.shape[0]
            out = MagicMock()
            out.pooler_output = torch.randn(batch, hidden)
            return out

    encoder = _TinyEncoder().to(device)
    reranker = MultiTaskReranker(encoder).to(device)

    tokenizer = MagicMock()
    tokenizer.side_effect = lambda *a, **kw: {
        "input_ids":      torch.ones(len(a[0]) if isinstance(a[0], list) else 1, 8, dtype=torch.long),
        "attention_mask": torch.ones(len(a[0]) if isinstance(a[0], list) else 1, 8, dtype=torch.long),
    }
    return reranker, tokenizer


def _make_smoke_generator_components():
    """Return mocked Qwen generator + tokenizer that produces deterministic answers."""
    from unittest.mock import MagicMock
    tokenizer = MagicMock()
    tokenizer.side_effect = lambda *a, **kw: {
        "input_ids":      torch.ones(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }
    # decode returns the answer embedded in the query string
    tokenizer.decode.side_effect = lambda ids, **kw: "answer0"

    generator = MagicMock()
    generator.generate.return_value = torch.tensor([[1, 2, 3]])
    return generator, tokenizer


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_contriever(device: str):
    print("Loading Contriever …")
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
    model = AutoModel.from_pretrained("facebook/contriever-msmarco").to(device)
    model.eval()
    return model, tokenizer


def _load_qwen(device: str):
    print("Loading Qwen2.5-7B …")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def _load_roberta(device: str):
    print("Loading RoBERTa-large …")
    from transformers import RobertaModel
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    encoder = RobertaModel.from_pretrained("roberta-large").to(device)
    return encoder, tokenizer


# ---------------------------------------------------------------------------
# Step 0 — Preprocess
# ---------------------------------------------------------------------------

def step0_preprocess(args, out_dir: str) -> tuple[List[dict], List[dict]]:
    if args.smoke:
        print("\n=== Step 0: Preprocess [SMOKE] ===")
        qa = _make_smoke_qa_pairs()
        triplets = _make_smoke_triplets()
        _save_pkl(qa, _pkl_path(out_dir, "qa_train"))
        _save_pkl(qa, _pkl_path(out_dir, "qa_eval"))
        _save_pkl(triplets, _pkl_path(out_dir, "triplets_train"))
        print(f"  {len(qa)} fake QA pairs, {len(triplets)} fake triplets")
        return qa, qa

    train_path = _pkl_path(out_dir, "qa_train")
    eval_path  = _pkl_path(out_dir, "qa_eval")
    trip_path  = _pkl_path(out_dir, "triplets_train")

    if _should_skip(trip_path, args.resume):
        return _load_pkl(train_path), _load_pkl(eval_path)

    print("\n=== Step 0: Preprocess ===")
    corpus = WikipediaCorpus(os.path.join(args.data_dir, "wikipedia-dump/psgs_w100.tsv.gz"))
    cont_model, cont_tok = _load_contriever(args.device)
    retriever = ContrieverRetriever(cont_model, cont_tok, corpus, device=args.device)
    preprocessor = Preprocessor(retriever)

    triviaqa_path = os.path.join(args.data_dir, "TriviaQA")
    qa_train = preprocessor.load_triviaqa(triviaqa_path, split="train")
    qa_eval  = preprocessor.load_triviaqa(triviaqa_path, split="validation")

    if args.n_train:
        qa_train = qa_train[: args.n_train]
    if args.n_eval:
        qa_eval = qa_eval[: args.n_eval]

    print(f"  {len(qa_train)} train pairs, {len(qa_eval)} eval pairs")
    _save_pkl(qa_train, train_path)
    _save_pkl(qa_eval,  eval_path)

    print("  Building triplets …")
    triplets = preprocessor.build_triplets(qa_train)
    _save_pkl(triplets, trip_path)
    print(f"  {len(triplets)} triplets")

    del cont_model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    return qa_train, qa_eval


# ---------------------------------------------------------------------------
# Step 1 + 2 — Query categorization + DIG scoring
# ---------------------------------------------------------------------------

def step2_dig_score(args, out_dir: str) -> List[DIGResult]:
    dig_path = _pkl_path(out_dir, "dig_results")

    if args.smoke:
        print("\n=== Step 1+2: DIG Scoring [SMOKE] ===")
        dig_results = _make_smoke_dig_results()
        _save_pkl(dig_results, dig_path)
        print(f"  {len(dig_results)} fake DIG results")
        return dig_results

    if _should_skip(dig_path, args.resume):
        return _load_pkl(dig_path)

    print("\n=== Step 1+2: DIG Scoring ===")
    triplets: List[Triplet] = _load_pkl(_pkl_path(out_dir, "triplets_train"))
    qwen_model, qwen_tok = _load_qwen(args.device)

    scorer = DIGScorer(qwen_model, qwen_tok, device=args.device)
    print(f"  Scoring {len(triplets)} triplets …")
    dig_results = scorer.score_triplets(triplets)

    _save_pkl(dig_results, dig_path)
    print(f"  {len(dig_results)} DIG results saved")

    del qwen_model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    return dig_results


# ---------------------------------------------------------------------------
# Step 3 — Build training batches and train reranker
# ---------------------------------------------------------------------------

def _balance_dig_results(dig_results: List[DIGResult]) -> list:
    """Return equal-sized shuffled list of (record, label) pairs from positives/negatives."""
    from dig_scorer import DIGCategory
    import random
    positives = [r for r in dig_results if r.category == DIGCategory.POSITIVE]
    negatives = [r for r in dig_results if r.category == DIGCategory.NEGATIVE]
    random.shuffle(positives)
    random.shuffle(negatives)
    n = min(len(positives), len(negatives))
    balanced = [(r, 1.0) for r in positives[:n]] + [(r, 0.0) for r in negatives[:n]]
    random.shuffle(balanced)
    return balanced


def _tokenize_chunk(chunk: list, tokenizer, device: str) -> dict:
    """Tokenize a batch of (record, label) pairs into padded input tensors."""
    queries = [r.query    for r, _ in chunk]
    docs    = [r.document for r, _ in chunk]
    enc = tokenizer(
        queries, docs,
        return_tensors="pt", truncation=True, max_length=512, padding=True,
    )
    return {k: v.to(device) for k, v in enc.items()}


def _make_training_batch(chunk: list, inputs: dict, device: str) -> TrainingBatch:
    """Assemble a TrainingBatch from a tokenized chunk."""
    labels     = torch.tensor([lbl for _, lbl in chunk], dtype=torch.float32).to(device)
    pos_scores = torch.tensor([r.dig_score for r, lbl in chunk if lbl == 1.0]).to(device)
    neg_scores = torch.tensor([r.dig_score for r, lbl in chunk if lbl == 0.0]).to(device)
    return TrainingBatch(
        query_doc_inputs=inputs,
        ce_labels=labels,
        positive_scores=pos_scores,
        negative_scores=neg_scores,
    )


def _dig_results_to_batches(
    dig_results: List[DIGResult],
    reranker_tokenizer,
    batch_size: int,
    device: str,
) -> List[TrainingBatch]:
    balanced = _balance_dig_results(dig_results)
    batches = []
    for i in range(0, len(balanced), batch_size):
        chunk = balanced[i : i + batch_size]
        if not chunk:
            continue
        inputs = _tokenize_chunk(chunk, reranker_tokenizer, device)
        batch  = _make_training_batch(chunk, inputs, device)
        if len(batch.positive_scores) == 0 or len(batch.negative_scores) == 0:
            continue
        batches.append(batch)
    return batches


def _run_training_epochs(
    trainer: "RerankerTrainer",
    dig_results: List["DIGResult"],
    reranker_tokenizer,
    epochs: int,
    batch_size: int,
    device: str,
) -> None:
    """Run the full epoch loop, printing avg loss per epoch."""
    for epoch in range(epochs):
        batches = _dig_results_to_batches(dig_results, reranker_tokenizer, batch_size, device)
        total_loss = 0.0
        for batch in tqdm(batches, desc=f"  Epoch {epoch+1}/{epochs}"):
            result = trainer.train_step(batch)
            total_loss += result["total_loss"]
        avg = total_loss / max(len(batches), 1)
        print(f"  Epoch {epoch+1} avg loss: {avg:.4f}")


def step3_train_reranker(args, out_dir: str):
    if args.smoke:
        print("\n=== Step 3: Train Reranker [SMOKE] ===")
        reranker, reranker_tok = _make_smoke_reranker_components(args.device)
        trainer = RerankerTrainer(reranker, device=args.device)
        dig_results = _load_pkl(_pkl_path(out_dir, "dig_results"))
        batches = _dig_results_to_batches(dig_results, reranker_tok, batch_size=2, device=args.device)
        for batch in tqdm(batches, desc="  Epoch 1/1 [smoke]"):
            result = trainer.train_step(batch)
        trainer.save(args.checkpoint)
        print(f"  Smoke reranker saved → {args.checkpoint}")
        return reranker, reranker_tok

    if _should_skip(args.checkpoint, args.resume):
        encoder, reranker_tok = _load_roberta(args.device)
        reranker = MultiTaskReranker(encoder)
        trainer = RerankerTrainer(reranker, device=args.device)
        trainer.load(args.checkpoint)
        return reranker, reranker_tok

    print("\n=== Step 3: Train Reranker ===")
    dig_results = _load_pkl(_pkl_path(out_dir, "dig_results"))
    encoder, reranker_tok = _load_roberta(args.device)
    reranker = MultiTaskReranker(encoder).to(args.device)
    trainer = RerankerTrainer(reranker, device=args.device)

    _run_training_epochs(trainer, dig_results, reranker_tok, args.epochs, args.batch_size, args.device)

    trainer.save(args.checkpoint)
    print(f"  Reranker saved → {args.checkpoint}")
    return reranker, reranker_tok


# ---------------------------------------------------------------------------
# Step 4 — Inference + Evaluation
# ---------------------------------------------------------------------------

def _run_evaluation(
    qa_pairs: List[dict],
    pipeline: InferencePipeline,
    dataset_name: str,
    approach: str,
) -> EvalRecord:
    predictions, ground_truths = [], []
    for qa in tqdm(qa_pairs, desc=f"  Eval {dataset_name}/{approach}"):
        result = pipeline.run(qa["question"])
        predictions.append(result.answer)
        ground_truths.append(qa["answer"])

    em = evaluate_exact_match(predictions, ground_truths)
    print(f"  {dataset_name} {approach}: EM = {em:.4f}")
    return EvalRecord(
        model="Qwen2.5-7B",
        dataset=dataset_name,
        approach=approach,
        em_score=round(em, 4),
        n_samples=len(predictions),
    )


def step4_inference(args, out_dir: str, qa_eval: List[dict]) -> List[EvalRecord]:
    results_path = os.path.join(out_dir, "results.json")

    if args.smoke:
        print("\n=== Step 4: Inference + Evaluation [SMOKE] ===")
        retriever = _make_smoke_retriever(qa_eval)
        reranker, reranker_tok = _make_smoke_reranker_components(args.device)
        generator, gen_tok = _make_smoke_generator_components()
        pipeline = InferencePipeline(
            retriever=retriever,
            reranker=reranker,
            reranker_tokenizer=reranker_tok,
            generator=generator,
            generator_tokenizer=gen_tok,
            device=args.device,
        )
        records = [_run_evaluation(qa_eval, pipeline, "TriviaQA", "infogain_rag")]
        _save_results(records, results_path)
        return records

    if _should_skip(results_path, args.resume):
        with open(results_path) as f:
            raw = json.load(f)
        return [EvalRecord(**r) for r in raw]

    print("\n=== Step 4: Inference + Evaluation ===")
    corpus = WikipediaCorpus(os.path.join(args.data_dir, "wikipedia-dump/psgs_w100.tsv.gz"))
    cont_model, cont_tok = _load_contriever(args.device)
    retriever = ContrieverRetriever(cont_model, cont_tok, corpus, device=args.device)

    qwen_model, qwen_tok = _load_qwen(args.device)
    reranker, reranker_tok = step3_train_reranker(args, out_dir)

    pipeline = InferencePipeline(
        retriever=retriever,
        reranker=reranker,
        reranker_tokenizer=reranker_tok,
        generator=qwen_model,
        generator_tokenizer=qwen_tok,
        device=args.device,
    )

    records = [_run_evaluation(qa_eval, pipeline, "TriviaQA", "infogain_rag")]
    _save_results(records, results_path)
    return records


# ---------------------------------------------------------------------------
# Step 5 — Save results and update main.py data
# ---------------------------------------------------------------------------

def _save_results(records: List[EvalRecord], path: str) -> None:
    data = [
        {
            "model":    r.model,
            "dataset":  r.dataset,
            "approach": r.approach,
            "em_score": r.em_score,
            "n_samples": r.n_samples,
        }
        for r in records
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved → {path}")


def _save_result_charts(suite: EvalSuite, out_dir: str) -> None:
    """Save a bar chart per dataset present in suite."""
    viz = ResultsVisualizer(suite)
    df = suite.to_dataframe()
    for dataset in df["dataset"].unique():
        try:
            viz.bar_chart_by_dataset(
                dataset,
                save_path=os.path.join(out_dir, f"bar_{dataset.lower()}_ours.png"),
            )
        except KeyError:
            pass
    print(f"\n  Charts saved to {out_dir}")


def step5_report(records: List[EvalRecord], out_dir: str) -> None:
    print("\n=== Step 5: Report ===")
    suite = EvalSuite()
    for r in records:
        suite.add(r)

    df = suite.to_dataframe()
    print(df.to_string(index=False))

    _save_result_charts(suite, out_dir)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _print_run_header(args: argparse.Namespace) -> None:
    """Print a one-line title and the key run parameters."""
    if args.smoke:
        print("InfoGain-RAG — Smoke Test (fake data, mocked models, CPU)")
        print("  Verifies full pipeline wiring without any downloads or GPU.")
    else:
        print("InfoGain-RAG — Full Pipeline Run")
    print(f"  device     : {args.device}")
    print(f"  data_dir   : {args.data_dir}")
    print(f"  out_dir    : {args.out_dir}")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  resume     : {args.resume}")


def main() -> None:
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    _print_run_header(args)

    _, qa_eval = step0_preprocess(args, args.out_dir)
    step2_dig_score(args, args.out_dir)
    step3_train_reranker(args, args.out_dir)
    records = step4_inference(args, args.out_dir, qa_eval)
    step5_report(records, args.out_dir)

    print("\n=== Done. ===")
    print(f"Results JSON : {os.path.join(args.out_dir, 'results.json')}")
    print("Copy outputs/results.json back to your local machine and run:")
    print("  python src/main.py")


if __name__ == "__main__":
    main()
