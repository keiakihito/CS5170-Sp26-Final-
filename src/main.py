"""
InfoGain-RAG — evaluation visualizer.

If outputs/results.json exists (written by run_pipeline.py), it loads our real
reproduced results and overlays them against the paper's numbers.
Otherwise it falls back to paper-only numbers so the demo still runs without a GPU.

Run:
    python src/main.py
"""

import json
import os
import sys

from evaluator import EvalRecord, EvalSuite, ResultsVisualizer


# ---------------------------------------------------------------------------
# Tee: write every print() to both stdout and a log file simultaneously
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def _setup_tee(log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)

# ---------------------------------------------------------------------------
# 1. Paper-reported numbers (replication target)
# ---------------------------------------------------------------------------

PAPER_RESULTS = [
    # TriviaQA — Qwen2.5-7B
    ("Qwen2.5-7B", "TriviaQA", "naive_rag",    0.529, 1000),
    ("Qwen2.5-7B", "TriviaQA", "bge_reranker",  0.670, 1000),
    ("Qwen2.5-7B", "TriviaQA", "infogain_rag",  0.720, 1000),
    # NaturalQA — Qwen2.5-7B
    ("Qwen2.5-7B", "NaturalQA", "naive_rag",    0.481, 1000),
    ("Qwen2.5-7B", "NaturalQA", "bge_reranker",  0.612, 1000),
    ("Qwen2.5-7B", "NaturalQA", "infogain_rag",  0.658, 1000),
    # TriviaQA — ablation variants
    ("Qwen2.5-7B", "TriviaQA", "ce_only",       0.681, 1000),
    ("Qwen2.5-7B", "TriviaQA", "margin_only",   0.694, 1000),
    ("Qwen2.5-7B", "TriviaQA", "multi_task",    0.720, 1000),
]

BETA_SENSITIVITY = {0.0: 0.694, 0.25: 0.706, 0.5: 0.713, 0.75: 0.720, 1.0: 0.681}

RESULTS_JSON = os.path.join("outputs", "results.json")


def _load_our_results() -> list:
    """Load results produced by run_pipeline.py, or empty list if not yet run."""
    if not os.path.exists(RESULTS_JSON):
        return []
    with open(RESULTS_JSON) as f:
        return json.load(f)


def build_suite() -> tuple[EvalSuite, bool]:
    """Build suite from paper numbers + our reproduced results (if available).
    Returns (suite, has_our_results).
    """
    our_raw = _load_our_results()
    has_ours = bool(our_raw)

    suite = EvalSuite()
    for model, dataset, approach, em, n in PAPER_RESULTS:
        suite.add(EvalRecord(model=model, dataset=dataset, approach=approach,
                             em_score=em, n_samples=n))
    for r in our_raw:
        # tag our results with a distinct approach name so they show up in charts
        suite.add(EvalRecord(
            model=r["model"],
            dataset=r["dataset"],
            approach=r["approach"] + "_ours",
            em_score=r["em_score"],
            n_samples=r["n_samples"],
        ))
    return suite, has_ours


# ---------------------------------------------------------------------------
# 2. Print summary table
# ---------------------------------------------------------------------------

def _divider(width: int = 60) -> str:
    return "─" * width


def print_summary(suite: EvalSuite, has_ours: bool) -> None:
    print(_divider())
    label = "InfoGain-RAG — Results (paper vs. ours)" if has_ours else "InfoGain-RAG — Paper-reported Results"
    print(f"  {label}")
    print(_divider())

    df = suite.to_dataframe()
    for dataset in ("TriviaQA", "NaturalQA"):
        print(f"\n  Dataset: {dataset}")
        paper_approaches = ["naive_rag", "bge_reranker", "infogain_rag"]
        subset = df[(df["dataset"] == dataset) & (df["approach"].isin(paper_approaches))]
        for _, row in subset.iterrows():
            bar = "█" * int(row["em_score"] * 30)
            print(f"    [paper] {row['approach']:<16}  EM={row['em_score']:.3f}  {bar}")

        if has_ours:
            our_subset = df[(df["dataset"] == dataset) &
                            (df["approach"].str.endswith("_ours"))]
            for _, row in our_subset.iterrows():
                bar = "█" * int(row["em_score"] * 30)
                approach_label = row["approach"].replace("_ours", "")
                paper_em = df[(df["dataset"] == dataset) &
                              (df["approach"] == approach_label)]["em_score"]
                diff = ""
                if not paper_em.empty:
                    delta = row["em_score"] - paper_em.iloc[0]
                    diff = f"  (Δ {delta:+.3f} vs paper)"
                print(f"    [ours]  {approach_label:<16}  EM={row['em_score']:.3f}  {bar}{diff}")

        try:
            gain = suite.improvement_over_baseline(dataset, "Qwen2.5-7B", "naive_rag", "infogain_rag")
            print(f"\n    Paper InfoGain gain over naive RAG: +{gain:.3f} EM ({gain*100:.1f} pp)")
        except KeyError:
            pass


def print_ablation(suite: EvalSuite) -> None:
    print(f"\n{_divider()}")
    print("  Ablation Study — TriviaQA")
    print(_divider())
    df = suite.to_dataframe()
    subset = df[(df["dataset"] == "TriviaQA") &
                (df["approach"].isin(["ce_only", "margin_only", "multi_task"]))]
    for _, row in subset.iterrows():
        bar = "█" * int(row["em_score"] * 30)
        print(f"    {row['approach']:<14}  EM={row['em_score']:.3f}  {bar}")


def print_beta(beta_records: dict) -> None:
    print(f"\n{_divider()}")
    print("  β Sensitivity (TriviaQA, Qwen2.5-7B)")
    print(_divider())
    for beta, em in sorted(beta_records.items()):
        bar = "█" * int(em * 30)
        marker = "  ← best" if beta == 0.75 else ""
        print(f"    β={beta:.2f}  EM={em:.3f}  {bar}{marker}")


# ---------------------------------------------------------------------------
# 3. Save charts
# ---------------------------------------------------------------------------

def save_charts(suite: EvalSuite, beta_records: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    viz = ResultsVisualizer(suite)

    paths = {
        "bar_triviaqa":  os.path.join(out_dir, "bar_triviaqa.png"),
        "bar_naturalqa": os.path.join(out_dir, "bar_naturalqa.png"),
        "beta":          os.path.join(out_dir, "beta_sensitivity.png"),
        "ablation":      os.path.join(out_dir, "ablation.png"),
    }

    viz.bar_chart_by_dataset("TriviaQA",  save_path=paths["bar_triviaqa"])
    viz.bar_chart_by_dataset("NaturalQA", save_path=paths["bar_naturalqa"])
    viz.beta_sensitivity_plot(beta_records, model="Qwen2.5-7B", save_path=paths["beta"])
    viz.ablation_bar_chart("TriviaQA", save_path=paths["ablation"])

    print(f"\n{_divider()}")
    print("  Charts saved")
    print(_divider())
    for name, path in paths.items():
        print(f"    {path}")


# ---------------------------------------------------------------------------
# 4. Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = "outputs"
    _setup_tee(os.path.join(out_dir, "sample-run.log"))
    suite, has_ours = build_suite()
    if has_ours:
        print("  [outputs/results.json found — showing our reproduced results alongside paper numbers]")
    else:
        print("  [No outputs/results.json found — showing paper-reported numbers only]")
        print("  [Run `python src/run_pipeline.py --device cuda` to generate your own results]")
    print()
    print_summary(suite, has_ours)
    print_ablation(suite)
    print_beta(BETA_SENSITIVITY)
    save_charts(suite, BETA_SENSITIVITY, out_dir=out_dir)
    print(f"\n{_divider()}")
    print("  Done.")
    print(_divider())


if __name__ == "__main__":
    main()
