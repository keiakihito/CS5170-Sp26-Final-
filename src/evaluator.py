from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass
class EvalRecord:
    """EM result for one (model, dataset, approach) combination."""
    model: str
    dataset: str
    approach: str   # e.g. "naive_rag", "bge_reranker", "infogain_rag"
    em_score: float
    n_samples: int


@dataclass
class EvalSuite:
    """Collection of EvalRecords with summary and comparison utilities."""
    records: List[EvalRecord] = field(default_factory=list)

    def add(self, record: EvalRecord) -> None:
        """Append one EvalRecord to the suite."""
        self.records.append(record)

    _COLUMNS = ["model", "dataset", "approach", "em_score", "n_samples"]

    def _record_to_dict(self, r: EvalRecord) -> dict:
        return {c: getattr(r, c) for c in self._COLUMNS}

    def _empty_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self._COLUMNS)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all records as a pandas DataFrame."""
        if not self.records:
            return self._empty_dataframe()
        return pd.DataFrame([self._record_to_dict(r) for r in self.records])

    def _lookup(self, dataset: str, model: str, approach: Optional[str] = None) -> List[EvalRecord]:
        results = [
            r for r in self.records
            if r.dataset == dataset and r.model == model
            and (approach is None or r.approach == approach)
        ]
        return results

    def _missing_key_label(self, dataset: str, model: str, approach: Optional[str]) -> str:
        label = f"dataset={dataset!r}, model={model!r}"
        if approach:
            label += f", approach={approach!r}"
        return label

    def _require_records(self, dataset: str, model: str, approach: Optional[str] = None) -> List[EvalRecord]:
        results = self._lookup(dataset, model, approach)
        if not results:
            raise KeyError(self._missing_key_label(dataset, model, approach))
        return results

    def best_approach(self, dataset: str, model: str) -> str:
        """Return the approach with the highest EM score for a given dataset+model."""
        records = self._require_records(dataset, model)
        return max(records, key=lambda r: r.em_score).approach

    def _score_for(self, dataset: str, model: str, approach: str) -> float:
        records = self._require_records(dataset, model, approach)
        return records[0].em_score

    def improvement_over_baseline(
        self, dataset: str, model: str, baseline: str, approach: str
    ) -> float:
        """Return EM improvement (percentage points) of approach over baseline."""
        baseline_em = self._score_for(dataset, model, baseline)
        approach_em = self._score_for(dataset, model, approach)
        return approach_em - baseline_em


_COMPARISON_APPROACHES = ["naive_rag", "bge_reranker", "infogain_rag"]
_ABLATION_APPROACHES   = ["ce_only", "margin_only", "multi_task"]
_PALETTE = "Set2"


class ResultsVisualizer:
    """Produces seaborn charts matching the paper's figures."""

    def __init__(self, suite: EvalSuite):
        sns.set_theme(style="whitegrid", font_scale=1.1)
        self.suite = suite

    def _filter_by_dataset(self, df: pd.DataFrame, dataset: str) -> pd.DataFrame:
        return df[df["dataset"] == dataset]

    def _filter_by_approaches(self, df: pd.DataFrame, approaches: List[str]) -> pd.DataFrame:
        return df[df["approach"].isin(approaches)]

    def _dataset_records(self, dataset: str) -> pd.DataFrame:
        subset = self._filter_by_dataset(self.suite.to_dataframe(), dataset)
        if subset.empty:
            raise KeyError(f"dataset={dataset!r} not found in suite")
        return subset

    def _save_or_close(self, save_path: Optional[str]) -> None:
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()

    def _labeled_axes(self, figsize, title: str, xlabel: str, ylabel: str):
        _, ax = plt.subplots(figsize=figsize)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    def _apply_bar_labels(self, ax) -> None:
        for bar in ax.patches:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.005,
                    f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9,
                )

    def bar_chart_by_dataset(self, dataset: str, save_path: Optional[str] = None):
        """Figure 3 style: grouped bar chart comparing approaches on one dataset."""
        df = self._dataset_records(dataset)
        df = self._filter_by_approaches(df, _COMPARISON_APPROACHES)
        ax = self._labeled_axes((8, 5), f"EM Score by Approach — {dataset}", "Approach", "Exact Match")
        sns.barplot(data=df, x="approach", y="em_score", hue="model", palette=_PALETTE, ax=ax)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
        self._apply_bar_labels(ax)
        self._save_or_close(save_path)

    def _beta_series(self, records_by_beta: Dict[float, float]):
        betas = sorted(records_by_beta)
        return betas, [records_by_beta[b] for b in betas]

    def beta_sensitivity_plot(
        self, records_by_beta: Dict[float, float], model: str, save_path: Optional[str] = None
    ):
        """Figure 2 style: line plot of EM vs β hyper-parameter."""
        if not records_by_beta:
            raise ValueError("records_by_beta must not be empty")
        betas, scores = self._beta_series(records_by_beta)
        ax = self._labeled_axes((7, 4), f"β Sensitivity — {model}", "β", "Exact Match")
        color = sns.color_palette(_PALETTE)[0]
        ax.plot(betas, scores, marker="o", linewidth=2, markersize=7, color=color)
        ax.set_xticks(betas)
        ax.set_ylim(min(scores) - 0.02, max(scores) + 0.02)
        self._save_or_close(save_path)

    def ablation_bar_chart(
        self, dataset: str, save_path: Optional[str] = None
    ):
        """Table 3 style: CE-only vs Margin-only vs Multi-task grouped bar chart."""
        df = self._dataset_records(dataset)
        df = self._filter_by_approaches(df, _ABLATION_APPROACHES)
        ax = self._labeled_axes((7, 4), f"Ablation Study — {dataset}", "Approach", "Exact Match")
        sns.barplot(data=df, x="approach", y="em_score", hue="approach", palette=_PALETTE, legend=False, ax=ax)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
        self._apply_bar_labels(ax)
        self._save_or_close(save_path)
