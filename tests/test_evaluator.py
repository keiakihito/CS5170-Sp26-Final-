import os
import tempfile

import pandas as pd
import pytest

from src.evaluator import EvalRecord, EvalSuite, ResultsVisualizer


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_record(
    model: str = "Qwen2.5-7B",
    dataset: str = "TriviaQA",
    approach: str = "infogain_rag",
    em_score: float = 0.72,
    n_samples: int = 1000,
) -> EvalRecord:
    return EvalRecord(
        model=model,
        dataset=dataset,
        approach=approach,
        em_score=em_score,
        n_samples=n_samples,
    )


def make_suite_with_records() -> EvalSuite:
    suite = EvalSuite()
    suite.add(make_record(approach="naive_rag",     em_score=0.529))
    suite.add(make_record(approach="bge_reranker",  em_score=0.670))
    suite.add(make_record(approach="infogain_rag",  em_score=0.720))
    return suite


def make_multi_dataset_suite() -> EvalSuite:
    suite = EvalSuite()
    for dataset in ("TriviaQA", "NaturalQA"):
        for approach, score in [("naive_rag", 0.50), ("infogain_rag", 0.72)]:
            suite.add(make_record(dataset=dataset, approach=approach, em_score=score))
    return suite


# ---------------------------------------------------------------------------
# EvalRecord
# ---------------------------------------------------------------------------

class TestEvalRecord:
    def test_stores_all_fields(self):
        # Guarantee: EvalRecord stores model, dataset, approach, em_score, n_samples
        r = make_record()
        assert r.model == "Qwen2.5-7B"
        assert r.dataset == "TriviaQA"
        assert r.approach == "infogain_rag"
        assert r.em_score == 0.72
        assert r.n_samples == 1000

    def test_em_score_is_float(self):
        # Guarantee: em_score is stored as a float
        r = make_record(em_score=0.72)
        assert isinstance(r.em_score, float)


# ---------------------------------------------------------------------------
# EvalSuite.add
# ---------------------------------------------------------------------------

class TestEvalSuiteAdd:
    def test_add_increases_record_count(self):
        # Guarantee: each add() call appends exactly one record to the suite
        suite = EvalSuite()
        suite.add(make_record())
        assert len(suite.records) == 1

    def test_add_multiple_records(self):
        # Guarantee: multiple add() calls accumulate all records in order
        suite = EvalSuite()
        suite.add(make_record(approach="naive_rag"))
        suite.add(make_record(approach="infogain_rag"))
        assert len(suite.records) == 2

    def test_empty_suite_has_no_records(self):
        # Guarantee: a freshly created suite starts with zero records
        assert len(EvalSuite().records) == 0


# ---------------------------------------------------------------------------
# EvalSuite.to_dataframe
# ---------------------------------------------------------------------------

class TestToDataframe:
    def test_returns_dataframe(self):
        # Guarantee: to_dataframe() returns a pandas DataFrame
        suite = make_suite_with_records()
        assert isinstance(suite.to_dataframe(), pd.DataFrame)

    def test_row_count_matches_records(self):
        # Guarantee: DataFrame has one row per EvalRecord
        suite = make_suite_with_records()
        assert len(suite.to_dataframe()) == len(suite.records)

    def test_has_required_columns(self):
        # Guarantee: DataFrame contains model, dataset, approach, em_score, n_samples columns
        suite = make_suite_with_records()
        df = suite.to_dataframe()
        assert {"model", "dataset", "approach", "em_score", "n_samples"} <= set(df.columns)

    def test_em_score_values_preserved(self):
        # Guarantee: em_score values in the DataFrame match the original records
        suite = make_suite_with_records()
        df = suite.to_dataframe()
        assert set(df["em_score"]) == {r.em_score for r in suite.records}

    def test_empty_suite_returns_empty_dataframe(self):
        # Guarantee: an empty suite produces an empty DataFrame without error
        df = EvalSuite().to_dataframe()
        assert isinstance(df, pd.DataFrame) and len(df) == 0


# ---------------------------------------------------------------------------
# EvalSuite.best_approach
# ---------------------------------------------------------------------------

class TestBestApproach:
    def test_returns_highest_em_approach(self):
        # Guarantee: best_approach returns the approach with the highest EM score
        suite = make_suite_with_records()
        assert suite.best_approach("TriviaQA", "Qwen2.5-7B") == "infogain_rag"

    def test_returns_string(self):
        # Guarantee: best_approach always returns a string
        suite = make_suite_with_records()
        assert isinstance(suite.best_approach("TriviaQA", "Qwen2.5-7B"), str)

    def test_unknown_dataset_raises(self):
        # Guarantee: querying a dataset not in the suite raises KeyError
        suite = make_suite_with_records()
        with pytest.raises(KeyError):
            suite.best_approach("PopQA", "Qwen2.5-7B")

    def test_unknown_model_raises(self):
        # Guarantee: querying a model not in the suite raises KeyError
        suite = make_suite_with_records()
        with pytest.raises(KeyError):
            suite.best_approach("TriviaQA", "LLaMA3.1-8B")


# ---------------------------------------------------------------------------
# EvalSuite.improvement_over_baseline
# ---------------------------------------------------------------------------

class TestImprovementOverBaseline:
    def test_returns_float(self):
        # Guarantee: improvement_over_baseline always returns a float
        suite = make_suite_with_records()
        result = suite.improvement_over_baseline(
            "TriviaQA", "Qwen2.5-7B", "naive_rag", "infogain_rag"
        )
        assert isinstance(result, float)

    def test_correct_improvement_value(self):
        # Guarantee: improvement equals approach_em - baseline_em in percentage points
        suite = make_suite_with_records()
        result = suite.improvement_over_baseline(
            "TriviaQA", "Qwen2.5-7B", "naive_rag", "infogain_rag"
        )
        assert abs(result - (0.720 - 0.529)) < 1e-6

    def test_negative_improvement_when_baseline_is_better(self):
        # Guarantee: negative value is returned when approach underperforms the baseline
        suite = EvalSuite()
        suite.add(make_record(approach="naive_rag",    em_score=0.80))
        suite.add(make_record(approach="infogain_rag", em_score=0.60))
        result = suite.improvement_over_baseline(
            "TriviaQA", "Qwen2.5-7B", "naive_rag", "infogain_rag"
        )
        assert result < 0.0

    def test_unknown_approach_raises(self):
        # Guarantee: querying an approach not in the suite raises KeyError
        suite = make_suite_with_records()
        with pytest.raises(KeyError):
            suite.improvement_over_baseline(
                "TriviaQA", "Qwen2.5-7B", "naive_rag", "gte_reranker"
            )


# ---------------------------------------------------------------------------
# ResultsVisualizer — init
# ---------------------------------------------------------------------------

class TestVisualizerInit:
    def test_stores_suite(self):
        # Guarantee: visualizer retains the EvalSuite passed at construction
        suite = make_suite_with_records()
        viz = ResultsVisualizer(suite)
        assert viz.suite is suite


# ---------------------------------------------------------------------------
# ResultsVisualizer — bar_chart_by_dataset
# ---------------------------------------------------------------------------

class TestBarChartByDataset:
    def test_runs_without_error(self):
        # Guarantee: bar_chart_by_dataset completes without raising for valid dataset
        suite = make_suite_with_records()
        viz = ResultsVisualizer(suite)
        viz.bar_chart_by_dataset("TriviaQA")

    def test_saves_file_when_path_given(self):
        # Guarantee: a PNG file is written when save_path is provided
        suite = make_suite_with_records()
        viz = ResultsVisualizer(suite)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bar.png")
            viz.bar_chart_by_dataset("TriviaQA", save_path=path)
            assert os.path.exists(path)

    def test_unknown_dataset_raises(self):
        # Guarantee: requesting a chart for an unknown dataset raises KeyError
        suite = make_suite_with_records()
        viz = ResultsVisualizer(suite)
        with pytest.raises(KeyError):
            viz.bar_chart_by_dataset("PopQA")


# ---------------------------------------------------------------------------
# ResultsVisualizer — beta_sensitivity_plot
# ---------------------------------------------------------------------------

class TestBetaSensitivityPlot:
    def test_runs_without_error(self):
        # Guarantee: beta_sensitivity_plot completes without raising for valid input
        suite = make_suite_with_records()
        viz = ResultsVisualizer(suite)
        beta_records = {0.0: 0.66, 0.25: 0.68, 0.5: 0.70, 0.75: 0.72, 1.0: 0.69}
        viz.beta_sensitivity_plot(beta_records, model="Qwen2.5-7B")

    def test_saves_file_when_path_given(self):
        # Guarantee: a PNG file is written when save_path is provided
        suite = make_suite_with_records()
        viz = ResultsVisualizer(suite)
        beta_records = {0.0: 0.66, 0.5: 0.70, 1.0: 0.69}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "beta.png")
            viz.beta_sensitivity_plot(beta_records, model="Qwen2.5-7B", save_path=path)
            assert os.path.exists(path)

    def test_empty_beta_records_raises(self):
        # Guarantee: empty beta_records dict raises ValueError
        suite = make_suite_with_records()
        viz = ResultsVisualizer(suite)
        with pytest.raises(ValueError):
            viz.beta_sensitivity_plot({}, model="Qwen2.5-7B")


# ---------------------------------------------------------------------------
# ResultsVisualizer — ablation_bar_chart
# ---------------------------------------------------------------------------

class TestAblationBarChart:
    def test_runs_without_error(self):
        # Guarantee: ablation_bar_chart completes without raising for valid input
        suite = EvalSuite()
        for approach, score in [("ce_only", 0.67), ("margin_only", 0.68), ("multi_task", 0.72)]:
            suite.add(make_record(approach=approach, em_score=score))
        viz = ResultsVisualizer(suite)
        viz.ablation_bar_chart("TriviaQA")

    def test_saves_file_when_path_given(self):
        # Guarantee: a PNG file is written when save_path is provided
        suite = EvalSuite()
        for approach, score in [("ce_only", 0.67), ("margin_only", 0.68), ("multi_task", 0.72)]:
            suite.add(make_record(approach=approach, em_score=score))
        viz = ResultsVisualizer(suite)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ablation.png")
            viz.ablation_bar_chart("TriviaQA", save_path=path)
            assert os.path.exists(path)
