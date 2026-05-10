import os
import tempfile
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from src.reranker import MultiTaskReranker, RerankerTrainer, TrainingBatch


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_encoder_output(hidden: int, batch: int) -> MagicMock:
    output = MagicMock()
    output.pooler_output = torch.randn(batch, hidden)
    return output


def _make_mock_encoder(hidden: int = 32, batch: int = 2) -> MagicMock:
    encoder = MagicMock()
    encoder.return_value = _make_mock_encoder_output(hidden, batch)
    encoder.config = MagicMock()
    encoder.config.hidden_size = hidden
    return encoder


def make_reranker(hidden: int = 32, batch: int = 2) -> MultiTaskReranker:
    encoder = _make_mock_encoder(hidden=hidden, batch=batch)
    return MultiTaskReranker(encoder)


def make_batch(batch: int = 2, seq_len: int = 8, n_pos: int = 2, n_neg: int = 2) -> TrainingBatch:
    return TrainingBatch(
        query_doc_inputs={
            "input_ids": torch.ones(batch, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(batch, seq_len, dtype=torch.long),
        },
        ce_labels=torch.tensor([1.0, 0.0] * (batch // 2)),
        positive_scores=torch.randn(n_pos),
        negative_scores=torch.randn(n_neg),
    )


# ---------------------------------------------------------------------------
# MultiTaskReranker — init
# ---------------------------------------------------------------------------

class TestRerankerInit:
    def test_stores_encoder(self):
        # Guarantee: encoder passed at construction is accessible on the reranker
        encoder = _make_mock_encoder()
        r = MultiTaskReranker(encoder)
        assert r.encoder is encoder

    def test_has_score_head(self):
        # Guarantee: reranker has a linear projection layer to produce a scalar score
        r = make_reranker()
        assert hasattr(r, "score_head")
        assert isinstance(r.score_head, nn.Linear)

    def test_score_head_output_dim_is_one(self):
        # Guarantee: score_head outputs a single relevance score per document
        r = make_reranker(hidden=32)
        assert r.score_head.out_features == 1

    def test_hyperparams_match_paper(self):
        # Guarantee: default hyperparameters match paper Section 4.1 values
        r = make_reranker()
        assert r.BETA == 0.75
        assert r.GAMMA == 15.0
        assert r.LR == 5e-6


# ---------------------------------------------------------------------------
# MultiTaskReranker — forward
# ---------------------------------------------------------------------------

class TestForward:
    def test_returns_tensor(self):
        # Guarantee: forward always returns a torch.Tensor
        r = make_reranker(batch=2)
        out = r(torch.ones(2, 8, dtype=torch.long), torch.ones(2, 8, dtype=torch.long))
        assert isinstance(out, torch.Tensor)

    def test_output_shape_is_batch(self):
        # Guarantee: one score is returned per (query, document) pair in the batch
        r = make_reranker(batch=4)
        out = r(torch.ones(4, 8, dtype=torch.long), torch.ones(4, 8, dtype=torch.long))
        assert out.shape == (4,)

    def test_output_is_scalar_per_sample(self):
        # Guarantee: each score is a plain scalar (no extra dimensions)
        r = make_reranker(batch=2)
        out = r(torch.ones(2, 8, dtype=torch.long), torch.ones(2, 8, dtype=torch.long))
        assert out.dim() == 1


# ---------------------------------------------------------------------------
# ce_loss  (eq 4)
# ---------------------------------------------------------------------------

class TestCELoss:
    def test_returns_scalar_tensor(self):
        # Guarantee: CE loss returns a scalar (0-dim) tensor
        r = make_reranker()
        scores = torch.tensor([0.8, 0.2])
        labels = torch.tensor([1.0, 0.0])
        loss = r.ce_loss(scores, labels)
        assert loss.dim() == 0

    def test_loss_is_non_negative(self):
        # Guarantee: CE loss is always ≥ 0
        r = make_reranker()
        scores = torch.randn(4)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert r.ce_loss(scores, labels).item() >= 0.0

    def test_perfect_prediction_gives_low_loss(self):
        # Guarantee: highly confident correct predictions yield near-zero CE loss
        r = make_reranker()
        scores = torch.tensor([10.0, -10.0])   # sigmoid → 1.0, 0.0
        labels = torch.tensor([1.0, 0.0])
        assert r.ce_loss(scores, labels).item() < 0.01

    def test_wrong_prediction_gives_high_loss(self):
        # Guarantee: confidently wrong predictions yield high CE loss
        r = make_reranker()
        scores = torch.tensor([-10.0, 10.0])   # sigmoid → 0.0, 1.0
        labels = torch.tensor([1.0, 0.0])
        assert r.ce_loss(scores, labels).item() > 5.0


# ---------------------------------------------------------------------------
# _softplus  (eq 8)
# ---------------------------------------------------------------------------

class TestSoftplus:
    def test_positive_input_close_to_input(self):
        # Guarantee: softplus(x) ≈ x for large positive x (smoothed ReLU behaviour)
        r = make_reranker()
        x = torch.tensor(20.0)
        assert abs(r._softplus(x).item() - 20.0) < 0.01

    def test_zero_input_gives_log2(self):
        # Guarantee: softplus(0) = log(2) ≈ 0.693
        r = make_reranker()
        result = r._softplus(torch.tensor(0.0)).item()
        assert abs(result - 0.6931) < 1e-3

    def test_large_negative_near_zero(self):
        # Guarantee: softplus(x) ≈ 0 for large negative x
        r = make_reranker()
        result = r._softplus(torch.tensor(-20.0)).item()
        assert result < 0.01


# ---------------------------------------------------------------------------
# margin_loss  (eq 7)
# ---------------------------------------------------------------------------

class TestMarginLoss:
    def test_returns_scalar_tensor(self):
        # Guarantee: margin loss returns a scalar (0-dim) tensor
        r = make_reranker()
        pos = torch.tensor([0.9, 0.8])
        neg = torch.tensor([0.1, 0.2])
        assert r.margin_loss(pos, neg).dim() == 0

    def test_loss_is_non_negative(self):
        # Guarantee: margin loss is always ≥ 0
        r = make_reranker()
        pos = torch.randn(3)
        neg = torch.randn(3)
        assert r.margin_loss(pos, neg).item() >= 0.0

    def test_well_separated_scores_give_low_loss(self):
        # Guarantee: positives scoring much higher than negatives yield near-zero margin loss
        r = make_reranker()
        pos = torch.tensor([5.0, 5.0])
        neg = torch.tensor([-5.0, -5.0])
        assert r.margin_loss(pos, neg).item() < 0.01

    def test_inverted_scores_give_high_loss(self):
        # Guarantee: negatives scoring higher than positives yield large margin loss
        r = make_reranker()
        pos = torch.tensor([-5.0, -5.0])
        neg = torch.tensor([5.0, 5.0])
        assert r.margin_loss(pos, neg).item() > 1.0


# ---------------------------------------------------------------------------
# combined_loss  (eq 9)
# ---------------------------------------------------------------------------

class TestCombinedLoss:
    def test_returns_scalar_tensor(self):
        # Guarantee: combined loss returns a scalar (0-dim) tensor
        r = make_reranker()
        scores = torch.tensor([0.8, 0.2])
        labels = torch.tensor([1.0, 0.0])
        loss = r.combined_loss(scores, labels, scores[:1], scores[1:])
        assert loss.dim() == 0

    def test_is_weighted_sum_of_ce_and_margin(self):
        # Guarantee: combined loss equals β·CE + (1-β)·Margin exactly
        r = make_reranker()
        scores = torch.tensor([0.8, 0.2])
        labels = torch.tensor([1.0, 0.0])
        pos = scores[:1]
        neg = scores[1:]
        l_ce = r.ce_loss(scores, labels)
        l_margin = r.margin_loss(pos, neg)
        expected = r.BETA * l_ce + (1 - r.BETA) * l_margin
        result = r.combined_loss(scores, labels, pos, neg)
        assert abs(result.item() - expected.item()) < 1e-5

    def test_loss_is_non_negative(self):
        # Guarantee: combined loss is always ≥ 0
        r = make_reranker()
        scores = torch.randn(4)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert r.combined_loss(scores, labels, scores[:2], scores[2:]).item() >= 0.0


# ---------------------------------------------------------------------------
# RerankerTrainer — init
# ---------------------------------------------------------------------------

class TestTrainerInit:
    def test_stores_reranker(self):
        # Guarantee: trainer holds a reference to the reranker it was given
        r = make_reranker()
        trainer = RerankerTrainer(r, device="cpu")
        assert trainer.reranker is r

    def test_has_optimizer(self):
        # Guarantee: trainer creates an Adam optimizer on construction
        r = make_reranker()
        trainer = RerankerTrainer(r, device="cpu")
        assert trainer.optimizer is not None

    def test_optimizer_lr_matches_paper(self):
        # Guarantee: optimizer learning rate matches paper value 5e-6
        r = make_reranker()
        trainer = RerankerTrainer(r, device="cpu")
        lr = trainer.optimizer.param_groups[0]["lr"]
        assert abs(lr - MultiTaskReranker.LR) < 1e-10


# ---------------------------------------------------------------------------
# RerankerTrainer — train_step
# ---------------------------------------------------------------------------

class TestTrainStep:
    def test_returns_dict_with_loss_keys(self):
        # Guarantee: train_step returns a dict containing ce_loss, margin_loss, total_loss
        r = make_reranker(batch=2)
        trainer = RerankerTrainer(r, device="cpu")
        result = trainer.train_step(make_batch(batch=2))
        assert {"ce_loss", "margin_loss", "total_loss"} <= result.keys()

    def test_losses_are_floats(self):
        # Guarantee: all reported loss values are plain Python floats
        r = make_reranker(batch=2)
        trainer = RerankerTrainer(r, device="cpu")
        result = trainer.train_step(make_batch(batch=2))
        for key in ("ce_loss", "margin_loss", "total_loss"):
            assert isinstance(result[key], float)

    def test_total_loss_matches_formula(self):
        # Guarantee: reported total_loss equals β·ce_loss + (1-β)·margin_loss
        r = make_reranker(batch=2)
        trainer = RerankerTrainer(r, device="cpu")
        result = trainer.train_step(make_batch(batch=2))
        expected = r.BETA * result["ce_loss"] + (1 - r.BETA) * result["margin_loss"]
        assert abs(result["total_loss"] - expected) < 1e-5

    def test_step_updates_parameters(self):
        # Guarantee: model parameters change after a train step (gradients applied)
        r = make_reranker(batch=2)
        params_before = [p.clone() for p in r.parameters()]
        trainer = RerankerTrainer(r, device="cpu")
        trainer.train_step(make_batch(batch=2))
        params_after = list(r.parameters())
        changed = any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
        assert changed


# ---------------------------------------------------------------------------
# RerankerTrainer — save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_file(self):
        # Guarantee: save() writes a file at the given path
        r = make_reranker()
        trainer = RerankerTrainer(r, device="cpu")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reranker.pt")
            trainer.save(path)
            assert os.path.exists(path)

    def test_load_restores_weights(self):
        # Guarantee: load() restores exactly the weights that were saved
        r = make_reranker(hidden=32, batch=2)
        trainer = RerankerTrainer(r, device="cpu")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reranker.pt")
            trainer.save(path)
            original_weights = {k: v.clone() for k, v in r.state_dict().items()}
            # Mutate weights
            for p in r.parameters():
                p.data.fill_(999.0)
            trainer.load(path)
            for k, v in r.state_dict().items():
                assert torch.allclose(v, original_weights[k])
