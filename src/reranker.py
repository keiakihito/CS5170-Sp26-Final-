from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainingBatch:
    """One batch for the multi-task reranker training loop."""
    query_doc_inputs: dict        # tokenized (query, document) pairs
    ce_labels: torch.Tensor       # binary labels for CE loss: 1=positive, 0=negative
    positive_scores: torch.Tensor # reranker scores for positive docs (margin loss)
    negative_scores: torch.Tensor # reranker scores for negative docs (margin loss)


class MultiTaskReranker(nn.Module):
    """Step 3: RoBERTa-large reranker trained with CE loss + Margin loss (eqs 4, 7, 9)."""

    # Paper hyperparameters (Section 4.1)
    BETA = 0.75    # β — weight balancing CE vs Margin loss
    GAMMA = 15.0   # γ — scaling factor in margin loss
    LR = 5e-6      # learning rate for Adam optimizer

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.score_head = nn.Linear(encoder.config.hidden_size, 1)

    # -- forward --

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return output.pooler_output  # (batch, hidden)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return relevance score for each (query, document) pair in the batch."""
        pooled = self._encode(input_ids, attention_mask)
        return self.score_head(pooled).squeeze(-1)  # (batch,)

    # -- eq 4: CE loss --

    def ce_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute eq 4: binary cross-entropy loss for document relevance classification."""
        return F.binary_cross_entropy_with_logits(scores, labels)

    # -- eq 8: softplus --

    def _softplus(self, x: torch.Tensor) -> torch.Tensor:
        """Eq 8: softplus(x) = log(1 + exp(x)) to smooth ReLU in margin loss."""
        return F.softplus(x)

    # -- eq 7: margin loss --

    def _pairwise_gaps(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        # All (neg - pos) pairs: shape (K, L) where K=|neg|, L=|pos|
        return neg_scores.unsqueeze(1) - pos_scores.unsqueeze(0)  # (K, L)

    def _logsumexp_margin(self, gaps: torch.Tensor) -> torch.Tensor:
        return torch.log(1 + (self.GAMMA * gaps).exp().sum())

    def margin_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """Compute eq 7: LogSumExp-approximated margin loss for document ranking."""
        gaps = self._pairwise_gaps(pos_scores, neg_scores)
        return self._logsumexp_margin(gaps)

    # -- eq 9: combined loss --

    def combined_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute eq 9: L_total = β·L_CE + (1-β)·L_Margin."""
        l_ce = self.ce_loss(scores, labels)
        l_margin = self.margin_loss(pos_scores, neg_scores)
        return self.BETA * l_ce + (1 - self.BETA) * l_margin


class RerankerTrainer:
    """Wraps MultiTaskReranker with an optimizer and a training step."""

    def __init__(self, reranker: MultiTaskReranker, device: str = "cpu"):
        self.reranker = reranker
        self.device = device
        self.optimizer = torch.optim.Adam(reranker.parameters(), lr=MultiTaskReranker.LR)

    # -- training step helpers --

    def _forward_scores(self, batch: TrainingBatch) -> torch.Tensor:
        inputs = batch.query_doc_inputs
        return self.reranker(inputs["input_ids"], inputs["attention_mask"])

    def _compute_losses(
        self, scores: torch.Tensor, batch: TrainingBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_ce = self.reranker.ce_loss(scores, batch.ce_labels)
        l_margin = self.reranker.margin_loss(batch.positive_scores, batch.negative_scores)
        l_total = self.reranker.combined_loss(
            scores, batch.ce_labels, batch.positive_scores, batch.negative_scores
        )
        return l_ce, l_margin, l_total

    def _step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_step(self, batch: TrainingBatch) -> dict:
        """Run one forward+backward pass; return dict with loss values."""
        self.reranker.train()
        scores = self._forward_scores(batch)
        l_ce, l_margin, l_total = self._compute_losses(scores, batch)
        self._step(l_total)
        return {
            "ce_loss": l_ce.item(),
            "margin_loss": l_margin.item(),
            "total_loss": l_total.item(),
        }

    # -- persistence --

    def save(self, path: str) -> None:
        """Save model weights to path."""
        torch.save(self.reranker.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model weights from path."""
        self.reranker.load_state_dict(torch.load(path, weights_only=True))
