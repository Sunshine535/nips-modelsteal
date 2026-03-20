"""
Active Query Selection for Progressive Parameter Inversion.

Selects inputs that maximally differentiate the student (with hypothesized
weights) from the teacher, thereby providing maximum information gain about
unknown parameters.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class QueryPool:
    """Maintains a pool of candidate inputs for active selection."""

    def __init__(
        self,
        tokenizer,
        pool_size: int = 10000,
        max_seq_len: int = 512,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.pool_size = pool_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self._pool: Optional[torch.Tensor] = None

    def build_from_dataset(self, dataset, text_column: str = "text") -> None:
        """Build query pool from a HuggingFace dataset."""
        input_ids_list = []
        for i, example in enumerate(dataset):
            if i >= self.pool_size:
                break
            text = example[text_column] if isinstance(example, dict) else str(example)
            tokens = self.tokenizer(
                text,
                max_length=self.max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids_list.append(tokens["input_ids"].squeeze(0))

        self._pool = torch.stack(input_ids_list)
        logger.info("Built query pool with %d samples", len(self._pool))

    def build_random(self, vocab_size: int) -> None:
        """Build random query pool from vocabulary."""
        self._pool = torch.randint(
            3,  # skip special tokens 0,1,2
            vocab_size,
            (self.pool_size, self.max_seq_len),
            generator=self.rng,
        )
        logger.info("Built random query pool: %s", self._pool.shape)

    @property
    def pool(self) -> torch.Tensor:
        if self._pool is None:
            raise RuntimeError("Query pool not initialized. Call build_*() first.")
        return self._pool

    def get_dataloader(self, batch_size: int = 64) -> DataLoader:
        return DataLoader(
            TensorDataset(self.pool),
            batch_size=batch_size,
            shuffle=False,
        )


class ActiveQuerySelector:
    """Selects maximally informative queries for parameter inversion."""

    def __init__(
        self,
        strategy: str = "gradient_magnitude",
        selection_batch: int = 64,
        device: str = "cuda",
    ):
        self.strategy = strategy
        self.selection_batch = selection_batch
        self.device = device

        self._strategies = {
            "random": self._select_random,
            "gradient_magnitude": self._select_gradient_magnitude,
            "fisher_information": self._select_fisher_information,
            "divergence": self._select_divergence,
        }

        if strategy not in self._strategies:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Options: {list(self._strategies)}"
            )

    def select(
        self,
        query_pool: QueryPool,
        student_model: nn.Module,
        teacher_fn,
        target_params: Optional[list] = None,
        n_select: int = 64,
    ) -> torch.Tensor:
        """
        Select n_select most informative queries from the pool.

        Args:
            query_pool: Pool of candidate input sequences.
            student_model: Current student model with hypothesized weights.
            teacher_fn: Callable that returns teacher logits for input_ids.
            target_params: Parameters being optimized (for gradient-based methods).
            n_select: Number of queries to select.

        Returns:
            Selected input_ids tensor of shape (n_select, seq_len).
        """
        return self._strategies[self.strategy](
            query_pool, student_model, teacher_fn, target_params, n_select
        )

    def _select_random(
        self, query_pool, student_model, teacher_fn, target_params, n_select
    ) -> torch.Tensor:
        indices = torch.randperm(len(query_pool.pool))[:n_select]
        return query_pool.pool[indices]

    @torch.no_grad()
    def _select_divergence(
        self, query_pool, student_model, teacher_fn, target_params, n_select
    ) -> torch.Tensor:
        """Select inputs where student and teacher diverge most."""
        scores = []
        student_model.eval()
        loader = query_pool.get_dataloader(batch_size=self.selection_batch)

        for (batch_ids,) in loader:
            batch_ids = batch_ids.to(self.device)
            student_logits = student_model(batch_ids).logits
            teacher_logits = teacher_fn(batch_ids)

            divergence = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="none",
            ).sum(dim=-1).mean(dim=-1)  # per-sample divergence

            scores.append(divergence.cpu())

        scores = torch.cat(scores)
        _, top_indices = scores.topk(min(n_select, len(scores)))
        return query_pool.pool[top_indices]

    def _select_gradient_magnitude(
        self, query_pool, student_model, teacher_fn, target_params, n_select
    ) -> torch.Tensor:
        """
        Select inputs that produce the largest gradient on target parameters.
        Gradient magnitude indicates information content about unknown weights.
        """
        if target_params is None:
            logger.warning("No target_params; falling back to divergence selection")
            return self._select_divergence(
                query_pool, student_model, teacher_fn, target_params, n_select
            )

        scores = []
        student_model.train()
        loader = query_pool.get_dataloader(batch_size=self.selection_batch)

        for (batch_ids,) in loader:
            batch_ids = batch_ids.to(self.device)
            batch_scores = []

            for i in range(batch_ids.size(0)):
                single_input = batch_ids[i : i + 1]
                student_logits = student_model(single_input).logits
                with torch.no_grad():
                    teacher_logits = teacher_fn(single_input)

                loss = F.mse_loss(student_logits, teacher_logits)
                student_model.zero_grad()
                loss.backward()

                grad_norm = 0.0
                for p in target_params:
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                batch_scores.append(grad_norm)

                student_model.zero_grad()

            scores.extend(batch_scores)

        scores = torch.tensor(scores)
        _, top_indices = scores.topk(min(n_select, len(scores)))
        return query_pool.pool[top_indices]

    def _select_fisher_information(
        self, query_pool, student_model, teacher_fn, target_params, n_select
    ) -> torch.Tensor:
        """
        Select inputs that maximize the Fisher information trace for target params.
        Approximated by the squared gradient (diagonal Fisher).
        """
        if target_params is None:
            return self._select_divergence(
                query_pool, student_model, teacher_fn, target_params, n_select
            )

        scores = []
        student_model.train()
        loader = query_pool.get_dataloader(batch_size=self.selection_batch)

        for (batch_ids,) in loader:
            batch_ids = batch_ids.to(self.device)
            batch_scores = []

            for i in range(batch_ids.size(0)):
                single_input = batch_ids[i : i + 1]
                student_logits = student_model(single_input).logits

                log_probs = F.log_softmax(student_logits, dim=-1)
                sampled_tokens = torch.multinomial(
                    F.softmax(student_logits[:, -1, :], dim=-1), 1
                )
                nll = -log_probs[:, -1, :].gather(1, sampled_tokens).sum()

                student_model.zero_grad()
                nll.backward()

                fisher_trace = 0.0
                for p in target_params:
                    if p.grad is not None:
                        fisher_trace += (p.grad.data ** 2).sum().item()

                batch_scores.append(fisher_trace)
                student_model.zero_grad()

            scores.extend(batch_scores)

        scores = torch.tensor(scores)
        _, top_indices = scores.topk(min(n_select, len(scores)))
        return query_pool.pool[top_indices]


def compute_query_information_gain(
    student_model: nn.Module,
    teacher_fn,
    input_ids: torch.Tensor,
    target_params: list,
) -> float:
    """
    Compute information gain for a single query.
    Measured as gradient norm w.r.t. target parameters.
    """
    student_model.train()
    student_logits = student_model(input_ids).logits
    with torch.no_grad():
        teacher_logits = teacher_fn(input_ids)

    loss = F.mse_loss(student_logits, teacher_logits)
    student_model.zero_grad()
    loss.backward()

    grad_norm = 0.0
    for p in target_params:
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2

    student_model.zero_grad()
    return grad_norm ** 0.5
