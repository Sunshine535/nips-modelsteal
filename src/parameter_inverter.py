"""
Core module for Progressive Layer-wise Parameter Inversion (PLPI).

Given black-box query access to a teacher model (logits only), iteratively
recover the teacher's weight parameters layer by layer, starting from the
output projection and working inward.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .active_query import ActiveQuerySelector, QueryPool

logger = logging.getLogger(__name__)


@dataclass
class InversionConfig:
    query_budget: int = 500000
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_steps_per_layer: int = 10000
    convergence_threshold: float = 1e-6
    optimizer_type: str = "adam"
    regularization_lambda: float = 1e-4
    regularization_type: str = "l2"
    active_query_strategy: str = "gradient_magnitude"
    active_query_pool_size: int = 10000
    selection_batch: int = 64
    log_every: int = 100
    save_every: int = 1000
    eval_every: int = 500
    seed: int = 42


@dataclass
class LayerInversionResult:
    layer_name: str
    cosine_similarity: float
    l2_distance: float
    final_loss: float
    num_steps: int
    num_queries_used: int
    converged: bool


@dataclass
class InversionResult:
    layer_results: list = field(default_factory=list)
    total_queries: int = 0
    recovered_state_dict: Optional[dict] = None


class BlackBoxTeacher:
    """Wraps a model to expose only logit-level access (simulates API)."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        defense_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.model.eval()
        self.device = device
        self.defense_fn = defense_fn
        self.query_count = 0

    @torch.no_grad()
    def query(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits only — black-box access."""
        input_ids = input_ids.to(self.device)
        logits = self.model(input_ids).logits
        self.query_count += input_ids.size(0)

        if self.defense_fn is not None:
            logits = self.defense_fn(logits)

        return logits


class LayerWiseInverter:
    """
    Recovers teacher weight parameters one layer at a time.

    Strategy:
    1. Start from the output projection (lm_head) — easiest to invert
       because it directly maps to observed logits.
    2. Fix recovered layers, optimize the next deeper layer.
    3. Use active query selection to maximize information gain per query.
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher: BlackBoxTeacher,
        config: InversionConfig,
        device: str = "cuda",
    ):
        self.student = student_model
        self.teacher = teacher
        self.config = config
        self.device = device

        self.query_selector = ActiveQuerySelector(
            strategy=config.active_query_strategy,
            selection_batch=config.selection_batch,
            device=device,
        )

        self.query_pool: Optional[QueryPool] = None
        self._recovered_layers: dict[str, torch.Tensor] = {}

    def set_query_pool(self, query_pool: QueryPool) -> None:
        self.query_pool = query_pool

    def get_invertible_layers(self) -> list[tuple[str, nn.Parameter]]:
        """
        Return layers in inversion order (output → input).
        For a transformer LM: lm_head → last decoder block → ... → first block.

        Handles weight-tied models where lm_head.weight shares its tensor
        with the embedding layer (the shared param only appears once in
        named_parameters under the embedding name).
        """
        layers = []

        lm_head_ptrs: set[int] = set()
        for mod_name, module in self.student.named_modules():
            if "lm_head" in mod_name:
                for p in module.parameters():
                    lm_head_ptrs.add(p.data_ptr())

        for name, param in self.student.named_parameters():
            if "lm_head" in name or param.data_ptr() in lm_head_ptrs:
                layers.append((name, param))

        added_ptrs = {p.data_ptr() for _, p in layers}

        block_params: dict[int, list] = {}
        for name, param in self.student.named_parameters():
            if param.data_ptr() in added_ptrs:
                continue
            if "layers." in name or "h." in name or "blocks." in name:
                for part in name.split("."):
                    if part.isdigit():
                        block_idx = int(part)
                        if block_idx not in block_params:
                            block_params[block_idx] = []
                        block_params[block_idx].append((name, param))
                        added_ptrs.add(param.data_ptr())
                        break

        for block_idx in sorted(block_params.keys(), reverse=True):
            layers.extend(block_params[block_idx])

        for name, param in self.student.named_parameters():
            if param.data_ptr() not in added_ptrs and "embed" in name:
                layers.append((name, param))

        return layers

    def invert_layer(
        self,
        layer_names: list[str],
        target_params: list[nn.Parameter],
        teacher_ground_truth: Optional[dict] = None,
    ) -> LayerInversionResult:
        """
        Invert a single layer (or group of related parameters).

        Freezes all other parameters, optimizes target_params to minimize
        the difference between student and teacher logits.
        """
        if not layer_names:
            logger.info("No parameters to invert — returning empty result")
            return LayerInversionResult(
                layer_name="(empty)",
                cosine_similarity=-1.0,
                l2_distance=-1.0,
                final_loss=0.0,
                num_steps=0,
                num_queries_used=0,
                converged=True,
            )

        for name, param in self.student.named_parameters():
            param.requires_grad = name in layer_names

        trainable = [p for p in self.student.parameters() if p.requires_grad]
        logger.info(
            "Inverting %d parameters across %d tensors: %s",
            sum(p.numel() for p in trainable),
            len(trainable),
            layer_names[:3],
        )

        optimizer = self._build_optimizer(trainable)
        total_queries = 0
        best_loss = float("inf")
        steps_without_improvement = 0

        for step in range(self.config.max_steps_per_layer):
            if total_queries >= self.config.query_budget:
                logger.info("Query budget exhausted at step %d", step)
                break

            if self.query_pool is not None and self.config.active_query_strategy != "random":
                input_ids = self.query_selector.select(
                    self.query_pool,
                    self.student,
                    self.teacher.query,
                    target_params=trainable,
                    n_select=self.config.batch_size,
                )
            else:
                input_ids = self._get_random_batch()

            input_ids = input_ids.to(self.device)
            teacher_logits = self.teacher.query(input_ids)
            total_queries += input_ids.size(0)

            student_logits = self.student(input_ids).logits
            loss = F.mse_loss(student_logits, teacher_logits)

            if self.config.regularization_lambda > 0:
                reg = self._compute_regularization(trainable)
                loss = loss + self.config.regularization_lambda * reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss - self.config.convergence_threshold:
                best_loss = loss_val
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            if step % self.config.log_every == 0:
                cos_sim = -1.0
                if teacher_ground_truth is not None:
                    cos_sim = self._compute_layer_cosine_sim(
                        layer_names, teacher_ground_truth
                    )
                logger.info(
                    "Step %d | loss=%.6f | best=%.6f | cos_sim=%.4f | queries=%d",
                    step, loss_val, best_loss, cos_sim, total_queries,
                )

            if steps_without_improvement > 500:
                logger.info("Converged at step %d (500 steps without improvement)", step)
                break

        cos_sim = -1.0
        l2_dist = -1.0
        if teacher_ground_truth is not None:
            cos_sim = self._compute_layer_cosine_sim(layer_names, teacher_ground_truth)
            l2_dist = self._compute_layer_l2(layer_names, teacher_ground_truth)

        for name in layer_names:
            for n, p in self.student.named_parameters():
                if n == name:
                    self._recovered_layers[name] = p.data.clone()

        return LayerInversionResult(
            layer_name=",".join(layer_names[:3]),
            cosine_similarity=cos_sim,
            l2_distance=l2_dist,
            final_loss=best_loss,
            num_steps=step + 1,
            num_queries_used=total_queries,
            converged=steps_without_improvement > 500,
        )

    def run_progressive_inversion(
        self,
        teacher_ground_truth: Optional[dict] = None,
        output_dir: Optional[str] = None,
    ) -> InversionResult:
        """Run full progressive inversion pipeline."""
        result = InversionResult()
        invertible = self.get_invertible_layers()

        current_block = None
        block_names = []
        block_params = []

        def _flush_block():
            nonlocal block_names, block_params
            if not block_names:
                return
            layer_result = self.invert_layer(
                block_names, block_params, teacher_ground_truth
            )
            result.layer_results.append(layer_result)
            result.total_queries += layer_result.num_queries_used
            logger.info(
                "Layer %s: cos_sim=%.4f, loss=%.6f, queries=%d",
                layer_result.layer_name,
                layer_result.cosine_similarity,
                layer_result.final_loss,
                layer_result.num_queries_used,
            )
            block_names = []
            block_params = []

        for name, param in invertible:
            block_id = self._get_block_id(name)
            if block_id != current_block:
                _flush_block()
                current_block = block_id

            block_names.append(name)
            block_params.append(param)

        _flush_block()

        result.recovered_state_dict = {
            k: v.cpu() for k, v in self._recovered_layers.items()
        }

        if output_dir is not None:
            self._save_results(result, output_dir)

        return result

    def _build_optimizer(self, params: list) -> torch.optim.Optimizer:
        if self.config.optimizer_type == "adam":
            return torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )
        elif self.config.optimizer_type == "lbfgs":
            return torch.optim.LBFGS(params, lr=self.config.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer_type}")

    def _compute_regularization(self, params: list) -> torch.Tensor:
        reg = torch.tensor(0.0, device=self.device)
        if self.config.regularization_type == "l2":
            for p in params:
                reg = reg + p.norm(2) ** 2
        elif self.config.regularization_type == "l1":
            for p in params:
                reg = reg + p.norm(1)
        return reg

    def _compute_layer_cosine_sim(
        self, layer_names: list[str], ground_truth: dict
    ) -> float:
        sims = []
        for name in layer_names:
            if name not in ground_truth:
                continue
            for n, p in self.student.named_parameters():
                if n == name:
                    gt = ground_truth[name].to(p.device).flatten().float()
                    pred = p.data.flatten().float()
                    sim = F.cosine_similarity(pred.unsqueeze(0), gt.unsqueeze(0)).item()
                    sims.append(sim)
                    break
        return sum(sims) / len(sims) if sims else -1.0

    def _compute_layer_l2(
        self, layer_names: list[str], ground_truth: dict
    ) -> float:
        dists = []
        for name in layer_names:
            if name not in ground_truth:
                continue
            for n, p in self.student.named_parameters():
                if n == name:
                    gt = ground_truth[name].to(p.device).float()
                    pred = p.data.float()
                    dists.append((pred - gt).norm(2).item())
                    break
        return sum(dists) / len(dists) if dists else -1.0

    @staticmethod
    def _get_block_id(name: str) -> str:
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part.isdigit() and i > 0:
                return ".".join(parts[: i + 1])
        if "lm_head" in name:
            return "lm_head"
        if "embed" in name:
            return "embed"
        return name

    def _get_random_batch(self) -> torch.Tensor:
        vocab_size = self.student.config.vocab_size
        return torch.randint(3, vocab_size, (self.config.batch_size, 128))

    def _save_results(self, result: InversionResult, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if result.recovered_state_dict:
            torch.save(result.recovered_state_dict, out / "recovered_weights.pt")

        import json
        summary = {
            "total_queries": result.total_queries,
            "layers": [
                {
                    "name": lr.layer_name,
                    "cosine_similarity": lr.cosine_similarity,
                    "l2_distance": lr.l2_distance,
                    "final_loss": lr.final_loss,
                    "num_steps": lr.num_steps,
                    "num_queries_used": lr.num_queries_used,
                    "converged": lr.converged,
                }
                for lr in result.layer_results
            ],
        }
        with open(out / "inversion_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Saved inversion results to %s", out)
