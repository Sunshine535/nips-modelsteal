"""
Sensitivity-Guided Progressive Suffix Inversion (S-PSI).

Recovers transformer suffix parameters (lm_head + last K blocks) from
black-box logit access.  Two regimes are supported:

  - **Oracle-prefix**: teacher boundary states h_{i-1}^T(x) are injected,
    isolating suffix identifiability.
  - **Pure-logits**: student prefix is frozen random init — realistic attack.

Core loss per block:
  L = α·L_logit + β·L_sensitivity + γ·L_reg
where L_sensitivity matches how logits respond to token perturbations.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SPSIConfig:
    """Configuration for S-PSI inversion."""

    query_budget: int = 500_000
    batch_size: int = 16
    learning_rate: float = 1e-4
    lm_head_lr: float = 1e-3
    weight_decay: float = 0.0
    max_steps_per_block: int = 10_000
    lm_head_steps: int = 5_000
    convergence_threshold: float = 1e-7
    patience: int = 500

    alpha: float = 1.0
    beta: float = 0.1
    gamma: float = 1e-5

    num_perturbation_positions: int = 4
    num_replacement_tokens: int = 2
    max_seq_len: int = 256

    suffix_refine_enabled: bool = False
    suffix_refine_interval: int = 2000
    suffix_refine_steps: int = 200
    suffix_refine_lr: float = 1e-5

    log_every: int = 100
    save_every: int = 2000
    seed: int = 42


@dataclass
class BlockResult:
    """Result from inverting a single block."""

    block_name: str
    per_matrix_cosine: dict = field(default_factory=dict)
    mean_cosine: float = -1.0
    final_loss: float = 0.0
    num_steps: int = 0
    num_queries: int = 0
    converged: bool = False


@dataclass
class SPSIResult:
    """Full S-PSI inversion result."""

    regime: str = "oracle"
    block_results: list = field(default_factory=list)
    total_queries: int = 0
    init_seed: int = 42


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
        input_ids = input_ids.to(self.device)
        logits = self.model(input_ids).logits
        self.query_count += input_ids.size(0)
        if self.defense_fn is not None:
            logits = self.defense_fn(logits)
        return logits

    @torch.no_grad()
    def get_boundary_state(
        self, input_ids: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Extract intermediate hidden state after block `layer_idx`."""
        input_ids = input_ids.to(self.device)
        outputs = self.model(
            input_ids, output_hidden_states=True, return_dict=True
        )
        return outputs.hidden_states[layer_idx + 1].detach()


class TeacherCache:
    """Pre-computed teacher logits and perturbation responses."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.clean_logits: Optional[torch.Tensor] = None
        self.input_ids: Optional[torch.Tensor] = None
        self.perturbed_input_ids: Optional[torch.Tensor] = None
        self.perturbed_logits: Optional[torch.Tensor] = None
        self.boundary_states: dict[int, torch.Tensor] = {}

    def build(
        self,
        teacher: BlackBoxTeacher,
        input_ids: torch.Tensor,
        config: SPSIConfig,
        cache_boundary_layers: Optional[list[int]] = None,
    ) -> None:
        """Pre-compute all teacher outputs for the query pool."""
        logger.info("Building teacher cache for %d inputs...", len(input_ids))
        self.input_ids = input_ids
        vocab_size = teacher.model.config.vocab_size

        clean_logits_list = []
        for start in range(0, len(input_ids), 32):
            batch = input_ids[start : start + 32]
            logits = teacher.query(batch)
            clean_logits_list.append(logits.cpu())
        self.clean_logits = torch.cat(clean_logits_list)
        logger.info("  Clean logits cached: %s", self.clean_logits.shape)

        rng = torch.Generator().manual_seed(config.seed)
        P = config.num_perturbation_positions
        R = config.num_replacement_tokens
        seq_len = input_ids.size(1)
        n = len(input_ids)

        top_tokens = torch.arange(3, 3 + 50)

        all_perturbed = []
        all_perturbed_logits = []

        for i in range(0, n, 32):
            batch = input_ids[i : min(i + 32, n)]
            bsz = batch.size(0)

            positions = torch.randint(
                0, seq_len, (bsz, P), generator=rng
            )
            replacements = top_tokens[
                torch.randint(0, 50, (bsz, P, R), generator=rng)
            ]

            batch_perturbed = []
            for b in range(bsz):
                for p_idx in range(P):
                    pos = positions[b, p_idx].item()
                    for r_idx in range(R):
                        x_prime = batch[b].clone()
                        x_prime[pos] = replacements[b, p_idx, r_idx]
                        batch_perturbed.append(x_prime)

            if batch_perturbed:
                stacked = torch.stack(batch_perturbed)
                all_perturbed.append(stacked)
                for s in range(0, len(stacked), 32):
                    chunk = stacked[s : s + 32]
                    pl = teacher.query(chunk)
                    all_perturbed_logits.append(pl.cpu())

        self.perturbed_input_ids = torch.cat(all_perturbed)
        self.perturbed_logits = torch.cat(all_perturbed_logits)
        logger.info(
            "  Perturbed logits cached: %s (P=%d, R=%d)",
            self.perturbed_logits.shape, P, R,
        )

        if cache_boundary_layers:
            for layer_idx in cache_boundary_layers:
                states = []
                for start in range(0, n, 32):
                    batch = input_ids[start : start + 32]
                    h = teacher.get_boundary_state(batch, layer_idx)
                    states.append(h.cpu())
                self.boundary_states[layer_idx] = torch.cat(states)
                logger.info(
                    "  Boundary state cached for layer %d: %s",
                    layer_idx, self.boundary_states[layer_idx].shape,
                )

    def get_batch(
        self,
        indices: torch.Tensor,
        config: SPSIConfig,
    ) -> tuple:
        """Return (input_ids, clean_logits, perturbed_input_ids, perturbed_logits)."""
        bsz = len(indices)
        P = config.num_perturbation_positions
        R = config.num_replacement_tokens
        pert_per_input = P * R

        ids = self.input_ids[indices]
        cl = self.clean_logits[indices]

        pert_indices = []
        for idx in indices:
            base = idx.item() * pert_per_input
            pert_indices.extend(range(base, base + pert_per_input))

        if pert_indices and max(pert_indices) < len(self.perturbed_input_ids):
            pi = self.perturbed_input_ids[pert_indices]
            pl = self.perturbed_logits[pert_indices]
        else:
            pi = ids.repeat(pert_per_input, 1)
            pl = cl.repeat(pert_per_input, 1, 1)

        return ids, cl, pi, pl

    def get_boundary_batch(
        self, indices: torch.Tensor, layer_idx: int
    ) -> Optional[torch.Tensor]:
        if layer_idx in self.boundary_states:
            return self.boundary_states[layer_idx][indices]
        return None


def get_num_blocks(model: nn.Module) -> int:
    """Return the number of transformer blocks in the model."""
    max_idx = -1
    for name, _ in model.named_parameters():
        for part in name.split("."):
            if part.isdigit():
                max_idx = max(max_idx, int(part))
                break
    return max_idx + 1


def get_block_param_names(model: nn.Module, block_idx: int) -> list[str]:
    """Return parameter names belonging to a specific block."""
    names = []
    target = str(block_idx)
    for name, _ in model.named_parameters():
        parts = name.split(".")
        for p in parts:
            if p == target:
                names.append(name)
                break
    return names


def get_lm_head_param_names(model: nn.Module) -> list[str]:
    """Return lm_head parameter names, handling weight tying."""
    names = []
    lm_head_ptrs: set[int] = set()
    for mod_name, module in model.named_modules():
        if "lm_head" in mod_name:
            for p in module.parameters():
                lm_head_ptrs.add(p.data_ptr())

    for name, param in model.named_parameters():
        if "lm_head" in name or param.data_ptr() in lm_head_ptrs:
            names.append(name)
    return names


def compute_per_matrix_cosine(
    model: nn.Module,
    ground_truth: dict[str, torch.Tensor],
    param_names: list[str],
) -> dict[str, float]:
    """Compute cosine similarity per weight matrix."""
    metrics = {}
    for name in param_names:
        if name not in ground_truth:
            continue
        for n, p in model.named_parameters():
            if n == name:
                gt = ground_truth[name].to(p.device).float().flatten()
                pred = p.data.float().flatten()
                if gt.shape != pred.shape:
                    break
                sim = F.cosine_similarity(
                    pred.unsqueeze(0), gt.unsqueeze(0)
                ).item()
                short = name.split(".")[-1]
                metrics[short] = sim
                break
    return metrics


def _inject_boundary_state(
    student: nn.Module,
    boundary_state: torch.Tensor,
    block_idx: int,
) -> None:
    """Register a forward hook that replaces block input with teacher boundary state.

    The hook intercepts the forward pass of block `block_idx` and substitutes
    the incoming hidden state with the teacher's cached boundary state.
    """
    pass


class _BoundaryInjectionHook:
    """Context manager that injects teacher boundary states into a student block."""

    def __init__(
        self,
        student: nn.Module,
        cache: "TeacherCache",
        indices: torch.Tensor,
        boundary_layer_idx: int,
    ):
        self.student = student
        self.cache = cache
        self.indices = indices
        self.boundary_layer_idx = boundary_layer_idx
        self.hook = None
        self._boundary = None

    def __enter__(self):
        h = self.cache.get_boundary_batch(self.indices, self.boundary_layer_idx)
        if h is None:
            return self
        device = next(self.student.parameters()).device
        self._boundary = h.to(device)

        block = _get_block_module(self.student, self.boundary_layer_idx + 1)
        if block is not None:
            def hook_fn(module, args):
                if self._boundary is not None:
                    if isinstance(args, tuple) and len(args) > 0:
                        new_args = (self._boundary,) + args[1:]
                        return new_args
                return args

            self.hook = block.register_forward_pre_hook(hook_fn)
        return self

    def __exit__(self, *args):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
        self._boundary = None


def _get_block_module(model: nn.Module, block_idx: int) -> Optional[nn.Module]:
    """Get the module corresponding to a specific block index."""
    for name, module in model.named_modules():
        parts = name.split(".")
        if len(parts) >= 2 and parts[-1] == str(block_idx):
            return module
    return None


def invert_block(
    student: nn.Module,
    cache: TeacherCache,
    config: SPSIConfig,
    param_names: list[str],
    ground_truth: Optional[dict] = None,
    boundary_layer_idx: Optional[int] = None,
    is_lm_head: bool = False,
    checkpoint_dir: Optional[str] = None,
    use_oracle_boundary: bool = False,
    query_budget_remaining: Optional[int] = None,
) -> BlockResult:
    """Run S-PSI inversion for a single block (or lm_head).

    Args:
        use_oracle_boundary: If True and boundary_layer_idx is set, inject
            cached teacher boundary states as block input.
        query_budget_remaining: If set, stop when budget is exhausted.
    """
    block_label = "lm_head" if is_lm_head else f"block_{boundary_layer_idx}"
    lr = config.lm_head_lr if is_lm_head else config.learning_rate
    max_steps = config.lm_head_steps if is_lm_head else config.max_steps_per_block

    for name, param in student.named_parameters():
        param.requires_grad = name in param_names

    trainable = [p for p in student.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in trainable)
    logger.info(
        "Inverting %s: %d params across %d tensors (oracle=%s)",
        block_label, total_params, len(trainable), use_oracle_boundary,
    )

    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=config.weight_decay)
    device = next(student.parameters()).device
    n_pool = len(cache.input_ids)
    rng = torch.Generator().manual_seed(config.seed + hash(block_label) % 10000)

    best_loss = float("inf")
    no_improve = 0
    total_queries = 0
    start_step = 0

    if checkpoint_dir:
        import glob
        ckpt_pattern = str(Path(checkpoint_dir) / f"ckpt_{block_label}_step*.pt")
        existing = sorted(glob.glob(ckpt_pattern), key=lambda x: int(x.split("step")[-1].split(".")[0]))
        if existing:
            ckpt = torch.load(existing[-1], map_location=device, weights_only=False)
            target = student.module if hasattr(student, "module") else student
            target.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = ckpt.get("step", 0)
            best_loss = ckpt.get("best_loss", float("inf"))
            logger.info(
                "Resumed %s from step %d (best_loss=%.6f)",
                block_label, start_step, best_loss,
            )

    P = config.num_perturbation_positions
    R = config.num_replacement_tokens
    pert_per_input = P * R

    oracle_boundary = (
        use_oracle_boundary
        and boundary_layer_idx is not None
        and not is_lm_head
        and boundary_layer_idx in cache.boundary_states
    )

    for step in range(start_step, max_steps):
        if query_budget_remaining is not None and total_queries >= query_budget_remaining:
            logger.info("%s: query budget exhausted at step %d", block_label, step)
            break

        indices = torch.randint(0, n_pool, (config.batch_size,), generator=rng)
        input_ids, clean_logits_t, pert_ids, pert_logits_t = cache.get_batch(
            indices, config
        )
        input_ids = input_ids.to(device)
        clean_logits_t = clean_logits_t.to(device)
        pert_ids = pert_ids.to(device)
        pert_logits_t = pert_logits_t.to(device)

        student.train()

        from contextlib import nullcontext

        if oracle_boundary:
            boundary_idx = boundary_layer_idx - 1
            ctx_clean = _BoundaryInjectionHook(
                student, cache, indices, boundary_idx,
            )
        else:
            ctx_clean = nullcontext()

        with ctx_clean:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z_s = student(input_ids).logits
                loss_logit = F.mse_loss(z_s.float(), clean_logits_t.float())

        loss_sensitivity = torch.tensor(0.0, device=device)
        if config.beta > 0 and len(pert_ids) > 0:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z_s_pert = student(pert_ids).logits
            delta_s = z_s_pert.float() - z_s.detach().float().repeat_interleave(
                pert_per_input, dim=0
            )
            delta_t = pert_logits_t.float() - clean_logits_t.float().repeat_interleave(
                pert_per_input, dim=0
            )
            loss_sensitivity = F.mse_loss(delta_s, delta_t)

        with torch.no_grad():
            loss_reg = sum(p.float().norm(2) ** 2 for p in trainable)

        loss = (
            config.alpha * loss_logit
            + config.beta * loss_sensitivity
            + config.gamma * loss_reg
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        total_queries += config.batch_size * (1 + pert_per_input)
        loss_val = loss.item()

        if loss_val < best_loss - config.convergence_threshold:
            best_loss = loss_val
            no_improve = 0
        else:
            no_improve += 1

        if step % config.log_every == 0:
            cos_str = ""
            if ground_truth is not None:
                pm = compute_per_matrix_cosine(student, ground_truth, param_names)
                mean_cos = sum(pm.values()) / max(len(pm), 1)
                cos_str = f" | mean_cos={mean_cos:.4f}"
            logger.info(
                "%s step %d/%d | loss=%.6f (logit=%.6f sens=%.6f) | best=%.6f | queries=%d%s",
                block_label, step, max_steps,
                loss_val, loss_logit.item(), loss_sensitivity.item(),
                best_loss, total_queries, cos_str,
            )

        if no_improve >= config.patience:
            logger.info(
                "%s converged at step %d (%d steps without improvement)",
                block_label, step, config.patience,
            )
            break

        if checkpoint_dir and step > 0 and step % config.save_every == 0:
            _save_checkpoint(
                checkpoint_dir, block_label, step, student, optimizer, best_loss
            )

    per_matrix = {}
    mean_cos = -1.0
    if ground_truth is not None:
        per_matrix = compute_per_matrix_cosine(student, ground_truth, param_names)
        mean_cos = sum(per_matrix.values()) / max(len(per_matrix), 1)

    return BlockResult(
        block_name=block_label,
        per_matrix_cosine=per_matrix,
        mean_cosine=mean_cos,
        final_loss=best_loss,
        num_steps=step + 1,
        num_queries=total_queries,
        converged=no_improve >= config.patience,
    )


def run_spsi(
    student: nn.Module,
    teacher: BlackBoxTeacher,
    cache: TeacherCache,
    config: SPSIConfig,
    regime: str = "oracle",
    num_suffix_blocks: int = 2,
    ground_truth: Optional[dict] = None,
    output_dir: Optional[str] = None,
    init_seed: int = 42,
) -> SPSIResult:
    """Run complete S-PSI pipeline."""
    result = SPSIResult(regime=regime, init_seed=init_seed)
    num_blocks = get_num_blocks(student)
    ckpt_dir = str(Path(output_dir) / "checkpoints") if output_dir else None
    use_oracle = regime == "oracle"
    precompute_queries = teacher.query_count
    budget_remaining = config.query_budget - precompute_queries
    logger.info(
        "Pre-compute used %d queries, remaining budget: %d",
        precompute_queries, budget_remaining,
    )

    lm_head_names = get_lm_head_param_names(student)
    logger.info("Phase 0: lm_head recovery (%d params)", len(lm_head_names))
    lm_result = invert_block(
        student, cache, config, lm_head_names,
        ground_truth=ground_truth,
        is_lm_head=True,
        checkpoint_dir=ckpt_dir,
        use_oracle_boundary=False,
        query_budget_remaining=budget_remaining,
    )
    result.block_results.append(lm_result)
    result.total_queries += lm_result.num_queries
    budget_remaining -= lm_result.num_queries
    logger.info(
        "lm_head: mean_cos=%.4f, loss=%.6f, steps=%d, queries=%d (remaining=%d)",
        lm_result.mean_cosine, lm_result.final_loss, lm_result.num_steps,
        lm_result.num_queries, max(budget_remaining, 0),
    )

    for k in range(num_suffix_blocks):
        block_idx = num_blocks - 1 - k
        if block_idx < 0:
            break
        if budget_remaining <= 0:
            logger.info("Query budget exhausted, stopping at block %d", block_idx)
            break

        block_names = get_block_param_names(student, block_idx)
        if not block_names:
            continue

        logger.info(
            "Phase %d: Block %d recovery (%d params, oracle=%s)",
            k + 1, block_idx, len(block_names), use_oracle,
        )
        br = invert_block(
            student, cache, config, block_names,
            ground_truth=ground_truth,
            boundary_layer_idx=block_idx,
            checkpoint_dir=ckpt_dir,
            use_oracle_boundary=use_oracle,
            query_budget_remaining=budget_remaining,
        )
        result.block_results.append(br)
        result.total_queries += br.num_queries
        budget_remaining -= br.num_queries
        logger.info(
            "Block %d: mean_cos=%.4f, loss=%.6f, steps=%d, queries=%d (remaining=%d)",
            block_idx, br.mean_cosine, br.final_loss, br.num_steps,
            br.num_queries, max(budget_remaining, 0),
        )

    if output_dir:
        _save_results(result, output_dir, student=student)

    return result


def _save_checkpoint(
    ckpt_dir: str,
    block_label: str,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_loss: float,
) -> None:
    path = Path(ckpt_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "block": block_label,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
        },
        path / f"ckpt_{block_label}_step{step}.pt",
    )


def _save_results(
    result: SPSIResult,
    output_dir: str,
    student: Optional[nn.Module] = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "regime": result.regime,
        "init_seed": result.init_seed,
        "total_queries": result.total_queries,
        "blocks": [
            {
                "name": br.block_name,
                "per_matrix_cosine": br.per_matrix_cosine,
                "mean_cosine": br.mean_cosine,
                "final_loss": br.final_loss,
                "num_steps": br.num_steps,
                "num_queries": br.num_queries,
                "converged": br.converged,
            }
            for br in result.block_results
        ],
    }

    with open(out / "spsi_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if student is not None:
        model_dir = out / "recovered_model"
        model_to_save = student.module if hasattr(student, "module") else student
        model_to_save.save_pretrained(str(model_dir))
        logger.info("Recovered model saved to %s", model_dir)

    logger.info("S-PSI results saved to %s", out)
