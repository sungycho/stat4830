"""Seed-based layer-level perturbation utilities for ES.

Design (from LAES / EvoLLM paper):
- Only an int seed is stored, never the full noise vector.
- Parameters are perturbed in-place, layer by layer; peak overhead
  is the size of the largest single layer, not the entire model.
- Restoration is the exact inverse: re-generate the same noise from
  the same seed and apply with the opposite sign.

Device note:
- The RNG is created on the same device as the model's parameters.
- CPU and CUDA generators produce different sequences for the same seed,
  so noise is not portable across devices — but roundtrip invariance
  (perturb + restore) holds as long as the device doesn't change mid-run.
"""
from __future__ import annotations
import torch

# Set to False to suppress per-layer progress prints (recommended on GPU).
VERBOSE: bool = True


def _es_params(model):
    """Trainable parameters only — frozen layers (embeddings, LoRA base) excluded.

    Result is cached on the model object to avoid rebuilding the list every call.
    Call clear_es_param_cache(model) to force a refresh (e.g. after changing
    requires_grad flags).
    """
    if not hasattr(model, "_es_params_cache"):
        model._es_params_cache = [p for p in model.parameters() if p.requires_grad]
    return model._es_params_cache


def clear_es_param_cache(model) -> None:
    """Invalidate the cached trainable-parameter list.

    Call this whenever requires_grad changes on any parameter (e.g. after
    freezing or unfreezing layers mid-run).
    """
    if hasattr(model, "_es_params_cache"):
        del model._es_params_cache


def _make_rng(params: list, seed: int) -> torch.Generator:
    """Create a seeded RNG on the same device as the model parameters."""
    device = params[0].device if params else torch.device("cpu")
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    return rng


def perturb_inplace(model, seed: int, sigma: float, sign: int) -> None:
    """Perturb model parameters in-place, layer by layer.

    A single RNG stream is created from `seed` on the model's device and
    advanced across all layers in fixed order, so each layer receives distinct
    noise while the full perturbation remains fully determined by `seed`.

    Args:
        model: the PyTorch model to perturb.
        seed:  RNG seed that fully determines the noise realisation.
        sigma: perturbation scale.
        sign:  +1 for positive perturbation, -1 for antithetic.
    """
    params = _es_params(model)
    rng = _make_rng(params, seed)
    total = len(params)
    phase = "+eps" if sign > 0 else "-eps"

    if VERBOSE:
        print(f"[perturb] start phase={phase} seed={seed} sigma={sigma} layers={total}")

    with torch.no_grad():
        for i, p in enumerate(params, start=1):
            noise = torch.randn(p.shape, generator=rng, device=p.device, dtype=p.dtype)
            p.add_(noise, alpha=sign * sigma)
            if VERBOSE and (i % 100 == 0 or i == total):
                print(f"[perturb] {phase} layer {i}/{total} shape={tuple(p.shape)}")

    if VERBOSE:
        print(f"[perturb] done phase={phase}")


def restore_inplace(model, seed: int, sigma: float, sign: int) -> None:
    """Restore parameters perturbed by perturb_inplace — exact inverse."""
    perturb_inplace(model, seed, sigma, -sign)


def es_grad_update(
    model,
    seeds: list[int],
    rewards: list[float],
    lr: float,
) -> None:
    """Apply the ES gradient update in-place, layer by layer (paper §3.2 points 6+7).

    Uses z-score normalised rewards (point 4). sigma is digested into lr (point 7),
    so it does not appear here — set lr = alpha / sigma at the call site.
    Peak extra memory = size of the largest single layer (one noise buffer at a time).

    Args:
        model:   model whose parameters will be updated.
        seeds:   list of perturbation seeds used in this iteration.
        rewards: corresponding scalar rewards (one per seed).
        lr:      effective learning rate (= alpha / sigma, pre-digested).
    """
    N = len(seeds)
    if N == 1:
        raise ValueError("es_grad_update requires at least 2 seeds; z-score is undefined for N=1.")
    r = torch.tensor(rewards, dtype=torch.float32)
    std = r.std()  # Bessel-corrected, defined for N≥2
    r_norm = (r - r.mean()) / (std + 1e-8)  # z-score normalisation

    params = _es_params(model)
    # Outer loop over seeds so each seed's RNG stream is advanced in the same
    # layer order as perturb_inplace, guaranteeing identical noise reconstruction.
    # Contributions are applied directly — no full-model delta buffer needed.
    with torch.no_grad():
        for seed, rn in zip(seeds, r_norm.tolist()):
            rng = _make_rng(params, seed)
            for p in params:
                noise = torch.randn(p.shape, generator=rng, device=p.device, dtype=p.dtype)
                p.add_(noise, alpha=rn * lr / N)
