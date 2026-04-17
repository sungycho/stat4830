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
VERBOSE: bool = False


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


def _sample_noise(p, rng: torch.Generator, noise_type: str) -> torch.Tensor:
    """Sample noise for a single parameter tensor."""
    if noise_type == "rademacher":
        # ±1 uniform random (Rademacher distribution)
        return (
            torch.empty(p.shape, device=p.device)
            .bernoulli_(0.5, generator=rng)
            .mul_(2)
            .sub_(1)
            .to(p.dtype)
        )
    # default: Gaussian
    return torch.randn(p.shape, generator=rng, device=p.device, dtype=p.dtype)


def perturb_inplace(
    model, seed: int, sigma: float, sign: int, noise_type: str = "gaussian"
) -> None:
    """Perturb model parameters in-place, layer by layer.

    A single RNG stream is created from `seed` on the model's device and
    advanced across all layers in fixed order, so each layer receives distinct
    noise while the full perturbation remains fully determined by `seed`.

    Args:
        model:      the PyTorch model to perturb.
        seed:       RNG seed that fully determines the noise realisation.
        sigma:      perturbation scale.
        sign:       +1 for positive perturbation, -1 for antithetic.
        noise_type: "gaussian" (default) or "rademacher" (±1 uniform).
    """
    params = _es_params(model)
    rng = _make_rng(params, seed)
    total = len(params)
    phase = "+eps" if sign > 0 else "-eps"

    if VERBOSE:
        print(f"[perturb] start phase={phase} seed={seed} sigma={sigma} noise={noise_type} layers={total}")

    with torch.no_grad():
        for i, p in enumerate(params, start=1):
            noise = _sample_noise(p, rng, noise_type)
            p.add_(noise, alpha=sign * sigma)
            if VERBOSE and (i % 100 == 0 or i == total):
                print(f"[perturb] {phase} layer {i}/{total} shape={tuple(p.shape)}")

    if VERBOSE:
        print(f"[perturb] done phase={phase}")


def restore_inplace(
    model, seed: int, sigma: float, sign: int, noise_type: str = "gaussian"
) -> None:
    """Restore parameters perturbed by perturb_inplace — exact inverse."""
    perturb_inplace(model, seed, sigma, -sign, noise_type=noise_type)


def es_grad_update(
    model,
    seeds: list[int],
    rewards: list[float],
    lr: float,
    top_k: int = 0,
    normalize: bool = True,
    noise_type: str = "gaussian",
) -> None:
    """Apply the ES gradient update in-place, layer by layer (paper §3.2 points 6+7).

    Peak extra memory = size of the largest single layer (one noise buffer at a time).

    Args:
        model:      model whose parameters will be updated.
        seeds:      list of perturbation seeds used in this iteration.
        rewards:    corresponding scalar rewards (one per seed).
        lr:         effective learning rate (caller is responsible for any SPSA scaling,
                    e.g. dividing by 2σ for MeZO-style raw advantages).
        top_k:      ARS-style: keep only the top-k seeds by |reward| before updating.
                    0 = disabled (use all seeds).
        normalize:  if True (default), z-score normalise rewards before update.
        noise_type: "gaussian" or "rademacher" — must match what perturb_inplace used.
    """
    N = len(seeds)
    r = torch.tensor(rewards, dtype=torch.float32)

    # ARS-style top-k selection by absolute advantage
    if top_k > 0 and top_k < N:
        indices = torch.argsort(torch.abs(r), descending=True)[:top_k]
        seeds = [seeds[i] for i in indices.tolist()]
        r = r[indices]
        N = len(seeds)

    if normalize and N >= 2:
        std = r.std()  # Bessel-corrected, defined for N≥2
        r_norm = (r - r.mean()) / (std + 1e-8)
    else:
        r_norm = r

    params = _es_params(model)
    # Outer loop over seeds so each seed's RNG stream is advanced in the same
    # layer order as perturb_inplace, guaranteeing identical noise reconstruction.
    with torch.no_grad():
        for seed, rn in zip(seeds, r_norm.tolist()):
            rng = _make_rng(params, seed)
            for p in params:
                noise = _sample_noise(p, rng, noise_type)
                p.add_(noise, alpha=rn * lr / N)
