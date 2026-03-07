"""Regression tests for perturb/restore invariance.

Run with: uv run python -m pytest tests/test_perturb.py -v
"""
import torch
import torch.nn as nn

from src.utils.perturb import perturb_inplace, restore_inplace, es_grad_update


def _tiny_model():
    model = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 4))
    return model


def _snapshot(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def _assert_restored(before, model, context=""):
    for k, v in model.state_dict().items():
        diff = (v - before[k]).abs().max().item()
        assert diff < 1e-6, f"{context} param '{k}' not restored: max_diff={diff:.2e}"


def test_perturb_restore_roundtrip():
    """perturb then restore must be an exact no-op."""
    model = _tiny_model()
    before = _snapshot(model)

    perturb_inplace(model, seed=42, sigma=1e-3, sign=+1)
    restore_inplace(model, seed=42, sigma=1e-3, sign=+1)

    _assert_restored(before, model, "4-pass")


def test_antithetic_3pass_roundtrip():
    """3-pass antithetic (perturb+1, perturb 2σ -1, restore -1) must restore original."""
    model = _tiny_model()
    before = _snapshot(model)
    sigma, seed = 1e-3, 99

    perturb_inplace(model, seed, sigma, sign=+1)        # θ → θ+σε
    perturb_inplace(model, seed, 2 * sigma, sign=-1)    # θ+σε → θ-σε
    restore_inplace(model, seed, sigma, sign=-1)         # θ-σε → θ

    _assert_restored(before, model, "3-pass antithetic")


def test_perturb_actually_changes_params():
    """Sanity check: perturb must change at least one parameter."""
    model = _tiny_model()
    before = _snapshot(model)

    perturb_inplace(model, seed=1, sigma=1e-3, sign=+1)

    changed = any(
        not torch.equal(model.state_dict()[k], before[k])
        for k in before
    )
    assert changed, "perturb_inplace made no change to any parameter"


def test_es_grad_update_nonzero():
    """With varied rewards, es_grad_update must change parameters."""
    model = _tiny_model()
    before = _snapshot(model)

    # Two seeds with opposite rewards → non-zero z-score normalized update
    es_grad_update(model, seeds=[7, 13], rewards=[1.0, -1.0], lr=1e-2)

    changed = any(
        not torch.equal(model.state_dict()[k], before[k])
        for k in before
    )
    assert changed, "es_grad_update made no change to any parameter"


def test_different_seeds_give_different_noise():
    """Two different seeds must produce different perturbations."""
    model_a = _tiny_model()
    model_b = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 4))
    model_b.load_state_dict(model_a.state_dict())  # identical starting point

    perturb_inplace(model_a, seed=1, sigma=1e-3, sign=+1)
    perturb_inplace(model_b, seed=2, sigma=1e-3, sign=+1)

    different = any(
        not torch.equal(model_a.state_dict()[k], model_b.state_dict()[k])
        for k in model_a.state_dict()
    )
    assert different, "Different seeds produced identical perturbations"
