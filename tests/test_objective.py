"""Loss shape/finiteness, readout alignment, and the anti-self-deception metrics."""

from __future__ import annotations

import pytest
import torch

from qdif.diffusion.objective import diffusion_loss, diffusion_metrics

B, C, V = 2, 8, 32


def _perfect_logits(x0: torch.Tensor, magnitude: float = 20.0) -> torch.Tensor:
    logits = torch.zeros(B, C, V)
    return logits.scatter(2, x0.unsqueeze(-1), magnitude)


def test_loss_shape_and_finiteness():
    x0 = torch.randint(0, V, (B, C))
    out = diffusion_loss(torch.randn(B, C, V), x0)
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    assert out.per_position_loss.shape == (B, C)


def test_uniform_logits_give_log_vocab_loss():
    x0 = torch.randint(0, V, (B, C))
    out = diffusion_loss(torch.zeros(B, C, V), x0)
    assert float(out.loss) == pytest.approx(torch.log(torch.tensor(float(V))).item(), abs=1e-5)


def test_perfect_prediction_gives_near_zero_loss():
    x0 = torch.randint(0, V, (B, C))
    out = diffusion_loss(_perfect_logits(x0), x0)
    assert float(out.loss) < 1e-6


def test_loss_is_differentiable_into_the_logits():
    x0 = torch.randint(0, V, (B, C))
    logits = torch.randn(B, C, V, requires_grad=True)
    diffusion_loss(logits, x0).loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0


def test_readout_is_unshifted():
    """logits[i] must score x0[i], NOT x0[i+1]. This pins the core convention."""
    x0 = torch.randint(0, V, (B, C))
    aligned = diffusion_loss(_perfect_logits(x0), x0).loss
    shifted = _perfect_logits(torch.roll(x0, shifts=-1, dims=1))
    assert float(aligned) < 1e-6
    assert float(diffusion_loss(shifted, x0).loss) > 1.0


def test_loss_on_corrupted_ignores_clean_positions():
    x0 = torch.randint(0, V, (B, C))
    logits = _perfect_logits(x0)
    # Make the clean positions catastrophically wrong; they must not count.
    corrupted = torch.zeros(B, C, dtype=torch.bool)
    corrupted[:, :4] = True
    logits[:, 4:] = 0.0
    out = diffusion_loss(logits, x0, corrupted, loss_on="corrupted")
    assert float(out.loss) < 1e-6
    assert out.metrics["loss_positions"] == B * 4


def test_loss_on_corrupted_at_zero_noise_is_finite():
    """t=0 leaves no corrupted positions; the loss must stay finite and connected."""
    x0 = torch.randint(0, V, (B, C))
    logits = torch.randn(B, C, V, requires_grad=True)
    out = diffusion_loss(logits, x0, torch.zeros(B, C, dtype=torch.bool), loss_on="corrupted")
    assert torch.isfinite(out.loss)
    assert float(out.loss) == 0.0
    out.loss.backward()  # must not raise


def test_valid_mask_excludes_padding():
    x0 = torch.randint(0, V, (B, C))
    logits = _perfect_logits(x0)
    logits[:, 4:] = 0.0
    valid = torch.zeros(B, C, dtype=torch.bool)
    valid[:, :4] = True
    assert float(diffusion_loss(logits, x0, valid_mask=valid).loss) < 1e-6


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="disagree"):
        diffusion_loss(torch.randn(B, C, V), torch.randint(0, V, (B, C + 1)))


def test_unknown_loss_on_rejected():
    with pytest.raises(ValueError, match="loss_on"):
        diffusion_loss(torch.randn(B, C, V), torch.randint(0, V, (B, C)), loss_on="some")


# ----------------------------------------------------------------------- metrics


def test_metrics_detect_a_perfect_denoiser():
    x0 = torch.randint(0, V, (B, C))
    xt = x0.clone()
    corrupted = torch.zeros(B, C, dtype=torch.bool)
    corrupted[:, :4] = True
    xt[:, :4] = (xt[:, :4] + 1) % V

    m = diffusion_metrics(_perfect_logits(x0), x0, xt, corrupted)
    assert m["identity_accuracy"] == 1.0
    assert m["corrupted_accuracy"] == 1.0
    assert m["clean_accuracy"] == 1.0
    assert m["lift_over_copy"] == pytest.approx(0.5)


def test_metrics_expose_a_pure_copier():
    """The failure mode the project actually hit: identity accuracy rises, but
    only because the noise left those positions alone."""
    x0 = torch.randint(0, V, (B, C))
    xt = x0.clone()
    corrupted = torch.zeros(B, C, dtype=torch.bool)
    corrupted[:, :4] = True
    xt[:, :4] = (xt[:, :4] + 1) % V

    m = diffusion_metrics(_perfect_logits(xt), x0, xt, corrupted)  # predicts its input
    assert m["copy_rate"] == 1.0
    assert m["corrupted_accuracy"] == 0.0
    assert m["clean_accuracy"] == 1.0
    assert m["identity_accuracy"] == pytest.approx(0.5)
    assert m["copy_baseline_accuracy"] == pytest.approx(0.5)
    assert m["lift_over_copy"] == pytest.approx(0.0)


def test_next_token_accuracy_detects_an_unadapted_ar_model():
    x0 = torch.randint(0, V, (B, C))
    shifted = _perfect_logits(torch.roll(x0, shifts=-1, dims=1))
    m = diffusion_metrics(shifted, x0, x0, torch.zeros(B, C, dtype=torch.bool))
    assert m["next_token_accuracy"] == 1.0
    assert m["identity_accuracy"] < 0.2


def test_entropy_and_confidence_bounds():
    x0 = torch.randint(0, V, (B, C))
    sharp = diffusion_metrics(_perfect_logits(x0), x0, x0, torch.zeros(B, C, dtype=torch.bool))
    flat = diffusion_metrics(torch.zeros(B, C, V), x0, x0, torch.zeros(B, C, dtype=torch.bool))
    assert sharp["mean_entropy"] < flat["mean_entropy"]
    assert flat["mean_top1_prob"] == pytest.approx(1.0 / V, abs=1e-6)
    assert flat["mean_entropy"] == pytest.approx(torch.log(torch.tensor(float(V))).item(), abs=1e-4)
