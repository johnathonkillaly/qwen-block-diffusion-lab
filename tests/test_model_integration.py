"""Integration tests against a real Qwen3.5 checkpoint.

Marked `model`; skipped automatically when no local checkpoint is present.
Run with:  pytest -m model
"""

from __future__ import annotations

import pytest
import torch

from qdif.diffusion.corruption import corrupt_canvas
from qdif.diffusion.objective import diffusion_loss, diffusion_metrics

pytestmark = pytest.mark.model


# ---------------------------------------------------------------------- tokenizer


def test_tokenizer_round_trip(tokenizer):
    text = "The cat sat on the mat."
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert tokenizer.decode(ids) == text


def test_tokenizer_vocab_covers_the_config_vocab(tokenizer, diffusion_model):
    model, _ = diffusion_model
    # The model's embedding table is >= the tokenizer vocabulary (padded for
    # special/multimodal tokens); sampling from [0, config.vocab_size) is safe.
    assert model.config.vocab_size >= tokenizer.vocab_size


# ------------------------------------------------------------------- architecture


def test_architecture_matches_the_documented_hybrid_pattern(diffusion_model, model_path):
    from qdif.models.qwen35_adapter import inspect_architecture

    model, _ = diffusion_model
    r = inspect_architecture(model.base, checkpoint_path=model_path)

    assert r.model_class == "Qwen3_5ForCausalLM"
    assert set(r.layer_types) == {"linear_attention", "full_attention"}
    assert len(r.full_attention_layers) + len(r.linear_attention_layers) == r.num_hidden_layers
    # 3:1 DeltaNet:full-attention, full attention on the last layer of each group.
    assert len(r.linear_attention_layers) == 3 * len(r.full_attention_layers)
    interval = r.full_attention_interval
    assert all((i + 1) % interval == 0 for i in r.full_attention_layers)


def test_vision_tower_is_not_loaded(diffusion_model, model_path):
    """Text-only execution must bypass the vision tower cleanly."""
    from qdif.models.qwen35_adapter import inspect_architecture

    model, _ = diffusion_model
    r = inspect_architecture(model.base, checkpoint_path=model_path)
    assert r.checkpoint_has_vision, "this checkpoint should ship a vision tower"
    assert not r.vision_loaded
    assert not any(".visual." in n for n, _ in model.base.named_parameters())


def test_mtp_weights_are_not_loaded(diffusion_model, model_path):
    from qdif.models.qwen35_adapter import inspect_architecture

    model, _ = diffusion_model
    r = inspect_architecture(model.base, checkpoint_path=model_path)
    assert r.checkpoint_has_mtp
    assert not any(n.startswith("mtp.") for n, _ in model.base.named_parameters())


def test_embeddings_are_tied_to_the_output_head(diffusion_model):
    model, _ = diffusion_model
    assert model.config.tie_word_embeddings
    assert model.base.lm_head.weight.data_ptr() == model.base.model.embed_tokens.weight.data_ptr()


# ------------------------------------------------------------------ forward paths


def test_ar_path_still_works(diffusion_model, tokenizer):
    """The autoregressive path must survive the diffusion wrapper untouched."""
    model, _ = diffusion_model
    ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad(), model.no_diffusion():
        out = model.ar_forward(ids)
    assert out.logits.shape == (1, ids.shape[1], model.config.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_untrained_adapters_leave_the_ar_path_bit_identical(diffusion_model, tokenizer):
    """lora_B and the auxiliary output layers are zero-initialised, so enabling the
    adapters cannot change anything before the first optimizer step."""
    model, _ = diffusion_model
    ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        enabled = model.ar_forward(ids).logits
        with model.no_diffusion():
            disabled = model.ar_forward(ids).logits
    assert torch.equal(enabled, disabled)


def test_diffusion_forward_shapes_and_finiteness(diffusion_model, cfg):
    model, _ = diffusion_model
    C = cfg.diffusion.canvas_length
    canvas = torch.randint(0, model.config.vocab_size, (1, C), device=model.device)
    prefix = torch.randint(0, model.config.vocab_size, (1, 8), device=model.device)
    with torch.no_grad():
        out = model(canvas_ids=canvas, t=torch.tensor([0.5]), prefix_ids=prefix)
    assert out.logits.shape == (1, C, model.config.vocab_size)
    assert out.hidden_states.shape == (1, C, model.config.hidden_size)
    assert out.canvas_start == 8
    assert out.total_length == 8 + C
    assert torch.isfinite(out.logits).all()


def test_diffusion_forward_without_a_prefix(diffusion_model, cfg):
    model, _ = diffusion_model
    canvas = torch.randint(0, model.config.vocab_size, (1, 16), device=model.device)
    with torch.no_grad():
        out = model(canvas_ids=canvas, t=torch.tensor([1.0]), prefix_ids=None)
    assert out.canvas_start == 0
    assert torch.isfinite(out.logits).all()


def test_bidirectional_mask_changes_canvas_logits_only(diffusion_model, cfg):
    """The prefix must be unaffected by canvas bidirectionality, or block-
    autoregressive commitment would be unsound."""
    model, _ = diffusion_model
    canvas = torch.randint(0, model.config.vocab_size, (1, 16), device=model.device)
    prefix = torch.randint(0, model.config.vocab_size, (1, 8), device=model.device)
    with torch.no_grad():
        bidi = model(canvas_ids=canvas, t=torch.tensor([0.5]), prefix_ids=prefix, bidirectional=True)
        causal = model(canvas_ids=canvas, t=torch.tensor([0.5]), prefix_ids=prefix, bidirectional=False)
    assert not torch.equal(bidi.logits, causal.logits)


def test_deltanet_layers_cannot_carry_information_backwards(diffusion_model, tokenizer):
    """The central architectural claim, tested empirically rather than asserted."""
    from qdif.eval.reconstruction import probe_bidirectionality

    model, _ = diffusion_model
    text = (
        "The quick brown fox jumps over the lazy dog while the cat sat on the "
        "windowsill watching the rain fall on the quiet street below the old house."
    )
    result = probe_bidirectionality(model, tokenizer, text, canvas_length=16)
    # With bidirectional full attention, earlier canvas positions move.
    assert result["bidirectional"]["num_earlier_positions_affected"] > 0
    # With the mask made causal, nothing moves -- so the 18 DeltaNet layers
    # contribute exactly zero backwards information flow.
    assert result["causal_control"]["num_earlier_positions_affected"] == 0
    assert result["causal_control"]["max_delta_before_last_position"] == 0.0


def test_canvas_length_boundaries(diffusion_model):
    model, _ = diffusion_model
    for C in (1, 16, 32):
        canvas = torch.randint(0, model.config.vocab_size, (1, C), device=model.device)
        with torch.no_grad():
            out = model(canvas_ids=canvas, t=torch.tensor([0.5]))
        assert out.logits.shape[1] == C


# -------------------------------------------------------------------- loss / grad


def test_loss_is_finite_across_the_whole_noise_range(diffusion_model, cfg):
    model, _ = diffusion_model
    x0 = torch.randint(0, 1000, (1, cfg.diffusion.canvas_length))
    for t in (0.0, 0.1, 0.5, 0.9, 1.0):
        r = corrupt_canvas(
            x0, torch.tensor([t]), model.config.vocab_size,
            generator=torch.Generator().manual_seed(0),
        )
        with torch.no_grad():
            out = model(canvas_ids=r.xt.to(model.device), t=r.t.to(model.device))
        loss = diffusion_loss(out.logits, r.x0.to(model.device), r.corrupted_mask.to(model.device))
        assert torch.isfinite(loss.loss), f"non-finite loss at t={t}"


def test_gradients_reach_lora_and_leave_the_base_frozen(diffusion_model, cfg):
    model, _ = diffusion_model
    model.zero_grad(set_to_none=True)
    C = cfg.diffusion.canvas_length
    x0 = torch.randint(0, 1000, (1, C))
    r = corrupt_canvas(
        x0, torch.tensor([0.5]), model.config.vocab_size, generator=torch.Generator().manual_seed(0)
    )
    out = model(canvas_ids=r.xt.to(model.device), t=r.t.to(model.device))
    diffusion_loss(out.logits, r.x0.to(model.device), r.corrupted_mask.to(model.device)).loss.backward()

    lora_grads, base_grads = [], []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if ".lora_" in name:
            lora_grads.append(float(p.grad.abs().sum()))
        elif not name.startswith(("timestep_conditioner.", "self_conditioner.")):
            base_grads.append(name)

    assert lora_grads and max(lora_grads) > 0, "no LoRA parameter received gradient"
    assert not base_grads, f"frozen base weights received gradient: {base_grads[:3]}"
    model.zero_grad(set_to_none=True)


def test_timestep_conditioner_receives_gradient(diffusion_model, cfg):
    model, _ = diffusion_model
    model.zero_grad(set_to_none=True)
    canvas = torch.randint(0, 1000, (1, cfg.diffusion.canvas_length), device=model.device)
    out = model(canvas_ids=canvas, t=torch.tensor([0.5], device=model.device))
    diffusion_loss(out.logits, canvas).loss.backward()
    grads = [
        float(p.grad.abs().sum())
        for n, p in model.named_parameters()
        if n.startswith("timestep_conditioner.") and p.grad is not None
    ]
    assert grads and max(grads) > 0
    model.zero_grad(set_to_none=True)


def test_unadapted_model_behaves_autoregressively_under_the_diffusion_readout(
    diffusion_model, tokenizer, cfg
):
    """Expected starting point: identity accuracy near zero, next-token accuracy high.
    If this ever inverts before training, the readout alignment has drifted."""
    model, _ = diffusion_model
    ids = tokenizer(
        "It is a truth universally acknowledged, that a single man in possession of a "
        "good fortune, must be in want of a wife.",
        add_special_tokens=False, return_tensors="pt",
    )["input_ids"][:, :32].to(model.device)
    with torch.no_grad():
        out = model(canvas_ids=ids, t=torch.zeros(1, device=model.device))
    m = diffusion_metrics(out.logits, ids, ids, torch.zeros_like(ids, dtype=torch.bool))
    assert m["next_token_accuracy"] > m["identity_accuracy"]


# --------------------------------------------------------------------- checkpoint


def test_checkpoint_save_and_reload(diffusion_model, cfg, tmp_path):
    from qdif.training.checkpoint import load_checkpoint, save_checkpoint
    from qdif.training.lora import LoRALinear

    model, _ = diffusion_model
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(std=0.02)
        model.timestep_conditioner.mlp[-1].weight.normal_(std=0.02)

    canvas = torch.randint(0, 1000, (1, 16), device=model.device)
    with torch.no_grad():
        before = model(canvas_ids=canvas, t=torch.tensor([0.5], device=model.device)).logits.clone()

    path = save_checkpoint(model, cfg, tmp_path / "ckpt", step=7)
    assert (path / "adapters.safetensors").exists()
    assert (path / "config.yaml").exists()
    # Base weights must never be written into an adapter checkpoint.
    from safetensors.torch import load_file

    keys = load_file(str(path / "adapters.safetensors")).keys()
    assert all(k.startswith(("lora.", "aux.")) for k in keys)
    assert not any("base.weight" in k for k in keys)

    # Perturb, then restore.
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.zero_()
        model.timestep_conditioner.mlp[-1].weight.zero_()
    with torch.no_grad():
        perturbed = model(canvas_ids=canvas, t=torch.tensor([0.5], device=model.device)).logits
    assert not torch.equal(before, perturbed)

    meta = load_checkpoint(model, path)
    assert meta["step"] == 7
    assert meta["loaded_lora_tensors"] > 0
    with torch.no_grad():
        restored = model(canvas_ids=canvas, t=torch.tensor([0.5], device=model.device)).logits
    assert torch.equal(before, restored)


# ------------------------------------------------------------------------ sampler


@pytest.mark.parametrize("steps", [1, 2, 4, 8])
def test_sampler_step_counts(diffusion_model, steps):
    from qdif.diffusion.sampler import denoise

    model, _ = diffusion_model
    res = denoise(
        model, canvas_length=16, steps=steps, generator=torch.Generator().manual_seed(0)
    )
    assert res.tokens.shape == (1, 16)
    assert res.forwards == steps
    assert res.steps_run == steps
    assert len(res.history) == steps
    assert res.tokens_per_forward == pytest.approx(16 / steps)
    assert int(res.tokens.min()) >= 0
    assert int(res.tokens.max()) < model.config.vocab_size


def test_sampler_commits_every_position_by_the_last_step(diffusion_model):
    from qdif.diffusion.sampler import denoise

    model, _ = diffusion_model
    res = denoise(model, canvas_length=16, steps=4, generator=torch.Generator().manual_seed(0))
    assert res.history[-1].num_committed == 16


def test_sampler_commitment_is_monotone(diffusion_model):
    from qdif.diffusion.sampler import denoise

    model, _ = diffusion_model
    res = denoise(model, canvas_length=16, steps=4, generator=torch.Generator().manual_seed(0))
    counts = [h.num_committed for h in res.history]
    assert counts == sorted(counts)


def test_sampler_accepts_a_prefix(diffusion_model, tokenizer):
    from qdif.diffusion.sampler import denoise

    model, _ = diffusion_model
    prefix = tokenizer("Once upon a time", return_tensors="pt")["input_ids"].to(model.device)
    res = denoise(
        model, canvas_length=16, prefix_ids=prefix, steps=2,
        generator=torch.Generator().manual_seed(0),
    )
    assert res.tokens.shape == (1, 16)


def test_block_generation_extends_the_prefix(diffusion_model, tokenizer):
    from qdif.diffusion.sampler import block_generate

    model, _ = diffusion_model
    prompt = tokenizer("Once upon a time", return_tensors="pt")["input_ids"].to(model.device)
    full, blocks = block_generate(
        model, prompt_ids=prompt, canvas_length=8, num_blocks=2, steps=2,
        generator=torch.Generator().manual_seed(0),
    )
    assert full.shape[1] == prompt.shape[1] + 16
    assert len(blocks) == 2
    assert torch.equal(full[:, : prompt.shape[1]], prompt)


def test_sampler_rejects_zero_steps(diffusion_model):
    from qdif.diffusion.sampler import denoise

    model, _ = diffusion_model
    with pytest.raises(ValueError, match="steps must be"):
        denoise(model, canvas_length=8, steps=0)


# ------------------------------------------------------------------- noise sweep


def test_noise_sweep_runs_end_to_end(diffusion_model, cfg):
    from qdif.data.collator import DiffusionCollator
    from qdif.data.datasets import build_dataset
    from qdif.eval.reconstruction import noise_sweep

    model, tok = diffusion_model
    dataset = build_dataset(cfg.data, tok, cfg.diffusion.canvas_length)
    collator = DiffusionCollator(
        cfg.diffusion, vocab_size=model.config.vocab_size, mask_token_id=tok.eos_token_id
    )
    results = noise_sweep(
        model, tok, dataset, collator, noise_levels=(0.0, 0.5, 1.0), batch_size=1
    )
    assert len(results) == 3
    assert all(r.loss == r.loss for r in results)  # not NaN
    # More noise means fewer positions survive untouched.
    assert results[0].changed_fraction == 0.0
    assert results[-1].changed_fraction > results[1].changed_fraction
