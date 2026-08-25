"""`qdif` command line.

Subcommands are ordered by the workflow they belong to:

    inspect / arch-report / memory      what is this machine and this checkpoint?
    corrupt                             is the forward noising process right?
    smoke-forward / smoke-grad          does the diffusion path run and differentiate?
    probe-bidir                         is the canvas actually bidirectional?
    overfit-one-batch / train           can it learn?
    reconstruct / generate / compare    what does it do?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_CONFIG = "configs/qwen35_0_8b_lora_torch.yaml"
BANNER = "qdif -- Qwen Diffusion Lab (experimental research harness)"


# --------------------------------------------------------------------------- utils


def _echo(*args):
    print(*args, flush=True)


def _rule(title: str = ""):
    width = 78
    if title:
        pad = max(width - len(title) - 3, 0)
        _echo(f"== {title} " + "=" * pad)
    else:
        _echo("=" * width)


def _load_cfg(args) -> "ExperimentConfig":  # noqa: F821
    from .config import ExperimentConfig, load_config

    path = Path(getattr(args, "config", None) or DEFAULT_CONFIG)
    if not path.exists():
        if getattr(args, "config", None):
            raise SystemExit(f"config not found: {path}")
        _echo(f"[warn] {path} not found; using built-in defaults")
        return ExperimentConfig()
    cfg = load_config(path)
    _apply_overrides(cfg, args)
    return cfg


def _apply_overrides(cfg, args) -> None:
    """CLI flags win over the YAML file."""
    for attr, (section, field_name) in {
        "canvas": ("diffusion", "canvas_length"),
        "steps": ("sampler", "steps"),
        "max_steps": ("training", "max_steps"),
        "lora_rank": ("lora", "rank"),
        "seed": ("training", "seed"),
        "device": ("training", "device"),
        "dtype": ("training", "dtype"),
        "model_path": ("model", "local_path"),
        "run_name": ("run", "name"),
        "self_conditioning": ("diffusion", "self_conditioning"),
    }.items():
        value = getattr(args, attr, None)
        if value is not None:
            setattr(getattr(cfg, section), field_name, value)


def _load_model(cfg, echo=_echo, with_lora: bool = True):
    """Build a DiffusionQwen exactly as training would, for eval/inference commands."""
    from .models.qwen35_adapter import load_qwen35_text, resolve_device, resolve_dtype
    from .models.qwen35_diffusion import DiffusionQwen
    from .models.registry import resolve_model_path
    from .training.lora import freeze_base, inject_lora

    device = resolve_device(cfg.training.device)
    dtype = resolve_dtype(cfg.training.dtype)
    path = resolve_model_path(
        cfg.model.id, cfg.model.local_path, local_files_only=cfg.model.local_files_only
    )
    echo(f"[model] {path}")
    echo(f"[model] device={device} dtype={dtype} attn={cfg.model.attn_implementation}")
    base, tokenizer = load_qwen35_text(
        path, dtype=dtype, device=device, attn_implementation=cfg.model.attn_implementation
    )
    report = None
    if with_lora and cfg.training.mode == "lora":
        report = inject_lora(
            base,
            rank=cfg.lora.rank,
            alpha=cfg.lora.alpha,
            dropout=cfg.lora.dropout,
            target_preset=cfg.lora.target_preset,
            target_modules=cfg.lora.target_modules,
            layer_indices=cfg.lora.layer_indices,
            init_scale=cfg.lora.init_scale,
        )
        freeze_base(base)
    model = DiffusionQwen(base, cfg.diffusion).to(device)
    return model, tokenizer, path, report


# ------------------------------------------------------------------------ commands


def cmd_inspect(args) -> int:
    from .env.inspect import full_report
    from .env.memory import ModelShape, standard_profiles

    report = full_report(model_filter=args.filter)
    if args.json:
        _echo(json.dumps(report, indent=2, default=str))
        return 0

    hw = report["hardware"]
    _rule("hardware")
    _echo(f"  {hw['platform']} | {hw['machine']} | {hw['processor']}")
    _echo(f"  python {hw['python']}")
    _echo(f"  memory: {hw['total_memory_gb']:.1f} GB {hw['memory_kind']}")
    if hw.get("mps_recommended_working_set_gb"):
        _echo(
            f"  metal working set (practical GPU ceiling): "
            f"~{hw['mps_recommended_working_set_gb']} GB"
        )
    _echo(f"  torch device: {hw['torch_device']} (mps={hw['mps_available']} cuda={hw['cuda_available']})")

    _rule("packages")
    for name, version in report["packages"].items():
        _echo(f"  {name:<16} {version or '-- not installed'}")

    _rule("transformers Qwen support")
    for name, ok in report["transformers_qwen_support"].items():
        _echo(f"  {name:<16} {'yes' if ok else 'no'}")

    _rule("volumes")
    _echo("  " + ", ".join(report["volumes"]) or "  (none)")

    _rule(f"local models matching {args.filter!r}")
    if not report["local_models"]:
        _echo("  (none found)")
    for m in report["local_models"]:
        flags = []
        if m["quantization"]:
            flags.append(m["quantization"])
        if m["vision"]:
            flags.append("vision")
        if m["mtp"]:
            flags.append("mtp")
        if m["trainable_backbone"]:
            flags.append("TRAINABLE")
        _echo(
            f"  {m['repo_id']:<46} {m['size_gb']:>7.2f} GB  {m['format']:<12} "
            f"{'L' + str(m['layers']) if m['layers'] else '':<6} {','.join(flags)}"
        )
        _echo(f"      {m['path']}")

    _rule("baseline inference servers")
    for s in report["baseline_servers"]:
        _echo(f"  {s['name']:<10} {s['url']:<32} {'UP' if s['up'] else 'down'}")
        for mid in s["models"]:
            _echo(f"      {mid}")

    _rule("memory estimates")
    cfg = _load_cfg(args)
    shape, lora_params, source = _shape_for_memory(cfg)
    _echo(f"  model shape source: {source}")
    _echo(
        f"  {'configuration':<28}{'params':>9}{'grads':>9}{'optim':>9}"
        f"{'activ':>9}{'state':>9}{'TOTAL':>9}"
    )
    for est in standard_profiles(shape, lora_params, seq_len=args.seq_len, batch=args.batch):
        r = est.as_row()
        _echo(
            f"  {r['config']:<28}{r['parameters_gb']:>9.2f}{r['gradients_gb']:>9.2f}"
            f"{r['optimizer_gb']:>9.2f}{r['activations_gb']:>9.2f}"
            f"{r['runtime_state_gb']:>9.3f}{r['total_gb']:>9.2f}"
        )
        if r["notes"]:
            _echo(f"      note: {r['notes']}")
    unit = "unified memory" if hw["torch_device"] == "mps" else hw["memory_kind"]
    _echo(f"  (all figures in GB of {unit}; seq_len={args.seq_len} batch={args.batch})")
    return 0


def _shape_for_memory(cfg):
    """Get a ModelShape without loading weights when possible."""
    import json as _json

    from .env.memory import ModelShape
    from .models.registry import resolve_model_path

    try:
        path = resolve_model_path(
            cfg.model.id, cfg.model.local_path, local_files_only=cfg.model.local_files_only
        )
        raw = _json.loads((Path(path) / "config.json").read_text())
        text = raw.get("text_config", raw)
        layer_types = text.get("layer_types", [])
        n_layers = text["num_hidden_layers"]
        n_full = sum(1 for t in layer_types if t == "full_attention") or n_layers // 4
        n_lin = sum(1 for t in layer_types if t == "linear_attention") or n_layers - n_full
        hidden = text["hidden_size"]
        inter = text["intermediate_size"]
        vocab = text["vocab_size"]
        # Parameter count from config, so `inspect` never has to load weights.
        per_layer = 3 * hidden * inter  # MLP
        head_dim = text.get("head_dim", hidden // text["num_attention_heads"])
        attn = hidden * text["num_attention_heads"] * head_dim * (2 if text.get("attn_output_gate") else 1)
        attn += 2 * hidden * text["num_key_value_heads"] * head_dim
        attn += hidden * text["num_attention_heads"] * head_dim
        key_dim = text.get("linear_num_key_heads", 16) * text.get("linear_key_head_dim", 128)
        val_dim = text.get("linear_num_value_heads", 16) * text.get("linear_value_head_dim", 128)
        delta = hidden * (2 * key_dim + val_dim) + hidden * val_dim + 2 * hidden * text.get(
            "linear_num_value_heads", 16
        ) + val_dim * hidden
        total = hidden * vocab + n_layers * per_layer + n_full * attn + n_lin * delta
        shape = ModelShape(
            total_params=int(total),
            hidden_size=hidden,
            num_layers=n_layers,
            num_full_attention_layers=n_full,
            num_linear_attention_layers=n_lin,
            vocab_size=vocab,
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=head_dim,
            linear_num_value_heads=text.get("linear_num_value_heads", 16),
            linear_key_head_dim=text.get("linear_key_head_dim", 128),
            linear_value_head_dim=text.get("linear_value_head_dim", 128),
            linear_conv_kernel_dim=text.get("linear_conv_kernel_dim", 4),
        )
        # LoRA on q/k/v/o of the full-attention layers, at the configured rank.
        r = cfg.lora.rank
        lora = n_full * r * (
            (hidden + text["num_attention_heads"] * head_dim * 2)
            + 2 * (hidden + text["num_key_value_heads"] * head_dim)
            + (text["num_attention_heads"] * head_dim + hidden)
        )
        return shape, int(lora), f"{cfg.model.id} config.json (estimated, weights not loaded)"
    except (FileNotFoundError, KeyError, ValueError) as exc:
        shape = ModelShape(
            total_params=4_000_000_000,
            hidden_size=2560,
            num_layers=36,
            num_full_attention_layers=9,
            num_linear_attention_layers=27,
            vocab_size=248320,
        )
        return shape, 20_000_000, f"fallback defaults ({type(exc).__name__}: {exc})"


def cmd_arch_report(args) -> int:
    from .models.qwen35_adapter import inspect_architecture

    cfg = _load_cfg(args)
    model, tokenizer, path, _ = _load_model(cfg, with_lora=False)
    report = inspect_architecture(model.base, checkpoint_path=path)

    if args.json:
        payload = report.to_dict()
        payload["tokenizer"] = {
            "class": type(tokenizer).__name__,
            "vocab_size": tokenizer.vocab_size,
            "len": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        out = json.dumps(payload, indent=2, default=str)
        _echo(out)
        if args.out:
            Path(args.out).write_text(out)
        return 0

    _rule("Qwen3.5 architecture")
    _echo(f"  checkpoint          {path}")
    _echo(f"  model class         {report.model_class}")
    _echo(f"  config class        {report.config_class}")
    _echo(f"  layers              {report.num_hidden_layers}")
    _echo(f"  hidden / inter      {report.hidden_size} / {report.intermediate_size}")
    _echo(f"  vocab               {report.vocab_size:,}")
    _echo(f"  total params        {report.total_params:,}")
    _rule("hybrid layer pattern")
    _echo(f"  full attention      {len(report.full_attention_layers)} layers: {report.full_attention_layers}")
    _echo(f"  gated deltanet      {len(report.linear_attention_layers)} layers: {report.linear_attention_layers}")
    _echo(f"  full_attn_interval  {report.full_attention_interval}")
    _echo(f"  attn heads / kv     {report.num_attention_heads} / {report.num_key_value_heads}, head_dim {report.head_dim}")
    _echo(f"  attn output gate    {report.attn_output_gate}")
    _echo(
        f"  deltanet            k-heads {report.linear_num_key_heads} v-heads "
        f"{report.linear_num_value_heads}, k-dim {report.linear_key_head_dim} "
        f"v-dim {report.linear_value_head_dim}, conv {report.linear_conv_kernel_dim}"
    )
    _rule("readout / rope / mtp / vision")
    _echo(f"  tied embeddings     {report.tie_word_embeddings} (lm_head shares storage: {report.lm_head_shares_embedding})")
    _echo(f"  rope                {report.rope}")
    _echo(f"  max positions       {report.max_position_embeddings:,}")
    _echo(f"  mtp layers (config) {report.mtp_num_hidden_layers}; weights in checkpoint: {report.checkpoint_has_mtp}")
    _echo(f"  vision in ckpt      {report.checkpoint_has_vision}; vision loaded: {report.vision_loaded}")
    _rule("adaptable linear modules")
    for name, count in sorted(report.linear_module_suffixes.items()):
        shape = report.module_shapes.get(name, [])
        _echo(f"  {name:<44} x{count:<4} {shape[0] if shape else '?'} -> {shape[1] if len(shape) > 1 else '?'}")
    _rule("findings")
    for note in report.notes:
        _echo(f"  * {note}")

    if args.out:
        Path(args.out).write_text(json.dumps(report.to_dict(), indent=2, default=str))
        _echo(f"\nwrote {args.out}")
    return 0


def cmd_corrupt(args) -> int:
    import torch

    from .diffusion.corruption import corrupt_canvas
    from .models.qwen35_adapter import resolve_dtype  # noqa: F401  (keeps import graph honest)
    from .models.registry import resolve_model_path
    from transformers import AutoTokenizer

    cfg = _load_cfg(args)
    path = resolve_model_path(
        cfg.model.id, cfg.model.local_path, local_files_only=cfg.model.local_files_only
    )
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    ids = tokenizer(args.text, add_special_tokens=False, return_tensors="pt")["input_ids"]

    seed = args.seed if args.seed is not None else 0
    gen = torch.Generator().manual_seed(seed)
    result = corrupt_canvas(
        ids,
        torch.tensor([args.t]),
        vocab_size=tokenizer.vocab_size,
        mode=cfg.diffusion.corruption,
        schedule=cfg.diffusion.schedule,
        mask_token_id=cfg.diffusion.mask_token_id or tokenizer.eos_token_id,
        generator=gen,
    )

    _rule(f"corruption  t={args.t}  mode={cfg.diffusion.corruption}  schedule={cfg.diffusion.schedule}  seed={seed}")
    _echo(f"  tokens: {ids.shape[1]}")
    _echo(f"  targeted for replacement: {float(result.targeted_fraction[0]):.1%}")
    _echo(f"  actually changed:         {float(result.changed_fraction[0]):.1%}")
    _echo("")
    _echo("CLEAN:")
    _echo("  " + tokenizer.decode(result.x0[0]))
    _echo("")
    _echo("NOISY:")
    _echo("  " + tokenizer.decode(result.xt[0]))
    _echo("")
    _rule("per token")
    _echo(f"  {'idx':>4} {'clean id':>9} {'noisy id':>9}  {'clean':<20} {'noisy':<20} changed")
    for i in range(ids.shape[1]):
        c, n = int(result.x0[0, i]), int(result.xt[0, i])
        _echo(
            f"  {i:>4} {c:>9} {n:>9}  {tokenizer.decode([c])!r:<20} "
            f"{tokenizer.decode([n])!r:<20} {'X' if c != n else ''}"
        )
    return 0


def cmd_smoke_forward(args) -> int:
    import torch

    from .data.collator import DiffusionCollator
    from .data.datasets import build_dataset
    from .diffusion.objective import diffusion_loss, diffusion_metrics

    cfg = _load_cfg(args)
    model, tokenizer, path, lora = _load_model(cfg)
    dataset = build_dataset(cfg.data, tokenizer, cfg.diffusion.canvas_length)
    collator = DiffusionCollator(
        cfg.diffusion,
        vocab_size=model.config.vocab_size,
        mask_token_id=tokenizer.eos_token_id,
        generator=torch.Generator().manual_seed(cfg.training.seed),
    )
    examples = [dataset[i] for i in range(min(cfg.training.batch_size, len(dataset)))]
    batch = collator(examples, t=args.t).to(model.device)

    _rule("diffusion forward smoke test")
    _echo(f"  prefix {tuple(batch.prefix_ids.shape)} canvas {tuple(batch.canvas_xt.shape)} t={args.t}")

    model.eval()
    with torch.no_grad():
        out = model(canvas_ids=batch.canvas_xt, t=batch.t, prefix_ids=batch.prefix_ids)
        loss_out = diffusion_loss(out.logits, batch.canvas_x0, batch.corrupted_mask)
        metrics = diffusion_metrics(out.logits, batch.canvas_x0, batch.canvas_xt, batch.corrupted_mask)

    _echo(f"  logits shape        {tuple(out.logits.shape)}  dtype {out.logits.dtype}")
    _echo(f"  logits finite       {bool(torch.isfinite(out.logits).all())}")
    _echo(f"  canvas_start        {out.canvas_start}  total_length {out.total_length}")
    _echo(f"  loss                {float(loss_out.loss):.6f}   finite={bool(torch.isfinite(loss_out.loss))}")
    _echo(f"  per-position loss   {tuple(loss_out.per_position_loss.shape)}")
    _rule("metrics")
    for k, v in metrics.items():
        _echo(f"  {k:<24} {v:.4f}")
    _rule("interpretation")
    if metrics["identity_accuracy"] < 0.2 and metrics["next_token_accuracy"] > metrics["identity_accuracy"]:
        _echo("  Untrained model still behaves autoregressively (next_token_accuracy >")
        _echo("  identity_accuracy). This is the EXPECTED starting point: the diffusion")
        _echo("  readout is unshifted, so LoRA has to learn the new alignment.")
    else:
        _echo("  identity_accuracy already exceeds next_token_accuracy -- either adapters")
        _echo("  are trained, or check the readout alignment in diffusion/objective.py.")

    from .training.diagnostics import format_reconstruction

    _echo("")
    _echo(
        format_reconstruction(
            tokenizer,
            x0=batch.canvas_x0[0],
            xt=batch.canvas_xt[0],
            pred=out.logits[0].argmax(-1),
            prefix=batch.prefix_ids[0],
            t=args.t,
        )
    )
    return 0


def cmd_smoke_grad(args) -> int:
    import torch

    from .data.collator import DiffusionCollator
    from .data.datasets import build_dataset
    from .diffusion.objective import diffusion_loss
    from .models.qwen35_diffusion import parameter_report
    from .training.lora import LoRALinear

    cfg = _load_cfg(args)
    model, tokenizer, path, lora = _load_model(cfg)
    for module in model.auxiliary_modules().values():
        for p in module.parameters():
            p.requires_grad_(True)

    params = parameter_report(model)
    _rule("parameter report")
    _echo(f"  total          {params['total_params']:>15,}")
    _echo(f"  trainable      {params['trainable_params']:>15,}  ({params['percent_trainable']:.4f}%)")
    _echo(f"  lora           {params['lora_params']:>15,}")
    _echo(f"  auxiliary      {params['auxiliary_params']:>15,}")
    _echo(f"  frozen         {params['frozen_params']:>15,}")
    _echo(f"  param storage  {params['param_gb']:.2f} GB")
    if lora:
        _echo(f"  lora modules   {lora.num_injected} ({', '.join(lora.target_suffixes)})")
        _echo(f"  lora layers    {lora.layer_indices if lora.layer_indices else 'all'}")

    dataset = build_dataset(cfg.data, tokenizer, cfg.diffusion.canvas_length)
    collator = DiffusionCollator(
        cfg.diffusion,
        vocab_size=model.config.vocab_size,
        mask_token_id=tokenizer.eos_token_id,
        generator=torch.Generator().manual_seed(cfg.training.seed),
    )
    batch = collator([dataset[0]], t=args.t).to(model.device)

    model.train()
    out = model(canvas_ids=batch.canvas_xt, t=batch.t, prefix_ids=batch.prefix_ids)
    loss_out = diffusion_loss(out.logits, batch.canvas_x0, batch.corrupted_mask)
    loss_out.loss.backward()

    _rule("gradient verification")
    _echo(f"  loss {float(loss_out.loss.detach()):.6f}")

    lora_with_grad = lora_without = 0
    base_with_grad: list[str] = []
    lora_norms = []
    for name, p in model.named_parameters():
        is_lora = ".lora_A" in name or ".lora_B" in name
        is_aux = name.startswith(("timestep_conditioner.", "self_conditioner."))
        if is_lora:
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                lora_with_grad += 1
                lora_norms.append(float(p.grad.float().norm()))
            else:
                lora_without += 1
        elif not is_aux and p.grad is not None and float(p.grad.abs().sum()) > 0:
            base_with_grad.append(name)

    aux_grads = {
        name: float(p.grad.float().norm()) if p.grad is not None else None
        for name, p in model.named_parameters()
        if name.startswith(("timestep_conditioner.", "self_conditioner."))
    }

    frozen_ok = not base_with_grad
    _echo(f"  LoRA params with nonzero grad     {lora_with_grad}")
    _echo(f"  LoRA params with zero/no grad     {lora_without}")
    if lora_norms:
        _echo(f"  LoRA grad norm  min {min(lora_norms):.3e}  max {max(lora_norms):.3e}")
    _echo(f"  frozen base weights received grad {'NO  (correct)' if frozen_ok else 'YES (WRONG)'}")
    if not frozen_ok:
        _echo(f"    offenders: {base_with_grad[:5]}")
    if aux_grads:
        _echo("  auxiliary module grads:")
        for name, norm in aux_grads.items():
            _echo(f"    {name:<44} {norm if norm is None else f'{norm:.3e}'}")

    # lora_B is zero-initialised, so at step 0 its gradient is nonzero while
    # lora_A's is exactly zero. That is correct, not a failure.
    zero_a = [
        n for n, p in model.named_parameters()
        if ".lora_A" in n and p.grad is not None and float(p.grad.abs().sum()) == 0
    ]
    if zero_a:
        _echo(
            f"  note: {len(zero_a)} lora_A tensors have zero grad at step 0 -- expected, "
            f"because lora_B is zero-initialised (dL/dA = B^T * ... = 0)."
        )

    ok = frozen_ok and (lora_with_grad > 0 or any(v for v in aux_grads.values()))
    _rule("result")
    _echo("  PASS" if ok else "  FAIL")
    return 0 if ok else 1


def cmd_probe_bidir(args) -> int:
    from .eval.reconstruction import probe_bidirectionality

    cfg = _load_cfg(args)
    model, tokenizer, path, _ = _load_model(cfg)
    result = probe_bidirectionality(model, tokenizer, args.text, canvas_length=args.canvas or 16)

    if args.json:
        _echo(json.dumps(result, indent=2))
        return 0

    _rule("bidirectionality probe")
    _echo(f"  full-attention layers: {result['full_attention_layers']}")
    _echo("  perturbing the LAST canvas token; measuring hidden-state change at earlier positions")
    for label in ("bidirectional", "causal_control"):
        r = result[label]
        _echo(f"\n  [{label}]")
        _echo(f"    earlier canvas positions affected: {r['num_earlier_positions_affected']}/{r['canvas_positions'] - 1}")
        _echo(f"    max delta before last position:    {r['max_delta_before_last_position']:.6e}")
    bidi = result["bidirectional"]["max_delta_before_last_position"]
    causal = result["causal_control"]["max_delta_before_last_position"]
    _rule("result")
    if bidi > 0 and causal == 0:
        _echo("  PASS: canvas bidirectionality is real and is produced by the full-attention")
        _echo("  layers alone. The 18 Gated DeltaNet layers remain strictly causal, as")
        _echo("  documented in docs/QWEN35_NOTES.md.")
        return 0
    _echo(f"  UNEXPECTED: bidirectional={bidi:.3e} causal={causal:.3e}")
    return 1


def _trainer_for(cfg):
    """Dispatch to the backend named in the config. No silent fallback."""
    if cfg.training.backend == "mlx":
        from .mlx_backend.trainer import train

        return train
    if cfg.training.backend == "torch":
        from .training.torch_trainer import train

        return train
    raise ValueError(f"unknown training.backend {cfg.training.backend!r}; expected mlx or torch")


def cmd_train(args) -> int:
    cfg = _load_cfg(args)
    summary = _trainer_for(cfg)(cfg, echo=_echo, fixed_batch=args.fixed_batch)
    _rule("summary")
    _echo(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_overfit(args) -> int:
    """The mandatory first experiment: memorise one batch, or stop and debug."""
    cfg = _load_cfg(args)
    # Derive the run directory from the config's own name so two overfit runs from
    # two different configs cannot silently write into the same directory.
    cfg.run.name = args.run_name or f"{cfg.run.name}-overfit"
    # --max-steps wins; otherwise respect the config but never run fewer than 60
    # steps, which is too few for the memorisation signal to be readable.
    cfg.training.max_steps = args.max_steps or max(cfg.training.max_steps, 60)
    cfg.training.log_every = max(cfg.training.max_steps // 20, 1)
    cfg.training.sample_every = max(cfg.training.max_steps // 4, 1)
    # A single batch cannot be denoised at every noise level at once; clamp the
    # timestep range so the test is about memorisation, not about the full schedule.
    cfg.diffusion.t_max = args.t_max
    cfg.data.max_examples = max(cfg.training.batch_size, 4)

    _rule("overfit one batch")
    _echo(f"  backend {cfg.training.backend} | steps {cfg.training.max_steps} | "
          f"batch {cfg.training.batch_size} | canvas {cfg.diffusion.canvas_length} | "
          f"t in [{cfg.diffusion.t_min}, {cfg.diffusion.t_max}]")
    if cfg.training.backend == "mlx":
        b = cfg.bidirectional_deltanet
        _echo(f"  bidirectional DeltaNet: {b.enabled} (fusion {b.fusion})")
    summary = _trainer_for(cfg)(cfg, echo=_echo, fixed_batch=True)

    first, final = summary["first_loss"], summary["final_loss"]
    acc = summary["final_identity_accuracy"] or 0.0
    corr = summary["final_corrupted_accuracy"] or 0.0
    lift = summary["final_lift_over_copy"] or 0.0
    mean_corr = summary["mean_corrupted_accuracy_last_quarter"] or 0.0
    mean_lift = summary["mean_lift_last_quarter"] or 0.0

    _rule("verdict")
    _echo(f"  loss                          {first:.4f} -> {final:.4f}")
    _echo(f"  identity accuracy             {summary['first_identity_accuracy']:.1%} -> {acc:.1%}")
    _echo(f"  copy-baseline accuracy        {summary['final_copy_baseline_accuracy']:.1%}"
          "   (what a pure copier scores)")
    _echo(f"  lift over copy                {lift:+.1%}   (mean over last quarter: {mean_lift:+.1%})")
    _echo(f"  corrupted-position accuracy   {corr:.1%}   (mean over last quarter: {mean_corr:.1%})")
    _echo(f"  next-token accuracy           {summary['final_next_token_accuracy']:.1%}"
          "   (falls as AR behaviour is abandoned)")
    _echo(f"  peak memory                   {summary['peak_memory_gb']:.2f} GB {summary['memory_kind']}")

    loss_fell = final < first * 0.5
    denoising = mean_corr > 0.15 and mean_lift > 0.02
    copy_rate = summary["final_copy_rate"] or 0.0
    _echo("")

    if loss_fell and denoising:
        _echo("  PASS: loss fell AND corrupted positions are repaired above the copy")
        _echo("  baseline. The optimisation path works; proceed to a real training run.")
        _echo(f"  full log: {summary['run_dir']}")
        return 0

    if not loss_fell:
        _echo("  FAIL (no learning): the loss is not falling. Do NOT train longer --")
        _echo("  debug the readout alignment, the LoRA targets and the learning rate.")
    elif mean_lift < -0.10:
        _echo("  FAIL (mode collapse): the loss fell, but identity accuracy is now WELL")
        _echo("  BELOW what a pure copier would score. The model is emitting a few")
        _echo("  high-frequency tokens across the whole canvas and destroying the clean")
        _echo("  positions along with the corrupted ones.")
        _echo("")
        _echo("  Typical with loss_on='corrupted': nothing rewards leaving clean tokens")
        _echo("  alone, so the model stops distinguishing them. Consider a blended")
        _echo("  objective, or more adapter capacity (lora.target_preset=full_attn_mlp).")
    elif copy_rate > 0.5 and mean_corr < 0.15:
        _echo("  FAIL (copy collapse): the loss fell and identity accuracy rose, but")
        _echo("  corrupted-position accuracy did not, and the lift over a pure copier is")
        _echo("  ~zero. The model has learned to ECHO ITS INPUT, not to denoise.")
        _echo("")
        _echo("  Under uniform corruption with loss_on='all', copying is a strong attractor:")
        _echo("  most positions are uncorrupted, the model cannot tell which, and echoing")
        _echo("  scores (1-t) for free.")
        _echo("")
        _echo("  Measured behaviour (docs/EXPERIMENTS.md 001 vs 004): this is usually a")
        _echo("  TRANSIENT the optimisation passes through, not the end state. The same")
        _echo("  config at 150 steps reached 95% corrupted-position accuracy. Before")
        _echo("  changing the objective, re-run with --max-steps 150.")
    else:
        _echo("  FAIL (partial): the loss fell but corrupted-position accuracy stayed")
        _echo(f"  below the {0.15:.0%} threshold ({mean_corr:.1%}). Not yet denoising.")
    _echo("")
    _echo("  Do NOT compensate by training longer. See docs/EXPERIMENTS.md.")
    _echo(f"  full log: {summary['run_dir']}")
    return 1


def cmd_reconstruct(args) -> int:
    import torch

    from .data.collator import DiffusionCollator
    from .data.datasets import build_dataset
    from .eval.reconstruction import DEFAULT_NOISE_LEVELS, noise_sweep

    cfg = _load_cfg(args)
    model, tokenizer, path, _ = _load_model(cfg)
    if args.checkpoint:
        from .training.checkpoint import load_checkpoint

        meta = load_checkpoint(model, args.checkpoint)
        _echo(f"[checkpoint] loaded {args.checkpoint} ({meta.get('loaded_lora_tensors')} lora tensors, step {meta.get('step')})")
    else:
        _echo("[checkpoint] none -- evaluating the UNADAPTED base model")

    dataset = build_dataset(cfg.data, tokenizer, cfg.diffusion.canvas_length)
    collator = DiffusionCollator(
        cfg.diffusion,
        vocab_size=model.config.vocab_size,
        mask_token_id=tokenizer.eos_token_id,
        generator=torch.Generator().manual_seed(cfg.training.seed),
    )
    levels = [args.noise] if args.noise is not None else list(DEFAULT_NOISE_LEVELS)
    results = noise_sweep(
        model, tokenizer, dataset, collator, noise_levels=levels, batch_size=args.batch, seed=cfg.training.seed
    )

    _rule("reconstruction sweep")
    for r in results:
        _echo("  " + r.summary())
    if args.show_examples:
        for r in results:
            _echo("")
            _echo(r.example)
    _rule("how to read this")
    _echo("  Before adaptation, identity accuracy near 0 with high next-token accuracy is")
    _echo("  correct: the diffusion readout is unshifted while the model is still an AR")
    _echo("  predictor. After adaptation, t=0 identity accuracy must approach 100%.")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps([{**vars(r), "example": r.example} for r in results], indent=2, default=str)
        )
        _echo(f"\nwrote {args.json_out}")
    return 0


def cmd_generate(args) -> int:
    from .inference.diffusion_generate import generate

    cfg = _load_cfg(args)
    model, tokenizer, path, _ = _load_model(cfg)
    if args.checkpoint:
        from .training.checkpoint import load_checkpoint

        load_checkpoint(model, args.checkpoint)
        _echo(f"[checkpoint] {args.checkpoint}")
    else:
        _echo("[checkpoint] none -- generating with the UNADAPTED base model "
              "(output is expected to be incoherent; this is a plumbing check)")

    model.eval()
    result = generate(model, tokenizer, args.prompt, cfg, steps=args.steps, num_blocks=args.blocks)
    _rule(f"diffusion generation ({result.steps_per_block} steps x {result.num_blocks} blocks)")
    _echo("  " + result.summary())
    _echo("")
    _echo("PROMPT:")
    _echo("  " + args.prompt)
    _echo("")
    _echo("GENERATED:")
    _echo("  " + result.output_text.replace("\n", "\\n"))
    if args.show_steps and result.blocks:
        _rule("per-step trace (block 0)")
        _echo(f"  {'step':>4} {'t':>6} {'committed':>10} {'top1':>8} {'entropy':>9} {'changed':>8}")
        for rec in result.blocks[0].history:
            _echo(
                f"  {rec.step:>4} {rec.t:>6.3f} {rec.num_committed:>10} "
                f"{rec.mean_top1_prob:>8.4f} {rec.mean_entropy:>9.4f} {rec.num_changed:>8.1f}"
            )
    return 0


def cmd_compare(args) -> int:
    from .inference.ar_baseline import run_transformers_baseline
    from .inference.diffusion_generate import generate

    cfg = _load_cfg(args)
    model, tokenizer, path, _ = _load_model(cfg)
    if args.checkpoint:
        from .training.checkpoint import load_checkpoint

        load_checkpoint(model, args.checkpoint)
        _echo(f"[checkpoint] {args.checkpoint}")
    model.eval()

    _rule("AR baseline (same weights, adapters disabled)")
    ar = run_transformers_baseline(
        model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens, disable_adapters=True
    )
    _echo("  " + ar.summary())
    _echo("  OUTPUT: " + ar.output_text.replace("\n", "\\n")[:400])

    ar_on = None
    if args.checkpoint:
        _rule("AR path WITH diffusion adapters loaded (research question 8)")
        ar_on = run_transformers_baseline(
            model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens, disable_adapters=False
        )
        _echo("  " + ar_on.summary())
        _echo("  OUTPUT: " + ar_on.output_text.replace("\n", "\\n")[:400])
        off, on = ar.reference_perplexity, ar_on.reference_perplexity
        if off and on:
            _echo(
                f"  reference perplexity: {off:.3f} (adapters off) -> {on:.3f} (on), "
                f"x{on / off:.2f}"
            )
            _echo("  This -- not self-perplexity -- is the answer to research question 8.")
            _echo("  A large ratio means the diffusion adapters damage the AR path and the")
            _echo("  two modes cannot coexist without toggling.")

    _rule("diffusion generation")
    dif = generate(model, tokenizer, args.prompt, cfg, steps=args.steps)
    _echo("  " + dif.summary())
    _echo("  OUTPUT: " + dif.output_text.replace("\n", "\\n")[:400])

    if args.mlx:
        _rule("AR baseline via mlx-lm (independent implementation)")
        try:
            from .inference.ar_baseline import run_mlx_baseline

            mlx_res = run_mlx_baseline(str(path), args.prompt, max_new_tokens=args.max_new_tokens)
            _echo("  " + mlx_res.summary())
            _echo("  OUTPUT: " + mlx_res.output_text.replace("\n", "\\n")[:400])
        except Exception as exc:  # noqa: BLE001 - baseline is optional
            _echo(f"  unavailable: {type(exc).__name__}: {exc}")

    _rule("comparison")
    _echo(f"  {'mode':<26}{'tok':>6}{'sec':>8}{'tok/s':>10}{'tok/forward':>14}")
    _echo(f"  {'autoregressive':<26}{ar.output_tokens:>6}{ar.seconds:>8.2f}{ar.tokens_per_sec:>10.1f}{1.0:>14.2f}")
    _echo(
        f"  {'diffusion':<26}{dif.output_tokens:>6}{dif.seconds:>8.2f}"
        f"{dif.tokens_per_sec:>10.1f}{dif.tokens_per_forward:>14.2f}"
    )
    _echo("")
    _echo("  tok/forward is the number that matters for diffusion decoding: AR is 1.0 by")
    _echo("  construction. A speed advantage only exists if quality holds at high values.")
    return 0


def cmd_smoke_bidir(args) -> int:
    """v0.2 architecture smoke test: the 12 checks that gate any training run."""
    import mlx.core as mx

    from .mlx_backend.build import build_mlx_setup
    from .mlx_backend.probe import probe_directionality
    from .mlx_backend.smoke import (
        active_memory_gb,
        benchmark_directions,
        check_directional_outputs,
        check_loss_and_gradients,
        peak_unified_memory_gb,
    )

    cfg = _load_cfg(args)
    if cfg.training.backend != "mlx":
        _echo(f"smoke-bidir requires training.backend: mlx (config says {cfg.training.backend!r})")
        return 1

    _rule("1-2  load + discovered architecture")
    setup = build_mlx_setup(cfg, echo=_echo)
    r = setup.load_report
    _echo(f"  model_type          {r.model_type}")
    _echo(f"  layers              {r.num_hidden_layers}")
    _echo(f"  hidden / inter      {r.hidden_size} / {r.intermediate_size}")
    _echo(f"  vocab               {r.vocab_size:,}")
    _echo(f"  attn heads / kv     {r.num_attention_heads} / {r.num_key_value_heads}, head_dim {r.head_dim}")
    _echo(f"  full attention      {len(r.full_attention_layers)} layers: {r.full_attention_layers}")
    _echo(f"  gated deltanet      {len(r.linear_attention_layers)} layers")
    _echo(f"  full_attn_interval  {r.full_attention_interval}")
    _echo(
        f"  deltanet heads      k={r.linear_num_key_heads} v={r.linear_num_value_heads} "
        f"(repeat factor {r.head_repeat_factor}), k-dim {r.linear_key_head_dim} "
        f"v-dim {r.linear_value_head_dim}, conv {r.linear_conv_kernel_dim}"
    )
    _echo(f"  tied embeddings     {r.tie_word_embeddings}")
    _echo(f"  vision loaded       {r.vision_loaded}")
    _echo(f"  params              {r.total_params:,} ({r.param_bytes / 1e9:.2f} GB {r.dtype})")
    for note in r.notes:
        _echo(f"  * {note}")

    checks = []
    _rule("3-7  directional outputs, alignment, fusion, weight sharing")
    checks += check_directional_outputs(setup, canvas_length=16, prefix_length=8)
    for c in checks:
        _echo(c.line())

    _rule("8  bidirectionality probe (per-layer)")
    probe = probe_directionality(
        setup,
        args.text,
        canvas_length=16,
        layer_index=args.probe_layer,
    )
    _echo(f"  probing DeltaNet layer {probe['layer_probed']} of {probe['deltanet_layers'][:6]}...")
    from .mlx_backend.probe import DirectionStats

    for key in ("causal", "bidirectional"):
        _echo(DirectionStats(**probe[key]).summary())
    _echo(f"  [{'PASS' if probe['pass'] else 'FAIL'}] {probe['interpretation']}")

    _rule("9-10  diffusion loss and gradient verification")
    grad = check_loss_and_gradients(setup, cfg, canvas_length=16, prefix_length=8)
    _echo(f"  loss                      {grad['loss']:.6f}  finite={grad['finite']}")
    _echo(f"  grad tensors              {grad['num_grad_tensors']:,}")
    _echo(f"  nonzero grads             {grad['nonzero_grads']} by family {grad['nonzero_by_family']}")
    _echo(f"  zero grads                {grad['zero_grads']}")
    _echo(f"  frozen base untouched     {'YES (correct)' if grad['frozen_clean'] else 'NO (WRONG)'}")
    if grad["unexpected_base_grads"]:
        _echo(f"    offenders: {grad['unexpected_base_grads']}")
    _echo("  largest grad norms:")
    for path, n in grad["grad_norms"].items():
        _echo(f"    {path:<58} {n:.6f}")
    _echo("  metrics on the smoke batch (untrained model, random tokens):")
    for k in ("identity_accuracy", "corrupted_accuracy", "copy_baseline_accuracy",
              "lift_over_copy", "next_token_accuracy", "mean_entropy"):
        _echo(f"    {k:<26} {grad['metrics'][k]:.4f}")

    _rule("11-12  memory and runtime ratio")
    bench = benchmark_directions(setup, canvas_length=args.canvas or 32, prefix_length=16)
    c, b, ratio = bench["causal"], bench["bidirectional"], bench["ratio"]
    _echo(f"  canvas {bench['canvas_length']} + prefix {bench['prefix_length']}")
    _echo(f"  {'condition':<18}{'forward s':>12}{'fwd+bwd s':>12}")
    _echo(f"  {'causal (A)':<18}{c['forward_s']:>12.4f}{c['fwd_bwd_s']:>12.4f}")
    _echo(f"  {'bidirectional':<18}{b['forward_s']:>12.4f}{b['fwd_bwd_s']:>12.4f}")
    _echo(f"  {'ratio v0.2/v0.1':<18}{ratio['forward']:>12.2f}x{ratio['fwd_bwd']:>11.2f}x")
    _echo(f"  peak unified memory  {peak_unified_memory_gb():.2f} GB (Apple unified memory, not VRAM)")
    _echo(f"  active memory        {active_memory_gb():.2f} GB")

    _rule("parameter budget")
    p = setup.params
    _echo(f"  total          {p['total_params']:>15,}")
    _echo(f"  trainable      {p['trainable_params']:>15,}  ({p['percent_trainable']:.5f}%)")
    _echo(f"  frozen         {p['frozen_params']:>15,}")
    _echo(f"  by family      {p['trainable_by_family']}")
    _echo(f"  weights        {p['param_gb']:.2f} GB   trainable {p['trainable_mb']:.2f} MB")

    all_pass = all(c.passed for c in checks) and probe["pass"] and grad["finite"] and grad["frozen_clean"]
    _rule("verdict")
    _echo("  ALL CHECKS PASSED -- safe to run the one-batch test." if all_pass
          else "  FAILURES PRESENT -- stop and diagnose before training.")

    if args.json_out:
        payload = {
            "setup": setup.summary(),
            "checks": [c.__dict__ for c in checks],
            "probe": probe,
            "gradients": grad,
            "benchmark": bench,
            "peak_unified_gb": peak_unified_memory_gb(),
            "all_pass": all_pass,
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2, default=str))
        _echo(f"\n  wrote {args.json_out}")
    return 0 if all_pass else 1


def cmd_heldout(args) -> int:
    """Phase 3: train one arm and track held-out denoising throughout."""
    from .mlx_backend.heldout import run_arm

    cfg = _load_cfg(args)
    if cfg.training.backend != "mlx":
        _echo("heldout requires training.backend: mlx")
        return 1
    arm = args.arm or cfg.run.name
    _rule(f"Phase 3 held-out run: arm {arm}")
    b = cfg.bidirectional_deltanet
    _echo(f"  DeltaNet: enabled={b.enabled} mode={b.mode} fusion={b.fusion}")
    _echo(f"  {cfg.training.max_steps} steps, batch {cfg.training.batch_size}, "
          f"canvas {cfg.diffusion.canvas_length}, t in [{cfg.diffusion.t_min}, {cfg.diffusion.t_max}]")
    summary = run_arm(cfg, arm=arm, echo=_echo)
    _rule("arm summary")
    _echo(f"  wall {summary['wall_seconds']:.0f}s | {summary['tokens_per_sec']:.1f} tok/s "
          f"| peak {summary['peak_unified_gb']:.2f} GB unified")
    for key, p in summary["final"].items():
        _echo(f"  {key:<10} corr-acc {p['corrupted_accuracy']:6.2%} "
              f"lift {p['lift_over_copy']:+6.2%} ce {p['loss']:6.3f} "
              f"exact {p['exact_reconstruction']:5.1%}")
    return 0


def cmd_p3_report(args) -> int:
    """Compare Phase 3 arms and apply the pre-registered criteria."""
    import numpy as np

    runs = {}
    for spec in args.runs:
        arm, _, path = spec.partition("=")
        p = Path(path or f"runs/p3-{arm}")
        curve = p / "heldout_curve.jsonl"
        if not curve.exists():
            _echo(f"missing {curve}")
            return 1
        rows = [json.loads(line) for line in curve.read_text().splitlines() if line.strip()]
        summ = json.loads((p / "summary.json").read_text()) if (p / "summary.json").exists() else {}
        runs[arm] = {"rows": rows, "summary": summ}

    steps = sorted({r["step"] for r in runs[next(iter(runs))]["rows"]})
    final = max(steps)

    def get(arm, step, t, key="corrupted_accuracy"):
        for r in runs[arm]["rows"]:
            if r["step"] == step and abs(r["t"] - t) < 1e-9:
                return r[key]
        return None

    _rule("held-out corrupted-position accuracy at the final step")
    levels = sorted({r["t"] for r in runs[next(iter(runs))]["rows"] if r["step"] == final})
    _echo("  " + f"{'arm':<14}" + "".join(f"{f't={t:.2f}':>11}" for t in levels))
    for arm in runs:
        _echo("  " + f"{arm:<14}" + "".join(
            f"{(get(arm, final, t) or float('nan')):>11.2%}" for t in levels))

    _rule("held-out lift over copy at the final step")
    _echo("  " + f"{'arm':<14}" + "".join(f"{f't={t:.2f}':>11}" for t in levels))
    for arm in runs:
        _echo("  " + f"{arm:<14}" + "".join(
            f"{(get(arm, final, t, 'lift_over_copy') or float('nan')):>+11.2%}" for t in levels))

    _rule("cross entropy at the final step")
    _echo("  " + f"{'arm':<14}" + "".join(f"{f't={t:.2f}':>11}" for t in levels))
    for arm in runs:
        _echo("  " + f"{arm:<14}" + "".join(
            f"{(get(arm, final, t, 'loss') or float('nan')):>11.3f}" for t in levels))
    _echo("  NOTE: t=1.00 measures the prior, not denoising -- at full corruption the")
    _echo("  particular held-out continuation is not identifiable from the canvas.")

    if "B" in runs and "C" in runs:
        _rule("aligned - shuffled  (corrupted-position accuracy) over training")
        _echo("  This is the measurement the v0.2 hypothesis lives or dies on.")
        _echo("  " + f"{'step':>6}" + "".join(f"{f't={t:.2f}':>11}" for t in TRACKED))
        for s in steps:
            row = "".join(
                f"{((get('B', s, t) or 0) - (get('C', s, t) or 0)):>+11.2%}" for t in TRACKED
            )
            _echo(f"  {s:>6}{row}")

    _rule("gate behaviour / cost")
    _echo("  " + f"{'arm':<14}{'gate':>8}{'tok/s':>10}{'s/step':>9}{'peak GB':>10}{'wall s':>9}")
    for arm, d in runs.items():
        s = d["summary"]
        g = s.get("gates", {}).get("gate_mean")
        _echo("  " + f"{arm:<14}{(f'{g:.3f}' if g is not None else '--'):>8}"
              f"{s.get('tokens_per_sec', float('nan')):>10.1f}"
              f"{s.get('mean_seconds_per_step', float('nan')):>9.3f}"
              f"{s.get('peak_unified_gb', float('nan')):>10.2f}"
              f"{s.get('wall_seconds', float('nan')):>9.0f}")

    if "A" in runs and "B" in runs:
        _rule("PRE-REGISTERED CRITERIA (docs/EXPERIMENTS.md, fixed before the run)")
        hi = [t for t in levels if t >= 0.50 - 1e-9 and t < 1.0]
        lift_b = np.mean([get("B", final, t, "lift_over_copy") for t in hi])
        lift_a = np.mean([get("A", final, t, "lift_over_copy") for t in hi])
        margin = (lift_b - lift_a) * 100
        wins = all(
            (get("B", final, t) or 0) > (get("A", final, t) or 0) for t in (0.50, 0.75)
        )
        _echo(f"  mean lift over t>=0.50 : B {lift_b:+.2%} vs A {lift_a:+.2%} "
              f"-> margin {margin:+.2f} points (need >= +3.00)")
        _echo(f"  B beats A at BOTH t=0.50 and t=0.75 individually: {wins}")
        gate = runs["B"]["summary"].get("gates", {}).get("gate_mean")
        if gate is not None:
            _echo(f"  learned gate mean: {gate:.3f} (kill if >= 0.900)")
        cont = margin >= 3.0 and wins
        tie = abs(margin) <= 1.0
        _echo("")
        if cont:
            _echo("  VERDICT: CONTINUE v0.2 -- criterion met.")
        elif tie:
            _echo("  VERDICT: CONTINUE v0.1 -- A is within 1 point of B. The 1.2x compute")
            _echo("  and the extra machinery are not worth a tie.")
        else:
            _echo("  VERDICT: v0.2 continue criterion NOT met.")
        if "C" in runs:
            d_bc = np.mean([
                (get("B", final, t) or 0) - (get("C", final, t) or 0) for t in hi
            ]) * 100
            _echo(f"  aligned - shuffled at t>=0.50: {d_bc:+.2f} points")
            if d_bc <= 0:
                _echo("  KILL CRITERION FIRES: the shuffled control is not worse than aligned")
                _echo("  fusion after training, so the reverse pass is a perturbation, not")
                _echo("  position-aligned information.")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(
            {a: {"summary": d["summary"], "rows": d["rows"]} for a, d in runs.items()},
            indent=2, default=str))
        _echo(f"\n  wrote {args.json_out}")
    return 0


TRACKED = (0.50, 0.75, 0.90)


def cmd_compare_fusion(args) -> int:
    """Untrained loss per fusion strategy, plus the gate sweep that de-confounds it."""
    from .mlx_backend.build import build_mlx_setup
    from .mlx_backend.fusion_eval import compare_fusions, gate_sweep

    cfg = _load_cfg(args)
    if cfg.training.backend != "mlx":
        _echo("compare-fusion requires training.backend: mlx")
        return 1
    setup = build_mlx_setup(cfg, echo=_echo)

    levels = tuple(args.noise_levels)
    res = compare_fusions(setup, cfg, noise_levels=levels, num_batches=args.batches)
    _rule("untrained diffusion loss by fusion strategy")
    _echo(f"  {res['num_batches']} batches of {res['batch_size']}, canvas "
          f"{res['canvas_length']}, gate_init {res['gate_init']}, NO training")
    header = "  " + f"{'condition':<18}" + "".join(f"{f't={t}':>12}" for t in levels)
    _echo(header)
    baseline = res["results"]["A_causal"]
    for name, r in res["results"].items():
        row = f"  {name:<18}" + "".join(f"{r[t]['loss']:>12.4f}" for t in levels)
        _echo(row)
    _echo("")
    _echo("  delta vs A_causal (negative = fusion helps the untrained model):")
    for name, r in res["results"].items():
        if name == "A_causal":
            continue
        row = f"  {name:<18}" + "".join(
            f"{r[t]['loss'] - baseline[t]['loss']:>+12.4f}" for t in levels
        )
        _echo(row)

    sweep = gate_sweep(setup, cfg, t_val=args.gate_sweep_t, num_batches=args.batches)
    _rule(f"scalar-gate sweep at t={sweep['t']}  (g=1 is causal, g=0.5 is mean, g=0 is reverse-only)")
    _echo(f"  {'g':>6}{'loss':>12}{'delta vs g=1':>16}")
    base_g = sweep["gates"][1.0]["loss"]
    for g, r in sweep["gates"].items():
        _echo(f"  {g:>6.2f}{r['loss']:>12.4f}{r['loss'] - base_g:>+16.4f}")
    from .mlx_backend.fusion_eval import control_sweep

    ctrl = control_sweep(setup, cfg, t_val=args.gate_sweep_t, num_batches=args.batches)
    _rule(f"de-confounding controls at t={ctrl['t']}")
    _echo("  scalar_gate              = g*H_f + (1-g)*H_r        (the real thing)")
    _echo("  forward_scaled_control   = g*H_f                    (attenuation only)")
    _echo("  shuffled_reverse_control = g*H_f + (1-g)*shuffle(H_r) (alignment destroyed)")
    _echo("")
    _echo("  " + f"{'condition':<26}" + "".join(f"{f'g={g}':>10}" for g in ctrl["gates"]))
    for name, rows in ctrl["conditions"].items():
        _echo("  " + f"{name:<26}" + "".join(f"{rows[g]['loss']:>10.4f}" for g in ctrl["gates"]))
    _rule("how to read this")
    _echo("  The gate sweep alone is confounded: lowering g both admits the reverse")
    _echo("  direction AND attenuates the pretrained forward path. Compare against the")
    _echo("  controls. If scalar_gate beats forward_scaled_control at the same g, the")
    _echo("  reverse recurrence carries real information; if it only beats")
    _echo("  shuffled_reverse_control too, that information is position-aligned.")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(
                {"fusions": res, "gate_sweep": sweep, "controls": ctrl}, indent=2, default=str
            )
        )
        _echo(f"\n  wrote {args.json_out}")
    return 0


def cmd_unsloth_report(args) -> int:
    from .mlx_backend.capability import capability_report, render

    report = capability_report()
    if args.json:
        _echo(json.dumps(report, indent=2, default=str))
        return 0
    render(report, _echo, _rule)
    return 0


def cmd_memory(args) -> int:
    from .env.memory import standard_profiles

    cfg = _load_cfg(args)
    shape, lora_params, source = _shape_for_memory(cfg)
    profiles = standard_profiles(shape, lora_params, seq_len=args.seq_len, batch=args.batch)
    if args.json:
        _echo(json.dumps([p.as_row() for p in profiles], indent=2))
        return 0
    _rule(f"memory estimate: {cfg.model.id}")
    _echo(f"  source: {source}")
    _echo(f"  params {shape.total_params:,} | layers {shape.num_layers} "
          f"({shape.num_full_attention_layers} full attn / {shape.num_linear_attention_layers} deltanet)")
    _echo(f"  lora params at r={cfg.lora.rank}: {lora_params:,}")
    for p in profiles:
        r = p.as_row()
        _echo("")
        _echo(f"  {r['config']}   TOTAL {r['total_gb']:.2f} GB")
        _echo(f"    parameters {r['parameters_gb']:.2f} | gradients {r['gradients_gb']:.2f} "
              f"| optimizer {r['optimizer_gb']:.2f} | activations {r['activations_gb']:.2f} "
              f"| runtime state {r['runtime_state_gb']:.3f}")
        if r["notes"]:
            _echo(f"    note: {r['notes']}")
    return 0


# ---------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qdif", description=BANNER)
    p.add_argument("--config", "-c", default=None, help=f"YAML config (default {DEFAULT_CONFIG})")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--config", "-c", default=None)
        sp.add_argument("--model-path", default=None, help="override model.local_path")
        sp.add_argument("--device", default=None)
        sp.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"])
        sp.add_argument("--seed", type=int, default=None)
        sp.add_argument("--canvas", type=int, default=None, help="override diffusion.canvas_length")
        sp.add_argument("--lora-rank", type=int, default=None)
        sp.add_argument("--run-name", default=None)
        sp.add_argument("--self-conditioning", action="store_true", default=None)
        return sp

    sp = common(sub.add_parser("inspect", help="environment, hardware, local models, memory"))
    sp.add_argument("--filter", default="qwen", help="substring filter for local model scan")
    sp.add_argument("--seq-len", type=int, default=512)
    sp.add_argument("--batch", type=int, default=1)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_inspect, steps=None, max_steps=None)

    sp = common(sub.add_parser("arch-report", help="inspect the loaded Qwen3.5 architecture"))
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--out", default=None, help="also write JSON here")
    sp.set_defaults(func=cmd_arch_report, steps=None, max_steps=None)

    sp = common(sub.add_parser("corrupt", help="show the forward noising process on real text"))
    sp.add_argument("--text", required=True)
    sp.add_argument("--t", type=float, required=True)
    sp.set_defaults(func=cmd_corrupt, steps=None, max_steps=None)

    sp = common(sub.add_parser("smoke-forward", help="one diffusion forward pass + loss"))
    sp.add_argument("--t", type=float, default=0.5)
    sp.set_defaults(func=cmd_smoke_forward, steps=None, max_steps=None)

    sp = common(sub.add_parser("smoke-grad", help="verify gradients reach LoRA and only LoRA"))
    sp.add_argument("--t", type=float, default=0.5)
    sp.set_defaults(func=cmd_smoke_grad, steps=None, max_steps=None)

    sp = common(sub.add_parser("probe-bidir", help="measure real information flow across the canvas"))
    sp.add_argument(
        "--text",
        default="The quick brown fox jumps over the lazy dog while the cat sat on the "
        "windowsill watching the rain fall on the quiet street below the old house.",
    )
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_probe_bidir, steps=None, max_steps=None)

    sp = common(sub.add_parser("train", help="run a training experiment"))
    sp.add_argument("--max-steps", type=int, default=None)
    sp.add_argument("--fixed-batch", action="store_true")
    sp.set_defaults(func=cmd_train, steps=None)

    sp = common(sub.add_parser("overfit-one-batch", help="the mandatory first experiment"))
    sp.add_argument("--max-steps", type=int, default=None)
    sp.add_argument("--t-max", type=float, default=0.5, help="upper noise level for the test")
    sp.set_defaults(func=cmd_overfit, steps=None)

    sp = common(sub.add_parser("reconstruct", help="reconstruction accuracy across noise levels"))
    sp.add_argument("--checkpoint", default=None)
    sp.add_argument("--noise", type=float, default=None, help="single level; default sweeps 0/0.1/0.5/0.9/1.0")
    sp.add_argument("--batch", type=int, default=2)
    sp.add_argument("--show-examples", action="store_true")
    sp.add_argument("--json-out", default=None)
    sp.set_defaults(func=cmd_reconstruct, steps=None, max_steps=None)

    sp = common(sub.add_parser("generate", help="block-diffusion generation"))
    sp.add_argument("--checkpoint", default=None)
    sp.add_argument("--prompt", required=True)
    sp.add_argument("--steps", type=int, default=None)
    sp.add_argument("--blocks", type=int, default=None)
    sp.add_argument("--show-steps", action="store_true")
    sp.set_defaults(func=cmd_generate, max_steps=None)

    sp = common(sub.add_parser("compare", help="AR baseline vs diffusion, same weights"))
    sp.add_argument("--checkpoint", default=None)
    sp.add_argument("--prompt", required=True)
    sp.add_argument("--steps", type=int, default=None)
    sp.add_argument("--max-new-tokens", type=int, default=32)
    sp.add_argument("--mlx", action="store_true", help="also run the mlx-lm AR baseline")
    sp.set_defaults(func=cmd_compare, max_steps=None)

    sp = common(sub.add_parser("smoke-bidir", help="v0.2 bidirectional-DeltaNet architecture smoke test"))
    sp.add_argument(
        "--text",
        default="The quick brown fox jumps over the lazy dog while the cat sat on the "
        "windowsill watching the rain fall on the quiet street below the old house.",
    )
    sp.add_argument("--probe-layer", type=int, default=None, help="DeltaNet layer to probe")
    sp.add_argument("--json-out", default=None)
    sp.set_defaults(func=cmd_smoke_bidir, steps=None, max_steps=None)

    sp = common(sub.add_parser("heldout", help="Phase 3: train one arm, track held-out denoising"))
    sp.add_argument("--arm", default=None, help="label for this arm (A/B/C/D)")
    sp.add_argument("--max-steps", type=int, default=None)
    sp.set_defaults(func=cmd_heldout, steps=None)

    sp = common(sub.add_parser("p3-report", help="compare Phase 3 arms against the frozen criteria"))
    sp.add_argument("--runs", nargs="+", required=True,
                    help="arm=path pairs, e.g. A=runs/p3-A-causal B=runs/p3-B-aligned")
    sp.add_argument("--json-out", default=None)
    sp.set_defaults(func=cmd_p3_report, steps=None, max_steps=None)

    sp = common(sub.add_parser("compare-fusion", help="untrained loss per fusion strategy + gate sweep"))
    sp.add_argument("--noise-levels", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 0.9])
    sp.add_argument("--gate-sweep-t", type=float, default=0.5)
    sp.add_argument("--batches", type=int, default=4)
    sp.add_argument("--json-out", default=None)
    sp.set_defaults(func=cmd_compare_fusion, steps=None, max_steps=None)

    sp = common(sub.add_parser("unsloth-report", help="verified Unsloth/MLX backend capabilities"))
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_unsloth_report, steps=None, max_steps=None)

    sp = common(sub.add_parser("memory", help="detailed memory breakdown"))
    sp.add_argument("--seq-len", type=int, default=512)
    sp.add_argument("--batch", type=int, default=1)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_memory, steps=None, max_steps=None)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except NotImplementedError as exc:
        _echo(f"\nnot implemented: {exc}")
        return 2
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _echo(f"\n{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
