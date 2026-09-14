#!/usr/bin/env python
"""Act IV-U6 gate U6-1: does a wider draft canvas change the first four slots?

No wall clock. For every context, the frozen drafter drafts at D = 4, 6, 8 and 16, and
slots 1–4 are compared under three noise conditions:

* **decoder** — noise drawn exactly as the decoder draws it (shape `(1, D-1)` from the
  cycle key), so each width sees a different realisation;
* **aligned** — one width-16 draw, prefix-shared, so slots 1–4 see identical noise at
  every width and only the canvas width differs;
* **redraw** — D=4 with an independent key: the control for "different noise".

The draft pass is causal, so the design predicts that slots 1–4 depend on width only
through the noise draw and bf16 reduction order. The gate decides with the recorded
`prefix₄` statistic; the aligned condition shows the mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

WIDTHS = (4, 6, 8, 16)
SLOTS = 4


def adapter_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_adapter(adapter: str) -> str:
    manifest = json.loads((REPO / "results/checkpoint_manifest.json").read_text())
    expected = next(a["adapter_sha256"] for a in manifest["adapters"] if a["id"] == "act4s_u6_drafter")
    actual = adapter_sha256(REPO / adapter)
    if actual != expected:
        raise SystemExit(f"adapter {adapter} sha256 {actual} != manifest {expected}")
    return actual


def leading_matches(proposal: list[int], target: list[int]) -> int:
    """Accepted slots among p₂…p₄ against the target's greedy continuation."""
    n = 0
    for j in range(1, SLOTS):
        if proposal[j] != target[j]:
            break
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter", default="runs/u4b1/step-16000/adapter.safetensors")
    ap.add_argument("--windows", type=int, default=229)
    ap.add_argument("--context", type=int, default=64)
    ap.add_argument("--out", default="results/act4u6/diagnostic_wide_draft.json")
    args = ap.parse_args()

    import mlx.core as mx

    from qdif.uno.cache_utils import restore_caches, snapshot_caches
    from qdif.uno.data import load_wikitext, pack_windows
    from qdif.uno.decode import ar_greedy_generate
    from qdif.uno.noise import build_draft_block, draft_lora_mask
    from qdif.uno.rng import BENCH_NOISE, stream_key
    from spec_decode import NOISE_STREAM_SEED, load, provenance, suite_prompts
    from uno import _paired_bootstrap

    sha = check_adapter(args.adapter)
    model = load(args.adapter)
    digest_before = model.fingerprint(full=True).digest
    tok = model.tokenizer

    contexts = [(cat, ids) for cat, _text, ids in suite_prompts(tok)]
    windows = pack_windows(load_wikitext("validation"), tok, width=args.context + 8,
                           max_windows=args.windows)
    contexts += [("wikitext", [int(t) for t in w[: args.context]]) for w in windows]
    print(f"[u6-diag] {len(contexts)} contexts ({len(contexts) - len(windows)} suite + "
          f"{len(windows)} WikiText)", flush=True)

    conditions = [(4, "decoder"), (4, "aligned"), (4, "redraw")] + [
        (d, c) for d in WIDTHS[1:] for c in ("decoder", "aligned")]
    per = {f"D{d}_{c}": {"prefix4": [], "slot_match_target": [[] for _ in range(SLOTS)],
                         "identical_to_d4": [[] for _ in range(SLOTS)],
                         "max_abs_logit_diff": [[] for _ in range(SLOTS)],
                         "top1": [[] for _ in range(SLOTS)], "p1_equals_target": []}
           for d, c in conditions}
    started = time.perf_counter()

    for idx, (category, ids) in enumerate(contexts):
        target, _ = ar_greedy_generate(model, ids, max_tokens=SLOTS + 1)
        caches = model.make_cache()
        if len(ids) > 1:
            mx.eval(model.ar_logits(mx.array([ids[:-1]], dtype=mx.int32), cache=caches))
        snap = snapshot_caches(caches)
        seed = mx.array([ids[-1]], dtype=mx.int32)
        wide = build_draft_block(seed, 16, "random_uniform", model.mask_token_id,
                                 model.vocab_size, key=stream_key(NOISE_STREAM_SEED, BENCH_NOISE, 100_000 + idx))
        references = {}
        for d, cond in conditions:
            if cond == "decoder":
                block = build_draft_block(seed, d, "random_uniform", model.mask_token_id,
                                          model.vocab_size, key=stream_key(NOISE_STREAM_SEED, BENCH_NOISE, idx))
            elif cond == "aligned":
                block = wide[:, :d]
            else:
                block = build_draft_block(seed, d, "random_uniform", model.mask_token_id,
                                          model.vocab_size, key=stream_key(NOISE_STREAM_SEED + 1, BENCH_NOISE, idx))
            restore_caches(caches, snap)
            logits = model.draft_logits(block, draft_lora_mask(1, d), cache=caches)
            rows = logits[0, :SLOTS].astype(mx.float32)
            probs = mx.softmax(rows, axis=-1)
            mx.eval(rows, probs)
            arg = [int(t) for t in mx.argmax(rows, axis=-1).tolist()]
            top1 = [float(v) for v in mx.max(probs, axis=-1).tolist()]
            rec = per[f"D{d}_{cond}"]
            rec["prefix4"].append(leading_matches(arg, target))
            rec["p1_equals_target"].append(arg[0] == target[0])
            for j in range(SLOTS):
                rec["slot_match_target"][j].append(arg[j] == target[j])
                rec["top1"][j].append(top1[j])
            ref_key = "aligned" if cond == "aligned" else "decoder"
            if d == 4 and cond in ("decoder", "aligned"):
                references[ref_key] = (arg, rows)
            else:
                ref_arg, ref_rows = references[ref_key]
                diff = mx.max(mx.abs(rows - ref_rows), axis=-1)
                mx.eval(diff)
                for j in range(SLOTS):
                    rec["identical_to_d4"][j].append(arg[j] == ref_arg[j])
                    rec["max_abs_logit_diff"][j].append(float(diff[j].item()))
        if (idx + 1) % 32 == 0:
            rate = (time.perf_counter() - started) / (idx + 1)
            print(f"  [{idx + 1}/{len(contexts)}] {rate:.2f}s/context", flush=True)

    digest_after = model.fingerprint(full=True).digest
    base = per["D4_decoder"]["prefix4"]

    def mean(xs):
        return statistics.fmean(xs) if xs else None

    summary = {}
    for name, rec in per.items():
        summary[name] = {
            "mean_prefix4": mean(rec["prefix4"]),
            "slot_agreement_with_target": [mean(x) for x in rec["slot_match_target"]],
            "p1_equals_target": mean(rec["p1_equals_target"]),
            "mean_top1": [mean(x) for x in rec["top1"]],
            "identical_to_d4_same_condition": [mean(x) for x in rec["identical_to_d4"]],
            "mean_max_abs_logit_diff_vs_d4": [mean(x) for x in rec["max_abs_logit_diff"]],
        }

    delta = {}
    for d in WIDTHS[1:]:
        for cond in ("decoder", "aligned"):
            ref = per[f"D4_{cond}"]["prefix4"]
            b = _paired_bootstrap(list(zip(ref, per[f"D{d}_{cond}"]["prefix4"])))
            delta[f"D{d}_{cond}"] = b
    delta["redraw_D4"] = _paired_bootstrap(list(zip(base, per["D4_redraw"]["prefix4"])))

    redraw_mag = abs(delta["redraw_D4"]["mean_diff"])
    alive = [d for d in WIDTHS[1:]
             if delta[f"D{d}_decoder"]["ci_low"] > 0 and delta[f"D{d}_decoder"]["mean_diff"] > redraw_mag]
    gate = {
        "rule": ("Variant 1 stays alive only if some D in {6,8,16} has a paired 95% CI on "
                 "prefix4(D)-prefix4(4) (decoder noise) with lower bound > 0 and a mean "
                 "greater than |mean redraw difference|."),
        "widths_passing": alive,
        "variant1_alive": bool(alive),
        "verdict": "VARIANT 1 ALIVE" if alive else "VARIANT 1 KILLED",
    }

    payload = {
        "gate": "U6-1", "provenance": provenance(), "adapter": args.adapter,
        "adapter_sha256": sha, "backbone_digest_before": digest_before,
        "backbone_digest_after": digest_after, "backbone_unchanged": digest_before == digest_after,
        "contexts": len(contexts), "context_tokens": args.context, "widths": list(WIDTHS),
        "noise_stream_seed": NOISE_STREAM_SEED, "summary": summary,
        "prefix4_delta_vs_d4": delta, "gate_result": gate,
        "seconds": round(time.perf_counter() - started, 1),
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    print("\n-- mean prefix4 (accepted slots among p2..p4 vs target greedy) --")
    for name in ("D4_decoder", "D4_redraw", "D6_decoder", "D8_decoder", "D16_decoder",
                 "D4_aligned", "D6_aligned", "D8_aligned", "D16_aligned"):
        s = summary[name]
        ident = s["identical_to_d4_same_condition"]
        print(f"  {name:12s} prefix4 {s['mean_prefix4']:.4f}  slot-identical-to-D4 "
              f"{['-' if v is None else f'{v:.3f}' for v in ident]}  max|dlogit| "
              f"{['-' if v is None else f'{v:.3f}' for v in s['mean_max_abs_logit_diff_vs_d4']]}")
    for k, b in delta.items():
        print(f"  delta {k:12s} {b['mean_diff']:+.4f} [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]")
    print(f"\n[u6-diag] {gate['verdict']}  (backbone unchanged: {payload['backbone_unchanged']})")
    print(f"[u6-diag] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
