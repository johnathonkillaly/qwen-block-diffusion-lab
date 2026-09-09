# CLAUDE.md

**The instructions for this repository live in [AGENTS.md](AGENTS.md).** Read that
file first — it is the shared contract for both Claude Code and Codex, and it is kept
current deliberately so the project can be handed back and forth.

Do not duplicate guidance here. If something needs saying, say it in `AGENTS.md`.

Three things worth knowing before you touch anything:

0. **`src/qdif/uno/` is supporting infrastructure, not its own reported result.** It is
   a from-source reproduction of IFM's public Uno (`github.com/ifm-ai/uno`,
   Apache-2.0), used to produce the adapter checkpoints that the **RPRM** experiment
   (`RPRM_DIFFUSION_*.md` at repo root, `scripts/rprm_stage1_*.py`) analyzes. RPRM's
   own verdict is **STOP** (frozen, complete) — do not reopen it, move its thresholds,
   or treat its diagnostic top-1 result as a pass.
1. **Use `.venv-unsloth` and `export HF_HOME=/Volumes/SHUTTLE`.** The other venv is a
   legacy torch path that is currently broken by a `transformers` version mismatch;
   its 19 test failures are pre-existing and not yours to fix unless asked.
2. **Pre-registered criteria are frozen.** `docs/ACT3_CRITERIA.md` and
   `RPRM_DIFFUSION_PREREG.md` must not be edited after results exist. Amendments get
   appended with a date and a reason.
3. **This project is run to disprove itself.** Controls and clean comparisons matter
   more than a good-looking metric.
