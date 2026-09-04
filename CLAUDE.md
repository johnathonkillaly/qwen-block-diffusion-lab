# CLAUDE.md

**The instructions for this repository live in [AGENTS.md](AGENTS.md).** Read that
file first — it is the shared contract for both Claude Code and Codex, and it is kept
current deliberately so the project can be handed back and forth.

Do not duplicate guidance here. If something needs saying, say it in `AGENTS.md`.

Four things worth knowing before you touch anything:

0. **There are two different "Act IV"s.** `Act IV-N` is the structured-noise alphabet
   (`configs/act4/`, `mlx_backend/act4*.py`, `docs/ACT4_CRITERIA.md` — UPPERCASE),
   paused at "ready to run Stage 2". `Act IV-U` is the Uno diffusion-distillation
   track (`src/qdif/uno/`, `scripts/uno.py`, `docs/act4u*_*.md`), which is the active
   one. It has three phases: `U` (Uno reproduction), `U2` (transactional recurrent
   verification) and `U3` (acceptance scaling). They share no code with Act IV-N.
   Don't cross the streams.

1. **Use `.venv-unsloth` and `export HF_HOME=/Volumes/SHUTTLE`.** The other venv is a
   legacy torch path that is currently broken by a `transformers` version mismatch;
   its 19 test failures are pre-existing and not yours to fix unless asked.
2. **Pre-registered criteria are frozen.** `docs/ACT3_CRITERIA.md`,
   `docs/ACT4_CRITERIA.md`, `docs/act4u_preregistered_criteria.md`,
   `docs/act4u2_preregistered_criteria.md` and `docs/act4u3_preregistered_criteria.md`
   must not be edited after results exist. Amendments get appended with a date and a reason.
3. **This project is run to disprove itself.** Controls and clean comparisons matter
   more than a good-looking metric.
