"""Backend-neutral discrete-diffusion math.

Nothing in this package imports a Qwen model. Corruption, schedules, the training
objective and the sampler policy are written against plain tensors so they can be
unit-tested without a checkpoint and reused by a future MLX backend.
"""
