"""v0.2 MLX / Unsloth backend.

On Apple silicon, Unsloth *is* an MLX stack: `unsloth.device_type` reports
`DEVICE_TYPE == "mlx"` and `unsloth_zoo` declares `mlx`, `mlx-lm` and `mlx-vlm` as
its darwin-arm64 dependencies while excluding torch, peft, trl and accelerate.
So "use Unsloth as the training engine on this machine" means "use the MLX path".

See docs/UNSLOTH_BACKEND.md for what that does and does not buy us, verified rather
than assumed.
"""
