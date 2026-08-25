"""Datasets and collation.

Datasets yield *clean* text only. Corruption is applied dynamically at batch time
by the trainer, so every epoch sees a fresh timestep and a fresh noise pattern for
the same example. Nothing here ever writes a pre-corrupted corpus to disk.
"""
