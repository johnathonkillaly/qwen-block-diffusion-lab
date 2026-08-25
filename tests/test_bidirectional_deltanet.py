"""v0.2: bidirectional Gated DeltaNet.

Split into two groups:

  * `mlx` + `model`: need MLX and a real Qwen3.5 checkpoint. These carry the claims
    that matter -- weight sharing, the leakage boundary, exactness of the causal
    fallback.
  * pure fusion-math tests: run anywhere MLX is installed, no checkpoint.

Run with:  .venv-unsloth/bin/python -m pytest tests/test_bidirectional_deltanet.py
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402

from qdif.mlx_backend.deltanet import (  # noqa: E402
    BidirectionalGatedDeltaNet,
    ConcatProjFusion,
    MeanFusion,
    ScalarGateFusion,
    ScaledForwardControl,
    ShuffledReverseControl,
    TokenGateFusion,
    build_fusion,
    deltanet_modules,
    set_bidirectional,
    set_canvas_start,
)

pytestmark = pytest.mark.mlx

B, C, HV, DV = 2, 6, 4, 8


def _pair(seed=0):
    mx.random.seed(seed)
    return mx.random.normal((B, C, HV, DV)), mx.random.normal((B, C, HV, DV))


# ---------------------------------------------------------------- fusion algebra


def test_mean_fusion_is_the_average():
    h_f, h_r = _pair()
    out = MeanFusion()(h_f, h_r)
    assert mx.allclose(out, (h_f + h_r) / 2, atol=1e-6)


def test_mean_fusion_has_no_parameters():
    assert MeanFusion().num_parameters == 0


def test_scalar_gate_starts_near_the_forward_path():
    h_f, h_r = _pair()
    fusion = ScalarGateFusion(gate_init=0.95)
    assert float(fusion.gate()) == pytest.approx(0.95, abs=1e-4)
    out = fusion(h_f, h_r)
    expected = 0.95 * h_f + 0.05 * h_r
    assert mx.allclose(out, expected, atol=1e-3)


def test_scalar_gate_at_one_is_exactly_the_forward_path():
    h_f, h_r = _pair()
    out = ScalarGateFusion(gate_init=1.0 - 1e-4)(h_f, h_r)
    assert mx.allclose(out, h_f, atol=1e-3)


def test_scalar_gate_has_exactly_one_parameter():
    assert ScalarGateFusion().num_parameters == 1


def test_token_gate_starts_uniform_at_gate_init():
    h_f, h_r = _pair()
    fusion = TokenGateFusion(HV, gate_init=0.9)
    out = fusion(h_f, h_r)
    assert mx.allclose(out, 0.9 * h_f + 0.1 * h_r, atol=1e-3)


def test_token_gate_stays_small():
    """Spec says keep it small: 2*Hv -> Hv, not 2*Hv*Dv -> Hv*Dv."""
    fusion = TokenGateFusion(num_v_heads=32)
    assert fusion.num_parameters == 2 * 32 * 32 + 32  # w_g + b_g
    assert fusion.num_parameters < 3000


def test_token_gate_becomes_position_dependent_once_trained():
    h_f, h_r = _pair()
    fusion = TokenGateFusion(HV, gate_init=0.9)
    fusion.w_g = mx.random.normal(fusion.w_g.shape) * 2.0
    out = fusion(h_f, h_r)
    naive = 0.9 * h_f + 0.1 * h_r
    assert not bool(mx.allclose(out, naive, atol=1e-3))


def test_concat_proj_starts_as_an_identity_on_the_forward_path():
    h_f, h_r = _pair()
    out = ConcatProjFusion(DV)(h_f, h_r)
    assert mx.allclose(out, h_f, atol=1e-5)


def test_build_fusion_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="unknown fusion"):
        build_fusion("bogus", HV, DV, 0.9)


# -------------------------------------------------------------------- controls


def test_forward_scaled_control_ignores_the_reverse_direction():
    h_f, h_r = _pair()
    out = ScaledForwardControl(0.5)(h_f, h_r)
    assert mx.allclose(out, 0.5 * h_f, atol=1e-5)
    other = ScaledForwardControl(0.5)(h_f, mx.zeros_like(h_r))
    assert mx.allclose(out, other, atol=1e-6)


def test_shuffled_reverse_control_preserves_magnitude_but_not_order():
    h_f, h_r = _pair()
    plain = ScalarGateFusion(0.5)(h_f, h_r)
    shuffled = ShuffledReverseControl(0.5)(h_f, h_r)
    assert not bool(mx.allclose(plain, shuffled, atol=1e-4))
    # Same set of reverse vectors, so the summed magnitude is comparable.
    a = float(mx.abs(plain.astype(mx.float32)).sum())
    b = float(mx.abs(shuffled.astype(mx.float32)).sum())
    assert 0.5 < a / b < 2.0


# ------------------------------------------------------- wrapper, no checkpoint


class _FakeGDN(nn.Module):
    """Minimal stand-in with the attribute surface the wrapper touches."""

    def __init__(self, hidden=16, num_v_heads=HV, head_v_dim=DV):
        super().__init__()
        self.hidden_size = hidden
        self.num_v_heads = num_v_heads
        self.num_k_heads = num_v_heads
        self.head_k_dim = head_v_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_v_heads * head_v_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = 4
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=4,
                                groups=self.conv_dim, bias=False, padding=0)
        self.in_proj_qkv = nn.Linear(hidden, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden, num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden, num_v_heads, bias=False)
        self.dt_bias = mx.ones(num_v_heads)
        self.A_log = mx.log(mx.random.uniform(low=0.1, high=16, shape=(num_v_heads,)))
        self.norm = nn.RMSNorm(head_v_dim)
        self.out_proj = nn.Linear(self.value_dim, hidden, bias=False)

    def __call__(self, x, mask=None, cache=None):  # not used by these tests
        raise NotImplementedError


def test_wrapper_holds_the_same_module_object():
    base = _FakeGDN()
    w = BidirectionalGatedDeltaNet(base, MeanFusion())
    assert w.base is base
    assert id(w.base.in_proj_qkv.weight) == id(base.in_proj_qkv.weight)
    assert id(w.base.out_proj.weight) == id(base.out_proj.weight)


def test_wrapper_refuses_incremental_cache():
    w = BidirectionalGatedDeltaNet(_FakeGDN(), MeanFusion())
    with pytest.raises(NotImplementedError, match="stale"):
        w(mx.zeros((1, 4, 16)), mask=None, cache=object())


def test_canvas_start_defaults_to_disabled():
    w = BidirectionalGatedDeltaNet(_FakeGDN(), MeanFusion())
    assert w._canvas_start == -1


# ----------------------------------------------------- real checkpoint required


@pytest.fixture(scope="module")
def mlx_setup():
    from qdif.config import load_config

    cfg_path = "configs/v02_B_mean_fusion.yaml"
    try:
        cfg = load_config(cfg_path)
    except FileNotFoundError:
        pytest.skip(f"{cfg_path} missing")
    cfg.diffusion.canvas_length = 16
    cfg.data.prefix_length = 8
    from qdif.mlx_backend.build import build_mlx_setup

    try:
        return build_mlx_setup(cfg, echo=lambda *a: None)
    except FileNotFoundError as exc:
        pytest.skip(f"no local Qwen3.5-4B checkpoint: {exc}")


model_test = pytest.mark.model


@model_test
def test_forward_path_matches_upstream(mlx_setup):
    """Our decomposition must reproduce mlx-lm's GatedDeltaNet bit-for-bit.

    This is the tripwire for upstream drift: if mlx-lm changes the recurrence, this
    fails loudly instead of silently changing the experiment.
    """
    from qdif.mlx_backend.deltanet import deltanet_front_end, deltanet_recurrence

    model = mlx_setup.model
    model.train()
    idx, mod = deltanet_modules(model.base_model)[0]
    base = mod.base
    x = mx.random.normal((1, 12, mlx_setup.load_report.hidden_size)).astype(mx.bfloat16)

    q, k, v, a, b, z = deltanet_front_end(base, x, None)
    h = deltanet_recurrence(base, q, k, v, a, b, None, use_kernel=False)
    mine = base.out_proj(base.norm(h, z).reshape(1, 12, -1))

    # `base` is now wrapped, but the wrapper with bidirectional off IS the upstream
    # computation, so compare against it directly.
    mod.bidirectional = False
    mod.set_canvas(-1)
    ref = mod(x)
    mx.eval(mine, ref)
    assert bool(mx.array_equal(mine, ref))


@model_test
def test_wrapper_disabled_is_exact(mlx_setup):
    """Ablation A must be the v0.1 computation exactly, not approximately."""
    model = mlx_setup.model
    model.train()
    _, mod = deltanet_modules(model.base_model)[0]
    x = mx.random.normal((1, 12, mlx_setup.load_report.hidden_size)).astype(mx.bfloat16)

    mod.set_canvas(6)
    mod.bidirectional = False
    a = mod(x)
    mod.bidirectional = True
    b = mod(x)
    mx.eval(a, b)
    assert not bool(mx.array_equal(a, b)), "bidirectional must change the canvas"

    mod.bidirectional = False
    a2 = mod(x)
    mx.eval(a2)
    assert bool(mx.array_equal(a, a2)), "toggling back must restore exactly"


@model_test
def test_prefix_is_never_touched_by_the_reverse_pass(mlx_setup):
    """THE leakage boundary: canvas information must not reach committed history."""
    model = mlx_setup.model
    model.train()
    _, mod = deltanet_modules(model.base_model)[0]
    P = 8
    x = mx.random.normal((1, 24, mlx_setup.load_report.hidden_size)).astype(mx.bfloat16)
    mod.set_canvas(P)

    mod.bidirectional = False
    causal = mod(x)
    mod.bidirectional = True
    bidi = mod(x)
    mx.eval(causal, bidi)

    assert bool(mx.array_equal(causal[:, :P], bidi[:, :P])), "prefix changed -- LEAK"
    assert not bool(mx.array_equal(causal[:, P:], bidi[:, P:]))


@model_test
def test_canvas_edit_cannot_change_prefix_representation(mlx_setup):
    """Stronger form: editing a canvas token must leave every prefix position fixed."""
    model = mlx_setup.model
    model.train()
    _, mod = deltanet_modules(model.base_model)[0]
    P = 8
    h = mlx_setup.load_report.hidden_size
    x = mx.random.normal((1, 24, h)).astype(mx.bfloat16)
    mod.set_canvas(P)
    mod.bidirectional = True

    before = mod(x)
    edited = np.asarray(x.astype(mx.float32)).copy()
    edited[:, -1, :] += 3.0
    after = mod(mx.array(edited).astype(mx.bfloat16))
    mx.eval(before, after)

    assert bool(mx.array_equal(before[:, :P], after[:, :P]))
    assert not bool(mx.array_equal(before[:, P:], after[:, P:]))


@model_test
def test_reverse_recurrence_carries_information_backwards(mlx_setup):
    """The point of v0.2: a LATER canvas token must influence an EARLIER one."""
    model = mlx_setup.model
    model.train()
    _, mod = deltanet_modules(model.base_model)[0]
    P = 8
    h = mlx_setup.load_report.hidden_size
    x = mx.random.normal((1, 24, h)).astype(mx.bfloat16)
    edited = np.asarray(x.astype(mx.float32)).copy()
    edited[:, -1, :] += 3.0
    x2 = mx.array(edited).astype(mx.bfloat16)
    mod.set_canvas(P)

    mod.bidirectional = False
    ca, cb = mod(x), mod(x2)
    mod.bidirectional = True
    ba, bb = mod(x), mod(x2)
    mx.eval(ca, cb, ba, bb)

    # earlier canvas positions, excluding the perturbed final one
    causal_delta = float(mx.abs((ca[:, P:-1] - cb[:, P:-1]).astype(mx.float32)).max())
    bidi_delta = float(mx.abs((ba[:, P:-1] - bb[:, P:-1]).astype(mx.float32)).max())
    assert causal_delta == 0.0, "causal DeltaNet leaked information backwards"
    assert bidi_delta > 0.0, "bidirectional DeltaNet carried nothing backwards"


@model_test
def test_weights_are_shared_not_duplicated(mlx_setup):
    """A 4B backbone must not become an 8B backbone."""
    from mlx.utils import tree_flatten

    model = mlx_setup.model
    mods = deltanet_modules(model.base_model)
    assert len(mods) == len(mlx_setup.load_report.linear_attention_layers)

    recorded = mlx_setup.bidi_report["shared_weight_ids"]
    for i, mod in mods:
        assert id(mod.base.in_proj_qkv.weight) == recorded[i]

    total = sum(v.size for _, v in tree_flatten(model.parameters()))
    base = mlx_setup.load_report.total_params
    extra = total - base
    lora = mlx_setup.lora_report.lora_params if mlx_setup.lora_report else 0
    fusion = mlx_setup.bidi_report["fusion_params"]
    cond = mlx_setup.params["trainable_by_family"].get("timestep_conditioner", 0)
    assert extra <= lora + fusion + cond + 8, (
        f"parameter count grew by {extra:,}, more than lora+fusion+conditioner "
        f"({lora + fusion + cond:,}) -- weights were duplicated"
    )


@model_test
def test_base_weights_are_frozen(mlx_setup):
    from mlx.utils import tree_flatten

    trainable = dict(tree_flatten(mlx_setup.model.trainable_parameters()))
    for path in trainable:
        assert (
            path.endswith(("lora_a", "lora_b"))
            or ".fusion." in path
            or path.startswith("timestep_conditioner")
        ), f"unexpected trainable parameter: {path}"


@model_test
def test_set_bidirectional_toggles_every_layer(mlx_setup):
    model = mlx_setup.model
    n = set_bidirectional(model.base_model, False)
    assert n == len(mlx_setup.load_report.linear_attention_layers)
    assert all(not m.bidirectional for _, m in deltanet_modules(model.base_model))
    set_bidirectional(model.base_model, True)
    assert all(m.bidirectional for _, m in deltanet_modules(model.base_model))


@model_test
def test_set_canvas_start_reaches_every_layer(mlx_setup):
    n = set_canvas_start(mlx_setup.model.base_model, 5)
    assert n == len(mlx_setup.load_report.linear_attention_layers)
    assert all(m._canvas_start == 5 for _, m in deltanet_modules(mlx_setup.model.base_model))


# ------------------------------------------------- FLARE block-end state readout


def test_block_end_readout_shape_and_head_replication():
    """q has Hk heads, the state has Hv; the readout must replicate q like the
    recurrence does (Qwen3.5-4B has Hv/Hk = 2)."""
    from qdif.mlx_backend.deltanet import block_end_readout

    Bn, Cn, Hk, Hv, Dk, Dv = 2, 5, 4, 8, 16, 16
    state = mx.random.normal((Bn, Hv, Dv, Dk))
    q = mx.random.normal((Bn, Cn, Hk, Dk))
    out = block_end_readout(None, state, q)
    assert out.shape == (Bn, Cn, Hv, Dv)


def test_block_end_readout_matches_the_reference_contraction():
    """Must equal the same contraction the per-step readout uses, with the state
    held fixed: y_i = (S * q_i[..., None, :]).sum(-1)."""
    from qdif.mlx_backend.deltanet import block_end_readout

    Bn, Cn, H, Dk, Dv = 1, 4, 3, 6, 6
    state = mx.random.normal((Bn, H, Dv, Dk))
    q = mx.random.normal((Bn, Cn, H, Dk))
    got = block_end_readout(None, state, q)
    for i in range(Cn):
        ref = (state.astype(mx.float32) * q[:, i][..., None, :].astype(mx.float32)).sum(axis=-1)
        assert mx.allclose(got[:, i].astype(mx.float32), ref, atol=1e-4)


def test_block_end_readout_is_position_invariant_given_equal_queries():
    """FLARE's state is shared across the block, so two positions with identical
    queries must produce identical outputs. This is the structural difference from
    our per-position reverse summary."""
    from qdif.mlx_backend.deltanet import block_end_readout

    state = mx.random.normal((1, 2, 4, 4))
    q = mx.random.normal((1, 1, 2, 4))
    q2 = mx.concatenate([q, q], axis=1)
    out = block_end_readout(None, state, q2)
    assert mx.allclose(out[:, 0], out[:, 1], atol=1e-6)


@model_test
def test_flare_mode_differs_from_causal_and_from_fusion(mlx_setup):
    from qdif.mlx_backend.deltanet import set_mode

    model = mlx_setup.model
    model.train()
    _, mod = deltanet_modules(model.base_model)[0]
    P = 8
    x = mx.random.normal((1, 24, mlx_setup.load_report.hidden_size)).astype(mx.bfloat16)
    mod.set_canvas(P)

    mod.bidirectional = False
    causal = mod(x)
    mod.bidirectional = True
    mod.mode = "fusion"
    fusion = mod(x)
    mod.mode = "flare"
    flare = mod(x)
    mx.eval(causal, fusion, flare)

    assert not bool(mx.array_equal(flare, causal)), "flare readout must change the canvas"
    assert not bool(mx.array_equal(flare, fusion)), "flare must differ from dual-recurrence fusion"
    # And it must respect the same leakage boundary.
    assert bool(mx.array_equal(flare[:, :P], causal[:, :P])), "flare leaked into the prefix"
    mod.mode = "fusion"


@model_test
def test_flare_mode_gives_every_canvas_position_block_wide_visibility(mlx_setup):
    """Perturbing the LAST canvas token must move EARLIER canvas positions, because
    they all read the block-end state."""
    model = mlx_setup.model
    model.train()
    _, mod = deltanet_modules(model.base_model)[0]
    P = 8
    h = mlx_setup.load_report.hidden_size
    x = mx.random.normal((1, 24, h)).astype(mx.bfloat16)
    edited = np.asarray(x.astype(mx.float32)).copy()
    edited[:, -1, :] += 3.0
    x2 = mx.array(edited).astype(mx.bfloat16)

    mod.set_canvas(P)
    mod.bidirectional = True
    mod.mode = "flare"
    a, b = mod(x), mod(x2)
    mx.eval(a, b)
    mod.mode = "fusion"

    earlier = float(mx.abs((a[:, P:-1] - b[:, P:-1]).astype(mx.float32)).max())
    prefix = float(mx.abs((a[:, :P] - b[:, :P]).astype(mx.float32)).max())
    assert earlier > 0.0, "block-end readout carried nothing backwards"
    assert prefix == 0.0, "block-end readout leaked into the prefix"
