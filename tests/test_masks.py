"""Canvas mask construction: causal prefix, bidirectional canvas, layer-type dict."""

from __future__ import annotations

import pytest
import torch

from qdif.models.masks import allow_to_additive, build_canvas_allow_matrix, build_diffusion_masks

T, P = 12, 4
SPANS = [(P, T)]


def test_prefix_rows_stay_causal():
    allow = build_canvas_allow_matrix(T, SPANS, bidirectional=True)
    for q in range(P):
        assert allow[q, : q + 1].all(), "a prefix query must see itself and everything before it"
        assert not allow[q, q + 1 :].any(), "a prefix query must never see the future"


def test_canvas_is_fully_connected_within_itself():
    allow = build_canvas_allow_matrix(T, SPANS, bidirectional=True)
    assert allow[P:, P:].all()


def test_canvas_still_sees_the_whole_prefix():
    allow = build_canvas_allow_matrix(T, SPANS, bidirectional=True)
    assert allow[P:, :P].all()


def test_canvas_never_leaks_into_the_prefix():
    """The property that makes block-autoregressive commitment sound."""
    allow = build_canvas_allow_matrix(T, SPANS, bidirectional=True)
    assert not allow[:P, P:].any()


def test_bidirectional_false_is_plain_causal():
    allow = build_canvas_allow_matrix(T, SPANS, bidirectional=False)
    assert torch.equal(allow, torch.tril(torch.ones(T, T, dtype=torch.bool)))


def test_bidirectional_strictly_adds_edges():
    causal = build_canvas_allow_matrix(T, SPANS, bidirectional=False)
    bidi = build_canvas_allow_matrix(T, SPANS, bidirectional=True)
    assert (bidi | causal).equal(bidi), "bidirectional must be a superset of causal"
    assert int(bidi.sum()) > int(causal.sum())


def test_multiple_canvas_blocks_do_not_merge():
    """Two committed blocks: each is internally bidirectional, but the earlier one
    must not be able to see the later one."""
    allow = build_canvas_allow_matrix(16, [(4, 8), (8, 12)], bidirectional=True)
    assert allow[4:8, 4:8].all()
    assert allow[8:12, 8:12].all()
    assert not allow[4:8, 8:12].any()
    assert allow[8:12, 4:8].all()


@pytest.mark.parametrize("span", [(-1, 4), (4, 40), (8, 4)])
def test_out_of_range_spans_rejected(span):
    with pytest.raises(ValueError, match="canvas span"):
        build_canvas_allow_matrix(T, [span])


def test_additive_mask_uses_zero_and_dtype_min():
    allow = build_canvas_allow_matrix(T, SPANS)
    add = allow_to_additive(allow, torch.float32, batch_size=2)
    assert add.shape == (2, 1, T, T)
    assert float(add[0, 0, 0, 0]) == 0.0
    assert float(add[0, 0, 0, 1]) == torch.finfo(torch.float32).min


def test_padding_mask_blocks_padded_keys_for_every_query():
    allow = build_canvas_allow_matrix(T, SPANS)
    padding = torch.ones(1, T, dtype=torch.bool)
    padding[0, -2:] = False
    add = allow_to_additive(allow, torch.float32, batch_size=1, padding_mask=padding)
    assert bool((add[0, 0, :, -2:] == torch.finfo(torch.float32).min).all())


def test_mask_dict_has_exactly_the_two_layer_type_keys():
    """These keys must match what Qwen3_5TextModel.forward indexes with."""
    masks = build_diffusion_masks(T, SPANS, torch.float32, batch_size=1)
    assert set(masks) == {"full_attention", "linear_attention"}
    assert masks["full_attention"].shape == (1, 1, T, T)
    # DeltaNet layers cannot take a bidirectional mask; with no padding there is
    # nothing to say to them at all.
    assert masks["linear_attention"] is None


def test_mask_dict_passes_padding_to_linear_layers():
    padding = torch.ones(2, T, dtype=torch.bool)
    masks = build_diffusion_masks(T, SPANS, torch.float32, batch_size=2, padding_mask=padding)
    assert masks["linear_attention"] is not None
    assert masks["linear_attention"].shape == (2, T)


def test_bf16_mask_uses_bf16_min():
    masks = build_diffusion_masks(T, SPANS, torch.bfloat16)
    assert masks["full_attention"].dtype == torch.bfloat16
    assert float(masks["full_attention"].min()) == float(torch.finfo(torch.bfloat16).min)
