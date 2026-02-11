import torch
import pytest

# -----------------------------
# CHANGE THIS IMPORT
# -----------------------------
# Example:
# from src.reasoning.video_transformer import MyVideoTransformerModel

from src.models.video.transformer_reasoning.event_transformer import VideoTransformer as MyVideoTransformerModel


def make_fake_bundle(B=2, K=64, d_model=256, S=8, num_event_types=32, device="cpu"):
    """
    Creates a fake encoder output bundle consistent with your pipeline.
    """

    tokens = torch.randn(B, K, d_model, device=device)
    scalars = torch.randn(B, K, S, device=device)
    token_conf = torch.rand(B, K, device=device)
    event_type_id = torch.randint(0, num_event_types, (B, K), device=device)

    # Random valid mask (at least 1 valid token per sample)
    attn_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
    for b in range(B):
        valid_count = torch.randint(low=1, high=max(2, K // 4), size=(1,)).item()
        idx = torch.randperm(K, device=device)[:valid_count]
        attn_mask[b, idx] = True

    # Zero out padded regions
    tokens[~attn_mask] = 0.0
    scalars[~attn_mask] = 0.0
    token_conf[~attn_mask] = 0.0
    event_type_id[~attn_mask] = 0

    return {
        "tokens": tokens,                # [B, K, 256]
        "attn_mask": attn_mask,          # [B, K]
        "event_type_id": event_type_id,  # [B, K]
        "token_conf": token_conf,        # [B, K]
        "event_scalars": scalars,        # [B, K, S]
    }


# =========================================================
# 1) INIT TEST (matches your init function)
# =========================================================
def test_model_init_config():
    model = MyVideoTransformerModel(
        T=256,
        Heads=8,
        num_encode=3,
        num_decode=0,
        dim_ff=2048,
        dropout=0.5,
        active="gelu",
        batch_first=True,
        norm_first=True,
        bias=True
    )

    assert model is not None
    assert hasattr(model, "transformer"), "Expected model.transformer"

    tr = model.transformer
    assert isinstance(tr, torch.nn.Transformer)

    # Encoder/decoder layer counts
    assert len(tr.encoder.layers) == 3
    assert len(tr.decoder.layers) == 0

    # Validate first encoder layer config
    layer0 = tr.encoder.layers[0]
    assert layer0.self_attn.embed_dim == 256
    assert layer0.self_attn.num_heads == 8

    assert layer0.linear1.out_features == 2048
    assert layer0.linear2.in_features == 2048

    assert pytest.approx(layer0.dropout.p, rel=1e-6) == 0.5
    assert layer0.norm_first is True

    # activation check
    assert layer0.activation.__name__ == "gelu"


# =========================================================
# 2) FORWARD PASS SHAPE TEST
# =========================================================
def test_forward_shapes():
    model = MyVideoTransformerModel(
        T=256,
        Heads=8,
        num_encode=3,
        num_decode=0,
        dim_ff=2048,
        dropout=0.5,
        active="gelu",
        batch_first=True,
        norm_first=True,
        bias=True
    )

    bundle = make_fake_bundle(B=4, K=64, d_model=256, S=8)

    out = model(bundle)

    assert isinstance(out, dict)

    # Check keys
    assert "z" in out
    assert "logit" in out
    assert "prob" in out
    assert "confidence score" in out

    # Check shapes
    assert out["z"].shape == (4, 256)
    assert out["logit"].shape == (4,)
    assert out["prob"].shape == (4,)
    assert out["confidence score"].shape == (4,)

    # Check ranges
    assert torch.all(out["prob"] >= 0.0)
    assert torch.all(out["prob"] <= 1.0)

    assert torch.all(out["confidence score"] >= 0.0)
    assert torch.all(out["confidence score"] <= 1.0)


# =========================================================
# 3) PADDING INVARIANCE TEST (mask correctness)
# =========================================================
def test_padding_invariance():
    """
    If attention masking is correct, changing padded tokens
    must not change output.
    """

    torch.manual_seed(0)

    # Dropout must be 0 for deterministic test
    model = MyVideoTransformerModel(
        T=256,
        Heads=8,
        num_encode=3,
        num_decode=0,
        dim_ff=2048,
        dropout=0.0,
        active="gelu",
        batch_first=True,
        norm_first=True,
        bias=True
    )

    bundle = make_fake_bundle(B=2, K=64, d_model=256, S=8)

    out1 = model(bundle)

    # Copy and corrupt padded tokens heavily
    bundle2 = {k: v.clone() for k, v in bundle.items()}
    mask = bundle2["attn_mask"]

    bundle2["tokens"][~mask] = torch.randn_like(bundle2["tokens"][~mask]) * 999.0
    bundle2["event_scalars"][~mask] = torch.randn_like(bundle2["event_scalars"][~mask]) * 999.0
    bundle2["token_conf"][~mask] = torch.rand_like(bundle2["token_conf"][~mask])
    bundle2["event_type_id"][~mask] = torch.randint(0, 32, bundle2["event_type_id"][~mask].shape)

    out2 = model(bundle2)

    # Must match if masking is correct
    assert torch.allclose(out1["prob"], out2["prob"], atol=1e-6)
    assert torch.allclose(out1["z"], out2["z"], atol=1e-6)


# =========================================================
# 4) GRADIENT FLOW TEST
# =========================================================
def test_backward_pass():
    model = MyVideoTransformerModel(
        T=256,
        Heads=8,
        num_encode=3,
        num_decode=0,
        dim_ff=2048,
        dropout=0.5,
        active="gelu",
        batch_first=True,
        norm_first=True,
        bias=True
    )

    bundle = make_fake_bundle(B=2, K=64, d_model=256, S=8)
    out = model(bundle)

    y = torch.tensor([0.0, 1.0])

    loss = torch.nn.functional.binary_cross_entropy(out["prob"], y)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(torch.isfinite(g).all() for g in grads)


# =========================================================
# 5) EDGE CASE: ALL TOKENS MASKED
# =========================================================
def test_all_masked_tokens():
    """
    If your model uses CLS token internally, this should still work.
    If not, you must handle the empty-event case explicitly.
    """

    model = MyVideoTransformerModel(
        T=256,
        Heads=8,
        num_encode=3,
        num_decode=0,
        dim_ff=2048,
        dropout=0.0,
        active="gelu",
        batch_first=True,
        norm_first=True,
        bias=True
    )

    B, K = 2, 64
    bundle = make_fake_bundle(B=B, K=K, d_model=256, S=8)

    bundle["attn_mask"][:] = False
    bundle["tokens"][:] = 0.0
    bundle["event_scalars"][:] = 0.0
    bundle["token_conf"][:] = 0.0
    bundle["event_type_id"][:] = 0

    out = model(bundle)

    assert out["prob"].shape == (B,)
    assert torch.all(out["prob"] >= 0.0)
    assert torch.all(out["prob"] <= 1.0)
