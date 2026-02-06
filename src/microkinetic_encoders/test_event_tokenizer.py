import torch
from event_tokenizer import EventTokenizer
print("🔥 FILE EXECUTED")

def test_token_shapes_and_padding():
    T = 300
    K_max = 64

    tokenizer = EventTokenizer(K_max=K_max)

    # ----- synthetic inputs -----
    features = torch.randn(T, 512)
    energy = torch.rand(T)
    frame_conf = torch.rand(T)
    stream_conf = torch.rand(T, 3)

    segments = [(0, 50), (60, 140), (150, 280)]
    event_type_id = [1, 3, 2]
    fps = 30

    out = tokenizer(
        features=features,
        energy=energy,
        segments=segments,
        frame_conf=frame_conf,
        stream_conf=stream_conf,
        fps=fps,
        event_type_id=event_type_id
    )

    # ----- CONTRACT CHECKS -----
    assert out["tokens"].shape == (K_max, 256)
    assert out["attn_mask"].shape == (K_max,)
    assert out["token_conf"].shape == (K_max,)
    assert out["event_scalars"].shape == (K_max, 8)

    # real events = 3
    assert out["attn_mask"].sum().item() == 3

    # padding tokens must be zero
    pad_idx = out["attn_mask"] == 0
    assert torch.all(out["tokens"][pad_idx] == 0)
    assert torch.all(out["token_conf"][pad_idx] == 0)

    print("✅ Token interface contract PASSED")

if __name__ == "__main__":
    print("Main block entered")
    test_token_shapes_and_padding()

