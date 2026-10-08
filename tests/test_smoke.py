import torch

from src.models.landmark_encoder import LightSignMoEProjector


def _streams(batch: int, frames: int) -> dict[str, torch.Tensor]:
    return {
        "left_hand": torch.randn(batch, frames, 21, 3),
        "right_hand": torch.randn(batch, frames, 21, 3),
        "face": torch.randn(batch, frames, 468, 3),
        "pose": torch.randn(batch, frames, 33, 3),
    }


def test_projector_output_shape_and_finite() -> None:
    model = LightSignMoEProjector(d_model=32, d_llm=64, window_size=4, stride=2)
    out = model(_streams(batch=2, frames=10))
    # 10 frames, ventana 4, stride 2 -> 4 ventanas
    assert out.shape == (2, 4, 64)
    assert torch.isfinite(out).all()
