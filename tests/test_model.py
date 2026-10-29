import torch

from ffr.losses import sdf_loss
from ffr.model import UNet3D, count_parameters


def test_unet_shapes_and_size():
    m = UNet3D(in_channels=4, base=8, depth=4)
    x = torch.randn(1, 4, 32, 48, 32)
    y = m(x)
    assert y.shape == (1, 1, 32, 48, 32)
    assert y.abs().max() <= 1
    assert count_parameters(UNet3D(base=16, depth=4)) < 6_000_000


def test_loss_is_zero_for_perfect_prediction_and_ignores_invalid():
    t = torch.rand(1, 1, 8, 8, 8) * 2 - 1
    assert sdf_loss(t, t).item() == 0
    v = torch.zeros_like(t)
    assert sdf_loss(t + 1, t, valid=v).item() == 0


def test_residual_variant_runs():
    m = UNet3D(base=4, depth=3, predict_residual=True)
    x = torch.rand(2, 1, 16, 16, 16) * 1.8 - 0.9
    assert m(x).shape == x.shape
