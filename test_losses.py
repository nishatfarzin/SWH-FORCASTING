import pytest

torch = pytest.importorskip("torch")

from waveshelf import lstm  # noqa: E402


def test_focal_matches_mse_when_gamma_zero():
    pred, target = torch.randn(64), torch.randn(64)
    focal0 = lstm.focal_mse(pred, target, gamma=0.0)
    assert torch.isclose(focal0, torch.mean((pred - target) ** 2))


def test_focal_weights_never_exceed_one_and_normalised_has_mean_one():
    pred, target = torch.randn(256), torch.randn(256)
    plain = lstm.focal_mse(pred, target)
    assert plain <= torch.mean((pred - target) ** 2) + 1e-9
    r = target - pred
    w = (r.abs() / r.abs().max()) ** 2
    assert (w / w.mean()).mean().item() == pytest.approx(1.0, abs=1e-6)


def test_unknown_loss_name():
    with pytest.raises(ValueError):
        lstm.get_loss("nope")
