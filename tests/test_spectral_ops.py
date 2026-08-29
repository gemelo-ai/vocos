import pytest
import scipy.signal
import torch

from vocos.spectral_ops import IMDCT, MDCT


@pytest.mark.parametrize("transform", [MDCT, IMDCT])
def test_mdct_transforms_use_scipy_windows_api(transform):
    frame_len = 16

    module = transform(frame_len=frame_len)
    expected = torch.from_numpy(scipy.signal.windows.cosine(frame_len)).float()

    torch.testing.assert_close(module.window, expected)


def test_centered_mdct_roundtrip():
    torch.manual_seed(0)
    audio = torch.randn(2, 64)

    coefficients = MDCT(frame_len=16, padding="center")(audio)
    reconstructed = IMDCT(frame_len=16, padding="center")(coefficients)

    assert reconstructed.shape == audio.shape
    torch.testing.assert_close(reconstructed, audio, atol=2e-6, rtol=1e-5)


def test_same_padding_preserves_waveform_length():
    torch.manual_seed(0)
    audio = torch.randn(2, 64)

    coefficients = MDCT(frame_len=16, padding="same")(audio)
    reconstructed = IMDCT(frame_len=16, padding="same")(coefficients)

    assert reconstructed.shape == audio.shape
    assert torch.isfinite(reconstructed).all()
    torch.testing.assert_close(reconstructed[:, 8:-8], audio[:, 8:-8], atol=2e-6, rtol=1e-5)
