import pytest
import torch

from vocos.heads import IMDCTCosHead, IMDCTSymExpHead, ISTFTHead


@pytest.mark.parametrize("head_class", [IMDCTSymExpHead, IMDCTCosHead])
def test_clip_audio_clips_waveform_without_changing_shape(head_class):
    raw_head = head_class(dim=4, mdct_frame_len=16, padding="center", clip_audio=False)
    clipped_head = head_class(dim=4, mdct_frame_len=16, padding="center", clip_audio=True)
    clipped_head.load_state_dict(raw_head.state_dict())

    with torch.no_grad():
        raw_head.out.weight.zero_()
        clipped_head.out.weight.zero_()
        raw_head.out.bias.fill_(3.0)
        clipped_head.out.bias.fill_(3.0)

    inputs = torch.zeros(2, 4, 4)
    raw_audio = raw_head(inputs)
    clipped_audio = clipped_head(inputs)

    assert raw_audio.shape == clipped_audio.shape
    assert raw_audio.abs().max() > 1.0
    torch.testing.assert_close(clipped_audio, raw_audio.clamp(-1.0, 1.0))


def test_istft_head_produces_finite_waveform():
    torch.manual_seed(0)
    head = ISTFTHead(dim=4, n_fft=16, hop_length=4, padding="same")

    audio = head(torch.randn(2, 5, 4))

    assert audio.shape == (2, 20)
    assert torch.isfinite(audio).all()
