# Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis

[Audio samples](https://gemelo-ai.github.io/vocos/) |
Paper [[abs]](https://arxiv.org/abs/2306.00814) [[pdf]](https://arxiv.org/pdf/2306.00814.pdf)

Vocos is a fast neural vocoder designed to synthesize audio waveforms from acoustic features. Trained using a Generative
Adversarial Network (GAN) objective, Vocos can generate waveforms in a single forward pass. Unlike other typical
GAN-based vocoders, Vocos does not model audio samples in the time domain. Instead, it generates spectral
coefficients, facilitating rapid audio reconstruction through inverse Fourier transform.

## Project maintenance

Vocos remains available for research and production use, and the maintainers are continuing to review focused,
backward-compatible improvements. Current priorities include documentation, project infrastructure, issue and pull
request triage, and making the contribution path clearer. If you use Vocos downstream, feedback and small,
well-scoped contributions are welcome.

## Installation

To use Vocos only in inference mode, install it using:

```bash
python -m pip install vocos
```

The training entry point, configs, and metrics live in the source repository. To train a model, clone the repository
and install it with the training dependencies:

```bash
git clone https://github.com/gemelo-ai/vocos.git
cd vocos
python -m pip install "setuptools<80"
python -m pip install -e ".[train]"
```

The checked-in training stack pins PyTorch Lightning 1.8.6, which still uses `pkg_resources`; keeping Setuptools below
version 80 in the training environment preserves that compatibility.

## Usage

### Reconstruct audio from mel-spectrogram

```python
import torch

from vocos import Vocos

vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

mel = torch.randn(1, 100, 256)  # B, C, T
audio = vocos.decode(mel)
```

Copy-synthesis from a file:

```python
import torchaudio

audio_path = "path/to/audio.wav"
y, sr = torchaudio.load(audio_path)
if y.size(0) > 1:  # mix to mono
    y = y.mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)
y_hat = vocos(y)
```

### Reconstruct audio from EnCodec tokens

Additionally, you need to provide a `bandwidth_id` which corresponds to the embedding for bandwidth from the
list: `[1.5, 3.0, 6.0, 12.0]`.

```python
vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz")

audio_tokens = torch.randint(low=0, high=1024, size=(8, 200))  # 8 codebooks, 200 frames
features = vocos.codes_to_features(audio_tokens)
bandwidth_id = torch.tensor([2])  # 6 kbps

audio = vocos.decode(features, bandwidth_id=bandwidth_id)
```

Copy-synthesis from a file: It extracts and quantizes features with EnCodec, then reconstructs them with Vocos in a
single forward pass.

```python
audio_path = "path/to/audio.wav"
y, sr = torchaudio.load(audio_path)
if y.size(0) > 1:  # mix to mono
    y = y.mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)

y_hat = vocos(y, bandwidth_id=bandwidth_id)
```

### Integrate with 🐶 [Bark](https://github.com/suno-ai/bark) text-to-audio model

See [example notebook](notebooks/Bark%2BVocos.ipynb).

## Pre-trained models

| Model Name                                                                          | Dataset       | Training Iterations | Parameters |
|-------------------------------------------------------------------------------------|---------------|---------------------|------------|
| [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz)         | LibriTTS      | 1M                  | 13.5M      |
| [charactr/vocos-encodec-24khz](https://huggingface.co/charactr/vocos-encodec-24khz) | DNS Challenge | 2M                  | 7.9M       |

## Training

Prepare a filelist of audio files for the training and validation set:

```bash
find "$TRAIN_DATASET_DIR" \( -type f -o -type l \) -name '*.wav' -print > filelist.train
find "$VAL_DATASET_DIR" \( -type f -o -type l \) -name '*.wav' -print > filelist.val
```

This includes symlinked `.wav` files without following symlinked directories.

Fill a config file, e.g. [vocos.yaml](configs/vocos.yaml), with your filelist paths and start training with:

```bash
python train.py -c configs/vocos.yaml
```

The checked-in training code and configs target PyTorch Lightning 1.8.6. Refer to the
[PyTorch Lightning 1.8.6 documentation](https://pytorch-lightning.readthedocs.io/en/1.8.6/) for details about
customizing that training pipeline.

## Contributing

Bug reports, documentation improvements, and focused fixes are welcome. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request, and include enough detail for maintainers to
understand and reproduce the change.

## Citation

If this code contributes to your research, please cite our work:

```
@article{siuzdak2023vocos,
  title={Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis},
  author={Siuzdak, Hubert},
  journal={arXiv preprint arXiv:2306.00814},
  year={2023}
}
```

## License

The code in this repository is released under the MIT license as found in the
[LICENSE](LICENSE) file.
