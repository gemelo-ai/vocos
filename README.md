# Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis

[Audio samples](https://gemelo-ai.github.io/vocos/) |
Paper [[abs]](https://arxiv.org/abs/2306.00814) [[pdf]](https://arxiv.org/pdf/2306.00814.pdf)

Vocos is a fast neural vocoder designed to synthesize audio waveforms from acoustic features. Trained using a Generative
Adversarial Network (GAN) objective, Vocos can generate waveforms in a single forward pass. Unlike other typical
GAN-based vocoders, Vocos does not model audio samples in the time domain. Instead, it generates spectral
coefficients, facilitating rapid audio reconstruction through inverse Fourier transform.

## Installation

To use Vocos only in inference mode, install it using:

```bash
pip install vocos
```

If you wish to train the model, install it with additional dependencies:

```bash
pip install vocos[train]
```

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

y, sr = torchaudio.load(YOUR_AUDIO_FILE)
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

audio_tokens = torch.randint(low=0, high=1024, size=(8, 200))  # 8 codeboooks, 200 frames
features = vocos.codes_to_features(audio_tokens)
bandwidth_id = torch.tensor([2])  # 6 kbps

audio = vocos.decode(features, bandwidth_id=bandwidth_id)
```

Copy-synthesis from a file: It extracts and quantizes features with EnCodec, then reconstructs them with Vocos in a
single forward pass.

```python
y, sr = torchaudio.load(YOUR_AUDIO_FILE)
if y.size(0) > 1:  # mix to mono
    y = y.mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)

y_hat = vocos(y, bandwidth_id=bandwidth_id)
```

### Integrate with 🐶 [Bark](https://github.com/suno-ai/bark) text-to-audio model

See [example notebook](notebooks%2FBark%2BVocos.ipynb).

## Pre-trained models

| Model Name                                                                          | Dataset       | Training Iterations | Parameters 
|-------------------------------------------------------------------------------------|---------------|-------------------|------------|
| [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz)         | LibriTTS      | 1M                | 13.5M
| [charactr/vocos-encodec-24khz](https://huggingface.co/charactr/vocos-encodec-24khz) | DNS Challenge | 2M                | 7.9M

## Training

Prepare a filelist of audio files for the training and validation set:

```bash
find $TRAIN_DATASET_DIR -name *.wav > filelist.train
find $VAL_DATASET_DIR -name *.wav > filelist.val
```

Fill a config file, e.g. [vocos.yaml](configs%2Fvocos.yaml), with your filelist paths and start training with:

```bash
python train.py -c configs/vocos.yaml
```

Refer to [Pytorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/) for details about customizing the
training pipeline.

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
