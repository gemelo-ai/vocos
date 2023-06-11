# Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis

[Audio samples](https://charactr-platform.github.io/vocos/) |
Paper [[abs]](https://arxiv.org/abs/2306.00814) [[pdf]](https://arxiv.org/pdf/2306.00814.pdf)

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

with torch.no_grad():
    audio = vocos.decode(mel)
```

Copy-synthesis from a file:

```python
import torchaudio

y, sr = torchaudio.load(YOUR_AUDIO_FILE)
if y.size(0) > 1:  # mix to mono
    y = y.mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)

with torch.no_grad():
    y_hat = vocos(y)
```

### Reconstruct audio from EnCodec

Additionally, you need to provide a `bandwidth_id` which corresponds to the lookup embedding for bandwidth from the
list: `[1.5, 3.0, 6.0, 12.0]`.

```python
vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz")

quantized_features = torch.randn(1, 128, 256)
bandwidth_id = torch.tensor([3])  # 12 kbps

with torch.no_grad():
    audio = vocos.decode(quantized_features, bandwidth_id=bandwidth_id)  
```

Copy-synthesis from a file: It extracts and quantizes features with EnCodec, then reconstructs them with Vocos in a
single forward pass.

```python
y, sr = torchaudio.load(YOUR_AUDIO_FILE)
if y.size(0) > 1:  # mix to mono
    y = y.mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)

with torch.no_grad():
    y_hat = vocos(y, bandwidth_id=bandwidth_id)
```

## Pre-trained models

The provided models were trained up to 2.5 million generator iterations, which resulted in slightly better objective
scores
compared to those reported in the paper.

| Model Name                                                                          | Dataset       | Training Iterations | Parameters 
|-------------------------------------------------------------------------------------|---------------|---------------------|------------|
| [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz)         | LibriTTS      | 2.5 M               | 13.5 M     
| [charactr/vocos-encodec-24khz](https://huggingface.co/charactr/vocos-encodec-24khz) | DNS Challenge | 2.5 M               | 7.9 M      

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