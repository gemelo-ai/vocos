import torch

from vocos import Vocos

vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

mel = torch.randn(1, 100, 256)  # B, C, T
audio = vocos.decode(mel)