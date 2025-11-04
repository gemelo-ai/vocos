"""Export Vocos model to ONNX format.

Usage: uv run scripts/export_onnx.py

Note: Outputs mag, cos(phase), sin(phase) instead of audio (ONNX doesn't support complex numbers).
Reconstruct audio: S = mag * (cos + i*sin), then apply ISTFT.
"""

import torch
from torch import nn
from vocos import Vocos


class ONNXCompatibleHead(nn.Module):
    """ONNX-compatible head that avoids complex numbers by outputting magnitude and phase components."""
    
    def __init__(self, head_module):
        super().__init__()
        self.out = head_module.out
    
    def forward(self, x):
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag).clip(max=1e2)
        # Return magnitude and phase as separate real components (cos, sin)
        return mag, torch.cos(p), torch.sin(p)


class VocosDecoder(nn.Module):
    """Vocos decoder (backbone + head) wrapper for ONNX export."""
    
    def __init__(self, vocos_model):
        super().__init__()
        self.backbone = vocos_model.backbone
        self.head = ONNXCompatibleHead(vocos_model.head)
    
    def forward(self, features):
        return self.head(self.backbone(features))


# Load and export model
print("Loading model...")
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
decoder = VocosDecoder(vocos)
decoder.eval()

print("Exporting to ONNX...")
mel = torch.randn(1, 100, 256)  # Dummy input: (batch, n_mels, time)
output_path = "vocos-mel-24khz.onnx"

with torch.no_grad():
    torch.onnx.export(
        decoder, mel, output_path,
        export_params=True, opset_version=18, do_constant_folding=True,
        input_names=['mel'], output_names=['mag', 'cos_phase', 'sin_phase'],
        # Allow dynamic batch size and time dimensions
        dynamic_axes={'mel': {0: 'batch_size', 2: 'time'},
                      'mag': {0: 'batch_size', 2: 'time'},
                      'cos_phase': {0: 'batch_size', 2: 'time'},
                      'sin_phase': {0: 'batch_size', 2: 'time'}},
        dynamo=False  # Use legacy exporter (dynamo has issues with this model)
    )

print(f"✓ Exported to {output_path}")


# Test the export
def test_export():
    """Verify ONNX output matches PyTorch and can reconstruct audio."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("⚠ Install onnxruntime to test: uv pip install onnxruntime")
        return
    
    print("\nTesting...")
    # Compare PyTorch vs ONNX outputs
    with torch.no_grad():
        mag_pt, cos_pt, sin_pt = decoder(mel)
    
    session = ort.InferenceSession(output_path)
    mag_ox, cos_ox, sin_ox = session.run(None, {"mel": mel.numpy()})
    
    diffs = [
        np.abs(mag_pt.numpy() - mag_ox).max(),
        np.abs(cos_pt.numpy() - cos_ox).max(),
        np.abs(sin_pt.numpy() - sin_ox).max()
    ]
    
    print(f"Max diff: {max(diffs):.2e} {'✓' if max(diffs) < 1e-4 else '⚠'}")
    
    # Verify audio reconstruction from ONNX outputs
    S = torch.from_numpy(mag_ox) * (torch.from_numpy(cos_ox) + 1j * torch.from_numpy(sin_ox))
    audio = vocos.head.istft(S)
    print(f"Audio: {audio.shape}, range [{audio.min():.3f}, {audio.max():.3f}]")


test_export()