"""CosyVoice3 flow-matching DiT on the AMD XDNA1 NPU (Ryzen AI 7040/8040)."""
from .flow import NpuEstimator, install
from .gate import NpuGate
from .steps import AdaptiveSteps

__all__ = ["NpuEstimator", "install", "NpuGate", "AdaptiveSteps", "soundfile_audio"]


def soundfile_audio():
    """Read audio files through soundfile in torchaudio.load.

    torchaudio >= 2.9 loads files only through the separate torchcodec package (and FFmpeg);
    CosyVoice only reads plain wav reference recordings, which soundfile handles.
    """
    import soundfile
    import torch
    import torchaudio

    def load(uri, *args, **kwargs):
        x, sr = soundfile.read(uri, dtype="float32", always_2d=True)
        return torch.from_numpy(x.T.copy()), sr

    torchaudio.load = load
