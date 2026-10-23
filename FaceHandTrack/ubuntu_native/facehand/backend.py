"""PyTorch device choice: the only place that knows about CUDA.

    accel="auto"  NVIDIA GPU (CUDA) if PyTorch can see one, otherwise the CPU
    accel="cuda"  the GPU (falls back to CPU with a message if there is none)
    accel="cpu"   the CPU
"""
import os
import time

import torch


def pick_device(accel: str = "auto"):
    """Returns (torch.device, human-readable name)."""
    if accel in ("auto", "cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # fixed input sizes: let cuDNN pick the fastest kernels once
        torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32 on RTX 30xx and newer
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda"), f"cuda ({torch.cuda.get_device_name(0)})"
    if accel == "cuda":
        print("[torch] CUDA was requested but PyTorch sees no NVIDIA GPU (no driver, or CPU-only PyTorch); "
              "using the CPU")
    elif accel not in ("auto", "cpu"):
        raise ValueError(f"accel must be auto, cuda or cpu, not {accel!r}")
    return torch.device("cpu"), f"cpu ({os.cpu_count()} threads)"


def sync(device: torch.device):
    """Wait for queued GPU work, so a stage's time is measured where it ends."""
    if device.type == "cuda":
        torch.cuda.synchronize()


class Timer:
    """Adds milliseconds per stage into a dict (GPU work included)."""

    def __init__(self, device):
        self.device = device
        self.timings = {}

    def start(self):
        return time.perf_counter()

    def stop(self, key, t0):
        sync(self.device)
        self.timings[key] = self.timings.get(key, 0.0) + (time.perf_counter() - t0) * 1000


class TorchModel:
    """Runs a torch module on NumPy arrays: arrays in, float32 arrays out."""

    def __init__(self, module: torch.nn.Module, device: torch.device):
        self.module = module.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, *arrays):
        out = self.module(*[torch.from_numpy(a).to(self.device) for a in arrays])
        out = out if isinstance(out, (tuple, list)) else (out,)
        return [o.float().cpu().numpy() for o in out]
