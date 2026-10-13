"""SSIM loss with the fused_ssim call signature.

The gsplat example imports fused_ssim, whose extension has to be compiled
against the same CUDA major version as PyTorch. When that compiler is not
available, the trainer still needs a differentiable scalar SSIM.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_window(
    window_size: int, sigma: float, channel: int, ref: torch.Tensor
) -> torch.Tensor:
    coords = torch.arange(window_size, device=ref.device, dtype=ref.dtype)
    coords = coords - window_size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel)
    return window.expand(channel, 1, window_size, window_size).contiguous()


def fused_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    padding: str = "same",
    train: bool = True,
) -> torch.Tensor:
    """Return the mean SSIM of NCHW images. `train` is accepted and unused."""
    del train
    if img1.shape != img2.shape:
        raise ValueError(
            f"SSIM inputs differ: {tuple(img1.shape)} vs {tuple(img2.shape)}"
        )
    if padding not in {"same", "valid"}:
        raise ValueError(f"padding must be 'same' or 'valid', got {padding!r}")
    window_size = 11
    channel = img1.shape[1]
    window = _gaussian_window(window_size, 1.5, channel, img1)
    pad = 0 if padding == "valid" else window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq = mu1.square()
    mu2_sq = mu2.square()
    mu12 = mu1 * mu2
    sigma1_sq = F.conv2d(img1.square(), window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2.square(), window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu12
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu12 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return (numerator / denominator).mean()
