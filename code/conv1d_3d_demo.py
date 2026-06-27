"""
conv1d_3d_demo.py  -  Chapter 10 companion
Convolutional Neural Networks for Beginners (AI Foundations Series, Book 4)
(c) 2026 David Grunwald. All rights reserved.

Runs a 1D convolution over a signal and a 3D convolution over a small volume,
generalizing the 2D operation to sequences (audio, time-series) and volumetric
data (video, medical scans). The mechanics never change; only the axes do.
"""
import numpy as np


def conv1d(signal, kernel):
    """1D 'valid' convolution: slide a kernel along a sequence."""
    n, k = len(signal), len(kernel)
    out = np.zeros(n - k + 1)
    for i in range(len(out)):
        out[i] = np.sum(signal[i:i + k] * kernel)
    return out


def conv3d(volume, kernel):
    """3D 'valid' convolution: slide a cube kernel through a volume."""
    d, h, w = volume.shape
    kd, kh, kw = kernel.shape
    out = np.zeros((d - kd + 1, h - kh + 1, w - kw + 1))
    for z in range(out.shape[0]):
        for y in range(out.shape[1]):
            for x in range(out.shape[2]):
                region = volume[z:z + kd, y:y + kh, x:x + kw]
                out[z, y, x] = np.sum(region * kernel)
    return out


if __name__ == "__main__":
    # --- 1D: a smoothing kernel on a noisy signal ---
    signal = np.array([0, 1, 3, 1, 0, 2, 4, 2, 0], dtype=float)
    smooth = np.array([1, 1, 1], dtype=float) / 3   # moving average
    print("1D convolution (moving-average smoothing):")
    print("  input: ", signal)
    print("  output:", np.round(conv1d(signal, smooth), 3))

    # --- 3D: an averaging kernel on a 3x3x3 volume ---
    rng = np.random.default_rng(7)
    volume = rng.integers(0, 5, size=(3, 3, 3)).astype(float)
    kernel = np.ones((2, 2, 2)) / 8
    print("\n3D convolution (3x3x3 volume, 2x2x2 averaging kernel):")
    print("  output shape:", conv3d(volume, kernel).shape)
    print("  output:\n", np.round(conv3d(volume, kernel), 3))

    print("\nIf you understood 2D convolutions, these are the same idea")
    print("with axes added or removed. That is the whole takeaway.")
