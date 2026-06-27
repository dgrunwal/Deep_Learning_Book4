"""
residual_block.py  -  Chapter 4 companion
Convolutional Neural Networks for Beginners (AI Foundations Series, Book 4)
(c) 2026 David Grunwald. All rights reserved.

A from-scratch residual block in NumPy. Runs a forward pass with and without
the skip connection so you can see how the identity path preserves the signal
and lets gradients flow in very deep networks.
"""
import numpy as np


def relu(x):
    return np.maximum(0, x)


def conv2d(image, kernel):
    """Single-channel 2D convolution with 'same' padding."""
    f = kernel.shape[0]
    p = f // 2
    padded = np.pad(image, p, mode="constant")
    h, w = image.shape
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            out[i, j] = np.sum(padded[i:i + f, j:j + f] * kernel)
    return out


def residual_block(x, k1, k2, use_skip=True):
    """
    A minimal residual block: x -> conv -> relu -> conv -> (+x) -> relu
    With use_skip=False it becomes a plain two-layer block for comparison.
    """
    out = relu(conv2d(x, k1))
    out = conv2d(out, k2)
    if use_skip:
        out = out + x          # the identity skip connection
    return relu(out)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    x = rng.normal(size=(5, 5))
    k1 = rng.normal(size=(3, 3)) * 0.1
    k2 = rng.normal(size=(3, 3)) * 0.1

    with_skip = residual_block(x, k1, k2, use_skip=True)
    without_skip = residual_block(x, k1, k2, use_skip=False)

    print("Input mean magnitude:        ", round(np.abs(x).mean(), 4))
    print("Output WITH skip (mean mag): ", round(np.abs(with_skip).mean(), 4))
    print("Output WITHOUT skip (mean):  ", round(np.abs(without_skip).mean(), 4))
    print("\nNotice the skip connection keeps the signal close to the input,")
    print("which is exactly what lets ResNets train at great depth.")
