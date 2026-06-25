"""
transpose_conv_demo.py  -  Chapter 7 companion
Convolutional Neural Networks for Beginners (AI Foundations Series, Book 4)
(c) 2026 David Grunwald. All rights reserved.

A tiny transpose convolution ("deconvolution") in NumPy that upsamples a 2x2
input to a larger grid, showing the learnable-upsampling mechanism behind the
decoder of a U-Net. This is the inverse-shaped sibling of a normal convolution.
"""
import numpy as np


def conv_transpose2d(x, kernel, stride=1):
    """
    Transpose convolution with 'valid'-style placement.
    Each input value scatters a scaled copy of the kernel into the output,
    and overlapping contributions are summed.
    """
    kh, kw = kernel.shape
    h, w = x.shape
    out_h = (h - 1) * stride + kh
    out_w = (w - 1) * stride + kw
    out = np.zeros((out_h, out_w))
    for i in range(h):
        for j in range(w):
            r, c = i * stride, j * stride
            out[r:r + kh, c:c + kw] += x[i, j] * kernel  # scatter and add
    return out


if __name__ == "__main__":
    x = np.array([[1, 2],
                  [3, 4]], dtype=float)

    kernel = np.array([[1, 1],
                       [1, 1]], dtype=float)

    print("Input (2x2):\n", x)
    print("\nKernel (2x2):\n", kernel)

    up1 = conv_transpose2d(x, kernel, stride=1)
    print("\nTranspose conv, stride 1 -> 3x3:\n", up1)

    up2 = conv_transpose2d(x, kernel, stride=2)
    print("\nTranspose conv, stride 2 -> 4x4 (true upsampling):\n", up2)

    print("\nU-Net's decoder uses exactly this to recover spatial resolution,")
    print("then skip connections add back the fine detail from the encoder.")
