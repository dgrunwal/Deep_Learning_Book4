"""
conv_from_scratch.py  -  Chapter 1 companion
Convolutional Neural Networks for Beginners (AI Foundations Series, Book 4)
(c) 2026 David Grunwald. All rights reserved.

A single 2D convolution in pure NumPy. No frameworks.
Slides one kernel over a small grayscale image, multiplies and sums,
and prints the resulting feature map. Every number is traceable by hand.
"""
import numpy as np


def conv2d(image, kernel):
    """Apply a 2D convolution with 'valid' padding (no border padding)."""
    kh, kw = kernel.shape
    h, w = image.shape
    out_h, out_w = h - kh + 1, w - kw + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = image[i:i + kh, j:j + kw]   # the receptive field
            out[i, j] = np.sum(region * kernel)  # multiply and sum
    return out


if __name__ == "__main__":
    image = np.array([[1, 2, 3, 0],
                      [4, 5, 6, 1],
                      [7, 8, 9, 2],
                      [1, 0, 1, 3]], dtype=float)

    edge = np.array([[-1, -1, -1],
                     [-1,  8, -1],
                     [-1, -1, -1]], dtype=float)   # simple edge detector

    print("Input image:\n", image)
    print("\nEdge-detection kernel:\n", edge)
    print("\nFeature map (4x4 image, 3x3 kernel -> 2x2 output):\n", conv2d(image, edge))
