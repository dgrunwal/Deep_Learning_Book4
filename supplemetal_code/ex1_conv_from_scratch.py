"""
ex1_conv_from_scratch.py  -  Supplemental in-line example (Chapters 1-2)
Convolutional Neural Networks for Beginners (AI Foundations Series, Book 4)
(c) 2026 David Grunwald. All rights reserved.

The forward pass of a convolutional layer, built by hand in NumPy.

This is the long-form companion to Chapters 1 and 2. It implements the three
operations every CNN is made of, with no framework hiding the arithmetic:

    1. zero_pad          - frame an image with a border of zeros
    2. conv_single_step  - one filter applied to one window (multiply and sum)
    3. conv_forward      - slide that step across the whole image
    4. pool_forward      - max / average pooling

Run it directly to see each operation on a tiny random example, with the
output shapes printed so you can check them against the formula from Chapter 2:

    out = (in - f + 2*pad) / stride + 1

Requires: numpy   (pip install numpy)
"""
import numpy as np


def zero_pad(X, pad):
    """Pad the height and width of a batch of images with zeros.

    X   -- array of shape (m, n_H, n_W, n_C): m images, height, width, channels
    pad -- number of zero pixels to add on every side

    Returns an array of shape (m, n_H + 2*pad, n_W + 2*pad, n_C).
    We pad only the two spatial axes (1 and 2), never the batch or channels.
    """
    return np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                  mode="constant", constant_values=0)


def conv_single_step(a_slice, W, b):
    """Apply one filter to one window of the input.

    a_slice -- a window of the input, shape (f, f, n_C_prev)
    W       -- one filter, same shape (f, f, n_C_prev)
    b       -- one bias, shape (1, 1, 1)

    This is the heart of a convolution: multiply element-wise, sum every
    number in the result to a single value, then add the bias.
    """
    s = a_slice * W              # element-wise product
    Z = np.sum(s)                # collapse the window to one number
    Z = Z + float(b.item())      # add the bias (pull out the single value)
    return Z


def conv_forward(A_prev, W, b, stride, pad):
    """Slide every filter across every image to build the output volume.

    A_prev -- input volume, shape (m, n_H_prev, n_W_prev, n_C_prev)
    W      -- all filters,  shape (f, f, n_C_prev, n_C)
    b      -- all biases,   shape (1, 1, 1, n_C)
    stride -- how many pixels the window jumps each step
    pad    -- zero-padding added around each image

    Returns the output volume Z of shape (m, n_H, n_W, n_C).
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    # Output spatial size - the Chapter 2 formula, floored to an integer.
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):                       # each image in the batch
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):                 # each output row
            vert_start = h * stride
            vert_end = vert_start + f
            for w in range(n_W):             # each output column
                horiz_start = w * stride
                horiz_end = horiz_start + f
                for c in range(n_C):         # each filter / output channel
                    window = a_prev_pad[vert_start:vert_end,
                                        horiz_start:horiz_end, :]
                    Z[i, h, w, c] = conv_single_step(
                        window, W[:, :, :, c], b[:, :, :, c])
    return Z


def pool_forward(A_prev, f, stride, mode="max"):
    """Downsample each channel with a sliding pooling window.

    A_prev -- input volume, shape (m, n_H_prev, n_W_prev, n_C)
    f      -- pooling window size (f x f)
    stride -- step between windows
    mode   -- "max" or "average"

    Pooling has no weights; it just summarizes each window. It shrinks the
    height and width while leaving the number of channels unchanged.
    """
    (m, n_H_prev, n_W_prev, n_C) = A_prev.shape
    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1

    A = np.zeros((m, n_H, n_W, n_C))
    for i in range(m):
        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f
            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f
                for c in range(n_C):
                    window = A_prev[i, vert_start:vert_end,
                                    horiz_start:horiz_end, c]
                    if mode == "max":
                        A[i, h, w, c] = np.max(window)
                    else:
                        A[i, h, w, c] = np.mean(window)
    return A


def main():
    np.random.seed(1)

    # A tiny "batch" of 2 images, 5x7 pixels, 4 channels.
    A_prev = np.random.randn(2, 5, 7, 4)
    # 8 filters, each 3x3, matching the 4 input channels.
    W = np.random.randn(3, 3, 4, 8)
    b = np.random.randn(1, 1, 1, 8)

    print("Convolution forward pass")
    print("  input shape :", A_prev.shape)
    Z = conv_forward(A_prev, W, b, stride=2, pad=1)
    print("  output shape:", Z.shape, "  (stride=2, pad=1)")
    print("  output mean :", round(float(np.mean(Z)), 4))

    print("\nPooling forward pass")
    A_prev2 = np.random.randn(2, 5, 5, 3)
    print("  input shape :", A_prev2.shape)
    A_max = pool_forward(A_prev2, f=3, stride=1, mode="max")
    A_avg = pool_forward(A_prev2, f=3, stride=1, mode="average")
    print("  max-pool out:", A_max.shape)
    print("  avg-pool out:", A_avg.shape)

    print("\nEverything above was plain NumPy - every number is traceable by hand.")


if __name__ == "__main__":
    main()
