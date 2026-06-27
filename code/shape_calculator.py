"""
shape_calculator.py  -  Chapter 2 companion
Convolutional Neural Networks for Beginners (AI Foundations Series, Book 4)
(c) 2026 David Grunwald. All rights reserved.

Given input size, kernel, padding, and stride, prints the output volume of
each layer in a stack. Use it to sanity-check an architecture before training.
Formula: out = floor((n + 2p - f) / s) + 1
"""
import math


def conv_out(n, f, p, s):
    """Output spatial size for one dimension of a conv or pool layer."""
    return math.floor((n + 2 * p - f) / s) + 1


def describe_stack(input_hw, channels, layers):
    """
    input_hw: (height, width) of the input
    channels: number of input channels
    layers:   list of dicts with keys: name, f (kernel), p (pad), s (stride),
              and out_ch (output channels; None for pooling -> keeps channels)
    """
    h, w = input_hw
    ch = channels
    print(f"{'Layer':<22}{'Output (HxWxC)':<22}{'Params note'}")
    print("-" * 60)
    print(f"{'input':<22}{f'{h}x{w}x{ch}':<22}")
    for layer in layers:
        h = conv_out(h, layer["f"], layer["p"], layer["s"])
        w = conv_out(w, layer["f"], layer["p"], layer["s"])
        if layer.get("out_ch"):
            ch = layer["out_ch"]
        note = f"{layer['f']}x{layer['f']}, pad {layer['p']}, stride {layer['s']}"
        print(f"{layer['name']:<22}{f'{h}x{w}x{ch}':<22}{note}")


if __name__ == "__main__":
    # A small VGG-style stack on a 32x32x3 image.
    stack = [
        {"name": "conv1 (same)", "f": 3, "p": 1, "s": 1, "out_ch": 32},
        {"name": "maxpool1",     "f": 2, "p": 0, "s": 2, "out_ch": None},
        {"name": "conv2 (same)", "f": 3, "p": 1, "s": 1, "out_ch": 64},
        {"name": "maxpool2",     "f": 2, "p": 0, "s": 2, "out_ch": None},
        {"name": "conv3 (valid)", "f": 3, "p": 0, "s": 1, "out_ch": 128},
    ]
    describe_stack((32, 32), 3, stack)
