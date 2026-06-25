"""
ex3_resnet_blocks.py  -  Supplemental in-line example (Chapter 4)
Convolutional Neural Networks for Beginners (AI Foundations Series, Book 4)
(c) 2026 David Grunwald. All rights reserved.

Residual learning: the two building blocks of ResNet.

Chapter 4 tours the classic architectures. ResNet's one durable idea is the
SKIP CONNECTION - adding a layer's input back to its output so the network
only has to learn the *difference* (the "residual"). This keeps gradients
flowing and lets networks go very deep.

This example implements the two blocks every ResNet is made of:

    identity_block       - input and output are the same shape, so the
                           shortcut is just X itself.
    convolutional_block  - the block changes shape (via stride), so the
                           shortcut needs its own 1x1 conv to match.

It then stacks a few blocks into a small ResNet and runs a forward pass on
synthetic images so the whole thing executes anywhere with no downloads.

Requires: tensorflow   (pip install tensorflow)
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TensorFlow logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization,
    Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D)
from tensorflow.keras.models import Model


def identity_block(X, f, filters):
    """A residual block where input and output shapes match.

    The shortcut path is the input X, untouched. The main path runs three
    convolutions, and at the end we ADD the shortcut back before the final
    ReLU. That addition is the whole trick.
    """
    F1, F2, F3 = filters
    X_shortcut = X                              # save the input for later

    # Main path: 1x1 -> f x f -> 1x1, each followed by BatchNorm.
    X = Conv2D(F1, 1, strides=1, padding="valid")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X = Conv2D(F2, f, strides=1, padding="same")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X = Conv2D(F3, 1, strides=1, padding="valid")(X)
    X = BatchNormalization(axis=3)(X)

    # Add the shortcut, THEN apply the final activation.
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    return X


def convolutional_block(X, f, filters, s=2):
    """A residual block that also changes the output shape.

    Because the main path uses stride s, its output is smaller than the
    input. The shortcut therefore can't be the raw input - it needs its own
    1x1 conv (with the same stride) to reshape X so the two paths can be
    added together.
    """
    F1, F2, F3 = filters
    X_shortcut = X

    # Main path (first conv uses the stride that shrinks the volume).
    X = Conv2D(F1, 1, strides=s, padding="valid")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X = Conv2D(F2, f, strides=1, padding="same")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X = Conv2D(F3, 1, strides=1, padding="valid")(X)
    X = BatchNormalization(axis=3)(X)

    # Shortcut path: reshape X to match the main path's new shape.
    X_shortcut = Conv2D(F3, 1, strides=s, padding="valid")(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    return X


def small_resnet(input_shape=(64, 64, 3), classes=6):
    """A compact ResNet built from the two blocks above.

    This is a trimmed-down ResNet (a handful of blocks instead of 50 layers)
    so it stays readable and runs fast, while keeping the real structure:
    stem -> convolutional_block -> identity_blocks -> pooling -> dense.
    """
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)

    # Stem
    X = Conv2D(64, 7, strides=2)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(3, strides=2)(X)

    # Stage 1: one conv block to set the shape, then identity blocks.
    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    # Stage 2: downsample, then more identity blocks.
    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2)
    X = identity_block(X, 3, [128, 128, 512])

    # Head
    X = GlobalAveragePooling2D()(X)
    X = Flatten()(X)
    outputs = Dense(classes, activation="softmax")(X)
    return Model(inputs=X_input, outputs=outputs)


def main():
    np.random.seed(1)

    model = small_resnet(input_shape=(64, 64, 3), classes=6)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    print("Small ResNet built from identity + convolutional blocks.")
    print("Total layers:", len(model.layers))

    # Forward pass on a few synthetic images - no training, no download.
    X = np.random.rand(4, 64, 64, 3).astype("float32")
    preds = model.predict(X, verbose=0)
    print("\nForward pass on 4 synthetic images")
    print("  prediction shape:", preds.shape, " (4 images x 6 class scores)")
    print("  row 0 sums to   :", round(float(preds[0].sum()), 4),
          "(softmax outputs are a probability distribution)")
    print("\nThe skip connections are what let this stack go deep without")
    print("the signal washing out. That is the one idea to take from ResNet.")


if __name__ == "__main__":
    main()
