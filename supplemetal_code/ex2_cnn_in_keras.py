"""
ex2_cnn_in_keras.py  -  Supplemental in-line example (Chapter 3)
Convolutional Neural Networks for Beginners (AI Foundations Series, Book 4)
(c) 2026 David Grunwald. All rights reserved.

The same CNN, built two ways in Keras.

Chapter 3 builds a CNN end to end. This example shows the two styles you will
meet everywhere in Keras code:

    1. The SEQUENTIAL API - a simple straight-line stack of layers.
       (CONV -> BatchNorm -> ReLU -> MaxPool -> Flatten -> Dense)

    2. The FUNCTIONAL API - layers wired together by hand, which is what you
       need once a network branches or merges (every later chapter uses it).
       (CONV -> ReLU -> MaxPool -> CONV -> ReLU -> MaxPool -> Flatten -> Dense)

To keep the example self-contained, it trains on a small batch of SYNTHETIC
images (random pixels with random labels). The point here is the architecture
and the build/compile/fit loop, not accuracy - swap in real images and the
code is unchanged.

Requires: tensorflow   (pip install tensorflow)
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TensorFlow logging

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl

IMG_SHAPE = (64, 64, 3)


def sequential_model():
    """A straight-line CNN using the Sequential API.

    Sequential is the simplest way to stack layers: each one feeds the next,
    in order. It is perfect for a plain CNN with no branching.
    """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=IMG_SHAPE),
        tfl.ZeroPadding2D(padding=3),
        tfl.Conv2D(32, (7, 7), strides=(1, 1)),
        tfl.BatchNormalization(axis=3),
        tfl.ReLU(),
        tfl.MaxPool2D(),
        tfl.Flatten(),
        tfl.Dense(1, activation="sigmoid"),   # one output: binary class
    ])
    return model


def functional_model(input_shape=IMG_SHAPE, classes=6):
    """The same idea using the Functional API.

    Here we create an Input tensor and pass it through each layer by calling
    the layer on the previous output. This explicit wiring is what lets later
    architectures (ResNet, U-Net) split and merge paths.
    """
    inputs = tf.keras.Input(shape=input_shape)
    Z1 = tfl.Conv2D(8, (4, 4), strides=1, padding="same")(inputs)
    A1 = tfl.ReLU()(Z1)
    P1 = tfl.MaxPool2D(pool_size=8, strides=8, padding="same")(A1)
    Z2 = tfl.Conv2D(16, (2, 2), strides=1, padding="same")(P1)
    A2 = tfl.ReLU()(Z2)
    P2 = tfl.MaxPool2D(pool_size=4, strides=4, padding="same")(A2)
    F = tfl.Flatten()(P2)
    outputs = tfl.Dense(classes, activation="softmax")(F)   # multi-class
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def fake_images(n, classes):
    """Make n random 64x64 RGB images and random labels - a stand-in for a
    real dataset so the script runs anywhere with no downloads."""
    X = np.random.rand(n, *IMG_SHAPE).astype("float32")
    y = np.random.randint(0, classes, size=(n,))
    return X, y


def main():
    print("=" * 60)
    print("1) Sequential API model (binary classification)")
    print("=" * 60)
    seq = sequential_model()
    seq.compile(optimizer="adam", loss="binary_crossentropy",
                metrics=["accuracy"])
    seq.summary()

    X, y = fake_images(64, classes=2)
    print("\nTraining on synthetic data for 2 epochs (demo only)...")
    seq.fit(X, y, epochs=2, batch_size=16, verbose=2)

    print("\n" + "=" * 60)
    print("2) Functional API model (6-class classification)")
    print("=" * 60)
    fun = functional_model(classes=6)
    fun.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    fun.summary()

    Xc, yc = fake_images(64, classes=6)
    print("\nTraining on synthetic data for 2 epochs (demo only)...")
    fun.fit(Xc, yc, epochs=2, batch_size=16, verbose=2)

    print("\nBoth models built, compiled, and trained. Swap in real images to")
    print("turn this scaffold into a working classifier.")


if __name__ == "__main__":
    main()
