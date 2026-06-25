"""
cifar10_baseline.py  -  Chapter 3 companion
Convolutional Neural Networks for Beginners (AI Foundations Series, Book 4)
(c) 2026 David Grunwald. All rights reserved.

A clean beginner CNN trained on CIFAR-10. Conv-pool blocks plus a dense head.
Establishes the book's BASELINE accuracy; every later improvement is measured
against this control experiment.

Run:  python cifar10_baseline.py
Requires: tensorflow
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TF on Windows/Py3.13

import tensorflow as tf
from tensorflow.keras import layers, models, datasets


def build_model():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10),  # logits; from_logits=True in the loss
    ])
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def main():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # normalize to [0, 1]

    model = build_model()
    model.summary()

    history = model.fit(
        x_train, y_train, epochs=10, batch_size=64,
        validation_data=(x_test, y_test), verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nBaseline test accuracy: {test_acc:.4f}")
    print("Keep this number. Every later chapter compares against it.")


if __name__ == "__main__":
    main()
