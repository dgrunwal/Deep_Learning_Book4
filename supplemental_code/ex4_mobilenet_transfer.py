r"""
ex4_mobilenet_transfer.py  -  Supplemental in-line example (Chapter 5)
Convolutional Neural Networks for Beginners (AI Foundations Series, Book 4)
(c) 2026 David Grunwald. All rights reserved.

Transfer learning with MobileNetV2.

Chapter 5's highest-leverage skill: instead of training a network from
scratch, stand on one already trained on millions of images and adapt it to
your own. This example shows the standard two-stage recipe:

    Stage 1 - FEATURE EXTRACTION
              Freeze the pretrained MobileNetV2 base and train only a new
              classification head.
    Stage 2 - FINE-TUNING
              Unfreeze the top of the base and continue training at a very
              LOW learning rate so the pretrained weights are nudged, not
              wrecked.

It also shows cheap DATA AUGMENTATION (random flips and rotations) as a
regularizer.

------------------------------------------------------------------------
DATA (one-time setup)
------------------------------------------------------------------------
Point DATA_DIR at any folder of class-subfolders, for example:

    my_data\
        class_a\   (.jpg files)
        class_b\   (.jpg files)

If DATA_DIR is missing, the script falls back to a tiny batch of SYNTHETIC
images so it still runs end to end and you can see the two-stage loop work.
Swap in a real folder to train a real classifier.

Requires: tensorflow   (pip install tensorflow)
------------------------------------------------------------------------
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TensorFlow logging

import contextlib

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl


@contextlib.contextmanager
def quiet_jpeg_warnings():
    """Hide the harmless "Corrupt JPEG data: N extraneous bytes" notes.

    These come from the C image decoder (libjpeg) writing straight to the
    system error stream, so Python's logging can't catch them. We briefly
    redirect that stream to nowhere. Training progress still prints normally,
    because Keras sends it to standard output, not the error stream.
    """
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


# >>> EDIT THIS to a folder of class-subfolders, or leave it to use the
#     synthetic fallback. <<<
DATA_DIR = r"C:\python3_13\cnn_env\data\my_images"

IMG_SIZE = (160, 160)
BATCH = 32
SEED = 42


def get_data():
    """Return (train_ds, val_ds).

    If DATA_DIR exists, load real images from it (80/20 train/val split).
    Otherwise, build a small synthetic dataset so the script always runs.
    """
    if os.path.isdir(DATA_DIR):
        print(f"Loading images from: {DATA_DIR}")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR, validation_split=0.2, subset="training",
            seed=SEED, image_size=IMG_SIZE, batch_size=BATCH)
        val_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR, validation_split=0.2, subset="validation",
            seed=SEED, image_size=IMG_SIZE, batch_size=BATCH)
        return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)

    print("DATA_DIR not found - using synthetic images so the demo still runs.")
    X = np.random.rand(64, *IMG_SIZE, 3).astype("float32") * 255.0
    y = np.random.randint(0, 2, size=(64,)).astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((X, y)).batch(BATCH)
    return ds, ds


def data_augmenter():
    """Two cheap augmentation layers applied during training only."""
    return tf.keras.Sequential([
        tfl.RandomFlip("horizontal"),
        tfl.RandomRotation(0.2),
    ])


def build_model():
    """A MobileNetV2 base (frozen) plus a fresh binary head.

    weights='imagenet' downloads the pretrained weights once (~14 MB) and
    caches them. include_top=False drops ImageNet's 1000-class head so we can
    attach our own.
    """
    augment = data_augmenter()
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False                      # Stage 1: freeze the base

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = augment(inputs)
    x = preprocess(x)
    x = base(x, training=False)
    x = tfl.GlobalAveragePooling2D()(x)
    x = tfl.Dropout(0.2)(x)
    outputs = tfl.Dense(1)(x)                   # one binary logit
    return tf.keras.Model(inputs, outputs), base


def main():
    train_ds, val_ds = get_data()
    model, base = build_model()

    # Stage 1: train the new head only.
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    print("\nStage 1: feature extraction (base frozen)")
    with quiet_jpeg_warnings():
        model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)

    # Stage 2: unfreeze the top layers and fine-tune at a tiny learning rate.
    base.trainable = True
    for layer in base.layers[:-30]:             # keep the lower layers frozen
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # note the tiny LR
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    print("\nStage 2: fine-tuning (top layers unfrozen, LR=1e-5)")
    with quiet_jpeg_warnings():
        model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)

    print("\nThat two-stage recipe - extract features, then gently fine-tune -")
    print("is the workhorse of practical computer vision.")


if __name__ == "__main__":
    main()
