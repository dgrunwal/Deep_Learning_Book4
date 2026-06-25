r"""
transfer_learning_demo.py  -  Chapter 5 companion
Convolutional Neural Networks for Beginners (AI Foundations Series, Book 4)
(c) 2026 David Grunwald. All rights reserved.

Two-stage transfer learning on a small dataset:
  Stage 1 - freeze a pretrained MobileNetV2 base, train a new head.
  Stage 2 - unfreeze the top layers and fine-tune at a LOW learning rate.
Includes basic data augmentation as cheap regularization.

------------------------------------------------------------------------
HOW TO GET THE DATA (one-time setup)
------------------------------------------------------------------------
This script reads images from a local folder, so there are no dataset
libraries to install and nothing that breaks on Windows.

1. Download Microsoft's "Kaggle Cats and Dogs Dataset" (a single .zip):
       https://www.microsoft.com/en-us/download/details.aspx?id=54765
2. Unzip it. Inside you will find a folder named  PetImages  that looks
   like this:

       PetImages\
           Cat\   (lots of .jpg files)
           Dog\   (lots of .jpg files)

3. Set DATA_DIR below to the full path of that PetImages folder.

That folder-of-class-subfolders layout is all the script needs. Any
dataset with the same shape (one subfolder per class) will work - just
point DATA_DIR at it.

Requires: tensorflow      (pip install tensorflow)
------------------------------------------------------------------------
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TensorFlow logging

import contextlib

import tensorflow as tf
from tensorflow.keras import layers, models


@contextlib.contextmanager
def quiet_jpeg_warnings():
    """Hide the harmless "Corrupt JPEG data: N extraneous bytes" notes.

    Those come from the C image decoder (libjpeg) writing straight to the
    system error stream, so Python's logging can't catch them. We briefly
    redirect that stream to nowhere. Training progress still prints normally,
    because Keras sends it to standard output, not the error stream.
    """
    saved = os.dup(2)                       # remember the real error stream
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)                      # send error stream to nowhere
    try:
        yield
    finally:
        os.dup2(saved, 2)                   # put the error stream back
        os.close(devnull)
        os.close(saved)

# >>> EDIT THIS ONE LINE to point at your unzipped PetImages folder <<<
DATA_DIR = r"C:\python3_13\cnn_env\data\PetImages"

IMG_SIZE = (160, 160)
BATCH = 32
SEED = 42


def clean_corrupt_images(folder):
    """The raw cats/dogs dataset ships a few unreadable files that crash
    training. Walk the folder once and delete any file TensorFlow cannot
    decode as a JPEG. Safe to run every time; it only removes bad files."""
    removed = 0
    with quiet_jpeg_warnings():
        for root, _, files in os.walk(folder):
            for name in files:
                path = os.path.join(root, name)
                try:
                    data = tf.io.read_file(path)
                    tf.io.decode_jpeg(data)      # raises if not a valid JPEG
                except Exception:
                    try:
                        os.remove(path)
                        removed += 1
                    except OSError:
                        pass
    if removed:
        print(f"Removed {removed} unreadable image file(s).")


def get_data():
    """Load images from DATA_DIR, split 80/20 into train/validation."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training",
        seed=SEED, image_size=IMG_SIZE, batch_size=BATCH)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation",
        seed=SEED, image_size=IMG_SIZE, batch_size=BATCH)

    # MobileNetV2 expects inputs scaled to [-1, 1]; its preprocess_input
    # does exactly that.
    pp = tf.keras.applications.mobilenet_v2.preprocess_input
    train_ds = train_ds.map(lambda x, y: (pp(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (pp(x), y)).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def build_model():
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ])
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False  # Stage 1: freeze the convolutional base

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = augment(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)  # one binary logit (cat vs dog)
    return models.Model(inputs, outputs), base


def main():
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(
            f"\nDATA_DIR not found:\n    {DATA_DIR}\n\n"
            "Edit the DATA_DIR line near the top of this file to point at\n"
            "your unzipped PetImages folder. See the instructions in the\n"
            "docstring at the top of the script.")

    clean_corrupt_images(DATA_DIR)
    train_ds, val_ds = get_data()
    model, base = build_model()

    # Stage 1: train the new head only.
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    print("\nStage 1: feature extraction (base frozen)")
    with quiet_jpeg_warnings():
        model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)

    # Stage 2: unfreeze top layers, fine-tune at a low learning rate.
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # note the tiny LR
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    print("\nStage 2: fine-tuning (top layers unfrozen, LR=1e-5)")
    with quiet_jpeg_warnings():
        model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)


if __name__ == "__main__":
    main()
